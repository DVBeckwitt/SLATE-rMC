"""Worker-owned hash-bound hBN preparation, frozen fitting and exact external result I/O."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
from hbn_state import HbnSession, encoded, hbn_session_from_document, payload_hash
from job_lifecycle import JobResult
from mask_state import mask_from_document
from osc_import import AXIS_LIMIT, DECODED_LIMIT_BYTES, PIXEL_LIMIT, SOURCE_LIMIT_BYTES
from simulation_io import (
    MAX_SIMULATION_CPU_BYTES,
    MAX_SIMULATION_GPU_BYTES,
    _external,
    _publish_bytes,
    _stop,
)


@dataclass(frozen=True, slots=True)
class HbnWorkResult:
    operation: str
    session: HbnSession
    curves: tuple[np.ndarray, ...] = ()
    presented_result_id: str | None = None
    spot_data: np.ndarray | None = None
    spot_model: np.ndarray | None = None
    spot_residual: np.ndarray | None = None

    @property
    def nbytes(self):
        return (
            self.session.nbytes
            + sum(
                a.nbytes
                for a in (*self.curves, self.spot_data, self.spot_model, self.spot_residual)
                if a is not None
            )
            + 8192
        )


def hbn_work_budget(operation, shape_rc, other_cpu_bytes=0, other_gpu_bytes=0):
    if (
        len(shape_rc) != 2
        or any(type(v) is not int or not 0 < v <= AXIS_LIMIT for v in shape_rc)
        or math.prod(shape_rc) > PIXEL_LIMIT
    ):
        raise ValueError("hBN shape exceeds the existing desktop admission")
    working = (
        96 * math.prod(shape_rc) if operation in ("load", "prepare", "spot") else 0
    ) + 64 * 1024**2
    if (
        type(other_cpu_bytes) is not int
        or other_cpu_bytes < 0
        or type(other_gpu_bytes) is not int
        or other_gpu_bytes < 0
        or working + other_cpu_bytes > MAX_SIMULATION_CPU_BYTES
        or other_gpu_bytes > MAX_SIMULATION_GPU_BYTES
    ):
        raise ValueError(
            "combined hBN working/display/history budget exceeds 2 GiB CPU or 512 MiB GPU"
        )
    return {
        "working_cpu_bytes": working,
        "other_cpu_bytes": other_cpu_bytes,
        "other_gpu_bytes": other_gpu_bytes,
        "shape_rc": list(shape_rc),
    }


def _hash_file(path, control):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            _stop(control)
            h.update(chunk)
    return h.hexdigest()


def _osc(path, control, expected=None):
    from rasim_next.io.osc import OscReadLimits, read_osc

    raw_sha = _hash_file(path, control)
    image = read_osc(
        Path(path),
        limits=OscReadLimits(SOURCE_LIMIT_BYTES, DECODED_LIMIT_BYTES, PIXEL_LIMIT, AXIS_LIMIT),
        canceled=lambda: control.canceled,
    )
    if raw_sha != _hash_file(path, control) or (
        expected is not None and image.decoded_sha256 != expected
    ):
        raise ValueError("hBN OSC bytes/decoded identity changed")
    return image, raw_sha


def _inputs_current(inputs, control):
    for key, sha in (
        ("source_path", "source_file_sha256"),
        ("dark_path", "dark_file_sha256"),
        ("configuration_path", "configuration_sha256"),
        ("cif_path", "cif_sha256"),
    ):
        if _hash_file(inputs[key], control) != inputs[sha]:
            raise ValueError("hBN input changed since admission: " + key)


def _geometry(inputs):
    return {
        name: inputs[name]
        for name in (
            "base_detector_rotation",
            "beam_direction_lab",
            "detector_column_pitch_m",
            "detector_row_pitch_m",
        )
    }


def _observations(pack):
    from rasim_next.fitting.hbn import HbnRingObservations

    row = pack.copy()
    digest = row.pop("sha256")
    if digest != payload_hash(row) or pack["schema"] != "slate.hbn-observations.v1":
        raise ValueError("frozen hBN pack changed")
    return HbnRingObservations(
        np.asarray(pack["coordinates_px"]),
        np.asarray(pack["ring_index"]),
        np.asarray(pack["two_theta_rad"]),
        np.asarray(pack["angular_sector"]),
    )


def _inclusion(inputs):
    mask = mask_from_document(inputs["mask"])
    if mask is None:
        return None
    if mask.source_sha256 != inputs["source_sha256"] or mask.shape != tuple(inputs["shape_rc"]):
        raise ValueError("hBN mask identity/shape mismatch")
    inclusion = np.ones(mask.shape, dtype=np.bool_)
    for row, start, stop, _reason in mask.spans:
        inclusion[row, start:stop] = False
    return inclusion


def _json_finite(value):
    if isinstance(value, dict):
        return {k: _json_finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_finite(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def validate_hbn_result(record, session):
    from rasim_next.fitting.hbn import evaluate_hbn_residual_px

    if record.get("schema") != "slate.hbn-result.v1" or record["acquisition_id"] != str(
        session.acquisition_id
    ):
        raise ValueError("result belongs to another hBN acquisition")
    row = record.copy()
    digest = row.pop("sha256")
    UUID(row["result_id"])
    if digest != payload_hash(row):
        raise ValueError("hBN result content hash mismatch")
    if type(record.get("name")) is not str or not 0 < len(record["name"]) <= 128:
        raise ValueError("hBN result needs a bounded nonempty name")
    launch = record["launch"]
    pack = launch["pack"]
    obs = _observations(pack)
    if (
        launch["inputs"] != pack["inputs"]
        or payload_hash(launch["inputs"]) != pack["inputs_sha256"]
    ):
        raise ValueError("hBN result launch/observation input mismatch")
    values = np.asarray(record["fitted_values"], dtype=float)
    if (
        values.shape != (5,)
        or not np.all(np.isfinite(values))
        or np.any(values < launch["lower"])
        or np.any(values > launch["upper"])
    ):
        raise ValueError("hBN fitted values violate their launch bounds")
    raw = evaluate_hbn_residual_px(values, obs, **_geometry(launch["inputs"]))
    if not np.array_equal(raw, np.asarray(record["residual_px"])):
        raise ValueError("hBN residual values do not match canonical result geometry")
    from rasim_next.fitting.hbn import HbnDetectorCalibration

    calibration = HbnDetectorCalibration(**record["calibration"])
    if (
        not np.array_equal(calibration.values, values)
        or type(calibration.success) is not bool
        or type(calibration.solver_success) is not bool
    ):
        raise ValueError("hBN calibration record disagrees with fitted values")
    if (
        len(calibration.ring_point_count) != 5
        or len(calibration.ring_rms_px) != 5
        or len(calibration.ring_angular_coverage_fraction) != 5
        or len(calibration.standard_error) != 5
        or len(calibration.active_bounds) != 5
        or not 0 <= calibration.jacobian_rank <= 5
        or not 1 <= calibration.function_evaluations <= launch["max_nfev"]
    ):
        raise ValueError("invalid hBN statistics dimensions/solver state")
    if calibration.residual_rms_px != float(
        np.sqrt(np.mean(raw**2))
    ) or calibration.residual_max_px != float(np.max(np.abs(raw))):
        raise ValueError("hBN summary disagrees with exact residuals")
    if list(calibration.ring_point_count) != np.bincount(obs.ring_index, minlength=5).tolist():
        raise ValueError("hBN per-ring counts disagree with the frozen pack")
    return record


def _export_hbn(request, session, inputs, control, record=None):
    path = Path(request["path"]).resolve(strict=False)
    protected = [
        Path(inputs[k]) for k in ("source_path", "dark_path", "configuration_path", "cif_path")
    ] + [Path(p) for p, _sha in session.exports]
    operation = request["operation"]
    if operation == "export_observations":
        if not session.frozen_json:
            raise ValueError("freeze reviewed observations before export")
        outputs = [(path, session.frozen_json.encode())]
    elif operation == "export":
        outputs = [(path, encoded(record).encode())]
    else:
        from io import BytesIO

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from rasim_next.fitting.hbn import hbn_ring_curves_px

        figure = Figure(figsize=(8, 7), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, 1, height_ratios=(3, 1))
        pack = record["launch"]["pack"]
        points = np.asarray(pack["coordinates_px"])
        curves = hbn_ring_curves_px(
            record["fitted_values"], pack["two_theta_rad"], **_geometry(record["launch"]["inputs"])
        )
        for curve in curves:
            axes[0].plot(curve[:, 0], curve[:, 1], linewidth=0.7)
        axes[0].scatter(points[:, 0], points[:, 1], s=8)
        rows, columns = record["launch"]["inputs"]["shape_rc"]
        axes[0].set(
            xlim=(0, columns - 1),
            ylim=(rows - 1, 0),
            xlabel="native column_px",
            ylabel="native row_px",
            title=record["name"] + " — " + record["qualification"],
        )
        axes[0].set_aspect("equal")
        axes[1].plot(record["residual_px"], ".")
        axes[1].set(xlabel="frozen observation index", ylabel="residual (px)")
        stream = BytesIO()
        figure.savefig(stream, format="png", dpi=140)
        outputs = [
            (path, stream.getvalue()),
            (
                path.with_suffix(path.suffix + ".values.json"),
                encoded(
                    {"result": record, "curves_px": [_json_finite(c.tolist()) for c in curves]}
                ).encode(),
            ),
        ]
    if len(session.exports) + len(outputs) > 16:
        raise ValueError("hBN export reference cap reached")
    if operation == "export_figure":
        outputs.reverse()
    for target, _raw in outputs:
        _external(target, protected)
    references = []
    for target, raw in outputs:
        _publish_bytes(target, raw, control)
        if target.read_bytes() != raw:
            raise ValueError("hBN export readback differs")
        references.append((str(target), hashlib.sha256(raw).hexdigest()))
    return replace(session, exports=(*session.exports, *references))


def hbn_work(argument, control):
    from threadpoolctl import threadpool_limits

    request = json.loads(argument)
    operation = request["operation"]
    _stop(control)
    if "resources" in request:
        resources = request["resources"]
        if resources != hbn_work_budget(
            operation,
            resources["shape_rc"],
            resources["other_cpu_bytes"],
            resources["other_gpu_bytes"],
        ):
            raise ValueError("hBN resource snapshot changed")
    with threadpool_limits(limits=1):
        if operation == "load":
            from metadata_review import bounded_reference_snapshot

            from rasim_next.pipeline.configured_simulation import (
                _compile_instrument,
                load_simulation_config,
            )

            image, source_raw = _osc(request["source_path"], control, request["source_sha256"])
            dark, dark_raw = _osc(request["dark_path"], control)
            if (
                "resources" in request
                and list(image.detector_native_counts.shape) != request["resources"]["shape_rc"]
            ):
                raise ValueError("hBN decoded shape differs from the admitted resource snapshot")
            if dark.detector_native_counts.shape != image.detector_native_counts.shape:
                raise ValueError("hBN and dark must have the same native shape")
            path = Path(request["configuration_path"]).resolve(strict=True)
            raw, _identity = bounded_reference_snapshot(path)
            config = load_simulation_config(path, source_bytes=raw)
            instrument = _compile_instrument(config.instrument)
            if instrument.detector_shape_rc != image.detector_native_counts.shape:
                raise ValueError("hBN native shape differs from the base detector configuration")
            inputs = {
                "schema": "slate.hbn-inputs.v1",
                "acquisition_id": request["acquisition_id"],
                "metadata_revision": request["metadata_revision"],
                "source_path": request["source_path"],
                "source_sha256": image.decoded_sha256,
                "source_file_sha256": source_raw,
                "dark_path": str(Path(request["dark_path"]).resolve()),
                "dark_sha256": dark.decoded_sha256,
                "dark_file_sha256": dark_raw,
                "configuration_path": str(path),
                "configuration_sha256": hashlib.sha256(raw).hexdigest(),
                "cif_path": str(config.material.cif_path),
                "cif_sha256": _hash_file(config.material.cif_path, control),
                "mask": request["mask"],
                "shape_rc": list(image.detector_native_counts.shape),
                "base_detector_rotation": instrument.lab_from_detector.rotation.tolist(),
                "beam_direction_lab": list(config.source.mean_direction_lab),
                "detector_column_pitch_m": instrument.detector_column_pitch_m,
                "detector_row_pitch_m": instrument.detector_row_pitch_m,
                "wavelength_A": 1.5406,
                "lattice_a_A": 2.504,
                "lattice_c_A": 6.661,
                "dark_handling": "unscaled native raw dark subtraction; no exposure normalization",
                "orientation": "accepted clockwise OSC conversion once at I/O",
                "cpu_workers": 1,
                "nested_blas_threads": 1,
            }
            session = HbnSession(
                uuid4(),
                UUID(request["acquisition_id"]),
                0,
                encoded(inputs),
                tuple(request.get("initial", (0, 0, 1453.12, 1596.422, 0.074))),
                (-0.15, -0.15, 0, 0, 0.04),
                (
                    0.15,
                    0.15,
                    image.detector_native_counts.shape[1] - 1,
                    image.detector_native_counts.shape[0] - 1,
                    0.12,
                ),
            )
            _inclusion(inputs)
            result = HbnWorkResult(operation, session)
        else:
            session = hbn_session_from_document(request["session"])
            inputs = json.loads(session.inputs_json)
            _inputs_current(inputs, control)
            if operation in ("prepare", "spot"):
                image, _ = _osc(inputs["source_path"], control, inputs["source_sha256"])
                dark, _ = _osc(inputs["dark_path"], control, inputs["dark_sha256"])
                mask = _inclusion(inputs)
                counts = image.detector_native_counts.astype(np.float64)
                dark_counts = dark.detector_native_counts
                _stop(control)
                if operation == "prepare":
                    from rasim_next.fitting.hbn import prepare_hbn_ring_observations

                    control.report(
                        "Discovering and preliminarily refining the five hBN rings for review"
                    )
                    obs, _preliminary = prepare_hbn_ring_observations(
                        counts,
                        dark_counts,
                        **_geometry(inputs),
                        initial_beam_center_px=session.initial[2:4],
                        initial_calibrant_distance_m=session.initial[4],
                        initial_tilts_rad=session.initial[:2],
                        inclusion_mask=mask,
                        lattice_a_A=inputs["lattice_a_A"],
                        lattice_c_A=inputs["lattice_c_A"],
                        wavelength_A=inputs["wavelength_A"],
                        lower_bounds=session.lower,
                        upper_bounds=session.upper,
                        f_scale=session.f_scale,
                        max_nfev=session.max_nfev,
                        canceled=lambda: control.canceled,
                    )
                    candidates = tuple(
                        (float(c), float(r), int(i), int(s))
                        for (c, r), i, s in zip(
                            obs.coordinates_px, obs.ring_index, obs.angular_sector, strict=True
                        )
                    )
                    result = HbnWorkResult(
                        operation,
                        replace(
                            session,
                            revision=session.revision + 1,
                            candidates=candidates,
                            exclusions=(),
                            frozen_json="",
                            selected_result_id=None,
                        ),
                    )
                else:
                    from rasim_next.fitting.spot import estimate_gaussian_spot

                    saturation = request.get("saturation_count")
                    if saturation is not None:
                        if not np.isfinite(saturation) or saturation <= 0:
                            raise ValueError("saturation threshold must be a positive raw count")
                        valid = image.detector_native_counts < saturation
                        mask = valid if mask is None else mask & valid
                    counts -= dark_counts
                    roi = tuple(request["roi"])
                    control.report("Fitting the declared native ROI Gaussian proposal")
                    proposal = estimate_gaussian_spot(
                        counts,
                        roi_column_row_bounds=roi,
                        inclusion_mask=mask,
                        canceled=lambda: control.canceled,
                    )
                    payload = {
                        "proposal_id": str(uuid4()),
                        "method": "elliptical_gaussian",
                        "inputs_sha256": session.inputs_sha256,
                        "draft_revision": session.revision,
                        "roi": roi,
                        "values": proposal.values,
                        "parameter_names": [
                            "background_count",
                            "amplitude_count",
                            "center_column_px",
                            "center_row_px",
                            "width_1_px",
                            "width_2_px",
                            "orientation_rad",
                        ],
                        "parameter_units": [
                            "count",
                            "count",
                            "native column px",
                            "native row px",
                            "pixel",
                            "pixel",
                            "radian",
                        ],
                        "center_standard_error_px": proposal.center_standard_error_px,
                        "residual_rms_count": proposal.residual_rms_count,
                        "valid_fraction": proposal.valid_fraction,
                        "jacobian_rank": proposal.jacobian_rank,
                        "condition": proposal.condition,
                        "reliable_initial_estimate": proposal.reliable_initial_estimate,
                        "limitations": proposal.limitations,
                        "interpretation": "observed spot center; approximate initial estimate, not a geometric beam intercept",
                        "saturation_raw_count": saturation,
                        "noise_model": "unweighted least squares of original valid raw-dark counts; model-dependent OLS errors",
                    }
                    c0, c1, r0, r1 = roi
                    data = np.array(counts[r0:r1, c0:c1], copy=True)
                    data.setflags(write=False)
                    result = HbnWorkResult(
                        operation,
                        replace(session, center_proposal_json=encoded(_json_finite(payload))),
                        spot_data=data,
                        spot_model=proposal.model,
                        spot_residual=proposal.residual,
                    )
            elif operation == "export_observations":
                result = HbnWorkResult(operation, _export_hbn(request, session, inputs, control))
            elif operation == "fit":
                from rasim_next.fitting.hbn import (
                    evaluate_hbn_residual_px,
                    fit_hbn_ring_observations,
                    hbn_ring_curves_px,
                )

                if not session.frozen_json:
                    raise ValueError("review and freeze observations before Fit")
                if len(session.results_json) >= 4:
                    raise ValueError(
                        "four retained hBN results; remove a result before another Fit"
                    )
                if type(request["name"]) is not str or not 0 < len(request["name"]) <= 128:
                    raise ValueError("hBN result needs a name of 1-128 characters")
                pack = json.loads(session.frozen_json)
                obs = _observations(pack)
                if pack["inputs_sha256"] != session.inputs_sha256:
                    raise ValueError("frozen hBN input identity is stale")
                control.report("Fitting exactly the frozen reviewed observation pack; no retracing")
                calibration = fit_hbn_ring_observations(
                    obs,
                    session.initial,
                    **_geometry(inputs),
                    detector_shape_rc=tuple(inputs["shape_rc"]),
                    lower_bounds=session.lower,
                    upper_bounds=session.upper,
                    f_scale=session.f_scale,
                    max_nfev=session.max_nfev,
                    canceled=lambda: control.canceled,
                )
                residual = evaluate_hbn_residual_px(calibration.values, obs, **_geometry(inputs))
                launch = {
                    "inputs": inputs,
                    "pack": pack,
                    "initial": session.initial,
                    "lower": session.lower,
                    "upper": session.upper,
                    "loss": "soft_l1",
                    "f_scale": session.f_scale,
                    "max_nfev": session.max_nfev,
                    "launch_sha256": session.launch_sha256,
                }
                record = _json_finite(
                    {
                        "schema": "slate.hbn-result.v1",
                        "result_id": str(uuid4()),
                        "name": request["name"],
                        "acquisition_id": str(session.acquisition_id),
                        "launch": launch,
                        "fitted_values": calibration.values.tolist(),
                        "residual_px": residual.tolist(),
                        "calibration": asdict(calibration),
                        "qualification": "qualified by existing hBN owner"
                        if calibration.success
                        else "unqualified candidate; existing hBN checks did not qualify",
                        "parameter_units": [
                            "radian",
                            "radian",
                            "native column px",
                            "native row px",
                            "metre; calibrant-private",
                        ],
                    }
                )
                record["sha256"] = payload_hash(record)
                text = encoded(record)
                updated = replace(session, results_json=(*session.results_json, text))
                result = HbnWorkResult(
                    operation,
                    updated,
                    presented_result_id=record["result_id"],
                    curves=hbn_ring_curves_px(
                        calibration.values, obs.two_theta_rad, **_geometry(inputs)
                    ),
                )
            elif operation in ("export", "export_figure", "import", "present"):
                if operation == "import":
                    path = Path(request["path"])
                    if path.stat().st_size > 192 * 1024:
                        raise ValueError("hBN imported result exceeds 192 KiB")
                    with path.open("rb") as stream:
                        raw = stream.read(192 * 1024 + 1)
                    if len(raw) > 192 * 1024:
                        raise ValueError("hBN imported result exceeds 192 KiB")
                    record = json.loads(raw)
                else:
                    record = json.loads(request["record"])
                    if not any(
                        json.loads(v)["sha256"] == record["sha256"] for v in session.results_json
                    ):
                        raise ValueError("hBN result is not retained in this session")
                validate_hbn_result(record, session)
                if operation == "import":
                    if (
                        not session.frozen_json
                        or record["launch"]["pack"]["sha256"]
                        != json.loads(session.frozen_json)["sha256"]
                        or record["launch"]["inputs"] != inputs
                    ):
                        raise ValueError(
                            "imported hBN record mismatches the current frozen pack/inputs"
                        )
                    if any(
                        json.loads(v)["result_id"] == record["result_id"]
                        for v in session.results_json
                    ):
                        raise ValueError("hBN result already retained")
                    session = replace(
                        session, results_json=(*session.results_json, encoded(record))
                    )
                if operation in ("export", "export_figure"):
                    session = _export_hbn(request, session, inputs, control, record)
                from rasim_next.fitting.hbn import hbn_ring_curves_px

                result = HbnWorkResult(
                    operation,
                    session,
                    presented_result_id=record["result_id"],
                    curves=hbn_ring_curves_px(
                        record["fitted_values"],
                        record["launch"]["pack"]["two_theta_rad"],
                        **_geometry(record["launch"]["inputs"]),
                    ),
                )
            else:
                raise ValueError("unknown hBN operation")
        _inputs_current(json.loads(result.session.inputs_json), control)
        _stop(control)
        return JobResult(result, result.nbytes)
