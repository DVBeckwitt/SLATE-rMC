"""Sample-only geometry operations on the application's single owning worker."""

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from uuid import uuid4

import numpy as np
from hbn_io import _hash_file, _osc
from job_lifecycle import JobResult
from mask_state import NativeMask, mask_document, mask_from_document
from sample_state import SampleSession, encoded, payload_hash, sample_session_from_document
from simulation_io import (
    MAX_SIMULATION_CPU_BYTES,
    MAX_SIMULATION_GPU_BYTES,
    _external,
    _publish_bytes,
    _stop,
)


@dataclass(frozen=True, slots=True)
class SampleWorkResult:
    operation: str
    session: SampleSession
    presented_result_id: str | None = None
    input_checks: tuple[tuple[str, str], ...] = ()

    @property
    def nbytes(self):
        return (
            self.session.nbytes
            + 8192
            + sum(
                len(identity) + len(detail.encode()) + 128 for identity, detail in self.input_checks
            )
        )


def sample_work_budget(other_cpu_bytes=0, other_gpu_bytes=0):
    # One 12-Mpixel discovery workspace plus eight admitted int32 native planes.
    working = 96 * 12_000_000 + 8 * 12_000_000 * 4 + 64 * 1024**2
    if (
        type(other_cpu_bytes) is not int
        or type(other_gpu_bytes) is not int
        or min(other_cpu_bytes, other_gpu_bytes) < 0
        or working + other_cpu_bytes > MAX_SIMULATION_CPU_BYTES
        or other_gpu_bytes > MAX_SIMULATION_GPU_BYTES
    ):
        raise ValueError(
            "sample preparation exceeds the combined 2 GiB CPU / 512 MiB GPU admission"
        )
    return {
        "working_cpu_bytes": working,
        "other_cpu_bytes": other_cpu_bytes,
        "other_gpu_bytes": other_gpu_bytes,
    }


def observation_id(image_id, key):
    return payload_hash([image_id, key])


class SampleInputsUnavailable(ValueError):
    """A recorded live input is missing, unreadable or has different bytes."""


def _current(inputs, control):
    for row in inputs["files"]:
        _stop(control)
        try:
            digest = _hash_file(row["path"], control)
        except OSError as exc:
            raise SampleInputsUnavailable(
                "sample input unavailable: " + row["path"] + ": " + str(exc)
            ) from exc
        if digest != row["sha256"]:
            raise SampleInputsUnavailable("sample input changed since admission: " + row["path"])


def _controls(image_count):
    from rasim_next.fitting.indexed_series import (
        DETECTOR_CALIBRATION_PARAMETER_NAMES,
        SHARED_GEOMETRY_PARAMETER_NAMES,
        SharedGeometryCorrectionBounds,
    )

    bounds = SharedGeometryCorrectionBounds.rasim_multi_angle_pose()
    names = (
        list(SHARED_GEOMETRY_PARAMETER_NAMES + DETECTOR_CALIBRATION_PARAMETER_NAMES)
        + ["incidence_angle_delta_rad"]
        + [f"incidence_angle_trim_helmert_{i + 1}_rad" for i in range(image_count - 1)]
    )
    spans = (
        list(bounds.half_span)
        + [10.0, 10.0, 0.01, math.radians(0.5)]
        + [math.radians(0.5)] * (image_count - 1)
    )
    return {
        "names": names,
        "initial": [0.0] * len(names),
        "lower": [-v for v in spans],
        "upper": spans,
        "fitted": list(SHARED_GEOMETRY_PARAMETER_NAMES),
        "trim_prior_sigma_rad": math.radians(0.25),
        "solver": {
            "method": "trf",
            "jac": "2-point",
            "ftol": 1e-12,
            "xtol": 1e-12,
            "gtol": 1e-12,
            "max_nfev": 150,
        },
    }


def _fit_arguments(controls, image_count):
    from rasim_next.fitting.indexed_series import (
        DETECTOR_CALIBRATION_PARAMETER_NAMES,
        SHARED_GEOMETRY_PARAMETER_NAMES,
        DetectorCalibrationCorrectionBounds,
        DetectorCalibrationCorrections,
        IncidenceAngleDeltaBounds,
        SharedGeometryCorrectionBounds,
        SharedGeometryCorrections,
    )

    default = _controls(image_count)
    if (
        set(controls) != set(default)
        or controls["names"] != default["names"]
        or controls["solver"] != default["solver"]
    ):
        raise ValueError("unsupported sample parameter ordering or solver defaults")
    values = np.array([controls[k] for k in ("initial", "lower", "upper")], dtype=np.float64)
    if (
        values.shape != (3, len(default["names"]))
        or not np.all(np.isfinite(values))
        or np.any(values[1] >= values[2])
        or np.any(values[0] < values[1])
        or np.any(values[0] > values[2])
    ):
        raise ValueError("sample initial values and bounds must be finite and ordered")
    fitted = controls["fitted"]
    if (
        not fitted
        or len(set(fitted)) != len(fitted)
        or any(n not in default["names"] for n in fitted)
    ):
        raise ValueError("choose actual unique fitted parameters")
    if "incidence_angle_delta_rad" in fitted and "sample_normal_x_tilt_rad" in fitted:
        raise ValueError("incidence delta and sample normal x share a gauge; fix one")
    trim_names = default["names"][13:]
    trim_on = any(n in fitted for n in trim_names)
    if trim_on and (
        not all(n in fitted for n in trim_names)
        or not np.all(values[1, 13:] == values[1, 13])
        or not np.all(values[2, 13:] == -values[1, 13])
    ):
        raise ValueError("fit the full zero-sum Helmert contrast span with equal symmetric bounds")
    if not trim_on and np.any(values[0, 13:] != 0):
        raise ValueError("fixed per-image trims must be zero under the current owner")
    if not math.isfinite(controls["trim_prior_sigma_rad"]) or controls["trim_prior_sigma_rad"] <= 0:
        raise ValueError("trim prior sigma must be positive")
    return dict(
        initial=SharedGeometryCorrections.from_array(values[0, :9]),
        bounds=SharedGeometryCorrectionBounds(
            SharedGeometryCorrections.from_array(values[1, :9]),
            SharedGeometryCorrections.from_array(values[2, :9]),
        ),
        fitted_parameter_names=tuple(n for n in SHARED_GEOMETRY_PARAMETER_NAMES if n in fitted),
        initial_detector_calibration_corrections=DetectorCalibrationCorrections.from_array(
            values[0, 9:12]
        ),
        detector_calibration_correction_bounds=DetectorCalibrationCorrectionBounds(
            DetectorCalibrationCorrections.from_array(values[1, 9:12]),
            DetectorCalibrationCorrections.from_array(values[2, 9:12]),
        ),
        fitted_detector_calibration_parameter_names=tuple(
            n for n in DETECTOR_CALIBRATION_PARAMETER_NAMES if n in fitted
        ),
        initial_incidence_angle_delta_rad=float(values[0, 12]),
        incidence_angle_delta_bounds=IncidenceAngleDeltaBounds(
            float(values[1, 12]), float(values[2, 12])
        )
        if "incidence_angle_delta_rad" in fitted
        else None,
        initial_incidence_angle_trim_contrast_rad=values[0, 13:] if trim_on else None,
        incidence_angle_trim_contrast_half_span_rad=float(values[2, 13]) if trim_on else None,
        incidence_angle_trim_prior_sigma_rad=controls["trim_prior_sigma_rad"] if trim_on else None,
    )


def _selection(document, exclusions=()):
    from rasim_next.fitting.geometry import IntegerLMarkerKey
    from rasim_next.selection.indexing import (
        MarkerIndexingDecision,
        MarkerIndexingStatus,
        MeasuredImageIndexingResult,
        PeakIndexingPolicy,
        select_confident_branch_tracks,
    )

    policy = PeakIndexingPolicy(**document["policy"])
    excluded = dict(exclusions)
    known = set()
    images = []
    for row in document["images"]:
        decisions = []
        for original in row["marker_decisions"]:
            data = dict(original)
            k = dict(data["key"])
            k["representative_rod_hk"] = tuple(k["representative_rod_hk"])
            key = IntegerLMarkerKey(**k)
            identity = observation_id(row["image_id"], asdict(key))
            known.add(identity)
            if identity in excluded:
                if data["observed_column_px"] is None:
                    raise ValueError("only a discovered observation may be excluded")
                data["status"] = MarkerIndexingStatus.MASKED_OR_CLIPPED
                data["reason"] = "Reviewed exclusion: " + excluded[identity]
            data["key"] = key
            decisions.append(MarkerIndexingDecision(**data))
        images.append(
            MeasuredImageIndexingResult(
                **{
                    k: row[k]
                    for k in (
                        "image_id",
                        "incidence_angle_rad",
                        "reference_wavelength_A",
                        "detector_data_hash",
                        "detector_mask_hash",
                        "detector_mask_revision",
                        "context_hash",
                    )
                },
                marker_decisions=tuple(decisions),
                policy=policy,
            )
        )
    if set(excluded) - known:
        raise ValueError("review refers to an unknown observation identity")
    return select_confident_branch_tracks(tuple(images), policy=policy)


def _selection_document(selection):
    return {
        "policy": asdict(selection.policy),
        "images": [
            {k: v for k, v in asdict(image).items() if k not in ("branch_tracks", "result_hash")}
            for image in selection.image_results
        ],
        "manifest_hash": selection.manifest_hash,
        "tracks": [asdict(t) for t in selection.branch_tracks],
    }


def freeze_sample(session):
    if session.prepared_json is None:
        raise ValueError("Prepare the complete declared series before review/freeze")
    prepared = json.loads(session.prepared_json)
    original = _selection(prepared["selection"])
    if encoded(_selection_document(original)) != encoded(prepared["selection"]):
        raise ValueError("prepared selection disagrees with canonical track admission")
    selection = _selection(prepared["selection"], session.exclusions)
    observations = []
    for row in json.loads(session.inputs_json)["images"]:
        obs = selection.observations_for(row["image_id"])
        observations.append(
            {
                "image_id": row["image_id"],
                "keys": [asdict(k) for k in obs.keys],
                "coordinates_px": obs.coordinates_px.tolist(),
                "covariance_px2": obs.covariance_px2.tolist(),
                "reference_wavelength_A": obs.reference_wavelength_A,
                "observation_ids": [observation_id(row["image_id"], asdict(k)) for k in obs.keys],
            }
        )
    pack = {
        "schema": "slate.sample-observations.v1",
        "inputs_sha256": payload_hash(json.loads(session.inputs_json)),
        "review": [list(v) for v in session.exclusions],
        "selection": _selection_document(selection),
        "observations": observations,
        "ordering": "declared image order; canonical marker key order; owner residual uses sorted image IDs",
        "measure": "detector-native continuous (column_px,row_px), full site covariance px2 plus canonical chord residual",
    }
    pack["sha256"] = payload_hash(pack)
    return replace(session, frozen_json=encoded(pack))


def _saved_discovery(document):
    """Read the canonical bounded discovery representation without a live geometry context."""
    from rasim_next.selection.blind import (
        BlindIndexingPolicy,
        DiscoveredCakePeak,
        MeasuredPeakDiscovery,
    )
    from rasim_next.selection.indexing import PeakIndexingPolicy

    data = {k: v for k, v in document.items() if k != "discovery_hash"}
    data["detector_shape_rc"] = tuple(data["detector_shape_rc"])
    data["peaks"] = tuple(DiscoveredCakePeak(**v) for v in data["peaks"])
    policy = dict(data["policy"])
    policy["track_policy"] = PeakIndexingPolicy(**policy["track_policy"])
    data["policy"] = BlindIndexingPolicy(**policy)
    discovery = MeasuredPeakDiscovery(**data)
    if discovery.discovery_hash != document["discovery_hash"]:
        raise ValueError("saved sample discovery hash mismatch")
    return discovery


def _images(session, control):
    from rasim_next.fitting.geometry import (
        ExactTagGeometryModel,
        IntegerLMarkerKey,
        IntegerLMarkerObservations,
    )
    from rasim_next.fitting.indexed_series import IndexedGeometryImage
    from rasim_next.pipeline.configured_simulation import (
        build_configured_geometry_inputs,
        build_geometry_only_ewald_context,
        load_simulation_config,
        rebind_configured_geometry_instrument,
    )
    from rasim_next.selection.blind import (
        _discovery_geometry_hash,
        _indexing_context_hash,
    )
    from rasim_next.selection.osc_series import (
        build_osc_angle_frame,
        load_osc_geometry_series,
        simulation_config_for_osc_image,
    )

    if session.frozen_json is None or freeze_sample(session).frozen_json != session.frozen_json:
        raise ValueError("Fit/import requires the exact canonical reviewed pack")
    inputs = json.loads(session.inputs_json)
    _current(inputs, control)
    series = load_osc_geometry_series(inputs["manifest_path"])
    if (
        [(r.image_id, str(r.osc_path), list(r.axis_rotation_angles_deg)) for r in series.images]
        != [(r["image_id"], r["path"], r["axis_rotation_angles_deg"]) for r in inputs["images"]]
        or str(series.config_path) != inputs["configuration_path"]
        or series.incidence_axis_index != inputs["incidence_axis_index"]
    ):
        raise ValueError("declared sample roster/configuration changed")
    base = load_simulation_config(series.config_path)
    shared = build_configured_geometry_inputs(
        simulation_config_for_osc_image(base, series.images[0])
    )
    prepared = json.loads(session.prepared_json)
    original_selection = _selection(prepared["selection"])
    pack = json.loads(session.frozen_json)
    images = []
    for index, declared in enumerate(series.images):
        _stop(control)
        geometry = (
            shared
            if index == 0
            else rebind_configured_geometry_instrument(
                shared, simulation_config_for_osc_image(base, declared)
            )
        )
        context = build_geometry_only_ewald_context(geometry)
        frame = build_osc_angle_frame(
            mean_direction_lab=geometry.config.source.mean_direction_lab,
            instrument=context.instrument,
            sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
            revision=f"osc-geometry-angle-frame.{declared.image_id}.v1",
        )
        d = next(v for v in prepared["discoveries"] if v["image_id"] == declared.image_id)
        discovery = _saved_discovery(d)
        selected = next(
            v for v in original_selection.image_results if v.image_id == declared.image_id
        )
        if (
            discovery.discovery_hash != d["discovery_hash"]
            or discovery.geometry_context_hash
            != _discovery_geometry_hash(context.instrument, frame)
            or selected.context_hash != _indexing_context_hash(discovery, context, frame)
            or selected.detector_data_hash != discovery.detector_data_hash
            or selected.detector_mask_hash != discovery.detector_mask_hash
            or selected.detector_mask_revision != discovery.detector_mask_revision
        ):
            raise ValueError("sample discovery/indexing provenance disagrees with geometry")
        obs = next(v for v in pack["observations"] if v["image_id"] == declared.image_id)
        keys = tuple(
            IntegerLMarkerKey(**{**k, "representative_rod_hk": tuple(k["representative_rod_hk"])})
            for k in obs["keys"]
        )
        observations = IntegerLMarkerObservations(
            keys=keys,
            coordinates_px=obs["coordinates_px"],
            covariance_px2=obs["covariance_px2"],
            reference_wavelength_A=obs["reference_wavelength_A"],
        )
        images.append(
            IndexedGeometryImage(
                declared.image_id,
                math.radians(declared.axis_rotation_angles_deg[series.incidence_axis_index]),
                ExactTagGeometryModel(geometry),
                observations,
            )
        )
    return tuple(images)


def _points(images, corrections, calibration, delta, trims):
    points = []
    for image in images:
        prediction = image.predict_integer_l_tags(
            image.observations.keys,
            corrections,
            detector_calibration_corrections=calibration,
            incidence_angle_delta_rad=delta,
            incidence_angle_trim_rad=trims[image.image_id],
        )
        if not np.all(prediction.active_panel):
            raise ValueError("sample predictions do not all remain on the active panel")
        for key, observed, predicted in zip(
            image.observations.keys,
            image.observations.coordinates_px,
            prediction.coordinates_px,
            strict=True,
        ):
            points.append(
                {
                    "image_id": image.image_id,
                    "observation_id": observation_id(image.image_id, asdict(key)),
                    "key": asdict(key),
                    "observed_px": observed.tolist(),
                    "predicted_px": predicted.tolist(),
                    "residual_px": (predicted - observed).tolist(),
                }
            )
    return points


def _fit_from_document(data):
    from rasim_next.fitting.indexed_series import (
        DetectorCalibrationCorrections,
        IndexedGeometryFitResult,
        IndexedGeometryImageMetrics,
        SharedGeometryCorrections,
    )

    values = dict(data)
    values["corrections"] = SharedGeometryCorrections(**values["corrections"])
    values["detector_calibration_corrections"] = DetectorCalibrationCorrections(
        **values["detector_calibration_corrections"]
    )
    values["per_image"] = tuple(IndexedGeometryImageMetrics(**v) for v in values["per_image"])
    for name in (
        "image_ids",
        "fitted_parameter_names",
        "fixed_parameter_names",
        "fitted_detector_calibration_parameter_names",
        "fixed_detector_calibration_parameter_names",
    ):
        values[name] = tuple(values[name])
    return IndexedGeometryFitResult(**values)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_sample_record(record, session):
    """Admit saved integrity and recorded scope; live predictions remain a separate gate."""

    if (
        record["schema"] != "slate.sample-result.v1"
        or record["sha256"] != payload_hash({k: v for k, v in record.items() if k != "sha256"})
        or record["session_id"] != str(session.session_id)
    ):
        raise ValueError("sample result structure/identity/hash mismatch")
    if (
        record["dataset_qualified"] is not False
        or record["parameter_precision_qualified"] is not False
        or record["downstream_admitted"] is not False
    ):
        raise ValueError(
            "standalone sample fit cannot claim missing dataset/precision/root/outer/heldout qualification"
        )
    if type(record["name"]) is not str or not record["name"].strip() or len(record["name"]) > 256:
        raise ValueError("sample result name must be nonempty and at most 256 characters")
    launch = record["launch"]
    snapshot = SampleSession(
        session.session_id,
        encoded(launch["inputs"]),
        encoded(launch["controls"]),
        prepared_json=encoded(launch["prepared"]),
        exclusions=tuple(tuple(v) for v in launch["pack"]["review"]),
        frozen_json=encoded(launch["pack"]),
    )
    if record["launch_sha256"] != snapshot.launch_sha256:
        raise ValueError("sample result launch hash mismatch")
    if freeze_sample(snapshot).frozen_json != snapshot.frozen_json:
        raise ValueError("sample historical launch differs from its canonical reviewed pack")
    inputs = launch["inputs"]
    _validate_input_bindings(inputs)
    discoveries = launch["prepared"]["discoveries"]
    if [v["image_id"] for v in discoveries] != [v["image_id"] for v in inputs["images"]]:
        raise ValueError("saved sample discoveries differ from the launch image roster")
    for document in discoveries:
        _saved_discovery(document)
    image_ids = tuple(sorted(v["image_id"] for v in inputs["images"]))
    args = _fit_arguments(launch["controls"], len(image_ids))
    if record["outcome"] == "unavailable":
        if (
            record["fit"] is not None
            or not record["unavailable_reason"]
            or record["points"]
            or record["objective_residual"]
        ):
            raise ValueError("unavailable sample record cannot carry a fitted outcome")
        return record
    if record["outcome"] != "candidate" or record["unavailable_reason"] is not None:
        raise ValueError("unsupported sample result outcome")
    fit = _fit_from_document(record["fit"])
    if encoded(_json_value(asdict(fit))) != encoded(record["fit"]):
        raise ValueError("sample fit record is not the canonical owner result representation")
    # This uses the core result's existing structural/rank/condition predicate.
    if (
        fit.image_ids != image_ids
        or fit.fitted_parameter_names != args["fitted_parameter_names"]
        or fit.fitted_detector_calibration_parameter_names
        != args["fitted_detector_calibration_parameter_names"]
        or fit.incidence_angle_delta_fitted != (args["incidence_angle_delta_bounds"] is not None)
        or fit.incidence_angle_trim_fitted
        != (args["initial_incidence_angle_trim_contrast_rad"] is not None)
    ):
        raise ValueError("sample result fitted scopes disagree with its launch")
    controls = launch["controls"]
    fitted_values = (
        list(fit.corrections.as_array())
        + list(fit.detector_calibration_corrections.as_array())
        + [fit.incidence_angle_delta_rad]
        + list(
            fit.incidence_angle_trim_contrast_rad
            if fit.incidence_angle_trim_fitted
            else np.zeros(len(image_ids) - 1)
        )
    )
    for i, name in enumerate(controls["names"]):
        if not controls["lower"][i] <= fitted_values[i] <= controls["upper"][i] or (
            name not in controls["fitted"] and fitted_values[i] != controls["initial"][i]
        ):
            raise ValueError("sample fitted values violate recorded bounds/fixed references")
    if fit.incidence_angle_trim_fitted:
        from rasim_next.fitting.indexed_series import zero_sum_helmert_basis

        if (
            not np.array_equal(
                zero_sum_helmert_basis(len(image_ids)) @ fit.incidence_angle_trim_contrast_rad,
                fit.incidence_angle_trim_by_image_id_rad,
            )
            or fit.incidence_angle_trim_prior_sigma_rad
            != args["incidence_angle_trim_prior_sigma_rad"]
            or fit.incidence_angle_trim_contrast_half_span_rad
            != args["incidence_angle_trim_contrast_half_span_rad"]
        ):
            raise ValueError("sample trims disagree with the Helmert basis/launch")
    expected_points = [
        (obs["image_id"], identity, key, xy)
        for obs in launch["pack"]["observations"]
        for identity, key, xy in zip(
            obs["observation_ids"], obs["keys"], obs["coordinates_px"], strict=True
        )
    ]
    actual_points = [
        (p["image_id"], p["observation_id"], p["key"], p["observed_px"]) for p in record["points"]
    ]
    if actual_points != expected_points:
        raise ValueError("sample historical points disagree with frozen observations")
    for point in record["points"]:
        values = np.asarray(
            [point[n] for n in ("observed_px", "predicted_px", "residual_px")], dtype=np.float64
        )
        if (
            values.shape != (3, 2)
            or not np.all(np.isfinite(values))
            or not np.array_equal(values[1] - values[0], values[2])
        ):
            raise ValueError("sample historical point coordinates/residuals are inconsistent")
    counts = {obs["image_id"]: len(obs["keys"]) for obs in launch["pack"]["observations"]}
    if any(metric.site_count != counts[metric.image_id] for metric in fit.per_image):
        raise ValueError("sample historical metric counts disagree with frozen observations")
    residual = np.asarray(record["objective_residual"], dtype=np.float64)
    expected_count = sum(2 * v.site_count + v.chord_count for v in fit.per_image)
    if residual.shape != (expected_count,) or not np.all(np.isfinite(residual)):
        raise ValueError("sample historical objective residual shape/values are invalid")
    return record


def _validate_input_bindings(inputs):
    if inputs["schema"] != "slate.sample-inputs.v1":
        raise ValueError("unsupported saved sample input schema")
    files = inputs["files"]
    paths = [v["path"] for v in files]
    required = {inputs[k] for k in ("manifest_path", "configuration_path", "cif_path")} | {
        v["path"] for v in inputs["images"]
    }
    if len(paths) != len(set(paths)) or set(paths) != required:
        raise ValueError("saved sample input file bindings are incomplete or duplicated")
    for row in files:
        if (
            not isinstance(row["path"], str)
            or not 0 < len(row["path"]) <= 4096
            or not Path(row["path"]).is_absolute()
        ):
            raise ValueError("saved sample inputs need bounded absolute paths")
        digest = row["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("saved sample inputs need SHA-256 identities")
    for image in inputs["images"]:
        mask = mask_from_document(image["mask"])
        if mask.source_sha256 != image["decoded_sha256"] or list(mask.shape) != image["shape_rc"]:
            raise ValueError("saved sample source/native mask binding is inconsistent")


def validate_sample_result(record, session, control):
    """Require full original live geometry and prediction checks before active use/export."""
    from rasim_next.fitting.indexed_series import (
        evaluate_indexed_geometry_series_metrics,
        evaluate_indexed_geometry_series_residual,
    )

    validate_sample_record(record, session)
    launch = record["launch"]
    snapshot = SampleSession(
        session.session_id,
        encoded(launch["inputs"]),
        encoded(launch["controls"]),
        prepared_json=encoded(launch["prepared"]),
        exclusions=tuple(tuple(v) for v in launch["pack"]["review"]),
        frozen_json=encoded(launch["pack"]),
    )
    images = _images(snapshot, control)
    if record["outcome"] == "unavailable":
        return record
    fit = _fit_from_document(record["fit"])
    trims = dict(zip(fit.image_ids, fit.incidence_angle_trim_by_image_id_rad, strict=True))
    kwargs = dict(
        detector_calibration_corrections=fit.detector_calibration_corrections,
        incidence_angle_delta_rad=fit.incidence_angle_delta_rad,
        incidence_angle_trim_by_image_id_rad=trims,
    )
    raw = evaluate_indexed_geometry_series_residual(images, fit.corrections, **kwargs)
    metrics = evaluate_indexed_geometry_series_metrics(images, fit.corrections, **kwargs)
    if (
        raw.tolist() != record["objective_residual"]
        or encoded(
            _points(
                images,
                fit.corrections,
                fit.detector_calibration_corrections,
                fit.incidence_angle_delta_rad,
                trims,
            )
        )
        != encoded(record["points"])
        or encoded([asdict(v) for v in metrics.per_image]) != encoded(record["fit"]["per_image"])
        or metrics.site_rms_px != fit.training_site_rms_px
        or metrics.site_max_px != fit.training_site_max_px
        or metrics.chord_angle_rms_rad != fit.training_chord_angle_rms_rad
    ):
        raise ValueError(
            "sample record coordinates/residuals/metrics disagree with the canonical owner"
        )
    return record


def sample_input_checks(session, control):
    """Keep unavailable live files distinct from malformed saved records and model failures."""
    _validate_input_bindings(json.loads(session.inputs_json))
    records = [json.loads(text) for text in session.results_json]
    for record in records:
        validate_sample_record(record, session)
    inputs_by_hash = {
        payload_hash(json.loads(session.inputs_json)): json.loads(session.inputs_json)
    }
    inputs_by_hash.update(
        (payload_hash(v["launch"]["inputs"]), v["launch"]["inputs"]) for v in records
    )
    checks = []
    for identity, inputs in inputs_by_hash.items():
        _stop(control)
        try:
            _current(inputs, control)
            for record in records:
                if payload_hash(record["launch"]["inputs"]) == identity:
                    validate_sample_result(record, session, control)
        except SampleInputsUnavailable as exc:
            checks.append((identity, str(exc)))
        else:
            checks.append((identity, ""))
    return tuple(checks)


def sample_protected_paths(session):
    files = list(json.loads(session.inputs_json)["files"])
    for text in session.results_json:
        files.extend(json.loads(text)["launch"]["inputs"]["files"])
    return [
        Path(p)
        for p in dict.fromkeys([v["path"] for v in files] + [p for p, _sha in session.exports])
    ]


def _export(request, session, control):
    inputs = json.loads(session.inputs_json)
    protected = sample_protected_paths(session)
    path = Path(request["path"]).resolve(strict=False)
    operation = request["operation"]
    if operation == "export_observations":
        if not session.frozen_json:
            raise ValueError("Freeze a reviewed pack before exporting observations")
        outputs = [
            (
                path,
                encoded(
                    {
                        "schema": "slate.sample-observations-export.v1",
                        "inputs": inputs,
                        "pack": json.loads(session.frozen_json),
                    }
                ).encode(),
            )
        ]
    else:
        record = json.loads(request["record"])
        if not any(json.loads(v)["sha256"] == record["sha256"] for v in session.results_json):
            raise ValueError("export requires a retained sample result")
        validate_sample_result(record, session, control)
        outputs = [(path, encoded(record).encode())]
        if operation == "export_figure":
            from io import BytesIO

            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            figure = Figure(figsize=(9, 7), layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.subplots(2, 1, height_ratios=(3, 1))
            for image in inputs["images"]:
                rows = [v for v in record["points"] if v["image_id"] == image["image_id"]]
                if not rows:
                    continue
                observed = np.asarray([v["observed_px"] for v in rows])
                predicted = np.asarray([v["predicted_px"] for v in rows])
                axes[0].scatter(
                    observed[:, 0], observed[:, 1], s=12, label=image["image_id"] + " observed"
                )
                axes[0].scatter(
                    predicted[:, 0],
                    predicted[:, 1],
                    s=18,
                    marker="+",
                    label=image["image_id"] + " predicted",
                )
                axes[1].plot(
                    np.linalg.norm(predicted - observed, axis=1), ".", label=image["image_id"]
                )
            axes[0].set(
                xlabel="native column_px",
                ylabel="native row_px",
                title=record["name"] + " · unqualified " + record["outcome"],
            )
            axes[0].invert_yaxis()
            axes[0].set_aspect("equal")
            if record["points"]:
                axes[0].legend(fontsize="small")
                axes[1].legend(fontsize="small")
            else:
                axes[0].text(
                    0.5,
                    0.5,
                    record["unavailable_reason"],
                    ha="center",
                    wrap=True,
                    transform=axes[0].transAxes,
                )
            axes[1].set(xlabel="per-image frozen observation index", ylabel="site distance (px)")
            stream = BytesIO()
            figure.savefig(stream, format="png", dpi=140)
            outputs = [
                (path.with_suffix(path.suffix + ".values.json"), encoded(record).encode()),
                (path, stream.getvalue()),
            ]
    if len(session.exports) + len(outputs) > 16:
        raise ValueError("sample export reference cap reached")
    for target, _raw in outputs:
        _external(target, protected)
    references = []
    for target, raw in outputs:
        _publish_bytes(target, raw, control)
        if target.read_bytes() != raw:
            raise ValueError("sample export differs on readback")
        references.append((str(target), _hash_file(target, control)))
    return replace(session, exports=(*session.exports, *references))


def sample_work(argument, control):
    from threadpoolctl import threadpool_limits

    from rasim_next.fitting.geometry import GeometryPredictionError, GeometryRankError
    from rasim_next.fitting.indexed_series import (
        evaluate_indexed_geometry_series_residual,
        fit_indexed_geometry_series,
    )
    from rasim_next.pipeline.configured_simulation import load_simulation_config
    from rasim_next.selection.osc_series import index_osc_geometry_series, load_osc_geometry_series

    request = json.loads(argument)
    operation = request["operation"]
    resources = request.get("resources")
    if resources is None or resources != sample_work_budget(
        resources["other_cpu_bytes"], resources["other_gpu_bytes"]
    ):
        raise ValueError("sample worker requires an admitted resource snapshot")

    def checkpoint(message):
        _stop(control)
        control.report(message)

    with threadpool_limits(limits=1):
        _stop(control)
        presented = None
        checks = ()
        if operation == "revalidate":
            session = sample_session_from_document(request["session"])
            checks = sample_input_checks(session, control)
        elif operation == "load":
            manifest = Path(request["manifest_path"]).resolve(strict=True)
            manifest_hash = _hash_file(manifest, control)
            series = load_osc_geometry_series(manifest)
            if not 2 <= len(series.images) <= 8:
                raise ValueError(
                    "sample-only geometry requires a complete supported multi-image series (2-8 images)"
                )
            config_hash = _hash_file(series.config_path, control)
            config = load_simulation_config(series.config_path)
            if len(config.instrument.axis_rotations) != 1 or series.incidence_axis_index != 0:
                raise ValueError(
                    "current sample owner requires one configured incidence axis at index 0"
                )
            files = [
                {"path": str(manifest), "sha256": manifest_hash},
                {"path": str(series.config_path), "sha256": config_hash},
                {
                    "path": str(config.material.cif_path),
                    "sha256": _hash_file(config.material.cif_path, control),
                },
            ]
            images = []
            for declared in series.images:
                checkpoint("Admitting " + declared.image_id)
                image, raw_sha = _osc(declared.osc_path, control)
                mask = request["project_masks"].get(str(declared.osc_path))
                if mask is None:
                    mask = mask_document(
                        NativeMask(image.decoded_sha256, tuple(image.detector_native_counts.shape))
                    )
                state = mask_from_document(mask)
                if (
                    state.source_sha256 != image.decoded_sha256
                    or state.shape != image.detector_native_counts.shape
                ):
                    raise ValueError(
                        "sample fitting mask does not match decoded source/native shape"
                    )
                files.append({"path": str(declared.osc_path), "sha256": raw_sha})
                images.append(
                    {
                        "image_id": declared.image_id,
                        "path": str(declared.osc_path),
                        "axis_rotation_angles_deg": list(declared.axis_rotation_angles_deg),
                        "decoded_sha256": image.decoded_sha256,
                        "shape_rc": list(image.detector_native_counts.shape),
                        "mask": mask,
                    }
                )
            inputs = {
                "schema": "slate.sample-inputs.v1",
                "manifest_path": str(manifest),
                "configuration_path": str(series.config_path),
                "cif_path": str(config.material.cif_path),
                "incidence_axis_index": series.incidence_axis_index,
                "qualification_profile": series.qualification_profile,
                "files": files,
                "images": images,
                "orientation": "accepted clockwise conversion once at OSC I/O",
                "fixed_geometry_material": "hash-bound configuration/CIF; no experiment adoption",
                "fixed_instrument": _json_value(asdict(config.instrument)),
                "nominal_mean_wavelength_A": config.source.mean_wavelength_A,
                "nominal_mean_direction_lab": list(config.source.mean_direction_lab),
                "phase_id": config.material.phase_id,
                "source_policy": "nominal_source_center.zero_divergence.mean_wavelength.v1",
                "cpu_workers": 1,
                "nested_blas_threads": 1,
            }
            previous = sample_session_from_document(request.get("previous_session"))
            session = SampleSession(uuid4(), encoded(inputs), encoded(_controls(len(images))))
            if previous is not None:
                same_roster = [
                    v["image_id"] for v in json.loads(previous.inputs_json)["images"]
                ] == [v["image_id"] for v in images]
                session = replace(
                    session,
                    session_id=previous.session_id,
                    controls_json=previous.controls_json if same_roster else session.controls_json,
                    results_json=previous.results_json,
                    selected_result_id=previous.selected_result_id,
                    exports=previous.exports,
                    revision=previous.revision + 1,
                )
        else:
            session = sample_session_from_document(request["session"])
            inputs = json.loads(session.inputs_json)
            _current(inputs, control)
            if operation == "prepare":
                counts = {}
                masks = {}
                revisions = {}
                for row in inputs["images"]:
                    checkpoint("Reading " + row["image_id"])
                    image, raw_sha = _osc(row["path"], control, row["decoded_sha256"])
                    counts[row["image_id"]] = image.detector_native_counts
                    mask = mask_from_document(row["mask"])
                    inclusion = np.ones(mask.shape, dtype=bool)
                    for r, start, stop, _reason in mask.spans:
                        inclusion[r, start:stop] = False
                    inclusion.setflags(write=False)
                    masks[row["image_id"]] = inclusion
                    revisions[row["image_id"]] = (
                        f"slate.native-mask.{mask.revision}.{payload_hash(row['mask'])}"
                    )
                run = index_osc_geometry_series(
                    load_osc_geometry_series(inputs["manifest_path"]),
                    detector_native_counts_by_image_id=counts,
                    detector_valid_mask_by_image_id=masks,
                    detector_mask_revision_by_image_id=revisions,
                    checkpoint=checkpoint,
                )
                admitted = []
                if run.indexed_images is not None:
                    admitted = [
                        observation_id(v.image_id, asdict(k))
                        for v in run.indexed_images
                        for k in v.observations.keys
                    ]
                prepared = {
                    "admitted_observation_ids": admitted,
                    "selection": _selection_document(run.selection),
                    "discoveries": [_json_value(asdict(v)) for v in run.discoveries],
                    "geometry_setup_seconds": run.geometry_setup_seconds,
                    "elapsed_seconds": run.elapsed_seconds,
                    "availability": "ready for review/freeze"
                    if run.indexed_images
                    else "unavailable: one or more images have no accepted visible marker observations",
                }
                session = replace(
                    session,
                    prepared_json=encoded(prepared),
                    frozen_json=None,
                    exclusions=(),
                    revision=session.revision + 1,
                )
            elif operation == "freeze":
                session = freeze_sample(session)
            elif operation == "fit":
                if len(session.results_json) >= 4:
                    raise ValueError(
                        "archive/export and explicitly remove an old result before fitting; history cap is four"
                    )
                images = _images(session, control)
                controls = json.loads(session.controls_json)
                args = _fit_arguments(controls, len(images))
                record = {
                    "schema": "slate.sample-result.v1",
                    "result_id": str(uuid4()),
                    "session_id": str(session.session_id),
                    "name": request["name"],
                    "launch_sha256": session.launch_sha256,
                    "launch": {
                        "inputs": inputs,
                        "controls": controls,
                        "prepared": json.loads(session.prepared_json),
                        "pack": json.loads(session.frozen_json),
                    },
                    "outcome": "candidate",
                    "fit": None,
                    "unavailable_reason": None,
                    "objective_residual": [],
                    "points": [],
                    "dataset_qualified": False,
                    "parameter_precision_qualified": False,
                    "downstream_admitted": False,
                    "qualification_reason": "Nominal core fit; root/outer/heldout audits and dataset/parameter-precision qualification are unavailable",
                    "uncertainty_reason": "Existing indexed-series owner reports scaled singular values and weakest direction; calibrated covariance/standard errors are unavailable",
                }
                try:
                    fit = fit_indexed_geometry_series(images, **args, checkpoint=checkpoint)
                    trims = dict(
                        zip(fit.image_ids, fit.incidence_angle_trim_by_image_id_rad, strict=True)
                    )
                    record["fit"] = _json_value(asdict(fit))
                    record["points"] = _points(
                        images,
                        fit.corrections,
                        fit.detector_calibration_corrections,
                        fit.incidence_angle_delta_rad,
                        trims,
                    )
                    record["objective_residual"] = evaluate_indexed_geometry_series_residual(
                        images,
                        fit.corrections,
                        detector_calibration_corrections=fit.detector_calibration_corrections,
                        incidence_angle_delta_rad=fit.incidence_angle_delta_rad,
                        incidence_angle_trim_by_image_id_rad=trims,
                    ).tolist()
                except (GeometryRankError, GeometryPredictionError) as exc:
                    record["outcome"] = "unavailable"
                    record["unavailable_reason"] = str(exc)
                record["sha256"] = payload_hash(record)
                session = replace(session, results_json=(*session.results_json, encoded(record)))
                presented = record["result_id"]
            elif operation == "import":
                path = Path(request["path"])
                if path.stat().st_size > 2 * 1024**2:
                    raise ValueError("sample imported record exceeds 2 MiB")
                with path.open("rb") as stream:
                    raw = stream.read(2 * 1024**2 + 1)
                if len(raw) > 2 * 1024**2:
                    raise ValueError("sample imported record exceeds 2 MiB")
                record = json.loads(raw)
                validate_sample_result(record, session, control)
                if record["launch"]["inputs"] != inputs or record["launch"]["pack"] != json.loads(
                    session.frozen_json or "null"
                ):
                    raise ValueError(
                        "imported sample result differs from the current reviewed inputs/pack"
                    )
                session = replace(session, results_json=(*session.results_json, encoded(record)))
                presented = record["result_id"]
            elif operation in ("export", "export_figure", "export_observations"):
                session = _export(request, session, control)
            else:
                raise ValueError("unknown sample operation")
        if operation != "revalidate":
            _current(json.loads(session.inputs_json), control)
            checks = ((payload_hash(json.loads(session.inputs_json)), ""),)
        _stop(control)
        result = SampleWorkResult(operation, session, presented, checks)
        return JobResult(result, result.nbytes)
