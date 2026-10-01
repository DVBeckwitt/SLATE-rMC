"""Joint geometry worker: bind frozen inputs, call the owner, publish exact records."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from uuid import uuid4

import numpy as np
from hbn_io import _geometry, _hash_file, _inputs_current, _json_finite, _observations
from hbn_state import hbn_session_from_document
from job_lifecycle import JobResult
from joint_state import joint_session_from_document, validate_joint_record
from sample_io import _current, _images, _validate_input_bindings, observation_id
from sample_state import encoded, payload_hash, sample_session_from_document
from simulation_io import (
    MAX_SIMULATION_CPU_BYTES,
    MAX_SIMULATION_GPU_BYTES,
    _external,
    _publish_bytes,
    _stop,
)


@dataclass(frozen=True, slots=True)
class JointWorkResult:
    operation: str
    session: object
    presented_result_id: str | None = None
    detail: str = ""

    @property
    def nbytes(self):
        return self.session.nbytes + 8192


def joint_work_budget(other_cpu_bytes=0, other_gpu_bytes=0):
    # Frozen geometry only: no decoded images or discovery workspaces.
    working = 256 * 1024**2
    if (
        type(other_cpu_bytes) is not int
        or type(other_gpu_bytes) is not int
        or min(other_cpu_bytes, other_gpu_bytes) < 0
        or working + other_cpu_bytes > MAX_SIMULATION_CPU_BYTES
        or other_gpu_bytes > MAX_SIMULATION_GPU_BYTES
    ):
        raise ValueError("joint geometry exceeds the shared 2 GiB CPU / 512 MiB GPU admission")
    return {
        "working_cpu_bytes": working,
        "other_cpu_bytes": other_cpu_bytes,
        "other_gpu_bytes": other_gpu_bytes,
    }


def joint_protected_paths(session):
    paths = []
    launches = [session.launch] + [json.loads(v)["launch"] for v in session.results_json]
    for launch in launches:
        if launch is None:
            continue
        for group, capture in launch["captures"].items():
            inputs = json.loads(capture["session"]["inputs_json"])
            if group == "hbn":
                paths.extend(
                    inputs[k]
                    for k in ("source_path", "dark_path", "configuration_path", "cif_path")
                )
            else:
                paths.extend(v["path"] for v in inputs["files"])
    paths.extend(p for p, _ in session.exports)
    paths.extend(
        json.loads(v)["report_origin"]["path"]
        for v in session.results_json
        if json.loads(v)["report_origin"] is not None
    )
    return [Path(p) for p in dict.fromkeys(paths)]


def _arguments(session, control):
    from rasim_next.fitting.hbn import HbnDetectorCalibration
    from rasim_next.fitting.joint_geometry import JointGeometryBounds, JointGeometryState

    captures = session.launch["captures"]
    if set(captures) != {"hbn", "bi2se3", "bi2te3"} or session.controls_json is None:
        raise ValueError("capture frozen hBN, Bi2Se3 and Bi2Te3 and commit joint controls first")
    hbn = hbn_session_from_document(captures["hbn"]["session"])
    inputs = json.loads(hbn.inputs_json)
    _inputs_current(inputs, control)
    obs = _observations(json.loads(hbn.frozen_json))
    calibration = HbnDetectorCalibration(**captures["hbn"]["calibration_record"]["calibration"])
    groups = {}
    for group in ("bi2se3", "bi2te3"):
        _stop(control)
        ss = sample_session_from_document(captures[group]["session"])
        groups[group] = _images(ss, control)
    geometry = _geometry(inputs)
    for image in (*groups["bi2se3"], *groups["bi2te3"]):
        instrument = image.model.instrument
        config = image.model.inputs.config
        if (
            not np.array_equal(
                instrument.lab_from_detector.rotation, geometry["base_detector_rotation"]
            )
            or not np.array_equal(config.source.mean_direction_lab, geometry["beam_direction_lab"])
            or instrument.detector_column_pitch_m != geometry["detector_column_pitch_m"]
            or instrument.detector_row_pitch_m != geometry["detector_row_pitch_m"]
        ):
            raise ValueError("joint captures disagree on shared detector/source references")
    controls = json.loads(session.controls_json)
    return dict(
        hbn_observations=obs,
        hbn_calibration=calibration,
        bi2se3_images=groups["bi2se3"],
        bi2te3_images=groups["bi2te3"],
        pbi2_y1_images=(),
        pbi2_y2_images=(),
        base_detector_rotation=np.asarray(geometry["base_detector_rotation"], dtype=np.float64),
        initial=JointGeometryState.from_array(controls["initial"]),
        bounds=JointGeometryBounds(
            JointGeometryState.from_array(controls["lower"]),
            JointGeometryState.from_array(controls["upper"]),
        ),
    )


def _fit(session, control, name):
    from rasim_next.fitting.hbn import evaluate_hbn_residual_px, hbn_ring_curves_px
    from rasim_next.fitting.joint_geometry import _predict_image, fit_joint_geometry
    from rasim_next.fitting.joint_geometry_report import (
        joint_geometry_report,
        joint_geometry_static,
    )

    def checkpoint(phase):
        _stop(control)
        control.report("Joint: " + phase)

    checkpoint("binding frozen observations")
    args = _arguments(session, control)
    result = fit_joint_geometry(**args, checkpoint=checkpoint)
    se = args["bi2se3_images"][0]
    report = joint_geometry_report(
        result,
        static=joint_geometry_static(se.model.inputs.config, se.model.instrument),
        hbn_calibration=args["hbn_calibration"],
        bi2se3_images=args["bi2se3_images"],
        bi2te3_images=args["bi2te3_images"],
    )
    points = []
    for group in ("bi2se3", "bi2te3"):
        for image in args[group + "_images"]:
            checkpoint("saved sites " + image.image_id)
            _, errors, _ = _predict_image(
                group, image, result.state, base_detector_rotation=args["base_detector_rotation"]
            )
            for key, observed, error in zip(
                image.observations.keys, image.observations.coordinates_px, errors, strict=True
            ):
                points.append(
                    {
                        "specimen_id": group,
                        "image_id": image.image_id,
                        "observation_id": observation_id(image.image_id, asdict(key)),
                        "key": asdict(key),
                        "observed_px": observed.tolist(),
                        "predicted_px": (observed + error).tolist(),
                        "residual_px": error.tolist(),
                    }
                )
    hbn = session.launch["captures"]["hbn"]
    hs = hbn_session_from_document(hbn["session"])
    values = result.state.as_array()
    hbn_residual = evaluate_hbn_residual_px(
        values[[0, 1, 2, 3, 20]], args["hbn_observations"], **_geometry(json.loads(hs.inputs_json))
    )
    record = {
        "schema": "slate.joint-result.v1",
        "result_id": str(uuid4()),
        "name": name,
        "launch": session.launch,
        "launch_sha256": session.launch_sha256,
        "report": _json_finite(report),
        "points": points,
        "hbn_residual_px": hbn_residual.tolist(),
        "hbn_curves_px": _json_finite(
            [
                curve.tolist()
                for curve in hbn_ring_curves_px(
                    values[[0, 1, 2, 3, 20]],
                    args["hbn_observations"].two_theta_rad,
                    **_geometry(json.loads(hs.inputs_json)),
                )
            ]
        ),
        "report_origin": None,
    }
    record["sha256"] = payload_hash(record)
    validate_joint_record(record)
    return record


def _handoff_predecessors(record, request, control):
    launch = record["launch"]
    if launch is None:
        raise ValueError(
            "New handoff unavailable: original predecessor provenance is missing; export the exact result or verify an existing handoff"
        )
    specimen = request["specimen_id"]
    if specimen not in ("bi2se3", "bi2te3"):
        raise ValueError("joint handoff specimen must be a captured Bi2Se3 or Bi2Te3 group")
    captures = launch["captures"]
    inputs = {
        group: json.loads(capture["session"]["inputs_json"]) for group, capture in captures.items()
    }
    selected = inputs[specimen]
    if Path(request["manifest_path"]).resolve(strict=True) != Path(
        selected["manifest_path"]
    ) or Path(request["detector_base_config_path"]).resolve(strict=True) != Path(
        inputs["bi2se3"]["configuration_path"]
    ):
        raise ValueError("handoff paths differ from the chosen result's original predecessors")
    expected = {}
    for group, values in inputs.items():
        if group == "hbn":
            _inputs_current(values, control)
            files = [
                {"path": values[path], "sha256": values[sha]}
                for path, sha in (
                    ("source_path", "source_file_sha256"),
                    ("dark_path", "dark_file_sha256"),
                    ("configuration_path", "configuration_sha256"),
                    ("cif_path", "cif_sha256"),
                )
            ]
        else:
            _validate_input_bindings(values)
            _current(values, control)
            files = values["files"]
        for row in files:
            path = str(Path(row["path"]).resolve(strict=True))
            if path in expected and expected[path] != row["sha256"]:
                raise ValueError("joint captured predecessors disagree on a file identity")
            expected[path] = row["sha256"]
    return expected, selected


def joint_work(argument, control):
    from threadpoolctl import threadpool_limits

    from rasim_next.fitting.joint_geometry_handoff import (
        joint_geometry_handoff_document,
        load_joint_geometry_handoff,
    )
    from rasim_next.fitting.joint_geometry_report import validate_joint_geometry_report

    request = json.loads(argument)
    operation = request["operation"]
    resources = request.get("resources")
    if resources is None or resources != joint_work_budget(
        resources["other_cpu_bytes"], resources["other_gpu_bytes"]
    ):
        raise ValueError("joint worker requires the shared resource admission")
    session = joint_session_from_document(request["session"])
    presented = None
    detail = ""
    with threadpool_limits(limits=1):
        _stop(control)
        if operation == "fit":
            if len(session.results_json) >= 4:
                raise ValueError(
                    "joint history is full; remove an unselected historical result first"
                )
            record = _fit(session, control, request["name"])
            presented = record["result_id"]
            session = replace(
                session,
                results_json=(*session.results_json, encoded(record)),
                revision=session.revision + 1,
            )
        elif operation == "import_result":
            path = Path(request["path"]).resolve(strict=True)
            if path.stat().st_size > 6 * 1024**2:
                raise ValueError("joint import exceeds 6 MiB")
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema") == "slate.joint-result.v1":
                record = validate_joint_record(data)
            else:
                validate_joint_geometry_report(data)
                record = {
                    "schema": "slate.joint-result.v1",
                    "result_id": str(uuid4()),
                    "name": request["name"],
                    "launch": None,
                    "launch_sha256": None,
                    "report": data,
                    "points": [],
                    "hbn_residual_px": [],
                    "report_origin": {"path": str(path), "sha256": _hash_file(path, control)},
                }
                record["sha256"] = payload_hash(record)
            if len(session.results_json) >= 4 or any(
                json.loads(v)["result_id"] == record["result_id"] for v in session.results_json
            ):
                raise ValueError("joint history is full or this result is already retained")
            session = replace(
                session,
                results_json=(*session.results_json, encoded(record)),
                revision=session.revision + 1,
            )
            presented = record["result_id"]
        elif operation in ("export_result", "save_handoff"):
            record = validate_joint_record(json.loads(request["record"]))
            if not any(json.loads(v)["sha256"] == record["sha256"] for v in session.results_json):
                raise ValueError("export requires a retained immutable joint result")
            path = Path(request["path"]).resolve(strict=False)
            _external(path, protected=joint_protected_paths(session))
            if len(session.exports) >= 15:
                raise ValueError("joint export reference history is full")
            if operation == "export_result":
                outputs = [(path, encoded(record).encode())]
                _publish_bytes(path, outputs[0][1], control)
            else:
                if not record["report"]["confidence_qualified"]:
                    raise ValueError("joint handoff requires a qualified report")
                report_path = path.with_suffix(".joint-report.json")
                _external(report_path, protected=joint_protected_paths(session))
                expected, inputs = _handoff_predecessors(record, request, control)
                report_bytes = encoded(record["report"]).encode()
                document = joint_geometry_handoff_document(
                    report_path=report_path,
                    report_bytes=report_bytes,
                    geometry_manifest_path=Path(request["manifest_path"]),
                    detector_base_config_path=Path(request["detector_base_config_path"]),
                    specimen_id=request["specimen_id"],
                    expected_file_hashes=expected,
                )
                if tuple(document["image_ids"]) != tuple(
                    v["image_id"] for v in inputs["images"]
                ) or tuple(document["commanded_incidence_angles_rad"]) != tuple(
                    math.radians(v["axis_rotation_angles_deg"][inputs["incidence_axis_index"]])
                    for v in inputs["images"]
                ):
                    raise ValueError(
                        "handoff image order or commanded angles differ from the chosen result"
                    )
                _handoff_predecessors(record, request, control)
                outputs = [(report_path, report_bytes), (path, encoded(document).encode())]
                written = []
                try:
                    for output, raw in outputs:
                        _publish_bytes(output, raw, control)
                        written.append((output, raw))
                    _stop(control)
                except BaseException:
                    for output, raw in written:
                        if output.is_file() and output.read_bytes() == raw:
                            output.unlink()
                    raise
            session = replace(
                session,
                exports=(
                    *session.exports,
                    *tuple((str(p), hashlib.sha256(raw).hexdigest()) for p, raw in outputs),
                ),
            )
            detail = (
                "Exact joint result exported"
                if operation == "export_result"
                else "Qualified hash-bound geometry handoff saved; no experiment adoption"
            )
        elif operation == "reload_handoff":
            from archive_storage import storage_files, stored_path

            storage = storage_files(request.get("storage_json", "{}"))
            path = stored_path(request["path"], storage).resolve(strict=True)
            handoff = load_joint_geometry_handoff(
                path,
                stored_paths={Path(r["original"]).resolve(): Path(r["stored"]) for r in storage},
            )
            if not any(p == str(path) for p, _ in session.exports):
                if len(session.exports) >= 16:
                    raise ValueError("joint export reference history is full")
                session = replace(
                    session, exports=(*session.exports, (str(path), _hash_file(path, control)))
                )
            detail = (
                "Verified joint handoff "
                + handoff.position.artifact_revision
                + "; serialized predecessor bytes verified; missing original fit lineage is not reconstructed; no experiment adoption"
            )
        else:
            raise ValueError("unsupported joint operation")
        _stop(control)
        value = JointWorkResult(operation, session, presented, detail)
        return JobResult(value, value.nbytes)
