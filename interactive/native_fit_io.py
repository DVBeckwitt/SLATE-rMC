"""Prepared-data reads and draft validation on the existing shared worker; no execution."""

import hashlib
import json
import math
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import numpy as np
from archive_storage import storage_files, stored_path
from hbn_io import _hash_file
from job_lifecycle import JobResult
from native_fit_profiles import NativeProfiles
from native_fit_state import (
    NativeFitSession,
    description,
    document,
    native_fit_session_from_document,
)
from sample_state import encoded, payload_hash
from simulation_io import (
    MAX_SIMULATION_CPU_BYTES,
    MAX_SIMULATION_GPU_BYTES,
    _external,
    _publish_bytes,
    _stop,
)


def prepared_budget(other_cpu_bytes=0, other_gpu_bytes=0):
    working = 1024 * 1024**2
    if (
        type(other_cpu_bytes) is not int
        or type(other_gpu_bytes) is not int
        or min(other_cpu_bytes, other_gpu_bytes) < 0
        or working + other_cpu_bytes > MAX_SIMULATION_CPU_BYTES
        or other_gpu_bytes > MAX_SIMULATION_GPU_BYTES
    ):
        raise ValueError("Prepared inspection exceeds shared resource admission")
    return dict(
        working_cpu_bytes=working, other_cpu_bytes=other_cpu_bytes, other_gpu_bytes=other_gpu_bytes
    )


def prepared_paths(session):
    if session is None:
        return []
    return [
        Path(v["path"])
        for text in (session.current_json, *session.history_json)
        for v in json.loads(text)["inputs"]["files"]
    ] + [Path(p) for p, _ in session.exports]


def _identities(paths, control, storage=()):
    rows = []
    for kind, path in paths:
        _stop(control)
        original = Path(path).resolve()
        path = stored_path(original, storage).resolve(strict=True)
        if path.stat().st_size > 512 * 1024**2:
            raise ValueError("Prepared file exceeds 512 MiB")
        rows.append(
            dict(
                kind=kind,
                path=str(original),
                sha256=_hash_file(path, control),
                size=path.stat().st_size,
            )
        )
    return rows


def _check_files(inputs, control, storage=()):
    for row in inputs["files"]:
        _stop(control)
        path = stored_path(row["path"], storage, row["sha256"])
        if path.stat().st_size != row["size"] or _hash_file(path, control) != row["sha256"]:
            raise ValueError("Prepared predecessor changed: " + row["path"])


def _definition(physics, plan):
    from rasim_next.fitting import bi_joint, bi_native, native_instrument, native_search, pb_native

    if physics.structure.stacking_phases:
        model = pb_native.PbJointModel(pb_native.PbNativeStructureModel(physics))
    else:
        model = bi_joint.BiJointModel(bi_native.BiNativeStructureModel(physics))
    names, units = tuple(model.parameter_names), tuple(model.parameter_units)
    owners = ("specimen:" + physics.sample_id,) * len(names)
    if plan.get("fit_instrument") is True:
        native_instrument.NativeInstrumentModel(physics, plan["acquisition_id"])
        names += native_instrument.NATIVE_INSTRUMENT_PARAMETER_NAMES
        units += native_instrument.NativeInstrumentModel.parameter_units
        owners += (plan["acquisition_id"],) * 18
    capabilities = tuple(zip(owners, names, units, strict=True))
    revision = payload_hash(
        {
            m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (bi_joint, bi_native, native_instrument, native_search, pb_native)
        }
    )
    rows = []
    for p in plan["parameters"]:
        reason = ""
        try:
            native_search.FitParameter(**p)
        except TypeError as exc:
            reason = "Unsupported parameter definition fields: " + str(exc)
        if (p["owner"], p["name"], p["unit"]) not in capabilities:
            reason = "Name/unit/owner is unavailable in the current material model"
        rows.append({"owner": p["owner"], "name": p["name"], "unit": p["unit"], "reason": reason})
    settings_reasons = {}
    for stage in plan.get("stages", ()):
        if "method" in stage and stage["method"] not in ("trf", "slsqp"):
            settings_reasons[stage["name"]] = (
                "Stage method is unavailable in the current search owner"
            )
    if "method" in plan and plan["method"] not in ("trf", "slsqp"):
        settings_reasons["plan"] = "Plan method is unavailable in the current search owner"
    value = {
        "settings_reasons": settings_reasons,
        "engine_revision": revision,
        "model_id": type(model).__module__ + "." + type(model).__name__,
        "ordered_identities": [[p["owner"], p["name"]] for p in plan["parameters"]],
        "capabilities": rows,
        "declared_parameters": plan["parameters"],
        "canonical_order": [list(v) for v in capabilities],
        "domain_policy": "FitParameter declarations and search validation only; full coupled model/domain/gauge launch admission remains unavailable",
    }
    value["sha256"] = payload_hash(value)
    return value


def _validate_plan(plan, observations, definition):
    from rasim_next.fitting.native_search import (
        FitParameter,
        GaussianCalibration,
        validate_native_search_request,
    )

    if definition["settings_reasons"] or any(p["reason"] for p in definition["capabilities"]):
        return "Unsupported rows retained as inert data; full launch admission unavailable"
    parameters = tuple(FitParameter(**p) for p in plan["parameters"])
    calibration = tuple(GaussianCalibration(**v) for v in plan.get("calibration", ()))
    settings = {
        k: plan[k]
        for k in (
            "method",
            "maximum_iterations",
            "maximum_function_evaluations",
            "finite_difference_step",
        )
        if k in plan
    }
    validate_native_search_request(
        observations,
        parameters,
        plan["starts"],
        fixed_values=plan.get("fixed_parameters"),
        calibration=calibration,
        **settings,
    )
    names = tuple(p.name for p in parameters)
    fixed = plan.get("fixed_parameters", {})
    starts = np.asarray(plan["starts"])
    if any(np.any(starts[:, names.index(k)] != v) for k, v in fixed.items()):
        raise ValueError("Fixed controls must equal every start")
    stages = plan.get("stages", ())
    for stage in stages:
        active = stage["active_parameters"]
        if (
            not active
            or len(set(active)) != len(active)
            or set(active) - set(names)
            or set(active) & set(fixed)
        ):
            raise ValueError("Stage active scope differs from declared free parameters")
        kwargs = {
            k: stage[k]
            for k in (
                "method",
                "maximum_iterations",
                "maximum_function_evaluations",
                "enforce_historical_guards",
            )
            if k in stage
        }
        if "finite_difference_step" in plan:
            kwargs["finite_difference_step"] = plan["finite_difference_step"]
        validate_native_search_request(
            observations,
            parameters,
            starts,
            fixed_values={
                p.name: float(starts[0, i])
                for i, p in enumerate(parameters)
                if p.name not in active
            },
            calibration=calibration,
            **kwargs,
        )
    if stages and tuple(stages[-1]["active_parameters"]) != tuple(
        n for n in names if n not in fixed
    ):
        raise ValueError("Final stage order must preserve the declared free roster")
    return "Draft search fields structurally checked; full engine launch admission unavailable (R4)"


def _load(physics_path, observation_path, plan, plan_path, control, storage=()):
    from rasim_next.fitting.native_input import load_native_fit_physics
    from rasim_next.fitting.native_observations import load_native_fit_observations

    originals = [Path(p).resolve() for p in (physics_path, observation_path, plan_path)]
    paths = [stored_path(p, storage).resolve(strict=True) for p in originals]
    if paths[0].stat().st_size > 2 * 1024**2 or paths[1].stat().st_size > 2 * 1024**2:
        raise ValueError("Prepared JSON exceeds 2 MiB")
    receipt = json.loads(paths[1].read_bytes())
    original_array = (originals[1].parent / receipt["arrays"]["path"]).resolve()
    array_path = stored_path(original_array, storage, receipt["arrays"]["sha256"]).resolve(
        strict=True
    )
    count = receipt["signal_projection"]["observation_count"]
    if type(count) is not int or not 0 < count <= 4096:
        raise ValueError("Prepared profile row count exceeds 4096")
    with zipfile.ZipFile(array_path) as archive:
        members = archive.infolist()
        if (
            len(members) > 64
            or len({v.filename for v in members}) != len(members)
            or sum(v.file_size for v in members) > 384 * 1024**2
        ):
            raise ValueError("Prepared NPZ exceeds bounded expansion or has duplicate arrays")
        for member in members:
            _stop(control)
            if not member.filename.endswith(".npy"):
                raise ValueError("Prepared NPZ must contain numeric NPY arrays only")
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported prepared NPY header version")
                if (
                    dtype.kind not in "biuf"
                    or dtype.hasobject
                    or dtype.fields is not None
                    or len(shape) > 3
                    or any(type(v) is not int or v < 0 for v in shape)
                    or math.prod(shape) * dtype.itemsize != member.file_size - stream.tell()
                ):
                    raise ValueError("Prepared numeric array shape/dtype/payload binding differs")
    raw_path = Path(receipt["raw_acquisition"]["path"])
    if not raw_path.is_absolute():
        raw_path = originals[1].parent / raw_path
    files = _identities(
        zip(
            ("physics", "observations", "plan", "arrays", "raw_acquisition"),
            (*originals, original_array, raw_path),
            strict=True,
        ),
        control,
        storage,
    )
    if (
        files[3]["sha256"] != receipt["arrays"]["sha256"]
        or files[4]["sha256"] != receipt["raw_acquisition"]["sha256"]
    ):
        raise ValueError("Prepared arrays/raw acquisition binding differs")
    control.report("Reading typed prepared inputs; no preparation or prediction")
    physics = load_native_fit_physics(paths[0])
    observations = load_native_fit_observations(paths[1], arrays_path=array_path)
    if (
        physics.input_revision != receipt["physical_input"]["sha256"]
        or observations.input_revision != files[1]["sha256"]
        or physics.input_revision != files[0]["sha256"]
        or plan["acquisition_id"] != files[4]["sha256"]
        or plan.get("sample_id", physics.sample_id) != physics.sample_id
    ):
        raise ValueError("Prepared physics/observations/plan ownership differs")
    if "experiment_binding" in plan and plan["experiment_binding"] != {
        "physics_sha256": physics.input_revision,
        "observation_sha256": receipt.get("preparation", {}).get(
            "source_observation_sha256", observations.input_revision
        ),
    }:
        raise ValueError("Plan belongs to different prepared inputs")
    definition = _definition(physics, plan)
    detail = _validate_plan(plan, observations, definition)
    with np.load(array_path, allow_pickle=False) as arrays:
        raw, background = arrays["measured"], arrays["background"]
    if not np.array_equal(raw - background, observations.net_count):
        raise ValueError("Prepared count snapshot differs")
    provenance = json.loads(paths[0].read_bytes())
    inputs = {
        "schema": "slate.native-fit-inputs.v1",
        "physics_sha256": physics.input_revision,
        "observation_sha256": observations.input_revision,
        "projection_revision": observations.projection.projection_revision,
        "raw_acquisition_sha256": files[4]["sha256"],
        "files": files,
        "sample_id": physics.sample_id,
        "objective": plan.get("objective", "gls"),
        "covariance_policy": "Full count_covariance + background_modes.T @ background_modes; displayed sigma is marginal only; no diagonal replacement",
        "provenance": {
            k: provenance[k]
            for k in (
                "material",
                "instrument",
                "source",
                "source_rule",
                "structure",
                "rod_catalog_revision",
            )
        },
        "row_count": count,
    }
    _check_files(inputs, control, storage)
    profiles = NativeProfiles(
        raw,
        background,
        observations.net_count,
        np.sqrt(np.diag(observations.covariance_count2)),
        observations.valid,
        tuple(f"{observations.input_revision}:{i}" for i in range(count)),
        observations.input_revision,
        observations.covariance_count2,
    )
    return inputs, definition, profiles, detail


@dataclass(frozen=True, slots=True)
class PreparedWorkResult:
    session: NativeFitSession
    profiles: NativeProfiles | None
    detail: str

    @property
    def nbytes(self):
        return self.session.nbytes + (self.profiles.nbytes if self.profiles else 0) + 4096


def prepared_work(argument, control):
    from threadpoolctl import threadpool_limits

    request = json.loads(argument)
    resources = request.get("resources", {})
    if resources != prepared_budget(
        resources.get("other_cpu_bytes", 0), resources.get("other_gpu_bytes", 0)
    ):
        raise ValueError("Prepared worker requires shared resource admission")
    session = native_fit_session_from_document(request.get("session"))
    storage = storage_files(request.get("storage_json", "{}"))
    operation = request["operation"]
    with threadpool_limits(limits=1):
        _stop(control)
        if operation == "export_profiles":
            from io import BytesIO

            current = json.loads(session.current_json)
            _check_files(current["inputs"], control, storage)
            paths = {v["kind"]: v["path"] for v in current["inputs"]["files"]}
            inputs, definition, profiles, detail = _load(
                paths["physics"],
                paths["observations"],
                current["plan"],
                paths["plan"],
                control,
                storage,
            )
            if description(inputs, current["plan"], definition) != session.current_json:
                raise ValueError("Prepared definitions changed; review before export")
            path = Path(request["path"]).resolve()
            if path.suffix.lower() != ".npz" or len(session.exports) >= 16:
                raise ValueError("Choose a new NPZ within the bounded export history")
            _external(
                path, protected=prepared_paths(session) + [Path(r["stored"]) for r in storage]
            )
            buffer = BytesIO()
            manifest = dict(
                schema="slate.prepared-count-export.v1",
                description_sha256=current["sha256"],
                inputs=current["inputs"],
                covariance_policy=inputs["covariance_policy"],
                unit="counts; covariance counts^2",
                row_order="frozen observation order",
            )
            np.savez(
                buffer,
                raw_count=profiles.raw_count,
                background_count=profiles.background_count,
                signed_corrected_count=profiles.corrected_count,
                marginal_standard_error_count=profiles.standard_error_count,
                valid=profiles.valid,
                covariance_count2=profiles.covariance_count2,
                row_ids=np.asarray(profiles.row_ids),
                manifest_utf8=np.frombuffer(encoded(manifest).encode(), dtype=np.uint8),
            )
            _stop(control)
            raw = buffer.getvalue()
            _publish_bytes(path, raw, control)
            session = replace(
                session, exports=(*session.exports, (str(path), hashlib.sha256(raw).hexdigest()))
            )
            profiles, detail = (
                None,
                "Exact measured vectors/row IDs/full covariance exported as numeric NPZ; no prediction or normalization",
            )
        elif operation == "export":
            current = json.loads(session.current_json)
            _check_files(current["inputs"], control, storage)
            path = Path(request["path"]).resolve()
            _external(
                path, protected=prepared_paths(session) + [Path(r["stored"]) for r in storage]
            )
            if len(session.exports) >= 16:
                raise ValueError("Prepared export history is full")
            raw = encoded(current["plan"]).encode()
            _publish_bytes(path, raw, control)
            session = replace(
                session, exports=(*session.exports, (str(path), hashlib.sha256(raw).hexdigest()))
            )
            profiles, detail = (
                None,
                "Exact committed draft plan exported; no launch admission or fit",
            )
        else:
            if operation == "load":
                original_plan = Path(request["plan_path"]).resolve()
                plan_path = stored_path(original_plan, storage).resolve(strict=True)
                if plan_path.stat().st_size > 96 * 1024:
                    raise ValueError("Prepared plan exceeds 96 KiB")
                plan_bytes = plan_path.read_bytes()
                plan = document(plan_bytes.decode("utf-8"), 96 * 1024)
                physics_path, observation_path = (
                    request["physics_path"],
                    request["observation_path"],
                )
            else:
                if operation not in ("reload", "commit") or session is None:
                    raise ValueError("Unsupported prepared operation")
                old = json.loads(session.current_json)
                _check_files(old["inputs"], control, storage)
                paths = {v["kind"]: v["path"] for v in old["inputs"]["files"]}
                physics_path, observation_path, plan_path = (
                    paths["physics"],
                    paths["observations"],
                    paths["plan"],
                )
                plan = (
                    document(request["plan_json"], 96 * 1024)
                    if operation == "commit"
                    else old["plan"]
                )
            if plan.get("schema") != "rasim-native-refinement-plan-v1":
                raise ValueError("Unsupported prepared plan schema")
            inputs, definition, profiles, detail = _load(
                physics_path,
                observation_path,
                plan,
                original_plan if operation == "load" else plan_path,
                control,
                storage,
            )
            if operation == "load" and hashlib.sha256(plan_bytes).hexdigest() != next(
                v["sha256"] for v in inputs["files"] if v["kind"] == "plan"
            ):
                raise ValueError("Plan changed during prepared input admission")
            if (
                operation == "commit"
                and definition["engine_revision"] != old["definition"]["engine_revision"]
            ):
                raise ValueError(
                    "Engine/model definition changed; explicitly load and review the current plan before editing"
                )
            if operation == "commit" and definition["settings_reasons"] != old["definition"].get(
                "settings_reasons", {}
            ):
                raise ValueError("Unsupported stage choice; current owner method controls only")
            current = description(inputs, plan, definition)
            if operation == "reload":
                if current != session.current_json:
                    raise ValueError(
                        "Original parameter/model definitions changed; load the current plan for explicit review; history retained"
                    )
            else:
                history = () if session is None else (*session.history_json, session.current_json)
                session = NativeFitSession(
                    session.session_id if session else uuid4(),
                    current,
                    history,
                    revision=0 if session is None else session.revision + 1,
                    exports=() if session is None else session.exports,
                )
        _stop(control)
        value = PreparedWorkResult(session, profiles, detail)
        return JobResult(value, value.nbytes)
