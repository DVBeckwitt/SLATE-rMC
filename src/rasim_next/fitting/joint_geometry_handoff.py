"""Exact handoff from qualified joint geometry to fixed-experiment fitting."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from rasim_next.fitting.fixed_experiment import FixedPositionState
from rasim_next.fitting.indexed_series import SharedGeometryCorrections
from rasim_next.fitting.joint_geometry import (
    DEFAULT_FIXED_REFERENCE_PARAMETERS,
    GLOBAL_PARAMETER_NAMES,
    JOINT_GEOMETRY_PARAMETER_NAMES,
    LOCAL_PARAMETER_NAMES,
    NUISANCE_PARAMETER_NAMES,
    JointGeometryState,
    SpecimenId,
    joint_beam_origin_lab_m,
    specimen_local_geometry,
)
from rasim_next.io.json_publication import publish_json_document
from rasim_next.pipeline.configured_simulation import (
    SimulationConfiguration,
    load_simulation_config,
)
from rasim_next.selection.osc_series import load_osc_geometry_series

JOINT_GEOMETRY_FIT_RESULT_SCHEMA_VERSION = "rasim-joint-hbn-crystal-geometry-fit-result-v3"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_vector(value: object, size: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {size} finite values") from error
    if len(result) != size or any(not math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _parameter_value(
    section: Mapping[str, object],
    name: str,
    *,
    role: str,
) -> float:
    entry = _mapping(section.get(name), f"joint geometry parameter {name!r}")
    expected_confidence: bool | None = True if role == "fitted" else None
    if (
        entry.get("role") != role
        or entry.get("confidence_qualified") is not expected_confidence
        or entry.get("active_bound") is not False
    ):
        raise ValueError(f"joint geometry parameter {name!r} is not qualified")
    value = float(entry.get("value", math.nan))
    if not math.isfinite(value):
        raise ValueError(f"joint geometry parameter {name!r} must be finite")
    return value


def qualified_joint_geometry_state(record: object) -> tuple[JointGeometryState, tuple[float, ...]]:
    """Read the fitted state and absolute beam origin from one qualified report."""

    root = _mapping(record, "joint geometry result")
    failures = root.get("qualification_failures")
    if (
        root.get("schema_version") != JOINT_GEOMETRY_FIT_RESULT_SCHEMA_VERSION
        or root.get("success") is not True
        or root.get("confidence_qualified") is not True
        or not isinstance(failures, (list, tuple))
        or failures
    ):
        raise ValueError("joint geometry result is not confidence qualified")

    global_section = _mapping(root.get("global"), "joint geometry global parameters")
    local_section = _mapping(root.get("local"), "joint geometry local parameters")
    nuisance_section = _mapping(root.get("nuisance"), "joint geometry nuisance parameters")
    if set(global_section) != {*GLOBAL_PARAMETER_NAMES, "z_b_m"}:
        raise ValueError("joint geometry global parameter roster changed")
    if set(local_section) != set(LOCAL_PARAMETER_NAMES):
        raise ValueError("joint geometry local parameter roster changed")
    if set(nuisance_section) != set(NUISANCE_PARAMETER_NAMES):
        raise ValueError("joint geometry nuisance parameter roster changed")

    fixed = dict(DEFAULT_FIXED_REFERENCE_PARAMETERS)
    parameterization = _mapping(root.get("parameterization"), "joint parameterization")
    unobserved = tuple(parameterization.get("unobserved_specimen_parameters", ()))
    metrics = _mapping(root.get("crystalline_metrics"), "crystalline metrics")
    per_image = metrics.get("per_image")
    if not isinstance(per_image, (tuple, list)):
        raise ValueError("joint geometry image roster is missing")
    observed = {
        _mapping(metric, "crystalline image metric").get("specimen_id") for metric in per_image
    }
    expected_unobserved = tuple(
        name
        for name in LOCAL_PARAMETER_NAMES
        if (name.startswith("pbi2_y1_") and "pbi2_y1" not in observed)
        or (name.startswith("pbi2_y2_") and "pbi2_y2" not in observed)
    )
    if unobserved != expected_unobserved:
        raise ValueError("joint geometry unobserved parameters disagree with image roster")
    expected_fitted = tuple(
        name
        for name in JOINT_GEOMETRY_PARAMETER_NAMES
        if name not in fixed and name not in unobserved
    )
    if tuple(parameterization.get("fitted_parameter_names", ())) != expected_fitted:
        raise ValueError("joint geometry fitted parameter roster changed")
    sections = {
        **dict.fromkeys(GLOBAL_PARAMETER_NAMES, global_section),
        **dict.fromkeys(LOCAL_PARAMETER_NAMES, local_section),
        **dict.fromkeys(NUISANCE_PARAMETER_NAMES, nuisance_section),
    }
    values = tuple(
        _parameter_value(
            sections[name],
            name,
            role=(
                "fixed_reference"
                if name in fixed
                else "unobserved_specimen"
                if name in unobserved
                else "fitted"
            ),
        )
        for name in JOINT_GEOMETRY_PARAMETER_NAMES
    )
    state = JointGeometryState.from_array(values)
    if any(getattr(state, name) != expected for name, expected in fixed.items()):
        raise ValueError("joint geometry fixed references changed")
    if any(getattr(state, name) != 0.0 for name in unobserved):
        raise ValueError("joint geometry unobserved specimen references changed")

    derived = _mapping(root.get("derived"), "joint geometry derived values")
    beam_origin = _finite_vector(
        derived.get("beam_origin_lab_m"),
        3,
        "derived beam_origin_lab_m",
    )
    return state, beam_origin


def _require_equal(actual: object, expected: object, name: str) -> None:
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise ValueError(f"joint geometry {name} differs from the detector base")


def _validate_detector_base(
    record: Mapping[str, object],
    detector_base: SimulationConfiguration,
) -> None:
    static = _mapping(record.get("static"), "joint geometry static values")
    instrument = detector_base.instrument
    axis = instrument.axis_rotations
    if len(axis) != 1:
        raise ValueError("joint geometry detector base requires one goniometer axis")
    declarations = (
        (instrument.detector_shape_rc, static.get("detector_shape_rc"), "detector shape"),
        (
            instrument.detector_row_pitch_m,
            static.get("detector_row_pitch_m"),
            "detector row pitch",
        ),
        (
            instrument.detector_column_pitch_m,
            static.get("detector_column_pitch_m"),
            "detector column pitch",
        ),
        (
            instrument.detector_reference_coordinate_px,
            static.get("detector_reference_coordinate_px"),
            "detector reference coordinate",
        ),
        (
            instrument.lab_from_detector.translation_m,
            static.get("detector_plane_reference_translation_lab_m"),
            "detector translation",
        ),
        (
            instrument.lab_from_detector.rotation,
            static.get("detector_base_rotation_lab_from_detector"),
            "detector base rotation",
        ),
        (
            detector_base.source.mean_direction_lab,
            static.get("beam_direction_lab"),
            "beam direction",
        ),
        (axis[0].axis_lab, static.get("goniometer_nominal_axis_lab"), "goniometer axis"),
        (
            axis[0].pivot_lab_m,
            static.get("goniometer_nominal_pivot_lab_m"),
            "goniometer pivot",
        ),
    )
    for actual, expected, name in declarations:
        _require_equal(actual, expected, name)
    if static.get("bi2se3_sample_x_tilt_rad") != 0.0:
        raise ValueError("joint geometry Bi2Se3 incidence gauge changed")


def _validate_specimen_base(
    specimen: SimulationConfiguration,
    detector_base: SimulationConfiguration,
) -> None:
    specimen_instrument = specimen.instrument
    base_instrument = detector_base.instrument
    declarations = (
        (
            specimen_instrument.detector_shape_rc,
            base_instrument.detector_shape_rc,
            "specimen detector shape",
        ),
        (
            specimen_instrument.detector_row_pitch_m,
            base_instrument.detector_row_pitch_m,
            "specimen detector row pitch",
        ),
        (
            specimen_instrument.detector_column_pitch_m,
            base_instrument.detector_column_pitch_m,
            "specimen detector column pitch",
        ),
        (
            specimen_instrument.detector_reference_coordinate_px,
            base_instrument.detector_reference_coordinate_px,
            "specimen detector reference coordinate",
        ),
        (
            specimen_instrument.lab_from_detector.translation_m,
            base_instrument.lab_from_detector.translation_m,
            "specimen detector translation",
        ),
        (
            specimen.source.mean_direction_lab,
            detector_base.source.mean_direction_lab,
            "specimen beam direction",
        ),
        (
            specimen.source.mean_origin_lab_m,
            detector_base.source.mean_origin_lab_m,
            "specimen nominal beam origin",
        ),
    )
    for actual, expected, name in declarations:
        _require_equal(actual, expected, name)
    if len(specimen_instrument.axis_rotations) != 1 or len(base_instrument.axis_rotations) != 1:
        raise ValueError("joint geometry handoff requires one goniometer axis")
    _require_equal(
        specimen_instrument.axis_rotations[0].axis_lab,
        base_instrument.axis_rotations[0].axis_lab,
        "specimen nominal goniometer axis",
    )
    _require_equal(
        specimen_instrument.axis_rotations[0].pivot_lab_m,
        base_instrument.axis_rotations[0].pivot_lab_m,
        "specimen nominal goniometer pivot",
    )


@dataclass(frozen=True, slots=True)
class JointGeometryFixedExperimentHandoff:
    """One rebased configuration and its once-only fixed corrections."""

    config: SimulationConfiguration
    position: FixedPositionState

    def __post_init__(self) -> None:
        if not isinstance(self.config, SimulationConfiguration):
            raise TypeError("config must be SimulationConfiguration")
        if not isinstance(self.position, FixedPositionState):
            raise TypeError("position must be FixedPositionState")


def _build_joint_geometry_fixed_experiment_handoff(
    record: object,
    *,
    artifact_revision: str,
    specimen_id: SpecimenId,
    specimen_config: SimulationConfiguration,
    detector_base_config: SimulationConfiguration,
    commanded_incidence_angles_rad: tuple[float, ...],
) -> JointGeometryFixedExperimentHandoff:
    """Translate qualified absolute joint geometry without applying any term twice.

    The fitted detector tilts are absolute relative to the joint fit's detector
    base. The fitted beam center has already been converted to an absolute beam
    line in ``beam_origin_lab_m``. Consequently the returned position keeps the
    detector reference coordinate unchanged and disables the separate detector
    calibration correction.
    """

    if not isinstance(specimen_config, SimulationConfiguration) or not isinstance(
        detector_base_config,
        SimulationConfiguration,
    ):
        raise TypeError("specimen and detector-base configs must be SimulationConfiguration")
    root = _mapping(record, "joint geometry result")
    state, beam_origin = qualified_joint_geometry_state(root)
    _validate_detector_base(root, detector_base_config)
    _validate_specimen_base(specimen_config, detector_base_config)
    expected_phase = "pbi2" if specimen_id in {"pbi2_y1", "pbi2_y2"} else specimen_id
    if specimen_config.material.phase_id != expected_phase:
        raise ValueError("specimen identity differs from the simulation material")

    commanded = tuple(float(value) for value in commanded_incidence_angles_rad)
    if not commanded or any(not math.isfinite(value) for value in commanded):
        raise ValueError("commanded incidence angles must be finite and nonempty")
    expected_beam_origin = joint_beam_origin_lab_m(state, detector_base_config)
    if not np.allclose(beam_origin, expected_beam_origin, rtol=0.0, atol=2.0e-15):
        raise ValueError("derived beam origin disagrees with fitted detector beam center")
    sample_x_tilt, sample_y_tilt, z_s_m = specimen_local_geometry(specimen_id, state)
    corrections = SharedGeometryCorrections(
        detector_column_tilt_rad=state.detector_column_tilt_rad,
        detector_row_tilt_rad=state.detector_row_tilt_rad,
        sample_normal_x_tilt_rad=sample_x_tilt,
        sample_normal_y_tilt_rad=sample_y_tilt,
        goniometer_axis_pitch_rad=state.goniometer_axis_pitch_rad,
        goniometer_axis_yaw_rad=state.goniometer_axis_yaw_rad,
        sample_plane_normal_offset_m=z_s_m,
        goniometer_pivot_pitch_offset_m=state.goniometer_pivot_pitch_offset_m,
        goniometer_pivot_yaw_offset_m=state.goniometer_pivot_yaw_offset_m,
    )
    base_rotation = detector_base_config.instrument.lab_from_detector.rotation
    rebased_config = replace(
        specimen_config,
        source=replace(specimen_config.source, mean_origin_lab_m=beam_origin),
        instrument=replace(
            specimen_config.instrument,
            lab_from_detector=replace(
                specimen_config.instrument.lab_from_detector,
                rotation=base_rotation,
            ),
        ),
    )
    reference = tuple(
        float(value) for value in rebased_config.instrument.detector_reference_coordinate_px
    )
    position = FixedPositionState(
        artifact_revision=artifact_revision,
        corrections=corrections,
        incidence_angle_delta_rad=state.incidence_angle_delta_rad,
        commanded_incidence_angles_rad=commanded,
        beam_center_column_row_px=reference,
        detector_calibration_active=False,
    )
    return JointGeometryFixedExperimentHandoff(rebased_config, position)


def _file_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def _load_identity(value: object, name: str) -> Path:
    identity = _mapping(value, name)
    if set(identity) != {"path", "sha256"}:
        raise ValueError(f"{name} identity is incomplete")
    path = Path(str(identity["path"])).resolve(strict=True)
    if _file_identity(path) != dict(identity):
        raise ValueError(f"{name} bytes changed")
    return path


def _handoff_revision(
    report: Mapping[str, str],
    manifest: Mapping[str, str],
    specimen: Mapping[str, str],
    detector: Mapping[str, str],
    specimen_cif: Mapping[str, str],
    detector_cif: Mapping[str, str],
    specimen_id: SpecimenId,
    osc_images: tuple[dict[str, str], ...],
) -> str:
    payload = json.dumps(
        (report, manifest, specimen, detector, specimen_cif, detector_cif, specimen_id, osc_images),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256-{hashlib.sha256(payload).hexdigest()}"


def _require_report_image_roster(
    report: Mapping[str, object], specimen_id: SpecimenId, image_ids: tuple[str, ...]
) -> None:
    metrics = _mapping(report.get("crystalline_metrics"), "crystalline metrics")
    per_image = metrics.get("per_image")
    if not isinstance(per_image, (list, tuple)):
        raise ValueError("joint report lacks crystalline image metrics")
    reported = []
    for item in per_image:
        metric = _mapping(item, "crystalline image metric")
        if metric.get("specimen_id") == specimen_id:
            reported.append(metric.get("image_id"))
    if tuple(reported) != image_ids:
        raise ValueError("joint report image roster differs from the OSC manifest")


def save_joint_geometry_handoff(
    destination: Path,
    *,
    report_path: Path,
    geometry_manifest_path: Path,
    detector_base_config_path: Path,
    specimen_id: SpecimenId,
) -> Path:
    """Save one hash-bound, reloadable geometry handoff outside the repository."""

    destination = destination.resolve()
    repository_root = Path(__file__).resolve().parents[3]
    if destination == repository_root or destination.is_relative_to(repository_root):
        raise ValueError("joint geometry handoff must be written outside the repository")
    if destination.exists():
        raise FileExistsError(destination)
    manifest_id = _file_identity(geometry_manifest_path)
    series = load_osc_geometry_series(Path(manifest_id["path"]))
    commanded_incidence_angles_rad = tuple(
        math.radians(image.axis_rotation_angles_deg[series.incidence_axis_index])
        for image in series.images
    )
    report_id = _file_identity(report_path)
    specimen_id_file = _file_identity(series.config_path)
    detector_id = _file_identity(detector_base_config_path)
    specimen_config = load_simulation_config(Path(specimen_id_file["path"]))
    detector_config = load_simulation_config(Path(detector_id["path"]))
    specimen_cif_id = _file_identity(specimen_config.material.cif_path)
    detector_cif_id = _file_identity(detector_config.material.cif_path)
    osc_ids = tuple(_file_identity(image.osc_path) for image in series.images)
    report = _mapping(
        json.loads(Path(report_id["path"]).read_text(encoding="utf-8")), "joint report"
    )
    _require_report_image_roster(
        report, specimen_id, tuple(image.image_id for image in series.images)
    )
    handoff = _build_joint_geometry_fixed_experiment_handoff(
        report,
        artifact_revision=_handoff_revision(
            report_id,
            manifest_id,
            specimen_id_file,
            detector_id,
            specimen_cif_id,
            detector_cif_id,
            specimen_id,
            osc_ids,
        ),
        specimen_id=specimen_id,
        specimen_config=specimen_config,
        detector_base_config=detector_config,
        commanded_incidence_angles_rad=commanded_incidence_angles_rad,
    )
    document = {
        "schema_version": "rasim-joint-geometry-handoff-v1",
        "status": "GEOMETRY_ONLY",
        "mosaic_qualified": False,
        "report": report_id,
        "geometry_manifest": manifest_id,
        "specimen_config": specimen_id_file,
        "detector_base_config": detector_id,
        "specimen_cif": specimen_cif_id,
        "detector_base_cif": detector_cif_id,
        "specimen_id": specimen_id,
        "commanded_incidence_angles_rad": commanded_incidence_angles_rad,
        "image_ids": tuple(image.image_id for image in series.images),
        "osc_images": osc_ids,
        "rebased_beam_origin_lab_m": handoff.config.source.mean_origin_lab_m,
        "rebased_detector_rotation": np.asarray(
            handoff.config.instrument.lab_from_detector.rotation
        ).tolist(),
        "fixed_position": handoff.position.to_record(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    return publish_json_document(destination, document)


def load_joint_geometry_handoff(path: Path) -> JointGeometryFixedExperimentHandoff:
    """Verify predecessor bytes and reconstruct the full detector and source state."""

    document = _mapping(json.loads(path.read_text(encoding="utf-8")), "handoff")
    if (
        document.get("schema_version") != "rasim-joint-geometry-handoff-v1"
        or document.get("status") != "GEOMETRY_ONLY"
        or document.get("mosaic_qualified") is not False
    ):
        raise ValueError("unsupported joint geometry handoff")
    report = _load_identity(document.get("report"), "joint report")
    manifest = _load_identity(document.get("geometry_manifest"), "geometry manifest")
    specimen = _load_identity(document.get("specimen_config"), "specimen config")
    detector = _load_identity(document.get("detector_base_config"), "detector base config")
    specimen_cif = _load_identity(document.get("specimen_cif"), "specimen CIF")
    detector_cif = _load_identity(document.get("detector_base_cif"), "detector base CIF")
    report_id = _file_identity(report)
    manifest_id = _file_identity(manifest)
    specimen_file_id = _file_identity(specimen)
    detector_id = _file_identity(detector)
    specimen_cif_id = _file_identity(specimen_cif)
    detector_cif_id = _file_identity(detector_cif)
    specimen_config = load_simulation_config(specimen)
    detector_config = load_simulation_config(detector)
    if (
        specimen_config.material.cif_path != specimen_cif
        or detector_config.material.cif_path != detector_cif
    ):
        raise ValueError("joint geometry handoff CIF references changed")
    series = load_osc_geometry_series(manifest)
    osc_ids = tuple(_file_identity(image.osc_path) for image in series.images)
    report_record = _mapping(json.loads(report.read_text(encoding="utf-8")), "joint report")
    _require_report_image_roster(
        report_record,
        str(document["specimen_id"]),
        tuple(image.image_id for image in series.images),
    )
    commanded = tuple(
        math.radians(image.axis_rotation_angles_deg[series.incidence_axis_index])
        for image in series.images
    )
    if (
        series.config_path != specimen
        or tuple(image.image_id for image in series.images) != tuple(document["image_ids"])
        or osc_ids != tuple(document["osc_images"])
        or not np.allclose(
            commanded, document["commanded_incidence_angles_rad"], rtol=0.0, atol=0.0
        )
    ):
        raise ValueError("joint geometry handoff image roster changed")
    handoff = _build_joint_geometry_fixed_experiment_handoff(
        report_record,
        artifact_revision=_handoff_revision(
            report_id,
            manifest_id,
            specimen_file_id,
            detector_id,
            specimen_cif_id,
            detector_cif_id,
            str(document["specimen_id"]),
            osc_ids,
        ),
        specimen_id=str(document["specimen_id"]),
        specimen_config=specimen_config,
        detector_base_config=detector_config,
        commanded_incidence_angles_rad=commanded,
    )
    if (
        json.loads(json.dumps(handoff.position.to_record())) != document.get("fixed_position")
        or not np.array_equal(
            handoff.config.source.mean_origin_lab_m,
            document.get("rebased_beam_origin_lab_m"),
        )
        or not np.array_equal(
            handoff.config.instrument.lab_from_detector.rotation,
            document.get("rebased_detector_rotation"),
        )
    ):
        raise ValueError("joint geometry handoff changed on reload")
    return handoff


__all__ = [
    "JOINT_GEOMETRY_FIT_RESULT_SCHEMA_VERSION",
    "JointGeometryFixedExperimentHandoff",
    "load_joint_geometry_handoff",
    "qualified_joint_geometry_state",
    "save_joint_geometry_handoff",
]
