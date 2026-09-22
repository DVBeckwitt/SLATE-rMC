"""Automatically fit shared hBN/Bi detector and goniometer geometry."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rasim_next.fitting.hbn import (  # noqa: E402
    CU_K_ALPHA_WAVELENGTH_A,
    HBN_LATTICE_A_A,
    HBN_LATTICE_C_A,
    HBN_RING_HKL,
    fit_hbn_detector_calibration,
    hbn_two_theta_rad,
)
from rasim_next.fitting.joint_geometry import (  # noqa: E402
    GLOBAL_PARAMETER_NAMES,
    JOINT_GEOMETRY_PARAMETER_NAMES,
    LOCAL_PARAMETER_NAMES,
    NUISANCE_PARAMETER_NAMES,
    fit_joint_geometry,
)
from rasim_next.io.osc import read_osc  # noqa: E402
from rasim_next.pipeline.configured_simulation import (  # noqa: E402
    build_configured_geometry_inputs,
    load_simulation_config,
    load_strict_yaml_mapping,
)
from rasim_next.selection.osc_series import (  # noqa: E402
    index_osc_geometry_series,
    load_osc_geometry_series,
)


def _mapping(value: object, name: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return value


def _path(base: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty path")
    result = (base / value).resolve()
    if not result.is_file():
        raise ValueError(f"{name} does not exist: {result}")
    return result


def _write_external_json(destination: Path, payload: dict[str, object]) -> Path:
    path = destination.resolve()
    if path == ROOT or path.is_relative_to(ROOT):
        raise ValueError("joint geometry artifact must be written outside the repository")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _parameter_payload(result: object, names: tuple[str, ...]) -> dict[str, object]:
    state = result.state
    return {
        name: {
            "value": float(getattr(state, name)),
            "standard_error": float(result.standard_error[JOINT_GEOMETRY_PARAMETER_NAMES.index(name)]),
            "confidence_qualified": bool(
                result.parameter_confident[JOINT_GEOMETRY_PARAMETER_NAMES.index(name)]
            ),
            "active_bound": bool(result.active_bounds[JOINT_GEOMETRY_PARAMETER_NAMES.index(name)]),
        }
        for name in names
    }


def run(manifest_path: Path) -> dict[str, object]:
    manifest = load_strict_yaml_mapping(manifest_path)
    root = _mapping(
        manifest,
        "joint geometry manifest",
        {
            "schema_version",
            "detector_base_simulation_config",
            "hbn",
            "bi2se3_series",
            "bi2te3_series",
        },
    )
    if root["schema_version"] != "rasim-joint-hbn-bi-geometry-v1":
        raise ValueError("unsupported joint geometry schema_version")
    hbn = _mapping(
        root["hbn"],
        "hbn",
        {
            "osc_path",
            "dark_path",
            "initial_beam_center_px",
            "initial_calibrant_distance_m",
        },
    )
    base_directory = manifest_path.parent
    base_config_path = _path(
        base_directory,
        root["detector_base_simulation_config"],
        "detector_base_simulation_config",
    )
    hbn_path = _path(base_directory, hbn["osc_path"], "hbn.osc_path")
    dark_path = _path(base_directory, hbn["dark_path"], "hbn.dark_path")
    bi2se3_path = _path(base_directory, root["bi2se3_series"], "bi2se3_series")
    bi2te3_path = _path(base_directory, root["bi2te3_series"], "bi2te3_series")
    center = np.asarray(hbn["initial_beam_center_px"], dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("hbn.initial_beam_center_px must contain finite column and row")

    base_config = load_simulation_config(base_config_path)
    base_inputs = build_configured_geometry_inputs(base_config)
    base_instrument = base_inputs.instrument
    hbn_image = read_osc(hbn_path).detector_native_counts
    dark_image = read_osc(dark_path).detector_native_counts
    hbn_observations, hbn_calibration = fit_hbn_detector_calibration(
        hbn_image,
        dark_image,
        base_detector_rotation=base_instrument.lab_from_detector.rotation,
        beam_direction_lab=base_config.source.mean_direction_lab,
        detector_column_pitch_m=base_instrument.detector_column_pitch_m,
        detector_row_pitch_m=base_instrument.detector_row_pitch_m,
        initial_beam_center_px=(float(center[0]), float(center[1])),
        initial_calibrant_distance_m=float(hbn["initial_calibrant_distance_m"]),
    )
    se3_run = index_osc_geometry_series(load_osc_geometry_series(bi2se3_path))
    te3_run = index_osc_geometry_series(load_osc_geometry_series(bi2te3_path))
    if se3_run.indexed_images is None or te3_run.indexed_images is None:
        raise RuntimeError("both indexed material series must retain fit-ready observations")
    result = fit_joint_geometry(
        hbn_observations=hbn_observations,
        hbn_calibration=hbn_calibration,
        bi2se3_images=se3_run.indexed_images,
        bi2te3_images=te3_run.indexed_images,
        base_detector_rotation=base_instrument.lab_from_detector.rotation,
    )
    nominal_axis = base_config.instrument.axis_rotations[0]
    beam_direction = np.asarray(base_config.source.mean_direction_lab, dtype=np.float64)
    beam_direction /= np.linalg.norm(beam_direction)
    static = {
        "detector_shape_rc": base_instrument.detector_shape_rc,
        "detector_row_pitch_m": base_instrument.detector_row_pitch_m,
        "detector_column_pitch_m": base_instrument.detector_column_pitch_m,
        "detector_reference_coordinate_px": base_instrument.detector_reference_coordinate_px,
        "detector_plane_reference_translation_lab_m": (
            base_instrument.lab_from_detector.translation_m.tolist()
        ),
        "detector_reference_distance_along_beam_m": float(
            base_instrument.lab_from_detector.translation_m @ beam_direction
        ),
        "detector_base_rotation_lab_from_detector": (
            base_instrument.lab_from_detector.rotation.tolist()
        ),
        "beam_direction_lab": list(base_config.source.mean_direction_lab),
        "goniometer_nominal_axis_lab": nominal_axis.axis_lab,
        "goniometer_nominal_pivot_lab_m": nominal_axis.pivot_lab_m,
        "goniometer_zero_translation_lab_m": (
            base_config.instrument.lab_from_goniometer_zero.translation_m
        ),
        "sample_translation_in_goniometer_m": (
            base_config.instrument.goniometer_from_sample.translation_m
        ),
        "crystal_translation_in_sample_m": (
            base_config.instrument.sample_from_crystal.translation_m
        ),
        "detector_roll_rad": 0.0,
        "bi2se3_sample_x_tilt_rad": 0.0,
        "bi2se3_sample_x_tilt_role": "fixed gauge reference for common incidence-angle zero",
        "hbn_lattice_a_A": HBN_LATTICE_A_A,
        "hbn_lattice_c_A": HBN_LATTICE_C_A,
        "hbn_wavelength_A": CU_K_ALPHA_WAVELENGTH_A,
        "hbn_ring_hkl": HBN_RING_HKL,
        "hbn_two_theta_deg": np.degrees(hbn_two_theta_rad()).tolist(),
    }
    global_parameters = _parameter_payload(result, GLOBAL_PARAMETER_NAMES)
    global_parameters["z_b_m"] = {
        "value": result.z_b_m,
        "standard_error": result.z_b_standard_error_m,
        "confidence_qualified": False,
        "active_bound": bool(
            result.active_bounds[
                JOINT_GEOMETRY_PARAMETER_NAMES.index("goniometer_pivot_pitch_offset_m")
            ]
        ),
        "derived_from": "fitted beam line minus corrected goniometer pivot along lab z",
    }
    payload = {
        "schema_version": "rasim-joint-hbn-bi-geometry-fit-result-v1",
        "success": result.success,
        "confidence_qualified": result.confidence_qualified,
        "message": result.message,
        "parameterization": {
            "shared_hbn_coordinates": (
                "detector_column_tilt_rad",
                "detector_row_tilt_rad",
                "beam_center_column_px",
                "beam_center_row_px",
            ),
            "hbn_excluded_shared_coordinates": (
                "detector distance",
                "goniometer axis",
                "goniometer pivot",
                "z_b",
                "sample pose",
                "z_s",
            ),
            "gauge": "Bi2Se3 sample-x tilt fixed; common incidence-angle delta fitted",
        },
        "static": static,
        "global": global_parameters,
        "local": _parameter_payload(result, LOCAL_PARAMETER_NAMES),
        "nuisance": _parameter_payload(result, NUISANCE_PARAMETER_NAMES),
        "derived": {
            "beam_origin_lab_m": result.beam_origin_lab_m.tolist(),
            "corrected_goniometer_axis_lab": result.corrected_goniometer_axis_lab.tolist(),
            "corrected_goniometer_pivot_lab_m": (
                result.corrected_goniometer_pivot_lab_m.tolist()
            ),
        },
        "hbn_automatic_trace": {
            "success": hbn_calibration.success,
            "point_count_by_ring": hbn_calibration.ring_point_count,
            "angular_coverage_fraction_by_ring": (
                hbn_calibration.ring_angular_coverage_fraction
            ),
            "standalone_rms_px": hbn_calibration.residual_rms_px,
            "joint_rms_px": result.hbn_residual_rms_px,
            "joint_max_px": result.hbn_residual_max_px,
        },
        "bi_metrics": {
            "pooled_site_rms_px": result.pooled_bi_site_rms_px,
            "pooled_site_max_px": result.pooled_bi_site_max_px,
            "per_image": tuple(asdict(metric) for metric in result.per_image),
        },
        "identifiability": {
            "jacobian_rank": result.jacobian_rank,
            "parameter_count": len(JOINT_GEOMETRY_PARAMETER_NAMES),
            "scaled_jacobian_condition": result.scaled_jacobian_condition,
            "scaled_jacobian_singular_values": (
                result.scaled_jacobian_singular_values.tolist()
            ),
            "weakest_direction": {
                name: float(value)
                for name, value in zip(
                    JOINT_GEOMETRY_PARAMETER_NAMES,
                    result.weakest_direction,
                    strict=True,
                )
            },
            "active_bounds": {
                name: bool(value)
                for name, value in zip(
                    JOINT_GEOMETRY_PARAMETER_NAMES,
                    result.active_bounds,
                    strict=True,
                )
            },
        },
        "work": {
            "model_evaluation_count": result.model_evaluation_count,
            "optimizer_function_evaluation_count": (
                result.optimizer_function_evaluation_count
            ),
            "indexed_site_count": {
                "bi2se3": sum(len(image.observations.keys) for image in se3_run.indexed_images),
                "bi2te3": sum(len(image.observations.keys) for image in te3_run.indexed_images),
            },
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=ROOT / "configs" / "joint_hbn_bi_geometry_fit.yaml",
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    payload = run(arguments.manifest.resolve())
    if arguments.destination is not None:
        payload["artifact_path"] = str(_write_external_json(arguments.destination, payload))
    if arguments.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["confidence_qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
