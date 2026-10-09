"""Canonical joint geometry report serialization shared by CLI and desktop."""

import math
from dataclasses import asdict
from itertools import pairwise

import numpy as np

from rasim_next.fitting.hbn import (
    CU_K_ALPHA_WAVELENGTH_A,
    HBN_LATTICE_A_A,
    HBN_LATTICE_C_A,
    HBN_RING_HKL,
    hbn_two_theta_rad,
)
from rasim_next.fitting.joint_geometry import (
    GLOBAL_PARAMETER_NAMES,
    JOINT_GEOMETRY_PARAMETER_NAMES,
    LOCAL_PARAMETER_NAMES,
    NUISANCE_PARAMETER_NAMES,
)


def _parameter_payload(result: object, names: tuple[str, ...]) -> dict[str, object]:
    state = result.state
    fixed = dict(result.fixed_reference_parameters)
    unobserved = set(result.unobserved_specimen_parameters)
    payload = {}
    for name in names:
        index = JOINT_GEOMETRY_PARAMETER_NAMES.index(name)
        if name in fixed or name in unobserved:
            payload[name] = {
                "value": float(getattr(state, name)),
                "standard_error": None,
                "confidence_qualified": None,
                "active_bound": False,
                "role": "fixed_reference" if name in fixed else "unobserved_specimen",
            }
        else:
            payload[name] = {
                "value": float(getattr(state, name)),
                "standard_error": float(result.standard_error[index])
                if math.isfinite(result.standard_error[index])
                else None,
                "confidence_qualified": bool(result.parameter_confident[index]),
                "active_bound": bool(result.active_bounds[index]),
                "role": "fitted",
            }
    return payload


def joint_geometry_static(base_config, base_instrument, *, pbi2_parent=None):
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
        "pbi2_parent": pbi2_parent,
        "hbn_lattice_a_A": HBN_LATTICE_A_A,
        "hbn_lattice_c_A": HBN_LATTICE_C_A,
        "hbn_wavelength_A": CU_K_ALPHA_WAVELENGTH_A,
        "hbn_ring_hkl": HBN_RING_HKL,
        "hbn_two_theta_deg": np.degrees(hbn_two_theta_rad()).tolist(),
    }
    return static


def joint_geometry_report(
    result,
    *,
    static,
    hbn_calibration,
    bi2se3_images,
    bi2te3_images,
    pbi2_y1_images=(),
    pbi2_y2_images=(),
):
    has_pbi2 = bool(pbi2_y1_images or pbi2_y2_images)
    global_parameters = _parameter_payload(result, GLOBAL_PARAMETER_NAMES)
    global_parameters["z_b_m"] = {
        "value": result.z_b_m,
        "standard_error": result.z_b_standard_error_m,
        "confidence_qualified": result.confidence_qualified,
        "active_bound": False,
        "role": "derived_conditional_on_fixed_reference",
        "derived_from": "fitted beam line minus nominal-reference goniometer pivot along lab z",
        "mechanical_interpretation_qualified": False,
    }
    qualification_failures = joint_geometry_qualification_failures(result)
    payload = {
        "schema_version": "rasim-joint-hbn-crystal-geometry-fit-result-v3",
        "success": result.success,
        "confidence_qualified": result.confidence_qualified,
        "message": result.message,
        "qualification_failures": tuple(qualification_failures),
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
            "gauge": (
                "Bi2Se3 sample-x tilt, goniometer-axis pitch, and pivot pitch displacement "
                "fixed to nominal references; common incidence-angle delta fitted"
            ),
            "fixed_reference_parameters": dict(result.fixed_reference_parameters),
            "unobserved_specimen_parameters": result.unobserved_specimen_parameters,
            "fitted_parameter_names": result.fitted_parameter_names,
            "qualification_scope": (
                "detector-predictive geometry over the observed angle range; fixed mechanical "
                "references and physical zB are not independently measured"
            ),
            "pbi2_admission": (
                "same exact 2H integer-L key recovered blindly at both distinct incidences, "
                "with observed incidence motion coherent to 8 px RMS; at least two retained "
                "keys per image"
            )
            if has_pbi2
            else None,
        },
        "static": static,
        "global": global_parameters,
        "local": _parameter_payload(result, LOCAL_PARAMETER_NAMES),
        "nuisance": _parameter_payload(result, NUISANCE_PARAMETER_NAMES),
        "derived": {
            "beam_origin_lab_m": result.beam_origin_lab_m.tolist(),
            "corrected_goniometer_axis_lab": result.corrected_goniometer_axis_lab.tolist(),
            "corrected_goniometer_pivot_lab_m": (result.corrected_goniometer_pivot_lab_m.tolist()),
        },
        "hbn_automatic_trace": {
            "success": hbn_calibration.success,
            "point_count_by_ring": hbn_calibration.ring_point_count,
            "angular_coverage_fraction_by_ring": (hbn_calibration.ring_angular_coverage_fraction),
            "standalone_rms_px": hbn_calibration.residual_rms_px,
            "joint_rms_px": result.hbn_residual_rms_px,
            "joint_max_px": result.hbn_residual_max_px,
        },
        "crystalline_metrics": {
            "pooled_site_rms_px": result.pooled_crystalline_site_rms_px,
            "pooled_site_max_px": result.pooled_crystalline_site_max_px,
            "per_image": tuple(asdict(metric) for metric in result.per_image),
        },
        "identifiability": {
            "jacobian_rank": result.jacobian_rank,
            "parameter_count": len(result.fitted_parameter_names),
            "scaled_jacobian_condition": result.scaled_jacobian_condition
            if math.isfinite(result.scaled_jacobian_condition)
            else None,
            "scaled_jacobian_singular_values": (result.scaled_jacobian_singular_values.tolist()),
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
            "optimizer_function_evaluation_count": (result.optimizer_function_evaluation_count),
            "indexed_site_count": {
                "bi2se3": sum(len(image.observations.keys) for image in bi2se3_images),
                "bi2te3": sum(len(image.observations.keys) for image in bi2te3_images),
                "pbi2_y1": sum(len(image.observations.keys) for image in pbi2_y1_images),
                "pbi2_y2": sum(len(image.observations.keys) for image in pbi2_y2_images),
            },
            "pbi2_retained_keys": {
                image.image_id: tuple(asdict(key) for key in image.observations.keys)
                for image in (*pbi2_y1_images, *pbi2_y2_images)
            },
        },
    }
    return payload


def joint_geometry_qualification_failures(result):
    fitted_names = set(result.fitted_parameter_names)
    active_bound_names = tuple(
        name
        for name, active in zip(
            JOINT_GEOMETRY_PARAMETER_NAMES,
            result.active_bounds,
            strict=True,
        )
        if active and name in fitted_names
    )
    unconfident_names = tuple(
        name
        for name, confident in zip(
            JOINT_GEOMETRY_PARAMETER_NAMES,
            result.parameter_confident,
            strict=True,
        )
        if not confident and name in fitted_names
    )
    qualification_failures = []
    if not result.success:
        qualification_failures.append("optimizer did not converge")
    if result.jacobian_rank != len(result.fitted_parameter_names):
        qualification_failures.append(
            f"Jacobian rank {result.jacobian_rank}/{len(result.fitted_parameter_names)}"
        )
    if result.scaled_jacobian_condition > 1.0e8:
        qualification_failures.append(
            f"scaled Jacobian condition {result.scaled_jacobian_condition:.6g} exceeds 1e8"
        )
    if active_bound_names:
        qualification_failures.append("active bounds: " + ", ".join(active_bound_names))
    if unconfident_names:
        qualification_failures.append("uncertain parameters: " + ", ".join(unconfident_names))
    if result.hbn_residual_rms_px > 2.0:
        qualification_failures.append("hBN residual RMS exceeds 2 px")
    if (
        result.pooled_crystalline_site_rms_px > 3.0
        or result.pooled_crystalline_site_max_px > 8.0
        or any(metric.site_rms_px > 4.0 for metric in result.per_image)
    ):
        qualification_failures.append("crystalline detector residual gate failed")
    return tuple(qualification_failures)


def validate_joint_geometry_report(report):
    """Check recorded roles and qualification without resolving files or solving."""
    from rasim_next.fitting.joint_geometry import (
        DEFAULT_FIXED_REFERENCE_PARAMETERS,
        JointGeometryState,
    )

    if report.get("schema_version") != "rasim-joint-hbn-crystal-geometry-fit-result-v3":
        raise ValueError("unsupported joint geometry report")
    fixed = dict(DEFAULT_FIXED_REFERENCE_PARAMETERS)
    metrics = report["crystalline_metrics"]
    per_image = metrics["per_image"]
    scopes = {v["specimen_id"] for v in per_image}
    if not {"bi2se3", "bi2te3"} <= scopes or not scopes <= {
        "bi2se3",
        "bi2te3",
        "pbi2_y1",
        "pbi2_y2",
    }:
        raise ValueError("joint report requires both canonical bismuth groups")
    identities = [(v["specimen_id"], v["image_id"]) for v in per_image]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate joint report image identity")
    unobserved = tuple(
        n
        for n in LOCAL_PARAMETER_NAMES
        if (n.startswith("pbi2_y1_") and "pbi2_y1" not in scopes)
        or (n.startswith("pbi2_y2_") and "pbi2_y2" not in scopes)
    )
    fitted = tuple(
        n for n in JOINT_GEOMETRY_PARAMETER_NAMES if n not in fixed and n not in unobserved
    )
    parameterization = report["parameterization"]
    if (
        parameterization["fixed_reference_parameters"] != fixed
        or tuple(parameterization.get("unobserved_specimen_parameters", ())) != unobserved
        or tuple(parameterization["fitted_parameter_names"]) != fitted
        or tuple(parameterization["shared_hbn_coordinates"]) != JOINT_GEOMETRY_PARAMETER_NAMES[:4]
    ):
        raise ValueError("joint report gauge/scopes disagree with its roster")
    values, flags, active = {}, {}, {}
    for section, names in (
        ("global", GLOBAL_PARAMETER_NAMES),
        ("local", LOCAL_PARAMETER_NAMES),
        ("nuisance", NUISANCE_PARAMETER_NAMES),
    ):
        if set(report[section]) != set(names) | ({"z_b_m"} if section == "global" else set()):
            raise ValueError("joint report canonical parameter roster changed")
        for n in names:
            row = report[section][n]
            role = (
                "fixed_reference"
                if n in fixed
                else "unobserved_specimen"
                if n in unobserved
                else "fitted"
            )
            if (
                row["role"] != role
                or type(row["value"]) not in (int, float)
                or not math.isfinite(row["value"])
            ):
                raise ValueError("joint report parameter role/value changed: " + n)
            if role != "fitted":
                if (
                    row["value"] != 0.0
                    or row["standard_error"] is not None
                    or row["confidence_qualified"] is not None
                    or row["active_bound"] is not False
                ):
                    raise ValueError("joint report fixed/unobserved reference changed: " + n)
            elif (
                type(row["confidence_qualified"]) is not bool
                or type(row["active_bound"]) is not bool
                or (
                    row["standard_error"] is not None
                    and (not math.isfinite(row["standard_error"]) or row["standard_error"] < 0)
                )
                or (row["confidence_qualified"] and row["standard_error"] is None)
            ):
                raise ValueError("joint report uncertainty/flags invalid: " + n)
            values[n], flags[n], active[n] = (
                row["value"],
                row["confidence_qualified"],
                row["active_bound"],
            )
    JointGeometryState.from_array([values[n] for n in JOINT_GEOMETRY_PARAMETER_NAMES])
    if values["hbn_calibrant_distance_m"] <= 0:
        raise ValueError("joint private calibrant distance must be positive")
    ident = report["identifiability"]
    count = len(fitted)
    if (
        type(ident["jacobian_rank"]) is not int
        or not 0 <= ident["jacobian_rank"] <= count
        or ident["parameter_count"] != count
        or ident["active_bounds"] != active
        or set(ident["weakest_direction"]) != set(JOINT_GEOMETRY_PARAMETER_NAMES)
    ):
        raise ValueError("joint report rank/active-bound scopes changed")
    singular = ident["scaled_jacobian_singular_values"]
    if (
        len(singular) != count
        or any(not math.isfinite(v) or v < 0 for v in singular)
        or any(a < b for a, b in pairwise(singular))
    ):
        raise ValueError("joint recorded singular values invalid")
    condition = ident["scaled_jacobian_condition"]
    expected_condition = singular[0] / singular[-1] if singular[-1] > 0 else None
    if condition != expected_condition:
        raise ValueError("joint recorded condition disagrees with singular values")
    for v in per_image:
        if (
            type(v["site_count"]) is not int
            or v["site_count"] < 1
            or not 0 <= v["site_rms_px"] <= v["site_max_px"]
            or not math.isfinite(v["site_max_px"])
        ):
            raise ValueError("joint per-image metric invalid")
    pooled = math.sqrt(
        sum(v["site_count"] * v["site_rms_px"] ** 2 for v in per_image)
        / sum(v["site_count"] for v in per_image)
    )
    if not math.isclose(
        pooled, metrics["pooled_site_rms_px"], rel_tol=1e-12, abs_tol=1e-12
    ) or metrics["pooled_site_max_px"] != max(v["site_max_px"] for v in per_image):
        raise ValueError("joint pooled diagnostics disagree with image metrics")
    hbn = report["hbn_automatic_trace"]
    if not 0 <= hbn["joint_rms_px"] <= hbn["joint_max_px"] or not math.isfinite(
        hbn["joint_max_px"]
    ):
        raise ValueError("joint hBN metrics must be finite nonnegative pixel residuals")
    if any(not math.isfinite(v) for v in ident["weakest_direction"].values()):
        raise ValueError("joint weakest direction must be finite")
    # Use the CLI's existing qualification rule on recorded checks.
    from types import SimpleNamespace

    proxy = SimpleNamespace(
        fitted_parameter_names=fitted,
        active_bounds=[active[n] for n in JOINT_GEOMETRY_PARAMETER_NAMES],
        parameter_confident=[flags[n] is True for n in JOINT_GEOMETRY_PARAMETER_NAMES],
        success=report["success"],
        jacobian_rank=ident["jacobian_rank"],
        scaled_jacobian_condition=math.inf if condition is None else condition,
        hbn_residual_rms_px=report["hbn_automatic_trace"]["joint_rms_px"],
        pooled_crystalline_site_rms_px=metrics["pooled_site_rms_px"],
        pooled_crystalline_site_max_px=metrics["pooled_site_max_px"],
        per_image=[SimpleNamespace(**v) for v in per_image],
    )
    failures = joint_geometry_qualification_failures(proxy)
    if (
        type(report["success"]) is not bool
        or type(report["confidence_qualified"]) is not bool
        or tuple(report["qualification_failures"]) != failures
        or report["confidence_qualified"] != (not failures)
    ):
        raise ValueError("joint report qualification contradicts recorded checks")
    z = report["global"]["z_b_m"]
    if (
        z["role"] != "derived_conditional_on_fixed_reference"
        or z["mechanical_interpretation_qualified"] is not False
        or z["confidence_qualified"] != report["confidence_qualified"]
    ):
        raise ValueError("joint conditional zB interpretation changed")
    for vector in report["derived"].values():
        if len(vector) != 3 or any(not math.isfinite(v) for v in vector):
            raise ValueError("joint derived vector invalid")
    return report
