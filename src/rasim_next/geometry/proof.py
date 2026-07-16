"""Compact analytic and reference proof for T02 geometry and optics."""

from __future__ import annotations

import cmath
import hashlib
import json
import platform
import subprocess
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from rasim_next.core.contracts import (
    CONTRACT_API_VERSION,
    IncidentSampleBatch,
    MaterialOptics,
    ScatteringEventBatch,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.validity import ValidityCode
from rasim_next.geometry.detector import detector_coordinate_to_ray, project_detector_ray
from rasim_next.geometry.instrument import (
    AxisRotation,
    CompiledInstrument,
    InstrumentConfiguration,
    compile_instrument,
)
from rasim_next.geometry.sample import intersect_sample_ray
from rasim_next.geometry.transport import build_incident_states, transport_scattering_events
from rasim_next.io.osc import read_osc
from rasim_next.optics.attenuation import (
    mode_decay_constant,
    scalar_optical_weight,
    uniform_depth_attenuation,
)
from rasim_next.optics.refraction import solve_exit_mode, solve_incident_mode

_PROOF_BASE_SHA = "812f896fde5b8365ff5c218fc606df674ad7dcad"
_REFERENCE_PACK_SHA256 = "e958703426ebea7a3fd62a8bb52447f9a5a8d7d5d4ad0eb0ce3b3706bbca1f06"
_TOLERANCES = {
    "version": "geometry-optics-tolerances-v1",
    "discrete": {"atol": 0.0, "rtol": 0.0},
    "rotation_direction_residual": {"atol": 2e-12, "rtol": 0.0},
    "position_m": {"atol": 1e-12, "rtol": 0.0},
    "wavevector_Ainv": {"atol": 5e-13, "rtol": 2e-12},
    "detector_coordinate_px": {"atol": 1e-9, "rtol": 0.0},
    "amplitude_weight": {"atol": 5e-13, "rtol": 2e-12},
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _environment_sha256() -> str:
    return _hash_json(
        {
            "machine": platform.machine(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
    )


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _check(check_id: str, passed: bool, evidence: str, **details: object) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
        **details,
    }


def _material(wavelength_A: float, index: complex, material_id: str) -> MaterialOptics:
    return MaterialOptics(
        material_id=material_id,
        wavelength_A=np.array([wavelength_A]),
        n_complex=np.array([index]),
        delta=np.array([1.0 - index.real]),
        beta=np.array([max(index.imag, 0.0)]),
        mu_Ainv=np.array([0.0]),
        provenance="T02 compact analytic fixture",
    )


def _instrument(
    *,
    lab_from_detector: RigidTransform | None = None,
    sample_width_m: float = 4.0e-4,
    sample_length_m: float = 6.0e-4,
) -> CompiledInstrument:
    identity = np.eye(3)
    zero = np.zeros(3)
    return compile_instrument(
        InstrumentConfiguration(
            axis_rotations=(),
            lab_from_goniometer_zero=RigidTransform(
                identity, zero, FrameId.GONIOMETER, FrameId.LAB
            ),
            goniometer_from_sample=RigidTransform(
                identity, zero, FrameId.SAMPLE, FrameId.GONIOMETER
            ),
            sample_from_crystal=RigidTransform(identity, zero, FrameId.CRYSTAL, FrameId.SAMPLE),
            lab_from_detector=lab_from_detector
            or RigidTransform(
                identity,
                np.array([0.0, 0.0, 1.0]),
                FrameId.DETECTOR,
                FrameId.LAB,
            ),
            detector_shape_rc=(11, 7),
            detector_row_pitch_m=2.0e-4,
            detector_column_pitch_m=1.0e-4,
            detector_reference_coordinate_px=(3.0, 5.0),
            sample_width_m=sample_width_m,
            sample_length_m=sample_length_m,
            film_thickness_A=500.0,
        )
    )


def _classified_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "osc.real.bi2se3",
            "classification": "MATCH",
            "ledger_ids": [
                "PHY-IO-001",
                "PHY-IO-002",
                "PHY-IO-003",
                "PHY-IO-004",
                "PHY-IO-005",
            ],
            "first_divergence_stage": None,
        },
        {
            "case_id": "osc.synthetic.non_square",
            "classification": "MATCH",
            "ledger_ids": [
                "PHY-IO-001",
                "PHY-IO-002",
                "PHY-IO-003",
                "PHY-IO-004",
                "PHY-IO-005",
            ],
            "first_divergence_stage": None,
        },
        {
            "case_id": "geometry.line_plane",
            "classification": "MATCH",
            "ledger_ids": ["PHY-GEO-005", "PHY-GEO-006", "PHY-THK-001"],
            "first_divergence_stage": None,
        },
        {
            "case_id": "geometry.sample_origin_nonrigid",
            "classification": "CORRECTED",
            "ledger_ids": ["PHY-GEO-001", "PHY-GEO-002", "PHY-GEO-003"],
            "first_divergence_stage": "geometry.instrument_transforms",
            "pack_first_divergence_stage": "geometry.sample_origin_lab",
            "last_matching_stage": "declared instrument inputs",
        },
        {
            "case_id": "optics.interface_fresnel",
            "classification": "CORRECTED",
            "ledger_ids": [
                "PHY-OPT-001",
                "PHY-OPT-002",
                "PHY-OPT-003",
                "PHY-OPT-005",
                "PHY-OPT-007",
                "PHY-OPT-010",
            ],
            "first_divergence_stage": "measurement.optical_weight",
            "pack_first_divergence_stage": "optics.interface_weight",
            "last_matching_stage": "optics.kz_incident_film",
        },
        {
            "case_id": "optics.depth_attenuation",
            "classification": "CORRECTED",
            "ledger_ids": ["PHY-OPT-008", "PHY-THK-002"],
            "first_divergence_stage": "optics.uniform_depth_attenuation",
            "pack_first_divergence_stage": "optics.depth_weight",
            "last_matching_stage": "declared decay constants",
        },
        {
            "case_id": "optics.external_exit",
            "classification": "MATCH",
            "ledger_ids": [
                "PHY-OPT-004",
                "PHY-OPT-006",
                "PHY-OPT-007",
                "PHY-OPT-010",
            ],
            "first_divergence_stage": None,
        },
        {
            "case_id": "geometry.detector_rectangular_anisotropic",
            "classification": "NO_ORACLE",
            "ledger_ids": [
                "PHY-GEO-004",
                "PHY-GEO-008",
                "PHY-GEO-009",
                "PHY-GEO-010",
                "PHY-GEO-011",
            ],
            "first_divergence_stage": None,
        },
        {
            "case_id": "geometry.global_rigid_covariance",
            "classification": "NO_ORACLE",
            "ledger_ids": ["PHY-GEO-007", "PHY-GEO-012"],
            "first_divergence_stage": None,
        },
        {
            "case_id": "optics.critical_limit",
            "classification": "NO_ORACLE",
            "ledger_ids": ["PHY-OPT-006", "PHY-OPT-009"],
            "first_divergence_stage": None,
        },
    ]


def _reference_integrity(
    pack_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    observed = _hash_file(pack_path)
    passed = (
        observed == _REFERENCE_PACK_SHA256
        and manifest.get("schema_version") == "rasim-reference-pack-v1"
    )
    return _check(
        "reference_pack_integrity",
        passed,
        "tracked pack hash and schema match the immutable v1 authority",
        expected_sha256=_REFERENCE_PACK_SHA256,
        observed_sha256=observed,
    )


def _classification_check(
    arrays: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        item["case_id"]: (
            item["classification"],
            item.get("pack_first_divergence_stage", item["first_divergence_stage"]),
        )
        for item in _classified_cases()
        if item["case_id"]
        in {
            "osc.real.bi2se3",
            "osc.synthetic.non_square",
            "geometry.line_plane",
            "geometry.sample_origin_nonrigid",
            "optics.interface_fresnel",
            "optics.depth_attenuation",
            "optics.external_exit",
        }
    }
    observed = {
        item["case_id"]: (item["classification"], item["first_divergence"])
        for item in manifest["cases"]
        if item["subsystem"] == "geometry_optics"
    }
    differences = {
        "nonrigid_origin_m": float(
            np.linalg.norm(
                arrays["geometry_sample_origin_rigid"] - arrays["geometry_sample_origin_legacy"]
            )
        ),
        "legacy_interface_weight": float(
            np.max(
                np.abs(
                    arrays["optics_selected_local_field_weight"]
                    - arrays["optics_legacy_power_average"]
                )
            )
        ),
        "legacy_full_depth": float(
            np.max(
                np.abs(
                    arrays["optics_attenuation_uniform_depth_average"]
                    - arrays["optics_attenuation_legacy_full_depth"]
                )
            )
        ),
    }
    return _check(
        "classification_and_first_divergence",
        observed == expected and all(difference > 0.0 for difference in differences.values()),
        "all T02 pack labels agree and each corrected legacy path visibly diverges",
        correction_differences=differences,
    )


def _exact_stage(
    stage_id: str,
    reference: object,
    mutated: object,
    *,
    failure_metric: str = "exact_value",
) -> tuple[str, str, object, object, float, float]:
    return stage_id, failure_metric, reference, mutated, 0.0, 0.0


def _numeric_stage(
    stage_id: str,
    reference: object,
    mutated: object,
    tolerance: dict[str, float],
) -> tuple[str, str, object, object, float, float]:
    return (
        stage_id,
        "numeric_value",
        reference,
        mutated,
        tolerance["atol"],
        tolerance["rtol"],
    )


def _validation_outcome(action: Callable[[], object]) -> str:
    try:
        action()
    except (TypeError, ValueError) as error:
        return type(error).__name__
    return "accepted"


def _compare_mutation_stage(
    stage: tuple[str, str, object, object, float, float],
) -> tuple[bool, dict[str, Any]]:
    stage_id, metric, reference, mutated, atol, rtol = stage
    if metric == "validation":
        diverged = reference != mutated
        return diverged, {
            "stage_id": stage_id,
            "failure_metric": metric,
            "reference_outcome": reference,
            "mutated_outcome": mutated,
            "diverged": diverged,
        }

    reference_array = np.asarray(reference)
    mutated_array = np.asarray(mutated)
    if reference_array.shape != mutated_array.shape:
        return True, {
            "stage_id": stage_id,
            "failure_metric": metric,
            "reference_shape": list(reference_array.shape),
            "mutated_shape": list(mutated_array.shape),
            "diverged": True,
        }
    if metric == "exact_value":
        mismatch = reference_array != mutated_array
        mismatch_count = int(np.count_nonzero(mismatch))
        first = (
            [int(index) for index in np.argwhere(mismatch)[0]]
            if mismatch_count and mismatch.ndim > 0
            else []
        )
        return mismatch_count > 0, {
            "stage_id": stage_id,
            "failure_metric": metric,
            "mismatch_count": mismatch_count,
            "first_failing_element": first,
            "diverged": mismatch_count > 0,
        }

    reference_numeric = np.asarray(reference_array, dtype=np.complex128)
    mutated_numeric = np.asarray(mutated_array, dtype=np.complex128)
    difference = np.abs(mutated_numeric - reference_numeric)
    limit = atol + rtol * np.abs(reference_numeric)
    failing = difference > limit
    failed_count = int(np.count_nonzero(failing))
    maximum_index = int(np.argmax(difference))
    failing_element = (
        [int(index) for index in np.unravel_index(maximum_index, difference.shape)]
        if difference.ndim > 0
        else []
    )
    return failed_count > 0, {
        "stage_id": stage_id,
        "failure_metric": metric,
        "maximum_error": float(np.max(difference)),
        "percentile_95_error": float(np.percentile(difference, 95.0)),
        "failing_element": failing_element,
        "failed_count": failed_count,
        "atol": atol,
        "rtol": rtol,
        "diverged": failed_count > 0,
    }


def _mutation(
    mutation_id: str,
    fixture_id: str,
    expected_first_stage: str,
    expected_failure_metric: str,
    stages: list[tuple[str, str, object, object, float, float]],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    observed_stage: str | None = None
    observed_metric: str | None = None
    for stage in stages:
        diverged, comparison = _compare_mutation_stage(stage)
        comparisons.append(comparison)
        if diverged and observed_stage is None:
            observed_stage = stage[0]
            observed_metric = stage[1]
    detected = observed_stage == expected_first_stage and observed_metric == expected_failure_metric
    return {
        "mutation_id": mutation_id,
        "fixture_id": fixture_id,
        "expected_first_stage": expected_first_stage,
        "expected_failure_metric": expected_failure_metric,
        "observed_first_stage": observed_stage,
        "observed_failure_metric": observed_metric,
        "detected": detected,
        "stage_comparisons": comparisons,
    }


def _error_injections(root: Path, arrays: Any) -> list[dict[str, Any]]:
    image = read_osc(root / "examples/common/osc/non_square_big_endian.osc")
    raw = image.raw_counts
    native = image.detector_native_counts
    direct_hit = project_detector_ray(
        np.zeros(3),
        [0.0, 0.0, 1.0],
        _instrument(),
    )
    mutations = [
        _mutation(
            "osc_counterclockwise",
            "osc.synthetic.non_square",
            "osc.detector_native_array",
            "exact_value",
            [
                _exact_stage("osc.raw_array", raw, raw),
                _exact_stage(
                    "osc.detector_native_array",
                    native,
                    np.rot90(raw, 1),
                ),
            ],
        ),
        _mutation(
            "osc_transpose",
            "osc.synthetic.non_square",
            "osc.detector_native_array",
            "exact_value",
            [
                _exact_stage("osc.raw_array", raw, raw),
                _exact_stage("osc.detector_native_array", native, raw.T),
            ],
        ),
        _mutation(
            "swap_row_column",
            "osc.synthetic.non_square",
            "osc.beam_center_native",
            "numeric_value",
            [
                _exact_stage("osc.detector_native_array", native, native),
                _numeric_stage(
                    "osc.beam_center_native",
                    [direct_hit.column_px, direct_hit.row_px],
                    [direct_hit.row_px, direct_hit.column_px],
                    _TOLERANCES["detector_coordinate_px"],
                ),
            ],
        ),
        _mutation(
            "half_pixel",
            "detector.reference_coordinate",
            "osc.beam_center_native",
            "numeric_value",
            [
                _numeric_stage(
                    "osc.beam_center_native",
                    [direct_hit.column_px, direct_hit.row_px],
                    [direct_hit.column_px + 0.5, direct_hit.row_px + 0.5],
                    _TOLERANCES["detector_coordinate_px"],
                )
            ],
        ),
    ]

    rotation_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    pivot_z = np.array([1.0, 0.0, 0.0])
    pivot_x = np.array([0.0, 2.0, 0.0])
    point = np.array([2.0, 0.0, 1.0])
    z_step = RigidTransform.around_pivot(
        rotation=rotation_z,
        pivot_m=pivot_z,
        frame=FrameId.LAB,
    )
    x_step = RigidTransform.around_pivot(
        rotation=rotation_x,
        pivot_m=pivot_x,
        frame=FrameId.LAB,
    )
    ordered = x_step.compose(z_step).apply_point(point)
    reversed_order = z_step.compose(x_step).apply_point(point)
    wrong_pivot = RigidTransform.around_pivot(
        rotation=rotation_z,
        pivot_m=np.zeros(3),
        frame=FrameId.LAB,
    ).apply_point(point)
    shift = RigidTransform(
        np.eye(3),
        [0.2, -0.1, 0.3],
        FrameId.LAB,
        FrameId.LAB,
    )
    mutations.extend(
        (
            _mutation(
                "transform_order",
                "geometry.noncommuting_pivots",
                "geometry.instrument_transforms",
                "numeric_value",
                [
                    _numeric_stage(
                        "geometry.instrument_transforms",
                        ordered,
                        reversed_order,
                        _TOLERANCES["position_m"],
                    )
                ],
            ),
            _mutation(
                "wrong_rotation_pivot",
                "geometry.nonzero_pivot",
                "geometry.instrument_transforms",
                "numeric_value",
                [
                    _numeric_stage(
                        "geometry.instrument_transforms",
                        z_step.apply_point(point),
                        wrong_pivot,
                        _TOLERANCES["position_m"],
                    )
                ],
            ),
            _mutation(
                "translate_vector",
                "geometry.rigid_transform",
                "geometry.lab_ray",
                "numeric_value",
                [
                    _numeric_stage(
                        "geometry.lab_ray",
                        shift.apply_vector(point),
                        shift.apply_point(point),
                        _TOLERANCES["position_m"],
                    )
                ],
            ),
            _mutation(
                "alter_transformed_coordinate",
                "geometry.sample_origin_nonrigid",
                "geometry.instrument_transforms",
                "numeric_value",
                [
                    _numeric_stage(
                        "geometry.instrument_transforms",
                        arrays["geometry_sample_origin_rigid"],
                        arrays["geometry_sample_origin_legacy"],
                        _TOLERANCES["position_m"],
                    )
                ],
            ),
        )
    )

    k0_Ainv = float(arrays["optics_exit_inputs"][0, 3])
    wavelength_A = 2.0 * np.pi / k0_Ainv
    film = _material(wavelength_A, 0.999979 + 3.2e-7j, "mutation-film")
    alpha = float(arrays["optics_alpha_rad"][0])
    incident = solve_incident_mode(
        [np.cos(alpha), 0.0, np.sin(alpha)],
        wavelength_A,
        film,
    )
    wrong_root = -incident.kz_film_Ainv
    correct_decay_outcome = _validation_outcome(
        lambda: mode_decay_constant(
            incident.kz_film_Ainv,
            incident.propagation_direction,
        )
    )
    reversed_decay_outcome = _validation_outcome(
        lambda: mode_decay_constant(
            incident.kz_film_Ainv,
            -incident.propagation_direction,
        )
    )

    exit_mode = solve_exit_mode(
        [
            incident.k_parallel_sample_Ainv[0],
            0.0,
            abs(incident.kz_film_Ainv.real),
        ],
        wavelength_A,
        film,
    )
    kappa_i = mode_decay_constant(
        incident.kz_film_Ainv,
        incident.propagation_direction,
    )
    kappa_f = mode_decay_constant(
        exit_mode.kz_film_Ainv,
        exit_mode.propagation_direction,
    )
    attenuation = float(uniform_depth_attenuation(kappa_i, kappa_f, 500.0))
    optical = float(
        scalar_optical_weight(
            incident.entrance_amplitude,
            exit_mode.exit_amplitude,
            attenuation,
        )
    )
    omit_entrance = abs(exit_mode.exit_amplitude) ** 2 * attenuation
    omit_exit = abs(incident.entrance_amplitude) ** 2 * attenuation
    full_depth = np.exp(-2.0 * (float(kappa_i) + float(kappa_f)) * 500.0)
    real_geometry_outcome = _validation_outcome(
        lambda: solve_exit_mode([0.3, 0.0, 1.0], wavelength_A, film)
    )
    complex_geometry_outcome = _validation_outcome(
        lambda: solve_exit_mode([0.3 + 1e-6j, 0.0, 1.0], wavelength_A, film)
    )
    right_handed_outcome = _validation_outcome(
        lambda: RigidTransform(
            np.eye(3),
            np.zeros(3),
            FrameId.DETECTOR,
            FrameId.LAB,
        )
    )
    left_handed_outcome = _validation_outcome(
        lambda: RigidTransform(
            np.diag([-1.0, 1.0, 1.0]),
            np.zeros(3),
            FrameId.DETECTOR,
            FrameId.LAB,
        )
    )

    mutations.extend(
        (
            _mutation(
                "opposite_complex_root",
                "optics.absorbing_incident",
                "optics.kz_incident_film",
                "numeric_value",
                [
                    _numeric_stage(
                        "optics.kz_incident_film",
                        incident.kz_film_Ainv,
                        wrong_root,
                        _TOLERANCES["wavevector_Ainv"],
                    )
                ],
            ),
            _mutation(
                "reverse_propagation_sign",
                "optics.absorbing_incident",
                "optics.kz_incident_film",
                "validation",
                [
                    _exact_stage(
                        "optics.kz_incident_film",
                        correct_decay_outcome,
                        reversed_decay_outcome,
                        failure_metric="validation",
                    )
                ],
            ),
            _mutation(
                "legacy_sp_power_average",
                "optics.interface_fresnel",
                "measurement.optical_weight",
                "numeric_value",
                [
                    _numeric_stage(
                        "measurement.optical_weight",
                        arrays["optics_selected_local_field_weight"],
                        arrays["optics_legacy_power_average"],
                        _TOLERANCES["amplitude_weight"],
                    )
                ],
            ),
            _mutation(
                "omit_entrance_transmission",
                "optics.public_transport",
                "measurement.optical_weight",
                "numeric_value",
                [
                    _numeric_stage(
                        "measurement.optical_weight",
                        optical,
                        omit_entrance,
                        _TOLERANCES["amplitude_weight"],
                    )
                ],
            ),
            _mutation(
                "omit_exit_transmission",
                "optics.public_transport",
                "measurement.optical_weight",
                "numeric_value",
                [
                    _numeric_stage(
                        "measurement.optical_weight",
                        optical,
                        omit_exit,
                        _TOLERANCES["amplitude_weight"],
                    )
                ],
            ),
            _mutation(
                "legacy_full_depth",
                "optics.depth_attenuation",
                "optics.uniform_depth_attenuation",
                "numeric_value",
                [
                    _numeric_stage(
                        "optics.uniform_depth_attenuation",
                        attenuation,
                        full_depth,
                        _TOLERANCES["amplitude_weight"],
                    )
                ],
            ),
            _mutation(
                "omit_attenuation",
                "optics.depth_attenuation",
                "optics.uniform_depth_attenuation",
                "numeric_value",
                [
                    _numeric_stage(
                        "optics.uniform_depth_attenuation",
                        attenuation,
                        1.0,
                        _TOLERANCES["amplitude_weight"],
                    )
                ],
            ),
            _mutation(
                "complex_wavevector_in_geometry",
                "optics.real_phase_boundary",
                "optics.kf_film_sample",
                "validation",
                [
                    _exact_stage(
                        "optics.kf_film_sample",
                        real_geometry_outcome,
                        complex_geometry_outcome,
                        failure_metric="validation",
                    )
                ],
            ),
            _mutation(
                "left_handed_detector_basis",
                "geometry.detector_frame",
                "geometry.detector_frame",
                "validation",
                [
                    _exact_stage(
                        "geometry.detector_frame",
                        right_handed_outcome,
                        left_handed_outcome,
                        failure_metric="validation",
                    )
                ],
            ),
        )
    )
    return mutations


def _osc_checks(root: Path, arrays: Any, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    big = read_osc(root / "examples/common/osc/non_square_big_endian.osc")
    little = read_osc(root / "examples/common/osc/non_square_little_endian.osc")
    synthetic_passed = all(
        (
            np.array_equal(big.raw_counts, arrays["osc_synthetic_big_endian_raw"]),
            np.array_equal(
                big.detector_native_counts,
                arrays["osc_synthetic_big_endian_native"],
            ),
            np.array_equal(little.raw_counts, arrays["osc_synthetic_little_endian_raw"]),
            np.array_equal(
                little.detector_native_counts,
                arrays["osc_synthetic_little_endian_native"],
            ),
        )
    )

    positions = arrays["osc_selected_positions_row_col"]
    real_passed = True
    for index, item in enumerate(manifest["osc_files"]):
        image = read_osc(root / "examples/bi2se3/osc" / f"{item['name']}.gz")
        raw = image.raw_counts
        native = image.detector_native_counts
        summary = np.array(
            [
                image.metadata.version,
                raw.shape[0],
                raw.shape[1],
                int(raw.min()),
                int(raw.max()),
                int(raw.sum(dtype=np.int64)),
            ]
        )
        real_passed &= all(
            (
                np.array_equal(summary, arrays["osc_summary"][index, :6]),
                np.array_equal(
                    raw[positions[:, 0], positions[:, 1]],
                    arrays["osc_selected_raw_values"][index],
                ),
                np.array_equal(
                    native[positions[:, 0], positions[:, 1]],
                    arrays["osc_selected_native_values"][index],
                ),
                np.array_equal(
                    np.unravel_index(int(np.argmax(raw)), raw.shape),
                    arrays["osc_argmax_raw_row_col"][index],
                ),
                np.array_equal(
                    np.unravel_index(int(np.argmax(native)), native.shape),
                    arrays["osc_argmax_native_row_col"][index],
                ),
            )
        )
    return [
        _check(
            "osc_synthetic_reference",
            synthetic_passed,
            "both endian layouts, high-range expansion, and clockwise native orientation match",
        ),
        _check(
            "osc_real_reference",
            real_passed,
            "three tracked Bi2Se3 images match summary, selected pixels, and argmax positions",
        ),
    ]


def _line_plane_check(arrays: Any) -> dict[str, Any]:
    identity = RigidTransform(np.eye(3), np.zeros(3), FrameId.SAMPLE, FrameId.LAB)
    maximum_error = 0.0
    passed = True
    for inputs, expected in zip(
        arrays["geometry_line_plane_inputs"],
        arrays["geometry_line_plane_outputs"],
        strict=True,
    ):
        result = intersect_sample_ray(
            inputs[0],
            inputs[1],
            lab_from_sample=identity,
            sample_width_m=10.0,
            sample_length_m=10.0,
        )
        expected_valid = bool(expected[3])
        passed &= (result.status is ValidityCode.VALID) == expected_valid
        if expected_valid:
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(result.point_lab_m - expected[:3]))),
            )
    passed &= maximum_error <= _TOLERANCES["position_m"]["atol"]
    return _check(
        "geometry_line_plane_reference",
        passed,
        "public sample intersection matches the tracked analytic line-plane cases",
        maximum_error_m=maximum_error,
    )


def _rigid_instrument_check(arrays: Any) -> dict[str, Any]:
    zero = np.zeros(3)
    crystal_angle = 0.2
    crystal_rotation = np.array(
        [
            [np.cos(crystal_angle), -np.sin(crystal_angle), 0.0],
            [np.sin(crystal_angle), np.cos(crystal_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tilt = np.deg2rad(1.5)
    yaw = np.deg2rad(0.7)
    cor_axis = np.array(
        [
            np.cos(tilt) * np.cos(yaw),
            -np.cos(tilt) * np.sin(yaw),
            np.sin(tilt),
        ]
    )
    compiled = compile_instrument(
        InstrumentConfiguration(
            axis_rotations=(
                AxisRotation([0.0, 1.0, 0.0], np.deg2rad(0.8), zero),
                AxisRotation([0.0, 0.0, 1.0], np.deg2rad(2.3), zero),
                AxisRotation(cor_axis, np.deg2rad(12.0), zero),
            ),
            lab_from_goniometer_zero=RigidTransform(
                np.eye(3), zero, FrameId.GONIOMETER, FrameId.LAB
            ),
            goniometer_from_sample=RigidTransform(
                np.eye(3),
                [0.0, 0.0, -0.0006],
                FrameId.SAMPLE,
                FrameId.GONIOMETER,
            ),
            sample_from_crystal=RigidTransform(
                crystal_rotation,
                [1e-5, -2e-5, 3e-5],
                FrameId.CRYSTAL,
                FrameId.SAMPLE,
            ),
            lab_from_detector=RigidTransform(
                np.eye(3), [0.5, 0.0, 0.0], FrameId.DETECTOR, FrameId.LAB
            ),
            detector_shape_rc=(11, 7),
            detector_row_pitch_m=2e-4,
            detector_column_pitch_m=1e-4,
            detector_reference_coordinate_px=(3.0, 5.0),
            sample_width_m=4e-4,
            sample_length_m=6e-4,
            film_thickness_A=100.0,
        )
    )
    rotation_error = float(
        np.max(np.abs(compiled.lab_from_sample.rotation - arrays["geometry_sample_rotation"]))
    )
    origin = compiled.lab_from_sample.apply_point(zero)
    origin_error = float(np.max(np.abs(origin - arrays["geometry_sample_origin_rigid"])))
    rejected_legacy_difference = float(
        np.linalg.norm(origin - arrays["geometry_sample_origin_legacy"])
    )
    passed = (
        rotation_error <= 2e-12
        and origin_error <= _TOLERANCES["position_m"]["atol"]
        and rejected_legacy_difference > 1e-9
    )
    return _check(
        "geometry_rigid_instrument_reference",
        passed,
        "ordered active rotations reproduce the rigid oracle and reject the legacy coordinate overwrite",
        maximum_rotation_error=rotation_error,
        maximum_origin_error_m=origin_error,
        rejected_legacy_difference_m=rejected_legacy_difference,
    )


def _detector_check() -> dict[str, Any]:
    instrument = _instrument()
    origin = np.zeros(3)
    coordinates = ((3.0, 5.0), (-0.5, -0.5), (6.5, 10.5), (1.25, 8.75))
    errors: list[float] = []
    for column, row in coordinates:
        ray = detector_coordinate_to_ray(
            column,
            row,
            origin_lab_m=origin,
            instrument=instrument,
        )
        hit = project_detector_ray(origin, ray.direction_lab, instrument)
        errors.append(max(abs(hit.column_px - column), abs(hit.row_px - row)))

    detector_center = instrument.lab_from_detector.apply_point(np.zeros(3))
    detector_normal = instrument.lab_from_detector.apply_vector([0.0, 0.0, 1.0])
    near = detector_coordinate_to_ray(
        3.0,
        5.0,
        origin_lab_m=detector_center - 5e-13 * detector_normal,
        instrument=instrument,
    )

    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    global_motion = RigidTransform(rotation, np.zeros(3), FrameId.LAB, FrameId.LAB)
    rotated = _instrument(lab_from_detector=global_motion.compose(instrument.lab_from_detector))
    ray = detector_coordinate_to_ray(
        1.25,
        8.75,
        origin_lab_m=origin,
        instrument=instrument,
    )
    covariant = project_detector_ray(
        global_motion.apply_point(origin),
        global_motion.apply_vector(ray.direction_lab),
        rotated,
    )
    covariance_error = max(
        abs(covariant.column_px - 1.25),
        abs(covariant.row_px - 8.75),
    )
    maximum_error = max((*errors, covariance_error))
    passed = (
        maximum_error <= _TOLERANCES["detector_coordinate_px"]["atol"]
        and near.status is ValidityCode.VALID
        and 0.0 < near.ray_distance_m <= 1e-12
    )
    return _check(
        "detector_round_trip_and_covariance",
        passed,
        "rectangular anisotropic coordinates round-trip at centers and edges under rigid lab motion",
        maximum_error_px=maximum_error,
        near_detector_distance_m=near.ray_distance_m,
    )


def _optics_check(arrays: Any) -> dict[str, Any]:
    k0_Ainv = float(arrays["optics_exit_inputs"][0, 3])
    wavelength_A = 2.0 * np.pi / k0_Ainv
    film = _material(wavelength_A, 0.999979 + 3.2e-7j, "pack-film")
    wavevector_errors: list[float] = []
    amplitude_errors: list[float] = []
    for row, alpha in enumerate(arrays["optics_alpha_rad"]):
        mode = solve_incident_mode(
            [np.cos(alpha), 0.0, np.sin(alpha)],
            wavelength_A,
            film,
        )
        radicand = (film.n_complex[0] * k0_Ainv) ** 2 - (k0_Ainv * np.cos(alpha)) ** 2
        root = cmath.sqrt(radicand)
        if root.imag < 0.0 or (root.imag == 0.0 and root.real < 0.0):
            root = -root
        amplitude = 2.0 * mode.kz_air_Ainv / (mode.kz_air_Ainv + root)
        wavevector_errors.extend(
            (
                abs(mode.kz_air_Ainv - arrays["optics_kz_incident"][row]),
                abs(mode.kz_film_Ainv - arrays["optics_kz_transmitted"][row]),
                abs(mode.kz_film_Ainv - root),
            )
        )
        amplitude_errors.extend(
            (
                abs(mode.entrance_amplitude - arrays["optics_t_scalar"][row]),
                abs(mode.entrance_amplitude - amplitude),
            )
        )

    kappa_i = arrays["optics_attenuation_kappa_i"]
    kappa_f = arrays["optics_attenuation_kappa_f"]
    thickness_A = float(arrays["optics_attenuation_thickness_angstrom"])
    attenuation = np.asarray(uniform_depth_attenuation(kappa_i, kappa_f, thickness_A))
    nodes, weights = np.polynomial.legendre.leggauss(24)
    depth_A = 0.5 * thickness_A * (nodes + 1.0)
    quadrature = 0.5 * np.sum(
        weights[None, :] * np.exp(-2.0 * (kappa_i + kappa_f)[:, None] * depth_A[None, :]),
        axis=1,
    )
    attenuation_error = float(
        max(
            np.max(np.abs(attenuation - arrays["optics_attenuation_uniform_depth_average"])),
            np.max(np.abs(attenuation - quadrature)),
        )
    )

    vacuum = _material(wavelength_A, 1.0 + 0.0j, "vacuum")
    exit_errors: list[float] = []
    exit_statuses: list[bool] = []
    for inputs, expected in zip(
        arrays["optics_exit_inputs"],
        arrays["optics_exit_outputs"],
        strict=True,
    ):
        kx, ky, kz_sign, case_k0 = inputs
        mode = solve_exit_mode([kx, ky, kz_sign], 2.0 * np.pi / case_k0, vacuum)
        expected_valid = bool(expected[0])
        exit_statuses.append((mode.status is ValidityCode.VALID) == expected_valid)
        if expected_valid:
            exit_errors.append(float(np.max(np.abs(mode.k_air_phase_sample_Ainv - expected[1:4]))))
            angle = np.arctan2(
                mode.k_air_phase_sample_Ainv[2],
                np.hypot(
                    mode.k_air_phase_sample_Ainv[0],
                    mode.k_air_phase_sample_Ainv[1],
                ),
            )
            exit_errors.append(abs(float(angle - expected[4])))

    critical_index = 0.8
    critical = solve_incident_mode(
        [critical_index, 0.0, np.sqrt(1.0 - critical_index**2)],
        wavelength_A,
        _material(wavelength_A, complex(critical_index), "critical"),
    )
    dispersion = critical.kz_film_Ainv**2 + np.dot(
        critical.k_parallel_sample_Ainv,
        critical.k_parallel_sample_Ainv,
    )
    critical_residual = abs(dispersion - (critical_index * k0_Ainv) ** 2)
    wavevector_limit = (
        _TOLERANCES["wavevector_Ainv"]["atol"] + _TOLERANCES["wavevector_Ainv"]["rtol"] * k0_Ainv
    )
    amplitude_limit = _TOLERANCES["amplitude_weight"]["atol"] + _TOLERANCES["amplitude_weight"][
        "rtol"
    ] * float(np.max(np.abs(arrays["optics_t_scalar"])))
    passed = (
        max(wavevector_errors) <= wavevector_limit
        and max(amplitude_errors) <= amplitude_limit
        and attenuation_error <= amplitude_limit
        and all(exit_statuses)
        and max(exit_errors) <= max(wavevector_limit, 2e-12)
        and critical_residual <= 2e-12 * max((critical_index * k0_Ainv) ** 2, 1.0)
    )
    return _check(
        "optics_equations_and_reference",
        passed,
        "complex-k branch, scalar amplitudes, exit modes, and uniform-depth attenuation match independent equations and the pack",
        maximum_wavevector_error_Ainv=float(max(wavevector_errors)),
        maximum_amplitude_error=float(max(amplitude_errors)),
        maximum_attenuation_error=attenuation_error,
        maximum_exit_error=float(max(exit_errors)),
        normalized_critical_residual=float(
            critical_residual / max((critical_index * k0_Ainv) ** 2, 1.0)
        ),
    )


def _transport_check() -> dict[str, Any]:
    wavelength_A = 1.54
    instrument = _instrument()
    material = _material(wavelength_A, 0.999979 + 3.2e-7j, "transport-film")
    samples = IncidentSampleBatch(
        incident_sample_id=np.array([7, 3]),
        origin_lab_m=np.array([[0.0, 0.0, 1.0], [2.1e-4, 0.0, 1.0]]),
        direction_lab=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        wavelength_A=np.full(2, wavelength_A),
        source_weight=np.array([0.75, 0.25]),
        polarization_state_id=("p7", "p3"),
        correlation_model="independent",
    )
    incident = build_incident_states(
        samples,
        material,
        instrument,
        trace_case_id="compact",
    )
    film_normal_Ainv = abs(float(incident.states.kz_film_Ainv[0].real))
    events = ScatteringEventBatch(
        event_id=np.array([9, 4]),
        incident_state_id=np.array([7, 3]),
        rod_id=np.array([2, 5]),
        wavelength_A=np.full(2, wavelength_A),
        q_internal_sample_Ainv=np.zeros((2, 3)),
        qz_Ainv=np.zeros(2),
        l_coordinate=np.zeros(2),
        kf_film_phase_sample_Ainv=np.array(
            [[0.0, 0.0, film_normal_Ainv], [0.0, 0.0, film_normal_Ainv]]
        ),
        reciprocal_weight=np.array([0.2, 0.8]),
        ewald_residual_Ainv=np.zeros(2),
        valid=np.ones(2, dtype=bool),
    )
    transported = transport_scattering_events(
        events,
        incident,
        material,
        instrument,
        trace_case_id="compact",
    )
    exit_mode = solve_exit_mode(
        [0.0, 0.0, film_normal_Ainv],
        wavelength_A,
        material,
    )
    incident_mode = solve_incident_mode([0.0, 0.0, -1.0], wavelength_A, material)
    attenuation = uniform_depth_attenuation(
        mode_decay_constant(
            incident_mode.kz_film_Ainv,
            incident_mode.propagation_direction,
        ),
        mode_decay_constant(exit_mode.kz_film_Ainv, exit_mode.propagation_direction),
        instrument.film_thickness_A,
    )
    expected_optical = scalar_optical_weight(
        incident_mode.entrance_amplitude,
        exit_mode.exit_amplitude,
        attenuation,
    )
    stage_ids = {record.stage_id for record in transported.traces}
    passed = all(
        (
            incident.status == (ValidityCode.VALID, ValidityCode.OUTSIDE_SUPPORT),
            transported.outgoing_status == (ValidityCode.VALID, ValidityCode.OUTSIDE_SUPPORT),
            transported.detector_status == (ValidityCode.VALID, ValidityCode.OUTSIDE_SUPPORT),
            np.array_equal(transported.outgoing_waves.event_id, events.event_id),
            np.array_equal(transported.detector_hits.event_id, events.event_id),
            abs(transported.outgoing_waves.optical_weight[0] - expected_optical) <= 5e-13,
            transported.detector_hits.column_px[0] == 3.0,
            transported.detector_hits.row_px[0] == 5.0,
            {
                "optics.kz_exit_air",
                "optics.uniform_depth_attenuation",
                "geometry.detector_column_px",
                "measurement.optical_weight",
                "measurement.pixel_solid_angle",
            }
            <= stage_ids,
        )
    )
    return _check(
        "public_transport_contract",
        passed,
        "IDs, first-failure status, optical factors, detector hits, and trace stages stay aligned",
        valid_event_id=int(events.event_id[0]),
        invalid_event_id=int(events.event_id[1]),
    )


def _convergence_checks() -> list[dict[str, Any]]:
    identity = RigidTransform(np.eye(3), np.zeros(3), FrameId.SAMPLE, FrameId.LAB)
    plane_errors: list[float] = []
    for exponent in range(8, 25, 4):
        epsilon = 2.0**-exponent
        direction = np.array([np.sqrt(1.0 - epsilon**2), 0.0, -epsilon])
        result = intersect_sample_ray(
            [0.0, 0.0, epsilon],
            direction,
            lab_from_sample=identity,
            sample_width_m=4.0,
            sample_length_m=4.0,
        )
        expected = np.array([direction[0], 0.0, 0.0])
        plane_errors.append(float(np.max(np.abs(result.point_sample_m - expected))))
        if result.status is not ValidityCode.VALID:
            plane_errors.append(float("inf"))

    wavelength_A = 1.54
    k0_Ainv = 2.0 * np.pi / wavelength_A
    index = 0.8
    critical_errors: list[float] = []
    material = _material(wavelength_A, complex(index), "critical-convergence")
    for exponent in range(8, 25, 4):
        epsilon = 2.0**-exponent
        for sign in (-1.0, 1.0):
            parallel = index * (1.0 + sign * epsilon)
            direction = [parallel, 0.0, np.sqrt(1.0 - parallel**2)]
            mode = solve_incident_mode(direction, wavelength_A, material)
            expected = cmath.sqrt((index * k0_Ainv) ** 2 - (parallel * k0_Ainv) ** 2)
            if expected.imag < 0.0 or (expected.imag == 0.0 and expected.real < 0.0):
                expected = -expected
            critical_errors.append(abs(mode.kz_film_Ainv - expected))

    instrument = _instrument()
    edge_statuses = [
        detector_coordinate_to_ray(
            coordinate,
            5.0,
            origin_lab_m=np.zeros(3),
            instrument=instrument,
        ).status
        for coordinate in (
            -0.5,
            np.nextafter(-0.5, -np.inf),
            6.5,
            np.nextafter(6.5, np.inf),
        )
    ]
    return [
        _check(
            "near_parallel_plane_convergence",
            max(plane_errors) <= 1e-12,
            "analytic intersections remain stable over five powers-of-two refinements",
            maximum_error_m=float(max(plane_errors)),
            refinement_count=5,
        ),
        _check(
            "critical_mode_convergence",
            max(critical_errors) <= 5e-13 + 2e-12 * k0_Ainv,
            "complex normal roots agree from both sides of the critical boundary",
            maximum_error_Ainv=float(max(critical_errors)),
            refinement_count=10,
        ),
        _check(
            "detector_edge_boundary",
            edge_statuses
            == [
                ValidityCode.VALID,
                ValidityCode.OUTSIDE_SUPPORT,
                ValidityCode.VALID,
                ValidityCode.OUTSIDE_SUPPORT,
            ],
            "half-pixel edges are inclusive and adjacent outside floats are rejected",
            statuses=[status.value for status in edge_statuses],
        ),
    ]


def _numeric_bytes(instance: object) -> int:
    total = 0
    for field in fields(instance):
        value = getattr(instance, field.name)
        if isinstance(value, np.ndarray):
            total += value.nbytes
    return total


def _scalar_incident_oracle(
    origin_lab_m: np.ndarray,
    direction_lab: np.ndarray,
    wavelength_A: float,
    refractive_index: complex,
    instrument: CompiledInstrument,
) -> tuple[np.ndarray, complex, complex, ValidityCode]:
    """Evaluate one benchmark ray from the defining equations, without T02 kernels."""

    sample_from_lab = instrument.lab_from_sample.inverse()
    origin_sample_m = sample_from_lab.apply_point(origin_lab_m)
    direction_sample = sample_from_lab.apply_vector(direction_lab)
    denominator = float(direction_sample[2])
    offset_m = float(origin_sample_m[2])
    if abs(denominator) <= 1e-14:
        if abs(offset_m) > 1e-12:
            return np.zeros(3), 0.0j, 0.0j, ValidityCode.PARALLEL
        distance_m = 0.0
    else:
        distance_m = -offset_m / denominator
        if distance_m < -1e-12:
            return np.zeros(3), 0.0j, 0.0j, ValidityCode.BACKWARD
        distance_m = max(distance_m, 0.0)

    point_sample_m = origin_sample_m + distance_m * direction_sample
    if (
        abs(point_sample_m[0]) > 0.5 * instrument.sample_width_m
        or abs(point_sample_m[1]) > 0.5 * instrument.sample_length_m
    ):
        return np.zeros(3), 0.0j, 0.0j, ValidityCode.OUTSIDE_SUPPORT
    point_lab_m = instrument.lab_from_sample.apply_point(point_sample_m)

    k0_Ainv = 2.0 * np.pi / wavelength_A
    k_air_sample_Ainv = k0_Ainv * direction_sample
    squared_parallel = float(np.dot(k_air_sample_Ainv[:2], k_air_sample_Ainv[:2]))
    root = cmath.sqrt((refractive_index * k0_Ainv) ** 2 - squared_parallel)
    propagation_direction = 1 if k_air_sample_Ainv[2] >= 0.0 else -1
    wrong_branch = (
        propagation_direction * root.imag < 0.0
        if root.imag != 0.0
        else propagation_direction * root.real < 0.0
    )
    if wrong_branch:
        root = -root
    kz_air_Ainv = complex(k_air_sample_Ainv[2])
    denominator_Ainv = kz_air_Ainv + root
    if denominator_Ainv == 0.0:
        return point_lab_m, root, 0.0j, ValidityCode.NUMERIC_FAILURE
    amplitude = 2.0 * kz_air_Ainv / denominator_Ainv
    return point_lab_m, root, amplitude, ValidityCode.VALID


def _benchmark() -> dict[str, Any]:
    size = 512
    wavelength_A = 1.54
    instrument = _instrument(sample_width_m=4.0, sample_length_m=4.0)
    material = _material(wavelength_A, 0.999979 + 3.2e-7j, "benchmark-film")
    x = np.linspace(-1e-3, 1e-3, size)
    directions = np.column_stack((x, np.zeros(size), -np.sqrt(1.0 - x**2)))
    samples = IncidentSampleBatch(
        incident_sample_id=np.arange(size),
        origin_lab_m=np.tile([0.0, 0.0, 1.0], (size, 1)),
        direction_lab=directions,
        wavelength_A=np.full(size, wavelength_A),
        source_weight=np.full(size, 1.0 / size),
        polarization_state_id=("benchmark",) * size,
        correlation_model="independent",
    )

    scalar_points = np.empty((size, 3))
    scalar_kz = np.empty(size, dtype=np.complex128)
    scalar_amplitude = np.empty(size, dtype=np.complex128)
    scalar_status: list[ValidityCode] = []
    start = time.perf_counter()
    for row in range(size):
        point_lab_m, kz_film_Ainv, entrance_amplitude, status = _scalar_incident_oracle(
            samples.origin_lab_m[row],
            samples.direction_lab[row],
            wavelength_A,
            complex(material.n_complex[0]),
            instrument,
        )
        scalar_points[row] = point_lab_m
        scalar_kz[row] = kz_film_Ainv
        scalar_amplitude[row] = entrance_amplitude
        scalar_status.append(status)
    scalar_seconds = time.perf_counter() - start

    build_incident_states(samples, material, instrument)
    start = time.perf_counter()
    vector = build_incident_states(samples, material, instrument)
    vector_seconds = time.perf_counter() - start

    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    measured = build_incident_states(samples, material, instrument)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    point_error = float(np.max(np.abs(vector.states.sample_intersection_lab_m - scalar_points)))
    kz_error = float(np.max(np.abs(vector.states.kz_film_Ainv - scalar_kz)))
    amplitude_error = float(np.max(np.abs(vector.states.entrance_amplitude - scalar_amplitude)))
    status_agreement = vector.status == tuple(scalar_status)
    passed = (
        status_agreement
        and point_error <= 1e-12
        and kz_error <= 5e-13 + 2e-12 * (2.0 * np.pi / wavelength_A)
        and amplitude_error <= 5e-13 + 2e-12 * float(np.max(np.abs(scalar_amplitude)))
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "equivalent_work_items": size,
        "precision": {"real": "float64", "complex": "complex128"},
        "scalar_seconds": scalar_seconds,
        "vector_seconds": vector_seconds,
        "scalar_to_vector_ratio": scalar_seconds / vector_seconds,
        "maximum_point_error_m": point_error,
        "maximum_kz_error_Ainv": kz_error,
        "maximum_amplitude_error": amplitude_error,
        "status_agreement": status_agreement,
        "memory": {
            "input_numeric_bytes": _numeric_bytes(samples),
            "output_numeric_bytes": _numeric_bytes(measured.states) + measured.wavelength_A.nbytes,
            "incremental_tracemalloc_peak_bytes": max(0, peak - baseline),
            "method": "untimed call; retained numeric output is reported separately",
        },
    }


def _missing_pack_result() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": "T02",
        "status": "FAIL",
        "base_sha": _PROOF_BASE_SHA,
        "commit_sha": None,
        "contract_version": CONTRACT_API_VERSION,
        "trace_schema_version": 4,
        "reference_pack_sha256s": {},
        "environment_sha256": _environment_sha256(),
        "checks": [
            {
                "check_id": "reference_pack",
                "status": "SKIP",
                "evidence": "immutable reference pack is unavailable",
            }
        ],
        "classifications": [],
        "limitations": ["immutable reference pack is unavailable"],
        "contract_requests": [],
    }


def run_proof(*, allow_missing_pack: bool = False) -> dict[str, Any]:
    """Run the compact T02 proof without importing or executing legacy source."""

    started = time.perf_counter()
    root = Path(__file__).resolve().parents[3]
    pack_path = root / "reference/rasim_reference_v1.npz"
    if not pack_path.is_file():
        if allow_missing_pack:
            return _missing_pack_result()
        raise FileNotFoundError(pack_path)

    with np.load(pack_path, allow_pickle=False) as arrays:
        manifest = json.loads(arrays["manifest_json"].tobytes())
        classification_check = _classification_check(arrays, manifest)
        injections = _error_injections(root, arrays)
        checks = [
            _reference_integrity(pack_path, manifest),
            classification_check,
            _check(
                "required_error_injections",
                all(item["detected"] for item in injections),
                "all assigned T02 mutations fail at their first affected scientific stage",
                detected_count=sum(item["detected"] for item in injections),
                assigned_count=len(injections),
            ),
            *_osc_checks(root, arrays, manifest),
            _line_plane_check(arrays),
            _rigid_instrument_check(arrays),
            _detector_check(),
            _optics_check(arrays),
            _transport_check(),
        ]

    base_process = _git(root, "merge-base", "--is-ancestor", _PROOF_BASE_SHA, "HEAD")
    checks.insert(
        0,
        _check(
            "proof_base_ancestry",
            base_process.returncode == 0,
            "the frozen proof base is an ancestor of the current T02 candidate",
            proof_base_sha=_PROOF_BASE_SHA,
        ),
    )
    convergence = _convergence_checks()
    benchmark = _benchmark()
    scientific_passed = (
        all(check["status"] == "PASS" for check in checks)
        and all(check["status"] == "PASS" for check in convergence)
        and benchmark["status"] == "PASS"
        and all(item["detected"] for item in injections)
    )

    commit_process = _git(root, "rev-parse", "HEAD")
    status_process = _git(root, "status", "--porcelain")
    worktree_changes = [line for line in status_process.stdout.splitlines() if line.strip()]
    worktree_clean = status_process.returncode == 0 and not worktree_changes
    status = (
        "PASS"
        if scientific_passed and worktree_clean
        else "BLOCKED"
        if scientific_passed
        else "FAIL"
    )
    return {
        "schema_version": 1,
        "task_id": "T02",
        "status": status,
        "base_sha": _PROOF_BASE_SHA,
        "commit_sha": (commit_process.stdout.strip() if commit_process.returncode == 0 else None),
        "contract_version": CONTRACT_API_VERSION,
        "trace_schema_version": 4,
        "reference_pack_sha256s": {"rasim_reference_v1": _hash_file(pack_path)},
        "environment_sha256": _environment_sha256(),
        "owned_paths": [
            "src/rasim_next/io/osc.py",
            "src/rasim_next/geometry/",
            "src/rasim_next/optics/",
            "tests/test_geometry_optics.py",
            "tasks/02_geometry_optics.md",
        ],
        "checks": checks,
        "classifications": _classified_cases(),
        "convergence": convergence,
        "benchmark": benchmark,
        "limitations": [
            "exact-zero internal exit normal uses the documented +normal limiting side",
            "the v1 reference pack has a pack hash but no per-array tolerance metadata",
            "tracemalloc does not include driver or operating-system cache memory",
            "multilayer distorted fields, deposition, caking, and fitting remain outside T02",
        ],
        "contract_requests": [
            {
                "request_id": "SHARED-EVENT-FIRST-FAILURE-STATUS",
                "owner": "shared contracts/T03",
                "blocking": True,
                "blocks": "exact GEO-07 event first-failure propagation",
                "required_action": "Add an aligned lossless status to ScatteringEventBatch, require valid == (status == VALID), and have T03 supply each event failure reason.",
            },
            {
                "request_id": "SHARED-TRACE-TYPE-LAYER",
                "owner": "shared proof-base/core",
                "blocking": True,
                "blocks": "production-neutral GEO-07 public trace values",
                "required_action": "Move TraceRecord, Measure, and QuantityKind from the proof namespace to one shared production-neutral core module; keep comparator machinery proof-only.",
            },
            {
                "request_id": "SHARED-T02-TOLERANCE-POLICY",
                "owner": "shared proof-base/integration",
                "blocking": True,
                "blocks": "T02 acceptance after comparison with the shared reference pack",
                "required_action": "Publish a reviewed versioned shared stage-tolerance artifact with canonical stage keys, scale semantics, and a stable hash; T02 must load that shared artifact before acceptance.",
            },
        ],
        "tolerance_version": _TOLERANCES["version"],
        "tolerance_sha256": _hash_json(_TOLERANCES),
        "tolerances": _TOLERANCES,
        "error_injections": injections,
        "scientific_gates_passed": scientific_passed,
        "worktree_clean": worktree_clean,
        "worktree_changes": worktree_changes,
        "proof_seconds": time.perf_counter() - started,
    }
