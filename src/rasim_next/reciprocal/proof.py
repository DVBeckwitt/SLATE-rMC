"""Compact analytic, tracked-reference, and sparse-memory proof for T03."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rasim_next.core.contracts import (
    CONTRACT_API_VERSION,
    IncidentSampleBatch,
    IncidentStateBatch,
    RodCatalog,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.reciprocal.events import EventBuildResult, build_scattering_events
from rasim_next.reciprocal.ewald import EwaldRootStatus, solve_continuous_rod_ewald
from rasim_next.sampling.mosaic import (
    MosaicOrientationBatch,
    WrappedMosaicParameters,
    manuscript_axisymmetric_v1_orientation_quadrature,
)
from rasim_next.sampling.source import (
    compile_independent_source_samples,
    compile_joint_source_samples,
)

_PROOF_BASE_SHA = "812f896fde5b8365ff5c218fc606df674ad7dcad"
_REFERENCE_PACK_SHA256 = "e958703426ebea7a3fd62a8bb52447f9a5a8d7d5d4ad0eb0ce3b3706bbca1f06"
_EVENT_FILL_BYTES_PER_ROW = 96


@dataclass(frozen=True, slots=True)
class _DenseRoot:
    l_coordinate: float
    q_sample_Ainv: NDArray[np.float64]
    kf_sample_Ainv: NDArray[np.float64]
    ewald_residual_Ainv: float
    coarea_jacobian: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _environment_sha256() -> str:
    return _canonical_sha256(
        {
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "python": platform.python_version(),
        }
    )


def _verified_commit_sha(root: Path) -> str:
    git_environment = {
        name: value for name, value in os.environ.items() if not name.upper().startswith("GIT_")
    }
    identity = subprocess.run(
        ["git", "rev-parse", "--show-toplevel", "--verify", "HEAD^{commit}"],
        check=False,
        capture_output=True,
        cwd=root,
        env=git_environment,
        text=True,
        timeout=10.0,
    )
    _require(identity.returncode == 0, "cannot verify proof checkout identity")
    identity_lines = identity.stdout.splitlines()
    _require(len(identity_lines) == 2, "proof checkout identity is malformed")
    reported_root, commit_sha = identity_lines
    _require(Path(reported_root).resolve() == root.resolve(), "proof checkout root mismatch")
    _require(
        len(commit_sha) == 40
        and commit_sha == commit_sha.lower()
        and all(character in "0123456789abcdef" for character in commit_sha),
        "proof checkout commit is malformed",
    )
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        check=False,
        capture_output=True,
        cwd=root,
        env=git_environment,
        text=True,
        timeout=10.0,
    )
    _require(status.returncode == 0, "cannot verify proof checkout cleanliness")
    _require(not status.stdout, "proof checkout is dirty")
    return commit_sha


def _load_reference_pack(root: Path) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    path = root / "reference" / "rasim_reference_v1.npz"
    _require(path.is_file(), "reference pack is missing")
    pack_bytes = path.read_bytes()
    _require(
        hashlib.sha256(pack_bytes).hexdigest() == _REFERENCE_PACK_SHA256,
        "reference pack hash mismatch",
    )
    with np.load(io.BytesIO(pack_bytes), allow_pickle=False) as data:
        manifest = json.loads(data["manifest_json"].tobytes().decode("utf-8"))
        arrays = {
            name: np.array(data[name], copy=True) for name in data.files if name != "manifest_json"
        }
    return manifest, arrays


def _legacy_density(
    q_sample_Ainv: NDArray[np.float64],
    g_sample_Ainv: NDArray[np.float64],
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    sigma_rad, half_width_rad, tail_probability = map(float, parameters)
    q_elevation = np.arctan2(q_sample_Ainv[:, 2], np.linalg.norm(q_sample_Ainv[:, :2], axis=1))
    g_elevation = np.arctan2(g_sample_Ainv[2], np.linalg.norm(g_sample_Ainv[:2]))
    offset = np.remainder(q_elevation - g_elevation + np.pi, 2.0 * np.pi) - np.pi
    gaussian = np.exp(-0.5 * (offset / sigma_rad) ** 2) / (sigma_rad * np.sqrt(2.0 * np.pi))
    lorentzian = half_width_rad / (np.pi * (offset**2 + half_width_rad**2))
    line_density = (1.0 - tail_probability) * gaussian + tail_probability * lorentzian
    return line_density / (2.0 * np.pi * np.dot(g_sample_Ainv, g_sample_Ainv))


def _legacy_circle_errors(
    q_sample_Ainv: NDArray[np.float64],
    incident_sample_Ainv: NDArray[np.float64],
    scattered_norm_Ainv: float,
    reciprocal_norm_Ainv: float,
) -> tuple[float, float]:
    center = -incident_sample_Ainv
    center_norm = float(np.linalg.norm(center))
    center_hat = center / center_norm
    plane_distance = (
        reciprocal_norm_Ainv**2 + center_norm**2 - scattered_norm_Ainv**2
    ) / (2.0 * center_norm)
    circle_center = plane_distance * center_hat
    circle_radius = math.sqrt(reciprocal_norm_Ainv**2 - plane_distance**2)
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(center_hat)))] = 1.0
    first_axis = seed - np.dot(seed, center_hat) * center_hat
    first_axis /= np.linalg.norm(first_axis)
    second_axis = np.cross(center_hat, first_axis)
    relative = q_sample_Ainv - circle_center
    azimuth = np.arctan2(relative @ second_axis, relative @ first_axis)
    reconstructed = circle_center + circle_radius * (
        np.cos(azimuth)[:, None] * first_axis + np.sin(azimuth)[:, None] * second_axis
    )
    dense_azimuth = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    dense_circle = circle_center + circle_radius * (
        np.cos(dense_azimuth)[:, None] * first_axis
        + np.sin(dense_azimuth)[:, None] * second_axis
    )
    coordinate_error = float(np.max(np.abs(reconstructed - q_sample_Ainv)))
    dense_invariant_error = max(
        float(np.max(np.abs(np.linalg.norm(dense_circle, axis=1) - reciprocal_norm_Ainv))),
        float(
            np.max(
                np.abs(
                    np.linalg.norm(incident_sample_Ainv + dense_circle, axis=1)
                    - scattered_norm_Ainv
                )
            )
        ),
    )
    return coordinate_error, dense_invariant_error


def _reference_evidence(
    manifest: dict[str, Any], arrays: dict[str, NDArray[Any]]
) -> tuple[list[dict[str, object]], dict[str, float]]:
    cases = {case["case_id"]: case for case in manifest["cases"]}
    density_case = cases["mosaic.legacy_density"]
    ewald_case = cases["mosaic.ewald_intersection"]
    _require(
        density_case["classification"] == "CORRECTED"
        and density_case["first_divergence"] == "mosaic.probability_measure",
        "tracked mosaic classification mismatch",
    )
    _require(
        ewald_case["classification"] == "MATCH" and ewald_case["first_divergence"] is None,
        "tracked Ewald classification mismatch",
    )
    density = _legacy_density(
        arrays["mosaic_q_xyz"], arrays["mosaic_G"], arrays["mosaic_parameters"]
    )
    density_error = float(np.max(np.abs(density - arrays["mosaic_legacy_density"])))
    density_bound = 64.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(arrays["mosaic_legacy_density"]))), 1.0
    )
    _require(density_error <= density_bound, "tracked legacy density trace mismatch")

    incident = arrays["ewald_k_in"]
    scattered_norm = float(arrays["ewald_k_scat"])
    reciprocal_norm = float(np.linalg.norm(arrays["mosaic_G"]))
    ewald_bound = 64.0 * np.finfo(np.float64).eps * max(
        scattered_norm, reciprocal_norm, 1.0
    )
    elastic_error = 0.0
    reciprocal_error = 0.0
    coordinate_error = 0.0
    dense_circle_error = 0.0
    for prefix in ("ewald_uniform", "ewald_adaptive"):
        q = arrays[f"{prefix}_events"][:, :3]
        elastic_error = max(
            elastic_error,
            float(np.max(np.abs(np.linalg.norm(incident + q, axis=1) - scattered_norm))),
        )
        reciprocal_error = max(
            reciprocal_error,
            float(np.max(np.abs(np.linalg.norm(q, axis=1) - reciprocal_norm))),
        )
        case_coordinate_error, case_dense_circle_error = _legacy_circle_errors(
            q,
            incident,
            scattered_norm,
            reciprocal_norm,
        )
        coordinate_error = max(coordinate_error, case_coordinate_error)
        dense_circle_error = max(dense_circle_error, case_dense_circle_error)
        _require(int(arrays[f"{prefix}_status"]) == 0, "tracked Ewald status mismatch")
    _require(
        max(elastic_error, reciprocal_error, coordinate_error, dense_circle_error) <= ewald_bound,
        "tracked Ewald events disagree with the independent Bragg-circle oracle",
    )
    classifications = [
        {
            "case_id": "mosaic.ewald_intersection",
            "classification": "MATCH",
            "ledger_ids": ["PHY-REC-002", "PHY-REC-003"],
            "first_divergence_stage": None,
            "evidence": "tracked coordinates reconstruct on an independent 256-node Bragg-circle oracle",
        },
        {
            "case_id": "mosaic.legacy_density",
            "classification": "CORRECTED",
            "ledger_ids": ["PHY-MOS-001", "PHY-MOS-002", "PHY-MOS-003", "PHY-MOS-004"],
            "first_divergence_stage": "mosaic.probability_measure",
            "evidence": "tracked line-density calculation matches through the named measure divergence",
        },
        {
            "case_id": "mosaic.deterministic_source",
            "classification": "NO_ORACLE",
            "ledger_ids": ["PHY-SRC-001", "PHY-SRC-002", "PHY-SRC-003", "PHY-SRC-005", "PHY-SRC-006"],
            "first_divergence_stage": None,
            "evidence": "analytic normalization and deterministic Cartesian ordering",
        },
        {
            "case_id": "mosaic.continuous_rod_events",
            "classification": "NO_ORACLE",
            "ledger_ids": ["PHY-MOS-005", "PHY-REC-004", "PHY-REC-005", "PHY-REC-006", "PHY-REC-007", "PHY-REC-008", "PHY-REC-009"],
            "first_divergence_stage": None,
            "evidence": "analytic roots plus an independent residual-scan oracle",
        },
    ]
    return classifications, {
        "legacy_density_max_error": density_error,
        "legacy_elastic_max_error_Ainv": elastic_error,
        "legacy_reciprocal_sphere_max_error_Ainv": reciprocal_error,
        "legacy_coordinate_max_error_Ainv": coordinate_error,
        "legacy_dense_circle_max_error_Ainv": dense_circle_error,
    }


def _signed_residual(
    u_Ainv: float,
    incident: NDArray[np.float64],
    q0: NDArray[np.float64],
    direction: NDArray[np.float64],
) -> float:
    return float(np.linalg.norm(incident + q0 + u_Ainv * direction) - np.linalg.norm(incident))


def _bisect_root(
    left: float,
    right: float,
    incident: NDArray[np.float64],
    q0: NDArray[np.float64],
    direction: NDArray[np.float64],
) -> float:
    left_value = _signed_residual(left, incident, q0, direction)
    for _ in range(60):
        midpoint = 0.5 * (left + right)
        midpoint_value = _signed_residual(midpoint, incident, q0, direction)
        if midpoint_value == 0.0:
            return midpoint
        if left_value * midpoint_value < 0.0:
            right = midpoint
        else:
            left = midpoint
            left_value = midpoint_value
    return 0.5 * (left + right)


def _dense_roots(
    incident: NDArray[np.float64],
    q0: NDArray[np.float64],
    direction: NDArray[np.float64],
    b3_norm_Ainv: float,
) -> tuple[EwaldRootStatus, tuple[_DenseRoot, ...], int]:
    direction = direction / np.linalg.norm(direction)
    incident_norm = float(np.linalg.norm(incident))
    center = -float(np.dot(incident + q0, direction))
    center_residual = _signed_residual(center, incident, q0, direction)
    if center_residual > 0.0:
        return EwaldRootStatus.NO_ROOT, (), 0
    if center_residual == 0.0:
        tangent_q = q0 + center * direction
        return EwaldRootStatus.TANGENT, (), int(np.count_nonzero(tangent_q) == 0)
    coordinates = np.linspace(center - incident_norm, center + incident_norm, 65)
    residuals = np.array(
        [_signed_residual(value, incident, q0, direction) for value in coordinates]
    )
    roots: list[float] = []
    for index, (left_value, right_value) in enumerate(pairwise(residuals)):
        if left_value == 0.0:
            roots.append(float(coordinates[index]))
        if left_value * right_value < 0.0:
            roots.append(
                _bisect_root(
                    float(coordinates[index]),
                    float(coordinates[index + 1]),
                    incident,
                    q0,
                    direction,
                )
            )
    if residuals[-1] == 0.0:
        roots.append(float(coordinates[-1]))
    roots = sorted(set(roots))
    _require(len(roots) == 2, "dense regular line did not find two roots")
    direct_count = 0
    dense_roots: list[_DenseRoot] = []
    for root in roots:
        q = np.asarray(q0 + root * direction, dtype=np.float64)
        if np.count_nonzero(q) == 0:
            direct_count += 1
            continue
        kf = np.asarray(incident + q, dtype=np.float64)
        kf_norm = float(np.linalg.norm(kf))
        derivative = abs(float(np.dot(kf / kf_norm, direction)))
        _require(derivative > 0.0, "dense regular root has zero derivative")
        dense_roots.append(
            _DenseRoot(
                root / b3_norm_Ainv,
                q,
                kf,
                abs(kf_norm - incident_norm),
                1.0 / derivative,
            )
        )
    return EwaldRootStatus.TWO_ROOT, tuple(dense_roots), direct_count


def _build_sparse_fixture() -> tuple[
    IncidentSampleBatch,
    IncidentStateBatch,
    RodCatalog,
    MosaicOrientationBatch,
    RigidTransform,
]:
    rod_count = 16
    samples = IncidentSampleBatch(
        incident_sample_id=np.array([0], dtype=np.int64),
        origin_lab_m=np.zeros((1, 3)),
        direction_lab=np.array([[0.0, 0.0, 1.0]]),
        wavelength_A=np.array([np.pi / 2.0]),
        source_weight=np.array([1.0]),
        polarization_state_id=("unity_scalar",),
        correlation_model="explicit_joint",
    )
    states = IncidentStateBatch(
        incident_state_id=np.array([100], dtype=np.int64),
        incident_sample_id=np.array([0], dtype=np.int64),
        sample_intersection_lab_m=np.zeros((1, 3)),
        direction_sample=np.array([[0.0, 0.0, 1.0]]),
        k_air_sample_Ainv=np.array([[0.0, 0.0, 7.0]]),
        k_film_phase_sample_Ainv=np.array([[0.0, 0.0, 4.0]]),
        kz_film_Ainv=np.array([4.0 + 0.0j]),
        entrance_amplitude=np.array([1.0 + 0.0j]),
        footprint_acceptance=np.array([1.0]),
        source_weight=np.array([1.0]),
        valid=np.array([True]),
    )
    basis = np.diag([1.0, 1.0, 2.0])
    h = np.arange(5, rod_count + 5, dtype=np.int32)
    h[rod_count // 2 - 1 : rod_count // 2 + 2] = (0, 1, 4)
    rods = RodCatalog(
        rod_id=np.arange(10_000, 10_000 + rod_count, dtype=np.int64),
        phase_id=("sparse",) * rod_count,
        h=h,
        k=np.zeros(rod_count, dtype=np.int32),
        family_id=("sparse",) * rod_count,
        family_key=("sparse",) * rod_count,
        qr_Ainv=h.astype(np.float64),
        reciprocal_basis_Ainv=basis,
        symmetry_metadata=("none",) * rod_count,
    )
    orientations = MosaicOrientationBatch(
        orientation_id=np.array([0], dtype=np.int64),
        alpha_rad=np.array([0.0]),
        azimuth_rad=np.array([0.0]),
        rotation_crystal=np.eye(3)[None, :, :],
        probability_mass=np.array([1.0]),
        reciprocal_basis_Ainv=basis,
        model_id="manuscript_axisymmetric_v1",
    )
    transform = RigidTransform(np.eye(3), np.zeros(3), FrameId.CRYSTAL, FrameId.SAMPLE)
    return samples, states, rods, orientations, transform


def _dense_sparse_events(
    samples: IncidentSampleBatch,
    states: IncidentStateBatch,
    rods: RodCatalog,
    orientations: MosaicOrientationBatch,
) -> dict[str, object]:
    incident = states.k_film_phase_sample_Ainv[0]
    basis = rods.reciprocal_basis_Ainv
    b3_norm = float(np.linalg.norm(basis[:, 2]))
    direction = basis[:, 2] / b3_norm
    wavelength = float(samples.wavelength_A[0])
    event_rod_ids: list[int] = []
    q_values: list[NDArray[np.float64]] = []
    l_values: list[float] = []
    kf_values: list[NDArray[np.float64]] = []
    weights: list[float] = []
    residuals: list[float] = []
    statuses: list[EwaldRootStatus] = []
    emitted_counts: list[int] = []
    direct_counts: list[int] = []
    for rod_id, h, k in zip(rods.rod_id, rods.h, rods.k, strict=True):
        q0 = int(h) * basis[:, 0] + int(k) * basis[:, 1]
        status, roots, direct_count = _dense_roots(incident, q0, direction, b3_norm)
        statuses.append(status)
        emitted_counts.append(len(roots))
        direct_counts.append(direct_count)
        for root in roots:
            event_rod_ids.append(int(rod_id))
            q_values.append(root.q_sample_Ainv)
            l_values.append(root.l_coordinate)
            kf_values.append(root.kf_sample_Ainv)
            weights.append(float(orientations.probability_mass[0]) * root.coarea_jacobian)
            residuals.append(root.ewald_residual_Ainv)
    event_count = len(q_values)
    attempt_count = len(statuses)
    q_array = np.asarray(q_values, dtype=np.float64).reshape((-1, 3))
    return {
        "event_id": np.arange(event_count, dtype=np.int64),
        "event_state_id": np.full(event_count, 100, dtype=np.int64),
        "event_rod_id": np.asarray(event_rod_ids, dtype=np.int64),
        "wavelength": np.full(event_count, wavelength),
        "q": q_array,
        "qz": q_array[:, 2],
        "l": np.asarray(l_values),
        "kf": np.asarray(kf_values).reshape((-1, 3)),
        "weight": np.asarray(weights),
        "residual": np.asarray(residuals),
        "valid": np.ones(event_count, dtype=np.bool_),
        "attempt_id": np.arange(attempt_count, dtype=np.int64),
        "attempt_state_id": np.full(attempt_count, 100, dtype=np.int64),
        "attempt_rod_id": np.array(rods.rod_id, copy=True),
        "attempt_orientation_id": np.zeros(attempt_count, dtype=np.int64),
        "status": tuple(statuses),
        "emitted_count": np.asarray(emitted_counts, dtype=np.int8),
        "direct_count": np.asarray(direct_counts, dtype=np.int8),
    }


def _require_exact_public_dense(public: EventBuildResult, dense: dict[str, object]) -> None:
    event_fields = {
        "event_id": "event_id",
        "incident_state_id": "event_state_id",
        "rod_id": "event_rod_id",
        "wavelength_A": "wavelength",
        "q_internal_sample_Ainv": "q",
        "qz_Ainv": "qz",
        "l_coordinate": "l",
        "kf_film_phase_sample_Ainv": "kf",
        "reciprocal_weight": "weight",
        "ewald_residual_Ainv": "residual",
        "valid": "valid",
    }
    status_fields = {
        "attempt_id": "attempt_id",
        "incident_state_id": "attempt_state_id",
        "rod_id": "attempt_rod_id",
        "orientation_id": "attempt_orientation_id",
        "emitted_root_count": "emitted_count",
        "direct_beam_root_count": "direct_count",
    }
    _require(
        all(
            np.array_equal(getattr(public.events, public_name), dense[dense_name])
            for public_name, dense_name in event_fields.items()
        )
        and all(
            np.array_equal(getattr(public.status, public_name), dense[dense_name])
            for public_name, dense_name in status_fields.items()
        )
        and public.status.root_status == dense["status"],
        "public sparse events differ from the ordered independent oracle",
    )


def _sparse_event_evidence() -> dict[str, object]:
    samples, states, rods, orientations, transform = _build_sparse_fixture()

    def build() -> EventBuildResult:
        return build_scattering_events(
            incident_samples=samples,
            incident_states=states,
            rods=rods,
            orientations=orientations,
            sample_from_crystal=transform,
        )

    initial = build()
    dense = _dense_sparse_events(samples, states, rods, orientations)
    _require_exact_public_dense(initial, dense)
    attempt_count = int(initial.status.attempt_id.size)
    event_count = int(initial.events.event_id.size)
    _require(
        attempt_count == 16
        and event_count == 3
        and initial.status.root_status.count(EwaldRootStatus.NO_ROOT) == 13
        and initial.status.root_status.count(EwaldRootStatus.TWO_ROOT) == 2
        and initial.status.root_status.count(EwaldRootStatus.TANGENT) == 1,
        "sparse fixture support changed",
    )
    event_fields = (
        "event_id",
        "incident_state_id",
        "rod_id",
        "wavelength_A",
        "q_internal_sample_Ainv",
        "qz_Ainv",
        "l_coordinate",
        "kf_film_phase_sample_Ainv",
        "reciprocal_weight",
        "ewald_residual_Ainv",
        "valid",
    )
    status_fields = (
        "attempt_id",
        "incident_state_id",
        "rod_id",
        "orientation_id",
        "emitted_root_count",
        "direct_beam_root_count",
    )
    event_bytes = sum(getattr(initial.events, name).nbytes for name in event_fields)
    status_bytes = sum(getattr(initial.status, name).nbytes for name in status_fields)
    actual_fill_bytes = _EVENT_FILL_BYTES_PER_ROW * event_count
    forbidden_cartesian_bytes = 2 * _EVENT_FILL_BYTES_PER_ROW * attempt_count
    return {
        "workload": {
            "attempted_lines": attempt_count,
            "emitted_events": event_count,
            "suppressed_direct_roots": int(initial.status.direct_beam_root_count.sum()),
        },
        "ordered_dense_oracle_match": True,
        "returned_event_numpy_element_bytes": int(event_bytes),
        "returned_status_numpy_element_bytes": int(status_bytes),
        "actual_event_fill_numpy_element_bytes": actual_fill_bytes,
        "rejected_two_root_cartesian_numpy_element_bytes": forbidden_cartesian_bytes,
    }


def _scientific_evidence() -> tuple[list[dict[str, str]], dict[str, float]]:
    joint = compile_joint_source_samples(
        origin_lab_m=np.array([[0.0, 0.0, 0.0], [0.0, 1.0e-3, 0.0]]),
        direction_lab=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        wavelength_A=np.array([1.0, 1.5]),
        probability_mass=np.array([0.25, 0.75]),
        polarization_state_id=("unity_scalar", "unity_scalar"),
    )
    independent = compile_independent_source_samples(
        origin_lab_m=np.array([[0.0, 0.0, 0.0], [0.0, 1.0e-3, 0.0]]),
        origin_probability_mass=np.array([0.25, 0.75]),
        direction_lab=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        direction_probability_mass=np.array([0.5, 0.5]),
        wavelength_A=np.array([1.0]),
        wavelength_probability_mass=np.array([1.0]),
        polarization_state_id="unity_scalar",
    )
    source_mass_error = max(
        abs(float(joint.source_weight.sum()) - 1.0),
        abs(float(independent.source_weight.sum()) - 1.0),
    )
    _require(
        source_mass_error == 0.0
        and joint.correlation_model == "explicit_joint"
        and independent.correlation_model == "independent_product"
        and joint.incident_sample_id.size == 2
        and independent.incident_sample_id.size == 4,
        "source mass, correlation, or ordering invariant failed",
    )

    parameters = WrappedMosaicParameters(0.0, 0.3, 0.25)
    orientations = manuscript_axisymmetric_v1_orientation_quadrature(
        parameters,
        reciprocal_basis_Ainv=np.eye(3),
        alpha_cell_count=4,
        azimuth_cell_count=4,
    )
    atom_mass = float(orientations.probability_mass[orientations.alpha_rad == 0.0].sum())
    orientation_mass_error = abs(float(orientations.probability_mass.sum()) - 1.0)
    _require(
        np.all(orientations.probability_mass >= 0.0)
        and atom_mass == 0.75
        and orientation_mass_error <= np.spacing(1.0),
        "folded spherical mosaic probability mass failed",
    )

    incident = np.array([0.0, 0.0, 4.0])
    direction = np.array([0.0, 0.0, 1.0])
    two = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([1.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    expected_jacobian = 4.0 / math.sqrt(15.0)
    jacobian_error = max(
        abs(root.coarea_jacobian - expected_jacobian) for root in two.emittable_roots
    )
    tangent = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([4.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    no_root = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([5.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    direct = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.zeros(3),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    direct_tangent = solve_continuous_rod_ewald(
        ki_sample_Ainv=np.array([4.0, 0.0, 0.0]),
        q0_sample_Ainv=np.zeros(3),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    near_tangent = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([np.nextafter(4.0, 0.0), 0.0, 2.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    _require(
        two.status is EwaldRootStatus.TWO_ROOT
        and len(two.emittable_roots) == 2
        and jacobian_error <= 4.0 * abs(np.spacing(expected_jacobian))
        and tangent.status is EwaldRootStatus.TANGENT
        and no_root.status is EwaldRootStatus.NO_ROOT
        and direct.status is EwaldRootStatus.TWO_ROOT
        and len(direct.emittable_roots) == 1
        and direct.direct_beam_root_count == 1
        and direct_tangent.status is EwaldRootStatus.TANGENT
        and direct_tangent.direct_beam_root_count == 1
        and min(root.coarea_jacobian for root in near_tangent.emittable_roots) > 1.0e6,
        "analytic Ewald status, direct-root, or Jacobian invariant failed",
    )
    residual_max = max(root.ewald_residual_Ainv for root in two.emittable_roots)
    residual_bound = 64.0 * np.finfo(np.float64).eps * 4.0
    _require(residual_max <= residual_bound, "unsquared elastic residual failed")
    return [
        {
            "check_id": "source_probability",
            "status": "PASS",
            "evidence": "joint and independent source masses/correlation models preserve deterministic ordering",
        },
        {
            "check_id": "spherical_mosaic_measure",
            "status": "PASS",
            "evidence": "mixed atom plus continuous folded-alpha mass is nonnegative and normalized",
        },
        {
            "check_id": "analytic_ewald",
            "status": "PASS",
            "evidence": "two/tangent/no-root, direct suppression, unsquared residual, and unclipped coarea Jacobian pass",
        },
    ], {
        "source_mass_error": source_mass_error,
        "orientation_mass_error": orientation_mass_error,
        "zero_tilt_atom_mass": atom_mass,
        "analytic_jacobian_error": jacobian_error,
        "elastic_residual_max_Ainv": residual_max,
    }


def run_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    """Run the compact T03 proof without writing diagnostics."""

    del allow_missing_pack
    root = Path(__file__).resolve().parents[3]
    proof_base = os.environ.get("PROOF_BASE_SHA", "")
    _require(proof_base == _PROOF_BASE_SHA, "PROOF_BASE_SHA is unset or does not match T03")
    commit_sha = _verified_commit_sha(root)
    manifest, arrays = _load_reference_pack(root)
    classifications, reference_metrics = _reference_evidence(manifest, arrays)
    scientific_checks, scientific_metrics = _scientific_evidence()
    sparse_evidence = _sparse_event_evidence()
    checks = [
        {
            "check_id": "tracked_reference",
            "status": "PASS",
            "evidence": "immutable pack hash, legacy density trace, and two-sphere coordinates pass",
        },
        *scientific_checks,
        {
            "check_id": "sparse_event_memory",
            "status": "PASS",
            "evidence": "16 attempted lines and 3 events match every ordered independent-oracle field; production allocates exact event rows",
        },
    ]
    return {
        "schema_version": 1,
        "task_id": "T03",
        "status": "READY",
        "base_sha": proof_base,
        "commit_sha": commit_sha,
        "contract_version": CONTRACT_API_VERSION,
        "trace_schema_version": 4,
        "reference_pack_sha256s": {"rasim_reference_v1": _REFERENCE_PACK_SHA256},
        "environment_sha256": _environment_sha256(),
        "owned_paths": [
            "src/rasim_next/sampling/source.py",
            "src/rasim_next/sampling/mosaic.py",
            "src/rasim_next/reciprocal/ewald.py",
            "src/rasim_next/reciprocal/events.py",
            "src/rasim_next/reciprocal/proof.py",
            "tests/test_mosaic_ewald.py",
            "tasks/03_mosaic_ewald.md",
        ],
        "checks": checks,
        "metrics": {**reference_metrics, **scientific_metrics, "sparse_event": sparse_evidence},
        "classifications": classifications,
        "convergence": [],
        "benchmark": None,
        "limitations": [
            "UNITY_APPROXIMATION: polarization_state_id='unity_scalar' carries no Stokes or polarization factor",
            "reciprocal_weight contains orientation probability mass times exactly one coarea Jacobian; optics and detector factors remain downstream",
            "localized/adaptive acceleration and a generic SO(3) sampler are not implemented",
        ],
        "contract_requests": [
            {
                "request_id": "T03-SHARED-TOLERANCE-MEASURE-CONTRACT",
                "owner": "proof-base/integration",
                "required_action": "publish a versioned shared tolerance and result-measure contract before cross-workstream integration",
                "evidence": "T03 retains analytic and machine-precision gates only; no shared versioned tolerance artifact exists in the proof base",
                "blocking_for_local_merge": False,
            }
        ],
    }
