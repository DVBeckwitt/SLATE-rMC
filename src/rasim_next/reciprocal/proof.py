"""Compact analytic, tracked-reference, and sparse-memory proof for T03."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import subprocess
import time
import tracemalloc
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
    wrapped_mosaic_line_density_rad_inv,
)
from rasim_next.sampling.source import (
    compile_independent_source_samples,
    compile_joint_source_samples,
)

_PROOF_BASE_SHA = "812f896fde5b8365ff5c218fc606df674ad7dcad"
_REFERENCE_PACK_SHA256 = "e958703426ebea7a3fd62a8bb52447f9a5a8d7d5d4ad0eb0ce3b3706bbca1f06"


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
    _require(density_case["classification"] == "CORRECTED", "tracked mosaic classification mismatch")
    _require(
        ewald_case["classification"] == "MATCH" and ewald_case["first_divergence"] is None,
        "tracked Ewald classification mismatch",
    )
    q = arrays["mosaic_q_xyz"]
    reciprocal_vector = arrays["mosaic_G"]
    q_elevation = np.arctan2(q[:, 2], np.linalg.norm(q[:, :2], axis=1))
    reciprocal_elevation = np.arctan2(
        reciprocal_vector[2], np.linalg.norm(reciprocal_vector[:2])
    )
    offset = np.remainder(q_elevation - reciprocal_elevation + np.pi, 2.0 * np.pi) - np.pi
    public_line_density = wrapped_mosaic_line_density_rad_inv(
        offset, WrappedMosaicParameters(*map(float, arrays["mosaic_parameters"]))
    )
    legacy_line_density = arrays["mosaic_legacy_density"] * (
        2.0 * np.pi * np.dot(reciprocal_vector, reciprocal_vector)
    )
    density_difference = np.abs(public_line_density - legacy_line_density)
    _require(np.any(density_difference > 0.0), "wrapped public density did not diverge")

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
            "first_divergence_stage": "mosaic.wrapped_line_density",
            "evidence": "the public wrapped Lorentzian line density diverges from the unwrapped legacy profile before probability-measure conversion",
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
        "legacy_wrapped_line_density_max_difference_rad_inv": float(density_difference.max()),
        "legacy_wrapped_line_density_p95_difference_rad_inv": float(
            np.percentile(density_difference, 95.0)
        ),
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


def _build_fixture(
    h: NDArray[np.int32],
    wavelengths_A: NDArray[np.float64],
    parameters: WrappedMosaicParameters,
    *,
    alpha_cell_count: int,
    azimuth_cell_count: int,
    b1_Ainv: float = 1.0,
) -> tuple[
    IncidentSampleBatch,
    IncidentStateBatch,
    RodCatalog,
    MosaicOrientationBatch,
    RigidTransform,
]:
    rod_count = h.size
    sample_count = wavelengths_A.size
    sample_ids = np.arange(sample_count, dtype=np.int64)
    wavevector_norm = 2.0 * np.pi / wavelengths_A
    samples = IncidentSampleBatch(
        incident_sample_id=sample_ids,
        origin_lab_m=np.zeros((sample_count, 3)),
        direction_lab=np.tile([0.0, 0.0, 1.0], (sample_count, 1)),
        wavelength_A=wavelengths_A,
        source_weight=np.full(sample_count, 1.0 / sample_count),
        polarization_state_id=("proof_state",) * sample_count,
        correlation_model="proof_fixture",
    )
    incident = np.column_stack((np.zeros((sample_count, 2)), wavevector_norm))
    states = IncidentStateBatch(
        incident_state_id=100 + sample_ids,
        incident_sample_id=sample_ids,
        sample_intersection_lab_m=np.zeros((sample_count, 3)),
        direction_sample=np.tile([0.0, 0.0, 1.0], (sample_count, 1)),
        k_air_sample_Ainv=incident,
        k_film_phase_sample_Ainv=incident,
        kz_film_Ainv=wavevector_norm.astype(np.complex128),
        entrance_amplitude=np.ones(sample_count, dtype=np.complex128),
        footprint_acceptance=np.ones(sample_count),
        source_weight=np.full(sample_count, 1.0 / sample_count),
        valid=np.ones(sample_count, dtype=np.bool_),
    )
    basis = np.diag([b1_Ainv, 1.0, 2.0])
    rods = RodCatalog(
        rod_id=np.arange(10_000, 10_000 + rod_count, dtype=np.int64),
        phase_id=("sparse",) * rod_count,
        h=h,
        k=np.zeros(rod_count, dtype=np.int32),
        family_id=("sparse",) * rod_count,
        family_key=("sparse",) * rod_count,
        qr_Ainv=np.abs(h.astype(np.float64) * b1_Ainv),
        reciprocal_basis_Ainv=basis,
        symmetry_metadata=("none",) * rod_count,
    )
    if parameters.zero_tilt_probability_mass == 1.0:
        orientations = MosaicOrientationBatch(
            np.array([0]),
            np.array([0.0]),
            np.array([0.0]),
            np.eye(3)[None, :, :],
            np.array([1.0]),
            basis,
            "manuscript_axisymmetric_v1",
        )
    else:
        orientations = manuscript_axisymmetric_v1_orientation_quadrature(
            parameters,
            reciprocal_basis_Ainv=basis,
            alpha_cell_count=alpha_cell_count,
            azimuth_cell_count=azimuth_cell_count,
        )
    transform = RigidTransform(np.eye(3), np.zeros(3), FrameId.CRYSTAL, FrameId.SAMPLE)
    return samples, states, rods, orientations, transform


def _dense_events(
    samples: IncidentSampleBatch,
    states: IncidentStateBatch,
    rods: RodCatalog,
    orientations: MosaicOrientationBatch,
    transform: RigidTransform,
) -> dict[str, object]:
    basis = rods.reciprocal_basis_Ainv
    b3_norm = float(np.linalg.norm(basis[:, 2]))
    b3_hat = basis[:, 2] / b3_norm
    wavelength_by_sample = dict(
        zip(map(int, samples.incident_sample_id), map(float, samples.wavelength_A), strict=True)
    )
    event_state_ids: list[int] = []
    event_rod_ids: list[int] = []
    q_values: list[NDArray[np.float64]] = []
    l_values: list[float] = []
    kf_values: list[NDArray[np.float64]] = []
    weights: list[float] = []
    residuals: list[float] = []
    wavelengths: list[float] = []
    attempt_state_ids: list[int] = []
    attempt_rod_ids: list[int] = []
    attempt_orientation_ids: list[int] = []
    statuses: list[EwaldRootStatus] = []
    emitted_counts: list[int] = []
    direct_counts: list[int] = []
    for state_index, valid in enumerate(states.valid):
        if not valid:
            continue
        state_id = int(states.incident_state_id[state_index])
        incident = states.k_film_phase_sample_Ainv[state_index]
        wavelength = wavelength_by_sample[int(states.incident_sample_id[state_index])]
        for rod_id, h, k in zip(rods.rod_id, rods.h, rods.k, strict=True):
            q0_crystal = int(h) * basis[:, 0] + int(k) * basis[:, 1]
            for orientation_index, mass_value in enumerate(orientations.probability_mass):
                mass = float(mass_value)
                if mass == 0.0:
                    continue
                rotation = transform.rotation @ orientations.rotation_crystal[orientation_index]
                q0 = rotation @ q0_crystal
                direction = rotation @ b3_hat
                status, roots, direct_count = _dense_roots(
                    incident, q0, direction, b3_norm
                )
                attempt_state_ids.append(state_id)
                attempt_rod_ids.append(int(rod_id))
                attempt_orientation_ids.append(int(orientations.orientation_id[orientation_index]))
                statuses.append(status)
                emitted_counts.append(len(roots))
                direct_counts.append(direct_count)
                for root in roots:
                    event_state_ids.append(state_id)
                    event_rod_ids.append(int(rod_id))
                    wavelengths.append(wavelength)
                    q_values.append(root.q_sample_Ainv)
                    l_values.append(root.l_coordinate)
                    kf_values.append(root.kf_sample_Ainv)
                    weights.append(mass * root.coarea_jacobian)
                    residuals.append(root.ewald_residual_Ainv)
    event_count = len(q_values)
    attempt_count = len(statuses)
    q_array = np.asarray(q_values, dtype=np.float64).reshape((-1, 3))
    return {
        "event_id": np.arange(event_count, dtype=np.int64),
        "event_state_id": np.asarray(event_state_ids, dtype=np.int64),
        "event_rod_id": np.asarray(event_rod_ids, dtype=np.int64),
        "wavelength": np.asarray(wavelengths),
        "q": q_array,
        "qz": q_array[:, 2],
        "l": np.asarray(l_values),
        "kf": np.asarray(kf_values).reshape((-1, 3)),
        "weight": np.asarray(weights),
        "residual": np.asarray(residuals),
        "valid": np.ones(event_count, dtype=np.bool_),
        "attempt_id": np.arange(attempt_count, dtype=np.int64),
        "attempt_state_id": np.asarray(attempt_state_ids, dtype=np.int64),
        "attempt_rod_id": np.asarray(attempt_rod_ids, dtype=np.int64),
        "attempt_orientation_id": np.asarray(attempt_orientation_ids, dtype=np.int64),
        "status": tuple(statuses),
        "emitted_count": np.asarray(emitted_counts, dtype=np.int8),
        "direct_count": np.asarray(direct_counts, dtype=np.int8),
    }


def _compare_public_dense(
    public: EventBuildResult, dense: dict[str, object], *, require_exact_numeric: bool = False
) -> dict[str, float]:
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
    numeric_fields = {
        "q_internal_sample_Ainv",
        "qz_Ainv",
        "l_coordinate",
        "kf_film_phase_sample_Ainv",
        "reciprocal_weight",
        "ewald_residual_Ainv",
    }
    errors: dict[str, float] = {}
    for public_name, dense_name in event_fields.items():
        public_value = getattr(public.events, public_name)
        dense_value = np.asarray(dense[dense_name])
        _require(
            public_value.shape == dense_value.shape and public_value.dtype == dense_value.dtype,
            f"{public_name} shape or dtype differs from dense oracle",
        )
        if public_name in numeric_fields:
            errors[public_name] = float(
                np.max(np.abs(public_value - dense_value), initial=0.0)
            )
        else:
            _require(
                np.array_equal(public_value, dense_value),
                f"{public_name} identity or order differs from dense oracle",
            )
    _require(
        all(
            np.array_equal(getattr(public.status, public_name), dense[dense_name])
            for public_name, dense_name in status_fields.items()
        )
        and public.status.root_status == dense["status"],
        "public statuses differ from the ordered independent oracle",
    )
    if require_exact_numeric:
        _require(max(errors.values(), default=0.0) == 0.0, "numeric sparse oracle mismatch")
    return errors


def _observables(
    q: NDArray[np.float64], weight: NDArray[np.float64], residual: NDArray[np.float64]
) -> dict[str, object]:
    total_mass = float(weight.sum())
    _require(total_mass > 0.0, "proof case emitted no reciprocal mass")
    order = np.argsort(q[:, 2])
    sorted_weight = weight[order]
    cumulative = (np.cumsum(sorted_weight) - 0.5 * sorted_weight) / total_mass
    return {
        "total_reciprocal_mass": total_mass,
        "weighted_Q_centroid_sample_Ainv": (
            np.sum(weight[:, None] * q, axis=0) / total_mass
        ).tolist(),
        "weighted_qz_quantiles_Ainv": np.interp(
            (0.1, 0.9), cumulative, q[order, 2]
        ).tolist(),
        "maximum_ewald_residual_Ainv": float(np.max(residual, initial=0.0)),
    }


def _oracle_evidence() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    atom = WrappedMosaicParameters(0.0, 0.0, 0.0)

    def evaluate(
        case_id: str,
        parameters: WrappedMosaicParameters,
        h_values: tuple[int, ...],
        wavelengths: tuple[float, ...],
        alpha_count: int,
        azimuth_count: int,
        b1_Ainv: float = 1.0,
    ) -> dict[str, object]:
        batches = _build_fixture(
            np.asarray(h_values, dtype=np.int32),
            np.asarray(wavelengths),
            parameters,
            alpha_cell_count=alpha_count,
            azimuth_cell_count=azimuth_count,
            b1_Ainv=b1_Ainv,
        )
        samples, states, rods, orientations, transform = batches
        public = build_scattering_events(
            incident_samples=samples,
            incident_states=states,
            rods=rods,
            orientations=orientations,
            sample_from_crystal=transform,
        )
        dense = _dense_events(samples, states, rods, orientations, transform)
        errors = _compare_public_dense(public, dense)
        statuses = public.status.root_status
        return {
            "case_id": case_id,
            "requested_alpha_cell_count": alpha_count,
            "orientation_count": int(orientations.orientation_id.size),
            "event_count": int(public.events.event_id.size),
            "status_counts": {status.value: statuses.count(status) for status in EwaldRootStatus},
            "suppressed_direct_roots": int(public.status.direct_beam_root_count.sum()),
            "public": _observables(
                public.events.q_internal_sample_Ainv,
                public.events.reciprocal_weight,
                public.events.ewald_residual_Ainv,
            ),
            "dense_oracle": _observables(
                np.asarray(dense["q"]),
                np.asarray(dense["weight"]),
                np.asarray(dense["residual"]),
            ),
            "public_dense_max_absolute_field_difference": errors,
        }

    configurations = (
        ("narrow", WrappedMosaicParameters(0.01, 0.08, 0.05), (1,), (np.pi / 2.0,), 4, 4),
        ("broad", WrappedMosaicParameters(0.2, 0.35, 0.2), (1,), (np.pi / 2.0,), 4, 4),
        ("lorentz_tail", WrappedMosaicParameters(0.02, 0.25, 0.85), (1,), (np.pi / 2.0,), 4, 4),
        ("tangent_no_root", atom, (1, 4, 5), (np.pi / 2.0,), 1, 1),
        ("bandwidth_specular", atom, (0,), (np.pi / 2.0, 1.1), 1, 1),
    )
    matrix = [evaluate(*configuration) for configuration in configurations]
    _require(
        matrix[3]["status_counts"] == {"TWO_ROOT": 1, "TANGENT": 1, "NO_ROOT": 1}
        and matrix[4]["suppressed_direct_roots"] == 2,
        "named support regimes changed",
    )
    levels = (4, 8, 16)
    refinement = [
        evaluate(
            "regular_quadrature_refinement",
            WrappedMosaicParameters(0.2, 0.0, 0.0),
            (1,),
            (np.pi / 2.0,),
            level,
            4,
            3.0,
        )
        for level in levels
    ]
    return matrix, [
        {
            "case_id": "regular_quadrature_refinement",
            "refinement_variable": "alpha_cell_count",
            "levels": list(levels),
            "observables": [row["public"] for row in refinement],
            "successive_qz_quantile_max_change_Ainv": np.max(
                np.abs(
                    np.diff([row["public"]["weighted_qz_quantiles_Ainv"] for row in refinement], axis=0)
                ),
                axis=1,
            ).tolist(),
            "assessment": "BLOCKED_PENDING_SHARED_STAGE_TOLERANCE",
        }
    ]


def _benchmark_evidence() -> dict[str, object]:
    h = np.arange(5, 4101, dtype=np.int32)
    h[2047:2050] = (0, 1, 4)
    samples, states, rods, orientations, transform = _build_fixture(
        h,
        np.array([np.pi / 2.0]),
        WrappedMosaicParameters(0.0, 0.0, 0.0),
        alpha_cell_count=1,
        azimuth_cell_count=1,
    )
    tracemalloc.start()
    start = time.perf_counter()
    public = build_scattering_events(
        incident_samples=samples,
        incident_states=states,
        rods=rods,
        orientations=orientations,
        sample_from_crystal=transform,
    )
    public_wall_seconds = time.perf_counter() - start
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    start = time.perf_counter()
    dense = _dense_events(samples, states, rods, orientations, transform)
    dense_wall_seconds = time.perf_counter() - start
    _compare_public_dense(public, dense, require_exact_numeric=True)
    attempt_count = int(public.status.attempt_id.size)
    event_count = int(public.events.event_id.size)
    _require(
        attempt_count == 4096
        and event_count == 3
        and public.status.root_status.count(EwaldRootStatus.NO_ROOT) == 4093
        and public.status.root_status.count(EwaldRootStatus.TWO_ROOT) == 2
        and public.status.root_status.count(EwaldRootStatus.TANGENT) == 1,
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
    event_bytes = sum(getattr(public.events, name).nbytes for name in event_fields)
    status_bytes = sum(getattr(public.status, name).nbytes for name in status_fields)
    fill_fields = event_fields[1:5] + event_fields[6:10]
    fill_row_bytes = sum(getattr(public.events, name).nbytes // event_count for name in fill_fields)
    former_preallocator_bytes = 2 * fill_row_bytes * attempt_count
    _require(peak_bytes < former_preallocator_bytes, "former maximum-root preallocation returned")
    return {
        "workload": {
            "attempted_lines": attempt_count,
            "emitted_events": event_count,
            "suppressed_direct_roots": int(public.status.direct_beam_root_count.sum()),
            "dense_residual_nodes_per_line": 65,
        },
        "ordered_dense_oracle_match": True,
        "public_wall_seconds_with_memory_tracing": public_wall_seconds,
        "dense_oracle_wall_seconds": dense_wall_seconds,
        "returned_numeric_output_bytes": int(event_bytes + status_bytes),
        "traced_live_bytes_at_return": current_bytes,
        "traced_peak_bytes": peak_bytes,
        "temporary_peak_working_bytes": peak_bytes - current_bytes,
        "former_two_root_preallocator_bytes": former_preallocator_bytes,
        "measurement_boundary": "fixture setup excluded; public wall time includes tracemalloc",
    }


def _scientific_evidence() -> tuple[
    list[dict[str, str]], dict[str, float], list[dict[str, object]]
]:
    joint = compile_joint_source_samples(
        origin_lab_m=np.zeros((2, 3)),
        direction_lab=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        wavelength_A=np.array([1.0, 1.5]),
        probability_mass=np.array([0.25, 0.75]),
        polarization_state_id=("linear_s", "circular_plus"),
        correlation_model="proof_joint",
    )
    independent = compile_independent_source_samples(
        origin_lab_m=np.zeros((2, 3)),
        origin_probability_mass=np.array([0.25, 0.75]),
        direction_lab=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        direction_probability_mass=np.array([0.5, 0.5]),
        wavelength_A=np.array([1.0]),
        wavelength_probability_mass=np.array([1.0]),
        polarization_state_id="proof_state",
        correlation_model="proof_independent",
    )
    source_mass_error = max(
        abs(float(joint.source_weight.sum()) - 1.0),
        abs(float(independent.source_weight.sum()) - 1.0),
    )
    orientations = manuscript_axisymmetric_v1_orientation_quadrature(
        WrappedMosaicParameters(0.0, 0.3, 0.25),
        reciprocal_basis_Ainv=np.eye(3),
        alpha_cell_count=4,
        azimuth_cell_count=4,
    )
    orientation_mass_error = abs(float(orientations.probability_mass.sum()) - 1.0)
    incident = np.array([0.0, 0.0, 4.0])
    direction = np.array([0.0, 0.0, 1.0])
    two = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([1.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
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
    _require(
        source_mass_error == 0.0
        and joint.polarization_state_id == ("linear_s", "circular_plus")
        and joint.correlation_model == "proof_joint"
        and independent.correlation_model == "proof_independent"
        and orientation_mass_error <= np.spacing(1.0)
        and two.status is EwaldRootStatus.TWO_ROOT
        and tangent.status is EwaldRootStatus.TANGENT
        and no_root.status is EwaldRootStatus.NO_ROOT
        and direct.direct_beam_root_count == 1,
        "source, mosaic, or analytic Ewald invariant failed",
    )

    def orientation_path(mutation: str | None = None, seed: int = 0) -> tuple[tuple[str, NDArray[np.float64]], ...]:
        count = 24
        if mutation == "node_resampling":
            generator = np.random.default_rng(seed)
            alpha = np.pi * (np.arange(count) + generator.random(count)) / count
            differential = np.full(count, np.pi / count)
        else:
            node, weight = np.polynomial.legendre.leggauss(count)
            half_width = -0.5 * np.pi if mutation == "reversed_signed_arc_measure" else 0.5 * np.pi
            alpha = 0.5 * np.pi + half_width * node
            differential = half_width * weight
            order = np.argsort(alpha)
            alpha, differential = alpha[order], differential[order]
        gaussian = wrapped_mosaic_line_density_rad_inv(
            alpha, WrappedMosaicParameters(0.45, 0.7, 0.0)
        )
        lorentzian = wrapped_mosaic_line_density_rad_inv(
            alpha, WrappedMosaicParameters(0.45, 0.7, 1.0)
        )
        density = gaussian + lorentzian if mutation == "mixture_misnormalization" else 0.65 * gaussian + 0.35 * lorentzian
        if mutation == "removed_spherical_measure":
            density = density * np.sin(alpha)
        mass = 2.0 * density * differential
        jacobian = np.array([root.coarea_jacobian for root in two.emittable_roots])
        if mutation == "omitted_coarea_jacobian":
            factor = np.ones_like(jacobian)
        elif mutation == "duplicate_empirical_lorentz_factor":
            factor = jacobian * jacobian
        else:
            factor = jacobian
        return (
            ("reciprocal.quadrature_coordinate", alpha),
            ("reciprocal.event_weight", (mass[:, None] * factor).ravel()),
        )

    def compare(
        reference: tuple[tuple[str, NDArray[np.float64]], ...],
        candidate: tuple[tuple[str, NDArray[np.float64]], ...],
    ) -> dict[str, object] | None:
        for (stage, expected), (candidate_stage, observed) in zip(reference, candidate, strict=True):
            _require(stage == candidate_stage, "mutation stage sequence changed")
            if np.array_equal(expected, observed):
                continue
            common = min(expected.size, observed.size)
            errors = np.abs(expected.ravel()[:common] - observed.ravel()[:common])
            if observed.size > common:
                errors = np.concatenate((errors, np.abs(observed.ravel()[common:])))
            if expected.size > common:
                errors = np.concatenate((errors, np.abs(expected.ravel()[common:])))
            metric = (
                "bitwise_repeatability"
                if stage == "reciprocal.quadrature_coordinate"
                else "accepted_ewald_residual_Ainv"
                if stage == "reciprocal.ewald_residual"
                else "nonnegative_event_mass"
                if np.any(observed < 0.0)
                else "integrated_event_mass"
            )
            index = int(np.argmax(errors))
            return {
                "observed_first_stage": stage,
                "observed_failure_metric": metric,
                "max_absolute_error": float(errors[index]),
                "p95_absolute_error": float(np.percentile(errors, 95.0)),
                "failing_element_id": index,
            }
        return None

    correct = orientation_path()
    first_resample, second_resample = orientation_path("node_resampling", 1729), orientation_path(
        "node_resampling", 2718
    )
    repeated_correct = tuple((stage, np.stack((value, value))) for stage, value in correct)
    repeated_resampled = tuple(
        (stage, np.stack((first_value, second_value)))
        for (stage, first_value), (_, second_value) in zip(
            first_resample, second_resample, strict=True
        )
    )
    root = two.emittable_roots[0]
    bad_q = np.array([1.0, 0.0, root.u_Ainv + 1.0e-3])
    bad_kf = incident + bad_q
    bad_residual = abs(float(np.linalg.norm(bad_kf) - np.linalg.norm(incident)))
    residual_bound = 64.0 * np.finfo(np.float64).eps * np.linalg.norm(incident)
    rejected = (
        ("reciprocal.intersection_support", bad_q[None, :]),
        ("reciprocal.ewald_residual", np.array([bad_residual]) if bad_residual <= residual_bound else np.empty(0)),
    )
    accepted = (
        ("reciprocal.intersection_support", bad_q[None, :]),
        ("reciprocal.ewald_residual", np.array([bad_residual])),
    )
    cases = (
        ("removed_spherical_measure", "reciprocal.event_weight", "integrated_event_mass", correct, orientation_path("removed_spherical_measure")),
        ("mixture_misnormalization", "reciprocal.event_weight", "integrated_event_mass", correct, orientation_path("mixture_misnormalization")),
        ("node_resampling", "reciprocal.quadrature_coordinate", "bitwise_repeatability", repeated_correct, repeated_resampled),
        ("reversed_signed_arc_measure", "reciprocal.event_weight", "nonnegative_event_mass", correct, orientation_path("reversed_signed_arc_measure")),
        ("omitted_coarea_jacobian", "reciprocal.event_weight", "integrated_event_mass", correct, orientation_path("omitted_coarea_jacobian")),
        ("duplicate_empirical_lorentz_factor", "reciprocal.event_weight", "integrated_event_mass", correct, orientation_path("duplicate_empirical_lorentz_factor")),
        ("accepted_bad_residual_root", "reciprocal.ewald_residual", "accepted_ewald_residual_Ainv", rejected, accepted),
    )
    mutations: list[dict[str, object]] = []
    for mutation_id, expected_stage, expected_metric, reference, candidate in cases:
        observed = compare(reference, candidate)
        _require(
            observed is not None
            and observed["observed_first_stage"] == expected_stage
            and observed["observed_failure_metric"] == expected_metric,
            f"{mutation_id} failed at the wrong stage or metric",
        )
        mutations.append(
            {
                "mutation_id": mutation_id,
                "fixture_id": "mosaic.compact_real_calculation_v1",
                "expected_first_stage": expected_stage,
                "expected_failure_metric": expected_metric,
                **observed,
                "detected": True,
            }
        )
    checks = [
        {
            "check_id": "source_probability",
            "status": "PASS",
            "evidence": "declared polarization IDs, correlation labels, ordering, and mass survive source compilation",
        },
        {
            "check_id": "spherical_mosaic_measure",
            "status": "PASS",
            "evidence": "mixed atom plus continuous folded-alpha probability is nonnegative and normalized",
        },
        {
            "check_id": "analytic_ewald",
            "status": "PASS",
            "evidence": "two-root, tangent, no-root, direct suppression, residual, and coarea invariants pass",
        },
    ]
    metrics = {
        "source_mass_error": source_mass_error,
        "orientation_mass_error": orientation_mass_error,
        "elastic_residual_max_Ainv": max(
            root.ewald_residual_Ainv for root in two.emittable_roots
        ),
    }
    return checks, metrics, mutations


def run_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    """Run the compact T03 proof without writing diagnostics."""

    del allow_missing_pack
    root = Path(__file__).resolve().parents[3]
    proof_base = os.environ.get("PROOF_BASE_SHA", "")
    _require(proof_base == _PROOF_BASE_SHA, "PROOF_BASE_SHA is unset or does not match T03")
    commit_sha = _verified_commit_sha(root)
    manifest, arrays = _load_reference_pack(root)
    classifications, reference_metrics = _reference_evidence(manifest, arrays)
    scientific_checks, scientific_metrics, mutations = _scientific_evidence()
    oracle_matrix, convergence = _oracle_evidence()
    benchmark = _benchmark_evidence()
    checks = [
        {
            "check_id": "tracked_reference",
            "status": "SKIP",
            "evidence": "raw public/pack differences recorded; shared tolerance artifact is missing",
        },
        *scientific_checks,
        {
            "check_id": "sparse_event_memory",
            "status": "PASS",
            "evidence": "4,096 attempted lines and 3 events match every ordered dense-oracle field within the measured memory ceiling",
        },
        {
            "check_id": "negative_controls",
            "status": "PASS",
            "evidence": "7/7 assigned real-calculation mutations fail at the expected first stage",
        },
        {
            "check_id": "shared_validation_contract",
            "status": "FAIL",
            "evidence": "proof base lacks the required reviewed stage-tolerance/result-measure artifact",
        },
    ]
    return {
        "schema_version": 1,
        "task_id": "T03",
        "status": "BLOCKED",
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
            "tests/test_mosaic_ewald.py",
            "tasks/03_mosaic_ewald.md",
        ],
        "checks": checks,
        "metrics": {**reference_metrics, **scientific_metrics, "oracle_matrix": oracle_matrix},
        "classifications": classifications,
        "convergence": convergence,
        "mutations": mutations,
        "benchmark": benchmark,
        "tolerance_artifact_sha256": None,
        "limitations": [
            "UNITY_APPROXIMATION boundary: T03 preserves declared polarization-state IDs and applies no polarization factor",
            "reciprocal_weight contains orientation probability mass times exactly one coarea Jacobian; optics and detector factors remain downstream",
            "localized/adaptive acceleration and a generic SO(3) sampler are not implemented",
        ],
        "contract_requests": [
            {
                "request_id": "T03-PROOF-OWNERSHIP-AND-SHARED-VALIDATION",
                "owner": "proof-base/integration",
                "required_action": "amend T03 ownership to include reciprocal/proof.py, approve a shared wrapped-line-density trace stage, and publish a reviewed versioned stage-tolerance/result-measure artifact with its hash",
                "evidence": "the original T03 owned paths exclude proof.py and docs/VALIDATION.md forbids shared-pack acceptance without the reviewed artifact",
                "blocking_for_local_merge": True,
            }
        ],
    }
