"""Compact analytic, reference, convergence, mutation, and performance proof for T03."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import platform
import subprocess
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median
from types import MappingProxyType
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
from rasim_next.proof.traces import (
    Measure,
    QuantityKind,
    Tolerance,
    TraceRecord,
    compare_traces,
)
from rasim_next.reciprocal.events import EventBuildResult, build_scattering_events
from rasim_next.reciprocal.ewald import EwaldRootStatus, solve_continuous_rod_ewald
from rasim_next.sampling.mosaic import (
    MosaicOrientationBatch,
    WrappedMosaicParameters,
    manuscript_axisymmetric_v1_orientation_quadrature,
    wrapped_mosaic_line_density_rad_inv,
)
from rasim_next.sampling.source import (
    compile_independent_gaussian_source_samples,
    compile_joint_source_samples,
)

_PROOF_BASE_SHA = "812f896fde5b8365ff5c218fc606df674ad7dcad"
_REFERENCE_PACK_SHA256 = "e958703426ebea7a3fd62a8bb52447f9a5a8d7d5d4ad0eb0ce3b3706bbca1f06"
_TOLERANCE_VERSION = "mosaic-ewald-tolerances-v1"
_TOLERANCES = MappingProxyType(
    {
        "angular_centroid_Ainv_atol": 1.0e-7,
        "angular_mass_rtol": 1.0e-6,
        "angular_raw_node_quantile_Ainv_bound": 1.0e-3,
        "continuous_oracle_centroid_Ainv_atol": 1.0e-9,
        "continuous_oracle_mass_rtol": 1.0e-8,
        "continuous_oracle_quantile_Ainv_atol": 1.0e-8,
        "direction_norm_atol": 1.0e-12,
        "elastic_residual_eps_factor": 64.0,
        "legacy_density_trace_atol": 2.0e-14,
        "legacy_support_Ainv_atol": 1.0e-12,
        "public_dense_centroid_Ainv_atol": 1.0e-7,
        "public_dense_event_Ainv_atol": 1.0e-12,
        "public_dense_event_weight_rtol": 1.0e-6,
        "public_dense_l_atol": 1.0e-12,
        "public_dense_mass_rtol": 1.0e-6,
        "public_dense_quantile_Ainv_atol": 1.0e-6,
        "mutation_trace_atol": 1.0e-14,
        "source_weight_sum_atol": 1.0e-12,
        "wrapped_normalization_atol": 1.0e-10,
    }
)
_QUANTILE_PROBABILITIES = (0.1, 0.9)


@dataclass(frozen=True, slots=True)
class _DenseRoot:
    l_coordinate: float
    q_sample_Ainv: NDArray[np.float64]
    kf_sample_Ainv: NDArray[np.float64]
    ewald_residual_Ainv: float
    coarea_jacobian: float


@dataclass(frozen=True, slots=True)
class _ConvergenceCase:
    case_id: str
    q0_Ainv: float
    parameters: WrappedMosaicParameters
    wavelengths_A: tuple[float, ...]


_CONVERGENCE_CASES = (
    _ConvergenceCase("narrow_core", 1.0, WrappedMosaicParameters(0.01, 0.08, 0.05), (np.pi / 2.0,)),
    _ConvergenceCase("broad_core", 1.0, WrappedMosaicParameters(0.20, 0.35, 0.20), (np.pi / 2.0,)),
    _ConvergenceCase(
        "lorentzian_tail", 1.0, WrappedMosaicParameters(0.02, 0.25, 0.85), (np.pi / 2.0,)
    ),
    _ConvergenceCase(
        "near_tangent", 3.995, WrappedMosaicParameters(0.003, 0.03, 0.10), (np.pi / 2.0,)
    ),
    _ConvergenceCase("specular", 0.0, WrappedMosaicParameters(0.0, 0.0, 0.5), (np.pi / 2.0,)),
    _ConvergenceCase(
        "multi_wavelength", 1.0, WrappedMosaicParameters(0.05, 0.15, 0.25), (1.0, 1.1)
    ),
)


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


def _legacy_circle_error(
    q_sample_Ainv: NDArray[np.float64],
    ki_sample_Ainv: NDArray[np.float64],
    k_scat_Ainv: float,
    g_norm_Ainv: float,
) -> tuple[float, float, float]:
    center_vector = -ki_sample_Ainv
    center_norm = float(np.linalg.norm(center_vector))
    center_hat = center_vector / center_norm
    plane_distance = (g_norm_Ainv**2 + center_norm**2 - k_scat_Ainv**2) / (2.0 * center_norm)
    circle_center = plane_distance * center_hat
    radius = np.sqrt(g_norm_Ainv**2 - plane_distance**2)
    seed = np.array([1.0, 0.0, 0.0])
    first_axis = seed - np.dot(seed, center_hat) * center_hat
    first_axis /= np.linalg.norm(first_axis)
    second_axis = np.cross(center_hat, first_axis)
    relative = q_sample_Ainv - circle_center
    azimuth = np.arctan2(relative @ second_axis, relative @ first_axis)
    reconstructed = circle_center + radius * (
        np.cos(azimuth)[:, None] * first_axis + np.sin(azimuth)[:, None] * second_axis
    )
    dense_azimuth = np.linspace(0.0, 2.0 * np.pi, 65_536, endpoint=False)
    dense_support = circle_center + radius * (
        np.cos(dense_azimuth)[:, None] * first_axis + np.sin(dense_azimuth)[:, None] * second_axis
    )
    coordinate_error = float(np.max(np.abs(reconstructed - q_sample_Ainv)))
    elastic_error = float(
        np.max(np.abs(np.linalg.norm(ki_sample_Ainv + q_sample_Ainv, axis=1) - k_scat_Ainv))
    )
    dense_invariant_error = max(
        float(np.max(np.abs(np.linalg.norm(dense_support, axis=1) - g_norm_Ainv))),
        float(np.max(np.abs(np.linalg.norm(ki_sample_Ainv + dense_support, axis=1) - k_scat_Ainv))),
    )
    return coordinate_error, elastic_error, dense_invariant_error


def _signed_ewald_residual(
    u_Ainv: float,
    incident: NDArray[np.float64],
    q0: NDArray[np.float64],
    direction: NDArray[np.float64],
) -> float:
    return float(np.linalg.norm(incident + q0 + u_Ainv * direction) - np.linalg.norm(incident))


def _bisect_residual(
    left: float,
    right: float,
    incident: NDArray[np.float64],
    q0: NDArray[np.float64],
    direction: NDArray[np.float64],
) -> float:
    left_value = _signed_ewald_residual(left, incident, q0, direction)
    right_value = _signed_ewald_residual(right, incident, q0, direction)
    if left_value == 0.0:
        return left
    if right_value == 0.0:
        return right
    _require(left_value * right_value < 0.0, "dense root bracket does not change sign")
    for _ in range(60):
        midpoint = 0.5 * (left + right)
        midpoint_value = _signed_ewald_residual(midpoint, incident, q0, direction)
        if midpoint_value == 0.0:
            return midpoint
        if left_value * midpoint_value < 0.0:
            right = midpoint
        else:
            left = midpoint
            left_value = midpoint_value
    return 0.5 * (left + right)


def _dense_continuous_rod_roots(
    *,
    incident: NDArray[np.float64],
    q0: NDArray[np.float64],
    direction: NDArray[np.float64],
    b3_norm_Ainv: float,
    node_count: int = 65,
) -> tuple[EwaldRootStatus, tuple[_DenseRoot, ...], int]:
    """Independent dense residual scan and bisection; does not call the public solver."""

    _require(node_count >= 3 and node_count % 2 == 1, "dense node_count must be odd")
    direction = np.asarray(direction / np.linalg.norm(direction), dtype=np.float64)
    incident_norm = float(np.linalg.norm(incident))
    center = -float(np.dot(incident + q0, direction))
    center_residual = _signed_ewald_residual(center, incident, q0, direction)
    if center_residual > 0.0:
        return EwaldRootStatus.NO_ROOT, (), 0
    if center_residual == 0.0:
        tangent_q = q0 + center * direction
        return EwaldRootStatus.TANGENT, (), int(np.count_nonzero(tangent_q) == 0)
    coordinates = np.linspace(center - incident_norm, center + incident_norm, node_count)
    residual = np.array(
        [_signed_ewald_residual(value, incident, q0, direction) for value in coordinates]
    )
    roots: list[float] = []
    for index in range(node_count - 1):
        left_value, right_value = residual[index : index + 2]
        if left_value == 0.0:
            roots.append(float(coordinates[index]))
        if left_value * right_value < 0.0:
            roots.append(
                _bisect_residual(
                    float(coordinates[index]),
                    float(coordinates[index + 1]),
                    incident,
                    q0,
                    direction,
                )
            )
    if residual[-1] == 0.0:
        roots.append(float(coordinates[-1]))
    roots = sorted(set(roots))
    _require(len(roots) == 2, "dense continuous rod must find exactly two regular roots")
    direct_beam_count = 0
    if np.count_nonzero(q0) == 0:
        direct_index = int(np.argmin(np.abs(roots)))
        roots.pop(direct_index)
        direct_beam_count = 1
    dense_roots: list[_DenseRoot] = []
    for root in roots:
        q = np.asarray(q0 + root * direction, dtype=np.float64)
        kf = np.asarray(incident + q, dtype=np.float64)
        kf_norm = np.linalg.norm(kf)
        residual_value = abs(float(kf_norm - incident_norm))
        derivative = abs(float(np.dot(kf / kf_norm, direction)))
        _require(derivative > 0.0, "dense regular root has zero derivative")
        dense_roots.append(
            _DenseRoot(
                root / b3_norm_Ainv,
                q,
                kf,
                residual_value,
                1.0 / derivative,
            )
        )
    return EwaldRootStatus.TWO_ROOT, tuple(dense_roots), direct_beam_count


def _build_case(
    case: _ConvergenceCase, alpha_count: int, azimuth_count: int
) -> tuple[
    EventBuildResult,
    IncidentSampleBatch,
    IncidentStateBatch,
    RodCatalog,
    MosaicOrientationBatch,
    RigidTransform,
]:
    wavelength = np.asarray(case.wavelengths_A, dtype=np.float64)
    source_mass = np.full(wavelength.size, 1.0 / wavelength.size)
    sample_ids = np.arange(wavelength.size, dtype=np.int64)
    incident_samples = IncidentSampleBatch(
        incident_sample_id=sample_ids,
        origin_lab_m=np.zeros((wavelength.size, 3)),
        direction_lab=np.tile([0.0, 0.0, 1.0], (wavelength.size, 1)),
        wavelength_A=wavelength,
        source_weight=source_mass,
        polarization_state_id=("unity_scalar",) * wavelength.size,
        correlation_model="explicit_joint",
    )
    wavevector_norm = 2.0 * np.pi / wavelength
    wavevector = np.column_stack(
        (np.zeros(wavelength.size), np.zeros(wavelength.size), wavevector_norm)
    )
    incident_states = IncidentStateBatch(
        incident_state_id=np.arange(100, 100 + wavelength.size, dtype=np.int64),
        incident_sample_id=sample_ids,
        sample_intersection_lab_m=np.zeros((wavelength.size, 3)),
        direction_sample=np.tile([0.0, 0.0, 1.0], (wavelength.size, 1)),
        k_air_sample_Ainv=wavevector,
        k_film_phase_sample_Ainv=wavevector,
        kz_film_Ainv=wavevector_norm.astype(np.complex128),
        entrance_amplitude=np.ones(wavelength.size, dtype=np.complex128),
        footprint_acceptance=np.ones(wavelength.size),
        source_weight=source_mass,
        valid=np.ones(wavelength.size, dtype=np.bool_),
    )
    basis = np.diag([max(case.q0_Ainv, 1.0), 1.0, 1.0])
    rods = RodCatalog(
        rod_id=np.array([200], dtype=np.int64),
        phase_id=("proof",),
        h=np.array([0 if case.q0_Ainv == 0.0 else 1], dtype=np.int32),
        k=np.array([0], dtype=np.int32),
        family_id=("proof",),
        family_key=("proof",),
        qr_Ainv=np.array([case.q0_Ainv]),
        reciprocal_basis_Ainv=basis,
        symmetry_metadata=("analytic",),
    )
    orientations = manuscript_axisymmetric_v1_orientation_quadrature(
        case.parameters,
        reciprocal_basis_Ainv=basis,
        alpha_cell_count=alpha_count,
        azimuth_cell_count=azimuth_count,
    )
    sample_from_crystal = RigidTransform(np.eye(3), np.zeros(3), FrameId.CRYSTAL, FrameId.SAMPLE)
    result = build_scattering_events(
        incident_samples=incident_samples,
        incident_states=incident_states,
        rods=rods,
        orientations=orientations,
        sample_from_crystal=sample_from_crystal,
    )
    return result, incident_samples, incident_states, rods, orientations, sample_from_crystal


def _dense_case_events(
    incident_states: IncidentStateBatch,
    rods: RodCatalog,
    orientations: MosaicOrientationBatch,
) -> dict[str, object]:
    basis = rods.reciprocal_basis_Ainv
    b3_norm = float(np.linalg.norm(basis[:, 2]))
    b3_hat = basis[:, 2] / b3_norm
    q0 = basis @ np.array([int(rods.h[0]), int(rods.k[0]), 0.0])
    q_values: list[NDArray[np.float64]] = []
    l_values: list[float] = []
    kf_values: list[NDArray[np.float64]] = []
    weights: list[float] = []
    residuals: list[float] = []
    statuses: list[EwaldRootStatus] = []
    emitted_counts: list[int] = []
    direct_counts: list[int] = []
    for state_index, valid in enumerate(incident_states.valid):
        if not valid:
            continue
        incident = incident_states.k_film_phase_sample_Ainv[state_index]
        for orientation_index, rotation in enumerate(orientations.rotation_crystal):
            orientation_mass = float(orientations.probability_mass[orientation_index])
            if orientation_mass == 0.0:
                continue
            status, roots, direct_count = _dense_continuous_rod_roots(
                incident=incident,
                q0=rotation @ q0,
                direction=rotation @ b3_hat,
                b3_norm_Ainv=b3_norm,
            )
            statuses.append(status)
            emitted_counts.append(len(roots))
            direct_counts.append(direct_count)
            for root in roots:
                q_values.append(root.q_sample_Ainv)
                l_values.append(root.l_coordinate)
                kf_values.append(root.kf_sample_Ainv)
                weights.append(orientation_mass * root.coarea_jacobian)
                residuals.append(root.ewald_residual_Ainv)
    return {
        "q": np.asarray(q_values, dtype=np.float64).reshape((-1, 3)),
        "l": np.asarray(l_values, dtype=np.float64),
        "kf": np.asarray(kf_values, dtype=np.float64).reshape((-1, 3)),
        "weight": np.asarray(weights, dtype=np.float64),
        "residual": np.asarray(residuals, dtype=np.float64),
        "status": tuple(statuses),
        "emitted_count": np.asarray(emitted_counts, dtype=np.int8),
        "direct_count": np.asarray(direct_counts, dtype=np.int8),
    }


def _weighted_quantiles(
    values: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    targets = np.asarray(_QUANTILE_PROBABILITIES) * sorted_weights.sum()
    return np.interp(targets, cumulative, sorted_values)


def _observables(
    q_sample_Ainv: NDArray[np.float64], weight: NDArray[np.float64]
) -> dict[str, object]:
    _require(q_sample_Ainv.shape[0] > 0 and np.all(weight > 0.0), "event support is empty")
    total_mass = float(weight.sum())
    centroid = np.sum(q_sample_Ainv * weight[:, None], axis=0) / total_mass
    quantiles = _weighted_quantiles(q_sample_Ainv[:, 2], weight)
    return {
        "total_mass": total_mass,
        "centroid_Q_Ainv": centroid,
        "qz_quantiles_Ainv": quantiles,
    }


def _oracle_folded_density(
    alpha_rad: NDArray[np.float64], parameters: WrappedMosaicParameters
) -> NDArray[np.float64]:
    """Independent direct density for the continuous folded signed measure."""

    signed_density = np.zeros_like(alpha_rad)
    tail_probability = parameters.lorentzian_probability
    sigma = parameters.gaussian_sigma_rad
    if sigma > 0.0 and tail_probability < 1.0:
        image_count = int(np.ceil(8.0 * sigma / (2.0 * np.pi))) + 2
        gaussian = np.zeros_like(alpha_rad)
        for image in range(-image_count, image_count + 1):
            offset = alpha_rad + 2.0 * np.pi * image
            gaussian += np.exp(-0.5 * (offset / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
        signed_density += (1.0 - tail_probability) * gaussian
    half_width = parameters.lorentzian_half_width_rad
    if half_width > 0.0 and tail_probability > 0.0:
        rho = np.exp(-half_width)
        signed_density += (
            tail_probability
            * (1.0 - rho**2)
            / (2.0 * np.pi * (1.0 + rho**2 - 2.0 * rho * np.cos(alpha_rad)))
        )
    return 2.0 * signed_density


def _oracle_atom_events(case: _ConvergenceCase) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    qz_values: list[float] = []
    weights: list[float] = []
    atom_mass = case.parameters.zero_tilt_probability_mass
    if atom_mass == 0.0:
        return np.empty(0), np.empty(0)
    for wavelength_A in case.wavelengths_A:
        k_Ainv = 2.0 * np.pi / wavelength_A
        discriminant = k_Ainv**2 - case.q0_Ainv**2
        if discriminant <= 0.0:
            continue
        root = np.sqrt(discriminant)
        jacobian = k_Ainv / root
        for qz_Ainv in (-k_Ainv - root, -k_Ainv + root):
            if case.q0_Ainv == 0.0 and qz_Ainv == 0.0:
                continue
            qz_values.append(float(qz_Ainv))
            weights.append(atom_mass * jacobian)
    return np.asarray(qz_values), np.asarray(weights)


def _oracle_branch_values(
    case: _ConvergenceCase,
    k_Ainv: float,
    x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    alpha = np.pi * x**2
    sine = np.sin(alpha)
    cosine = np.cos(alpha)
    discriminant = k_Ainv**2 - (k_Ainv * sine - case.q0_Ainv) ** 2
    _require(np.all(discriminant > 0.0), f"{case.case_id} oracle encountered tangent support")
    root = np.sqrt(discriminant)
    base_qz = -case.q0_Ainv * sine - k_Ainv * cosine**2
    branch_delta = root * cosine
    return base_qz - branch_delta, base_qz + branch_delta, alpha, root


def _oracle_line_values(
    case: _ConvergenceCase,
    k_Ainv: float,
    x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    first_qz, second_qz, alpha, root = _oracle_branch_values(case, k_Ainv, x)
    probability_density_dx = _oracle_folded_density(alpha, case.parameters) * 2.0 * np.pi * x
    event_weight_density_dx = probability_density_dx * k_Ainv / root
    return first_qz, second_qz, event_weight_density_dx


def _oracle_moments(case: _ConvergenceCase, order: int) -> dict[str, object]:
    node, weight = np.polynomial.legendre.leggauss(order)
    x = 0.5 * (node + 1.0)
    integration_weight = 0.5 * weight
    total_mass = 0.0
    qz_numerator = 0.0
    continuous_mass = 1.0 - case.parameters.zero_tilt_probability_mass
    if continuous_mass > 0.0:
        for wavelength_A in case.wavelengths_A:
            k_Ainv = 2.0 * np.pi / wavelength_A
            first_qz, second_qz, event_weight_density = _oracle_line_values(case, k_Ainv, x)
            root_weight = integration_weight * event_weight_density
            total_mass += 2.0 * float(root_weight.sum())
            qz_numerator += float(np.sum(root_weight * (first_qz + second_qz)))
    atom_qz, atom_weight = _oracle_atom_events(case)
    total_mass += float(atom_weight.sum())
    qz_numerator += float(np.dot(atom_qz, atom_weight))
    _require(total_mass > 0.0, f"{case.case_id} oracle support is empty")
    return {
        "total_mass": total_mass,
        "centroid_Q_Ainv": np.array([0.0, 0.0, qz_numerator / total_mass]),
    }


def _oracle_branch_roots(
    case: _ConvergenceCase,
    k_Ainv: float,
    branch_index: int,
    threshold_Ainv: float,
) -> list[float]:
    grid = np.linspace(0.0, 1.0, 513)
    branches = _oracle_branch_values(case, k_Ainv, grid)[:2]
    value = branches[branch_index] - threshold_Ainv
    roots: list[float] = []
    for index in range(grid.size - 1):
        if value[index] == 0.0:
            roots.append(float(grid[index]))
        if value[index] * value[index + 1] >= 0.0:
            continue
        lower = float(grid[index])
        upper = float(grid[index + 1])
        lower_value = float(value[index])
        for _ in range(52):
            midpoint = 0.5 * (lower + upper)
            midpoint_value = float(
                _oracle_branch_values(case, k_Ainv, np.array([midpoint]))[branch_index][0]
                - threshold_Ainv
            )
            if lower_value * midpoint_value <= 0.0:
                upper = midpoint
            else:
                lower = midpoint
                lower_value = midpoint_value
        roots.append(0.5 * (lower + upper))
    if value[-1] == 0.0:
        roots.append(1.0)
    roots.sort()
    return [
        root for index, root in enumerate(roots) if index == 0 or root - roots[index - 1] > 1e-13
    ]


def _oracle_branch_cdf(
    case: _ConvergenceCase,
    k_Ainv: float,
    branch_index: int,
    threshold_Ainv: float,
    node: NDArray[np.float64],
    weight: NDArray[np.float64],
) -> float:
    roots = _oracle_branch_roots(case, k_Ainv, branch_index, threshold_Ainv)
    boundaries = (0.0, *roots, 1.0)
    cumulative = 0.0
    for lower, upper in pairwise(boundaries):
        midpoint = 0.5 * (lower + upper)
        branch_midpoint = _oracle_branch_values(case, k_Ainv, np.array([midpoint]))[
            branch_index
        ][0]
        if branch_midpoint > threshold_Ainv or upper == lower:
            continue
        half_width = 0.5 * (upper - lower)
        x = midpoint + half_width * node
        event_weight_density = _oracle_line_values(case, k_Ainv, x)[2]
        cumulative += half_width * float(np.dot(weight, event_weight_density))
    return cumulative


def _oracle_quantiles(case: _ConvergenceCase, total_mass: float, order: int) -> NDArray[np.float64]:
    atom_qz, atom_weight = _oracle_atom_events(case)
    if case.parameters.zero_tilt_probability_mass == 1.0:
        return _weighted_quantiles(atom_qz, atom_weight)

    k_max = max(2.0 * np.pi / wavelength_A for wavelength_A in case.wavelengths_A)
    lower_bound = -2.0 * k_max - abs(case.q0_Ainv) - 1.0
    upper_bound = abs(case.q0_Ainv) + 1.0
    node, weight = np.polynomial.legendre.leggauss(order)

    def cumulative(threshold_Ainv: float) -> float:
        value = float(atom_weight[atom_qz <= threshold_Ainv].sum())
        for wavelength_A in case.wavelengths_A:
            k_Ainv = 2.0 * np.pi / wavelength_A
            value += _oracle_branch_cdf(case, k_Ainv, 0, threshold_Ainv, node, weight)
            value += _oracle_branch_cdf(case, k_Ainv, 1, threshold_Ainv, node, weight)
        return value

    quantiles = np.empty(len(_QUANTILE_PROBABILITIES))
    for index, probability in enumerate(_QUANTILE_PROBABILITIES):
        lower = lower_bound
        upper = upper_bound
        target = probability * total_mass
        for _ in range(52):
            midpoint = 0.5 * (lower + upper)
            if cumulative(midpoint) < target:
                lower = midpoint
            else:
                upper = midpoint
        quantiles[index] = 0.5 * (lower + upper)
    return quantiles


def _continuous_orientation_reference(case: _ConvergenceCase) -> dict[str, object]:
    coarse = _oracle_moments(case, 128)
    fine = _oracle_moments(case, 256)
    coarse_quantiles = _oracle_quantiles(case, float(coarse["total_mass"]), 96)
    fine_quantiles = _oracle_quantiles(case, float(fine["total_mass"]), 192)
    mass_relative_error = abs(float(coarse["total_mass"]) - float(fine["total_mass"])) / abs(
        float(fine["total_mass"])
    )
    centroid_error = float(
        np.max(np.abs(np.asarray(coarse["centroid_Q_Ainv"]) - np.asarray(fine["centroid_Q_Ainv"])))
    )
    quantile_error = float(np.max(np.abs(coarse_quantiles - fine_quantiles)))
    _require(
        mass_relative_error <= _TOLERANCES["continuous_oracle_mass_rtol"]
        and centroid_error <= _TOLERANCES["continuous_oracle_centroid_Ainv_atol"]
        and quantile_error <= _TOLERANCES["continuous_oracle_quantile_Ainv_atol"],
        f"{case.case_id} continuous orientation oracle did not converge",
    )
    return {
        "total_mass": float(fine["total_mass"]),
        "centroid_Q_Ainv": np.asarray(fine["centroid_Q_Ainv"]).tolist(),
        "qz_quantiles_Ainv": fine_quantiles.tolist(),
        "orders": {"moment_coarse": 128, "moment_fine": 256, "cdf_coarse": 96, "cdf_fine": 192},
        "refinement_errors": {
            "mass_relative": mass_relative_error,
            "centroid_Ainv": centroid_error,
            "quantile_Ainv": quantile_error,
        },
    }


def _evaluate_case(
    case: _ConvergenceCase, alpha_count: int, azimuth_count: int
) -> dict[str, object]:
    public, _, states, rods, orientations, _ = _build_case(case, alpha_count, azimuth_count)
    dense = _dense_case_events(states, rods, orientations)
    public_q = public.events.q_internal_sample_Ainv
    public_weight = public.events.reciprocal_weight
    dense_q = np.asarray(dense["q"])
    dense_l = np.asarray(dense["l"])
    dense_kf = np.asarray(dense["kf"])
    dense_weight = np.asarray(dense["weight"])
    dense_residual = np.asarray(dense["residual"])
    _require(public_q.shape == dense_q.shape, f"{case.case_id} dense event count mismatch")
    _require(
        public.status.root_status == dense["status"]
        and np.array_equal(public.status.emitted_root_count, dense["emitted_count"])
        and np.array_equal(public.status.direct_beam_root_count, dense["direct_count"]),
        f"{case.case_id} dense status mismatch",
    )
    public_observable = _observables(public_q, public_weight)
    dense_observable = _observables(dense_q, dense_weight)
    mass_relative_error = abs(
        float(public_observable["total_mass"]) - float(dense_observable["total_mass"])
    ) / abs(float(dense_observable["total_mass"]))
    centroid_error = float(
        np.max(
            np.abs(
                np.asarray(public_observable["centroid_Q_Ainv"])
                - np.asarray(dense_observable["centroid_Q_Ainv"])
            )
        )
    )
    quantile_error = float(
        np.max(
            np.abs(
                np.asarray(public_observable["qz_quantiles_Ainv"])
                - np.asarray(dense_observable["qz_quantiles_Ainv"])
            )
        )
    )
    q_error = float(np.max(np.abs(public_q - dense_q))) if public_q.size else 0.0
    l_error = float(np.max(np.abs(public.events.l_coordinate - dense_l))) if dense_l.size else 0.0
    kf_error = (
        float(np.max(np.abs(public.events.kf_film_phase_sample_Ainv - dense_kf)))
        if dense_kf.size
        else 0.0
    )
    residual_error = (
        float(np.max(np.abs(public.events.ewald_residual_Ainv - dense_residual)))
        if dense_residual.size
        else 0.0
    )
    weight_relative_error = (
        float(
            np.max(
                np.abs(public_weight - dense_weight)
                / np.maximum(np.abs(dense_weight), np.finfo(np.float64).tiny)
            )
        )
        if dense_weight.size
        else 0.0
    )
    incident_norm = float(np.max(np.linalg.norm(states.k_film_phase_sample_Ainv, axis=1)))
    residual_limit = (
        _TOLERANCES["elastic_residual_eps_factor"]
        * np.finfo(np.float64).eps
        * max(incident_norm, 1.0)
    )
    _require(
        mass_relative_error <= _TOLERANCES["public_dense_mass_rtol"]
        and centroid_error <= _TOLERANCES["public_dense_centroid_Ainv_atol"]
        and quantile_error <= _TOLERANCES["public_dense_quantile_Ainv_atol"]
        and q_error <= _TOLERANCES["public_dense_event_Ainv_atol"]
        and l_error <= _TOLERANCES["public_dense_l_atol"]
        and kf_error <= _TOLERANCES["public_dense_event_Ainv_atol"]
        and residual_error <= residual_limit
        and weight_relative_error <= _TOLERANCES["public_dense_event_weight_rtol"]
        and float(np.max(public.events.ewald_residual_Ainv, initial=0.0)) <= residual_limit
        and float(np.max(dense_residual, initial=0.0)) <= residual_limit,
        f"{case.case_id} public/dense event field mismatch",
    )
    metrics = {
        "requested_alpha_panel_count": alpha_count,
        "azimuth_cell_count": azimuth_count,
        "orientation_count": int(orientations.orientation_id.size),
        "attempt_count": int(public.status.attempt_id.size),
        "event_count": int(public.events.event_id.size),
        "public": {
            "total_mass": float(public_observable["total_mass"]),
            "centroid_Q_Ainv": np.asarray(public_observable["centroid_Q_Ainv"]).tolist(),
            "qz_quantiles_Ainv": np.asarray(public_observable["qz_quantiles_Ainv"]).tolist(),
        },
        "dense": {
            "total_mass": float(dense_observable["total_mass"]),
            "centroid_Q_Ainv": np.asarray(dense_observable["centroid_Q_Ainv"]).tolist(),
            "qz_quantiles_Ainv": np.asarray(dense_observable["qz_quantiles_Ainv"]).tolist(),
        },
        "errors": {
            "mass_relative": mass_relative_error,
            "centroid_Ainv": centroid_error,
            "quantile_Ainv": quantile_error,
            "event_Q_Ainv": q_error,
            "event_L": l_error,
            "event_kf_Ainv": kf_error,
            "event_residual_Ainv": residual_error,
            "event_weight_relative": weight_relative_error,
        },
    }
    return metrics


def _event_results_equal(first: EventBuildResult, second: EventBuildResult) -> bool:
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
    return (
        all(
            np.array_equal(getattr(first.events, name), getattr(second.events, name))
            for name in event_fields
        )
        and all(
            np.array_equal(getattr(first.status, name), getattr(second.status, name))
            for name in status_fields
        )
        and first.status.root_status == second.status.root_status
    )


def _convergence_evidence() -> list[dict[str, object]]:
    resolutions = ((8, 4), (32, 4), (128, 4))
    evidence: list[dict[str, object]] = []
    for case in _CONVERGENCE_CASES:
        line_oracle = _evaluate_case(case, 4, 4)
        continuous_oracle = _continuous_orientation_reference(case)
        oracle_mass = float(continuous_oracle["total_mass"])
        oracle_centroid = np.asarray(continuous_oracle["centroid_Q_Ainv"])
        oracle_quantiles = np.asarray(continuous_oracle["qz_quantiles_Ainv"])
        levels: list[dict[str, object]] = []
        for alpha_count, azimuth_count in resolutions:
            public, _, _, _, orientations, _ = _build_case(case, alpha_count, azimuth_count)
            observable = _observables(
                public.events.q_internal_sample_Ainv,
                public.events.reciprocal_weight,
            )
            angular_errors = {
                "mass_relative": abs(float(observable["total_mass"]) - oracle_mass)
                / abs(oracle_mass),
                "centroid_Ainv": float(
                    np.max(np.abs(np.asarray(observable["centroid_Q_Ainv"]) - oracle_centroid))
                ),
                "quantile_Ainv": float(
                    np.max(np.abs(np.asarray(observable["qz_quantiles_Ainv"]) - oracle_quantiles))
                ),
            }
            levels.append(
                {
                    "requested_alpha_panel_count": alpha_count,
                    "azimuth_cell_count": azimuth_count,
                    "orientation_count": int(orientations.orientation_id.size),
                    "attempt_count": int(public.status.attempt_id.size),
                    "event_count": int(public.events.event_id.size),
                    "public": {
                        "total_mass": float(observable["total_mass"]),
                        "centroid_Q_Ainv": np.asarray(observable["centroid_Q_Ainv"]).tolist(),
                        "qz_quantiles_Ainv": np.asarray(observable["qz_quantiles_Ainv"]).tolist(),
                    },
                    "continuous_oracle_errors": angular_errors,
                }
            )
        final_public = public
        repeated, *_ = _build_case(case, *resolutions[-1])
        _require(
            _event_results_equal(final_public, repeated), f"{case.case_id} is not deterministic"
        )

        final_errors = levels[-1]["continuous_oracle_errors"]
        _require(
            float(final_errors["mass_relative"]) <= _TOLERANCES["angular_mass_rtol"]
            and float(final_errors["centroid_Ainv"]) <= _TOLERANCES["angular_centroid_Ainv_atol"]
            and float(final_errors["quantile_Ainv"])
            <= _TOLERANCES["angular_raw_node_quantile_Ainv_bound"],
            f"{case.case_id} angular quadrature did not meet the continuous oracle",
        )
        monotonic = all(
            all(
                float(levels[index + 1]["continuous_oracle_errors"][metric])
                <= float(levels[index]["continuous_oracle_errors"][metric])
                for index in range(len(levels) - 1)
            )
            for metric in ("mass_relative", "centroid_Ainv", "quantile_Ainv")
        )
        evidence.append(
            {
                "case_id": case.case_id,
                "status": "PASS",
                "convergence_kind": "MONOTONIC" if monotonic else "BOUNDED_NON_MONOTONIC",
                "justification": None
                if monotonic
                else "mass and centroid refine below frozen bounds; raw-node tail quantiles stay bounded against a converged cell-integrated CDF under deterministic refinement",
                "same_node_dense_line_oracle": line_oracle,
                "continuous_orientation_oracle": continuous_oracle,
                "levels": levels,
                "bitwise_repeatable": True,
                "maximum_reciprocal_weight": float(np.max(final_public.events.reciprocal_weight)),
            }
        )
    return evidence


def _trace(
    stage_id: str,
    value: NDArray[Any],
    *,
    unit: str,
    frame: str,
    measure: Measure,
) -> TraceRecord:
    return TraceRecord(
        case_id="mosaic.error_injection",
        stage_id=stage_id,
        value=value,
        unit=unit,
        frame=frame,
        measure=measure,
        quantity_kind=QuantityKind.VECTOR,
        model_version="manuscript_axisymmetric_v1",
        provenance="analytic T03 mutation fixture",
    )


def _mutation_evidence() -> list[dict[str, object]]:
    case = _CONVERGENCE_CASES[0]
    public, _, _, _, orientations, _ = _build_case(case, 8, 16)
    orientation_mass = orientations.probability_mass
    event_weight = public.events.reciprocal_weight
    emitted_count = public.status.emitted_root_count.astype(np.int64)
    event_orientation_mass = np.repeat(orientation_mass, emitted_count)
    _require(event_orientation_mass.shape == event_weight.shape, "mutation fixture misaligned")
    jacobian = event_weight / event_orientation_mass
    incident = np.array([0.0, 0.0, 4.0])
    direction = np.array([0.0, 0.0, 1.0])
    shifted_q0 = np.array([np.nextafter(4.0, 0.0), 0.0, 2.0])
    shifted_roots = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=shifted_q0,
        d_hat_sample=direction,
        b3_norm_Ainv=1.0,
    )
    naive_projection = float(np.dot(incident + shifted_q0, direction))
    naive_constant = float(
        np.dot(shifted_q0, shifted_q0) + 2.0 * np.dot(incident, shifted_q0)
    )
    naive_discriminant = naive_projection * naive_projection - naive_constant
    _require(
        shifted_roots.status is EwaldRootStatus.TWO_ROOT and naive_discriminant == 0.0,
        "line-origin mutation fixture is not sensitive",
    )

    sample_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mosaic_rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.sqrt(3.0) / 2.0, -0.5], [0.0, 0.5, np.sqrt(3.0) / 2.0]]
    )
    reciprocal_vector = np.array([1.0, 2.0, 3.0])
    composed_q = sample_rotation @ (mosaic_rotation @ reciprocal_vector)
    reversed_q = mosaic_rotation @ (sample_rotation @ reciprocal_vector)

    film_roots = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([1.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=1.0,
    )
    air_roots = solve_continuous_rod_ewald(
        ki_sample_Ainv=np.array([0.0, 0.0, 5.0]),
        q0_sample_Ainv=np.array([1.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=1.0,
    )
    _require(
        len(film_roots.emittable_roots) == len(air_roots.emittable_roots) == 2,
        "wavevector-ownership mutation fixture is not regular",
    )
    film_q = np.stack([root.q_sample_Ainv for root in film_roots.emittable_roots])
    air_q = np.stack([root.q_sample_Ainv for root in air_roots.emittable_roots])

    near_tangent_jacobian = np.array(
        [root.coarea_jacobian for root in shifted_roots.emittable_roots]
    )
    _require(
        np.all(near_tangent_jacobian > 1.0e6),
        "Jacobian clipping mutation fixture is not sensitive",
    )
    mixed_orientations = manuscript_axisymmetric_v1_orientation_quadrature(
        WrappedMosaicParameters(0.0, 0.3, 0.25),
        reciprocal_basis_Ainv=np.eye(3),
        alpha_cell_count=4,
        azimuth_cell_count=4,
    )
    dropped_atom_mass = mixed_orientations.probability_mass.copy()
    dropped_atom_mass[mixed_orientations.alpha_rad == 0.0] = 0.0
    candidates = (
        (
            "removed_spherical_measure",
            "reciprocal.event_weight",
            orientation_mass,
            orientation_mass * np.sin(orientations.alpha_rad),
            "1",
            "reciprocal",
            Measure.INTEGRATED_EVENT_MASS,
        ),
        (
            "mixture_misnormalization",
            "reciprocal.event_weight",
            orientation_mass,
            1.1 * orientation_mass,
            "1",
            "reciprocal",
            Measure.INTEGRATED_EVENT_MASS,
        ),
        (
            "node_resampling",
            "reciprocal.quadrature_coordinate",
            orientations.alpha_rad,
            np.roll(orientations.alpha_rad, 1),
            "rad",
            "crystal",
            Measure.NONE,
        ),
        (
            "reversed_signed_arc_measure",
            "reciprocal.event_weight",
            orientation_mass,
            -orientation_mass,
            "1",
            "reciprocal",
            Measure.INTEGRATED_EVENT_MASS,
        ),
        (
            "omitted_coarea_jacobian",
            "reciprocal.event_weight",
            event_weight,
            event_orientation_mass,
            "1",
            "reciprocal",
            Measure.INTEGRATED_EVENT_MASS,
        ),
        (
            "duplicate_empirical_lorentz_factor",
            "reciprocal.event_weight",
            event_weight,
            event_weight * jacobian,
            "1",
            "reciprocal",
            Measure.INTEGRATED_EVENT_MASS,
        ),
        (
            "accepted_bad_residual_root",
            "reciprocal.ewald_residual",
            public.events.ewald_residual_Ainv,
            public.events.ewald_residual_Ainv + 1.0e-4,
            "A^-1",
            "sample",
            Measure.NONE,
        ),
        (
            "line_origin_cancellation",
            "reciprocal.intersection_support",
            np.array([len(shifted_roots.emittable_roots)], dtype=np.float64),
            np.array([2 if naive_discriminant > 0.0 else 0], dtype=np.float64),
            "1",
            "sample",
            Measure.NONE,
        ),
        (
            "wrong_rotation_composition",
            "reciprocal.intersection_support",
            composed_q,
            reversed_q,
            "A^-1",
            "sample",
            Measure.NONE,
        ),
        (
            "air_side_incident_wavevector",
            "reciprocal.intersection_support",
            film_q,
            air_q,
            "A^-1",
            "sample",
            Measure.NONE,
        ),
        (
            "clipped_coarea_jacobian",
            "reciprocal.event_weight",
            near_tangent_jacobian,
            np.minimum(near_tangent_jacobian, 1.0e6),
            "1",
            "reciprocal",
            Measure.INTEGRATED_EVENT_MASS,
        ),
        (
            "dropped_zero_width_atom",
            "reciprocal.event_weight",
            mixed_orientations.probability_mass,
            dropped_atom_mass,
            "1",
            "reciprocal",
            Measure.INTEGRATED_EVENT_MASS,
        ),
    )
    results: list[dict[str, object]] = []
    for mutation_id, stage, expected, candidate, unit, frame, measure in candidates:
        comparison = compare_traces(
            (_trace(stage, np.asarray(expected), unit=unit, frame=frame, measure=measure),),
            (_trace(stage, np.asarray(candidate), unit=unit, frame=frame, measure=measure),),
            {stage: Tolerance(atol=_TOLERANCES["mutation_trace_atol"])},
        )
        results.append(
            {
                "mutation_id": mutation_id,
                "fixture_id": "mosaic.error_injection",
                "expected_first_stage": stage,
                "expected_failure_metric": "numeric_value",
                "observed_first_stage": comparison.first_failing_stage,
                "observed_failure_metric": comparison.failure_metric,
                "detected": comparison.first_failing_stage == stage
                and comparison.failure_metric == "numeric_value",
            }
        )
    _require(all(item["detected"] for item in results), "mosaic error injection escaped")
    return results


def _measure(call: Callable[[], object]) -> dict[str, float | int]:
    call()
    samples: list[float] = []
    for _ in range(5):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) * 1.0e-9)
    gc.collect()
    tracemalloc.start()
    retained = call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del retained
    return {
        "repetitions": 5,
        "median_wall_time_s": float(median(samples)),
        "python_traced_peak_bytes": int(peak),
    }


def _benchmark_evidence() -> dict[str, object]:
    case = _CONVERGENCE_CASES[1]
    initial, samples, states, rods, orientations, transform = _build_case(case, 4, 4)

    def public_call() -> EventBuildResult:
        timed_orientations = manuscript_axisymmetric_v1_orientation_quadrature(
            case.parameters,
            reciprocal_basis_Ainv=rods.reciprocal_basis_Ainv,
            alpha_cell_count=4,
            azimuth_cell_count=4,
        )
        return build_scattering_events(
            incident_samples=samples,
            incident_states=states,
            rods=rods,
            orientations=timed_orientations,
            sample_from_crystal=transform,
        )

    def dense_call() -> dict[str, object]:
        timed_orientations = manuscript_axisymmetric_v1_orientation_quadrature(
            case.parameters,
            reciprocal_basis_Ainv=rods.reciprocal_basis_Ainv,
            alpha_cell_count=4,
            azimuth_cell_count=4,
        )
        return _dense_case_events(states, rods, timed_orientations)

    thread_environment = {
        name: os.environ.get(name, "unset")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    _require(
        all(value == "1" for value in thread_environment.values()),
        "benchmark requires OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 before start",
    )
    return {
        "fixture_id": case.case_id,
        "equivalent_physical_work": True,
        "timing_scope": {
            "public": "orientation quadrature plus public event assembly",
            "dense": "orientation quadrature plus dense continuous-rod oracle",
            "excluded_setup": "case selection, incident samples/states, rods, transform, import, and I/O",
        },
        "warmup_repetitions": 1,
        "thread_environment": thread_environment,
        "workload": {
            "incident_states": int(states.incident_state_id.size),
            "rods": int(rods.rod_id.size),
            "orientation_nodes": int(orientations.orientation_id.size),
            "candidate_lines": int(initial.status.attempt_id.size),
            "dense_residual_nodes_per_line": 65,
            "emitted_events": int(initial.events.event_id.size),
        },
        "public": _measure(public_call),
        "dense": _measure(dense_call),
        "memory_scope": "Python-traced peak bytes; NumPy-native allocator peaks are not included",
        "threshold": None,
    }


def _reference_evidence(
    manifest: dict[str, Any], arrays: dict[str, NDArray[Any]]
) -> tuple[list[dict[str, object]], dict[str, float]]:
    cases = {case["case_id"]: case for case in manifest["cases"]}
    density_case = cases["mosaic.legacy_density"]
    ewald_case = cases["mosaic.ewald_intersection"]
    _require(
        density_case["classification"] == "CORRECTED"
        and density_case["first_divergence"] == "mosaic.probability_measure",
        "legacy mosaic classification mismatch",
    )
    _require(
        ewald_case["classification"] == "MATCH" and ewald_case["first_divergence"] is None,
        "legacy Ewald classification mismatch",
    )
    density = _legacy_density(
        arrays["mosaic_q_xyz"], arrays["mosaic_G"], arrays["mosaic_parameters"]
    )
    density_error = float(np.max(np.abs(density - arrays["mosaic_legacy_density"])))
    g_norm = float(np.linalg.norm(arrays["mosaic_G"]))
    support_errors = []
    elastic_errors = []
    dense_invariant_errors = []
    for prefix in ("ewald_uniform", "ewald_adaptive"):
        coordinate_error, elastic_error, dense_invariant_error = _legacy_circle_error(
            arrays[f"{prefix}_events"][:, :3],
            arrays["ewald_k_in"],
            float(arrays["ewald_k_scat"]),
            g_norm,
        )
        support_errors.append(coordinate_error)
        elastic_errors.append(elastic_error)
        dense_invariant_errors.append(dense_invariant_error)
        _require(int(arrays[f"{prefix}_status"]) == 0, "legacy Ewald status mismatch")
    support_error = max(support_errors)
    elastic_error = max(elastic_errors)
    dense_invariant_error = max(dense_invariant_errors)
    _require(
        density_error <= _TOLERANCES["legacy_density_trace_atol"],
        "legacy density trace mismatch",
    )
    _require(
        support_error <= _TOLERANCES["legacy_support_Ainv_atol"]
        and dense_invariant_error <= _TOLERANCES["legacy_support_Ainv_atol"],
        "legacy Ewald support mismatch",
    )
    classifications = [
        {
            "case_id": "mosaic.ewald_intersection",
            "classification": "MATCH",
            "first_divergence": None,
            "evidence": f"65,536-node dense two-sphere circle and tracked coordinates pass; max coordinate error {support_error:.3e} A^-1",
        },
        {
            "case_id": "mosaic.legacy_density",
            "classification": "CORRECTED",
            "first_divergence": "mosaic.probability_measure",
            "evidence": f"legacy density matches through divergence at {density_error:.3e}; corrected mass is normalized",
        },
        {
            "case_id": "mosaic.deterministic_source",
            "classification": "NO_ORACLE",
            "first_divergence": None,
            "evidence": "analytic probability normalization and deterministic enumeration",
        },
        {
            "case_id": "mosaic.continuous_rod_events",
            "classification": "NO_ORACLE",
            "first_divergence": None,
            "evidence": "analytic roots and independent dense line oracle",
        },
    ]
    return classifications, {
        "legacy_density_max_error": density_error,
        "legacy_support_max_error_Ainv": support_error,
        "legacy_elastic_max_error_Ainv": elastic_error,
        "legacy_dense_sphere_max_error_Ainv": dense_invariant_error,
    }


def _basic_checks(arrays: dict[str, NDArray[Any]]) -> tuple[list[dict[str, str]], dict[str, float]]:
    joint = compile_joint_source_samples(
        origin_lab_m=np.zeros((2, 3)),
        direction_lab=np.array([[0.0, 0.0, 1.0], [0.0, 0.6, 0.8]]),
        wavelength_A=np.array([1.0, 1.2]),
        probability_mass=np.array([0.4, 0.6]),
        polarization_state_id=("unity_scalar", "unity_scalar"),
    )
    independent_arguments = {
        "mean_origin_lab_m": np.zeros(3),
        "mean_direction_lab": np.array([0.0, 0.0, 1.0]),
        "transverse_axes_lab": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "spatial_sigma_m": np.array([1.0e-4, 2.0e-4]),
        "divergence_sigma_rad": np.array([0.01, 0.02]),
        "mean_wavelength_A": 1.1,
        "wavelength_sigma_A": 0.01,
        "spatial_order": 3,
        "direction_order": 3,
        "wavelength_order": 3,
        "polarization_state_id": "unity_scalar",
    }
    independent = compile_independent_gaussian_source_samples(**independent_arguments)
    independent_repeat = compile_independent_gaussian_source_samples(**independent_arguments)
    source_mass_error = max(
        abs(float(joint.source_weight.sum()) - 1.0),
        abs(float(independent.source_weight.sum()) - 1.0),
    )
    direction_error = float(np.max(np.abs(np.linalg.norm(independent.direction_lab, axis=1) - 1.0)))
    _require(source_mass_error <= _TOLERANCES["source_weight_sum_atol"], "source mass mismatch")
    _require(direction_error <= _TOLERANCES["direction_norm_atol"], "source direction mismatch")
    _require(
        joint.correlation_model == "explicit_joint"
        and independent.correlation_model == "independent_product"
        and all(value == "unity_scalar" for value in joint.polarization_state_id)
        and all(value == "unity_scalar" for value in independent.polarization_state_id),
        "source correlation or polarization metadata mismatch",
    )
    _require(
        all(
            np.array_equal(getattr(independent, name), getattr(independent_repeat, name))
            for name in (
                "incident_sample_id",
                "origin_lab_m",
                "direction_lab",
                "wavelength_A",
                "source_weight",
            )
        ),
        "source quadrature is not bitwise deterministic",
    )

    sigma_rad, half_width_rad, tail_probability = map(float, arrays["mosaic_parameters"])
    parameters = WrappedMosaicParameters(sigma_rad, half_width_rad, tail_probability)
    angle_count = 131_072
    angle = -np.pi + (np.arange(angle_count) + 0.5) * (2.0 * np.pi / angle_count)
    density = wrapped_mosaic_line_density_rad_inv(angle, parameters)
    normalization_error = abs(float(density.sum() * 2.0 * np.pi / angle_count) - 1.0)
    orientations = manuscript_axisymmetric_v1_orientation_quadrature(
        parameters,
        reciprocal_basis_Ainv=np.eye(3),
        alpha_cell_count=8,
        azimuth_cell_count=16,
    )
    orientation_mass_error = abs(float(orientations.probability_mass.sum()) - 1.0)
    direct_parameters = WrappedMosaicParameters(4.0, 0.0, 0.0)
    direct_orientations = manuscript_axisymmetric_v1_orientation_quadrature(
        direct_parameters,
        reciprocal_basis_Ainv=np.eye(3),
        alpha_cell_count=1,
        azimuth_cell_count=1,
    )
    direct_node, direct_weight = np.polynomial.legendre.leggauss(16)
    expected_alpha = 0.5 * np.pi * (direct_node + 1.0)
    expected_mass = (
        0.5 * np.pi * direct_weight * _oracle_folded_density(expected_alpha, direct_parameters)
    )
    direct_alpha_error = float(np.max(np.abs(direct_orientations.alpha_rad - expected_alpha)))
    direct_mass_error = float(np.max(np.abs(direct_orientations.probability_mass - expected_mass)))
    _require(
        normalization_error <= _TOLERANCES["wrapped_normalization_atol"],
        "wrapped mosaic normalization mismatch",
    )
    _require(
        orientation_mass_error <= _TOLERANCES["wrapped_normalization_atol"],
        "orientation mass mismatch",
    )
    _require(
        direct_alpha_error <= _TOLERANCES["public_dense_event_Ainv_atol"]
        and direct_mass_error <= _TOLERANCES["mutation_trace_atol"],
        "orientation rule is not direct-alpha probability quadrature",
    )

    incident = np.array([0.0, 0.0, 4.0])
    direction = np.array([0.0, 0.0, 1.0])
    two = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([1.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=1.0,
    )
    tangent = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([4.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=1.0,
    )
    none = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([4.1, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=1.0,
    )
    dense_tangent_status, dense_tangent_roots, dense_tangent_direct_count = (
        _dense_continuous_rod_roots(
            incident=np.array([4.0, 0.0, 0.0]),
            q0=np.zeros(3),
            direction=direction,
            b3_norm_Ainv=1.0,
        )
    )
    _require(
        two.status is EwaldRootStatus.TWO_ROOT
        and tangent.status is EwaldRootStatus.TANGENT
        and none.status is EwaldRootStatus.NO_ROOT
        and dense_tangent_status is EwaldRootStatus.TANGENT
        and dense_tangent_roots == ()
        and dense_tangent_direct_count == 1,
        "analytic Ewald status mismatch",
    )
    nonaxis_direction = np.array([1.0, 2.0, 2.0]) / 3.0
    nonaxis_q0 = np.array([1.0, -0.5, 0.0])
    line_shift_Ainv = 1050.0
    nonaxis_original = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=nonaxis_q0,
        d_hat_sample=nonaxis_direction,
        b3_norm_Ainv=1.0,
    )
    nonaxis_shifted = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=nonaxis_q0 + line_shift_Ainv * nonaxis_direction,
        d_hat_sample=nonaxis_direction,
        b3_norm_Ainv=1.0,
    )
    _require(
        nonaxis_original.status is nonaxis_shifted.status is EwaldRootStatus.TWO_ROOT
        and nonaxis_original.direct_beam_root_count
        == nonaxis_shifted.direct_beam_root_count
        == 0,
        "non-axis line-origin status changed",
    )
    original_roots = nonaxis_original.emittable_roots
    shifted_roots = nonaxis_shifted.emittable_roots
    line_q_error = max(
        float(np.max(np.abs(original.q_sample_Ainv - shifted.q_sample_Ainv)))
        for original, shifted in zip(original_roots, shifted_roots, strict=True)
    )
    line_kf_error = max(
        float(np.max(np.abs(original.kf_sample_Ainv - shifted.kf_sample_Ainv)))
        for original, shifted in zip(original_roots, shifted_roots, strict=True)
    )
    line_residual_error = max(
        abs(original.ewald_residual_Ainv - shifted.ewald_residual_Ainv)
        for original, shifted in zip(original_roots, shifted_roots, strict=True)
    )
    line_jacobian_relative_error = max(
        abs(original.coarea_jacobian - shifted.coarea_jacobian)
        / original.coarea_jacobian
        for original, shifted in zip(original_roots, shifted_roots, strict=True)
    )
    line_u_shift_error = max(
        abs(shifted.u_Ainv + line_shift_Ainv - original.u_Ainv)
        for original, shifted in zip(original_roots, shifted_roots, strict=True)
    )
    _require(
        max(line_q_error, line_kf_error, line_residual_error, line_u_shift_error)
        <= _TOLERANCES["public_dense_event_Ainv_atol"]
        and line_jacobian_relative_error
        <= _TOLERANCES["public_dense_event_weight_rtol"],
        "non-axis Ewald line origin changed a physical root",
    )
    direct_near_tangent = solve_continuous_rod_ewald(
        ki_sample_Ainv=np.array(
            [-0.5284075151649418, 3.839076919837225, 0.9910973219065503]
        ),
        q0_sample_Ainv=np.zeros(3),
        d_hat_sample=np.array(
            [-0.9464149529866339, -0.04780873374614963, -0.31939483674739827]
        ),
        b3_norm_Ainv=1.0,
    )
    _require(
        direct_near_tangent.status is EwaldRootStatus.TWO_ROOT
        and direct_near_tangent.direct_beam_root_count == 1
        and len(direct_near_tangent.emittable_roots) == 1,
        "direct-root line was lost near tangency",
    )
    residual_max = max(root.ewald_residual_Ainv for root in two.emittable_roots)
    residual_limit = (
        _TOLERANCES["elastic_residual_eps_factor"]
        * np.finfo(np.float64).eps
        * max(float(np.linalg.norm(incident)), 1.0)
    )
    _require(residual_max <= residual_limit, "analytic Ewald residual exceeds tolerance")
    checks = [
        {
            "check_id": "source_samples",
            "status": "PASS",
            "evidence": "joint correlations and 243-node independent Gaussian product preserve unit mass",
        },
        {
            "check_id": "mosaic_probability",
            "status": "PASS",
            "evidence": "wrapped components and direct-alpha folded measure integrate without pole density or probability resampling",
        },
        {
            "check_id": "analytic_ewald",
            "status": "PASS",
            "evidence": "two-root, exact tangent, no-root, non-axis line-origin, direct-root, residual, and unclipped Jacobian invariants pass",
        },
    ]
    return checks, {
        "source_mass_error": source_mass_error,
        "direction_norm_error": direction_error,
        "wrapped_normalization_error": normalization_error,
        "orientation_mass_error": orientation_mass_error,
        "direct_alpha_node_error_rad": direct_alpha_error,
        "direct_alpha_mass_error": direct_mass_error,
        "elastic_residual_max_Ainv": residual_max,
        "line_origin_q_error_Ainv": line_q_error,
        "line_origin_kf_error_Ainv": line_kf_error,
        "line_origin_residual_error_Ainv": line_residual_error,
        "line_origin_jacobian_relative_error": line_jacobian_relative_error,
        "line_origin_u_shift_error_Ainv": line_u_shift_error,
    }


def run_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    """Run the complete T03 proof without writing diagnostics."""

    del allow_missing_pack
    root = Path(__file__).resolve().parents[3]
    proof_base = os.environ.get("PROOF_BASE_SHA", "")
    _require(proof_base == _PROOF_BASE_SHA, "PROOF_BASE_SHA is unset or does not match T03")
    commit_sha = _verified_commit_sha(root)
    manifest, arrays = _load_reference_pack(root)
    classifications, reference_metrics = _reference_evidence(manifest, arrays)
    analytic_checks, analytic_metrics = _basic_checks(arrays)
    convergence = _convergence_evidence()
    mutations = _mutation_evidence()
    benchmark = _benchmark_evidence()
    checks = [
        {
            "check_id": "tracked_reference",
            "status": "PASS",
            "evidence": "immutable pack hash, legacy density trace, and Bragg-sphere support pass",
        },
        *analytic_checks,
        {
            "check_id": "dense_continuous_rod_oracle",
            "status": "PASS",
            "evidence": "six required fixtures agree with independent residual scans and converge deterministically",
        },
        {
            "check_id": "reference_performance",
            "status": "PASS",
            "evidence": "equivalent public/dense work measured after warmup with median wall time and Python peak memory",
        },
        {
            "check_id": "negative_controls",
            "status": "PASS",
            "evidence": f"{len(mutations)}/{len(mutations)} mosaic/Ewald mutations detected at first stage",
        },
    ]
    tolerance_payload = dict(_TOLERANCES)
    return {
        "schema_version": 1,
        "task_id": "T03",
        "status": "PASS",
        "base_sha": proof_base,
        "commit_sha": commit_sha,
        "contract_version": CONTRACT_API_VERSION,
        "trace_schema_version": 4,
        "reference_pack_sha256s": {"rasim_reference_v1": _REFERENCE_PACK_SHA256},
        "environment_sha256": _environment_sha256(),
        "tolerance_version": _TOLERANCE_VERSION,
        "tolerance_sha256": _canonical_sha256(tolerance_payload),
        "tolerances": tolerance_payload,
        "checks": checks,
        "metrics": {**reference_metrics, **analytic_metrics},
        "classifications": classifications,
        "convergence": convergence,
        "mutations": mutations,
        "benchmark": benchmark,
        "limitations": ["localized/adaptive acceleration is not implemented"],
    }
