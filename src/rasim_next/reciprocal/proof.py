"""Compact analytic, tracked-reference, and sparse-memory proof for T03."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import subprocess
import time
import tracemalloc
from dataclasses import dataclass, replace
from itertools import pairwise
from math import fsum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from painted_ewald import (
    EwaldSpherePainter,
    PaintMeasure,
    enumerate_rods_within_ewald_sphere,
)
from painted_ewald import (
    ForwardPolicy as PaintedForwardPolicy,
)
from painted_ewald import (
    MosaicParameters as PaintedMosaicParameters,
)
from painted_ewald import (
    PainterConfig as PaintedPainterConfig,
)
from painted_ewald import (
    RasterParameters as PaintedRasterParameters,
)
from painted_ewald import (
    Rod as PaintedRod,
)
from rasim_next.core.contracts import (
    CONTRACT_API_VERSION,
    IncidentSampleBatch,
    IncidentStateBatch,
    RodCatalog,
    canonical_revision_sha256,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.validity import ValidityCode
from rasim_next.proof.tolerances import STAGE_TOLERANCE_SHA256, load_stage_tolerances
from rasim_next.reciprocal.events import EventBuildResult, build_scattering_events
from rasim_next.reciprocal.ewald import EwaldRootStatus, solve_continuous_rod_ewald
from rasim_next.sampling.mosaic import (
    MosaicOrientationBatch,
    WrappedMosaicParameters,
    manuscript_axisymmetric_v1_orientation_quadrature,
    wrapped_mosaic_line_density_rad_inv,
)
from rasim_next.sampling.source import sample_gaussian_source_rays

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


def _environment_sha256() -> str:
    identity = f"{platform.python_implementation()}|{platform.python_version()}|{np.__version__}"
    return hashlib.sha256(identity.encode()).hexdigest()


def _verified_checkout(root: Path) -> tuple[str, tuple[str, ...]]:
    git_environment = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")
    }
    options = {
        "check": True,
        "capture_output": True,
        "cwd": root,
        "env": git_environment,
        "text": True,
        "timeout": 10.0,
    }
    identity = subprocess.run(
        ["git", "rev-parse", "--show-toplevel", "--verify", "HEAD^{commit}"], **options
    )
    reported_root, commit_sha = identity.stdout.splitlines()
    _require(Path(reported_root).resolve() == root.resolve(), "proof checkout root mismatch")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
        **options,
    )
    changed_paths = tuple(line[3:].replace("\\", "/") for line in status.stdout.splitlines())
    return commit_sha, changed_paths


def _load_reference_pack(root: Path) -> dict[str, NDArray[Any]]:
    path = root / "reference" / "rasim_reference_v1.npz"
    _require(path.is_file(), "reference pack is missing")
    pack_bytes = path.read_bytes()
    _require(
        hashlib.sha256(pack_bytes).hexdigest() == _REFERENCE_PACK_SHA256,
        "reference pack hash mismatch",
    )
    with np.load(io.BytesIO(pack_bytes), allow_pickle=False) as data:
        arrays = {
            name: np.array(data[name], copy=True) for name in data.files if name != "manifest_json"
        }
    return arrays


def _reference_evidence(
    arrays: dict[str, NDArray[Any]], tolerances: Any
) -> list[dict[str, object]]:
    q = arrays["mosaic_q_xyz"]
    reciprocal_vector = arrays["mosaic_G"]
    q_elevation = np.arctan2(q[:, 2], np.linalg.norm(q[:, :2], axis=1))
    reciprocal_elevation = np.arctan2(reciprocal_vector[2], np.linalg.norm(reciprocal_vector[:2]))
    offset = np.remainder(q_elevation - reciprocal_elevation + np.pi, 2.0 * np.pi) - np.pi
    parameters = WrappedMosaicParameters(*map(float, arrays["mosaic_parameters"]))
    public_line_density = wrapped_mosaic_line_density_rad_inv(offset, parameters)
    legacy_line_density = arrays["mosaic_legacy_density"] * (
        2.0 * np.pi * np.dot(reciprocal_vector, reciprocal_vector)
    )
    density_difference = np.abs(public_line_density - legacy_line_density)
    density_peak = float(wrapped_mosaic_line_density_rad_inv(np.zeros(1), parameters)[0])
    density_limit = tolerances["mosaic.wrapped_line_density"].bind(density_peak).limit
    _require(
        float(density_difference.max()) > density_limit,
        "wrapped public density did not diverge beyond the shared tolerance",
    )

    incident = arrays["ewald_k_in"]
    scattered_norm = float(arrays["ewald_k_scat"])
    reciprocal_norm = float(np.linalg.norm(arrays["mosaic_G"]))
    elastic_bound = (
        tolerances["reciprocal.ewald_residual"].bind(float(np.linalg.norm(incident))).limit
    )
    reciprocal_bound = tolerances["reciprocal.event_q_internal"].bind(reciprocal_norm).limit
    elastic_error = 0.0
    reciprocal_error = 0.0
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
        _require(int(arrays[f"{prefix}_status"]) == 0, "tracked Ewald status mismatch")
    _require(
        elastic_error <= elastic_bound and reciprocal_error <= reciprocal_bound,
        "tracked Ewald events violate elastic or reciprocal-sphere invariants",
    )
    rows = (
        ("mosaic.ewald_intersection", "MATCH", "PHY-REC-002 PHY-REC-003"),
        ("mosaic.legacy_density", "CORRECTED", "PHY-MOS-001 PHY-MOS-002 PHY-MOS-003 PHY-MOS-004"),
        (
            "mosaic.seeded_gaussian_source",
            "NO_ORACLE",
            "PHY-SRC-001 PHY-SRC-002 PHY-SRC-003 PHY-SRC-005 PHY-SRC-006",
        ),
        (
            "mosaic.continuous_rod_events",
            "NO_ORACLE",
            "PHY-MOS-005 PHY-REC-004 PHY-REC-005 PHY-REC-006 PHY-REC-007 PHY-REC-009",
        ),
    )
    classifications = [
        {
            "case_id": case_id,
            "classification": classification,
            "ledger_ids": ledger_ids.split(),
        }
        for case_id, classification, ledger_ids in rows
    ]
    classifications[1]["first_divergence_stage"] = "mosaic.wrapped_line_density"
    return classifications


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
    analytic_zero_tilt: bool = False,
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
    provenance = "T03 analytic reciprocal fixture.v1"
    source_weight = np.full(sample_count, 1.0 / sample_count)
    origin_lab_m = np.zeros((sample_count, 3))
    direction_lab = np.tile([0.0, 0.0, 1.0], (sample_count, 1))
    polarization_ids = ("proof_state",) * sample_count
    samples = IncidentSampleBatch(
        incident_sample_id=sample_ids,
        origin_lab_m=origin_lab_m,
        direction_lab=direction_lab,
        wavelength_A=wavelengths_A,
        source_weight=source_weight,
        polarization_state_id=polarization_ids,
        source_sampling_model_id="explicit_proof_source.v1",
        source_rng_model_id="no_rng.v1",
        source_seed=0,
        source_parameter_provenance=provenance,
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
        wavelength_A=wavelengths_A,
        polarization_state_id=polarization_ids,
        status=(ValidityCode.VALID,) * sample_count,
        valid=np.ones(sample_count, dtype=np.bool_),
        source_sampling_model_id=samples.source_sampling_model_id,
        source_rng_model_id=samples.source_rng_model_id,
        source_seed=samples.source_seed,
        source_parameter_provenance=samples.source_parameter_provenance,
        source_parameter_revision=samples.source_parameter_revision,
        source_revision=samples.source_revision,
        sample_geometry_revision=canonical_revision_sha256(
            ("sample_geometry", "T03 analytic reciprocal fixture.v1")
        ),
        material_revision=canonical_revision_sha256(
            ("material", "T03 analytic reciprocal fixture.v1")
        ),
        incident_model_id="one_transmitted_channel.v1",
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
    orientations = manuscript_axisymmetric_v1_orientation_quadrature(
        parameters,
        reciprocal_basis_Ainv=basis,
        alpha_cell_count=alpha_cell_count,
        azimuth_cell_count=azimuth_cell_count,
    )
    if analytic_zero_tilt:
        _require(
            orientations.orientation_id.size == 1 and parameters.zero_tilt_probability_mass == 1.0,
            "analytic zero-tilt fixture requires one atomic orientation",
        )
        orientations = MosaicOrientationBatch(
            np.array([0]),
            np.array([0.0]),
            np.array([np.pi]),
            np.diag([-1.0, -1.0, 1.0])[None, :, :],
            np.array([1.0]),
            basis,
            "manuscript_axisymmetric_v1",
        )
    transform = RigidTransform(np.eye(3), np.zeros(3), FrameId.CRYSTAL, FrameId.SAMPLE)
    return samples, states, rods, orientations, transform


def _dense_events(
    states: IncidentStateBatch,
    rods: RodCatalog,
    orientations: MosaicOrientationBatch,
    transform: RigidTransform,
) -> dict[str, object]:
    basis = rods.reciprocal_basis_Ainv
    b3_norm = float(np.linalg.norm(basis[:, 2]))
    b3_hat = basis[:, 2] / b3_norm
    event_rows: list[tuple[Any, ...]] = []
    attempt_rows: list[tuple[Any, ...]] = []
    for state_index, valid in enumerate(states.valid):
        if not valid:
            continue
        state_id = int(states.incident_state_id[state_index])
        incident = states.k_film_phase_sample_Ainv[state_index]
        wavelength = float(states.wavelength_A[state_index])
        for rod_id, h, k in zip(rods.rod_id, rods.h, rods.k, strict=True):
            q0_crystal = int(h) * basis[:, 0] + int(k) * basis[:, 1]
            for orientation_index, mass_value in enumerate(orientations.probability_mass):
                mass = float(mass_value)
                if mass == 0.0:
                    continue
                rotation = transform.rotation @ orientations.rotation_crystal[orientation_index]
                q0 = rotation @ q0_crystal
                direction = rotation @ b3_hat
                status, roots, direct_count = _dense_roots(incident, q0, direction, b3_norm)
                orientation_id = int(orientations.orientation_id[orientation_index])
                attempt_rows.append(
                    (state_id, int(rod_id), orientation_id, status, len(roots), direct_count)
                )
                for root in roots:
                    event_rows.append(
                        (
                            state_id,
                            orientation_id,
                            int(rod_id),
                            wavelength,
                            root.q_sample_Ainv,
                            root.l_coordinate,
                            root.kf_sample_Ainv,
                            mass * root.coarea_jacobian,
                            root.ewald_residual_Ainv,
                        )
                    )
    event_count, attempt_count = len(event_rows), len(attempt_rows)
    event_columns = list(zip(*event_rows, strict=True)) if event_rows else [()] * 9
    attempt_columns = list(zip(*attempt_rows, strict=True)) if attempt_rows else [()] * 6
    q_array = np.asarray(event_columns[4], dtype=np.float64).reshape((-1, 3))
    return {
        "event_id": np.arange(event_count, dtype=np.int64),
        "event_state_id": np.asarray(event_columns[0], dtype=np.int64),
        "event_orientation_id": np.asarray(event_columns[1], dtype=np.int64),
        "event_rod_id": np.asarray(event_columns[2], dtype=np.int64),
        "wavelength": np.asarray(event_columns[3], dtype=np.float64),
        "q": q_array,
        "q_sample_normal": q_array[:, 2],
        "l": np.asarray(event_columns[5], dtype=np.float64),
        "kf": np.asarray(event_columns[6], dtype=np.float64).reshape((-1, 3)),
        "weight": np.asarray(event_columns[7], dtype=np.float64),
        "residual": np.asarray(event_columns[8], dtype=np.float64),
        "valid": np.ones(event_count, dtype=np.bool_),
        "event_status": (ValidityCode.VALID,) * event_count,
        "attempt_id": np.arange(attempt_count, dtype=np.int64),
        "attempt_state_id": np.asarray(attempt_columns[0], dtype=np.int64),
        "attempt_rod_id": np.asarray(attempt_columns[1], dtype=np.int64),
        "attempt_orientation_id": np.asarray(attempt_columns[2], dtype=np.int64),
        "status": tuple(attempt_columns[3]),
        "emitted_count": np.asarray(attempt_columns[4], dtype=np.int8),
        "direct_count": np.asarray(attempt_columns[5], dtype=np.int8),
    }


def _compare_public_dense(
    public: EventBuildResult,
    dense: dict[str, object],
    tolerances: Any,
    incident_norm_Ainv: float,
    b3_norm_Ainv: float,
    *,
    require_exact_numeric: bool = False,
) -> None:
    exact_pairs = (
        (public.events.event_id, dense["event_id"]),
        (public.events.incident_state_id, dense["event_state_id"]),
        (public.events.orientation_id, dense["event_orientation_id"]),
        (public.events.rod_id, dense["event_rod_id"]),
        (public.events.wavelength_A, dense["wavelength"]),
        (public.events.valid, dense["valid"]),
        (public.status.attempt_id, dense["attempt_id"]),
        (public.status.incident_state_id, dense["attempt_state_id"]),
        (public.status.rod_id, dense["attempt_rod_id"]),
        (public.status.orientation_id, dense["attempt_orientation_id"]),
        (public.status.emitted_root_count, dense["emitted_count"]),
        (public.status.direct_beam_root_count, dense["direct_count"]),
    )
    _require(
        all(
            left.dtype == np.asarray(right).dtype and np.array_equal(left, right)
            for left, right in exact_pairs
        )
        and public.events.status == dense["event_status"]
        and public.status.root_status == dense["status"],
        "public identities, ordering, or statuses differ from the dense oracle",
    )

    dense_q = np.asarray(dense["q"])
    dense_kf = np.asarray(dense["kf"])
    q_scale = max(
        incident_norm_Ainv,
        float(np.max(np.linalg.norm(dense_q, axis=1), initial=0.0)),
    )
    kf_scale = max(
        incident_norm_Ainv,
        float(np.max(np.linalg.norm(dense_kf, axis=1), initial=0.0)),
    )
    weight_scale = float(np.max(np.abs(np.asarray(dense["weight"])), initial=0.0))
    numeric_fields = (
        ("q_internal_sample_Ainv", "q", "reciprocal.event_q_internal", q_scale),
        ("q_sample_normal_Ainv", "q_sample_normal", "reciprocal.event_q_internal", q_scale),
        ("kf_film_phase_sample_Ainv", "kf", "optics.kf_film_sample", kf_scale),
        ("reciprocal_weight", "weight", "reciprocal.event_weight", weight_scale),
        ("ewald_residual_Ainv", "residual", "reciprocal.ewald_residual", incident_norm_Ainv),
    )
    maximum_error = 0.0
    for public_name, dense_name, stage, scale in numeric_fields:
        observed = getattr(public.events, public_name)
        expected = np.asarray(dense[dense_name])
        _require(
            observed.shape == expected.shape and observed.dtype == expected.dtype,
            f"{public_name} shape or dtype differs from dense oracle",
        )
        error = float(np.max(np.abs(observed - expected), initial=0.0))
        _require(error <= tolerances[stage].bind(scale).limit, f"{public_name} exceeds tolerance")
        maximum_error = max(maximum_error, error)

    dense_l = np.asarray(dense["l"])
    _require(
        public.events.l_coordinate.shape == dense_l.shape
        and public.events.l_coordinate.dtype == dense_l.dtype,
        "l_coordinate shape or dtype differs from dense oracle",
    )
    l_error_Ainv = b3_norm_Ainv * float(
        np.max(np.abs(public.events.l_coordinate - dense_l), initial=0.0)
    )
    l_limit = tolerances["reciprocal.intersection_support"].bind(q_scale).limit
    _require(l_error_Ainv <= l_limit, "L displacement exceeds reciprocal tolerance")
    maximum_error = max(maximum_error, l_error_Ainv)
    if require_exact_numeric:
        _require(maximum_error == 0.0, "numeric sparse oracle mismatch")


def _observables(
    q: NDArray[np.float64], weight: NDArray[np.float64], residual: NDArray[np.float64]
) -> dict[str, object]:
    total_mass = float(weight.sum())
    _require(total_mass > 0.0, "proof case emitted no reciprocal mass")
    order = np.argsort(q[:, 2])
    sorted_weight = weight[order]
    cumulative = (np.cumsum(sorted_weight) - 0.5 * sorted_weight) / total_mass
    centroid = (np.sum(weight[:, None] * q, axis=0) / total_mass).tolist()
    quantiles = np.interp((0.1, 0.9), cumulative, q[order, 2]).tolist()
    return {
        "total_reciprocal_mass": total_mass,
        "weighted_Q_centroid_sample_Ainv": centroid,
        "weighted_q_sample_normal_quantiles_Ainv": quantiles,
        "maximum_ewald_residual_Ainv": float(np.max(residual, initial=0.0)),
    }


def _oracle_evidence(tolerances: Any) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    atom = WrappedMosaicParameters(0.0, 0.0, 0.0)

    def evaluate(
        case_id: str,
        parameters: WrappedMosaicParameters,
        h_values: tuple[int, ...],
        wavelengths: tuple[float, ...],
        alpha_count: int,
        azimuth_count: int,
        b1_Ainv: float = 1.0,
        analytic_tangent: bool = False,
    ) -> dict[str, object]:
        batches = _build_fixture(
            np.asarray(h_values, dtype=np.int32),
            np.asarray(wavelengths),
            parameters,
            alpha_cell_count=alpha_count,
            azimuth_cell_count=azimuth_count,
            b1_Ainv=b1_Ainv,
            analytic_zero_tilt=analytic_tangent,
        )
        _, states, rods, orientations, transform = batches
        public = build_scattering_events(
            incident_states=states,
            rods=rods,
            orientations=orientations,
            sample_from_crystal=transform,
        )
        dense = _dense_events(states, rods, orientations, transform)
        incident_norm = float(np.max(np.linalg.norm(states.k_film_phase_sample_Ainv, axis=1)))
        _compare_public_dense(
            public,
            dense,
            tolerances,
            incident_norm,
            float(np.linalg.norm(rods.reciprocal_basis_Ainv[:, 2])),
            require_exact_numeric=analytic_tangent,
        )
        statuses = public.status.root_status
        return {
            "case_id": case_id,
            "event_count": int(public.events.event_id.size),
            "status_counts": {status.value: statuses.count(status) for status in EwaldRootStatus},
            "suppressed_direct_roots": int(public.status.direct_beam_root_count.sum()),
            "public": _observables(
                public.events.q_internal_sample_Ainv,
                public.events.reciprocal_weight,
                public.events.ewald_residual_Ainv,
            ),
            "ordered_dense_oracle_match": True,
            "_dense_weight": np.asarray(dense["weight"]),
            "_alpha": orientations.alpha_rad,
        }

    configurations = (
        ("narrow", WrappedMosaicParameters(0.01, 0.08, 0.05), (1,), (np.pi / 2.0,), 4, 4),
        ("broad", WrappedMosaicParameters(0.2, 0.35, 0.2), (1,), (np.pi / 2.0,), 4, 4),
        ("lorentz_tail", WrappedMosaicParameters(0.02, 0.25, 0.85), (1,), (np.pi / 2.0,), 4, 4),
        ("analytic_tangent_no_root", atom, (1, 4, 5), (np.pi / 2.0,), 1, 1, 1.0, True),
        ("bandwidth_specular", atom, (0,), (np.pi / 2.0, 1.1), 1, 1),
    )
    matrix = [evaluate(*configuration) for configuration in configurations]
    _require(
        matrix[3]["status_counts"] == {"TWO_ROOT": 1, "TANGENT": 1, "NO_ROOT": 1}
        and matrix[4]["suppressed_direct_roots"] == 2,
        "named support regimes changed",
    )
    for row in matrix:
        del row["_dense_weight"], row["_alpha"]

    levels = (5, 10, 20)
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
    dense_weights = [np.asarray(row.pop("_dense_weight")) for row in refinement]
    alpha_nodes = [np.asarray(row.pop("_alpha")) for row in refinement]
    panel_edges = np.array([0.0, 0.2, 0.4, 0.8, 1.6, np.pi])
    node_counts = np.array([np.histogram(alpha, bins=panel_edges)[0] for alpha in alpha_nodes])
    _require(
        np.array_equal(node_counts, np.array([[64] * 5, [128] * 5, [256] * 5])),
        "5/10/20 quadrature is not exact per-segment panel doubling",
    )

    mass = np.array([row["public"]["total_reciprocal_mass"] for row in refinement])
    mass_changes = np.abs(np.diff(mass))
    epsilon = np.finfo(np.float64).eps
    mass_floor = 0.0
    for weight in dense_weights[1:]:
        count = weight.size
        gamma = (count - 1) * epsilon / (1.0 - (count - 1) * epsilon)
        mass_floor += gamma * float(np.sum(np.abs(weight)))
    quantile_changes = np.max(
        np.abs(
            np.diff(
                [row["public"]["weighted_q_sample_normal_quantiles_Ainv"] for row in refinement],
                axis=0,
            )
        ),
        axis=1,
    )
    q_floor = tolerances["reciprocal.event_q_internal"].bind(8.0).limit
    _require(
        mass_changes[1] <= max(0.5 * mass_changes[0], mass_floor)
        and quantile_changes[1] <= max(0.5 * quantile_changes[0], q_floor),
        "quadrature refinement failed mass or Q-normal contraction",
    )
    return matrix, [
        {
            "case_id": "regular_quadrature_refinement",
            "refinement_variable": "alpha_cell_count",
            "levels": list(levels),
            "observables": [row["public"] for row in refinement],
            "panel_node_counts": node_counts.tolist(),
            "successive_total_mass_change": mass_changes.tolist(),
            "binary64_mass_floor": mass_floor,
            "successive_q_sample_normal_quantile_max_change_Ainv": quantile_changes.tolist(),
            "shared_q_floor_Ainv": q_floor,
            "criterion": "fine <= max(0.5*coarse, independent floor)",
            "assessment": "PASS",
        }
    ]


def _benchmark_evidence(tolerances: Any) -> dict[str, object]:
    h = np.arange(5, 4101, dtype=np.int32)
    h[2047:2050] = (0, 1, 4)
    _, states, rods, orientations, transform = _build_fixture(
        h,
        np.array([np.pi / 2.0]),
        WrappedMosaicParameters(0.0, 0.0, 0.0),
        alpha_cell_count=1,
        azimuth_cell_count=1,
        analytic_zero_tilt=True,
    )
    tracemalloc.start()
    start = time.perf_counter()
    public = build_scattering_events(
        incident_states=states,
        rods=rods,
        orientations=orientations,
        sample_from_crystal=transform,
    )
    public_wall_seconds = time.perf_counter() - start
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    start = time.perf_counter()
    dense = _dense_events(states, rods, orientations, transform)
    dense_wall_seconds = time.perf_counter() - start
    _compare_public_dense(
        public,
        dense,
        tolerances,
        float(np.linalg.norm(states.k_film_phase_sample_Ainv[0])),
        float(np.linalg.norm(rods.reciprocal_basis_Ainv[:, 2])),
        require_exact_numeric=True,
    )
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
    fill_fields = (
        "incident_state_id",
        "orientation_id",
        "rod_id",
        "wavelength_A",
        "q_internal_sample_Ainv",
        "l_coordinate",
        "kf_film_phase_sample_Ainv",
        "reciprocal_weight",
        "ewald_residual_Ainv",
    )
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
        "traced_live_bytes_at_return": current_bytes,
        "traced_peak_bytes": peak_bytes,
        "temporary_peak_working_bytes": peak_bytes - current_bytes,
        "former_two_root_preallocator_bytes": former_preallocator_bytes,
        "measurement_boundary": "fixture setup excluded; public wall time includes tracemalloc",
    }


def _scientific_evidence(
    tolerances: Any,
) -> list[dict[str, object]]:
    source = sample_gaussian_source_rays(
        mean_origin_lab_m=np.zeros(3),
        mean_direction_lab=np.array([0.0, 0.0, 1.0]),
        transverse_axes_lab=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        spatial_sigma_m=np.array([2.0e-4, 3.0e-4]),
        divergence_sigma_rad=np.array([0.02, 0.03]),
        mean_wavelength_A=1.2,
        wavelength_sigma_A=0.04,
        sample_count=4097,
        seed=1729,
        polarization_state_id="proof_state",
    )
    radius = np.arccos(np.clip(source.direction_lab[:, 2], -1.0, 1.0))
    inverse_sine = np.divide(radius, np.sin(radius), out=np.ones_like(radius), where=radius != 0.0)
    standardized = np.column_stack(
        (
            source.origin_lab_m[:, 0] / 2.0e-4,
            source.origin_lab_m[:, 1] / 3.0e-4,
            source.direction_lab[:, 0] * inverse_sine / 0.02,
            source.direction_lab[:, 1] * inverse_sine / 0.03,
            (source.wavelength_A - 1.2) / 0.04,
        )
    )
    source_mass_error = abs(float(source.source_weight.sum()) - 1.0)
    source_correlation = np.corrcoef(standardized, rowvar=False)
    source_variance = np.sum(source.source_weight[:, None] * standardized**2, axis=0)
    pdf_weight = np.exp(-0.5 * np.sum(standardized * standardized, axis=1))
    pdf_weight /= pdf_weight.sum()
    double_weighted_variance = np.sum(pdf_weight[:, None] * standardized**2, axis=0)
    incident = np.array([0.0, 0.0, 4.0])
    direction = np.array([0.0, 0.0, 1.0])
    two = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([1.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    _require(
        source_mass_error <= tolerances["sampling.source_empirical_mass"].bind(1.0).limit
        and np.array_equal(source.source_weight, np.full(4097, 1.0 / 4097))
        and source.polarization_state_id == ("proof_state",) * 4097
        and source.source_sampling_model_id == "independent_gaussian_antithetic_lhs.v2"
        and source.source_rng_model_id == "numpy_pcg64.v1"
        and source.source_seed == 1729
        and len(source.source_parameter_revision) == 64
        and len(source.source_revision) == 64
        and np.max(np.abs(np.mean(standardized, axis=0))) < 1.0e-12
        and np.max(np.abs(source_variance - 1.0)) < 0.25
        and np.max(np.abs(source_correlation - np.eye(5))) < 0.05
        and np.max(np.abs(double_weighted_variance - 1.0)) >= 0.25
        and two.status is EwaldRootStatus.TWO_ROOT,
        "sampled source, mosaic, or coarea fixture invariant failed",
    )

    control = _build_fixture(
        np.array([1], dtype=np.int32),
        np.array([np.pi / 2.0]),
        WrappedMosaicParameters(0.45, 0.7, 0.35),
        alpha_cell_count=5,
        azimuth_cell_count=1,
    )
    _, states, rods, orientations, transform = control
    public = build_scattering_events(
        incident_states=states,
        rods=rods,
        orientations=orientations,
        sample_from_crystal=transform,
    )
    dense = _dense_events(states, rods, orientations, transform)
    incident_norm = float(np.linalg.norm(states.k_film_phase_sample_Ainv[0]))
    b3_norm = float(np.linalg.norm(rods.reciprocal_basis_Ainv[:, 2]))
    _compare_public_dense(public, dense, tolerances, incident_norm, b3_norm)

    reference_weight = np.asarray(dense["weight"])
    correct_weight = public.events.reciprocal_weight
    orientation_id = public.events.orientation_id
    orientation_mass = orientations.probability_mass[orientation_id]
    alpha = orientations.alpha_rad[orientation_id]
    jacobian = reference_weight / orientation_mass
    gaussian_count = manuscript_axisymmetric_v1_orientation_quadrature(
        WrappedMosaicParameters(0.45, 0.0, 0.0),
        reciprocal_basis_Ainv=rods.reciprocal_basis_Ainv,
        alpha_cell_count=5,
        azimuth_cell_count=1,
    ).orientation_id.size
    component_probability = np.where(orientation_id < gaussian_count, 0.65, 0.35)
    event_limit = (
        tolerances["reciprocal.event_weight"].bind(float(np.max(np.abs(reference_weight)))).limit
    )
    correct_event_error = float(np.max(np.abs(correct_weight - reference_weight)))
    event_mutants = (
        (
            "removed_spherical_measure",
            "mosaic.spherical_measure.v1",
            correct_weight * np.sin(alpha),
        ),
        (
            "mixture_misnormalization",
            "mosaic.mixture_normalization.v1",
            correct_weight / component_probability,
        ),
        ("reversed_signed_arc_measure", "mosaic.signed_arc.v1", -correct_weight),
        ("omitted_coarea_jacobian", "reciprocal.coarea.v1", orientation_mass),
        (
            "duplicate_empirical_lorentz_factor",
            "reciprocal.single_lorentz.v1",
            correct_weight * jacobian,
        ),
    )
    control_rows: list[tuple[str, str, str, str, float, float, float]] = []
    for mutation_id, fixture_id, mutant_weight in event_mutants:
        mutant_error = float(np.max(np.abs(mutant_weight - reference_weight)))
        control_rows.append(
            (
                mutation_id,
                fixture_id,
                "reciprocal.event_weight",
                "maximum_event_weight_error",
                event_limit,
                correct_event_error,
                mutant_error,
            )
        )

    source_correct_error = float(np.max(np.abs(source_variance - 1.0)))
    source_mutant_error = float(np.max(np.abs(double_weighted_variance - 1.0)))
    control_rows.append(
        (
            "pdf_double_weighting",
            "source.pdf_double_weight.v1",
            "sampling.source_empirical_mass",
            "variance_from_analytic_one",
            0.25,
            source_correct_error,
            source_mutant_error,
        )
    )

    root = two.emittable_roots[0]
    bad_q = root.q_sample_Ainv + 1.0e-3 * direction
    bad_residual = abs(float(np.linalg.norm(incident + bad_q) - np.linalg.norm(incident)))
    residual_limit = (
        tolerances["reciprocal.ewald_residual"].bind(float(np.linalg.norm(incident))).limit
    )
    correct_residual = max(item.ewald_residual_Ainv for item in two.emittable_roots)
    control_rows.append(
        (
            "accepted_bad_residual_root",
            "reciprocal.residual_rejection.v1",
            "reciprocal.ewald_residual",
            "accepted_ewald_residual_Ainv",
            residual_limit,
            correct_residual,
            bad_residual,
        )
    )
    _require(
        len(control_rows) == 7
        and all(correct < limit <= mutant for *_, limit, correct, mutant in control_rows),
        "one or more controls failed the common correct-PASS/mutant-FAIL rule",
    )
    mutations = [
        {
            "mutation_id": mutation_id,
            "fixture_id": fixture_id,
            "observed_first_stage": stage,
            "observed_failure_metric": metric,
            "gate_limit": limit,
            "correct_error": correct,
            "correct_status": "PASS",
            "mutant_error": mutant,
            "mutant_status": "FAIL",
            "detected": True,
        }
        for mutation_id, fixture_id, stage, metric, limit, correct, mutant in control_rows
    ]
    return mutations


class _PaintedProofStrength:
    __slots__ = ()

    def evaluate(self, *, rod: PaintedRod, L: float, k_norm_Ainv: float) -> float:
        return 1.0 + 0.03 * rod.h + 0.05 * rod.k + 1.0e-10 * L * L + 0.02 * k_norm_Ainv


def _painted_skew_config(*, azimuth_count: int, azimuth_phase_rad: float) -> PaintedPainterConfig:
    direction = np.array([1.0, 2.0, 3.0])
    direction /= np.linalg.norm(direction)
    first_perpendicular = np.array([2.0, -1.0, 0.0])
    first_perpendicular -= np.dot(first_perpendicular, direction) * direction
    first_perpendicular *= 1.5 / np.linalg.norm(first_perpendicular)
    second_perpendicular = np.cos(np.pi / 3.0) * first_perpendicular + np.sin(
        np.pi / 3.0
    ) * np.cross(direction, first_perpendicular)
    reciprocal_basis = np.column_stack(
        (
            first_perpendicular + 1.0e4 * direction,
            second_perpendicular - 3.0e3 * direction,
            0.3 * direction,
        )
    )
    return PaintedPainterConfig(
        reciprocal_basis_Ainv=reciprocal_basis,
        crystal_to_sample=np.eye(3),
        rods=(PaintedRod(1, 0), PaintedRod(0, 1)),
        mosaic=PaintedMosaicParameters(
            gaussian_sigma_rad=np.deg2rad(1.0),
            lorentzian_half_width_rad=np.deg2rad(0.5),
            lorentzian_probability=0.2,
            alpha_panel_count=8,
            alpha_gauss_order=8,
            azimuth_count=azimuth_count,
            azimuth_phase_rad=azimuth_phase_rad,
        ),
        raster=PaintedRasterParameters(8, 16),
    )


def _painted_tilted_config(*, azimuth_count: int, azimuth_phase_rad: float) -> PaintedPainterConfig:
    reciprocal_basis = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.0, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 0.2],
        ]
    )
    return PaintedPainterConfig(
        reciprocal_basis_Ainv=reciprocal_basis,
        crystal_to_sample=np.eye(3),
        rods=(PaintedRod(1, 0),),
        mosaic=PaintedMosaicParameters(
            gaussian_sigma_rad=np.deg2rad(5.0),
            lorentzian_half_width_rad=np.deg2rad(2.0),
            lorentzian_probability=0.1,
            alpha_panel_count=8,
            alpha_gauss_order=8,
            azimuth_count=azimuth_count,
            azimuth_phase_rad=azimuth_phase_rad,
        ),
        raster=PaintedRasterParameters(16, 32),
    )


def _painted_point_moments(points: tuple[Any, ...]) -> tuple[list[float], list[float]]:
    weights = np.fromiter((point.weight for point in points), dtype=np.float64, count=len(points))
    total = fsum(float(value) for value in weights)
    _require(total > 0.0, "painted point coating has no retained mass")
    directions = np.stack([point.kf_sample_Ainv for point in points])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    first = np.einsum("i,ij->j", weights, directions, optimize=True) / total
    second = np.einsum("i,ij,ik->jk", weights, directions, directions, optimize=True) / total
    return first.tolist(), second[np.triu_indices(3)].tolist()


def _symmetric_relative_difference(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    scale = np.maximum(0.5 * (np.abs(left) + np.abs(right)), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(left - right) / scale, initial=0.0))


def _painted_convergence_evidence() -> dict[str, object]:
    incident = np.array([0.2, 0.4, -10.0])
    phases = (0.371, 1.137)
    levels = (8, 16, 32)
    rows: list[dict[str, object]] = []
    numerical: dict[tuple[float, int], tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for phase in phases:
        for level in levels:
            painter = EwaldSpherePainter(
                _painted_tilted_config(
                    azimuth_count=level,
                    azimuth_phase_rad=phase,
                ),
                _PaintedProofStrength(),
            )
            painted = painter.paint(incident)
            branch_mass = np.asarray(
                [
                    fsum(point.weight for point in painted.points if point.branch == branch)
                    for branch in (1, 2)
                ]
            )
            branch_moments: list[float] = []
            for branch in (1, 2):
                first, second = _painted_point_moments(
                    tuple(point for point in painted.points if point.branch == branch)
                )
                branch_moments.extend(first + second)
            moments = np.asarray(branch_moments)
            numerical[(phase, level)] = (branch_mass, moments)
            rows.append(
                {
                    "azimuth_phase_rad": phase,
                    "azimuth_count": level,
                    "orientation_count": painter.mosaic_space.orientation_id.size,
                    "branch_mass": branch_mass.tolist(),
                    "per_branch_normalized_first_and_second_direction_moments": moments.tolist(),
                }
            )

    refinements: list[dict[str, float]] = []
    changes_by_phase: dict[float, list[tuple[float, float]]] = {}
    for phase in phases:
        phase_changes: list[tuple[float, float]] = []
        for coarse, fine in pairwise(levels):
            coarse_branch, coarse_moments = numerical[(phase, coarse)]
            fine_branch, fine_moments = numerical[(phase, fine)]
            branch_change = _symmetric_relative_difference(coarse_branch, fine_branch)
            moment_change = float(np.max(np.abs(coarse_moments - fine_moments)))
            phase_changes.append((branch_change, moment_change))
            refinements.append(
                {
                    "azimuth_phase_rad": phase,
                    "coarse_azimuth_count": coarse,
                    "fine_azimuth_count": fine,
                    "branch_mass_change": branch_change,
                    "moment_change": moment_change,
                }
            )
        changes_by_phase[phase] = phase_changes

    left_branch, left_moments = numerical[(phases[0], levels[-1])]
    right_branch, right_moments = numerical[(phases[1], levels[-1])]
    final_phase_branch = _symmetric_relative_difference(left_branch, right_branch)
    final_phase_moment = float(np.max(np.abs(left_moments - right_moments)))
    _require(
        all(
            changes[1][0] <= 0.3 * changes[0][0]
            and changes[1][1] <= 0.3 * changes[0][1]
            and changes[1][0] <= 1.0e-5
            and changes[1][1] <= 1.0e-5
            for changes in changes_by_phase.values()
        )
        and final_phase_branch <= 1.0e-5
        and final_phase_moment <= 1.0e-5,
        "painted coating failed azimuth phase/refinement convergence",
    )
    return {
        "case_id": "painted_ewald.azimuth_phase_refinement",
        "refinement_variable": "azimuth_count",
        "levels": list(levels),
        "phases_rad": list(phases),
        "observables": rows,
        "successive_per_phase_refinements": refinements,
        "final_cross_phase_discrepancy": {
            "azimuth_count": levels[-1],
            "branch_mass": final_phase_branch,
            "low_order_moments": final_phase_moment,
        },
        "criterion": (
            "for each phase, the 16->32 change is <=0.3 of the 8->16 change and <=1e-5; "
            "the final cross-phase branch and low-order-moment discrepancies are <=1e-5"
        ),
        "assessment": "PASS",
    }


def _painted_texture_direction_moments(texture: Any) -> NDArray[np.float64]:
    mu = 0.5 * (texture.mu_edges[:-1] + texture.mu_edges[1:])
    phi = 0.5 * (texture.phi_edges_rad[:-1] + texture.phi_edges_rad[1:])
    radial = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))[:, None]
    directions = np.stack(
        (
            radial * np.cos(phi)[None, :],
            radial * np.sin(phi)[None, :],
            np.broadcast_to(mu[:, None], texture.mass.shape),
        ),
        axis=-1,
    )
    total = fsum(float(value) for value in texture.mass.ravel())
    _require(total > 0.0, "painted Bi2Se3 texture has no retained mass")
    normalized_mass = texture.mass / total
    first = np.einsum("ij,ijk->k", normalized_mass, directions, optimize=True)
    second = np.einsum(
        "ij,ijk,ijl->kl",
        normalized_mass,
        directions,
        directions,
        optimize=True,
    )
    return np.concatenate((first, second[np.triu_indices(3)]))


def _painted_bi2se3_convergence_evidence() -> dict[str, object]:
    reciprocal_basis = np.array(
        [
            [1.516578640400576, 0.0, 0.0],
            [0.8755970862825088, 1.7511941725650182, 0.0],
            [0.0, 0.0, 0.2194156064806393],
        ]
    )
    k_norm_Ainv = 4.078341560576728
    grazing_angle_rad = np.deg2rad(5.0)
    incident = np.array(
        [
            0.0,
            k_norm_Ainv * np.cos(grazing_angle_rad),
            -k_norm_Ainv * np.sin(grazing_angle_rad),
        ]
    )
    rods = enumerate_rods_within_ewald_sphere(
        reciprocal_basis_Ainv=reciprocal_basis,
        k_norm_Ainv=k_norm_Ainv,
    )
    _require(len(rods) == 85, "Bi2Se3 convergence fixture lost its complete rod catalog")
    phases = (0.371, 1.137)
    levels = (256, 512, 1024)
    rows: list[dict[str, object]] = []
    numerical: dict[
        tuple[float, int],
        tuple[NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64]],
    ] = {}
    branch_keys: tuple[tuple[int, int], ...] | None = None

    for phase in phases:
        for level in levels:
            painter = EwaldSpherePainter(
                PaintedPainterConfig(
                    reciprocal_basis_Ainv=reciprocal_basis,
                    crystal_to_sample=np.eye(3),
                    rods=rods,
                    mosaic=PaintedMosaicParameters(
                        gaussian_sigma_rad=np.deg2rad(5.0),
                        lorentzian_half_width_rad=np.deg2rad(2.0),
                        lorentzian_probability=0.1,
                        alpha_panel_count=64,
                        alpha_gauss_order=12,
                        azimuth_count=level,
                        azimuth_phase_rad=phase,
                    ),
                    raster=PaintedRasterParameters(),
                )
            )
            coating = painter.paint_coating(incident)
            keys = tuple(
                (family.family_m, branch.branch)
                for family in coating.family_summaries
                for branch in family.branches
            )
            if branch_keys is None:
                branch_keys = keys
            _require(keys == branch_keys, "Bi2Se3 convergence branch ordering changed")
            branch_weight = np.asarray(
                [
                    branch.painted_weight
                    for family in coating.family_summaries
                    for branch in family.branches
                ]
            )
            moments = _painted_texture_direction_moments(coating.texture)
            total = coating.ledger.painted_weight
            normalized_texture = coating.texture.mass / total
            numerical[(phase, level)] = (branch_weight, moments, total, normalized_texture)
            rows.append(
                {
                    "azimuth_phase_rad": phase,
                    "azimuth_count": level,
                    "orientation_count": painter.mosaic_space.orientation_id.size,
                    "retained_root_count": coating.retained_root_count,
                    "total_painted_weight": total,
                    "family_branch_keys": [list(key) for key in keys],
                    "family_branch_weights": branch_weight.tolist(),
                    "raster_center_first_and_second_direction_moments": moments.tolist(),
                    "maximum_ewald_residual_Ainv": coating.maximum_ewald_residual_Ainv,
                }
            )

    refinements: list[dict[str, float]] = []
    changes_by_phase: dict[float, list[tuple[float, float, float, float]]] = {}
    for phase in phases:
        phase_changes: list[tuple[float, float, float, float]] = []
        for coarse, fine in pairwise(levels):
            coarse_branch, coarse_moments, coarse_total, coarse_texture = numerical[(phase, coarse)]
            fine_branch, fine_moments, fine_total, fine_texture = numerical[(phase, fine)]
            changes = (
                _symmetric_relative_difference(coarse_branch, fine_branch),
                float(np.max(np.abs(coarse_moments - fine_moments))),
                _symmetric_relative_difference(
                    np.asarray([coarse_total]),
                    np.asarray([fine_total]),
                ),
                float(np.sum(np.abs(coarse_texture - fine_texture), dtype=np.float64)),
            )
            phase_changes.append(changes)
            refinements.append(
                {
                    "azimuth_phase_rad": phase,
                    "coarse_azimuth_count": coarse,
                    "fine_azimuth_count": fine,
                    "maximum_family_branch_relative_change": changes[0],
                    "moment_change": changes[1],
                    "total_weight_relative_change": changes[2],
                    "hard_bin_normalized_texture_l1_change_not_gated": changes[3],
                }
            )
        changes_by_phase[phase] = phase_changes

    left_branch, left_moments, left_total, left_texture = numerical[(phases[0], levels[-1])]
    right_branch, right_moments, right_total, right_texture = numerical[(phases[1], levels[-1])]
    final_phase = {
        "azimuth_count": levels[-1],
        "maximum_family_branch_relative_difference": _symmetric_relative_difference(
            left_branch,
            right_branch,
        ),
        "moment_difference": float(np.max(np.abs(left_moments - right_moments))),
        "total_weight_relative_difference": _symmetric_relative_difference(
            np.asarray([left_total]),
            np.asarray([right_total]),
        ),
        "hard_bin_normalized_texture_l1_difference_not_gated": float(
            np.sum(np.abs(left_texture - right_texture), dtype=np.float64)
        ),
    }
    _require(
        all(
            changes[1][0] <= 0.85 * changes[0][0]
            and changes[1][1] <= 0.85 * changes[0][1]
            and changes[1][0] <= 2.0e-3
            and changes[1][1] <= 4.0e-4
            and changes[1][2] <= 1.0e-4
            for changes in changes_by_phase.values()
        )
        and final_phase["maximum_family_branch_relative_difference"] <= 5.0e-4
        and final_phase["moment_difference"] <= 1.5e-4
        and final_phase["total_weight_relative_difference"] <= 1.0e-4,
        "full Bi2Se3 coating failed integrated azimuth convergence",
    )
    return {
        "case_id": "painted_ewald.bi2se3_5deg_full_catalog",
        "refinement_variable": "azimuth_count",
        "levels": list(levels),
        "phases_rad": list(phases),
        "rod_count": len(rods),
        "observables": rows,
        "successive_per_phase_refinements": refinements,
        "final_cross_phase_discrepancy": final_phase,
        "criterion": (
            "each 512->1024 family/branch and moment change contracts by >=15%; fine "
            "family/branch <=2e-3, moment <=4e-4, and total <=1e-4; final cross-phase "
            "family/branch <=5e-4, moment <=1.5e-4, and total <=1e-4"
        ),
        "hard_bin_texture_is_not_a_pointwise_acceptance_observable": True,
        "assessment": "PASS",
    }


def _painted_coating_evidence(tolerances: Any) -> dict[str, object]:
    incident = np.array([0.2, 4.062900581047559, -0.3545543022596421])
    config = _painted_skew_config(azimuth_count=8, azimuth_phase_rad=0.371)
    painter = EwaldSpherePainter(config, _PaintedProofStrength())

    tracemalloc.start()
    start = time.perf_counter()
    scalar = painter.paint(incident)
    scalar_wall_seconds = time.perf_counter() - start
    _, scalar_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    start = time.perf_counter()
    coating = painter.paint_coating(incident)
    streaming_wall_seconds = time.perf_counter() - start
    _, streaming_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    texture_error = float(np.max(np.abs(scalar.texture.mass - coating.texture.mass), initial=0.0))
    ledger_fields = (
        "painted_weight",
        "collapsed_direct_base_mass",
        "tangent_base_mass",
        "no_root_base_mass",
        "forward_excluded_base_mass",
    )
    ledger_error = max(
        abs(getattr(scalar.ledger, name) - getattr(coating.ledger, name)) for name in ledger_fields
    )
    branch_error = 0.0
    branch_counts_match = True
    branch_equivalence: list[dict[str, object]] = []
    for rod_summary in coating.rod_summaries:
        for branch in rod_summary.branches:
            scalar_count = sum(
                1
                for point in scalar.points
                if (point.rod_h, point.rod_k, point.branch)
                == (rod_summary.rod.h, rod_summary.rod.k, branch.branch)
            )
            scalar_weight = fsum(
                point.weight
                for point in scalar.points
                if (point.rod_h, point.rod_k, point.branch)
                == (rod_summary.rod.h, rod_summary.rod.k, branch.branch)
            )
            branch_error = max(branch_error, abs(scalar_weight - branch.painted_weight))
            branch_counts_match &= scalar_count == branch.retained_root_count
            branch_equivalence.append(
                {
                    "rod_h": rod_summary.rod.h,
                    "rod_k": rod_summary.rod.k,
                    "branch": branch.branch,
                    "scalar_retained_root_count": scalar_count,
                    "streaming_retained_root_count": branch.retained_root_count,
                }
            )
    scalar_stream_error = max(texture_error, ledger_error, branch_error)
    scalar_stream_limit = 5.0e-12
    _require(
        scalar_stream_error <= scalar_stream_limit
        and branch_counts_match
        and coating.retained_root_count == len(scalar.points)
        and coating.ledger.suppressed_algebraic_direct_root_count
        == scalar.ledger.suppressed_algebraic_direct_root_count,
        "painted scalar and streaming paths disagree",
    )

    complete_family_rods = tuple(
        PaintedRod(h, k) for h, k in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))
    )
    complete_coating = EwaldSpherePainter(
        replace(config, rods=complete_family_rods),
        _PaintedProofStrength(),
    ).paint_coating(incident)
    family = complete_coating.family_summaries[0]
    _require(
        len(complete_coating.family_summaries) == 1
        and family.family_m == 1
        and family.is_complete_hexagonal_family,
        "representative-rod mutation fixture is not one complete m=1 family",
    )
    family_sum_error = abs(
        fsum(summary.ledger.painted_weight for summary in family.rod_summaries)
        - family.ledger.painted_weight
    )
    first_rod_weight = family.rod_summaries[0].ledger.painted_weight
    representative_mutant = len(family.rod_summaries) * first_rod_weight
    representative_error = abs(representative_mutant - family.ledger.painted_weight)
    representative_limit = 1.0e-8 * max(family.ledger.painted_weight, 1.0)

    direction_slice = painter.mosaic_slice(rod=PaintedRod(0, 0), u_Ainv=1.0)
    direction_by_id = {
        int(orientation_id): direction / np.linalg.norm(direction)
        for orientation_id, direction in zip(
            painter.mosaic_space.orientation_id,
            direction_slice.q_sample_Ainv,
            strict=True,
        )
    }
    root_displacement_Ainv = 1.0e-3
    correct_root_error = max(point.ewald_residual_Ainv for point in scalar.points)
    displaced_root_error = max(
        abs(
            np.linalg.norm(
                point.kf_sample_Ainv
                + root_displacement_Ainv * direction_by_id[point.orientation_id]
            )
            - scalar.radius_Ainv
        )
        for point in scalar.points
    )
    displaced_q_error = max(
        np.linalg.norm(root_displacement_Ainv * direction_by_id[point.orientation_id])
        for point in scalar.points
    )
    displaced_root_limit = tolerances["reciprocal.ewald_residual"].bind(scalar.radius_Ainv).limit
    displaced_L_delta = root_displacement_Ainv / float(
        np.linalg.norm(config.reciprocal_basis_Ainv[:, 2])
    )

    cutoff_mosaic = PaintedMosaicParameters(
        gaussian_sigma_rad=np.deg2rad(5.0),
        lorentzian_half_width_rad=np.deg2rad(2.0),
        lorentzian_probability=0.1,
        alpha_panel_count=8,
        alpha_gauss_order=8,
        azimuth_count=16,
        azimuth_phase_rad=0.371,
    )
    cutoff_arguments = {
        "reciprocal_basis_Ainv": np.diag([1.0, 1.0, 0.2]),
        "crystal_to_sample": np.eye(3),
        "rods": (PaintedRod(0, 0),),
        "mosaic": cutoff_mosaic,
        "raster": PaintedRasterParameters(8, 16),
    }
    pushforward = EwaldSpherePainter(
        PaintedPainterConfig(
            **cutoff_arguments,
            measure=PaintMeasure.MOSAIC_PUSHFORWARD,
        )
    ).paint(incident)
    target_cutoff_Ainv = 0.7
    intended_painter = EwaldSpherePainter(
        PaintedPainterConfig(
            **cutoff_arguments,
            measure=PaintMeasure.COAREA_INTENSITY,
            forward_policy=PaintedForwardPolicy(q_min_Ainv=target_cutoff_Ainv),
        )
    )
    intended_scalar = intended_painter.paint(incident)
    intended_stream = intended_painter.paint_coating(incident)
    q_norm_and_point = tuple(
        (float(np.linalg.norm(point.q_sample_Ainv)), point) for point in pushforward.points
    )
    retained_weight = fsum(
        point.base_weight * (2.0 * pushforward.radius_Ainv / q_norm)
        for q_norm, point in q_norm_and_point
        if q_norm >= target_cutoff_Ainv
    )
    reversed_weight = fsum(
        point.base_weight * (2.0 * pushforward.radius_Ainv / q_norm)
        for q_norm, point in q_norm_and_point
        if q_norm < target_cutoff_Ainv
    )
    retained_base_weight = fsum(
        point.base_weight for q_norm, point in q_norm_and_point if q_norm >= target_cutoff_Ainv
    )
    excluded_base_weight = fsum(
        point.base_weight for q_norm, point in q_norm_and_point if q_norm < target_cutoff_Ainv
    )
    partition_error = abs(
        retained_base_weight + excluded_base_weight - pushforward.ledger.painted_weight
    )
    pushforward_total_error = abs(
        pushforward.ledger.painted_weight + pushforward.ledger.collapsed_direct_base_mass - 1.0
    )
    cutoff_error = max(
        abs(intended_scalar.ledger.painted_weight - intended_stream.ledger.painted_weight),
        abs(retained_weight - intended_stream.ledger.painted_weight),
        abs(excluded_base_weight - intended_stream.ledger.forward_excluded_base_mass),
        partition_error,
        pushforward_total_error,
    )
    reversed_cutoff_error = max(
        abs(reversed_weight - intended_stream.ledger.painted_weight),
        abs(retained_base_weight - intended_stream.ledger.forward_excluded_base_mass),
    )
    cutoff_limit = 1.0e-10 * max(intended_stream.ledger.painted_weight, 1.0)

    _require(
        family_sum_error <= representative_limit < representative_error
        and correct_root_error <= displaced_root_limit < displaced_root_error
        and cutoff_error <= cutoff_limit < reversed_cutoff_error,
        "one or more painted coating mutations escaped detection",
    )
    _require(
        retained_weight > 0.0
        and reversed_weight > 0.0
        and retained_base_weight > 0.0
        and excluded_base_weight > 0.0
        and pushforward.ledger.tangent_base_mass == 0.0
        and pushforward.ledger.no_root_base_mass == 0.0
        and streaming_peak_bytes < scalar_peak_bytes,
        "painted cutoff or bounded-memory benchmark lost sensitivity",
    )
    _require(
        streaming_wall_seconds <= 0.75 * scalar_wall_seconds,
        "streaming coating benchmark lost its wall-time margin",
    )
    mutations = [
        {
            "mutation_id": "representative_rod_replaces_exact_family_sum",
            "fixture_id": "painted_ewald.complete_m1_subtotal.v1",
            "observed_first_stage": "painted_ewald.family_weight",
            "observed_failure_metric": "absolute_family_weight_error",
            "gate_limit": representative_limit,
            "correct_error": family_sum_error,
            "correct_status": "PASS",
            "mutant_error": representative_error,
            "mutant_status": "FAIL",
            "detected": True,
        },
        {
            "mutation_id": "reversed_m0_forward_cutoff",
            "fixture_id": "painted_ewald.forward_cutoff.v1",
            "observed_first_stage": "painted_ewald.forward_policy",
            "observed_failure_metric": "absolute_cutoff_weight_error",
            "gate_limit": cutoff_limit,
            "correct_error": cutoff_error,
            "correct_status": "PASS",
            "mutant_error": reversed_cutoff_error,
            "mutant_status": "FAIL",
            "detected": True,
        },
        {
            "mutation_id": "displaced_batched_ewald_roots",
            "fixture_id": "painted_ewald.scalar_stream.v1",
            "observed_first_stage": "painted_ewald.ewald_residual",
            "observed_failure_metric": "maximum_unsquared_shell_residual_Ainv",
            "gate_limit": displaced_root_limit,
            "correct_error": correct_root_error,
            "correct_status": "PASS",
            "mutant_error": displaced_root_error,
            "mutant_status": "FAIL",
            "detected": True,
        },
    ]
    return {
        "equivalence": {
            "orientation_count": painter.mosaic_space.orientation_id.size,
            "rod_count": len(config.rods),
            "retained_root_count": coating.retained_root_count,
            "maximum_scalar_stream_error": scalar_stream_error,
            "error_limit": scalar_stream_limit,
            "per_rod_strength_at_exact_L": True,
            "configured_m_subtotal_after_rod_evaluation": True,
            "per_rod_branch_counts": branch_equivalence,
        },
        "mutations": mutations,
        "benchmark": {
            "workload": "identical two-rod skew-basis exact-strength coating",
            "orientation_count": painter.mosaic_space.orientation_id.size,
            "scalar_wall_seconds_with_memory_tracing": scalar_wall_seconds,
            "streaming_wall_seconds_with_memory_tracing": streaming_wall_seconds,
            "scalar_traced_peak_bytes": scalar_peak_bytes,
            "streaming_traced_peak_bytes": streaming_peak_bytes,
            "streaming_wall_gate_ratio": 0.75,
            "per_rod_branch_counts_match": branch_counts_match,
            "equivalent_observable_match": True,
        },
        "mutation_details": {
            "displaced_root_q_delta_Ainv": displaced_q_error,
            "displaced_root_u_delta_Ainv": root_displacement_Ainv,
            "displaced_root_L_delta": displaced_L_delta,
            "cutoff_regular_pushforward_partition_error": partition_error,
        },
    }


def run_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    """Run the compact T03 proof without writing diagnostics."""
    del allow_missing_pack
    root = Path(__file__).resolve().parents[3]
    proof_base = os.environ.get("PROOF_BASE_SHA", "")
    _require(proof_base == _PROOF_BASE_SHA, "PROOF_BASE_SHA is unset or does not match T03")
    commit_sha, changed_paths = _verified_checkout(root)
    tolerances = load_stage_tolerances()
    arrays = _load_reference_pack(root)
    classifications = _reference_evidence(arrays, tolerances)
    mutations = _scientific_evidence(tolerances)
    oracle_matrix, convergence = _oracle_evidence(tolerances)
    benchmark = _benchmark_evidence(tolerances)
    painted = _painted_coating_evidence(tolerances)
    mutations.extend(painted["mutations"])
    convergence.append(_painted_convergence_evidence())
    convergence.append(_painted_bi2se3_convergence_evidence())
    benchmark["painted_coating"] = painted["benchmark"]
    checks = [
        {
            "check_id": "mosaic_ewald_science",
            "status": "PASS",
            "evidence": (
                "shared gates, dense and scalar coating oracles, full Bi2Se3 integrated "
                "phase convergence, memory, "
                f"and {len(mutations)}/{len(mutations)} controls pass"
            ),
        },
        {
            "check_id": "worktree_clean",
            "status": "PASS" if not changed_paths else "FAIL",
            "evidence": "clean checkout"
            if not changed_paths
            else "dirty paths reported separately",
        },
    ]
    return {
        "schema_version": 1,
        "task_id": "T03",
        "status": "READY" if not changed_paths else "BLOCKED",
        "scientific_status": "PASS",
        "worktree_status": {
            "status": "READY" if not changed_paths else "BLOCKED",
            "changed_paths": list(changed_paths),
        },
        "base_sha": proof_base,
        "commit_sha": commit_sha,
        "commit_sha_scope": "HEAD_ONLY",
        "contract_version": CONTRACT_API_VERSION,
        "trace_schema_version": 4,
        "reference_pack_sha256s": {"rasim_reference_v1": _REFERENCE_PACK_SHA256},
        "environment_sha256": _environment_sha256(),
        "checks": checks,
        "metrics": {
            "oracle_matrix": oracle_matrix,
            "painted_coating_equivalence": painted["equivalence"],
        },
        "classifications": classifications,
        "convergence": convergence,
        "mutations": mutations,
        "benchmark": benchmark,
        "tolerance_artifact_sha256": STAGE_TOLERANCE_SHA256,
        "limitations": [
            "T03 preserves polarization IDs; reciprocal_weight is orientation mass times one coarea Jacobian, with all downstream factors excluded",
            "The full Bi2Se3 visualization uses the specification's unit-strength geometry mode; material structure factors enter only through the exact-L StrengthModel boundary",
            "The conservative hard-bin texture is not a pointwise convergence oracle: acceptance uses integrated family/branch totals and low-order moments, while the continuous visualization is a periodic reconstruction of equal-solid-angle cell averages",
        ],
    }
