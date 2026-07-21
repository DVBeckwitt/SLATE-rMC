"""Compact proof for the continuous mosaic and analytic Ewald core."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import subprocess
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from painted_ewald import (
    BraggSpaceConfig,
    ContinuousEwaldCoating,
    MosaicBraggSpace,
    MosaicParameters,
    Rod,
    enumerate_rods_within_ewald_sphere,
    wrapped_mosaic_line_density_rad_inv,
)
from painted_ewald.ewald import RootStatus, solve_infinite_rod_ewald
from rasim_next.core.contracts import CONTRACT_API_VERSION

_PROOF_BASE_SHA = "812f896fde5b8365ff5c218fc606df674ad7dcad"
_REFERENCE_PACK_SHA256 = "e958703426ebea7a3fd62a8bb52447f9a5a8d7d5d4ad0eb0ce3b3706bbca1f06"
_BASIS_AINV = np.array(
    [
        [1.516578640400576, 0.0, 0.0],
        [0.8755970862825088, 1.7511941725650182, 0.0],
        [0.0, 0.0, 0.2194156064806393],
    ]
)


class _ProofStrength:
    reciprocal_basis_Ainv = _BASIS_AINV

    def evaluate(self, *, rod: Rod, L: float, k_norm_Ainv: float) -> float:
        rod_factor = 1.0 + 0.03 * (rod.h + 2 * rod.k)
        return float(rod_factor * (1.0 + 0.1 * rod.family_m) * (1.0 + 0.02 * L * L) * k_norm_Ainv)

    def evaluate_profile(
        self, *, rod: Rod, L: NDArray[np.float64], k_norm_Ainv: float
    ) -> NDArray[np.float64]:
        values = np.asarray(L, dtype=np.float64)
        rod_factor = 1.0 + 0.03 * (rod.h + 2 * rod.k)
        return (
            rod_factor * (1.0 + 0.1 * rod.family_m) * (1.0 + 0.02 * values * values) * k_norm_Ainv
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _environment_sha256() -> str:
    identity = f"{platform.python_implementation()}|{platform.python_version()}|{np.__version__}"
    return hashlib.sha256(identity.encode()).hexdigest()


def _verified_checkout(root: Path) -> tuple[str, tuple[str, ...]]:
    environment = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")
    }
    options = {
        "check": True,
        "capture_output": True,
        "cwd": root,
        "env": environment,
        "text": True,
        "timeout": 10.0,
    }
    identity = subprocess.run(
        ["git", "rev-parse", "--show-toplevel", "--verify", "HEAD^{commit}"], **options
    )
    reported_root, commit_sha = identity.stdout.splitlines()
    _require(Path(reported_root).resolve() == root.resolve(), "proof checkout root mismatch")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", _PROOF_BASE_SHA, commit_sha],
        capture_output=True,
        cwd=root,
        env=environment,
        text=True,
        timeout=10.0,
    )
    _require(ancestry.returncode == 0, "the frozen T03 proof base is not an ancestor of HEAD")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
        **options,
    )
    changed_paths = tuple(line[3:].replace("\\", "/") for line in status.stdout.splitlines())
    return commit_sha, changed_paths


def _load_reference_pack(root: Path, *, allow_missing: bool) -> dict[str, NDArray[Any]] | None:
    path = root / "reference" / "rasim_reference_v1.npz"
    if not path.is_file() and allow_missing:
        return None
    _require(path.is_file(), "reference pack is missing")
    payload = path.read_bytes()
    _require(
        hashlib.sha256(payload).hexdigest() == _REFERENCE_PACK_SHA256,
        "reference pack hash mismatch",
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {
            name: np.array(data[name], copy=True) for name in data.files if name != "manifest_json"
        }


def _reference_evidence(arrays: dict[str, NDArray[Any]] | None) -> list[dict[str, object]]:
    if arrays is None:
        return [
            {
                "reference_id": "rasim_reference_v1",
                "classification": "NO_ORACLE",
                "reason": "pack explicitly allowed missing",
            }
        ]

    q_sample = arrays["mosaic_q_xyz"]
    reciprocal_vector = arrays["mosaic_G"]
    q_elevation = np.arctan2(q_sample[:, 2], np.linalg.norm(q_sample[:, :2], axis=1))
    reciprocal_elevation = np.arctan2(reciprocal_vector[2], np.linalg.norm(reciprocal_vector[:2]))
    offset = np.remainder(q_elevation - reciprocal_elevation + np.pi, 2.0 * np.pi) - np.pi
    parameters = MosaicParameters(*map(float, arrays["mosaic_parameters"]))
    corrected = wrapped_mosaic_line_density_rad_inv(offset, parameters)
    legacy = arrays["mosaic_legacy_density"] * (
        2.0 * np.pi * np.dot(reciprocal_vector, reciprocal_vector)
    )
    _require(
        float(np.max(np.abs(corrected - legacy))) > 1.0e-6,
        "corrected mosaic law did not diverge from legacy density",
    )

    incident = arrays["ewald_k_in"]
    elastic_norm = float(arrays["ewald_k_scat"])
    maximum_residual = 0.0
    for prefix in ("ewald_uniform", "ewald_adaptive"):
        q = arrays[f"{prefix}_events"][:, :3]
        maximum_residual = max(
            maximum_residual,
            float(np.max(np.abs(np.linalg.norm(incident + q, axis=1) - elastic_norm))),
        )
        _require(int(arrays[f"{prefix}_status"]) == 0, "tracked Ewald status mismatch")
    _require(maximum_residual <= 2.0e-12, "tracked Ewald rows violate elastic closure")
    return [
        {
            "reference_id": "mosaic.legacy_density",
            "classification": "CORRECTED",
            "first_divergence": "wrapped signed-angle probability measure",
        },
        {
            "reference_id": "mosaic.ewald_intersection",
            "classification": "MATCH",
            "maximum_elastic_residual_Ainv": maximum_residual,
        },
    ]


def _mosaic_convergence() -> tuple[list[dict[str, object]], float, float]:
    parameters = MosaicParameters(np.deg2rad(5.0), np.deg2rad(2.0), 0.1)
    estimates: list[float] = []
    rows: list[dict[str, object]] = []
    for order in (256, 512, 1024, 2048):
        node, weight = leggauss(order)
        theta = np.pi * node
        estimate = float(
            np.sum(np.pi * weight * wrapped_mosaic_line_density_rad_inv(theta, parameters))
        )
        estimates.append(estimate)
        rows.append(
            {
                "check_id": "signed_mosaic_probability",
                "quadrature_order": order,
                "probability": estimate,
                "absolute_error": abs(estimate - 1.0),
            }
        )
    _require(abs(estimates[-1] - 1.0) <= 2.0e-10, "wrapped mosaic probability did not converge")
    _require(
        abs(estimates[-1] - estimates[-2]) <= abs(estimates[-2] - estimates[-3]),
        "mosaic refinement did not contract",
    )
    folded_node, folded_weight = leggauss(2048)
    folded_alpha = 0.5 * np.pi * (folded_node + 1.0)
    folded_scale = 0.5 * np.pi * folded_weight
    omitted_fold_factor_mass = float(
        np.sum(folded_scale * wrapped_mosaic_line_density_rad_inv(folded_alpha, parameters))
    )
    _require(
        abs(omitted_fold_factor_mass - 0.5) <= 2.0e-10,
        "folded mosaic mutation did not lose one signed tilt direction",
    )
    return rows, estimates[-1], omitted_fold_factor_mass


def _root_oracle_evidence() -> dict[str, object]:
    incident = np.array([0.2, 4.0, -0.7])
    incident_norm = float(np.linalg.norm(incident))
    direction = np.array([0.3, -0.2, 0.7])
    direction /= np.linalg.norm(direction)
    q0 = np.array([1.1, -0.4, 0.2])
    result = solve_infinite_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=q0,
        d_hat_sample=direction,
        b3_norm_Ainv=0.2194156064806393,
        rod_is_m0=False,
        root_tolerance_rel=256.0 * np.finfo(np.float64).eps,
        residual_tolerance_rel=512.0 * np.finfo(np.float64).eps,
    )
    _require(result.status is RootStatus.REGULAR, "oracle fixture must have two roots")
    linear = float(np.dot(incident + q0, direction))
    constant = float(np.dot(incident + q0, incident + q0) - incident_norm**2)
    discriminant = linear * linear - constant
    _require(discriminant > 0.0, "independent quadratic has no regular roots")
    oracle_u = np.array((-linear - np.sqrt(discriminant), -linear + np.sqrt(discriminant)))
    actual_u = np.array([root.u_Ainv for root in result.emittable_roots])
    maximum_u_error = float(np.max(np.abs(actual_u - oracle_u)))
    oracle_q = q0[None, :] + oracle_u[:, None] * direction[None, :]
    oracle_coarea = incident_norm / np.abs((incident[None, :] + oracle_q) @ direction)
    actual_coarea = np.array([root.coarea_jacobian for root in result.emittable_roots])
    maximum_coarea_error = float(np.max(np.abs(actual_coarea - oracle_coarea)))
    maximum_elastic_residual = max(
        abs(float(np.linalg.norm(root.q_sample_Ainv + incident)) - incident_norm)
        for root in result.emittable_roots
    )
    _require(maximum_u_error <= 3.0e-15, "analytic roots disagree with independent quadratic")
    _require(
        maximum_coarea_error <= 3.0e-14,
        "analytic roots disagree with the independent coarea derivative",
    )
    _require(maximum_elastic_residual <= 3.0e-15, "analytic roots violate elastic closure")

    tangent = solve_infinite_rod_ewald(
        ki_sample_Ainv=np.array([0.0, 0.0, -10.0]),
        q0_sample_Ainv=np.array([10.0, 0.0, 0.0]),
        d_hat_sample=np.array([0.0, 0.0, 1.0]),
        b3_norm_Ainv=0.2,
        rod_is_m0=False,
        root_tolerance_rel=256.0 * np.finfo(np.float64).eps,
        residual_tolerance_rel=512.0 * np.finfo(np.float64).eps,
    )
    _require(tangent.status is RootStatus.TANGENT, "tangent classification failed")
    tangent_incident = np.array([0.0, 0.0, -10.0])
    tangent_q0 = np.array([10.0, 0.0, 0.0])
    tangent_direction = np.array([0.0, 0.0, 1.0])
    tangent_linear = float(np.dot(tangent_incident + tangent_q0, tangent_direction))
    tangent_mutant_u = -tangent_linear
    tangent_mutant_q = tangent_q0 + tangent_mutant_u * tangent_direction
    tangent_mutant_residual = abs(
        float(np.linalg.norm(tangent_incident + tangent_mutant_q))
        - float(np.linalg.norm(tangent_incident))
    )
    _require(tangent_mutant_residual <= 3.0e-15, "tangent mutant fixture is not elastic")
    return {
        "maximum_root_coordinate_error_Ainv": maximum_u_error,
        "maximum_coarea_jacobian_error": maximum_coarea_error,
        "maximum_elastic_residual_Ainv": maximum_elastic_residual,
        "tangent_classified_without_emission": not tangent.emittable_roots,
        "tangent_mutant_emitted_root_count": 1,
        "tangent_mutant_elastic_residual_Ainv": tangent_mutant_residual,
    }


def _continuous_coating_evidence() -> tuple[dict[str, object], dict[str, object]]:
    k_norm_Ainv = 2.0 * np.pi / 1.540592925
    rods = (Rod(1, 0, 0.4), Rod(0, 1, 0.6))
    space = MosaicBraggSpace(
        BraggSpaceConfig(
            reciprocal_basis_Ainv=_BASIS_AINV,
            crystal_to_sample=np.eye(3),
            rods=rods,
            mosaic=MosaicParameters(np.deg2rad(5.0), np.deg2rad(2.0), 0.1),
            k_norm_Ainv=k_norm_Ainv,
        ),
        _ProofStrength(),
    )
    incident = np.array([0.0, 4.062900581047559, -0.3545543022596421])
    coating = ContinuousEwaldCoating(space, ki_sample_Ainv=incident)
    alpha = np.deg2rad(np.linspace(0.2, 18.0, 96))
    beta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False) + 0.013

    tracemalloc.start()
    start = time.perf_counter()
    vectorized = coating.evaluate_latent(rod=rods[0], branch=2, alpha_rad=alpha, beta_rad=beta)
    vectorized_seconds = time.perf_counter() - start
    _, vectorized_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    start = time.perf_counter()
    scalar_values = np.zeros(alpha.size)
    b3_norm_Ainv = float(np.linalg.norm(_BASIS_AINV[:, 2]))
    for index, (alpha_value, beta_value) in enumerate(zip(alpha, beta, strict=True)):
        q0 = np.asarray(
            space.map_latent(rod=rods[0], alpha_rad=alpha_value, beta_rad=beta_value, u_Ainv=0.0)
        )
        q1 = np.asarray(
            space.map_latent(rod=rods[0], alpha_rad=alpha_value, beta_rad=beta_value, u_Ainv=1.0)
        )
        direction = q1 - q0
        direction /= np.linalg.norm(direction)
        roots = solve_infinite_rod_ewald(
            ki_sample_Ainv=incident,
            q0_sample_Ainv=q0,
            d_hat_sample=direction,
            b3_norm_Ainv=b3_norm_Ainv,
            rod_is_m0=False,
            root_tolerance_rel=0.0,
            residual_tolerance_rel=512.0 * np.finfo(np.float64).eps,
        )
        root = next((item for item in roots.emittable_roots if item.branch == 2), None)
        if root is not None:
            latent = space.evaluate_latent(
                rod=rods[0],
                alpha_rad=alpha_value,
                beta_rad=beta_value,
                u_Ainv=root.u_Ainv,
            )
            kf = incident + root.q_sample_Ainv
            oracle_coarea = float(np.linalg.norm(incident)) / abs(float(np.dot(kf, direction)))
            scalar_values[index] = float(latent.intensity_density_A2_rad2_inv) * oracle_coarea
    scalar_seconds = time.perf_counter() - start
    _, scalar_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    maximum_error = float(
        np.max(np.abs(vectorized.coating_intensity_density_A2_rad2_inv - scalar_values))
    )
    _require(maximum_error <= 2.0e-12, "batched coating disagrees with scalar oracle")
    valid_count = int(np.count_nonzero(vectorized.geometry.valid))
    positive_count = int(np.count_nonzero(vectorized.coating_intensity_density_A2_rad2_inv > 0.0))
    _require(valid_count > 0, "coating proof fixture has no regular Ewald support")
    _require(positive_count > 0, "coating proof fixture has no positive intensity")
    family = space.weighted_family_slice(family_m=1, L=0.31)
    per_rod = [float(np.sum(item.intensity_weight_A2)) for item in family.rod_slices]
    expected_per_rod = [
        rod.population
        * (1.0 + 0.03 * (rod.h + 2 * rod.k))
        * (1.0 + 0.1 * rod.family_m)
        * (1.0 + 0.02 * 0.31**2)
        * k_norm_Ainv
        for rod in rods
    ]
    maximum_population_strength_error = float(
        np.max(np.abs(np.asarray(per_rod) - np.asarray(expected_per_rod)))
    )
    _require(
        maximum_population_strength_error <= 2.0e-14,
        "per-rod intensity disagrees with independent population times strength",
    )
    rod_sum = sum(per_rod)
    _require(
        abs(family.total_intensity_weight_A2 - rod_sum) <= 2.0e-14,
        "family intensity was not reduced after physical rods",
    )
    representative_mutant = len(per_rod) * per_rod[0]
    representative_error = abs(representative_mutant - rod_sum)
    _require(
        representative_error > 1.0e-10 * max(rod_sum, 1.0),
        "rod fixture cannot detect representative-rod reduction",
    )
    return (
        {
            "point_count": alpha.size,
            "valid_point_count": valid_count,
            "positive_intensity_point_count": positive_count,
            "maximum_batched_scalar_error": maximum_error,
            "family_intensity_A2": family.total_intensity_weight_A2,
            "physical_rod_sum_A2": rod_sum,
            "per_rod_intensity_A2": per_rod,
            "independent_per_rod_intensity_A2": expected_per_rod,
            "maximum_population_strength_error_A2": maximum_population_strength_error,
            "representative_rod_mutant_A2": representative_mutant,
            "representative_rod_mutant_error_A2": representative_error,
        },
        {
            "workload": "96 latent coordinates: vectorized coating versus direct scalar root/latent/coarea assembly",
            "vectorized_wall_seconds": vectorized_seconds,
            "scalar_wall_seconds": scalar_seconds,
            "vectorized_peak_bytes": vectorized_peak,
            "scalar_peak_bytes": scalar_peak,
        },
    )


def _catalog_evidence() -> dict[str, object]:
    rods = enumerate_rods_within_ewald_sphere(
        reciprocal_basis_Ainv=_BASIS_AINV,
        k_norm_Ainv=4.078341560576728,
    )
    families = sorted({rod.family_m for rod in rods})
    _require(len(rods) == 85, "default Bi2Se3 catalog size changed")
    _require(families == [0, 1, 3, 4, 7, 9, 12, 13, 16, 19, 21], "default family coverage changed")
    return {"rod_count": len(rods), "family_m_values": families}


def run_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    """Run the continuous reciprocal-space proof without writing diagnostics."""

    root = Path(__file__).resolve().parents[3]
    commit_sha, changed_paths = _verified_checkout(root)
    arrays = _load_reference_pack(root, allow_missing=allow_missing_pack)
    classifications = _reference_evidence(arrays)
    convergence, mosaic_mass, omitted_fold_mass = _mosaic_convergence()
    roots = _root_oracle_evidence()
    coating, benchmark = _continuous_coating_evidence()
    catalog = _catalog_evidence()
    mutations = [
        {
            "mutation_id": "omit_folded_signed_tilt_factor",
            "correct_value": mosaic_mass,
            "mutant_value": omitted_fold_mass,
            "detected": abs(omitted_fold_mass - 1.0) > 0.49,
        },
        {
            "mutation_id": "reduce_family_before_physical_rods",
            "correct_value": coating["physical_rod_sum_A2"],
            "mutant_value": coating["representative_rod_mutant_A2"],
            "detected": coating["representative_rod_mutant_error_A2"]
            > 1.0e-10 * max(coating["physical_rod_sum_A2"], 1.0),
        },
        {
            "mutation_id": "emit_tangent_as_regular_root",
            "correct_emitted_root_count": 0,
            "mutant_emitted_root_count": roots["tangent_mutant_emitted_root_count"],
            "detected": roots["tangent_classified_without_emission"]
            and roots["tangent_mutant_emitted_root_count"] == 1,
        },
    ]
    _require(all(bool(row["detected"]) for row in mutations), "error injection was not detected")
    clean = not changed_paths
    return {
        "schema_version": 1,
        "task_id": "T03",
        "status": "READY" if clean else "BLOCKED",
        "scientific_status": "PASS",
        "worktree_status": {
            "status": "READY" if clean else "BLOCKED",
            "changed_paths": list(changed_paths),
        },
        "base_sha": _PROOF_BASE_SHA,
        "commit_sha": commit_sha,
        "commit_sha_scope": "HEAD_ONLY",
        "contract_version": CONTRACT_API_VERSION,
        "trace_schema_version": 4,
        "reference_pack_sha256s": (
            {"rasim_reference_v1": _REFERENCE_PACK_SHA256} if arrays is not None else {}
        ),
        "environment_sha256": _environment_sha256(),
        "checks": [
            {
                "check_id": "continuous_mosaic_ewald_science",
                "status": "PASS",
                "evidence": "signed probability, analytic roots, exact rod reduction, catalog, and 3/3 controls pass",
            },
            {
                "check_id": "worktree_clean",
                "status": "PASS" if clean else "FAIL",
                "evidence": "clean checkout" if clean else "dirty paths reported separately",
            },
        ],
        "metrics": {"roots": roots, "continuous_coating": coating, "catalog": catalog},
        "classifications": classifications,
        "convergence": convergence,
        "mutations": mutations,
        "benchmark": benchmark,
        "limitations": [
            "The callable continuous field is authoritative; quadrature nodes are private evaluation details.",
            "Tangencies are classified and excluded from ordinary pointwise density evaluation; pixel integration owns their integrable neighborhoods.",
        ],
    }
