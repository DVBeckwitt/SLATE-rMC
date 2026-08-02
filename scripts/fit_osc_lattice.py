"""Regularized direct-basis sensitivity fit conditioned on a completed OSC position fit."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rasim_next.fitting import (
    LATTICE_FIT_SCHEMA_VERSION,
    LATTICE_MAXIMUM_ABSOLUTE_LOG_STRAIN,
    LATTICE_MAXIMUM_ABSOLUTE_PRIOR_PULL,
    LATTICE_MAXIMUM_DATA_CONDITION,
    LATTICE_MAXIMUM_PRIOR_SIGMA_LOG_STRAIN,
    LATTICE_MINIMUM_DATA_IMPROVEMENT_FRACTION,
    LATTICE_PARAMETER_NAMES,
    LATTICE_REQUIRED_DATA_PRACTICAL_RANK,
    LATTICE_SENSITIVITY_RELATIVE_TOLERANCE,
    IndexedGeometryImage,
    audit_indexed_geometry_series_roots,
    evaluate_indexed_geometry_series_metrics,
    evaluate_indexed_geometry_series_residual,
    fixed_position_from_fit_record,
    hexagonal_direct_basis,
)
from rasim_next.fitting.geometry import ExactTagGeometryModel
from rasim_next.pipeline.configured_simulation import (
    rebind_configured_geometry_direct_basis,
    rebind_configured_geometry_instrument,
)
from rasim_next.selection import index_osc_geometry_series, load_osc_geometry_series

SCHEMA = LATTICE_FIT_SCHEMA_VERSION
SENSITIVITY_RELATIVE_TOLERANCE = LATTICE_SENSITIVITY_RELATIVE_TOLERANCE
MAXIMUM_DATA_CONDITION = LATTICE_MAXIMUM_DATA_CONDITION


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _external_file(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or resolved.is_relative_to(ROOT):
        raise ValueError("lattice artifact must be written outside the repository")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _scaled_sensitivity_diagnostics(
    jacobian: np.ndarray,
    parameter_scales: np.ndarray,
    *,
    relative_tolerance: float = SENSITIVITY_RELATIVE_TOLERANCE,
) -> tuple[np.ndarray, int, int, float]:
    """Diagnose data sensitivity in dimensionless parameter-scale coordinates."""
    matrix = np.asarray(jacobian, dtype=np.float64)
    scales = np.asarray(parameter_scales, dtype=np.float64)
    if matrix.ndim != 2 or scales.shape != (matrix.shape[1],):
        raise ValueError("jacobian and parameter scales are incompatible")
    if (
        matrix.size == 0
        or np.any(~np.isfinite(matrix))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError("sensitivity inputs must be finite and nonempty")
    tolerance = float(relative_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative sensitivity tolerance must be finite and positive")

    singular_values = np.linalg.svd(matrix * scales[None, :], compute_uv=False)
    maximum = float(singular_values[0])
    numerical_cutoff = 64.0 * np.finfo(np.float64).eps * max(matrix.shape) * maximum
    practical_cutoff = max(numerical_cutoff, tolerance * maximum)
    numerical_rank = int(np.count_nonzero(singular_values > numerical_cutoff))
    practical_rank = int(np.count_nonzero(singular_values >= practical_cutoff))
    condition = (
        maximum / float(singular_values[-1])
        if numerical_rank == matrix.shape[1] and singular_values[-1] > 0.0
        else math.inf
    )
    return singular_values, practical_rank, numerical_rank, condition


def _lattice_fit_is_accepted(
    *,
    optimizer_success: bool,
    selection_rebased: bool,
    root_classification: str,
    active_bounds: np.ndarray,
    prior_pull: np.ndarray,
    improvement_fraction: float,
    data_rank: int,
    data_condition: float,
    maximum_data_condition: float = MAXIMUM_DATA_CONDITION,
) -> bool:
    """Apply the data-only promotion gate for the two lattice parameters."""
    bounds = np.asarray(active_bounds, dtype=np.bool_)
    pulls = np.asarray(prior_pull, dtype=np.float64)
    return bool(
        optimizer_success
        and not selection_rebased
        and root_classification == "SAME"
        and bounds.shape == (2,)
        and not np.any(bounds)
        and pulls.shape == (2,)
        and np.all(np.isfinite(pulls))
        and np.max(np.abs(pulls)) <= LATTICE_MAXIMUM_ABSOLUTE_PRIOR_PULL
        and math.isfinite(improvement_fraction)
        and improvement_fraction >= LATTICE_MINIMUM_DATA_IMPROVEMENT_FRACTION
        and data_rank == LATTICE_REQUIRED_DATA_PRACTICAL_RANK
        and math.isfinite(data_condition)
        and data_condition <= maximum_data_condition
    )


def _lattice_bound_images(
    images: tuple[IndexedGeometryImage, ...],
    log_strain: np.ndarray,
) -> tuple[IndexedGeometryImage, ...]:
    if not images:
        raise ValueError("indexed geometry images are required")
    reference_A = images[0].model.inputs.crystal.direct_basis_A
    direct_basis_A = hexagonal_direct_basis(reference_A, log_strain)
    shared = rebind_configured_geometry_direct_basis(images[0].model.inputs, direct_basis_A)
    return tuple(
        IndexedGeometryImage(
            image_id=image.image_id,
            commanded_angle_rad=image.commanded_angle_rad,
            model=ExactTagGeometryModel(
                rebind_configured_geometry_instrument(shared, image.model.inputs.config)
            ),
            observations=image.observations,
        )
        for image in images
    )


def fit_lattice_sensitivity(
    *,
    manifest_path: Path,
    position_path: Path,
    destination: Path,
    prior_sigma: tuple[float, float] = (5.0e-4, 5.0e-4),
    bound_half_span: float = 2.0e-3,
    allow_selection_rebase: bool = False,
) -> dict[str, Any]:
    destination = _external_file(destination)
    if destination.exists():
        raise FileExistsError(destination)
    sigma = np.asarray(prior_sigma, dtype=np.float64)
    bound = float(bound_half_span)
    if sigma.shape != (2,) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("prior_sigma must contain two finite positive values")
    if np.any(sigma > LATTICE_MAXIMUM_PRIOR_SIGMA_LOG_STRAIN):
        raise ValueError("prior_sigma exceeds the near-CIF adoption limit")
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("bound_half_span must be finite and positive")
    if bound > LATTICE_MAXIMUM_ABSOLUTE_LOG_STRAIN:
        raise ValueError("bound_half_span exceeds the near-CIF adoption limit")
    resolved_manifest = manifest_path.resolve()
    position_document = json.loads(position_path.resolve().read_text(encoding="utf-8"))
    position, _, declared_selection = fixed_position_from_fit_record(
        position_document,
        expected_manifest_path=resolved_manifest,
        expected_manifest_sha256=_sha256(resolved_manifest),
    )
    corrections = position.corrections
    incidence_delta = position.incidence_angle_delta_rad
    incidence_trims = (
        dict(
            zip(
                position.incidence_angle_image_ids,
                position.incidence_angle_trim_rad,
                strict=True,
            )
        )
        if position.incidence_angle_image_ids
        else None
    )
    series = load_osc_geometry_series(manifest_path)
    indexing = index_osc_geometry_series(series)
    if indexing.indexed_images is None:
        raise RuntimeError("position indexing did not retain exact-tag observations")
    images = tuple(indexing.indexed_images)
    selection_rebased = declared_selection != indexing.selection.manifest_hash
    if selection_rebased and not allow_selection_rebase:
        raise ValueError(
            "position artifact and rediscovered marker selection differ: "
            f"{declared_selection!r} != {indexing.selection.manifest_hash!r}"
        )

    def data_residual(log_strain: np.ndarray) -> np.ndarray:
        return evaluate_indexed_geometry_series_residual(
            _lattice_bound_images(images, log_strain),
            corrections,
            incidence_angle_delta_rad=incidence_delta,
            incidence_angle_trim_by_image_id_rad=incidence_trims,
        )

    def posterior_residual(log_strain: np.ndarray) -> np.ndarray:
        return np.concatenate((data_residual(log_strain), log_strain / sigma))

    zero = np.zeros(2, dtype=np.float64)
    baseline_residual = data_residual(zero)
    optimized = least_squares(
        posterior_residual,
        zero,
        bounds=(-bound, bound),
        x_scale=sigma,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=80,
    )
    fitted_strain = np.asarray(optimized.x, dtype=np.float64)
    fitted_images = _lattice_bound_images(images, fitted_strain)
    fitted_residual = evaluate_indexed_geometry_series_residual(
        fitted_images,
        corrections,
        incidence_angle_delta_rad=incidence_delta,
        incidence_angle_trim_by_image_id_rad=incidence_trims,
    )
    baseline_sum = float(baseline_residual @ baseline_residual)
    fitted_sum = float(fitted_residual @ fitted_residual)
    improvement_fraction = (baseline_sum - fitted_sum) / baseline_sum
    root_audit = audit_indexed_geometry_series_roots(
        fitted_images,
        corrections,
        incidence_angle_delta_rad=incidence_delta,
        incidence_angle_trim_by_image_id_rad=incidence_trims,
    )
    prior_pull = fitted_strain / sigma
    active_bound = np.abs(fitted_strain) >= bound - 1.0e-8 * bound
    data_sensitivity = _scaled_sensitivity_diagnostics(
        optimized.jac[: baseline_residual.size],
        sigma,
    )
    posterior_sensitivity = _scaled_sensitivity_diagnostics(optimized.jac, sigma)
    accepted = _lattice_fit_is_accepted(
        optimizer_success=bool(optimized.success),
        selection_rebased=selection_rebased,
        root_classification=root_audit.classification,
        active_bounds=active_bound,
        prior_pull=prior_pull,
        improvement_fraction=improvement_fraction,
        data_rank=data_sensitivity[1],
        data_condition=data_sensitivity[3],
    )
    reference_A = images[0].model.inputs.crystal.direct_basis_A
    fitted_A = hexagonal_direct_basis(reference_A, fitted_strain)
    accepted_A = fitted_A if accepted else reference_A
    reference_a = float(np.linalg.norm(reference_A[:, 0]))
    reference_c = float(np.linalg.norm(reference_A[:, 2]))
    fitted_a = float(np.linalg.norm(fitted_A[:, 0]))
    fitted_c = float(np.linalg.norm(fitted_A[:, 2]))
    metrics = evaluate_indexed_geometry_series_metrics(
        fitted_images,
        corrections,
        incidence_angle_delta_rad=incidence_delta,
        incidence_angle_trim_by_image_id_rad=incidence_trims,
    )
    document = {
        "schema_version": SCHEMA,
        "status": "ACCEPT_FITTED_LATTICE" if accepted else "RETAIN_CIF_LATTICE",
        "accepted": accepted,
        "selection_identity_rebased": selection_rebased,
        "parameter_names": list(LATTICE_PARAMETER_NAMES),
        "reference": {
            "a_A": reference_a,
            "c_A": reference_c,
            "direct_basis_A": reference_A.tolist(),
        },
        "sensitivity_fit": {
            "log_strain": fitted_strain.tolist(),
            "a_A": fitted_a,
            "c_A": fitted_c,
            "direct_basis_A": fitted_A.tolist(),
            "prior_pull": prior_pull.tolist(),
            "data_sum_squares": fitted_sum,
            "baseline_data_sum_squares": baseline_sum,
            "data_improvement_fraction": improvement_fraction,
            "posterior_half_sum_squares": float(optimized.cost),
            "optimizer_success": bool(optimized.success),
            "optimizer_message": str(optimized.message),
            "function_evaluations": int(optimized.nfev),
            "active_bounds": active_bound.tolist(),
            "site_rms_px": metrics.site_rms_px,
            "site_max_px": metrics.site_max_px,
            "data_sensitivity": {
                "coordinate_system": "dimensionless parameter-scale coordinates",
                "parameter_scales": sigma.tolist(),
                "singular_values": data_sensitivity[0].tolist(),
                "practical_rank": data_sensitivity[1],
                "numerical_rank": data_sensitivity[2],
                "condition": data_sensitivity[3],
                "relative_tolerance": SENSITIVITY_RELATIVE_TOLERANCE,
                "prior_rows_included": False,
            },
            "posterior_sensitivity": {
                "coordinate_system": "dimensionless parameter-scale coordinates",
                "parameter_scales": sigma.tolist(),
                "singular_values": posterior_sensitivity[0].tolist(),
                "practical_rank": posterior_sensitivity[1],
                "numerical_rank": posterior_sensitivity[2],
                "condition": posterior_sensitivity[3],
                "relative_tolerance": SENSITIVITY_RELATIVE_TOLERANCE,
                "prior_rows_included": True,
            },
        },
        "accepted_state": {
            "a_A": float(np.linalg.norm(accepted_A[:, 0])),
            "c_A": float(np.linalg.norm(accepted_A[:, 2])),
            "direct_basis_A": accepted_A.tolist(),
        },
        "regularization": {
            "mean_log_strain": [0.0, 0.0],
            "sigma_log_strain": sigma.tolist(),
            "hard_bound_half_span": bound,
            "interpretation": "algorithmic near-CIF trust prior; not a crystallographic uncertainty",
        },
        "acceptance": {
            "minimum_data_improvement_fraction": LATTICE_MINIMUM_DATA_IMPROVEMENT_FRACTION,
            "maximum_absolute_prior_pull": LATTICE_MAXIMUM_ABSOLUTE_PRIOR_PULL,
            "required_data_practical_rank": LATTICE_REQUIRED_DATA_PRACTICAL_RANK,
            "maximum_data_condition": MAXIMUM_DATA_CONDITION,
            "sensitivity_relative_tolerance": SENSITIVITY_RELATIVE_TOLERANCE,
            "root_audit": root_audit.classification,
        },
        "model_pixelized": False,
        "intensity_evaluated": False,
        "provenance": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path.resolve()),
            "position": str(position_path.resolve()),
            "position_sha256": _sha256(position_path.resolve()),
            "selection_revision": indexing.selection.manifest_hash,
            "predecessor_selection_revision": declared_selection,
            "position_predecessor_qualified": True,
            "adapter_sha256": _sha256(Path(__file__)),
        },
    }
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--position", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--prior-sigma", type=float, nargs=2, default=(5.0e-4, 5.0e-4))
    parser.add_argument("--bound-half-span", type=float, default=2.0e-3)
    parser.add_argument("--allow-selection-rebase", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    document = fit_lattice_sensitivity(
        manifest_path=arguments.manifest,
        position_path=arguments.position,
        destination=arguments.destination,
        prior_sigma=tuple(arguments.prior_sigma),
        bound_half_span=arguments.bound_half_span,
        allow_selection_rebase=arguments.allow_selection_rebase,
    )
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
