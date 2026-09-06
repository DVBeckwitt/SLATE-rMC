"""Seeded Gaussian source-ray quadrature with explicit probability masses."""

from __future__ import annotations

import json
import math
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

from rasim_next.core.contracts import IncidentSampleBatch

_DIMENSION_COUNT = 5
SOURCE_QUADRATURE_MODEL_ID = "weighted_correlated_gaussian_source_quadrature.v2"
_SAMPLING_MODEL_ID = "independent_gaussian_antithetic_lhs.v2"
_CORRELATED_SAMPLING_MODEL_ID = "moment_matched_gaussian_antithetic_lhs.v2"
_FINITE_CORRELATED_SAMPLING_MODEL_ID = "gaussian_antithetic_lhs_correlated_transform.v1"
_RNG_MODEL_ID = "numpy_pcg64.v1"
_NO_RNG_MODEL_ID = "no_rng.v1"
_DISCRETE_LINE_SAMPLING_MODEL_ID = "independent_gaussian_spatial_angular_stratified_lines.v1"
_CORRELATED_DISCRETE_LINE_SAMPLING_MODEL_ID = (
    "moment_matched_gaussian_spatial_angular_stratified_lines.v2"
)
_FINITE_CORRELATED_DISCRETE_LINE_SAMPLING_MODEL_ID = (
    "gaussian_spatial_angular_stratified_lines_correlated_transform.v1"
)
_FINITE_UNEQUAL_DISCRETE_LINE_SAMPLING_MODEL_ID = "finite_unequal_line_geometry_quadrature.v1"
_FINITE_UNEQUAL_CORRELATED_DISCRETE_LINE_SAMPLING_MODEL_ID = (
    "finite_unequal_line_geometry_correlated_transform.v1"
)
_FINITE_UNEQUAL_MATCHED_DISCRETE_LINE_SAMPLING_MODEL_ID = (
    "finite_unequal_line_geometry_moment_matched.v1"
)
NOMINAL_MEAN_GEOMETRY_REFERENCE_MODEL_ID = "nominal_mean_geometry_reference.v1"


def require_physical_intensity_source_model(source_sampling_model_id: str) -> None:
    """Reject the geometry-only centroid companion at intensity boundaries."""

    if source_sampling_model_id == NOMINAL_MEAN_GEOMETRY_REFERENCE_MODEL_ID:
        raise ValueError("nominal mean geometry reference cannot contribute to intensity")


def _contains_only_real_numbers(supplied: np.ndarray) -> bool:
    return supplied.dtype.kind in "iuf" or (
        supplied.dtype.kind == "O"
        and all(
            isinstance(item, Real) and not isinstance(item, (bool, np.bool_))
            for item in supplied.flat
        )
    )


def _finite_real(value: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    supplied = np.asarray(value, dtype=object)
    if not _contains_only_real_numbers(supplied):
        raise ValueError(f"{name} must be real")
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite real array with shape {shape}")
    return array


def _antithetic_lhs(*, size: int, dimension_count: int, seed: int) -> np.ndarray:
    unit = np.empty((size, dimension_count), dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(seed))
    pair_count = size // 2
    if pair_count:
        for dimension in range(dimension_count):
            lower_strata = generator.permutation(pair_count)
            offsets = generator.random(pair_count)
            lower = (lower_strata + offsets) / size
            lower_edges = lower_strata / size
            upper_edges = (lower_strata + 1) / size
            paired_upper_strata = size - 1 - lower_strata
            paired_lower_edges = paired_upper_strata / size
            paired_upper_edges = (paired_upper_strata + 1) / size
            strict_lower = np.maximum(
                np.nextafter(lower_edges, upper_edges),
                np.nextafter(1.0 - paired_upper_edges, np.inf),
            )
            strict_upper = np.minimum(
                np.nextafter(upper_edges, lower_edges),
                np.nextafter(1.0 - paired_lower_edges, -np.inf),
            )
            lower = np.clip(lower, strict_lower, strict_upper)
            upper = 1.0 - lower
            if not (
                np.all((lower > lower_edges) & (lower < upper_edges))
                and np.all((upper > paired_lower_edges) & (upper < paired_upper_edges))
            ):
                raise RuntimeError("failed to construct endpoint-safe antithetic LHS coordinates")
            lower_on_first_row = generator.integers(0, 2, pair_count, dtype=np.int8).astype(bool)
            unit[: 2 * pair_count : 2, dimension] = np.where(lower_on_first_row, lower, upper)
            unit[1 : 2 * pair_count : 2, dimension] = np.where(lower_on_first_row, upper, lower)
    if size % 2:
        unit[-1] = 0.5
    return unit


def _sample_antithetic_geometry(
    *,
    size: int,
    seed: int,
    mean_origin: np.ndarray,
    mean_direction: np.ndarray,
    axes: np.ndarray,
    spatial_sigma: np.ndarray,
    divergence_sigma: np.ndarray,
    position_divergence_correlation: np.ndarray,
    source_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    attempt_count = 32 if np.any(position_divergence_correlation != 0.0) and size >= 8 else 1
    failure: RuntimeError | None = None
    for attempt in range(attempt_count):
        realization_seed = (seed + attempt) % 2**64
        gaussian = ndtri(
            _antithetic_lhs(
                size=size,
                dimension_count=_DIMENSION_COUNT,
                seed=realization_seed,
            )
        )
        try:
            origin, direction, moment_matched = _sample_origin_and_direction(
                gaussian[:, :4],
                mean_origin=mean_origin,
                mean_direction=mean_direction,
                axes=axes,
                spatial_sigma=spatial_sigma,
                divergence_sigma=divergence_sigma,
                position_divergence_correlation=position_divergence_correlation,
                source_weight=source_weight,
            )
        except RuntimeError as error:
            failure = error
            continue
        return gaussian, origin, direction, moment_matched
    raise RuntimeError(
        "source quadrature could not realize the declared covariance after deterministic retries"
    ) from failure


def _validated_source_geometry(
    *,
    mean_origin_lab_m: ArrayLike,
    mean_direction_lab: ArrayLike,
    transverse_axes_lab: ArrayLike,
    spatial_sigma_m: ArrayLike,
    divergence_sigma_rad: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_origin = _finite_real(mean_origin_lab_m, (3,), "mean_origin_lab_m")
    mean_direction = _finite_real(mean_direction_lab, (3,), "mean_direction_lab")
    axes = _finite_real(transverse_axes_lab, (2, 3), "transverse_axes_lab")
    spatial_sigma = _finite_real(spatial_sigma_m, (2,), "spatial_sigma_m")
    divergence_sigma = _finite_real(divergence_sigma_rad, (2,), "divergence_sigma_rad")
    if np.any(spatial_sigma < 0.0) or np.any(divergence_sigma < 0.0):
        raise ValueError("source standard deviations must be nonnegative")
    if not np.isclose(np.linalg.norm(mean_direction), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("mean_direction_lab must be unit length")
    if not np.allclose(axes @ axes.T, np.eye(2), rtol=0.0, atol=1.0e-12) or not np.allclose(
        axes @ mean_direction, 0.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("transverse_axes_lab must be orthonormal and tangent")
    return mean_origin, mean_direction, axes, spatial_sigma, divergence_sigma


def _sample_origin_and_direction(
    gaussian: np.ndarray,
    *,
    mean_origin: np.ndarray,
    mean_direction: np.ndarray,
    axes: np.ndarray,
    spatial_sigma: np.ndarray,
    divergence_sigma: np.ndarray,
    position_divergence_correlation: np.ndarray,
    source_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if gaussian.ndim != 2 or gaussian.shape[1] != 4:
        raise ValueError("gaussian source geometry must have shape (N, 4)")
    has_correlation = bool(np.any(position_divergence_correlation != 0.0))
    moment_matched = False
    if has_correlation and gaussian.shape[0] >= 8:
        supplied_weight = np.asarray(source_weight, dtype=np.float64)
        weight = supplied_weight / np.sum(supplied_weight, dtype=np.float64)
        centered = gaussian - weight @ gaussian
        weighted_centered = np.sqrt(weight)[:, None] * centered
        _, raw_upper = np.linalg.qr(weighted_centered, mode="reduced")
        diagonal_sign = np.where(np.diag(raw_upper) < 0.0, -1.0, 1.0)
        raw_upper = diagonal_sign[:, None] * raw_upper
        if np.any(np.diag(raw_upper) <= np.finfo(np.float64).eps):
            raise RuntimeError(
                "source quadrature cannot realize the declared four-dimensional covariance"
            )
        whitened = np.linalg.solve(raw_upper.T, centered.T).T
        target_covariance = np.eye(4, dtype=np.float64)
        target_covariance[0, 2] = target_covariance[2, 0] = position_divergence_correlation[0]
        target_covariance[1, 3] = target_covariance[3, 1] = position_divergence_correlation[1]
        target_lower = np.linalg.cholesky(target_covariance)
        standardized = whitened @ target_lower.T
        realized_covariance = standardized.T @ (weight[:, None] * standardized)
        if not np.allclose(
            realized_covariance,
            target_covariance,
            rtol=0.0,
            atol=3.0e-12,
        ):
            raise RuntimeError(
                "source quadrature cannot stably realize the declared four-dimensional covariance"
            )
        position = standardized[:, :2]
        correlated_divergence = standardized[:, 2:4]
        moment_matched = True
    else:
        position = gaussian[:, :2]
        independent_divergence = gaussian[:, 2:4]
        correlated_divergence = (
            position_divergence_correlation * position
            + np.sqrt(1.0 - position_divergence_correlation**2) * independent_divergence
        )
    origin = mean_origin + (position * spatial_sigma) @ axes
    tangent = (correlated_divergence * divergence_sigma) @ axes
    radius = np.linalg.norm(tangent, axis=1)
    sine_scale = np.divide(np.sin(radius), radius, out=np.ones_like(radius), where=radius != 0.0)
    direction = np.cos(radius)[:, None] * mean_direction + sine_scale[:, None] * tangent
    return origin, direction, moment_matched


def _source_batch(
    *,
    origin: np.ndarray,
    direction: np.ndarray,
    wavelength_A: np.ndarray,
    source_weight: np.ndarray,
    polarization_state_id: str,
    source_sampling_model_id: str,
    seed: int,
    parameter_provenance: str,
    rng_model_id: str = _RNG_MODEL_ID,
) -> IncidentSampleBatch:
    size = wavelength_A.size
    return IncidentSampleBatch(
        incident_sample_id=np.arange(size, dtype=np.int64),
        origin_lab_m=origin,
        direction_lab=direction,
        wavelength_A=wavelength_A,
        source_weight=source_weight,
        polarization_state_id=(polarization_state_id,) * size,
        source_sampling_model_id=source_sampling_model_id,
        source_rng_model_id=rng_model_id,
        source_seed=seed,
        source_parameter_provenance=parameter_provenance,
    )


def sample_gaussian_source_rays(
    *,
    mean_origin_lab_m: ArrayLike,
    mean_direction_lab: ArrayLike,
    transverse_axes_lab: ArrayLike,
    spatial_sigma_m: ArrayLike,
    divergence_sigma_rad: ArrayLike,
    mean_wavelength_A: float,
    wavelength_sigma_A: float,
    sample_count: int,
    seed: int,
    polarization_state_id: str,
    position_divergence_correlation: ArrayLike = (0.0, 0.0),
) -> IncidentSampleBatch:
    """Sample Gaussian source rays with optional per-axis position-angle correlation."""

    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, (int, np.integer))
        or sample_count <= 0
    ):
        raise ValueError("sample_count must be a positive integer")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or seed < 0
        or seed > 2**64 - 1
    ):
        raise ValueError("seed must be a nonnegative unsigned 64-bit integer")
    if not isinstance(polarization_state_id, str) or not polarization_state_id:
        raise ValueError("polarization_state_id must be a nonempty string")
    correlation = _finite_real(
        position_divergence_correlation,
        (2,),
        "position_divergence_correlation",
    )
    if np.any(np.abs(correlation) >= 1.0):
        raise ValueError("position_divergence_correlation must lie strictly within (-1, 1)")

    mean_origin, mean_direction, axes, spatial_sigma, divergence_sigma = _validated_source_geometry(
        mean_origin_lab_m=mean_origin_lab_m,
        mean_direction_lab=mean_direction_lab,
        transverse_axes_lab=transverse_axes_lab,
        spatial_sigma_m=spatial_sigma_m,
        divergence_sigma_rad=divergence_sigma_rad,
    )
    mean_wavelength = float(_finite_real(mean_wavelength_A, (), "mean_wavelength_A"))
    wavelength_sigma = float(_finite_real(wavelength_sigma_A, (), "wavelength_sigma_A"))
    if wavelength_sigma < 0.0:
        raise ValueError("source standard deviations must be nonnegative")
    if mean_wavelength <= 0.0:
        raise ValueError("mean_wavelength_A must be positive")
    size = int(sample_count)
    source_weight = np.full(size, 1.0 / size)
    gaussian, origin, direction, moment_matched = _sample_antithetic_geometry(
        size=size,
        seed=int(seed),
        mean_origin=mean_origin,
        mean_direction=mean_direction,
        axes=axes,
        spatial_sigma=spatial_sigma,
        divergence_sigma=divergence_sigma,
        position_divergence_correlation=correlation,
        source_weight=source_weight,
    )
    wavelength = mean_wavelength + wavelength_sigma * gaussian[:, 4]
    if np.any(wavelength <= 0.0):
        raise ValueError("sampled wavelengths must be positive")

    parameter_values = {
        "divergence_sigma_rad": [float(value).hex() for value in divergence_sigma],
        "mean_direction_lab": [float(value).hex() for value in mean_direction],
        "mean_origin_lab_m": [float(value).hex() for value in mean_origin],
        "mean_wavelength_A": mean_wavelength.hex(),
        "polarization_state_id": polarization_state_id,
        "sample_count": size,
        "spatial_sigma_m": [float(value).hex() for value in spatial_sigma],
        "transverse_axes_lab": [[float(value).hex() for value in row] for row in axes],
        "wavelength_sigma_A": wavelength_sigma.hex(),
    }
    if np.any(correlation != 0.0):
        parameter_values["position_divergence_correlation"] = [
            float(value).hex() for value in correlation
        ]
    parameter_units = {
        "divergence_sigma_rad": "rad",
        "mean_direction_lab": "1",
        "mean_origin_lab_m": "m",
        "mean_wavelength_A": "angstrom",
        "polarization_state_id": "id",
        "sample_count": "1",
        "spatial_sigma_m": "m",
        "transverse_axes_lab": "1",
        "wavelength_sigma_A": "angstrom",
    }
    if np.any(correlation != 0.0):
        parameter_units["position_divergence_correlation"] = "1"
    parameter_provenance = json.dumps(
        {
            "frames": {
                "mean_direction_lab": "lab",
                "mean_origin_lab_m": "lab",
                "transverse_axes_lab": "lab",
            },
            "units": parameter_units,
            "values": parameter_values,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _source_batch(
        origin=origin,
        direction=direction,
        wavelength_A=wavelength,
        source_weight=source_weight,
        polarization_state_id=polarization_state_id,
        source_sampling_model_id=(
            _CORRELATED_SAMPLING_MODEL_ID
            if np.any(correlation != 0.0) and moment_matched
            else _FINITE_CORRELATED_SAMPLING_MODEL_ID
            if np.any(correlation != 0.0)
            else _SAMPLING_MODEL_ID
        ),
        seed=int(seed),
        parameter_provenance=parameter_provenance,
    )


def sample_nominal_mean_geometry_source_ray(
    *,
    mean_origin_lab_m: ArrayLike,
    mean_direction_lab: ArrayLike,
    transverse_axes_lab: ArrayLike,
    spatial_sigma_m: ArrayLike,
    divergence_sigma_rad: ArrayLike,
    reference_wavelength_A: float,
    polarization_state_id: str,
    position_divergence_correlation: ArrayLike = (0.0, 0.0),
) -> IncidentSampleBatch:
    """Return the explicit one-ray centroid companion used only for nominal geometry."""

    sampled = sample_gaussian_source_rays(
        mean_origin_lab_m=mean_origin_lab_m,
        mean_direction_lab=mean_direction_lab,
        transverse_axes_lab=transverse_axes_lab,
        spatial_sigma_m=spatial_sigma_m,
        divergence_sigma_rad=divergence_sigma_rad,
        mean_wavelength_A=reference_wavelength_A,
        wavelength_sigma_A=0.0,
        sample_count=1,
        seed=0,
        polarization_state_id=polarization_state_id,
        position_divergence_correlation=position_divergence_correlation,
    )
    provenance = json.loads(sampled.source_parameter_provenance)
    provenance["model_id"] = NOMINAL_MEAN_GEOMETRY_REFERENCE_MODEL_ID
    provenance["purpose"] = "nominal_geometry_only_no_intensity"
    return _source_batch(
        origin=sampled.origin_lab_m,
        direction=sampled.direction_lab,
        wavelength_A=sampled.wavelength_A,
        source_weight=sampled.source_weight,
        polarization_state_id=polarization_state_id,
        source_sampling_model_id=NOMINAL_MEAN_GEOMETRY_REFERENCE_MODEL_ID,
        seed=0,
        parameter_provenance=json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
        rng_model_id=_NO_RNG_MODEL_ID,
    )


def sample_discrete_gaussian_line_source_rays(
    *,
    mean_origin_lab_m: ArrayLike,
    mean_direction_lab: ArrayLike,
    transverse_axes_lab: ArrayLike,
    spatial_sigma_m: ArrayLike,
    divergence_sigma_rad: ArrayLike,
    line_wavelength_A: ArrayLike,
    line_probability: ArrayLike,
    common_wavelength_sigma_A: float,
    sample_count: int,
    seed: int,
    polarization_state_id: str,
    position_divergence_correlation: ArrayLike = (0.0, 0.0),
) -> IncidentSampleBatch:
    """Sample Gaussian ray geometry and an exactly weighted discrete line mixture."""

    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, (int, np.integer))
        or sample_count <= 0
    ):
        raise ValueError("sample_count must be a positive integer")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or seed < 0
        or seed > 2**64 - 1
    ):
        raise ValueError("seed must be a nonnegative unsigned 64-bit integer")
    if not isinstance(polarization_state_id, str) or not polarization_state_id:
        raise ValueError("polarization_state_id must be a nonempty string")
    correlation = _finite_real(
        position_divergence_correlation,
        (2,),
        "position_divergence_correlation",
    )
    if np.any(np.abs(correlation) >= 1.0):
        raise ValueError("position_divergence_correlation must lie strictly within (-1, 1)")
    mean_origin, mean_direction, axes, spatial_sigma, divergence_sigma = _validated_source_geometry(
        mean_origin_lab_m=mean_origin_lab_m,
        mean_direction_lab=mean_direction_lab,
        transverse_axes_lab=transverse_axes_lab,
        spatial_sigma_m=spatial_sigma_m,
        divergence_sigma_rad=divergence_sigma_rad,
    )
    supplied_lines = np.asarray(line_wavelength_A, dtype=object)
    supplied_probabilities = np.asarray(line_probability, dtype=object)
    if not _contains_only_real_numbers(supplied_lines) or not _contains_only_real_numbers(
        supplied_probabilities
    ):
        raise ValueError("source line wavelengths and probabilities must be real numbers")
    lines = np.asarray(supplied_lines, dtype=np.float64)
    probabilities = np.asarray(supplied_probabilities, dtype=np.float64)
    if (
        lines.ndim != 1
        or probabilities.shape != lines.shape
        or lines.size == 0
        or np.any(~np.isfinite(lines))
        or np.any(lines <= 0.0)
        or np.unique(lines).size != lines.size
    ):
        raise ValueError("line_wavelength_A must contain distinct finite positive wavelengths")
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or not np.isclose(np.sum(probabilities, dtype=np.float64), 1.0, rtol=0.0, atol=2.0e-15)
    ):
        raise ValueError("line_probability must contain positive masses summing to one")
    if sample_count < lines.size:
        raise ValueError("sample_count must be at least the number of source lines")
    sigma = float(_finite_real(common_wavelength_sigma_A, (), "common_wavelength_sigma_A"))
    if sigma < 0.0:
        raise ValueError("line width must be nonnegative")

    size = int(sample_count)
    count = np.full(lines.size, size // lines.size, dtype=np.int64)
    count[: size % lines.size] += 1
    origin_parts: list[np.ndarray] = []
    direction_parts: list[np.ndarray] = []
    wavelength_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    line_moment_matched: list[bool] = []
    for index, line_count in enumerate(count):
        local_row_weight = probabilities[index] / float(line_count)
        realized_line_mass = math.fsum([local_row_weight] * int(line_count))
        if local_row_weight <= 0.0 or not math.isclose(
            realized_line_mass,
            float(probabilities[index]),
            rel_tol=8.0 * np.finfo(np.float64).eps,
            abs_tol=0.0,
        ):
            raise ValueError("line probability is too small for the allocated source rows")
        local_gaussian, local_origin, local_direction, local_moment_matched = (
            _sample_antithetic_geometry(
                size=int(line_count),
                seed=int(seed),
                mean_origin=mean_origin,
                mean_direction=mean_direction,
                axes=axes,
                spatial_sigma=spatial_sigma,
                divergence_sigma=divergence_sigma,
                position_divergence_correlation=correlation,
                source_weight=np.full(int(line_count), 1.0 / float(line_count)),
            )
        )
        local_wavelength = np.full(int(line_count), lines[index], dtype=np.float64)
        if sigma > 0.0:
            local_wavelength += sigma * local_gaussian[:, 4]
        origin_parts.append(local_origin)
        direction_parts.append(local_direction)
        wavelength_parts.append(local_wavelength)
        weight_parts.append(np.full(int(line_count), local_row_weight))
        line_moment_matched.append(local_moment_matched)
    origin = np.concatenate(origin_parts, axis=0)
    direction = np.concatenate(direction_parts, axis=0)
    wavelength = np.concatenate(wavelength_parts)
    source_weight = np.concatenate(weight_parts)
    if np.any(wavelength <= 0.0):
        raise ValueError("sampled wavelengths must be positive")
    moment_matched = all(line_moment_matched)

    parameter_provenance = json.dumps(
        {
            "frames": {
                "mean_direction_lab": "lab",
                "mean_origin_lab_m": "lab",
                "transverse_axes_lab": "lab",
            },
            "model_id": "discrete_gaussian_lines.v1",
            "units": {
                "common_wavelength_sigma_A": "angstrom",
                "divergence_sigma_rad": "rad",
                "line_probability": "1",
                "line_wavelength_A": "angstrom",
                "mean_direction_lab": "1",
                "mean_origin_lab_m": "m",
                "polarization_state_id": "id",
                "position_divergence_correlation": "1",
                "sample_count": "1",
                "spatial_sigma_m": "m",
                "transverse_axes_lab": "1",
            },
            "values": {
                "common_wavelength_sigma_A": sigma.hex(),
                "divergence_sigma_rad": [float(value).hex() for value in divergence_sigma],
                "line_probability": [float(value).hex() for value in probabilities],
                "line_wavelength_A": [float(value).hex() for value in lines],
                "mean_direction_lab": [float(value).hex() for value in mean_direction],
                "mean_origin_lab_m": [float(value).hex() for value in mean_origin],
                "polarization_state_id": polarization_state_id,
                "position_divergence_correlation": [float(value).hex() for value in correlation],
                "sample_count": size,
                "spatial_sigma_m": [float(value).hex() for value in spatial_sigma],
                "transverse_axes_lab": [[float(value).hex() for value in row] for row in axes],
            },
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _source_batch(
        origin=origin,
        direction=direction,
        wavelength_A=wavelength,
        source_weight=source_weight,
        polarization_state_id=polarization_state_id,
        source_sampling_model_id=(
            _FINITE_UNEQUAL_MATCHED_DISCRETE_LINE_SAMPLING_MODEL_ID
            if size % lines.size != 0 and np.any(correlation != 0.0) and moment_matched
            else _FINITE_UNEQUAL_CORRELATED_DISCRETE_LINE_SAMPLING_MODEL_ID
            if size % lines.size != 0 and np.any(correlation != 0.0)
            else _FINITE_UNEQUAL_DISCRETE_LINE_SAMPLING_MODEL_ID
            if size % lines.size != 0
            else _CORRELATED_DISCRETE_LINE_SAMPLING_MODEL_ID
            if np.any(correlation != 0.0) and moment_matched
            else _FINITE_CORRELATED_DISCRETE_LINE_SAMPLING_MODEL_ID
            if np.any(correlation != 0.0)
            else _DISCRETE_LINE_SAMPLING_MODEL_ID
        ),
        seed=int(seed),
        parameter_provenance=parameter_provenance,
    )
