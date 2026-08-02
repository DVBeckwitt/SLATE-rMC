"""Joint fitting of heterogeneous detector-region observations."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares, nnls

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FixedMatchedRegionBackground:
    """Frozen background mass, uncertainty, and anchor interpolation operator."""

    count_mass: ArrayLike
    covariance_count2: ArrayLike
    revision: str
    anchor_projection: ArrayLike | None = None

    def __post_init__(self) -> None:
        mass = np.array(self.count_mass, dtype=np.float64, copy=True, order="C")
        covariance = np.array(
            self.covariance_count2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        projection = (
            np.zeros((mass.size, mass.size), dtype=np.float64)
            if self.anchor_projection is None
            else np.array(self.anchor_projection, dtype=np.float64, copy=True, order="C")
        )
        covariance_eigenvalue = (
            np.linalg.eigvalsh(covariance)
            if covariance.shape == (mass.size, mass.size)
            else np.asarray((-math.inf,))
        )
        if (
            mass.ndim != 1
            or not mass.size
            or covariance.shape != (mass.size, mass.size)
            or projection.shape != (mass.size, mass.size)
            or np.any(~np.isfinite(mass))
            or np.any(mass < 0.0)
            or np.any(~np.isfinite(covariance))
            or np.any(~np.isfinite(projection))
            or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-10)
            or np.min(covariance_eigenvalue)
            < -1.0e-10 * max(float(np.max(np.abs(covariance_eigenvalue))), 1.0)
            or not isinstance(self.revision, str)
            or not self.revision
        ):
            raise ValueError("fixed matched-region background is invalid")
        mass.setflags(write=False)
        covariance.setflags(write=False)
        projection.setflags(write=False)
        object.__setattr__(self, "count_mass", mass)
        object.__setattr__(self, "covariance_count2", covariance)
        object.__setattr__(self, "anchor_projection", projection)


@dataclass(frozen=True, slots=True)
class MatchedRegionObservations:
    """Raw count masses grouped into complete signal/background blocks."""

    dataset_ids: tuple[str, ...]
    dataset_index: ArrayLike
    block_index: ArrayLike
    signal_family: ArrayLike
    is_background: ArrayLike
    count_mass: ArrayLike
    support_px2: ArrayLike
    background_coordinate: ArrayLike
    required_signal_families: tuple[int, ...]
    count_covariance_count2: ArrayLike | None = None

    def __post_init__(self) -> None:
        dataset_ids = tuple(self.dataset_ids)
        required = tuple(int(value) for value in self.required_signal_families)
        if (
            not dataset_ids
            or any(not isinstance(value, str) or not value for value in dataset_ids)
            or len(set(dataset_ids)) != len(dataset_ids)
        ):
            raise ValueError("dataset_ids must contain unique nonempty strings")
        if (
            not required
            or len(set(required)) != len(required)
            or any(value < 0 for value in required)
        ):
            raise ValueError("required_signal_families must contain unique nonnegative values")
        dataset = np.array(self.dataset_index, dtype=np.int64, copy=True, order="C")
        block = np.array(self.block_index, dtype=np.int64, copy=True, order="C")
        family = np.array(self.signal_family, dtype=np.int64, copy=True, order="C")
        background = np.array(self.is_background, dtype=np.bool_, copy=True, order="C")
        count = np.array(self.count_mass, dtype=np.float64, copy=True, order="C")
        support = np.array(self.support_px2, dtype=np.float64, copy=True, order="C")
        coordinate = np.array(
            self.background_coordinate,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        shape = count.shape
        count_covariance = (
            np.diag(np.maximum(count, 1.0))
            if self.count_covariance_count2 is None
            else np.array(
                self.count_covariance_count2,
                dtype=np.float64,
                copy=True,
                order="C",
            )
        )
        covariance_eigenvalue = (
            np.linalg.eigvalsh(count_covariance)
            if count_covariance.shape == (count.size, count.size)
            else np.asarray((-math.inf,))
        )
        if (
            count.ndim != 1
            or not count.size
            or dataset.shape != shape
            or block.shape != shape
            or family.shape != shape
            or background.shape != shape
            or support.shape != shape
            or coordinate.shape != shape
            or count_covariance.shape != (count.size, count.size)
        ):
            raise ValueError("matched-region observation arrays must be aligned nonempty vectors")
        if (
            np.any((dataset < 0) | (dataset >= len(dataset_ids)))
            or np.any(block < 0)
            or np.any(count < 0.0)
            or np.any(~np.isfinite(count))
            or np.any(~np.isfinite(support))
            or np.any(support <= 0.0)
            or np.any(~np.isfinite(coordinate))
            or np.any(~np.isfinite(count_covariance))
            or not np.allclose(count_covariance, count_covariance.T, rtol=0.0, atol=1.0e-10)
            or np.min(covariance_eigenvalue)
            < -1.0e-10 * max(float(np.max(np.abs(covariance_eigenvalue))), 1.0)
            or np.any(background & (family != -1))
            or np.any((~background) & ~np.isin(family, required))
        ):
            raise ValueError("matched-region observations contain invalid values")
        unique_blocks = np.unique(block)
        if not np.array_equal(unique_blocks, np.arange(unique_blocks.size)):
            raise ValueError("block_index must be contiguous from zero")
        for block_id in unique_blocks:
            selected = block == block_id
            if np.unique(dataset[selected]).size != 1:
                raise ValueError("one background block cannot cross datasets")
            if np.count_nonzero(background[selected]) != 2 or not np.any(~background[selected]):
                raise ValueError("every block requires signals and exactly two background anchors")
            design = np.column_stack((support[selected], support[selected] * coordinate[selected]))
            if np.linalg.matrix_rank(design) != 2:
                raise ValueError("every block requires a rank-two affine background design")
        represented_globally = set(family[~background].tolist())
        missing = set(required) - represented_globally
        if missing:
            raise ValueError(f"joint observations lack signal families {sorted(missing)}")
        for value in (
            dataset,
            block,
            family,
            background,
            count,
            support,
            coordinate,
            count_covariance,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(self, "dataset_index", dataset)
        object.__setattr__(self, "block_index", block)
        object.__setattr__(self, "signal_family", family)
        object.__setattr__(self, "is_background", background)
        object.__setattr__(self, "count_mass", count)
        object.__setattr__(self, "support_px2", support)
        object.__setattr__(self, "background_coordinate", coordinate)
        object.__setattr__(self, "required_signal_families", required)
        object.__setattr__(self, "count_covariance_count2", count_covariance)


@dataclass(frozen=True, slots=True)
class MatchedRegionFitResult:
    """One joint solution with dataset scales and affine block backgrounds."""

    parameter_names: tuple[str, ...]
    parameters: FloatArray
    dataset_scales: FloatArray
    fitted_model_mass: FloatArray
    fitted_objective_model_mass: FloatArray
    fitted_background_mass: FloatArray
    fitted_signal_row: BoolArray
    weighted_residual: FloatArray
    prior_weighted_residual: FloatArray
    objective_half_chi_squared: float
    data_objective_half_chi_squared: float
    prior_objective_half_chi_squared: float
    jacobian_singular_values: FloatArray
    sensitivity_rank: int
    sensitivity_numerical_rank: int
    sensitivity_condition: float
    sensitivity_relative_tolerance: float
    penalized_jacobian_singular_values: FloatArray
    penalized_sensitivity_rank: int
    penalized_sensitivity_numerical_rank: int
    penalized_sensitivity_condition: float
    parameter_correlation: FloatArray
    success: bool
    optimizer_message: str
    function_evaluations: int


def condition_matched_region_background_from_anchors(
    observations: MatchedRegionObservations,
    baseline: FixedMatchedRegionBackground,
) -> FixedMatchedRegionBackground:
    """Condition a frozen baseline on the two measured anchors in every block.

    The anchor residual above the baseline is treated as an affine density in
    the block coordinate.  Its prediction is frozen before structure fitting;
    anchor counting noise and baseline covariance are propagated into the
    resulting signal-background covariance.
    """

    if not isinstance(observations, MatchedRegionObservations):
        raise TypeError("observations must be MatchedRegionObservations")
    if not isinstance(baseline, FixedMatchedRegionBackground):
        raise TypeError("baseline must be FixedMatchedRegionBackground")
    count = np.asarray(observations.count_mass)
    if baseline.count_mass.shape != count.shape:
        raise ValueError("baseline background and observations must align")

    row_count = count.size
    conditioned_mass = np.array(baseline.count_mass, copy=True)
    if np.any(np.asarray(baseline.anchor_projection) != 0.0):
        raise ValueError("background baseline is already conditioned on anchors")
    baseline_transform = np.eye(row_count, dtype=np.float64)
    anchor_projection = np.zeros((row_count, row_count), dtype=np.float64)
    support = np.asarray(observations.support_px2)
    coordinate = np.asarray(observations.background_coordinate)
    is_background = np.asarray(observations.is_background)
    for block_id in range(int(np.max(observations.block_index)) + 1):
        index = np.flatnonzero(observations.block_index == block_id)
        anchor_local = np.flatnonzero(is_background[index])
        anchor_index = index[anchor_local]
        anchor_coordinate = coordinate[anchor_index]
        signal_coordinate = coordinate[index[~is_background[index]]]
        if (
            anchor_coordinate[0] == anchor_coordinate[1]
            or np.any(signal_coordinate < np.min(anchor_coordinate))
            or np.any(signal_coordinate > np.max(anchor_coordinate))
        ):
            raise ValueError("adjacent background anchors must bracket every signal coordinate")
        design = np.column_stack((support[index], support[index] * coordinate[index]))
        anchor_design = design[anchor_local]
        interpolation = np.linalg.solve(anchor_design.T, design.T).T
        residual_anchor_mass = count[anchor_index] - baseline.count_mass[anchor_index]
        conditioned_mass[index] += interpolation @ residual_anchor_mass

        selector = np.zeros((2, index.size), dtype=np.float64)
        selector[np.arange(2), anchor_local] = 1.0
        block_projection = interpolation @ selector
        anchor_projection[np.ix_(index, index)] = block_projection
        baseline_transform[np.ix_(index, index)] -= block_projection
    count_covariance = np.asarray(observations.count_covariance_count2)
    covariance = (
        baseline_transform @ np.asarray(baseline.covariance_count2) @ baseline_transform.T
        + anchor_projection @ count_covariance @ anchor_projection.T
    )
    covariance = 0.5 * (covariance + covariance.T)
    digest = hashlib.sha256()
    digest.update(b"matched-region-adjacent-affine-background.v1\0")
    digest.update(baseline.revision.encode("utf-8"))
    digest.update(np.ascontiguousarray(conditioned_mass).tobytes())
    digest.update(np.ascontiguousarray(covariance).tobytes())
    digest.update(np.ascontiguousarray(anchor_projection).tobytes())
    return FixedMatchedRegionBackground(
        count_mass=conditioned_mass,
        covariance_count2=covariance,
        revision=f"sha256-{digest.hexdigest()}.adjacent-affine.v1",
        anchor_projection=anchor_projection,
    )


def condition_matched_region_model_from_anchors(
    background: FixedMatchedRegionBackground,
    model_mass: ArrayLike,
) -> FloatArray:
    """Apply the data-side anchor subtraction operator to a diffraction model."""

    if not isinstance(background, FixedMatchedRegionBackground):
        raise TypeError("background must be FixedMatchedRegionBackground")
    model = np.asarray(model_mass, dtype=np.float64)
    if (
        model.shape != background.count_mass.shape
        or np.any(~np.isfinite(model))
        or np.any(model < 0.0)
    ):
        raise ValueError("model_mass must be a finite aligned nonnegative vector")
    return model - np.asarray(background.anchor_projection) @ model


def profile_matched_region_nuisance(
    model_mass: ArrayLike,
    observations: MatchedRegionObservations,
    fixed_background: FixedMatchedRegionBackground,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Profile dataset scales against one frozen conditioned background."""

    if not isinstance(observations, MatchedRegionObservations):
        raise TypeError("observations must be MatchedRegionObservations")
    model = np.asarray(model_mass, dtype=np.float64)
    count = np.asarray(observations.count_mass)
    if model.shape != count.shape or not np.all(np.isfinite(model)) or np.any(model < 0.0):
        raise ValueError("model_mass must be a finite nonnegative aligned vector")
    if not isinstance(fixed_background, FixedMatchedRegionBackground):
        raise TypeError("fixed_background must be FixedMatchedRegionBackground")
    if fixed_background.count_mass.shape != count.shape:
        raise ValueError("fixed background and observations must align")
    signal = ~np.asarray(observations.is_background)
    signal_index = np.flatnonzero(signal)
    count_covariance = np.asarray(observations.count_covariance_count2)
    anchor_projection = np.asarray(fixed_background.anchor_projection)
    covariance = (
        count_covariance
        + np.asarray(fixed_background.covariance_count2)
        - anchor_projection @ count_covariance
        - count_covariance @ anchor_projection.T
    )
    covariance = covariance[np.ix_(signal_index, signal_index)]
    covariance = 0.5 * (covariance + covariance.T)
    root_covariance = cholesky(covariance, lower=True, check_finite=False)
    conditioned_model = condition_matched_region_model_from_anchors(fixed_background, model)
    corrected = count[signal] - fixed_background.count_mass[signal]
    design = np.zeros((signal_index.size, len(observations.dataset_ids)), dtype=np.float64)
    design[
        np.arange(signal_index.size),
        np.asarray(observations.dataset_index)[signal],
    ] = conditioned_model[signal]
    whitened_count = solve_triangular(
        root_covariance,
        corrected,
        lower=True,
        check_finite=False,
    )
    whitened_design = solve_triangular(
        root_covariance,
        design,
        lower=True,
        check_finite=False,
    )
    if np.any(np.linalg.norm(whitened_design, axis=0) <= np.finfo(np.float64).tiny):
        raise ValueError("candidate model has no scale-identifying signal in one dataset")
    scales, _ = nnls(whitened_design, whitened_count)
    residual = np.zeros_like(count)
    residual[signal] = whitened_count - whitened_design @ scales
    return residual, scales, np.array(fixed_background.count_mass, copy=True)


def fit_matched_regions(
    observations: MatchedRegionObservations,
    predict_model_mass: Callable[[FloatArray], ArrayLike],
    *,
    fixed_background: FixedMatchedRegionBackground,
    parameter_names: Sequence[str],
    initial_parameters: Sequence[ArrayLike],
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    parameter_scales: ArrayLike | None = None,
    prior_residual: Callable[[FloatArray], ArrayLike] | None = None,
    sensitivity_relative_tolerance: float = 1.0e-5,
    maximum_function_evaluations: int = 200,
) -> MatchedRegionFitResult:
    """Fit one parameter vector to every family and dataset simultaneously."""

    if not isinstance(observations, MatchedRegionObservations):
        raise TypeError("observations must be MatchedRegionObservations")
    names = tuple(parameter_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("parameter_names must contain nonempty strings")
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if (
        lower.shape != (len(names),)
        or upper.shape != lower.shape
        or np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or np.any(lower >= upper)
    ):
        raise ValueError("parameter bounds must be aligned, finite, and increasing")
    starts = tuple(np.asarray(value, dtype=np.float64) for value in initial_parameters)
    if not starts or any(
        value.shape != lower.shape or np.any(value < lower) or np.any(value > upper)
        for value in starts
    ):
        raise ValueError("initial parameters must lie inside the declared bounds")
    maximum = int(maximum_function_evaluations)
    if maximum <= 0:
        raise ValueError("maximum_function_evaluations must be positive")
    relative_tolerance = float(sensitivity_relative_tolerance)
    if (
        not math.isfinite(relative_tolerance)
        or relative_tolerance <= 0.0
        or relative_tolerance >= 1.0
    ):
        raise ValueError("sensitivity_relative_tolerance must lie strictly between zero and one")
    if parameter_scales is None:
        parameter_coordinate_scales = np.ones(len(names), dtype=np.float64)
        optimizer_scales: FloatArray | str = "jac"
    else:
        parameter_coordinate_scales = np.asarray(parameter_scales, dtype=np.float64)
        if (
            parameter_coordinate_scales.shape != lower.shape
            or np.any(~np.isfinite(parameter_coordinate_scales))
            or np.any(parameter_coordinate_scales <= 0.0)
        ):
            raise ValueError("parameter_scales must be aligned, finite, and positive")
        optimizer_scales = parameter_coordinate_scales

    def evaluated_prior(parameters: FloatArray) -> FloatArray:
        if prior_residual is None:
            return np.empty(0, dtype=np.float64)
        values = np.asarray(prior_residual(parameters), dtype=np.float64)
        if values.ndim != 1 or np.any(~np.isfinite(values)):
            raise ValueError("prior_residual must return one finite vector")
        return values

    prior_size = evaluated_prior(starts[0]).size
    if any(evaluated_prior(start).size != prior_size for start in starts[1:]):
        raise ValueError("prior_residual shape must be parameter independent")

    def data_residual(parameters: FloatArray) -> FloatArray:
        model = np.asarray(predict_model_mass(parameters), dtype=np.float64)
        values, _, _ = profile_matched_region_nuisance(
            model,
            observations,
            fixed_background,
        )
        return values[~observations.is_background]

    def residual(parameters: FloatArray) -> FloatArray:
        return np.concatenate((data_residual(parameters), evaluated_prior(parameters)))

    fitted = tuple(
        least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=maximum,
            x_scale=optimizer_scales,
        )
        for start in starts
    )
    selected = min(fitted, key=lambda result: float(result.cost))
    model = np.asarray(predict_model_mass(selected.x), dtype=np.float64)
    weighted_residual, dataset_scales, background = profile_matched_region_nuisance(
        model,
        observations,
        fixed_background,
    )
    signal_row = ~np.asarray(observations.is_background)
    data_row_count = int(np.count_nonzero(signal_row))
    data_jacobian = (
        np.asarray(selected.jac[:data_row_count], dtype=np.float64)
        * parameter_coordinate_scales[None, :]
    )
    penalized_jacobian = (
        np.asarray(selected.jac, dtype=np.float64) * parameter_coordinate_scales[None, :]
    )

    def diagnostics(jacobian: FloatArray) -> tuple[FloatArray, int, int, float]:
        singular = np.linalg.svd(jacobian, compute_uv=False)
        if singular.size < len(names):
            singular = np.pad(singular, (0, len(names) - singular.size))
        if not singular.size or singular[0] == 0.0:
            return singular, 0, 0, math.inf
        numerical_tolerance = 64.0 * np.finfo(np.float64).eps * max(jacobian.shape) * singular[0]
        practical_tolerance = max(
            numerical_tolerance,
            relative_tolerance * singular[0],
        )
        numerical_rank = int(np.count_nonzero(singular > numerical_tolerance))
        practical_rank = int(np.count_nonzero(singular >= practical_tolerance))
        condition = (
            float(singular[0] / singular[-1])
            if numerical_rank == len(names) and singular[-1] > 0.0
            else math.inf
        )
        return singular, practical_rank, numerical_rank, condition

    singular_values, rank, numerical_rank, condition = diagnostics(data_jacobian)
    penalized_singular, penalized_rank, penalized_numerical_rank, penalized_condition = diagnostics(
        penalized_jacobian
    )
    if rank == len(names):
        covariance = np.linalg.inv(data_jacobian.T @ data_jacobian)
        standard_deviation = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(standard_deviation, standard_deviation)
    else:
        correlation = np.full((len(names), len(names)), np.nan, dtype=np.float64)
    prior_weighted_residual = evaluated_prior(np.asarray(selected.x, dtype=np.float64))
    data_objective = 0.5 * float(weighted_residual[signal_row] @ weighted_residual[signal_row])
    prior_objective = 0.5 * float(prior_weighted_residual @ prior_weighted_residual)
    frozen = (
        np.array(selected.x, copy=True),
        np.array(dataset_scales, copy=True),
        np.array(dataset_scales[np.asarray(observations.dataset_index)] * model, copy=True),
        np.array(
            dataset_scales[np.asarray(observations.dataset_index)]
            * condition_matched_region_model_from_anchors(fixed_background, model),
            copy=True,
        ),
        np.array(background, copy=True),
        np.array(signal_row, copy=True),
        np.array(weighted_residual, copy=True),
        np.array(prior_weighted_residual, copy=True),
        np.array(singular_values, copy=True),
        np.array(penalized_singular, copy=True),
        np.array(correlation, copy=True),
    )
    for value in frozen:
        value.setflags(write=False)
    return MatchedRegionFitResult(
        parameter_names=names,
        parameters=frozen[0],
        dataset_scales=frozen[1],
        fitted_model_mass=frozen[2],
        fitted_objective_model_mass=frozen[3],
        fitted_background_mass=frozen[4],
        fitted_signal_row=frozen[5],
        weighted_residual=frozen[6],
        prior_weighted_residual=frozen[7],
        objective_half_chi_squared=data_objective + prior_objective,
        data_objective_half_chi_squared=data_objective,
        prior_objective_half_chi_squared=prior_objective,
        jacobian_singular_values=frozen[8],
        sensitivity_rank=rank,
        sensitivity_numerical_rank=numerical_rank,
        sensitivity_condition=condition,
        sensitivity_relative_tolerance=relative_tolerance,
        penalized_jacobian_singular_values=frozen[9],
        penalized_sensitivity_rank=penalized_rank,
        penalized_sensitivity_numerical_rank=penalized_numerical_rank,
        penalized_sensitivity_condition=penalized_condition,
        parameter_correlation=frozen[10],
        success=bool(selected.success),
        optimizer_message=str(selected.message),
        function_evaluations=int(sum(result.nfev for result in fitted)),
    )


__all__ = [
    "FixedMatchedRegionBackground",
    "MatchedRegionFitResult",
    "MatchedRegionObservations",
    "condition_matched_region_background_from_anchors",
    "condition_matched_region_model_from_anchors",
    "fit_matched_regions",
    "profile_matched_region_nuisance",
]
