"""Shared radial detector-background calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from rasim_next.core.contracts import canonical_revision_sha256

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _frozen_float(value: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    if result.ndim != ndim or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class RadialBackgroundProfiles:
    """Robust radial-by-azimuth background summaries from detector-native pixels."""

    dataset_ids: tuple[str, ...]
    dataset_index: ArrayLike
    radius_px: ArrayLike
    azimuth_sector_index: ArrayLike
    density_count_per_px: ArrayLike
    support_px2: ArrayLike
    is_training: ArrayLike

    def __post_init__(self) -> None:
        dataset_ids = tuple(self.dataset_ids)
        if (
            not dataset_ids
            or len(set(dataset_ids)) != len(dataset_ids)
            or any(not isinstance(value, str) or not value for value in dataset_ids)
        ):
            raise ValueError("dataset_ids must contain unique nonempty strings")
        radius = _frozen_float(self.radius_px, name="radius_px", ndim=1)
        density = _frozen_float(
            self.density_count_per_px,
            name="density_count_per_px",
            ndim=1,
        )
        support = _frozen_float(self.support_px2, name="support_px2", ndim=1)
        dataset = np.array(self.dataset_index, dtype=np.int64, copy=True, order="C")
        sector = np.array(self.azimuth_sector_index, dtype=np.int64, copy=True, order="C")
        training = np.array(self.is_training, dtype=np.bool_, copy=True, order="C")
        shape = radius.shape
        if (
            not radius.size
            or density.shape != shape
            or support.shape != shape
            or dataset.shape != shape
            or sector.shape != shape
            or training.shape != shape
            or np.any(radius < 0.0)
            or np.any(support <= 0.0)
            or np.any((dataset < 0) | (dataset >= len(dataset_ids)))
            or np.any(sector < 0)
            or not np.any(training)
            or not np.any(~training)
        ):
            raise ValueError("radial background profiles are invalid or misaligned")
        for dataset_index in range(len(dataset_ids)):
            selected = dataset == dataset_index
            if not np.any(selected & training) or not np.any(selected & ~training):
                raise ValueError("every dataset requires training and held-out profiles")
        dataset.setflags(write=False)
        sector.setflags(write=False)
        training.setflags(write=False)
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(self, "dataset_index", dataset)
        object.__setattr__(self, "radius_px", radius)
        object.__setattr__(self, "azimuth_sector_index", sector)
        object.__setattr__(self, "density_count_per_px", density)
        object.__setattr__(self, "support_px2", support)
        object.__setattr__(self, "is_training", training)


def _halo_shape(
    radius_px: FloatArray,
    inner_scale_px: float,
    inner_power: float,
    outer_scale_px: float,
    outer_power: float,
) -> FloatArray:
    radius = np.asarray(radius_px, dtype=np.float64)
    inner = np.power(radius / inner_scale_px, inner_power)
    outer = np.power(radius / outer_scale_px, outer_power)
    return -np.expm1(-inner) * np.exp(-outer)


@dataclass(frozen=True, slots=True)
class RadialBackgroundState:
    """One shared rise-times-decay halo with per-dataset amplitude and pedestal."""

    dataset_ids: tuple[str, ...]
    inner_scale_px: float
    inner_power: float
    outer_scale_px: float
    outer_power: float
    amplitude_count_per_px: ArrayLike
    pedestal_count_per_px: ArrayLike
    parameter_covariance: ArrayLike
    revision: str = field(init=False)

    def __post_init__(self) -> None:
        dataset_ids = tuple(self.dataset_ids)
        shape_values = tuple(
            float(value)
            for value in (
                self.inner_scale_px,
                self.inner_power,
                self.outer_scale_px,
                self.outer_power,
            )
        )
        amplitude = _frozen_float(
            self.amplitude_count_per_px,
            name="amplitude_count_per_px",
            ndim=1,
        )
        pedestal = _frozen_float(
            self.pedestal_count_per_px,
            name="pedestal_count_per_px",
            ndim=1,
        )
        parameter_count = 4 + 2 * len(dataset_ids)
        covariance = _frozen_float(
            self.parameter_covariance,
            name="parameter_covariance",
            ndim=2,
        )
        covariance_eigenvalues = (
            np.linalg.eigvalsh(covariance)
            if covariance.shape == (parameter_count, parameter_count)
            else np.asarray((-math.inf,))
        )
        if not dataset_ids or len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("radial background dataset_ids are invalid")
        if amplitude.shape != (len(dataset_ids),) or pedestal.shape != amplitude.shape:
            raise ValueError("radial background dataset parameters are misaligned")
        if covariance.shape != (parameter_count, parameter_count):
            raise ValueError("radial background covariance has the wrong shape")
        if any(not math.isfinite(value) or value <= 0.0 for value in shape_values):
            raise ValueError("radial background shape parameters must be finite and positive")
        if shape_values[0] >= shape_values[2]:
            raise ValueError("radial background inner scale must precede the outer scale")
        if np.any(amplitude < 0.0) or np.any(pedestal < 0.0):
            raise ValueError("radial background amplitudes and pedestals must be nonnegative")
        if not np.allclose(covariance, covariance.T, rtol=1.0e-12, atol=1.0e-10):
            raise ValueError("radial background covariance must be symmetric")
        if np.min(covariance_eigenvalues) < -1.0e-10 * max(
            float(np.max(np.abs(covariance_eigenvalues))), 1.0
        ):
            raise ValueError("radial background covariance must be positive semidefinite")
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(self, "inner_scale_px", shape_values[0])
        object.__setattr__(self, "inner_power", shape_values[1])
        object.__setattr__(self, "outer_scale_px", shape_values[2])
        object.__setattr__(self, "outer_power", shape_values[3])
        object.__setattr__(self, "amplitude_count_per_px", amplitude)
        object.__setattr__(self, "pedestal_count_per_px", pedestal)
        object.__setattr__(self, "parameter_covariance", covariance)
        object.__setattr__(
            self,
            "revision",
            canonical_revision_sha256(
                ("definition_id", "shared_rise_decay_radial_background.v2"),
                ("dataset_ids", dataset_ids),
                ("shape_parameters", np.asarray(shape_values, dtype=np.float64)),
                ("amplitude_count_per_px", amplitude),
                ("pedestal_count_per_px", pedestal),
                ("parameter_covariance", covariance),
            ),
        )

    @property
    def parameter_vector(self) -> FloatArray:
        value = np.concatenate(
            (
                np.asarray(
                    (
                        self.inner_scale_px,
                        self.inner_power,
                        self.outer_scale_px,
                        self.outer_power,
                    )
                ),
                self.amplitude_count_per_px,
                self.pedestal_count_per_px,
            )
        )
        value.setflags(write=False)
        return value

    @classmethod
    def from_parameter_vector(
        cls,
        dataset_ids: tuple[str, ...],
        parameter_vector: ArrayLike,
        *,
        parameter_covariance: ArrayLike,
    ) -> RadialBackgroundState:
        values = np.asarray(parameter_vector, dtype=np.float64)
        dataset_count = len(tuple(dataset_ids))
        if values.shape != (4 + 2 * dataset_count,) or np.any(~np.isfinite(values)):
            raise ValueError("radial background parameter vector is invalid")
        return cls(
            dataset_ids=tuple(dataset_ids),
            inner_scale_px=float(values[0]),
            inner_power=float(values[1]),
            outer_scale_px=float(values[2]),
            outer_power=float(values[3]),
            amplitude_count_per_px=values[4 : 4 + dataset_count],
            pedestal_count_per_px=values[4 + dataset_count :],
            parameter_covariance=parameter_covariance,
        )

    def _dataset_index(self, dataset: int | str | ArrayLike, shape: tuple[int, ...]) -> IntArray:
        if isinstance(dataset, str):
            try:
                value = self.dataset_ids.index(dataset)
            except ValueError as error:
                raise ValueError(f"unknown background dataset {dataset!r}") from error
            result = np.full(shape, value, dtype=np.int64)
        elif np.isscalar(dataset):
            result = np.full(shape, int(dataset), dtype=np.int64)
        else:
            result = np.asarray(dataset, dtype=np.int64)
            if result.shape != shape:
                raise ValueError("dataset indices and radii must align")
        if np.any((result < 0) | (result >= len(self.dataset_ids))):
            raise ValueError("background dataset index is outside the calibrated series")
        return result

    def count_density(self, dataset: int | str | ArrayLike, radius_px: ArrayLike) -> FloatArray:
        radius = np.asarray(radius_px, dtype=np.float64)
        if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
            raise ValueError("radius_px must be finite and nonnegative")
        index = self._dataset_index(dataset, radius.shape)
        shape = _halo_shape(
            radius,
            self.inner_scale_px,
            self.inner_power,
            self.outer_scale_px,
            self.outer_power,
        )
        return self.pedestal_count_per_px[index] + self.amplitude_count_per_px[index] * shape

    def count_density_parameter_jacobian(
        self,
        dataset: int | str | ArrayLike,
        radius_px: ArrayLike,
    ) -> FloatArray:
        radius = np.asarray(radius_px, dtype=np.float64)
        if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
            raise ValueError("radius_px must be finite and nonnegative")
        index = self._dataset_index(dataset, radius.shape)
        safe_radius = np.maximum(radius, np.finfo(np.float64).tiny)
        inner = np.power(radius / self.inner_scale_px, self.inner_power)
        outer = np.power(radius / self.outer_scale_px, self.outer_power)
        inner_exp = np.exp(-inner)
        outer_exp = np.exp(-outer)
        rise = 1.0 - inner_exp
        shape = rise * outer_exp
        amplitude = self.amplitude_count_per_px[index]
        jacobian = np.zeros((*radius.shape, self.parameter_vector.size), dtype=np.float64)
        jacobian[..., 0] = (
            amplitude * outer_exp * inner_exp * (-self.inner_power * inner / self.inner_scale_px)
        )
        jacobian[..., 1] = (
            amplitude * outer_exp * inner_exp * inner * np.log(safe_radius / self.inner_scale_px)
        )
        jacobian[..., 2] = amplitude * shape * self.outer_power * outer / self.outer_scale_px
        jacobian[..., 3] = -amplitude * shape * outer * np.log(safe_radius / self.outer_scale_px)
        dataset_count = len(self.dataset_ids)
        flat_index = index.reshape(-1)
        flat_shape = shape.reshape(-1)
        flat_jacobian = jacobian.reshape(-1, jacobian.shape[-1])
        row = np.arange(flat_index.size)
        flat_jacobian[row, 4 + flat_index] = flat_shape
        flat_jacobian[row, 4 + dataset_count + flat_index] = 1.0
        return jacobian


@dataclass(frozen=True, slots=True)
class RadialBackgroundFitResult:
    state: RadialBackgroundState
    fitted_density_count_per_px: FloatArray
    residual_density_count_per_px: FloatArray
    training_rmse_count_per_px: float
    heldout_rmse_count_per_px: float
    heldout_mae_count_per_px: float
    heldout_bias_count_per_px: float
    success: bool
    optimizer_message: str
    function_evaluations: int


def fit_shared_radial_background(
    profiles: RadialBackgroundProfiles,
    *,
    inner_scale_bounds_px: tuple[float, float] = (30.0, 240.0),
    outer_scale_bounds_px: tuple[float, float] = (260.0, 1400.0),
    power_bounds: tuple[float, float] = (0.5, 4.0),
) -> RadialBackgroundFitResult:
    """Fit one frozen radial shape using only declared training sectors."""

    if not isinstance(profiles, RadialBackgroundProfiles):
        raise TypeError("profiles must be RadialBackgroundProfiles")
    for bounds, name in (
        (inner_scale_bounds_px, "inner_scale_bounds_px"),
        (outer_scale_bounds_px, "outer_scale_bounds_px"),
        (power_bounds, "power_bounds"),
    ):
        if (
            len(bounds) != 2
            or not all(math.isfinite(float(value)) for value in bounds)
            or float(bounds[0]) <= 0.0
            or float(bounds[0]) >= float(bounds[1])
        ):
            raise ValueError(f"{name} must contain two increasing positive values")
    if float(inner_scale_bounds_px[1]) >= float(outer_scale_bounds_px[0]):
        raise ValueError("inner and outer scale bounds must preserve rise-before-decay ordering")
    dataset_count = len(profiles.dataset_ids)
    observed_maximum = float(np.max(profiles.density_count_per_px))
    lower = np.concatenate(
        (
            (inner_scale_bounds_px[0], power_bounds[0], outer_scale_bounds_px[0], power_bounds[0]),
            np.zeros(2 * dataset_count),
        )
    )
    upper = np.concatenate(
        (
            (inner_scale_bounds_px[1], power_bounds[1], outer_scale_bounds_px[1], power_bounds[1]),
            np.full(dataset_count, max(4.0 * observed_maximum, 1.0)),
            np.full(dataset_count, max(2.0 * observed_maximum, 1.0)),
        )
    )
    amplitude = np.empty(dataset_count, dtype=np.float64)
    pedestal = np.empty(dataset_count, dtype=np.float64)
    for dataset_index in range(dataset_count):
        values = profiles.density_count_per_px[profiles.dataset_index == dataset_index]
        pedestal[dataset_index] = max(float(np.percentile(values, 5.0)), 0.0)
        amplitude[dataset_index] = max(float(np.max(values) - pedestal[dataset_index]), 1.0)
    initial = np.concatenate(
        (
            (103.0, 1.8, 427.0, 1.5),
            amplitude,
            pedestal,
        )
    )
    initial = np.clip(initial, lower + 1.0e-10, upper - 1.0e-10)
    training = np.asarray(profiles.is_training)
    support_weight = np.sqrt(
        np.minimum(profiles.support_px2, np.percentile(profiles.support_px2[training], 75.0))
    )
    support_weight /= np.median(support_weight[training])

    def density(parameters: FloatArray) -> FloatArray:
        state = RadialBackgroundState.from_parameter_vector(
            profiles.dataset_ids,
            parameters,
            parameter_covariance=np.zeros((parameters.size, parameters.size)),
        )
        return state.count_density(profiles.dataset_index, profiles.radius_px)

    def residual(parameters: FloatArray) -> FloatArray:
        return support_weight[training] * (
            density(parameters)[training] - profiles.density_count_per_px[training]
        )

    fitted = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        x_scale=np.maximum(np.abs(initial), 1.0),
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=2000,
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
    )
    degrees_of_freedom = max(int(np.count_nonzero(training)) - fitted.x.size, 1)
    raw_residual = residual(np.asarray(fitted.x, dtype=np.float64))
    residual_variance = float(raw_residual @ raw_residual) / degrees_of_freedom
    covariance = residual_variance * np.linalg.pinv(fitted.jac.T @ fitted.jac, rcond=1.0e-12)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    covariance = (eigenvector * np.maximum(eigenvalue, 0.0)) @ eigenvector.T
    state = RadialBackgroundState.from_parameter_vector(
        profiles.dataset_ids,
        fitted.x,
        parameter_covariance=covariance,
    )
    fitted_density = state.count_density(profiles.dataset_index, profiles.radius_px)
    residual_density = profiles.density_count_per_px - fitted_density
    heldout = ~training

    def rmse(selected: BoolArray) -> float:
        return float(np.sqrt(np.mean(residual_density[selected] ** 2)))

    for value in (fitted_density, residual_density):
        value.setflags(write=False)
    return RadialBackgroundFitResult(
        state=state,
        fitted_density_count_per_px=fitted_density,
        residual_density_count_per_px=residual_density,
        training_rmse_count_per_px=rmse(training),
        heldout_rmse_count_per_px=rmse(heldout),
        heldout_mae_count_per_px=float(np.mean(np.abs(residual_density[heldout]))),
        heldout_bias_count_per_px=float(np.mean(residual_density[heldout])),
        success=bool(fitted.success),
        optimizer_message=str(fitted.message),
        function_evaluations=int(fitted.nfev),
    )


__all__ = [
    "RadialBackgroundFitResult",
    "RadialBackgroundProfiles",
    "RadialBackgroundState",
    "fit_shared_radial_background",
]
