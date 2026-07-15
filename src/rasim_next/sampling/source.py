"""Deterministic source samples with explicit correlation semantics."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import ArrayLike

from rasim_next.core.contracts import IncidentSampleBatch

UNITY_SCALAR_POLARIZATION = "unity_scalar"
MAX_GENERATED_SOURCE_ROWS = 262_144


def _reject_complex(value: ArrayLike, name: str) -> None:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) or (
        supplied.dtype.kind == "O" and any(np.iscomplexobj(item) for item in supplied.flat)
    ):
        raise ValueError(f"{name} must be real")


def _finite_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    _reject_complex(value, name)
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array


def _normal_rule(standard_deviation: float, order: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 1 <= order <= 100:
        raise ValueError(f"{name}_order must be an integer from 1 through 100")
    _reject_complex(standard_deviation, f"{name} standard deviation")
    standard_deviation = float(standard_deviation)
    if not np.isfinite(standard_deviation) or standard_deviation < 0.0:
        raise ValueError(f"{name} standard deviation must be finite and nonnegative")
    if standard_deviation == 0.0:
        return np.zeros(1), np.ones(1)
    nodes, weights = hermgauss(int(order))
    return np.sqrt(2.0) * standard_deviation * nodes, weights / np.sqrt(np.pi)


def _normal_product(
    standard_deviation: np.ndarray, order: int, name: str
) -> tuple[np.ndarray, np.ndarray]:
    first, first_mass = _normal_rule(standard_deviation[0], order, name)
    second, second_mass = _normal_rule(standard_deviation[1], order, name)
    coordinates = np.column_stack((np.repeat(first, second.size), np.tile(second, first.size)))
    mass = np.repeat(first_mass, second.size) * np.tile(second_mass, first.size)
    return coordinates, mass


def compile_joint_source_samples(
    *,
    origin_lab_m: ArrayLike,
    direction_lab: ArrayLike,
    wavelength_A: ArrayLike,
    probability_mass: ArrayLike,
    polarization_state_id: tuple[str, ...],
) -> IncidentSampleBatch:
    """Preserve explicitly correlated source rows as one immutable batch."""

    polarization_ids = tuple(polarization_state_id)
    if not polarization_ids or any(
        state_id != UNITY_SCALAR_POLARIZATION for state_id in polarization_ids
    ):
        raise ValueError('polarization_state_id must explicitly contain only "unity_scalar"')
    for name, value in (
        ("origin_lab_m", origin_lab_m),
        ("direction_lab", direction_lab),
        ("wavelength_A", wavelength_A),
        ("probability_mass", probability_mass),
    ):
        _reject_complex(value, name)
    return IncidentSampleBatch(
        incident_sample_id=np.arange(len(polarization_ids), dtype=np.int64),
        origin_lab_m=origin_lab_m,
        direction_lab=direction_lab,
        wavelength_A=wavelength_A,
        source_weight=probability_mass,
        polarization_state_id=polarization_ids,
        correlation_model="explicit_joint",
    )


def _component_mass(value: ArrayLike, size: int, name: str) -> np.ndarray:
    _reject_complex(value, name)
    mass = np.array(value, dtype=np.float64, copy=True)
    if mass.shape != (size,) or not np.all(np.isfinite(mass)) or np.any(mass < 0.0):
        raise ValueError(f"{name} must be a finite nonnegative array with shape {(size,)}")
    if not np.isclose(mass.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must sum to one")
    return mass


def compile_independent_source_samples(
    *,
    origin_lab_m: ArrayLike,
    origin_probability_mass: ArrayLike,
    direction_lab: ArrayLike,
    direction_probability_mass: ArrayLike,
    wavelength_A: ArrayLike,
    wavelength_probability_mass: ArrayLike,
    polarization_state_id: str,
) -> IncidentSampleBatch:
    """Form the declared independent source product in lexicographic order."""

    _reject_complex(origin_lab_m, "origin_lab_m")
    _reject_complex(direction_lab, "direction_lab")
    _reject_complex(wavelength_A, "wavelength_A")
    origins = np.array(origin_lab_m, dtype=np.float64, copy=True)
    directions = np.array(direction_lab, dtype=np.float64, copy=True)
    wavelengths = np.array(wavelength_A, dtype=np.float64, copy=True)
    if origins.ndim != 2 or origins.shape[1:] != (3,) or not np.all(np.isfinite(origins)):
        raise ValueError("origin_lab_m must be a finite array with shape (N, 3)")
    if directions.ndim != 2 or directions.shape[1:] != (3,) or not np.all(np.isfinite(directions)):
        raise ValueError("direction_lab must be a finite array with shape (N, 3)")
    if wavelengths.ndim != 1 or not np.all(np.isfinite(wavelengths)):
        raise ValueError("wavelength_A must be a finite one-dimensional array")
    if polarization_state_id != UNITY_SCALAR_POLARIZATION:
        raise ValueError('polarization_state_id must be "unity_scalar"')
    origin_mass = _component_mass(
        origin_probability_mass, origins.shape[0], "origin_probability_mass"
    )
    direction_mass = _component_mass(
        direction_probability_mass, directions.shape[0], "direction_probability_mass"
    )
    wavelength_mass = _component_mass(
        wavelength_probability_mass, wavelengths.size, "wavelength_probability_mass"
    )
    origin_size, direction_size, wavelength_size = (
        origins.shape[0],
        directions.shape[0],
        wavelengths.size,
    )
    size = origin_size * direction_size * wavelength_size
    if size > MAX_GENERATED_SOURCE_ROWS:
        raise ValueError(
            f"requested {size} source rows exceeds allowed {MAX_GENERATED_SOURCE_ROWS} rows"
        )
    probability_mass = (
        np.repeat(origin_mass, direction_size * wavelength_size)
        * np.tile(np.repeat(direction_mass, wavelength_size), origin_size)
        * np.tile(wavelength_mass, origin_size * direction_size)
    )
    return IncidentSampleBatch(
        incident_sample_id=np.arange(size, dtype=np.int64),
        origin_lab_m=np.repeat(origins, direction_size * wavelength_size, axis=0),
        direction_lab=np.tile(np.repeat(directions, wavelength_size, axis=0), (origin_size, 1)),
        wavelength_A=np.tile(wavelengths, origin_size * direction_size),
        source_weight=probability_mass,
        polarization_state_id=(UNITY_SCALAR_POLARIZATION,) * size,
        correlation_model="independent_product",
    )


def compile_independent_gaussian_source_samples(
    *,
    mean_origin_lab_m: ArrayLike,
    mean_direction_lab: ArrayLike,
    transverse_axes_lab: ArrayLike,
    spatial_sigma_m: ArrayLike,
    divergence_sigma_rad: ArrayLike,
    mean_wavelength_A: float,
    wavelength_sigma_A: float,
    spatial_order: int,
    direction_order: int,
    wavelength_order: int,
    polarization_state_id: str,
) -> IncidentSampleBatch:
    """Compile an explicitly independent deterministic Gaussian source product."""

    mean_origin = _finite_array(mean_origin_lab_m, (3,), "mean_origin_lab_m")
    mean_direction = _finite_array(mean_direction_lab, (3,), "mean_direction_lab")
    axes = _finite_array(transverse_axes_lab, (2, 3), "transverse_axes_lab")
    spatial_sigma = _finite_array(spatial_sigma_m, (2,), "spatial_sigma_m")
    divergence_sigma = _finite_array(divergence_sigma_rad, (2,), "divergence_sigma_rad")
    if np.any(spatial_sigma < 0.0) or np.any(divergence_sigma < 0.0):
        raise ValueError("source standard deviations must be nonnegative")
    if not np.isclose(np.linalg.norm(mean_direction), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("mean_direction_lab must be unit length")
    if not np.allclose(axes @ axes.T, np.eye(2), rtol=0.0, atol=1.0e-12) or not np.allclose(
        axes @ mean_direction, 0.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "transverse_axes_lab must be orthonormal and tangent to mean_direction_lab"
        )
    if polarization_state_id != UNITY_SCALAR_POLARIZATION:
        raise ValueError('polarization_state_id must be "unity_scalar"')
    _reject_complex(mean_wavelength_A, "mean_wavelength_A")
    mean_wavelength = float(mean_wavelength_A)
    if not np.isfinite(mean_wavelength) or mean_wavelength <= 0.0:
        raise ValueError("mean_wavelength_A must be positive")

    spatial_offset, spatial_mass = _normal_product(spatial_sigma, spatial_order, "spatial")
    angular_offset, direction_mass = _normal_product(divergence_sigma, direction_order, "direction")
    wavelength_offset, wavelength_mass = _normal_rule(
        wavelength_sigma_A, wavelength_order, "wavelength"
    )
    spatial_nodes = mean_origin + spatial_offset @ axes
    tangent = angular_offset @ axes
    angular_radius = np.linalg.norm(angular_offset, axis=1)
    sine_scale = np.divide(
        np.sin(angular_radius),
        angular_radius,
        out=np.ones_like(angular_radius),
        where=angular_radius != 0.0,
    )
    direction_nodes = (
        np.cos(angular_radius)[:, None] * mean_direction + sine_scale[:, None] * tangent
    )
    wavelength_nodes = mean_wavelength + wavelength_offset
    if np.any(wavelength_nodes <= 0.0):
        raise ValueError("wavelength quadrature nodes must be positive")

    return compile_independent_source_samples(
        origin_lab_m=spatial_nodes,
        origin_probability_mass=spatial_mass,
        direction_lab=direction_nodes,
        direction_probability_mass=direction_mass,
        wavelength_A=wavelength_nodes,
        wavelength_probability_mass=wavelength_mass,
        polarization_state_id=polarization_state_id,
    )
