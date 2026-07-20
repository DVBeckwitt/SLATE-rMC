"""Conservative equal-solid-angle rasterization of painted Ewald points."""

from __future__ import annotations

import numpy as np

from painted_ewald.types import PaintedPoint, RasterParameters, SphereTexture
from painted_ewald.validation import finite_scalar


def _equal_solid_angle_edges(
    parameters: RasterParameters,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.linspace(-1.0, 1.0, parameters.mu_bin_count + 1, dtype=np.float64),
        np.linspace(
            0.0,
            2.0 * np.pi,
            parameters.phi_bin_count + 1,
            dtype=np.float64,
        ),
    )


def _rasterize_arrays_equal_solid_angle(
    *,
    kf_sample_Ainv: np.ndarray,
    weights: np.ndarray,
    radius_Ainv: float,
    parameters: RasterParameters,
) -> np.ndarray:
    """Return conservative bin mass for validated final-wavevector arrays."""

    shape = (parameters.mu_bin_count, parameters.phi_bin_count)
    if weights.size == 0:
        return np.zeros(shape, dtype=np.float64)
    if (
        kf_sample_Ainv.shape != (weights.size, 3)
        or not np.all(np.isfinite(kf_sample_Ainv))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("final wavevectors and weights must be finite, aligned, and nonnegative")
    mu_edges, phi_edges = _equal_solid_angle_edges(parameters)
    mu = np.clip(kf_sample_Ainv[:, 2] / radius_Ainv, -1.0, 1.0)
    phi = np.remainder(
        np.arctan2(kf_sample_Ainv[:, 1], kf_sample_Ainv[:, 0]),
        2.0 * np.pi,
    )
    mu_index = np.searchsorted(mu_edges, mu, side="right") - 1
    phi_index = np.searchsorted(phi_edges, phi, side="right") - 1
    np.clip(mu_index, 0, parameters.mu_bin_count - 1, out=mu_index)
    np.clip(phi_index, 0, parameters.phi_bin_count - 1, out=phi_index)
    linear_index = mu_index * parameters.phi_bin_count + phi_index
    order = np.argsort(linear_index, kind="stable")
    sorted_index = linear_index[order]
    starts = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.flatnonzero(np.diff(sorted_index)) + 1)
    )
    mass = np.zeros(parameters.mu_bin_count * parameters.phi_bin_count, dtype=np.float64)
    mass[sorted_index[starts]] = np.add.reduceat(weights[order], starts)
    mass = mass.reshape(shape)
    point_total = float(np.sum(weights, dtype=np.float64))
    texture_total = float(np.sum(mass, dtype=np.float64))
    tolerance = 1024.0 * np.finfo(np.float64).eps * max(point_total, 1.0)
    if abs(texture_total - point_total) > tolerance:
        raise FloatingPointError("equal-solid-angle rasterization did not conserve weight")
    return mass


def _sphere_texture_from_mass(*, mass: np.ndarray, parameters: RasterParameters) -> SphereTexture:
    mu_edges, phi_edges = _equal_solid_angle_edges(parameters)
    solid_angle = np.diff(mu_edges)[:, None] * np.diff(phi_edges)[None, :]
    return SphereTexture(
        mu_edges=mu_edges,
        phi_edges_rad=phi_edges,
        mass=mass,
        density_per_sr=mass / solid_angle,
    )


def rasterize_equal_solid_angle(
    *,
    points: tuple[PaintedPoint, ...],
    radius_Ainv: float,
    parameters: RasterParameters,
) -> SphereTexture:
    """Accumulate each complete point weight once in uniform ``mu``/``phi`` bins."""

    radius = finite_scalar(radius_Ainv, "radius_Ainv")
    if radius <= 0.0:
        raise ValueError("radius_Ainv must be positive")
    if not isinstance(parameters, RasterParameters):
        raise TypeError("parameters must be RasterParameters")
    if not all(isinstance(point, PaintedPoint) for point in points):
        raise TypeError("points must contain only PaintedPoint values")

    shape = (parameters.mu_bin_count, parameters.phi_bin_count)
    if not points:
        mass = np.zeros(shape, dtype=np.float64)
    else:
        kf_sample = np.stack([point.kf_sample_Ainv for point in points])
        weights = np.fromiter(
            (point.weight for point in points), dtype=np.float64, count=len(points)
        )
        mass = _rasterize_arrays_equal_solid_angle(
            kf_sample_Ainv=kf_sample,
            weights=weights,
            radius_Ainv=radius,
            parameters=parameters,
        )
    return _sphere_texture_from_mass(
        mass=mass,
        parameters=parameters,
    )


__all__ = ["rasterize_equal_solid_angle"]
