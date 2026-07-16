"""Seeded equal-mass Gaussian source-ray sampling."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

from rasim_next.core.contracts import IncidentSampleBatch

_DIMENSION_COUNT = 5
_UNIT_CLIP_EPS = np.finfo(np.float64).eps
_CORRELATION_MODEL = "independent_gaussian_lhs.v1"


def _finite_real(value: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) or (
        supplied.dtype.kind == "O" and any(np.iscomplexobj(item) for item in supplied.flat)
    ):
        raise ValueError(f"{name} must be real")
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite real array with shape {shape}")
    return array


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
) -> IncidentSampleBatch:
    """Sample five independent Gaussian dimensions with empirical mass ``1/N``."""

    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, (int, np.integer))
        or sample_count <= 0
    ):
        raise ValueError("sample_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(polarization_state_id, str) or not polarization_state_id:
        raise ValueError("polarization_state_id must be a nonempty string")

    mean_origin = _finite_real(mean_origin_lab_m, (3,), "mean_origin_lab_m")
    mean_direction = _finite_real(mean_direction_lab, (3,), "mean_direction_lab")
    axes = _finite_real(transverse_axes_lab, (2, 3), "transverse_axes_lab")
    spatial_sigma = _finite_real(spatial_sigma_m, (2,), "spatial_sigma_m")
    divergence_sigma = _finite_real(divergence_sigma_rad, (2,), "divergence_sigma_rad")
    mean_wavelength = float(_finite_real(mean_wavelength_A, (), "mean_wavelength_A"))
    wavelength_sigma = float(_finite_real(wavelength_sigma_A, (), "wavelength_sigma_A"))
    if np.any(spatial_sigma < 0.0) or np.any(divergence_sigma < 0.0) or wavelength_sigma < 0.0:
        raise ValueError("source standard deviations must be nonnegative")
    if mean_wavelength <= 0.0:
        raise ValueError("mean_wavelength_A must be positive")
    if not np.isclose(np.linalg.norm(mean_direction), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("mean_direction_lab must be unit length")
    if not np.allclose(axes @ axes.T, np.eye(2), rtol=0.0, atol=1.0e-12) or not np.allclose(
        axes @ mean_direction, 0.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("transverse_axes_lab must be orthonormal and tangent")

    size = int(sample_count)
    pair_count = size // 2
    unit = np.empty((size, _DIMENSION_COUNT), dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    if pair_count:
        for dimension in range(_DIMENSION_COUNT):
            lhs = (generator.permutation(pair_count) + generator.random(pair_count)) / pair_count
            unit[: 2 * pair_count : 2, dimension] = lhs
            unit[1 : 2 * pair_count : 2, dimension] = 1.0 - lhs
    if size % 2:
        unit[-1] = 0.5
    gaussian = ndtri(np.clip(unit, _UNIT_CLIP_EPS, 1.0 - _UNIT_CLIP_EPS))

    origin = mean_origin + (gaussian[:, :2] * spatial_sigma) @ axes
    tangent = (gaussian[:, 2:4] * divergence_sigma) @ axes
    radius = np.linalg.norm(tangent, axis=1)
    sine_scale = np.divide(np.sin(radius), radius, out=np.ones_like(radius), where=radius != 0.0)
    direction = np.cos(radius)[:, None] * mean_direction + sine_scale[:, None] * tangent
    wavelength = mean_wavelength + wavelength_sigma * gaussian[:, 4]
    if np.any(wavelength <= 0.0):
        raise ValueError("sampled wavelengths must be positive")

    return IncidentSampleBatch(
        incident_sample_id=np.arange(size, dtype=np.int64),
        origin_lab_m=origin,
        direction_lab=direction,
        wavelength_A=wavelength,
        source_weight=np.full(size, 1.0 / size),
        polarization_state_id=(polarization_state_id,) * size,
        correlation_model=_CORRELATION_MODEL,
    )
