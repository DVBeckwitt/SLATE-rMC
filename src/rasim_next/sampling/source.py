"""Seeded equal-mass Gaussian source-ray sampling."""

from __future__ import annotations

import json

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

from rasim_next.core.contracts import IncidentSampleBatch

_DIMENSION_COUNT = 5
_SAMPLING_MODEL_ID = "independent_gaussian_antithetic_lhs.v2"
_RNG_MODEL_ID = "numpy_pcg64.v1"


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
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or seed < 0
        or seed > 2**64 - 1
    ):
        raise ValueError("seed must be a nonnegative unsigned 64-bit integer")
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
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    if pair_count:
        for dimension in range(_DIMENSION_COUNT):
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
    gaussian = ndtri(unit)

    origin = mean_origin + (gaussian[:, :2] * spatial_sigma) @ axes
    tangent = (gaussian[:, 2:4] * divergence_sigma) @ axes
    radius = np.linalg.norm(tangent, axis=1)
    sine_scale = np.divide(np.sin(radius), radius, out=np.ones_like(radius), where=radius != 0.0)
    direction = np.cos(radius)[:, None] * mean_direction + sine_scale[:, None] * tangent
    wavelength = mean_wavelength + wavelength_sigma * gaussian[:, 4]
    if np.any(wavelength <= 0.0):
        raise ValueError("sampled wavelengths must be positive")

    incident_sample_id = np.arange(size, dtype=np.int64)
    source_weight = np.full(size, 1.0 / size)
    polarization_ids = (polarization_state_id,) * size
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
    parameter_provenance = json.dumps(
        {
            "frames": {
                "mean_direction_lab": "lab",
                "mean_origin_lab_m": "lab",
                "transverse_axes_lab": "lab",
            },
            "units": {
                "divergence_sigma_rad": "rad",
                "mean_direction_lab": "1",
                "mean_origin_lab_m": "m",
                "mean_wavelength_A": "angstrom",
                "polarization_state_id": "id",
                "sample_count": "1",
                "spatial_sigma_m": "m",
                "transverse_axes_lab": "1",
                "wavelength_sigma_A": "angstrom",
            },
            "values": parameter_values,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return IncidentSampleBatch(
        incident_sample_id=incident_sample_id,
        origin_lab_m=origin,
        direction_lab=direction,
        wavelength_A=wavelength,
        source_weight=source_weight,
        polarization_state_id=polarization_ids,
        source_sampling_model_id=_SAMPLING_MODEL_ID,
        source_rng_model_id=_RNG_MODEL_ID,
        source_seed=int(seed),
        source_parameter_provenance=parameter_provenance,
    )
