
"""Random mosaic block profile generation for diffraction simulations."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri


_PROFILE_DIMENSIONS = 5
_UNIT_CLIP_EPS = np.finfo(np.float64).eps
RANDOM_GAUSSIAN_SAMPLING = "random_gaussian"


def generate_random_profiles(
    num_samples,
    divergence_sigma,
    bw_sigma,
    lambda0,
    bandwidth,
    *,
    rng: np.random.Generator | int | None = None,
):
    """Generate low-discrepancy Gaussian beam profiles with antithetic pairing."""

    sample_count = max(int(num_samples), 0)
    if sample_count == 0:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty, empty, empty, empty

    rng_obj = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    pair_count = sample_count // 2

    if pair_count > 0:
        base_unit = np.empty((pair_count, _PROFILE_DIMENSIONS), dtype=np.float64)
        for axis in range(_PROFILE_DIMENSIONS):
            perm = rng_obj.permutation(pair_count).astype(np.float64)
            base_unit[:, axis] = (perm + rng_obj.random(pair_count)) / float(pair_count)
        anti_unit = 1.0 - base_unit
        unit = np.empty((sample_count, _PROFILE_DIMENSIONS), dtype=np.float64)
        unit[0 : 2 * pair_count : 2, :] = base_unit
        unit[1 : 2 * pair_count : 2, :] = anti_unit
    else:
        unit = np.empty((sample_count, _PROFILE_DIMENSIONS), dtype=np.float64)

    if sample_count % 2 == 1:
        unit[-1, :] = 0.5

    gaussian = ndtri(np.clip(unit, _UNIT_CLIP_EPS, 1.0 - _UNIT_CLIP_EPS))

    theta_array = divergence_sigma * gaussian[:, 0]
    phi_array = divergence_sigma * gaussian[:, 1]
    beam_x_array = bw_sigma * gaussian[:, 2]
    beam_y_array = bw_sigma * gaussian[:, 3]
    wavelength_array = lambda0 + (lambda0 * bandwidth) * gaussian[:, 4]

    return (
        np.asarray(beam_x_array, dtype=np.float64),
        np.asarray(beam_y_array, dtype=np.float64),
        np.asarray(theta_array, dtype=np.float64),
        np.asarray(phi_array, dtype=np.float64),
        np.asarray(wavelength_array, dtype=np.float64),
    )
