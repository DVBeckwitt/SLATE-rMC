"""Scalar field-amplitude primitive shared by optics and reflectivity."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def scalar_interface_amplitude(
    k1z_Ainv: ArrayLike, k2z_Ainv: ArrayLike
) -> complex | NDArray[np.complex128]:
    """Return the scalar field amplitude ``2*k1z/(k1z+k2z)``."""

    k1z = np.asarray(k1z_Ainv, dtype=np.complex128)
    k2z = np.asarray(k2z_Ainv, dtype=np.complex128)
    if not np.all(np.isfinite(k1z)) or not np.all(np.isfinite(k2z)):
        raise ValueError("normal wavevectors must be finite")
    denominator = k1z + k2z
    if np.any(denominator == 0.0):
        raise ZeroDivisionError("scalar interface amplitude is undefined when k1z + k2z is zero")
    amplitude = 2.0 * k1z / denominator
    if amplitude.ndim == 0:
        return complex(amplitude)
    return np.asarray(amplitude, dtype=np.complex128)
