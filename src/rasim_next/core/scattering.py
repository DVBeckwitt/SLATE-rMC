from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

CLASSICAL_ELECTRON_RADIUS_A = 2.8179403262e-5


def electron_squared_to_intensity_per_sr(raw_electron_squared: ArrayLike) -> NDArray[np.float64]:
    """Return polarization-neutral ``r_e**2 * electron**2`` in angstrom²/sr."""

    supplied = np.asarray(raw_electron_squared)
    if supplied.dtype.kind not in "fiu":
        raise TypeError("raw_electron_squared must be a real numeric array")
    values = np.array(raw_electron_squared, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("raw_electron_squared must be finite and nonnegative")
    values *= CLASSICAL_ELECTRON_RADIUS_A**2
    values.setflags(write=False)
    return values
