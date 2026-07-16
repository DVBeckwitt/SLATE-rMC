"""Finite float-vector validation shared by geometry kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def finite_vector3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    array.setflags(write=False)
    return array


def finite_vectors3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape (N, 3) with finite values")
    return array
