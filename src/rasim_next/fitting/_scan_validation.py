"""Array normalization helpers shared by scan integration and acceptance."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def float_array(value: ArrayLike, name: str, *, ndim: int) -> FloatArray:
    supplied = np.asarray(value)
    if supplied.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real")
    result = np.array(supplied, dtype=np.float64, copy=True, order="C")
    if result.ndim != ndim or not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite nonempty {ndim}-dimensional array")
    result.setflags(write=False)
    return result


def int_array(value: ArrayLike, name: str, *, ndim: int) -> IntArray:
    supplied = np.asarray(value)
    if supplied.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    limits = np.iinfo(np.int64)
    if np.any(supplied < limits.min) or np.any(supplied > limits.max):
        raise ValueError(f"{name} values are outside the int64 range")
    result = np.array(supplied, dtype=np.int64, copy=True, order="C")
    if result.ndim != ndim or not result.size:
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    result.setflags(write=False)
    return result


def bool_array(value: ArrayLike, name: str, *, ndim: int) -> BoolArray:
    supplied = np.asarray(value)
    if supplied.dtype.kind != "b":
        raise ValueError(f"{name} must contain booleans")
    result = np.array(supplied, dtype=np.bool_, copy=True, order="C")
    if result.ndim != ndim or not result.size:
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    result.setflags(write=False)
    return result
