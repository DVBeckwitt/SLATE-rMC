"""Narrow validation helpers for the continuous reciprocal and Ewald core."""

from __future__ import annotations

from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def reject_complex(value: object, name: str) -> None:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) or (
        supplied.dtype.kind == "O" and any(np.iscomplexobj(item) for item in supplied.flat)
    ):
        raise ValueError(f"{name} must be real")


def finite_scalar(value: object, name: str) -> float:
    reject_complex(value, name)
    supplied = np.asarray(value)
    if supplied.shape != () or supplied.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite scalar")
    result = float(supplied)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def positive_integer(value: object, name: str) -> int:
    try:
        result = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if isinstance(value, (bool, np.bool_)) or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def integer(value: object, name: str) -> int:
    try:
        result = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    return result


def readonly_float_array(value: ArrayLike, shape: tuple[int | None, ...], name: str) -> FloatArray:
    reject_complex(value, name)
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has invalid shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def reciprocal_basis(value: ArrayLike) -> FloatArray:
    basis = readonly_float_array(value, (3, 3), "reciprocal_basis_Ainv")
    if np.linalg.matrix_rank(basis) != 3:
        raise ValueError("reciprocal_basis_Ainv must be nonsingular")
    return basis


def proper_rotation(value: ArrayLike, name: str = "crystal_to_sample") -> FloatArray:
    rotation = readonly_float_array(value, (3, 3), name)
    tolerance = 128.0 * np.finfo(np.float64).eps
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0.0, atol=tolerance
    ) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} must be a proper orthogonal rotation")
    return rotation
