"""Scalar optical attenuation and total local-field intensity weight."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mode_decay_constant(
    kz_Ainv: Any,
    propagation_direction: ArrayLike,
) -> float | NDArray[np.float64]:
    """Return nonnegative field decay along each mode's signed propagation direction."""

    mode, direction = np.broadcast_arrays(
        np.asarray(kz_Ainv, dtype=np.complex128),
        np.asarray(propagation_direction),
    )
    if not np.all(np.isfinite(mode)):
        raise ValueError("kz_Ainv must be finite")
    if not np.all((direction == -1) | (direction == 1)):
        raise ValueError("propagation_direction must be +1 or -1")
    decay = direction.astype(np.float64) * mode.imag
    tolerance = 16.0 * np.finfo(np.float64).eps * np.maximum(np.abs(mode), 1.0)
    if np.any(decay < -tolerance):
        raise ValueError("kz_Ainv grows along the declared propagation direction")
    return _result(np.maximum(decay, 0.0))


def _nonnegative(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return array


def _result(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        return float(result)
    result.setflags(write=False)
    return result


def uniform_depth_attenuation(
    kappa_i_Ainv: ArrayLike,
    kappa_f_Ainv: ArrayLike,
    thickness_A: ArrayLike,
) -> float | NDArray[np.float64]:
    """Return the uniform-depth average ``(1-exp(-x))/x`` with ``x=2*(ki+kf)*t``."""

    kappa_i, kappa_f, thickness = np.broadcast_arrays(
        _nonnegative(kappa_i_Ainv, "kappa_i_Ainv"),
        _nonnegative(kappa_f_Ainv, "kappa_f_Ainv"),
        _nonnegative(thickness_A, "thickness_A"),
    )
    exponent = 2.0 * (kappa_i + kappa_f) * thickness
    attenuation = np.ones_like(exponent)
    np.divide(
        -np.expm1(-exponent),
        exponent,
        out=attenuation,
        where=exponent != 0.0,
    )
    return _result(attenuation)


def path_attenuation(
    kappa_i_Ainv: ArrayLike,
    kappa_f_Ainv: ArrayLike,
    incident_path_A: ArrayLike,
    exit_path_A: ArrayLike,
) -> float | NDArray[np.float64]:
    """Return explicit-path intensity damping; no path is inferred from film thickness."""

    kappa_i, kappa_f, incident_path, exit_path = np.broadcast_arrays(
        _nonnegative(kappa_i_Ainv, "kappa_i_Ainv"),
        _nonnegative(kappa_f_Ainv, "kappa_f_Ainv"),
        _nonnegative(incident_path_A, "incident_path_A"),
        _nonnegative(exit_path_A, "exit_path_A"),
    )
    attenuation = np.exp(-2.0 * kappa_i * incident_path - 2.0 * kappa_f * exit_path)
    return _result(attenuation)


def _complex(value: Any, name: str) -> NDArray[np.complex128]:
    array = np.asarray(value, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def scalar_optical_weight(
    entrance_amplitude: Any,
    exit_amplitude: Any,
    attenuation_weight: ArrayLike,
) -> float | NDArray[np.float64]:
    """Return ``|t_in*t_out|^2`` times one declared attenuation factor."""

    entrance, exit_mode, attenuation = np.broadcast_arrays(
        _complex(entrance_amplitude, "entrance_amplitude"),
        _complex(exit_amplitude, "exit_amplitude"),
        _nonnegative(attenuation_weight, "attenuation_weight"),
    )
    weight = np.asarray(np.abs(entrance * exit_mode) ** 2 * attenuation, dtype=np.float64)
    if not np.all(np.isfinite(weight)):
        raise ValueError("scalar optical weight is not finite")
    return _result(weight)
