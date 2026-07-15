"""Shared complex normal-wavevector branch selection.

The core uses fields proportional to ``exp(i k·r - i omega t)``. A wave travelling in the
requested normal direction therefore has the same sign for its real normal component and a
non-growing evanescent component in that direction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]


def select_normal_wavevector(
    squared_normal_wavevector: ArrayLike, propagation_direction: int
) -> complex | ComplexArray:
    """Select the propagating or non-growing square-root branch for direction ``+1`` or ``-1``."""

    if propagation_direction not in (-1, 1):
        raise ValueError("propagation_direction must be +1 or -1")
    radicand = np.asarray(squared_normal_wavevector, dtype=np.complex128)
    if not np.all(np.isfinite(radicand)):
        raise ValueError("squared_normal_wavevector must be finite")
    root = np.sqrt(radicand)
    has_decay = root.imag != 0.0
    wrong_decay = propagation_direction * root.imag < 0.0
    wrong_phase = propagation_direction * root.real < 0.0
    selected = np.where(np.where(has_decay, wrong_decay, wrong_phase), -root, root)
    if selected.ndim == 0:
        return complex(selected)
    return np.asarray(selected, dtype=np.complex128)


def normal_wavevector(
    *,
    k0_Ainv: ArrayLike,
    refractive_index: ArrayLike,
    k_parallel_Ainv: ArrayLike,
    propagation_direction: int,
) -> complex | ComplexArray:
    """Return ``sqrt((n*k0)^2 - k_parallel·k_parallel)`` on the shared branch."""

    k0 = np.asarray(k0_Ainv, dtype=np.float64)
    index = np.asarray(refractive_index, dtype=np.complex128)
    if not np.all(np.isfinite(k0)) or np.any(k0 <= 0.0):
        raise ValueError("k0_Ainv must be finite and positive")
    if not np.all(np.isfinite(index)):
        raise ValueError("refractive_index must be finite")
    supplied_parallel = np.asarray(k_parallel_Ainv)
    if np.iscomplexobj(supplied_parallel) and np.any(supplied_parallel.imag != 0.0):
        raise ValueError("k_parallel_Ainv must contain real phase vectors")
    parallel = np.asarray(supplied_parallel.real, dtype=np.float64)
    if parallel.ndim == 0 or parallel.shape[-1] not in (2, 3):
        raise ValueError("k_parallel_Ainv must end with a length-2 or length-3 vector axis")
    if not np.all(np.isfinite(parallel)):
        raise ValueError("k_parallel_Ainv must be finite")
    squared_parallel = np.sum(parallel * parallel, axis=-1)
    squared_normal = (index * k0) ** 2 - squared_parallel
    return select_normal_wavevector(squared_normal, propagation_direction)
