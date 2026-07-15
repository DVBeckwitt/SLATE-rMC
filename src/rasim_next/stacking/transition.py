"""Typed state, phase, and transition conventions for finite stacking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class StackingState(StrEnum):
    REGISTRY_0_PLUS = "0F+"
    REGISTRY_1_PLUS = "1F+"
    REGISTRY_2_PLUS = "2F+"
    REGISTRY_0_MINUS = "0F-"
    REGISTRY_1_MINUS = "1F-"
    REGISTRY_2_MINUS = "2F-"


STATE_ORDER = tuple(StackingState)

_REGISTRY_ROOTS = np.array(
    [
        1.0 + 0.0j,
        -0.5 + 0.5j * np.sqrt(3.0),
        -0.5 - 0.5j * np.sqrt(3.0),
    ],
    dtype=np.complex128,
)
_REGISTRY_ROOTS.setflags(write=False)


class RegistryPhaseModel(StrEnum):
    """Fixed phase models; arbitrary evaluated expressions are not accepted."""

    FORWARD_H_PLUS_2K = "exp[2pi*i*(h+2k)/3]"
    LEGACY_2H_PLUS_K = "exp[2pi*i*(2h+k)/3]"


class Parent(StrEnum):
    TWO_H = "2H"
    FOUR_H_PLUS = "4H+"
    FOUR_H_MINUS = "4H-"
    SIX_H_PLUS = "6H+"
    SIX_H_MINUS = "6H-"


def _probability(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return result


@dataclass(frozen=True, slots=True)
class TransitionLaw:
    """Five mutually exclusive row-current, column-next interface events."""

    a: float
    b_plus: float
    b_minus: float
    d_plus: float
    d_minus: float

    def __post_init__(self) -> None:
        for name in ("a", "b_plus", "b_minus", "d_plus", "d_minus"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if not np.isclose(self.as_array().sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("transition probabilities must sum to one")

    def as_array(self) -> NDArray[np.float64]:
        result = np.array(
            [self.a, self.b_plus, self.b_minus, self.d_plus, self.d_minus],
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    @classmethod
    def from_array(cls, values: ArrayLike) -> TransitionLaw:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (5,):
            raise ValueError("transition law must have shape (5,)")
        return cls(*map(float, array))

    @classmethod
    def for_parent(cls, parent: Parent) -> TransitionLaw:
        selected = {
            Parent.TWO_H: 0,
            Parent.SIX_H_PLUS: 1,
            Parent.SIX_H_MINUS: 2,
            Parent.FOUR_H_PLUS: 3,
            Parent.FOUR_H_MINUS: 4,
        }[Parent(parent)]
        values = np.zeros(5, dtype=np.float64)
        values[selected] = 1.0
        return cls.from_array(values)


@dataclass(frozen=True, slots=True)
class InitialPopulation:
    """Orientation probabilities at registry zero for the first layer."""

    plus: float
    minus: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "plus", _probability(self.plus, "plus"))
        object.__setattr__(self, "minus", _probability(self.minus, "minus"))
        if not np.isclose(self.plus + self.minus, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("initial orientation probabilities must sum to one")

    def as_array(self) -> NDArray[np.float64]:
        result = np.array([self.plus, self.minus], dtype=np.float64)
        result.setflags(write=False)
        return result

    @classmethod
    def plus_only(cls) -> InitialPopulation:
        return cls(1.0, 0.0)


def registry_phase(
    h: ArrayLike,
    k: ArrayLike,
    model: RegistryPhaseModel = RegistryPhaseModel.FORWARD_H_PLUS_2K,
) -> NDArray[np.complex128] | np.complex128:
    """Return registry phase with positive Fourier sign."""

    h_array = np.asarray(h)
    k_array = np.asarray(k)
    if h_array.dtype.kind not in "iu" or k_array.dtype.kind not in "iu":
        raise ValueError("Miller h and k must contain integers")
    h_array, k_array = np.broadcast_arrays(h_array, k_array)
    phase_model = RegistryPhaseModel(model)
    h_mod = np.remainder(h_array, 3).astype(np.int8, copy=False)
    k_mod = np.remainder(k_array, 3).astype(np.int8, copy=False)
    index = np.remainder(
        h_mod + 2 * k_mod
        if phase_model is RegistryPhaseModel.FORWARD_H_PLUS_2K
        else 2 * h_mod + k_mod,
        3,
    )
    result = np.array(_REGISTRY_ROOTS[index], dtype=np.complex128, copy=True)
    if result.ndim == 0:
        return np.complex128(result)
    result.setflags(write=False)
    return result


def full_transition_matrix(law: TransitionLaw) -> NDArray[np.float64]:
    """Return manuscript T6 with rows current and columns next."""

    shift_plus = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    shift_minus = shift_plus.T
    identity = np.eye(3)
    same = law.a * identity + law.b_plus * shift_plus + law.b_minus * shift_minus
    plus_to_minus = law.d_plus * shift_plus + law.d_minus * shift_minus
    minus_to_plus = law.d_plus * shift_minus + law.d_minus * shift_plus
    result = np.block([[same, plus_to_minus], [minus_to_plus, same]])
    result.setflags(write=False)
    return result


def orientation_transition_matrix(law: TransitionLaw) -> NDArray[np.float64]:
    same = law.a + law.b_plus + law.b_minus
    flip = law.d_plus + law.d_minus
    result = np.array([[same, flip], [flip, same]], dtype=np.float64)
    result.setflags(write=False)
    return result


def _validated_registry_phase(omega: ArrayLike) -> NDArray[np.complex128]:
    phase = np.asarray(omega, dtype=np.complex128)
    if not np.all(np.isfinite(phase)):
        raise ValueError("registry phase must be a finite cube root of unity")
    distances = np.abs(phase[..., None] - _REGISTRY_ROOTS)
    nearest = np.argmin(distances, axis=-1)
    if np.any(np.take_along_axis(distances, nearest[..., None], axis=-1)[..., 0] > 1e-12):
        raise ValueError("registry phase must be a finite cube root of unity")
    result = np.array(_REGISTRY_ROOTS[nearest], dtype=np.complex128, copy=True)
    result.setflags(write=False)
    return result


def reduced_transition_matrix(law: TransitionLaw, omega: ArrayLike) -> NDArray[np.complex128]:
    """Return exact two-orientation Fourier block selected by ``omega``."""

    phase = _validated_registry_phase(omega)
    inverse = np.conj(phase)
    diagonal = law.a + law.b_plus * phase + law.b_minus * inverse
    upper = law.d_plus * phase + law.d_minus * inverse
    lower = law.d_plus * inverse + law.d_minus * phase
    result = np.empty((*phase.shape, 2, 2), dtype=np.complex128)
    result[..., 0, 0] = diagonal
    result[..., 0, 1] = upper
    result[..., 1, 0] = lower
    result[..., 1, 1] = diagonal
    result.setflags(write=False)
    return result
