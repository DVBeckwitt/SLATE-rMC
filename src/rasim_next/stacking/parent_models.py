"""Typed deterministic, parent-rich, and reduced stacking models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from rasim_next.stacking.transition import InitialPopulation, Parent, TransitionLaw


class Handedness(StrEnum):
    PLUS = "plus"
    MINUS = "minus"


@dataclass(frozen=True, slots=True)
class RichEpsilonModel:
    parent: Parent
    epsilon: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent", Parent(self.parent))
        epsilon = float(self.epsilon)
        if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be finite and between zero and one")
        object.__setattr__(self, "epsilon", epsilon)

    def transition_law(self) -> TransitionLaw:
        parent = TransitionLaw.for_parent(self.parent).as_array()
        alternatives = (1.0 - parent) / 4.0
        return TransitionLaw.from_array((1.0 - self.epsilon) * parent + self.epsilon * alternatives)


@dataclass(frozen=True, slots=True)
class ReducedABDModel:
    """Legacy reduced ``a,b,d`` law with one explicitly selected handedness."""

    a: float
    b: float
    d: float
    handedness: Handedness = Handedness.PLUS

    def __post_init__(self) -> None:
        values = np.array([self.a, self.b, self.d], dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("a, b, and d must be finite and nonnegative")
        if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("a, b, and d must sum to one")
        object.__setattr__(self, "a", float(values[0]))
        object.__setattr__(self, "b", float(values[1]))
        object.__setattr__(self, "d", float(values[2]))
        object.__setattr__(self, "handedness", Handedness(self.handedness))

    def transition_law(self) -> TransitionLaw:
        if self.handedness is Handedness.PLUS:
            return TransitionLaw(self.a, self.b, 0.0, self.d, 0.0)
        return TransitionLaw(self.a, 0.0, self.b, 0.0, self.d)


@dataclass(frozen=True, slots=True)
class StackingPopulation:
    """One explicit incoherent population of a stacking transition model."""

    population_id: str
    model: TransitionLaw
    initial: InitialPopulation

    def __post_init__(self) -> None:
        if not isinstance(self.population_id, str) or not self.population_id:
            raise ValueError("population_id must be a nonempty string")
        if not isinstance(self.model, TransitionLaw):
            raise TypeError("model must be a TransitionLaw")
        if not isinstance(self.initial, InitialPopulation):
            raise TypeError("initial must be an InitialPopulation")
