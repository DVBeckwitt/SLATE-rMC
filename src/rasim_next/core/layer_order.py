"""Exact commensurate coordinates along one declared layer repeat."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import total_ordering
from typing import Any


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


@total_ordering
@dataclass(frozen=True, slots=True)
class CommensurateLayerOrder:
    """Reduced exact value of ``L`` in a declared one-layer reciprocal basis."""

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        numerator = _integer(self.numerator, "numerator")
        denominator = _integer(self.denominator, "denominator")
        if denominator == 0:
            raise ValueError("denominator must be nonzero")
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        divisor = math.gcd(numerator, denominator)
        numerator //= divisor
        denominator //= divisor
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CommensurateLayerOrder):
            return NotImplemented
        return self.numerator * other.denominator < other.numerator * self.denominator

    @property
    def is_integer(self) -> bool:
        return self.denominator == 1

    def as_float(self) -> float:
        return self.numerator / self.denominator

    def __str__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"


__all__ = ["CommensurateLayerOrder"]
