"""Unit labels used in public contracts and proof traces."""

from enum import StrEnum


class Unit(StrEnum):
    NONE = "none"
    DIMENSIONLESS = "1"
    METRE = "m"
    ANGSTROM = "angstrom"
    INVERSE_ANGSTROM = "angstrom^-1"
    RADIAN = "rad"
    PIXEL = "px"
    STERADIAN = "sr"
