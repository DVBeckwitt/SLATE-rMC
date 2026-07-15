"""Common explicit validity codes; invalid data never use sentinel values."""

from enum import StrEnum


class ValidityCode(StrEnum):
    VALID = "VALID"
    PARALLEL = "PARALLEL"
    BACKWARD = "BACKWARD"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"
    NO_SOLUTION = "NO_SOLUTION"
    NON_PROPAGATING = "NON_PROPAGATING"
    RESIDUAL_EXCEEDED = "RESIDUAL_EXCEEDED"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"
