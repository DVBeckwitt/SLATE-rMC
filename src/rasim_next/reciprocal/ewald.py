"""Compatibility contract for analytic continuous-rod Ewald intersections."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald.ewald import RootStatus, solve_infinite_rod_ewald
from painted_ewald.validation import readonly_float_array

FloatArray = NDArray[np.float64]


class EwaldRootStatus(StrEnum):
    TWO_ROOT = "TWO_ROOT"
    TANGENT = "TANGENT"
    NO_ROOT = "NO_ROOT"


def _vector(value: ArrayLike, name: str) -> FloatArray:
    return readonly_float_array(value, (3,), name)


def _exactly_collinear(left: FloatArray, right: FloatArray) -> bool:
    if np.count_nonzero(left) == 0:
        return True
    for first_index, second_index in ((1, 2), (2, 0), (0, 1)):
        first_product = float(left[first_index]) * float(right[second_index])
        second_product = float(left[second_index]) * float(right[first_index])
        if abs(first_product - second_product) > (
            8.0 * np.finfo(np.float64).eps * (abs(first_product) + abs(second_product))
        ):
            return False
    anchor = int(np.argmax(np.abs(right)))
    left_anchor = Fraction.from_float(float(left[anchor]))
    right_anchor = Fraction.from_float(float(right[anchor]))
    return all(
        Fraction.from_float(float(left_value)) * right_anchor
        == left_anchor * Fraction.from_float(float(right_value))
        for left_value, right_value in zip(left, right, strict=True)
    )


@dataclass(frozen=True, slots=True)
class EwaldRoot:
    u_Ainv: float
    l_coordinate: float
    q_sample_Ainv: FloatArray
    kf_sample_Ainv: FloatArray
    ewald_residual_Ainv: float
    coarea_jacobian: float

    def __post_init__(self) -> None:
        for name in ("q_sample_Ainv", "kf_sample_Ainv"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        scalars = (
            self.u_Ainv,
            self.l_coordinate,
            self.ewald_residual_Ainv,
            self.coarea_jacobian,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("Ewald root scalars must be finite")
        if self.ewald_residual_Ainv < 0.0 or self.coarea_jacobian <= 0.0:
            raise ValueError("Ewald residual must be nonnegative and Jacobian positive")


@dataclass(frozen=True, slots=True)
class EwaldRootResult:
    status: EwaldRootStatus
    emittable_roots: tuple[EwaldRoot, ...]
    direct_beam_root_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", EwaldRootStatus(self.status))
        object.__setattr__(self, "emittable_roots", tuple(self.emittable_roots))
        if (
            isinstance(self.direct_beam_root_count, bool)
            or not isinstance(self.direct_beam_root_count, (int, np.integer))
            or self.direct_beam_root_count < 0
        ):
            raise ValueError("direct_beam_root_count must be a nonnegative integer")
        object.__setattr__(self, "direct_beam_root_count", int(self.direct_beam_root_count))
        if self.status is not EwaldRootStatus.TWO_ROOT and self.emittable_roots:
            raise ValueError("tangent and no-root results cannot emit roots")
        classified_count = len(self.emittable_roots) + self.direct_beam_root_count
        if self.status is EwaldRootStatus.NO_ROOT and classified_count != 0:
            raise ValueError("NO_ROOT cannot classify any root")
        if self.status is EwaldRootStatus.TANGENT and self.direct_beam_root_count > 1:
            raise ValueError("TANGENT can classify at most one direct-beam root")
        if self.status is EwaldRootStatus.TWO_ROOT and classified_count != 2:
            raise ValueError("TWO_ROOT must classify exactly two roots")


def solve_continuous_rod_ewald(
    *,
    ki_sample_Ainv: ArrayLike,
    q0_sample_Ainv: ArrayLike,
    d_hat_sample: ArrayLike,
    b3_norm_Ainv: float,
) -> EwaldRootResult:
    """Preserve the accepted strict-status API using the shared root authority."""

    q0 = _vector(q0_sample_Ainv, "q0_sample_Ainv")
    direction = _vector(d_hat_sample, "d_hat_sample")
    result = solve_infinite_rod_ewald(
        ki_sample_Ainv=ki_sample_Ainv,
        q0_sample_Ainv=q0,
        d_hat_sample=direction,
        b3_norm_Ainv=b3_norm_Ainv,
        rod_is_m0=_exactly_collinear(q0, direction),
        root_tolerance_rel=0.0,
        residual_tolerance_rel=64.0 * np.finfo(np.float64).eps,
    )
    status = {
        RootStatus.NO_ROOT: EwaldRootStatus.NO_ROOT,
        RootStatus.TANGENT: EwaldRootStatus.TANGENT,
        RootStatus.COLLAPSED_DIRECT: EwaldRootStatus.TANGENT,
        RootStatus.REGULAR: EwaldRootStatus.TWO_ROOT,
    }[result.status]
    roots = tuple(
        EwaldRoot(
            u_Ainv=root.u_Ainv,
            l_coordinate=root.L,
            q_sample_Ainv=root.q_sample_Ainv,
            kf_sample_Ainv=root.kf_sample_Ainv,
            ewald_residual_Ainv=root.ewald_residual_Ainv,
            coarea_jacobian=root.coarea_jacobian,
        )
        for root in result.emittable_roots
    )
    return EwaldRootResult(status, roots, result.direct_root_count)


__all__ = ["EwaldRoot", "EwaldRootResult", "EwaldRootStatus", "solve_continuous_rod_ewald"]
