"""Analytic continuous-rod intersections with the elastic Ewald sphere."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class EwaldRootStatus(StrEnum):
    TWO_ROOT = "TWO_ROOT"
    TANGENT = "TANGENT"
    NO_ROOT = "NO_ROOT"


def _vector(value: ArrayLike, name: str) -> FloatArray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) or (
        supplied.dtype.kind == "O" and any(np.iscomplexobj(item) for item in supplied.flat)
    ):
        raise ValueError(f"{name} must be real")
    array = np.array(supplied, dtype=np.float64, copy=True)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    array.setflags(write=False)
    return array


def _compensated_dot(left: FloatArray, right: FloatArray) -> float:
    return math.fsum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _exactly_collinear(left: FloatArray, right: FloatArray) -> bool:
    if np.count_nonzero(left) == 0:
        return True
    for first_index, second_index in ((1, 2), (2, 0), (0, 1)):
        first_product = float(left[first_index]) * float(right[second_index])
        second_product = float(left[second_index]) * float(right[first_index])
        if abs(first_product - second_product) > (
            8.0
            * np.finfo(np.float64).eps
            * (abs(first_product) + abs(second_product))
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
    """Solve ``|ki + q0 + u*d_hat| = |ki|`` and validate unsquared residuals."""

    incident = _vector(ki_sample_Ainv, "ki_sample_Ainv")
    q0 = _vector(q0_sample_Ainv, "q0_sample_Ainv")
    direction = _vector(d_hat_sample, "d_hat_sample")
    incident_norm = math.sqrt(_compensated_dot(incident, incident))
    if incident_norm == 0.0:
        raise ValueError("ki_sample_Ainv must be nonzero")
    direction_norm = math.sqrt(_compensated_dot(direction, direction))
    if not np.isclose(direction_norm, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("d_hat_sample must be unit length")
    line_contains_direct_root = _exactly_collinear(q0, direction)
    direction = direction / direction_norm
    if np.iscomplexobj(np.asarray(b3_norm_Ainv)):
        raise ValueError("b3_norm_Ainv must be real")
    b3_norm = float(b3_norm_Ainv)
    if not np.isfinite(b3_norm) or b3_norm <= 0.0:
        raise ValueError("b3_norm_Ainv must be positive")

    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(direction)))] = 1.0
    perpendicular_1 = np.cross(direction, seed)
    perpendicular_1 /= np.linalg.norm(perpendicular_1)
    perpendicular_2 = np.cross(direction, perpendicular_1)
    perpendicular_2 /= np.linalg.norm(perpendicular_2)

    incident_perpendicular_1 = _compensated_dot(incident, perpendicular_1)
    incident_perpendicular_2 = _compensated_dot(incident, perpendicular_2)
    incident_parallel = _compensated_dot(incident, direction)
    q0_perpendicular_1 = _compensated_dot(q0, perpendicular_1)
    q0_perpendicular_2 = _compensated_dot(q0, perpendicular_2)
    q0_parallel = _compensated_dot(q0, direction)
    sphere_parallel_offset = math.fsum((incident_parallel, q0_parallel))
    if line_contains_direct_root:
        sphere_perpendicular_1 = incident_perpendicular_1
        sphere_perpendicular_2 = incident_perpendicular_2
        sphere_coordinate_magnitude = abs(incident_parallel)
    else:
        sphere_perpendicular_1 = math.fsum((incident_perpendicular_1, q0_perpendicular_1))
        sphere_perpendicular_2 = math.fsum((incident_perpendicular_2, q0_perpendicular_2))
        radicand = math.fsum(
            (
                incident_norm * incident_norm,
                -sphere_perpendicular_1 * sphere_perpendicular_1,
                -sphere_perpendicular_2 * sphere_perpendicular_2,
            )
        )
        if radicand < 0.0:
            return EwaldRootResult(EwaldRootStatus.NO_ROOT, (), 0)
        sphere_coordinate_magnitude = math.sqrt(radicand)
    if sphere_coordinate_magnitude == 0.0:
        return EwaldRootResult(
            EwaldRootStatus.TANGENT,
            (),
            int(line_contains_direct_root),
        )

    sphere_coordinates = (-sphere_coordinate_magnitude, sphere_coordinate_magnitude)
    emittable: list[EwaldRoot] = []
    residual_limit = 64.0 * np.finfo(np.float64).eps * max(incident_norm, 1.0)
    for sphere_coordinate in sphere_coordinates:
        if line_contains_direct_root and sphere_coordinate == incident_parallel:
            continue
        u_Ainv = sphere_coordinate - sphere_parallel_offset
        kf = np.array(
            [
                math.fsum(
                    (
                        sphere_perpendicular_1 * perpendicular_1[index],
                        sphere_perpendicular_2 * perpendicular_2[index],
                        sphere_coordinate * direction[index],
                    )
                )
                for index in range(3)
            ]
        )
        q = kf - incident
        kf_norm = math.sqrt(_compensated_dot(kf, kf))
        residual = abs(kf_norm - incident_norm)
        if residual > residual_limit:
            raise FloatingPointError("quadratic Ewald root failed the unsquared residual")
        derivative = abs(sphere_coordinate) / kf_norm
        if derivative == 0.0:
            raise FloatingPointError("regular Ewald root has zero coarea derivative")
        emittable.append(
            EwaldRoot(
                u_Ainv=u_Ainv,
                l_coordinate=u_Ainv / b3_norm,
                q_sample_Ainv=q,
                kf_sample_Ainv=kf,
                ewald_residual_Ainv=residual,
                coarea_jacobian=1.0 / derivative,
            )
        )
    return EwaldRootResult(
        EwaldRootStatus.TWO_ROOT,
        tuple(emittable),
        int(line_contains_direct_root),
    )
