"""Analytic intersections of complete reciprocal rods with an elastic Ewald sphere."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from math import fsum, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald.types import EwaldRoot, EwaldRootResult, RootStatus
from painted_ewald.validation import finite_scalar, readonly_float_array

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _RootBatch:
    branch: int
    orientation_index: IntArray
    u_Ainv: FloatArray
    L: FloatArray
    q_sample_Ainv: FloatArray
    kf_sample_Ainv: FloatArray
    coarea_jacobian: FloatArray
    ewald_residual_Ainv: FloatArray


@dataclass(frozen=True, slots=True)
class _RootBatchResult:
    roots: tuple[_RootBatch, ...]
    collapsed_direct: NDArray[np.bool_]
    tangent: NDArray[np.bool_]
    no_root: NDArray[np.bool_]
    suppressed_algebraic_direct_root_count: int


def _dot(left: FloatArray, right: FloatArray) -> float:
    return fsum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _validated_unit_vector(value: ArrayLike, name: str) -> FloatArray:
    vector = readonly_float_array(value, (3,), name)
    norm = sqrt(_dot(vector, vector))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must be unit length")
    normalized = np.asarray(vector / norm, dtype=np.float64)
    normalized.setflags(write=False)
    return normalized


def _nonnegative_tolerance(value: object, name: str) -> float:
    tolerance = finite_scalar(value, name)
    if tolerance < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return tolerance


def _stable_line_components(
    q0: FloatArray,
    direction: FloatArray,
) -> tuple[float, FloatArray]:
    """Return affine coordinate and perpendicular anchor without cancellation."""

    with localcontext() as context:
        context.prec = 80
        q_decimal = tuple(Decimal.from_float(float(value)) for value in q0)
        d_decimal = tuple(Decimal.from_float(float(value)) for value in direction)
        direction_squared = sum(value * value for value in d_decimal)
        parallel = (
            sum(q_value * d_value for q_value, d_value in zip(q_decimal, d_decimal, strict=True))
            / direction_squared
        )
        perpendicular = np.asarray(
            [
                float(q_value - parallel * d_value)
                for q_value, d_value in zip(q_decimal, d_decimal, strict=True)
            ],
            dtype=np.float64,
        )
    return float(parallel), perpendicular


def _solve_validated_infinite_rod_ewald(
    *,
    incident: FloatArray,
    incident_norm: float,
    q0: FloatArray,
    q0_parallel_Ainv: float | None,
    direction: FloatArray,
    b3_norm_Ainv: float,
    rod_is_m0: bool,
    root_tolerance_rel: float,
    residual_tolerance_rel: float,
) -> EwaldRootResult:
    """Objectize one row from the authoritative batched equation path."""

    if q0_parallel_Ainv is not None:
        q0_parallel = q0_parallel_Ainv
        q0_perpendicular = q0
    elif rod_is_m0 and not np.any(q0):
        q0_parallel = 0.0
        q0_perpendicular = q0
    else:
        q0_parallel, q0_perpendicular = _stable_line_components(q0, direction)
    batch = _solve_batched_infinite_rod_ewald(
        incident=incident,
        incident_norm=incident_norm,
        q0=q0_perpendicular[None, :],
        q0_parallel_Ainv=q0_parallel,
        direction=direction[None, :],
        b3_norm_Ainv=b3_norm_Ainv,
        rod_is_m0=rod_is_m0,
        root_tolerance_rel=root_tolerance_rel,
        residual_tolerance_rel=residual_tolerance_rel,
    )
    if batch.collapsed_direct[0]:
        return EwaldRootResult(RootStatus.COLLAPSED_DIRECT, (), 1)
    if batch.no_root[0]:
        return EwaldRootResult(RootStatus.NO_ROOT, (), 0)
    if batch.tangent[0]:
        return EwaldRootResult(RootStatus.TANGENT, (), 0)
    roots = tuple(
        EwaldRoot(
            u_Ainv=float(root.u_Ainv[0]),
            L=float(root.L[0]),
            q_sample_Ainv=root.q_sample_Ainv[0],
            kf_sample_Ainv=root.kf_sample_Ainv[0],
            ewald_residual_Ainv=float(root.ewald_residual_Ainv[0]),
            coarea_jacobian=float(root.coarea_jacobian[0]),
            branch=root.branch,
        )
        for root in batch.roots
    )
    return EwaldRootResult(
        RootStatus.REGULAR,
        roots,
        batch.suppressed_algebraic_direct_root_count,
    )


def _validated_root_batch(
    *,
    branch: int,
    orientation_index: IntArray,
    u_Ainv: FloatArray,
    q_sample_Ainv: FloatArray,
    kf_sample_Ainv: FloatArray,
    coarea_jacobian: FloatArray,
    incident_norm: float,
    b3_norm_Ainv: float,
    residual_tolerance_rel: float,
) -> _RootBatch:
    residual = np.abs(np.linalg.norm(kf_sample_Ainv, axis=1) - incident_norm)
    l_coordinate = u_Ainv / b3_norm_Ainv
    q_norm = np.linalg.norm(q_sample_Ainv, axis=1)
    if not all(
        np.all(np.isfinite(array))
        for array in (
            u_Ainv,
            l_coordinate,
            q_sample_Ainv,
            kf_sample_Ainv,
            coarea_jacobian,
            residual,
            q_norm,
        )
    ):
        raise FloatingPointError("analytic Ewald root produced a nonfinite value")
    residual_limit = residual_tolerance_rel * max(incident_norm, 1.0)
    if np.any(residual > residual_limit):
        raise FloatingPointError("analytic Ewald root failed the unsquared residual")
    if np.any(q_norm > 2.0 * incident_norm + residual_limit):
        raise FloatingPointError("elastic Ewald root exceeds the 2K transfer bound")
    return _RootBatch(
        branch=branch,
        orientation_index=orientation_index,
        u_Ainv=u_Ainv,
        L=l_coordinate,
        q_sample_Ainv=q_sample_Ainv,
        kf_sample_Ainv=kf_sample_Ainv,
        coarea_jacobian=coarea_jacobian,
        ewald_residual_Ainv=residual,
    )


def _solve_batched_infinite_rod_ewald(
    *,
    incident: FloatArray,
    incident_norm: float,
    q0: FloatArray | None,
    q0_parallel_Ainv: FloatArray | float | None,
    direction: FloatArray,
    b3_norm_Ainv: float,
    rod_is_m0: bool,
    root_tolerance_rel: float,
    residual_tolerance_rel: float,
) -> _RootBatchResult:
    """Vectorized optimized path matching the scalar analytic-root contract."""

    orientation_count = direction.shape[0]
    direction_norm_squared = np.einsum("ij,ij->i", direction, direction, optimize=True)
    if not np.all(np.isfinite(direction_norm_squared)) or np.any(direction_norm_squared <= 0.0):
        raise FloatingPointError("Ewald rod direction has invalid norm")
    direction_norm = np.sqrt(direction_norm_squared)
    if q0 is None:
        if not rod_is_m0:
            raise ValueError("q0 is required for a nonzero rod")
        q0_parallel = np.zeros(orientation_count, dtype=np.float64)
        q0_perpendicular = np.zeros_like(direction)
    else:
        parallel_correction = (
            np.einsum("ij,ij->i", q0, direction, optimize=True) / direction_norm_squared
        )
        q0_perpendicular = q0 - parallel_correction[:, None] * direction
        if q0_parallel_Ainv is None:
            q0_parallel = parallel_correction
        else:
            supplied_parallel = np.asarray(q0_parallel_Ainv, dtype=np.float64)
            try:
                supplied_parallel = np.broadcast_to(
                    supplied_parallel,
                    (orientation_count,),
                )
            except ValueError as error:
                raise ValueError(
                    "q0_parallel_Ainv must broadcast to the orientation count"
                ) from error
            q0_parallel = supplied_parallel + parallel_correction
    incident_dot_direction = direction @ incident
    incident_parallel = incident_dot_direction / direction_norm_squared
    if not all(
        np.all(np.isfinite(array))
        for array in (
            q0_parallel,
            q0_perpendicular,
            incident_dot_direction,
            incident_parallel,
        )
    ):
        raise FloatingPointError("Ewald line projection produced a nonfinite value")
    false_mask = np.zeros(orientation_count, dtype=np.bool_)

    if rod_is_m0:
        parallel_anchor = q0_parallel[:, None] * direction
        represented_anchor = q0_perpendicular + parallel_anchor
        line_component_scale = np.maximum(
            np.abs(represented_anchor) + np.abs(parallel_anchor),
            1.0,
        )
        if np.any(np.abs(q0_perpendicular) > residual_tolerance_rel * line_component_scale):
            raise ValueError("an m=0 rod line must contain q=0")
        physical_u = -2.0 * incident_parallel
        physical_q_norm = np.abs(physical_u) * direction_norm
        collapse_tolerance = root_tolerance_rel * max(incident_norm, 1.0)
        collapsed = physical_q_norm <= collapse_tolerance
        orientation_index = np.flatnonzero(~collapsed)
        selected_physical_u = physical_u[orientation_index]
        selected_direction = direction[orientation_index]
        q_sample = selected_physical_u[:, None] * selected_direction
        kf_sample = incident + q_sample
        root = _validated_root_batch(
            branch=0,
            orientation_index=orientation_index,
            u_Ainv=selected_physical_u - q0_parallel[orientation_index],
            q_sample_Ainv=q_sample,
            kf_sample_Ainv=kf_sample,
            coarea_jacobian=(incident_norm / np.abs(incident_dot_direction[orientation_index])),
            incident_norm=incident_norm,
            b3_norm_Ainv=b3_norm_Ainv,
            residual_tolerance_rel=residual_tolerance_rel,
        )
        return _RootBatchResult(
            roots=(root,),
            collapsed_direct=collapsed,
            tangent=false_mask,
            no_root=false_mask,
            suppressed_algebraic_direct_root_count=orientation_count,
        )

    incident_perpendicular = incident - incident_parallel[:, None] * direction
    sphere_perpendicular = incident_perpendicular + q0_perpendicular
    perpendicular_norm_squared = np.einsum(
        "ij,ij->i", sphere_perpendicular, sphere_perpendicular, optimize=True
    )
    discriminant = incident_norm * incident_norm - perpendicular_norm_squared
    if not np.all(np.isfinite(discriminant)):
        raise FloatingPointError("Ewald discriminant produced a nonfinite value")
    # Scaling by the physical perpendicular line distance is invariant under
    # q0 -> q0 + t*d. The raw max(A**2, |Cq|, K**2, 1) prescription is not:
    # a sufficiently large anchor shift can relabel an unchanged regular line
    # as tangent. The retained anchor-shift test fixes this corrected boundary.
    discriminant_scale = np.maximum(
        np.maximum(incident_norm * incident_norm, perpendicular_norm_squared),
        1.0,
    )
    discriminant_tolerance = root_tolerance_rel * discriminant_scale
    no_root = discriminant < -discriminant_tolerance
    tangent = np.abs(discriminant) <= discriminant_tolerance
    orientation_index = np.flatnonzero(~no_root & ~tangent)
    sqrt_discriminant = np.sqrt(discriminant[orientation_index])
    root_coordinate_magnitude = sqrt_discriminant / direction_norm[orientation_index]
    parallel_offset = incident_parallel[orientation_index] + q0_parallel[orientation_index]
    selected_direction = direction[orientation_index]
    selected_sphere_perpendicular = sphere_perpendicular[orientation_index]
    coarea_jacobian = incident_norm / (sqrt_discriminant * direction_norm[orientation_index])
    roots: list[_RootBatch] = []
    for branch, signed_discriminant in (
        (1, -root_coordinate_magnitude),
        (2, root_coordinate_magnitude),
    ):
        kf_sample = (
            selected_sphere_perpendicular + signed_discriminant[:, None] * selected_direction
        )
        q_sample = kf_sample - incident
        roots.append(
            _validated_root_batch(
                branch=branch,
                orientation_index=orientation_index,
                u_Ainv=-parallel_offset + signed_discriminant,
                q_sample_Ainv=q_sample,
                kf_sample_Ainv=kf_sample,
                coarea_jacobian=coarea_jacobian,
                incident_norm=incident_norm,
                b3_norm_Ainv=b3_norm_Ainv,
                residual_tolerance_rel=residual_tolerance_rel,
            )
        )
    return _RootBatchResult(
        roots=tuple(roots),
        collapsed_direct=false_mask,
        tangent=tangent,
        no_root=no_root,
        suppressed_algebraic_direct_root_count=0,
    )


def solve_infinite_rod_ewald(
    *,
    ki_sample_Ainv: ArrayLike,
    q0_sample_Ainv: ArrayLike,
    d_hat_sample: ArrayLike,
    b3_norm_Ainv: float,
    rod_is_m0: bool,
    root_tolerance_rel: float,
    residual_tolerance_rel: float,
) -> EwaldRootResult:
    """Solve ``|ki + q0 + u*d| = |ki|`` without truncating the rod.

    The discriminant is evaluated from the line's perpendicular offset, an
    anchor-invariant form of ``A**2 - Cq``. A zero root tolerance requests
    the strict sign classification required by the detector pushforward.
    """

    incident = readonly_float_array(ki_sample_Ainv, (3,), "ki_sample_Ainv")
    q0 = readonly_float_array(q0_sample_Ainv, (3,), "q0_sample_Ainv")
    direction = _validated_unit_vector(d_hat_sample, "d_hat_sample")
    incident_norm = sqrt(_dot(incident, incident))
    if incident_norm == 0.0:
        raise ValueError("ki_sample_Ainv must be nonzero")
    b3_norm = finite_scalar(b3_norm_Ainv, "b3_norm_Ainv")
    if b3_norm <= 0.0:
        raise ValueError("b3_norm_Ainv must be positive")
    if not isinstance(rod_is_m0, (bool, np.bool_)):
        raise ValueError("rod_is_m0 must be boolean")
    root_tolerance = _nonnegative_tolerance(root_tolerance_rel, "root_tolerance_rel")
    residual_tolerance = _nonnegative_tolerance(residual_tolerance_rel, "residual_tolerance_rel")
    return _solve_validated_infinite_rod_ewald(
        incident=incident,
        incident_norm=incident_norm,
        q0=q0,
        q0_parallel_Ainv=None,
        direction=direction,
        b3_norm_Ainv=b3_norm,
        rod_is_m0=bool(rod_is_m0),
        root_tolerance_rel=root_tolerance,
        residual_tolerance_rel=residual_tolerance,
    )


__all__ = ["EwaldRoot", "EwaldRootResult", "RootStatus", "solve_infinite_rod_ewald"]
