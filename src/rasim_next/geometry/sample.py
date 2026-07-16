"""Conditioned ray intersections with the finite sample surface."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.validity import ValidityCode
from rasim_next.geometry._vectors import finite_vector3, finite_vectors3

_PARALLEL_TOL = 1e-14
_POSITION_TOL_M = 1e-12


@dataclass(frozen=True, slots=True)
class SampleIntersection:
    """A scalar sample-plane result; numeric fields are finite for every status."""

    point_lab_m: NDArray[np.float64]
    point_sample_m: NDArray[np.float64]
    ray_distance_m: float
    footprint_acceptance: float
    status: ValidityCode


@dataclass(frozen=True, slots=True)
class _SampleIntersectionArrays:
    point_lab_m: NDArray[np.float64]
    point_sample_m: NDArray[np.float64]
    ray_distance_m: NDArray[np.float64]
    footprint_acceptance: NDArray[np.float64]
    status: NDArray[np.str_]


def _intersect_sample_rays(
    origin_lab_m: ArrayLike,
    direction_lab: ArrayLike,
    *,
    lab_from_sample: RigidTransform,
    sample_width_m: float,
    sample_length_m: float,
) -> _SampleIntersectionArrays:
    origins = finite_vectors3(origin_lab_m, "origin_lab_m")
    directions = finite_vectors3(direction_lab, "direction_lab")
    if origins.shape != directions.shape:
        raise ValueError("origin_lab_m and direction_lab must have equal shapes")
    if not np.allclose(
        np.linalg.norm(directions, axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("direction_lab must contain unit vectors")
    if not isinstance(lab_from_sample, RigidTransform):
        raise TypeError("lab_from_sample must be a RigidTransform")
    if (
        lab_from_sample.source_frame != FrameId.SAMPLE
        or lab_from_sample.target_frame != FrameId.LAB
    ):
        raise ValueError("lab_from_sample must map sample to lab")
    width = float(sample_width_m)
    length = float(sample_length_m)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("sample_width_m must be finite and positive")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("sample_length_m must be finite and positive")

    sample_from_lab = lab_from_sample.inverse()
    origin_sample_m = sample_from_lab.apply_point(origins)
    direction_sample = sample_from_lab.apply_vector(directions)
    denominator = direction_sample[:, 2]
    offset_m = origin_sample_m[:, 2]
    parallel = np.abs(denominator) <= _PARALLEL_TOL
    coplanar = parallel & (np.abs(offset_m) <= _POSITION_TOL_M)
    status = np.full(origins.shape[0], ValidityCode.VALID, dtype="U16")
    status[parallel & ~coplanar] = ValidityCode.PARALLEL

    distance_m = np.zeros(origins.shape[0], dtype=np.float64)
    nonparallel = ~parallel
    distance_m[nonparallel] = -offset_m[nonparallel] / denominator[nonparallel]
    backward = nonparallel & (distance_m < -_POSITION_TOL_M)
    status[backward] = ValidityCode.BACKWARD
    distance_m = np.maximum(distance_m, 0.0)
    point_sample_m = origin_sample_m + distance_m[:, None] * direction_sample
    active = status == ValidityCode.VALID
    outside = active & (
        (np.abs(point_sample_m[:, 0]) > 0.5 * width) | (np.abs(point_sample_m[:, 1]) > 0.5 * length)
    )
    status[outside] = ValidityCode.OUTSIDE_SUPPORT
    valid = status == ValidityCode.VALID

    point_lab_output = np.zeros_like(origins)
    point_sample_output = np.zeros_like(origins)
    distance_output = np.zeros(origins.shape[0], dtype=np.float64)
    point_lab_output[valid] = lab_from_sample.apply_point(point_sample_m[valid])
    point_sample_output[valid] = point_sample_m[valid]
    distance_output[valid] = distance_m[valid]
    return _SampleIntersectionArrays(
        point_lab_output,
        point_sample_output,
        distance_output,
        valid.astype(np.float64),
        status,
    )


def intersect_sample_ray(
    origin_lab_m: ArrayLike,
    direction_lab: ArrayLike,
    *,
    lab_from_sample: RigidTransform,
    sample_width_m: float,
    sample_length_m: float,
) -> SampleIntersection:
    """Intersect a forward unit ray with local sample ``z=0`` and its rectangular footprint."""

    origin = finite_vector3(origin_lab_m, "origin_lab_m")
    direction = finite_vector3(direction_lab, "direction_lab")
    intersections = _intersect_sample_rays(
        origin[None, :],
        direction[None, :],
        lab_from_sample=lab_from_sample,
        sample_width_m=sample_width_m,
        sample_length_m=sample_length_m,
    )
    return SampleIntersection(
        finite_vector3(intersections.point_lab_m[0], "point_lab_m"),
        finite_vector3(intersections.point_sample_m[0], "point_sample_m"),
        float(intersections.ray_distance_m[0]),
        float(intersections.footprint_acceptance[0]),
        ValidityCode(intersections.status[0]),
    )
