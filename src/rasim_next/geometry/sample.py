"""Conditioned ray intersections with the finite sample surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.validity import ValidityCode
from rasim_next.geometry._vectors import finite_vector3, finite_vectors3
from rasim_next.geometry.instrument import _sample_support

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
    direction_sample: NDArray[np.float64]
    status: NDArray[np.str_]


def _intersect_sample_rays(
    origin_lab_m: ArrayLike,
    direction_lab: ArrayLike,
    *,
    lab_from_sample: RigidTransform,
    sample_from_lab: RigidTransform,
    sample_support_model_id: str,
    sample_width_m: float | None,
    sample_length_m: float | None,
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
    if not isinstance(sample_from_lab, RigidTransform):
        raise TypeError("sample_from_lab must be a RigidTransform")
    if (
        sample_from_lab.source_frame != FrameId.LAB
        or sample_from_lab.target_frame != FrameId.SAMPLE
    ):
        raise ValueError("sample_from_lab must map lab to sample")
    support_model_id, width, length = _sample_support(
        sample_support_model_id,
        sample_width_m,
        sample_length_m,
    )

    origin_sample_m = sample_from_lab.apply_point(origins)
    direction_sample = sample_from_lab.apply_vector(directions)
    denominator = direction_sample[:, 2]
    offset_m = origin_sample_m[:, 2]
    parallel = np.abs(denominator) <= _PARALLEL_TOL
    status = np.full(origins.shape[0], ValidityCode.VALID, dtype="U16")
    status[parallel] = ValidityCode.PARALLEL

    distance_m = np.zeros(origins.shape[0], dtype=np.float64)
    nonparallel = ~parallel
    distance_m[nonparallel] = -offset_m[nonparallel] / denominator[nonparallel]
    backward = nonparallel & (distance_m < -_POSITION_TOL_M)
    status[backward] = ValidityCode.BACKWARD
    distance_m = np.maximum(distance_m, 0.0)
    point_sample_m = origin_sample_m + distance_m[:, None] * direction_sample
    active = status == ValidityCode.VALID
    outside = np.zeros(origins.shape[0], dtype=np.bool_)
    if support_model_id == "finite_rectangle.v1":
        assert width is not None and length is not None
        outside = active & (
            (np.abs(point_sample_m[:, 0]) > 0.5 * width)
            | (np.abs(point_sample_m[:, 1]) > 0.5 * length)
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
        direction_sample,
        status,
    )


def intersect_sample_ray(
    origin_lab_m: ArrayLike,
    direction_lab: ArrayLike,
    *,
    lab_from_sample: RigidTransform,
    sample_support_model_id: str,
    sample_width_m: float | None,
    sample_length_m: float | None,
) -> SampleIntersection:
    """Intersect a forward unit ray with local sample ``z=0`` and its rectangular footprint."""

    origin = finite_vector3(origin_lab_m, "origin_lab_m")
    direction = finite_vector3(direction_lab, "direction_lab")
    intersections = _intersect_sample_rays(
        origin[None, :],
        direction[None, :],
        lab_from_sample=lab_from_sample,
        sample_from_lab=lab_from_sample.inverse(),
        sample_support_model_id=sample_support_model_id,
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
