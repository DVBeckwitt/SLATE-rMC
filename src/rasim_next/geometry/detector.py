"""Forward and inverse geometry for a rectangular detector plane."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.validity import ValidityCode
from rasim_next.geometry._vectors import finite_vector3, finite_vectors3
from rasim_next.geometry.instrument import CompiledInstrument

_PARALLEL_TOL = 1e-14
_POSITION_TOL_M = 1e-12


@dataclass(frozen=True, slots=True)
class DetectorProjection:
    """One ray-to-detector result with continuous detector-native coordinates."""

    point_lab_m: NDArray[np.float64]
    column_px: float
    row_px: float
    ray_distance_m: float
    pixel_solid_angle_sr: float
    status: ValidityCode


@dataclass(frozen=True, slots=True)
class DetectorRay:
    """A ray from a lab point through one continuous detector coordinate."""

    detector_point_lab_m: NDArray[np.float64]
    direction_lab: NDArray[np.float64]
    ray_distance_m: float
    status: ValidityCode


@dataclass(frozen=True, slots=True)
class _DetectorProjectionArrays:
    point_lab_m: NDArray[np.float64]
    column_px: NDArray[np.float64]
    row_px: NDArray[np.float64]
    ray_distance_m: NDArray[np.float64]
    pixel_solid_angle_sr: NDArray[np.float64]
    status: NDArray[np.str_]


def _ray(
    detector_point_lab_m: ArrayLike,
    direction_lab: ArrayLike,
    ray_distance_m: float,
    status: ValidityCode,
) -> DetectorRay:
    return DetectorRay(
        finite_vector3(detector_point_lab_m, "detector_point_lab_m"),
        finite_vector3(direction_lab, "direction_lab"),
        float(ray_distance_m),
        status,
    )


def _inside(column_px: float, row_px: float, shape_rc: tuple[int, int]) -> bool:
    rows, columns = shape_rc
    return -0.5 <= column_px <= columns - 0.5 and -0.5 <= row_px <= rows - 0.5


def _project_detector_rays(
    origin_lab_m: ArrayLike,
    direction_lab: ArrayLike,
    instrument: CompiledInstrument,
) -> _DetectorProjectionArrays:
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
    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be a CompiledInstrument")

    detector_from_lab = instrument.lab_from_detector.inverse()
    origin_detector_m = detector_from_lab.apply_point(origins)
    direction_detector = detector_from_lab.apply_vector(directions)
    denominator = direction_detector[:, 2]
    offset_m = origin_detector_m[:, 2]
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
    point_detector_m = origin_detector_m + distance_m[:, None] * direction_detector
    reference_column, reference_row = instrument.detector_reference_coordinate_px
    column_px = reference_column + point_detector_m[:, 0] / instrument.detector_column_pitch_m
    row_px = reference_row + point_detector_m[:, 1] / instrument.detector_row_pitch_m
    rows, columns = instrument.detector_shape_rc
    active = status == ValidityCode.VALID
    column_lower_m = (-0.5 - reference_column) * instrument.detector_column_pitch_m
    column_upper_m = (columns - 0.5 - reference_column) * instrument.detector_column_pitch_m
    row_lower_m = (-0.5 - reference_row) * instrument.detector_row_pitch_m
    row_upper_m = (rows - 0.5 - reference_row) * instrument.detector_row_pitch_m
    outside = active & (
        (point_detector_m[:, 0] < column_lower_m)
        | (point_detector_m[:, 0] > column_upper_m)
        | (point_detector_m[:, 1] < row_lower_m)
        | (point_detector_m[:, 1] > row_upper_m)
    )
    status[outside] = ValidityCode.OUTSIDE_SUPPORT

    solid_angle_sr = np.zeros(origins.shape[0], dtype=np.float64)
    positive_distance = (status == ValidityCode.VALID) & (distance_m > 0.0)
    pixel_area_m2 = instrument.detector_column_pitch_m * instrument.detector_row_pitch_m
    solid_angle_sr[positive_distance] = (
        pixel_area_m2
        * np.abs(direction_detector[positive_distance, 2])
        / distance_m[positive_distance] ** 2
    )
    numeric_failure = (status == ValidityCode.VALID) & ~(
        np.isfinite(column_px) & np.isfinite(row_px) & np.isfinite(solid_angle_sr)
    )
    status[numeric_failure] = ValidityCode.NUMERIC_FAILURE
    valid = status == ValidityCode.VALID

    point_lab_output = np.zeros_like(origins)
    column_output = np.zeros(origins.shape[0], dtype=np.float64)
    row_output = np.zeros(origins.shape[0], dtype=np.float64)
    distance_output = np.zeros(origins.shape[0], dtype=np.float64)
    solid_angle_output = np.zeros(origins.shape[0], dtype=np.float64)
    point_lab_output[valid] = instrument.lab_from_detector.apply_point(point_detector_m[valid])
    column_output[valid] = column_px[valid]
    row_output[valid] = row_px[valid]
    distance_output[valid] = distance_m[valid]
    solid_angle_output[valid] = solid_angle_sr[valid]
    return _DetectorProjectionArrays(
        point_lab_output,
        column_output,
        row_output,
        distance_output,
        solid_angle_output,
        status,
    )


def project_detector_ray(
    origin_lab_m: ArrayLike,
    direction_lab: ArrayLike,
    instrument: CompiledInstrument,
) -> DetectorProjection:
    """Project a forward unit ray to the compiled detector without applying solid angle."""

    origin = finite_vector3(origin_lab_m, "origin_lab_m")
    direction = finite_vector3(direction_lab, "direction_lab")
    projections = _project_detector_rays(
        origin[None, :],
        direction[None, :],
        instrument,
    )
    return DetectorProjection(
        finite_vector3(projections.point_lab_m[0], "point_lab_m"),
        float(projections.column_px[0]),
        float(projections.row_px[0]),
        float(projections.ray_distance_m[0]),
        float(projections.pixel_solid_angle_sr[0]),
        ValidityCode(projections.status[0]),
    )


def detector_coordinate_to_ray(
    column_px: float,
    row_px: float,
    *,
    origin_lab_m: ArrayLike,
    instrument: CompiledInstrument,
) -> DetectorRay:
    """Return the normalized lab ray from ``origin_lab_m`` through a detector coordinate."""

    column = float(column_px)
    row = float(row_px)
    if not math.isfinite(column) or not math.isfinite(row):
        raise ValueError("detector coordinates must be finite")
    origin = finite_vector3(origin_lab_m, "origin_lab_m")
    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be a CompiledInstrument")
    if not _inside(column, row, instrument.detector_shape_rc):
        return _ray(
            np.zeros(3),
            np.zeros(3),
            0.0,
            ValidityCode.OUTSIDE_SUPPORT,
        )

    reference_column, reference_row = instrument.detector_reference_coordinate_px
    point_detector_m = np.array(
        [
            (column - reference_column) * instrument.detector_column_pitch_m,
            (row - reference_row) * instrument.detector_row_pitch_m,
            0.0,
        ]
    )
    detector_point_lab_m = instrument.lab_from_detector.apply_point(point_detector_m)
    displacement = detector_point_lab_m - origin
    distance_m = float(np.linalg.norm(displacement))
    if not math.isfinite(distance_m) or distance_m == 0.0:
        return _ray(np.zeros(3), np.zeros(3), 0.0, ValidityCode.NO_SOLUTION)
    return _ray(
        detector_point_lab_m,
        displacement / distance_m,
        distance_m,
        ValidityCode.VALID,
    )
