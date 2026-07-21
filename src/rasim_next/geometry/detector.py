"""Forward and inverse geometry for a rectangular detector plane."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.validity import ValidityCode
from rasim_next.geometry._vectors import finite_vector3, finite_vectors3
from rasim_next.geometry.instrument import CompiledInstrument

_DETECTOR_INCIDENCE_COSINE_TOL = 1e-14
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
class DetectorProjectionBatch:
    """Aligned immutable results for a batch of detector-ray projections."""

    point_lab_m: NDArray[np.float64]
    column_px: NDArray[np.float64]
    row_px: NDArray[np.float64]
    ray_distance_m: NDArray[np.float64]
    pixel_solid_angle_sr: NDArray[np.float64]
    status: NDArray[np.str_]
    valid: NDArray[np.bool_] = field(init=False)

    def __post_init__(self) -> None:
        points = np.array(self.point_lab_m, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point_lab_m must have shape (N, 3)")
        size = points.shape[0]
        arrays: dict[str, NDArray[np.float64]] = {}
        for name in (
            "column_px",
            "row_px",
            "ray_distance_m",
            "pixel_solid_angle_sr",
        ):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != (size,):
                raise ValueError(f"{name} must have shape {(size,)}")
            arrays[name] = value
        if not np.all(np.isfinite(points)) or any(
            not np.all(np.isfinite(value)) for value in arrays.values()
        ):
            raise ValueError("detector projection values must be finite")
        if np.any(arrays["ray_distance_m"] < 0.0):
            raise ValueError("ray_distance_m must be nonnegative")
        if np.any(arrays["pixel_solid_angle_sr"] < 0.0):
            raise ValueError("pixel_solid_angle_sr must be nonnegative")

        supplied_status = np.asarray(self.status)
        if supplied_status.shape != (size,):
            raise ValueError(f"status must have shape {(size,)}")
        status = np.asarray(
            [ValidityCode(item).value for item in supplied_status],
            dtype="U32",
        )
        valid = status == ValidityCode.VALID
        for value in (points, *arrays.values(), status, valid):
            value.setflags(write=False)
        object.__setattr__(self, "point_lab_m", points)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "valid", valid)


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


@dataclass(frozen=True, slots=True)
class _DetectorPlaneIntersections:
    point_detector_m: NDArray[np.float64]
    direction_detector: NDArray[np.float64]
    distance_m: NDArray[np.float64]
    column_px: NDArray[np.float64]
    row_px: NDArray[np.float64]
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


def _detector_coordinates_to_lab_points(
    column_px: ArrayLike,
    row_px: ArrayLike,
    instrument: CompiledInstrument,
) -> NDArray[np.float64]:
    """Map broadcast-compatible continuous detector coordinates to lab points."""

    columns, rows = np.broadcast_arrays(
        np.asarray(column_px, dtype=np.float64),
        np.asarray(row_px, dtype=np.float64),
    )
    reference_column, reference_row = instrument.detector_reference_coordinate_px
    point_detector_m = np.stack(
        (
            (columns - reference_column) * instrument.detector_column_pitch_m,
            (rows - reference_row) * instrument.detector_row_pitch_m,
            np.zeros_like(columns),
        ),
        axis=-1,
    )
    return instrument.lab_from_detector.apply_point(point_detector_m)


def _detector_incidence_cosine(
    direction_lab: ArrayLike,
    instrument: CompiledInstrument,
) -> NDArray[np.float64]:
    """Return signed incidence on the detector's outward normal."""

    directions = np.asarray(direction_lab, dtype=np.float64)
    if directions.shape[-1:] != (3,):
        raise ValueError("direction_lab must end with a length-3 coordinate axis")
    return np.asarray(directions @ instrument.lab_from_detector.rotation[:, 2])


def _intersect_detector_plane(
    origin_lab_m: ArrayLike,
    direction_lab: ArrayLike,
    instrument: CompiledInstrument,
) -> _DetectorPlaneIntersections:
    """Return unbounded detector-plane intersections in native continuous coordinates."""

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
    parallel = np.abs(denominator) <= _DETECTOR_INCIDENCE_COSINE_TOL
    status = np.full(origins.shape[0], ValidityCode.VALID, dtype="U16")
    status[parallel] = ValidityCode.PARALLEL

    distance_m = np.zeros(origins.shape[0], dtype=np.float64)
    nonparallel = ~parallel
    distance_m[nonparallel] = -offset_m[nonparallel] / denominator[nonparallel]
    backward = nonparallel & (distance_m < -_POSITION_TOL_M)
    status[backward] = ValidityCode.BACKWARD
    no_forward_distance = nonparallel & ~backward & (distance_m <= 0.0)
    status[no_forward_distance] = ValidityCode.NO_SOLUTION
    back_facing = (status == ValidityCode.VALID) & (denominator < -_DETECTOR_INCIDENCE_COSINE_TOL)
    status[back_facing] = ValidityCode.BACKWARD
    distance_m = np.maximum(distance_m, 0.0)
    point_detector_m = origin_detector_m + distance_m[:, None] * direction_detector
    reference_column, reference_row = instrument.detector_reference_coordinate_px
    column_px = reference_column + point_detector_m[:, 0] / instrument.detector_column_pitch_m
    row_px = reference_row + point_detector_m[:, 1] / instrument.detector_row_pitch_m
    return _DetectorPlaneIntersections(
        point_detector_m=point_detector_m,
        direction_detector=direction_detector,
        distance_m=distance_m,
        column_px=column_px,
        row_px=row_px,
        status=status,
    )


def _project_detector_rays(
    origin_lab_m: ArrayLike,
    direction_lab: ArrayLike,
    instrument: CompiledInstrument,
    *,
    support_tolerance_px: float = 0.0,
) -> _DetectorProjectionArrays:
    support_tolerance = float(support_tolerance_px)
    if not math.isfinite(support_tolerance) or support_tolerance < 0.0:
        raise ValueError("support_tolerance_px must be finite and nonnegative")
    intersections = _intersect_detector_plane(origin_lab_m, direction_lab, instrument)
    point_detector_m = intersections.point_detector_m
    direction_detector = intersections.direction_detector
    distance_m = intersections.distance_m
    column_px = np.array(intersections.column_px, copy=True)
    row_px = np.array(intersections.row_px, copy=True)
    status = np.array(intersections.status, copy=True)
    origin_count = point_detector_m.shape[0]
    reference_column, reference_row = instrument.detector_reference_coordinate_px
    rows, columns = instrument.detector_shape_rc
    active = status == ValidityCode.VALID
    column_lower_m = (
        -0.5 - support_tolerance - reference_column
    ) * instrument.detector_column_pitch_m
    column_upper_m = (
        columns - 0.5 + support_tolerance - reference_column
    ) * instrument.detector_column_pitch_m
    row_lower_m = (-0.5 - support_tolerance - reference_row) * instrument.detector_row_pitch_m
    row_upper_m = (rows - 0.5 + support_tolerance - reference_row) * instrument.detector_row_pitch_m
    outside = active & (
        (point_detector_m[:, 0] < column_lower_m)
        | (point_detector_m[:, 0] > column_upper_m)
        | (point_detector_m[:, 1] < row_lower_m)
        | (point_detector_m[:, 1] > row_upper_m)
    )
    status[outside] = ValidityCode.OUTSIDE_SUPPORT
    active = status == ValidityCode.VALID
    column_px[active] = np.clip(column_px[active], -0.5, columns - 0.5)
    row_px[active] = np.clip(row_px[active], -0.5, rows - 0.5)

    solid_angle_sr = np.zeros(origin_count, dtype=np.float64)
    positive_distance = (status == ValidityCode.VALID) & (distance_m > 0.0)
    pixel_area_m2 = instrument.detector_column_pitch_m * instrument.detector_row_pitch_m
    solid_angle_sr[positive_distance] = (
        pixel_area_m2
        * direction_detector[positive_distance, 2]
        / distance_m[positive_distance] ** 2
    )
    numeric_failure = (status == ValidityCode.VALID) & ~(
        np.isfinite(column_px) & np.isfinite(row_px) & np.isfinite(solid_angle_sr)
    )
    status[numeric_failure] = ValidityCode.NUMERIC_FAILURE
    valid = status == ValidityCode.VALID

    point_lab_output = np.zeros((origin_count, 3), dtype=np.float64)
    column_output = np.zeros(origin_count, dtype=np.float64)
    row_output = np.zeros(origin_count, dtype=np.float64)
    distance_output = np.zeros(origin_count, dtype=np.float64)
    solid_angle_output = np.zeros(origin_count, dtype=np.float64)
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


def project_detector_rays(
    origin_lab_m: ArrayLike,
    direction_lab: ArrayLike,
    instrument: CompiledInstrument,
) -> DetectorProjectionBatch:
    """Project aligned forward unit rays without applying detector solid angle."""

    projections = _project_detector_rays(origin_lab_m, direction_lab, instrument)
    return DetectorProjectionBatch(
        point_lab_m=projections.point_lab_m,
        column_px=projections.column_px,
        row_px=projections.row_px,
        ray_distance_m=projections.ray_distance_m,
        pixel_solid_angle_sr=projections.pixel_solid_angle_sr,
        status=projections.status,
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

    detector_point_lab_m = _detector_coordinates_to_lab_points(column, row, instrument)
    displacement = detector_point_lab_m - origin
    distance_m = float(np.linalg.norm(displacement))
    if not math.isfinite(distance_m) or distance_m == 0.0:
        return _ray(np.zeros(3), np.zeros(3), 0.0, ValidityCode.NO_SOLUTION)
    direction_lab = displacement / distance_m
    incidence_cosine = float(_detector_incidence_cosine(direction_lab, instrument))
    if abs(incidence_cosine) <= _DETECTOR_INCIDENCE_COSINE_TOL:
        return _ray(np.zeros(3), np.zeros(3), 0.0, ValidityCode.PARALLEL)
    if incidence_cosine < 0.0:
        return _ray(np.zeros(3), np.zeros(3), 0.0, ValidityCode.BACKWARD)
    return _ray(
        detector_point_lab_m,
        direction_lab,
        distance_m,
        ValidityCode.VALID,
    )
