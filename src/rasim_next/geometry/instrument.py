"""Compilation of declared instrument motions into named rigid transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.geometry._vectors import finite_vector3


def _transform(
    value: object,
    name: str,
    source: FrameId,
    target: FrameId,
) -> RigidTransform:
    if not isinstance(value, RigidTransform):
        raise TypeError(f"{name} must be a RigidTransform")
    if value.source_frame != source or value.target_frame != target:
        raise ValueError(f"{name} must map {source} to {target}")
    return value


def _shape_rc(value: tuple[int, int]) -> tuple[int, int]:
    shape = tuple(value)
    if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("detector_shape_rc must contain positive integer row/column sizes")
    return shape


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _detector_reference(value: tuple[float, float]) -> tuple[float, float]:
    coordinate = tuple(float(item) for item in value)
    if len(coordinate) != 2 or not all(math.isfinite(item) for item in coordinate):
        raise ValueError("detector_reference_coordinate_px must contain finite (column_px, row_px)")
    return coordinate


@dataclass(frozen=True, slots=True)
class AxisRotation:
    """One active right-handed rotation about a unit lab axis and lab pivot."""

    axis_lab: NDArray[np.float64]
    angle_rad: float
    pivot_lab_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        axis = finite_vector3(self.axis_lab, "axis_lab")
        pivot = finite_vector3(self.pivot_lab_m, "pivot_lab_m")
        angle = float(self.angle_rad)
        if not math.isfinite(angle):
            raise ValueError("angle_rad must be finite")
        norm = float(np.linalg.norm(axis))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("axis_lab must have unit length")
        axis = axis / norm
        axis.setflags(write=False)
        object.__setattr__(self, "axis_lab", axis)
        object.__setattr__(self, "angle_rad", angle)
        object.__setattr__(self, "pivot_lab_m", pivot)


@dataclass(frozen=True, slots=True)
class InstrumentConfiguration:
    """Explicit zero poses, ordered motions, detector dimensions, and sample dimensions."""

    axis_rotations: tuple[AxisRotation, ...]
    lab_from_goniometer_zero: RigidTransform
    goniometer_from_sample: RigidTransform
    sample_from_crystal: RigidTransform
    lab_from_detector: RigidTransform
    detector_shape_rc: tuple[int, int]
    detector_row_pitch_m: float
    detector_column_pitch_m: float
    detector_reference_coordinate_px: tuple[float, float]
    sample_width_m: float
    sample_length_m: float
    film_thickness_A: float

    def __post_init__(self) -> None:
        rotations = tuple(self.axis_rotations)
        if any(not isinstance(rotation, AxisRotation) for rotation in rotations):
            raise TypeError("axis_rotations must contain AxisRotation values")
        object.__setattr__(self, "axis_rotations", rotations)
        expected_frames = (
            ("lab_from_goniometer_zero", FrameId.GONIOMETER, FrameId.LAB),
            ("goniometer_from_sample", FrameId.SAMPLE, FrameId.GONIOMETER),
            ("sample_from_crystal", FrameId.CRYSTAL, FrameId.SAMPLE),
            ("lab_from_detector", FrameId.DETECTOR, FrameId.LAB),
        )
        for name, source, target in expected_frames:
            _transform(getattr(self, name), name, source, target)

        object.__setattr__(self, "detector_shape_rc", _shape_rc(self.detector_shape_rc))
        object.__setattr__(
            self,
            "detector_reference_coordinate_px",
            _detector_reference(self.detector_reference_coordinate_px),
        )
        for name in (
            "detector_row_pitch_m",
            "detector_column_pitch_m",
            "sample_width_m",
            "sample_length_m",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        thickness = float(self.film_thickness_A)
        if not math.isfinite(thickness) or thickness < 0.0:
            raise ValueError("film_thickness_A must be finite and nonnegative")
        object.__setattr__(self, "film_thickness_A", thickness)


@dataclass(frozen=True, slots=True)
class CompiledInstrument:
    """Named transforms and dimensions consumed by T02 geometry and optics.

    Detector-local ``x``, ``y``, and ``z`` are column, row, and outward-normal axes. Its origin
    corresponds to ``detector_reference_coordinate_px`` in ``(column_px, row_px)`` order.
    """

    lab_from_goniometer: RigidTransform
    lab_from_sample: RigidTransform
    sample_from_crystal: RigidTransform
    lab_from_crystal: RigidTransform
    lab_from_detector: RigidTransform
    detector_shape_rc: tuple[int, int]
    detector_row_pitch_m: float
    detector_column_pitch_m: float
    detector_reference_coordinate_px: tuple[float, float]
    sample_width_m: float
    sample_length_m: float
    film_thickness_A: float

    def __post_init__(self) -> None:
        expected_frames = (
            ("lab_from_goniometer", FrameId.GONIOMETER, FrameId.LAB),
            ("lab_from_sample", FrameId.SAMPLE, FrameId.LAB),
            ("sample_from_crystal", FrameId.CRYSTAL, FrameId.SAMPLE),
            ("lab_from_crystal", FrameId.CRYSTAL, FrameId.LAB),
            ("lab_from_detector", FrameId.DETECTOR, FrameId.LAB),
        )
        for name, source, target in expected_frames:
            _transform(getattr(self, name), name, source, target)
        object.__setattr__(self, "detector_shape_rc", _shape_rc(self.detector_shape_rc))
        object.__setattr__(
            self,
            "detector_reference_coordinate_px",
            _detector_reference(self.detector_reference_coordinate_px),
        )
        for name in (
            "detector_row_pitch_m",
            "detector_column_pitch_m",
            "sample_width_m",
            "sample_length_m",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        thickness = float(self.film_thickness_A)
        if not math.isfinite(thickness) or thickness < 0.0:
            raise ValueError("film_thickness_A must be finite and nonnegative")
        object.__setattr__(self, "film_thickness_A", thickness)


def _rotation_matrix(rotation: AxisRotation) -> NDArray[np.float64]:
    x, y, z = rotation.axis_lab
    cosine = math.cos(rotation.angle_rad)
    sine = math.sin(rotation.angle_rad)
    complement = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * complement,
                x * y * complement - z * sine,
                x * z * complement + y * sine,
            ],
            [
                y * x * complement + z * sine,
                cosine + y * y * complement,
                y * z * complement - x * sine,
            ],
            [
                z * x * complement - y * sine,
                z * y * complement + x * sine,
                cosine + z * z * complement,
            ],
        ],
        dtype=np.float64,
    )


def compile_instrument(configuration: InstrumentConfiguration) -> CompiledInstrument:
    """Apply declared rotations in tuple order and compile target-from-source transforms."""

    if not isinstance(configuration, InstrumentConfiguration):
        raise TypeError("configuration must be an InstrumentConfiguration")
    lab_motion = RigidTransform.identity(FrameId.LAB)
    for rotation in configuration.axis_rotations:
        step = RigidTransform.around_pivot(
            rotation=_rotation_matrix(rotation),
            pivot_m=rotation.pivot_lab_m,
            frame=FrameId.LAB,
        )
        lab_motion = step.compose(lab_motion)

    lab_from_goniometer = lab_motion.compose(configuration.lab_from_goniometer_zero)
    lab_from_sample = lab_from_goniometer.compose(configuration.goniometer_from_sample)
    lab_from_crystal = lab_from_sample.compose(configuration.sample_from_crystal)
    return CompiledInstrument(
        lab_from_goniometer=lab_from_goniometer,
        lab_from_sample=lab_from_sample,
        sample_from_crystal=configuration.sample_from_crystal,
        lab_from_crystal=lab_from_crystal,
        lab_from_detector=configuration.lab_from_detector,
        detector_shape_rc=configuration.detector_shape_rc,
        detector_row_pitch_m=configuration.detector_row_pitch_m,
        detector_column_pitch_m=configuration.detector_column_pitch_m,
        detector_reference_coordinate_px=configuration.detector_reference_coordinate_px,
        sample_width_m=configuration.sample_width_m,
        sample_length_m=configuration.sample_length_m,
        film_thickness_A=configuration.film_thickness_A,
    )
