"""Joint hBN and multi-material indexed-image geometry fitting."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.fitting.geometry import (
    ExactTagGeometryModel,
    GeometryPredictionError,
    IntegerLMarkerObservations,
    evaluate_layer_l_geometry_objective_residual,
    evaluate_tagged_geometry_objective_residual,
)
from rasim_next.fitting.hbn import (
    HbnDetectorCalibration,
    HbnRingObservations,
    evaluate_hbn_residual_px,
)
from rasim_next.fitting.indexed_series import (
    IndexedGeometryImage,
    SharedGeometryCorrections,
    apply_shared_geometry_corrections,
)
from rasim_next.geometry.instrument import CompiledInstrument, compose_intrinsic_xy_rotation
from rasim_next.pipeline.configured_simulation import (
    SimulationConfiguration,
    rebind_configured_geometry_instrument,
    sample_configured_nominal_geometry_source,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
SpecimenId = Literal["bi2se3", "bi2te3", "pbi2_y1", "pbi2_y2"]

JOINT_GEOMETRY_PARAMETER_NAMES = (
    "detector_column_tilt_rad",
    "detector_row_tilt_rad",
    "beam_center_column_px",
    "beam_center_row_px",
    "goniometer_axis_pitch_rad",
    "goniometer_axis_yaw_rad",
    "goniometer_pivot_pitch_offset_m",
    "goniometer_pivot_yaw_offset_m",
    "incidence_angle_delta_rad",
    "bi2se3_sample_y_tilt_rad",
    "bi2se3_zs_m",
    "bi2te3_sample_x_tilt_rad",
    "bi2te3_sample_y_tilt_rad",
    "bi2te3_zs_m",
    "pbi2_y1_sample_x_tilt_rad",
    "pbi2_y1_sample_y_tilt_rad",
    "pbi2_y1_zs_m",
    "pbi2_y2_sample_x_tilt_rad",
    "pbi2_y2_sample_y_tilt_rad",
    "pbi2_y2_zs_m",
    "hbn_calibrant_distance_m",
)

GLOBAL_PARAMETER_NAMES = JOINT_GEOMETRY_PARAMETER_NAMES[:9]
LOCAL_PARAMETER_NAMES = JOINT_GEOMETRY_PARAMETER_NAMES[9:20]
NUISANCE_PARAMETER_NAMES = JOINT_GEOMETRY_PARAMETER_NAMES[20:]

DEFAULT_FIXED_REFERENCE_PARAMETERS = (
    ("goniometer_axis_pitch_rad", 0.0),
    ("goniometer_pivot_pitch_offset_m", 0.0),
)
_DEFAULT_FIXED_PARAMETER_NAMES = frozenset(name for name, _ in DEFAULT_FIXED_REFERENCE_PARAMETERS)
DEFAULT_FITTED_PARAMETER_NAMES = tuple(
    name for name in JOINT_GEOMETRY_PARAMETER_NAMES if name not in _DEFAULT_FIXED_PARAMETER_NAMES
)


@dataclass(frozen=True, slots=True)
class JointGeometryState:
    detector_column_tilt_rad: float
    detector_row_tilt_rad: float
    beam_center_column_px: float
    beam_center_row_px: float
    goniometer_axis_pitch_rad: float
    goniometer_axis_yaw_rad: float
    goniometer_pivot_pitch_offset_m: float
    goniometer_pivot_yaw_offset_m: float
    incidence_angle_delta_rad: float
    bi2se3_sample_y_tilt_rad: float
    bi2se3_zs_m: float
    bi2te3_sample_x_tilt_rad: float
    bi2te3_sample_y_tilt_rad: float
    bi2te3_zs_m: float
    pbi2_y1_sample_x_tilt_rad: float
    pbi2_y1_sample_y_tilt_rad: float
    pbi2_y1_zs_m: float
    pbi2_y2_sample_x_tilt_rad: float
    pbi2_y2_sample_y_tilt_rad: float
    pbi2_y2_zs_m: float
    hbn_calibrant_distance_m: float

    def __post_init__(self) -> None:
        for name in JOINT_GEOMETRY_PARAMETER_NAMES:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    @classmethod
    def from_array(cls, values: ArrayLike) -> JointGeometryState:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(JOINT_GEOMETRY_PARAMETER_NAMES),) or not np.all(np.isfinite(array)):
            raise ValueError("joint geometry state has the wrong shape or nonfinite values")
        return cls(*(float(value) for value in array))

    @classmethod
    def from_hbn(cls, calibration: HbnDetectorCalibration) -> JointGeometryState:
        if not isinstance(calibration, HbnDetectorCalibration):
            raise TypeError("calibration must be HbnDetectorCalibration")
        return cls(
            calibration.detector_column_tilt_rad,
            calibration.detector_row_tilt_rad,
            calibration.beam_center_column_px,
            calibration.beam_center_row_px,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            calibration.calibrant_distance_m,
        )

    def as_array(self) -> FloatArray:
        result = np.asarray(
            tuple(getattr(self, name) for name in JOINT_GEOMETRY_PARAMETER_NAMES),
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class JointGeometryBounds:
    lower: JointGeometryState
    upper: JointGeometryState

    def __post_init__(self) -> None:
        if not isinstance(self.lower, JointGeometryState) or not isinstance(
            self.upper,
            JointGeometryState,
        ):
            raise TypeError("joint geometry bounds require two states")
        if np.any(self.lower.as_array() >= self.upper.as_array()):
            raise ValueError("every joint geometry lower bound must precede its upper bound")

    @classmethod
    def around_hbn(cls, calibration: HbnDetectorCalibration) -> JointGeometryBounds:
        center_column = calibration.beam_center_column_px
        center_row = calibration.beam_center_row_px
        angle = math.radians(5.0)
        incidence = math.radians(0.5)
        lower = JointGeometryState(
            -0.15,
            -0.15,
            center_column - 20.0,
            center_row - 20.0,
            -angle,
            -angle,
            -1.0e-3,
            -1.0e-3,
            -incidence,
            -angle,
            -5.0e-4,
            -angle,
            -angle,
            -5.0e-4,
            -angle,
            -angle,
            -5.0e-4,
            -angle,
            -angle,
            -5.0e-4,
            0.04,
        )
        upper = JointGeometryState(
            0.15,
            0.15,
            center_column + 20.0,
            center_row + 20.0,
            angle,
            angle,
            1.0e-3,
            1.0e-3,
            incidence,
            angle,
            5.0e-4,
            angle,
            angle,
            5.0e-4,
            angle,
            angle,
            5.0e-4,
            angle,
            angle,
            5.0e-4,
            0.12,
        )
        return cls(lower, upper)

    @property
    def half_span(self) -> FloatArray:
        result = 0.5 * (self.upper.as_array() - self.lower.as_array())
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class JointGeometryImageMetric:
    specimen_id: SpecimenId
    image_id: str
    site_count: int
    site_rms_px: float
    site_max_px: float


@dataclass(frozen=True, slots=True)
class JointGeometryFitResult:
    state: JointGeometryState
    success: bool
    confidence_qualified: bool
    message: str
    standard_error: FloatArray
    parameter_confident: BoolArray
    jacobian_rank: int
    scaled_jacobian_condition: float
    scaled_jacobian_singular_values: FloatArray
    weakest_direction: FloatArray
    active_bounds: BoolArray
    hbn_residual_rms_px: float
    hbn_residual_max_px: float
    per_image: tuple[JointGeometryImageMetric, ...]
    pooled_crystalline_site_rms_px: float
    pooled_crystalline_site_max_px: float
    model_evaluation_count: int
    optimizer_function_evaluation_count: int
    beam_origin_lab_m: FloatArray
    corrected_goniometer_axis_lab: FloatArray
    corrected_goniometer_pivot_lab_m: FloatArray
    z_b_m: float
    z_b_standard_error_m: float
    fitted_parameter_names: tuple[str, ...]
    fixed_reference_parameters: tuple[tuple[str, float], ...]
    unobserved_specimen_parameters: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(JOINT_GEOMETRY_PARAMETER_NAMES)
        standard_error = np.asarray(self.standard_error, dtype=np.float64)
        confident = np.asarray(self.parameter_confident, dtype=np.bool_)
        singular = np.asarray(self.scaled_jacobian_singular_values, dtype=np.float64)
        weakest = np.asarray(self.weakest_direction, dtype=np.float64)
        active = np.asarray(self.active_bounds, dtype=np.bool_)
        origin = np.asarray(self.beam_origin_lab_m, dtype=np.float64)
        axis = np.asarray(self.corrected_goniometer_axis_lab, dtype=np.float64)
        pivot = np.asarray(self.corrected_goniometer_pivot_lab_m, dtype=np.float64)
        fixed_names = tuple(name for name, _ in self.fixed_reference_parameters)
        excluded_names = set(fixed_names) | set(self.unobserved_specimen_parameters)
        expected_fitted_names = tuple(
            name for name in JOINT_GEOMETRY_PARAMETER_NAMES if name not in excluded_names
        )
        if (
            standard_error.shape != (count,)
            or confident.shape != (count,)
            or singular.shape != (len(self.fitted_parameter_names),)
            or weakest.shape != (count,)
            or active.shape != (count,)
            or origin.shape != (3,)
            or axis.shape != (3,)
            or pivot.shape != (3,)
        ):
            raise ValueError("joint geometry result arrays have invalid shapes")
        if fixed_names != tuple(
            name for name in JOINT_GEOMETRY_PARAMETER_NAMES if name in fixed_names
        ):
            raise ValueError("fixed reference parameters must use canonical parameter order")
        if self.fitted_parameter_names != expected_fitted_names:
            raise ValueError("fitted parameters must exclude references and absent specimens")
        observed_specimens = {metric.specimen_id for metric in self.per_image}
        expected_unobserved = tuple(
            name
            for name in LOCAL_PARAMETER_NAMES
            if (name.startswith("pbi2_y1_") and "pbi2_y1" not in observed_specimens)
            or (name.startswith("pbi2_y2_") and "pbi2_y2" not in observed_specimens)
        )
        if self.unobserved_specimen_parameters != expected_unobserved:
            raise ValueError("unobserved parameters disagree with the image roster")
        for value in (standard_error, confident, singular, weakest, active, origin, axis, pivot):
            value.setflags(write=False)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "parameter_confident", confident)
        object.__setattr__(self, "scaled_jacobian_singular_values", singular)
        object.__setattr__(self, "weakest_direction", weakest)
        object.__setattr__(self, "active_bounds", active)
        object.__setattr__(self, "beam_origin_lab_m", origin)
        object.__setattr__(self, "corrected_goniometer_axis_lab", axis)
        object.__setattr__(self, "corrected_goniometer_pivot_lab_m", pivot)


def _axis_and_pivot(
    instrument: CompiledInstrument,
    image: IndexedGeometryImage,
    state: JointGeometryState,
) -> tuple[FloatArray, FloatArray]:
    configured = image.model.inputs.config.instrument.axis_rotations[0]
    base_axis = np.asarray(configured.axis_lab, dtype=np.float64)
    horizontal = math.hypot(float(base_axis[0]), float(base_axis[1]))
    base_pitch = math.atan2(float(base_axis[2]), horizontal)
    base_yaw = math.atan2(-float(base_axis[1]), float(base_axis[0]))
    pitch = base_pitch + state.goniometer_axis_pitch_rad
    yaw = base_yaw + state.goniometer_axis_yaw_rad
    cosine_pitch = math.cos(pitch)
    axis = np.asarray(
        (math.cos(yaw) * cosine_pitch, -math.sin(yaw) * cosine_pitch, math.sin(pitch))
    )
    pitch_tangent = np.asarray(
        (-math.cos(yaw) * math.sin(pitch), math.sin(yaw) * math.sin(pitch), math.cos(pitch))
    )
    yaw_tangent = np.asarray((-math.sin(yaw), -math.cos(yaw), 0.0))
    pivot = (
        np.asarray(configured.pivot_lab_m, dtype=np.float64)
        + state.goniometer_pivot_pitch_offset_m * pitch_tangent
        + state.goniometer_pivot_yaw_offset_m * yaw_tangent
    )
    return axis, pivot


def specimen_local_geometry(
    specimen_id: SpecimenId, state: JointGeometryState
) -> tuple[float, float, float]:
    """Return sample-x tilt, sample-y tilt, and local plane offset."""

    if specimen_id == "bi2se3":
        return 0.0, state.bi2se3_sample_y_tilt_rad, state.bi2se3_zs_m
    if specimen_id == "bi2te3":
        return state.bi2te3_sample_x_tilt_rad, state.bi2te3_sample_y_tilt_rad, state.bi2te3_zs_m
    if specimen_id == "pbi2_y1":
        return state.pbi2_y1_sample_x_tilt_rad, state.pbi2_y1_sample_y_tilt_rad, state.pbi2_y1_zs_m
    if specimen_id == "pbi2_y2":
        return state.pbi2_y2_sample_x_tilt_rad, state.pbi2_y2_sample_y_tilt_rad, state.pbi2_y2_zs_m
    raise ValueError(f"unknown specimen_id {specimen_id!r}")


def joint_beam_origin_lab_m(
    state: JointGeometryState,
    detector_base: SimulationConfiguration,
    base_detector_rotation: FloatArray | None = None,
) -> FloatArray:
    """Project the fitted detector beam center onto the declared source line."""

    base_instrument = detector_base.instrument
    detector_rotation = compose_intrinsic_xy_rotation(
        (
            base_instrument.lab_from_detector.rotation
            if base_detector_rotation is None
            else base_detector_rotation
        ),
        state.detector_column_tilt_rad,
        state.detector_row_tilt_rad,
    )
    reference_column, reference_row = base_instrument.detector_reference_coordinate_px
    detector_offset_m = np.asarray(
        (
            (state.beam_center_column_px - reference_column)
            * base_instrument.detector_column_pitch_m,
            (state.beam_center_row_px - reference_row) * base_instrument.detector_row_pitch_m,
            0.0,
        )
    )
    beam_hit_lab_m = (
        base_instrument.lab_from_detector.translation_m + detector_rotation @ detector_offset_m
    )
    direction = np.asarray(detector_base.source.mean_direction_lab, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    nominal_origin = np.asarray(detector_base.source.mean_origin_lab_m, dtype=np.float64)
    return beam_hit_lab_m - direction * float((beam_hit_lab_m - nominal_origin) @ direction)


def _absolute_instrument_and_model(
    specimen_id: SpecimenId,
    image: IndexedGeometryImage,
    state: JointGeometryState,
    *,
    base_detector_rotation: FloatArray,
) -> tuple[ExactTagGeometryModel, CompiledInstrument, FloatArray]:
    config = image.model.inputs.config
    configured_axis = config.instrument.axis_rotations[0]
    shifted_config = replace(
        config,
        instrument=replace(
            config.instrument,
            axis_rotations=(
                replace(
                    configured_axis,
                    angle_deg=(
                        configured_axis.angle_deg + math.degrees(state.incidence_angle_delta_rad)
                    ),
                ),
            ),
        ),
    )
    inputs = rebind_configured_geometry_instrument(image.model.inputs, shifted_config)
    detector = inputs.instrument.lab_from_detector
    detector_rotation = compose_intrinsic_xy_rotation(
        base_detector_rotation,
        state.detector_column_tilt_rad,
        state.detector_row_tilt_rad,
    )
    instrument = replace(
        inputs.instrument,
        lab_from_detector=RigidTransform(
            detector_rotation,
            detector.translation_m,
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
    )
    beam_origin = joint_beam_origin_lab_m(state, shifted_config, base_detector_rotation)
    source = replace(shifted_config.source, mean_origin_lab_m=tuple(float(v) for v in beam_origin))
    shifted_config = replace(shifted_config, source=source)
    inputs = replace(
        inputs,
        config=shifted_config,
        samples=sample_configured_nominal_geometry_source(source),
    )
    sample_x_tilt, sample_y_tilt, z_s_m = specimen_local_geometry(specimen_id, state)
    corrections = SharedGeometryCorrections(
        detector_column_tilt_rad=0.0,
        detector_row_tilt_rad=0.0,
        sample_normal_x_tilt_rad=sample_x_tilt,
        sample_normal_y_tilt_rad=sample_y_tilt,
        goniometer_axis_pitch_rad=state.goniometer_axis_pitch_rad,
        goniometer_axis_yaw_rad=state.goniometer_axis_yaw_rad,
        sample_plane_normal_offset_m=z_s_m,
        goniometer_pivot_pitch_offset_m=state.goniometer_pivot_pitch_offset_m,
        goniometer_pivot_yaw_offset_m=state.goniometer_pivot_yaw_offset_m,
    )
    instrument = apply_shared_geometry_corrections(
        instrument,
        shifted_config.instrument.axis_rotations,
        corrections,
    )
    return ExactTagGeometryModel(inputs), instrument, beam_origin


def _predict_image(
    specimen_id: SpecimenId,
    image: IndexedGeometryImage,
    state: JointGeometryState,
    *,
    base_detector_rotation: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    model, instrument, beam_origin = _absolute_instrument_and_model(
        specimen_id,
        image,
        state,
        base_detector_rotation=base_detector_rotation,
    )
    observations = image.observations
    if isinstance(observations, IntegerLMarkerObservations):
        prediction = model.predict_integer_l_tags(observations.keys, instrument=instrument)
        residual = evaluate_tagged_geometry_objective_residual(observations, prediction)
    else:
        prediction = model.predict_layer_l_tags(observations.definitions, instrument=instrument)
        residual = evaluate_layer_l_geometry_objective_residual(observations, prediction)
    if not np.all(prediction.active_panel):
        raise ValueError(f"frozen marker topology changed for {image.image_id}")
    site_error = prediction.coordinates_px - observations.coordinates_px
    return residual, site_error, beam_origin


def evaluate_joint_geometry_residual(
    state: JointGeometryState,
    *,
    hbn_observations: HbnRingObservations,
    bi2se3_images: tuple[IndexedGeometryImage, ...],
    bi2te3_images: tuple[IndexedGeometryImage, ...],
    pbi2_y1_images: tuple[IndexedGeometryImage, ...],
    pbi2_y2_images: tuple[IndexedGeometryImage, ...],
    base_detector_rotation: ArrayLike,
) -> FloatArray:
    """Evaluate hBN and the supplied indexed specimen series in one residual vector."""

    if not isinstance(state, JointGeometryState):
        raise TypeError("state must be JointGeometryState")
    rotation = np.asarray(base_detector_rotation, dtype=np.float64)
    images = (
        tuple(bi2se3_images) + tuple(bi2te3_images) + tuple(pbi2_y1_images) + tuple(pbi2_y2_images)
    )
    if not images:
        raise ValueError("joint geometry fit requires indexed material images")
    reference = images[0].model.instrument
    hbn_values = np.asarray(
        (
            state.detector_column_tilt_rad,
            state.detector_row_tilt_rad,
            state.beam_center_column_px,
            state.beam_center_row_px,
            state.hbn_calibrant_distance_m,
        )
    )
    blocks = [
        evaluate_hbn_residual_px(
            hbn_values,
            hbn_observations,
            base_detector_rotation=rotation,
            beam_direction_lab=images[0].model.inputs.config.source.mean_direction_lab,
            detector_column_pitch_m=reference.detector_column_pitch_m,
            detector_row_pitch_m=reference.detector_row_pitch_m,
        )
    ]
    for specimen_id, specimen_images in (
        ("bi2se3", tuple(bi2se3_images)),
        ("bi2te3", tuple(bi2te3_images)),
        ("pbi2_y1", tuple(pbi2_y1_images)),
        ("pbi2_y2", tuple(pbi2_y2_images)),
    ):
        for image in specimen_images:
            blocks.append(
                _predict_image(
                    specimen_id,
                    image,
                    state,
                    base_detector_rotation=rotation,
                )[0]
            )
    result = np.concatenate(blocks)
    result.setflags(write=False)
    return result


def _finite_jacobian(
    function: object,
    values: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    steps: FloatArray,
) -> FloatArray:
    baseline = np.asarray(function(values), dtype=np.float64)  # type: ignore[operator]
    jacobian = np.empty((baseline.size, values.size), dtype=np.float64)
    for index, requested_step in enumerate(steps):
        step = min(float(requested_step), 0.2 * float(upper[index] - lower[index]))
        for _ in range(12):
            forward_value = None
            backward_value = None
            if values[index] + step <= upper[index]:
                forward = values.copy()
                forward[index] += step
                with suppress(GeometryPredictionError):
                    forward_value = np.asarray(function(forward))  # type: ignore[operator]
            if values[index] - step >= lower[index]:
                backward = values.copy()
                backward[index] -= step
                with suppress(GeometryPredictionError):
                    backward_value = np.asarray(function(backward))  # type: ignore[operator]
            if forward_value is not None and backward_value is not None:
                jacobian[:, index] = (forward_value - backward_value) / (2.0 * step)
                break
            if forward_value is not None:
                jacobian[:, index] = (forward_value - baseline) / step
                break
            if backward_value is not None:
                jacobian[:, index] = (baseline - backward_value) / step
                break
            step *= 0.5
        else:
            raise GeometryPredictionError(
                f"no topology-preserving finite-difference step for parameter {index}"
            )
    return jacobian


def validate_joint_geometry_start(
    initial: JointGeometryState,
    bounds: JointGeometryBounds,
    *,
    unobserved_parameters: tuple[str, ...] = (),
) -> None:
    """Validate declared starts without replacing adjustable or fixed coordinates."""
    if not isinstance(initial, JointGeometryState) or not isinstance(bounds, JointGeometryBounds):
        raise TypeError("joint initial state and bounds must use their canonical types")
    values, lower, upper = initial.as_array(), bounds.lower.as_array(), bounds.upper.as_array()
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("joint starting values lie outside the declared bounds")
    if lower[-1] <= 0.0:
        raise ValueError("private hBN calibrant distance bounds must be positive metres")
    for name, expected in DEFAULT_FIXED_REFERENCE_PARAMETERS:
        if getattr(initial, name) != expected:
            raise ValueError(f"joint fixed reference {name} must remain {expected}")
    if any(name not in LOCAL_PARAMETER_NAMES for name in unobserved_parameters):
        raise ValueError("unknown unobserved joint parameter")
    for name in unobserved_parameters:
        if getattr(initial, name) != 0.0:
            raise ValueError(f"unobserved joint reference {name} must remain zero")


def fit_joint_geometry(
    *,
    hbn_observations: HbnRingObservations,
    hbn_calibration: HbnDetectorCalibration,
    bi2se3_images: tuple[IndexedGeometryImage, ...],
    bi2te3_images: tuple[IndexedGeometryImage, ...],
    pbi2_y1_images: tuple[IndexedGeometryImage, ...],
    pbi2_y2_images: tuple[IndexedGeometryImage, ...],
    base_detector_rotation: ArrayLike,
    bounds: JointGeometryBounds | None = None,
    initial: JointGeometryState | None = None,
    checkpoint: Callable[[str], None] | None = None,
) -> JointGeometryFitResult:
    """Fit the identifiable geometry after fixing the declared mechanical references."""

    if not isinstance(hbn_observations, HbnRingObservations) or not isinstance(
        hbn_calibration,
        HbnDetectorCalibration,
    ):
        raise TypeError("hBN observations and calibration have invalid types")
    se3 = tuple(bi2se3_images)
    te3 = tuple(bi2te3_images)
    y1 = tuple(pbi2_y1_images)
    y2 = tuple(pbi2_y2_images)
    if not se3 or not te3:
        raise ValueError("Bi2Se3 and Bi2Te3 image series are required")
    rotation = np.asarray(base_detector_rotation, dtype=np.float64)
    explicit_initial = initial is not None
    if initial is None:
        initial = JointGeometryState.from_hbn(hbn_calibration)
    if not isinstance(initial, JointGeometryState):
        raise TypeError("initial must be a JointGeometryState")
    active_bounds = JointGeometryBounds.around_hbn(hbn_calibration) if bounds is None else bounds
    lower = active_bounds.lower.as_array()
    upper = active_bounds.upper.as_array()
    initial_values = np.array(initial.as_array(), copy=True)
    fixed_reference = dict(DEFAULT_FIXED_REFERENCE_PARAMETERS)
    unobserved = tuple(
        name
        for name in LOCAL_PARAMETER_NAMES
        if (name.startswith("pbi2_y1_") and not y1) or (name.startswith("pbi2_y2_") and not y2)
    )
    validate_joint_geometry_start(initial, active_bounds, unobserved_parameters=unobserved)
    for name, value in fixed_reference.items():
        if not explicit_initial:
            initial_values[JOINT_GEOMETRY_PARAMETER_NAMES.index(name)] = value

    def check(phase: str) -> None:
        if checkpoint is not None:
            checkpoint(phase)

    check("initial residual")
    fitted_indices = np.asarray(
        [
            index
            for index, name in enumerate(JOINT_GEOMETRY_PARAMETER_NAMES)
            if name not in fixed_reference and name not in unobserved
        ],
        dtype=np.int64,
    )
    half_span = active_bounds.half_span
    scale = np.asarray(
        (
            0.02,
            0.02,
            5.0,
            5.0,
            0.01,
            0.01,
            3.0e-4,
            3.0e-4,
            math.radians(0.1),
            0.01,
            1.0e-4,
            0.01,
            0.01,
            1.0e-4,
            0.01,
            0.01,
            1.0e-4,
            0.01,
            0.01,
            1.0e-4,
            2.0e-3,
        )
    )
    steps = np.asarray(
        (
            1.0e-5,
            1.0e-5,
            0.01,
            0.01,
            1.0e-5,
            1.0e-5,
            1.0e-6,
            1.0e-6,
            1.0e-5,
            1.0e-5,
            1.0e-6,
            1.0e-5,
            1.0e-5,
            1.0e-6,
            1.0e-5,
            1.0e-5,
            1.0e-6,
            1.0e-5,
            1.0e-5,
            1.0e-6,
            1.0e-5,
        )
    )

    def expand_fitted_values(fitted_values: FloatArray) -> FloatArray:
        values = initial_values.copy()
        values[fitted_indices] = fitted_values
        return values

    evaluation_count = 1
    residual_size = evaluate_joint_geometry_residual(
        JointGeometryState.from_array(initial_values),
        hbn_observations=hbn_observations,
        bi2se3_images=se3,
        bi2te3_images=te3,
        pbi2_y1_images=y1,
        pbi2_y2_images=y2,
        base_detector_rotation=rotation,
    ).size

    def raw_residual_array(fitted_values: FloatArray) -> FloatArray:
        nonlocal evaluation_count
        evaluation_count += 1
        check(f"residual evaluation {evaluation_count}")
        return np.array(
            evaluate_joint_geometry_residual(
                JointGeometryState.from_array(expand_fitted_values(fitted_values)),
                hbn_observations=hbn_observations,
                bi2se3_images=se3,
                bi2te3_images=te3,
                pbi2_y1_images=y1,
                pbi2_y2_images=y2,
                base_detector_rotation=rotation,
            ),
            copy=True,
        )

    def optimizer_residual_array(fitted_values: FloatArray) -> FloatArray:
        try:
            return raw_residual_array(fitted_values)
        except GeometryPredictionError:
            return np.full(residual_size, 1.0e6, dtype=np.float64)

    check("optimization")
    optimized = least_squares(
        optimizer_residual_array,
        initial_values[fitted_indices],
        bounds=(lower[fitted_indices], upper[fitted_indices]),
        method="trf",
        jac="2-point",
        x_scale=scale[fitted_indices],
        loss="soft_l1",
        f_scale=2.0,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=250,
    )
    optimized_values = expand_fitted_values(np.asarray(optimized.x, dtype=np.float64))
    state = JointGeometryState.from_array(optimized_values)
    raw_residual = raw_residual_array(optimized.x)
    check("qualification Jacobian")
    jacobian = _finite_jacobian(
        raw_residual_array,
        np.asarray(optimized.x, dtype=np.float64),
        lower[fitted_indices],
        upper[fitted_indices],
        steps[fitted_indices],
    )
    scaled_jacobian = jacobian * half_span[fitted_indices][None, :]
    check("rank and covariance")
    _, singular, right = np.linalg.svd(scaled_jacobian, full_matrices=False)
    tolerance = singular[0] * max(scaled_jacobian.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
    fitted_weakest = np.asarray(right[-1], dtype=np.float64)
    if fitted_weakest[int(np.argmax(np.abs(fitted_weakest)))] < 0.0:
        fitted_weakest = -fitted_weakest
    weakest = np.zeros(len(JOINT_GEOMETRY_PARAMETER_NAMES), dtype=np.float64)
    weakest[fitted_indices] = fitted_weakest
    bound_tolerance = 1.0e-6 * half_span[fitted_indices]
    fitted_on_bounds = np.asarray(
        (optimized.x - lower[fitted_indices] <= bound_tolerance)
        | (upper[fitted_indices] - optimized.x <= bound_tolerance),
        dtype=np.bool_,
    )
    on_bounds = np.zeros(len(JOINT_GEOMETRY_PARAMETER_NAMES), dtype=np.bool_)
    on_bounds[fitted_indices] = fitted_on_bounds
    degrees_of_freedom = max(raw_residual.size - fitted_indices.size, 1)
    covariance = (
        np.linalg.pinv(jacobian.T @ jacobian)
        * float(raw_residual @ raw_residual)
        / (degrees_of_freedom)
    )
    fitted_standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    standard_error = np.full(len(JOINT_GEOMETRY_PARAMETER_NAMES), np.nan, dtype=np.float64)
    standard_error[fitted_indices] = fitted_standard_error
    fitted_confident = np.asarray(
        (~fitted_on_bounds)
        & np.isfinite(fitted_standard_error)
        & (fitted_standard_error < 0.5 * half_span[fitted_indices]),
        dtype=np.bool_,
    )
    parameter_confident = np.zeros(len(JOINT_GEOMETRY_PARAMETER_NAMES), dtype=np.bool_)
    parameter_confident[fitted_indices] = fitted_confident

    hbn_residual = evaluate_hbn_residual_px(
        np.asarray(
            (
                state.detector_column_tilt_rad,
                state.detector_row_tilt_rad,
                state.beam_center_column_px,
                state.beam_center_row_px,
                state.hbn_calibrant_distance_m,
            )
        ),
        hbn_observations,
        base_detector_rotation=rotation,
        beam_direction_lab=se3[0].model.inputs.config.source.mean_direction_lab,
        detector_column_pitch_m=se3[0].model.instrument.detector_column_pitch_m,
        detector_row_pitch_m=se3[0].model.instrument.detector_row_pitch_m,
    )
    per_image = []
    all_site_errors = []
    beam_origin = None
    for specimen_id, images in (
        ("bi2se3", se3),
        ("bi2te3", te3),
        ("pbi2_y1", y1),
        ("pbi2_y2", y2),
    ):
        for image in images:
            check(f"metrics {specimen_id}/{image.image_id}")
            _, site_error, image_beam_origin = _predict_image(
                specimen_id,
                image,
                state,
                base_detector_rotation=rotation,
            )
            magnitude = np.linalg.norm(site_error, axis=1)
            all_site_errors.append(magnitude)
            per_image.append(
                JointGeometryImageMetric(
                    specimen_id=specimen_id,
                    image_id=image.image_id,
                    site_count=magnitude.size,
                    site_rms_px=float(np.sqrt(np.mean(magnitude**2))),
                    site_max_px=float(np.max(magnitude)),
                )
            )
            beam_origin = image_beam_origin
    pooled = np.concatenate(all_site_errors)
    axis, pivot = _axis_and_pivot(se3[0].model.instrument, se3[0], state)
    direction = np.asarray(se3[0].model.inputs.config.source.mean_direction_lab, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    assert beam_origin is not None
    closest_beam_point = beam_origin + direction * float((pivot - beam_origin) @ direction)
    z_b_m = float(closest_beam_point[2] - pivot[2])

    def derived_z_b(fitted_values: FloatArray) -> float:
        check("conditional zB uncertainty")
        candidate = JointGeometryState.from_array(expand_fitted_values(fitted_values))
        _, _, candidate_origin = _absolute_instrument_and_model(
            "bi2se3",
            se3[0],
            candidate,
            base_detector_rotation=rotation,
        )
        _, candidate_pivot = _axis_and_pivot(se3[0].model.instrument, se3[0], candidate)
        candidate_beam_point = candidate_origin + direction * float(
            (candidate_pivot - candidate_origin) @ direction
        )
        return float(candidate_beam_point[2] - candidate_pivot[2])

    z_b_gradient = np.zeros(fitted_indices.size, dtype=np.float64)
    for fitted_index, (parameter_index, step) in enumerate(
        zip(fitted_indices, steps[fitted_indices], strict=True)
    ):
        forward = np.asarray(optimized.x, dtype=np.float64).copy()
        backward = forward.copy()
        if (
            optimized.x[fitted_index] - step >= lower[parameter_index]
            and optimized.x[fitted_index] + step <= upper[parameter_index]
        ):
            forward[fitted_index] += step
            backward[fitted_index] -= step
            z_b_gradient[fitted_index] = (derived_z_b(forward) - derived_z_b(backward)) / (
                2.0 * step
            )
        elif optimized.x[fitted_index] + step <= upper[parameter_index]:
            forward[fitted_index] += step
            z_b_gradient[fitted_index] = (derived_z_b(forward) - z_b_m) / step
        else:
            backward[fitted_index] -= step
            z_b_gradient[fitted_index] = (z_b_m - derived_z_b(backward)) / step
    z_b_variance = float(z_b_gradient @ covariance @ z_b_gradient)
    z_b_standard_error_m = math.sqrt(max(z_b_variance, 0.0))
    metrics_qualified = bool(
        np.sqrt(np.mean(pooled**2)) <= 3.0
        and np.max(pooled) <= 8.0
        and all(metric.site_rms_px <= 4.0 for metric in per_image)
        and np.sqrt(np.mean(hbn_residual**2)) <= 2.0
    )
    confidence = bool(
        optimized.success
        and rank == fitted_indices.size
        and condition <= 1.0e8
        and not np.any(on_bounds)
        and np.all(fitted_confident)
        and metrics_qualified
    )
    check("result")
    return JointGeometryFitResult(
        state=state,
        success=bool(optimized.success),
        confidence_qualified=confidence,
        message=str(optimized.message),
        standard_error=standard_error,
        parameter_confident=parameter_confident,
        jacobian_rank=rank,
        scaled_jacobian_condition=condition,
        scaled_jacobian_singular_values=singular,
        weakest_direction=weakest,
        active_bounds=on_bounds,
        hbn_residual_rms_px=float(np.sqrt(np.mean(hbn_residual**2))),
        hbn_residual_max_px=float(np.max(np.abs(hbn_residual))),
        per_image=tuple(per_image),
        pooled_crystalline_site_rms_px=float(np.sqrt(np.mean(pooled**2))),
        pooled_crystalline_site_max_px=float(np.max(pooled)),
        model_evaluation_count=evaluation_count,
        optimizer_function_evaluation_count=int(optimized.nfev),
        beam_origin_lab_m=beam_origin,
        corrected_goniometer_axis_lab=axis,
        corrected_goniometer_pivot_lab_m=pivot,
        z_b_m=z_b_m,
        z_b_standard_error_m=z_b_standard_error_m,
        fitted_parameter_names=tuple(
            name
            for name in JOINT_GEOMETRY_PARAMETER_NAMES
            if name not in fixed_reference and name not in unobserved
        ),
        fixed_reference_parameters=DEFAULT_FIXED_REFERENCE_PARAMETERS,
        unobserved_specimen_parameters=unobserved,
    )
