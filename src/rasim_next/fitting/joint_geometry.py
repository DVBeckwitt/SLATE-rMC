"""Joint hBN and multi-material indexed-image geometry fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.fitting.geometry import (
    ExactTagGeometryModel,
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
    rebind_configured_geometry_instrument,
    sample_configured_nominal_geometry_source,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
MaterialId = Literal["bi2se3", "bi2te3"]

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
    "hbn_calibrant_distance_m",
)

GLOBAL_PARAMETER_NAMES = JOINT_GEOMETRY_PARAMETER_NAMES[:9]
LOCAL_PARAMETER_NAMES = JOINT_GEOMETRY_PARAMETER_NAMES[9:14]
NUISANCE_PARAMETER_NAMES = JOINT_GEOMETRY_PARAMETER_NAMES[14:]


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
        if array.shape != (len(JOINT_GEOMETRY_PARAMETER_NAMES),) or not np.all(
            np.isfinite(array)
        ):
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
    material_id: MaterialId
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
    pooled_bi_site_rms_px: float
    pooled_bi_site_max_px: float
    model_evaluation_count: int
    optimizer_function_evaluation_count: int
    beam_origin_lab_m: FloatArray
    corrected_goniometer_axis_lab: FloatArray
    corrected_goniometer_pivot_lab_m: FloatArray
    z_b_m: float
    z_b_standard_error_m: float

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
        if (
            standard_error.shape != (count,)
            or confident.shape != (count,)
            or singular.shape != (count,)
            or weakest.shape != (count,)
            or active.shape != (count,)
            or origin.shape != (3,)
            or axis.shape != (3,)
            or pivot.shape != (3,)
        ):
            raise ValueError("joint geometry result arrays have invalid shapes")
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


def _absolute_instrument_and_model(
    material_id: MaterialId,
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
                        configured_axis.angle_deg
                        + math.degrees(state.incidence_angle_delta_rad)
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
    reference_column, reference_row = instrument.detector_reference_coordinate_px
    detector_offset_m = np.asarray(
        (
            (state.beam_center_column_px - reference_column)
            * instrument.detector_column_pitch_m,
            (state.beam_center_row_px - reference_row) * instrument.detector_row_pitch_m,
            0.0,
        )
    )
    beam_hit_lab_m = instrument.lab_from_detector.translation_m + detector_rotation @ detector_offset_m
    direction = np.asarray(shifted_config.source.mean_direction_lab, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    nominal_origin = np.asarray(shifted_config.source.mean_origin_lab_m, dtype=np.float64)
    beam_origin = beam_hit_lab_m - direction * float((beam_hit_lab_m - nominal_origin) @ direction)
    source = replace(shifted_config.source, mean_origin_lab_m=tuple(float(v) for v in beam_origin))
    shifted_config = replace(shifted_config, source=source)
    inputs = replace(
        inputs,
        config=shifted_config,
        samples=sample_configured_nominal_geometry_source(source),
    )
    if material_id == "bi2se3":
        sample_x_tilt = 0.0
        sample_y_tilt = state.bi2se3_sample_y_tilt_rad
        z_s_m = state.bi2se3_zs_m
    else:
        sample_x_tilt = state.bi2te3_sample_x_tilt_rad
        sample_y_tilt = state.bi2te3_sample_y_tilt_rad
        z_s_m = state.bi2te3_zs_m
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
    material_id: MaterialId,
    image: IndexedGeometryImage,
    state: JointGeometryState,
    *,
    base_detector_rotation: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    model, instrument, beam_origin = _absolute_instrument_and_model(
        material_id,
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
    base_detector_rotation: ArrayLike,
) -> FloatArray:
    """Evaluate hBN and both fixed indexed material series in one residual vector."""

    if not isinstance(state, JointGeometryState):
        raise TypeError("state must be JointGeometryState")
    rotation = np.asarray(base_detector_rotation, dtype=np.float64)
    images = tuple(bi2se3_images) + tuple(bi2te3_images)
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
    for material_id, material_images in (
        ("bi2se3", tuple(bi2se3_images)),
        ("bi2te3", tuple(bi2te3_images)),
    ):
        for image in material_images:
            blocks.append(
                _predict_image(
                    material_id,
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
        forward = values.copy()
        backward = values.copy()
        if values[index] - step >= lower[index] and values[index] + step <= upper[index]:
            forward[index] += step
            backward[index] -= step
            jacobian[:, index] = (
                np.asarray(function(forward)) - np.asarray(function(backward))  # type: ignore[operator]
            ) / (2.0 * step)
        elif values[index] + step <= upper[index]:
            forward[index] += step
            jacobian[:, index] = (np.asarray(function(forward)) - baseline) / step  # type: ignore[operator]
        else:
            backward[index] -= step
            jacobian[:, index] = (baseline - np.asarray(function(backward))) / step  # type: ignore[operator]
    return jacobian


def fit_joint_geometry(
    *,
    hbn_observations: HbnRingObservations,
    hbn_calibration: HbnDetectorCalibration,
    bi2se3_images: tuple[IndexedGeometryImage, ...],
    bi2te3_images: tuple[IndexedGeometryImage, ...],
    base_detector_rotation: ArrayLike,
    bounds: JointGeometryBounds | None = None,
) -> JointGeometryFitResult:
    """Fit the declared shared and sample-local geometry with fail-closed confidence gates."""

    if not isinstance(hbn_observations, HbnRingObservations) or not isinstance(
        hbn_calibration,
        HbnDetectorCalibration,
    ):
        raise TypeError("hBN observations and calibration have invalid types")
    se3 = tuple(bi2se3_images)
    te3 = tuple(bi2te3_images)
    if not se3 or not te3:
        raise ValueError("both Bi2Se3 and Bi2Te3 image series are required")
    rotation = np.asarray(base_detector_rotation, dtype=np.float64)
    initial = JointGeometryState.from_hbn(hbn_calibration)
    active_bounds = JointGeometryBounds.around_hbn(hbn_calibration) if bounds is None else bounds
    lower = active_bounds.lower.as_array()
    upper = active_bounds.upper.as_array()
    initial_values = initial.as_array()
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
        )
    )
    evaluation_count = 0

    def residual_array(values: FloatArray) -> FloatArray:
        nonlocal evaluation_count
        evaluation_count += 1
        return np.array(
            evaluate_joint_geometry_residual(
                JointGeometryState.from_array(values),
                hbn_observations=hbn_observations,
                bi2se3_images=se3,
                bi2te3_images=te3,
                base_detector_rotation=rotation,
            ),
            copy=True,
        )

    optimized = least_squares(
        residual_array,
        initial_values,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale=scale,
        loss="soft_l1",
        f_scale=2.0,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=250,
    )
    state = JointGeometryState.from_array(optimized.x)
    raw_residual = residual_array(optimized.x)
    jacobian = _finite_jacobian(
        residual_array,
        np.asarray(optimized.x, dtype=np.float64),
        lower,
        upper,
        steps,
    )
    scaled_jacobian = jacobian * half_span[None, :]
    _, singular, right = np.linalg.svd(scaled_jacobian, full_matrices=False)
    tolerance = singular[0] * max(scaled_jacobian.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
    weakest = np.asarray(right[-1], dtype=np.float64)
    if weakest[int(np.argmax(np.abs(weakest)))] < 0.0:
        weakest = -weakest
    bound_tolerance = 1.0e-6 * half_span
    on_bounds = np.asarray(
        (optimized.x - lower <= bound_tolerance) | (upper - optimized.x <= bound_tolerance),
        dtype=np.bool_,
    )
    degrees_of_freedom = max(raw_residual.size - optimized.x.size, 1)
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * float(raw_residual @ raw_residual) / (
        degrees_of_freedom
    )
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    parameter_confident = np.asarray(
        (~on_bounds)
        & np.isfinite(standard_error)
        & (standard_error < 0.5 * half_span),
        dtype=np.bool_,
    )

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
    for material_id, images in (("bi2se3", se3), ("bi2te3", te3)):
        for image in images:
            _, site_error, image_beam_origin = _predict_image(
                material_id,
                image,
                state,
                base_detector_rotation=rotation,
            )
            magnitude = np.linalg.norm(site_error, axis=1)
            all_site_errors.append(magnitude)
            per_image.append(
                JointGeometryImageMetric(
                    material_id=material_id,
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

    def derived_z_b(values: FloatArray) -> float:
        candidate = JointGeometryState.from_array(values)
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

    z_b_gradient = np.zeros(len(JOINT_GEOMETRY_PARAMETER_NAMES), dtype=np.float64)
    for index, step in enumerate(steps):
        forward = np.asarray(optimized.x, dtype=np.float64).copy()
        backward = forward.copy()
        if optimized.x[index] - step >= lower[index] and optimized.x[index] + step <= upper[index]:
            forward[index] += step
            backward[index] -= step
            z_b_gradient[index] = (derived_z_b(forward) - derived_z_b(backward)) / (2.0 * step)
        elif optimized.x[index] + step <= upper[index]:
            forward[index] += step
            z_b_gradient[index] = (derived_z_b(forward) - z_b_m) / step
        else:
            backward[index] -= step
            z_b_gradient[index] = (z_b_m - derived_z_b(backward)) / step
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
        and rank == len(JOINT_GEOMETRY_PARAMETER_NAMES)
        and condition <= 1.0e8
        and not np.any(on_bounds)
        and np.all(parameter_confident)
        and metrics_qualified
    )
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
        pooled_bi_site_rms_px=float(np.sqrt(np.mean(pooled**2))),
        pooled_bi_site_max_px=float(np.max(pooled)),
        model_evaluation_count=evaluation_count,
        optimizer_function_evaluation_count=int(optimized.nfev),
        beam_origin_lab_m=beam_origin,
        corrected_goniometer_axis_lab=axis,
        corrected_goniometer_pivot_lab_m=pivot,
        z_b_m=z_b_m,
        z_b_standard_error_m=z_b_standard_error_m,
    )
