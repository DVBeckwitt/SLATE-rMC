"""Shared detector-native geometry fitting across indexed OSC images."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.fitting.geometry import (
    ExactTagGeometryModel,
    GeometryPredictionError,
    GeometryRankError,
    IntegerLMarkerKey,
    IntegerLMarkerObservations,
    IntegerLMarkerPrediction,
    IntegerLSelectionAudit,
    LayerLMarkerDefinition,
    LayerLMarkerObservations,
    LayerLMarkerPrediction,
    LayerLSelectionAudit,
    _finite_difference_jacobian,
    _nonzero_chord_angles_and_residual_px,
    _rank_diagnostics,
    _readonly_float_array,
    audit_exact_layer_l_geometry_roots,
    audit_exact_tag_geometry_roots,
    evaluate_layer_l_geometry_objective_residual,
    evaluate_tagged_geometry_objective_residual,
)
from rasim_next.geometry.instrument import (
    AxisRotation,
    CompiledInstrument,
    axis_rotation_transform,
    compose_intrinsic_xy_rotation,
)
from rasim_next.pipeline.configured_simulation import (
    AxisRotationConfiguration,
    rebind_configured_geometry_instrument,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

INCIDENCE_ANGLE_DELTA_PARAMETER_NAME = "incidence_angle_delta_rad"
INCIDENCE_ANGLE_TRIM_PARAMETER_PREFIX = "incidence_angle_trim_helmert"
SHARED_GEOMETRY_PARAMETER_NAMES = (
    "detector_column_tilt_rad",
    "detector_row_tilt_rad",
    "sample_normal_x_tilt_rad",
    "sample_normal_y_tilt_rad",
    "goniometer_axis_pitch_rad",
    "goniometer_axis_yaw_rad",
    "sample_plane_normal_offset_m",
    "goniometer_pivot_pitch_offset_m",
    "goniometer_pivot_yaw_offset_m",
)
DETECTOR_CALIBRATION_PARAMETER_NAMES = (
    "detector_reference_column_offset_px",
    "detector_reference_row_offset_px",
    "detector_plane_normal_offset_m",
)
_PARAMETER_NAMES = SHARED_GEOMETRY_PARAMETER_NAMES
_PARAMETERIZATION_ID = (
    "shared_detector_xy_axis_tangent_xy_pivot_tangent_xy_sample_normal_xy_plane_offset.v2"
)
_MAXIMUM_JACOBIAN_CONDITION = 1.0e8
_ACTIVE_BOUND_RELATIVE_TOLERANCE = 1.0e-6
_RANK_STEP = (1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5, 1.0e-6, 1.0e-6, 1.0e-6)
_OPTIMIZER_SCALE = (
    math.radians(0.5),
    math.radians(0.5),
    math.radians(0.5),
    math.radians(0.5),
    math.radians(0.5),
    math.radians(0.5),
    5.0e-5,
    5.0e-5,
    5.0e-5,
)
_INCIDENCE_ANGLE_DELTA_RANK_STEP_RAD = 1.0e-5
_INCIDENCE_ANGLE_DELTA_OPTIMIZER_SCALE_RAD = math.radians(0.5)
_INCIDENCE_ANGLE_TRIM_RANK_STEP_RAD = 1.0e-5
_DETECTOR_CALIBRATION_RANK_STEP = (0.01, 0.01, 1.0e-5)
_DETECTOR_CALIBRATION_OPTIMIZER_SCALE = (1.0, 1.0, 1.0e-3)
_CALIBRATED_PARAMETERIZATION_ID = f"{_PARAMETERIZATION_ID}.detector_reference_polish.v1"


def zero_sum_helmert_basis(image_count: int) -> FloatArray:
    """Return a deterministic orthonormal basis for zero-sum image offsets."""

    if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count < 2:
        raise ValueError("image_count must be an integer of at least two")
    basis = np.zeros((image_count, image_count - 1), dtype=np.float64)
    for column in range(image_count - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    basis.setflags(write=False)
    return basis


def _canonical_incidence_angle_trims(
    images: tuple[IndexedGeometryImage, ...],
    value: Mapping[str, float] | None,
) -> dict[str, float]:
    image_ids = tuple(image.image_id for image in images)
    if value is None:
        return dict.fromkeys(image_ids, 0.0)
    if not isinstance(value, Mapping):
        raise TypeError("incidence_angle_trim_by_image_id_rad must be a mapping")
    if set(value) != set(image_ids):
        raise ValueError("incidence-angle trims must match the canonical image IDs exactly")
    trims = {image_id: float(value[image_id]) for image_id in image_ids}
    if any(not math.isfinite(trim) for trim in trims.values()):
        raise ValueError("incidence-angle trims must be finite")
    if not math.isclose(math.fsum(trims.values()), 0.0, rel_tol=0.0, abs_tol=1.0e-14):
        raise ValueError("incidence-angle trims must sum to zero")
    return trims


def _canonical_fitted_parameter_names(
    value: tuple[str, ...],
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("fitted_parameter_names must be a sequence of parameter names")
    names = tuple(value)
    if not names and not allow_empty:
        raise ValueError("fitted_parameter_names must contain at least one parameter")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("fitted_parameter_names must contain only strings")
    if len(set(names)) != len(names):
        raise ValueError("fitted_parameter_names must not contain duplicates")
    unknown = set(names) - set(SHARED_GEOMETRY_PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown shared geometry parameter names: {sorted(unknown)}")
    selected = set(names)
    return tuple(name for name in SHARED_GEOMETRY_PARAMETER_NAMES if name in selected)


def _canonical_detector_calibration_parameter_names(
    value: tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("fitted detector calibration names must be a sequence")
    names = tuple(value)
    if any(not isinstance(name, str) for name in names):
        raise TypeError("fitted detector calibration names must contain only strings")
    if len(set(names)) != len(names):
        raise ValueError("fitted detector calibration names must not contain duplicates")
    unknown = set(names) - set(DETECTOR_CALIBRATION_PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown detector calibration parameter names: {sorted(unknown)}")
    selected = set(names)
    return tuple(name for name in DETECTOR_CALIBRATION_PARAMETER_NAMES if name in selected)


@dataclass(frozen=True, slots=True)
class SharedGeometryCorrections:
    """Nine uniquely owned corrections shared by one commanded-angle series."""

    detector_column_tilt_rad: float
    detector_row_tilt_rad: float
    sample_normal_x_tilt_rad: float
    sample_normal_y_tilt_rad: float
    goniometer_axis_pitch_rad: float
    goniometer_axis_yaw_rad: float
    sample_plane_normal_offset_m: float
    goniometer_pivot_pitch_offset_m: float
    goniometer_pivot_yaw_offset_m: float

    def __post_init__(self) -> None:
        for name in _PARAMETER_NAMES:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    @classmethod
    def zero(cls) -> SharedGeometryCorrections:
        return cls(*(0.0 for _ in _PARAMETER_NAMES))

    @classmethod
    def from_array(cls, value: ArrayLike) -> SharedGeometryCorrections:
        values = _readonly_float_array(value, (9,), "shared geometry corrections")
        return cls(*(float(item) for item in values))

    def as_array(self) -> FloatArray:
        return _readonly_float_array(
            tuple(getattr(self, name) for name in _PARAMETER_NAMES),
            (9,),
            "shared geometry corrections",
        )


@dataclass(frozen=True, slots=True)
class SharedGeometryCorrectionBounds:
    """Hard bounds for the nine shared geometry coordinates."""

    lower: SharedGeometryCorrections
    upper: SharedGeometryCorrections

    def __post_init__(self) -> None:
        if not isinstance(self.lower, SharedGeometryCorrections) or not isinstance(
            self.upper, SharedGeometryCorrections
        ):
            raise TypeError("lower and upper must be SharedGeometryCorrections")
        if np.any(self.lower.as_array() >= self.upper.as_array()):
            raise ValueError("every shared geometry lower bound must be smaller than its upper")

    @classmethod
    def rasim_multi_angle_pose(cls) -> SharedGeometryCorrectionBounds:
        half_span = np.asarray(
            (
                math.radians(10.0),
                math.radians(10.0),
                math.radians(5.0),
                math.radians(5.0),
                math.radians(5.0),
                math.radians(5.0),
                1.0e-4,
                1.0e-4,
                1.0e-4,
            )
        )
        return cls(
            lower=SharedGeometryCorrections.from_array(-half_span),
            upper=SharedGeometryCorrections.from_array(half_span),
        )

    @property
    def half_span(self) -> FloatArray:
        return _readonly_float_array(
            0.5 * (self.upper.as_array() - self.lower.as_array()),
            (9,),
            "shared geometry half span",
        )


@dataclass(frozen=True, slots=True)
class DetectorCalibrationCorrections:
    """Optional detector reference-coordinate and plane-normal calibration polish."""

    detector_reference_column_offset_px: float = 0.0
    detector_reference_row_offset_px: float = 0.0
    detector_plane_normal_offset_m: float = 0.0

    def __post_init__(self) -> None:
        for name in DETECTOR_CALIBRATION_PARAMETER_NAMES:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    @classmethod
    def zero(cls) -> DetectorCalibrationCorrections:
        return cls()

    @classmethod
    def from_array(cls, value: ArrayLike) -> DetectorCalibrationCorrections:
        values = _readonly_float_array(value, (3,), "detector calibration corrections")
        return cls(*(float(item) for item in values))

    def as_array(self) -> FloatArray:
        return _readonly_float_array(
            tuple(getattr(self, name) for name in DETECTOR_CALIBRATION_PARAMETER_NAMES),
            (3,),
            "detector calibration corrections",
        )


@dataclass(frozen=True, slots=True)
class DetectorCalibrationCorrectionBounds:
    """Hard bounds for optional detector reference calibration coordinates."""

    lower: DetectorCalibrationCorrections
    upper: DetectorCalibrationCorrections

    def __post_init__(self) -> None:
        if not isinstance(self.lower, DetectorCalibrationCorrections) or not isinstance(
            self.upper, DetectorCalibrationCorrections
        ):
            raise TypeError("lower and upper must be DetectorCalibrationCorrections")
        if np.any(self.lower.as_array() >= self.upper.as_array()):
            raise ValueError("every detector calibration lower bound must be smaller than upper")

    @property
    def half_span(self) -> FloatArray:
        return _readonly_float_array(
            0.5 * (self.upper.as_array() - self.lower.as_array()),
            (3,),
            "detector calibration half span",
        )


@dataclass(frozen=True, slots=True)
class IncidenceAngleDeltaBounds:
    """Hard bounds for one common additive commanded-incidence correction."""

    lower_rad: float
    upper_rad: float

    def __post_init__(self) -> None:
        lower = float(self.lower_rad)
        upper = float(self.upper_rad)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("incidence-angle delta bounds must be finite and increasing")
        object.__setattr__(self, "lower_rad", lower)
        object.__setattr__(self, "upper_rad", upper)

    @classmethod
    def rasim_shared_offset(cls) -> IncidenceAngleDeltaBounds:
        half_span_rad = math.radians(0.5)
        return cls(lower_rad=-half_span_rad, upper_rad=half_span_rad)

    @property
    def half_span_rad(self) -> float:
        return 0.5 * (self.upper_rad - self.lower_rad)


def _axis_pitch_yaw(axis_lab: ArrayLike) -> tuple[float, float]:
    axis = np.asarray(axis_lab, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("configured goniometer axis must be a finite three-vector")
    horizontal = math.hypot(float(axis[0]), float(axis[1]))
    if horizontal <= 1.0e-12:
        raise ValueError("goniometer-axis corrections require a nonvertical configured axis")
    return math.atan2(float(axis[2]), horizontal), math.atan2(-float(axis[1]), float(axis[0]))


def _axis_rotation(
    configuration: AxisRotationConfiguration,
    axis_lab: ArrayLike,
    pivot_lab_m: ArrayLike,
) -> AxisRotation:
    return AxisRotation(
        axis_lab=np.asarray(axis_lab, dtype=np.float64),
        angle_rad=math.radians(configuration.angle_deg),
        pivot_lab_m=np.asarray(pivot_lab_m, dtype=np.float64),
    )


def apply_detector_calibration_corrections(
    instrument: CompiledInstrument,
    corrections: DetectorCalibrationCorrections,
) -> CompiledInstrument:
    """Apply native center offsets and a translation along the nominal panel normal."""

    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be CompiledInstrument")
    if not isinstance(corrections, DetectorCalibrationCorrections):
        raise TypeError("corrections must be DetectorCalibrationCorrections")
    if np.all(corrections.as_array() == 0.0):
        return instrument
    detector = instrument.lab_from_detector
    detector_translation = (
        detector.translation_m
        + corrections.detector_plane_normal_offset_m * detector.rotation[:, 2]
    )
    reference_column, reference_row = instrument.detector_reference_coordinate_px
    return replace(
        instrument,
        lab_from_detector=RigidTransform(
            detector.rotation,
            detector_translation,
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        detector_reference_coordinate_px=(
            reference_column + corrections.detector_reference_column_offset_px,
            reference_row + corrections.detector_reference_row_offset_px,
        ),
    )


def apply_shared_geometry_corrections(
    instrument: CompiledInstrument,
    configured_axis_rotations: tuple[AxisRotationConfiguration, ...],
    corrections: SharedGeometryCorrections,
) -> CompiledInstrument:
    """Apply the canonical axis, pivoted end-pose, and signed-plane corrections."""

    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be CompiledInstrument")
    if not isinstance(corrections, SharedGeometryCorrections):
        raise TypeError("corrections must be SharedGeometryCorrections")
    configured = tuple(configured_axis_rotations)
    if len(configured) != 1 or not isinstance(configured[0], AxisRotationConfiguration):
        raise ValueError("shared geometry fitting requires exactly one configured goniometer axis")
    if instrument.sample_support_model_id != "unbounded_plane.v1":
        raise ValueError("shared geometry fitting currently requires unbounded_plane.v1 support")

    rotation = configured[0]
    base_pitch, base_yaw = _axis_pitch_yaw(rotation.axis_lab)
    pitch = base_pitch + corrections.goniometer_axis_pitch_rad
    yaw = base_yaw + corrections.goniometer_axis_yaw_rad
    cosine_pitch = math.cos(pitch)
    corrected_axis = np.asarray(
        (
            math.cos(yaw) * cosine_pitch,
            -math.sin(yaw) * cosine_pitch,
            math.sin(pitch),
        ),
        dtype=np.float64,
    )
    pivot_pitch_tangent = np.asarray(
        (
            -math.cos(yaw) * math.sin(pitch),
            math.sin(yaw) * math.sin(pitch),
            math.cos(pitch),
        ),
        dtype=np.float64,
    )
    pivot_yaw_tangent = np.asarray(
        (-math.sin(yaw), -math.cos(yaw), 0.0),
        dtype=np.float64,
    )
    base_pivot = np.asarray(rotation.pivot_lab_m, dtype=np.float64)
    corrected_pivot = (
        base_pivot
        + corrections.goniometer_pivot_pitch_offset_m * pivot_pitch_tangent
        + corrections.goniometer_pivot_yaw_offset_m * pivot_yaw_tangent
    )
    base_motion = axis_rotation_transform(_axis_rotation(rotation, rotation.axis_lab, base_pivot))
    corrected_motion = axis_rotation_transform(
        _axis_rotation(rotation, corrected_axis, corrected_pivot)
    )
    sample_after_axis = corrected_motion.compose(base_motion.inverse()).compose(
        instrument.lab_from_sample
    )

    sample_rotation = compose_intrinsic_xy_rotation(
        sample_after_axis.rotation,
        corrections.sample_normal_x_tilt_rad,
        corrections.sample_normal_y_tilt_rad,
    )
    sample_delta_lab = sample_rotation @ sample_after_axis.rotation.T
    sample_translation = corrected_pivot + sample_delta_lab @ (
        sample_after_axis.translation_m - corrected_pivot
    )
    sample_translation = (
        sample_translation + corrections.sample_plane_normal_offset_m * sample_rotation[:, 2]
    )
    detector = instrument.lab_from_detector
    detector_rotation = compose_intrinsic_xy_rotation(
        detector.rotation,
        corrections.detector_column_tilt_rad,
        corrections.detector_row_tilt_rad,
    )
    return replace(
        instrument,
        lab_from_detector=RigidTransform(
            detector_rotation,
            detector.translation_m,
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        lab_from_sample=RigidTransform(
            sample_rotation,
            sample_translation,
            FrameId.SAMPLE,
            FrameId.LAB,
        ),
    )


def _array_state_signature(value: ArrayLike) -> tuple[str, tuple[int, ...], bytes]:
    array = np.ascontiguousarray(value)
    return array.dtype.str, array.shape, array.tobytes()


def _transform_state_signature(transform: RigidTransform) -> tuple[object, ...]:
    return (
        transform.source_frame.value,
        transform.target_frame.value,
        _array_state_signature(transform.rotation),
        _array_state_signature(transform.translation_m),
    )


def _instrument_state_signature(
    instrument: CompiledInstrument,
    *,
    include_commanded_sample_pose: bool,
) -> tuple[object, ...]:
    return (
        _transform_state_signature(instrument.lab_from_sample)
        if include_commanded_sample_pose
        else None,
        _transform_state_signature(instrument.sample_from_crystal),
        _transform_state_signature(instrument.lab_from_detector),
        instrument.detector_shape_rc,
        instrument.detector_row_pitch_m,
        instrument.detector_column_pitch_m,
        instrument.detector_reference_coordinate_px,
        instrument.sample_support_model_id,
        instrument.sample_width_m,
        instrument.sample_length_m,
        instrument.film_thickness_A,
    )


@dataclass(frozen=True, slots=True)
class IndexedGeometryImage:
    """One immutable indexed image and its exact nominal geometry context."""

    image_id: str
    commanded_angle_rad: float
    model: ExactTagGeometryModel
    observations: IntegerLMarkerObservations | LayerLMarkerObservations

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id.strip():
            raise ValueError("image_id must be nonempty")
        if not isinstance(self.model, ExactTagGeometryModel):
            raise TypeError("model must be ExactTagGeometryModel")
        if not isinstance(
            self.observations,
            (IntegerLMarkerObservations, LayerLMarkerObservations),
        ):
            raise TypeError(
                "observations must be IntegerLMarkerObservations or LayerLMarkerObservations"
            )
        angle = float(self.commanded_angle_rad)
        if not math.isfinite(angle):
            raise ValueError("commanded_angle_rad must be finite")
        configured = self.model.inputs.config.instrument.axis_rotations
        if len(configured) != 1:
            raise ValueError("indexed geometry images require exactly one configured axis")
        expected_angle = math.radians(configured[0].angle_deg)
        if not math.isclose(angle, expected_angle, rel_tol=0.0, abs_tol=1.0e-14):
            raise ValueError("commanded angle does not match the image geometry context")
        expected_instrument = rebind_configured_geometry_instrument(
            self.model.inputs,
            self.model.inputs.config,
        ).instrument
        if _instrument_state_signature(
            self.model.instrument,
            include_commanded_sample_pose=True,
        ) != _instrument_state_signature(
            expected_instrument,
            include_commanded_sample_pose=True,
        ):
            raise ValueError("indexed geometry model instrument does not match its declared config")
        if (
            isinstance(self.observations, LayerLMarkerObservations)
            and self.observations.keys[0].reciprocal_basis_revision
            != self.model.reciprocal_basis_revision
        ):
            raise ValueError(
                "observation reciprocal-basis revision does not match the image geometry context"
            )
        wavelength_scale = max(
            self.observations.reference_wavelength_A,
            self.model.reference_wavelength_A,
            1.0,
        )
        if not math.isclose(
            self.observations.reference_wavelength_A,
            self.model.reference_wavelength_A,
            rel_tol=0.0,
            abs_tol=256.0 * np.finfo(np.float64).eps * wavelength_scale,
        ):
            raise ValueError("observation wavelength does not match the image geometry context")
        object.__setattr__(self, "image_id", self.image_id.strip())
        object.__setattr__(self, "commanded_angle_rad", angle)

    def corrected_instrument(
        self,
        corrections: SharedGeometryCorrections,
        *,
        detector_calibration_corrections: DetectorCalibrationCorrections | None = None,
        incidence_angle_delta_rad: float = 0.0,
        incidence_angle_trim_rad: float = 0.0,
    ) -> CompiledInstrument:
        delta = float(incidence_angle_delta_rad)
        if not math.isfinite(delta):
            raise ValueError("incidence_angle_delta_rad must be finite")
        trim = float(incidence_angle_trim_rad)
        if not math.isfinite(trim):
            raise ValueError("incidence_angle_trim_rad must be finite")
        config = self.model.inputs.config
        axis = config.instrument.axis_rotations[0]
        shifted_config = replace(
            config,
            instrument=replace(
                config.instrument,
                axis_rotations=(
                    replace(axis, angle_deg=axis.angle_deg + math.degrees(delta + trim)),
                ),
            ),
        )
        shifted_inputs = rebind_configured_geometry_instrument(
            self.model.inputs,
            shifted_config,
        )
        calibrated_instrument = apply_detector_calibration_corrections(
            shifted_inputs.instrument,
            (
                DetectorCalibrationCorrections.zero()
                if detector_calibration_corrections is None
                else detector_calibration_corrections
            ),
        )
        return apply_shared_geometry_corrections(
            calibrated_instrument,
            shifted_config.instrument.axis_rotations,
            corrections,
        )

    def predict_integer_l_tags(
        self,
        keys: tuple[IntegerLMarkerKey, ...],
        corrections: SharedGeometryCorrections,
        *,
        detector_calibration_corrections: DetectorCalibrationCorrections | None = None,
        incidence_angle_delta_rad: float = 0.0,
        incidence_angle_trim_rad: float = 0.0,
    ) -> IntegerLMarkerPrediction:
        return self.model.predict_integer_l_tags(
            keys,
            instrument=self.corrected_instrument(
                corrections,
                detector_calibration_corrections=detector_calibration_corrections,
                incidence_angle_delta_rad=incidence_angle_delta_rad,
                incidence_angle_trim_rad=incidence_angle_trim_rad,
            ),
        )

    def predict_layer_l_tags(
        self,
        definitions: tuple[LayerLMarkerDefinition, ...],
        corrections: SharedGeometryCorrections,
        *,
        detector_calibration_corrections: DetectorCalibrationCorrections | None = None,
        incidence_angle_delta_rad: float = 0.0,
        incidence_angle_trim_rad: float = 0.0,
    ) -> LayerLMarkerPrediction:
        return self.model.predict_layer_l_tags(
            definitions,
            instrument=self.corrected_instrument(
                corrections,
                detector_calibration_corrections=detector_calibration_corrections,
                incidence_angle_delta_rad=incidence_angle_delta_rad,
                incidence_angle_trim_rad=incidence_angle_trim_rad,
            ),
        )

    def _predict_observation_tags(
        self,
        corrections: SharedGeometryCorrections,
        *,
        detector_calibration_corrections: DetectorCalibrationCorrections | None = None,
        incidence_angle_delta_rad: float = 0.0,
        incidence_angle_trim_rad: float = 0.0,
    ) -> IntegerLMarkerPrediction | LayerLMarkerPrediction:
        if isinstance(self.observations, LayerLMarkerObservations):
            return self.predict_layer_l_tags(
                self.observations.definitions,
                corrections,
                detector_calibration_corrections=detector_calibration_corrections,
                incidence_angle_delta_rad=incidence_angle_delta_rad,
                incidence_angle_trim_rad=incidence_angle_trim_rad,
            )
        return self.predict_integer_l_tags(
            self.observations.keys,
            corrections,
            detector_calibration_corrections=detector_calibration_corrections,
            incidence_angle_delta_rad=incidence_angle_delta_rad,
            incidence_angle_trim_rad=incidence_angle_trim_rad,
        )


def _series_geometry_signature(image: IndexedGeometryImage) -> tuple[object, ...]:
    inputs = image.model.inputs
    config = inputs.config
    normalized_axes = tuple(
        replace(axis, angle_deg=0.0) for axis in config.instrument.axis_rotations
    )

    instrument = inputs.instrument
    actual_state = (
        inputs.samples.source_revision,
        inputs.material.material_revision,
        _array_state_signature(inputs.reciprocal.basis_Ainv),
        tuple((rod.h, rod.k, rod.family_m, rod.population) for rod in inputs.rods),
        _instrument_state_signature(instrument, include_commanded_sample_pose=False),
    )
    return (
        config.material,
        config.source,
        replace(config.instrument, axis_rotations=normalized_axes),
        config.bragg,
        actual_state,
    )


def _canonical_images(images: tuple[IndexedGeometryImage, ...]) -> tuple[IndexedGeometryImage, ...]:
    supplied = tuple(images)
    if not supplied or any(not isinstance(image, IndexedGeometryImage) for image in supplied):
        raise ValueError("images must contain at least one IndexedGeometryImage")
    ordered = tuple(sorted(supplied, key=lambda image: image.image_id))
    image_ids = tuple(image.image_id for image in ordered)
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("indexed geometry image IDs must be unique")
    signature = _series_geometry_signature(ordered[0])
    if any(_series_geometry_signature(image) != signature for image in ordered[1:]):
        raise ValueError(
            "indexed geometry images must share declared and actual material, source, mount, "
            "detector, axis, reciprocal, and Bragg state"
        )
    return ordered


def evaluate_indexed_geometry_series_residual(
    images: tuple[IndexedGeometryImage, ...],
    corrections: SharedGeometryCorrections,
    *,
    detector_calibration_corrections: DetectorCalibrationCorrections | None = None,
    incidence_angle_delta_rad: float = 0.0,
    incidence_angle_trim_by_image_id_rad: Mapping[str, float] | None = None,
) -> FloatArray:
    """Concatenate canonical per-image detector-native residual blocks."""

    if not isinstance(corrections, SharedGeometryCorrections):
        raise TypeError("corrections must be SharedGeometryCorrections")
    ordered = _canonical_images(images)
    trims = _canonical_incidence_angle_trims(ordered, incidence_angle_trim_by_image_id_rad)
    blocks = []
    for image in ordered:
        prediction = image._predict_observation_tags(
            corrections,
            detector_calibration_corrections=detector_calibration_corrections,
            incidence_angle_delta_rad=incidence_angle_delta_rad,
            incidence_angle_trim_rad=trims[image.image_id],
        )
        if isinstance(image.observations, LayerLMarkerObservations):
            if not isinstance(prediction, LayerLMarkerPrediction):
                raise AssertionError("layer-L observations produced the wrong prediction type")
            blocks.append(
                evaluate_layer_l_geometry_objective_residual(image.observations, prediction)
            )
        else:
            if not isinstance(prediction, IntegerLMarkerPrediction):
                raise AssertionError("integer-L observations produced the wrong prediction type")
            blocks.append(
                evaluate_tagged_geometry_objective_residual(image.observations, prediction)
            )
    residual = np.concatenate(blocks)
    residual.setflags(write=False)
    return residual


@dataclass(frozen=True, slots=True)
class IndexedGeometryImageMetrics:
    image_id: str
    site_count: int
    chord_count: int
    site_rms_px: float
    site_max_px: float
    chord_angle_rms_rad: float

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id:
            raise ValueError("image metric image_id must be nonempty")
        for name in ("site_count", "chord_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("site_rms_px", "site_max_px", "chord_angle_rms_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class IndexedGeometryFitResult:
    corrections: SharedGeometryCorrections
    incidence_angle_delta_rad: float
    incidence_angle_delta_fitted: bool
    success: bool
    message: str
    image_ids: tuple[str, ...]
    per_image: tuple[IndexedGeometryImageMetrics, ...]
    training_site_rms_px: float
    training_site_max_px: float
    training_chord_angle_rms_rad: float
    jacobian_rank: int
    jacobian_condition: float
    scaled_jacobian_singular_values: FloatArray
    scaled_jacobian_weakest_direction: FloatArray
    active_bounds: BoolArray
    model_evaluation_count: int
    optimizer_function_evaluation_count: int
    optimizer_jacobian_evaluation_count: int
    parameterization_id: str = _PARAMETERIZATION_ID
    fitted_parameter_names: tuple[str, ...] = SHARED_GEOMETRY_PARAMETER_NAMES
    fixed_parameter_names: tuple[str, ...] = ()
    detector_calibration_corrections: DetectorCalibrationCorrections = field(
        default_factory=DetectorCalibrationCorrections.zero
    )
    fitted_detector_calibration_parameter_names: tuple[str, ...] = ()
    fixed_detector_calibration_parameter_names: tuple[str, ...] = (
        DETECTOR_CALIBRATION_PARAMETER_NAMES
    )
    incidence_angle_trim_contrast_rad: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    incidence_angle_trim_by_image_id_rad: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    incidence_angle_trim_fitted: bool = False
    incidence_angle_trim_prior_sigma_rad: float | None = None
    incidence_angle_trim_contrast_half_span_rad: float | None = None
    posterior_jacobian_rank: int | None = None
    posterior_jacobian_condition: float | None = None

    @property
    def jacobian_parameter_names(self) -> tuple[str, ...]:
        trim_names = tuple(
            f"{INCIDENCE_ANGLE_TRIM_PARAMETER_PREFIX}_{index + 1}_rad"
            for index in range(len(self.incidence_angle_trim_contrast_rad))
        )
        return (
            self.fitted_parameter_names
            + self.fitted_detector_calibration_parameter_names
            + ((INCIDENCE_ANGLE_DELTA_PARAMETER_NAME,) if self.incidence_angle_delta_fitted else ())
            + trim_names
        )

    def __post_init__(self) -> None:
        if not isinstance(self.corrections, SharedGeometryCorrections):
            raise TypeError("corrections must be SharedGeometryCorrections")
        if not isinstance(
            self.detector_calibration_corrections,
            DetectorCalibrationCorrections,
        ):
            raise TypeError(
                "detector_calibration_corrections must be DetectorCalibrationCorrections"
            )
        if not isinstance(self.success, bool):
            raise TypeError("success must be bool")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be nonempty")
        if not isinstance(self.incidence_angle_delta_fitted, bool):
            raise TypeError("incidence_angle_delta_fitted must be bool")
        if not isinstance(self.incidence_angle_trim_fitted, bool):
            raise TypeError("incidence_angle_trim_fitted must be bool")
        incidence_delta = float(self.incidence_angle_delta_rad)
        if not math.isfinite(incidence_delta):
            raise ValueError("incidence_angle_delta_rad must be finite")
        fitted_calibration_names = _canonical_detector_calibration_parameter_names(
            self.fitted_detector_calibration_parameter_names
        )
        fitted_names = _canonical_fitted_parameter_names(
            self.fitted_parameter_names,
            allow_empty=(
                bool(fitted_calibration_names)
                or self.incidence_angle_delta_fitted
                or self.incidence_angle_trim_fitted
            ),
        )
        fixed_names = tuple(
            name for name in SHARED_GEOMETRY_PARAMETER_NAMES if name not in fitted_names
        )
        fixed_calibration_names = tuple(
            name
            for name in DETECTOR_CALIBRATION_PARAMETER_NAMES
            if name not in fitted_calibration_names
        )
        if tuple(self.fitted_parameter_names) != fitted_names:
            raise ValueError("fitted_parameter_names must use canonical parameter order")
        if tuple(self.fixed_parameter_names) != fixed_names:
            raise ValueError("fixed_parameter_names must be the canonical fitted complement")
        if tuple(self.fitted_detector_calibration_parameter_names) != fitted_calibration_names:
            raise ValueError("fitted_detector_calibration_parameter_names must use canonical order")
        if tuple(self.fixed_detector_calibration_parameter_names) != fixed_calibration_names:
            raise ValueError(
                "fixed_detector_calibration_parameter_names must be the canonical complement"
            )
        contrast = np.asarray(self.incidence_angle_trim_contrast_rad, dtype=np.float64)
        trims = np.asarray(self.incidence_angle_trim_by_image_id_rad, dtype=np.float64)
        expected_contrast_count = len(self.image_ids) - 1 if self.incidence_angle_trim_fitted else 0
        if contrast.shape != (expected_contrast_count,) or not np.all(np.isfinite(contrast)):
            raise ValueError("incidence-angle trim contrasts have the wrong shape or values")
        if trims.shape != (len(self.image_ids),) or not np.all(np.isfinite(trims)):
            raise ValueError("incidence-angle image trims have the wrong shape or values")
        if not math.isclose(float(np.sum(trims)), 0.0, rel_tol=0.0, abs_tol=1.0e-14):
            raise ValueError("incidence-angle image trims must sum to zero")
        if not self.incidence_angle_trim_fitted and np.any(trims != 0.0):
            raise ValueError("unfitted incidence-angle trims must be zero")
        contrast.setflags(write=False)
        trims.setflags(write=False)
        trim_prior = self.incidence_angle_trim_prior_sigma_rad
        trim_half_span = self.incidence_angle_trim_contrast_half_span_rad
        if self.incidence_angle_trim_fitted:
            if trim_prior is None or not math.isfinite(trim_prior) or trim_prior <= 0.0:
                raise ValueError("a fitted incidence-angle trim requires a positive prior sigma")
            if trim_half_span is None or not math.isfinite(trim_half_span) or trim_half_span <= 0.0:
                raise ValueError("a fitted incidence-angle trim requires a positive half-span")
        elif trim_prior is not None or trim_half_span is not None:
            raise ValueError("unfitted incidence-angle trims cannot declare controls")
        fitted_count = (
            len(fitted_names)
            + len(fitted_calibration_names)
            + int(self.incidence_angle_delta_fitted)
            + expected_contrast_count
        )
        if not self.image_ids or len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("image_ids must contain unique nonempty IDs")
        if any(not isinstance(value, str) or not value for value in self.image_ids):
            raise ValueError("image_ids must contain unique nonempty IDs")
        if any(not isinstance(metric, IndexedGeometryImageMetrics) for metric in self.per_image):
            raise TypeError("per_image must contain IndexedGeometryImageMetrics")
        if self.image_ids != tuple(metric.image_id for metric in self.per_image):
            raise ValueError("per-image metrics must match the canonical image IDs")
        for name in (
            "training_site_rms_px",
            "training_site_max_px",
            "training_chord_angle_rms_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.jacobian_rank, bool)
            or not isinstance(self.jacobian_rank, int)
            or not 0 <= self.jacobian_rank <= fitted_count
        ):
            raise ValueError("jacobian_rank must not exceed the fitted parameter count")
        condition = float(self.jacobian_condition)
        if not math.isfinite(condition) or condition < 1.0:
            raise ValueError("jacobian_condition must be finite and at least one")
        object.__setattr__(self, "jacobian_condition", condition)
        singular = _readonly_float_array(
            self.scaled_jacobian_singular_values,
            (fitted_count,),
            "scaled_jacobian_singular_values",
        )
        if np.any(singular < 0.0) or np.any(np.diff(singular) > 0.0):
            raise ValueError("scaled singular values must be nonnegative and descending")
        weakest = _readonly_float_array(
            self.scaled_jacobian_weakest_direction,
            (fitted_count,),
            "scaled_jacobian_weakest_direction",
        )
        if not math.isclose(float(np.linalg.norm(weakest)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("scaled weakest direction must have unit norm")
        active = np.array(self.active_bounds, dtype=np.bool_, copy=True)
        if active.shape != (fitted_count,):
            raise ValueError("active_bounds must contain one flag per fitted parameter")
        active.setflags(write=False)
        for name in (
            "model_evaluation_count",
            "optimizer_function_evaluation_count",
            "optimizer_jacobian_evaluation_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        calibration_active = bool(fitted_calibration_names) or np.any(
            self.detector_calibration_corrections.as_array() != 0.0
        )
        expected_parameterization_id = (
            _CALIBRATED_PARAMETERIZATION_ID if calibration_active else _PARAMETERIZATION_ID
        )
        if self.parameterization_id != expected_parameterization_id:
            raise ValueError("unsupported indexed geometry parameterization")
        if self.success and (
            self.jacobian_rank != fitted_count
            or self.jacobian_condition > _MAXIMUM_JACOBIAN_CONDITION
        ):
            raise ValueError("a successful fit must have a full acceptable Jacobian")
        posterior_rank = self.posterior_jacobian_rank
        posterior_condition = self.posterior_jacobian_condition
        if posterior_rank is None:
            posterior_rank = self.jacobian_rank
        if posterior_condition is None:
            posterior_condition = condition
        if (
            isinstance(posterior_rank, bool)
            or not isinstance(posterior_rank, int)
            or not 0 <= posterior_rank <= fitted_count
        ):
            raise ValueError("posterior_jacobian_rank is invalid")
        posterior_condition = float(posterior_condition)
        if not math.isfinite(posterior_condition) or posterior_condition < 1.0:
            raise ValueError("posterior_jacobian_condition must be finite and at least one")
        object.__setattr__(self, "fitted_parameter_names", fitted_names)
        object.__setattr__(self, "fixed_parameter_names", fixed_names)
        object.__setattr__(
            self,
            "fitted_detector_calibration_parameter_names",
            fitted_calibration_names,
        )
        object.__setattr__(
            self,
            "fixed_detector_calibration_parameter_names",
            fixed_calibration_names,
        )
        object.__setattr__(self, "incidence_angle_delta_rad", incidence_delta)
        object.__setattr__(self, "incidence_angle_trim_contrast_rad", contrast)
        object.__setattr__(self, "incidence_angle_trim_by_image_id_rad", trims)
        object.__setattr__(self, "incidence_angle_trim_prior_sigma_rad", trim_prior)
        object.__setattr__(
            self,
            "incidence_angle_trim_contrast_half_span_rad",
            trim_half_span,
        )
        object.__setattr__(self, "posterior_jacobian_rank", posterior_rank)
        object.__setattr__(self, "posterior_jacobian_condition", posterior_condition)
        object.__setattr__(self, "scaled_jacobian_singular_values", singular)
        object.__setattr__(self, "scaled_jacobian_weakest_direction", weakest)
        object.__setattr__(self, "active_bounds", active)


@dataclass(frozen=True, slots=True)
class IndexedGeometrySeriesMetrics:
    image_ids: tuple[str, ...]
    per_image: tuple[IndexedGeometryImageMetrics, ...]
    site_rms_px: float
    site_max_px: float
    chord_angle_rms_rad: float

    def __post_init__(self) -> None:
        if self.image_ids != tuple(metric.image_id for metric in self.per_image):
            raise ValueError("series metrics must match the canonical image IDs")
        if not self.image_ids or any(
            not isinstance(metric, IndexedGeometryImageMetrics) for metric in self.per_image
        ):
            raise ValueError("series metrics require at least one per-image metric")
        for name in ("site_rms_px", "site_max_px", "chord_angle_rms_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class IndexedGeometryRootAuditImage:
    image_id: str
    audit: IntegerLSelectionAudit | LayerLSelectionAudit

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id:
            raise ValueError("root audit image_id must be nonempty")
        if not isinstance(self.audit, (IntegerLSelectionAudit, LayerLSelectionAudit)):
            raise TypeError("audit must be IntegerLSelectionAudit or LayerLSelectionAudit")


@dataclass(frozen=True, slots=True)
class IndexedGeometrySeriesRootAudit:
    classification: str
    images: tuple[IndexedGeometryRootAuditImage, ...]

    def __post_init__(self) -> None:
        images = tuple(self.images)
        if not images or any(
            not isinstance(item, IndexedGeometryRootAuditImage) for item in images
        ):
            raise ValueError("images must contain at least one root audit")
        if tuple(item.image_id for item in images) != tuple(
            sorted(item.image_id for item in images)
        ) or len({item.image_id for item in images}) != len(images):
            raise ValueError("root audits must have unique canonical image IDs")
        expected = (
            "SAME" if all(item.audit.classification == "SAME" for item in images) else "CHANGED"
        )
        if self.classification != expected:
            raise ValueError("series root-audit classification disagrees with its images")
        object.__setattr__(self, "images", images)


def audit_indexed_geometry_series_roots(
    images: tuple[IndexedGeometryImage, ...],
    corrections: SharedGeometryCorrections,
    *,
    detector_calibration_corrections: DetectorCalibrationCorrections | None = None,
    incidence_angle_delta_rad: float = 0.0,
    incidence_angle_trim_by_image_id_rad: Mapping[str, float] | None = None,
) -> IndexedGeometrySeriesRootAudit:
    """Audit every frozen root through the independent continuous-surface solver."""

    if not isinstance(corrections, SharedGeometryCorrections):
        raise TypeError("corrections must be SharedGeometryCorrections")
    ordered = _canonical_images(images)
    trims = _canonical_incidence_angle_trims(ordered, incidence_angle_trim_by_image_id_rad)
    audits = []
    for image in ordered:
        instrument = image.corrected_instrument(
            corrections,
            detector_calibration_corrections=detector_calibration_corrections,
            incidence_angle_delta_rad=incidence_angle_delta_rad,
            incidence_angle_trim_rad=trims[image.image_id],
        )
        audit = (
            audit_exact_layer_l_geometry_roots(
                image.model,
                image.observations.definitions,
                instrument=instrument,
            )
            if isinstance(image.observations, LayerLMarkerObservations)
            else audit_exact_tag_geometry_roots(
                image.model,
                image.observations.keys,
                instrument=instrument,
            )
        )
        audits.append(IndexedGeometryRootAuditImage(image_id=image.image_id, audit=audit))
    frozen_audits = tuple(audits)
    return IndexedGeometrySeriesRootAudit(
        classification=(
            "SAME"
            if all(item.audit.classification == "SAME" for item in frozen_audits)
            else "CHANGED"
        ),
        images=frozen_audits,
    )


def _fit_metrics(
    images: tuple[IndexedGeometryImage, ...],
    corrections: SharedGeometryCorrections,
    incidence_angle_delta_rad: float,
    incidence_angle_trim_by_image_id_rad: Mapping[str, float] | None = None,
    detector_calibration_corrections: DetectorCalibrationCorrections | None = None,
) -> tuple[tuple[IndexedGeometryImageMetrics, ...], FloatArray, FloatArray]:
    per_image: list[IndexedGeometryImageMetrics] = []
    all_site_error: list[FloatArray] = []
    all_chord_angle: list[FloatArray] = []
    trims = _canonical_incidence_angle_trims(images, incidence_angle_trim_by_image_id_rad)
    for image in images:
        prediction = image._predict_observation_tags(
            corrections,
            detector_calibration_corrections=detector_calibration_corrections,
            incidence_angle_delta_rad=incidence_angle_delta_rad,
            incidence_angle_trim_rad=trims[image.image_id],
        )
        if not np.all(prediction.active_panel):
            raise GeometryPredictionError(
                "the fitted marker set does not remain on the active panel"
            )
        site_error = np.linalg.norm(
            prediction.coordinates_px - image.observations.coordinates_px,
            axis=1,
        )
        chord_angle, _ = _nonzero_chord_angles_and_residual_px(image.observations, prediction)
        per_image.append(
            IndexedGeometryImageMetrics(
                image_id=image.image_id,
                site_count=len(image.observations.keys),
                chord_count=int(chord_angle.size),
                site_rms_px=float(np.sqrt(np.mean(site_error**2))),
                site_max_px=float(np.max(site_error)),
                chord_angle_rms_rad=(
                    float(np.sqrt(np.mean(chord_angle**2))) if chord_angle.size else 0.0
                ),
            )
        )
        all_site_error.append(site_error)
        all_chord_angle.append(chord_angle)
    return (
        tuple(per_image),
        np.concatenate(all_site_error),
        np.concatenate(all_chord_angle)
        if any(value.size for value in all_chord_angle)
        else np.zeros(0),
    )


def evaluate_indexed_geometry_series_metrics(
    images: tuple[IndexedGeometryImage, ...],
    corrections: SharedGeometryCorrections,
    *,
    detector_calibration_corrections: DetectorCalibrationCorrections | None = None,
    incidence_angle_delta_rad: float = 0.0,
    incidence_angle_trim_by_image_id_rad: Mapping[str, float] | None = None,
) -> IndexedGeometrySeriesMetrics:
    """Report raw detector-pixel and chord metrics without optimizing."""

    ordered = _canonical_images(images)
    per_image, site_error, chord_angle = _fit_metrics(
        ordered,
        corrections,
        incidence_angle_delta_rad,
        incidence_angle_trim_by_image_id_rad,
        detector_calibration_corrections,
    )
    return IndexedGeometrySeriesMetrics(
        image_ids=tuple(image.image_id for image in ordered),
        per_image=per_image,
        site_rms_px=float(np.sqrt(np.mean(site_error**2))),
        site_max_px=float(np.max(site_error)),
        chord_angle_rms_rad=(float(np.sqrt(np.mean(chord_angle**2))) if chord_angle.size else 0.0),
    )


def fit_indexed_geometry_series(
    images: tuple[IndexedGeometryImage, ...],
    *,
    initial: SharedGeometryCorrections,
    bounds: SharedGeometryCorrectionBounds,
    fitted_parameter_names: tuple[str, ...] = SHARED_GEOMETRY_PARAMETER_NAMES,
    initial_detector_calibration_corrections: DetectorCalibrationCorrections | None = None,
    detector_calibration_correction_bounds: DetectorCalibrationCorrectionBounds | None = None,
    fitted_detector_calibration_parameter_names: tuple[str, ...] = (),
    initial_incidence_angle_delta_rad: float = 0.0,
    incidence_angle_delta_bounds: IncidenceAngleDeltaBounds | None = None,
    initial_incidence_angle_trim_contrast_rad: ArrayLike | None = None,
    incidence_angle_trim_contrast_half_span_rad: float | None = None,
    incidence_angle_trim_prior_sigma_rad: float | None = None,
) -> IndexedGeometryFitResult:
    """Fit shared geometry, optional detector calibration, incidence, and trims."""

    ordered = _canonical_images(images)
    if not isinstance(initial, SharedGeometryCorrections):
        raise TypeError("initial must be SharedGeometryCorrections")
    if not isinstance(bounds, SharedGeometryCorrectionBounds):
        raise TypeError("bounds must be SharedGeometryCorrectionBounds")
    initial_calibration = (
        DetectorCalibrationCorrections.zero()
        if initial_detector_calibration_corrections is None
        else initial_detector_calibration_corrections
    )
    if not isinstance(initial_calibration, DetectorCalibrationCorrections):
        raise TypeError(
            "initial_detector_calibration_corrections must be DetectorCalibrationCorrections"
        )
    fitted_calibration_names = _canonical_detector_calibration_parameter_names(
        fitted_detector_calibration_parameter_names
    )
    if fitted_calibration_names and detector_calibration_correction_bounds is None:
        raise ValueError(
            "fitted detector calibration parameters require detector calibration bounds"
        )
    if detector_calibration_correction_bounds is not None and not isinstance(
        detector_calibration_correction_bounds,
        DetectorCalibrationCorrectionBounds,
    ):
        raise TypeError(
            "detector_calibration_correction_bounds must be DetectorCalibrationCorrectionBounds"
        )
    incidence_delta_fitted = incidence_angle_delta_bounds is not None
    trim_fitted = (
        incidence_angle_trim_contrast_half_span_rad is not None
        or incidence_angle_trim_prior_sigma_rad is not None
        or initial_incidence_angle_trim_contrast_rad is not None
    )
    if trim_fitted and (
        incidence_angle_trim_contrast_half_span_rad is None
        or incidence_angle_trim_prior_sigma_rad is None
    ):
        raise ValueError("incidence-angle trims require both a contrast half-span and prior sigma")
    fitted_names = _canonical_fitted_parameter_names(
        fitted_parameter_names,
        allow_empty=bool(fitted_calibration_names) or incidence_delta_fitted or trim_fitted,
    )
    if incidence_delta_fitted and "sample_normal_x_tilt_rad" in fitted_names:
        raise ValueError(
            "incidence_angle_delta_rad and sample_normal_x_tilt_rad share the nominal "
            "incidence-axis gauge; fix sample_normal_x_tilt_rad when fitting the common delta"
        )
    fitted_indices = np.asarray(
        [SHARED_GEOMETRY_PARAMETER_NAMES.index(name) for name in fitted_names],
        dtype=np.int64,
    )
    fitted_calibration_indices = np.asarray(
        [DETECTOR_CALIBRATION_PARAMETER_NAMES.index(name) for name in fitted_calibration_names],
        dtype=np.int64,
    )
    lower = bounds.lower.as_array()
    upper = bounds.upper.as_array()
    initial_values = initial.as_array()
    if np.any(initial_values < lower) or np.any(initial_values > upper):
        raise ValueError("initial shared geometry corrections must lie inside the bounds")
    initial_calibration_values = initial_calibration.as_array()
    if detector_calibration_correction_bounds is None:
        calibration_lower = np.full(3, -np.inf, dtype=np.float64)
        calibration_upper = np.full(3, np.inf, dtype=np.float64)
        calibration_half_span = np.ones(3, dtype=np.float64)
    else:
        calibration_lower = detector_calibration_correction_bounds.lower.as_array()
        calibration_upper = detector_calibration_correction_bounds.upper.as_array()
        calibration_half_span = detector_calibration_correction_bounds.half_span
        if np.any(initial_calibration_values < calibration_lower) or np.any(
            initial_calibration_values > calibration_upper
        ):
            raise ValueError("initial detector calibration corrections must lie inside the bounds")
    initial_incidence_delta = float(initial_incidence_angle_delta_rad)
    if not math.isfinite(initial_incidence_delta):
        raise ValueError("initial_incidence_angle_delta_rad must be finite")
    if incidence_angle_delta_bounds is not None:
        if not isinstance(incidence_angle_delta_bounds, IncidenceAngleDeltaBounds):
            raise TypeError("incidence_angle_delta_bounds must be IncidenceAngleDeltaBounds")
        if not (
            incidence_angle_delta_bounds.lower_rad
            <= initial_incidence_delta
            <= incidence_angle_delta_bounds.upper_rad
        ):
            raise ValueError("initial incidence-angle delta must lie inside its bounds")

    image_ids = tuple(image.image_id for image in ordered)
    trim_basis = (
        zero_sum_helmert_basis(len(ordered)) if trim_fitted else np.empty((len(ordered), 0))
    )
    trim_count = trim_basis.shape[1]
    if trim_fitted:
        trim_half_span = float(incidence_angle_trim_contrast_half_span_rad)
        trim_prior_sigma = float(incidence_angle_trim_prior_sigma_rad)
        if not math.isfinite(trim_half_span) or trim_half_span <= 0.0:
            raise ValueError("incidence-angle trim contrast half-span must be positive and finite")
        if not math.isfinite(trim_prior_sigma) or trim_prior_sigma <= 0.0:
            raise ValueError("incidence-angle trim prior sigma must be positive and finite")
        initial_trim_contrast = (
            np.zeros(trim_count, dtype=np.float64)
            if initial_incidence_angle_trim_contrast_rad is None
            else np.asarray(initial_incidence_angle_trim_contrast_rad, dtype=np.float64)
        )
        if initial_trim_contrast.shape != (trim_count,) or not np.all(
            np.isfinite(initial_trim_contrast)
        ):
            raise ValueError("initial incidence-angle trim contrasts have the wrong shape")
        if np.any(np.abs(initial_trim_contrast) > trim_half_span):
            raise ValueError("initial incidence-angle trim contrasts lie outside the bounds")
    else:
        trim_half_span = None
        trim_prior_sigma = None
        initial_trim_contrast = np.empty(0, dtype=np.float64)

    model_evaluation_count = 0

    shared_value_count = len(fitted_indices)
    calibration_value_count = len(fitted_calibration_indices)
    calibration_start = shared_value_count
    delta_index = shared_value_count + calibration_value_count if incidence_delta_fitted else None
    trim_start = shared_value_count + calibration_value_count + int(incidence_delta_fitted)

    def unpack(
        value: FloatArray,
    ) -> tuple[
        SharedGeometryCorrections,
        DetectorCalibrationCorrections,
        float,
        FloatArray,
        dict[str, float],
    ]:
        full_value = np.array(initial_values, copy=True)
        full_value[fitted_indices] = value[:shared_value_count]
        full_calibration_value = np.array(initial_calibration_values, copy=True)
        full_calibration_value[fitted_calibration_indices] = value[
            calibration_start : calibration_start + calibration_value_count
        ]
        incidence_delta = (
            float(value[delta_index]) if delta_index is not None else initial_incidence_delta
        )
        contrast = (
            np.asarray(value[trim_start:], dtype=np.float64)
            if trim_fitted
            else initial_trim_contrast
        )
        trim_values = trim_basis @ contrast if trim_fitted else np.zeros(len(ordered))
        trim_by_id = {
            image_id: float(trim_values[index]) for index, image_id in enumerate(image_ids)
        }
        return (
            SharedGeometryCorrections.from_array(full_value),
            DetectorCalibrationCorrections.from_array(full_calibration_value),
            incidence_delta,
            contrast,
            trim_by_id,
        )

    def data_residual(value: FloatArray) -> FloatArray:
        nonlocal model_evaluation_count
        model_evaluation_count += 1
        corrections, calibration, incidence_delta, _, trim_by_id = unpack(value)
        return evaluate_indexed_geometry_series_residual(
            ordered,
            corrections,
            detector_calibration_corrections=calibration,
            incidence_angle_delta_rad=incidence_delta,
            incidence_angle_trim_by_image_id_rad=trim_by_id,
        )

    def optimizer_residual(value: FloatArray) -> FloatArray:
        measured = data_residual(value)
        if not trim_fitted:
            return measured
        return np.concatenate((measured, value[trim_start:] / trim_prior_sigma))

    fitted_lower = lower[fitted_indices]
    fitted_upper = upper[fitted_indices]
    fitted_initial = initial_values[fitted_indices]
    fitted_half_span = bounds.half_span[fitted_indices]
    fitted_rank_step = np.asarray(_RANK_STEP)[fitted_indices]
    fitted_optimizer_scale = np.asarray(_OPTIMIZER_SCALE)[fitted_indices]
    if calibration_value_count:
        fitted_lower = np.append(
            fitted_lower,
            calibration_lower[fitted_calibration_indices],
        )
        fitted_upper = np.append(
            fitted_upper,
            calibration_upper[fitted_calibration_indices],
        )
        fitted_initial = np.append(
            fitted_initial,
            initial_calibration_values[fitted_calibration_indices],
        )
        fitted_half_span = np.append(
            fitted_half_span,
            calibration_half_span[fitted_calibration_indices],
        )
        fitted_rank_step = np.append(
            fitted_rank_step,
            np.asarray(_DETECTOR_CALIBRATION_RANK_STEP)[fitted_calibration_indices],
        )
        fitted_optimizer_scale = np.append(
            fitted_optimizer_scale,
            np.asarray(_DETECTOR_CALIBRATION_OPTIMIZER_SCALE)[fitted_calibration_indices],
        )
    if incidence_angle_delta_bounds is not None:
        fitted_lower = np.append(fitted_lower, incidence_angle_delta_bounds.lower_rad)
        fitted_upper = np.append(fitted_upper, incidence_angle_delta_bounds.upper_rad)
        fitted_initial = np.append(fitted_initial, initial_incidence_delta)
        fitted_half_span = np.append(
            fitted_half_span,
            incidence_angle_delta_bounds.half_span_rad,
        )
        fitted_rank_step = np.append(
            fitted_rank_step,
            _INCIDENCE_ANGLE_DELTA_RANK_STEP_RAD,
        )
        fitted_optimizer_scale = np.append(
            fitted_optimizer_scale,
            _INCIDENCE_ANGLE_DELTA_OPTIMIZER_SCALE_RAD,
        )
    if trim_fitted:
        fitted_lower = np.append(fitted_lower, np.full(trim_count, -trim_half_span))
        fitted_upper = np.append(fitted_upper, np.full(trim_count, trim_half_span))
        fitted_initial = np.append(fitted_initial, initial_trim_contrast)
        fitted_half_span = np.append(fitted_half_span, np.full(trim_count, trim_half_span))
        fitted_rank_step = np.append(
            fitted_rank_step,
            np.full(trim_count, _INCIDENCE_ANGLE_TRIM_RANK_STEP_RAD),
        )
        fitted_optimizer_scale = np.append(
            fitted_optimizer_scale,
            np.full(trim_count, trim_prior_sigma),
        )
    preflight = _finite_difference_jacobian(
        data_residual,
        fitted_initial,
        fitted_lower,
        fitted_upper,
        step_size=fitted_rank_step,
    )
    fitted_count = (
        len(fitted_names) + len(fitted_calibration_names) + int(incidence_delta_fitted) + trim_count
    )
    rank, condition, _ = _rank_diagnostics(preflight, fitted_half_span)
    if rank < fitted_count or condition > _MAXIMUM_JACOBIAN_CONDITION:
        raise GeometryRankError(
            "indexed geometry Jacobian rank/conditioning failed: "
            f"rank={rank}/{fitted_count}, "
            f"condition={condition:.6g}"
        )

    optimized = least_squares(
        optimizer_residual,
        fitted_initial,
        bounds=(fitted_lower, fitted_upper),
        method="trf",
        jac="2-point",
        x_scale=fitted_optimizer_scale,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=150,
    )
    (
        corrections,
        detector_calibration_corrections,
        incidence_angle_delta_rad,
        trim_contrast,
        trim_by_id,
    ) = unpack(optimized.x)
    trim_values = np.asarray(tuple(trim_by_id[image_id] for image_id in image_ids))
    per_image, site_error, chord_angle = _fit_metrics(
        ordered,
        corrections,
        incidence_angle_delta_rad,
        trim_by_id,
        detector_calibration_corrections,
    )
    fitted_data_jacobian = _finite_difference_jacobian(
        data_residual,
        np.asarray(optimized.x, dtype=np.float64),
        fitted_lower,
        fitted_upper,
        step_size=fitted_rank_step,
    )
    rank, condition, singular = _rank_diagnostics(
        fitted_data_jacobian,
        fitted_half_span,
    )
    if rank < fitted_count or condition > _MAXIMUM_JACOBIAN_CONDITION:
        raise GeometryRankError(
            "fitted indexed geometry Jacobian rank/conditioning failed: "
            f"rank={rank}/{fitted_count}, "
            f"condition={condition:.6g}"
        )
    posterior_rank, posterior_condition, _ = _rank_diagnostics(
        np.asarray(optimized.jac, dtype=np.float64),
        fitted_half_span,
    )
    scaled_jacobian = fitted_data_jacobian * fitted_half_span[None, :]
    weakest = np.linalg.svd(scaled_jacobian, full_matrices=False)[2][-1]
    largest_component = int(np.argmax(np.abs(weakest)))
    if weakest[largest_component] < 0.0:
        weakest = -weakest
    bound_proximity = _ACTIVE_BOUND_RELATIVE_TOLERANCE * fitted_half_span
    active_bounds = np.asarray(
        (optimized.active_mask != 0)
        | (optimized.x - fitted_lower <= bound_proximity)
        | (fitted_upper - optimized.x <= bound_proximity),
        dtype=np.bool_,
    )
    return IndexedGeometryFitResult(
        corrections=corrections,
        detector_calibration_corrections=detector_calibration_corrections,
        incidence_angle_delta_rad=incidence_angle_delta_rad,
        incidence_angle_delta_fitted=incidence_delta_fitted,
        incidence_angle_trim_contrast_rad=np.array(trim_contrast, copy=True),
        incidence_angle_trim_by_image_id_rad=np.array(trim_values, copy=True),
        incidence_angle_trim_fitted=trim_fitted,
        incidence_angle_trim_prior_sigma_rad=trim_prior_sigma,
        incidence_angle_trim_contrast_half_span_rad=trim_half_span,
        posterior_jacobian_rank=posterior_rank,
        posterior_jacobian_condition=posterior_condition,
        success=bool(optimized.success),
        message=str(optimized.message),
        fitted_parameter_names=fitted_names,
        fixed_parameter_names=tuple(
            name for name in SHARED_GEOMETRY_PARAMETER_NAMES if name not in fitted_names
        ),
        fitted_detector_calibration_parameter_names=fitted_calibration_names,
        fixed_detector_calibration_parameter_names=tuple(
            name
            for name in DETECTOR_CALIBRATION_PARAMETER_NAMES
            if name not in fitted_calibration_names
        ),
        parameterization_id=(
            _CALIBRATED_PARAMETERIZATION_ID
            if fitted_calibration_names
            or np.any(detector_calibration_corrections.as_array() != 0.0)
            else _PARAMETERIZATION_ID
        ),
        image_ids=tuple(image.image_id for image in ordered),
        per_image=per_image,
        training_site_rms_px=float(np.sqrt(np.mean(site_error**2))),
        training_site_max_px=float(np.max(site_error)),
        training_chord_angle_rms_rad=(
            float(np.sqrt(np.mean(chord_angle**2))) if chord_angle.size else 0.0
        ),
        jacobian_rank=rank,
        jacobian_condition=condition,
        scaled_jacobian_singular_values=singular,
        scaled_jacobian_weakest_direction=weakest,
        active_bounds=active_bounds,
        model_evaluation_count=model_evaluation_count,
        optimizer_function_evaluation_count=int(optimized.nfev),
        optimizer_jacobian_evaluation_count=int(optimized.njev or 0),
    )


__all__ = [
    "DETECTOR_CALIBRATION_PARAMETER_NAMES",
    "INCIDENCE_ANGLE_DELTA_PARAMETER_NAME",
    "SHARED_GEOMETRY_PARAMETER_NAMES",
    "DetectorCalibrationCorrectionBounds",
    "DetectorCalibrationCorrections",
    "IncidenceAngleDeltaBounds",
    "IndexedGeometryFitResult",
    "IndexedGeometryImage",
    "IndexedGeometryImageMetrics",
    "IndexedGeometryRootAuditImage",
    "IndexedGeometrySeriesMetrics",
    "IndexedGeometrySeriesRootAudit",
    "SharedGeometryCorrectionBounds",
    "SharedGeometryCorrections",
    "apply_detector_calibration_corrections",
    "apply_shared_geometry_corrections",
    "audit_indexed_geometry_series_roots",
    "evaluate_indexed_geometry_series_metrics",
    "evaluate_indexed_geometry_series_residual",
    "fit_indexed_geometry_series",
]
