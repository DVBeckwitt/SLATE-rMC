"""Detector-native bounded geometry fitting for exact integer-L marker sites."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from painted_ewald import ContinuousEwaldCoating
from painted_ewald.rotations import mosaic_axes
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.validity import ValidityCode
from rasim_next.geometry import build_incident_states, compose_intrinsic_xy_rotation
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.materials import material_optics
from rasim_next.pipeline.configured_simulation import (
    ConfiguredSimulationInputs,
    DetectorIntegerLMarkers,
    build_nominal_ewald_context,
    build_source_averaged_detector,
    evaluate_nominal_integer_l_markers,
    sample_configured_source,
    solve_integer_l_ewald_roots,
)
from rasim_next.pipeline.continuous_detector import DetectorEwaldMeasure
from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorCoordinateIntensity

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_PARAMETER_NAMES = (
    "detector_column_tilt_rad",
    "detector_row_tilt_rad",
    "sample_normal_x_tilt_rad",
    "sample_normal_y_tilt_rad",
)
_RANK_RELATIVE_TOLERANCE = 1.0e-8
_MAXIMUM_JACOBIAN_CONDITION = 1.0e8
_GEOMETRY_PARAMETERIZATION_ID = "detector_xy_plus_pivoted_effective_sample_normal_xy.v2"
_PREDICTION_STATUSES = frozenset(code.value for code in ValidityCode) | {
    "BRANCH_CHANGED",
    "INTEGER_L_MISMATCH",
    "ROOT_MISSING",
    "ROOT_TANGENT",
}


class GeometryRankError(ValueError):
    """Raised when the declared marker/parameter pack is not identifiable."""


class GeometryPredictionError(ValueError):
    """Raised when a frozen marker changes physical topology during a fit."""


def _readonly_float_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    array = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class GeometryCorrections:
    """Identifiable local pose corrections relative to one frozen instrument."""

    detector_column_tilt_rad: float
    detector_row_tilt_rad: float
    sample_normal_x_tilt_rad: float
    sample_normal_y_tilt_rad: float

    def __post_init__(self) -> None:
        for name in _PARAMETER_NAMES:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    @classmethod
    def zero(cls) -> GeometryCorrections:
        return cls(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_array(cls, value: ArrayLike) -> GeometryCorrections:
        array = _readonly_float_array(value, (4,), "geometry corrections")
        return cls(*(float(item) for item in array))

    def as_array(self) -> FloatArray:
        return _readonly_float_array(
            tuple(getattr(self, name) for name in _PARAMETER_NAMES),
            (4,),
            "geometry corrections",
        )


@dataclass(frozen=True, slots=True)
class GeometryCorrectionBounds:
    """Hard local correction bounds in the same order as ``GeometryCorrections``."""

    lower: GeometryCorrections
    upper: GeometryCorrections

    def __post_init__(self) -> None:
        if not isinstance(self.lower, GeometryCorrections) or not isinstance(
            self.upper, GeometryCorrections
        ):
            raise TypeError("lower and upper must be GeometryCorrections")
        if np.any(self.lower.as_array() >= self.upper.as_array()):
            raise ValueError("every geometry lower bound must be smaller than its upper bound")

    @classmethod
    def rasim_reduced_pose(cls) -> GeometryCorrectionBounds:
        """Map RA-SIM's local angular shells onto the identifiable correction pack.

        The detector limits are direct legacy pitch/yaw bounds. RA-SIM bounds each raw
        sample/goniometer angular correction by five degrees, but has no separate second
        effective-normal coordinate; both reduced sample-normal components conservatively use
        that same shell.
        """

        return cls(
            lower=GeometryCorrections.from_array(np.radians((-10.0, -10.0, -5.0, -5.0))),
            upper=GeometryCorrections.from_array(np.radians((10.0, 10.0, 5.0, 5.0))),
        )


@dataclass(frozen=True, slots=True, order=True)
class IntegerLMarkerKey:
    """Physical marker identity, including the seam-safe analytic beta-root side."""

    family_m: int
    integer_L: int
    branch: int
    root_sign: int
    representative_rod_hk: tuple[int, int]

    def __post_init__(self) -> None:
        values = (self.family_m, self.integer_L, self.branch, self.root_sign)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values
        ):
            raise TypeError("integer-L marker identity fields must be integers")
        rod_hk = tuple(self.representative_rod_hk)
        if (
            self.family_m <= 0
            or self.branch not in {1, 2}
            or self.root_sign not in {-1, 1}
            or len(rod_hk) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in rod_hk
            )
        ):
            raise ValueError("invalid non-specular integer-L marker identity")
        object.__setattr__(self, "family_m", int(self.family_m))
        object.__setattr__(self, "integer_L", int(self.integer_L))
        object.__setattr__(self, "branch", int(self.branch))
        object.__setattr__(self, "root_sign", int(self.root_sign))
        object.__setattr__(self, "representative_rod_hk", (int(rod_hk[0]), int(rod_hk[1])))

    @property
    def tag_branch(self) -> int:
        """User-facing detector branch: negative beta-root side 1, positive side 2."""

        return 1 if self.root_sign < 0 else 2


def _validate_tag_key_pack(keys: tuple[IntegerLMarkerKey, ...], record_name: str) -> None:
    if len(set(keys)) != len(keys):
        raise ValueError(f"integer-L {record_name} identities must be unique")
    tag_identities = tuple((key.family_m, key.integer_L, key.tag_branch) for key in keys)
    if len(set(tag_identities)) != len(tag_identities):
        raise ValueError("each (m,L,tag_branch) may have only one detector tag")
    paired: dict[tuple[int, int], dict[int, IntegerLMarkerKey]] = {}
    for key in keys:
        paired.setdefault((key.family_m, key.integer_L), {})[key.tag_branch] = key
    for sides in paired.values():
        if set(sides) != {1, 2}:
            continue
        if sides[1].branch != sides[2].branch:
            raise ValueError("paired detector tags must share one analytic Ewald branch")
        if sides[1].representative_rod_hk != sides[2].representative_rod_hk:
            raise ValueError("paired detector tags must share one representative physical rod")


def _marker_keys(
    markers: DetectorIntegerLMarkers,
    indices: NDArray[np.int64],
) -> tuple[IntegerLMarkerKey, ...]:
    keys: list[IntegerLMarkerKey] = []
    for index in indices:
        row = int(index)
        if int(markers.family_m[row]) == 0:
            raise ValueError("m=0 specular markers are not part of this geometry-fit contract")
        keys.append(
            IntegerLMarkerKey(
                family_m=int(markers.family_m[row]),
                integer_L=int(markers.integer_L[row]),
                branch=int(markers.branch[row]),
                root_sign=int(markers.root_sign[row]),
                representative_rod_hk=markers.contributing_rod_hk[row][0],
            )
        )
    return tuple(keys)


def _selection_indices(selection: ArrayLike | None, size: int) -> NDArray[np.int64]:
    if selection is None:
        return np.arange(size, dtype=np.int64)
    supplied = np.asarray(selection)
    if supplied.dtype == np.bool_:
        if supplied.shape != (size,):
            raise ValueError("boolean marker selection must match the observation count")
        return np.flatnonzero(supplied)
    if supplied.ndim != 1 or not np.issubdtype(supplied.dtype, np.integer):
        raise ValueError("marker selection must be a one-dimensional integer or boolean array")
    indices = np.asarray(supplied, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= size) or np.unique(indices).size != indices.size:
        raise ValueError("marker selection indices must be unique and in range")
    return indices


@dataclass(frozen=True, slots=True)
class IntegerLMarkerObservations:
    """Frozen detector-native marker positions and their per-site covariance."""

    keys: tuple[IntegerLMarkerKey, ...]
    coordinates_px: FloatArray
    covariance_px2: FloatArray
    reference_wavelength_A: float
    whitening_matrix_px_inv: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        keys = tuple(self.keys)
        if not keys or any(not isinstance(key, IntegerLMarkerKey) for key in keys):
            raise ValueError("keys must contain at least one IntegerLMarkerKey")
        _validate_tag_key_pack(keys, "observation")
        coordinates = _readonly_float_array(self.coordinates_px, (len(keys), 2), "coordinates_px")
        covariance = _readonly_float_array(self.covariance_px2, (len(keys), 2, 2), "covariance_px2")
        if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), rtol=0.0, atol=0.0):
            raise ValueError("each marker covariance must be symmetric")
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("each marker covariance must be positive definite") from error
        whitening = np.linalg.inv(cholesky)
        whitening.setflags(write=False)
        wavelength = float(self.reference_wavelength_A)
        if not math.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("reference_wavelength_A must be finite and positive")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "coordinates_px", coordinates)
        object.__setattr__(self, "covariance_px2", covariance)
        object.__setattr__(self, "reference_wavelength_A", wavelength)
        object.__setattr__(self, "whitening_matrix_px_inv", whitening)

    @classmethod
    def from_markers(
        cls,
        markers: DetectorIntegerLMarkers,
        *,
        selection: ArrayLike | None = None,
        sigma_px: float = 1.0,
    ) -> IntegerLMarkerObservations:
        if not isinstance(markers, DetectorIntegerLMarkers):
            raise TypeError("markers must be DetectorIntegerLMarkers")
        sigma = float(sigma_px)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_px must be finite and positive")
        indices = _selection_indices(selection, markers.column_px.size)
        keys = _marker_keys(markers, indices)
        coordinates = np.column_stack((markers.column_px[indices], markers.row_px[indices]))
        covariance = np.broadcast_to(np.eye(2) * sigma**2, (indices.size, 2, 2)).copy()
        return cls(keys, coordinates, covariance, markers.reference_wavelength_A)

    @classmethod
    def from_prediction(
        cls,
        prediction: IntegerLMarkerPrediction,
        *,
        reference_wavelength_A: float,
        sigma_px: float = 1.0,
    ) -> IntegerLMarkerObservations:
        """Freeze exact tagged landmarks supplied by one reference detector function."""

        if not isinstance(prediction, IntegerLMarkerPrediction):
            raise TypeError("prediction must be IntegerLMarkerPrediction")
        if not np.all(prediction.active_panel):
            raise ValueError("every reference tagged landmark must lie on the active panel")
        sigma = float(sigma_px)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_px must be finite and positive")
        covariance = np.broadcast_to(
            np.eye(2) * sigma**2,
            (len(prediction.keys), 2, 2),
        ).copy()
        return cls(
            keys=prediction.keys,
            coordinates_px=prediction.coordinates_px,
            covariance_px2=covariance,
            reference_wavelength_A=reference_wavelength_A,
        )

    def subset(self, selection: ArrayLike) -> IntegerLMarkerObservations:
        indices = _selection_indices(selection, len(self.keys))
        return IntegerLMarkerObservations(
            keys=tuple(self.keys[int(index)] for index in indices),
            coordinates_px=self.coordinates_px[indices],
            covariance_px2=self.covariance_px2[indices],
            reference_wavelength_A=self.reference_wavelength_A,
        )


@dataclass(frozen=True, slots=True)
class IntegerLMarkerPrediction:
    keys: tuple[IntegerLMarkerKey, ...]
    coordinates_px: FloatArray
    detector_status: NDArray[np.str_]
    ewald_residual_Ainv: FloatArray
    active_panel: BoolArray = field(init=False)

    def __post_init__(self) -> None:
        keys = tuple(self.keys)
        if not keys or any(not isinstance(key, IntegerLMarkerKey) for key in keys):
            raise ValueError("keys must contain at least one IntegerLMarkerKey")
        _validate_tag_key_pack(keys, "prediction")
        coordinates = _readonly_float_array(self.coordinates_px, (len(keys), 2), "coordinates_px")
        residual = _readonly_float_array(
            self.ewald_residual_Ainv, (len(keys),), "ewald_residual_Ainv"
        )
        if np.any(residual < 0.0):
            raise ValueError("ewald_residual_Ainv must be nonnegative")
        supplied_status = np.asarray(self.detector_status)
        if supplied_status.shape != (len(keys),):
            raise ValueError("detector_status must contain one value per marker")
        status = np.asarray(tuple(str(value) for value in supplied_status), dtype="U32")
        invalid_status = sorted(set(status) - _PREDICTION_STATUSES)
        if invalid_status:
            raise ValueError(f"unsupported detector prediction status: {invalid_status}")
        active = status == ValidityCode.VALID.value
        for value in (status, active):
            value.setflags(write=False)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "coordinates_px", coordinates)
        object.__setattr__(self, "detector_status", status)
        object.__setattr__(self, "ewald_residual_Ainv", residual)
        object.__setattr__(self, "active_panel", active)


@dataclass(frozen=True, slots=True)
class M0IntegerLPrediction:
    """Minimum-mosaic-tilt exact-L landmarks on the m=0 detector function."""

    integer_L: tuple[int, ...]
    coordinates_px: FloatArray
    alpha_rad: FloatArray
    beta_rad: FloatArray
    detector_status: NDArray[np.str_]
    ewald_residual_Ainv: FloatArray
    reference_wavelength_A: float
    active_panel: BoolArray = field(init=False)
    landmark_policy: str = "minimum_mosaic_tilt_exact_L.m0.mean_source_state.v1"

    def __post_init__(self) -> None:
        integer_l = tuple(self.integer_L)
        if (
            not integer_l
            or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in integer_l
            )
            or any(int(value) == 0 for value in integer_l)
            or len(set(integer_l)) != len(integer_l)
        ):
            raise ValueError("integer_L must contain unique nonzero integer identities")
        integer_l = tuple(int(value) for value in integer_l)
        size = len(integer_l)
        coordinates = _readonly_float_array(self.coordinates_px, (size, 2), "coordinates_px")
        alpha = _readonly_float_array(self.alpha_rad, (size,), "alpha_rad")
        beta = _readonly_float_array(self.beta_rad, (size,), "beta_rad")
        residual = _readonly_float_array(
            self.ewald_residual_Ainv,
            (size,),
            "ewald_residual_Ainv",
        )
        if np.any(alpha < 0.0) or np.any(alpha > np.pi):
            raise ValueError("alpha_rad must use the folded interval [0, pi]")
        if np.any(beta < 0.0) or np.any(beta >= 2.0 * np.pi):
            raise ValueError("beta_rad must use [0, 2*pi)")
        if np.any(residual < 0.0):
            raise ValueError("ewald_residual_Ainv must be nonnegative")
        wavelength = float(self.reference_wavelength_A)
        if not math.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("reference_wavelength_A must be finite and positive")
        supplied_status = np.asarray(self.detector_status)
        if supplied_status.shape != (size,):
            raise ValueError("detector_status must contain one value per m=0 landmark")
        status = np.asarray(tuple(str(value) for value in supplied_status), dtype="U32")
        invalid_status = sorted(set(status) - _PREDICTION_STATUSES)
        if invalid_status:
            raise ValueError(f"unsupported detector prediction status: {invalid_status}")
        active = status == ValidityCode.VALID.value
        if self.landmark_policy != "minimum_mosaic_tilt_exact_L.m0.mean_source_state.v1":
            raise ValueError("unsupported m=0 exact-L landmark policy")
        for value in (status, active):
            value.setflags(write=False)
        object.__setattr__(self, "integer_L", integer_l)
        object.__setattr__(self, "coordinates_px", coordinates)
        object.__setattr__(self, "alpha_rad", alpha)
        object.__setattr__(self, "beta_rad", beta)
        object.__setattr__(self, "detector_status", status)
        object.__setattr__(self, "ewald_residual_Ainv", residual)
        object.__setattr__(self, "reference_wavelength_A", wavelength)
        object.__setattr__(self, "active_panel", active)

    @property
    def tag_branch(self) -> int:
        """Return the unique specular-axis branch used by every m=0 landmark."""

        return 0


@dataclass(frozen=True, slots=True)
class M0IntegerLObservations:
    """Frozen m=0 exact-L detector landmarks from one reference function."""

    integer_L: tuple[int, ...]
    coordinates_px: FloatArray
    covariance_px2: FloatArray
    reference_wavelength_A: float
    whitening_matrix_px_inv: FloatArray = field(init=False, repr=False)
    landmark_policy: str = "minimum_mosaic_tilt_exact_L.m0.mean_source_state.v1"

    def __post_init__(self) -> None:
        integer_l = tuple(self.integer_L)
        if (
            len(integer_l) < 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in integer_l
            )
            or any(int(value) == 0 for value in integer_l)
            or len(set(integer_l)) != len(integer_l)
        ):
            raise ValueError(
                "m=0 line observations require at least two unique nonzero integer L values"
            )
        integer_l = tuple(int(value) for value in integer_l)
        size = len(integer_l)
        coordinates = _readonly_float_array(self.coordinates_px, (size, 2), "coordinates_px")
        covariance = _readonly_float_array(
            self.covariance_px2,
            (size, 2, 2),
            "covariance_px2",
        )
        if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), rtol=0.0, atol=0.0):
            raise ValueError("each m=0 landmark covariance must be symmetric")
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("each m=0 landmark covariance must be positive definite") from error
        whitening = np.linalg.inv(cholesky)
        whitening.setflags(write=False)
        wavelength = float(self.reference_wavelength_A)
        if not math.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("reference_wavelength_A must be finite and positive")
        if self.landmark_policy != "minimum_mosaic_tilt_exact_L.m0.mean_source_state.v1":
            raise ValueError("unsupported m=0 exact-L landmark policy")
        object.__setattr__(self, "integer_L", integer_l)
        object.__setattr__(self, "coordinates_px", coordinates)
        object.__setattr__(self, "covariance_px2", covariance)
        object.__setattr__(self, "reference_wavelength_A", wavelength)
        object.__setattr__(self, "whitening_matrix_px_inv", whitening)

    @classmethod
    def from_prediction(
        cls,
        prediction: M0IntegerLPrediction,
        *,
        sigma_px: float = 1.0,
    ) -> M0IntegerLObservations:
        if not isinstance(prediction, M0IntegerLPrediction):
            raise TypeError("prediction must be M0IntegerLPrediction")
        if not np.all(prediction.active_panel):
            raise ValueError("every reference m=0 landmark must lie on the active panel")
        sigma = float(sigma_px)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_px must be finite and positive")
        covariance = np.broadcast_to(
            np.eye(2) * sigma**2,
            (len(prediction.integer_L), 2, 2),
        ).copy()
        return cls(
            integer_L=prediction.integer_L,
            coordinates_px=prediction.coordinates_px,
            covariance_px2=covariance,
            reference_wavelength_A=prediction.reference_wavelength_A,
            landmark_policy=prediction.landmark_policy,
        )


def _corrected_instrument(
    base: CompiledInstrument,
    corrections: GeometryCorrections,
    sample_correction_pivot_lab_m: FloatArray,
) -> CompiledInstrument:
    if not isinstance(corrections, GeometryCorrections):
        raise TypeError("corrections must be GeometryCorrections")
    detector = base.lab_from_detector
    sample = base.lab_from_sample
    detector_rotation = compose_intrinsic_xy_rotation(
        detector.rotation,
        corrections.detector_column_tilt_rad,
        corrections.detector_row_tilt_rad,
    )
    if corrections.sample_normal_x_tilt_rad == 0.0 and corrections.sample_normal_y_tilt_rad == 0.0:
        corrected_sample = sample
    else:
        sample_rotation = compose_intrinsic_xy_rotation(
            sample.rotation,
            corrections.sample_normal_x_tilt_rad,
            corrections.sample_normal_y_tilt_rad,
        )
        sample_delta_lab = sample_rotation @ sample.rotation.T
        sample_motion_lab = RigidTransform.around_pivot(
            rotation=sample_delta_lab,
            pivot_m=sample_correction_pivot_lab_m,
            frame=FrameId.LAB,
        )
        corrected_sample = sample_motion_lab.compose(sample)
    return replace(
        base,
        lab_from_detector=RigidTransform(
            detector_rotation,
            detector.translation_m,
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        lab_from_sample=corrected_sample,
    )


def _resolved_sample_correction_pivot(
    inputs: ConfiguredSimulationInputs,
    supplied_pivot_lab_m: ArrayLike | None,
) -> FloatArray:
    if supplied_pivot_lab_m is not None:
        return _readonly_float_array(
            supplied_pivot_lab_m,
            (3,),
            "sample_correction_pivot_lab_m",
        )
    configured_rotations = inputs.config.instrument.axis_rotations
    if not configured_rotations:
        raise ValueError(
            "geometry fitting requires one common configured goniometer pivot or an explicit "
            "sample_correction_pivot_lab_m"
        )
    common_pivot_lab_m = configured_rotations[0].pivot_lab_m
    if any(
        not np.array_equal(rotation.pivot_lab_m, common_pivot_lab_m)
        for rotation in configured_rotations[1:]
    ):
        raise ValueError(
            "geometry fitting requires one common configured goniometer pivot or an explicit "
            "sample_correction_pivot_lab_m"
        )
    return _readonly_float_array(
        common_pivot_lab_m,
        (3,),
        "sample_correction_pivot_lab_m",
    )


class ContinuousDetectorGeometryModel:
    """Prepared all-state continuous detector field under four rigid-angle corrections."""

    __slots__ = (
        "_inputs",
        "_prepared_detector",
        "_sample_correction_pivot_lab_m",
        "_tag_geometry",
    )

    def __init__(
        self,
        inputs: ConfiguredSimulationInputs,
        *,
        sample_correction_pivot_lab_m: ArrayLike | None = None,
    ) -> None:
        if not isinstance(inputs, ConfiguredSimulationInputs):
            raise TypeError("inputs must be ConfiguredSimulationInputs")
        sample_pivot = _resolved_sample_correction_pivot(
            inputs,
            sample_correction_pivot_lab_m,
        )
        prepared = build_source_averaged_detector(inputs)
        object.__setattr__(self, "_inputs", inputs)
        object.__setattr__(self, "_prepared_detector", prepared)
        object.__setattr__(self, "_sample_correction_pivot_lab_m", sample_pivot)
        object.__setattr__(self, "_tag_geometry", _ExactTagGeometry(inputs, sample_pivot))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ContinuousDetectorGeometryModel is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ContinuousDetectorGeometryModel is immutable")

    @property
    def source_state_count(self) -> int:
        return int(self._inputs.incident.states.incident_state_id.size)

    @property
    def sample_correction_pivot_lab_m(self) -> FloatArray:
        """Return the fixed LAB pivot used by effective sample-normal corrections."""

        return self._sample_correction_pivot_lab_m

    def bind(self, corrections: GeometryCorrections) -> ContinuousDetectorFunction:
        """Bind one immutable callable detector function without evaluating a raster."""

        if not isinstance(corrections, GeometryCorrections):
            raise TypeError("corrections must be GeometryCorrections")
        return ContinuousDetectorFunction(self, corrections)

    def evaluate_detector_coordinates(
        self,
        corrections: GeometryCorrections,
        column_px: ArrayLike,
        row_px: ArrayLike,
    ) -> SourceAveragedDetectorCoordinateIntensity:
        """Evaluate the pre-binned raw density at arbitrary continuous coordinates."""

        instrument = _corrected_instrument(
            self._inputs.instrument,
            corrections,
            self._sample_correction_pivot_lab_m,
        )
        incident = build_incident_states(
            self._inputs.samples,
            self._inputs.material,
            instrument,
        )
        try:
            detector = self._prepared_detector.rebind_geometry(
                incident=incident,
                instrument=instrument,
            )
        except ValueError as error:
            raise GeometryPredictionError(
                f"continuous-field geometry became invalid: {error}"
            ) from error
        return detector.evaluate_detector_coordinates_all_roots(
            column_px,
            row_px,
            execution_backend="cpu",
        )


class _ExactTagGeometry:
    """Internal one-state exact-tag geometry owned by the continuous field model."""

    __slots__ = (
        "_inputs",
        "_nominal_material",
        "_nominal_samples",
        "_sample_correction_pivot_lab_m",
    )

    def __init__(
        self,
        inputs: ConfiguredSimulationInputs,
        sample_correction_pivot_lab_m: FloatArray,
    ) -> None:
        if not isinstance(inputs, ConfiguredSimulationInputs):
            raise TypeError("inputs must be ConfiguredSimulationInputs")
        build_nominal_ewald_context(inputs)
        nominal_samples = sample_configured_source(inputs.config.source, sample_count=1)
        nominal_material = material_optics(inputs.crystal, nominal_samples.wavelength_A)
        object.__setattr__(self, "_inputs", inputs)
        object.__setattr__(self, "_nominal_samples", nominal_samples)
        object.__setattr__(self, "_nominal_material", nominal_material)
        object.__setattr__(self, "_sample_correction_pivot_lab_m", sample_correction_pivot_lab_m)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("exact tag geometry is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("exact tag geometry is immutable")

    @property
    def reference_wavelength_A(self) -> float:
        return float(self._nominal_samples.wavelength_A[0])

    @staticmethod
    def _frozen_nonzero_keys(
        keys: tuple[IntegerLMarkerKey, ...],
    ) -> tuple[IntegerLMarkerKey, ...]:
        frozen = tuple(keys)
        if not frozen or any(not isinstance(key, IntegerLMarkerKey) for key in frozen):
            raise ValueError("keys must contain at least one IntegerLMarkerKey")
        if len(set(frozen)) != len(frozen):
            raise ValueError("prediction keys must be unique")
        return frozen

    @staticmethod
    def _frozen_m0_integer_l(integer_L: tuple[int, ...]) -> tuple[int, ...]:
        frozen = tuple(integer_L)
        if (
            not frozen
            or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in frozen
            )
            or any(int(value) == 0 for value in frozen)
            or len(set(frozen)) != len(frozen)
        ):
            raise ValueError("integer_L must contain unique nonzero integer identities")
        return tuple(int(value) for value in frozen)

    def _detector_context(
        self,
        corrections: GeometryCorrections,
    ) -> tuple[ContinuousEwaldCoating, DetectorEwaldMeasure]:
        if not isinstance(corrections, GeometryCorrections):
            raise TypeError("corrections must be GeometryCorrections")
        instrument = _corrected_instrument(
            self._inputs.instrument,
            corrections,
            self._sample_correction_pivot_lab_m,
        )
        incident = build_incident_states(
            self._nominal_samples,
            self._nominal_material,
            instrument,
        )
        if not bool(incident.states.valid[0]):
            raise GeometryPredictionError(
                f"nominal incident state became {incident.states.status[0].value}"
            )
        coating = ContinuousEwaldCoating(
            self._inputs.bragg_space,
            ki_sample_Ainv=incident.states.k_film_phase_sample_Ainv[0],
        )
        return coating, DetectorEwaldMeasure(
            coating=coating,
            incident=incident,
            material=self._nominal_material,
            instrument=instrument,
        )

    def predict(
        self,
        keys: tuple[IntegerLMarkerKey, ...],
        corrections: GeometryCorrections,
    ) -> IntegerLMarkerPrediction:
        frozen_keys = self._frozen_nonzero_keys(keys)
        coating, detector = self._detector_context(corrections)
        return self._predict_nonzero_with_context(frozen_keys, coating, detector)

    def _predict_nonzero_with_context(
        self,
        frozen_keys: tuple[IntegerLMarkerKey, ...],
        coating: ContinuousEwaldCoating,
        detector: DetectorEwaldMeasure,
    ) -> IntegerLMarkerPrediction:
        rods = {(rod.h, rod.k): rod for rod in self._inputs.bragg_space.config.rods}
        basis = self._inputs.bragg_space.config.reciprocal_basis_Ainv
        crystal_to_sample = self._inputs.bragg_space.config.crystal_to_sample
        ki_sample_Ainv = coating.ki_sample_Ainv
        size = len(frozen_keys)
        coordinates = np.zeros((size, 2), dtype=np.float64)
        residual = np.zeros(size, dtype=np.float64)
        status = np.full(size, "ROOT_MISSING", dtype="U32")
        batches: dict[tuple[tuple[int, int], int], list[tuple[int, float]]] = {}
        for index, key in enumerate(frozen_keys):
            rod = rods.get(key.representative_rod_hk)
            if rod is None or rod.family_m != key.family_m:
                raise ValueError(
                    f"marker rod {key.representative_rod_hk} does not belong to m={key.family_m}"
                )
            roots = solve_integer_l_ewald_roots(
                rod=rod,
                integer_l=key.integer_L,
                reciprocal_basis_Ainv=basis,
                crystal_to_sample=crystal_to_sample,
                ki_sample_Ainv=ki_sample_Ainv,
            )
            if roots is None:
                continue
            if roots.branch != key.branch:
                status[index] = "BRANCH_CHANGED"
                continue
            if key.root_sign not in roots.root_sign:
                status[index] = "ROOT_TANGENT" if roots.root_sign == (0,) else "ROOT_MISSING"
                continue
            root_index = roots.root_sign.index(key.root_sign)
            batches.setdefault((key.representative_rod_hk, key.branch), []).append(
                (index, roots.beta_rad[root_index])
            )

        for (rod_hk, branch), candidates in batches.items():
            indices = np.asarray([item[0] for item in candidates], dtype=np.int64)
            beta = np.asarray([item[1] for item in candidates], dtype=np.float64)
            mapped = detector.map_latent_geometry(
                rod=rods[rod_hk],
                branch=branch,
                alpha_rad=np.zeros(beta.size, dtype=np.float64),
                beta_rad=beta,
            )
            actual_l = mapped.ewald_geometry.L
            for batch_index, marker_index in enumerate(indices):
                key = frozen_keys[int(marker_index)]
                l_tolerance = (
                    131072.0
                    * np.finfo(np.float64).eps
                    * max(abs(float(actual_l[batch_index])), abs(key.integer_L), 1.0)
                )
                if abs(float(actual_l[batch_index]) - key.integer_L) > l_tolerance:
                    status[marker_index] = "INTEGER_L_MISMATCH"
                    continue
                coordinates[marker_index] = (
                    float(mapped.column_px[batch_index]),
                    float(mapped.row_px[batch_index]),
                )
                residual[marker_index] = float(
                    mapped.ewald_geometry.ewald_residual_Ainv[batch_index]
                )
                status[marker_index] = str(mapped.detector_status[batch_index])
        return IntegerLMarkerPrediction(
            keys=frozen_keys,
            coordinates_px=coordinates,
            detector_status=status,
            ewald_residual_Ainv=residual,
        )

    def predict_m0_minimum_tilt_exact_l_landmarks(
        self,
        integer_L: tuple[int, ...],
        corrections: GeometryCorrections,
    ) -> M0IntegerLPrediction:
        """Map the unique minimum-tilt m=0 orientation at each exact nonzero L.

        An exact m=0 L section intersects the mosaic/Ewald manifold in a curve. This method
        fixes one reproducible landmark by choosing the reciprocal-axis direction closest to the
        unmosaicked axis. It is a geometry landmark on the continuous detector function, not an
        intensity maximum or centroid.
        """

        frozen_l = self._frozen_m0_integer_l(integer_L)
        coating, detector = self._detector_context(corrections)
        return self._predict_m0_with_context(frozen_l, coating, detector)

    def _predict_m0_with_context(
        self,
        frozen_l: tuple[int, ...],
        coating: ContinuousEwaldCoating,
        detector: DetectorEwaldMeasure,
    ) -> M0IntegerLPrediction:
        specular_rods = tuple(
            rod for rod in self._inputs.bragg_space.config.rods if rod.family_m == 0
        )
        if len(specular_rods) != 1:
            raise ValueError("minimum-tilt exact-L landmarks require one physical m=0 rod")
        rod = specular_rods[0]
        basis = self._inputs.bragg_space.config.reciprocal_basis_Ainv
        crystal_to_sample = self._inputs.bragg_space.config.crystal_to_sample
        mean_axis_crystal, tilt_axis_crystal = mosaic_axes(basis)
        reference_axis_crystal = np.cross(tilt_axis_crystal, mean_axis_crystal)
        mean_axis_sample = crystal_to_sample @ mean_axis_crystal
        ki_sample = coating.ki_sample_Ainv
        k_norm = float(np.linalg.norm(ki_sample))
        incident_direction = ki_sample / k_norm
        perpendicular_mean = mean_axis_sample - float(mean_axis_sample @ incident_direction) * (
            incident_direction
        )
        perpendicular_norm = float(np.linalg.norm(perpendicular_mean))
        if perpendicular_norm <= 1024.0 * np.finfo(np.float64).eps:
            raise GeometryPredictionError(
                "minimum-tilt m=0 landmark is ambiguous when ki is parallel to the mean axis"
            )
        closest_perpendicular = perpendicular_mean / perpendicular_norm
        b3_norm = float(np.linalg.norm(basis[:, 2]))
        size = len(frozen_l)
        coordinates = np.zeros((size, 2), dtype=np.float64)
        alpha = np.zeros(size, dtype=np.float64)
        beta = np.zeros(size, dtype=np.float64)
        residual = np.zeros(size, dtype=np.float64)
        status = np.full(size, "ROOT_MISSING", dtype="U32")
        feasible_indices: list[int] = []
        for index, integer_l in enumerate(frozen_l):
            u_Ainv = integer_l * b3_norm
            z = -u_Ainv / (2.0 * k_norm)
            tolerance = 4096.0 * np.finfo(np.float64).eps
            if abs(z) > 1.0 + tolerance:
                continue
            z = min(1.0, max(-1.0, z))
            direction_sample = (
                z * incident_direction + math.sqrt(max(0.0, 1.0 - z * z)) * closest_perpendicular
            )
            direction_crystal = crystal_to_sample.T @ direction_sample
            cosine_alpha = min(
                1.0,
                max(-1.0, float(direction_crystal @ mean_axis_crystal)),
            )
            alpha[index] = math.acos(cosine_alpha)
            beta[index] = math.atan2(
                float(direction_crystal @ tilt_axis_crystal),
                float(direction_crystal @ reference_axis_crystal),
            ) % (2.0 * np.pi)
            feasible_indices.append(index)

        if feasible_indices:
            indices = np.asarray(feasible_indices, dtype=np.int64)
            mapped = detector.map_specular_geometry(
                rod=rod,
                alpha_rad=alpha[indices],
                beta_rad=beta[indices],
            ).geometry
            actual_l = mapped.ewald_geometry.L
            for batch_index, landmark_index in enumerate(indices):
                integer_l = frozen_l[int(landmark_index)]
                l_tolerance = (
                    131072.0
                    * np.finfo(np.float64).eps
                    * max(abs(float(actual_l[batch_index])), abs(integer_l), 1.0)
                )
                if abs(float(actual_l[batch_index]) - integer_l) > l_tolerance:
                    status[landmark_index] = "INTEGER_L_MISMATCH"
                    continue
                coordinates[landmark_index] = (
                    float(mapped.column_px[batch_index]),
                    float(mapped.row_px[batch_index]),
                )
                residual[landmark_index] = float(
                    mapped.ewald_geometry.ewald_residual_Ainv[batch_index]
                )
                status[landmark_index] = str(mapped.detector_status[batch_index])
        return M0IntegerLPrediction(
            integer_L=frozen_l,
            coordinates_px=coordinates,
            alpha_rad=alpha,
            beta_rad=beta,
            detector_status=status,
            ewald_residual_Ainv=residual,
            reference_wavelength_A=self.reference_wavelength_A,
        )

    def predict_tagged_landmarks(
        self,
        nonzero_keys: tuple[IntegerLMarkerKey, ...],
        m0_integer_L: tuple[int, ...],
        corrections: GeometryCorrections,
    ) -> tuple[IntegerLMarkerPrediction, M0IntegerLPrediction]:
        """Predict both tag groups while constructing incident/detector geometry once."""

        frozen_keys = self._frozen_nonzero_keys(nonzero_keys)
        frozen_l = self._frozen_m0_integer_l(m0_integer_L)
        coating, detector = self._detector_context(corrections)
        return (
            self._predict_nonzero_with_context(frozen_keys, coating, detector),
            self._predict_m0_with_context(frozen_l, coating, detector),
        )


@dataclass(frozen=True, slots=True)
class ContinuousDetectorFunction:
    """One immutable continuous detector field with associated exact-L landmarks."""

    model: ContinuousDetectorGeometryModel = field(repr=False)
    corrections: GeometryCorrections

    def __post_init__(self) -> None:
        if not isinstance(self.model, ContinuousDetectorGeometryModel):
            raise TypeError("model must be ContinuousDetectorGeometryModel")
        if not isinstance(self.corrections, GeometryCorrections):
            raise TypeError("corrections must be GeometryCorrections")

    @property
    def source_state_count(self) -> int:
        return self.model.source_state_count

    @property
    def measure_id(self) -> str:
        return "raw_detector_coordinate_density_A2_per_px2.v1"

    @property
    def instrument(self) -> CompiledInstrument:
        """Return the corrected pose owned by this immutable detector function."""

        return _corrected_instrument(
            self.model._inputs.instrument,
            self.corrections,
            self.model._sample_correction_pivot_lab_m,
        )

    @property
    def tag_incident_state_policy(self) -> str:
        """Return the single deterministic incident state used for all exact tags."""

        return "nominal_source_center.zero_divergence.mean_wavelength.v1"

    @property
    def tag_incident_state_contributes_to_intensity(self) -> bool:
        """The nominal companion carries tag geometry, not empirical source mass."""

        return False

    def evaluate_detector_coordinates(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
    ) -> SourceAveragedDetectorCoordinateIntensity:
        return self.model.evaluate_detector_coordinates(
            self.corrections,
            column_px,
            row_px,
        )

    def __call__(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
    ) -> SourceAveragedDetectorCoordinateIntensity:
        return self.evaluate_detector_coordinates(column_px, row_px)

    def predict_integer_l_tags(
        self,
        keys: tuple[IntegerLMarkerKey, ...],
    ) -> IntegerLMarkerPrediction:
        """Return nominal-source exact-L landmarks on this detector-function domain."""

        return self.model._tag_geometry.predict(keys, self.corrections)

    def predict_m0_minimum_tilt_exact_l_landmarks(
        self,
        integer_L: tuple[int, ...],
    ) -> M0IntegerLPrediction:
        return self.model._tag_geometry.predict_m0_minimum_tilt_exact_l_landmarks(
            integer_L,
            self.corrections,
        )

    def predict_tagged_landmarks(
        self,
        nonzero_keys: tuple[IntegerLMarkerKey, ...],
        m0_integer_L: tuple[int, ...],
    ) -> tuple[IntegerLMarkerPrediction, M0IntegerLPrediction]:
        return self.model._tag_geometry.predict_tagged_landmarks(
            nonzero_keys,
            m0_integer_L,
            self.corrections,
        )


def _marker_chord_pairs(
    keys: tuple[IntegerLMarkerKey, ...],
) -> tuple[tuple[int, int], ...]:
    groups: dict[tuple[int, int], dict[int, int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault((key.family_m, key.integer_L), {})[key.root_sign] = index
    pairs: list[tuple[int, int]] = []
    for _, sides in sorted(groups.items()):
        if set(sides) != {-1, 1}:
            continue
        pairs.append((sides[-1], sides[1]))
    return tuple(pairs)


def _signed_line_angle_rad(target_vector: FloatArray, trial_vector: FloatArray) -> float:
    target_norm = float(np.linalg.norm(target_vector))
    trial_norm = float(np.linalg.norm(trial_vector))
    scale = max(target_norm, trial_norm, 1.0)
    if min(target_norm, trial_norm) <= 1024.0 * np.finfo(np.float64).eps * scale:
        raise GeometryPredictionError("a tagged detector line collapsed to zero span")
    target = target_vector / target_norm
    trial = trial_vector / trial_norm
    determinant = float(target[0] * trial[1] - target[1] * trial[0])
    dot = float(target @ trial)
    return math.atan2(determinant, dot)


def _nonzero_chord_angles_and_residual_px(
    observations: IntegerLMarkerObservations,
    prediction: IntegerLMarkerPrediction,
) -> tuple[FloatArray, FloatArray]:
    pairs = _marker_chord_pairs(observations.keys)
    angle = np.empty(len(pairs), dtype=np.float64)
    residual = np.empty(len(pairs), dtype=np.float64)
    for pair_index, (negative_index, positive_index) in enumerate(pairs):
        target_vector = (
            observations.coordinates_px[positive_index]
            - observations.coordinates_px[negative_index]
        )
        trial_vector = (
            prediction.coordinates_px[positive_index] - prediction.coordinates_px[negative_index]
        )
        delta = _signed_line_angle_rad(target_vector, trial_vector)
        angle[pair_index] = delta
        residual[pair_index] = float(np.linalg.norm(target_vector)) * math.sin(0.5 * delta)
    return angle, residual


def _m0_principal_line(
    coordinates_px: FloatArray, integer_L: tuple[int, ...]
) -> tuple[FloatArray, float]:
    order = np.argsort(np.asarray(integer_L, dtype=np.int64))
    coordinate = np.asarray(coordinates_px[order], dtype=np.float64)
    ell = np.asarray(integer_L, dtype=np.float64)[order]
    centered = coordinate - np.mean(coordinate, axis=0)
    covariance = centered.T @ centered
    trace = float(np.trace(covariance))
    eigengap = math.hypot(
        float(covariance[0, 0] - covariance[1, 1]),
        2.0 * float(covariance[0, 1]),
    )
    if trace <= 0.0 or eigengap <= 1024.0 * np.finfo(np.float64).eps * max(trace, 1.0):
        raise GeometryPredictionError("m=0 exact-L landmarks do not define a stable detector line")
    angle = 0.5 * math.atan2(
        2.0 * float(covariance[0, 1]),
        float(covariance[0, 0] - covariance[1, 1]),
    )
    direction = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
    orientation = float(np.sum((ell - np.mean(ell)) * (centered @ direction)))
    if abs(orientation) <= 1024.0 * np.finfo(np.float64).eps * max(trace, 1.0):
        raise GeometryPredictionError("m=0 exact-L line has ambiguous increasing-L orientation")
    if orientation < 0.0:
        direction = -direction
    projection = coordinate @ direction
    span = float(np.max(projection) - np.min(projection))
    if span <= 1024.0 * np.finfo(np.float64).eps * max(float(np.max(np.abs(coordinate))), 1.0):
        raise GeometryPredictionError("m=0 exact-L line has negligible span")
    return direction, span


def _m0_line_angle_and_residual_px(
    observations: M0IntegerLObservations,
    prediction: M0IntegerLPrediction,
) -> tuple[float, float]:
    target_direction, target_span = _m0_principal_line(
        observations.coordinates_px,
        observations.integer_L,
    )
    trial_direction, _ = _m0_principal_line(
        prediction.coordinates_px,
        prediction.integer_L,
    )
    delta = _signed_line_angle_rad(target_direction, trial_direction)
    return delta, target_span * math.sin(0.5 * delta)


def _typical_observation_sigma_px(covariance_px2: FloatArray) -> float:
    variance = 0.5 * (covariance_px2[:, 0, 0] + covariance_px2[:, 1, 1])
    return math.sqrt(float(np.mean(variance)))


_TrialLandmarkPredictor = Callable[
    [GeometryCorrections],
    tuple[IntegerLMarkerPrediction, M0IntegerLPrediction | None],
]


def evaluate_tagged_geometry_objective_residual(
    observations: IntegerLMarkerObservations,
    prediction: IntegerLMarkerPrediction,
    *,
    m0_observations: M0IntegerLObservations | None = None,
    m0_prediction: M0IntegerLPrediction | None = None,
) -> FloatArray:
    """Evaluate the declared whitened site-plus-line objective without optimization.

    Ordering is nonzero-tag ``(column,row)`` residuals in key order, paired nonzero
    half-angle residuals in sorted ``(m,L)`` order, then—when supplied—m=0
    ``(column,row)`` residuals in L-record order and the single increasing-L TLS-line
    half-angle residual.
    """

    if not isinstance(observations, IntegerLMarkerObservations):
        raise TypeError("observations must be IntegerLMarkerObservations")
    if not isinstance(prediction, IntegerLMarkerPrediction):
        raise TypeError("prediction must be IntegerLMarkerPrediction")
    if prediction.keys != observations.keys:
        raise ValueError("prediction keys must exactly match observation keys")
    if (m0_observations is None) != (m0_prediction is None):
        raise ValueError("m=0 observations and prediction must be supplied together")
    if m0_observations is not None:
        if not isinstance(m0_observations, M0IntegerLObservations) or not isinstance(
            m0_prediction,
            M0IntegerLPrediction,
        ):
            raise TypeError("m=0 records must use the declared observation and prediction types")
        if m0_prediction.integer_L != m0_observations.integer_L:
            raise ValueError("m=0 prediction L identities must exactly match observations")
        wavelength_scale = max(
            m0_prediction.reference_wavelength_A,
            m0_observations.reference_wavelength_A,
            1.0,
        )
        if not math.isclose(
            m0_prediction.reference_wavelength_A,
            m0_observations.reference_wavelength_A,
            rel_tol=0.0,
            abs_tol=256.0 * np.finfo(np.float64).eps * wavelength_scale,
        ):
            raise ValueError("m=0 prediction wavelength must match observations")
    if not np.all(prediction.active_panel):
        invalid = tuple(
            f"{observations.keys[index]}:{prediction.detector_status[index]}"
            for index in np.flatnonzero(~prediction.active_panel)
        )
        raise GeometryPredictionError("frozen marker topology changed: " + "; ".join(invalid))
    delta = prediction.coordinates_px - observations.coordinates_px
    weighted = np.einsum(
        "nij,nj->ni",
        observations.whitening_matrix_px_inv,
        delta,
        optimize=True,
    )
    _, chord_residual_px = _nonzero_chord_angles_and_residual_px(observations, prediction)
    pieces = [
        np.asarray(weighted.reshape(-1), dtype=np.float64),
        chord_residual_px / _typical_observation_sigma_px(observations.covariance_px2),
    ]
    if m0_observations is not None:
        if m0_prediction is None:
            raise AssertionError("validated m=0 prediction unexpectedly disappeared")
        if not np.all(m0_prediction.active_panel):
            invalid = tuple(
                f"L={m0_observations.integer_L[index]}:{m0_prediction.detector_status[index]}"
                for index in np.flatnonzero(~m0_prediction.active_panel)
            )
            raise GeometryPredictionError(
                "frozen m=0 landmark topology changed: " + "; ".join(invalid)
            )
        m0_delta = m0_prediction.coordinates_px - m0_observations.coordinates_px
        m0_weighted = np.einsum(
            "nij,nj->ni",
            m0_observations.whitening_matrix_px_inv,
            m0_delta,
            optimize=True,
        )
        _, m0_line_residual_px = _m0_line_angle_and_residual_px(
            m0_observations,
            m0_prediction,
        )
        pieces.extend(
            (
                np.asarray(m0_weighted.reshape(-1), dtype=np.float64),
                np.asarray(
                    [
                        m0_line_residual_px
                        / _typical_observation_sigma_px(m0_observations.covariance_px2)
                    ],
                    dtype=np.float64,
                ),
            )
        )
    result = np.concatenate(pieces)
    result.setflags(write=False)
    return result


def _finite_difference_jacobian(
    function: Callable[[FloatArray], FloatArray],
    values: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    step_rad: float = 1.0e-5,
) -> FloatArray:
    baseline = np.asarray(function(values), dtype=np.float64)
    jacobian = np.empty((baseline.size, values.size), dtype=np.float64)
    for parameter_index in range(values.size):
        step = min(step_rad, 0.25 * float(upper[parameter_index] - lower[parameter_index]))
        forward = values.copy()
        backward = values.copy()
        if values[parameter_index] - step >= lower[parameter_index] and (
            values[parameter_index] + step <= upper[parameter_index]
        ):
            forward[parameter_index] += step
            backward[parameter_index] -= step
            jacobian[:, parameter_index] = (
                np.asarray(function(forward)) - np.asarray(function(backward))
            ) / (2.0 * step)
        elif values[parameter_index] + step <= upper[parameter_index]:
            forward[parameter_index] += step
            jacobian[:, parameter_index] = (np.asarray(function(forward)) - baseline) / step
        else:
            backward[parameter_index] -= step
            jacobian[:, parameter_index] = (baseline - np.asarray(function(backward))) / step
    return jacobian


def _rank_diagnostics(
    jacobian: FloatArray, parameter_scale: FloatArray
) -> tuple[int, float, FloatArray]:
    scaled = jacobian * parameter_scale[None, :]
    singular = np.linalg.svd(scaled, compute_uv=False)
    threshold = (
        _RANK_RELATIVE_TOLERANCE * singular[0] if singular.size and singular[0] > 0.0 else math.inf
    )
    rank = int(np.count_nonzero(singular > threshold))
    condition = (
        float(singular[0] / singular[-1])
        if rank == scaled.shape[1] and singular[-1] > 0.0
        else math.inf
    )
    singular.setflags(write=False)
    return rank, condition, singular


@dataclass(frozen=True, slots=True)
class GeometryFitResult:
    corrections: GeometryCorrections
    success: bool
    message: str
    training_site_rms_px: float
    training_site_max_px: float
    training_chord_angle_rms_rad: float
    training_m0_line_angle_rad: float
    chord_count: int
    m0_landmark_count: int
    jacobian_rank: int
    jacobian_condition: float
    scaled_jacobian_singular_values: FloatArray
    active_bounds: BoolArray
    model_evaluation_count: int
    optimizer_function_evaluation_count: int
    optimizer_jacobian_evaluation_count: int
    parameterization_id: str = _GEOMETRY_PARAMETERIZATION_ID

    def __post_init__(self) -> None:
        if not isinstance(self.corrections, GeometryCorrections):
            raise TypeError("corrections must be GeometryCorrections")
        if type(self.success) is not bool:
            raise TypeError("success must be a boolean")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be nonempty")
        for name in (
            "training_site_rms_px",
            "training_site_max_px",
            "training_chord_angle_rms_rad",
            "training_m0_line_angle_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.training_site_max_px < self.training_site_rms_px:
            raise ValueError("training site maximum cannot be smaller than its RMS")
        for name in ("chord_count", "m0_landmark_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        if (
            isinstance(self.jacobian_rank, bool)
            or not isinstance(self.jacobian_rank, (int, np.integer))
            or not 0 <= self.jacobian_rank <= 4
        ):
            raise ValueError("jacobian_rank must be an integer from zero through four")
        condition = float(self.jacobian_condition)
        if math.isnan(condition) or condition < 1.0:
            raise ValueError("jacobian_condition must be at least one")
        if self.success and (
            self.jacobian_rank != 4
            or not math.isfinite(condition)
            or condition > _MAXIMUM_JACOBIAN_CONDITION
        ):
            raise ValueError("a successful fit requires a full-rank, conditioned Jacobian")
        for name in (
            "model_evaluation_count",
            "optimizer_function_evaluation_count",
            "optimizer_jacobian_evaluation_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        if self.model_evaluation_count < self.optimizer_function_evaluation_count:
            raise ValueError("model evaluation count cannot be smaller than optimizer nfev")
        if self.parameterization_id != _GEOMETRY_PARAMETERIZATION_ID:
            raise ValueError("unsupported geometry-fit parameterization")
        singular = _readonly_float_array(
            self.scaled_jacobian_singular_values,
            (4,),
            "scaled_jacobian_singular_values",
        )
        if np.any(singular < 0.0) or np.any(np.diff(singular) > 0.0):
            raise ValueError("scaled Jacobian singular values must be nonnegative and descending")
        active = np.array(self.active_bounds, dtype=np.bool_, copy=True)
        if active.shape != (4,):
            raise ValueError("active_bounds must contain one flag per geometry parameter")
        active.setflags(write=False)
        object.__setattr__(self, "scaled_jacobian_singular_values", singular)
        object.__setattr__(self, "active_bounds", active)
        object.__setattr__(self, "jacobian_rank", int(self.jacobian_rank))
        object.__setattr__(self, "jacobian_condition", condition)


def _fit_landmark_geometry(
    predictor: _TrialLandmarkPredictor,
    reference_wavelength_A: float,
    observations: IntegerLMarkerObservations,
    *,
    initial: GeometryCorrections,
    bounds: GeometryCorrectionBounds,
    m0_observations: M0IntegerLObservations | None = None,
) -> GeometryFitResult:
    """Fit exact detector landmarks through one declared trial-function feature map."""

    if not callable(predictor):
        raise TypeError("predictor must be callable")
    if not isinstance(observations, IntegerLMarkerObservations):
        raise TypeError("observations must be IntegerLMarkerObservations")
    if not isinstance(initial, GeometryCorrections):
        raise TypeError("initial must be GeometryCorrections")
    if not isinstance(bounds, GeometryCorrectionBounds):
        raise TypeError("bounds must be GeometryCorrectionBounds")
    if m0_observations is not None and not isinstance(
        m0_observations,
        M0IntegerLObservations,
    ):
        raise TypeError("m0_observations must be M0IntegerLObservations or None")
    reference_wavelength_A = float(reference_wavelength_A)
    if not math.isfinite(reference_wavelength_A) or reference_wavelength_A <= 0.0:
        raise ValueError("reference_wavelength_A must be finite and positive")
    wavelength_scale = max(reference_wavelength_A, observations.reference_wavelength_A, 1.0)
    if not math.isclose(
        reference_wavelength_A,
        observations.reference_wavelength_A,
        rel_tol=0.0,
        abs_tol=256.0 * np.finfo(np.float64).eps * wavelength_scale,
    ):
        raise ValueError("observation wavelength does not match the geometry model")
    if m0_observations is not None and not math.isclose(
        reference_wavelength_A,
        m0_observations.reference_wavelength_A,
        rel_tol=0.0,
        abs_tol=256.0 * np.finfo(np.float64).eps * wavelength_scale,
    ):
        raise ValueError("m=0 observation wavelength does not match the geometry model")
    lower = bounds.lower.as_array()
    upper = bounds.upper.as_array()
    initial_values = initial.as_array()
    if np.any(initial_values < lower) or np.any(initial_values > upper):
        raise ValueError("initial geometry corrections must lie inside the bounds")
    if observations.coordinates_px.size < initial_values.size:
        raise GeometryRankError("marker residual count is smaller than the parameter rank")

    model_evaluation_count = 0

    def residual(value: FloatArray) -> FloatArray:
        nonlocal model_evaluation_count
        model_evaluation_count += 1
        prediction, m0_prediction = predictor(GeometryCorrections.from_array(value))
        return evaluate_tagged_geometry_objective_residual(
            observations,
            prediction,
            m0_observations=m0_observations,
            m0_prediction=m0_prediction,
        )

    parameter_scale = 0.5 * (upper - lower)
    preflight_jacobian = _finite_difference_jacobian(
        residual,
        initial_values,
        lower,
        upper,
    )
    rank, condition, _ = _rank_diagnostics(preflight_jacobian, parameter_scale)
    if rank < initial_values.size or condition > _MAXIMUM_JACOBIAN_CONDITION:
        raise GeometryRankError(
            f"geometry Jacobian rank/conditioning failed: rank={rank}/4, condition={condition:.6g}"
        )

    optimized = least_squares(
        residual,
        initial_values,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale=np.radians((0.5, 0.5, 0.5, 0.5)),
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=100,
    )
    corrections = GeometryCorrections.from_array(optimized.x)
    prediction, fitted_m0_prediction = predictor(corrections)
    if not np.all(prediction.active_panel):
        raise GeometryPredictionError("the fitted marker set does not remain on the active panel")
    raw_error = prediction.coordinates_px - observations.coordinates_px
    site_error_parts = [np.linalg.norm(raw_error, axis=1)]
    chord_angles, _ = _nonzero_chord_angles_and_residual_px(observations, prediction)
    m0_line_angle = 0.0
    if m0_observations is not None:
        if fitted_m0_prediction is None:
            raise AssertionError("combined tagged prediction did not return m=0 landmarks")
        m0_prediction = fitted_m0_prediction
        if not np.all(m0_prediction.active_panel):
            raise GeometryPredictionError("the fitted m=0 landmark set does not remain on panel")
        site_error_parts.append(
            np.linalg.norm(
                m0_prediction.coordinates_px - m0_observations.coordinates_px,
                axis=1,
            )
        )
        m0_line_angle, _ = _m0_line_angle_and_residual_px(
            m0_observations,
            m0_prediction,
        )
    site_error_px = np.concatenate(site_error_parts)
    final_jacobian = np.asarray(optimized.jac, dtype=np.float64)
    rank, condition, singular = _rank_diagnostics(final_jacobian, parameter_scale)
    active_bounds = np.isclose(optimized.x, lower, rtol=0.0, atol=1.0e-10) | np.isclose(
        optimized.x, upper, rtol=0.0, atol=1.0e-10
    )
    return GeometryFitResult(
        corrections=corrections,
        success=bool(optimized.success and rank == 4 and condition <= _MAXIMUM_JACOBIAN_CONDITION),
        message=str(optimized.message),
        training_site_rms_px=float(np.sqrt(np.mean(site_error_px**2))),
        training_site_max_px=float(np.max(site_error_px)),
        training_chord_angle_rms_rad=(
            float(np.sqrt(np.mean(chord_angles**2))) if chord_angles.size else 0.0
        ),
        training_m0_line_angle_rad=abs(float(m0_line_angle)),
        chord_count=int(chord_angles.size),
        m0_landmark_count=(0 if m0_observations is None else len(m0_observations.integer_L)),
        jacobian_rank=rank,
        jacobian_condition=condition,
        scaled_jacobian_singular_values=singular,
        active_bounds=active_bounds,
        model_evaluation_count=model_evaluation_count + 1,
        optimizer_function_evaluation_count=int(optimized.nfev),
        optimizer_jacobian_evaluation_count=int(optimized.njev or 0),
    )


def fit_tagged_detector_function_geometry(
    model: ContinuousDetectorGeometryModel,
    reference_function: ContinuousDetectorFunction,
    *,
    nonzero_keys: tuple[IntegerLMarkerKey, ...],
    m0_integer_L: tuple[int, ...] = (),
    sigma_px: float = 1.0,
    initial: GeometryCorrections,
    bounds: GeometryCorrectionBounds,
) -> GeometryFitResult:
    """Fit exact tagged landmarks associated with two continuous detector functions.

    The reference remains a callable detector field. Each trial correction binds another callable
    field from the same prepared model. Only exact tagged detector coordinates and the line angles
    derived from those tags enter the objective; no raster, pixel integration, field sampling, or
    centroid is constructed.
    """

    if not isinstance(model, ContinuousDetectorGeometryModel):
        raise TypeError("model must be ContinuousDetectorGeometryModel")
    if not isinstance(reference_function, ContinuousDetectorFunction):
        raise TypeError("reference_function must be ContinuousDetectorFunction")
    if reference_function.model is not model:
        raise ValueError("reference and trial functions must share one prepared detector model")
    frozen_keys = tuple(nonzero_keys)
    frozen_m0 = tuple(m0_integer_L)
    reference_m0: M0IntegerLPrediction | None = None
    if frozen_m0:
        reference_prediction, reference_m0 = reference_function.predict_tagged_landmarks(
            frozen_keys,
            frozen_m0,
        )
    else:
        reference_prediction = reference_function.predict_integer_l_tags(frozen_keys)
    observations = IntegerLMarkerObservations.from_prediction(
        reference_prediction,
        reference_wavelength_A=model._tag_geometry.reference_wavelength_A,
        sigma_px=sigma_px,
    )
    m0_observations: M0IntegerLObservations | None = None
    if frozen_m0:
        if reference_m0 is None:
            raise AssertionError("combined reference prediction did not return m=0 landmarks")
        m0_observations = M0IntegerLObservations.from_prediction(
            reference_m0,
            sigma_px=sigma_px,
        )

    def trial_function_predictor(
        corrections: GeometryCorrections,
    ) -> tuple[IntegerLMarkerPrediction, M0IntegerLPrediction | None]:
        trial_function = model.bind(corrections)
        if frozen_m0:
            return trial_function.predict_tagged_landmarks(frozen_keys, frozen_m0)
        return trial_function.predict_integer_l_tags(frozen_keys), None

    return _fit_landmark_geometry(
        trial_function_predictor,
        model._tag_geometry.reference_wavelength_A,
        observations,
        initial=initial,
        bounds=bounds,
        m0_observations=m0_observations,
    )


@dataclass(frozen=True, slots=True)
class IntegerLSelectionAudit:
    classification: str
    missing_keys: tuple[IntegerLMarkerKey, ...]
    unexpected_keys: tuple[IntegerLMarkerKey, ...]
    expected_count: int
    enumerated_count: int

    def __post_init__(self) -> None:
        if self.classification not in {"SAME", "CHANGED", "MISSING", "AMBIGUOUS"}:
            raise ValueError("unsupported integer-L marker selection classification")
        missing = tuple(self.missing_keys)
        unexpected = tuple(self.unexpected_keys)
        if any(not isinstance(key, IntegerLMarkerKey) for key in missing + unexpected):
            raise TypeError("audit differences must contain IntegerLMarkerKey values")
        if len(set(missing)) != len(missing) or len(set(unexpected)) != len(unexpected):
            raise ValueError("audit differences must contain unique identities")
        for name in ("expected_count", "enumerated_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        if self.classification == "SAME" and (
            missing or unexpected or self.expected_count != self.enumerated_count
        ):
            raise ValueError("a SAME audit cannot contain differences or unequal counts")
        if self.classification == "MISSING" and (not missing or unexpected):
            raise ValueError("a MISSING audit requires missing identities only")
        if self.classification == "CHANGED" and not unexpected:
            raise ValueError("a CHANGED audit requires unexpected identities")
        if self.classification != "AMBIGUOUS" and (
            self.expected_count - len(missing) != self.enumerated_count - len(unexpected)
        ):
            raise ValueError("audit counts and identity differences are inconsistent")
        object.__setattr__(self, "missing_keys", missing)
        object.__setattr__(self, "unexpected_keys", unexpected)


def audit_integer_l_marker_selection(
    detector_function: ContinuousDetectorFunction,
    expected_keys: tuple[IntegerLMarkerKey, ...],
) -> IntegerLSelectionAudit:
    """Independently re-enumerate visible roots after fitting without reassignment."""

    if not isinstance(detector_function, ContinuousDetectorFunction):
        raise TypeError("detector_function must be a ContinuousDetectorFunction")
    expected = tuple(expected_keys)
    if any(not isinstance(key, IntegerLMarkerKey) for key in expected):
        raise TypeError("expected_keys must contain IntegerLMarkerKey values")
    model = detector_function.model
    trial_inputs = replace(
        model._inputs,
        instrument=_corrected_instrument(
            model._inputs.instrument,
            detector_function.corrections,
            model._sample_correction_pivot_lab_m,
        ),
    )
    markers = evaluate_nominal_integer_l_markers(build_nominal_ewald_context(trial_inputs))
    indices = np.flatnonzero(markers.family_m != 0)
    tangent_count = int(np.count_nonzero(markers.root_sign[indices] == 0))
    if tangent_count:
        return IntegerLSelectionAudit(
            classification="AMBIGUOUS",
            missing_keys=(),
            unexpected_keys=(),
            expected_count=len(expected),
            enumerated_count=int(indices.size),
        )
    enumerated = _marker_keys(markers, indices)
    expected_set = set(expected)
    enumerated_set = set(enumerated)
    missing = tuple(sorted(expected_set - enumerated_set))
    unexpected = tuple(sorted(enumerated_set - expected_set))
    if len(expected_set) != len(expected) or len(enumerated_set) != len(enumerated):
        classification = "AMBIGUOUS"
    elif not missing and not unexpected:
        classification = "SAME"
    elif missing and not unexpected:
        classification = "MISSING"
    else:
        classification = "CHANGED"
    return IntegerLSelectionAudit(
        classification=classification,
        missing_keys=missing,
        unexpected_keys=unexpected,
        expected_count=len(expected),
        enumerated_count=len(enumerated),
    )
