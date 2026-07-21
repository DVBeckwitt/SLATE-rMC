"""Detector-native bounded geometry fitting for exact integer-L marker sites."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from painted_ewald import ContinuousEwaldCoating
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
    evaluate_nominal_integer_l_markers,
    sample_configured_source,
    solve_integer_l_ewald_roots,
)
from rasim_next.pipeline.continuous_detector import DetectorEwaldMeasure

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
        if len(set(keys)) != len(keys):
            raise ValueError("integer-L observation identities must be unique")
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
        if len(set(keys)) != len(keys):
            raise ValueError("integer-L prediction identities must be unique")
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


class IntegerLGeometryModel:
    """Compiled one-state exact-marker predictor with frozen non-intensity physics."""

    __slots__ = ("_inputs", "_nominal_material", "_nominal_samples")

    def __init__(self, inputs: ConfiguredSimulationInputs) -> None:
        if not isinstance(inputs, ConfiguredSimulationInputs):
            raise TypeError("inputs must be ConfiguredSimulationInputs")
        build_nominal_ewald_context(inputs)
        nominal_samples = sample_configured_source(inputs.config.source, sample_count=1)
        nominal_material = material_optics(inputs.crystal, nominal_samples.wavelength_A)
        object.__setattr__(self, "_inputs", inputs)
        object.__setattr__(self, "_nominal_samples", nominal_samples)
        object.__setattr__(self, "_nominal_material", nominal_material)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("IntegerLGeometryModel is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("IntegerLGeometryModel is immutable")

    @property
    def inputs(self) -> ConfiguredSimulationInputs:
        return self._inputs

    @property
    def reference_wavelength_A(self) -> float:
        return float(self._nominal_samples.wavelength_A[0])

    def corrected_instrument(self, corrections: GeometryCorrections) -> CompiledInstrument:
        if not isinstance(corrections, GeometryCorrections):
            raise TypeError("corrections must be GeometryCorrections")
        base = self._inputs.instrument
        detector = base.lab_from_detector
        sample = base.lab_from_sample
        detector_rotation = compose_intrinsic_xy_rotation(
            detector.rotation,
            corrections.detector_column_tilt_rad,
            corrections.detector_row_tilt_rad,
        )
        sample_rotation = compose_intrinsic_xy_rotation(
            sample.rotation,
            corrections.sample_normal_x_tilt_rad,
            corrections.sample_normal_y_tilt_rad,
        )
        return replace(
            base,
            lab_from_detector=RigidTransform(
                detector_rotation,
                detector.translation_m,
                FrameId.DETECTOR,
                FrameId.LAB,
            ),
            lab_from_sample=RigidTransform(
                sample_rotation,
                sample.translation_m,
                FrameId.SAMPLE,
                FrameId.LAB,
            ),
        )

    def predict(
        self,
        keys: tuple[IntegerLMarkerKey, ...],
        corrections: GeometryCorrections,
    ) -> IntegerLMarkerPrediction:
        frozen_keys = tuple(keys)
        if not frozen_keys or any(not isinstance(key, IntegerLMarkerKey) for key in frozen_keys):
            raise ValueError("keys must contain at least one IntegerLMarkerKey")
        if len(set(frozen_keys)) != len(frozen_keys):
            raise ValueError("prediction keys must be unique")
        instrument = self.corrected_instrument(corrections)
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
        detector = DetectorEwaldMeasure(
            coating=coating,
            incident=incident,
            material=self._nominal_material,
            instrument=instrument,
        )
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


def _weighted_residual(
    model: IntegerLGeometryModel,
    observations: IntegerLMarkerObservations,
    correction_values: ArrayLike,
) -> FloatArray:
    corrections = GeometryCorrections.from_array(correction_values)
    prediction = model.predict(observations.keys, corrections)
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
    return np.asarray(weighted.reshape(-1), dtype=np.float64)


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
    jacobian_rank: int
    jacobian_condition: float
    scaled_jacobian_singular_values: FloatArray
    active_bounds: BoolArray
    model_evaluation_count: int
    optimizer_function_evaluation_count: int
    optimizer_jacobian_evaluation_count: int
    parameterization_id: str = "detector_xy_plus_effective_sample_normal_xy.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.corrections, GeometryCorrections):
            raise TypeError("corrections must be GeometryCorrections")
        if type(self.success) is not bool:
            raise TypeError("success must be a boolean")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be nonempty")
        for name in ("training_site_rms_px", "training_site_max_px"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.training_site_max_px < self.training_site_rms_px:
            raise ValueError("training site maximum cannot be smaller than its RMS")
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
        if self.parameterization_id != "detector_xy_plus_effective_sample_normal_xy.v1":
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


def fit_integer_l_marker_geometry(
    model: IntegerLGeometryModel,
    observations: IntegerLMarkerObservations,
    *,
    initial: GeometryCorrections,
    bounds: GeometryCorrectionBounds,
) -> GeometryFitResult:
    """Fit the identifiable four-angle pose pack to frozen detector-native sites."""

    if not isinstance(model, IntegerLGeometryModel):
        raise TypeError("model must be IntegerLGeometryModel")
    if not isinstance(observations, IntegerLMarkerObservations):
        raise TypeError("observations must be IntegerLMarkerObservations")
    if not isinstance(initial, GeometryCorrections):
        raise TypeError("initial must be GeometryCorrections")
    if not isinstance(bounds, GeometryCorrectionBounds):
        raise TypeError("bounds must be GeometryCorrectionBounds")
    wavelength_scale = max(model.reference_wavelength_A, observations.reference_wavelength_A, 1.0)
    if not math.isclose(
        model.reference_wavelength_A,
        observations.reference_wavelength_A,
        rel_tol=0.0,
        abs_tol=256.0 * np.finfo(np.float64).eps * wavelength_scale,
    ):
        raise ValueError("observation wavelength does not match the geometry model")
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
        return _weighted_residual(model, observations, value)

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
    prediction = model.predict(observations.keys, corrections)
    if not np.all(prediction.active_panel):
        raise GeometryPredictionError("the fitted marker set does not remain on the active panel")
    raw_error = prediction.coordinates_px - observations.coordinates_px
    site_error_px = np.linalg.norm(raw_error, axis=1)
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
        jacobian_rank=rank,
        jacobian_condition=condition,
        scaled_jacobian_singular_values=singular,
        active_bounds=active_bounds,
        model_evaluation_count=model_evaluation_count + 1,
        optimizer_function_evaluation_count=int(optimized.nfev),
        optimizer_jacobian_evaluation_count=int(optimized.njev or 0),
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
    model: IntegerLGeometryModel,
    corrections: GeometryCorrections,
    expected_keys: tuple[IntegerLMarkerKey, ...],
) -> IntegerLSelectionAudit:
    """Independently re-enumerate visible roots after fitting without reassignment."""

    if not isinstance(model, IntegerLGeometryModel):
        raise TypeError("model must be IntegerLGeometryModel")
    expected = tuple(expected_keys)
    if any(not isinstance(key, IntegerLMarkerKey) for key in expected):
        raise TypeError("expected_keys must contain IntegerLMarkerKey values")
    trial_inputs = replace(model.inputs, instrument=model.corrected_instrument(corrections))
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
