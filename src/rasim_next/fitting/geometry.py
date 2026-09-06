"""Detector-native bounded geometry fitting at exact layer-coordinate landmarks."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq, least_squares

from painted_ewald import (
    EwaldLatentGeometry,
    map_tied_rotation_latent,
)
from painted_ewald.rotations import mosaic_axes
from painted_ewald.types import Rod, RootStatus
from rasim_next.core.contracts import MaterialOptics, canonical_revision_sha256
from rasim_next.core.frames import FrameId
from rasim_next.core.layer_order import CommensurateLayerOrder
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.validity import ValidityCode
from rasim_next.geometry import build_incident_states, compose_intrinsic_xy_rotation
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.materials import material_optics
from rasim_next.pipeline.configured_simulation import (
    ConfiguredGeometryInputs,
    ConfiguredSimulationInputs,
    DetectorIntegerLMarkers,
    build_nominal_ewald_context,
    build_source_averaged_detector,
    evaluate_nominal_integer_l_markers,
    sample_configured_nominal_geometry_source,
    solve_integer_l_ewald_roots,
    solve_layer_l_ewald_roots,
)
from rasim_next.pipeline.continuous_detector import (
    map_ewald_geometry_to_detector,
)
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
_M0_MINIMUM_TILT_LANDMARK_POLICY = "minimum_mosaic_tilt_exact_L.m0.mean_source_state.v1"
_PREDICTION_STATUSES = frozenset(code.value for code in ValidityCode) | {
    "BRANCH_CHANGED",
    "ROOT_MISSING",
    "ROOT_TANGENT",
    "LOCUS_SPLIT",
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


def _marker_whitening(covariance: FloatArray, subject: str) -> FloatArray:
    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), rtol=0.0, atol=0.0):
        raise ValueError(f"each {subject} covariance must be symmetric")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"each {subject} covariance must be positive definite") from error
    whitening = np.linalg.inv(cholesky)
    whitening.setflags(write=False)
    return whitening


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


@dataclass(frozen=True, slots=True, order=True)
class LayerLMarkerKey:
    """Rod-free physical identity for an exact commensurate layer landmark."""

    family_m: int
    layer_order: CommensurateLayerOrder
    branch: int
    root_sign: int
    reciprocal_basis_revision: str

    def __post_init__(self) -> None:
        values = (self.family_m, self.branch, self.root_sign)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values
        ):
            raise TypeError("layer-L marker identity fields must be integers")
        if self.family_m <= 0 or self.branch not in {1, 2} or self.root_sign not in {-1, 1}:
            raise ValueError("invalid non-specular layer-L marker identity")
        if not isinstance(self.layer_order, CommensurateLayerOrder):
            raise TypeError("layer_order must be CommensurateLayerOrder")
        revision = self.reciprocal_basis_revision
        if (
            not isinstance(revision, str)
            or len(revision) != 64
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValueError("reciprocal_basis_revision must be a lowercase SHA-256 revision")
        object.__setattr__(self, "family_m", int(self.family_m))
        object.__setattr__(self, "branch", int(self.branch))
        object.__setattr__(self, "root_sign", int(self.root_sign))

    @property
    def tag_branch(self) -> int:
        return 1 if self.root_sign < 0 else 2


@dataclass(frozen=True, slots=True)
class LayerLMarkerDefinition:
    """A physical key plus the explicit signed rods that realize its detector locus."""

    key: LayerLMarkerKey
    contributing_rod_hk: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, LayerLMarkerKey):
            raise TypeError("key must be LayerLMarkerKey")
        rods: list[tuple[int, int]] = []
        for supplied in tuple(self.contributing_rod_hk):
            rod = tuple(supplied)
            if len(rod) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in rod
            ):
                raise ValueError("contributing_rod_hk must contain integer (h, k) pairs")
            hk = (int(rod[0]), int(rod[1]))
            if hk[0] * hk[0] + hk[0] * hk[1] + hk[1] * hk[1] != self.key.family_m:
                raise ValueError("each contributing rod must belong to the marker family")
            rods.append(hk)
        canonical = tuple(sorted(set(rods)))
        if not canonical:
            raise ValueError("a layer-L marker definition requires at least one contributing rod")
        object.__setattr__(self, "contributing_rod_hk", canonical)

    @property
    def representative_rod_hk(self) -> tuple[int, int]:
        return self.contributing_rod_hk[0]


def _validate_layer_l_definitions(
    definitions: tuple[LayerLMarkerDefinition, ...],
    record_name: str,
) -> None:
    if not definitions or any(
        not isinstance(definition, LayerLMarkerDefinition) for definition in definitions
    ):
        raise ValueError("definitions must contain at least one LayerLMarkerDefinition")
    keys = tuple(definition.key for definition in definitions)
    if len(set(keys)) != len(keys):
        raise ValueError(f"layer-L {record_name} physical keys must be unique")
    if len({key.reciprocal_basis_revision for key in keys}) != 1:
        raise ValueError(f"layer-L {record_name} must use one reciprocal-basis revision")
    tag_identities = tuple(
        (
            key.family_m,
            key.layer_order,
            key.tag_branch,
            key.reciprocal_basis_revision,
        )
        for key in keys
    )
    if len(set(tag_identities)) != len(tag_identities):
        raise ValueError("each (m,L,tag_branch,basis) may have only one detector tag")
    paired: dict[
        tuple[int, CommensurateLayerOrder, str],
        dict[int, LayerLMarkerDefinition],
    ] = {}
    for definition in definitions:
        key = definition.key
        paired.setdefault((key.family_m, key.layer_order, key.reciprocal_basis_revision), {})[
            key.tag_branch
        ] = definition
    for sides in paired.values():
        if set(sides) != {1, 2}:
            continue
        if sides[1].key.branch != sides[2].key.branch:
            raise ValueError("paired detector tags must share one analytic Ewald branch")
        if sides[1].contributing_rod_hk != sides[2].contributing_rod_hk:
            raise ValueError("paired detector tags must share contributing physical rods")


def _validate_tag_key_pack(keys: tuple[IntegerLMarkerKey, ...], record_name: str) -> None:
    if len(set(keys)) != len(keys):
        raise ValueError(f"integer-L {record_name} identities must be unique")
    tag_identities = tuple(
        (key.representative_rod_hk, key.integer_L, key.tag_branch) for key in keys
    )
    if len(set(tag_identities)) != len(tag_identities):
        raise ValueError("each physical (rod,L,tag_branch) may have only one detector tag")
    paired: dict[tuple[tuple[int, int], int], dict[int, IntegerLMarkerKey]] = {}
    for key in keys:
        paired.setdefault((key.representative_rod_hk, key.integer_L), {})[key.tag_branch] = key
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
        whitening = _marker_whitening(covariance, "marker")
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
class LayerLMarkerPrediction:
    """Detector prediction aligned to rod-free exact-layer definitions."""

    definitions: tuple[LayerLMarkerDefinition, ...]
    coordinates_px: FloatArray
    detector_status: NDArray[np.str_]
    ewald_residual_Ainv: FloatArray
    active_panel: BoolArray = field(init=False)

    def __post_init__(self) -> None:
        definitions = tuple(self.definitions)
        _validate_layer_l_definitions(definitions, "prediction")
        size = len(definitions)
        coordinates = _readonly_float_array(self.coordinates_px, (size, 2), "coordinates_px")
        residual = _readonly_float_array(
            self.ewald_residual_Ainv,
            (size,),
            "ewald_residual_Ainv",
        )
        if np.any(residual < 0.0):
            raise ValueError("ewald_residual_Ainv must be nonnegative")
        supplied_status = np.asarray(self.detector_status)
        if supplied_status.shape != (size,):
            raise ValueError("detector_status must contain one value per marker")
        status = np.asarray(tuple(str(value) for value in supplied_status), dtype="U32")
        invalid_status = sorted(set(status) - _PREDICTION_STATUSES)
        if invalid_status:
            raise ValueError(f"unsupported detector prediction status: {invalid_status}")
        active = status == ValidityCode.VALID.value
        for value in (status, active):
            value.setflags(write=False)
        object.__setattr__(self, "definitions", definitions)
        object.__setattr__(self, "coordinates_px", coordinates)
        object.__setattr__(self, "detector_status", status)
        object.__setattr__(self, "ewald_residual_Ainv", residual)
        object.__setattr__(self, "active_panel", active)

    @property
    def keys(self) -> tuple[LayerLMarkerKey, ...]:
        return tuple(definition.key for definition in self.definitions)

    def subset(self, selection: ArrayLike) -> LayerLMarkerPrediction:
        indices = _selection_indices(selection, len(self.definitions))
        return LayerLMarkerPrediction(
            definitions=tuple(self.definitions[int(index)] for index in indices),
            coordinates_px=self.coordinates_px[indices],
            detector_status=self.detector_status[indices],
            ewald_residual_Ainv=self.ewald_residual_Ainv[indices],
        )


@dataclass(frozen=True, slots=True)
class LayerLMarkerObservations:
    """Frozen detector-native observations at exact commensurate layer coordinates."""

    definitions: tuple[LayerLMarkerDefinition, ...]
    coordinates_px: FloatArray
    covariance_px2: FloatArray
    reference_wavelength_A: float
    whitening_matrix_px_inv: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        definitions = tuple(self.definitions)
        _validate_layer_l_definitions(definitions, "observation")
        size = len(definitions)
        coordinates = _readonly_float_array(self.coordinates_px, (size, 2), "coordinates_px")
        covariance = _readonly_float_array(
            self.covariance_px2,
            (size, 2, 2),
            "covariance_px2",
        )
        whitening = _marker_whitening(covariance, "marker")
        wavelength = float(self.reference_wavelength_A)
        if not math.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("reference_wavelength_A must be finite and positive")
        object.__setattr__(self, "definitions", definitions)
        object.__setattr__(self, "coordinates_px", coordinates)
        object.__setattr__(self, "covariance_px2", covariance)
        object.__setattr__(self, "reference_wavelength_A", wavelength)
        object.__setattr__(self, "whitening_matrix_px_inv", whitening)

    @property
    def keys(self) -> tuple[LayerLMarkerKey, ...]:
        return tuple(definition.key for definition in self.definitions)

    @classmethod
    def from_prediction(
        cls,
        prediction: LayerLMarkerPrediction,
        *,
        reference_wavelength_A: float,
        sigma_px: float = 1.0,
    ) -> LayerLMarkerObservations:
        if not isinstance(prediction, LayerLMarkerPrediction):
            raise TypeError("prediction must be LayerLMarkerPrediction")
        if not np.all(prediction.active_panel):
            raise ValueError("every reference layer-L landmark must lie on the active panel")
        sigma = float(sigma_px)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_px must be finite and positive")
        covariance = np.broadcast_to(
            np.eye(2) * sigma**2,
            (len(prediction.definitions), 2, 2),
        ).copy()
        return cls(
            definitions=prediction.definitions,
            coordinates_px=prediction.coordinates_px,
            covariance_px2=covariance,
            reference_wavelength_A=reference_wavelength_A,
        )

    def subset(self, selection: ArrayLike) -> LayerLMarkerObservations:
        indices = _selection_indices(selection, len(self.definitions))
        return LayerLMarkerObservations(
            definitions=tuple(self.definitions[int(index)] for index in indices),
            coordinates_px=self.coordinates_px[indices],
            covariance_px2=self.covariance_px2[indices],
            reference_wavelength_A=self.reference_wavelength_A,
        )


def merge_layer_l_marker_observations(
    baseline: IntegerLMarkerObservations | LayerLMarkerObservations,
    *optional_groups: LayerLMarkerObservations,
    reciprocal_basis_revision: str | None = None,
) -> IntegerLMarkerObservations | LayerLMarkerObservations:
    """Merge already-qualified optional landmarks without duplicating physical loci."""

    if not isinstance(baseline, (IntegerLMarkerObservations, LayerLMarkerObservations)):
        raise TypeError("baseline must be integer- or layer-L marker observations")
    if any(not isinstance(group, LayerLMarkerObservations) for group in optional_groups):
        raise TypeError("optional_groups must contain LayerLMarkerObservations")
    if not optional_groups:
        return baseline
    if isinstance(baseline, IntegerLMarkerObservations):
        if reciprocal_basis_revision is None:
            raise ValueError(
                "reciprocal_basis_revision is required when augmenting integer-L observations"
            )
        baseline = LayerLMarkerObservations(
            definitions=tuple(
                LayerLMarkerDefinition(
                    key=LayerLMarkerKey(
                        family_m=key.family_m,
                        layer_order=CommensurateLayerOrder(key.integer_L),
                        branch=key.branch,
                        root_sign=key.root_sign,
                        reciprocal_basis_revision=reciprocal_basis_revision,
                    ),
                    contributing_rod_hk=(key.representative_rod_hk,),
                )
                for key in baseline.keys
            ),
            coordinates_px=baseline.coordinates_px,
            covariance_px2=baseline.covariance_px2,
            reference_wavelength_A=baseline.reference_wavelength_A,
        )
    elif reciprocal_basis_revision is not None and any(
        key.reciprocal_basis_revision != reciprocal_basis_revision for key in baseline.keys
    ):
        raise ValueError("baseline reciprocal-basis revision does not match the requested revision")
    groups = (baseline, *optional_groups)
    revisions = {key.reciprocal_basis_revision for group in groups for key in group.keys}
    if len(revisions) != 1:
        raise ValueError("merged layer-L observations must share one reciprocal-basis revision")
    wavelength = groups[0].reference_wavelength_A
    scale = max(1.0, *(group.reference_wavelength_A for group in groups))
    if any(
        not math.isclose(
            group.reference_wavelength_A,
            wavelength,
            rel_tol=0.0,
            abs_tol=256.0 * np.finfo(np.float64).eps * scale,
        )
        for group in groups[1:]
    ):
        raise ValueError("merged layer-L observations must share one reference wavelength")

    rows: dict[
        LayerLMarkerKey,
        tuple[LayerLMarkerDefinition, FloatArray, FloatArray],
    ] = {}
    for group in groups:
        for index, definition in enumerate(group.definitions):
            key = definition.key
            existing = rows.get(key)
            coordinate = group.coordinates_px[index]
            covariance = group.covariance_px2[index]
            if existing is None:
                rows[key] = (definition, coordinate, covariance)
                continue
            prior_definition, prior_coordinate, prior_covariance = existing
            if not np.array_equal(prior_coordinate, coordinate) or not np.array_equal(
                prior_covariance,
                covariance,
            ):
                raise ValueError("duplicate physical layer-L observations conflict")
            rows[key] = (
                LayerLMarkerDefinition(
                    key=key,
                    contributing_rod_hk=(
                        *prior_definition.contributing_rod_hk,
                        *definition.contributing_rod_hk,
                    ),
                ),
                prior_coordinate,
                prior_covariance,
            )
    ordered = tuple(rows[key] for key in sorted(rows))
    return LayerLMarkerObservations(
        definitions=tuple(item[0] for item in ordered),
        coordinates_px=np.asarray([item[1] for item in ordered]),
        covariance_px2=np.asarray([item[2] for item in ordered]),
        reference_wavelength_A=wavelength,
    )


@dataclass(frozen=True, slots=True)
class M0IntegerLPrediction:
    """Declared exact-L landmarks on the m=0 detector function."""

    integer_L: tuple[int, ...]
    coordinates_px: FloatArray
    alpha_rad: FloatArray
    beta_rad: FloatArray
    detector_status: NDArray[np.str_]
    ewald_residual_Ainv: FloatArray
    reference_wavelength_A: float
    active_panel: BoolArray = field(init=False)
    landmark_policy: str = _M0_MINIMUM_TILT_LANDMARK_POLICY

    def __post_init__(self) -> None:
        integer_l = tuple(self.integer_L)
        if (
            not integer_l
            or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in integer_l
            )
            or any(int(value) <= 0 for value in integer_l)
            or len(set(integer_l)) != len(integer_l)
        ):
            raise ValueError("integer_L must contain unique positive |L| identities")
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
        if self.landmark_policy != _M0_MINIMUM_TILT_LANDMARK_POLICY:
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
    landmark_policy: str = _M0_MINIMUM_TILT_LANDMARK_POLICY

    def __post_init__(self) -> None:
        integer_l = tuple(self.integer_L)
        if (
            len(integer_l) < 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in integer_l
            )
            or any(int(value) <= 0 for value in integer_l)
            or len(set(integer_l)) != len(integer_l)
        ):
            raise ValueError(
                "m=0 line observations require at least two unique positive |L| identities"
            )
        integer_l = tuple(int(value) for value in integer_l)
        size = len(integer_l)
        coordinates = _readonly_float_array(self.coordinates_px, (size, 2), "coordinates_px")
        covariance = _readonly_float_array(
            self.covariance_px2,
            (size, 2, 2),
            "covariance_px2",
        )
        whitening = _marker_whitening(covariance, "m=0 landmark")
        wavelength = float(self.reference_wavelength_A)
        if not math.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("reference_wavelength_A must be finite and positive")
        if self.landmark_policy != _M0_MINIMUM_TILT_LANDMARK_POLICY:
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


def _frozen_nonzero_keys(
    keys: tuple[IntegerLMarkerKey, ...],
) -> tuple[IntegerLMarkerKey, ...]:
    frozen = tuple(keys)
    if not frozen or any(not isinstance(key, IntegerLMarkerKey) for key in frozen):
        raise ValueError("keys must contain at least one IntegerLMarkerKey")
    if len(set(frozen)) != len(frozen):
        raise ValueError("prediction keys must be unique")
    return frozen


def _frozen_m0_integer_l(integer_L: tuple[int, ...]) -> tuple[int, ...]:
    frozen = tuple(integer_L)
    if (
        not frozen
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in frozen
        )
        or any(int(value) <= 0 for value in frozen)
        or len(set(frozen)) != len(frozen)
    ):
        raise ValueError("integer_L must contain unique positive |L| identities")
    return tuple(int(value) for value in frozen)


def _map_fixed_layer_l_roots_to_detector(
    *,
    rod: Rod,
    branch: int,
    beta_rad: FloatArray,
    layer_l: FloatArray,
    reciprocal_basis_Ainv: FloatArray,
    crystal_to_sample: FloatArray,
    ki_sample_Ainv: FloatArray,
    incident: IncidentTransportResult,
    material: MaterialOptics,
    instrument: CompiledInstrument,
) -> tuple[FloatArray, NDArray[np.str_], FloatArray, FloatArray]:
    """Map already-solved fixed-L roots through the one canonical detector projection."""

    beta = np.asarray(beta_rad, dtype=np.float64)
    ell = np.asarray(layer_l, dtype=np.float64)
    if beta.ndim != 1 or ell.shape != beta.shape or not np.all(np.isfinite(beta + ell)):
        raise ValueError("beta_rad and layer_l must be aligned finite vectors")
    b3_norm_Ainv = float(np.linalg.norm(reciprocal_basis_Ainv[:, 2]))
    q_sample = map_tied_rotation_latent(
        rod=rod,
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        crystal_to_sample=crystal_to_sample,
        alpha_rad=np.zeros(beta.size, dtype=np.float64),
        beta_rad=beta,
        u_Ainv=ell * b3_norm_Ainv,
    )
    kf_sample = q_sample + ki_sample_Ainv[None, :]
    ewald_residual = np.abs(np.linalg.norm(kf_sample, axis=1) - np.linalg.norm(ki_sample_Ainv))
    geometry = EwaldLatentGeometry(
        rod=rod,
        branch=branch,
        alpha_rad=np.zeros(beta.size, dtype=np.float64),
        beta_rad=beta,
        u_Ainv=ell * b3_norm_Ainv,
        L=ell,
        q_sample_Ainv=q_sample,
        kf_sample_Ainv=kf_sample,
        ewald_residual_Ainv=ewald_residual,
        status=np.full(beta.size, RootStatus.REGULAR.value, dtype="U32"),
    )
    mapped = map_ewald_geometry_to_detector(
        geometry,
        incident=incident,
        material=material,
        instrument=instrument,
    )
    coordinates = np.column_stack((mapped.column_px, mapped.row_px))
    status = np.asarray(mapped.detector_status, dtype="U32")
    return coordinates, status, ewald_residual, q_sample


def _predict_single_layer_l_rod(
    *,
    key: LayerLMarkerKey,
    rod: Rod,
    reciprocal_basis_Ainv: FloatArray,
    crystal_to_sample: FloatArray,
    ki_sample_Ainv: FloatArray,
    incident: IncidentTransportResult,
    material: MaterialOptics,
    instrument: CompiledInstrument,
) -> tuple[FloatArray, str, float, FloatArray]:
    roots = solve_layer_l_ewald_roots(
        rod=rod,
        layer_order=key.layer_order,
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        crystal_to_sample=crystal_to_sample,
        ki_sample_Ainv=ki_sample_Ainv,
    )
    empty_coordinate = np.zeros(2, dtype=np.float64)
    empty_q = np.zeros(3, dtype=np.float64)
    if roots is None:
        return empty_coordinate, "ROOT_MISSING", 0.0, empty_q
    if roots.branch != key.branch:
        return empty_coordinate, "BRANCH_CHANGED", 0.0, empty_q
    if key.root_sign not in roots.root_sign:
        status = "ROOT_TANGENT" if roots.root_sign == (0,) else "ROOT_MISSING"
        return empty_coordinate, status, 0.0, empty_q
    beta = roots.beta_rad[roots.root_sign.index(key.root_sign)]
    coordinates, status, residual, q_sample = _map_fixed_layer_l_roots_to_detector(
        rod=rod,
        branch=key.branch,
        beta_rad=np.asarray([beta]),
        layer_l=np.asarray([key.layer_order.as_float()]),
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        crystal_to_sample=crystal_to_sample,
        ki_sample_Ainv=ki_sample_Ainv,
        incident=incident,
        material=material,
        instrument=instrument,
    )
    return coordinates[0], str(status[0]), float(residual[0]), q_sample[0]


class ExactTagGeometryModel:
    """One-ray exact integer-L predictor with no intensity or mosaic dependency."""

    __slots__ = ("_inputs",)

    def __init__(self, inputs: ConfiguredGeometryInputs) -> None:
        if not isinstance(inputs, ConfiguredGeometryInputs):
            raise TypeError("inputs must be ConfiguredGeometryInputs")
        if inputs.samples.incident_sample_id.size != 1:
            raise ValueError("exact-tag geometry requires exactly one nominal source state")
        object.__setattr__(self, "_inputs", inputs)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ExactTagGeometryModel is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ExactTagGeometryModel is immutable")

    def _require_hexagonal_layer_l_metric(self) -> None:
        """Fail before legacy rod-free rational identities can merge split metric shells."""

        direct = self._inputs.crystal.direct_basis_A
        a1 = direct[:, 0]
        a2 = direct[:, 1]
        norm1 = float(np.linalg.norm(a1))
        norm2 = float(np.linalg.norm(a2))
        cosine = float(a1 @ a2) / (norm1 * norm2)
        tolerance = 4096.0 * np.finfo(np.float64).eps
        if not np.isclose(norm1, norm2, rtol=tolerance, atol=0.0) or not np.isclose(
            abs(cosine),
            0.5,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                "rational layer-L marker identity currently requires a hexagonal surface metric"
            )

    @property
    def inputs(self) -> ConfiguredGeometryInputs:
        return self._inputs

    @property
    def instrument(self) -> CompiledInstrument:
        return self._inputs.instrument

    @property
    def reference_wavelength_A(self) -> float:
        return float(self._inputs.samples.wavelength_A[0])

    @property
    def reciprocal_basis_revision(self) -> str:
        return canonical_revision_sha256(
            ("definition_id", "commensurate_layer_coordinate_basis.v1"),
            ("phase_id", self._inputs.config.material.phase_id),
            ("cif_sha256", self._inputs.config.cif_sha256),
            ("reciprocal_basis_Ainv", self._inputs.reciprocal.basis_Ainv),
        )

    @property
    def geometry_context_revision(self) -> str:
        instrument = self._inputs.instrument
        return canonical_revision_sha256(
            ("definition_id", "exact_tag_geometry_context.v1"),
            ("source_revision", self._inputs.samples.source_revision),
            ("material_revision", self._inputs.material.material_revision),
            ("reciprocal_basis_revision", self.reciprocal_basis_revision),
            (
                "rod_hkm",
                np.asarray(
                    [(rod.h, rod.k, rod.family_m) for rod in self._inputs.rods],
                    dtype=np.int64,
                ),
            ),
            ("sample_geometry_revision", instrument.sample_geometry_revision),
            ("sample_from_crystal_rotation", instrument.sample_from_crystal.rotation),
            ("sample_from_crystal_translation_m", instrument.sample_from_crystal.translation_m),
            ("lab_from_detector_rotation", instrument.lab_from_detector.rotation),
            ("lab_from_detector_translation_m", instrument.lab_from_detector.translation_m),
            ("detector_shape_rc", np.asarray(instrument.detector_shape_rc, dtype=np.int64)),
            ("detector_row_pitch_m", np.asarray(instrument.detector_row_pitch_m)),
            ("detector_column_pitch_m", np.asarray(instrument.detector_column_pitch_m)),
            (
                "detector_reference_coordinate_px",
                np.asarray(instrument.detector_reference_coordinate_px),
            ),
            ("film_thickness_A", np.asarray(instrument.film_thickness_A)),
        )

    def predict_integer_l_tags(
        self,
        keys: tuple[IntegerLMarkerKey, ...],
        *,
        instrument: CompiledInstrument | None = None,
    ) -> IntegerLMarkerPrediction:
        """Predict frozen tags through direct Ewald geometry and native projection."""

        frozen_keys = _frozen_nonzero_keys(keys)
        active_instrument = self._inputs.instrument if instrument is None else instrument
        if not isinstance(active_instrument, CompiledInstrument):
            raise TypeError("instrument must be CompiledInstrument")
        incident = build_incident_states(
            self._inputs.samples,
            self._inputs.material,
            active_instrument,
        )
        if not bool(incident.states.valid[0]):
            raise GeometryPredictionError(
                f"nominal incident state became {incident.states.status[0].value}"
            )

        rods = {(rod.h, rod.k): rod for rod in self._inputs.rods}
        basis = self._inputs.reciprocal.basis_Ainv
        crystal_to_sample = active_instrument.sample_from_crystal.rotation
        ki_sample_Ainv = incident.states.k_film_phase_sample_Ainv[0]
        size = len(frozen_keys)
        coordinates = np.zeros((size, 2), dtype=np.float64)
        residual = np.zeros(size, dtype=np.float64)
        status = np.full(size, "ROOT_MISSING", dtype="U32")
        root_problems: dict[tuple[tuple[int, int], int], list[tuple[int, IntegerLMarkerKey]]] = {}
        for marker_index, key in enumerate(frozen_keys):
            rod = rods.get(key.representative_rod_hk)
            if rod is None or rod.family_m != key.family_m:
                raise ValueError(
                    f"marker rod {key.representative_rod_hk} does not belong to m={key.family_m}"
                )
            root_problems.setdefault((key.representative_rod_hk, key.integer_L), []).append(
                (marker_index, key)
            )

        batches: dict[tuple[tuple[int, int], int], list[tuple[int, float]]] = {}
        for (rod_hk, integer_l), markers in root_problems.items():
            roots = solve_integer_l_ewald_roots(
                rod=rods[rod_hk],
                integer_l=integer_l,
                reciprocal_basis_Ainv=basis,
                crystal_to_sample=crystal_to_sample,
                ki_sample_Ainv=ki_sample_Ainv,
            )
            for marker_index, key in markers:
                if roots is None:
                    continue
                if roots.branch != key.branch:
                    status[marker_index] = "BRANCH_CHANGED"
                    continue
                if key.root_sign not in roots.root_sign:
                    status[marker_index] = (
                        "ROOT_TANGENT" if roots.root_sign == (0,) else "ROOT_MISSING"
                    )
                    continue
                root_index = roots.root_sign.index(key.root_sign)
                batches.setdefault((rod_hk, key.branch), []).append(
                    (marker_index, roots.beta_rad[root_index])
                )

        for (rod_hk, branch), candidates in batches.items():
            rod = rods[rod_hk]
            indices = np.asarray([item[0] for item in candidates], dtype=np.int64)
            beta = np.asarray([item[1] for item in candidates], dtype=np.float64)
            integer_l = np.asarray(
                [frozen_keys[int(index)].integer_L for index in indices],
                dtype=np.float64,
            )
            mapped_coordinates, mapped_status, ewald_residual, _ = (
                _map_fixed_layer_l_roots_to_detector(
                    rod=rod,
                    branch=branch,
                    beta_rad=beta,
                    layer_l=integer_l,
                    reciprocal_basis_Ainv=basis,
                    crystal_to_sample=crystal_to_sample,
                    ki_sample_Ainv=ki_sample_Ainv,
                    incident=incident,
                    material=self._inputs.material,
                    instrument=active_instrument,
                )
            )
            coordinates[indices] = mapped_coordinates
            residual[indices] = ewald_residual
            status[indices] = mapped_status
        return IntegerLMarkerPrediction(
            keys=frozen_keys,
            coordinates_px=coordinates,
            detector_status=status,
            ewald_residual_Ainv=residual,
        )

    def predict_layer_l_tags(
        self,
        definitions: tuple[LayerLMarkerDefinition, ...],
        *,
        instrument: CompiledInstrument | None = None,
    ) -> LayerLMarkerPrediction:
        """Predict exact rational-layer tags without structure or mosaic evaluation."""

        self._require_hexagonal_layer_l_metric()
        frozen = tuple(definitions)
        _validate_layer_l_definitions(frozen, "prediction input")
        basis_revision = self.reciprocal_basis_revision
        if any(definition.key.reciprocal_basis_revision != basis_revision for definition in frozen):
            raise ValueError("layer-L marker reciprocal-basis revision does not match the model")
        active_instrument = self._inputs.instrument if instrument is None else instrument
        if not isinstance(active_instrument, CompiledInstrument):
            raise TypeError("instrument must be CompiledInstrument")
        incident = build_incident_states(
            self._inputs.samples,
            self._inputs.material,
            active_instrument,
        )
        if not bool(incident.states.valid[0]):
            raise GeometryPredictionError(
                f"nominal incident state became {incident.states.status[0].value}"
            )

        rods = {(rod.h, rod.k): rod for rod in self._inputs.rods}
        for definition in frozen:
            if any(
                rod_hk not in rods or rods[rod_hk].family_m != definition.key.family_m
                for rod_hk in definition.contributing_rod_hk
            ):
                raise ValueError("a layer-L contributing rod is not present in the model family")

        size = len(frozen)
        coordinates = np.zeros((size, 2), dtype=np.float64)
        residual = np.zeros(size, dtype=np.float64)
        status = np.full(size, "ROOT_MISSING", dtype="U32")

        integer_indices = tuple(
            index
            for index, definition in enumerate(frozen)
            if definition.key.layer_order.is_integer
        )
        if integer_indices:
            integer_keys = tuple(
                IntegerLMarkerKey(
                    family_m=frozen[index].key.family_m,
                    integer_L=frozen[index].key.layer_order.numerator,
                    branch=frozen[index].key.branch,
                    root_sign=frozen[index].key.root_sign,
                    representative_rod_hk=frozen[index].representative_rod_hk,
                )
                for index in integer_indices
            )
            integer_prediction = self.predict_integer_l_tags(
                integer_keys,
                instrument=active_instrument,
            )
            target = np.asarray(integer_indices, dtype=np.int64)
            coordinates[target] = integer_prediction.coordinates_px
            residual[target] = integer_prediction.ewald_residual_Ainv
            status[target] = integer_prediction.detector_status

        basis = self._inputs.reciprocal.basis_Ainv
        crystal_to_sample = active_instrument.sample_from_crystal.rotation
        ki_sample = incident.states.k_film_phase_sample_Ainv[0]
        for index, definition in enumerate(frozen):
            key = definition.key
            if not key.layer_order.is_integer:
                coordinate, marker_status, marker_residual, _ = _predict_single_layer_l_rod(
                    key=key,
                    rod=rods[definition.representative_rod_hk],
                    reciprocal_basis_Ainv=basis,
                    crystal_to_sample=crystal_to_sample,
                    ki_sample_Ainv=ki_sample,
                    incident=incident,
                    material=self._inputs.material,
                    instrument=active_instrument,
                )
                coordinates[index] = coordinate
                residual[index] = marker_residual
                status[index] = marker_status
            if len(definition.contributing_rod_hk) == 1:
                continue
            contributing = tuple(
                _predict_single_layer_l_rod(
                    key=key,
                    rod=rods[rod_hk],
                    reciprocal_basis_Ainv=basis,
                    crystal_to_sample=crystal_to_sample,
                    ki_sample_Ainv=ki_sample,
                    incident=incident,
                    material=self._inputs.material,
                    instrument=active_instrument,
                )
                for rod_hk in definition.contributing_rod_hk
            )
            reference_coordinate, reference_status, _, reference_q = contributing[0]
            coordinate_scale = max(float(max(active_instrument.detector_shape_rc)), 1.0)
            coordinate_tolerance = 32768.0 * np.finfo(np.float64).eps * coordinate_scale
            q_tolerance = (
                32768.0
                * np.finfo(np.float64).eps
                * max(
                    float(np.linalg.norm(ki_sample)),
                    1.0,
                )
            )
            if any(
                marker_status != reference_status
                or float(np.linalg.norm(marker_coordinate - reference_coordinate))
                > coordinate_tolerance
                or float(np.linalg.norm(marker_q - reference_q)) > q_tolerance
                for marker_coordinate, marker_status, _, marker_q in contributing[1:]
            ):
                status[index] = "LOCUS_SPLIT"
                continue
            residual[index] = max(item[2] for item in contributing)
        return LayerLMarkerPrediction(
            definitions=frozen,
            coordinates_px=coordinates,
            detector_status=status,
            ewald_residual_Ainv=residual,
        )

    def enumerate_layer_l_tags(
        self,
        layer_orders: tuple[CommensurateLayerOrder, ...],
        *,
        instrument: CompiledInstrument | None = None,
    ) -> LayerLMarkerPrediction:
        """Enumerate panel-visible exact-layer loci before any intensity selection."""

        self._require_hexagonal_layer_l_metric()
        orders = tuple(layer_orders)
        if (
            not orders
            or any(not isinstance(order, CommensurateLayerOrder) for order in orders)
            or len(set(orders)) != len(orders)
        ):
            raise ValueError("layer_orders must contain unique CommensurateLayerOrder values")
        active_instrument = self._inputs.instrument if instrument is None else instrument
        if not isinstance(active_instrument, CompiledInstrument):
            raise TypeError("instrument must be CompiledInstrument")
        incident = build_incident_states(
            self._inputs.samples,
            self._inputs.material,
            active_instrument,
        )
        if not bool(incident.states.valid[0]):
            raise GeometryPredictionError(
                f"nominal incident state became {incident.states.status[0].value}"
            )
        basis = self._inputs.reciprocal.basis_Ainv
        crystal_to_sample = active_instrument.sample_from_crystal.rotation
        ki_sample = incident.states.k_film_phase_sample_Ainv[0]
        groups: dict[
            LayerLMarkerKey,
            tuple[list[tuple[int, int]], FloatArray, FloatArray, float],
        ] = {}
        coordinate_scale = max(float(max(active_instrument.detector_shape_rc)), 1.0)
        coordinate_tolerance = 32768.0 * np.finfo(np.float64).eps * coordinate_scale
        q_tolerance = (
            32768.0
            * np.finfo(np.float64).eps
            * max(
                float(np.linalg.norm(ki_sample)),
                1.0,
            )
        )
        basis_revision = self.reciprocal_basis_revision
        for rod in self._inputs.rods:
            if rod.family_m == 0:
                continue
            for layer_order in sorted(orders):
                roots = solve_layer_l_ewald_roots(
                    rod=rod,
                    layer_order=layer_order,
                    reciprocal_basis_Ainv=basis,
                    crystal_to_sample=crystal_to_sample,
                    ki_sample_Ainv=ki_sample,
                )
                if roots is None or roots.root_sign == (0,):
                    continue
                for root_sign in roots.root_sign:
                    key = LayerLMarkerKey(
                        family_m=rod.family_m,
                        layer_order=layer_order,
                        branch=roots.branch,
                        root_sign=root_sign,
                        reciprocal_basis_revision=basis_revision,
                    )
                    coordinate, status, residual, q_sample = _predict_single_layer_l_rod(
                        key=key,
                        rod=rod,
                        reciprocal_basis_Ainv=basis,
                        crystal_to_sample=crystal_to_sample,
                        ki_sample_Ainv=ki_sample,
                        incident=incident,
                        material=self._inputs.material,
                        instrument=active_instrument,
                    )
                    if status != ValidityCode.VALID.value:
                        continue
                    existing = groups.get(key)
                    if existing is None:
                        groups[key] = ([(rod.h, rod.k)], coordinate, q_sample, residual)
                        continue
                    rods, reference_coordinate, reference_q, reference_residual = existing
                    if (
                        float(np.linalg.norm(coordinate - reference_coordinate))
                        > coordinate_tolerance
                        or float(np.linalg.norm(q_sample - reference_q)) > q_tolerance
                    ):
                        raise GeometryPredictionError(
                            "one rod-free layer-L key maps to multiple detector loci"
                        )
                    rods.append((rod.h, rod.k))
                    groups[key] = (
                        rods,
                        reference_coordinate,
                        reference_q,
                        max(reference_residual, residual),
                    )
        if not groups:
            raise GeometryPredictionError("no exact layer-L landmarks reach the active panel")
        ordered = tuple((key, groups[key]) for key in sorted(groups))
        return LayerLMarkerPrediction(
            definitions=tuple(
                LayerLMarkerDefinition(key=key, contributing_rod_hk=tuple(value[0]))
                for key, value in ordered
            ),
            coordinates_px=np.asarray([value[1] for _, value in ordered]),
            detector_status=np.full(len(ordered), ValidityCode.VALID.value, dtype="U32"),
            ewald_residual_Ainv=np.asarray([value[3] for _, value in ordered]),
        )

    def predict_m0_minimum_tilt_exact_l_landmarks(
        self,
        integer_L: tuple[int, ...],
        *,
        instrument: CompiledInstrument | None = None,
    ) -> M0IntegerLPrediction:
        """Map the unconstrained minimum-tilt point on each exact nonzero ``00L`` curve."""

        frozen_l = _frozen_m0_integer_l(integer_L)
        active_instrument = self._inputs.instrument if instrument is None else instrument
        if not isinstance(active_instrument, CompiledInstrument):
            raise TypeError("instrument must be CompiledInstrument")
        incident = build_incident_states(
            self._inputs.samples,
            self._inputs.material,
            active_instrument,
        )
        if not bool(incident.states.valid[0]):
            raise GeometryPredictionError(
                f"nominal incident state became {incident.states.status[0].value}"
            )
        specular_rods = tuple(rod for rod in self._inputs.rods if rod.family_m == 0)
        if len(specular_rods) != 1:
            raise ValueError("minimum-tilt exact-L landmarks require one physical m=0 rod")
        rod = specular_rods[0]
        basis = self._inputs.reciprocal.basis_Ainv
        crystal_to_sample = active_instrument.sample_from_crystal.rotation
        mean_axis_crystal, tilt_axis_crystal = mosaic_axes(basis)
        reference_axis_crystal = np.cross(tilt_axis_crystal, mean_axis_crystal)
        mean_axis_sample = crystal_to_sample @ mean_axis_crystal
        ki_sample = incident.states.k_film_phase_sample_Ainv[0]
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
        q_sample = np.zeros((size, 3), dtype=np.float64)
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
            q_sample[index] = u_Ainv * direction_sample
            feasible_indices.append(index)

        if feasible_indices:
            indices = np.asarray(feasible_indices, dtype=np.int64)
            integer_l = np.asarray([frozen_l[int(index)] for index in indices], dtype=np.float64)
            selected_q = q_sample[indices]
            kf_sample = selected_q + ki_sample[None, :]
            selected_residual = np.abs(np.linalg.norm(kf_sample, axis=1) - k_norm)
            geometry = EwaldLatentGeometry(
                rod=rod,
                branch=0,
                alpha_rad=alpha[indices],
                beta_rad=beta[indices],
                u_Ainv=integer_l * b3_norm,
                L=integer_l,
                q_sample_Ainv=selected_q,
                kf_sample_Ainv=kf_sample,
                ewald_residual_Ainv=selected_residual,
                status=np.full(indices.size, RootStatus.REGULAR.value, dtype="U32"),
            )
            mapped = map_ewald_geometry_to_detector(
                geometry,
                incident=incident,
                material=self._inputs.material,
                instrument=active_instrument,
            )
            coordinates[indices, 0] = mapped.column_px
            coordinates[indices, 1] = mapped.row_px
            residual[indices] = selected_residual
            status[indices] = mapped.detector_status
        return M0IntegerLPrediction(
            integer_L=frozen_l,
            coordinates_px=coordinates,
            alpha_rad=alpha,
            beta_rad=beta,
            detector_status=status,
            ewald_residual_Ainv=residual,
            reference_wavelength_A=self.reference_wavelength_A,
        )


def _direct_layer_l_root_coordinates(
    *,
    rod: Rod,
    layer_order: CommensurateLayerOrder,
    reciprocal_basis_Ainv: FloatArray,
    crystal_to_sample: FloatArray,
    ki_sample_Ainv: FloatArray,
    incident: IncidentTransportResult,
    material: MaterialOptics,
    instrument: CompiledInstrument,
) -> tuple[dict[tuple[int, int], FloatArray], bool]:
    """Directly bracket fixed-L elastic roots without the prediction solver."""

    tau = 2.0 * np.pi
    layer_l = layer_order.as_float()
    b3_norm_Ainv = float(np.linalg.norm(reciprocal_basis_Ainv[:, 2]))
    u_Ainv = layer_l * b3_norm_Ainv
    mean_axis, _ = mosaic_axes(reciprocal_basis_Ainv)
    mean_axis_sample = crystal_to_sample @ mean_axis

    def q_at_beta(beta_rad: float) -> FloatArray:
        return map_tied_rotation_latent(
            rod=rod,
            reciprocal_basis_Ainv=reciprocal_basis_Ainv,
            crystal_to_sample=crystal_to_sample,
            alpha_rad=0.0,
            beta_rad=beta_rad % tau,
            u_Ainv=u_Ainv,
        )

    q_zero = q_at_beta(0.0)
    incident_norm = float(np.linalg.norm(ki_sample_Ainv))
    q_norm = float(np.linalg.norm(q_zero))
    q_perpendicular_norm = float(np.linalg.norm(np.cross(mean_axis_sample, q_zero)))
    equation_scale = max(q_norm**2, 2.0 * incident_norm * q_norm, incident_norm**2, 1.0)
    derivative_scale = max(2.0 * incident_norm * q_perpendicular_norm, 1.0)
    equation_tolerance = 4096.0 * np.finfo(np.float64).eps * equation_scale
    derivative_tolerance = 4096.0 * np.finfo(np.float64).eps * derivative_scale

    def elastic_equation(beta_rad: float) -> float:
        q_sample = q_at_beta(beta_rad)
        return math.fsum(
            float(q_component * (q_component + 2.0 * ki_component))
            for q_component, ki_component in zip(q_sample, ki_sample_Ainv, strict=True)
        )

    def elastic_derivative(beta_rad: float) -> float:
        q_sample = q_at_beta(beta_rad)
        derivative = np.cross(mean_axis_sample, q_sample)
        return 2.0 * math.fsum(
            float((ki_component + q_component) * derivative_component)
            for ki_component, q_component, derivative_component in zip(
                ki_sample_Ainv,
                q_sample,
                derivative,
                strict=True,
            )
        )

    edges = np.linspace(0.0, tau, 9)
    derivative_at_edges = tuple(elastic_derivative(float(edge)) for edge in edges)
    if max(abs(value) for value in derivative_at_edges) <= derivative_tolerance:
        equation_at_edges = tuple(elastic_equation(float(edge)) for edge in edges)
        separated = (
            min(equation_at_edges) > equation_tolerance
            or max(equation_at_edges) < -equation_tolerance
        )
        return ({}, False) if separated else ({}, True)
    extrema: list[float] = []
    for left, right, left_value, right_value in zip(
        edges[:-1],
        edges[1:],
        derivative_at_edges[:-1],
        derivative_at_edges[1:],
        strict=True,
    ):
        if abs(left_value) <= derivative_tolerance:
            extrema.append(float(left % tau))
        elif left_value * right_value < 0.0:
            extrema.append(
                float(
                    brentq(
                        elastic_derivative,
                        float(left),
                        float(right),
                        xtol=64.0 * np.finfo(np.float64).eps,
                        rtol=8.0 * np.finfo(np.float64).eps,
                    )
                    % tau
                )
            )
    unique_extrema: list[float] = []
    angular_tolerance = 1024.0 * np.finfo(np.float64).eps * tau
    for beta in sorted(extrema):
        if not any(
            abs((beta - existing + np.pi) % tau - np.pi) <= angular_tolerance
            for existing in unique_extrema
        ):
            unique_extrema.append(beta)
    if len(unique_extrema) != 2:
        return {}, True
    first_extremum, second_extremum = unique_extrema
    first_value = elastic_equation(first_extremum)
    second_value = elastic_equation(second_extremum)
    if abs(first_value) <= equation_tolerance or abs(second_value) <= equation_tolerance:
        return {}, True
    if first_value * second_value > 0.0:
        return {}, False

    roots = (
        (
            float(
                brentq(
                    elastic_equation,
                    first_extremum,
                    second_extremum,
                    xtol=64.0 * np.finfo(np.float64).eps,
                    rtol=8.0 * np.finfo(np.float64).eps,
                )
                % tau
            ),
            -1 if first_value < second_value else 1,
        ),
        (
            float(
                brentq(
                    elastic_equation,
                    second_extremum,
                    first_extremum + tau,
                    xtol=64.0 * np.finfo(np.float64).eps,
                    rtol=8.0 * np.finfo(np.float64).eps,
                )
                % tau
            ),
            -1 if second_value < first_value else 1,
        ),
    )

    coordinates: dict[tuple[int, int], FloatArray] = {}
    branch_tolerance = (
        4096.0 * np.finfo(np.float64).eps * max(incident_norm, q_norm, abs(u_Ainv), 1.0)
    )
    residual_tolerance = 512.0 * np.finfo(np.float64).eps * max(incident_norm, 1.0)
    for beta, root_sign in roots:
        q_sample = q_at_beta(beta)
        kf_sample = q_sample + ki_sample_Ainv
        branch_derivative = float(kf_sample @ mean_axis_sample)
        if abs(branch_derivative) <= branch_tolerance:
            return {}, True
        branch = 1 if branch_derivative < 0.0 else 2
        ewald_residual = abs(float(np.linalg.norm(kf_sample)) - incident_norm)
        if ewald_residual > residual_tolerance:
            return {}, True
        geometry = EwaldLatentGeometry(
            rod=rod,
            branch=branch,
            alpha_rad=np.asarray(0.0),
            beta_rad=np.asarray(beta),
            u_Ainv=np.asarray(u_Ainv),
            L=np.asarray(layer_l),
            q_sample_Ainv=q_sample,
            kf_sample_Ainv=kf_sample,
            ewald_residual_Ainv=np.asarray(ewald_residual),
            status=np.asarray(RootStatus.REGULAR.value),
        )
        mapped = map_ewald_geometry_to_detector(
            geometry,
            incident=incident,
            material=material,
            instrument=instrument,
        )
        if str(mapped.detector_status) != ValidityCode.VALID.value:
            continue
        coordinate = np.asarray(
            (float(mapped.column_px), float(mapped.row_px)),
            dtype=np.float64,
        )
        coordinate.setflags(write=False)
        coordinates[(branch, root_sign)] = coordinate
    return coordinates, False


def audit_exact_tag_geometry_roots(
    model: ExactTagGeometryModel,
    expected_keys: tuple[IntegerLMarkerKey, ...],
    *,
    instrument: CompiledInstrument | None = None,
) -> IntegerLSelectionAudit:
    """Compare predicted roots with direct fixed-L elastic-root enumeration."""

    if not isinstance(model, ExactTagGeometryModel):
        raise TypeError("model must be ExactTagGeometryModel")
    expected = _frozen_nonzero_keys(expected_keys)
    active_instrument = model.instrument if instrument is None else instrument
    if not isinstance(active_instrument, CompiledInstrument):
        raise TypeError("instrument must be CompiledInstrument")
    incident = build_incident_states(model.inputs.samples, model.inputs.material, active_instrument)
    if not bool(incident.states.valid[0]):
        return IntegerLSelectionAudit(
            classification="MISSING",
            missing_keys=expected,
            unexpected_keys=(),
            expected_count=len(expected),
            enumerated_count=0,
        )
    rods = {(rod.h, rod.k): rod for rod in model.inputs.rods}
    basis = model.inputs.reciprocal.basis_Ainv
    crystal_to_sample = active_instrument.sample_from_crystal.rotation
    ki_sample = incident.states.k_film_phase_sample_Ainv[0]
    try:
        predicted = model.predict_integer_l_tags(expected, instrument=active_instrument)
    except GeometryPredictionError:
        return IntegerLSelectionAudit(
            classification="MISSING",
            missing_keys=expected,
            unexpected_keys=(),
            expected_count=len(expected),
            enumerated_count=0,
        )
    direct_by_problem: dict[
        tuple[tuple[int, int], int], tuple[dict[tuple[int, int], FloatArray], bool]
    ] = {}
    for key in expected:
        rod = rods.get(key.representative_rod_hk)
        if rod is None or rod.family_m != key.family_m:
            continue
        problem = (key.representative_rod_hk, key.integer_L)
        if problem in direct_by_problem:
            continue
        direct_by_problem[problem] = _direct_layer_l_root_coordinates(
            rod=rod,
            layer_order=CommensurateLayerOrder(key.integer_L),
            reciprocal_basis_Ainv=basis,
            crystal_to_sample=crystal_to_sample,
            ki_sample_Ainv=ki_sample,
            incident=incident,
            material=model.inputs.material,
            instrument=active_instrument,
        )
    if any(ambiguous for _, ambiguous in direct_by_problem.values()):
        return IntegerLSelectionAudit(
            classification="AMBIGUOUS",
            missing_keys=(),
            unexpected_keys=(),
            expected_count=len(expected),
            enumerated_count=sum(len(item) for item, _ in direct_by_problem.values()),
        )

    missing: list[IntegerLMarkerKey] = []
    mismatched: list[IntegerLMarkerKey] = []
    matched_count = 0
    coordinate_tolerance_px = 1.0e-5
    for index, key in enumerate(expected):
        direct = direct_by_problem.get((key.representative_rod_hk, key.integer_L))
        oracle_coordinate = None if direct is None else direct[0].get((key.branch, key.root_sign))
        if (
            oracle_coordinate is None
            or str(predicted.detector_status[index]) != ValidityCode.VALID.value
        ):
            missing.append(key)
            continue
        coordinate_error = float(
            np.linalg.norm(predicted.coordinates_px[index] - oracle_coordinate)
        )
        if coordinate_error > coordinate_tolerance_px:
            missing.append(key)
            mismatched.append(key)
            continue
        matched_count += 1
    missing_keys = tuple(sorted(missing))
    unexpected_keys = tuple(sorted(mismatched))
    if unexpected_keys:
        classification = "CHANGED"
    elif missing_keys:
        classification = "MISSING"
    else:
        classification = "SAME"
    return IntegerLSelectionAudit(
        classification=classification,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        expected_count=len(expected),
        enumerated_count=matched_count + len(unexpected_keys),
    )


def audit_exact_layer_l_geometry_roots(
    model: ExactTagGeometryModel,
    expected_definitions: tuple[LayerLMarkerDefinition, ...],
    *,
    instrument: CompiledInstrument | None = None,
) -> LayerLSelectionAudit:
    """Audit rational-layer loci with an independent bracketed elastic-root solve."""

    if not isinstance(model, ExactTagGeometryModel):
        raise TypeError("model must be ExactTagGeometryModel")
    expected = tuple(expected_definitions)
    _validate_layer_l_definitions(expected, "root audit")
    active_instrument = model.instrument if instrument is None else instrument
    if not isinstance(active_instrument, CompiledInstrument):
        raise TypeError("instrument must be CompiledInstrument")
    keys = tuple(definition.key for definition in expected)
    incident = build_incident_states(model.inputs.samples, model.inputs.material, active_instrument)
    if not bool(incident.states.valid[0]):
        return LayerLSelectionAudit(
            classification="MISSING",
            missing_keys=keys,
            unexpected_keys=(),
            expected_count=len(keys),
            enumerated_count=0,
        )
    try:
        predicted = model.predict_layer_l_tags(expected, instrument=active_instrument)
    except GeometryPredictionError:
        return LayerLSelectionAudit(
            classification="MISSING",
            missing_keys=keys,
            unexpected_keys=(),
            expected_count=len(keys),
            enumerated_count=0,
        )

    rods = {(rod.h, rod.k): rod for rod in model.inputs.rods}
    basis = model.inputs.reciprocal.basis_Ainv
    crystal_to_sample = active_instrument.sample_from_crystal.rotation
    ki_sample = incident.states.k_film_phase_sample_Ainv[0]
    direct_by_problem: dict[
        tuple[tuple[int, int], CommensurateLayerOrder],
        tuple[dict[tuple[int, int], FloatArray], bool],
    ] = {}
    for definition in expected:
        for rod_hk in definition.contributing_rod_hk:
            rod = rods.get(rod_hk)
            if rod is None or rod.family_m != definition.key.family_m:
                continue
            problem = (rod_hk, definition.key.layer_order)
            if problem in direct_by_problem:
                continue
            direct_by_problem[problem] = _direct_layer_l_root_coordinates(
                rod=rod,
                layer_order=definition.key.layer_order,
                reciprocal_basis_Ainv=basis,
                crystal_to_sample=crystal_to_sample,
                ki_sample_Ainv=ki_sample,
                incident=incident,
                material=model.inputs.material,
                instrument=active_instrument,
            )
    if any(ambiguous for _, ambiguous in direct_by_problem.values()):
        return LayerLSelectionAudit(
            classification="AMBIGUOUS",
            missing_keys=(),
            unexpected_keys=(),
            expected_count=len(keys),
            enumerated_count=sum(
                all(
                    (rod_hk, definition.key.layer_order) in direct_by_problem
                    for rod_hk in definition.contributing_rod_hk
                )
                for definition in expected
            ),
        )

    missing: list[LayerLMarkerKey] = []
    mismatched: list[LayerLMarkerKey] = []
    matched_count = 0
    coordinate_tolerance_px = 1.0e-5
    for index, definition in enumerate(expected):
        key = definition.key
        oracle_coordinates = []
        for rod_hk in definition.contributing_rod_hk:
            direct = direct_by_problem.get((rod_hk, key.layer_order))
            coordinate = None if direct is None else direct[0].get((key.branch, key.root_sign))
            if coordinate is None:
                oracle_coordinates = []
                break
            oracle_coordinates.append(coordinate)
        if not oracle_coordinates or not predicted.active_panel[index]:
            missing.append(key)
            continue
        if any(
            float(np.linalg.norm(predicted.coordinates_px[index] - coordinate))
            > coordinate_tolerance_px
            for coordinate in oracle_coordinates
        ):
            missing.append(key)
            mismatched.append(key)
            continue
        matched_count += 1
    missing_keys = tuple(sorted(missing))
    unexpected_keys = tuple(sorted(mismatched))
    classification = "CHANGED" if unexpected_keys else "MISSING" if missing_keys else "SAME"
    return LayerLSelectionAudit(
        classification=classification,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        expected_count=len(keys),
        enumerated_count=matched_count + len(unexpected_keys),
    )


class _ExactTagGeometry:
    """Internal one-state exact-tag geometry owned by the continuous field model."""

    __slots__ = (
        "_exact_model",
        "_inputs",
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
        nominal_samples = sample_configured_nominal_geometry_source(inputs.config.source)
        nominal_material = material_optics(inputs.crystal, nominal_samples.wavelength_A)
        exact_model = ExactTagGeometryModel(
            ConfiguredGeometryInputs(
                config=inputs.config,
                samples=nominal_samples,
                instrument=inputs.instrument,
                crystal=inputs.crystal,
                material=nominal_material,
                reciprocal=inputs.reciprocal,
                rods=inputs.bragg_space.config.rods,
            )
        )
        object.__setattr__(self, "_exact_model", exact_model)
        object.__setattr__(self, "_inputs", inputs)
        object.__setattr__(self, "_nominal_samples", nominal_samples)
        object.__setattr__(self, "_sample_correction_pivot_lab_m", sample_correction_pivot_lab_m)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("exact tag geometry is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("exact tag geometry is immutable")

    @property
    def reference_wavelength_A(self) -> float:
        return float(self._nominal_samples.wavelength_A[0])

    def predict(
        self,
        keys: tuple[IntegerLMarkerKey, ...],
        corrections: GeometryCorrections,
    ) -> IntegerLMarkerPrediction:
        instrument = _corrected_instrument(
            self._inputs.instrument,
            corrections,
            self._sample_correction_pivot_lab_m,
        )
        return self._exact_model.predict_integer_l_tags(keys, instrument=instrument)

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

        instrument = _corrected_instrument(
            self._inputs.instrument,
            corrections,
            self._sample_correction_pivot_lab_m,
        )
        return self._exact_model.predict_m0_minimum_tilt_exact_l_landmarks(
            integer_L,
            instrument=instrument,
        )

    def predict_tagged_landmarks(
        self,
        nonzero_keys: tuple[IntegerLMarkerKey, ...],
        m0_integer_L: tuple[int, ...],
        corrections: GeometryCorrections,
    ) -> tuple[IntegerLMarkerPrediction, M0IntegerLPrediction]:
        """Predict both exact tag groups for one correction state."""

        frozen_keys = _frozen_nonzero_keys(nonzero_keys)
        frozen_l = _frozen_m0_integer_l(m0_integer_L)
        instrument = _corrected_instrument(
            self._inputs.instrument,
            corrections,
            self._sample_correction_pivot_lab_m,
        )
        return (
            self._exact_model.predict_integer_l_tags(
                frozen_keys,
                instrument=instrument,
            ),
            self._exact_model.predict_m0_minimum_tilt_exact_l_landmarks(
                frozen_l,
                instrument=instrument,
            ),
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
    keys: tuple[IntegerLMarkerKey | LayerLMarkerKey, ...],
) -> tuple[tuple[int, int], ...]:
    groups: dict[
        tuple[str, int, int, CommensurateLayerOrder],
        dict[int, int],
    ] = {}
    for index, key in enumerate(keys):
        layer_order = (
            key.layer_order
            if isinstance(key, LayerLMarkerKey)
            else CommensurateLayerOrder(key.integer_L)
        )
        group_id = (
            ("layer", key.family_m, 0, layer_order)
            if isinstance(key, LayerLMarkerKey)
            else (
                "rod",
                key.representative_rod_hk[0],
                key.representative_rod_hk[1],
                layer_order,
            )
        )
        groups.setdefault(group_id, {})[key.root_sign] = index
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
    observations: IntegerLMarkerObservations | LayerLMarkerObservations,
    prediction: IntegerLMarkerPrediction | LayerLMarkerPrediction,
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


def _evaluate_nonzero_geometry_objective_residual(
    observations: IntegerLMarkerObservations | LayerLMarkerObservations,
    prediction: IntegerLMarkerPrediction | LayerLMarkerPrediction,
    *,
    m0_observations: M0IntegerLObservations | None = None,
    m0_prediction: M0IntegerLPrediction | None = None,
) -> FloatArray:
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


def evaluate_tagged_geometry_objective_residual(
    observations: IntegerLMarkerObservations,
    prediction: IntegerLMarkerPrediction,
    *,
    m0_observations: M0IntegerLObservations | None = None,
    m0_prediction: M0IntegerLPrediction | None = None,
) -> FloatArray:
    """Evaluate the legacy integer-L site-plus-line objective without optimization."""

    if not isinstance(observations, IntegerLMarkerObservations):
        raise TypeError("observations must be IntegerLMarkerObservations")
    if not isinstance(prediction, IntegerLMarkerPrediction):
        raise TypeError("prediction must be IntegerLMarkerPrediction")
    return _evaluate_nonzero_geometry_objective_residual(
        observations,
        prediction,
        m0_observations=m0_observations,
        m0_prediction=m0_prediction,
    )


def evaluate_layer_l_geometry_objective_residual(
    observations: LayerLMarkerObservations,
    prediction: LayerLMarkerPrediction,
) -> FloatArray:
    """Evaluate the same geometry objective at exact rational-layer landmarks."""

    if not isinstance(observations, LayerLMarkerObservations):
        raise TypeError("observations must be LayerLMarkerObservations")
    if not isinstance(prediction, LayerLMarkerPrediction):
        raise TypeError("prediction must be LayerLMarkerPrediction")
    if prediction.definitions != observations.definitions:
        raise ValueError("prediction definitions must exactly match observation definitions")
    return _evaluate_nonzero_geometry_objective_residual(observations, prediction)


def _finite_difference_jacobian(
    function: Callable[[FloatArray], FloatArray],
    values: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    step_size: ArrayLike = 1.0e-5,
) -> FloatArray:
    baseline = np.asarray(function(values), dtype=np.float64)
    jacobian = np.empty((baseline.size, values.size), dtype=np.float64)
    supplied_step = np.asarray(step_size, dtype=np.float64)
    if supplied_step.ndim == 0:
        steps = np.full(values.size, float(supplied_step), dtype=np.float64)
    else:
        steps = np.array(supplied_step, dtype=np.float64, copy=True)
    if steps.shape != values.shape or not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError(
            "finite-difference step_size must be positive with one value per parameter"
        )
    for parameter_index in range(values.size):
        step = min(
            float(steps[parameter_index]),
            0.25 * float(upper[parameter_index] - lower[parameter_index]),
        )
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


@dataclass(frozen=True, slots=True)
class LayerLSelectionAudit:
    classification: str
    missing_keys: tuple[LayerLMarkerKey, ...]
    unexpected_keys: tuple[LayerLMarkerKey, ...]
    expected_count: int
    enumerated_count: int

    def __post_init__(self) -> None:
        if self.classification not in {"SAME", "CHANGED", "MISSING", "AMBIGUOUS"}:
            raise ValueError("unsupported layer-L marker selection classification")
        missing = tuple(self.missing_keys)
        unexpected = tuple(self.unexpected_keys)
        if any(not isinstance(key, LayerLMarkerKey) for key in missing + unexpected):
            raise TypeError("audit differences must contain LayerLMarkerKey values")
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
