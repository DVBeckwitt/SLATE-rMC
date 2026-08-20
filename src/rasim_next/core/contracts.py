"""Immutable array contracts shared by the detector-native numerical core."""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Complex, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.validity import ValidityCode

CONTRACT_API_VERSION = 14
_ArraySpec = tuple[str, np.dtype[Any] | type[np.generic], tuple[int, ...], bool]


class LayerAmplitudeNormalization(StrEnum):
    ONE_REGISTRY_FREE_LAYER = "ONE_REGISTRY_FREE_LAYER"


class LayerPhaseSign(StrEnum):
    POSITIVE_Q_DOT_R = "POSITIVE_Q_DOT_R"


class EventIntensityNormalization(StrEnum):
    UNIT_CELL = "UNIT_CELL"
    FINITE_TOTAL = "FINITE_TOTAL"
    FINITE_PER_LAYER = "FINITE_PER_LAYER"


def _array(
    value: ArrayLike,
    dtype: np.dtype[Any] | type[np.generic],
    shape: tuple[int | None, ...],
    name: str,
    nonnegative: bool = False,
) -> NDArray[Any]:
    supplied = np.asarray(value)
    target = np.dtype(dtype)
    inspect_elements = not isinstance(value, np.ndarray) or supplied.dtype.kind == "O"
    supplied_object = np.asarray(value, dtype=object) if inspect_elements else None
    if np.issubdtype(target, np.integer) and not (
        supplied.dtype.kind in "iu"
        and (
            supplied_object is None
            or all(
                isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_))
                for item in supplied_object.flat
            )
        )
    ):
        raise ValueError(f"{name} must contain integers")
    if np.issubdtype(target, np.integer):
        limits = np.iinfo(target)
        if np.any(supplied < limits.min) or np.any(supplied > limits.max):
            raise ValueError(f"{name} values are outside the {target.name} range")
    if np.issubdtype(target, np.floating) and not (
        supplied.dtype.kind in "iuf"
        and (
            supplied_object is None
            or all(
                isinstance(item, Real) and not isinstance(item, (bool, np.bool_))
                for item in supplied_object.flat
            )
        )
    ):
        raise ValueError(f"{name} must contain real numbers")
    if np.issubdtype(target, np.complexfloating) and not (
        supplied.dtype.kind in "iufc"
        and (
            supplied_object is None
            or all(
                isinstance(item, Complex) and not isinstance(item, (bool, np.bool_))
                for item in supplied_object.flat
            )
        )
    ):
        raise ValueError(f"{name} must contain numeric values")
    if np.issubdtype(target, np.bool_) and not (
        supplied.dtype.kind == "b"
        and (
            supplied_object is None
            or all(isinstance(item, (bool, np.bool_)) for item in supplied_object.flat)
        )
    ):
        raise ValueError(f"{name} must contain booleans")
    array = np.array(supplied, dtype=target, copy=True, order="C")
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        expected = tuple("N" if item is None else item for item in shape)
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    array.setflags(write=False)
    return array


def _texts(value: tuple[str, ...], size: int, name: str) -> tuple[str, ...]:
    result = tuple(value)
    if len(result) != size or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain {size} nonempty strings")
    return result


def _versioned_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]*\.v[1-9][0-9]*", value) is None
    ):
        raise ValueError(f"{name} must be a nonempty versioned identifier ending in .vN")
    return value


def _sha256_revision(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 revision")
    return value


def canonical_revision_sha256(*fields: tuple[str, object]) -> str:
    """Hash named canonical values with one typed, length-prefixed encoding."""

    if not fields:
        raise ValueError("canonical revision requires at least one field")
    names = tuple(name for name, _ in fields)
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
        names
    ):
        raise ValueError("canonical revision field names must be unique nonempty strings")

    digest = hashlib.sha256()

    def update_bytes(value: bytes) -> None:
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)

    update_bytes(b"rasim_next.canonical_typed_sha256.v1")
    for name, value in sorted(fields, key=lambda item: item[0]):
        update_bytes(name.encode("utf-8"))
        if isinstance(value, str):
            update_bytes(b"text")
            update_bytes(value.encode("utf-8"))
            continue
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            integer = int(value)
            if integer < -(2**63) or integer > 2**64 - 1:
                raise ValueError(f"canonical integer field {name!r} is outside 64-bit range")
            value = np.asarray(integer, dtype=np.int64 if integer < 0 else np.uint64)
        if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
            update_bytes(b"text-sequence")
            update_bytes(struct.pack("<Q", len(value)))
            for item in value:
                update_bytes(item.encode("utf-8"))
            continue

        supplied = np.asarray(value)
        if supplied.dtype.kind not in "biufc" or supplied.dtype.hasobject:
            raise ValueError(f"canonical numeric field {name!r} has unsupported dtype")
        if supplied.dtype.kind in "fc" and not np.all(np.isfinite(supplied)):
            raise ValueError(f"canonical numeric field {name!r} must be finite")
        dtype = supplied.dtype.newbyteorder("<")
        array = np.ascontiguousarray(supplied, dtype=dtype)
        update_bytes(b"numeric-array")
        update_bytes(dtype.str.encode("ascii"))
        update_bytes(struct.pack("<Q", array.ndim))
        update_bytes(struct.pack(f"<{array.ndim}Q", *array.shape))
        update_bytes(array.tobytes(order="C"))
    return digest.hexdigest()


def incidence_scan_calibration_binding_revision(
    *,
    scan_calibration_revision: str,
    component_sample_geometry_revision: tuple[str, ...],
    component_incidence_axis_angle_rad: ArrayLike,
    effective_incidence_angle_rad: ArrayLike,
    detector_panel_revision: str,
    source_revision: str,
    source_state_count: int,
) -> str:
    """Hash the complete calibrated identity of one incidence-node series."""

    if not isinstance(scan_calibration_revision, str) or not scan_calibration_revision:
        raise ValueError("scan_calibration_revision must be nonempty")
    geometry_revisions = tuple(component_sample_geometry_revision)
    if not geometry_revisions or any(
        not isinstance(value, str) or not value for value in geometry_revisions
    ):
        raise ValueError("component sample-geometry revisions must be nonempty strings")

    def real_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
        supplied = np.asarray(value)
        if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
            raise ValueError(f"{name} must be real")
        result = np.asarray(supplied.real, dtype=np.float64)
        if result.ndim != 1 or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite one-dimensional array")
        return result

    component_angles = real_vector(
        component_incidence_axis_angle_rad,
        "component_incidence_axis_angle_rad",
    )
    effective_angles = real_vector(
        effective_incidence_angle_rad,
        "effective_incidence_angle_rad",
    )
    if component_angles.size != len(geometry_revisions) or effective_angles.shape != (
        len(geometry_revisions),
    ):
        raise ValueError("incidence calibration vectors and component revisions must align")
    if any(
        not isinstance(value, str) or not value
        for value in (detector_panel_revision, source_revision)
    ):
        raise ValueError("detector-panel and source revisions must be nonempty")
    if (
        isinstance(source_state_count, (bool, np.bool_))
        or not isinstance(source_state_count, (int, np.integer))
        or int(source_state_count) < 1
    ):
        raise ValueError("source_state_count must be a positive integer")
    return canonical_revision_sha256(
        ("definition_id", "incidence_angle_component_calibration_binding.v1"),
        ("scan_calibration_revision", scan_calibration_revision),
        ("component_sample_geometry_revision", geometry_revisions),
        (
            "component_incidence_axis_angle_rad",
            component_angles,
        ),
        (
            "effective_incidence_angle_rad",
            effective_angles,
        ),
        ("detector_panel_revision", detector_panel_revision),
        ("source_revision", source_revision),
        ("source_state_count", source_state_count),
    )


def source_realization_revision(
    *,
    source_sampling_model_id: str,
    source_rng_model_id: str,
    source_seed: int,
    source_parameter_revision: str,
    incident_sample_id: ArrayLike,
    origin_lab_m: ArrayLike,
    direction_lab: ArrayLike,
    wavelength_A: ArrayLike,
    source_weight: ArrayLike,
    polarization_state_id: tuple[str, ...],
) -> str:
    """Return the complete canonical source-realization revision."""

    return canonical_revision_sha256(
        ("direction_lab", direction_lab),
        ("incident_sample_id", incident_sample_id),
        ("origin_lab_m", origin_lab_m),
        ("polarization_state_id", polarization_state_id),
        ("source_parameter_revision", source_parameter_revision),
        ("source_rng_model_id", source_rng_model_id),
        ("source_sampling_model_id", source_sampling_model_id),
        ("source_seed", source_seed),
        ("source_weight", source_weight),
        ("wavelength_A", wavelength_A),
    )


SAMPLE_INTERSECTION_MODEL_ID = "unique_forward_plane_intersection.v1"
_SAMPLE_PLANE_OFFSET_RESOLUTION_M = 1.0e-12


def _canonical_sample_plane_offset_m(
    rotation: NDArray[np.float64],
    translation_m: NDArray[np.float64],
) -> float:
    """Return the signed plane offset at the frozen geometry position resolution."""

    offset_m = float(np.dot(rotation[:, 2], translation_m))
    if not np.isfinite(offset_m):
        raise ValueError("sample plane offset must be finite")
    scaled_offset = offset_m / _SAMPLE_PLANE_OFFSET_RESOLUTION_M
    if np.isfinite(scaled_offset) and abs(scaled_offset) < 2**52:
        offset_m = float(round(scaled_offset)) * _SAMPLE_PLANE_OFFSET_RESOLUTION_M
    return 0.0 if offset_m == 0.0 else offset_m


def sample_geometry_revision_sha256(
    *,
    lab_from_sample_rotation: ArrayLike,
    lab_from_sample_translation_m: ArrayLike,
    sample_support_model_id: str,
    sample_width_m: float | None,
    sample_length_m: float | None,
) -> str:
    """Return the canonical sample-entrance geometry revision."""

    rotation = _array(
        lab_from_sample_rotation,
        np.float64,
        (3, 3),
        "lab_from_sample_rotation",
    )
    translation_m = _array(
        lab_from_sample_translation_m,
        np.float64,
        (3,),
        "lab_from_sample_translation_m",
    )
    fields: list[tuple[str, object]] = [
        ("intersection_model_id", SAMPLE_INTERSECTION_MODEL_ID),
        ("lab_from_sample_rotation", rotation),
        ("lab_from_sample_source_frame", "sample"),
        ("lab_from_sample_target_frame", "lab"),
        ("sample_entrance_revision_schema", "sample_entrance_revision.v2"),
        ("sample_support_model_id", sample_support_model_id),
    ]
    if sample_support_model_id == "finite_rectangle.v1":
        if sample_width_m is None or sample_length_m is None:
            raise ValueError("finite sample geometry revision requires width and length")
        fields.extend(
            (
                ("lab_from_sample_translation_m", translation_m),
                ("sample_length_m", sample_length_m),
                ("sample_width_m", sample_width_m),
            )
        )
    elif sample_support_model_id == "unbounded_plane.v1":
        if sample_width_m is not None or sample_length_m is not None:
            raise ValueError("unbounded sample geometry revision requires absent dimensions")
        fields.append(
            (
                "sample_plane_signed_normal_offset_lab_m",
                _canonical_sample_plane_offset_m(rotation, translation_m),
            )
        )
    else:
        raise ValueError("sample geometry revision requires a supported model ID")
    return canonical_revision_sha256(*fields)


def material_optics_revision_sha256(
    *,
    material_id: str,
    wavelength_A: ArrayLike,
    n_complex: ArrayLike,
    provenance: str,
) -> str:
    """Return the canonical exact material-optics revision."""

    return canonical_revision_sha256(
        ("material_id", material_id),
        ("material_optics_revision_schema", "material_optics_revision.v2"),
        ("n_complex", n_complex),
        ("provenance", provenance),
        ("wavelength_A", wavelength_A),
    )


def _batch(
    instance: object,
    identity_name: str,
    specs: tuple[_ArraySpec, ...],
    text_names: tuple[str, ...] = (),
) -> int:
    ids = _array(getattr(instance, identity_name), np.int64, (None,), identity_name, True)
    if np.unique(ids).size != ids.size:
        raise ValueError(f"{identity_name} must be unique")
    object.__setattr__(instance, identity_name, ids)
    for name, dtype, trailing_shape, nonnegative in specs:
        object.__setattr__(
            instance,
            name,
            _array(getattr(instance, name), dtype, (ids.size, *trailing_shape), name, nonnegative),
        )
    for name in text_names:
        object.__setattr__(instance, name, _texts(getattr(instance, name), ids.size, name))
    return ids.size


@dataclass(frozen=True, slots=True)
class IncidentSampleBatch:
    incident_sample_id: NDArray[np.int64]
    origin_lab_m: NDArray[np.float64]
    direction_lab: NDArray[np.float64]
    wavelength_A: NDArray[np.float64]
    source_weight: NDArray[np.float64]
    polarization_state_id: tuple[str, ...]
    source_sampling_model_id: str
    source_rng_model_id: str
    source_seed: int
    source_parameter_provenance: str
    source_parameter_revision: str = field(init=False)
    source_weight_revision: str = field(init=False)
    source_revision: str = field(init=False)

    def __post_init__(self) -> None:
        size = _batch(
            self,
            "incident_sample_id",
            (
                ("origin_lab_m", np.float64, (3,), False),
                ("direction_lab", np.float64, (3,), False),
                ("wavelength_A", np.float64, (), True),
                ("source_weight", np.float64, (), True),
            ),
            ("polarization_state_id",),
        )
        if np.any(self.wavelength_A == 0) or not np.allclose(
            np.linalg.norm(self.direction_lab, axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("wavelengths must be positive and directions unit length")
        if size == 0 or not np.isclose(
            np.sum(self.source_weight, dtype=np.float64),
            1.0,
            rtol=0.0,
            atol=2.0e-15,
        ):
            raise ValueError(
                "source_weight must be finite nonnegative probability mass summing to one"
            )
        object.__setattr__(
            self,
            "source_sampling_model_id",
            _versioned_id(self.source_sampling_model_id, "source_sampling_model_id"),
        )
        object.__setattr__(
            self,
            "source_rng_model_id",
            _versioned_id(self.source_rng_model_id, "source_rng_model_id"),
        )
        if (
            isinstance(self.source_seed, bool)
            or not isinstance(self.source_seed, (int, np.integer))
            or self.source_seed < 0
            or self.source_seed > 2**64 - 1
        ):
            raise ValueError("source_seed must be a nonnegative unsigned 64-bit integer")
        object.__setattr__(self, "source_seed", int(self.source_seed))
        if (
            not isinstance(self.source_parameter_provenance, str)
            or not self.source_parameter_provenance
        ):
            raise ValueError("source_parameter_provenance must be nonempty canonical text")
        parameter_revision = canonical_revision_sha256(
            ("source_parameter_provenance", self.source_parameter_provenance),
        )
        object.__setattr__(self, "source_parameter_revision", parameter_revision)
        weight_revision = canonical_revision_sha256(
            ("incident_sample_id", self.incident_sample_id),
            ("source_weight", self.source_weight),
        )
        object.__setattr__(
            self,
            "source_weight_revision",
            weight_revision,
        )
        source_revision = source_realization_revision(
            source_sampling_model_id=self.source_sampling_model_id,
            source_rng_model_id=self.source_rng_model_id,
            source_seed=self.source_seed,
            source_parameter_revision=parameter_revision,
            incident_sample_id=self.incident_sample_id,
            origin_lab_m=self.origin_lab_m,
            direction_lab=self.direction_lab,
            wavelength_A=self.wavelength_A,
            source_weight=self.source_weight,
            polarization_state_id=self.polarization_state_id,
        )
        object.__setattr__(self, "source_revision", source_revision)


@dataclass(frozen=True, slots=True)
class MaterialOptics:
    material_id: str
    wavelength_A: NDArray[np.float64]
    n_complex: NDArray[np.complex128]
    provenance: str
    material_revision: str = field(init=False)

    def __post_init__(self) -> None:
        wavelength = _array(self.wavelength_A, np.float64, (None,), "wavelength_A", True)
        object.__setattr__(self, "wavelength_A", wavelength)
        object.__setattr__(
            self,
            "n_complex",
            _array(self.n_complex, np.complex128, (wavelength.size,), "n_complex"),
        )
        if (
            wavelength.size == 0
            or np.any(wavelength == 0)
            or np.any(np.diff(wavelength) <= 0.0)
            or not self.material_id
            or not self.provenance
        ):
            raise ValueError("material identity, provenance, and positive wavelengths are required")
        if np.any(self.n_complex.imag < 0.0):
            raise ValueError("n_complex imaginary part must be nonnegative for absorption")
        object.__setattr__(
            self,
            "material_revision",
            material_optics_revision_sha256(
                material_id=self.material_id,
                wavelength_A=self.wavelength_A,
                n_complex=self.n_complex,
                provenance=self.provenance,
            ),
        )


@dataclass(frozen=True, slots=True)
class IncidentStateBatch:
    incident_state_id: NDArray[np.int64]
    incident_sample_id: NDArray[np.int64]
    source_origin_lab_m: NDArray[np.float64]
    source_direction_lab: NDArray[np.float64]
    sample_intersection_lab_m: NDArray[np.float64]
    direction_sample: NDArray[np.float64]
    k_air_sample_Ainv: NDArray[np.float64]
    k_film_phase_sample_Ainv: NDArray[np.float64]
    kz_film_Ainv: NDArray[np.complex128]
    entrance_amplitude: NDArray[np.complex128]
    footprint_acceptance: NDArray[np.float64]
    source_weight: NDArray[np.float64]
    wavelength_A: NDArray[np.float64]
    polarization_state_id: tuple[str, ...]
    status: tuple[ValidityCode, ...]
    valid: NDArray[np.bool_]
    source_sampling_model_id: str
    source_rng_model_id: str
    source_seed: int
    source_parameter_provenance: str
    source_parameter_revision: str
    source_weight_revision: str
    source_revision: str
    sample_geometry_revision: str
    material_revision: str
    incident_model_id: str

    def __post_init__(self) -> None:
        size = _batch(
            self,
            "incident_state_id",
            (
                ("incident_sample_id", np.int64, (), True),
                ("source_origin_lab_m", np.float64, (3,), False),
                ("source_direction_lab", np.float64, (3,), False),
                ("sample_intersection_lab_m", np.float64, (3,), False),
                ("direction_sample", np.float64, (3,), False),
                ("k_air_sample_Ainv", np.float64, (3,), False),
                ("k_film_phase_sample_Ainv", np.float64, (3,), False),
                ("kz_film_Ainv", np.complex128, (), False),
                ("entrance_amplitude", np.complex128, (), False),
                ("footprint_acceptance", np.float64, (), True),
                ("source_weight", np.float64, (), True),
                ("wavelength_A", np.float64, (), True),
                ("valid", np.bool_, (), False),
            ),
            ("polarization_state_id",),
        )
        if size == 0 or not np.isclose(
            np.sum(self.source_weight, dtype=np.float64),
            1.0,
            rtol=0.0,
            atol=2.0e-15,
        ):
            raise ValueError(
                "source_weight must be finite nonnegative probability mass summing to one"
            )
        if np.any(self.wavelength_A == 0):
            raise ValueError("wavelength_A must be positive")
        if not np.allclose(
            np.linalg.norm(self.source_direction_lab, axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("source_direction_lab must contain unit vectors")

        status = tuple(ValidityCode(item) for item in self.status)
        if len(status) != size:
            raise ValueError(f"status must contain {size} ValidityCode values")
        object.__setattr__(self, "status", status)
        status_valid = np.fromiter(
            (item is ValidityCode.VALID for item in status), dtype=np.bool_, count=size
        )
        if not np.array_equal(self.valid, status_valid):
            raise ValueError("valid must agree exactly with status == ValidityCode.VALID")

        for name in ("source_sampling_model_id", "source_rng_model_id", "incident_model_id"):
            object.__setattr__(self, name, _versioned_id(getattr(self, name), name))
        if (
            isinstance(self.source_seed, bool)
            or not isinstance(self.source_seed, (int, np.integer))
            or self.source_seed < 0
            or self.source_seed > 2**64 - 1
        ):
            raise ValueError("source_seed must be a nonnegative unsigned 64-bit integer")
        object.__setattr__(self, "source_seed", int(self.source_seed))
        if (
            not isinstance(self.source_parameter_provenance, str)
            or not self.source_parameter_provenance
        ):
            raise ValueError("source_parameter_provenance must be nonempty canonical text")
        for name in (
            "source_parameter_revision",
            "source_weight_revision",
            "source_revision",
            "sample_geometry_revision",
            "material_revision",
        ):
            object.__setattr__(self, name, _sha256_revision(getattr(self, name), name))
        expected_parameter_revision = canonical_revision_sha256(
            ("source_parameter_provenance", self.source_parameter_provenance),
        )
        if self.source_parameter_revision != expected_parameter_revision:
            raise ValueError("source_parameter_revision does not match canonical provenance")
        if np.unique(self.incident_sample_id).size != size:
            raise ValueError(
                "one_transmitted_channel.v1 requires one state per unique source sample"
            )
        expected_weight_revision = canonical_revision_sha256(
            ("incident_sample_id", self.incident_sample_id),
            ("source_weight", self.source_weight),
        )
        if self.source_weight_revision != expected_weight_revision:
            raise ValueError("source_weight_revision does not match the transported source mass")
        if self.incident_model_id != "one_transmitted_channel.v1":
            raise ValueError("unsupported incident_model_id")

        geometry_failure = np.fromiter(
            (
                item
                in (
                    ValidityCode.PARALLEL,
                    ValidityCode.BACKWARD,
                    ValidityCode.OUTSIDE_SUPPORT,
                )
                for item in status
            ),
            dtype=np.bool_,
            count=size,
        )
        optical_failure = np.fromiter(
            (item in (ValidityCode.NO_SOLUTION, ValidityCode.NUMERIC_FAILURE) for item in status),
            dtype=np.bool_,
            count=size,
        )
        if not np.all(status_valid | geometry_failure | optical_failure):
            raise ValueError("status is not valid at the incident boundary")

        geometry_zero_arrays = (
            self.sample_intersection_lab_m,
            self.direction_sample,
            self.k_air_sample_Ainv,
            self.k_film_phase_sample_Ainv,
            self.kz_film_Ainv,
            self.entrance_amplitude,
            self.footprint_acceptance,
        )
        if any(np.any(array[geometry_failure] != 0) for array in geometry_zero_arrays):
            raise ValueError("geometry-failure payload must be zero after source provenance")

        accepted_geometry = status_valid | optical_failure
        if np.any((self.footprint_acceptance < 0.0) | (self.footprint_acceptance > 1.0)):
            raise ValueError("footprint_acceptance must lie in [0, 1]")
        if not np.allclose(
            np.linalg.norm(self.direction_sample[accepted_geometry], axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("accepted geometry requires unit direction_sample")
        expected_k_air = (2.0 * np.pi / self.wavelength_A[accepted_geometry])[
            :, None
        ] * self.direction_sample[accepted_geometry]
        wavevector_atol = 1.4210854715202206e-14
        wavevector_rtol = 2.2737367544328376e-13
        if not np.allclose(
            self.k_air_sample_Ainv[accepted_geometry],
            expected_k_air,
            rtol=wavevector_rtol,
            atol=wavevector_atol,
        ):
            raise ValueError("k_air_sample_Ainv must equal (2*pi/wavelength_A)*direction_sample")

        optical_zero_arrays = (
            self.k_film_phase_sample_Ainv,
            self.kz_film_Ainv,
            self.entrance_amplitude,
        )
        if any(np.any(array[optical_failure] != 0) for array in optical_zero_arrays):
            raise ValueError("optical-failure film payload must be zero")
        if not np.allclose(
            self.k_film_phase_sample_Ainv[status_valid, :2],
            self.k_air_sample_Ainv[status_valid, :2],
            rtol=wavevector_rtol,
            atol=wavevector_atol,
        ):
            raise ValueError("valid incident modes must conserve tangential wavevector")
        if not np.allclose(
            self.k_film_phase_sample_Ainv[status_valid, 2],
            self.kz_film_Ainv[status_valid].real,
            rtol=wavevector_rtol,
            atol=wavevector_atol,
        ):
            raise ValueError("film phase normal must equal real(kz_film_Ainv)")
        expected_source_revision = source_realization_revision(
            source_sampling_model_id=self.source_sampling_model_id,
            source_rng_model_id=self.source_rng_model_id,
            source_seed=self.source_seed,
            source_parameter_revision=self.source_parameter_revision,
            incident_sample_id=self.incident_sample_id,
            origin_lab_m=self.source_origin_lab_m,
            direction_lab=self.source_direction_lab,
            wavelength_A=self.wavelength_A,
            source_weight=self.source_weight,
            polarization_state_id=self.polarization_state_id,
        )
        if self.source_revision != expected_source_revision:
            raise ValueError(
                "source_revision does not bind the transported source support and mass"
            )


@dataclass(frozen=True, slots=True)
class RodCatalog:
    rod_id: NDArray[np.int64]
    phase_id: tuple[str, ...]
    h: NDArray[np.int32]
    k: NDArray[np.int32]
    family_id: tuple[str, ...]
    family_key: tuple[str, ...]
    qr_Ainv: NDArray[np.float64]
    reciprocal_basis_Ainv: NDArray[np.float64]
    symmetry_metadata: tuple[str, ...]

    def __post_init__(self) -> None:
        _batch(
            self,
            "rod_id",
            (
                ("h", np.int32, (), False),
                ("k", np.int32, (), False),
                ("qr_Ainv", np.float64, (), True),
            ),
            ("phase_id", "family_id", "family_key", "symmetry_metadata"),
        )
        basis = _array(self.reciprocal_basis_Ainv, np.float64, (3, 3), "reciprocal_basis_Ainv")
        if np.isclose(np.linalg.det(basis), 0.0):
            raise ValueError("reciprocal_basis_Ainv must be nonsingular")
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)


@dataclass(frozen=True, slots=True)
class RodQueryBatch:
    event_id: NDArray[np.int64]
    rod_id: NDArray[np.int64]
    phase_id: tuple[str, ...]
    h: NDArray[np.int32]
    k: NDArray[np.int32]
    q_sample_normal_Ainv: NDArray[np.float64]
    l_coordinate: NDArray[np.float64]
    wavelength_A: NDArray[np.float64]

    def __post_init__(self) -> None:
        _batch(
            self,
            "event_id",
            (
                ("rod_id", np.int64, (), True),
                ("h", np.int32, (), False),
                ("k", np.int32, (), False),
                ("q_sample_normal_Ainv", np.float64, (), False),
                ("l_coordinate", np.float64, (), False),
                ("wavelength_A", np.float64, (), True),
            ),
            ("phase_id",),
        )
        if np.any(self.wavelength_A == 0):
            raise ValueError("wavelength_A must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class LayerAmplitudeResult:
    event_id: NDArray[np.int64]
    rod_id: NDArray[np.int64]
    phase_id: tuple[str, ...]
    f_plus_e: NDArray[np.complex128]
    f_minus_e: NDArray[np.complex128] | None
    normalization: LayerAmplitudeNormalization
    phase_sign: LayerPhaseSign
    gauge_id: str
    layer_normal_crystal: NDArray[np.float64]
    layer_repeat_A: float

    def __post_init__(self) -> None:
        size = _batch(
            self,
            "event_id",
            (("rod_id", np.int64, (), True), ("f_plus_e", np.complex128, (), False)),
            ("phase_id",),
        )
        if self.f_minus_e is not None:
            object.__setattr__(
                self,
                "f_minus_e",
                _array(self.f_minus_e, np.complex128, (size,), "f_minus_e"),
            )
        normalization = LayerAmplitudeNormalization(self.normalization)
        phase_sign = LayerPhaseSign(self.phase_sign)
        normal = _array(self.layer_normal_crystal, np.float64, (3,), "layer_normal_crystal")
        if not np.isclose(np.linalg.norm(normal), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("layer_normal_crystal must be a unit vector")
        repeat = float(self.layer_repeat_A)
        if not np.isfinite(repeat) or repeat <= 0.0:
            raise ValueError("layer_repeat_A must be finite and positive")
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "phase_sign", phase_sign)
        object.__setattr__(self, "gauge_id", _versioned_id(self.gauge_id, "gauge_id"))
        object.__setattr__(self, "layer_normal_crystal", normal)
        object.__setattr__(self, "layer_repeat_A", repeat)


@dataclass(frozen=True, slots=True, kw_only=True)
class LayerNormalQBatch:
    event_id: NDArray[np.int64]
    rod_id: NDArray[np.int64]
    phase_id: tuple[str, ...]
    layer_normal_q_Ainv: NDArray[np.float64]
    gauge_id: str

    def __post_init__(self) -> None:
        _batch(
            self,
            "event_id",
            (
                ("rod_id", np.int64, (), True),
                ("layer_normal_q_Ainv", np.float64, (), False),
            ),
            ("phase_id",),
        )
        object.__setattr__(self, "gauge_id", _versioned_id(self.gauge_id, "gauge_id"))


@dataclass(frozen=True, slots=True, kw_only=True)
class EventIntensityResult:
    event_id: NDArray[np.int64]
    scattering_strength_A2: NDArray[np.float64]
    model_id: str
    model_component_id: str
    population_group_id: str | None
    normalization: EventIntensityNormalization

    def __post_init__(self) -> None:
        _batch(self, "event_id", (("scattering_strength_A2", np.float64, (), True),))
        if (
            not self.model_id
            or not self.model_component_id
            or (self.population_group_id is not None and not self.population_group_id)
        ):
            raise ValueError("model identity and population group are required")
        object.__setattr__(self, "normalization", EventIntensityNormalization(self.normalization))
