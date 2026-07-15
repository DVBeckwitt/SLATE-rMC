"""Immutable array contracts shared by T02 through T05."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

CONTRACT_API_VERSION = 4
_ArraySpec = tuple[str, np.dtype[Any] | type[np.generic], tuple[int, ...], bool]


def _array(
    value: ArrayLike,
    dtype: np.dtype[Any] | type[np.generic],
    shape: tuple[int | None, ...],
    name: str,
    nonnegative: bool = False,
) -> NDArray[Any]:
    supplied = np.asarray(value)
    target = np.dtype(dtype)
    if np.issubdtype(target, np.integer) and supplied.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    array = np.array(value, dtype=target, copy=True, order="C")
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
    correlation_model: str

    def __post_init__(self) -> None:
        _batch(
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
        if not np.isclose(self.source_weight.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("source_weight must sum to one")
        if not self.correlation_model:
            raise ValueError("correlation_model is required")


@dataclass(frozen=True, slots=True)
class MaterialOptics:
    material_id: str
    wavelength_A: NDArray[np.float64]
    n_complex: NDArray[np.complex128]
    delta: NDArray[np.float64]
    beta: NDArray[np.float64]
    mu_Ainv: NDArray[np.float64]
    provenance: str

    def __post_init__(self) -> None:
        wavelength = _array(self.wavelength_A, np.float64, (None,), "wavelength_A", True)
        object.__setattr__(self, "wavelength_A", wavelength)
        for name, dtype, nonnegative in (
            ("n_complex", np.complex128, False),
            ("delta", np.float64, False),
            ("beta", np.float64, True),
            ("mu_Ainv", np.float64, True),
        ):
            object.__setattr__(
                self, name, _array(getattr(self, name), dtype, (wavelength.size,), name, nonnegative)
            )
        if np.any(wavelength == 0) or not self.material_id or not self.provenance:
            raise ValueError("material identity, provenance, and positive wavelengths are required")


@dataclass(frozen=True, slots=True)
class IncidentStateBatch:
    incident_state_id: NDArray[np.int64]
    incident_sample_id: NDArray[np.int64]
    sample_intersection_lab_m: NDArray[np.float64]
    direction_sample: NDArray[np.float64]
    k_air_sample_Ainv: NDArray[np.float64]
    k_film_phase_sample_Ainv: NDArray[np.float64]
    kz_film_Ainv: NDArray[np.complex128]
    entrance_amplitude: NDArray[np.complex128]
    footprint_acceptance: NDArray[np.float64]
    source_weight: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        _batch(
            self,
            "incident_state_id",
            (
                ("incident_sample_id", np.int64, (), True),
                ("sample_intersection_lab_m", np.float64, (3,), False),
                ("direction_sample", np.float64, (3,), False),
                ("k_air_sample_Ainv", np.float64, (3,), False),
                ("k_film_phase_sample_Ainv", np.float64, (3,), False),
                ("kz_film_Ainv", np.complex128, (), False),
                ("entrance_amplitude", np.complex128, (), False),
                ("footprint_acceptance", np.float64, (), True),
                ("source_weight", np.float64, (), True),
                ("valid", np.bool_, (), False),
            ),
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
            (("h", np.int32, (), False), ("k", np.int32, (), False), ("qr_Ainv", np.float64, (), True)),
            ("phase_id", "family_id", "family_key", "symmetry_metadata"),
        )
        basis = _array(self.reciprocal_basis_Ainv, np.float64, (3, 3), "reciprocal_basis_Ainv")
        if np.isclose(np.linalg.det(basis), 0.0):
            raise ValueError("reciprocal_basis_Ainv must be nonsingular")
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)


@dataclass(frozen=True, slots=True)
class ScatteringEventBatch:
    event_id: NDArray[np.int64]
    incident_state_id: NDArray[np.int64]
    rod_id: NDArray[np.int64]
    wavelength_A: NDArray[np.float64]
    q_internal_sample_Ainv: NDArray[np.float64]
    qz_Ainv: NDArray[np.float64]
    l_coordinate: NDArray[np.float64]
    kf_film_phase_sample_Ainv: NDArray[np.float64]
    reciprocal_weight: NDArray[np.float64]
    ewald_residual_Ainv: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        _batch(
            self,
            "event_id",
            (
                ("incident_state_id", np.int64, (), True),
                ("rod_id", np.int64, (), True),
                ("wavelength_A", np.float64, (), True),
                ("q_internal_sample_Ainv", np.float64, (3,), False),
                ("qz_Ainv", np.float64, (), False),
                ("l_coordinate", np.float64, (), False),
                ("kf_film_phase_sample_Ainv", np.float64, (3,), False),
                ("reciprocal_weight", np.float64, (), True),
                ("ewald_residual_Ainv", np.float64, (), False),
                ("valid", np.bool_, (), False),
            ),
        )
        if np.any(self.wavelength_A == 0):
            raise ValueError("wavelength_A must be positive")


@dataclass(frozen=True, slots=True)
class RodQueryBatch:
    event_id: NDArray[np.int64]
    rod_id: NDArray[np.int64]
    phase_id: tuple[str, ...]
    h: NDArray[np.int32]
    k: NDArray[np.int32]
    qz_Ainv: NDArray[np.float64]
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
                ("qz_Ainv", np.float64, (), False),
                ("l_coordinate", np.float64, (), False),
                ("wavelength_A", np.float64, (), True),
            ),
            ("phase_id",),
        )
        if np.any(self.wavelength_A == 0):
            raise ValueError("wavelength_A must be positive")


@dataclass(frozen=True, slots=True)
class LayerAmplitudeResult:
    event_id: NDArray[np.int64]
    f_plus: NDArray[np.complex128]
    f_minus: NDArray[np.complex128] | None

    def __post_init__(self) -> None:
        size = _batch(self, "event_id", (("f_plus", np.complex128, (), False),))
        if self.f_minus is not None:
            object.__setattr__(
                self, "f_minus", _array(self.f_minus, np.complex128, (size,), "f_minus")
            )


@dataclass(frozen=True, slots=True)
class EventIntensityResult:
    event_id: NDArray[np.int64]
    intensity_per_sr: NDArray[np.float64]
    model_id: str
    model_component_id: str
    population_group_id: str | None
    normalization: str

    def __post_init__(self) -> None:
        _batch(self, "event_id", (("intensity_per_sr", np.float64, (), True),))
        if not self.model_id or not self.model_component_id or not self.normalization:
            raise ValueError("model identity and normalization are required")


@dataclass(frozen=True, slots=True)
class OutgoingWaveBatch:
    event_id: NDArray[np.int64]
    kf_air_lab_Ainv: NDArray[np.float64]
    exit_amplitude: NDArray[np.complex128]
    attenuation_weight: NDArray[np.float64]
    optical_weight: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        _batch(
            self,
            "event_id",
            (
                ("kf_air_lab_Ainv", np.float64, (3,), False),
                ("exit_amplitude", np.complex128, (), False),
                ("attenuation_weight", np.float64, (), True),
                ("optical_weight", np.float64, (), True),
                ("valid", np.bool_, (), False),
            ),
        )


@dataclass(frozen=True, slots=True)
class DetectorHitBatch:
    event_id: NDArray[np.int64]
    column_px: NDArray[np.float64]
    row_px: NDArray[np.float64]
    pixel_solid_angle_sr: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        _batch(
            self,
            "event_id",
            (
                ("column_px", np.float64, (), False),
                ("row_px", np.float64, (), False),
                ("pixel_solid_angle_sr", np.float64, (), True),
                ("valid", np.bool_, (), False),
            ),
        )
