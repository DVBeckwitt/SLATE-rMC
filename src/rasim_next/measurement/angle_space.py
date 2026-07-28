"""Sparse full-pixel splitting and normalized finite-bin angle fields."""

from __future__ import annotations

import hashlib
import math
from array import array
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.validity import ValidityCode
from rasim_next.geometry.angles import (
    AngleFrame,
    _raw_chi_to_phi,
    _raw_chi_to_unwrapped_phi,
    angles_to_detector_coordinates,
    detector_coordinates_to_angles,
)
from rasim_next.geometry.detector import _intersect_detector_plane
from rasim_next.geometry.instrument import CompiledInstrument

_FLOAT_EPS = np.finfo(np.float64).eps
_GRID_TOL = 128.0 * _FLOAT_EPS
_POLE_TOL_FACTOR = 128.0 * _FLOAT_EPS
_CONSERVATION_TOL = 3.0e-11
_PROFILE_CONSERVATION_TOL = 3.0e-9
_CORNER_ROW_TILE_SIZE = 64
_COVERAGE_ENTRY_TILE_SIZE = 262_144
_LOSS_FIELD_NAMES = (
    "detector_mask_excluded_signal",
    "detector_mask_excluded_normalization",
    "angular_lost_signal",
    "angular_lost_normalization",
    "angle_mask_excluded_signal",
    "angle_mask_excluded_normalization",
)


def _readonly_float(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied):
        raise ValueError(f"{name} must be real")
    result = np.array(supplied, dtype=np.float64, copy=True, order="C")
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    result.setflags(write=False)
    return result


def _readonly_bool(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.bool_]:
    supplied = np.asarray(value)
    if supplied.dtype.kind != "b":
        raise ValueError(f"{name} must be boolean")
    result = np.array(supplied, dtype=np.bool_, copy=True, order="C")
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    result.setflags(write=False)
    return result


def _readonly_int(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.int64]:
    supplied = np.asarray(value)
    if supplied.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    result = np.array(supplied, dtype=np.int64, copy=True, order="C")
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    result.setflags(write=False)
    return result


def _version(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _freeze_nonnegative_losses(instance: object) -> None:
    for name in _LOSS_FIELD_NAMES:
        value = float(getattr(instance, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        object.__setattr__(instance, name, value)


def _raw_chi_to_phi_permutation(chi_bin_count: int) -> NDArray[np.int64]:
    output_row = np.arange(chi_bin_count, dtype=np.int64)
    return (3 * chi_bin_count // 4 - 1 - output_row) % chi_bin_count


@dataclass(frozen=True, slots=True)
class AngleBinGrid:
    """Uniform canonical grid indexed ``[raw_chi_bin, two_theta_bin]``."""

    two_theta_edges_rad: NDArray[np.float64]
    chi_raw_edges_rad: NDArray[np.float64]
    revision: str
    two_theta_centers_rad: NDArray[np.float64] = field(init=False, repr=False)
    chi_raw_centers_rad: NDArray[np.float64] = field(init=False, repr=False)

    bin_measure: ClassVar[str] = "one-per-valid-bin.v1"
    canonical_order: ClassVar[str] = "raw-chi-major.two-theta-minor.v1"
    grid_contract_revision: ClassVar[str] = "uniform-zero-radial.full-raw-chi.v1"

    def __post_init__(self) -> None:
        theta_supplied = np.asarray(self.two_theta_edges_rad)
        chi_supplied = np.asarray(self.chi_raw_edges_rad)
        if np.iscomplexobj(theta_supplied) or np.iscomplexobj(chi_supplied):
            raise ValueError("angle-bin edges must be real")
        theta = np.array(theta_supplied, dtype=np.float64, copy=True, order="C")
        chi = np.array(chi_supplied, dtype=np.float64, copy=True, order="C")
        if (
            theta.ndim != 1
            or chi.ndim != 1
            or theta.size < 2
            or chi.size < 2
            or not np.all(np.isfinite(theta))
            or not np.all(np.isfinite(chi))
            or np.any(np.diff(theta) <= 0.0)
            or np.any(np.diff(chi) <= 0.0)
        ):
            raise ValueError("angle-bin edges must be finite, one-dimensional, and increasing")
        if (
            theta[0] < 0.0
            or not np.isclose(theta[0], 0.0, rtol=0.0, atol=_GRID_TOL)
            or theta[-1] > np.pi
        ):
            raise ValueError("two_theta_edges_rad must start at zero and end no later than pi")
        if not np.allclose(chi[[0, -1]], [-np.pi, np.pi], rtol=0.0, atol=_GRID_TOL):
            raise ValueError("chi_raw_edges_rad must span exactly [-pi, pi]")
        canonical_theta = np.linspace(0.0, float(theta[-1]), theta.size)
        canonical_chi = np.linspace(-np.pi, np.pi, chi.size)
        if not np.allclose(theta, canonical_theta, rtol=_GRID_TOL, atol=_GRID_TOL):
            raise ValueError("two_theta_edges_rad must be uniform")
        if not np.allclose(chi, canonical_chi, rtol=_GRID_TOL, atol=_GRID_TOL):
            raise ValueError("chi_raw_edges_rad must be uniform")
        theta = canonical_theta
        chi = canonical_chi
        chi_bins = chi.size - 1
        if chi_bins % 4 != 0:
            raise ValueError(
                "the raw-chi bin count must be divisible by four for an exact phi view"
            )

        theta_centers = 0.5 * (theta[:-1] + theta[1:])
        chi_centers = 0.5 * (chi[:-1] + chi[1:])
        mapped_phi = _raw_chi_to_phi(chi_centers)
        permutation = _raw_chi_to_phi_permutation(chi_bins)
        phi_centers = mapped_phi[permutation]
        if not np.allclose(phi_centers, chi_centers, rtol=0.0, atol=4.0 * _GRID_TOL):
            raise ValueError("raw-chi edges do not map to the frozen increasing-phi bin grid")

        for item in (theta, chi, theta_centers, chi_centers):
            item.setflags(write=False)
        object.__setattr__(self, "two_theta_edges_rad", theta)
        object.__setattr__(self, "chi_raw_edges_rad", chi)
        object.__setattr__(self, "revision", _version(self.revision, "revision"))
        object.__setattr__(self, "two_theta_centers_rad", theta_centers)
        object.__setattr__(self, "chi_raw_centers_rad", chi_centers)

    @property
    def shape(self) -> tuple[int, int]:
        return self.chi_raw_edges_rad.size - 1, self.two_theta_edges_rad.size - 1

    @property
    def seam_rad(self) -> float:
        return float(self.chi_raw_edges_rad[0])

    @property
    def phi_edges_rad(self) -> NDArray[np.float64]:
        """Canonical increasing-phi edges, sharing the full-period edge storage."""

        return self.chi_raw_edges_rad

    @property
    def phi_centers_rad(self) -> NDArray[np.float64]:
        """Canonical increasing-phi centers, sharing the uniform center storage."""

        return self.chi_raw_centers_rad


@dataclass(frozen=True, slots=True)
class SparseDetectorAngleProjector:
    """Immutable sparse coverage ``M[b, k]`` in detector-pixel-major records."""

    instrument: CompiledInstrument
    angle_frame: AngleFrame
    grid: AngleBinGrid
    detector_valid_mask: NDArray[np.bool_]
    angle_bin_valid_mask: NDArray[np.bool_]
    coverage_pixel_index: NDArray[np.int64]
    coverage_bin_index: NDArray[np.int64]
    weight: NDArray[np.float64]
    lost_support: NDArray[np.float64]
    instrument_fingerprint: str
    cache_key: str

    projector_revision: ClassVar[str] = "physical-corner-full-pixel-split.v1"
    dtype: ClassVar[str] = "float64"
    sparse_engine_revision: ClassVar[str] = "pixel-major-sorted-coverage-records.v1"
    summation_engine_revision: ClassVar[str] = "tiled-numpy-bincount.v1"
    polygon_revision: ClassVar[str] = "fixed-tl-br-diagonal.sutherland-hodgman.v1"
    unwrap_revision: ClassVar[str] = "first-vertex-shortest-arc.pi-tie-negative.v1"
    pole_revision: ClassVar[str] = "physical-pixel-fan.duplicated-pole-limits.v1"
    tie_policy_revision: ClassVar[str] = "lower-inclusive.upper-exclusive.outer-max-last.v1"
    clipping_policy: ClassVar[str] = "explicit-loss.no-renormalization.v1"
    corner_row_tile_size: ClassVar[int] = _CORNER_ROW_TILE_SIZE
    coverage_entry_tile_size: ClassVar[int] = _COVERAGE_ENTRY_TILE_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, CompiledInstrument):
            raise TypeError("instrument must be a CompiledInstrument")
        if not isinstance(self.angle_frame, AngleFrame):
            raise TypeError("angle_frame must be an AngleFrame")
        if not isinstance(self.grid, AngleBinGrid):
            raise TypeError("grid must be an AngleBinGrid")
        shape = self.instrument.detector_shape_rc
        pixel_count = math.prod(shape)
        bin_count = math.prod(self.grid.shape)
        detector_mask = _readonly_bool(self.detector_valid_mask, shape, "detector_valid_mask")
        bin_mask = _readonly_bool(
            self.angle_bin_valid_mask,
            self.grid.shape,
            "angle_bin_valid_mask",
        )
        supplied_weight = np.asarray(self.weight)
        if np.iscomplexobj(supplied_weight):
            raise ValueError("weight must be real")
        weight = np.array(supplied_weight, dtype=np.float64, copy=True, order="C")
        if weight.ndim != 1 or not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
            raise ValueError("weight must be a finite, positive one-dimensional array")
        nnz = weight.size
        pixel_index = _readonly_int(
            self.coverage_pixel_index,
            (nnz,),
            "coverage_pixel_index",
        )
        bin_index = _readonly_int(self.coverage_bin_index, (nnz,), "coverage_bin_index")
        lost = _readonly_float(self.lost_support, shape, "lost_support")
        if (
            np.any(pixel_index < 0)
            or np.any(pixel_index >= pixel_count)
            or np.any(bin_index < 0)
            or np.any(bin_index >= bin_count)
            or np.any(lost < 0.0)
            or np.any(lost > 1.0 + _CONSERVATION_TOL)
        ):
            raise ValueError("sparse coverage indices or lost support are invalid")
        if nnz:
            if np.any(np.diff(pixel_index) < 0):
                raise ValueError("coverage records must be in detector-pixel-major order")
            same_pixel = pixel_index[1:] == pixel_index[:-1]
            if np.any(bin_index[1:][same_pixel] <= bin_index[:-1][same_pixel]):
                raise ValueError("coverage bins must be unique and increasing within each pixel")
        column_sum = np.bincount(pixel_index, weights=weight, minlength=pixel_count)
        if not np.allclose(
            column_sum + lost.ravel(),
            1.0,
            rtol=0.0,
            atol=_CONSERVATION_TOL,
        ):
            raise ValueError("every sparse column plus lost support must conserve unity")
        weight.setflags(write=False)
        object.__setattr__(self, "detector_valid_mask", detector_mask)
        object.__setattr__(self, "angle_bin_valid_mask", bin_mask)
        object.__setattr__(self, "coverage_pixel_index", pixel_index)
        object.__setattr__(self, "coverage_bin_index", bin_index)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "lost_support", lost)
        object.__setattr__(
            self,
            "instrument_fingerprint",
            _version(self.instrument_fingerprint, "instrument_fingerprint"),
        )
        object.__setattr__(self, "cache_key", _version(self.cache_key, "cache_key"))
        expected_instrument_fingerprint = _instrument_fingerprint(self.instrument)
        if self.instrument_fingerprint != expected_instrument_fingerprint:
            raise ValueError("instrument_fingerprint does not match the frozen instrument")
        expected_cache_key = _cache_key(
            instrument_fingerprint=expected_instrument_fingerprint,
            angle_frame=self.angle_frame,
            grid=self.grid,
            detector_mask=detector_mask,
            angle_mask=bin_mask,
        )
        if self.cache_key != expected_cache_key:
            raise ValueError("cache_key does not match the frozen projector inputs and revisions")


@dataclass(frozen=True, slots=True)
class SparseDetectorProfileProjector:
    """Sparse physical-pixel coverage for independent local ``(2theta, phi)`` profiles."""

    instrument: CompiledInstrument
    angle_frame: AngleFrame
    two_theta_bounds_rad: NDArray[np.float64]
    phi_bin_edges_rad: NDArray[np.float64]
    detector_valid_mask: NDArray[np.bool_]
    profile_bin_valid_mask: NDArray[np.bool_]
    coverage_pixel_index: NDArray[np.int64]
    coverage_profile_bin_index: NDArray[np.int64]
    weight: NDArray[np.float64]
    profile_pixel_bounds_cr: NDArray[np.int64]
    instrument_fingerprint: str
    cache_key: str

    projector_revision: ClassVar[str] = "cropped-physical-pixel-profile-split.v2"
    polygon_revision: ClassVar[str] = SparseDetectorAngleProjector.polygon_revision
    unwrap_revision: ClassVar[str] = SparseDetectorAngleProjector.unwrap_revision
    pole_revision: ClassVar[str] = SparseDetectorAngleProjector.pole_revision
    clipping_policy: ClassVar[str] = "independent-profile-windows.no-renormalization.v1"
    boundary_sampling_revision: ClassVar[str] = "inverse-boundary-65.expand-until-clear.v1"
    topology_contract: ClassVar[str] = "fully-panel-contained-connected-local-window.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, CompiledInstrument):
            raise TypeError("instrument must be a CompiledInstrument")
        if not isinstance(self.angle_frame, AngleFrame):
            raise TypeError("angle_frame must be an AngleFrame")
        theta = np.asarray(self.two_theta_bounds_rad)
        phi = np.asarray(self.phi_bin_edges_rad)
        if theta.ndim != 2 or theta.shape[1] != 2 or theta.shape[0] == 0:
            raise ValueError("two_theta_bounds_rad must have shape (profile, 2)")
        profile_count = theta.shape[0]
        if phi.ndim != 2 or phi.shape[0] != profile_count or phi.shape[1] < 4:
            raise ValueError("phi_bin_edges_rad must have shape (profile, bin + 1)")
        theta = _readonly_float(theta, (profile_count, 2), "two_theta_bounds_rad")
        phi = _readonly_float(phi, phi.shape, "phi_bin_edges_rad")
        if (
            np.any(theta[:, 0] < 0.0)
            or np.any(theta[:, 1] > np.pi)
            or np.any(theta[:, 0] >= theta[:, 1])
        ):
            raise ValueError("two-theta profile bounds must be ordered inside [0, pi]")
        if np.any(np.diff(phi, axis=1) <= 0.0):
            raise ValueError("phi-bin edges must increase within each profile")
        if np.any(phi[:, -1] - phi[:, 0] >= 2.0 * np.pi):
            raise ValueError("each local phi profile must span less than one azimuth period")
        rows, columns = self.instrument.detector_shape_rc
        detector_mask = _readonly_bool(
            self.detector_valid_mask,
            (rows, columns),
            "detector_valid_mask",
        )
        profile_mask = _readonly_bool(
            self.profile_bin_valid_mask,
            (profile_count, phi.shape[1] - 1),
            "profile_bin_valid_mask",
        )
        if np.any(np.sum(profile_mask, axis=1) < 3):
            raise ValueError("each local profile must retain at least three bins")
        supplied_weight = np.asarray(self.weight)
        if np.iscomplexobj(supplied_weight):
            raise ValueError("weight must be real")
        weight = np.array(supplied_weight, dtype=np.float64, copy=True, order="C")
        if weight.ndim != 1 or not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
            raise ValueError("weight must be a finite positive vector")
        pixel_count = rows * columns
        profile_bin_count = profile_count * profile_mask.shape[1]
        pixel_index = _readonly_int(
            self.coverage_pixel_index,
            (weight.size,),
            "coverage_pixel_index",
        )
        profile_bin_index = _readonly_int(
            self.coverage_profile_bin_index,
            (weight.size,),
            "coverage_profile_bin_index",
        )
        if (
            np.any(pixel_index < 0)
            or np.any(pixel_index >= pixel_count)
            or np.any(profile_bin_index < 0)
            or np.any(profile_bin_index >= profile_bin_count)
        ):
            raise ValueError("profile coverage indices are outside their frozen arrays")
        if weight.size:
            if not np.all(detector_mask.ravel()[pixel_index]) or not np.all(
                profile_mask.ravel()[profile_bin_index]
            ):
                raise ValueError("profile coverage records may reference only valid mask entries")
            pair = profile_bin_index.astype(np.int64) * pixel_count + pixel_index
            if np.unique(pair).size != pair.size:
                raise ValueError("profile coverage records must be unique per bin and pixel")
            profile_pixel = (profile_bin_index // profile_mask.shape[1]) * pixel_count + pixel_index
            _, compact_profile_pixel = np.unique(profile_pixel, return_inverse=True)
            covered_fraction = np.bincount(compact_profile_pixel, weights=weight)
            if np.any(covered_fraction > 1.0 + _PROFILE_CONSERVATION_TOL):
                maximum = float(np.max(covered_fraction))
                raise ValueError(
                    "one profile assigned more than one pixel mass "
                    f"(maximum fraction {maximum:.17g})"
                )
        bounds = _readonly_int(
            self.profile_pixel_bounds_cr,
            (profile_count, 4),
            "profile_pixel_bounds_cr",
        )
        if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] >= columns):
            raise ValueError("profile column bounds are outside the detector")
        if np.any(bounds[:, 2] < 0) or np.any(bounds[:, 3] >= rows):
            raise ValueError("profile row bounds are outside the detector")
        if np.any(bounds[:, 0] > bounds[:, 1]) or np.any(bounds[:, 2] > bounds[:, 3]):
            raise ValueError("profile pixel bounds must be ordered")
        weight.setflags(write=False)
        object.__setattr__(self, "two_theta_bounds_rad", theta)
        object.__setattr__(self, "phi_bin_edges_rad", phi)
        object.__setattr__(self, "detector_valid_mask", detector_mask)
        object.__setattr__(self, "profile_bin_valid_mask", profile_mask)
        object.__setattr__(self, "coverage_pixel_index", pixel_index)
        object.__setattr__(self, "coverage_profile_bin_index", profile_bin_index)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "profile_pixel_bounds_cr", bounds)
        object.__setattr__(
            self,
            "instrument_fingerprint",
            _version(self.instrument_fingerprint, "instrument_fingerprint"),
        )
        object.__setattr__(self, "cache_key", _version(self.cache_key, "cache_key"))
        expected_fingerprint = _instrument_fingerprint(self.instrument)
        if self.instrument_fingerprint != expected_fingerprint:
            raise ValueError("instrument_fingerprint does not match the frozen instrument")
        expected_key = _profile_cache_key(
            instrument_fingerprint=expected_fingerprint,
            angle_frame=self.angle_frame,
            two_theta_bounds_rad=theta,
            phi_bin_edges_rad=phi,
            detector_mask=detector_mask,
            profile_mask=profile_mask,
        )
        if self.cache_key != expected_key:
            raise ValueError("cache_key does not match the frozen local-profile projector")


@dataclass(frozen=True, slots=True)
class NormalizedAngleProfiles:
    """Independent local profiles formed by reducing detector-pixel mass before division."""

    S: NDArray[np.float64]
    N: NDArray[np.float64]
    I: NDArray[np.float64]  # noqa: E741 - scientific contract uses S, N, I
    valid: NDArray[np.bool_]
    profile_bin_valid_mask: NDArray[np.bool_]
    projector_cache_key: str

    observable_kind: ClassVar[str] = "normalized-local-angle-profiles.v1"
    detector_signal_kind: ClassVar[str] = "nonnegative-detector-pixel-mass.v2"
    detector_normalization_kind: ClassVar[str] = "nonnegative-detector-support-weight.v1"
    input_correction_policy: ClassVar[str] = "no-corrections-declared.v1"
    detector_solid_angle_applied: ClassVar[bool] = False

    def __post_init__(self) -> None:
        supplied = np.asarray(self.S)
        if supplied.ndim != 2 or supplied.shape[0] == 0 or supplied.shape[1] < 3:
            raise ValueError("local profile fields must have shape (profile, bin)")
        shape = supplied.shape
        signal = _readonly_float(self.S, shape, "S")
        normalization = _readonly_float(self.N, shape, "N")
        intensity = _readonly_float(self.I, shape, "I")
        valid = _readonly_bool(self.valid, shape, "valid")
        profile_mask = _readonly_bool(
            self.profile_bin_valid_mask,
            shape,
            "profile_bin_valid_mask",
        )
        if np.any(signal < 0.0) or np.any(normalization < 0.0) or np.any(intensity < 0.0):
            raise ValueError("S, N, and I must be nonnegative")
        expected_valid = profile_mask & (normalization > 0.0)
        expected_intensity = np.zeros(shape, dtype=np.float64)
        np.divide(signal, normalization, out=expected_intensity, where=expected_valid)
        if not np.array_equal(valid, expected_valid) or not np.allclose(
            intensity,
            expected_intensity,
            rtol=3e-15,
            atol=0.0,
        ):
            raise ValueError("local-profile validity and intensity must follow S/N")
        object.__setattr__(self, "S", signal)
        object.__setattr__(self, "N", normalization)
        object.__setattr__(self, "I", intensity)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "profile_bin_valid_mask", profile_mask)
        object.__setattr__(
            self,
            "projector_cache_key",
            _version(self.projector_cache_key, "projector_cache_key"),
        )


@dataclass(frozen=True, slots=True)
class NormalizedAngleField:
    """Canonical finite-bin ``S``, ``N``, and post-reduction ``I=S/N``."""

    S: NDArray[np.float64]
    N: NDArray[np.float64]
    I: NDArray[np.float64]  # noqa: E741 - scientific contract uses S, N, I
    valid: NDArray[np.bool_]
    grid: AngleBinGrid
    angle_bin_valid_mask: NDArray[np.bool_]
    projector_cache_key: str
    detector_mask_excluded_signal: float
    detector_mask_excluded_normalization: float
    angular_lost_signal: float
    angular_lost_normalization: float
    angle_mask_excluded_signal: float
    angle_mask_excluded_normalization: float

    observable_kind: ClassVar[str] = "normalized-finite-bin-angle-field.v1"
    detector_signal_kind: ClassVar[str] = "nonnegative-detector-pixel-mass.v2"
    detector_normalization_kind: ClassVar[str] = "nonnegative-detector-support-weight.v1"
    input_correction_policy: ClassVar[str] = "no-corrections-declared.v1"
    invalid_bin_rule: ClassVar[str] = "normalization-strictly-positive.v1"
    correction_ledger: ClassVar[tuple[str, ...]] = ()
    detector_solid_angle_applied: ClassVar[bool] = False
    unit_area_normalized: ClassVar[bool] = False
    bin_measure: ClassVar[str] = AngleBinGrid.bin_measure
    canonical_order: ClassVar[str] = AngleBinGrid.canonical_order
    unwrap_revision: ClassVar[str] = SparseDetectorAngleProjector.unwrap_revision
    clipping_policy: ClassVar[str] = SparseDetectorAngleProjector.clipping_policy

    def __post_init__(self) -> None:
        if not isinstance(self.grid, AngleBinGrid):
            raise TypeError("grid must be an AngleBinGrid")
        shape = self.grid.shape
        signal = _readonly_float(self.S, shape, "S")
        normalization = _readonly_float(self.N, shape, "N")
        intensity = _readonly_float(self.I, shape, "I")
        valid = _readonly_bool(self.valid, shape, "valid")
        bin_mask = _readonly_bool(
            self.angle_bin_valid_mask,
            shape,
            "angle_bin_valid_mask",
        )
        if np.any(signal < 0.0) or np.any(normalization < 0.0) or np.any(intensity < 0.0):
            raise ValueError("S, N, and I must be nonnegative")
        expected_valid = bin_mask & (normalization > 0.0)
        if not np.array_equal(valid, expected_valid):
            raise ValueError("valid must equal the frozen bin mask with N > 0")
        expected_intensity = np.zeros(shape, dtype=np.float64)
        np.divide(signal, normalization, out=expected_intensity, where=valid)
        if not np.allclose(intensity, expected_intensity, rtol=3e-15, atol=0.0):
            raise ValueError("I must equal S/N after bin reduction and be zero outside valid bins")
        _freeze_nonnegative_losses(self)
        object.__setattr__(self, "S", signal)
        object.__setattr__(self, "N", normalization)
        object.__setattr__(self, "I", intensity)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "angle_bin_valid_mask", bin_mask)
        object.__setattr__(
            self,
            "projector_cache_key",
            _version(self.projector_cache_key, "projector_cache_key"),
        )


@dataclass(frozen=True, slots=True)
class IncreasingPhiAngleField:
    """Display view with raw-chi rows permuted into increasing ``phi``."""

    grid: AngleBinGrid
    S: NDArray[np.float64]
    N: NDArray[np.float64]
    I: NDArray[np.float64]  # noqa: E741 - synchronized view of the named I field
    valid: NDArray[np.bool_]
    angle_bin_valid_mask: NDArray[np.bool_]
    projector_cache_key: str
    detector_mask_excluded_signal: float
    detector_mask_excluded_normalization: float
    angular_lost_signal: float
    angular_lost_normalization: float
    angle_mask_excluded_signal: float
    angle_mask_excluded_normalization: float

    observable_kind: ClassVar[str] = "normalized-finite-bin-increasing-phi-view.v1"
    canonical_order: ClassVar[str] = "increasing-phi-major.two-theta-minor.v1"
    unwrap_revision: ClassVar[str] = SparseDetectorAngleProjector.unwrap_revision
    clipping_policy: ClassVar[str] = SparseDetectorAngleProjector.clipping_policy
    bin_measure: ClassVar[str] = NormalizedAngleField.bin_measure
    detector_signal_kind: ClassVar[str] = NormalizedAngleField.detector_signal_kind
    detector_normalization_kind: ClassVar[str] = NormalizedAngleField.detector_normalization_kind
    input_correction_policy: ClassVar[str] = NormalizedAngleField.input_correction_policy
    invalid_bin_rule: ClassVar[str] = NormalizedAngleField.invalid_bin_rule
    correction_ledger: ClassVar[tuple[str, ...]] = NormalizedAngleField.correction_ledger
    detector_solid_angle_applied: ClassVar[bool] = NormalizedAngleField.detector_solid_angle_applied
    unit_area_normalized: ClassVar[bool] = NormalizedAngleField.unit_area_normalized

    def __post_init__(self) -> None:
        if not isinstance(self.grid, AngleBinGrid):
            raise TypeError("grid must be an AngleBinGrid")
        shape = self.grid.shape
        signal = _readonly_float(self.S, shape, "S")
        normalization = _readonly_float(self.N, shape, "N")
        intensity = _readonly_float(self.I, shape, "I")
        valid = _readonly_bool(self.valid, shape, "valid")
        bin_mask = _readonly_bool(
            self.angle_bin_valid_mask,
            shape,
            "angle_bin_valid_mask",
        )
        if np.any(signal < 0.0) or np.any(normalization < 0.0) or np.any(intensity < 0.0):
            raise ValueError("S, N, and I must be nonnegative")
        expected_valid = bin_mask & (normalization > 0.0)
        expected_intensity = np.zeros(shape, dtype=np.float64)
        np.divide(signal, normalization, out=expected_intensity, where=expected_valid)
        if not np.array_equal(valid, expected_valid) or not np.allclose(
            intensity,
            expected_intensity,
            rtol=3e-15,
            atol=0.0,
        ):
            raise ValueError("increasing-phi validity and I must follow the normalized contract")
        _freeze_nonnegative_losses(self)
        object.__setattr__(self, "S", signal)
        object.__setattr__(self, "N", normalization)
        object.__setattr__(self, "I", intensity)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "angle_bin_valid_mask", bin_mask)
        object.__setattr__(
            self,
            "projector_cache_key",
            _version(self.projector_cache_key, "projector_cache_key"),
        )


def _unwrap_chi(raw_chi_rad: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.array(raw_chi_rad, dtype=np.float64, copy=True)
    for index in range(1, result.size):
        delta = (raw_chi_rad[index] - raw_chi_rad[index - 1] + np.pi) % (2.0 * np.pi) - np.pi
        result[index] = result[index - 1] + delta
    return result


def _polygon_area(polygon: NDArray[np.float64]) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    products = (
        polygon[index, 0] * polygon[(index + 1) % polygon.shape[0], 1]
        - polygon[index, 1] * polygon[(index + 1) % polygon.shape[0], 0]
        for index in range(polygon.shape[0])
    )
    return 0.5 * abs(math.fsum(products))


def _clip_boundary(
    polygon: NDArray[np.float64],
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> NDArray[np.float64]:
    if polygon.shape[0] == 0:
        return polygon

    def inside(point: NDArray[np.float64]) -> bool:
        return bool(point[axis] >= boundary if keep_greater else point[axis] <= boundary)

    output: list[NDArray[np.float64]] = []
    start = polygon[-1]
    start_inside = inside(start)
    for end in polygon:
        end_inside = inside(end)
        if start_inside != end_inside:
            denominator = end[axis] - start[axis]
            if denominator != 0.0:
                fraction = (boundary - start[axis]) / denominator
                intersection = start + fraction * (end - start)
                intersection[axis] = boundary
                output.append(intersection)
        if end_inside:
            output.append(end)
        start = end
        start_inside = end_inside
    if not output:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def _rectangle_overlap_area(
    polygon: NDArray[np.float64],
    theta_lower: float,
    theta_upper: float,
    chi_lower: float,
    chi_upper: float,
) -> float:
    clipped = polygon
    for axis, boundary, keep_greater in (
        (0, theta_lower, True),
        (0, theta_upper, False),
        (1, chi_lower, True),
        (1, chi_upper, False),
    ):
        clipped = _clip_boundary(
            clipped,
            axis=axis,
            boundary=boundary,
            keep_greater=keep_greater,
        )
        if clipped.shape[0] == 0:
            return 0.0
    return _polygon_area(clipped)


def _detector_axis_pole(
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
) -> tuple[float, float, float] | None:
    two_theta = np.array([0.0, np.pi])
    directions = np.stack(
        (angle_frame.direct_beam_lab, -angle_frame.direct_beam_lab),
        axis=0,
    )
    intersections = _intersect_detector_plane(
        np.broadcast_to(angle_frame.origin_lab_m, directions.shape),
        directions,
        instrument,
    )
    indices = np.flatnonzero(intersections.status == ValidityCode.VALID)
    if indices.size == 0:
        return None
    if indices.size != 1:
        raise ValueError("the detector plane has ambiguous direct-axis ownership")
    index = int(indices[0])
    return (
        float(intersections.column_px[index]),
        float(intersections.row_px[index]),
        float(two_theta[index]),
    )


def _pixel_angular_pieces(
    *,
    column: int,
    row: int,
    corner_theta: NDArray[np.float64],
    corner_chi: NDArray[np.float64],
    corner_azimuth_valid: NDArray[np.bool_],
    pole: tuple[float, float, float] | None,
    pole_tolerance_px: float,
) -> tuple[NDArray[np.float64], ...]:
    physical_corners = np.array(
        [
            [column - 0.5, row - 0.5],
            [column + 0.5, row - 0.5],
            [column + 0.5, row + 0.5],
            [column - 0.5, row + 0.5],
        ],
        dtype=np.float64,
    )
    if pole is not None:
        pole_point = np.asarray(pole[:2])
        inside = bool(
            column - 0.5 - pole_tolerance_px <= pole_point[0]
            and pole_point[0] <= column + 0.5 + pole_tolerance_px
            and row - 0.5 - pole_tolerance_px <= pole_point[1]
            and pole_point[1] <= row + 0.5 + pole_tolerance_px
        )
        if inside:
            for axis, lower, upper in (
                (0, column - 0.5, column + 0.5),
                (1, row - 0.5, row + 0.5),
            ):
                if abs(float(pole_point[axis]) - lower) <= pole_tolerance_px:
                    pole_point[axis] = lower
                elif abs(float(pole_point[axis]) - upper) <= pole_tolerance_px:
                    pole_point[axis] = upper
            pieces: list[NDArray[np.float64]] = []
            for index in range(4):
                following = (index + 1) % 4
                first = physical_corners[index] - pole_point
                second = physical_corners[following] - pole_point
                physical_double_area = abs(first[0] * second[1] - first[1] * second[0])
                if physical_double_area <= _POLE_TOL_FACTOR:
                    continue
                if not corner_azimuth_valid[index] or not corner_azimuth_valid[following]:
                    raise ValueError(
                        "a nondegenerate direct-beam fan edge lacks a limiting azimuth"
                    )
                unwrapped = _unwrap_chi(corner_chi[[index, following]])
                piece = np.array(
                    [
                        [pole[2], unwrapped[0]],
                        [corner_theta[index], unwrapped[0]],
                        [corner_theta[following], unwrapped[1]],
                        [pole[2], unwrapped[1]],
                    ],
                    dtype=np.float64,
                )
                if _polygon_area(piece) > 0.0:
                    pieces.append(piece)
            if not pieces:
                raise ValueError("a direct-beam pixel has zero angular support")
            return tuple(pieces)

    if not np.all(corner_azimuth_valid):
        raise ValueError("a detector-pixel corner has undefined azimuth outside the pole tie case")
    pieces = []
    for indices in ((0, 1, 2), (0, 2, 3)):
        index = np.asarray(indices)
        piece = np.column_stack((corner_theta[index], _unwrap_chi(corner_chi[index])))
        if _polygon_area(piece) > 0.0:
            pieces.append(piece)
    if not pieces:
        raise ValueError("a detector pixel has zero angular support")
    return tuple(pieces)


def _pixel_bin_weights(
    pieces: tuple[NDArray[np.float64], ...],
    grid: AngleBinGrid,
) -> tuple[NDArray[np.int64], NDArray[np.float64], float]:
    full_area = math.fsum(_polygon_area(piece) for piece in pieces)
    if not math.isfinite(full_area) or full_area <= 0.0:
        raise ValueError("detector pixel has invalid full angular area")
    theta_edges = grid.two_theta_edges_rad
    chi_edges = grid.chi_raw_edges_rad
    theta_bins = grid.shape[1]
    chi_bins = grid.shape[0]
    chi_step = float(chi_edges[1] - chi_edges[0])
    seam = grid.seam_rad
    overlap_by_bin: dict[int, float] = {}
    for piece in pieces:
        theta_min = float(np.min(piece[:, 0]))
        theta_max = float(np.max(piece[:, 0]))
        theta_start = max(0, int(np.searchsorted(theta_edges, theta_min, side="right") - 1))
        theta_stop = min(theta_bins, int(np.searchsorted(theta_edges, theta_max, side="left")))
        if theta_stop <= theta_start:
            continue
        chi_min = float(np.min(piece[:, 1]))
        chi_max = float(np.max(piece[:, 1]))
        global_chi_start = math.floor((chi_min - seam) / chi_step)
        global_chi_stop = math.floor((chi_max - seam) / chi_step)
        for global_chi in range(global_chi_start, global_chi_stop + 1):
            canonical_chi = global_chi % chi_bins
            period_index = (global_chi - canonical_chi) // chi_bins
            chi_lower = float(chi_edges[canonical_chi] + period_index * 2.0 * np.pi)
            chi_upper = float(chi_edges[canonical_chi + 1] + period_index * 2.0 * np.pi)
            for theta_bin in range(theta_start, theta_stop):
                area = _rectangle_overlap_area(
                    piece,
                    float(theta_edges[theta_bin]),
                    float(theta_edges[theta_bin + 1]),
                    chi_lower,
                    chi_upper,
                )
                if area <= 0.0:
                    continue
                bin_index = canonical_chi * theta_bins + theta_bin
                overlap_by_bin[bin_index] = math.fsum((overlap_by_bin.get(bin_index, 0.0), area))

    ordered_bins = np.array(sorted(overlap_by_bin), dtype=np.int64)
    weights = np.array(
        [overlap_by_bin[int(bin_index)] / full_area for bin_index in ordered_bins],
        dtype=np.float64,
    )
    retained = math.fsum(float(value) for value in weights)
    lost = 1.0 - retained
    if lost < 0.0 and abs(lost) <= _CONSERVATION_TOL:
        lost = 0.0
    if lost < 0.0 or lost > 1.0 + _CONSERVATION_TOL:
        raise ValueError("angular overlap violates the no-renormalization conservation contract")
    return ordered_bins, weights, min(1.0, lost)


def _hash_text(digest: object, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
    digest.update(encoded)


def _hash_array(digest: object, name: str, value: ArrayLike, dtype: str) -> None:
    array_value = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    _hash_text(digest, name)
    _hash_text(digest, str(array_value.shape))
    digest.update(array_value.tobytes())


def _instrument_fingerprint(instrument: CompiledInstrument) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, "detector_angle_instrument_fingerprint.v2")
    _hash_array(
        digest,
        "lab_from_detector.rotation",
        instrument.lab_from_detector.rotation,
        "<f8",
    )
    _hash_array(
        digest,
        "lab_from_detector.translation_m",
        instrument.lab_from_detector.translation_m,
        "<f8",
    )
    _hash_array(digest, "detector_shape_rc", instrument.detector_shape_rc, "<i8")
    _hash_array(
        digest,
        "detector_row_pitch_m",
        (instrument.detector_row_pitch_m,),
        "<f8",
    )
    _hash_array(
        digest,
        "detector_column_pitch_m",
        (instrument.detector_column_pitch_m,),
        "<f8",
    )
    _hash_array(
        digest,
        "detector_reference_coordinate_px",
        instrument.detector_reference_coordinate_px,
        "<f8",
    )
    return f"sha256-{digest.hexdigest()}.v2"


def _cache_key(
    *,
    instrument_fingerprint: str,
    angle_frame: AngleFrame,
    grid: AngleBinGrid,
    detector_mask: NDArray[np.bool_],
    angle_mask: NDArray[np.bool_],
) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, instrument_fingerprint)
    for name, value in (
        ("origin_lab_m", angle_frame.origin_lab_m),
        ("row_down_lab", angle_frame.row_down_lab),
        ("column_right_lab", angle_frame.column_right_lab),
        ("direct_beam_lab", angle_frame.direct_beam_lab),
        ("two_theta_edges_rad", grid.two_theta_edges_rad),
        ("chi_raw_edges_rad", grid.chi_raw_edges_rad),
        ("detector_mask", detector_mask),
        ("angle_mask", angle_mask),
    ):
        dtype = "u1" if np.asarray(value).dtype.kind == "b" else "<f8"
        _hash_array(digest, name, value, dtype)
    for value in (
        angle_frame.revision,
        grid.revision,
        SparseDetectorAngleProjector.projector_revision,
        SparseDetectorAngleProjector.dtype,
        SparseDetectorAngleProjector.sparse_engine_revision,
        SparseDetectorAngleProjector.summation_engine_revision,
        SparseDetectorAngleProjector.polygon_revision,
        SparseDetectorAngleProjector.unwrap_revision,
        SparseDetectorAngleProjector.pole_revision,
        SparseDetectorAngleProjector.tie_policy_revision,
        SparseDetectorAngleProjector.clipping_policy,
        AngleBinGrid.bin_measure,
        AngleBinGrid.canonical_order,
        AngleBinGrid.grid_contract_revision,
    ):
        _hash_text(digest, value)
    _hash_array(
        digest,
        "tile_sizes",
        (
            SparseDetectorAngleProjector.corner_row_tile_size,
            SparseDetectorAngleProjector.coverage_entry_tile_size,
        ),
        "<i8",
    )
    return f"sha256-{digest.hexdigest()}.v1"


def _profile_cache_key(
    *,
    instrument_fingerprint: str,
    angle_frame: AngleFrame,
    two_theta_bounds_rad: NDArray[np.float64],
    phi_bin_edges_rad: NDArray[np.float64],
    detector_mask: NDArray[np.bool_],
    profile_mask: NDArray[np.bool_],
) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, "detector_local_profile_projector.v1")
    _hash_text(digest, instrument_fingerprint)
    for name, value in (
        ("origin_lab_m", angle_frame.origin_lab_m),
        ("row_down_lab", angle_frame.row_down_lab),
        ("column_right_lab", angle_frame.column_right_lab),
        ("direct_beam_lab", angle_frame.direct_beam_lab),
        ("two_theta_bounds_rad", two_theta_bounds_rad),
        ("phi_bin_edges_rad", phi_bin_edges_rad),
        ("detector_mask", detector_mask),
        ("profile_mask", profile_mask),
    ):
        dtype = "u1" if np.asarray(value).dtype.kind == "b" else "<f8"
        _hash_array(digest, name, value, dtype)
    for value in (
        angle_frame.revision,
        SparseDetectorProfileProjector.projector_revision,
        SparseDetectorProfileProjector.polygon_revision,
        SparseDetectorProfileProjector.unwrap_revision,
        SparseDetectorProfileProjector.pole_revision,
        SparseDetectorProfileProjector.clipping_policy,
        SparseDetectorProfileProjector.boundary_sampling_revision,
        SparseDetectorProfileProjector.topology_contract,
    ):
        _hash_text(digest, value)
    return f"sha256-{digest.hexdigest()}.v1"


def _profile_seed_pixel_bounds(
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    two_theta_bounds_rad: NDArray[np.float64],
    phi_bin_edges_rad: NDArray[np.float64],
) -> tuple[int, int, int, int]:
    edge_coordinate = np.linspace(0.0, 1.0, 65)
    theta_lower, theta_upper = (float(value) for value in two_theta_bounds_rad)
    phi_lower = float(phi_bin_edges_rad[0])
    phi_upper = float(phi_bin_edges_rad[-1])
    theta_span = theta_lower + edge_coordinate * (theta_upper - theta_lower)
    phi_span = phi_lower + edge_coordinate * (phi_upper - phi_lower)
    interior_theta, interior_phi = np.meshgrid(theta_span[::4], phi_span[::4])
    theta = np.concatenate(
        (
            np.full(edge_coordinate.shape, theta_lower),
            np.full(edge_coordinate.shape, theta_upper),
            theta_span,
            theta_span,
            interior_theta.ravel(),
        )
    )
    phi = np.concatenate(
        (
            phi_span,
            phi_span,
            np.full(edge_coordinate.shape, phi_lower),
            np.full(edge_coordinate.shape, phi_upper),
            interior_phi.ravel(),
        )
    )
    coordinates = angles_to_detector_coordinates(
        theta,
        phi,
        instrument=instrument,
        angle_frame=angle_frame,
    )
    if not np.all(coordinates.valid):
        raise ValueError(
            "a local angular profile must be fully panel-contained at its topology samples"
        )
    columns = coordinates.column_px
    rows = coordinates.row_px
    detector_rows, detector_columns = instrument.detector_shape_rc
    column_lower = max(0, math.floor(float(np.min(columns)) - 0.5) - 1)
    column_upper = min(
        detector_columns - 1,
        math.ceil(float(np.max(columns)) + 0.5) + 1,
    )
    row_lower = max(0, math.floor(float(np.min(rows)) - 0.5) - 1)
    row_upper = min(
        detector_rows - 1,
        math.ceil(float(np.max(rows)) + 0.5) + 1,
    )
    if column_lower > column_upper or row_lower > row_upper:
        raise ValueError("a local angular profile misses the finite detector")
    return column_lower, column_upper, row_lower, row_upper


def _local_profile_coverage(
    *,
    profile_index: int,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    two_theta_bounds_rad: NDArray[np.float64],
    phi_bin_edges_rad: NDArray[np.float64],
    detector_mask: NDArray[np.bool_],
    profile_mask: NDArray[np.bool_],
    pixel_bounds_cr: tuple[int, int, int, int],
    pole: tuple[float, float, float] | None,
    pole_tolerance_px: float,
) -> tuple[list[int], list[int], list[float], tuple[bool, bool, bool, bool]]:
    column_lower, column_upper, row_lower, row_upper = pixel_bounds_cr
    corner_columns, corner_rows = np.meshgrid(
        np.arange(column_lower, column_upper + 2, dtype=np.float64) - 0.5,
        np.arange(row_lower, row_upper + 2, dtype=np.float64) - 0.5,
    )
    angles = detector_coordinates_to_angles(
        corner_columns,
        corner_rows,
        instrument=instrument,
        angle_frame=angle_frame,
    )
    if not np.all(angles.valid):
        invalid = tuple(sorted(set(angles.status[~angles.valid].ravel())))
        raise ValueError(f"local-profile detector corners are not projectable: {invalid}")
    theta_lower, theta_upper = (float(value) for value in two_theta_bounds_rad)
    bin_count = phi_bin_edges_rad.size - 1
    detector_columns = instrument.detector_shape_rc[1]
    pixel_indices: list[int] = []
    profile_bin_indices: list[int] = []
    weights: list[float] = []
    touched = [False, False, False, False]
    for row in range(row_lower, row_upper + 1):
        local_row = row - row_lower
        for column in range(column_lower, column_upper + 1):
            local_column = column - column_lower
            lattice_indices = (
                (local_row, local_column),
                (local_row, local_column + 1),
                (local_row + 1, local_column + 1),
                (local_row + 1, local_column),
            )
            corner_theta = np.asarray(
                [angles.two_theta_rad[index] for index in lattice_indices],
                dtype=np.float64,
            )
            corner_chi = np.asarray(
                [angles.chi_raw_rad[index] for index in lattice_indices],
                dtype=np.float64,
            )
            corner_azimuth_valid = np.asarray(
                [angles.azimuth_valid[index] for index in lattice_indices],
                dtype=np.bool_,
            )
            pieces = _pixel_angular_pieces(
                column=column,
                row=row,
                corner_theta=corner_theta,
                corner_chi=corner_chi,
                corner_azimuth_valid=corner_azimuth_valid,
                pole=pole,
                pole_tolerance_px=pole_tolerance_px,
            )
            full_area = math.fsum(_polygon_area(piece) for piece in pieces)
            overlap_by_bin: dict[int, float] = {}
            for piece in pieces:
                unwrapped_phi = _raw_chi_to_unwrapped_phi(piece[:, 1])
                if (
                    float(np.max(piece[:, 0])) <= theta_lower
                    or float(np.min(piece[:, 0])) >= theta_upper
                ):
                    continue
                minimum_period = math.floor(
                    (float(phi_bin_edges_rad[0]) - float(np.max(unwrapped_phi))) / (2.0 * np.pi)
                )
                maximum_period = math.ceil(
                    (float(phi_bin_edges_rad[-1]) - float(np.min(unwrapped_phi))) / (2.0 * np.pi)
                )
                for period_index in range(minimum_period, maximum_period + 1):
                    mapped = np.empty(piece.shape, dtype=np.float64)
                    mapped[:, 0] = piece[:, 0]
                    mapped[:, 1] = unwrapped_phi + period_index * 2.0 * np.pi
                    if float(np.max(mapped[:, 1])) <= float(phi_bin_edges_rad[0]) or float(
                        np.min(mapped[:, 1])
                    ) >= float(phi_bin_edges_rad[-1]):
                        continue
                    phi_start = max(
                        0,
                        int(
                            np.searchsorted(
                                phi_bin_edges_rad,
                                np.min(mapped[:, 1]),
                                side="right",
                            )
                            - 1
                        ),
                    )
                    phi_stop = min(
                        bin_count,
                        int(
                            np.searchsorted(
                                phi_bin_edges_rad,
                                np.max(mapped[:, 1]),
                                side="left",
                            )
                        ),
                    )
                    for phi_bin in range(phi_start, phi_stop):
                        if not profile_mask[phi_bin]:
                            continue
                        area = _rectangle_overlap_area(
                            mapped,
                            theta_lower,
                            theta_upper,
                            float(phi_bin_edges_rad[phi_bin]),
                            float(phi_bin_edges_rad[phi_bin + 1]),
                        )
                        if area > 0.0:
                            overlap_by_bin[phi_bin] = math.fsum(
                                (overlap_by_bin.get(phi_bin, 0.0), area)
                            )
            if not overlap_by_bin:
                continue
            if column == column_lower:
                touched[0] = True
            if column == column_upper:
                touched[1] = True
            if row == row_lower:
                touched[2] = True
            if row == row_upper:
                touched[3] = True
            if not detector_mask[row, column]:
                continue
            pixel = row * detector_columns + column
            for phi_bin in sorted(overlap_by_bin):
                fraction = overlap_by_bin[phi_bin] / full_area
                if fraction <= 0.0:
                    continue
                pixel_indices.append(pixel)
                profile_bin_indices.append(profile_index * bin_count + phi_bin)
                weights.append(fraction)
    return pixel_indices, profile_bin_indices, weights, tuple(touched)


def compile_detector_profile_projector(
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    two_theta_bounds_rad: ArrayLike,
    phi_bin_edges_rad: ArrayLike,
    detector_valid_mask: ArrayLike | None = None,
    profile_bin_valid_mask: ArrayLike | None = None,
) -> SparseDetectorProfileProjector:
    """Compile physical pixels for fully panel-contained, connected local angle windows."""

    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be a CompiledInstrument")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")
    theta = np.asarray(two_theta_bounds_rad, dtype=np.float64)
    phi = np.asarray(phi_bin_edges_rad, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != 2 or theta.shape[0] == 0:
        raise ValueError("two_theta_bounds_rad must have shape (profile, 2)")
    if phi.ndim != 2 or phi.shape[0] != theta.shape[0] or phi.shape[1] < 4:
        raise ValueError("phi_bin_edges_rad must have shape (profile, bin + 1)")
    if np.any(phi[:, -1] - phi[:, 0] >= 2.0 * np.pi):
        raise ValueError("each local phi profile must span less than one azimuth period")
    rows, columns = instrument.detector_shape_rc
    detector_mask = (
        np.ones((rows, columns), dtype=np.bool_)
        if detector_valid_mask is None
        else _readonly_bool(detector_valid_mask, (rows, columns), "detector_valid_mask")
    )
    profile_mask = (
        np.ones((theta.shape[0], phi.shape[1] - 1), dtype=np.bool_)
        if profile_bin_valid_mask is None
        else _readonly_bool(
            profile_bin_valid_mask,
            (theta.shape[0], phi.shape[1] - 1),
            "profile_bin_valid_mask",
        )
    )
    pole = _detector_axis_pole(instrument, angle_frame)
    pole_scale = max(
        1.0,
        float(rows),
        float(columns),
        0.0 if pole is None else abs(pole[0]),
        0.0 if pole is None else abs(pole[1]),
    )
    pole_tolerance = _POLE_TOL_FACTOR * pole_scale
    coverage_pixel: list[int] = []
    coverage_profile_bin: list[int] = []
    coverage_weight: list[float] = []
    bounds = np.empty((theta.shape[0], 4), dtype=np.int64)
    for profile_index in range(theta.shape[0]):
        pixel_bounds = _profile_seed_pixel_bounds(
            instrument=instrument,
            angle_frame=angle_frame,
            two_theta_bounds_rad=theta[profile_index],
            phi_bin_edges_rad=phi[profile_index],
        )
        expansion_px = 2
        expansion_limit = math.ceil(math.log2(max(rows, columns))) + 2
        for _ in range(expansion_limit):
            local_pixel, local_bin, local_weight, touched = _local_profile_coverage(
                profile_index=profile_index,
                instrument=instrument,
                angle_frame=angle_frame,
                two_theta_bounds_rad=theta[profile_index],
                phi_bin_edges_rad=phi[profile_index],
                detector_mask=detector_mask,
                profile_mask=profile_mask[profile_index],
                pixel_bounds_cr=pixel_bounds,
                pole=pole,
                pole_tolerance_px=pole_tolerance,
            )
            column_lower, column_upper, row_lower, row_upper = pixel_bounds
            expand = (
                touched[0] and column_lower > 0,
                touched[1] and column_upper < columns - 1,
                touched[2] and row_lower > 0,
                touched[3] and row_upper < rows - 1,
            )
            touches_panel_edge = (
                touched[0] and column_lower == 0,
                touched[1] and column_upper == columns - 1,
                touched[2] and row_lower == 0,
                touched[3] and row_upper == rows - 1,
            )
            if any(touches_panel_edge):
                raise ValueError(
                    "a fully panel-contained local angular profile may not touch "
                    "the physical detector edge"
                )
            if not any(expand):
                break
            pixel_bounds = (
                max(0, column_lower - (expansion_px if expand[0] else 0)),
                min(columns - 1, column_upper + (expansion_px if expand[1] else 0)),
                max(0, row_lower - (expansion_px if expand[2] else 0)),
                min(rows - 1, row_upper + (expansion_px if expand[3] else 0)),
            )
            expansion_px *= 2
        else:
            raise RuntimeError("local-profile crop did not converge to a clear pixel boundary")
        if not local_weight:
            raise ValueError("a local angular profile has no valid detector-pixel support")
        coverage_pixel.extend(local_pixel)
        coverage_profile_bin.extend(local_bin)
        coverage_weight.extend(local_weight)
        bounds[profile_index] = pixel_bounds
    instrument_key = _instrument_fingerprint(instrument)
    return SparseDetectorProfileProjector(
        instrument=instrument,
        angle_frame=angle_frame,
        two_theta_bounds_rad=theta,
        phi_bin_edges_rad=phi,
        detector_valid_mask=detector_mask,
        profile_bin_valid_mask=profile_mask,
        coverage_pixel_index=np.asarray(coverage_pixel, dtype=np.int64),
        coverage_profile_bin_index=np.asarray(coverage_profile_bin, dtype=np.int64),
        weight=np.asarray(coverage_weight, dtype=np.float64),
        profile_pixel_bounds_cr=bounds,
        instrument_fingerprint=instrument_key,
        cache_key=_profile_cache_key(
            instrument_fingerprint=instrument_key,
            angle_frame=angle_frame,
            two_theta_bounds_rad=theta,
            phi_bin_edges_rad=phi,
            detector_mask=detector_mask,
            profile_mask=profile_mask,
        ),
    )


def compile_detector_angle_projector(
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    grid: AngleBinGrid,
    detector_valid_mask: ArrayLike | None = None,
    angle_bin_valid_mask: ArrayLike | None = None,
) -> SparseDetectorAngleProjector:
    """Compile physical detector pixels into canonical finite angle bins."""

    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be a CompiledInstrument")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")
    if not isinstance(grid, AngleBinGrid):
        raise TypeError("grid must be an AngleBinGrid")
    rows, columns = instrument.detector_shape_rc
    detector_mask = (
        np.ones((rows, columns), dtype=np.bool_)
        if detector_valid_mask is None
        else _readonly_bool(detector_valid_mask, (rows, columns), "detector_valid_mask")
    )
    angle_mask = (
        np.ones(grid.shape, dtype=np.bool_)
        if angle_bin_valid_mask is None
        else _readonly_bool(angle_bin_valid_mask, grid.shape, "angle_bin_valid_mask")
    )
    pole = _detector_axis_pole(instrument, angle_frame)
    pole_scale = max(
        1.0,
        float(rows),
        float(columns),
        0.0 if pole is None else abs(pole[0]),
        0.0 if pole is None else abs(pole[1]),
    )
    pole_tolerance = _POLE_TOL_FACTOR * pole_scale

    coverage_pixel = array("q")
    coverage_bin = array("q")
    coverage_weight = array("d")
    lost_support = np.zeros((rows, columns), dtype=np.float64)
    column_edges = np.arange(columns + 1, dtype=np.float64) - 0.5

    for row_start in range(0, rows, SparseDetectorAngleProjector.corner_row_tile_size):
        row_stop = min(rows, row_start + SparseDetectorAngleProjector.corner_row_tile_size)
        row_edges = np.arange(row_start, row_stop + 1, dtype=np.float64) - 0.5
        corner_columns, corner_rows = np.meshgrid(column_edges, row_edges)
        angles = detector_coordinates_to_angles(
            corner_columns,
            corner_rows,
            instrument=instrument,
            angle_frame=angle_frame,
        )
        if not np.all(angles.valid):
            invalid = tuple(sorted(set(angles.status[~angles.valid].ravel())))
            raise ValueError(f"detector physical corners are not projectable: {invalid}")
        for row in range(row_start, row_stop):
            local_row = row - row_start
            for column in range(columns):
                pixel = row * columns + column
                lattice_indices = (
                    (local_row, column),
                    (local_row, column + 1),
                    (local_row + 1, column + 1),
                    (local_row + 1, column),
                )
                corner_theta = np.array(
                    [angles.two_theta_rad[index] for index in lattice_indices],
                    dtype=np.float64,
                )
                corner_chi = np.array(
                    [angles.chi_raw_rad[index] for index in lattice_indices],
                    dtype=np.float64,
                )
                corner_azimuth_valid = np.array(
                    [angles.azimuth_valid[index] for index in lattice_indices],
                    dtype=np.bool_,
                )
                pieces = _pixel_angular_pieces(
                    column=column,
                    row=row,
                    corner_theta=corner_theta,
                    corner_chi=corner_chi,
                    corner_azimuth_valid=corner_azimuth_valid,
                    pole=pole,
                    pole_tolerance_px=pole_tolerance,
                )
                bins, weights, lost = _pixel_bin_weights(pieces, grid)
                coverage_pixel.extend([pixel] * bins.size)
                coverage_bin.extend(int(value) for value in bins)
                coverage_weight.extend(float(value) for value in weights)
                lost_support[row, column] = lost
    pixel_index = np.array(coverage_pixel, dtype=np.int64)
    bin_index = np.array(coverage_bin, dtype=np.int64)
    weight = np.array(coverage_weight, dtype=np.float64)
    instrument_key = _instrument_fingerprint(instrument)
    return SparseDetectorAngleProjector(
        instrument=instrument,
        angle_frame=angle_frame,
        grid=grid,
        detector_valid_mask=detector_mask,
        angle_bin_valid_mask=angle_mask,
        coverage_pixel_index=pixel_index,
        coverage_bin_index=bin_index,
        weight=weight,
        lost_support=lost_support,
        instrument_fingerprint=instrument_key,
        cache_key=_cache_key(
            instrument_fingerprint=instrument_key,
            angle_frame=angle_frame,
            grid=grid,
            detector_mask=detector_mask,
            angle_mask=angle_mask,
        ),
    )


def _detector_field(value: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[np.float64]:
    result = _readonly_float(value, shape, name)
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative; signed input needs a separate contract")
    return result


def _apply_projector(
    projector: SparseDetectorAngleProjector,
    detector_field: NDArray[np.float64],
) -> NDArray[np.float64]:
    output = np.zeros(math.prod(projector.grid.shape), dtype=np.float64)
    flat = detector_field.ravel()
    for start in range(0, projector.weight.size, projector.coverage_entry_tile_size):
        stop = min(projector.weight.size, start + projector.coverage_entry_tile_size)
        contribution = (
            projector.weight[start:stop] * flat[projector.coverage_pixel_index[start:stop]]
        )
        output += np.bincount(
            projector.coverage_bin_index[start:stop],
            weights=contribution,
            minlength=output.size,
        )
    return output.reshape(projector.grid.shape)


def project_detector_profiles(
    projector: SparseDetectorProfileProjector,
    detector_signal: ArrayLike,
    detector_normalization: ArrayLike | None = None,
) -> NormalizedAngleProfiles:
    """Reduce nonnegative detector-pixel mass into local profiles, then form ``I=S/N``."""

    if not isinstance(projector, SparseDetectorProfileProjector):
        raise TypeError("projector must be a SparseDetectorProfileProjector")
    detector_shape = projector.instrument.detector_shape_rc
    signal = np.asarray(detector_signal)
    if (
        signal.shape != detector_shape
        or not np.issubdtype(signal.dtype, np.number)
        or np.iscomplexobj(signal)
        or not np.all(np.isfinite(signal))
        or np.any(signal < 0.0)
    ):
        raise ValueError(
            f"detector_signal must be a finite nonnegative numeric array with shape {detector_shape}"
        )
    flat_signal = signal.ravel()
    output_size = int(np.prod(projector.profile_bin_valid_mask.shape))
    signal_bins = np.bincount(
        projector.coverage_profile_bin_index,
        weights=projector.weight * flat_signal[projector.coverage_pixel_index],
        minlength=output_size,
    ).reshape(projector.profile_bin_valid_mask.shape)
    normalization_weight = projector.weight
    if detector_normalization is not None:
        normalization = np.asarray(detector_normalization)
        if (
            normalization.shape != detector_shape
            or not np.issubdtype(normalization.dtype, np.number)
            or np.iscomplexobj(normalization)
            or not np.all(np.isfinite(normalization))
            or np.any(normalization < 0.0)
        ):
            raise ValueError(
                "detector_normalization must be a finite nonnegative numeric array "
                f"with shape {detector_shape}"
            )
        flat_normalization = normalization.ravel()
        normalization_weight = projector.weight * flat_normalization[projector.coverage_pixel_index]
    normalization_bins = np.bincount(
        projector.coverage_profile_bin_index,
        weights=normalization_weight,
        minlength=output_size,
    ).reshape(projector.profile_bin_valid_mask.shape)
    valid = projector.profile_bin_valid_mask & (normalization_bins > 0.0)
    intensity = np.zeros(signal_bins.shape, dtype=np.float64)
    np.divide(signal_bins, normalization_bins, out=intensity, where=valid)
    return NormalizedAngleProfiles(
        S=signal_bins,
        N=normalization_bins,
        I=intensity,
        valid=valid,
        profile_bin_valid_mask=projector.profile_bin_valid_mask,
        projector_cache_key=projector.cache_key,
    )


def _nonnegative_fsum(values: NDArray[np.float64]) -> float:
    result = math.fsum(float(value) for value in values.ravel())
    if result < 0.0 and abs(result) <= _CONSERVATION_TOL:
        return 0.0
    return result


def project_normalized_angle_field(
    projector: SparseDetectorAngleProjector,
    detector_signal: ArrayLike,
    detector_normalization: ArrayLike,
) -> NormalizedAngleField:
    """Apply ``M`` to uncorrected nonnegative ``s`` and ``n``, then form ``I=S/N``."""

    if not isinstance(projector, SparseDetectorAngleProjector):
        raise TypeError("projector must be a SparseDetectorAngleProjector")
    detector_shape_rc = projector.instrument.detector_shape_rc
    signal = _detector_field(detector_signal, detector_shape_rc, "detector_signal")
    normalization = _detector_field(
        detector_normalization,
        detector_shape_rc,
        "detector_normalization",
    )
    detector_mask = projector.detector_valid_mask
    masked_signal = np.where(detector_mask, signal, 0.0)
    masked_normalization = np.where(detector_mask, normalization, 0.0)
    detector_mask_excluded_signal = _nonnegative_fsum(signal[~detector_mask])
    detector_mask_excluded_normalization = _nonnegative_fsum(normalization[~detector_mask])
    angular_lost_signal = _nonnegative_fsum(projector.lost_support * masked_signal)
    angular_lost_normalization = _nonnegative_fsum(projector.lost_support * masked_normalization)

    signal_bins = _apply_projector(projector, masked_signal)
    normalization_bins = _apply_projector(projector, masked_normalization)
    angle_mask = projector.angle_bin_valid_mask
    angle_mask_excluded_signal = _nonnegative_fsum(signal_bins[~angle_mask])
    angle_mask_excluded_normalization = _nonnegative_fsum(normalization_bins[~angle_mask])
    signal_bins[~angle_mask] = 0.0
    normalization_bins[~angle_mask] = 0.0
    valid = angle_mask & (normalization_bins > 0.0)
    intensity = np.zeros(projector.grid.shape, dtype=np.float64)
    np.divide(signal_bins, normalization_bins, out=intensity, where=valid)
    return NormalizedAngleField(
        S=signal_bins,
        N=normalization_bins,
        I=intensity,
        valid=valid,
        grid=projector.grid,
        angle_bin_valid_mask=angle_mask,
        projector_cache_key=projector.cache_key,
        detector_mask_excluded_signal=detector_mask_excluded_signal,
        detector_mask_excluded_normalization=detector_mask_excluded_normalization,
        angular_lost_signal=angular_lost_signal,
        angular_lost_normalization=angular_lost_normalization,
        angle_mask_excluded_signal=angle_mask_excluded_signal,
        angle_mask_excluded_normalization=angle_mask_excluded_normalization,
    )


def to_increasing_phi(field: NormalizedAngleField) -> IncreasingPhiAngleField:
    """Apply one identical row permutation to the phi axis, fields, and validity mask."""

    if not isinstance(field, NormalizedAngleField):
        raise TypeError("field must be a NormalizedAngleField")
    grid = field.grid
    permutation = _raw_chi_to_phi_permutation(grid.shape[0])
    return IncreasingPhiAngleField(
        grid=grid,
        S=field.S[permutation],
        N=field.N[permutation],
        I=field.I[permutation],
        valid=field.valid[permutation],
        angle_bin_valid_mask=field.angle_bin_valid_mask[permutation],
        projector_cache_key=field.projector_cache_key,
        detector_mask_excluded_signal=field.detector_mask_excluded_signal,
        detector_mask_excluded_normalization=field.detector_mask_excluded_normalization,
        angular_lost_signal=field.angular_lost_signal,
        angular_lost_normalization=field.angular_lost_normalization,
        angle_mask_excluded_signal=field.angle_mask_excluded_signal,
        angle_mask_excluded_normalization=field.angle_mask_excluded_normalization,
    )
