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
    detector_coordinates_to_angles,
)
from rasim_next.geometry.detector import _intersect_detector_plane
from rasim_next.geometry.instrument import CompiledInstrument

_FLOAT_EPS = np.finfo(np.float64).eps
_GRID_TOL = 128.0 * _FLOAT_EPS
_POLE_TOL_FACTOR = 128.0 * _FLOAT_EPS
_CONSERVATION_TOL = 3.0e-11
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
    detector_signal_kind: ClassVar[str] = "nonnegative-deposited-detector-signal.v1"
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
    for name in (
        "lab_from_goniometer",
        "lab_from_sample",
        "sample_from_crystal",
        "lab_from_crystal",
        "lab_from_detector",
    ):
        transform = getattr(instrument, name)
        _hash_array(digest, f"{name}.rotation", transform.rotation, "<f8")
        _hash_array(digest, f"{name}.translation_m", transform.translation_m, "<f8")
    _hash_array(digest, "detector_shape_rc", instrument.detector_shape_rc, "<i8")
    _hash_array(
        digest,
        "detector_calibration",
        (
            instrument.detector_row_pitch_m,
            instrument.detector_column_pitch_m,
            *instrument.detector_reference_coordinate_px,
            instrument.sample_width_m,
            instrument.sample_length_m,
            instrument.film_thickness_A,
        ),
        "<f8",
    )
    return f"sha256-{digest.hexdigest()}.v1"


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
