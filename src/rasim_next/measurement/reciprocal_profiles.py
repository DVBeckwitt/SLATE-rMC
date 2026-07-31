"""Material-neutral reciprocal-rod regions and finite-bin sample reduction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _finite_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    array = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class LayeredReciprocalFrame:
    """Map sample-frame wavevectors into one declared layered reciprocal frame."""

    reciprocal_basis_Ainv: FloatArray
    sample_from_crystal_rotation: FloatArray
    axial_basis_index: int = 2

    def __post_init__(self) -> None:
        basis = _finite_array(self.reciprocal_basis_Ainv, (3, 3), "reciprocal_basis_Ainv")
        rotation = _finite_array(
            self.sample_from_crystal_rotation,
            (3, 3),
            "sample_from_crystal_rotation",
        )
        if abs(float(np.linalg.det(basis))) <= np.finfo(np.float64).tiny:
            raise ValueError("reciprocal_basis_Ainv must be nonsingular")
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), rtol=0.0, atol=2.0e-12
        ) or not math.isclose(
            float(np.linalg.det(rotation)),
            1.0,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        ):
            raise ValueError("sample_from_crystal_rotation must be a proper rotation")
        index = self.axial_basis_index
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("axial_basis_index must be an integer")
        if int(index) not in {0, 1, 2}:
            raise ValueError("axial_basis_index must be 0, 1, or 2")
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)
        object.__setattr__(self, "sample_from_crystal_rotation", rotation)
        object.__setattr__(self, "axial_basis_index", int(index))

    def coordinates(self, q_sample_Ainv: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Return radial reciprocal distance and fractional axial coordinate."""

        supplied = np.asarray(q_sample_Ainv)
        if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
            raise ValueError("q_sample_Ainv must be real")
        q_sample = np.asarray(supplied.real, dtype=np.float64)
        if q_sample.ndim < 1 or q_sample.shape[-1] != 3 or not np.all(np.isfinite(q_sample)):
            raise ValueError("q_sample_Ainv must end in three finite Cartesian components")
        q_crystal = q_sample @ self.sample_from_crystal_rotation
        axial_vector = self.reciprocal_basis_Ainv[:, self.axial_basis_index]
        axial_axis = axial_vector / np.linalg.norm(axial_vector)
        normal_component = q_crystal @ axial_axis
        radial = q_crystal - normal_component[..., None] * axial_axis
        hkl = q_crystal @ np.linalg.inv(self.reciprocal_basis_Ainv).T
        radial_coordinate = np.asarray(np.linalg.norm(radial, axis=-1), dtype=np.float64)
        axial_coordinate = np.asarray(hkl[..., self.axial_basis_index], dtype=np.float64)
        radial_coordinate.setflags(write=False)
        axial_coordinate.setflags(write=False)
        return radial_coordinate, axial_coordinate

    def rod_radial_coordinate_Ainv(self, reciprocal_indices: ArrayLike) -> float:
        """Return one rod's distance from the declared reciprocal texture axis."""

        indices = np.asarray(reciprocal_indices)
        if np.iscomplexobj(indices) and np.any(indices.imag != 0.0):
            raise ValueError("reciprocal_indices must be real")
        values = np.asarray(indices.real, dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("reciprocal_indices must contain three finite values")
        if values[self.axial_basis_index] != 0.0:
            raise ValueError("a rod radial coordinate requires zero axial index")
        q_crystal = self.reciprocal_basis_Ainv @ values
        axial_vector = self.reciprocal_basis_Ainv[:, self.axial_basis_index]
        axial_axis = axial_vector / np.linalg.norm(axial_vector)
        radial = q_crystal - axial_axis * float(q_crystal @ axial_axis)
        return float(np.linalg.norm(radial))


@dataclass(frozen=True, slots=True)
class ReciprocalProfileRegion:
    """One radial reciprocal-space band, axial binning, and detector-side interval."""

    identity: str
    qr_center_Ainv: float
    qr_half_width_Ainv: float
    axial_bin_edges: FloatArray
    detector_column_interval_px: tuple[float, float]
    sideband_gap_Ainv: float
    sideband_width_Ainv: float

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError("identity must be a nonempty string")
        scalars = np.asarray(
            (
                self.qr_center_Ainv,
                self.qr_half_width_Ainv,
                self.sideband_gap_Ainv,
                self.sideband_width_Ainv,
            ),
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(scalars))
            or self.qr_center_Ainv < 0.0
            or self.qr_half_width_Ainv <= 0.0
            or self.sideband_gap_Ainv < 0.0
            or self.sideband_width_Ainv <= 0.0
        ):
            raise ValueError("radial bounds must be finite and nonnegative with positive widths")
        edges = np.array(self.axial_bin_edges, dtype=np.float64, copy=True, order="C")
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("axial_bin_edges must be a strictly increasing finite vector")
        interval = tuple(self.detector_column_interval_px)
        if len(interval) != 2:
            raise ValueError("detector_column_interval_px must contain two bounds")
        lower, upper = (float(value) for value in interval)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("detector column bounds must be finite and ordered")
        edges.setflags(write=False)
        object.__setattr__(self, "qr_center_Ainv", float(self.qr_center_Ainv))
        object.__setattr__(self, "qr_half_width_Ainv", float(self.qr_half_width_Ainv))
        object.__setattr__(self, "sideband_gap_Ainv", float(self.sideband_gap_Ainv))
        object.__setattr__(self, "sideband_width_Ainv", float(self.sideband_width_Ainv))
        object.__setattr__(self, "axial_bin_edges", edges)
        object.__setattr__(self, "detector_column_interval_px", (lower, upper))

    @property
    def bin_count(self) -> int:
        return int(self.axial_bin_edges.size - 1)

    @property
    def axial_bin_centers(self) -> FloatArray:
        centers = 0.5 * (self.axial_bin_edges[:-1] + self.axial_bin_edges[1:])
        centers.setflags(write=False)
        return centers


@dataclass(frozen=True, slots=True)
class ReciprocalProfileMembership:
    """Per-sample signal and sideband bin identities; ``-1`` means outside."""

    signal_bin_index: IntArray
    sideband_bin_index: IntArray

    def __post_init__(self) -> None:
        signal = np.array(self.signal_bin_index, dtype=np.int64, copy=True, order="C")
        sideband = np.array(self.sideband_bin_index, dtype=np.int64, copy=True, order="C")
        if signal.shape != sideband.shape or np.any(signal < -1) or np.any(sideband < -1):
            raise ValueError("signal and sideband bin identities must share a shape and be >= -1")
        signal.setflags(write=False)
        sideband.setflags(write=False)
        object.__setattr__(self, "signal_bin_index", signal)
        object.__setattr__(self, "sideband_bin_index", sideband)


def _axial_bin_indices(axial: FloatArray, edges: FloatArray) -> IntArray:
    indices = np.searchsorted(edges, axial, side="right") - 1
    indices = np.asarray(indices, dtype=np.int64)
    indices = np.where(axial == edges[-1], edges.size - 2, indices)
    return np.where((indices >= 0) & (indices < edges.size - 1), indices, -1)


def reciprocal_profile_membership(
    qr_Ainv: ArrayLike,
    axial_coordinate: ArrayLike,
    detector_column_px: ArrayLike,
    valid: ArrayLike,
    *,
    region: ReciprocalProfileRegion,
) -> ReciprocalProfileMembership:
    """Assign continuous or detector-native samples to one declared profile and sidebands."""

    if not isinstance(region, ReciprocalProfileRegion):
        raise TypeError("region must be a ReciprocalProfileRegion")
    qr, axial, column, accepted = np.broadcast_arrays(
        np.asarray(qr_Ainv, dtype=np.float64),
        np.asarray(axial_coordinate, dtype=np.float64),
        np.asarray(detector_column_px, dtype=np.float64),
        np.asarray(valid, dtype=np.bool_),
    )
    finite = accepted & np.isfinite(qr) & np.isfinite(axial) & np.isfinite(column) & (qr >= 0.0)
    lower_column, upper_column = region.detector_column_interval_px
    branch = finite & (column >= lower_column) & (column < upper_column)
    axial_bin = _axial_bin_indices(axial, region.axial_bin_edges)
    branch &= axial_bin >= 0

    signal_lower = max(0.0, region.qr_center_Ainv - region.qr_half_width_Ainv)
    signal_upper = region.qr_center_Ainv + region.qr_half_width_Ainv
    signal = branch & (qr >= signal_lower) & (qr <= signal_upper)

    inner_upper = region.qr_center_Ainv - region.qr_half_width_Ainv - region.sideband_gap_Ainv
    inner_lower = inner_upper - region.sideband_width_Ainv
    outer_lower = region.qr_center_Ainv + region.qr_half_width_Ainv + region.sideband_gap_Ainv
    outer_upper = outer_lower + region.sideband_width_Ainv
    sideband = branch & (
        ((inner_upper > 0.0) & (qr >= max(0.0, inner_lower)) & (qr <= inner_upper))
        | ((qr >= outer_lower) & (qr <= outer_upper))
    )
    return ReciprocalProfileMembership(
        signal_bin_index=np.where(signal, axial_bin, -1).astype(np.int64, copy=False),
        sideband_bin_index=np.where(sideband, axial_bin, -1).astype(np.int64, copy=False),
    )


@dataclass(frozen=True, slots=True)
class BinnedSampleIntegral:
    """Signal and integration measure accumulated independently in finite bins."""

    signal_sum: FloatArray
    measure_sum: FloatArray

    def __post_init__(self) -> None:
        signal = np.array(self.signal_sum, dtype=np.float64, copy=True, order="C")
        measure = np.array(self.measure_sum, dtype=np.float64, copy=True, order="C")
        if (
            signal.ndim != 1
            or signal.shape != measure.shape
            or signal.size == 0
            or not np.all(np.isfinite(signal))
            or not np.all(np.isfinite(measure))
            or np.any(measure < 0.0)
        ):
            raise ValueError("signal_sum and measure_sum must be finite equal-length vectors")
        signal.setflags(write=False)
        measure.setflags(write=False)
        object.__setattr__(self, "signal_sum", signal)
        object.__setattr__(self, "measure_sum", measure)

    @property
    def mean(self) -> FloatArray:
        mean = np.divide(
            self.signal_sum,
            self.measure_sum,
            out=np.full_like(self.signal_sum, np.nan),
            where=self.measure_sum > 0.0,
        )
        mean.setflags(write=False)
        return mean


def accumulate_binned_samples(
    *,
    values: ArrayLike,
    measure_weights: ArrayLike,
    bin_index: ArrayLike,
    bin_count: int,
) -> BinnedSampleIntegral:
    """Accumulate sampled density and measure without pointwise pre-normalization."""

    if isinstance(bin_count, bool) or not isinstance(bin_count, (int, np.integer)):
        raise TypeError("bin_count must be an integer")
    count = int(bin_count)
    if count < 1:
        raise ValueError("bin_count must be positive")
    value, weight, index = np.broadcast_arrays(
        np.asarray(values, dtype=np.float64),
        np.asarray(measure_weights, dtype=np.float64),
        np.asarray(bin_index, dtype=np.int64),
    )
    if (
        not np.all(np.isfinite(value))
        or not np.all(np.isfinite(weight))
        or np.any(weight < 0.0)
        or np.any((index < -1) | (index >= count))
    ):
        raise ValueError("sample values, weights, or bin identities are invalid")
    selected = index >= 0
    signal_sum = np.bincount(
        index[selected],
        weights=value[selected] * weight[selected],
        minlength=count,
    ).astype(np.float64, copy=False)
    measure_sum = np.bincount(
        index[selected],
        weights=weight[selected],
        minlength=count,
    ).astype(np.float64, copy=False)
    return BinnedSampleIntegral(signal_sum=signal_sum, measure_sum=measure_sum)
