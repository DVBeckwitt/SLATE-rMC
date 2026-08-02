"""Native-pixel memberships for matched angular and reciprocal observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

IntArray = NDArray[np.int64]


def _interval(value: tuple[float, float], name: str) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value tuple")
    lower, upper = map(float, value)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"{name} must contain increasing finite values")
    return lower, upper


def _edges(value: ArrayLike, name: str) -> NDArray[np.float64]:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    edges = np.array(supplied.real, dtype=np.float64, copy=True)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    edges.setflags(write=False)
    return edges


def _overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return max(first[0], second[0]) < min(first[1], second[1])


@dataclass(frozen=True, slots=True)
class SpecularAngularProfileRegion:
    """One m=0 profile binned in 2theta with explicit phi anchors."""

    identity: str
    two_theta_bin_edges_rad: ArrayLike
    phi_signal_interval_rad: tuple[float, float]
    phi_background_intervals_rad: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError("identity must be a nonempty string")
        edges = _edges(self.two_theta_bin_edges_rad, "two_theta_bin_edges_rad")
        signal = _interval(self.phi_signal_interval_rad, "phi_signal_interval_rad")
        background = tuple(
            _interval(interval, f"phi_background_intervals_rad[{index}]")
            for index, interval in enumerate(self.phi_background_intervals_rad)
        )
        if not background:
            raise ValueError("phi_background_intervals_rad must not be empty")
        if any(_overlap(signal, interval) for interval in background):
            raise ValueError("phi background intervals must be disjoint from the signal interval")
        if any(
            _overlap(first, second)
            for index, first in enumerate(background)
            for second in background[index + 1 :]
        ):
            raise ValueError("phi background intervals must be mutually disjoint")
        object.__setattr__(self, "two_theta_bin_edges_rad", edges)
        object.__setattr__(self, "phi_signal_interval_rad", signal)
        object.__setattr__(self, "phi_background_intervals_rad", background)

    @property
    def bin_count(self) -> int:
        return int(np.asarray(self.two_theta_bin_edges_rad).size - 1)


@dataclass(frozen=True, slots=True)
class OffSpecularBand:
    """One named Qr signal interval within a joint background layout."""

    identity: str
    qr_interval_Ainv: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError("identity must be a nonempty string")
        object.__setattr__(
            self,
            "qr_interval_Ainv",
            _interval(self.qr_interval_Ainv, "qr_interval_Ainv"),
        )


@dataclass(frozen=True, slots=True)
class OffSpecularBandLayout:
    """Joint Qr/L observation with globally disjoint signal and anchor bands."""

    axial_bin_edges: ArrayLike
    detector_column_interval_px: tuple[float, float]
    signal_bands: tuple[OffSpecularBand, ...]
    background_intervals_Ainv: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        edges = _edges(self.axial_bin_edges, "axial_bin_edges")
        columns = _interval(self.detector_column_interval_px, "detector_column_interval_px")
        signals = tuple(self.signal_bands)
        if not signals or not all(isinstance(item, OffSpecularBand) for item in signals):
            raise TypeError("signal_bands must contain OffSpecularBand values")
        identities = tuple(item.identity for item in signals)
        if len(set(identities)) != len(identities):
            raise ValueError("signal band identities must be unique")
        if any(
            _overlap(first.qr_interval_Ainv, second.qr_interval_Ainv)
            for index, first in enumerate(signals)
            for second in signals[index + 1 :]
        ):
            raise ValueError("signal bands must be mutually disjoint")
        background = tuple(
            _interval(interval, f"background_intervals_Ainv[{index}]")
            for index, interval in enumerate(self.background_intervals_Ainv)
        )
        if not background:
            raise ValueError("background_intervals_Ainv must not be empty")
        for interval in background:
            for signal in signals:
                if _overlap(interval, signal.qr_interval_Ainv):
                    raise ValueError(
                        f"background interval intersects signal band {signal.identity!r}"
                    )
        if any(
            _overlap(first, second)
            for index, first in enumerate(background)
            for second in background[index + 1 :]
        ):
            raise ValueError("background intervals must be mutually disjoint")
        object.__setattr__(self, "axial_bin_edges", edges)
        object.__setattr__(self, "detector_column_interval_px", columns)
        object.__setattr__(self, "signal_bands", signals)
        object.__setattr__(self, "background_intervals_Ainv", background)

    @property
    def bin_count(self) -> int:
        return int(np.asarray(self.axial_bin_edges).size - 1)


@dataclass(frozen=True, slots=True)
class SpecularAngularMembership:
    signal_bin_index: IntArray
    background_bin_index: IntArray


@dataclass(frozen=True, slots=True)
class OffSpecularMembership:
    signal_bin_index: Mapping[str, IntArray]
    background_bin_index: IntArray


def _aligned_coordinates(
    values: tuple[ArrayLike, ...],
    names: tuple[str, ...],
) -> tuple[NDArray[np.float64], ...]:
    arrays: list[NDArray[np.float64]] = []
    for supplied, name in zip(values, names, strict=True):
        raw = np.asarray(supplied)
        if np.iscomplexobj(raw) and np.any(raw.imag != 0.0):
            raise ValueError(f"{name} must be real")
        arrays.append(np.asarray(raw.real, dtype=np.float64))
    result = tuple(np.broadcast_arrays(*arrays))
    if any(not np.all(np.isfinite(value)) for value in result):
        raise ValueError("observation coordinates must be finite")
    return result


def _bin_index(coordinate: NDArray[np.float64], edges: NDArray[np.float64]) -> IntArray:
    index = np.searchsorted(edges, coordinate, side="right") - 1
    index = np.where(coordinate == edges[-1], edges.size - 2, index)
    accepted = (coordinate >= edges[0]) & (coordinate <= edges[-1])
    result = np.where(accepted, index, -1).astype(np.int64, copy=False)
    return result


def _frozen_index(index: NDArray[np.int64], selected: NDArray[np.bool_]) -> IntArray:
    result = np.array(np.where(selected, index, -1), dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def specular_angular_membership(
    *,
    two_theta_rad: ArrayLike,
    phi_rad: ArrayLike,
    valid: ArrayLike,
    region: SpecularAngularProfileRegion,
) -> SpecularAngularMembership:
    """Assign native pixel centers to m=0 signal or phi-background bins."""

    if not isinstance(region, SpecularAngularProfileRegion):
        raise TypeError("region must be a SpecularAngularProfileRegion")
    two_theta, phi = _aligned_coordinates(
        (two_theta_rad, phi_rad),
        ("two_theta_rad", "phi_rad"),
    )
    accepted = np.broadcast_to(np.asarray(valid, dtype=np.bool_), two_theta.shape)
    bin_index = _bin_index(two_theta, np.asarray(region.two_theta_bin_edges_rad))
    signal_lower, signal_upper = region.phi_signal_interval_rad
    signal = accepted & (bin_index >= 0) & (phi >= signal_lower) & (phi < signal_upper)
    background = np.zeros(two_theta.shape, dtype=np.bool_)
    for lower, upper in region.phi_background_intervals_rad:
        background |= (phi >= lower) & (phi < upper)
    background &= accepted & (bin_index >= 0)
    return SpecularAngularMembership(
        signal_bin_index=_frozen_index(bin_index, signal),
        background_bin_index=_frozen_index(bin_index, background),
    )


def offspecular_region_membership(
    *,
    qr_Ainv: ArrayLike,
    axial_coordinate: ArrayLike,
    detector_column_px: ArrayLike,
    valid: ArrayLike,
    layout: OffSpecularBandLayout,
) -> OffSpecularMembership:
    """Assign native pixel centers to named Qr signals and clean shared anchors."""

    if not isinstance(layout, OffSpecularBandLayout):
        raise TypeError("layout must be an OffSpecularBandLayout")
    qr, axial, column = _aligned_coordinates(
        (qr_Ainv, axial_coordinate, detector_column_px),
        ("qr_Ainv", "axial_coordinate", "detector_column_px"),
    )
    accepted = np.array(
        np.broadcast_to(np.asarray(valid, dtype=np.bool_), qr.shape),
        dtype=np.bool_,
        copy=True,
    )
    lower_column, upper_column = layout.detector_column_interval_px
    accepted &= (column >= lower_column) & (column < upper_column)
    bin_index = _bin_index(axial, np.asarray(layout.axial_bin_edges))
    accepted &= bin_index >= 0
    signals: dict[str, IntArray] = {}
    for band in layout.signal_bands:
        lower, upper = band.qr_interval_Ainv
        selected = accepted & (qr >= lower) & (qr < upper)
        signals[band.identity] = _frozen_index(bin_index, selected)
    background = np.zeros(qr.shape, dtype=np.bool_)
    for lower, upper in layout.background_intervals_Ainv:
        background |= (qr >= lower) & (qr < upper)
    background &= accepted
    return OffSpecularMembership(
        signal_bin_index=MappingProxyType(signals),
        background_bin_index=_frozen_index(bin_index, background),
    )


__all__ = [
    "OffSpecularBand",
    "OffSpecularBandLayout",
    "OffSpecularMembership",
    "SpecularAngularMembership",
    "SpecularAngularProfileRegion",
    "offspecular_region_membership",
    "specular_angular_membership",
]
