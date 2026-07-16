"""Coherent finite ordered-stack sums in raw electron units."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import EventIntensityResult, RodQueryBatch
from rasim_next.ordered.amplitudes import OrderedEventResult


@dataclass(frozen=True, slots=True)
class FiniteStackResult:
    """Event-aligned finite total before any universal scattering scale."""

    event_id: NDArray[np.int64]
    amplitude_e: NDArray[np.complex128]
    intensity: EventIntensityResult

    def __post_init__(self) -> None:
        event_id = np.array(self.event_id, dtype=np.int64, copy=True, order="C")
        amplitude = np.array(self.amplitude_e, dtype=np.complex128, copy=True, order="C")
        if event_id.ndim != 1 or amplitude.shape != event_id.shape:
            raise ValueError("event_id and amplitude_e must be aligned one-dimensional arrays")
        if not np.array_equal(event_id, self.intensity.event_id) or not np.all(
            np.isfinite(amplitude)
        ):
            raise ValueError(
                "finite stack amplitude and intensity must be finite and event-aligned"
            )
        event_id.setflags(write=False)
        amplitude.setflags(write=False)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "amplitude_e", amplitude)


@dataclass(frozen=True, slots=True)
class Bi2Se3WholeCellCompatResult:
    """Raw finite stack plus the separate historical legacy-unit observable."""

    event_id: NDArray[np.int64]
    external_l_coordinate: NDArray[np.float64]
    unit_cell_amplitude_e: NDArray[np.complex128]
    finite_stack: FiniteStackResult
    legacy_intensity: NDArray[np.float64]
    cache_identity: str
    provenance: str
    layer_count: int = 17
    phi_l_divisor: float = 1.0
    pole_clamp: float = 1e-6
    legacy_normalization: str = "legacy AREA*(pole-clamped pair sum/17)*|F|^2"

    def __post_init__(self) -> None:
        event_id = np.array(self.event_id, dtype=np.int64, copy=True, order="C")
        l_coordinate = np.array(self.external_l_coordinate, dtype=np.float64, copy=True, order="C")
        unit_cell = np.array(self.unit_cell_amplitude_e, dtype=np.complex128, copy=True, order="C")
        legacy = np.array(self.legacy_intensity, dtype=np.float64, copy=True, order="C")
        if (
            event_id.ndim != 1
            or l_coordinate.shape != event_id.shape
            or unit_cell.shape != event_id.shape
            or legacy.shape != event_id.shape
            or not np.array_equal(event_id, self.finite_stack.event_id)
            or not np.all(np.isfinite(l_coordinate))
            or not np.all(np.isfinite(unit_cell))
            or not np.all(np.isfinite(legacy))
            or np.any(legacy < 0.0)
            or self.layer_count != 17
            or self.phi_l_divisor != 1.0
            or self.pole_clamp != 1e-6
            or not self.cache_identity
            or not self.provenance
            or not self.legacy_normalization
        ):
            raise ValueError("Bi2Se3 compatibility result is invalid or misaligned")
        for array in (event_id, l_coordinate, unit_cell, legacy):
            array.setflags(write=False)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "external_l_coordinate", l_coordinate)
        object.__setattr__(self, "unit_cell_amplitude_e", unit_cell)
        object.__setattr__(self, "legacy_intensity", legacy)


def _result(event_id: ArrayLike, amplitude_e: ArrayLike) -> FiniteStackResult:
    event = np.asarray(event_id, dtype=np.int64)
    amplitude = np.asarray(amplitude_e, dtype=np.complex128)
    intensity = EventIntensityResult(
        event_id=event,
        intensity_per_sr=np.abs(amplitude) ** 2,
        model_id="ordered",
        model_component_id="finite_coherent_total",
        population_group_id=None,
        normalization="finite total |sum(F_e exp(i phase))|^2; electron2",
    )
    return FiniteStackResult(event_id=event, amplitude_e=amplitude, intensity=intensity)


def coherent_finite_stack(
    event_id: ArrayLike,
    qz_Ainv: ArrayLike,
    layer_amplitude_e: ArrayLike,
    layer_depth_A: ArrayLike,
    *,
    registry_phase_rad: ArrayLike = 0.0,
) -> FiniteStackResult:
    """Sum explicit layer amplitudes with external depth and registry phases."""

    event = np.asarray(event_id, dtype=np.int64)
    qz = np.asarray(qz_Ainv, dtype=np.float64)
    amplitude = np.asarray(layer_amplitude_e, dtype=np.complex128)
    depth = np.asarray(layer_depth_A, dtype=np.float64)
    if event.ndim != 1 or qz.shape != event.shape or amplitude.ndim != 2:
        raise ValueError("event, qz, and layer amplitudes must form an event-by-layer batch")
    if amplitude.shape[0] != event.size or amplitude.shape[1] == 0:
        raise ValueError("layer amplitudes must contain at least one layer for every event")
    try:
        depth_batch = np.broadcast_to(depth, amplitude.shape)
        registry = np.broadcast_to(
            np.asarray(registry_phase_rad, dtype=np.float64), amplitude.shape
        )
    except ValueError as error:
        raise ValueError(
            "layer depths and registry phases must broadcast to event-by-layer shape"
        ) from error
    if not (
        np.all(np.isfinite(qz))
        and np.all(np.isfinite(amplitude))
        and np.all(np.isfinite(depth_batch))
        and np.all(np.isfinite(registry))
    ):
        raise ValueError("finite stack inputs must be finite")
    phase = qz[:, None] * depth_batch + registry
    return _result(event, np.sum(amplitude * np.exp(1.0j * phase), axis=1))


def uniform_finite_stack(
    event_id: ArrayLike,
    qz_Ainv: ArrayLike,
    repeat_amplitude_e: ArrayLike,
    repeat_spacing_A: float,
    repeat_count: int,
    *,
    registry_step_phase_rad: ArrayLike = 0.0,
) -> FiniteStackResult:
    """Evaluate a uniform finite geometric sum stably at and near Bragg points."""

    event = np.asarray(event_id, dtype=np.int64)
    qz = np.asarray(qz_Ainv, dtype=np.float64)
    repeat = np.asarray(repeat_amplitude_e, dtype=np.complex128)
    if event.ndim != 1 or qz.shape != event.shape or repeat.shape != event.shape:
        raise ValueError("event, qz, and repeat amplitude must be aligned one-dimensional arrays")
    try:
        count = index(repeat_count)
    except TypeError as error:
        raise ValueError("repeat_count must be a positive integer") from error
    if isinstance(repeat_count, bool) or count < 1:
        raise ValueError("repeat_count must be a positive integer")
    spacing = float(repeat_spacing_A)
    try:
        registry = np.broadcast_to(
            np.asarray(registry_step_phase_rad, dtype=np.float64), event.shape
        )
    except ValueError as error:
        raise ValueError("registry step phase must broadcast to the event batch") from error
    if not (
        np.isfinite(spacing)
        and spacing >= 0.0
        and np.all(np.isfinite(qz))
        and np.all(np.isfinite(repeat))
        and np.all(np.isfinite(registry))
    ):
        raise ValueError("uniform finite stack inputs must be finite with nonnegative spacing")
    step_phase = qz * spacing + registry
    wrapped = np.remainder(step_phase + np.pi, 2.0 * np.pi) - np.pi
    geometric = (
        count
        * np.exp(0.5j * (count - 1) * wrapped)
        * np.sinc(count * wrapped / (2.0 * np.pi))
        / np.sinc(wrapped / (2.0 * np.pi))
    )
    return _result(event, repeat * geometric)


def bi2se3_whole_cell_compat_curve(
    query: RodQueryBatch,
    ordered: OrderedEventResult,
) -> Bi2Se3WholeCellCompatResult:
    """Build the frozen 17-cell raw and legacy-unit compatibility observables."""

    if ordered.basis_mode != "bi2se3_whole_cell_compat":
        raise ValueError("ordered result must use bi2se3_whole_cell_compat")
    if not np.array_equal(query.event_id, ordered.event_id):
        raise ValueError("query and ordered result must preserve identical event identity")
    if not np.all(query.wavelength_A == 1.54):
        raise ValueError("Bi2Se3 compatibility requires wavelength_A=1.54")
    if not np.allclose(
        query.qz_Ainv,
        query.l_coordinate * (2.0 * np.pi / 28.636),
        rtol=2e-12,
        atol=1e-12,
    ):
        raise ValueError("query qz_Ainv and external L are inconsistent for c=28.636 A")

    finite_stack = uniform_finite_stack(
        query.event_id,
        query.qz_Ainv,
        ordered.amplitude_e,
        repeat_spacing_A=28.636,
        repeat_count=17,
    )
    damped_step = (1.0 - 1e-6) * np.exp(2.0j * np.pi * query.l_coordinate / 1.0)
    weighted_pair_sum = np.zeros(query.event_id.shape, dtype=np.complex128)
    power = np.ones(query.event_id.shape, dtype=np.complex128)
    for separation in range(1, 17):
        power *= damped_step
        weighted_pair_sum += (17 - separation) * power
    pair_factor_per_layer = np.maximum(
        (17.0 + 2.0 * np.real(weighted_pair_sum)) / 17.0,
        0.0,
    )
    legacy_area = (2.0 * np.pi) ** 2 / 17.98e-10 * 3.0
    legacy_intensity = legacy_area * np.abs(ordered.amplitude_e) ** 2 * pair_factor_per_layer
    identity = (
        f"{ordered.cache_identity};finite_layers=17;phi_l_divisor=1;"
        "legacy_area=(2*pi)^2/17.98e-10*3;pole_clamp=1e-6"
    )
    return Bi2Se3WholeCellCompatResult(
        event_id=query.event_id,
        external_l_coordinate=query.l_coordinate,
        unit_cell_amplitude_e=ordered.amplitude_e,
        finite_stack=finite_stack,
        legacy_intensity=legacy_intensity,
        cache_identity=identity,
        provenance=identity,
    )
