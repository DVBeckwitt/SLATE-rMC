"""Coherent finite ordered-stack sums in raw electron units."""

from __future__ import annotations

from operator import index

import numpy as np
from numpy.typing import ArrayLike

from rasim_next.core.contracts import EventIntensityResult
from rasim_next.core.scattering import electron_squared_to_scattering_strength_A2


def _result(event_id: ArrayLike, amplitude_e: ArrayLike) -> EventIntensityResult:
    event = np.asarray(event_id, dtype=np.int64)
    amplitude = np.asarray(amplitude_e, dtype=np.complex128)
    return EventIntensityResult(
        event_id=event,
        scattering_strength_A2=electron_squared_to_scattering_strength_A2(np.abs(amplitude) ** 2),
        model_id="ordered",
        model_component_id="finite_coherent_total",
        population_group_id=None,
        normalization="FINITE_TOTAL",
    )


def coherent_finite_stack(
    event_id: ArrayLike,
    q_sample_normal_Ainv: ArrayLike,
    layer_amplitude_e: ArrayLike,
    layer_depth_A: ArrayLike,
    *,
    registry_phase_rad: ArrayLike = 0.0,
) -> EventIntensityResult:
    """Sum explicit layer amplitudes with external depth and registry phases."""

    event = np.asarray(event_id, dtype=np.int64)
    q_sample_normal = np.asarray(q_sample_normal_Ainv, dtype=np.float64)
    amplitude = np.asarray(layer_amplitude_e, dtype=np.complex128)
    depth = np.asarray(layer_depth_A, dtype=np.float64)
    if event.ndim != 1 or q_sample_normal.shape != event.shape or amplitude.ndim != 2:
        raise ValueError("inputs must form an event-by-layer batch with sample-normal Q")
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
        np.all(np.isfinite(q_sample_normal))
        and np.all(np.isfinite(amplitude))
        and np.all(np.isfinite(depth_batch))
        and np.all(np.isfinite(registry))
    ):
        raise ValueError("finite stack inputs must be finite")
    phase = q_sample_normal[:, None] * depth_batch + registry
    return _result(event, np.sum(amplitude * np.exp(1.0j * phase), axis=1))


def uniform_finite_stack(
    event_id: ArrayLike,
    q_sample_normal_Ainv: ArrayLike,
    repeat_amplitude_e: ArrayLike,
    repeat_spacing_A: float,
    repeat_count: int,
    *,
    registry_step_phase_rad: ArrayLike = 0.0,
) -> EventIntensityResult:
    """Evaluate a uniform finite geometric sum stably at and near Bragg points."""

    event = np.asarray(event_id, dtype=np.int64)
    q_sample_normal = np.asarray(q_sample_normal_Ainv, dtype=np.float64)
    repeat = np.asarray(repeat_amplitude_e, dtype=np.complex128)
    if event.ndim != 1 or q_sample_normal.shape != event.shape or repeat.shape != event.shape:
        raise ValueError("event, sample-normal Q, and amplitude must be aligned arrays")
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
        and np.all(np.isfinite(q_sample_normal))
        and np.all(np.isfinite(repeat))
        and np.all(np.isfinite(registry))
    ):
        raise ValueError("uniform finite stack inputs must be finite with nonnegative spacing")
    step_phase = q_sample_normal * spacing + registry
    wrapped = np.remainder(step_phase + np.pi, 2.0 * np.pi) - np.pi
    geometric = (
        count
        * np.exp(0.5j * (count - 1) * wrapped)
        * np.sinc(count * wrapped / (2.0 * np.pi))
        / np.sinc(wrapped / (2.0 * np.pi))
    )
    return _result(event, repeat * geometric)
