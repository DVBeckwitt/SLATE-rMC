"""Finite full-state and exact reduced stacking intensities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import (
    EventIntensityNormalization,
    EventIntensityResult,
    LayerAmplitudeNormalization,
    LayerAmplitudeResult,
    LayerNormalQBatch,
    LayerPhaseSign,
    RodQueryBatch,
)
from rasim_next.core.scattering import electron_squared_to_scattering_strength_A2
from rasim_next.stacking.parent_models import StackingPopulation
from rasim_next.stacking.transition import (
    InitialPopulation,
    RegistryPhaseModel,
    TransitionLaw,
    _validated_registry_phase,
    full_transition_matrix,
    orientation_transition_matrix,
    registry_phase,
)


def _layers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError("layers must be a positive integer")
    return int(value)


def _readonly_nonnegative(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    tolerance = 256.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(result))
    if np.any(result < -tolerance):
        raise ValueError(f"{name} is negative beyond roundoff")
    result[result < 0.0] = 0.0
    result.setflags(write=False)
    return result


def _broadcast_inputs(
    f_plus: ArrayLike,
    f_minus: ArrayLike,
    omega: ArrayLike,
    vertical_phase: ArrayLike,
) -> tuple[NDArray[np.complex128], ...]:
    arrays = tuple(
        np.asarray(value, dtype=np.complex128) for value in (f_plus, f_minus, omega, vertical_phase)
    )
    broadcast = tuple(np.broadcast_arrays(*arrays))
    if any(np.any(~np.isfinite(value)) for value in broadcast):
        raise ValueError("amplitudes and phases must be finite")
    omega_array = _validated_registry_phase(broadcast[2])
    if not np.allclose(np.abs(broadcast[3]), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("finite-stack vertical phase must have unit magnitude")
    return broadcast[0], broadcast[1], omega_array, broadcast[3]


def _reduced_moment_intensity(
    layers: int,
    f_plus: NDArray[np.complex128],
    f_minus: NDArray[np.complex128],
    omega: NDArray[np.complex128],
    vertical_phase: NDArray[np.complex128],
    law: TransitionLaw,
    initial: InitialPopulation,
) -> NDArray[np.float64]:
    """Propagate exact orientation-conditioned amplitude moments in a registry gauge."""

    amplitudes = np.stack((f_plus, f_minus), axis=-1)
    probability = initial.as_array().copy()
    mean = np.array(amplitudes, dtype=np.complex128, copy=True, order="C")
    variance = np.zeros(amplitudes.shape, dtype=np.float64)
    inverse = np.conj(omega)
    edges = (
        (0, 0, law.a, 1.0 + 0.0j),
        (0, 0, law.b_plus, inverse),
        (0, 0, law.b_minus, omega),
        (0, 1, law.d_plus, inverse),
        (0, 1, law.d_minus, omega),
        (1, 1, law.a, 1.0 + 0.0j),
        (1, 1, law.b_plus, inverse),
        (1, 1, law.b_minus, omega),
        (1, 0, law.d_plus, omega),
        (1, 0, law.d_minus, inverse),
    )
    phase_power = np.ones_like(vertical_phase)
    orientation_transition = orientation_transition_matrix(law)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(1, layers):
            phase_power *= vertical_phase
            contribution = phase_power[..., None] * amplitudes
            next_probability = probability @ orientation_transition
            next_mean = np.zeros_like(mean)
            for source, target, transition_probability, registry_gauge in edges:
                weight = probability[source] * transition_probability
                if weight:
                    next_mean[..., target] += weight * (
                        registry_gauge * mean[..., source] + contribution[..., target]
                    )
            next_mean = np.divide(
                next_mean,
                next_probability,
                out=np.zeros_like(next_mean),
                where=next_probability > 0.0,
            )
            next_variance = np.zeros_like(variance)
            for source, target, transition_probability, registry_gauge in edges:
                weight = probability[source] * transition_probability
                if weight:
                    candidate_mean = registry_gauge * mean[..., source] + contribution[..., target]
                    next_variance[..., target] += weight * (
                        variance[..., source] + np.abs(candidate_mean - next_mean[..., target]) ** 2
                    )
            variance = np.divide(
                next_variance,
                next_probability,
                out=np.zeros_like(next_variance),
                where=next_probability > 0.0,
            )
            probability = next_probability
            mean = next_mean
        intensity = np.sum(
            probability * (variance + np.abs(mean) ** 2),
            axis=-1,
        )
    return _readonly_nonnegative(intensity, "finite reduced moment intensity")


def _full_moment_intensity(
    layers: int,
    amplitudes: NDArray[np.complex128],
    vertical_phase: complex,
    transition: NDArray[np.float64],
    initial: InitialPopulation,
) -> NDArray[np.float64]:
    """Propagate exact state-conditioned amplitude moments for one event."""

    probability = np.array(
        [initial.plus, 0.0, 0.0, initial.minus, 0.0, 0.0],
        dtype=np.float64,
    )
    mean = np.zeros(6, dtype=np.complex128)
    mean[0] = amplitudes[0]
    mean[3] = amplitudes[3]
    variance = np.zeros(6, dtype=np.float64)
    phase_power = 1.0 + 0.0j
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(1, layers):
            phase_power *= vertical_phase
            contribution = phase_power * amplitudes
            incoming_weight = probability[:, None] * transition
            next_probability = incoming_weight.sum(axis=0)
            candidate_mean = mean[:, None] + contribution[None, :]
            next_mean = np.divide(
                np.sum(incoming_weight * candidate_mean, axis=0),
                next_probability,
                out=np.zeros(6, dtype=np.complex128),
                where=next_probability > 0.0,
            )
            next_variance = np.divide(
                np.sum(
                    incoming_weight
                    * (variance[:, None] + np.abs(candidate_mean - next_mean[None, :]) ** 2),
                    axis=0,
                ),
                next_probability,
                out=np.zeros(6, dtype=np.float64),
                where=next_probability > 0.0,
            )
            probability = next_probability
            mean = next_mean
            variance = next_variance
        intensity = np.sum(probability * (variance + np.abs(mean) ** 2))
    return _readonly_nonnegative(intensity, "finite full moment intensity")


def finite_intensity_reduced(
    layers: int,
    f_plus: ArrayLike,
    f_minus: ArrayLike,
    omega: ArrayLike,
    vertical_phase: ArrayLike,
    law: TransitionLaw,
    initial: InitialPopulation,
) -> NDArray[np.float64]:
    """Evaluate an exact two-orientation finite moment recurrence."""

    count = _layers(layers)
    f_plus_array, f_minus_array, omega_array, phase_array = _broadcast_inputs(
        f_plus, f_minus, omega, vertical_phase
    )
    return _reduced_moment_intensity(
        count,
        f_plus_array,
        f_minus_array,
        omega_array,
        phase_array,
        law,
        initial,
    )


def finite_intensity_full(
    layers: int,
    f_plus: complex,
    f_minus: complex,
    omega: complex,
    vertical_phase: complex,
    law: TransitionLaw,
    initial: InitialPopulation,
) -> NDArray[np.float64]:
    """Evaluate a stable six-state finite moment recurrence for one event."""

    count = _layers(layers)
    f_plus_array, f_minus_array, omega_array, phase_array = _broadcast_inputs(
        f_plus, f_minus, omega, vertical_phase
    )
    if f_plus_array.ndim or f_minus_array.ndim or omega_array.ndim or phase_array.ndim:
        raise ValueError("full-state oracle accepts one event")
    omega_value = complex(omega_array)
    amplitudes = np.array(
        [
            complex(f_plus_array),
            omega_value * complex(f_plus_array),
            omega_value**2 * complex(f_plus_array),
            complex(f_minus_array),
            omega_value * complex(f_minus_array),
            omega_value**2 * complex(f_minus_array),
        ],
        dtype=np.complex128,
    )
    amplitude_intensity = np.abs(amplitudes) ** 2
    phase_value = complex(phase_array)
    transition = full_transition_matrix(law)
    if np.any(~np.isfinite(amplitude_intensity)):
        raise ValueError("full-state intensity must remain finite")
    return _full_moment_intensity(count, amplitudes, phase_value, transition, initial)


def _event_phases(
    query: RodQueryBatch,
    amplitudes: LayerAmplitudeResult,
    layer_normal_q: LayerNormalQBatch,
    phase_model: RegistryPhaseModel,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    layer_normal_q_Ainv = _aligned_layer_normal_q(query, amplitudes, layer_normal_q)
    omega = np.asarray(registry_phase(query.h, query.k, phase_model), dtype=np.complex128)
    with np.errstate(over="ignore", invalid="ignore"):
        phase_argument = layer_normal_q_Ainv * amplitudes.layer_repeat_A
    if np.any(~np.isfinite(phase_argument)):
        raise ValueError("layer_normal_q_Ainv * layer_repeat_A must be finite")
    vertical = np.exp(1j * phase_argument)
    return omega, vertical


def _aligned_amplitudes(
    query: RodQueryBatch, amplitudes: LayerAmplitudeResult
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    if not (
        np.array_equal(query.event_id, amplitudes.event_id)
        and np.array_equal(query.rod_id, amplitudes.rod_id)
        and query.phase_id == amplitudes.phase_id
    ):
        raise ValueError("layer amplitudes must be exactly event/rod/phase-aligned")
    if amplitudes.normalization is not LayerAmplitudeNormalization.ONE_REGISTRY_FREE_LAYER:
        raise ValueError("stacking requires one-registry-free-layer amplitudes")
    if amplitudes.phase_sign is not LayerPhaseSign.POSITIVE_Q_DOT_R:
        raise ValueError("stacking requires the positive-Q-dot-R phase convention")
    if amplitudes.f_minus_e is None:
        raise ValueError("stacking intensity requires both f_plus and f_minus")
    return amplitudes.f_plus_e, amplitudes.f_minus_e


def _aligned_layer_normal_q(
    query: RodQueryBatch,
    amplitudes: LayerAmplitudeResult,
    layer_normal_q: LayerNormalQBatch,
) -> NDArray[np.float64]:
    if not (
        np.array_equal(query.event_id, layer_normal_q.event_id)
        and np.array_equal(query.rod_id, layer_normal_q.rod_id)
        and query.phase_id == layer_normal_q.phase_id
    ):
        raise ValueError("layer-normal wavevectors must be exactly event/rod/phase-aligned")
    if amplitudes.gauge_id != layer_normal_q.gauge_id:
        raise ValueError("layer amplitudes and layer-normal wavevectors must share one gauge_id")
    return layer_normal_q.layer_normal_q_Ainv


def finite_event_intensity(
    query: RodQueryBatch,
    amplitudes: LayerAmplitudeResult,
    law: TransitionLaw,
    *,
    layer_normal_q: LayerNormalQBatch,
    layers: int,
    initial: InitialPopulation,
    model_component_id: str,
    population_group_id: str | None,
    normalization: EventIntensityNormalization,
    phase_model: RegistryPhaseModel = RegistryPhaseModel.FORWARD_H_PLUS_2K,
) -> EventIntensityResult:
    """Return one unweighted finite stacking component in angstrom squared."""

    count = _layers(layers)
    normalization = EventIntensityNormalization(normalization)
    if normalization is EventIntensityNormalization.UNIT_CELL:
        raise ValueError("finite stacking normalization must be FINITE_TOTAL or FINITE_PER_LAYER")
    f_plus, f_minus = _aligned_amplitudes(query, amplitudes)
    omega, vertical = _event_phases(query, amplitudes, layer_normal_q, phase_model)
    raw = finite_intensity_reduced(count, f_plus, f_minus, omega, vertical, law, initial)
    if normalization is EventIntensityNormalization.FINITE_PER_LAYER:
        raw = raw / float(count)
    return EventIntensityResult(
        event_id=query.event_id,
        scattering_strength_A2=electron_squared_to_scattering_strength_A2(raw),
        model_id="stacking",
        model_component_id=model_component_id,
        population_group_id=population_group_id,
        normalization=normalization,
    )


def finite_population_event_intensity(
    query: RodQueryBatch,
    amplitudes: LayerAmplitudeResult,
    populations: tuple[StackingPopulation, ...],
    *,
    layer_normal_q: LayerNormalQBatch,
    layers: int,
    population_group_id: str,
    normalization: EventIntensityNormalization,
    phase_model: RegistryPhaseModel = RegistryPhaseModel.FORWARD_H_PLUS_2K,
) -> tuple[EventIntensityResult, ...]:
    """Return sorted event-aligned components without applying population weights."""

    count = _layers(layers)
    normalization = EventIntensityNormalization(normalization)
    if normalization is EventIntensityNormalization.UNIT_CELL:
        raise ValueError("finite stacking normalization must be FINITE_TOTAL or FINITE_PER_LAYER")
    supplied = tuple(populations)
    if not supplied:
        raise ValueError("populations must be nonempty")
    if any(not isinstance(population, StackingPopulation) for population in supplied):
        raise TypeError("populations must contain StackingPopulation values")
    ordered = tuple(sorted(supplied, key=lambda population: population.population_id))
    population_id = tuple(population.population_id for population in ordered)
    if len(set(population_id)) != len(population_id):
        raise ValueError("population_id values must be unique")
    f_plus, f_minus = _aligned_amplitudes(query, amplitudes)
    omega, vertical = _event_phases(query, amplitudes, layer_normal_q, phase_model)
    component = np.stack(
        [
            finite_intensity_reduced(
                count,
                f_plus,
                f_minus,
                omega,
                vertical,
                population.model,
                population.initial,
            )
            for population in ordered
        ]
    )
    if normalization is EventIntensityNormalization.FINITE_PER_LAYER:
        component = component / float(count)
    scattering_strength_A2 = electron_squared_to_scattering_strength_A2(component)
    return tuple(
        EventIntensityResult(
            event_id=query.event_id,
            scattering_strength_A2=scattering_strength_A2[index],
            model_id="stacking",
            model_component_id=population.population_id,
            population_group_id=population_group_id,
            normalization=normalization,
        )
        for index, population in enumerate(ordered)
    )
