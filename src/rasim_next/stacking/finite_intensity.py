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
    registry_phase,
)

_REDUCED_MOMENT_EVENT_CHUNK_SIZE = 131_072


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


def _finite_moment_intensity(
    layers: int,
    f_plus: complex | NDArray[np.complex128],
    f_minus: complex | NDArray[np.complex128],
    omega: complex | NDArray[np.complex128],
    vertical_phase: complex | NDArray[np.complex128],
    a: float,
    b_plus: float,
    b_minus: float,
    d_plus: float,
    d_minus: float,
    probability_plus: float,
    probability_minus: float,
) -> float | NDArray[np.float64]:
    """Exact centered registry-gauge recurrence for arrays or compiled scalar lanes.

    Inputs are validated by the caller. CPU and CUDA compile this same arithmetic;
    the independent full-state recurrence remains the proof oracle.
    """

    inverse = omega.real - 1j * omega.imag
    same = a + b_plus + b_minus
    flip = d_plus + d_minus
    same_gauge = 0.0j
    same_gauge_variance = 0.0
    flip_gauge_plus = 0.0j
    flip_gauge_minus = 0.0j
    flip_gauge_variance = 0.0
    if same > 0.0:
        weight_a, weight_b_plus, weight_b_minus = a / same, b_plus / same, b_minus / same
        same_gauge = 1.0 + weight_b_plus * (inverse - 1.0) + weight_b_minus * (omega - 1.0)
        # Positive pairwise variance avoids cancellation and is exactly zero
        # when registry is invisible. Normalize before multiplying tiny weights.
        same_gauge_variance = (
            weight_a * weight_b_plus * abs(1.0 - inverse) ** 2
            + weight_a * weight_b_minus * abs(1.0 - omega) ** 2
            + weight_b_plus * weight_b_minus * abs(inverse - omega) ** 2
        )
    if flip > 0.0:
        weight_d_plus, weight_d_minus = d_plus / flip, d_minus / flip
        flip_gauge_plus = omega + weight_d_minus * (inverse - omega)
        flip_gauge_minus = inverse + weight_d_minus * (omega - inverse)
        flip_gauge_variance = weight_d_plus * weight_d_minus * abs(inverse - omega) ** 2

    mean_plus, mean_minus = f_plus, f_minus
    variance_plus = 0.0 * f_plus.real
    variance_minus = 0.0 * f_minus.real
    phase_power = 1.0 + 0.0j
    for _ in range(1, layers):
        phase_power = phase_power * vertical_phase
        next_probability_plus = probability_plus * same + probability_minus * flip
        next_probability_minus = probability_minus * same + probability_plus * flip
        stay_plus, stay_minus = same_gauge * mean_plus, same_gauge * mean_minus
        flipped_plus = flip_gauge_plus * mean_minus
        flipped_minus = flip_gauge_minus * mean_plus
        plus_squared, minus_squared = abs(mean_plus) ** 2, abs(mean_minus) ** 2
        if next_probability_plus > 0.0:
            stay_weight = probability_plus * same / next_probability_plus
            flip_weight = probability_minus * flip / next_probability_plus
            difference = flipped_plus - stay_plus
            # Anchor on the heavier component; identical means remain identical.
            transported_plus = (
                stay_plus + flip_weight * difference
                if stay_weight >= flip_weight
                else flipped_plus - stay_weight * difference
            )
            next_variance_plus = (
                stay_weight * (variance_plus + same_gauge_variance * plus_squared)
                + flip_weight * (variance_minus + flip_gauge_variance * minus_squared)
                + stay_weight * flip_weight * abs(difference) ** 2
            )
            next_mean_plus = transported_plus + phase_power * f_plus
        else:
            next_mean_plus = 0.0 * f_plus
            next_variance_plus = 0.0 * f_plus.real
        if next_probability_minus > 0.0:
            stay_weight = probability_minus * same / next_probability_minus
            flip_weight = probability_plus * flip / next_probability_minus
            difference = flipped_minus - stay_minus
            transported_minus = (
                stay_minus + flip_weight * difference
                if stay_weight >= flip_weight
                else flipped_minus - stay_weight * difference
            )
            next_variance_minus = (
                stay_weight * (variance_minus + same_gauge_variance * minus_squared)
                + flip_weight * (variance_plus + flip_gauge_variance * plus_squared)
                + stay_weight * flip_weight * abs(difference) ** 2
            )
            next_mean_minus = transported_minus + phase_power * f_minus
        else:
            next_mean_minus = 0.0 * f_minus
            next_variance_minus = 0.0 * f_minus.real
        probability_plus, probability_minus = next_probability_plus, next_probability_minus
        mean_plus, mean_minus = next_mean_plus, next_mean_minus
        variance_plus, variance_minus = next_variance_plus, next_variance_minus
    return probability_plus * (variance_plus + abs(mean_plus) ** 2) + probability_minus * (
        variance_minus + abs(mean_minus) ** 2
    )


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
    probabilities = (
        law.a,
        law.b_plus,
        law.b_minus,
        law.d_plus,
        law.d_minus,
        initial.plus,
        initial.minus,
    )
    if f_plus_array.size > _REDUCED_MOMENT_EVENT_CHUNK_SIZE:
        shape = f_plus_array.shape
        result = np.empty(f_plus_array.size, dtype=np.float64)
        flattened = tuple(
            np.ravel(value) for value in (f_plus_array, f_minus_array, omega_array, phase_array)
        )
        for start in range(0, result.size, _REDUCED_MOMENT_EVENT_CHUNK_SIZE):
            stop = min(start + _REDUCED_MOMENT_EVENT_CHUNK_SIZE, result.size)
            result[start:stop] = _finite_moment_intensity(
                count,
                flattened[0][start:stop],
                flattened[1][start:stop],
                flattened[2][start:stop],
                flattened[3][start:stop],
                *probabilities,
            )
        return _readonly_nonnegative(
            result.reshape(shape),
            "finite reduced moment intensity",
        )
    with np.errstate(over="ignore", invalid="ignore"):
        result = _finite_moment_intensity(
            count,
            f_plus_array,
            f_minus_array,
            omega_array,
            phase_array,
            *probabilities,
        )
    return _readonly_nonnegative(
        result,
        "finite reduced moment intensity",
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
