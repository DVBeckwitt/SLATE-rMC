"""Exact reduced finite-stack intensities."""

from __future__ import annotations

import numba
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
    first_plus=None,
    first_minus=None,
    last_plus=None,
    last_minus=None,
) -> float | NDArray[np.float64]:
    """Exact centered registry-gauge recurrence for arrays or compiled scalar lanes.

    Inputs are validated by the caller. CPU and CUDA compile this same arithmetic.
    Independent full-state and enumeration evidence is archived in Git.
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

    mean_plus = f_plus if first_plus is None else first_plus
    mean_minus = f_minus if first_minus is None else first_minus
    variance_plus = 0.0 * f_plus.real
    variance_minus = 0.0 * f_minus.real
    phase_power = 1.0 + 0.0j
    for layer in range(1, layers):
        phase_power = phase_power * vertical_phase
        amplitude_plus = last_plus if layer == layers - 1 and last_plus is not None else f_plus
        amplitude_minus = last_minus if layer == layers - 1 and last_minus is not None else f_minus
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
            next_mean_plus = transported_plus + phase_power * amplitude_plus
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
            next_mean_minus = transported_minus + phase_power * amplitude_minus
        else:
            next_mean_minus = 0.0 * f_minus
            next_variance_minus = 0.0 * f_minus.real
        probability_plus, probability_minus = next_probability_plus, next_probability_minus
        mean_plus, mean_minus = next_mean_plus, next_mean_minus
        variance_plus, variance_minus = next_variance_plus, next_variance_minus
    return probability_plus * (variance_plus + abs(mean_plus) ** 2) + probability_minus * (
        variance_minus + abs(mean_minus) ** 2
    )


_finite_moment_intensity_cpu = numba.njit(nogil=True, fastmath=False, cache=False)(
    _finite_moment_intensity
)


@numba.njit(nogil=True, fastmath=False, cache=False)
def _finite_moment_events(layers, arrays, probabilities):
    """Evaluate independent scalar lanes with constant per-event working storage."""
    result = np.empty(len(arrays[0]))
    for i in range(len(result)):
        result[i] = _finite_moment_intensity_cpu(
            layers,
            arrays[0][i],
            arrays[1][i],
            arrays[2][i],
            arrays[3][i],
            *probabilities,
            arrays[4][i],
            arrays[5][i],
            arrays[6][i],
            arrays[7][i],
        )
    return result


def finite_intensity_reduced(
    layers: int,
    f_plus: ArrayLike,
    f_minus: ArrayLike,
    omega: ArrayLike,
    vertical_phase: ArrayLike,
    law: TransitionLaw,
    initial: InitialPopulation,
    *,
    first_amplitudes_e: tuple[ArrayLike, ArrayLike] | None = None,
    last_amplitudes_e: tuple[ArrayLike, ArrayLike] | None = None,
) -> NDArray[np.float64]:
    """Evaluate exact finite moments with optional distinct endpoint motifs.

    ``layers`` counts amplitude slots. With endpoints both pairs are required,
    in (plus, minus) order, and at least two slots must be present. No thickness
    or per-repeat normalization is implicit in this raw intensity calculation.
    """

    count = _layers(layers)
    f_plus_array, f_minus_array, omega_array, phase_array = _broadcast_inputs(
        f_plus, f_minus, omega, vertical_phase
    )
    endpoints = ()
    if first_amplitudes_e is not None or last_amplitudes_e is not None:
        if (
            first_amplitudes_e is None
            or last_amplitudes_e is None
            or count < 2
            or len(first_amplitudes_e) != 2
            or len(last_amplitudes_e) != 2
        ):
            raise ValueError("endpoint motifs require two orientation pairs and at least two slots")
        arrays = np.broadcast_arrays(
            f_plus_array,
            f_minus_array,
            omega_array,
            phase_array,
            *(
                np.asarray(a, dtype=np.complex128)
                for a in (*first_amplitudes_e, *last_amplitudes_e)
            ),
        )
        if any(np.any(~np.isfinite(a)) for a in arrays):
            raise ValueError("endpoint amplitudes must be finite")
        f_plus_array, f_minus_array, omega_array, phase_array = arrays[:4]
        endpoints = tuple(arrays[4:])
    probabilities = (
        law.a,
        law.b_plus,
        law.b_minus,
        law.d_plus,
        law.d_minus,
        initial.plus,
        initial.minus,
    )
    flattened = tuple(
        np.ravel(value) for value in (f_plus_array, f_minus_array, omega_array, phase_array)
    )
    flattened += (
        tuple(np.ravel(value) for value in endpoints)
        if endpoints
        else (flattened[0], flattened[1], flattened[0], flattened[1])
    )
    result = _finite_moment_events(count, flattened, probabilities)
    return _readonly_nonnegative(
        result.reshape(f_plus_array.shape),
        "finite reduced moment intensity",
    )


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
