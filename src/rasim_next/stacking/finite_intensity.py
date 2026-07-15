"""Finite full-state and exact reduced stacking intensities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import chain
from math import fsum
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import EventIntensityResult, LayerAmplitudeResult, RodQueryBatch
from rasim_next.stacking.parent_models import StackingPopulation
from rasim_next.stacking.transition import (
    InitialPopulation,
    RegistryPhaseModel,
    TransitionLaw,
    _validated_registry_phase,
    full_transition_matrix,
    orientation_transition_matrix,
    reduced_transition_matrix,
    registry_phase,
)


class FiniteNormalization(StrEnum):
    TOTAL = "intensity_electron2"
    PER_LAYER = "intensity_per_layer_electron2"


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


def _clip_cancellation_roundoff(
    value: ArrayLike, cancellation_scale: ArrayLike
) -> NDArray[np.float64]:
    """Zero signed residues inside a forward-error cancellation bound."""

    result = np.array(value, dtype=np.float64, copy=True, order="C")
    scale = np.broadcast_to(np.asarray(cancellation_scale, dtype=np.float64), result.shape)
    tolerance = 256.0 * np.finfo(np.float64).eps * np.maximum(1.0, scale)
    if np.any(~np.isfinite(result)) or np.any(~np.isfinite(scale)):
        raise ValueError("ensemble intensity and cancellation scale must be finite")
    if np.any(result < -tolerance):
        raise ValueError("ensemble intensity is negative beyond roundoff")
    result[np.abs(result) <= tolerance] = 0.0
    return result


@dataclass(frozen=True, slots=True)
class FiniteIntensity:
    """Raw finite-stack intensity in electron2 and its explicit per-layer quotient."""

    intensity_electron2: NDArray[np.float64]
    intensity_per_layer_electron2: NDArray[np.float64]

    def __post_init__(self) -> None:
        total = _readonly_nonnegative(self.intensity_electron2, "intensity_electron2")
        per_layer = _readonly_nonnegative(
            self.intensity_per_layer_electron2,
            "intensity_per_layer_electron2",
        )
        if total.shape != per_layer.shape:
            raise ValueError("total and per-layer intensities must have identical shapes")
        object.__setattr__(self, "intensity_electron2", total)
        object.__setattr__(self, "intensity_per_layer_electron2", per_layer)


@dataclass(frozen=True, slots=True)
class PopulationIntensityResult:
    """Unweighted population components and their explicit incoherent electron2 total."""

    population_id: tuple[str, ...]
    weight: NDArray[np.float64]
    component_intensity_electron2: NDArray[np.float64]
    weighted_total_intensity_electron2: NDArray[np.float64]
    event_intensity: EventIntensityResult

    def __post_init__(self) -> None:
        if not isinstance(self.population_id, tuple):
            raise ValueError("population_id must be a tuple")
        population_id = tuple(self.population_id)
        if (
            not population_id
            or any(not isinstance(item, str) or not item for item in population_id)
            or len(set(population_id)) != len(population_id)
            or population_id != tuple(sorted(population_id))
        ):
            raise ValueError("population_id must be nonempty, unique, and sorted")
        weight = np.array(self.weight, dtype=np.float64, copy=True, order="C")
        if (
            weight.shape != (len(population_id),)
            or np.any(~np.isfinite(weight))
            or np.any(weight < 0.0)
            or not np.isclose(weight.sum(), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("population weights must be finite, nonnegative, and sum to one")
        if not isinstance(self.event_intensity, EventIntensityResult):
            raise TypeError("event_intensity must be an EventIntensityResult")
        event_count = self.event_intensity.event_id.size
        component = _readonly_nonnegative(
            self.component_intensity_electron2,
            "component_intensity_electron2",
        )
        total = _readonly_nonnegative(
            self.weighted_total_intensity_electron2,
            "weighted_total_intensity_electron2",
        )
        if component.shape != (len(population_id), event_count) or total.shape != (event_count,):
            raise ValueError("population intensity arrays have inconsistent shapes")
        expected = np.einsum("p,pe->e", weight, component, optimize=True)
        tolerance = 256.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(expected))
        if np.any(np.abs(total - expected) > tolerance):
            raise ValueError("weighted total must equal the incoherent component intensity sum")
        if (
            self.event_intensity.normalization != FiniteNormalization.TOTAL.value
            or not np.array_equal(self.event_intensity.intensity_per_sr, total)
        ):
            raise ValueError("event intensity must contain the total electron2 population result")
        weight.setflags(write=False)
        object.__setattr__(self, "population_id", population_id)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "component_intensity_electron2", component)
        object.__setattr__(self, "weighted_total_intensity_electron2", total)


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


def _orientation_prefixes(
    layers: int, law: TransitionLaw, initial: InitialPopulation
) -> NDArray[np.float64]:
    transition = orientation_transition_matrix(law)
    populations = np.empty((layers, 2), dtype=np.float64)
    populations[0] = initial.as_array()
    for index in range(1, layers):
        populations[index] = populations[index - 1] @ transition
    return np.vstack((np.zeros((1, 2), dtype=np.float64), np.cumsum(populations, axis=0)))


def finite_intensity_reduced(
    layers: int,
    f_plus: ArrayLike,
    f_minus: ArrayLike,
    omega: ArrayLike,
    vertical_phase: ArrayLike,
    law: TransitionLaw,
    initial: InitialPopulation,
) -> FiniteIntensity:
    """Evaluate exact finite sums in the selected two-state Fourier sector."""

    count = _layers(layers)
    f_plus_array, f_minus_array, omega_array, phase_array = _broadcast_inputs(
        f_plus, f_minus, omega, vertical_phase
    )
    amplitudes = np.stack((f_plus_array, f_minus_array), axis=-1)
    transition = reduced_transition_matrix(law, omega_array)
    prefixes = _orientation_prefixes(count, law, initial)
    self_total = np.einsum("i,...i->...", prefixes[count], np.abs(amplitudes) ** 2, optimize=True)
    pair_total = np.zeros(amplitudes.shape[:-1], dtype=np.complex128)
    pair_absolute_sum = np.zeros(amplitudes.shape[:-1], dtype=np.float64)
    power = np.broadcast_to(np.eye(2, dtype=np.complex128), transition.shape).copy()
    vertical_power = np.ones_like(phase_array)
    for separation in range(1, count):
        power = np.matmul(power, transition)
        vertical_power *= phase_array
        propagated = np.einsum("...ij,...j->...i", power, amplitudes, optimize=True)
        by_orientation = np.conj(amplitudes) * propagated
        pair_contribution = vertical_power * np.einsum(
            "i,...i->...", prefixes[count - separation], by_orientation, optimize=True
        )
        pair_total += pair_contribution
        pair_absolute_sum += np.abs(pair_contribution)
    total = _clip_cancellation_roundoff(
        self_total + 2.0 * pair_total.real,
        np.abs(self_total) + 2.0 * pair_absolute_sum,
    )
    return FiniteIntensity(total, total / float(count))


def finite_intensity_full(
    layers: int,
    f_plus: complex,
    f_minus: complex,
    omega: complex,
    vertical_phase: complex,
    law: TransitionLaw,
    initial: InitialPopulation,
) -> FiniteIntensity:
    """Transparent six-state finite self/pair oracle for one event."""

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
    transition = full_transition_matrix(law)
    population = np.array([initial.plus, 0.0, 0.0, initial.minus, 0.0, 0.0], dtype=np.float64)
    populations = [population]
    for _ in range(1, count):
        populations.append(populations[-1] @ transition)
    try:
        self_terms = tuple(float(current @ np.abs(amplitudes) ** 2) for current in populations)
        pair_real_terms: list[float] = []
        pair_absolute_terms: list[float] = []
        transition_power = np.eye(6, dtype=np.float64)
        vertical_power = 1.0 + 0.0j
        for separation in range(1, count):
            transition_power = transition_power @ transition
            vertical_power *= complex(phase_array)
            propagated = transition_power @ amplitudes
            separation_terms = tuple(
                complex(vertical_power * (populations[start] @ (np.conj(amplitudes) * propagated)))
                for start in range(count - separation)
            )
            pair_real_terms.append(fsum(2.0 * term.real for term in separation_terms))
            pair_absolute_terms.append(fsum(2.0 * abs(term) for term in separation_terms))
        summed_intensity = fsum(chain(self_terms, pair_real_terms))
        cancellation_scale = fsum(chain(self_terms, pair_absolute_terms))
    except (OverflowError, ValueError) as error:
        raise ValueError("full-state intensity must remain finite") from error
    total = _clip_cancellation_roundoff(summed_intensity, cancellation_scale)
    return FiniteIntensity(np.asarray(total), np.asarray(total / count))


def stationary_intensity_reduced(
    f_plus: ArrayLike,
    f_minus: ArrayLike,
    omega: ArrayLike,
    vertical_phase: ArrayLike,
    law: TransitionLaw,
    stationary_population: InitialPopulation,
    *,
    correlation_decay: float,
) -> NDArray[np.float64]:
    """Return separately named stationary per-layer intensity with explicit damping."""

    decay = float(correlation_decay)
    if not np.isfinite(decay) or not 0.0 < decay <= 1.0:
        raise ValueError("correlation_decay must be finite, positive, and at most one")
    if not np.allclose(
        stationary_population.as_array() @ orientation_transition_matrix(law),
        stationary_population.as_array(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("stationary_population must be invariant under the orientation law")
    f_plus_array, f_minus_array, omega_array, phase_array = _broadcast_inputs(
        f_plus, f_minus, omega, vertical_phase
    )
    amplitudes = np.stack((f_plus_array, f_minus_array), axis=-1)
    transition = reduced_transition_matrix(law, omega_array)
    z_transition = decay * phase_array[..., None, None] * transition
    identity = np.broadcast_to(np.eye(2, dtype=np.complex128), z_transition.shape)
    system = identity - z_transition
    try:
        minimum_singular_value = np.linalg.svd(system, compute_uv=False)[..., -1]
        if np.any(~np.isfinite(minimum_singular_value)) or np.any(minimum_singular_value == 0.0):
            raise ValueError("stationary intensity is singular at this phase")
        propagated = np.linalg.solve(
            system,
            np.einsum("...ij,...j->...i", z_transition, amplitudes, optimize=True),
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("stationary intensity is singular at this phase") from error
    population = stationary_population.as_array()
    self_term = np.einsum("i,...i->...", population, np.abs(amplitudes) ** 2, optimize=True)
    pair_term = np.einsum(
        "i,...i->...", population, np.conj(amplitudes) * propagated, optimize=True
    )
    intensity = _clip_cancellation_roundoff(
        self_term + 2.0 * pair_term.real,
        np.abs(self_term) + 2.0 * np.abs(pair_term) * np.maximum(1.0, 1.0 / minimum_singular_value),
    )
    return _readonly_nonnegative(intensity, "stationary intensity")


def _event_phases(
    query: RodQueryBatch,
    layer_repeat_A: float,
    phase_model: RegistryPhaseModel,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    if isinstance(layer_repeat_A, bool) or not isinstance(layer_repeat_A, Real):
        raise ValueError("layer_repeat_A must be a real scalar")
    try:
        repeat = float(layer_repeat_A)
    except (OverflowError, ValueError) as error:
        raise ValueError("layer_repeat_A must be finite and positive") from error
    if not np.isfinite(repeat) or repeat <= 0.0:
        raise ValueError("layer_repeat_A must be finite and positive")
    omega = np.asarray(registry_phase(query.h, query.k, phase_model), dtype=np.complex128)
    with np.errstate(over="ignore", invalid="ignore"):
        phase_argument = query.qz_Ainv * repeat
    if np.any(~np.isfinite(phase_argument)):
        raise ValueError("qz_Ainv * layer_repeat_A must be finite")
    vertical = np.exp(1j * phase_argument)
    return omega, vertical


def _aligned_amplitudes(
    query: RodQueryBatch, amplitudes: LayerAmplitudeResult
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    if not np.array_equal(query.event_id, amplitudes.event_id):
        raise ValueError("layer amplitudes must be exactly event-aligned")
    if amplitudes.f_minus is None:
        raise ValueError("stacking intensity requires both f_plus and f_minus")
    return amplitudes.f_plus, amplitudes.f_minus


def _identifier(value: object, name: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        optional = " or None" if allow_none else ""
        raise ValueError(f"{name} must be a nonempty string{optional}")
    return value


def finite_event_intensity(
    query: RodQueryBatch,
    amplitudes: LayerAmplitudeResult,
    law: TransitionLaw,
    *,
    layers: int,
    layer_repeat_A: float,
    initial: InitialPopulation,
    normalization: FiniteNormalization,
    model_component_id: str,
    population_group_id: str | None,
    phase_model: RegistryPhaseModel = RegistryPhaseModel.FORWARD_H_PLUS_2K,
) -> EventIntensityResult:
    """Return event-aligned total electron2 or its explicit per-layer quotient."""

    normalization = FiniteNormalization(normalization)
    component_id = _identifier(model_component_id, "model_component_id")
    group_id = _identifier(population_group_id, "population_group_id", allow_none=True)
    f_plus, f_minus = _aligned_amplitudes(query, amplitudes)
    omega, vertical = _event_phases(query, layer_repeat_A, phase_model)
    result = finite_intensity_reduced(layers, f_plus, f_minus, omega, vertical, law, initial)
    selected = (
        result.intensity_electron2
        if normalization is FiniteNormalization.TOTAL
        else result.intensity_per_layer_electron2
    )
    return EventIntensityResult(
        query.event_id,
        selected,
        "stacking-transition-finite-v1",
        component_id,
        group_id,
        normalization.value,
    )


def finite_population_event_intensity(
    query: RodQueryBatch,
    amplitudes: LayerAmplitudeResult,
    populations: tuple[StackingPopulation, ...],
    *,
    layers: int,
    layer_repeat_A: float,
    initial: InitialPopulation,
    population_group_id: str,
    phase_model: RegistryPhaseModel = RegistryPhaseModel.FORWARD_H_PLUS_2K,
) -> PopulationIntensityResult:
    """Return unweighted components and their incoherent total in electron2."""

    count = _layers(layers)
    group_id = _identifier(population_group_id, "population_group_id")
    supplied = tuple(populations)
    if not supplied:
        raise ValueError("populations must be nonempty")
    if any(not isinstance(population, StackingPopulation) for population in supplied):
        raise TypeError("populations must contain StackingPopulation values")
    ordered = tuple(sorted(supplied, key=lambda population: population.population_id))
    population_id = tuple(population.population_id for population in ordered)
    if len(set(population_id)) != len(population_id):
        raise ValueError("population_id values must be unique")
    weight = np.array([population.weight for population in ordered], dtype=np.float64)
    if not np.isclose(weight.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("population weights must sum to one")
    f_plus, f_minus = _aligned_amplitudes(query, amplitudes)
    omega, vertical = _event_phases(query, layer_repeat_A, phase_model)
    component = np.stack(
        [
            finite_intensity_reduced(
                count,
                f_plus,
                f_minus,
                omega,
                vertical,
                population.model,
                initial,
            ).intensity_electron2
            for population in ordered
        ]
    )
    total = np.einsum("p,pe->e", weight, component, optimize=True)
    event_intensity = EventIntensityResult(
        query.event_id,
        total,
        "stacking-transition-population-finite-v1",
        "weighted-stacking-population-total",
        group_id,
        FiniteNormalization.TOTAL.value,
    )
    return PopulationIntensityResult(
        population_id,
        weight,
        component,
        total,
        event_intensity,
    )
