"""Finite full-state and exact reduced stacking intensities."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import LayerAmplitudeResult, RodQueryBatch
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


def _normalization_is_consistent(
    total: NDArray[np.float64], per_layer: NDArray[np.float64], layers: int
) -> bool:
    reconstructed = layers * per_layer
    tolerance = 8.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(total))
    return bool(np.all(np.abs(total - reconstructed) <= tolerance))


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
class StackingEventIntensityResult:
    """Event-aligned finite-stack intensity in electron2, without solid-angle semantics."""

    event_id: NDArray[np.int64]
    layers: int
    intensity_electron2: NDArray[np.float64]
    intensity_per_layer_electron2: NDArray[np.float64]
    model_component_id: str
    population_group_id: str | None

    def __post_init__(self) -> None:
        event_id = np.array(self.event_id, dtype=np.int64, copy=True, order="C")
        count = _layers(self.layers)
        finite = FiniteIntensity(
            self.intensity_electron2,
            self.intensity_per_layer_electron2,
        )
        if event_id.ndim != 1 or finite.intensity_electron2.shape != event_id.shape:
            raise ValueError("event intensity must be one-dimensional and event-aligned")
        if not _normalization_is_consistent(
            finite.intensity_electron2,
            finite.intensity_per_layer_electron2,
            count,
        ):
            raise ValueError("total and per-layer event intensities are inconsistent")
        event_id.setflags(write=False)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "layers", count)
        object.__setattr__(self, "intensity_electron2", finite.intensity_electron2)
        object.__setattr__(
            self,
            "intensity_per_layer_electron2",
            finite.intensity_per_layer_electron2,
        )
        object.__setattr__(
            self,
            "model_component_id",
            _identifier(self.model_component_id, "model_component_id"),
        )
        object.__setattr__(
            self,
            "population_group_id",
            _identifier(self.population_group_id, "population_group_id", allow_none=True),
        )


@dataclass(frozen=True, slots=True)
class PopulationIntensityResult:
    """Event-aligned unweighted intensity components for independent populations."""

    event_id: NDArray[np.int64]
    population_id: tuple[str, ...]
    layers: int
    component_intensity_electron2: NDArray[np.float64]
    component_intensity_per_layer_electron2: NDArray[np.float64]
    population_group_id: str

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
        event_id = np.array(self.event_id, dtype=np.int64, copy=True, order="C")
        count = _layers(self.layers)
        component = _readonly_nonnegative(
            self.component_intensity_electron2,
            "component_intensity_electron2",
        )
        per_layer = _readonly_nonnegative(
            self.component_intensity_per_layer_electron2,
            "component_intensity_per_layer_electron2",
        )
        expected_shape = (len(population_id), event_id.size)
        if event_id.ndim != 1 or component.shape != expected_shape or per_layer.shape != expected_shape:
            raise ValueError("population intensity arrays have inconsistent shapes")
        if not _normalization_is_consistent(component, per_layer, count):
            raise ValueError("population total and per-layer component intensities are inconsistent")
        event_id.setflags(write=False)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "population_id", population_id)
        object.__setattr__(self, "layers", count)
        object.__setattr__(self, "component_intensity_electron2", component)
        object.__setattr__(self, "component_intensity_per_layer_electron2", per_layer)
        object.__setattr__(
            self,
            "population_group_id",
            _identifier(self.population_group_id, "population_group_id"),
        )


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
                    candidate_mean = (
                        registry_gauge * mean[..., source] + contribution[..., target]
                    )
                    next_variance[..., target] += weight * (
                        variance[..., source]
                        + np.abs(candidate_mean - next_mean[..., target]) ** 2
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
) -> FiniteIntensity:
    """Evaluate an exact two-orientation finite moment recurrence."""

    count = _layers(layers)
    f_plus_array, f_minus_array, omega_array, phase_array = _broadcast_inputs(
        f_plus, f_minus, omega, vertical_phase
    )
    total = _reduced_moment_intensity(
        count,
        f_plus_array,
        f_minus_array,
        omega_array,
        phase_array,
        law,
        initial,
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
    total = _full_moment_intensity(count, amplitudes, phase_value, transition, initial)
    return FiniteIntensity(total, total / count)


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
    model_component_id: str,
    population_group_id: str | None,
    phase_model: RegistryPhaseModel = RegistryPhaseModel.FORWARD_H_PLUS_2K,
) -> StackingEventIntensityResult:
    """Return event-aligned electron2 values without claiming a per-steradian measure."""

    component_id = _identifier(model_component_id, "model_component_id")
    group_id = _identifier(population_group_id, "population_group_id", allow_none=True)
    f_plus, f_minus = _aligned_amplitudes(query, amplitudes)
    omega, vertical = _event_phases(query, layer_repeat_A, phase_model)
    result = finite_intensity_reduced(layers, f_plus, f_minus, omega, vertical, law, initial)
    return StackingEventIntensityResult(
        query.event_id,
        layers,
        result.intensity_electron2,
        result.intensity_per_layer_electron2,
        component_id,
        group_id,
    )


def finite_population_event_intensity(
    query: RodQueryBatch,
    amplitudes: LayerAmplitudeResult,
    populations: tuple[StackingPopulation, ...],
    *,
    layers: int,
    layer_repeat_A: float,
    population_group_id: str,
    phase_model: RegistryPhaseModel = RegistryPhaseModel.FORWARD_H_PLUS_2K,
) -> PopulationIntensityResult:
    """Return event-aligned unweighted population components in electron2."""

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
                population.initial,
            ).intensity_electron2
            for population in ordered
        ]
    )
    return PopulationIntensityResult(
        query.event_id,
        population_id,
        count,
        component,
        component / float(count),
        group_id,
    )
