"""Direct short-sequence enumeration oracle."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from rasim_next.core.contracts import LayerAmplitudeResult, RodQueryBatch
from rasim_next.stacking.finite_intensity import (
    FiniteIntensity,
    _aligned_amplitudes,
    _broadcast_inputs,
    _event_phases,
    _layers,
)
from rasim_next.stacking.transition import (
    STATE_ORDER,
    InitialPopulation,
    RegistryPhaseModel,
    StackingState,
    TransitionLaw,
    full_transition_matrix,
)


def finite_explicit_sequence_intensity(
    query: RodQueryBatch,
    amplitudes: LayerAmplitudeResult,
    states: tuple[StackingState, ...],
    layer_depth_A: ArrayLike,
    *,
    layers: int,
    layer_repeat_A: float,
    phase_model: RegistryPhaseModel = RegistryPhaseModel.FORWARD_H_PLUS_2K,
) -> FiniteIntensity:
    """Directly sum one declared orientation/registry state at each physical depth."""

    count = _layers(layers)
    if (
        not isinstance(states, tuple)
        or len(states) != count
        or any(not isinstance(state, StackingState) for state in states)
    ):
        raise ValueError("states must contain one StackingState per layer")
    state_indices = tuple(STATE_ORDER.index(state) for state in states)

    depths = np.array(layer_depth_A, dtype=np.float64, copy=True, order="C")
    if depths.shape != (count,) or np.any(~np.isfinite(depths)):
        raise ValueError("layer_depth_A must contain one finite depth per layer")
    f_plus, f_minus = _aligned_amplitudes(query, amplitudes)
    omega, _ = _event_phases(query, layer_repeat_A, phase_model)
    repeat = float(layer_repeat_A)
    if count > 1:
        spacing = np.diff(depths)
        tolerance = (
            256.0
            * np.finfo(np.float64).eps
            * np.maximum(1.0, np.maximum(np.abs(spacing), abs(repeat)))
        )
        if np.any(np.abs(spacing - repeat) > tolerance):
            raise ValueError("layer_depth_A must advance by layer_repeat_A")

    coherent_amplitude = np.zeros(query.event_id.size, dtype=np.complex128)
    with np.errstate(over="ignore", invalid="ignore"):
        for state_index, depth in zip(state_indices, depths, strict=True):
            orientation_amplitude = f_plus if state_index < 3 else f_minus
            registry_factor = omega ** (state_index % 3)
            phase_argument = query.qz_Ainv * depth
            if np.any(~np.isfinite(phase_argument)):
                raise ValueError("qz_Ainv * layer_depth_A must be finite")
            coherent_amplitude += (
                orientation_amplitude * registry_factor * np.exp(1j * phase_argument)
            )
        intensity = np.abs(coherent_amplitude) ** 2
    return FiniteIntensity(intensity, intensity / float(count))


def finite_intensity_by_enumeration(
    layers: int,
    f_plus: complex,
    f_minus: complex,
    omega: complex,
    vertical_phase: complex,
    law: TransitionLaw,
    initial: InitialPopulation,
) -> FiniteIntensity:
    """Enumerate every nonzero state path; intended only for short proof sequences."""

    count = _layers(layers)
    if count > 10:
        raise ValueError("direct sequence oracle is limited to ten layers")
    f_plus_array, f_minus_array, omega_array, phase_array = _broadcast_inputs(
        f_plus, f_minus, omega, vertical_phase
    )
    if f_plus_array.ndim or f_minus_array.ndim or omega_array.ndim or phase_array.ndim:
        raise ValueError("direct sequence oracle accepts one event")
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
    paths = []
    if initial.plus:
        paths.append((0, initial.plus, amplitudes[0]))
    if initial.minus:
        paths.append((3, initial.minus, amplitudes[3]))
    transition = full_transition_matrix(law)
    vertical_power = 1.0 + 0.0j
    for _layer in range(1, count):
        vertical_power *= complex(phase_array)
        next_paths: list[tuple[int, float, complex]] = []
        for current_state, path_probability, path_amplitude in paths:
            for next_state, probability in enumerate(transition[current_state]):
                if probability:
                    next_paths.append(
                        (
                            next_state,
                            path_probability * float(probability),
                            path_amplitude + vertical_power * amplitudes[next_state],
                        )
                    )
        paths = next_paths
    total_probability = sum(item[1] for item in paths)
    if not np.isclose(total_probability, 1.0, rtol=0.0, atol=2e-12):
        raise ValueError("enumerated path probabilities do not sum to one")
    total = sum(probability * abs(amplitude) ** 2 for _, probability, amplitude in paths)
    return FiniteIntensity(np.asarray(total), np.asarray(total / count))
