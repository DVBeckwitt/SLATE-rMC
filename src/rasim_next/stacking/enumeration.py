"""Direct short-sequence enumeration oracle."""

from __future__ import annotations

import numpy as np

from rasim_next.stacking.finite_intensity import FiniteIntensity, _broadcast_inputs, _layers
from rasim_next.stacking.transition import InitialPopulation, TransitionLaw, full_transition_matrix


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
