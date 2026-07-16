from __future__ import annotations

import numpy as np
import pytest

from rasim_next.core.contracts import LayerAmplitudeResult, RodQueryBatch
from rasim_next.stacking import (
    STATE_ORDER,
    InitialPopulation,
    Parent,
    PopulationIntensityResult,
    StackingPopulation,
    TransitionLaw,
    finite_event_intensity,
    finite_intensity_by_enumeration,
    finite_intensity_full,
    finite_intensity_reduced,
    finite_population_event_intensity,
    full_transition_matrix,
    registry_phase,
)


def _query_and_amplitudes() -> tuple[RodQueryBatch, LayerAmplitudeResult]:
    query = RodQueryBatch(
        np.array([101, 102], dtype=np.int64),
        np.array([5, 6], dtype=np.int64),
        ("pbi2", "pbi2"),
        np.array([0, 1], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
        np.array([0.21, 0.37]),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
    )
    amplitudes = LayerAmplitudeResult(
        query.event_id,
        np.array([1.1 + 0.2j, 0.8 - 0.1j]),
        np.array([0.7 - 0.3j, 1.0 + 0.4j]),
    )
    return query, amplitudes


def test_transition_matrix_is_stochastic_in_the_declared_state_order() -> None:
    assert tuple(state.value for state in STATE_ORDER) == (
        "0F+",
        "1F+",
        "2F+",
        "0F-",
        "1F-",
        "2F-",
    )
    law = TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18)
    transition = full_transition_matrix(law)
    assert transition.shape == (6, 6)
    assert np.all(transition >= 0.0)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)


def test_short_stack_direct_full_and_reduced_results_agree() -> None:
    layers = 6
    f_plus = 1.2 + 0.3j
    f_minus = 0.7 - 0.5j
    omega = complex(registry_phase(1, 0))
    vertical_phase = np.exp(0.43j)
    law = TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18)
    initial = InitialPopulation(0.8, 0.2)
    arguments = (layers, f_plus, f_minus, omega, vertical_phase, law, initial)
    direct = finite_intensity_by_enumeration(*arguments).intensity_electron2
    full = finite_intensity_full(*arguments).intensity_electron2
    reduced = finite_intensity_reduced(*arguments).intensity_electron2
    np.testing.assert_allclose(full, direct, rtol=5e-13, atol=5e-13)
    np.testing.assert_allclose(reduced, direct, rtol=5e-13, atol=5e-13)


def test_single_layer_uses_only_the_declared_initial_population() -> None:
    f_plus = 1.2 + 0.3j
    f_minus = 0.8 - 0.4j
    initial = InitialPopulation(0.27, 0.73)
    expected = initial.plus * abs(f_plus) ** 2 + initial.minus * abs(f_minus) ** 2
    arguments = (
        1,
        f_plus,
        f_minus,
        complex(registry_phase(1, 0)),
        np.exp(0.44j),
        TransitionLaw(0.13, 0.21, 0.08, 0.34, 0.24),
        initial,
    )
    for evaluator in (finite_intensity_full, finite_intensity_reduced):
        np.testing.assert_allclose(
            evaluator(*arguments).intensity_electron2,
            expected,
            rtol=5e-13,
            atol=5e-13,
        )


def test_coherent_extinction_never_erases_nearby_positive_intensity() -> None:
    law = TransitionLaw.for_parent(Parent.TWO_H)
    initial = InitialPopulation.plus_only()
    for layers in (8, 64, 512):
        exact_phase = 2.0 * np.pi / layers
        nearby_offset = 1e-9 * (512 / layers) ** 2
        for phase_offset in (-nearby_offset, 0.0, nearby_offset):
            vertical_phase = np.exp(1j * (exact_phase + phase_offset))
            direct = abs(sum(vertical_phase**layer for layer in range(layers))) ** 2
            for evaluator in (finite_intensity_full, finite_intensity_reduced):
                observed = evaluator(
                    layers,
                    1.0,
                    1.0,
                    1.0,
                    vertical_phase,
                    law,
                    initial,
                ).intensity_electron2
                assert observed >= 0.0
                if phase_offset:
                    assert direct > 0.0
                    assert observed > 0.0
                np.testing.assert_allclose(observed, direct, rtol=5e-13, atol=5e-13)


def test_raw_event_intensity_is_aligned_and_explicitly_normalized() -> None:
    query, amplitudes = _query_and_amplitudes()
    result = finite_event_intensity(
        query,
        amplitudes,
        TransitionLaw.for_parent(Parent.TWO_H),
        layers=9,
        layer_repeat_A=3.4,
        initial=InitialPopulation.plus_only(),
        model_component_id="2H",
        population_group_id=None,
    )
    np.testing.assert_array_equal(result.event_id, query.event_id)
    np.testing.assert_allclose(
        result.intensity_electron2,
        9.0 * result.intensity_per_layer_electron2,
        rtol=5e-13,
        atol=5e-13,
    )
    assert not hasattr(result, "intensity_per_sr")
    misaligned = LayerAmplitudeResult(query.event_id[::-1], amplitudes.f_plus, amplitudes.f_minus)
    with pytest.raises(ValueError, match="event-aligned"):
        finite_event_intensity(
            query,
            misaligned,
            TransitionLaw.for_parent(Parent.TWO_H),
            layers=9,
            layer_repeat_A=3.4,
            initial=InitialPopulation.plus_only(),
            model_component_id="2H",
            population_group_id=None,
        )


def test_population_components_are_unweighted_and_use_individual_initial_states() -> None:
    query, amplitudes = _query_and_amplitudes()
    populations = (
        StackingPopulation(
            "2H",
            TransitionLaw.for_parent(Parent.TWO_H),
            InitialPopulation.minus_only(),
        ),
        StackingPopulation(
            "4H",
            TransitionLaw.for_parent(Parent.FOUR_H_PLUS),
            InitialPopulation.plus_only(),
        ),
    )
    arguments = {
        "query": query,
        "amplitudes": amplitudes,
        "layers": 6,
        "layer_repeat_A": 3.4,
        "population_group_id": "parents",
    }
    result = finite_population_event_intensity(populations=populations, **arguments)
    reversed_result = finite_population_event_intensity(
        populations=tuple(reversed(populations)), **arguments
    )
    assert isinstance(result, PopulationIntensityResult)
    assert result.population_id == ("2H", "4H")
    np.testing.assert_array_equal(result.event_id, query.event_id)
    np.testing.assert_array_equal(
        result.component_intensity_electron2,
        reversed_result.component_intensity_electron2,
    )
    omega = np.asarray(registry_phase(query.h, query.k))
    vertical_phase = np.exp(1j * query.qz_Ainv * 3.4)
    expected = np.array(
        [
            [
                finite_intensity_by_enumeration(
                    6,
                    amplitudes.f_plus[event],
                    amplitudes.f_minus[event],
                    omega[event],
                    vertical_phase[event],
                    population.model,
                    population.initial,
                ).intensity_electron2
                for event in range(query.event_id.size)
            ]
            for population in populations
        ]
    )
    np.testing.assert_allclose(
        result.component_intensity_electron2,
        expected,
        rtol=5e-13,
        atol=5e-13,
    )
    assert not hasattr(result, "weight")
    assert not hasattr(result, "weighted_total_intensity_electron2")
