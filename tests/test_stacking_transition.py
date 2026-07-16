from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rasim_next.core.contracts import (
    EventIntensityNormalization,
    LayerAmplitudeNormalization,
    LayerAmplitudeResult,
    LayerNormalQBatch,
    LayerPhaseSign,
    RodQueryBatch,
)
from rasim_next.core.scattering import electron_squared_to_scattering_strength_A2
from rasim_next.proof.tolerances import load_stage_tolerances
from rasim_next.stacking import (
    STATE_ORDER,
    InitialPopulation,
    Parent,
    StackingPopulation,
    StackingState,
    TransitionLaw,
    finite_event_intensity,
    finite_population_event_intensity,
    full_transition_matrix,
    registry_phase,
)
from rasim_next.stacking.enumeration import (
    finite_explicit_sequence_intensity,
    finite_intensity_by_enumeration,
)
from rasim_next.stacking.finite_intensity import (
    finite_intensity_full,
    finite_intensity_reduced,
)

_TOLERANCES = load_stage_tolerances()


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
        event_id=query.event_id,
        rod_id=query.rod_id,
        phase_id=query.phase_id,
        f_plus_e=np.array([1.1 + 0.2j, 0.8 - 0.1j]),
        f_minus_e=np.array([0.7 - 0.3j, 1.0 + 0.4j]),
        normalization=LayerAmplitudeNormalization.ONE_REGISTRY_FREE_LAYER,
        phase_sign=LayerPhaseSign.POSITIVE_Q_DOT_R,
        gauge_id="pbi2.pb_centered.v1",
        layer_normal_crystal=np.array([0.0, 0.0, 1.0]),
        layer_repeat_A=3.4,
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
    direct = finite_intensity_by_enumeration(*arguments)
    limit = (
        _TOLERANCES["stacking.pair_kernel"]
        .bind(layers**2 * max(abs(f_plus), abs(f_minus)) ** 2)
        .limit
    )
    np.testing.assert_allclose(finite_intensity_full(*arguments), direct, rtol=0.0, atol=limit)
    np.testing.assert_allclose(finite_intensity_reduced(*arguments), direct, rtol=0.0, atol=limit)


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
    limit = _TOLERANCES["stacking.pair_kernel"].bind(max(abs(f_plus), abs(f_minus)) ** 2).limit
    for evaluator in (finite_intensity_full, finite_intensity_reduced):
        np.testing.assert_allclose(evaluator(*arguments), expected, rtol=0.0, atol=limit)


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
                )
                assert observed >= 0.0
                if phase_offset:
                    assert direct > 0.0
                    assert observed > 0.0
                limit = _TOLERANCES["stacking.pair_kernel"].bind(float(layers**2)).limit
                np.testing.assert_allclose(observed, direct, rtol=0.0, atol=limit)


def test_event_intensity_uses_shared_alignment_measure_and_layer_normal_q() -> None:
    query, amplitudes = _query_and_amplitudes()
    layer_normal_q = LayerNormalQBatch(
        event_id=query.event_id,
        rod_id=query.rod_id,
        phase_id=query.phase_id,
        layer_normal_q_Ainv=np.array([0.63, -0.28]),
        gauge_id=amplitudes.gauge_id,
    )
    layers = 9
    total = finite_event_intensity(
        query,
        amplitudes,
        TransitionLaw.for_parent(Parent.TWO_H),
        layer_normal_q=layer_normal_q,
        layers=layers,
        initial=InitialPopulation.plus_only(),
        model_component_id="2H",
        population_group_id=None,
        normalization=EventIntensityNormalization.FINITE_TOTAL,
    )
    per_layer = finite_event_intensity(
        query,
        amplitudes,
        TransitionLaw.for_parent(Parent.TWO_H),
        layer_normal_q=layer_normal_q,
        layers=layers,
        initial=InitialPopulation.plus_only(),
        model_component_id="2H",
        population_group_id=None,
        normalization=EventIntensityNormalization.FINITE_PER_LAYER,
    )
    direct = finite_explicit_sequence_intensity(
        query,
        amplitudes,
        (StackingState.REGISTRY_0_PLUS,) * layers,
        np.arange(layers) * amplitudes.layer_repeat_A,
        layer_normal_q=layer_normal_q,
        layers=layers,
    )
    layer_frame_amplitude = amplitudes.f_plus_e * np.exp(
        1j
        * layer_normal_q.layer_normal_q_Ainv[:, np.newaxis]
        * np.arange(layers)
        * amplitudes.layer_repeat_A
    ).sum(axis=1)
    expected_raw = np.abs(layer_frame_amplitude) ** 2
    expected_A2 = electron_squared_to_scattering_strength_A2(expected_raw)
    raw_scale = layers**2 * float(np.max(np.abs(amplitudes.f_plus_e)) ** 2)
    raw_limit = _TOLERANCES["stacking.pair_kernel"].bind(raw_scale).limit
    A2_scale = float(electron_squared_to_scattering_strength_A2([raw_scale])[0])
    A2_limit = _TOLERANCES["stacking.finite_intensity"].bind(A2_scale).limit
    np.testing.assert_array_equal(total.event_id, query.event_id)
    np.testing.assert_allclose(total.scattering_strength_A2, expected_A2, rtol=0.0, atol=A2_limit)
    np.testing.assert_allclose(direct, expected_raw, rtol=0.0, atol=raw_limit)
    np.testing.assert_allclose(
        total.scattering_strength_A2,
        layers * per_layer.scattering_strength_A2,
        rtol=0.0,
        atol=A2_limit,
    )
    assert total.model_id == "stacking"
    assert total.model_component_id == "2H"
    assert total.population_group_id is None
    assert total.normalization is EventIntensityNormalization.FINITE_TOTAL
    assert per_layer.normalization is EventIntensityNormalization.FINITE_PER_LAYER
    sample_frame_amplitude = amplitudes.f_plus_e * np.exp(
        1j
        * query.q_sample_normal_Ainv[:, np.newaxis]
        * np.arange(layers)
        * amplitudes.layer_repeat_A
    ).sum(axis=1)
    assert not np.allclose(
        total.scattering_strength_A2,
        electron_squared_to_scattering_strength_A2(np.abs(sample_frame_amplitude) ** 2),
        rtol=0.0,
        atol=A2_limit,
    )
    assert not hasattr(total, "intensity_electron2")
    assert not hasattr(total, "intensity_per_sr")

    bad_inputs = (
        (replace(amplitudes, event_id=amplitudes.event_id[::-1]), layer_normal_q),
        (replace(amplitudes, rod_id=amplitudes.rod_id[::-1]), layer_normal_q),
        (replace(amplitudes, phase_id=("other",) * 2), layer_normal_q),
        (replace(amplitudes, gauge_id="other.gauge.v1"), layer_normal_q),
        (amplitudes, replace(layer_normal_q, event_id=layer_normal_q.event_id[::-1])),
        (amplitudes, replace(layer_normal_q, rod_id=layer_normal_q.rod_id[::-1])),
        (amplitudes, replace(layer_normal_q, phase_id=("other",) * 2)),
        (amplitudes, replace(layer_normal_q, gauge_id="other.gauge.v1")),
    )
    for bad_amplitudes, bad_q in bad_inputs:
        with pytest.raises(ValueError):
            finite_event_intensity(
                query,
                bad_amplitudes,
                TransitionLaw.for_parent(Parent.TWO_H),
                layer_normal_q=bad_q,
                layers=layers,
                initial=InitialPopulation.plus_only(),
                model_component_id="2H",
                population_group_id=None,
                normalization=EventIntensityNormalization.FINITE_TOTAL,
            )
    with pytest.raises(ValueError, match="FINITE_TOTAL or FINITE_PER_LAYER"):
        finite_event_intensity(
            query,
            amplitudes,
            TransitionLaw.for_parent(Parent.TWO_H),
            layer_normal_q=layer_normal_q,
            layers=layers,
            initial=InitialPopulation.plus_only(),
            model_component_id="2H",
            population_group_id=None,
            normalization=EventIntensityNormalization.UNIT_CELL,
        )


def test_population_components_are_unweighted_and_use_individual_initial_states() -> None:
    query, amplitudes = _query_and_amplitudes()
    layer_normal_q = LayerNormalQBatch(
        event_id=query.event_id,
        rod_id=query.rod_id,
        phase_id=query.phase_id,
        layer_normal_q_Ainv=np.array([0.63, -0.28]),
        gauge_id=amplitudes.gauge_id,
    )
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
        "layer_normal_q": layer_normal_q,
        "layers": 6,
        "population_group_id": "parents",
        "normalization": EventIntensityNormalization.FINITE_TOTAL,
    }
    result = finite_population_event_intensity(populations=populations, **arguments)
    reversed_result = finite_population_event_intensity(
        populations=tuple(reversed(populations)), **arguments
    )
    assert tuple(component.model_component_id for component in result) == ("2H", "4H")
    assert all(component.model_id == "stacking" for component in result)
    assert all(component.population_group_id == "parents" for component in result)
    assert all(
        component.normalization is EventIntensityNormalization.FINITE_TOTAL for component in result
    )
    assert all(np.array_equal(component.event_id, query.event_id) for component in result)
    np.testing.assert_array_equal(
        np.stack([component.scattering_strength_A2 for component in result]),
        np.stack([component.scattering_strength_A2 for component in reversed_result]),
    )
    omega = np.asarray(registry_phase(query.h, query.k))
    vertical_phase = np.exp(1j * layer_normal_q.layer_normal_q_Ainv * amplitudes.layer_repeat_A)
    expected_raw = np.array(
        [
            [
                finite_intensity_by_enumeration(
                    6,
                    amplitudes.f_plus_e[event],
                    amplitudes.f_minus_e[event],
                    omega[event],
                    vertical_phase[event],
                    population.model,
                    population.initial,
                )
                for event in range(query.event_id.size)
            ]
            for population in populations
        ]
    )
    expected_A2 = electron_squared_to_scattering_strength_A2(expected_raw)
    scale_raw = (
        len(populations)
        * 6**2
        * max(
            float(np.max(np.abs(amplitudes.f_plus_e) ** 2)),
            float(np.max(np.abs(amplitudes.f_minus_e) ** 2)),
        )
    )
    scale_A2 = float(electron_squared_to_scattering_strength_A2([scale_raw])[0])
    limit = _TOLERANCES["stacking.population_intensity"].bind(scale_A2).limit
    np.testing.assert_allclose(
        np.stack([component.scattering_strength_A2 for component in result]),
        expected_A2,
        rtol=0.0,
        atol=limit,
    )
    assert all(not hasattr(component, "weight") for component in result)
    assert all(
        not hasattr(component, "weighted_total_scattering_strength_A2") for component in result
    )
