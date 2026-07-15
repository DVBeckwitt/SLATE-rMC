from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rasim_next.core.contracts import EventIntensityResult, LayerAmplitudeResult, RodQueryBatch
from rasim_next.stacking import (
    STATE_ORDER,
    FiniteIntensity,
    FiniteNormalization,
    InitialPopulation,
    Parent,
    PopulationIntensityResult,
    ReducedABDModel,
    RegistryPhaseModel,
    RichEpsilonModel,
    StackingPopulation,
    StackingState,
    TransitionLaw,
    finite_event_intensity,
    finite_explicit_sequence_intensity,
    finite_intensity_by_enumeration,
    finite_intensity_full,
    finite_intensity_reduced,
    finite_population_event_intensity,
    full_transition_matrix,
    reduced_transition_matrix,
    registry_phase,
    stationary_intensity_reduced,
)


def test_state_order_transition_convention_and_exact_reduction() -> None:
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
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(transition[0], [0.17, 0.23, 0.11, 0.0, 0.31, 0.18])

    omega = registry_phase(1, 0, RegistryPhaseModel.FORWARD_H_PLUS_2K)
    omega_plus = np.complex128(-0.5 + 0.5j * np.sqrt(3.0))
    omega_minus = np.complex128(-0.5 - 0.5j * np.sqrt(3.0))
    np.testing.assert_array_equal(omega, omega_plus)
    np.testing.assert_array_equal(registry_phase(0, 1), omega_minus)
    np.testing.assert_array_equal(
        registry_phase(1, 0, RegistryPhaseModel.LEGACY_2H_PLUS_K), omega_minus
    )
    np.testing.assert_array_equal(
        registry_phase(0, 1, RegistryPhaseModel.LEGACY_2H_PLUS_K), omega_plus
    )
    reduced = reduced_transition_matrix(law, omega)
    inverse = np.conj(omega)
    expected_reduced = np.array(
        [
            [
                law.a + law.b_plus * omega + law.b_minus * inverse,
                law.d_plus * omega + law.d_minus * inverse,
            ],
            [
                law.d_plus * inverse + law.d_minus * omega,
                law.a + law.b_plus * omega + law.b_minus * inverse,
            ],
        ]
    )
    np.testing.assert_allclose(reduced, expected_reduced, rtol=0.0, atol=1e-15)
    same_orientation = law.a + law.b_plus + law.b_minus
    orientation_flip = law.d_plus + law.d_minus
    np.testing.assert_allclose(
        reduced_transition_matrix(law, 1.0 + 0.0j),
        [[same_orientation, orientation_flip], [orientation_flip, same_orientation]],
        rtol=0.0,
        atol=1e-15,
    )
    f_plus, f_minus = 1.1 + 0.4j, 0.7 - 0.2j
    xi = np.exp(0.37j)
    initial = InitialPopulation(0.63, 0.37)
    full = finite_intensity_full(7, f_plus, f_minus, omega, xi, law, initial)
    exact = finite_intensity_reduced(7, f_plus, f_minus, omega, xi, law, initial)
    np.testing.assert_allclose(
        exact.intensity_electron2, full.intensity_electron2, rtol=2e-14, atol=2e-14
    )
    assert reduced.shape == (2, 2)
    np.testing.assert_array_equal(registry_phase(10_000, 0), registry_phase(1, 0))
    np.testing.assert_array_equal(
        reduced_transition_matrix(law, np.exp(5e-13j)),
        reduced_transition_matrix(law, 1.0 + 0.0j),
    )
    with np.testing.assert_raises(ValueError):
        reduced_transition_matrix(law, np.exp(0.2j))

    with pytest.raises(ValueError, match="negative"):
        FiniteIntensity(np.array([1e20, -1.0]), np.array([1e20, -1.0]))
    with pytest.raises(ValueError, match="sum to one"):
        TransitionLaw(0.2, 0.2, 0.2, 0.2, 0.1)
    with pytest.raises(ValueError, match="finite"):
        TransitionLaw(np.nan, 0.25, 0.25, 0.25, 0.25)
    with pytest.raises(ValueError, match="sum to one"):
        InitialPopulation(0.6, 0.3)
    with pytest.raises(ValueError, match="finite"):
        finite_intensity_full(
            3,
            1e154 + 0.0j,
            1e154 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            TransitionLaw.for_parent(Parent.TWO_H),
            InitialPopulation.plus_only(),
        )

    with pytest.raises(ValueError, match="singular"):
        stationary_intensity_reduced(
            1.0,
            1.0,
            1.0,
            1.0,
            TransitionLaw.for_parent(Parent.TWO_H),
            InitialPopulation.plus_only(),
            correlation_decay=1.0,
        )
    undamped = stationary_intensity_reduced(
        1.0,
        1.0,
        1.0,
        np.exp(0.2j),
        TransitionLaw.for_parent(Parent.TWO_H),
        InitialPopulation.plus_only(),
        correlation_decay=1.0,
    )
    assert np.isfinite(undamped) and undamped >= 0.0
    near_bragg = stationary_intensity_reduced(
        1.0,
        1.0,
        1.0,
        np.exp(1e-6j),
        TransitionLaw.for_parent(Parent.TWO_H),
        InitialPopulation.plus_only(),
        correlation_decay=1.0,
    )
    np.testing.assert_array_equal(near_bragg, 0.0)


def test_direct_sequence_and_full_pair_oracles_include_start_and_end_effects() -> None:
    law = TransitionLaw(0.28, 0.14, 0.19, 0.25, 0.14)
    initial = InitialPopulation(0.8, 0.2)
    f_plus, f_minus = 1.3 - 0.1j, 0.6 + 0.7j
    omega = registry_phase(2, -1)
    xi = np.exp(0.51j)
    for layers in range(1, 7):
        enumerated = finite_intensity_by_enumeration(
            layers, f_plus, f_minus, omega, xi, law, initial
        )
        full = finite_intensity_full(layers, f_plus, f_minus, omega, xi, law, initial)
        reduced = finite_intensity_reduced(layers, f_plus, f_minus, omega, xi, law, initial)
        np.testing.assert_allclose(
            full.intensity_electron2,
            enumerated.intensity_electron2,
            rtol=3e-14,
            atol=3e-14,
        )
        np.testing.assert_allclose(
            reduced.intensity_electron2,
            enumerated.intensity_electron2,
            rtol=3e-14,
            atol=3e-14,
        )


def test_explicit_layer_sequence_is_direct_event_aligned_coherent_sum() -> None:
    query = RodQueryBatch(
        np.array([41, 42, 43]),
        np.array([7, 8, 9]),
        ("pbi2", "pbi2", "pbi2"),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0, 0, 1], dtype=np.int32),
        np.array([-0.19, 0.43, 0.91]),
        np.zeros(3),
        np.ones(3),
    )
    amplitudes = LayerAmplitudeResult(
        query.event_id,
        np.array([1.2 + 0.3j, 0.8 - 0.4j, 1.1 + 0.2j]),
        np.array([0.7 - 0.1j, 1.0 + 0.5j, 0.6 - 0.3j]),
    )
    repeat_A = 6.986
    states = (
        StackingState.REGISTRY_0_PLUS,
        StackingState.REGISTRY_1_MINUS,
        StackingState.REGISTRY_0_PLUS,
        StackingState.REGISTRY_1_MINUS,
    )
    depths_A = 1.25 + repeat_A * np.arange(len(states))
    explicit = finite_explicit_sequence_intensity(
        query,
        amplitudes,
        states,
        depths_A,
        layers=len(states),
        layer_repeat_A=repeat_A,
    )

    omega = registry_phase(query.h, query.k)
    expected_amplitude = (
        amplitudes.f_plus * np.exp(1j * query.qz_Ainv * depths_A[0])
        + omega * amplitudes.f_minus * np.exp(1j * query.qz_Ainv * depths_A[1])
        + amplitudes.f_plus * np.exp(1j * query.qz_Ainv * depths_A[2])
        + omega * amplitudes.f_minus * np.exp(1j * query.qz_Ainv * depths_A[3])
    )
    np.testing.assert_allclose(
        explicit.intensity_electron2,
        np.abs(expected_amplitude) ** 2,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_array_equal(
        explicit.intensity_per_layer_electron2,
        explicit.intensity_electron2 / len(states),
    )
    for event in range(query.event_id.size):
        expected_transition = finite_intensity_reduced(
            len(states),
            amplitudes.f_plus[event],
            amplitudes.f_minus[event],
            omega[event],
            np.exp(1j * query.qz_Ainv[event] * repeat_A),
            TransitionLaw.for_parent(Parent.FOUR_H_PLUS),
            InitialPopulation.plus_only(),
        )
        np.testing.assert_allclose(
            explicit.intensity_electron2[event],
            expected_transition.intensity_electron2,
            rtol=2e-14,
            atol=2e-14,
        )

    with pytest.raises(ValueError, match="event-aligned"):
        finite_explicit_sequence_intensity(
            query,
            LayerAmplitudeResult(query.event_id[::-1], amplitudes.f_plus, amplitudes.f_minus),
            states,
            depths_A,
            layers=len(states),
            layer_repeat_A=repeat_A,
        )
    with pytest.raises(ValueError, match="states must contain"):
        finite_explicit_sequence_intensity(
            query,
            amplitudes,
            states[:-1],
            depths_A,
            layers=len(states),
            layer_repeat_A=repeat_A,
        )
    with pytest.raises(ValueError, match="states must contain"):
        finite_explicit_sequence_intensity(
            query,
            amplitudes,
            (*states[:-1], "0F+"),
            depths_A,
            layers=len(states),
            layer_repeat_A=repeat_A,
        )
    with pytest.raises(ValueError, match="advance by layer_repeat_A"):
        finite_explicit_sequence_intensity(
            query,
            amplitudes,
            states,
            depths_A + np.array([0.0, 0.0, 0.1, 0.1]),
            layers=len(states),
            layer_repeat_A=repeat_A,
        )


def test_finite_normalization_parent_limits_and_laue_identity() -> None:
    initial = InitialPopulation.plus_only()
    f_plus, f_minus = 1.2 + 0.3j, 0.8 - 0.4j
    omega = registry_phase(1, 0)
    xi = np.exp(0.44j)
    expected_parent_laws = {
        Parent.TWO_H: [1.0, 0.0, 0.0, 0.0, 0.0],
        Parent.FOUR_H_PLUS: [0.0, 0.0, 0.0, 1.0, 0.0],
        Parent.FOUR_H_MINUS: [0.0, 0.0, 0.0, 0.0, 1.0],
        Parent.SIX_H_PLUS: [0.0, 1.0, 0.0, 0.0, 0.0],
        Parent.SIX_H_MINUS: [0.0, 0.0, 1.0, 0.0, 0.0],
    }
    deterministic_cycles = {
        Parent.TWO_H: (f_plus,),
        Parent.FOUR_H_PLUS: (f_plus, omega * f_minus),
        Parent.FOUR_H_MINUS: (f_plus, omega**2 * f_minus),
        Parent.SIX_H_PLUS: (f_plus, omega * f_plus, omega**2 * f_plus),
        Parent.SIX_H_MINUS: (f_plus, omega**2 * f_plus, omega * f_plus),
    }
    for parent, expected_law in expected_parent_laws.items():
        law = TransitionLaw.for_parent(parent)
        np.testing.assert_array_equal(law.as_array(), expected_law)
        direct = finite_intensity_by_enumeration(6, f_plus, f_minus, omega, xi, law, initial)
        reduced = finite_intensity_reduced(6, f_plus, f_minus, omega, xi, law, initial)
        np.testing.assert_allclose(
            reduced.intensity_electron2,
            direct.intensity_electron2,
            rtol=2e-14,
            atol=2e-14,
        )
        cycle = deterministic_cycles[parent]
        expected = abs(sum(xi**n * cycle[n % len(cycle)] for n in range(6))) ** 2
        np.testing.assert_allclose(reduced.intensity_electron2, expected, rtol=2e-14, atol=2e-14)

    single = finite_intensity_reduced(1, f_plus, f_minus, omega, xi, law, initial)
    np.testing.assert_allclose(single.intensity_electron2, abs(f_plus) ** 2, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        single.intensity_electron2,
        single.intensity_per_layer_electron2,
        rtol=0.0,
        atol=0.0,
    )

    layers = 13
    amplitude = 1.4 - 0.2j
    laue = finite_intensity_reduced(
        layers,
        amplitude,
        amplitude,
        1.0 + 0.0j,
        xi,
        TransitionLaw(0.13, 0.21, 0.08, 0.34, 0.24),
        InitialPopulation(0.27, 0.73),
    )
    expected_total = abs(amplitude * sum(xi**n for n in range(layers))) ** 2
    np.testing.assert_allclose(laue.intensity_electron2, expected_total, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(
        laue.intensity_per_layer_electron2 * layers,
        laue.intensity_electron2,
        rtol=0.0,
        atol=0.0,
    )

    extinction_layers = 512
    extinction_phase = np.exp(2j * np.pi / extinction_layers)
    for evaluator in (finite_intensity_full, finite_intensity_reduced):
        extinction = evaluator(
            extinction_layers,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            extinction_phase,
            TransitionLaw.for_parent(Parent.TWO_H),
            InitialPopulation.plus_only(),
        )
        np.testing.assert_allclose(extinction.intensity_electron2, 0.0, rtol=0.0, atol=1e-24)

    # T04 exact 2H motif amplitudes at h=k=0, qz=0.4 A^-1, wavelength=1 A, Uiso=0.
    pbi2_f_plus_e = 150.17616504979657 + 8.935295728678174j
    pbi2_f_minus_e = 150.17616504979657 + 8.935295728678163j
    material_layers = 9
    material_vertical_phase = np.exp(1j * 0.40000000000000036 * 6.986)
    explicit_amplitude = sum(
        pbi2_f_plus_e * material_vertical_phase**layer for layer in range(material_layers)
    )
    direct_material_intensity = abs(explicit_amplitude) ** 2
    full_material_intensity = finite_intensity_full(
        material_layers,
        pbi2_f_plus_e,
        pbi2_f_minus_e,
        1.0 + 0.0j,
        material_vertical_phase,
        TransitionLaw.for_parent(Parent.SIX_H_PLUS),
        InitialPopulation.plus_only(),
    ).intensity_electron2
    assert abs(full_material_intensity - direct_material_intensity) <= 1e-10 * max(
        1.0, abs(direct_material_intensity)
    )


def test_typed_parent_models_and_event_boundary() -> None:
    np.testing.assert_allclose(
        RichEpsilonModel(Parent.FOUR_H_PLUS, 0.1).transition_law().as_array(),
        [0.025, 0.025, 0.025, 0.9, 0.025],
    )
    np.testing.assert_allclose(
        ReducedABDModel(0.56, 0.26, 0.18).transition_law().as_array(),
        [0.56, 0.26, 0.0, 0.18, 0.0],
    )
    with pytest.raises(ValueError, match="epsilon"):
        RichEpsilonModel(Parent.TWO_H, 1.01)
    with pytest.raises(ValueError, match="nonnegative"):
        ReducedABDModel(0.8, -0.1, 0.3)
    with pytest.raises(ValueError, match="sum to one"):
        ReducedABDModel(0.5, 0.2, 0.2)

    query_values = {
        "event_id": np.array([101, 102], dtype=np.int64),
        "rod_id": np.array([5, 6], dtype=np.int64),
        "phase_id": ("pbi2", "pbi2"),
        "h": np.array([1, 2], dtype=np.int32),
        "k": np.array([0, 0], dtype=np.int32),
        "qz_Ainv": np.array([0.31, 0.47]),
        "l_coordinate": np.array([0.2, 0.3]),
        "wavelength_A": np.array([1.0, 1.0]),
    }
    query = RodQueryBatch(**query_values)
    amplitudes = LayerAmplitudeResult(
        query.event_id,
        np.array([1.1 + 0.2j, 0.8 - 0.1j]),
        np.array([0.7 - 0.3j, 1.0 + 0.4j]),
    )
    event_arguments = {
        "query": query,
        "amplitudes": amplitudes,
        "law": TransitionLaw.for_parent(Parent.TWO_H),
        "layers": 9,
        "layer_repeat_A": 3.4,
        "initial": InitialPopulation.plus_only(),
        "normalization": FiniteNormalization.PER_LAYER,
        "model_component_id": "2H-rich",
        "population_group_id": "parents",
    }
    one = finite_event_intensity(
        **event_arguments,
    )
    np.testing.assert_array_equal(one.event_id, query.event_id)
    assert np.all(one.intensity_per_sr >= 0.0)
    assert one.normalization == "intensity_per_layer_electron2"

    misaligned = LayerAmplitudeResult(query.event_id[::-1], amplitudes.f_plus, amplitudes.f_minus)
    with pytest.raises(ValueError, match="event-aligned"):
        finite_event_intensity(**(event_arguments | {"amplitudes": misaligned}))
    missing_orientation = LayerAmplitudeResult(query.event_id, amplitudes.f_plus, None)
    with pytest.raises(ValueError, match="both f_plus and f_minus"):
        finite_event_intensity(**(event_arguments | {"amplitudes": missing_orientation}))
    for invalid_id in ("", 3):
        with pytest.raises(ValueError, match="model_component_id"):
            finite_event_intensity(**(event_arguments | {"model_component_id": invalid_id}))
    for invalid_group in ("", object()):
        with pytest.raises(ValueError, match="population_group_id"):
            finite_event_intensity(**(event_arguments | {"population_group_id": invalid_group}))

    with pytest.raises(ValueError, match="unique"):
        LayerAmplitudeResult(
            np.array([101, 101]),
            amplitudes.f_plus,
            amplitudes.f_minus,
        )
    with pytest.raises(ValueError, match="unique"):
        RodQueryBatch(**(query_values | {"event_id": np.array([101, 101], dtype=np.int64)}))
    missing_event = LayerAmplitudeResult(
        np.array([101]),
        amplitudes.f_plus[:1],
        amplitudes.f_minus[:1],
    )
    with pytest.raises(ValueError, match="event-aligned"):
        finite_event_intensity(**(event_arguments | {"amplitudes": missing_event}))
    with pytest.raises(ValueError, match="finite"):
        LayerAmplitudeResult(
            query.event_id,
            np.array([np.nan + 0.0j, 1.0 + 0.0j]),
            amplitudes.f_minus,
        )
    with pytest.raises(ValueError, match="finite"):
        LayerAmplitudeResult(
            query.event_id,
            amplitudes.f_plus,
            np.array([1.0 + 0.0j, np.inf + 0.0j]),
        )
    with pytest.raises(ValueError, match="finite"):
        RodQueryBatch(**(query_values | {"qz_Ainv": np.array([np.nan, 0.47])}))
    for invalid_repeat in (0.0, -1.0, np.nan, np.inf, True, "3.4", 10**400):
        with pytest.raises(ValueError, match="layer_repeat_A"):
            finite_event_intensity(**(event_arguments | {"layer_repeat_A": invalid_repeat}))
    for invalid_layers in (0, True):
        with pytest.raises(ValueError, match="layers"):
            finite_event_intensity(**(event_arguments | {"layers": invalid_layers}))

    laue_query = RodQueryBatch(
        np.array([201]),
        np.array([8]),
        ("pbi2",),
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0.31]),
        np.array([0.0]),
        np.array([1.0]),
    )
    amplitude = 1.1 - 0.2j
    laue_amplitudes = LayerAmplitudeResult(
        laue_query.event_id,
        np.array([amplitude]),
        np.array([amplitude]),
    )
    laue_event = finite_event_intensity(
        laue_query,
        laue_amplitudes,
        TransitionLaw.for_parent(Parent.TWO_H),
        layers=2,
        layer_repeat_A=3.4,
        initial=InitialPopulation.plus_only(),
        normalization=FiniteNormalization.TOTAL,
        model_component_id="2H",
        population_group_id=None,
    )
    expected_electron2 = abs(amplitude * (1.0 + np.exp(1j * 0.31 * 3.4))) ** 2
    np.testing.assert_allclose(
        laue_event.intensity_per_sr,
        expected_electron2,
        rtol=2e-14,
        atol=2e-14,
    )
    assert laue_event.normalization == "intensity_electron2"


def test_immutable_stacking_pack_matches_with_declared_legacy_scale() -> None:
    root = Path(__file__).resolve().parents[1]
    legacy_area_scale = 3.0 * (2.0 * np.pi) ** 2 / 17.98e-10
    with np.load(root / "reference" / "rasim_reference_v1.npz", allow_pickle=False) as pack:
        for phase_index, parent in enumerate((Parent.TWO_H, Parent.FOUR_H_PLUS, Parent.SIX_H_PLUS)):
            for epsilon_index, epsilon in enumerate(pack["stacking_rich_epsilons"]):
                law = RichEpsilonModel(parent, float(epsilon)).transition_law()
                np.testing.assert_allclose(
                    law.as_array(),
                    pack["stacking_rich_probabilities"][phase_index, epsilon_index],
                    rtol=0.0,
                    atol=1e-15,
                )

        common = dict(
            layers=12,
            f_plus=pack["stacking_F_plus"],
            f_minus=pack["stacking_F_minus"],
            omega=complex(pack["stacking_omega"]),
            vertical_phase=pack["stacking_xi"],
            initial=InitialPopulation.plus_only(),
        )
        rich = finite_intensity_reduced(
            law=TransitionLaw.from_array(pack["stacking_rich_theta"]), **common
        )
        np.testing.assert_allclose(
            legacy_area_scale * rich.intensity_per_layer_electron2,
            pack["stacking_rich_intensity"],
            rtol=3e-14,
            atol=3e-4,
        )
        for index, values in enumerate(pack["stacking_abd_cases"]):
            law = ReducedABDModel(*map(float, values)).transition_law()
            reduced = finite_intensity_reduced(law=law, **common)
            np.testing.assert_allclose(
                legacy_area_scale * reduced.intensity_per_layer_electron2,
                pack["stacking_reduced_intensity"][index],
                rtol=3e-14,
                atol=3e-4,
            )


def test_population_intensities_are_incoherent_and_order_invariant() -> None:
    query = RodQueryBatch(
        np.array([301, 302]),
        np.array([11, 12]),
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
    second_weight = 0.6000000000005
    populations = (
        StackingPopulation("2H", TransitionLaw.for_parent(Parent.TWO_H), 0.4),
        StackingPopulation(
            "4H-rich",
            RichEpsilonModel(Parent.FOUR_H_PLUS, 0.2).transition_law(),
            second_weight,
        ),
    )
    arguments = {
        "query": query,
        "amplitudes": amplitudes,
        "layers": 6,
        "layer_repeat_A": 3.4,
        "initial": InitialPopulation.plus_only(),
        "population_group_id": "stacking-parents",
    }
    forward = finite_population_event_intensity(populations=populations, **arguments)
    reverse = finite_population_event_intensity(
        populations=tuple(reversed(populations)), **arguments
    )
    assert isinstance(forward, PopulationIntensityResult)
    assert forward.population_id == ("2H", "4H-rich")
    np.testing.assert_array_equal(forward.weight, [0.4, second_weight])
    assert not forward.weight.flags.writeable

    omega = registry_phase(query.h, query.k)
    vertical = np.exp(1j * query.qz_Ainv * 3.4)
    expected_components = np.array(
        [
            [
                finite_intensity_by_enumeration(
                    6,
                    amplitudes.f_plus[event],
                    amplitudes.f_minus[event],
                    omega[event],
                    vertical[event],
                    population.model,
                    InitialPopulation.plus_only(),
                ).intensity_electron2
                for event in range(query.event_id.size)
            ]
            for population in populations
        ]
    )
    np.testing.assert_allclose(
        forward.component_intensity_electron2,
        expected_components,
        rtol=5e-13,
        atol=5e-13,
    )
    expected_total = 0.4 * expected_components[0] + second_weight * expected_components[1]
    np.testing.assert_allclose(
        forward.weighted_total_intensity_electron2,
        expected_total,
        rtol=5e-13,
        atol=5e-13,
    )
    np.testing.assert_array_equal(
        forward.weighted_total_intensity_electron2,
        reverse.weighted_total_intensity_electron2,
    )
    np.testing.assert_array_equal(
        forward.event_intensity.intensity_per_sr,
        forward.weighted_total_intensity_electron2,
    )
    np.testing.assert_array_equal(forward.event_intensity.event_id, query.event_id)
    assert forward.event_intensity.normalization == "intensity_electron2"

    for invalid_weight in (-0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="weight"):
            StackingPopulation("invalid", populations[0].model, invalid_weight)
    with pytest.raises(ValueError, match="nonempty"):
        finite_population_event_intensity(populations=(), **arguments)
    with pytest.raises(ValueError, match="unique"):
        finite_population_event_intensity(
            populations=(
                populations[0],
                StackingPopulation("2H", populations[1].model, 0.6),
            ),
            **arguments,
        )
    with pytest.raises(ValueError, match="sum to one"):
        finite_population_event_intensity(
            populations=(
                StackingPopulation("2H", populations[0].model, 0.4),
                StackingPopulation("4H-rich", populations[1].model, 0.5),
            ),
            **arguments,
        )
    wrong_total = np.array([1e20, 1e6])
    wrong_event = EventIntensityResult(
        query.event_id,
        wrong_total,
        "stacking-transition-population-finite-v1",
        "weighted-stacking-population-total",
        "stacking-parents",
        "intensity_electron2",
    )
    with pytest.raises(ValueError, match="weighted total"):
        PopulationIntensityResult(
            ("only",),
            np.array([1.0]),
            np.array([[1e20, 1.0]]),
            wrong_total,
            wrong_event,
        )
