"""Compact analytic, legacy, mutation, convergence, and benchmark proof."""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import time
import tracemalloc
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rasim_next.core.contracts import (
    CONTRACT_API_VERSION,
    LayerAmplitudeResult,
    RodQueryBatch,
)
from rasim_next.proof.traces import Measure, QuantityKind, TraceRecord, compare_traces
from rasim_next.stacking.enumeration import finite_intensity_by_enumeration
from rasim_next.stacking.finite_intensity import (
    FiniteNormalization,
    finite_event_intensity,
    finite_intensity_full,
    finite_intensity_reduced,
    finite_population_event_intensity,
    stationary_intensity_reduced,
)
from rasim_next.stacking.parent_models import (
    ReducedABDModel,
    RichEpsilonModel,
    StackingPopulation,
)
from rasim_next.stacking.transition import (
    STATE_ORDER,
    InitialPopulation,
    Parent,
    TransitionLaw,
    full_transition_matrix,
    reduced_transition_matrix,
    registry_phase,
)

_LEGACY_AREA_SCALE = 3.0 * (2.0 * np.pi) ** 2 / 17.98e-10
_PACK_ATOL = 3e-4
_PACK_RTOL = 3e-14
_ORACLE_ATOL = 5e-13
_ORACLE_RTOL = 5e-13
_EXPECTED_MUTATION_STAGES = {
    "transition_convention_transposed": "stacking.transition_matrix_6",
    "wrong_layer_count_offset": "stacking.pair_kernel",
    "total_per_layer_swap": "stacking.finite_intensity",
    "coherent_population_mixture": "stacking.population_intensity",
    "registry_phase_omitted": "stacking.registry_phase",
    "stationary_substituted_for_finite": "stacking.finite_intensity",
    "reduced_sector_coefficient_perturbed": "stacking.transition_matrix_reduced",
    "gauge_amplitude_without_registry_compensation": "stacking.pair_kernel",
}
_EXPECTED_CLASSIFICATIONS = {
    "stacking.transition_matrix_6": ("MATCH", None),
    "stacking.transition_matrix_reduced": ("MATCH", None),
    "stacking.synthetic_finite": ("MATCH", None),
    "stacking.reference_pack_intensity": ("MATCH", None),
    "stacking.legacy_initial_population": ("CORRECTED", "stacking.pair_kernel"),
    "stacking.legacy_normalization": ("CORRECTED", "stacking.finite_intensity"),
    "stacking.legacy_phase_expression": ("CORRECTED", "stacking.registry_phase"),
    "stacking.legacy_layer_repeat": ("CORRECTED", "stacking.finite_intensity"),
    "stacking.legacy_epsilon_clipping": ("CORRECTED", "stacking.transition_matrix_6"),
    "stacking.legacy_abd_normalization": ("CORRECTED", "stacking.transition_matrix_6"),
    "stacking.incoherent_population_mixture": ("MATCH", None),
}
_CLASSIFICATION_EVIDENCE = {
    "stacking.transition_matrix_6": "tracked matrix convention and stochastic-row check",
    "stacking.transition_matrix_reduced": "exact six-state Fourier reduction in all three Miller sectors",
    "stacking.synthetic_finite": "direct enumeration, full pair sum, and reduced result agree",
    "stacking.reference_pack_intensity": "tracked pack agrees after its declared legacy area scale",
    "stacking.legacy_initial_population": "explicit normalized first-layer orientation replaces implicit occupancy",
    "stacking.legacy_normalization": "finite total and per-layer outputs are separately declared",
    "stacking.legacy_phase_expression": "typed positive-sign phase models replace evaluated expressions",
    "stacking.legacy_layer_repeat": "the explicit angstrom layer repeat controls the vertical phase",
    "stacking.legacy_epsilon_clipping": "out-of-domain epsilon is rejected before matrix construction",
    "stacking.legacy_abd_normalization": "raw a,b,d values must be nonnegative and normalized",
    "stacking.incoherent_population_mixture": "analytic weighted intensity sum matches PHY-STK-011",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _check(check_id: str, passed: bool, evidence: str) -> dict[str, str]:
    return {"check_id": check_id, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def _trace(stage: str, value: NDArray[Any] | np.generic | float | complex) -> TraceRecord:
    if "transition_matrix" in stage:
        quantity_kind = QuantityKind.MATRIX
    elif "intensity" in stage or "pair_kernel" in stage:
        quantity_kind = QuantityKind.INTENSITY
    else:
        quantity_kind = QuantityKind.AMPLITUDE
    return TraceRecord(
        "stacking.mutation",
        stage,
        np.asarray(value),
        "1",
        "crystal",
        Measure.NONE,
        quantity_kind,
        "stacking-transition-v1",
        "analytic mutation fixture",
    )


def _mutation(
    mutation_id: str,
    expected_stages: tuple[tuple[str, NDArray[Any] | np.generic | float | complex], ...],
    mutated: NDArray[Any] | np.generic | float | complex,
) -> dict[str, object]:
    stage = _EXPECTED_MUTATION_STAGES[mutation_id]
    reference = tuple(_trace(stage_id, value) for stage_id, value in expected_stages)
    candidate = tuple(
        _trace(stage_id, mutated if stage_id == stage else value)
        for stage_id, value in expected_stages
    )
    comparison = compare_traces(reference, candidate)
    failing_index = tuple(stage_id for stage_id, _ in expected_stages).index(stage)
    prior_stages_matched = all(
        np.array_equal(reference[index].value, candidate[index].value)
        for index in range(failing_index)
    )
    return {
        "mutation_id": mutation_id,
        "fixture_id": "stacking.synthetic.stage-local",
        "expected_first_stage": stage,
        "expected_failure_metric": "numeric_value",
        "observed_first_stage": comparison.first_failing_stage,
        "observed_failure_metric": comparison.failure_metric,
        "prior_stages_matched": prior_stages_matched,
        "detected": comparison.first_failing_stage == stage and prior_stages_matched,
    }


def _pair_kernel(
    layers: int,
    f_plus: complex,
    f_minus: complex,
    omega: complex,
    vertical_phase: complex,
    law: TransitionLaw,
    initial: InitialPopulation,
    *,
    exponent_offset: int = 0,
) -> NDArray[np.float64]:
    amplitudes = np.array(
        [
            f_plus,
            omega * f_plus,
            omega**2 * f_plus,
            f_minus,
            omega * f_minus,
            omega**2 * f_minus,
        ],
        dtype=np.complex128,
    )
    transition = full_transition_matrix(law)
    populations = [np.array([initial.plus, 0.0, 0.0, initial.minus, 0.0, 0.0], dtype=np.float64)]
    for _ in range(1, layers):
        populations.append(populations[-1] @ transition)
    terms = []
    for separation in range(1, layers):
        power = np.linalg.matrix_power(transition, separation + exponent_offset)
        propagated = power @ amplitudes
        pair = sum(
            vertical_phase**separation * (populations[start] @ (np.conj(amplitudes) * propagated))
            for start in range(layers - separation)
        )
        terms.append(2.0 * pair.real)
    return np.asarray(terms, dtype=np.float64)


def _reduced_pair_terms(
    layers: int,
    amplitudes: NDArray[np.complex128],
    transition: NDArray[np.complex128],
    vertical_phase: complex,
    law: TransitionLaw,
    initial: InitialPopulation,
) -> tuple[NDArray[np.float64], float]:
    same = law.a + law.b_plus + law.b_minus
    flip = law.d_plus + law.d_minus
    orientation = np.array([[same, flip], [flip, same]], dtype=np.float64)
    populations = [initial.as_array()]
    for _ in range(1, layers):
        populations.append(populations[-1] @ orientation)
    self_total = sum(float(population @ np.abs(amplitudes) ** 2) for population in populations)
    power = np.eye(2, dtype=np.complex128)
    vertical_power = 1.0 + 0.0j
    terms = []
    for separation in range(1, layers):
        power = power @ transition
        vertical_power *= vertical_phase
        propagated = power @ amplitudes
        pair = sum(
            vertical_power * (populations[start] @ (np.conj(amplitudes) * propagated))
            for start in range(layers - separation)
        )
        terms.append(2.0 * pair.real)
    pair_terms = np.asarray(terms, dtype=np.float64)
    return pair_terms, float(self_total + pair_terms.sum())


def _gauge_fixture() -> tuple[float, float, float, NDArray[np.float64], NDArray[np.float64], float]:
    law = TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18)
    initial = InitialPopulation.plus_only()
    omega = complex(registry_phase(1, 0))
    vertical = np.exp(0.43j)
    amplitudes = np.array([1.2 + 0.3j, 0.7 - 0.5j], dtype=np.complex128)
    transition = reduced_transition_matrix(law, omega)
    reference_pair, reference_total = _reduced_pair_terms(
        7, amplitudes, transition, vertical, law, initial
    )
    gauge = np.diag(np.array([1.0 + 0.0j, omega], dtype=np.complex128))
    transformed_amplitudes = gauge @ amplitudes
    transformed_transition = gauge @ transition @ gauge.conj().T
    transformed_pair, transformed_total = _reduced_pair_terms(
        7,
        transformed_amplitudes,
        transformed_transition,
        vertical,
        law,
        initial,
    )
    amplitude_only_pair, amplitude_only_total = _reduced_pair_terms(
        7,
        transformed_amplitudes,
        transition,
        vertical,
        law,
        initial,
    )
    return (
        float(np.max(np.abs(reference_pair - transformed_pair))),
        abs(reference_total - transformed_total),
        abs(reference_total - amplitude_only_total),
        reference_pair,
        amplitude_only_pair,
        reference_total,
    )


def _mutations() -> list[dict[str, object]]:
    law = TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18)
    initial = InitialPopulation.plus_only()
    omega = complex(registry_phase(1, 0))
    xi = np.exp(0.43j)
    f_plus, f_minus = 1.2 + 0.3j, 0.7 - 0.5j
    finite = finite_intensity_reduced(7, f_plus, f_minus, omega, xi, law, initial)
    reduced = reduced_transition_matrix(law, omega)
    perturbed = reduced.copy()
    perturbed[0, 1] += 1e-3
    stationary = stationary_intensity_reduced(
        f_plus,
        f_minus,
        omega,
        xi,
        law,
        InitialPopulation(0.5, 0.5),
        correlation_decay=0.9,
    )
    component_1 = finite_intensity_reduced(
        7, f_plus, f_minus, omega, xi, TransitionLaw.for_parent(Parent.TWO_H), initial
    ).intensity_electron2
    component_2 = finite_intensity_reduced(
        7,
        f_plus,
        f_minus,
        omega,
        xi,
        RichEpsilonModel(Parent.FOUR_H_PLUS, 0.2).transition_law(),
        initial,
    ).intensity_electron2
    incoherent = 0.4 * component_1 + 0.6 * component_2
    amplitude_mixture = abs(np.sqrt(0.4 * component_1) + np.sqrt(0.6 * component_2)) ** 2
    stages = (
        ("stacking.registry_phase", np.asarray(omega)),
        ("stacking.transition_matrix_6", full_transition_matrix(law)),
        ("stacking.transition_matrix_reduced", reduced),
        (
            "stacking.pair_kernel",
            _pair_kernel(7, f_plus, f_minus, omega, xi, law, initial),
        ),
        ("stacking.finite_intensity", finite.intensity_electron2),
        ("stacking.population_intensity", incoherent),
    )
    per_layer_stages = tuple(
        (
            stage,
            finite.intensity_per_layer_electron2 if stage == "stacking.finite_intensity" else value,
        )
        for stage, value in stages
    )
    _, _, _, gauge_pair, gauge_mutated_pair, gauge_total = _gauge_fixture()
    gauge_stages = tuple(
        (
            stage,
            gauge_pair
            if stage == "stacking.pair_kernel"
            else gauge_total
            if stage == "stacking.finite_intensity"
            else value,
        )
        for stage, value in stages
    )
    return [
        _mutation(
            "transition_convention_transposed",
            stages,
            full_transition_matrix(law).T,
        ),
        _mutation(
            "wrong_layer_count_offset",
            stages,
            _pair_kernel(7, f_plus, f_minus, omega, xi, law, initial, exponent_offset=1),
        ),
        _mutation(
            "total_per_layer_swap",
            stages,
            finite.intensity_per_layer_electron2,
        ),
        _mutation(
            "coherent_population_mixture",
            stages,
            amplitude_mixture,
        ),
        _mutation(
            "registry_phase_omitted",
            stages,
            np.asarray(1.0 + 0.0j),
        ),
        _mutation(
            "stationary_substituted_for_finite",
            per_layer_stages,
            stationary,
        ),
        _mutation(
            "reduced_sector_coefficient_perturbed",
            stages,
            perturbed,
        ),
        _mutation(
            "gauge_amplitude_without_registry_compensation",
            gauge_stages,
            gauge_mutated_pair,
        ),
    ]


def _oracle_error() -> tuple[float, float]:
    law = TransitionLaw(0.28, 0.14, 0.19, 0.25, 0.14)
    initial = InitialPopulation(0.8, 0.2)
    maximum_full = 0.0
    maximum_reduced = 0.0
    for h, k in ((0, 0), (1, 0), (0, 1), (1, 1)):
        omega = complex(registry_phase(h, k))
        for layers in range(1, 7):
            xi = np.exp(1j * (0.11 + 0.07 * layers))
            f_plus = 1.3 - 0.1j + 0.01 * layers
            f_minus = 0.6 + 0.7j - 0.02j * layers
            enumerated = finite_intensity_by_enumeration(
                layers, f_plus, f_minus, omega, xi, law, initial
            ).intensity_electron2
            full = finite_intensity_full(
                layers, f_plus, f_minus, omega, xi, law, initial
            ).intensity_electron2
            reduced = finite_intensity_reduced(
                layers, f_plus, f_minus, omega, xi, law, initial
            ).intensity_electron2
            maximum_full = max(maximum_full, float(abs(full - enumerated)))
            maximum_reduced = max(maximum_reduced, float(abs(reduced - enumerated)))
    return maximum_full, maximum_reduced


def _limits() -> tuple[float, float, float, float, float]:
    law = TransitionLaw(0.13, 0.21, 0.08, 0.34, 0.24)
    initial = InitialPopulation(0.27, 0.73)
    xi = np.exp(0.44j)
    amplitude = 1.4 - 0.2j
    layers = 13
    laue = finite_intensity_reduced(layers, amplitude, amplitude, 1.0 + 0.0j, xi, law, initial)
    expected = abs(amplitude * sum(xi**index for index in range(layers))) ** 2
    single = finite_intensity_reduced(
        1, 1.2 + 0.3j, 0.8 - 0.4j, complex(registry_phase(1, 0)), xi, law, initial
    )
    expected_laws = {
        Parent.TWO_H: np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        Parent.FOUR_H_PLUS: np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
        Parent.FOUR_H_MINUS: np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        Parent.SIX_H_PLUS: np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        Parent.SIX_H_MINUS: np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    }
    f_plus = 1.2 + 0.3j
    f_minus = 0.8 - 0.4j
    omega = complex(registry_phase(1, 0))
    cycles = {
        Parent.TWO_H: (f_plus,),
        Parent.FOUR_H_PLUS: (f_plus, omega * f_minus),
        Parent.FOUR_H_MINUS: (f_plus, omega**2 * f_minus),
        Parent.SIX_H_PLUS: (f_plus, omega * f_plus, omega**2 * f_plus),
        Parent.SIX_H_MINUS: (f_plus, omega**2 * f_plus, omega * f_plus),
    }
    maximum_template_error = 0.0
    maximum_parent_error = 0.0
    for parent, expected_law in expected_laws.items():
        parent_law = TransitionLaw.for_parent(parent)
        maximum_template_error = max(
            maximum_template_error,
            float(np.max(np.abs(parent_law.as_array() - expected_law))),
        )
        reduced = finite_intensity_reduced(
            6,
            f_plus,
            f_minus,
            omega,
            xi,
            parent_law,
            InitialPopulation.plus_only(),
        )
        cycle = cycles[parent]
        expected_parent = abs(sum(xi**n * cycle[n % len(cycle)] for n in range(6))) ** 2
        maximum_parent_error = max(
            maximum_parent_error,
            float(abs(reduced.intensity_electron2 - expected_parent)),
        )
    expected_single = initial.plus * abs(1.2 + 0.3j) ** 2 + initial.minus * abs(0.8 - 0.4j) ** 2
    extinction_layers = 512
    extinction_phase = np.exp(2j * np.pi / extinction_layers)
    extinction_error = max(
        float(
            evaluator(
                extinction_layers,
                1.0,
                1.0,
                1.0,
                extinction_phase,
                TransitionLaw.for_parent(Parent.TWO_H),
                InitialPopulation.plus_only(),
            ).intensity_electron2
        )
        for evaluator in (finite_intensity_full, finite_intensity_reduced)
    )
    return (
        float(abs(laue.intensity_electron2 - expected)),
        float(abs(single.intensity_electron2 - expected_single)),
        maximum_template_error,
        maximum_parent_error,
        extinction_error,
    )


def _rejects(call: Callable[[], object]) -> bool:
    try:
        call()
    except (TypeError, ValueError):
        return True
    return False


def _boundary_conventions() -> tuple[bool, bool]:
    omega_plus = np.complex128(-0.5 + 0.5j * np.sqrt(3.0))
    omega_minus = np.complex128(-0.5 - 0.5j * np.sqrt(3.0))
    phase_exact = all(
        (
            np.array_equal(registry_phase(1, 0), omega_plus),
            np.array_equal(registry_phase(0, 1), omega_minus),
            np.array_equal(registry_phase(10_000, 0), omega_plus),
            np.array_equal(registry_phase(1, 0, "exp[2pi*i*(2h+k)/3]"), omega_minus),
            np.array_equal(
                reduced_transition_matrix(TransitionLaw.for_parent(Parent.TWO_H), np.exp(5e-13j)),
                reduced_transition_matrix(TransitionLaw.for_parent(Parent.TWO_H), 1.0 + 0.0j),
            ),
        )
    )
    rejection_pass = all(
        (
            _rejects(lambda: TransitionLaw(-0.1, 0.3, 0.2, 0.3, 0.3)),
            _rejects(lambda: TransitionLaw(0.2, 0.2, 0.2, 0.2, 0.1)),
            _rejects(lambda: InitialPopulation(0.6, 0.3)),
            _rejects(lambda: RichEpsilonModel(Parent.TWO_H, 1.01)),
            _rejects(lambda: ReducedABDModel(0.8, -0.1, 0.3)),
            _rejects(lambda: ReducedABDModel(0.5, 0.2, 0.2)),
            _rejects(
                lambda: stationary_intensity_reduced(
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    TransitionLaw.for_parent(Parent.TWO_H),
                    InitialPopulation.plus_only(),
                    correlation_decay=1.0,
                )
            ),
        )
    ) and bool(
        np.isfinite(
            stationary_intensity_reduced(
                1.0,
                1.0,
                1.0,
                np.exp(0.2j),
                TransitionLaw.for_parent(Parent.TWO_H),
                InitialPopulation.plus_only(),
                correlation_decay=1.0,
            )
        )
    )
    return phase_exact, rejection_pass


def _pack_comparison(pack_path: Path) -> tuple[float, float, float]:
    maximum_absolute = 0.0
    maximum_scaled = 0.0
    maximum_probability = 0.0
    with np.load(pack_path, allow_pickle=False) as pack:
        for phase_index, parent in enumerate((Parent.TWO_H, Parent.FOUR_H_PLUS, Parent.SIX_H_PLUS)):
            for epsilon_index, epsilon in enumerate(pack["stacking_rich_epsilons"]):
                observed = RichEpsilonModel(parent, float(epsilon)).transition_law().as_array()
                expected = pack["stacking_rich_probabilities"][phase_index, epsilon_index]
                maximum_probability = max(
                    maximum_probability, float(np.max(np.abs(observed - expected)))
                )
        common = {
            "layers": 12,
            "f_plus": pack["stacking_F_plus"],
            "f_minus": pack["stacking_F_minus"],
            "omega": complex(pack["stacking_omega"]),
            "vertical_phase": pack["stacking_xi"],
            "initial": InitialPopulation.plus_only(),
        }
        comparisons = [
            (
                _LEGACY_AREA_SCALE
                * finite_intensity_reduced(
                    law=TransitionLaw.from_array(pack["stacking_rich_theta"]), **common
                ).intensity_per_layer_electron2,
                pack["stacking_rich_intensity"],
            )
        ]
        for index, values in enumerate(pack["stacking_abd_cases"]):
            comparisons.append(
                (
                    _LEGACY_AREA_SCALE
                    * finite_intensity_reduced(
                        law=ReducedABDModel(*map(float, values)).transition_law(), **common
                    ).intensity_per_layer_electron2,
                    pack["stacking_reduced_intensity"][index],
                )
            )
        for observed, expected in comparisons:
            error = np.abs(observed - expected)
            limit = _PACK_ATOL + _PACK_RTOL * np.abs(expected)
            maximum_absolute = max(maximum_absolute, float(np.max(error)))
            maximum_scaled = max(maximum_scaled, float(np.max(error / limit)))
    return maximum_probability, maximum_absolute, maximum_scaled


def _convergence() -> dict[str, object]:
    law = TransitionLaw(0.2, 0.2, 0.1, 0.3, 0.2)
    stationary_population = InitialPopulation(0.5, 0.5)
    omega = 1.0 + 0.0j
    xi = 1.0 + 0.0j
    f_plus, f_minus = 1.2 + 0.3j, 0.7 - 0.5j
    decay = 0.8
    target = float(
        stationary_intensity_reduced(
            f_plus,
            f_minus,
            omega,
            xi,
            law,
            stationary_population,
            correlation_decay=decay,
        )
    )
    amplitudes = np.array([f_plus, f_minus], dtype=np.complex128)
    transition = reduced_transition_matrix(law, omega)
    population = stationary_population.as_array()
    self_term = float(population @ np.abs(amplitudes) ** 2)

    def truncated(cutoff: int) -> float:
        power = np.eye(2, dtype=np.complex128)
        pair_total = 0.0 + 0.0j
        phase_power = 1.0 + 0.0j
        for _ in range(1, cutoff + 1):
            power = power @ transition
            phase_power *= decay * xi
            pair_total += phase_power * (population @ (np.conj(amplitudes) * (power @ amplitudes)))
        return self_term + 2.0 * pair_total.real

    cutoffs = (4, 8, 16, 32)
    errors = [abs(truncated(cutoff) - target) for cutoff in cutoffs]
    return {
        "refinement": "direct damped lag sum toward separately named stationary solve",
        "correlation_decay": decay,
        "lag_cutoffs": list(cutoffs),
        "absolute_errors": errors,
        "monotone": all(later < earlier for earlier, later in pairwise(errors)),
    }


def _event_contract() -> tuple[bool, float, float, float]:
    query = RodQueryBatch(
        np.array([101, 102], dtype=np.int64),
        np.array([5, 6], dtype=np.int64),
        ("pbi2", "pbi2"),
        np.array([1, 2], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
        np.array([0.31, 0.47]),
        np.array([0.2, 0.3]),
        np.array([1.0, 1.0]),
    )
    amplitudes = LayerAmplitudeResult(
        query.event_id,
        np.array([1.1 + 0.2j, 0.8 - 0.1j]),
        np.array([0.7 - 0.3j, 1.0 + 0.4j]),
    )
    initial = InitialPopulation.plus_only()
    law = TransitionLaw.for_parent(Parent.TWO_H)
    arguments = {
        "query": query,
        "amplitudes": amplitudes,
        "law": law,
        "layers": 9,
        "layer_repeat_A": 3.4,
        "initial": initial,
        "model_component_id": "2H-rich",
        "population_group_id": "parents",
    }
    total = finite_event_intensity(normalization=FiniteNormalization.TOTAL, **arguments)
    per_layer = finite_event_intensity(normalization=FiniteNormalization.PER_LAYER, **arguments)
    aligned = bool(
        np.array_equal(query.event_id, total.event_id)
        and np.all(total.intensity_per_sr >= 0.0)
        and total.model_component_id == "2H-rich"
        and total.population_group_id == "parents"
    )
    normalization_error = float(
        np.max(np.abs(total.intensity_per_sr - 9.0 * per_layer.intensity_per_sr))
    )
    populations = (
        StackingPopulation("2H", law, 0.4),
        StackingPopulation(
            "4H-rich",
            RichEpsilonModel(Parent.FOUR_H_PLUS, 0.2).transition_law(),
            0.6,
        ),
    )
    population_arguments = {
        "query": query,
        "amplitudes": amplitudes,
        "layers": 6,
        "layer_repeat_A": 3.4,
        "initial": initial,
        "population_group_id": "stacking-parents",
    }
    forward = finite_population_event_intensity(
        populations=populations,
        **population_arguments,
    )
    reverse = finite_population_event_intensity(
        populations=tuple(reversed(populations)),
        **population_arguments,
    )
    omega = np.asarray(registry_phase(query.h, query.k))
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
                    initial,
                ).intensity_electron2
                for event in range(query.event_id.size)
            ]
            for population in populations
        ]
    )
    component_error = float(
        np.max(np.abs(forward.component_intensity_electron2 - expected_components))
    )
    expected_total = np.einsum("p,pe->e", np.array([0.4, 0.6]), expected_components)
    population_oracle_error = max(
        component_error,
        float(np.max(np.abs(forward.weighted_total_intensity_electron2 - expected_total))),
    )
    population_order_error = max(
        float(
            np.max(
                np.abs(
                    forward.component_intensity_electron2 - reverse.component_intensity_electron2
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    forward.weighted_total_intensity_electron2
                    - reverse.weighted_total_intensity_electron2
                )
            )
        ),
    )
    aligned &= bool(np.array_equal(forward.event_intensity.event_id, query.event_id))
    return aligned, normalization_error, population_oracle_error, population_order_error


def _measure(call: Callable[[], NDArray[np.float64]]) -> tuple[NDArray[np.float64], float, int]:
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def _benchmark() -> dict[str, object]:
    events = 48
    layers = 24
    index = np.arange(events, dtype=np.float64)
    f_plus = 1.1 + 0.003 * index + 1j * (0.2 + 0.002 * index)
    f_minus = 0.8 - 0.002 * index + 1j * (-0.3 + 0.001 * index)
    omega = np.asarray(registry_phase(np.arange(events) % 3, np.arange(events) % 2))
    vertical = np.exp(1j * (0.17 + 0.013 * index))
    law = TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18)
    initial = InitialPopulation(0.63, 0.37)

    def full_work() -> NDArray[np.float64]:
        return np.array(
            [
                finite_intensity_full(
                    layers,
                    f_plus[event],
                    f_minus[event],
                    omega[event],
                    vertical[event],
                    law,
                    initial,
                ).intensity_electron2
                for event in range(events)
            ],
            dtype=np.float64,
        )

    def reduced_work() -> NDArray[np.float64]:
        return finite_intensity_reduced(
            layers, f_plus, f_minus, omega, vertical, law, initial
        ).intensity_electron2

    full, full_seconds, full_peak = _measure(full_work)
    reduced, reduced_seconds, reduced_peak = _measure(reduced_work)
    return {
        "equivalent_work": {"events": events, "layers": layers},
        "full_pair_oracle_seconds": full_seconds,
        "full_pair_oracle_peak_bytes": full_peak,
        "reduced_recurrence_seconds": reduced_seconds,
        "reduced_recurrence_peak_bytes": reduced_peak,
        "speedup": full_seconds / reduced_seconds,
        "maximum_absolute_error": float(np.max(np.abs(full - reduced))),
    }


def _legacy_classifications() -> list[dict[str, str | None]]:
    return [
        {
            "case_id": case_id,
            "classification": classification,
            "first_divergence": first_divergence,
            "evidence": _CLASSIFICATION_EVIDENCE[case_id],
        }
        for case_id, (classification, first_divergence) in _EXPECTED_CLASSIFICATIONS.items()
    ]


def _classifications_complete(classifications: list[dict[str, str | None]]) -> bool:
    observed = {
        item["case_id"]: (item["classification"], item["first_divergence"])
        for item in classifications
    }
    return (
        len(observed) == len(classifications) == len(_EXPECTED_CLASSIFICATIONS)
        and observed == _EXPECTED_CLASSIFICATIONS
        and set(_CLASSIFICATION_EVIDENCE) == set(_EXPECTED_CLASSIFICATIONS)
        and all(item["evidence"] for item in classifications)
    )


def run_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    del allow_missing_pack
    root = Path(__file__).resolve().parents[3]
    pack_path = root / "reference" / "rasim_reference_v1.npz"
    probability_error, pack_error, pack_scaled_error = _pack_comparison(pack_path)
    full_error, reduced_error = _oracle_error()
    (
        laue_error,
        single_error,
        parent_template_error,
        parent_intensity_error,
        extinction_error,
    ) = _limits()
    phase_exact, rejection_pass = _boundary_conventions()
    convergence = _convergence()
    (
        gauge_pair_error,
        gauge_total_error,
        gauge_one_sided_difference,
        _,
        _,
        _,
    ) = _gauge_fixture()
    (
        aligned,
        normalization_error,
        population_oracle_error,
        population_order_error,
    ) = _event_contract()
    mutations = _mutations()
    benchmark = _benchmark()
    classifications = _legacy_classifications()
    convention_law = TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18)
    convention_matrix = full_transition_matrix(convention_law)
    stochastic_error = float(np.max(np.abs(convention_matrix.sum(axis=1) - 1.0)))
    state_order = tuple(state.value for state in STATE_ORDER)
    observed_mutations = {
        str(item["mutation_id"]): item["observed_first_stage"] for item in mutations
    }
    mutations_pass = (
        len(observed_mutations) == len(mutations) == len(_EXPECTED_MUTATION_STAGES)
        and observed_mutations == _EXPECTED_MUTATION_STAGES
        and all(
            bool(item["detected"])
            and item["observed_failure_metric"] == item["expected_failure_metric"]
            for item in mutations
        )
    )
    checks = [
        _check(
            "probability_and_state_conventions",
            probability_error <= 1e-15
            and stochastic_error <= 1e-15
            and state_order == ("0F+", "1F+", "2F+", "0F-", "1F-", "2F-"),
            f"state order exact; row-sum error {stochastic_error:.3e}; template error {probability_error:.3e}",
        ),
        _check(
            "typed_phase_and_boundary_validation",
            phase_exact and rejection_pass,
            "both nontrivial roots and legacy mapping exact; invalid probabilities rejected; undamped nonsingular solve accepted",
        ),
        _check(
            "direct_sequence_and_full_pair",
            full_error <= _ORACLE_ATOL,
            f"N=1..6 and four Miller classes; maximum absolute error {full_error:.3e}",
        ),
        _check(
            "exact_fourier_reduction",
            reduced_error <= _ORACLE_ATOL,
            f"six-state/enumerated parity; maximum absolute error {reduced_error:.3e}",
        ),
        _check(
            "analytic_limits",
            laue_error <= _ORACLE_ATOL
            and single_error <= _ORACLE_ATOL
            and parent_template_error == 0.0
            and parent_intensity_error <= _ORACLE_ATOL
            and extinction_error <= 1e-24,
            f"Laue error {laue_error:.3e}; N=1 error {single_error:.3e}; parent template error {parent_template_error:.3e}; explicit-cycle error {parent_intensity_error:.3e}; N=512 extinction error {extinction_error:.3e}",
        ),
        _check(
            "gauge_invariance",
            gauge_pair_error <= _ORACLE_ATOL
            and gauge_total_error <= _ORACLE_ATOL
            and gauge_one_sided_difference > 1e-3,
            f"two-sided pair error {gauge_pair_error:.3e}; total error {gauge_total_error:.3e}; one-sided difference {gauge_one_sided_difference:.3e}",
        ),
        _check(
            "immutable_reference_pack",
            pack_scaled_error <= 1.0,
            f"declared legacy scale; max absolute {pack_error:.3e}, tolerance ratio {pack_scaled_error:.3e}",
        ),
        _check(
            "stationary_convergence",
            bool(convergence["monotone"]),
            "direct damped lag truncation converges monotonically to the stationary solve",
        ),
        _check(
            "event_contract_and_populations",
            aligned
            and normalization_error <= _ORACLE_ATOL
            and population_oracle_error <= _ORACLE_ATOL
            and population_order_error == 0.0,
            f"event IDs aligned; total/per-layer error {normalization_error:.3e}; population oracle error {population_oracle_error:.3e}; order error {population_order_error:.3e}",
        ),
        _check(
            "optimized_proof_agreement",
            float(benchmark["maximum_absolute_error"]) <= _ORACLE_ATOL,
            f"equivalent-work maximum absolute error {benchmark['maximum_absolute_error']:.3e}",
        ),
        _check(
            "error_injection",
            mutations_pass,
            f"{sum(bool(item['detected']) for item in mutations)}/{len(mutations)} isolated stage-local mutations detected with prior stages identical",
        ),
        _check(
            "legacy_classification_coverage",
            _classifications_complete(classifications),
            f"{len(classifications)} cases classified with evidence and every correction names its first divergence",
        ),
    ]
    tolerance_policy = json.dumps(
        {
            "pack_atol": _PACK_ATOL,
            "pack_rtol": _PACK_RTOL,
            "oracle_atol": _ORACLE_ATOL,
            "oracle_rtol": _ORACLE_RTOL,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    local_pass = all(item["status"] == "PASS" for item in checks)
    return {
        "schema_version": 1,
        "task_id": "T05",
        "status": "READY" if local_pass else "FAIL",
        "base_sha": _git(root, "merge-base", "HEAD", "main"),
        "commit_sha": _git(root, "rev-parse", "HEAD"),
        "contract_version": CONTRACT_API_VERSION,
        "trace_schema_version": 4,
        "reference_pack_sha256s": {"rasim_reference_v1": _sha256(pack_path)},
        "tolerance_policy_sha256": hashlib.sha256(tolerance_policy).hexdigest(),
        "checks": checks,
        "classifications": classifications,
        "convergence": convergence,
        "gauge_invariance": {
            "maximum_pair_error": gauge_pair_error,
            "total_error": gauge_total_error,
            "one_sided_difference": gauge_one_sided_difference,
        },
        "benchmark": benchmark,
        "mutations": mutations,
        "limitations": [
            "homogeneous first-order transition law and one explicit layer repeat per event batch",
            "direct sequence oracle intentionally limited to ten layers",
            "stationary result requires an explicit invariant population; singular undamped phases are rejected",
        ],
        "coordination_notes": [
            "T04 supplies raw event-aligned f_plus/f_minus in electrons with the shared Pb-site/layer-center gauge and no registry phase or downstream weight",
            "T05 returns raw finite-stack electron2 and applies stacking-population fractions once; T07 must not reapply them",
        ],
    }
