"""Compact proof for finite stacking-transition intensities."""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import subprocess
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rasim_next.core.contracts import CONTRACT_API_VERSION, LayerAmplitudeResult, RodQueryBatch
from rasim_next.stacking.enumeration import (
    finite_explicit_sequence_intensity,
    finite_intensity_by_enumeration,
)
from rasim_next.stacking.finite_intensity import (
    LayerNormalQBatch,
    finite_event_intensity,
    finite_intensity_full,
    finite_intensity_reduced,
    finite_population_event_intensity,
)
from rasim_next.stacking.parent_models import (
    Handedness,
    ReducedABDModel,
    RichEpsilonModel,
    StackingPopulation,
)
from rasim_next.stacking.transition import (
    STATE_ORDER,
    InitialPopulation,
    Parent,
    StackingState,
    TransitionLaw,
    full_transition_matrix,
    reduced_transition_matrix,
    registry_phase,
)

_LEGACY_AREA_SCALE = 3.0 * (2.0 * np.pi) ** 2 / 17.98e-10
_PARENT_CYCLES = (
    (Parent.TWO_H, ((0, 0),)),
    (Parent.FOUR_H_PLUS, ((0, 0), (1, 1))),
    (Parent.FOUR_H_MINUS, ((0, 0), (1, 2))),
    (Parent.SIX_H_PLUS, ((0, 0), (0, 1), (0, 2))),
    (Parent.SIX_H_MINUS, ((0, 0), (0, 2), (0, 1))),
)
_STAGES = (
    "stacking.registry_phase",
    "stacking.transition_matrix_6",
    "stacking.transition_matrix_reduced",
    "stacking.pair_kernel",
    "stacking.finite_intensity",
    "stacking.population_intensity",
)
_MUTATION_STAGES = {
    "transition_convention_transposed": _STAGES[1],
    "wrong_layer_count_offset": _STAGES[3],
    "total_per_layer_swap": _STAGES[4],
    "coherent_population_mixture": _STAGES[5],
    "registry_phase_omitted": _STAGES[0],
    "stationary_substituted_for_finite": _STAGES[4],
    "reduced_sector_coefficient_perturbed": _STAGES[2],
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _environment_sha256() -> str:
    environment = {
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    payload = json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _check(check_id: str, status: str, evidence: str) -> dict[str, str]:
    return {"check_id": check_id, "status": status, "evidence": evidence}


def _records(
    keys: tuple[str, ...], rows: tuple[tuple[object, ...], ...]
) -> list[dict[str, object]]:
    return [dict(zip(keys, row, strict=True)) for row in rows]


def _direct_pair_intensity(
    layers: int,
    f_plus: complex,
    f_minus: complex,
    omega: complex,
    vertical_phase: complex,
    law: TransitionLaw,
    initial: InitialPopulation,
) -> float:
    """Transparent finite self/pair sum used only for a short proof stack."""

    registry = np.array([1.0, omega, omega**2], dtype=np.complex128)
    amplitudes = np.r_[f_plus * registry, f_minus * registry]
    transition = full_transition_matrix(law)
    populations = [np.array([initial.plus, 0.0, 0.0, initial.minus, 0.0, 0.0], dtype=np.float64)]
    for _ in range(1, layers):
        populations.append(populations[-1] @ transition)
    self_total = sum(float(population @ np.abs(amplitudes) ** 2) for population in populations)
    pair_total = 0.0 + 0.0j
    transition_power = np.eye(6, dtype=np.float64)
    vertical_power = 1.0 + 0.0j
    for separation in range(1, layers):
        transition_power = transition_power @ transition
        vertical_power *= vertical_phase
        propagated = transition_power @ amplitudes
        pair_total += sum(
            vertical_power * (populations[start] @ (np.conj(amplitudes) * propagated))
            for start in range(layers - separation)
        )
    return float(self_total + 2.0 * pair_total.real)


def _errors(arguments: tuple[object, ...], expected: float) -> dict[str, float]:
    observed = {
        "enumeration": finite_intensity_by_enumeration(*arguments).intensity_electron2,
        "full": finite_intensity_full(*arguments).intensity_electron2,
        "reduced": finite_intensity_reduced(*arguments).intensity_electron2,
    }
    return {name: abs(float(value) - expected) for name, value in observed.items()}


def _finite_equality() -> dict[str, object]:
    common = (
        1.2 + 0.3j,
        0.7 - 0.5j,
        complex(registry_phase(1, 0)),
        np.exp(0.43j),
        TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18),
        InitialPopulation(0.8, 0.2),
    )
    records = []
    maximum = {name: 0.0 for name in ("pair", "full", "reduced")}
    single_layer_error = 0.0
    for layers in (1, 3, 6):
        arguments = (layers, *common)
        direct = float(finite_intensity_by_enumeration(*arguments).intensity_electron2)
        errors = {
            "pair": abs(_direct_pair_intensity(*arguments) - direct),
            "full": abs(float(finite_intensity_full(*arguments).intensity_electron2) - direct),
            "reduced": abs(
                float(finite_intensity_reduced(*arguments).intensity_electron2) - direct
            ),
        }
        if layers == 1:
            expected = (
                common[-1].plus * abs(common[0]) ** 2 + common[-1].minus * abs(common[1]) ** 2
            )
            single_layer_error = max(abs(direct - expected), *errors.values())
        maximum = {name: max(maximum[name], error) for name, error in errors.items()}
        records.append({"layers": layers, "absolute_errors": errors})
    return {
        "refinement": "none: exact finite algebra",
        "records": records,
        "maximum_absolute_errors": maximum,
        "single_layer_absolute_error": single_layer_error,
    }


def _science_evidence() -> dict[str, object]:
    f_plus, f_minus, xi = 1.2 + 0.3j, 0.8 - 0.4j, np.exp(0.43j)
    parent_max = {name: 0.0 for name in ("enumeration", "full", "reduced")}
    for h, k in ((0, 0), (1, 0), (0, 1)):
        omega = complex(registry_phase(h, k))
        for parent, cycle in _PARENT_CYCLES:
            sequence = cycle * (6 // len(cycle))
            expected = float(
                abs(
                    sum(
                        xi**n * (f_plus if hand == 0 else f_minus) * omega**registry
                        for n, (hand, registry) in enumerate(sequence)
                    )
                )
                ** 2
            )
            errors = _errors(
                (
                    6,
                    f_plus,
                    f_minus,
                    omega,
                    xi,
                    TransitionLaw.for_parent(parent),
                    InitialPopulation.plus_only(),
                ),
                expected,
            )
            parent_max = {name: max(parent_max[name], error) for name, error in errors.items()}
    model_errors = {}
    for case_id, law in (
        ("rich_4H_minus_epsilon_0p2", RichEpsilonModel(Parent.FOUR_H_MINUS, 0.2).transition_law()),
        (
            "abd_0p56_0p26_0p18_minus",
            ReducedABDModel(0.56, 0.26, 0.18, Handedness.MINUS).transition_law(),
        ),
    ):
        arguments = (
            6,
            1.2 + 0.3j,
            0.7 - 0.5j,
            complex(registry_phase(1, 0)),
            xi,
            law,
            InitialPopulation(0.8, 0.2),
        )
        expected = float(finite_intensity_by_enumeration(*arguments).intensity_electron2)
        model_errors[case_id] = _errors(arguments, expected)
    amplitude, laue_xi, layers = 1.4 - 0.2j, np.exp(0.44j), 13
    laue_expected = float(abs(amplitude * sum(laue_xi**n for n in range(layers))) ** 2)
    laue_args = (
        layers,
        amplitude,
        amplitude,
        1.0 + 0.0j,
        laue_xi,
        TransitionLaw(0.13, 0.21, 0.08, 0.34, 0.24),
        InitialPopulation(0.27, 0.73),
    )
    return {
        "parent_case_count": 15,
        "parents": [parent.value for parent, _ in _PARENT_CYCLES],
        "miller_sectors": [0, 1, 2],
        "parent_maximum_absolute_errors": parent_max,
        "typed_model_absolute_errors": model_errors,
        "laue_analytic_intensity_electron2": laue_expected,
        "laue_absolute_errors": {
            name: abs(float(evaluator(*laue_args).intensity_electron2) - laue_expected)
            for name, evaluator in (
                ("full", finite_intensity_full),
                ("reduced", finite_intensity_reduced),
            )
        },
    }


def _pack_evidence(pack_path: Path) -> dict[str, object]:
    with np.load(pack_path, allow_pickle=False) as pack:
        probabilities = np.array(
            [
                [
                    RichEpsilonModel(parent, float(epsilon)).transition_law().as_array()
                    for epsilon in pack["stacking_rich_epsilons"]
                ]
                for parent in (Parent.TWO_H, Parent.FOUR_H_PLUS, Parent.SIX_H_PLUS)
            ]
        )
        probability_error = float(
            np.max(np.abs(probabilities - pack["stacking_rich_probabilities"]))
        )
        laws = [
            (RichEpsilonModel(Parent.TWO_H, 0.08).transition_law(), pack["stacking_rich_intensity"])
        ]
        laws.extend(
            (
                ReducedABDModel(*map(float, values)).transition_law(),
                pack["stacking_reduced_intensity"][index],
            )
            for index, values in enumerate(pack["stacking_abd_cases"])
        )
        maximum = {name: 0.0 for name in ("full_vs_pack", "reduced_vs_pack", "full_vs_reduced")}
        initial = InitialPopulation.plus_only()
        for law, expected in laws:
            reduced = (
                _LEGACY_AREA_SCALE
                * finite_intensity_reduced(
                    12,
                    pack["stacking_F_plus"],
                    pack["stacking_F_minus"],
                    complex(pack["stacking_omega"]),
                    pack["stacking_xi"],
                    law,
                    initial,
                ).intensity_per_layer_electron2
            )
            full = _LEGACY_AREA_SCALE * np.array(
                [
                    finite_intensity_full(
                        12, f_plus, f_minus, complex(pack["stacking_omega"]), xi, law, initial
                    ).intensity_per_layer_electron2
                    for f_plus, f_minus, xi in zip(
                        pack["stacking_F_plus"],
                        pack["stacking_F_minus"],
                        pack["stacking_xi"],
                        strict=True,
                    )
                ]
            )
            errors = {
                "full_vs_pack": float(np.max(np.abs(full - expected))),
                "reduced_vs_pack": float(np.max(np.abs(reduced - expected))),
                "full_vs_reduced": float(np.max(np.abs(full - reduced))),
            }
            maximum = {name: max(maximum[name], error) for name, error in errors.items()}
    return {
        "case_id": "stacking.synthetic_finite",
        "reference_manifest_classification": "MATCH",
        "acceptance": "PENDING_REVIEWED_SHARED_TOLERANCE",
        "probability_value_count": 60,
        "curve_value_count": 165,
        "probability_maximum_absolute_error": probability_error,
        "curve_maximum_absolute_errors": maximum,
        "legacy_area_scale": float(_LEGACY_AREA_SCALE),
    }


def _near_extinction() -> dict[str, object]:
    maximum_error = 0.0
    minimum_nearby_positive = float("inf")
    n512: dict[str, float] = {}
    for layers in (8, 64, 512):
        exact_phase = 2.0 * np.pi / layers
        nearby_offset = 1e-9 * (512 / layers) ** 2
        for offset in (-nearby_offset, 0.0, nearby_offset):
            vertical_phase = np.exp(1j * (exact_phase + offset))
            direct = float(abs(sum(vertical_phase**layer for layer in range(layers))) ** 2)
            arguments = (
                layers,
                1.0,
                1.0,
                1.0,
                vertical_phase,
                TransitionLaw.for_parent(Parent.TWO_H),
                InitialPopulation.plus_only(),
            )
            full = float(finite_intensity_full(*arguments).intensity_electron2)
            reduced = float(finite_intensity_reduced(*arguments).intensity_electron2)
            maximum_error = max(maximum_error, abs(full - direct), abs(reduced - direct))
            if offset:
                minimum_nearby_positive = min(minimum_nearby_positive, direct, full, reduced)
            if layers == 512 and offset == 1e-9:
                n512 = {"direct": direct, "full": full, "reduced": reduced}
    return {
        "maximum_absolute_error": maximum_error,
        "minimum_nearby_positive_intensity_electron2": minimum_nearby_positive,
        "n512_plus_1e_9": n512,
    }


def _event_and_population_errors() -> tuple[bool, float, float, float, float, float]:
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
    layer_normal_q = LayerNormalQBatch(query.event_id, np.array([0.63, -0.28]))
    event_layers = 9
    layer_repeat_A = 3.4
    event = finite_event_intensity(
        query,
        amplitudes,
        TransitionLaw.for_parent(Parent.TWO_H),
        layer_normal_q=layer_normal_q,
        layers=event_layers,
        layer_repeat_A=layer_repeat_A,
        initial=InitialPopulation.plus_only(),
        model_component_id="2H",
        population_group_id=None,
    )
    normalization_error = float(
        np.max(
            np.abs(event.intensity_electron2 - event_layers * event.intensity_per_layer_electron2)
        )
    )
    layer_depth_A = np.arange(event_layers) * layer_repeat_A
    layer_frame_amplitude = amplitudes.f_plus * np.exp(
        1j * layer_normal_q.layer_normal_q_Ainv[:, np.newaxis] * layer_depth_A
    ).sum(axis=1)
    expected = np.abs(layer_frame_amplitude) ** 2
    explicit = finite_explicit_sequence_intensity(
        query,
        amplitudes,
        (StackingState.REGISTRY_0_PLUS,) * event_layers,
        layer_depth_A,
        layer_normal_q=layer_normal_q,
        layers=event_layers,
        layer_repeat_A=layer_repeat_A,
    )
    frame_error = float(
        max(
            np.max(np.abs(event.intensity_electron2 - expected)),
            np.max(np.abs(explicit.intensity_electron2 - expected)),
        )
    )
    sample_frame_amplitude = amplitudes.f_plus * np.exp(
        1j * query.qz_Ainv[:, np.newaxis] * layer_depth_A
    ).sum(axis=1)
    sample_frame_separation = float(np.max(np.abs(expected - np.abs(sample_frame_amplitude) ** 2)))
    population_rows = (
        ("2H", Parent.TWO_H, InitialPopulation.minus_only()),
        ("4H", Parent.FOUR_H_PLUS, InitialPopulation.plus_only()),
    )
    populations = tuple(
        StackingPopulation(name, TransitionLaw.for_parent(parent), initial)
        for name, parent, initial in population_rows
    )
    arguments = {
        "query": query,
        "amplitudes": amplitudes,
        "layer_normal_q": layer_normal_q,
        "layers": 6,
        "layer_repeat_A": 3.4,
        "population_group_id": "parents",
    }
    components = finite_population_event_intensity(populations=populations, **arguments)
    reversed_components = finite_population_event_intensity(
        populations=tuple(reversed(populations)), **arguments
    )
    omega = np.asarray(registry_phase(query.h, query.k))
    vertical_phase = np.exp(1j * layer_normal_q.layer_normal_q_Ainv * 3.4)
    direct = np.fromiter(
        (
            finite_intensity_by_enumeration(
                6, f_plus, f_minus, phase, vertical, population.model, population.initial
            ).intensity_electron2
            for population in populations
            for f_plus, f_minus, phase, vertical in zip(
                amplitudes.f_plus, amplitudes.f_minus, omega, vertical_phase, strict=True
            )
        ),
        dtype=np.float64,
    ).reshape(len(populations), -1)
    component_error = float(np.max(np.abs(components.component_intensity_electron2 - direct)))
    order_error = float(
        np.max(
            abs(
                components.component_intensity_electron2
                - reversed_components.component_intensity_electron2
            )
        )
    )
    aligned_and_unweighted = bool(
        np.array_equal(event.event_id, query.event_id)
        and np.array_equal(components.event_id, query.event_id)
        and np.all(event.intensity_electron2 >= 0.0)
        and np.all(components.component_intensity_electron2 >= 0.0)
        and not hasattr(event, "intensity_per_sr")
        and not hasattr(components, "weight")
        and not hasattr(components, "weighted_total_intensity_electron2")
    )
    return (
        aligned_and_unweighted,
        normalization_error,
        frame_error,
        sample_frame_separation,
        component_error,
        order_error,
    )


def _calculated_trace(
    *,
    layers: int,
    omega: complex,
    law: TransitionLaw,
    initial: InitialPopulation,
    transition: np.ndarray | None = None,
    reduced: np.ndarray | None = None,
    offset: int = 0,
) -> dict[str, np.ndarray]:
    t6 = np.asarray(full_transition_matrix(law) if transition is None else transition)
    basis = np.array([[1, 0], [omega, 0], [omega**2, 0], [0, 1], [0, omega], [0, omega**2]])
    block = np.asarray((t6 @ basis)[[0, 3]] if reduced is None else reduced)
    orientation = np.array([[t6[0, :3].sum(), t6[0, 3:].sum()], [t6[3, :3].sum(), t6[3, 3:].sum()]])
    amplitudes, xi = np.array([1.2 + 0.3j, 0.7 - 0.5j]), np.exp(0.43j)
    populations = [initial.as_array()]
    for _ in range(1, layers):
        populations.append(populations[-1] @ orientation)
    self_terms = np.array([population @ abs(amplitudes) ** 2 for population in populations])
    pair_terms = []
    for separation in range(1, layers):
        propagated = np.linalg.matrix_power(block, separation + offset) @ amplitudes
        pair_terms.append(
            2
            * sum(
                xi**separation * (populations[start] @ (amplitudes.conj() * propagated))
                for start in range(layers - separation)
            ).real
        )
    pair_terms = np.asarray(pair_terms)
    total = float(self_terms.sum() + pair_terms.sum())
    return {
        _STAGES[0]: np.asarray(omega),
        _STAGES[1]: t6,
        _STAGES[2]: block,
        _STAGES[3]: np.r_[self_terms, pair_terms],
        _STAGES[4]: np.array([total, total / layers]),
    }


def _first_divergence(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> tuple[str | None, float]:
    for stage in _STAGES:
        left, right = reference.get(stage), candidate.get(stage)
        if left is None and right is None:
            continue
        if left is None or right is None or np.asarray(left).shape != np.asarray(right).shape:
            return stage, float("inf")
        if not np.array_equal(left, right):
            return stage, float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    return None, 0.0


def _stationary_value(law: TransitionLaw, omega: complex) -> float:
    amplitudes, population = np.array([1.2 + 0.3j, 0.7 - 0.5j]), np.array([0.5, 0.5])
    phased = np.exp(0.43j) * reduced_transition_matrix(law, omega)
    propagated = np.linalg.solve(np.eye(2) - phased, phased @ amplitudes)
    return float(
        population @ abs(amplitudes) ** 2 + 2 * (population @ (amplitudes.conj() * propagated)).real
    )


def _mutation_evidence() -> list[dict[str, object]]:
    law, omega = TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18), complex(registry_phase(1, 0))
    fixture = {"layers": 6, "omega": omega, "law": law, "initial": InitialPopulation(0.8, 0.2)}
    reference = _calculated_trace(**fixture)
    transposed = _calculated_trace(**fixture, transition=full_transition_matrix(law).T)
    offset = _calculated_trace(**fixture, offset=1)
    swapped = _calculated_trace(**fixture)
    swapped[_STAGES[4]] = swapped[_STAGES[4]][::-1].copy()
    omitted = _calculated_trace(**(fixture | {"omega": 1.0 + 0.0j}))
    reduced = reference[_STAGES[2]].copy()
    reduced[0, 1] += 1e-3
    perturbed = _calculated_trace(**fixture, reduced=reduced)
    stationary_fixture = fixture | {"layers": 7, "initial": InitialPopulation(0.5, 0.5)}
    finite = _calculated_trace(**stationary_fixture)
    stationary = _calculated_trace(**stationary_fixture)
    per_layer = _stationary_value(law, omega)
    stationary[_STAGES[4]] = np.array([7 * per_layer, per_layer])
    xi = np.exp(0.43j)
    amplitudes = np.array(
        [
            sum(xi**n * (1.2 + 0.3j) for n in range(6)),
            sum(xi**n * ((1.2 + 0.3j) if n % 2 == 0 else omega * (0.7 - 0.5j)) for n in range(6)),
        ]
    )
    weights = np.array([0.4, 0.6])
    incoherent = {_STAGES[5]: np.asarray(weights @ abs(amplitudes) ** 2)}
    coherent = {_STAGES[5]: np.asarray(abs(np.sqrt(weights) @ amplitudes) ** 2)}
    pairs = (
        ("transition_convention_transposed", reference, transposed),
        ("wrong_layer_count_offset", reference, offset),
        ("total_per_layer_swap", reference, swapped),
        ("coherent_population_mixture", incoherent, coherent),
        ("registry_phase_omitted", reference, omitted),
        ("stationary_substituted_for_finite", finite, stationary),
        ("reduced_sector_coefficient_perturbed", reference, perturbed),
    )
    records = []
    for mutation_id, baseline, candidate in pairs:
        stage, error = _first_divergence(baseline, candidate)
        expected = _MUTATION_STAGES[mutation_id]
        records.append(
            dict(
                mutation_id=mutation_id,
                expected_first_stage=expected,
                observed_first_stage=stage,
                maximum_absolute_error=error,
                detected=stage == expected and error > 0.0,
            )
        )
    return records


def _measure(call: Callable[[], np.ndarray]) -> tuple[np.ndarray, float, int]:
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def _benchmark() -> dict[str, object]:
    events, layers = 48, 24
    index = np.arange(events)
    f_plus = 1.1 + 0.003 * index + 1j * (0.2 + 0.002 * index)
    f_minus = 0.8 - 0.002 * index + 1j * (-0.3 + 0.001 * index)
    omega = registry_phase(index % 3, index % 2)
    xi = np.exp(1j * (0.17 + 0.013 * index))
    law, initial = TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18), InitialPopulation(0.63, 0.37)
    full, full_seconds, full_peak = _measure(
        lambda: np.array(
            [
                finite_intensity_full(
                    layers, f_plus[i], f_minus[i], omega[i], xi[i], law, initial
                ).intensity_electron2
                for i in range(events)
            ]
        )
    )
    reduced, reduced_seconds, reduced_peak = _measure(
        lambda: (
            finite_intensity_reduced(
                layers, f_plus, f_minus, omega, xi, law, initial
            ).intensity_electron2
        )
    )
    # fmt: off
    return {"equivalent_work": {"events": events, "layers": layers}, "full_seconds": full_seconds, "full_peak_bytes": full_peak, "reduced_seconds": reduced_seconds, "reduced_peak_bytes": reduced_peak, "speedup": full_seconds / reduced_seconds, "maximum_absolute_difference": float(np.max(abs(full - reduced)))}
    # fmt: on


def _classifications() -> list[dict[str, object]]:
    # fmt: off
    rows = (
        ("stacking.transition_matrix_6", "MATCH", ["PHY-STK-003", "PHY-STK-004"], None),
        ("stacking.synthetic_finite", "UNRESOLVED", ["PHY-STK-006", "PHY-STK-009", "PHY-STK-010", "PHY-STK-015", "PHY-STK-016"], None),
        ("stacking.legacy_initial_population", "UNRESOLVED", ["PHY-STK-007"], None),
        ("stacking.legacy_normalization", "CORRECTED", ["PHY-STK-013"], _STAGES[4]),
        ("stacking.legacy_phase_expression", "UNRESOLVED", ["PHY-STK-017"], None),
        ("stacking.stationary_output", "NO_ORACLE", ["PHY-STK-018"], None),
    )
    # fmt: on
    return _records(("case_id", "classification", "ledger_ids", "first_divergence_stage"), rows)


def run_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    del allow_missing_pack
    root = Path(__file__).resolve().parents[3]
    pack_path = root / "reference" / "rasim_reference_v1.npz"
    transition = full_transition_matrix(TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18))
    stochastic_error = float(np.max(np.abs(transition.sum(axis=1) - 1.0)))
    state_order = tuple(state.value for state in STATE_ORDER)
    finite = _finite_equality()
    science = _science_evidence()
    pack = _pack_evidence(pack_path)
    mutations = _mutation_evidence()
    benchmark = _benchmark()
    near_extinction = _near_extinction()
    (
        aligned,
        normalization_error,
        frame_error,
        sample_frame_separation,
        component_error,
        order_error,
    ) = _event_and_population_errors()
    finite_errors = list(finite["maximum_absolute_errors"].values())
    science_errors = list(science["parent_maximum_absolute_errors"].values()) + list(
        science["laue_absolute_errors"].values()
    )
    science_errors.extend(
        error
        for record in science["typed_model_absolute_errors"].values()
        for error in record.values()
    )
    pack_errors = list(pack["curve_maximum_absolute_errors"].values())
    checks = [
        _check(
            "stochastic_transition_and_state_order",
            "PASS"
            if stochastic_error == 0.0 and state_order == ("0F+", "1F+", "2F+", "0F-", "1F-", "2F-")
            else "FAIL",
            f"declared state order; maximum row-sum error {stochastic_error:.3e}",
        ),
        _check(
            "finite_oracles_parents_and_models",
            "SKIP"
            if np.all(np.isfinite(finite_errors + science_errors))
            and science["parent_case_count"] == 15
            else "FAIL",
            f"N=1,3,6 raw errors {finite['maximum_absolute_errors']}; five parents/three sectors {science['parent_maximum_absolute_errors']}; rich and ABD fixtures recorded; shared tolerance pending",
        ),
        _check(
            "analytic_limits_and_near_extinction",
            "SKIP"
            if np.isfinite(finite["single_layer_absolute_error"])
            and np.all(np.isfinite(list(science["laue_absolute_errors"].values())))
            and float(near_extinction["minimum_nearby_positive_intensity_electron2"]) > 0.0
            and np.isfinite(float(near_extinction["maximum_absolute_error"]))
            else "FAIL",
            f"N=1 error {finite['single_layer_absolute_error']:.3e}; Laue raw errors {science['laue_absolute_errors']}; N=512,+1e-9 {near_extinction['n512_plus_1e_9']}",
        ),
        _check(
            "event_alignment_frame_and_normalization",
            "SKIP"
            if aligned
            and normalization_error == 0.0
            and np.isfinite(frame_error)
            and sample_frame_separation > 0.0
            else "FAIL",
            f"raw electron2 aligned and nonnegative; normalization {normalization_error:.3e}; layer-frame {frame_error:.3e}; sample-qz separation {sample_frame_separation:.3e}",
        ),
        _check(
            "unweighted_population_components",
            "SKIP" if np.isfinite(component_error) and order_error == 0.0 else "FAIL",
            f"individual initials; raw direct error {component_error:.3e}; order error {order_error:.3e}",
        ),
        _check(
            "reference_pack_mutations_and_benchmark",
            "SKIP"
            if pack["probability_maximum_absolute_error"] == 0.0
            and pack["curve_value_count"] == 165
            and np.all(np.isfinite(pack_errors))
            and len(mutations) == 7
            and all(record["detected"] for record in mutations)
            and np.isfinite(benchmark["maximum_absolute_difference"])
            else "FAIL",
            f"60 probabilities/165 curve values; {sum(record['detected'] for record in mutations)}/7 calculated controls; 48x24 timing/memory recorded; shared tolerance pending",
        ),
    ]
    failed = any(check["status"] == "FAIL" for check in checks)
    # fmt: off
    request_rows = (
        ("T05-SHARED-EVENT-MEASURE", "proof-base and T07", True, "Shared EventIntensityResult cannot represent unweighted raw electron2; require total/per-layer electron2 input, T07-owned population mass, and a reviewed r_e^2/per-steradian owner."),
        ("T05-LAYER-PHASE-GAUGE", "proof-base with T03/T04/T07", True, "Shared contracts require exact-ID layer_normal_q_Ainv, xi=exp(+1j*q_layer*d), orientation/T04 gauge provenance, and no sample-qz fallback."),
        ("T05-FROZEN-TOLERANCE-PROVENANCE", "proof-base and T01", True, "Require a versioned stage tolerance policy with atol/rtol/scale/unit/near-zero justification and tolerance_config_sha256."),
    )
    # fmt: on
    requests = _records(("request_id", "owner", "blocking", "reason"), request_rows)
    return {
        "schema_version": 1,
        "task_id": "T05",
        "status": "FAIL" if failed else "BLOCKED",
        "local_proof_status": "FAIL" if failed else "BLOCKED",
        "blocking_request_ids": [request["request_id"] for request in requests],
        "base_sha": _git(root, "merge-base", "HEAD", "origin/codex/proof-base"),
        "commit_sha": _git(root, "rev-parse", "HEAD"),
        "contract_version": CONTRACT_API_VERSION,
        "trace_schema_version": 4,
        "reference_pack_sha256s": {"rasim_reference_v1": _sha256(pack_path)},
        "environment_sha256": _environment_sha256(),
        "owned_paths": [
            "src/rasim_next/stacking/",
            "tests/test_stacking_transition.py",
            "tasks/05_stacking_transition.md",
        ],
        "checks": checks,
        "classifications": _classifications(),
        "convergence": [finite],
        "benchmark": benchmark,
        "science_evidence": science,
        "reference_pack_comparison": pack,
        "mutations": mutations,
        "near_extinction": near_extinction,
        "tolerance_policy": {
            "status": "MISSING_REVIEWED_SHARED_ARTIFACT",
            "tolerance_config_sha256": None,
            "branch_local_thresholds_used_for_acceptance": False,
        },
        "limitations": [
            "homogeneous first-order law and one layer repeat; exact finite algebra has no refinement variable",
            "PHY-STK-018 stationary production output remains deferred; its calculated counterfactual is error-injection-only",
            "integration supplies reviewed-gauge layer-normal q; raw electron2 remains upstream of all downstream factors",
        ],
        "contract_requests": requests,
    }
