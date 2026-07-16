"""Compact proof for finite stacking-transition intensities."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
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
from rasim_next.stacking.parent_models import StackingPopulation
from rasim_next.stacking.transition import (
    STATE_ORDER,
    InitialPopulation,
    Parent,
    StackingState,
    TransitionLaw,
    full_transition_matrix,
    registry_phase,
)

_ORACLE_ATOL = 5e-13


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_sha256() -> str:
    payload = json.dumps(
        {
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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
    populations = [
        np.array([initial.plus, 0.0, 0.0, initial.minus, 0.0, 0.0], dtype=np.float64)
    ]
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
            vertical_power
            * (populations[start] @ (np.conj(amplitudes) * propagated))
            for start in range(layers - separation)
        )
    return float(self_total + 2.0 * pair_total.real)


def _short_stack_errors() -> dict[str, float]:
    arguments = (
        6,
        1.2 + 0.3j,
        0.7 - 0.5j,
        complex(registry_phase(1, 0)),
        np.exp(0.43j),
        TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18),
        InitialPopulation(0.8, 0.2),
    )
    direct = float(finite_intensity_by_enumeration(*arguments).intensity_electron2)
    return {
        "pair": abs(_direct_pair_intensity(*arguments) - direct),
        "full": abs(float(finite_intensity_full(*arguments).intensity_electron2) - direct),
        "reduced": abs(float(finite_intensity_reduced(*arguments).intensity_electron2) - direct),
    }


def _single_layer_error() -> float:
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
    return max(
        abs(float(evaluator(*arguments).intensity_electron2) - expected)
        for evaluator in (finite_intensity_full, finite_intensity_reduced)
    )


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
        "layer_repeat_A": 3.4,
        "population_group_id": "parents",
    }
    components = finite_population_event_intensity(populations=populations, **arguments)
    reversed_components = finite_population_event_intensity(
        populations=tuple(reversed(populations)), **arguments
    )
    omega = np.asarray(registry_phase(query.h, query.k))
    vertical_phase = np.exp(1j * layer_normal_q.layer_normal_q_Ainv * 3.4)
    direct = np.array(
        [
            [
                finite_intensity_by_enumeration(
                    6,
                    amplitudes.f_plus[event_index],
                    amplitudes.f_minus[event_index],
                    omega[event_index],
                    vertical_phase[event_index],
                    population.model,
                    population.initial,
                ).intensity_electron2
                for event_index in range(query.event_id.size)
            ]
            for population in populations
        ]
    )
    component_error = float(np.max(np.abs(components.component_intensity_electron2 - direct)))
    order_error = float(
        np.max(
            np.abs(
                components.component_intensity_electron2
                - reversed_components.component_intensity_electron2
            )
        )
    )
    aligned_and_unweighted = bool(
        np.array_equal(event.event_id, query.event_id)
        and np.array_equal(components.event_id, query.event_id)
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


def _classifications() -> list[dict[str, object]]:
    return [
        {
            "case_id": "stacking.transition_matrix_6",
            "classification": "MATCH",
            "ledger_ids": ["PHY-STK-003", "PHY-STK-004"],
            "first_divergence_stage": None,
        },
        {
            "case_id": "stacking.legacy_initial_population",
            "classification": "CORRECTED",
            "ledger_ids": ["PHY-STK-007"],
            "first_divergence_stage": "stacking.pair_kernel",
        },
        {
            "case_id": "stacking.legacy_normalization",
            "classification": "CORRECTED",
            "ledger_ids": ["PHY-STK-010", "PHY-STK-018"],
            "first_divergence_stage": "stacking.finite_intensity",
        },
    ]


def run_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    del allow_missing_pack
    root = Path(__file__).resolve().parents[3]
    pack_path = root / "reference" / "rasim_reference_v1.npz"
    transition = full_transition_matrix(TransitionLaw(0.17, 0.23, 0.11, 0.31, 0.18))
    stochastic_error = float(np.max(np.abs(transition.sum(axis=1) - 1.0)))
    state_order = tuple(state.value for state in STATE_ORDER)
    short_stack = _short_stack_errors()
    single_layer_error = _single_layer_error()
    near_extinction = _near_extinction()
    (
        aligned,
        normalization_error,
        frame_error,
        sample_frame_separation,
        component_error,
        order_error,
    ) = _event_and_population_errors()
    checks = [
        _check(
            "stochastic_transition",
            stochastic_error <= 1e-15
            and state_order == ("0F+", "1F+", "2F+", "0F-", "1F-", "2F-"),
            f"declared state order; maximum row-sum error {stochastic_error:.3e}",
        ),
        _check(
            "short_stack_oracles",
            max(short_stack.values()) <= _ORACLE_ATOL,
            f"N=6 enumeration versus pair/full/reduced errors {short_stack}",
        ),
        _check(
            "single_layer_limit",
            single_layer_error <= _ORACLE_ATOL,
            f"maximum full/reduced error {single_layer_error:.3e}",
        ),
        _check(
            "coherent_zero_neighborhood",
            float(near_extinction["minimum_nearby_positive_intensity_electron2"]) > 0.0
            and float(near_extinction["maximum_absolute_error"]) <= _ORACLE_ATOL,
            f"N=512,+1e-9 direct/full/reduced {near_extinction['n512_plus_1e_9']}",
        ),
        _check(
            "event_alignment_and_normalization",
            aligned
            and normalization_error <= _ORACLE_ATOL
            and frame_error <= _ORACLE_ATOL
            and sample_frame_separation > _ORACLE_ATOL,
            "raw electron2 event IDs aligned; "
            f"total/per-layer error {normalization_error:.3e}; "
            f"layer-normal/direct error {frame_error:.3e}; "
            f"sample-qz counterfactual separation {sample_frame_separation:.3e}",
        ),
        _check(
            "unweighted_population_components",
            component_error <= _ORACLE_ATOL and order_error == 0.0,
            f"individual initials; direct error {component_error:.3e}; order error {order_error:.3e}",
        ),
    ]
    passed = all(check["status"] == "PASS" for check in checks)
    return {
        "schema_version": 1,
        "task_id": "T05",
        "status": "READY" if passed else "FAIL",
        "base_sha": _git(root, "merge-base", "HEAD", "main"),
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
        "near_extinction": near_extinction,
        "limitations": [
            "homogeneous first-order transition law with one explicit layer repeat per event batch",
            "integration must supply event-aligned layer-normal q from the declared crystallite/layer frame",
            "raw electron2 remains upstream of universal scattering-scale and per-steradian factors",
        ],
        "contract_requests": [
            {
                "request_id": "T05-T07-INTEGRATION-BOUNDARY",
                "owner": "T07 integration",
                "blocking": False,
                "reason": "The shared query exposes sample-frame qz rather than layer-normal q, the shared per-steradian result cannot represent T05 raw electron2 without a false unit, and T07 owns population mass.",
                "required": {
                    "layer_phase_input": "event_id[E] int64 in exact query order plus finite layer_normal_q_Ainv[E] float64 in inverse angstroms",
                    "layer_gauge": "T07 projects each real event Q onto its crystallite/layer normal in the same T04 motif and first-layer gauge; sample-frame qz is never substituted",
                    "raw_event_input": "event_id plus total/per-layer electron2 and model/population IDs",
                    "population_weights": "separate ID-aligned finite nonnegative masses summing to one",
                    "ownership": "T07 applies population mass and any r_e^2/per-steradian conversion exactly once",
                },
            }
        ],
    }
