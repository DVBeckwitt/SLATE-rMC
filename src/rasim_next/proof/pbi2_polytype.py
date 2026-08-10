"""Exact ideal-polytype CIF versus PbI2 transition-parent benchmark."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rasim_next.core.contracts import (
    EventIntensityNormalization,
    LayerAmplitudeResult,
    LayerNormalQBatch,
    RodQueryBatch,
)
from rasim_next.core.scattering import (
    CLASSICAL_ELECTRON_RADIUS_A,
    electron_squared_to_scattering_strength_A2,
)
from rasim_next.materials.crystal import CrystalStructure, read_crystal
from rasim_next.materials.optics import atomic_scattering_factor_e
from rasim_next.ordered.motifs import extract_pbi2_motifs, pbi2_layer_amplitudes
from rasim_next.proof.tolerances import (
    STAGE_TOLERANCE_SHA256,
    STAGE_TOLERANCE_VERSION,
    load_stage_tolerances,
)
from rasim_next.stacking.finite_intensity import finite_event_intensity
from rasim_next.stacking.parent_models import RichEpsilonModel
from rasim_next.stacking.transition import InitialPopulation, Parent, TransitionLaw

_WAVELENGTH_A = 1.540592925
_SIGNED_RODS = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))
_BENCHMARK_MODEL_ID = "pbi2.ideal_shared_2h_motif.pure_parents.v1"
_NATIVE_2H_SHA256 = "7cf2a5e1957ea63d277c704cff390724175f96e6d26f982287490eedc24afbf9"


@dataclass(frozen=True, slots=True)
class _BenchmarkCase:
    label: str
    filename: str
    sha256: str
    period_layers: int
    parent: Parent
    wrong_parent: Parent
    wrong_initial: InitialPopulation
    expected_extinction_count: int
    expected_states: tuple[tuple[str, tuple[float, float, float]], ...]


_CASES = (
    _BenchmarkCase(
        "2H",
        "PbI2_2H_ideal.cif",
        "c2ebb0c719d249f682f31dc5ce69113bce079485b71aada0981691634a0cfa50",
        1,
        Parent.TWO_H,
        Parent.TWO_H,
        InitialPopulation.minus_only(),
        0,
        (("plus", (0.0, 0.0, 0.0)),),
    ),
    _BenchmarkCase(
        "4H+",
        "PbI2_4H_plus_ideal.cif",
        "bfceb8b65e258d4f71fb8a996afa4e90f3678526ef0c435731fbf7ffaa35b9c1",
        2,
        Parent.FOUR_H_PLUS,
        Parent.FOUR_H_MINUS,
        InitialPopulation.plus_only(),
        4,
        (
            ("plus", (0.0, 0.0, 0.0)),
            ("minus", (1.0 / 3.0, 2.0 / 3.0, 0.5)),
        ),
    ),
    _BenchmarkCase(
        "6H+",
        "PbI2_6H_plus_ideal.cif",
        "f4e733e9b06d211a8ceb32a6c00bf5c0016b90733d1fe959e00b0dec352c60bf",
        3,
        Parent.SIX_H_PLUS,
        Parent.SIX_H_MINUS,
        InitialPopulation.plus_only(),
        44,
        (
            ("plus", (0.0, 0.0, 0.0)),
            ("plus", (1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0)),
            ("plus", (2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0)),
        ),
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(check_id: str, passed: bool, evidence: str) -> dict[str, str]:
    return {"check_id": check_id, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def _reflection_hkl(period_layers: int) -> NDArray[np.int32]:
    rows = [
        (h, k, cell_l)
        for h, k in _SIGNED_RODS
        for cell_l in range(-2 * period_layers, 2 * period_layers + 1)
        if (h, k, cell_l) != (0, 0, 0)
    ]
    return np.asarray(rows, dtype=np.int32)


def _direct_cif_amplitude(
    crystal: CrystalStructure,
    hkl: NDArray[np.int32],
    wavelength_A: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Literal positive-phase atom sum independent of ordered/stacking amplitude helpers."""

    q_cartesian_Ainv = hkl @ (2.0 * np.pi * np.linalg.inv(crystal.direct_basis_A))
    q_magnitude_Ainv = np.linalg.norm(q_cartesian_Ainv, axis=1)
    amplitude_e = np.zeros(hkl.shape[0], dtype=np.complex128)
    absolute_term_sum_e = np.zeros(hkl.shape[0], dtype=np.float64)
    factor_by_species: dict[tuple[str, str, int], NDArray[np.complex128]] = {}
    for site in crystal.sites:
        if site.u_iso_A2 is None:
            raise ValueError("benchmark CIF sites must declare Uiso explicitly")
        key = (site.species, site.element, site.charge)
        if key not in factor_by_species:
            factor_by_species[key] = atomic_scattering_factor_e(
                species=site.species,
                element=site.element,
                charge=site.charge,
                q_magnitude_Ainv=q_magnitude_Ainv,
                wavelength_A=wavelength_A,
            )[0]
        phase = np.exp(2.0j * np.pi * (hkl @ np.asarray(site.fractional)))
        damping = np.exp(-0.5 * site.u_iso_A2 * q_magnitude_Ainv**2)
        term = site.occupancy * factor_by_species[key] * damping * phase
        amplitude_e += term
        absolute_term_sum_e += np.abs(term)
    return amplitude_e, absolute_term_sum_e


def _fixture_matches_contract(
    source: CrystalStructure, crystal: CrystalStructure, case: _BenchmarkCase
) -> bool:
    motifs = extract_pbi2_motifs(crystal)
    observed_states = tuple(
        (motif.orientation, crystal.sites[motif.atoms[0].site_index].fractional) for motif in motifs
    )
    states_match = len(observed_states) == len(case.expected_states) and all(
        orientation == expected_orientation
        and np.allclose(center, expected_center, rtol=0.0, atol=2.0e-15)
        for (orientation, center), (expected_orientation, expected_center) in zip(
            observed_states, case.expected_states, strict=True
        )
    )
    metric_matches = np.allclose(
        crystal.direct_basis_A[:, :2], source.direct_basis_A[:, :2], rtol=0.0, atol=1.0e-12
    ) and np.allclose(
        crystal.direct_basis_A[:, 2] / case.period_layers,
        source.direct_basis_A[:, 2],
        rtol=0.0,
        atol=1.0e-12,
    )
    return bool(
        crystal.spacegroup_hm == "P 1"
        and len(crystal.sites) == 3 * case.period_layers
        and all(site.occupancy == 1.0 and site.u_iso_A2 == 0.0 for site in crystal.sites)
        and states_match
        and metric_matches
    )


def _canonical_plus_motif_signature(
    crystal: CrystalStructure,
) -> tuple[tuple[str, str, int, float, tuple[float, float, float]], ...] | None:
    motifs = extract_pbi2_motifs(crystal)
    if len(motifs) != 1 or motifs[0].orientation not in {"plus", "minus"}:
        return None
    motif = motifs[0]
    reflect = -1.0 if motif.orientation == "minus" else 1.0
    rows = tuple(
        (
            atom.species,
            atom.element,
            atom.charge,
            atom.occupancy,
            (
                atom.fractional_offset[0],
                atom.fractional_offset[1],
                reflect * atom.fractional_offset[2],
            ),
        )
        for atom in motif.atoms
    )
    return tuple(sorted(rows, key=lambda row: (row[1] != "Pb", row[4][2])))


def _benchmark_source_matches_native(source: CrystalStructure, native_2h: CrystalStructure) -> bool:
    source_signature = _canonical_plus_motif_signature(source)
    native_signature = _canonical_plus_motif_signature(native_2h)
    if (
        source_signature is None
        or native_signature is None
        or len(source_signature) != len(native_signature)
    ):
        return False
    identities_match = all(
        source_row[:4] == native_row[:4]
        for source_row, native_row in zip(source_signature, native_signature, strict=True)
    )
    positions_match = all(
        np.allclose(source_row[4][:2], native_row[4][:2], rtol=0.0, atol=5.0e-7)
        and np.isclose(source_row[4][2], native_row[4][2], rtol=0.0, atol=1.0e-12)
        for source_row, native_row in zip(source_signature, native_signature, strict=True)
    )
    return bool(
        extract_pbi2_motifs(source)[0].orientation == "plus"
        and extract_pbi2_motifs(native_2h)[0].orientation == "minus"
        and identities_match
        and positions_match
        and np.allclose(source.direct_basis_A, native_2h.direct_basis_A, rtol=0.0, atol=1.0e-12)
    )


def _stack_strength_A2(
    query: RodQueryBatch,
    amplitudes: LayerAmplitudeResult,
    layer_normal_q: LayerNormalQBatch,
    *,
    law: TransitionLaw,
    layers: int,
    initial: InitialPopulation,
    normalization: EventIntensityNormalization,
) -> NDArray[np.float64]:
    return finite_event_intensity(
        query,
        amplitudes,
        law,
        layer_normal_q=layer_normal_q,
        layers=layers,
        initial=initial,
        model_component_id="ideal-parent",
        population_group_id=None,
        normalization=normalization,
    ).scattering_strength_A2


def _case_result(
    source: CrystalStructure,
    benchmark_directory: Path,
    case: _BenchmarkCase,
) -> dict[str, object]:
    path = benchmark_directory / case.filename
    crystal = read_crystal(path, phase_id=case.label, expected_sha256=case.sha256)
    hkl = _reflection_hkl(case.period_layers)
    wavelength_A = np.full(hkl.shape[0], _WAVELENGTH_A)
    direct_amplitude_e, direct_term_sum_e = _direct_cif_amplitude(crystal, hkl, wavelength_A)
    direct_A2 = electron_squared_to_scattering_strength_A2(np.abs(direct_amplitude_e) ** 2)

    layer_l = hkl[:, 2].astype(np.float64) / case.period_layers
    event_id = np.arange(hkl.shape[0], dtype=np.int64)
    query = RodQueryBatch(
        event_id,
        event_id,
        (source.phase_id,) * hkl.shape[0],
        hkl[:, 0],
        hkl[:, 1],
        2.0 * np.pi * layer_l / np.linalg.norm(source.direct_basis_A[:, 2]),
        layer_l,
        wavelength_A,
    )
    amplitudes = pbi2_layer_amplitudes(source, query)
    layer_normal_q = LayerNormalQBatch(
        event_id=query.event_id,
        rod_id=query.rod_id,
        phase_id=query.phase_id,
        layer_normal_q_Ainv=query.q_sample_normal_Ainv,
        gauge_id=amplitudes.gauge_id,
    )
    pure_law = TransitionLaw.for_parent(case.parent)
    initial = InitialPopulation.plus_only()
    stack_A2 = _stack_strength_A2(
        query,
        amplitudes,
        layer_normal_q,
        law=pure_law,
        layers=case.period_layers,
        initial=initial,
        normalization=EventIntensityNormalization.FINITE_TOTAL,
    )
    repeated_A2 = _stack_strength_A2(
        query,
        amplitudes,
        layer_normal_q,
        law=pure_law,
        layers=2 * case.period_layers,
        initial=initial,
        normalization=EventIntensityNormalization.FINITE_TOTAL,
    )
    wrong_A2 = _stack_strength_A2(
        query,
        amplitudes,
        layer_normal_q,
        law=TransitionLaw.for_parent(case.wrong_parent),
        layers=case.period_layers,
        initial=case.wrong_initial,
        normalization=EventIntensityNormalization.FINITE_TOTAL,
    )
    epsilon_A2 = _stack_strength_A2(
        query,
        amplitudes,
        layer_normal_q,
        law=RichEpsilonModel(case.parent, 0.001).transition_law(),
        layers=2 * case.period_layers,
        initial=initial,
        normalization=EventIntensityNormalization.FINITE_TOTAL,
    )
    per_layer_A2 = _stack_strength_A2(
        query,
        amplitudes,
        layer_normal_q,
        law=pure_law,
        layers=2 * case.period_layers,
        initial=initial,
        normalization=EventIntensityNormalization.FINITE_PER_LAYER,
    )

    tolerance = load_stage_tolerances()["stacking.finite_intensity"]
    maximum_layer_amplitude_e = np.maximum(
        np.abs(amplitudes.f_plus_e), np.abs(amplitudes.f_minus_e)
    )
    one_period_scale_A2 = electron_squared_to_scattering_strength_A2(
        np.maximum(
            direct_term_sum_e**2,
            (case.period_layers * maximum_layer_amplitude_e) ** 2,
        )
    )
    two_period_scale_A2 = electron_squared_to_scattering_strength_A2(
        np.maximum(
            4.0 * direct_term_sum_e**2,
            (2 * case.period_layers * maximum_layer_amplitude_e) ** 2,
        )
    )
    one_period_limit_A2 = tolerance.atol + tolerance.rtol * one_period_scale_A2
    two_period_limit_A2 = tolerance.atol + tolerance.rtol * two_period_scale_A2
    one_period_error_A2 = np.abs(stack_A2 - direct_A2)
    repeated_error_A2 = np.abs(repeated_A2 - 4.0 * direct_A2)
    direct_norm = float(np.linalg.norm(direct_A2))
    direct_support = direct_A2 > one_period_limit_A2
    stack_support = stack_A2 > one_period_limit_A2
    wrong_parent_nrmse = float(np.linalg.norm(wrong_A2 - direct_A2) / direct_norm)
    epsilon_violation = float(np.max(np.abs(epsilon_A2 - 4.0 * direct_A2) / two_period_limit_A2))
    per_layer_violation = float(
        np.max(np.abs(per_layer_A2 - 4.0 * direct_A2) / two_period_limit_A2)
    )
    electron_area_A2 = CLASSICAL_ELECTRON_RADIUS_A**2
    return {
        "parent": case.parent.value,
        "period_layers": case.period_layers,
        "cif_sha256": _sha256(path),
        "reflection_count": int(hkl.shape[0]),
        "registry_sectors": sorted({int((h + 2 * k) % 3) for h, k in hkl[:, :2]}),
        "fixture_contract_passed": _fixture_matches_contract(source, crystal, case),
        "maximum_absolute_intensity_error_e2": float(
            np.max(one_period_error_A2) / electron_area_A2
        ),
        "maximum_scaled_intensity_error": float(np.max(one_period_error_A2 / one_period_limit_A2)),
        "intensity_nrmse": float(np.linalg.norm(stack_A2 - direct_A2) / direct_norm),
        "intensity_r_factor": float(np.sum(one_period_error_A2) / np.sum(direct_A2)),
        "support_disagreement_count": int(np.count_nonzero(direct_support != stack_support)),
        "extinction_count": int(np.count_nonzero(~direct_support)),
        "expected_extinction_count": case.expected_extinction_count,
        "two_period_maximum_scaled_error": float(np.max(repeated_error_A2 / two_period_limit_A2)),
        "wrong_parent_nrmse": wrong_parent_nrmse,
        "epsilon_0p001_maximum_scaled_violation": epsilon_violation,
        "per_layer_maximum_scaled_violation": per_layer_violation,
    }


def run_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    """Run the frozen pure-parent Bragg-intensity benchmark without writing artifacts."""

    del allow_missing_pack
    root = Path(__file__).resolve().parents[3]
    benchmark_directory = root / "examples" / "pbi2" / "benchmark"
    source_case = _CASES[0]
    source = read_crystal(
        benchmark_directory / source_case.filename,
        phase_id="pbi2-ideal-layer",
        expected_sha256=source_case.sha256,
    )
    native_2h_path = root / "examples" / "pbi2" / "structures" / "PbI2_2H.cif"
    native_2h = read_crystal(
        native_2h_path,
        phase_id="pbi2-native-2h",
        expected_sha256=_NATIVE_2H_SHA256,
    )
    source_matches_native = _benchmark_source_matches_native(source, native_2h)
    polytypes = {case.label: _case_result(source, benchmark_directory, case) for case in _CASES}
    fixtures_pass = source_matches_native and all(
        bool(item["fixture_contract_passed"]) for item in polytypes.values()
    )
    one_period_pass = all(
        float(item["maximum_scaled_intensity_error"]) <= 1.0
        and int(item["support_disagreement_count"]) == 0
        and int(item["extinction_count"]) == int(item["expected_extinction_count"])
        for item in polytypes.values()
    )
    two_period_pass = all(
        float(item["two_period_maximum_scaled_error"]) <= 1.0 for item in polytypes.values()
    )
    controls_pass = all(
        float(item["wrong_parent_nrmse"]) > 1.0e-3
        and float(item["epsilon_0p001_maximum_scaled_violation"]) > 1.0
        and float(item["per_layer_maximum_scaled_violation"]) > 1.0
        for item in polytypes.values()
    )
    checks = [
        _check(
            "frozen_ideal_cif_contracts",
            fixtures_pass,
            "the rationalized benchmark source matches the SHA-pinned native 2H metric and reflected motif; explicit P1 cells contain the frozen +A, +A/-B, +A/+B/+C cycles",
        ),
        _check(
            "one_period_cif_bragg_intensity_parity",
            one_period_pass,
            "raw FINITE_TOTAL strengths agree pointwise with independent signed-CIF atom sums within the frozen stacking tolerance",
        ),
        _check(
            "two_period_closing_transition_and_normalization",
            two_period_pass,
            "two pure periods equal four times one conventional-cell intensity at every Bragg index",
        ),
        _check(
            "orientation_hand_purity_and_measure_controls",
            controls_pass,
            "wrong orientation/hand, epsilon=0.001, and FINITE_PER_LAYER substitutions all exceed acceptance",
        ),
    ]
    return {
        "schema_version": 1,
        "benchmark_model_id": _BENCHMARK_MODEL_ID,
        "status": "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL",
        "checks": checks,
        "native_2h_source": {
            "sha256": _sha256(native_2h_path),
            "benchmark_source_matches_reflected_native_motif": source_matches_native,
        },
        "polytypes": polytypes,
        "wavelength_A": _WAVELENGTH_A,
        "reflection_roster": {
            "signed_hk": [list(rod) for rod in _SIGNED_RODS],
            "cell_l": "inclusive [-2P,2P] for period P; (0,0,0) excluded",
            "multiplicity": 1,
        },
        "conventions": {
            "phase_sign": "POSITIVE_Q_DOT_R",
            "registry_phase": "exp[2pi*i*(h+2k)/3]",
            "layer_coordinate": "L=l_cell/P",
            "initial_population": "plus_only",
            "parent_fault_probability": 0.0,
            "normalization": "FINITE_TOTAL",
            "fitted_scale": False,
        },
        "tolerance_policy": {
            "artifact_version": STAGE_TOLERANCE_VERSION,
            "artifact_sha256": STAGE_TOLERANCE_SHA256,
            "stage_id": "stacking.finite_intensity",
        },
        "limitations": [
            "These synthetic CIFs validate exact ideal-parent topology and intensity assembly, not independently relaxed material structures.",
            "The ideal 6H+ benchmark is the opposite hand from the supplied relaxed R-3m 6H- CIF.",
            "Atomic factors are the shared declared XrayDB authority; the CIF atom sum is independent of ordered and stacking amplitude helpers.",
        ],
    }
