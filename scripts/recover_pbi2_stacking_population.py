"""Independent direct-profile proof for the PbI2 stacking-population fit."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from rasim_next.core.contracts import RodQueryBatch
from rasim_next.core.scattering import electron_squared_to_scattering_strength_A2
from rasim_next.fitting import (
    STACKING_COMPONENT_IDS,
    StackingPopulationIdentifiabilityError,
    compile_pbi2_stacking_profile_response,
    fit_stacking_phase_totals,
)
from rasim_next.materials import CrystalStructure, read_crystal
from rasim_next.ordered import pbi2_layer_amplitudes
from rasim_next.stacking import InitialPopulation, TransitionLaw
from rasim_next.stacking.enumeration import finite_intensity_by_enumeration

ROOT = Path(__file__).resolve().parents[1]
CIF_SHA256 = "7cf2a5e1957ea63d277c704cff390724175f96e6d26f982287490eedc24afbf9"
LAYERS = 4
WAVELENGTH_A = 1.540592925
TRUTH_DOMAIN = np.asarray((0.60, 0.16, 0.09, 0.10, 0.05))
TRUTH_PHASE = np.asarray((0.60, 0.25, 0.15))
TRUTH_SCALE = 1.0e6
NOISE_SIGMA_A2 = 0.2
TRAINING_RODS = np.asarray(((-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1)))
HELDOUT_RODS = np.asarray(((-2, 0), (0, -2), (2, -2), (2, 0), (0, 2), (-2, 2)))
_PARENT_TRANSITION_INDEX = dict(zip(STACKING_COMPONENT_IDS, (0, 3, 4, 1, 2), strict=True))
_PARENT_EPSILON = 0.001


def _queries(rods: np.ndarray, l_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(rods, l_grid.size, axis=0), np.tile(l_grid, len(rods))


def _enumerated_parent_response_A2(
    crystal: CrystalStructure,
    *,
    signed_hk: np.ndarray,
    l_coordinate: np.ndarray,
    wavelength_A: float,
    layers: int,
) -> np.ndarray:
    """Evaluate the five parents by explicit short-stack path enumeration."""

    event_id = np.arange(l_coordinate.size, dtype=np.int64)
    _, rod_id = np.unique(signed_hk, axis=0, return_inverse=True)
    layer_normal = np.cross(crystal.direct_basis_A[:, 0], crystal.direct_basis_A[:, 1])
    layer_normal /= np.linalg.norm(layer_normal)
    layer_repeat_A = float(np.dot(crystal.direct_basis_A[:, 2], layer_normal))
    layer_q_Ainv = 2.0 * np.pi * l_coordinate / layer_repeat_A
    query = RodQueryBatch(
        event_id=event_id,
        rod_id=rod_id,
        phase_id=(crystal.phase_id,) * l_coordinate.size,
        h=signed_hk[:, 0].astype(np.int32),
        k=signed_hk[:, 1].astype(np.int32),
        q_sample_normal_Ainv=layer_q_Ainv,
        l_coordinate=l_coordinate,
        wavelength_A=np.full(l_coordinate.size, wavelength_A),
    )
    amplitudes = pbi2_layer_amplitudes(crystal, query, unknown_u_iso_A2=0.0)
    registry_index = np.remainder(query.h + 2 * query.k, 3)
    registry = np.exp(2j * np.pi * registry_index / 3.0)
    vertical = np.exp(2j * np.pi * l_coordinate)
    raw_per_layer = np.empty((l_coordinate.size, len(STACKING_COMPONENT_IDS)))
    initial = InitialPopulation.plus_only()
    for column, component_id in enumerate(STACKING_COMPONENT_IDS):
        parent = np.zeros(5)
        parent[_PARENT_TRANSITION_INDEX[component_id]] = 1.0
        law = TransitionLaw.from_array(
            (1.0 - _PARENT_EPSILON) * parent + _PARENT_EPSILON * (1.0 - parent) / 4.0
        )
        for row in range(l_coordinate.size):
            raw_per_layer[row, column] = finite_intensity_by_enumeration(
                layers,
                amplitudes.f_plus_e[row],
                amplitudes.f_minus_e[row],
                registry[row],
                vertical[row],
                law,
                initial,
            )
    return electron_squared_to_scattering_strength_A2(raw_per_layer / layers)


def run_proof() -> dict[str, object]:
    tracemalloc.start()
    started = time.perf_counter()
    training_rod_set = {tuple(map(int, rod)) for rod in TRAINING_RODS}
    heldout_rod_set = {tuple(map(int, rod)) for rod in HELDOUT_RODS}
    whole_rod_partition = training_rod_set.isdisjoint(heldout_rod_set)
    if not whole_rod_partition:
        raise ValueError("training and held-out signed rods must be disjoint")
    crystal = read_crystal(
        ROOT / "examples" / "pbi2" / "structures" / "PbI2_2H.cif",
        phase_id="pbi2-2h",
        expected_sha256=CIF_SHA256,
    )
    training_hk, training_l = _queries(TRAINING_RODS, np.linspace(-3.0, 3.0, 61))
    heldout_hk, heldout_l = _queries(HELDOUT_RODS, np.linspace(-2.95, 2.95, 60))
    training = compile_pbi2_stacking_profile_response(
        crystal,
        CIF_SHA256,
        signed_hk=training_hk,
        l_coordinate=training_l,
        wavelength_A=WAVELENGTH_A,
        layers=LAYERS,
    )
    heldout = compile_pbi2_stacking_profile_response(
        crystal,
        CIF_SHA256,
        signed_hk=heldout_hk,
        l_coordinate=heldout_l,
        wavelength_A=WAVELENGTH_A,
        layers=LAYERS,
    )
    training_oracle_A2 = _enumerated_parent_response_A2(
        crystal,
        signed_hk=training_hk,
        l_coordinate=training_l,
        wavelength_A=WAVELENGTH_A,
        layers=LAYERS,
    )
    heldout_oracle_A2 = _enumerated_parent_response_A2(
        crystal,
        signed_hk=heldout_hk,
        l_coordinate=heldout_l,
        wavelength_A=WAVELENGTH_A,
        layers=LAYERS,
    )
    truth_amount = TRUTH_SCALE * TRUTH_DOMAIN
    training_truth = training_oracle_A2 @ truth_amount
    variance = np.full(training_truth.size, NOISE_SIGMA_A2**2)
    noiseless = fit_stacking_phase_totals(training, training_truth, variance)
    random = np.random.default_rng(20260728)
    noisy = fit_stacking_phase_totals(
        training,
        training_truth + random.normal(0.0, NOISE_SIGMA_A2, training_truth.size),
        variance,
    )
    heldout_truth = heldout_oracle_A2 @ truth_amount
    heldout_observed = heldout_truth + random.normal(0.0, NOISE_SIGMA_A2, heldout_truth.size)
    heldout_prediction = heldout.component_response_A2 @ noisy.domain_amount
    heldout_weighted_rms = float(
        np.sqrt(np.mean(((heldout_observed - heldout_prediction) / NOISE_SIGMA_A2) ** 2))
    )
    heldout_rod_weighted_rms = {}
    for h, k in sorted(heldout_rod_set):
        mask = np.all(heldout_hk == (h, k), axis=1)
        weighted_residual = (heldout_observed[mask] - heldout_prediction[mask]) / NOISE_SIGMA_A2
        heldout_rod_weighted_rms[f"({h},{k})"] = float(np.sqrt(np.mean(weighted_residual**2)))
    maximum_heldout_rod_weighted_rms = max(heldout_rod_weighted_rms.values())

    control_hk = np.tile((-2, 1), (61, 1))
    control = compile_pbi2_stacking_profile_response(
        crystal,
        CIF_SHA256,
        signed_hk=control_hk,
        l_coordinate=np.linspace(-3.0, 3.0, control_hk.shape[0]),
        wavelength_A=WAVELENGTH_A,
        layers=LAYERS,
    )
    control_signal = control.component_response_A2 @ truth_amount
    try:
        fit_stacking_phase_totals(
            control,
            control_signal,
            np.full(control_signal.size, NOISE_SIGMA_A2**2),
        )
    except StackingPopulationIdentifiabilityError:
        identifiability_injection_detected = True
    else:
        identifiability_injection_detected = False

    runtime_seconds = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics = {
        "training_compiler_oracle_max_abs_error_A2": float(
            np.max(np.abs(training.component_response_A2 - training_oracle_A2))
        ),
        "heldout_compiler_oracle_max_abs_error_A2": float(
            np.max(np.abs(heldout.component_response_A2 - heldout_oracle_A2))
        ),
        "noiseless_phase_max_abs_error": float(
            np.max(np.abs(noiseless.phase_fraction - TRUTH_PHASE))
        ),
        "noisy_phase_max_abs_error": float(np.max(np.abs(noisy.phase_fraction - TRUTH_PHASE))),
        "heldout_weighted_rms": heldout_weighted_rms,
        "maximum_heldout_rod_weighted_rms": maximum_heldout_rod_weighted_rms,
        "runtime_seconds": runtime_seconds,
    }
    acceptance = {
        "training_compiler_matches_enumeration": bool(
            np.allclose(
                training.component_response_A2,
                training_oracle_A2,
                rtol=5.0e-13,
                atol=1.0e-18,
            )
        ),
        "heldout_compiler_matches_enumeration": bool(
            np.allclose(
                heldout.component_response_A2,
                heldout_oracle_A2,
                rtol=5.0e-13,
                atol=1.0e-18,
            )
        ),
        "noiseless_phase": metrics["noiseless_phase_max_abs_error"] < 1.0e-10,
        "noisy_phase": metrics["noisy_phase_max_abs_error"] < 0.03,
        "noiseless_profile_bounds": bool(
            np.all(noiseless.phase_profile_bounds[:, 0] <= TRUTH_PHASE)
            and np.all(noiseless.phase_profile_bounds[:, 1] >= TRUTH_PHASE)
        ),
        "whole_rod_partition": whole_rod_partition,
        "heldout_profiles": heldout_weighted_rms < 1.25,
        "every_heldout_rod": maximum_heldout_rod_weighted_rms < 1.25,
        "identifiability_injection": identifiability_injection_detected,
        "runtime": runtime_seconds < 60.0,
    }
    return {
        "schema": "rasim-pbi2-direct-stacking-profile-proof-v2",
        "legacy_classification": "NO_ORACLE",
        "first_divergence": "new direct signed-rod/L profile observable",
        "measure": "pointwise intrinsic scattering strength in A2; not a probability density",
        "oracle": "explicit six-state path enumeration; shares transition-law and layer-amplitude contracts but not the production finite-moment recurrence",
        "validation_scope": "synthetic numerical validation only; no measured PbI2 profiles are tracked",
        "convergence": "NOT_APPLICABLE: exact four-layer path enumeration and pointwise recurrence",
        "training_profile_count": int(training_l.size),
        "heldout_profile_count": int(heldout_l.size),
        "heldout_design": "all L points from six second-order signed rods absent from training",
        "heldout_rod_weighted_rms": heldout_rod_weighted_rms,
        "truth_domain_fraction": TRUTH_DOMAIN.tolist(),
        "noisy_domain_fraction": noisy.domain_fraction.tolist(),
        "truth_phase_fraction": TRUTH_PHASE.tolist(),
        "noisy_phase_fraction": noisy.phase_fraction.tolist(),
        "noiseless_phase_profile_bounds_delta_chi_square_1": (
            noiseless.phase_profile_bounds.tolist()
        ),
        "noisy_phase_profile_bounds_delta_chi_square_1": noisy.phase_profile_bounds.tolist(),
        "response_rank": noisy.response_rank,
        "phase_contrast_rank": noisy.phase_contrast_rank,
        "response_condition": noisy.response_condition,
        "phase_contrast_condition": noisy.phase_contrast_condition,
        "response_revision": training.response_revision,
        "python_peak_memory_MiB": peak_bytes / 2**20,
        "metrics": metrics,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
    }


def main() -> int:
    result = run_proof()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
