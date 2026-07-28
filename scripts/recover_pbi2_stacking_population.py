"""Compact direct-profile proof for the PbI2 stacking-population fit."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from rasim_next.fitting import (
    StackingPopulationIdentifiabilityError,
    compile_pbi2_stacking_profile_response,
    fit_stacking_phase_totals,
)
from rasim_next.materials import read_crystal

ROOT = Path(__file__).resolve().parents[1]
CIF_SHA256 = "7cf2a5e1957ea63d277c704cff390724175f96e6d26f982287490eedc24afbf9"
LAYERS = 50
WAVELENGTH_A = 1.540592925
TRUTH_DOMAIN = np.asarray((0.60, 0.16, 0.09, 0.10, 0.05))
TRUTH_PHASE = np.asarray((0.60, 0.25, 0.15))
TRUTH_SCALE = 1.0e6
NOISE_SIGMA_A2 = 0.2
TRAINING_RODS = np.asarray(((-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1)))
HELDOUT_RODS = np.asarray(((-2, 0), (0, -2), (2, -2), (2, 0), (0, 2), (-2, 2)))


def _queries(rods: np.ndarray, l_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(rods, l_grid.size, axis=0), np.tile(l_grid, len(rods))


def run_proof() -> dict[str, object]:
    tracemalloc.start()
    started = time.perf_counter()
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
    truth_amount = TRUTH_SCALE * TRUTH_DOMAIN
    training_truth = training.component_response_A2 @ truth_amount
    variance = np.full(training_truth.size, NOISE_SIGMA_A2**2)
    noiseless = fit_stacking_phase_totals(training, training_truth, variance)
    random = np.random.default_rng(20260728)
    noisy = fit_stacking_phase_totals(
        training,
        training_truth + random.normal(0.0, NOISE_SIGMA_A2, training_truth.size),
        variance,
    )
    heldout_truth = heldout.component_response_A2 @ truth_amount
    heldout_observed = heldout_truth + random.normal(0.0, NOISE_SIGMA_A2, heldout_truth.size)
    heldout_prediction = heldout.component_response_A2 @ noisy.domain_amount
    heldout_weighted_rms = float(
        np.sqrt(np.mean(((heldout_observed - heldout_prediction) / NOISE_SIGMA_A2) ** 2))
    )

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
        "noiseless_phase_max_abs_error": float(
            np.max(np.abs(noiseless.phase_fraction - TRUTH_PHASE))
        ),
        "noisy_phase_max_abs_error": float(np.max(np.abs(noisy.phase_fraction - TRUTH_PHASE))),
        "heldout_weighted_rms": heldout_weighted_rms,
        "runtime_seconds": runtime_seconds,
    }
    acceptance = {
        "noiseless_phase": metrics["noiseless_phase_max_abs_error"] < 1.0e-10,
        "noisy_phase": metrics["noisy_phase_max_abs_error"] < 0.03,
        "noisy_profile_bounds": bool(
            np.all(noisy.phase_profile_bounds[:, 0] <= TRUTH_PHASE)
            and np.all(noisy.phase_profile_bounds[:, 1] >= TRUTH_PHASE)
        ),
        "heldout_profiles": heldout_weighted_rms < 1.25,
        "identifiability_injection": identifiability_injection_detected,
        "runtime": runtime_seconds < 60.0,
    }
    return {
        "schema": "rasim-pbi2-direct-stacking-profile-proof-v1",
        "legacy_classification": "NO_ORACLE",
        "first_divergence": "new direct signed-rod/L profile observable",
        "measure": "pointwise intrinsic scattering strength in A2; not a probability density",
        "convergence": "NOT_APPLICABLE: exact pointwise finite-stack evaluation",
        "training_profile_count": int(training_l.size),
        "heldout_profile_count": int(heldout_l.size),
        "heldout_design": "six distinct second-order signed rods with interleaved L coordinates",
        "truth_domain_fraction": TRUTH_DOMAIN.tolist(),
        "noisy_domain_fraction": noisy.domain_fraction.tolist(),
        "truth_phase_fraction": TRUTH_PHASE.tolist(),
        "noisy_phase_fraction": noisy.phase_fraction.tolist(),
        "phase_profile_bounds_delta_chi_square_1": noisy.phase_profile_bounds.tolist(),
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
