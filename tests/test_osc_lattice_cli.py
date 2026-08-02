from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fit_osc_lattice.py"
SPEC = importlib.util.spec_from_file_location("fit_osc_lattice_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


def test_lattice_data_sensitivity_uses_scaled_practical_rank() -> None:
    full_rank = np.asarray(((1.0, 0.0), (0.0, 1.0e-4), (0.5, 0.2)))
    deficient = np.asarray(((1.0, 0.0), (0.0, 1.0e-7), (0.5, 2.0e-8)))

    accepted = ADAPTER._scaled_sensitivity_diagnostics(
        full_rank,
        np.ones(2),
        relative_tolerance=1.0e-5,
    )
    rejected = ADAPTER._scaled_sensitivity_diagnostics(
        deficient,
        np.ones(2),
        relative_tolerance=1.0e-5,
    )

    assert accepted[1] == 2
    assert accepted[3] <= 1.0e5
    assert rejected[1] == 1


def test_lattice_promotion_rejects_prior_determined_solution() -> None:
    assert ADAPTER._lattice_fit_is_accepted(
        optimizer_success=True,
        selection_rebased=False,
        root_classification="SAME",
        active_bounds=np.zeros(2, dtype=np.bool_),
        prior_pull=np.zeros(2),
        improvement_fraction=0.2,
        data_rank=2,
        data_condition=10.0,
        maximum_data_condition=1.0e5,
    )
    assert not ADAPTER._lattice_fit_is_accepted(
        optimizer_success=True,
        selection_rebased=False,
        root_classification="SAME",
        active_bounds=np.zeros(2, dtype=np.bool_),
        prior_pull=np.zeros(2),
        improvement_fraction=0.2,
        data_rank=1,
        data_condition=10.0,
        maximum_data_condition=1.0e5,
    )
