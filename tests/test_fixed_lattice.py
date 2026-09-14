from __future__ import annotations

import copy

import numpy as np
import pytest

from rasim_next.fitting.fixed_lattice import FixedLatticeState, fixed_lattice_from_fit_record

REFERENCE_BASIS_A = np.asarray(((4.0, -2.0, 0.0), (0.0, 3.4641016151377544, 0.0), (0.0, 0.0, 28.0)))


def test_implicit_and_retained_lattice_preserve_the_unmodified_cif_path() -> None:
    implicit = FixedLatticeState.implicit_cif(REFERENCE_BASIS_A)
    retained = FixedLatticeState.from_lattice_artifact(
        decision="RETAIN_CIF_LATTICE",
        reference_direct_basis_A=REFERENCE_BASIS_A,
        active_direct_basis_A=REFERENCE_BASIS_A,
        artifact_path="C:/evidence/lattice.json",
        artifact_sha256="a" * 64,
        position_artifact_sha256="b" * 64,
    )

    assert implicit.direct_basis_override_A is None
    assert retained.direct_basis_override_A is None
    assert (
        FixedLatticeState.from_record(
            retained.to_record(),
            reference_direct_basis_A=REFERENCE_BASIS_A,
        )
        == retained
    )


def test_accepted_lattice_round_trip_exposes_only_the_full_basis_override() -> None:
    active = REFERENCE_BASIS_A @ np.diag((1.0002, 1.0002, 0.9997))
    accepted = FixedLatticeState.from_lattice_artifact(
        decision="ACCEPT_FITTED_LATTICE",
        reference_direct_basis_A=REFERENCE_BASIS_A,
        active_direct_basis_A=active,
        artifact_path="C:/evidence/lattice.json",
        artifact_sha256="a" * 64,
        position_artifact_sha256="b" * 64,
    )

    np.testing.assert_array_equal(accepted.direct_basis_override_A, active)
    assert (
        FixedLatticeState.from_record(
            accepted.to_record(),
            reference_direct_basis_A=REFERENCE_BASIS_A,
        )
        == accepted
    )


def test_fixed_lattice_rejects_left_handed_or_changed_retained_basis() -> None:
    with pytest.raises(ValueError, match="right-handed"):
        FixedLatticeState.implicit_cif(REFERENCE_BASIS_A @ np.diag((-1.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match="retained"):
        FixedLatticeState.from_lattice_artifact(
            decision="RETAIN_CIF_LATTICE",
            reference_direct_basis_A=REFERENCE_BASIS_A,
            active_direct_basis_A=REFERENCE_BASIS_A * 1.001,
            artifact_path="C:/evidence/lattice.json",
            artifact_sha256="a" * 64,
            position_artifact_sha256="b" * 64,
        )


def _lattice_document(*, accepted: bool) -> dict[str, object]:
    active = REFERENCE_BASIS_A @ np.diag((1.0002, 1.0002, 0.9997))
    return {
        "schema_version": "rasim-osc-lattice-sensitivity-v2",
        "status": "ACCEPT_FITTED_LATTICE" if accepted else "RETAIN_CIF_LATTICE",
        "accepted": accepted,
        "selection_identity_rebased": False,
        "parameter_names": ["hexagonal_log_a_strain", "hexagonal_log_c_strain"],
        "reference": {"direct_basis_A": REFERENCE_BASIS_A.tolist()},
        "sensitivity_fit": {
            "log_strain": [np.log(1.0002), np.log(0.9997)],
            "direct_basis_A": active.tolist(),
            "prior_pull": [np.log(1.0002) / 5.0e-4, np.log(0.9997) / 5.0e-4],
            "data_improvement_fraction": 0.2,
            "optimizer_success": True,
            "active_bounds": [False, False],
            "data_sensitivity": {
                "practical_rank": 2,
                "condition": 20.0,
                "parameter_scales": [5.0e-4, 5.0e-4],
                "relative_tolerance": 1.0e-5,
                "prior_rows_included": False,
            },
        },
        "accepted_state": {"direct_basis_A": (active if accepted else REFERENCE_BASIS_A).tolist()},
        "regularization": {
            "mean_log_strain": [0.0, 0.0],
            "sigma_log_strain": [5.0e-4, 5.0e-4],
            "hard_bound_half_span": 2.0e-3,
        },
        "acceptance": {
            "minimum_data_improvement_fraction": 0.1,
            "maximum_absolute_prior_pull": 3.0,
            "required_data_practical_rank": 2,
            "maximum_data_condition": 1.0e5,
            "sensitivity_relative_tolerance": 1.0e-5,
            "root_audit": "SAME",
        },
        "model_pixelized": False,
        "intensity_evaluated": False,
        "provenance": {"position_sha256": "b" * 64},
    }


def test_lattice_artifact_adoption_requires_data_only_identifiability() -> None:
    accepted = fixed_lattice_from_fit_record(
        _lattice_document(accepted=True),
        artifact_path="C:/evidence/lattice.json",
        artifact_sha256="a" * 64,
        position_artifact_sha256="b" * 64,
        reference_direct_basis_A=REFERENCE_BASIS_A,
    )
    retained = fixed_lattice_from_fit_record(
        _lattice_document(accepted=False),
        artifact_path="C:/evidence/lattice.json",
        artifact_sha256="a" * 64,
        position_artifact_sha256="b" * 64,
        reference_direct_basis_A=REFERENCE_BASIS_A,
    )

    assert accepted.direct_basis_override_A is not None
    assert retained.direct_basis_override_A is None

    deficient = copy.deepcopy(_lattice_document(accepted=True))
    deficient["sensitivity_fit"]["data_sensitivity"]["practical_rank"] = 1
    with pytest.raises(ValueError, match="data-only"):
        fixed_lattice_from_fit_record(
            deficient,
            artifact_path="C:/evidence/lattice.json",
            artifact_sha256="a" * 64,
            position_artifact_sha256="b" * 64,
            reference_direct_basis_A=REFERENCE_BASIS_A,
        )

    self_authorized = copy.deepcopy(_lattice_document(accepted=True))
    self_authorized["acceptance"]["maximum_data_condition"] = 1.0e30
    with pytest.raises(ValueError, match="data-only"):
        fixed_lattice_from_fit_record(
            self_authorized,
            artifact_path="C:/evidence/lattice.json",
            artifact_sha256="a" * 64,
            position_artifact_sha256="b" * 64,
            reference_direct_basis_A=REFERENCE_BASIS_A,
        )

    loose_prior = copy.deepcopy(_lattice_document(accepted=True))
    loose_prior["regularization"]["sigma_log_strain"] = [1.0, 1.0]
    loose_prior["sensitivity_fit"]["data_sensitivity"]["parameter_scales"] = [1.0, 1.0]
    loose_prior["sensitivity_fit"]["prior_pull"] = [
        np.log(1.0002),
        np.log(0.9997),
    ]
    with pytest.raises(ValueError, match="data-only"):
        fixed_lattice_from_fit_record(
            loose_prior,
            artifact_path="C:/evidence/lattice.json",
            artifact_sha256="a" * 64,
            position_artifact_sha256="b" * 64,
            reference_direct_basis_A=REFERENCE_BASIS_A,
        )

    sheared = copy.deepcopy(_lattice_document(accepted=True))
    sheared_basis = np.asarray(sheared["accepted_state"]["direct_basis_A"], dtype=np.float64)
    sheared_basis[0, 1] += 0.01
    sheared["accepted_state"]["direct_basis_A"] = sheared_basis.tolist()
    sheared["sensitivity_fit"]["direct_basis_A"] = sheared_basis.tolist()
    with pytest.raises(ValueError, match="data-only"):
        fixed_lattice_from_fit_record(
            sheared,
            artifact_path="C:/evidence/lattice.json",
            artifact_sha256="a" * 64,
            position_artifact_sha256="b" * 64,
            reference_direct_basis_A=REFERENCE_BASIS_A,
        )
