"""Immutable adopted-lattice state shared by staged fitting adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
_SCHEMA = "rasim-fixed-lattice-state-v1"
LATTICE_FIT_SCHEMA_VERSION = "rasim-osc-lattice-sensitivity-v2"
LATTICE_PARAMETER_NAMES = (
    "hexagonal_log_a_strain",
    "hexagonal_log_c_strain",
)
LATTICE_SENSITIVITY_RELATIVE_TOLERANCE = 1.0e-5
LATTICE_MINIMUM_DATA_IMPROVEMENT_FRACTION = 0.10
LATTICE_MAXIMUM_ABSOLUTE_PRIOR_PULL = 3.0
LATTICE_MAXIMUM_PRIOR_SIGMA_LOG_STRAIN = 5.0e-4
LATTICE_MAXIMUM_ABSOLUTE_LOG_STRAIN = 2.0e-3
LATTICE_REQUIRED_DATA_PRACTICAL_RANK = 2
LATTICE_MAXIMUM_DATA_CONDITION = 1.0e5
_DECISIONS = frozenset(
    {
        "IMPLICIT_CIF_LATTICE",
        "RETAIN_CIF_LATTICE",
        "ACCEPT_FITTED_LATTICE",
    }
)


def _direct_basis(value: ArrayLike, name: str) -> FloatArray:
    basis = np.asarray(value, dtype=np.float64)
    if basis.shape != (3, 3) or np.any(~np.isfinite(basis)):
        raise ValueError(f"{name} must be one finite 3 by 3 direct basis")
    if float(np.linalg.det(basis)) <= 0.0:
        raise ValueError(f"{name} must be right-handed and nonsingular")
    result = np.array(basis, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _sha256(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def hexagonal_direct_basis(
    reference_direct_basis_A: ArrayLike,
    log_strain: ArrayLike,
) -> FloatArray:
    """Apply equal in-plane and independent normal log strains to a direct basis."""

    reference = _direct_basis(reference_direct_basis_A, "reference_direct_basis_A")
    strain = np.asarray(log_strain, dtype=np.float64)
    if strain.shape != (2,) or np.any(~np.isfinite(strain)):
        raise ValueError("hexagonal log strain must contain two finite values")
    result = np.asarray(reference @ np.diag(np.exp((strain[0], strain[0], strain[1]))))
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class FixedLatticeState:
    """The full direct basis adopted before mosaic and intensity fitting."""

    decision: str
    reference_direct_basis_A: FloatArray
    active_direct_basis_A: FloatArray
    artifact_path: str | None
    artifact_sha256: str | None
    position_artifact_sha256: str | None

    def __post_init__(self) -> None:
        decision = str(self.decision)
        if decision not in _DECISIONS:
            raise ValueError(f"unsupported fixed-lattice decision {decision!r}")
        reference = _direct_basis(self.reference_direct_basis_A, "reference_direct_basis_A")
        active = _direct_basis(self.active_direct_basis_A, "active_direct_basis_A")
        artifact_sha256 = _sha256(self.artifact_sha256, "artifact_sha256")
        position_sha256 = _sha256(
            self.position_artifact_sha256,
            "position_artifact_sha256",
        )
        if decision == "IMPLICIT_CIF_LATTICE":
            if any(
                value is not None
                for value in (self.artifact_path, artifact_sha256, position_sha256)
            ):
                raise ValueError("implicit CIF lattice cannot name an artifact")
        else:
            if not isinstance(self.artifact_path, str) or not self.artifact_path:
                raise ValueError("explicit lattice decision requires an artifact path")
            if artifact_sha256 is None or position_sha256 is None:
                raise ValueError("explicit lattice decision requires artifact hashes")
        if decision != "ACCEPT_FITTED_LATTICE" and not np.array_equal(active, reference):
            raise ValueError("implicit or retained lattice must equal the reference CIF basis")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "reference_direct_basis_A", reference)
        object.__setattr__(self, "active_direct_basis_A", active)
        object.__setattr__(self, "artifact_sha256", artifact_sha256)
        object.__setattr__(self, "position_artifact_sha256", position_sha256)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FixedLatticeState):
            return NotImplemented
        return bool(
            self.decision == other.decision
            and self.artifact_path == other.artifact_path
            and self.artifact_sha256 == other.artifact_sha256
            and self.position_artifact_sha256 == other.position_artifact_sha256
            and np.array_equal(self.reference_direct_basis_A, other.reference_direct_basis_A)
            and np.array_equal(self.active_direct_basis_A, other.active_direct_basis_A)
        )

    @classmethod
    def implicit_cif(cls, reference_direct_basis_A: ArrayLike) -> FixedLatticeState:
        basis = np.asarray(reference_direct_basis_A, dtype=np.float64)
        return cls(
            decision="IMPLICIT_CIF_LATTICE",
            reference_direct_basis_A=basis,
            active_direct_basis_A=basis,
            artifact_path=None,
            artifact_sha256=None,
            position_artifact_sha256=None,
        )

    @classmethod
    def from_lattice_artifact(
        cls,
        *,
        decision: str,
        reference_direct_basis_A: ArrayLike,
        active_direct_basis_A: ArrayLike,
        artifact_path: str,
        artifact_sha256: str,
        position_artifact_sha256: str,
    ) -> FixedLatticeState:
        return cls(
            decision=decision,
            reference_direct_basis_A=np.asarray(reference_direct_basis_A, dtype=np.float64),
            active_direct_basis_A=np.asarray(active_direct_basis_A, dtype=np.float64),
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            position_artifact_sha256=position_artifact_sha256,
        )

    @property
    def accepted(self) -> bool:
        return self.decision == "ACCEPT_FITTED_LATTICE"

    @property
    def direct_basis_override_A(self) -> FloatArray | None:
        return self.active_direct_basis_A if self.accepted else None

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA,
            "decision": self.decision,
            "accepted": self.accepted,
            "reference_direct_basis_A": self.reference_direct_basis_A.tolist(),
            "active_direct_basis_A": self.active_direct_basis_A.tolist(),
            "artifact": (
                None
                if self.artifact_path is None
                else {"path": self.artifact_path, "sha256": self.artifact_sha256}
            ),
            "position_artifact_sha256": self.position_artifact_sha256,
        }

    @classmethod
    def from_record(
        cls,
        record: object | None,
        *,
        reference_direct_basis_A: ArrayLike,
    ) -> FixedLatticeState:
        if record is None:
            return cls.implicit_cif(reference_direct_basis_A)
        if not isinstance(record, dict) or set(record) != {
            "schema_version",
            "decision",
            "accepted",
            "reference_direct_basis_A",
            "active_direct_basis_A",
            "artifact",
            "position_artifact_sha256",
        }:
            raise ValueError("fixed-lattice record has an invalid shape")
        if record["schema_version"] != _SCHEMA:
            raise ValueError("unsupported fixed-lattice record schema")
        decision = str(record["decision"])
        if record["accepted"] is not (decision == "ACCEPT_FITTED_LATTICE"):
            raise ValueError("fixed-lattice accepted flag contradicts its decision")
        recorded_reference = _direct_basis(
            record["reference_direct_basis_A"],
            "recorded reference_direct_basis_A",
        )
        expected_reference = _direct_basis(
            reference_direct_basis_A,
            "reference_direct_basis_A",
        )
        if not np.array_equal(recorded_reference, expected_reference):
            raise ValueError("fixed-lattice reference differs from the configured CIF")
        artifact = record["artifact"]
        if artifact is None:
            artifact_path = None
            artifact_sha256 = None
        elif isinstance(artifact, dict) and set(artifact) == {"path", "sha256"}:
            artifact_path = artifact["path"]
            artifact_sha256 = artifact["sha256"]
        else:
            raise ValueError("fixed-lattice artifact identity is invalid")
        return cls(
            decision=decision,
            reference_direct_basis_A=recorded_reference,
            active_direct_basis_A=record["active_direct_basis_A"],
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            position_artifact_sha256=record["position_artifact_sha256"],
        )


def fixed_lattice_from_fit_record(
    record: object,
    *,
    artifact_path: str,
    artifact_sha256: str,
    position_artifact_sha256: str,
    reference_direct_basis_A: ArrayLike,
) -> FixedLatticeState:
    """Adopt only a lattice result that passed its declared data-only gate."""

    if not isinstance(record, Mapping):
        raise ValueError("lattice result must contain one mapping")
    status = record.get("status")
    accepted = record.get("accepted")
    if (
        record.get("schema_version") != LATTICE_FIT_SCHEMA_VERSION
        or status not in {"RETAIN_CIF_LATTICE", "ACCEPT_FITTED_LATTICE"}
        or accepted is not (status == "ACCEPT_FITTED_LATTICE")
        or record.get("model_pixelized") is not False
        or record.get("intensity_evaluated") is not False
    ):
        raise ValueError("lattice result status or geometry-only contract is invalid")

    provenance = record.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("position_sha256") != position_artifact_sha256
    ):
        raise ValueError("lattice result is not bound to the supplied position fit")
    try:
        recorded_reference = _direct_basis(
            record["reference"]["direct_basis_A"],
            "lattice-result reference basis",
        )
        active = _direct_basis(
            record["accepted_state"]["direct_basis_A"],
            "lattice-result accepted basis",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("lattice result lacks a valid direct basis") from error
    expected_reference = _direct_basis(reference_direct_basis_A, "reference_direct_basis_A")
    if not np.array_equal(recorded_reference, expected_reference):
        raise ValueError("lattice result reference differs from the configured CIF")
    if not accepted:
        if not np.array_equal(active, recorded_reference):
            raise ValueError("retained lattice result changed the CIF basis")
    else:
        sensitivity = record.get("sensitivity_fit")
        policy = record.get("acceptance")
        regularization = record.get("regularization")
        if (
            not isinstance(sensitivity, Mapping)
            or not isinstance(policy, Mapping)
            or not isinstance(regularization, Mapping)
        ):
            raise ValueError("accepted lattice result lacks its data-only gate")
        data_sensitivity = sensitivity.get("data_sensitivity")
        if not isinstance(data_sensitivity, Mapping):
            raise ValueError("accepted lattice result lacks data-only sensitivity")
        try:
            fitted = _direct_basis(
                sensitivity["direct_basis_A"],
                "lattice-result fitted basis",
            )
            log_strain = np.asarray(sensitivity["log_strain"], dtype=np.float64)
            prior_pull = np.asarray(sensitivity["prior_pull"], dtype=np.float64)
            active_bounds = np.asarray(sensitivity["active_bounds"], dtype=np.bool_)
            improvement = float(sensitivity["data_improvement_fraction"])
            minimum_improvement = float(policy["minimum_data_improvement_fraction"])
            maximum_pull = float(policy["maximum_absolute_prior_pull"])
            required_rank = int(policy["required_data_practical_rank"])
            rank = int(data_sensitivity["practical_rank"])
            maximum_condition = float(policy["maximum_data_condition"])
            condition = float(data_sensitivity["condition"])
            parameter_scales = np.asarray(
                data_sensitivity["parameter_scales"],
                dtype=np.float64,
            )
            relative_tolerance = float(data_sensitivity["relative_tolerance"])
            declared_tolerance = float(policy["sensitivity_relative_tolerance"])
            regularization_mean = np.asarray(
                regularization["mean_log_strain"],
                dtype=np.float64,
            )
            regularization_sigma = np.asarray(
                regularization["sigma_log_strain"],
                dtype=np.float64,
            )
            hard_bound = float(regularization["hard_bound_half_span"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("accepted lattice result has malformed gate evidence") from error
        expected_fitted = hexagonal_direct_basis(recorded_reference, log_strain)
        if (
            tuple(record.get("parameter_names", ())) != LATTICE_PARAMETER_NAMES
            or record.get("selection_identity_rebased") is not False
            or policy.get("root_audit") != "SAME"
            or sensitivity.get("optimizer_success") is not True
            or data_sensitivity.get("prior_rows_included") is not False
            or active_bounds.shape != (2,)
            or np.any(active_bounds)
            or log_strain.shape != (2,)
            or np.any(~np.isfinite(log_strain))
            or parameter_scales.shape != (2,)
            or np.any(~np.isfinite(parameter_scales))
            or np.any(parameter_scales <= 0.0)
            or np.any(parameter_scales > LATTICE_MAXIMUM_PRIOR_SIGMA_LOG_STRAIN)
            or regularization_mean.shape != (2,)
            or np.any(regularization_mean != 0.0)
            or regularization_sigma.shape != (2,)
            or not np.array_equal(regularization_sigma, parameter_scales)
            or not math.isfinite(hard_bound)
            or hard_bound <= 0.0
            or hard_bound > LATTICE_MAXIMUM_ABSOLUTE_LOG_STRAIN
            or np.any(np.abs(log_strain) > hard_bound)
            or prior_pull.shape != (2,)
            or np.any(~np.isfinite(prior_pull))
            or not np.allclose(
                prior_pull,
                log_strain / parameter_scales,
                rtol=0.0,
                atol=1.0e-14,
            )
            or maximum_pull != LATTICE_MAXIMUM_ABSOLUTE_PRIOR_PULL
            or np.max(np.abs(prior_pull)) > maximum_pull
            or not math.isfinite(improvement)
            or minimum_improvement != LATTICE_MINIMUM_DATA_IMPROVEMENT_FRACTION
            or improvement < minimum_improvement
            or required_rank != LATTICE_REQUIRED_DATA_PRACTICAL_RANK
            or rank != required_rank
            or maximum_condition != LATTICE_MAXIMUM_DATA_CONDITION
            or not math.isfinite(condition)
            or condition < 1.0
            or condition > maximum_condition
            or relative_tolerance != LATTICE_SENSITIVITY_RELATIVE_TOLERANCE
            or declared_tolerance != LATTICE_SENSITIVITY_RELATIVE_TOLERANCE
            or not np.array_equal(fitted, expected_fitted)
            or not np.array_equal(active, fitted)
        ):
            raise ValueError("accepted lattice result failed its data-only gate")

    return FixedLatticeState.from_lattice_artifact(
        decision=str(status),
        reference_direct_basis_A=recorded_reference,
        active_direct_basis_A=active,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        position_artifact_sha256=position_artifact_sha256,
    )


__all__ = [
    "LATTICE_FIT_SCHEMA_VERSION",
    "LATTICE_MAXIMUM_ABSOLUTE_LOG_STRAIN",
    "LATTICE_MAXIMUM_ABSOLUTE_PRIOR_PULL",
    "LATTICE_MAXIMUM_DATA_CONDITION",
    "LATTICE_MAXIMUM_PRIOR_SIGMA_LOG_STRAIN",
    "LATTICE_MINIMUM_DATA_IMPROVEMENT_FRACTION",
    "LATTICE_PARAMETER_NAMES",
    "LATTICE_REQUIRED_DATA_PRACTICAL_RANK",
    "LATTICE_SENSITIVITY_RELATIVE_TOLERANCE",
    "FixedLatticeState",
    "fixed_lattice_from_fit_record",
    "hexagonal_direct_basis",
]
