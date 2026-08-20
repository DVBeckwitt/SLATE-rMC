"""Fail-closed delayed acceptance for fixed-dataset and continuous-scan fits."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from rasim_next.fitting._scan_validation import (
    FloatArray,
)
from rasim_next.fitting._scan_validation import (
    bool_array as _bool_array,
)
from rasim_next.fitting._scan_validation import (
    float_array as _float_array,
)


@dataclass(frozen=True, slots=True)
class FixedDatasetScore:
    """Exact fixed-dataset score with one profiled scale per dataset."""

    parameters: ArrayLike
    chi_squared: float
    dataset_wrms: ArrayLike
    dataset_family_wrms: ArrayLike
    dataset_scales: ArrayLike
    parameter_at_bound: ArrayLike
    comparison_revision: str

    def __post_init__(self) -> None:
        parameters = _float_array(self.parameters, "parameters", ndim=1)
        dataset_wrms = _float_array(self.dataset_wrms, "dataset_wrms", ndim=1)
        family_wrms = _float_array(
            self.dataset_family_wrms,
            "dataset_family_wrms",
            ndim=2,
        )
        scales = _float_array(self.dataset_scales, "dataset_scales", ndim=1)
        at_bound = _bool_array(self.parameter_at_bound, "parameter_at_bound", ndim=1)
        chi_squared = float(self.chi_squared)
        if (
            not math.isfinite(chi_squared)
            or chi_squared < 0.0
            or family_wrms.shape[0] != dataset_wrms.size
            or scales.shape != dataset_wrms.shape
            or np.any(dataset_wrms < 0.0)
            or np.any(family_wrms < 0.0)
            or np.any(scales < 0.0)
            or at_bound.shape != parameters.shape
            or not isinstance(self.comparison_revision, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.comparison_revision) is None
        ):
            raise ValueError("fixed-dataset score is invalid")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "chi_squared", chi_squared)
        object.__setattr__(self, "dataset_wrms", dataset_wrms)
        object.__setattr__(self, "dataset_family_wrms", family_wrms)
        object.__setattr__(self, "dataset_scales", scales)
        object.__setattr__(self, "parameter_at_bound", at_bound)


@dataclass(frozen=True, slots=True)
class ScanScore:
    """Exact scan score and the physical blocks used by acceptance gates."""

    parameters: ArrayLike
    chi_squared: float
    profiled_scale: float
    family_chi_squared: ArrayLike
    complete_group_chi_squared: ArrayLike
    dominance_block_chi_squared: ArrayLike
    panel_model_contrast_A2: ArrayLike
    angular_convergence_wrms: float
    effect_uncertainty_chi_squared: float
    comparison_revision: str

    def __post_init__(self) -> None:
        parameters = _float_array(self.parameters, "parameters", ndim=1)
        families = _float_array(self.family_chi_squared, "family_chi_squared", ndim=1)
        groups = _float_array(
            self.complete_group_chi_squared,
            "complete_group_chi_squared",
            ndim=1,
        )
        dominance = _float_array(
            self.dominance_block_chi_squared,
            "dominance_block_chi_squared",
            ndim=1,
        )
        panel_mass = _float_array(
            self.panel_model_contrast_A2,
            "panel_model_contrast_A2",
            ndim=2,
        )
        chi_squared = float(self.chi_squared)
        scale = float(self.profiled_scale)
        convergence = float(self.angular_convergence_wrms)
        uncertainty = float(self.effect_uncertainty_chi_squared)
        if (
            not math.isfinite(chi_squared)
            or chi_squared < 0.0
            or not math.isfinite(scale)
            or scale < 0.0
            or np.any(families < 0.0)
            or np.any(groups < 0.0)
            or np.any(dominance < 0.0)
            or not math.isfinite(convergence)
            or convergence < 0.0
            or not math.isfinite(uncertainty)
            or uncertainty < 0.0
        ):
            raise ValueError("scan score is invalid")
        if (
            not isinstance(self.comparison_revision, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.comparison_revision) is None
        ):
            raise ValueError("comparison_revision must be a SHA-256 revision")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "chi_squared", chi_squared)
        object.__setattr__(self, "profiled_scale", scale)
        object.__setattr__(self, "family_chi_squared", families)
        object.__setattr__(self, "complete_group_chi_squared", groups)
        object.__setattr__(self, "dominance_block_chi_squared", dominance)
        object.__setattr__(self, "panel_model_contrast_A2", panel_mass)
        object.__setattr__(self, "angular_convergence_wrms", convergence)
        object.__setattr__(self, "effect_uncertainty_chi_squared", uncertainty)


@dataclass(frozen=True, slots=True)
class StagedAcceptancePolicy:
    """Explicit numerical gates and work cap for delayed acceptance."""

    noninferiority_margin_wrms: float
    maximum_accepted_outer_iterations: int
    maximum_scan_convergence_wrms: float
    complete_group_noninferiority_margin_chi_squared: float
    maximum_dominance_fraction: float

    def __post_init__(self) -> None:
        margin = float(self.noninferiority_margin_wrms)
        convergence = float(self.maximum_scan_convergence_wrms)
        group_margin = float(self.complete_group_noninferiority_margin_chi_squared)
        dominance = float(self.maximum_dominance_fraction)
        iterations = self.maximum_accepted_outer_iterations
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("noninferiority margin must be finite and nonnegative")
        if not math.isfinite(convergence) or convergence <= 0.0:
            raise ValueError("scan convergence gate must be finite and positive")
        if not math.isfinite(group_margin) or group_margin < 0.0:
            raise ValueError("complete-group margin must be finite and nonnegative")
        if not math.isfinite(dominance) or not 0.0 <= dominance <= 1.0:
            raise ValueError("dominance fraction must lie in [0, 1]")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("maximum accepted outer iterations must be positive")
        object.__setattr__(self, "noninferiority_margin_wrms", margin)
        object.__setattr__(self, "maximum_scan_convergence_wrms", convergence)
        object.__setattr__(
            self,
            "complete_group_noninferiority_margin_chi_squared",
            group_margin,
        )
        object.__setattr__(self, "maximum_dominance_fraction", dominance)


@dataclass(frozen=True, slots=True)
class StagedCandidateRecord:
    outer_iteration: int
    replacement: bool
    parameters: FloatArray
    fixed_score: FixedDatasetScore | None
    scan_score: ScanScore | None
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        iteration = self.outer_iteration
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("outer_iteration must be a nonnegative integer")
        if not isinstance(self.replacement, bool) or not isinstance(self.accepted, bool):
            raise TypeError("candidate record flags must be booleans")
        parameters = _float_array(self.parameters, "candidate parameters", ndim=1)
        fixed = self.fixed_score
        scan = self.scan_score
        if fixed is not None and not isinstance(fixed, FixedDatasetScore):
            raise TypeError("fixed_score must be FixedDatasetScore or None")
        if scan is not None and not isinstance(scan, ScanScore):
            raise TypeError("scan_score must be ScanScore or None")
        if scan is not None and fixed is None:
            raise ValueError("a scan score requires a fixed-dataset score")
        if fixed is not None and not np.array_equal(fixed.parameters, parameters):
            raise ValueError("fixed score parameters do not match the candidate")
        if scan is not None and not np.array_equal(scan.parameters, parameters):
            raise ValueError("scan score parameters do not match the candidate")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("candidate reason must be nonempty")
        if self.accepted != (fixed is not None and scan is not None and self.reason == "accepted"):
            raise ValueError("accepted candidate records require both scores and reason 'accepted'")
        object.__setattr__(self, "parameters", parameters)


def _same_fixed_score(left: FixedDatasetScore, right: FixedDatasetScore) -> bool:
    return (
        left.chi_squared == right.chi_squared
        and left.comparison_revision == right.comparison_revision
        and np.array_equal(left.parameters, right.parameters)
        and np.array_equal(left.dataset_wrms, right.dataset_wrms)
        and np.array_equal(left.dataset_family_wrms, right.dataset_family_wrms)
        and np.array_equal(left.dataset_scales, right.dataset_scales)
        and np.array_equal(left.parameter_at_bound, right.parameter_at_bound)
    )


def _same_scan_score(left: ScanScore, right: ScanScore) -> bool:
    return (
        left.chi_squared == right.chi_squared
        and left.profiled_scale == right.profiled_scale
        and left.angular_convergence_wrms == right.angular_convergence_wrms
        and left.effect_uncertainty_chi_squared == right.effect_uncertainty_chi_squared
        and left.comparison_revision == right.comparison_revision
        and np.array_equal(left.parameters, right.parameters)
        and np.array_equal(left.family_chi_squared, right.family_chi_squared)
        and np.array_equal(left.complete_group_chi_squared, right.complete_group_chi_squared)
        and np.array_equal(left.dominance_block_chi_squared, right.dominance_block_chi_squared)
        and np.array_equal(left.panel_model_contrast_A2, right.panel_model_contrast_A2)
    )


@dataclass(frozen=True, slots=True)
class StagedScanFitResult:
    initial_fixed_score: FixedDatasetScore
    initial_scan_score: ScanScore
    policy: StagedAcceptancePolicy
    records: tuple[StagedCandidateRecord, ...]
    final_fixed_score: FixedDatasetScore
    final_scan_score: ScanScore
    exact_scan_call_count: int
    outer_iterations_attempted: int
    accepted_outer_iteration_count: int
    stop_reason: str

    @property
    def final_parameters(self) -> FloatArray:
        return self.final_fixed_score.parameters

    def __post_init__(self) -> None:
        if not isinstance(self.policy, StagedAcceptancePolicy):
            raise TypeError("policy must be StagedAcceptancePolicy")
        for name in (
            "initial_fixed_score",
            "final_fixed_score",
        ):
            if not isinstance(getattr(self, name), FixedDatasetScore):
                raise TypeError(f"{name} must be FixedDatasetScore")
        for name in ("initial_scan_score", "final_scan_score"):
            if not isinstance(getattr(self, name), ScanScore):
                raise TypeError(f"{name} must be ScanScore")
        records = tuple(self.records)
        if any(not isinstance(record, StagedCandidateRecord) for record in records):
            raise TypeError("records must contain StagedCandidateRecord values")
        if not np.array_equal(
            self.initial_fixed_score.parameters,
            self.initial_scan_score.parameters,
        ) or not np.array_equal(
            self.final_fixed_score.parameters,
            self.final_scan_score.parameters,
        ):
            raise ValueError("fixed and scan scores must share their parameter vector")
        fixed_layout = (
            self.initial_fixed_score.comparison_revision,
            self.initial_fixed_score.dataset_wrms.shape,
            self.initial_fixed_score.dataset_family_wrms.shape,
            self.initial_fixed_score.dataset_scales.shape,
        )
        scan_layout = (
            self.initial_scan_score.comparison_revision,
            self.initial_scan_score.family_chi_squared.shape,
            self.initial_scan_score.complete_group_chi_squared.shape,
            self.initial_scan_score.dominance_block_chi_squared.shape,
            self.initial_scan_score.panel_model_contrast_A2.shape,
        )
        if (
            self.final_fixed_score.comparison_revision,
            self.final_fixed_score.dataset_wrms.shape,
            self.final_fixed_score.dataset_family_wrms.shape,
            self.final_fixed_score.dataset_scales.shape,
        ) != fixed_layout or (
            self.final_scan_score.comparison_revision,
            self.final_scan_score.family_chi_squared.shape,
            self.final_scan_score.complete_group_chi_squared.shape,
            self.final_scan_score.dominance_block_chi_squared.shape,
            self.final_scan_score.panel_model_contrast_A2.shape,
        ) != scan_layout:
            raise ValueError("initial and final scores must retain their comparison layouts")
        for record in records:
            if (
                record.fixed_score is not None
                and (
                    record.fixed_score.comparison_revision,
                    record.fixed_score.dataset_wrms.shape,
                    record.fixed_score.dataset_family_wrms.shape,
                    record.fixed_score.dataset_scales.shape,
                )
                != fixed_layout
            ):
                raise ValueError("candidate fixed score changed its comparison layout")
            if (
                record.scan_score is not None
                and (
                    record.scan_score.comparison_revision,
                    record.scan_score.family_chi_squared.shape,
                    record.scan_score.complete_group_chi_squared.shape,
                    record.scan_score.dominance_block_chi_squared.shape,
                    record.scan_score.panel_model_contrast_A2.shape,
                )
                != scan_layout
            ):
                raise ValueError("candidate scan score changed its comparison layout")
        if records:
            outer_ids = tuple(record.outer_iteration for record in records)
            attempted_ids = tuple(range(max(outer_ids) + 1))
            if tuple(dict.fromkeys(outer_ids)) != attempted_ids or outer_ids != tuple(
                sorted(outer_ids)
            ):
                raise ValueError("candidate records must use ordered contiguous outer iterations")
            for outer_iteration in attempted_ids:
                group = tuple(
                    record for record in records if record.outer_iteration == outer_iteration
                )
                if (
                    len(group) not in (1, 2)
                    or group[0].replacement
                    or (len(group) == 1 and not group[0].accepted)
                    or (len(group) == 2 and (group[0].accepted or not group[1].replacement))
                    or (outer_iteration < attempted_ids[-1] and not group[-1].accepted)
                ):
                    raise ValueError("candidate records violate delayed-acceptance ordering")
        current_fixed = self.initial_fixed_score
        current_scan = self.initial_scan_score
        for record in records:
            if record.fixed_score is None:
                if record.reason != "cheap_proposal_outside_trust_or_bounds":
                    raise ValueError("unscored candidate has an invalid rejection reason")
                continue
            fixed_reason = _fixed_gate_reason(
                record.fixed_score,
                self.initial_fixed_score,
                noninferiority_margin_wrms=self.policy.noninferiority_margin_wrms,
            )
            if record.scan_score is None:
                if fixed_reason is None or record.reason != fixed_reason:
                    raise ValueError("fixed-only candidate has an invalid rejection reason")
                continue
            if fixed_reason is not None:
                raise ValueError("a fixed-gate rejection cannot carry a scan score")
            scan_reason = _scan_gate_reason(
                record.fixed_score,
                record.scan_score,
                current_fixed,
                current_scan,
                self.initial_scan_score,
                self.policy,
            )
            expected_reason = "accepted" if scan_reason is None else scan_reason
            if record.reason != expected_reason or record.accepted != (scan_reason is None):
                raise ValueError("scanned candidate has an invalid acceptance decision")
            if record.accepted:
                current_fixed = record.fixed_score
                current_scan = record.scan_score
        accepted_records = tuple(record for record in records if record.accepted)
        expected_final_fixed = (
            self.initial_fixed_score if not accepted_records else accepted_records[-1].fixed_score
        )
        expected_final_scan = (
            self.initial_scan_score if not accepted_records else accepted_records[-1].scan_score
        )
        if not _same_fixed_score(
            self.final_fixed_score, expected_final_fixed
        ) or not _same_scan_score(self.final_scan_score, expected_final_scan):
            raise ValueError("final scores do not match the last accepted candidate")
        expected_scan_calls = 1 + sum(record.scan_score is not None for record in records)
        expected_attempted = (
            0 if not records else max(record.outer_iteration for record in records) + 1
        )
        counters = (
            self.exact_scan_call_count,
            self.outer_iterations_attempted,
            self.accepted_outer_iteration_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters
        ):
            raise ValueError("staged fit counters must be nonnegative integers")
        if (
            self.exact_scan_call_count != expected_scan_calls
            or self.outer_iterations_attempted != expected_attempted
            or self.accepted_outer_iteration_count != len(accepted_records)
            or self.outer_iterations_attempted > self.policy.maximum_accepted_outer_iterations
            or self.accepted_outer_iteration_count > self.policy.maximum_accepted_outer_iterations
            or self.exact_scan_call_count > 1 + 2 * self.policy.maximum_accepted_outer_iterations
        ):
            raise ValueError("staged fit counters do not match candidate records")
        if not isinstance(self.stop_reason, str) or not self.stop_reason:
            raise ValueError("stop_reason must be nonempty")
        if (
            records
            and records[-1].accepted
            and self.stop_reason != ("maximum_accepted_outer_iterations_reached")
        ):
            raise ValueError("an accepted terminal record must exhaust the policy work cap")
        object.__setattr__(self, "records", records)


def _fixed_gate_reason(
    candidate: FixedDatasetScore,
    anchor: FixedDatasetScore,
    *,
    noninferiority_margin_wrms: float,
) -> str | None:
    if np.any(candidate.parameter_at_bound):
        return "candidate_parameter_at_bound"
    if np.any(candidate.dataset_wrms > anchor.dataset_wrms + noninferiority_margin_wrms):
        return "fixed_dataset_noninferiority_failed"
    if np.any(
        candidate.dataset_family_wrms > anchor.dataset_family_wrms + noninferiority_margin_wrms
    ):
        return "fixed_family_noninferiority_failed"
    return None


def _scan_gate_reason(
    candidate_fixed: FixedDatasetScore,
    candidate_scan: ScanScore,
    current_fixed: FixedDatasetScore,
    current_scan: ScanScore,
    scan_anchor: ScanScore,
    policy: StagedAcceptancePolicy,
) -> str | None:
    if (
        candidate_scan.comparison_revision != scan_anchor.comparison_revision
        or candidate_scan.family_chi_squared.shape != scan_anchor.family_chi_squared.shape
        or candidate_scan.complete_group_chi_squared.shape
        != scan_anchor.complete_group_chi_squared.shape
        or candidate_scan.dominance_block_chi_squared.shape
        != scan_anchor.dominance_block_chi_squared.shape
        or candidate_scan.panel_model_contrast_A2.shape != scan_anchor.panel_model_contrast_A2.shape
    ):
        raise ValueError("candidate scan score changed its comparison blocks")
    if candidate_scan.profiled_scale <= 0.0:
        return "scan_profiled_scale_nonphysical"
    if candidate_scan.angular_convergence_wrms >= policy.maximum_scan_convergence_wrms:
        return "candidate_absolute_scan_convergence_failed"
    if np.any(
        candidate_scan.complete_group_chi_squared
        > scan_anchor.complete_group_chi_squared
        + policy.complete_group_noninferiority_margin_chi_squared
    ):
        return "complete_scan_group_noninferiority_failed"
    current_joint = current_fixed.chi_squared + current_scan.chi_squared
    candidate_joint = candidate_fixed.chi_squared + candidate_scan.chi_squared
    joint_improvement = current_joint - candidate_joint
    if joint_improvement <= candidate_scan.effect_uncertainty_chi_squared:
        return "joint_improvement_not_above_numerical_uncertainty"
    scan_improvement = current_scan.chi_squared - candidate_scan.chi_squared
    if scan_improvement > 0.0:
        block_improvement = (
            current_scan.dominance_block_chi_squared - candidate_scan.dominance_block_chi_squared
        )
        if (
            np.max(np.maximum(block_improvement, 0.0), initial=0.0)
            > policy.maximum_dominance_fraction * scan_improvement
        ):
            return "scan_improvement_dominated_by_one_pair_or_band"
    return None


def run_staged_delayed_acceptance(
    *,
    initial_fixed_score: FixedDatasetScore,
    exact_fixed_evaluator: Callable[[FloatArray], FixedDatasetScore],
    exact_scan_evaluator: Callable[[FloatArray], ScanScore],
    cheap_proposal: Callable[[FloatArray, float, bool], ArrayLike],
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    parameter_scales: ArrayLike,
    initial_trust_radius: float,
    policy: StagedAcceptancePolicy,
    initial_scan_score: ScanScore | None = None,
    baseline_gate: Callable[[FixedDatasetScore, ScanScore], str | None] | None = None,
) -> StagedScanFitResult:
    """Run fixed-first delayed acceptance with one replacement per outer step."""

    if not isinstance(initial_fixed_score, FixedDatasetScore):
        raise TypeError("initial_fixed_score must be FixedDatasetScore")
    if not isinstance(policy, StagedAcceptancePolicy):
        raise TypeError("policy must be StagedAcceptancePolicy")
    lower = _float_array(lower_bounds, "lower_bounds", ndim=1)
    upper = _float_array(upper_bounds, "upper_bounds", ndim=1)
    scales = _float_array(parameter_scales, "parameter_scales", ndim=1)
    parameter_count = initial_fixed_score.parameters.size
    if (
        lower.shape != (parameter_count,)
        or upper.shape != (parameter_count,)
        or scales.shape != (parameter_count,)
        or np.any(lower >= upper)
        or np.any(scales <= 0.0)
    ):
        raise ValueError("parameter bounds and scales are invalid")
    trust_radius = float(initial_trust_radius)
    if not math.isfinite(trust_radius) or trust_radius <= 0.0:
        raise ValueError("initial_trust_radius must be positive")

    initial_scan = (
        exact_scan_evaluator(initial_fixed_score.parameters)
        if initial_scan_score is None
        else initial_scan_score
    )
    exact_scan_calls = 1
    if not isinstance(initial_scan, ScanScore) or not np.array_equal(
        initial_scan.parameters,
        initial_fixed_score.parameters,
    ):
        raise ValueError("baseline scan score does not match the fixed-dataset optimum")
    if initial_scan.angular_convergence_wrms >= policy.maximum_scan_convergence_wrms:
        stop_reason = "baseline_absolute_scan_convergence_failed"
        return StagedScanFitResult(
            initial_fixed_score,
            initial_scan,
            policy,
            (),
            initial_fixed_score,
            initial_scan,
            exact_scan_calls,
            0,
            0,
            stop_reason,
        )
    if initial_scan.profiled_scale <= 0.0:
        stop_reason = "baseline_scan_profiled_scale_nonphysical"
        return StagedScanFitResult(
            initial_fixed_score,
            initial_scan,
            policy,
            (),
            initial_fixed_score,
            initial_scan,
            exact_scan_calls,
            0,
            0,
            stop_reason,
        )
    if baseline_gate is not None:
        reason = baseline_gate(initial_fixed_score, initial_scan)
        if reason is not None:
            if not isinstance(reason, str) or not reason:
                raise ValueError("baseline_gate must return a nonempty reason or None")
            return StagedScanFitResult(
                initial_fixed_score,
                initial_scan,
                policy,
                (),
                initial_fixed_score,
                initial_scan,
                exact_scan_calls,
                0,
                0,
                reason,
            )

    current_fixed = initial_fixed_score
    current_scan = initial_scan
    records: list[StagedCandidateRecord] = []
    accepted_count = 0
    attempted_count = 0
    stop_reason = "maximum_accepted_outer_iterations_reached"
    for outer in range(policy.maximum_accepted_outer_iterations):
        attempted_count += 1
        accepted_this_outer = False
        for replacement in (False, True):
            if replacement:
                trust_radius *= 0.5
            proposed = _float_array(
                cheap_proposal(current_fixed.parameters, trust_radius, replacement),
                "proposed parameters",
                ndim=1,
            )
            if proposed.shape != (parameter_count,):
                raise ValueError("cheap proposal returned the wrong parameter shape")
            scaled_step = (proposed - current_fixed.parameters) / scales
            outside_trust = float(np.linalg.norm(scaled_step)) > trust_radius * (
                1.0 + 64.0 * np.finfo(np.float64).eps
            )
            outside_bounds = np.any((proposed < lower) | (proposed > upper))
            if outside_trust or outside_bounds:
                reason = "cheap_proposal_outside_trust_or_bounds"
                records.append(
                    StagedCandidateRecord(
                        outer,
                        replacement,
                        proposed,
                        None,
                        None,
                        False,
                        reason,
                    )
                )
                if replacement:
                    stop_reason = "replacement_cheap_proposal_invalid"
                continue

            fixed_candidate = exact_fixed_evaluator(proposed)
            if not isinstance(fixed_candidate, FixedDatasetScore) or not np.array_equal(
                fixed_candidate.parameters,
                proposed,
            ):
                raise ValueError("exact fixed-dataset score does not match its candidate")
            if (
                fixed_candidate.comparison_revision != initial_fixed_score.comparison_revision
                or fixed_candidate.dataset_wrms.shape != initial_fixed_score.dataset_wrms.shape
                or fixed_candidate.dataset_family_wrms.shape
                != initial_fixed_score.dataset_family_wrms.shape
                or fixed_candidate.dataset_scales.shape != initial_fixed_score.dataset_scales.shape
            ):
                raise ValueError("candidate fixed-dataset score changed its comparison blocks")
            fixed_reason = _fixed_gate_reason(
                fixed_candidate,
                initial_fixed_score,
                noninferiority_margin_wrms=policy.noninferiority_margin_wrms,
            )
            if fixed_reason is not None:
                records.append(
                    StagedCandidateRecord(
                        outer,
                        replacement,
                        proposed,
                        fixed_candidate,
                        None,
                        False,
                        fixed_reason,
                    )
                )
                if replacement:
                    stop_reason = "replacement_exact_fixed_dataset_gate_failed"
                continue

            scan_candidate = exact_scan_evaluator(proposed)
            exact_scan_calls += 1
            if not isinstance(scan_candidate, ScanScore) or not np.array_equal(
                scan_candidate.parameters,
                proposed,
            ):
                raise ValueError("exact scan score does not match its candidate")
            scan_reason = _scan_gate_reason(
                fixed_candidate,
                scan_candidate,
                current_fixed,
                current_scan,
                initial_scan,
                policy,
            )
            accepted = scan_reason is None
            records.append(
                StagedCandidateRecord(
                    outer,
                    replacement,
                    proposed,
                    fixed_candidate,
                    scan_candidate,
                    accepted,
                    "accepted" if accepted else scan_reason,
                )
            )
            if accepted:
                current_fixed = fixed_candidate
                current_scan = scan_candidate
                accepted_count += 1
                accepted_this_outer = True
                break
            if replacement:
                stop_reason = f"replacement_{scan_reason}"
        if not accepted_this_outer:
            break

    if exact_scan_calls > 1 + 2 * policy.maximum_accepted_outer_iterations:
        raise RuntimeError("exact scan call cap was violated")
    return StagedScanFitResult(
        initial_fixed_score=initial_fixed_score,
        initial_scan_score=initial_scan,
        policy=policy,
        records=tuple(records),
        final_fixed_score=current_fixed,
        final_scan_score=current_scan,
        exact_scan_call_count=exact_scan_calls,
        outer_iterations_attempted=attempted_count,
        accepted_outer_iteration_count=accepted_count,
        stop_reason=stop_reason,
    )


__all__ = [
    "FixedDatasetScore",
    "ScanScore",
    "StagedAcceptancePolicy",
    "StagedCandidateRecord",
    "StagedScanFitResult",
    "run_staged_delayed_acceptance",
]
