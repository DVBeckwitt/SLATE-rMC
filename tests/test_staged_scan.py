from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rasim_next.fitting.adaptive_scan import (
    ScanNodeEvaluation,
    evaluate_adaptive_scan_oracle,
)
from rasim_next.fitting.staged_scan import (
    FixedDatasetScore,
    ScanScore,
    StagedAcceptancePolicy,
    run_staged_delayed_acceptance,
)
from rasim_next.pipeline.incidence_acquisition import ContinuousIncidenceAcquisition

CALIBRATION_REVISION = f"sha256-{'a' * 64}"
EVALUATION_REVISION = "e" * 64
ROOT = Path(__file__).resolve().parents[1]
POLICY = StagedAcceptancePolicy(
    noninferiority_margin_wrms=0.25,
    maximum_accepted_outer_iterations=3,
    maximum_scan_convergence_wrms=0.25,
    complete_group_noninferiority_margin_chi_squared=0.0,
    maximum_dominance_fraction=0.5,
)


def test_uniform_scan_acquisition_integrates_constant_and_linear_functions() -> None:
    acquisition = ContinuousIncidenceAcquisition.from_mapping(
        json.loads(
            (ROOT / "examples/bi2se3/experiment/continuous_scan_acquisition.json").read_text(
                encoding="utf-8"
            )
        )
    )
    assert acquisition.to_mapping() == json.loads(
        (ROOT / "examples/bi2se3/experiment/continuous_scan_acquisition.json").read_text(
            encoding="utf-8"
        )
    )
    quadrature = acquisition.uniform_panel_quadrature(
        incidence_angle_calibration_revision=CALIBRATION_REVISION,
        panel_edges_deg=(5.0, 10.0, 15.0, 20.0, 25.0),
    )

    constant = np.full(quadrature.incidence_angle_rad.shape, 7.0)
    assert quadrature.integrate(constant) == 7.0
    np.testing.assert_array_equal(
        quadrature.integrate(np.full((64, 3), 7.0)),
        np.full(3, 7.0),
    )
    np.testing.assert_allclose(
        quadrature.integrate(quadrature.commanded_incidence_angle_rad),
        math.radians(15.0),
        rtol=0.0,
        atol=3.0e-16,
    )
    assert quadrature.incidence_angle_rad.size == 64
    np.testing.assert_array_equal(np.bincount(quadrature.panel_index), 16)

    delta = math.radians(0.4315724945)
    calibrated = acquisition.uniform_panel_quadrature(
        incidence_angle_calibration_revision=CALIBRATION_REVISION,
        effective_incidence_angle_rad=quadrature.commanded_incidence_angle_rad + delta,
        panel_edges_deg=(5.0, 10.0, 15.0, 20.0, 25.0),
    )
    np.testing.assert_allclose(
        calibrated.incidence_angle_rad,
        calibrated.commanded_incidence_angle_rad + delta,
        rtol=0.0,
        atol=2.0e-16,
    )


def test_scan_direction_and_step_metadata_do_not_change_the_prediction() -> None:
    forward = ContinuousIncidenceAcquisition(5.0, 25.0, scan_image_step_deg=0.1)
    reverse = ContinuousIncidenceAcquisition(25.0, 5.0, scan_image_step_deg=0.1)
    different_step = ContinuousIncidenceAcquisition(5.0, 25.0, scan_image_step_deg=0.25)
    different_motor_metadata = ContinuousIncidenceAcquisition(
        5.0,
        25.0,
        forward_reverse_speed_difference_percent=0.01,
        endpoint_excess_exposure_percent=0.05,
        interior_dwell_rms_deviation_percent=0.3,
    )

    forward_rule = forward.uniform_panel_quadrature(
        incidence_angle_calibration_revision=CALIBRATION_REVISION,
    )
    reverse_rule = reverse.uniform_panel_quadrature(
        incidence_angle_calibration_revision=CALIBRATION_REVISION,
    )
    step_rule = different_step.uniform_panel_quadrature(
        incidence_angle_calibration_revision=CALIBRATION_REVISION,
    )

    assert forward.nominal_start_deg == reverse.nominal_start_deg == 5.0
    assert forward.nominal_stop_deg == reverse.nominal_stop_deg == 25.0
    assert forward.exposure_time_s is None
    assert forward.one_way_traversal_count is None
    assert forward.complete_cycle_count is None
    assert forward.forward_reverse_speed_difference_percent is None
    assert forward.endpoint_excess_exposure_percent is None
    assert forward.interior_dwell_rms_deviation_percent is None
    assert forward.motor_exposure == "uniform_normalized"
    assert forward.prediction_revision == reverse.prediction_revision
    assert forward.prediction_revision == different_step.prediction_revision
    assert forward.prediction_revision == different_motor_metadata.prediction_revision
    assert forward.provenance_revision != reverse.provenance_revision
    assert forward.provenance_revision != different_step.provenance_revision
    assert forward.provenance_revision != different_motor_metadata.provenance_revision
    assert forward_rule.quadrature_revision == reverse_rule.quadrature_revision
    assert forward_rule.quadrature_revision == step_rule.quadrature_revision


def test_scan_cache_rejects_a_different_physical_support() -> None:
    old = ContinuousIncidenceAcquisition(5.0, 20.0)
    corrected = ContinuousIncidenceAcquisition(5.0, 25.0)
    cached_rule = old.uniform_panel_quadrature(
        incidence_angle_calibration_revision=CALIBRATION_REVISION,
        panel_edges_deg=(5.0, 10.0, 15.0, 20.0),
    )
    with pytest.raises(ValueError, match="acquisition support"):
        cached_rule.require_acquisition(corrected)


def _fixed_score(parameter: float, *, wrms: float = 1.0) -> FixedDatasetScore:
    return FixedDatasetScore(
        parameters=np.asarray((parameter,)),
        chi_squared=10.0 - parameter,
        dataset_wrms=np.full(2, wrms),
        dataset_family_wrms=np.full((2, 2), wrms),
        dataset_scales=np.ones(2),
        parameter_at_bound=np.zeros(1, dtype=np.bool_),
        comparison_revision="b" * 64,
    )


def _scan_score(parameter: float, *, accepted: bool = True) -> ScanScore:
    return ScanScore(
        parameters=np.asarray((parameter,)),
        chi_squared=(20.0 - 2.0 * parameter if accepted else 30.0 + parameter),
        profiled_scale=1.0,
        family_chi_squared=np.full(2, 5.0 - 0.2 * parameter),
        complete_group_chi_squared=np.full(2, 4.0 - 0.2 * parameter),
        dominance_block_chi_squared=np.full(3, 2.0 - 0.1 * parameter),
        panel_model_contrast_A2=np.full((3, 2), 1.0 + parameter),
        angular_convergence_wrms=0.1,
        effect_uncertainty_chi_squared=0.01,
        comparison_revision="a" * 64,
    )


def test_staged_fit_never_calls_candidate_scan_after_fixed_rejection() -> None:
    scan_calls: list[float] = []

    def exact_scan(parameters: np.ndarray) -> ScanScore:
        scan_calls.append(float(parameters[0]))
        return _scan_score(float(parameters[0]))

    result = run_staged_delayed_acceptance(
        initial_fixed_score=_fixed_score(0.0),
        exact_fixed_evaluator=lambda parameters: _fixed_score(
            float(parameters[0]),
            wrms=1.5,
        ),
        exact_scan_evaluator=exact_scan,
        cheap_proposal=lambda current, radius, replacement: current + 0.1,
        lower_bounds=np.asarray((-1.0,)),
        upper_bounds=np.asarray((1.0,)),
        parameter_scales=np.asarray((1.0,)),
        initial_trust_radius=0.2,
        policy=POLICY,
    )

    assert scan_calls == [0.0]
    assert result.exact_scan_call_count == 1
    assert result.outer_iterations_attempted == 1
    assert result.accepted_outer_iteration_count == 0
    assert result.stop_reason == "replacement_exact_fixed_dataset_gate_failed"


def test_panelwise_scan_oracle_retains_physical_panel_contributions() -> None:
    acquisition = ContinuousIncidenceAcquisition(25.0, 5.0)
    delta = math.radians(0.4315724945)

    def evaluate_nodes(commanded: np.ndarray) -> ScanNodeEvaluation:
        return ScanNodeEvaluation(
            effective_incidence_angle_rad=commanded + delta,
            roi_contrast_A2=np.column_stack((np.full(commanded.size, 2.0), commanded + 1.0)),
            root_topology_code=np.zeros((commanded.size, 2), dtype=np.int64),
            branch_near_fold=np.zeros((commanded.size, 2), dtype=np.bool_),
            signal_window_crossing=np.zeros((commanded.size, 2), dtype=np.bool_),
            background_window_crossing=np.zeros((commanded.size, 2), dtype=np.bool_),
            rapid_roi_change=np.zeros((commanded.size, 2), dtype=np.bool_),
            source_revision="2" * 64,
            evaluation_revision=EVALUATION_REVISION,
        )

    result = evaluate_adaptive_scan_oracle(
        acquisition=acquisition,
        panel_edges_deg=(5.0, 11.0, 18.0, 25.0),
        evaluate_nodes=evaluate_nodes,
        working_covariance=np.eye(2),
        model_scale_count_per_A2=1.0,
        maximum_refinement_depth=2,
    )

    assert result.converged
    assert result.initial_commanded_incidence_angle_rad.size == 48
    assert result.commanded_incidence_angle_rad.size == 96
    assert result.angle_evaluation_count == 144
    assert result.absolute_convergence_wrms < 1.0e-14
    np.testing.assert_allclose(
        result.fine_scan_contrast_A2,
        (2.0, 1.0 + math.radians(15.0)),
        rtol=0.0,
        atol=5.0e-16,
    )
    np.testing.assert_allclose(
        np.sum(result.physical_panel_contrast_A2, axis=0),
        result.fine_scan_contrast_A2,
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_array_equal(np.unique(result.physical_panel_index), np.arange(3))
    assert not result.physical_panel_contrast_A2.flags.writeable
    with pytest.raises(ValueError, match="panel masses"):
        replace(
            result,
            physical_panel_contrast_A2=result.physical_panel_contrast_A2 + 1.0,
        )
    with pytest.raises(ValueError, match="convergence flag"):
        replace(result, converged=False)
    with pytest.raises(ValueError, match="refinement metric"):
        replace(
            result,
            records=tuple(
                replace(record, whitened_difference_wrms=999.0) for record in result.records
            ),
        )
    with pytest.raises(ValueError, match="node masses"):
        replace(
            result,
            records=tuple(
                replace(
                    record,
                    coarse_mass_A2=record.coarse_mass_A2 + 7.0,
                    fine_mass_A2=record.fine_mass_A2 + 7.0,
                )
                for record in result.records
            ),
        )
    with pytest.raises(ValueError, match="node masses"):
        replace(
            result,
            physical_panel_contrast_A2=result.physical_panel_contrast_A2[::-1],
        )
    with pytest.raises(ValueError, match="one common calibration"):
        replace(
            result,
            effective_incidence_angle_rad=(
                result.effective_incidence_angle_rad
                + np.linspace(0.0, 1.0e-6, result.effective_incidence_angle_rad.size)
            ),
        )
    with pytest.raises(ValueError, match="partition the canonical acquisition support"):
        evaluate_adaptive_scan_oracle(
            acquisition=acquisition,
            panel_edges_deg=(6.0, 25.0),
            evaluate_nodes=evaluate_nodes,
            working_covariance=np.eye(2),
            model_scale_count_per_A2=1.0,
        )
    revision_call_count = 0

    def changed_evaluation_revision(commanded: np.ndarray) -> ScanNodeEvaluation:
        nonlocal revision_call_count
        revision_call_count += 1
        return replace(
            evaluate_nodes(commanded),
            evaluation_revision=("e" if revision_call_count == 1 else "f") * 64,
        )

    with pytest.raises(ValueError, match="one evaluation contract"):
        evaluate_adaptive_scan_oracle(
            acquisition=acquisition,
            panel_edges_deg=(5.0, 25.0),
            evaluate_nodes=changed_evaluation_revision,
            working_covariance=np.eye(2),
            model_scale_count_per_A2=1.0,
        )
    with pytest.raises(ValueError, match="must contain integers"):
        replace(
            evaluate_nodes(np.asarray((math.radians(10.0),))),
            root_topology_code=np.zeros((1, 2), dtype=np.float64),
        )


def test_staged_fit_obeys_outer_iteration_and_exact_scan_call_caps() -> None:
    scan_calls: list[float] = []

    def exact_scan(parameters: np.ndarray) -> ScanScore:
        value = float(parameters[0])
        scan_calls.append(value)
        is_replacement = len(scan_calls) % 2 == 1
        return _scan_score(value, accepted=is_replacement)

    def proposal(current: np.ndarray, radius: float, replacement: bool) -> np.ndarray:
        del replacement
        return current + 0.5 * radius

    result = run_staged_delayed_acceptance(
        initial_fixed_score=_fixed_score(0.0),
        exact_fixed_evaluator=lambda parameters: _fixed_score(float(parameters[0])),
        exact_scan_evaluator=exact_scan,
        cheap_proposal=proposal,
        lower_bounds=np.asarray((-1.0,)),
        upper_bounds=np.asarray((1.0,)),
        parameter_scales=np.asarray((1.0,)),
        initial_trust_radius=0.4,
        policy=replace(POLICY, maximum_accepted_outer_iterations=4),
    )

    assert result.accepted_outer_iteration_count == 4
    assert result.outer_iterations_attempted == 4
    assert result.exact_scan_call_count == 9
    assert len(scan_calls) == 9
    np.testing.assert_allclose(result.final_parameters, (0.1875,), rtol=0.0, atol=1.0e-15)
    with pytest.raises(ValueError, match="counters"):
        replace(result, exact_scan_call_count=result.exact_scan_call_count + 1)
    with pytest.raises(ValueError, match="ordered contiguous"):
        replace(
            result,
            records=(
                replace(result.records[0], outer_iteration=1),
                *result.records[1:],
            ),
        )


def test_staged_policy_controls_the_baseline_scan_gate() -> None:
    fixed = _fixed_score(0.0)
    scan = _scan_score(0.0)

    result = run_staged_delayed_acceptance(
        initial_fixed_score=fixed,
        initial_scan_score=scan,
        exact_fixed_evaluator=lambda _parameters: pytest.fail("fixed candidate evaluated"),
        exact_scan_evaluator=lambda _parameters: pytest.fail("scan candidate evaluated"),
        cheap_proposal=lambda *_args: pytest.fail("proposal evaluated"),
        lower_bounds=np.asarray((-1.0,)),
        upper_bounds=np.asarray((1.0,)),
        parameter_scales=np.asarray((1.0,)),
        initial_trust_radius=0.2,
        policy=replace(POLICY, maximum_scan_convergence_wrms=0.05),
    )

    assert result.stop_reason == "baseline_absolute_scan_convergence_failed"
    with pytest.raises(ValueError, match="dominance fraction"):
        replace(POLICY, maximum_dominance_fraction=1.01)


@pytest.mark.parametrize("changed_boundary", ("fixed", "scan"))
def test_staged_fit_rejects_changed_comparison_revisions(changed_boundary: str) -> None:
    def fixed_evaluator(parameters: np.ndarray) -> FixedDatasetScore:
        score = _fixed_score(float(parameters[0]))
        return (
            replace(score, comparison_revision="c" * 64) if changed_boundary == "fixed" else score
        )

    def scan_evaluator(parameters: np.ndarray) -> ScanScore:
        score = _scan_score(float(parameters[0]))
        return replace(score, comparison_revision="c" * 64) if changed_boundary == "scan" else score

    with pytest.raises(ValueError, match="comparison blocks"):
        run_staged_delayed_acceptance(
            initial_fixed_score=_fixed_score(0.0),
            initial_scan_score=_scan_score(0.0),
            exact_fixed_evaluator=fixed_evaluator,
            exact_scan_evaluator=scan_evaluator,
            cheap_proposal=lambda current, _radius, _replacement: current + 0.1,
            lower_bounds=np.asarray((-1.0,)),
            upper_bounds=np.asarray((1.0,)),
            parameter_scales=np.asarray((1.0,)),
            initial_trust_radius=0.2,
            policy=replace(POLICY, maximum_accepted_outer_iterations=1),
        )


def test_staged_policy_controls_complete_group_and_dominance_gates() -> None:
    def run_with(
        candidate_scan: Callable[[float], ScanScore],
        policy: StagedAcceptancePolicy,
    ) -> object:
        return run_staged_delayed_acceptance(
            initial_fixed_score=_fixed_score(0.0),
            initial_scan_score=_scan_score(0.0),
            exact_fixed_evaluator=lambda parameters: _fixed_score(float(parameters[0])),
            exact_scan_evaluator=lambda parameters: candidate_scan(float(parameters[0])),
            cheap_proposal=lambda current, _radius, _replacement: current + 0.1,
            lower_bounds=np.asarray((-1.0,)),
            upper_bounds=np.asarray((1.0,)),
            parameter_scales=np.asarray((1.0,)),
            initial_trust_radius=0.2,
            policy=policy,
        )

    one_iteration = replace(POLICY, maximum_accepted_outer_iterations=1)
    group_candidate = lambda value: replace(  # noqa: E731
        _scan_score(value),
        complete_group_chi_squared=np.full(2, 4.1),
    )
    strict_group = run_with(group_candidate, one_iteration)
    relaxed_group_policy = replace(
        one_iteration,
        complete_group_noninferiority_margin_chi_squared=0.2,
    )
    relaxed_group = run_with(group_candidate, relaxed_group_policy)
    assert strict_group.records[0].reason == "complete_scan_group_noninferiority_failed"
    assert relaxed_group.records[0].accepted
    assert relaxed_group.policy is relaxed_group_policy

    dominance_candidate = lambda value: replace(  # noqa: E731
        _scan_score(value),
        chi_squared=19.0,
        dominance_block_chi_squared=np.asarray((1.4, 2.0, 2.0)),
    )
    strict_dominance = run_with(dominance_candidate, one_iteration)
    relaxed_dominance = run_with(
        dominance_candidate,
        replace(one_iteration, maximum_dominance_fraction=0.7),
    )
    assert strict_dominance.records[0].reason == ("scan_improvement_dominated_by_one_pair_or_band")
    assert relaxed_dominance.records[0].accepted


def test_persistent_scan_event_and_signed_contrast_fail_closed() -> None:
    acquisition = ContinuousIncidenceAcquisition(5.0, 25.0)

    def evaluate_nodes(commanded: np.ndarray) -> ScanNodeEvaluation:
        shape = (commanded.size, 1)
        return ScanNodeEvaluation(
            effective_incidence_angle_rad=commanded,
            roi_contrast_A2=-(1.0 + (commanded[:, None] > math.radians(12.345))),
            root_topology_code=np.zeros(shape, dtype=np.int64),
            branch_near_fold=np.ones(shape, dtype=np.bool_),
            signal_window_crossing=np.zeros(shape, dtype=np.bool_),
            background_window_crossing=np.zeros(shape, dtype=np.bool_),
            rapid_roi_change=np.zeros(shape, dtype=np.bool_),
            source_revision="3" * 64,
            evaluation_revision=EVALUATION_REVISION,
        )

    result = evaluate_adaptive_scan_oracle(
        acquisition=acquisition,
        panel_edges_deg=(5.0, 10.0, 15.0, 20.0, 25.0),
        evaluate_nodes=evaluate_nodes,
        working_covariance=np.eye(1),
        model_scale_count_per_A2=1.0,
        maximum_refinement_depth=2,
    )

    assert not result.converged
    assert result.angle_evaluation_count == 448
    assert np.all(result.roi_contrast_A2 < 0.0)
    assert any(record.event_refinement_required for record in result.records)
    terminal = [
        record for record in result.records if record.accepted or record.refinement_depth + 1 == 2
    ]
    np.testing.assert_allclose(
        result.coarse_scan_contrast_A2,
        np.sum([record.coarse_mass_A2 for record in terminal], axis=0),
        rtol=0.0,
        atol=3.0e-16,
    )
    assert not np.array_equal(
        result.coarse_scan_contrast_A2,
        result.initial_scan_contrast_A2,
    )


def test_coarse_only_topology_event_cannot_pass_at_the_depth_limit() -> None:
    acquisition = ContinuousIncidenceAcquisition(5.0, 25.0)
    call_count = 0

    def evaluate_nodes(commanded: np.ndarray) -> ScanNodeEvaluation:
        nonlocal call_count
        call_count += 1
        shape = (commanded.size, 1)
        coarse_only = np.full(shape, call_count == 1, dtype=np.bool_)
        return ScanNodeEvaluation(
            effective_incidence_angle_rad=commanded,
            roi_contrast_A2=np.ones(shape),
            root_topology_code=np.zeros(shape, dtype=np.int64),
            branch_near_fold=coarse_only,
            signal_window_crossing=np.zeros(shape, dtype=np.bool_),
            background_window_crossing=np.zeros(shape, dtype=np.bool_),
            rapid_roi_change=np.zeros(shape, dtype=np.bool_),
            source_revision="4" * 64,
            evaluation_revision=EVALUATION_REVISION,
        )

    result = evaluate_adaptive_scan_oracle(
        acquisition=acquisition,
        panel_edges_deg=(5.0, 10.0, 15.0, 20.0, 25.0),
        evaluate_nodes=evaluate_nodes,
        working_covariance=np.eye(1),
        model_scale_count_per_A2=1.0,
        maximum_refinement_depth=1,
    )

    assert not result.converged
    assert result.records[0].event_refinement_required


def test_precomputed_baseline_gate_stops_before_every_proposal_callback() -> None:
    fixed = _fixed_score(0.0)
    scan = _scan_score(0.0)

    def forbidden(*_args: object) -> object:
        raise AssertionError("a stopped baseline must not evaluate a proposal")

    result = run_staged_delayed_acceptance(
        initial_fixed_score=fixed,
        initial_scan_score=scan,
        exact_fixed_evaluator=forbidden,
        exact_scan_evaluator=forbidden,
        cheap_proposal=forbidden,
        lower_bounds=np.asarray((-1.0,)),
        upper_bounds=np.asarray((1.0,)),
        parameter_scales=np.asarray((1.0,)),
        initial_trust_radius=0.2,
        policy=POLICY,
        baseline_gate=lambda _fixed, _scan: "finite_region_observation_unstable",
    )

    assert result.stop_reason == "finite_region_observation_unstable"
    assert result.exact_scan_call_count == 1
    assert result.outer_iterations_attempted == 0


def test_zero_profiled_scan_scale_reaches_the_scientific_stop_gate() -> None:
    acquisition = ContinuousIncidenceAcquisition(5.0, 25.0)

    def evaluate_nodes(commanded: np.ndarray) -> ScanNodeEvaluation:
        shape = (commanded.size, 1)
        return ScanNodeEvaluation(
            effective_incidence_angle_rad=commanded,
            roi_contrast_A2=np.ones(shape),
            root_topology_code=np.zeros(shape, dtype=np.int64),
            branch_near_fold=np.zeros(shape, dtype=np.bool_),
            signal_window_crossing=np.zeros(shape, dtype=np.bool_),
            background_window_crossing=np.zeros(shape, dtype=np.bool_),
            rapid_roi_change=np.zeros(shape, dtype=np.bool_),
            source_revision="5" * 64,
            evaluation_revision=EVALUATION_REVISION,
        )

    with pytest.raises(ValueError, match="finite positive scale"):
        evaluate_adaptive_scan_oracle(
            acquisition=acquisition,
            panel_edges_deg=(5.0, 10.0, 15.0, 20.0, 25.0),
            evaluate_nodes=evaluate_nodes,
            working_covariance=np.eye(1),
            profile_scale=lambda _model: 0.0,
            maximum_refinement_depth=1,
        )
    scan = replace(_scan_score(0.0), profiled_scale=0.0)
    result = run_staged_delayed_acceptance(
        initial_fixed_score=_fixed_score(0.0),
        initial_scan_score=scan,
        exact_fixed_evaluator=lambda _parameters: pytest.fail("fixed candidate evaluated"),
        exact_scan_evaluator=lambda _parameters: pytest.fail("scan candidate evaluated"),
        cheap_proposal=lambda *_args: pytest.fail("proposal evaluated"),
        lower_bounds=np.asarray((-1.0,)),
        upper_bounds=np.asarray((1.0,)),
        parameter_scales=np.asarray((1.0,)),
        initial_trust_radius=0.2,
        policy=POLICY,
    )

    assert result.stop_reason == "baseline_scan_profiled_scale_nonphysical"
