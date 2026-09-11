"""Run an explicitly configured native Bi fit and retain one external diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from painted_ewald import MosaicParameters
from rasim_next.fitting.bi_joint import (
    BI_JOINT_PARAMETER_NAMES,
    BiJointCandidate,
    BiNativeFitEvaluator,
    bi_joint_sensitivity,
    fit_bi_joint,
)
from rasim_next.fitting.bi_native import BiNativeStructureModel
from rasim_next.fitting.native_input import load_native_fit_physics
from rasim_next.fitting.native_observations import load_native_fit_observations
from rasim_next.proof.diagnostics import write_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if (
        plan.get("schema") != "rasim-bi-native-fit-plan-v1"
        or tuple(plan["parameter_names"]) != BI_JOINT_PARAMETER_NAMES
    ):
        raise ValueError("plan must declare the complete ordered Bi joint parameter contract")
    original = load_native_fit_physics(args.physics)
    observation_record = json.loads(args.observations.read_bytes())
    if observation_record["physical_input"]["sha256"] != original.input_revision:
        raise ValueError("observation bundle and supplied physical input are not bound together")
    physics = replace(
        original,
        integration_rule=replace(original.integration_rule, **plan["integration_override"]),
    )
    observations = load_native_fit_observations(args.observations)
    candidate = BiJointCandidate(plan["initial"], plan["coherent_repeats"])
    evaluator = BiNativeFitEvaluator(
        BiNativeStructureModel(physics),
        observations,
        MosaicParameters(*plan["proposal_mosaic"]),
        plan["workers"],
    )
    lower, upper = np.array(plan["lower"]), np.array(plan["upper"])
    coverage = evaluator.model.validate_rod_coverage(
        a_bounds_A=(lower[0], upper[0]), c_bounds_A=(lower[1], upper[1])
    )
    start = perf_counter()
    history, predictions, scales, losses, guards, stages = [], [], [], [], [], []
    repeat_history = []
    extra = {}
    manifest = dict(
        schema="rasim-bi-native-joint-fit-v1",
        plan=plan,
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        physics_input_revision=original.input_revision,
        observation_input_revision=observations.input_revision,
        projection_revision=observations.projection.projection_revision,
        integration_rule=asdict(physics.integration_rule),
        rod_coverage=coverage,
        numerical_status="not_qualified",
        acceptance="candidate_only",
        objective="GLS of net counts with full count plus background covariance; one profiled scale",
        limitations=[
            "One acquisition; source and mounting fixed.",
            "Bounds are search ranges, not uncertainty intervals.",
            "Optimizer completion and local sensitivity do not prove identifiability.",
            "The m0 channel is the named empirical Parratt/kinematic composite.",
        ],
    )

    def save():
        manifest.update(
            elapsed_seconds=perf_counter() - start,
            response_compilations=evaluator.compile_count,
            compile_seconds=evaluator.compile_seconds,
            stages=stages,
        )
        write_diagnostic(
            args.output,
            repository_root=Path(__file__).resolve().parents[1],
            arrays=dict(
                parameters=np.asarray(history),
                coherent_repeats=np.asarray(repeat_history, dtype=np.int64),
                prediction_count=np.asarray(predictions),
                scale=np.asarray(scales),
                gls_chi_square=np.asarray(losses),
                guard_scores=np.asarray(guards),
                guard_limit=observations.guard_limit,
                **extra,
            ),
            manifest=manifest,
        )

    def checkpoint(trial, scale, residual, prediction, scores):
        history.append(trial.values)
        repeat_history.append(trial.coherent_repeats)
        predictions.append(prediction)
        scales.append(scale)
        losses.append(float(residual @ residual))
        guards.append(scores["guard_scores"])
        save()
        print(
            json.dumps(
                dict(
                    evaluation=len(history),
                    chi_square=losses[-1],
                    guard_max_ratio=float(np.max(guards[-1] / observations.guard_limit)),
                    compiles=evaluator.compile_count,
                    elapsed_seconds=round(perf_counter() - start, 2),
                )
            ),
            flush=True,
        )

    if not plan["stages"]:
        raw = evaluator.predict(candidate)
        scale, residual = observations.profile_scale(raw)
        checkpoint(candidate, scale, residual, scale * raw, observations.scores(scale * raw))
    for stage in plan["stages"]:
        result = fit_bi_joint(
            evaluator,
            candidate,
            lower=lower,
            upper=upper,
            active_indices=tuple(stage["active_indices"]),
            maximum_iterations=stage["maximum_iterations"],
            finite_difference_step=plan["finite_difference_step"],
            enforce_historical_guards=stage["enforce_historical_guards"],
            callback=checkpoint,
        )
        candidate = result.candidate
        stages.append(
            dict(
                name=stage["name"],
                optimizer_success=bool(result.success),
                message=str(result.message),
                iterations=int(result.nit),
                active_parameters=result.active_parameters,
                gls_chi_square=result.gls_chi_square,
                guards_pass=result.scores["guards_pass"],
                scale=result.scale,
            )
        )
    # Refine every declared integer independently; compare the same frozen
    # observable at the same integration rule, after continuous nuisance fitting.
    guard_mode = bool(plan["stages"] and plan["stages"][-1]["enforce_historical_guards"])
    if plan.get("repeat_refits"):
        guard_mode = plan["repeat_refit_enforce_guards"]
        raw = evaluator.predict(candidate)
        scale, _ = observations.profile_scale(raw, enforce_guards=guard_mode)
        best_scores = observations.scores(scale * raw)
        best_key = (not best_scores["guards_pass"], best_scores["gls_chi_square"])
        seed = candidate
        for specification in plan["repeat_refits"]:
            repeats = specification["coherent_repeats"]
            values = seed.values.copy()
            values[18] = np.clip(seed.film_thickness_A - repeats * values[1], lower[18], upper[18])
            result = fit_bi_joint(
                evaluator,
                BiJointCandidate(values, repeats),
                lower=lower,
                upper=upper,
                maximum_iterations=specification["maximum_iterations"],
                finite_difference_step=plan["finite_difference_step"],
                enforce_historical_guards=guard_mode,
                callback=checkpoint,
            )
            stages.append(
                dict(
                    name=f"integer_repeat_refit_{repeats}",
                    coherent_repeats=repeats,
                    optimizer_success=bool(result.success),
                    message=str(result.message),
                    iterations=int(result.nit),
                    active_parameters=result.active_parameters,
                    gls_chi_square=result.gls_chi_square,
                    guards_pass=result.scores["guards_pass"],
                    scale=result.scale,
                    parameters=result.candidate.values.tolist(),
                )
            )
            key = (not result.scores["guards_pass"], result.gls_chi_square)
            if key < best_key:
                best_key, candidate = key, result.candidate
        manifest["repeat_selection"] = (
            "Prefer guard-feasible candidates, then minimum GLS; qualification separate"
        )
    manifest["selected_parameters"] = candidate.values.tolist()
    manifest["selected_coherent_repeats"] = candidate.coherent_repeats
    manifest["selected_film_thickness_A"] = candidate.film_thickness_A
    manifest["selected_surface_fractions"] = candidate.surface_fractions
    manifest["stitch_state"] = evaluator.stitch_state(candidate)
    raw = evaluator.predict(candidate)
    scale, _ = observations.profile_scale(raw, enforce_guards=guard_mode)
    extra["selected_prediction_count"] = scale * raw
    manifest["selected_scale"] = scale
    selected_scores = observations.scores(scale * raw)
    manifest["selected_scores"] = {
        key: value for key, value in selected_scores.items() if key != "guard_scores"
    }
    if plan["sensitivity"]:
        sensitivity_stitches = []

        def record_stitch(index, trial, state):
            sensitivity_stitches.append(dict(parameter_index=index, state=state))

        jacobian = bi_joint_sensitivity(
            evaluator,
            candidate,
            lower=lower,
            upper=upper,
            step=plan["finite_difference_step"],
            callback=record_stitch,
        )
        baseline_branches = [
            (s["selection"], s["bounds_q_over_qc"]) for s in sensitivity_stitches[0]["state"]
        ]
        manifest["sensitivity_stitch_states"] = sensitivity_stitches
        manifest["sensitivity_branch_changes"] = [
            entry["parameter_index"]
            for entry in sensitivity_stitches[1:]
            if [(s["selection"], s["bounds_q_over_qc"]) for s in entry["state"]]
            != baseline_branches
        ]
        _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
        extra.update(
            whitened_jacobian_per_range=jacobian,
            sensitivity_singular_values=singular,
            sensitivity_right_vectors=right,
        )
    # Integer repeats are evaluated separately, never rounded inside the continuous fit.
    for repeats in plan["repeat_checks"]:
        trial = BiJointCandidate(candidate.values, repeats)
        prediction = evaluator.predict(trial)
        scale, residual = observations.profile_scale(prediction)
        extra[f"repeat_{repeats}_prediction_count"] = scale * prediction
        stages.append(
            dict(
                name=f"integer_repeat_check_{repeats}",
                coherent_repeats=repeats,
                film_thickness_A=trial.film_thickness_A,
                gls_chi_square=float(residual @ residual),
                interpretation="conditional comparison; other parameters held fixed",
            )
        )
    save()
    print(json.dumps(dict(output=str(args.output), stages=stages)), flush=True)


if __name__ == "__main__":
    main()
