"""Observable-level numerical disagreements and physical sensitivity diagnostics."""

import numpy as np


def compare_native_predictions(
    observations,
    reference,
    refined,
    *,
    fixed_scale,
    maximum_whitened_rms,
    maximum_contrast_rms,
    maximum_objective_contrast_error,
    require_constrained_objective_agreement=False,
):
    """Compare complete fixed native support at baseline and candidate perturbations.

    Each matrix is (candidate, observation), baseline first. One fixed physical
    scale isolates prediction disagreement; independently profiled scales assess
    objective contrasts and guard decisions. Passing these finite comparisons is
    empirical numerical agreement, never a certified integration-error bound.
    """
    reference, refined = np.asarray(reference), np.asarray(refined)
    guard_diagnostics = observations.allow_guard_constraints
    if require_constrained_objective_agreement and not guard_diagnostics:
        raise ValueError("historical guard objectives cannot qualify training or synthetic fits")
    thresholds = np.array(
        [maximum_whitened_rms, maximum_contrast_rms, maximum_objective_contrast_error]
    )
    if (
        reference.ndim != 2
        or refined.shape != reference.shape
        or len(reference) < 2
        or reference.shape[1:] != observations.net_count.shape
        or np.iscomplexobj(reference)
        or np.iscomplexobj(refined)
        or np.any(~np.isfinite(reference))
        or np.any(~np.isfinite(refined))
        or not np.isfinite(fixed_scale)
        or fixed_scale <= 0
        or np.any(~np.isfinite(thresholds))
        or np.any(thresholds <= 0)
    ):
        raise ValueError(
            "accuracy comparison requires aligned baseline/perturbations, scale and tolerances"
        )
    delta = np.array(
        [
            observations.whiten(fixed_scale * (high - low))
            for low, high in zip(reference, refined, strict=True)
        ]
    )
    rms = np.sqrt(np.mean(delta**2, axis=1))
    contrast_rms = np.sqrt(np.mean((delta[1:] - delta[0]) ** 2, axis=1))
    objectives, guards, feasible, constrained_scores, constrained_objectives = [], [], [], [], []
    for predictions in (reference, refined):
        scores = []
        intervals, constrained, constrained_objective = [], [], []
        for raw in predictions:
            scale, _ = observations.profile_scale(raw)
            scores.append(observations.scores(scale * raw))
            if guard_diagnostics:
                intervals.append(observations.guard_scale_interval(raw) is not None)
                guarded_scale, _ = observations.profile_scale(raw, enforce_guards=True)
                guarded = observations.scores(guarded_scale * raw)
                constrained.append(guarded["guard_scores"])
                constrained_objective.append(guarded["gls_chi_square"])
        objectives.append(np.array([s["gls_chi_square"] for s in scores]))
        guards.append(
            np.array([s["guard_scores"] <= observations.guard_limit + 1e-6 for s in scores])
            if guard_diagnostics
            else np.empty((len(scores), 0), dtype=bool)
        )
        feasible.append(np.array(intervals))
        constrained_scores.append(np.array(constrained))
        constrained_objectives.append(np.array(constrained_objective))
    objective_error = (objectives[1][1:] - objectives[1][0]) - (
        objectives[0][1:] - objectives[0][0]
    )
    agreement = (
        np.max(rms) <= maximum_whitened_rms
        and np.max(contrast_rms) <= maximum_contrast_rms
        and np.max(abs(objective_error)) <= maximum_objective_contrast_error
        and np.array_equal(guards[0], guards[1])
        and np.array_equal(feasible[0], feasible[1])
    )
    constrained_error = (
        (constrained_objectives[1][1:] - constrained_objectives[1][0])
        - (constrained_objectives[0][1:] - constrained_objectives[0][0])
        if guard_diagnostics
        else np.empty(0)
    )
    if require_constrained_objective_agreement:
        agreement = (
            agreement
            and bool(np.all(feasible))
            and np.max(abs(constrained_error)) <= maximum_objective_contrast_error
        )
    return dict(
        whitened_rms=rms,
        contrast_whitened_rms=contrast_rms,
        objective_contrast_error=objective_error,
        constrained_objective_contrast_error=constrained_error,
        guard_decision_changed=guards[0] != guards[1],
        guard_scale_feasibility_changed=feasible[0] != feasible[1],
        reference_constrained_guard_scores=constrained_scores[0],
        refined_constrained_guard_scores=constrained_scores[1],
        reference_objective=objectives[0],
        refined_objective=objectives[1],
        reference_constrained_objective=constrained_objectives[0],
        refined_constrained_objective=constrained_objectives[1],
        empirical_agreement=bool(agreement),
        objective_observation_revision=observations.input_revision,
        guard_diagnostics_included=guard_diagnostics,
        error_bound_status="not_certified",
    )


def native_sensitivity(predict, observations, parameters, values, *, relative_step=1e-3):
    """Whitened scale-profiled Jacobian per declared physical sensitivity scale.

    A local SVD reveals weak combinations. It is neither a nuisance-refitted
    profile nor a confidence interval, and must itself pass numerical comparison.
    """
    values = np.asarray(values)
    if (
        values.shape != (len(parameters),)
        or np.iscomplexobj(values)
        or np.any(~np.isfinite(values))
        or not 0 < relative_step < 1
    ):
        raise ValueError("sensitivity requires an aligned candidate and positive fractional step")
    values = np.asarray(values, dtype=float)
    _, baseline = observations.profile_scale(predict(values))
    jacobian = np.empty((len(baseline), len(parameters)))
    probes = [values.copy()]
    for i, parameter in enumerate(parameters):
        if not parameter.lower <= values[i] <= parameter.upper:
            raise ValueError("sensitivity candidate lies outside declared ranges")
        step = relative_step * parameter.sensitivity_scale
        forward, backward = parameter.upper - values[i], values[i] - parameter.lower
        delta = min(step, forward) if forward >= min(step, backward) else -min(step, backward)
        trial = values.copy()
        trial[i] += delta
        probes.append(trial)
        _, residual = observations.profile_scale(predict(trial))
        jacobian[:, i] = (residual - baseline) * parameter.sensitivity_scale / delta
    _, singular_values, directions = np.linalg.svd(jacobian, full_matrices=False)
    return dict(
        jacobian=jacobian,
        singular_values=singular_values,
        right_singular_vectors=directions,
        probe_values=np.array(probes),
        interpretation="local sensitivity only; nuisance profiles and numerical checks required",
    )
