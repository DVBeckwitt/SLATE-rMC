"""Run a declared Bi/Pb native refinement experiment and retain one external diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from rasim_next.fitting.native_acceleration import reference_corrected_start
from rasim_next.fitting.native_accuracy import (
    compare_conditional_predictions,
    compare_native_predictions,
    native_sensitivity,
)
from rasim_next.fitting.native_execution import NativePredictionStore
from rasim_next.fitting.native_input import load_native_fit_physics
from rasim_next.fitting.native_observations import (
    load_native_background_controls,
    load_native_fit_observations,
)
from rasim_next.fitting.native_search import (
    FitParameter,
    GaussianCalibration,
    conditional_validation,
    fit_native_parameters,
    native_fit_candidate,
    profile_native_parameter,
    training_observations,
    validate_native_search_request,
)
from rasim_next.fitting.native_structure import validate_native_rod_coverage
from rasim_next.fitting.native_workflow import (
    make_native_evaluator,
    native_physics_with,
    native_prediction_group,
)
from rasim_next.io.diagnostics import validate_diagnostic_destination, write_diagnostic
from rasim_next.pipeline.detector_revisions import _instrument_revision


def _point_record(point):
    if point is None:
        return None
    return dict(
        parameters=point.parameter_values.tolist(),
        scale=point.scale,
        data_chi_square=point.data_chi_square,
        data_objective=point.data_objective,
        objective_kind=point.objective_kind,
        calibration_chi_square=point.calibration_chi_square,
        assumption_chi_square=point.assumption_chi_square,
        objective=point.objective,
        guards_pass=point.scores["guards_pass"],
        optimizer_converged=point.optimizer_converged,
        optimizer_message=point.get("optimizer_message"),
        method=point.get("method"),
        function_evaluations=point.get("function_evaluations"),
        jacobian_evaluations=point.get("jacobian_evaluations"),
        optimality=point.get("optimality"),
        physical_boundaries=point.get("physical_boundary_parameters"),
        search_limits=point.get("search_bound_parameters"),
    )


def _implementation_record():
    root = Path(__file__).resolve().parents[1]
    return dict(
        source_hash_scope="startup filesystem snapshot; not loaded-bytecode attestation",
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        git_worktree_status=subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ),
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src").rglob("*.py"))
        },
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "numba", "gemmi", "xraydb")
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse exact completed raw predictions; restart public optimizer",
    )
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if plan["schema"] != "rasim-native-refinement-plan-v1":
        raise ValueError("unsupported native refinement plan schema")
    require_qualification = plan.get("require_initial_qualification", True)
    if type(require_qualification) is not bool:
        raise ValueError("require_initial_qualification must be boolean")
    original = load_native_fit_physics(args.physics)
    observations = replace(
        load_native_fit_observations(args.observations),
        objective_kind=plan.get("objective", "gls"),
    )
    if observations.objective_kind == "historical" and "synthetic" in plan:
        raise ValueError("historical objective requires original frozen observations")
    observation_record = json.loads(args.observations.read_bytes())
    if observation_record["physical_input"]["sha256"] != original.input_revision:
        raise ValueError("observations and physical input are not bound together")
    if plan["acquisition_id"] != observation_record["raw_acquisition"]["sha256"]:
        raise ValueError("acquisition ownership must equal the frozen raw acquisition SHA256")
    parameters = tuple(FitParameter(**p) for p in plan["parameters"])
    calibration = tuple(GaussianCalibration(**b) for b in plan.get("calibration", ()))
    names = tuple(p.name for p in parameters)
    for block in calibration:
        if block.acquisition_id != plan["acquisition_id"]:
            raise ValueError("calibration must explicitly bind this target acquisition")
    binding = plan.get("experiment_binding")
    if binding is not None:
        descriptor = json.loads(args.observations.read_bytes())
        source_revision = descriptor.get("preparation", {}).get(
            "source_observation_sha256", observations.input_revision
        )
        if binding != {
            "physics_sha256": original.input_revision,
            "observation_sha256": source_revision,
        }:
            raise ValueError(
                "plan is bound to a different physical experiment or observation roster"
            )
    training = plan.get("training_indices")
    validation_masks = {}
    if training is not None:
        indices = np.asarray(training)
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or len(indices) == 0
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= len(observations.valid))
        ):
            raise ValueError("training indices must be distinct in-range nonnegative integers")
        mask = np.zeros_like(observations.valid)
        mask[indices] = True
        training_observations(observations, mask)
        if not np.any(observations.valid & ~mask) or not plan.get("validation_groups"):
            raise ValueError(
                "prospective validation requires nonempty held-out rows and declared groups"
            )
        coverage = np.zeros_like(mask, dtype=int)
        for group in plan["validation_groups"]:
            indices = np.asarray(group["indices"])
            if (
                not isinstance(group["name"], str)
                or not group["name"]
                or group["name"] in validation_masks
                or indices.ndim != 1
                or indices.dtype.kind not in "iu"
                or not len(indices)
                or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0)
                or np.any(indices >= len(mask))
            ):
                raise ValueError("validation groups require unique names and explicit row indices")
            group_mask = np.zeros_like(mask)
            group_mask[indices] = True
            validation_masks[group["name"]] = group_mask
            coverage += group_mask
        if not np.array_equal(coverage, (observations.valid & ~mask).astype(int)):
            raise ValueError("validation groups must partition the declared held-out valid rows")
    starts = np.asarray(plan["starts"], dtype=float)
    repeats = tuple(plan["repeat_choices"])
    if (
        not repeats
        or any(type(n) is not int or n < 1 for n in repeats)
        or len(set(repeats)) != len(repeats)
    ):
        raise ValueError("repeat choices must be distinct positive integers")

    validate_native_search_request(
        observations,
        parameters,
        starts,
        calibration=calibration,
        method="trf",
        maximum_iterations=1,
        maximum_function_evaluations=1,
        finite_difference_step=plan["finite_difference_step"],
    )
    stages = plan.get("stages", [])
    fixed_parameters = plan.get("fixed_parameters", {})
    if not isinstance(fixed_parameters, dict) or set(fixed_parameters) - set(names):
        raise ValueError("fixed parameters must explicitly name declared coordinates")
    for name, value in fixed_parameters.items():
        index = names.index(name)
        parameter = parameters[index]
        if (
            type(value) not in (int, float)
            or not np.isfinite(value)
            or not parameter.lower <= value <= parameter.upper
            or np.any(starts[:, index] != value)
        ):
            raise ValueError(
                "fixed values must be finite, within bounds and identical in all starts"
            )
    if fixed_parameters and (plan.get("sensitivity") or plan.get("profiles")):
        raise ValueError(
            "fixed-parameter controls require identification in the released full model"
        )
    free_names = tuple(name for name in names if name not in fixed_parameters)
    if stages and (not free_names or tuple(stages[-1]["active_parameters"]) != free_names):
        raise ValueError("the final fit stage must release every admitted continuous coordinate")
    stage_names = [stage["name"] for stage in stages]
    if any(not isinstance(name, str) or not name for name in stage_names) or len(
        set(stage_names)
    ) != len(stage_names):
        raise ValueError("fit stages require distinct nonempty names")
    search_observations = (
        training_observations(observations, mask) if training is not None else observations
    )
    for stage in stages:
        if type(stage["enforce_historical_guards"]) is not bool:
            raise ValueError("stage historical guard policy must be boolean")
        active = tuple(stage["active_parameters"])
        if not active or len(set(active)) != len(active) or set(active) - set(names):
            raise ValueError("fit stages require distinct declared active parameters")
        if set(stage["active_parameters"]) & set(fixed_parameters):
            raise ValueError("a fit stage cannot release a declared fixed control parameter")
        if (training is not None or "synthetic" in plan) and stage["enforce_historical_guards"]:
            raise ValueError(
                "historical guards cannot constrain synthetic or prospective training fits"
            )
        fixed = {name: float(starts[0, i]) for i, name in enumerate(names) if name not in active}
        method = stage.get(
            "method", plan.get("method", "slsqp" if stage["enforce_historical_guards"] else "trf")
        )
        budget = stage.get(
            "maximum_function_evaluations", plan.get("maximum_function_evaluations", 80)
        )
        validate_native_search_request(
            search_observations,
            parameters,
            starts,
            fixed_values=fixed,
            calibration=calibration,
            method=method,
            maximum_iterations=stage["maximum_iterations"],
            maximum_function_evaluations=budget,
            finite_difference_step=plan["finite_difference_step"],
            enforce_historical_guards=stage["enforce_historical_guards"],
            batched=True,
        )
        correction = stage.get("reference_correction")
        if correction is not None:
            native_physics_with(original, plan, correction["numerical_override"])
            radius = np.asarray(correction["trust_radii"])
            if (
                radius.shape != (len(parameters),)
                or np.iscomplexobj(radius)
                or np.any(~np.isfinite(radius))
                or np.any(radius <= 0)
                or type(correction.get("maximum_updates", 2)) is not int
                or correction.get("maximum_updates", 2) < 1
            ):
                raise ValueError(
                    "reference correction requires positive finite trust radii and updates"
                )
            validate_native_search_request(
                search_observations,
                parameters,
                starts,
                fixed_values=fixed,
                calibration=calibration,
                method=method,
                maximum_iterations=stage["maximum_iterations"],
                maximum_function_evaluations=correction.get("maximum_function_evaluations", 30),
                finite_difference_step=plan["finite_difference_step"],
                enforce_historical_guards=stage["enforce_historical_guards"],
            )
    for profile in plan.get("profiles", ()):
        if profile["name"] not in names:
            raise ValueError("profile names an undeclared parameter")
        grid = np.asarray(profile["grid"])
        profiled_parameter = parameters[names.index(profile["name"])]
        if (
            grid.ndim != 1
            or not len(grid)
            or np.iscomplexobj(grid)
            or np.any(~np.isfinite(grid))
            or len(np.unique(grid)) != len(grid)
            or np.any(grid < profiled_parameter.lower)
            or np.any(grid > profiled_parameter.upper)
        ):
            raise ValueError("profile grid must contain distinct finite in-range values")
        validate_native_search_request(
            search_observations,
            parameters,
            starts,
            calibration=calibration,
            method=profile.get("method", plan.get("method", "trf")),
            maximum_iterations=profile["maximum_iterations"],
            maximum_function_evaluations=profile.get(
                "maximum_function_evaluations", plan.get("maximum_function_evaluations", 80)
            ),
            finite_difference_step=plan["finite_difference_step"],
            batched=True,
        )
    validate_diagnostic_destination(
        args.output, repository_root=Path(__file__).resolve().parents[1]
    )
    if args.resume and not args.output.is_file():
        raise FileNotFoundError(args.output)
    if not args.resume and args.output.exists():
        raise ValueError("output already exists; use --resume or a new output path")
    prediction_workers = plan.get("prediction_workers", 1)
    group_size = plan.get("prediction_group_size", 16)
    if any(type(n) is not int or n < 1 for n in (prediction_workers, group_size)):
        raise ValueError("prediction worker and group counts must be positive integers")
    physics = native_physics_with(original, plan, {})
    evaluator = make_native_evaluator(physics, observations, plan)
    # Validate every supplied start and the complete physical candidate roster.
    for start in starts:
        for n in repeats:
            evaluator.bind(start, n)
    coverage = validate_native_rod_coverage(
        physics,
        a_bounds_A=(parameters[0].lower, parameters[0].upper),
        c_bounds_A=(parameters[1].lower, parameters[1].upper),
    )
    if plan["fit_instrument"]:
        spectral = starts[0].copy()
        spectral[-3] = parameters[-3].upper
        spectral[-2:] = [p.lower for p in parameters[-2:]]
        broadest, _, _, _ = evaluator.bind(spectral, repeats[0])
        coverage = validate_native_rod_coverage(
            broadest,
            a_bounds_A=(parameters[0].lower, parameters[0].upper),
            c_bounds_A=(parameters[1].lower, parameters[1].upper),
        )
    start_time = perf_counter()
    arrays, history = {}, []
    manifest = dict(
        schema="rasim-native-refinement-result-v2",
        plan=plan,
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        physics_input_revision=original.input_revision,
        observation_input_revision=observations.input_revision,
        projection_revision=observations.projection.projection_revision,
        rod_coverage=coverage,
        numerical_status="not_qualified",
        identification_status="not_profiled",
        acceptance="candidate_only",
        fit_scope="fixed_parameter_control" if fixed_parameters else "all_admitted_coordinates",
        fixed_parameters=fixed_parameters,
        implementation=_implementation_record(),
        model="finite detector-native Bi/Pb with composition-derived optics",
        numerical_checks=[],
        fits=[],
        profiles=[],
        limitations=[
            "Each acquisition has its own scale and source/instrument ownership.",
            "Historical guards are compatibility checks, not independent measurements.",
            "Prospective splits reuse an explored acquisition; they are not untouched experiments.",
            "No calibrated detector PSF, strain ensemble or finite footprint is inferred by this model.",
        ],
    )

    identity = hashlib.sha256(
        json.dumps(
            dict(
                plan=manifest["plan_sha256"],
                physics=original.input_revision,
                observations=observations.input_revision,
                implementation=manifest["implementation"]["source_sha256"],
                runner=manifest["implementation"]["runner_sha256"],
                dependencies=manifest["implementation"]["dependencies"],
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    predictions = NativePredictionStore(identity, len(parameters), len(observations.net_count))
    if args.resume:
        with np.load(io.BytesIO(args.output.read_bytes()), allow_pickle=False) as previous:
            previous_manifest = json.loads(previous["manifest_json"].tobytes())
            predictions.restore(previous, previous_manifest["prediction_store"])
            manifest["previous_execution"] = {
                key: previous_manifest.get(key)
                for key in (
                    "execution_status",
                    "fits",
                    "profiles",
                    "optimizer_candidate",
                    "selected",
                    "numerical_status",
                )
            }
        manifest["resumed_from_completed_predictions"] = len(predictions.raw)
    elif args.output.exists():
        raise ValueError("output already exists; use --resume or a new output path")

    def save():
        manifest.update(
            elapsed_seconds=perf_counter() - start_time,
            compile_metrics_scope="parent evaluator only; worker-group builds excluded",
            compile_count=evaluator.compile_count,
            compile_seconds=evaluator.compile_seconds,
        )
        arrays["evaluation_history"] = np.asarray(history)
        arrays.update(predictions.arrays())
        manifest["prediction_store"] = predictions.state()
        cache = evaluator.scattering_cache
        manifest["scattering_cache"] = dict(
            build_count=cache.build_count,
            reuse_count=cache.reuse_count,
            build_seconds=cache.build_seconds,
            retained_bytes=cache.retained_bytes,
            peak_retained_bytes=cache.peak_retained_bytes,
        )
        write_diagnostic(
            args.output,
            arrays=arrays,
            manifest=manifest,
            repository_root=Path(__file__).resolve().parents[1],
        )

    def execute_batch(values, n):
        if prediction_workers == 1:
            for index, row in enumerate(values):
                yield index, evaluator.predict(row, n)
            return
        # Group exact response dependencies together; each process owns its evaluator.
        excluded = {
            "source_line_0_probability",
            "detector_center_column_delta_px",
            "detector_center_row_delta_px",
            "detector_normal_distance_delta_m",
            "detector_local_x_tilt_rad",
            "detector_local_y_tilt_rad",
            "sample_normal_offset_m",
            "source_spatial_sigma_0_m",
            "source_spatial_sigma_1_m",
            "source_position_divergence_correlation_0",
            "source_position_divergence_correlation_1",
        }
        instrument_names = tuple(
            p.name
            for p in parameters
            if p.owner == plan["acquisition_id"] and p.name not in excluded
        )
        dependent = [
            i
            for i, p in enumerate(parameters)
            if p.name in instrument_names or p.name in ("a_A", "c_A") or "occupancy" in p.name
        ]
        groups = {}
        for i, row in enumerate(values):
            groups.setdefault(row[dependent].tobytes(), []).append(i)
        chunks = [
            indices[first : first + group_size]
            for indices in groups.values()
            for first in range(0, len(indices), group_size)
        ]
        futures = {
            executor.submit(
                native_prediction_group, physics, observations, plan, values[indices], n
            ): indices
            for indices in chunks
        }
        for future in as_completed(futures):
            for index, row in zip(futures[future], future.result(), strict=True):
                yield index, row

    def predict_many(values, n):
        return predictions.evaluate(values, n, execute_batch, checkpoint=save)

    def predict(values, n):
        return predict_many(np.asarray(values)[None, :], n)[0]

    def previous_predictions(n):
        saved = predictions.arrays()
        selected = saved["prediction_repeats"] == n
        return saved["prediction_values"][selected], saved["prediction_raw"][selected]

    def checkpoint(point):
        history.append(np.r_[point.parameter_values, point.scale, point.objective])
        arrays["latest_prediction_count"] = point.prediction_count
        print(
            json.dumps(
                dict(
                    evaluation=len(history),
                    objective=point.objective,
                    guards_pass=point.scores["guards_pass"],
                    elapsed_seconds=round(perf_counter() - start_time, 2),
                )
            ),
            flush=True,
        )

    executor = (
        ProcessPoolExecutor(max_workers=prediction_workers) if prediction_workers > 1 else None
    )
    try:
        baseline = predict(starts[0], repeats[0])
        bound, baseline_arguments, _, _ = evaluator.bind(starts[0], repeats[0])
        manifest["baseline_physics"] = dict(
            input_revision=bound.input_revision,
            source_revision=bound.source.revision,
            material_revision=bound.material.material_revision,
            material_provenance=bound.material.provenance,
            instrument_revision=_instrument_revision(
                replace(bound.instrument, film_thickness_A=baseline_arguments["film_thickness_A"])
            ),
        )
        baseline_scale, _ = observations.profile_scale(baseline)
        arrays["baseline_raw"] = baseline
        arrays["baseline_prediction_count"] = baseline_scale * baseline
        manifest["baseline_scale"] = baseline_scale
        manifest["baseline_inactive_parameters"] = evaluator.inactive_parameters(
            starts[0], repeats[0]
        )
        manifest["baseline_stitch"] = evaluator.stitch_state(starts[0], repeats[0])
        save()
        checks = plan.get("numerical_checks", [])

        center_errors = {}

        def run_checks(label, candidates, n, fixed_scale, objective_observations, guarded=False):
            candidates = np.asarray(candidates, dtype=float)
            for name, value in fixed_parameters.items():
                if np.any(candidates[:, names.index(name)] != value):
                    raise ValueError(
                        "qualification cannot change a declared fixed control parameter"
                    )
            agreement = True
            arrays[f"{label}_qualification_candidates"] = candidates
            reference = np.array([predict(v, n) for v in candidates])
            arrays[f"{label}_qualification_reference"] = reference
            reference_stitches = [evaluator.stitch_state(v, n) for v in candidates]
            for i, specification in enumerate(checks):
                print(
                    json.dumps(dict(numerical_check=specification["name"], state="running")),
                    flush=True,
                )
                refined_physics = native_physics_with(original, plan, specification)
                if (
                    refined_physics.integration_rule == physics.integration_rule
                    and refined_physics.spatial_quadrature_order == physics.spatial_quadrature_order
                    and refined_physics.source.revision == physics.source.revision
                    and refined_physics.source_definition.local_m0_divergence_order
                    == physics.source_definition.local_m0_divergence_order
                ):
                    raise ValueError("a numerical check must change an effective integration rule")
                refined_evaluator = make_native_evaluator(refined_physics, observations, plan)
                effective_rule = refined_physics.integration_rule
                if physics.specular_stitch_stack is None:
                    effective_rule = replace(
                        effective_rule, stitch_grid_size=physics.integration_rule.stitch_grid_size
                    )
                effective_change = (
                    effective_rule != physics.integration_rule
                    or refined_physics.spatial_quadrature_order != physics.spatial_quadrature_order
                )
                for candidate in candidates:
                    trial_physics, _, _, _ = refined_evaluator.bind(candidate, n)
                    low_physics, _, _, _ = evaluator.bind(candidate, n)
                    high_parts, low_parts = (
                        trial_physics.integration_parts(),
                        low_physics.integration_parts(),
                    )
                    effective_change = (
                        effective_change
                        or len(high_parts) != len(low_parts)
                        or any(
                            any(
                                not np.array_equal(
                                    getattr(high.source.mean_rays, field),
                                    getattr(low.source.mean_rays, field),
                                )
                                for field in (
                                    "origin_lab_m",
                                    "direction_lab",
                                    "wavelength_A",
                                    "source_weight",
                                )
                            )
                            or not np.array_equal(
                                high.source.conditional_origin_factor_lab_m,
                                low.source.conditional_origin_factor_lab_m,
                            )
                            for high, low in zip(high_parts, low_parts, strict=False)
                        )
                    )
                    validate_native_rod_coverage(
                        trial_physics,
                        a_bounds_A=(parameters[0].lower, parameters[0].upper),
                        c_bounds_A=(parameters[1].lower, parameters[1].upper),
                    )
                if not effective_change:
                    raise ValueError(
                        "numerical check leaves all effective probe quadratures unchanged"
                    )
                refined = np.array([refined_evaluator.predict(v, n) for v in candidates])
                refined_stitches = [refined_evaluator.stitch_state(v, n) for v in candidates]
                tolerances = dict(plan["numerical_tolerances"])
                tolerances["require_constrained_objective_agreement"] = guarded
                comparison = compare_native_predictions(
                    objective_observations,
                    reference,
                    refined,
                    fixed_scale=fixed_scale,
                    **tolerances,
                )
                prefix = "constrained_" if guarded else ""
                center_errors.setdefault(label, []).append(
                    float(
                        comparison[f"refined_{prefix}objective"][0]
                        - comparison[f"reference_{prefix}objective"][0]
                    )
                )
                for name, value in comparison.items():
                    if isinstance(value, np.ndarray):
                        arrays[f"{label}_numerical_{i}_{name}"] = value
                arrays[f"{label}_numerical_{i}_raw"] = refined
                manifest["numerical_checks"].append(
                    dict(
                        name=specification["name"],
                        scope=label,
                        N=n,
                        objective_observation_revision=objective_observations.input_revision,
                        guard_diagnostics_included=comparison["guard_diagnostics_included"],
                        reference_stitches=reference_stitches,
                        refined_stitches=refined_stitches,
                        empirical_agreement=comparison["empirical_agreement"],
                        compile_count=refined_evaluator.compile_count,
                        compile_seconds=refined_evaluator.compile_seconds,
                        source_partitions_candidate_index=len(candidates) - 1,
                        source_partitions=[
                            dict(
                                rods=[(r.h, r.k, r.population) for r in p.rods],
                                divergence_order=p.source_definition.divergence_order,
                                row_count=len(p.source.mean_rays.wavelength_A),
                                source_revision=p.source.revision,
                            )
                            for p in trial_physics.integration_parts()
                        ],
                    )
                )
                agreement = agreement and comparison["empirical_agreement"]
                refined_evaluator.clear_responses()
                save()
                print(
                    json.dumps(
                        dict(
                            numerical_check=specification["name"],
                            empirical_agreement=comparison["empirical_agreement"],
                            maximum_whitened_rms=float(np.max(comparison["whitened_rms"])),
                        )
                    ),
                    flush=True,
                )
            return "empirical_agreement_at_declared_probes" if agreement else "not_qualified"

        target_observations = observations
        if "synthetic" in plan:
            specification = plan["synthetic"]
            truth = np.asarray(specification["truth"], dtype=float)
            raw = predict(truth, specification["coherent_repeats"])
            count = specification["scale"] * raw
            if specification["add_noise"]:
                supported = observations.valid
                count[supported] += np.linalg.cholesky(
                    observations.covariance_count2[np.ix_(supported, supported)]
                ) @ np.random.default_rng(specification["seed"]).standard_normal(
                    int(supported.sum())
                )
            target_observations = replace(
                observations,
                net_count=count,
                fit_target=observations.fit_operator @ count,
                allow_guard_constraints=False,
                input_revision=hashlib.sha256(count.tobytes()).hexdigest(),
            )
            arrays["synthetic_target"] = count
            manifest["synthetic"] = specification
        complete_target = target_observations
        if training is not None:
            target_observations = training_observations(target_observations, mask)
            arrays["training_mask"] = mask
            arrays["validation_mask"] = observations.valid & ~mask
        if checks:
            guarded = bool(stages and stages[-1]["enforce_historical_guards"])
            qualification_scale = plan.get("qualification_scale")
            if qualification_scale is None:
                qualification_scale, _ = target_observations.profile_scale(
                    baseline, enforce_guards=guarded
                )
            manifest["numerical_status"] = run_checks(
                "initial",
                plan["qualification_candidates"],
                repeats[0],
                qualification_scale,
                target_observations,
                guarded,
            )
        manifest["initial_numerical_status"] = manifest["numerical_status"]
        if (
            require_qualification
            and (stages or plan.get("profiles"))
            and manifest["initial_numerical_status"] == "not_qualified"
        ):
            manifest.update(
                selected=None,
                workflow_complete=False,
                execution_status="initial_numerical_qualification_failed",
            )
            save()
            print(
                json.dumps(
                    dict(output=str(args.output), execution_status=manifest["execution_status"])
                ),
                flush=True,
            )
            raise SystemExit(2)
        options = dict(
            calibration=calibration,
            finite_difference_step=plan["finite_difference_step"],
            callback=checkpoint,
        )
        if plan.get("sensitivity"):
            sensitivity = native_sensitivity(
                lambda v: predict(v, repeats[0]),
                target_observations,
                parameters,
                starts[0],
                relative_step=plan["finite_difference_step"],
            )
            for name, value in sensitivity.items():
                if isinstance(value, np.ndarray):
                    arrays[f"initial_sensitivity_{name}"] = value
            save()
        fitted = []
        for n in repeats:
            warm_starts = starts
            for stage in stages:
                active = tuple(stage["active_parameters"])
                fixed = {
                    name: float(warm_starts[0, i])
                    for i, name in enumerate(names)
                    if name not in active
                }
                correction = stage.get("reference_correction")
                if correction is not None:
                    low_physics = native_physics_with(
                        original, plan, correction["numerical_override"]
                    )
                    low_evaluator = make_native_evaluator(low_physics, observations, plan)
                    warm, evidence = reference_corrected_start(
                        lambda v, n=n: predict(v, n),
                        lambda v, n=n, low=low_evaluator: low.predict(v, n),
                        target_observations,
                        parameters,
                        warm_starts[0],
                        trust_radii=correction["trust_radii"],
                        numerical_tolerances=plan["numerical_tolerances"],
                        maximum_updates=correction.get("maximum_updates", 2),
                        fixed_values=fixed,
                        calibration=calibration,
                        method=stage.get(
                            "method",
                            plan.get(
                                "method", "slsqp" if stage["enforce_historical_guards"] else "trf"
                            ),
                        ),
                        maximum_iterations=stage["maximum_iterations"],
                        maximum_function_evaluations=correction.get(
                            "maximum_function_evaluations", 30
                        ),
                        finite_difference_step=plan["finite_difference_step"],
                        enforce_historical_guards=stage["enforce_historical_guards"],
                    )
                    for record in evidence:
                        if "comparison" in record:
                            record["comparison"] = {
                                name: value.tolist() if isinstance(value, np.ndarray) else value
                                for name, value in record["comparison"].items()
                            }
                    manifest.setdefault("reference_acceleration", []).append(
                        dict(
                            N=n,
                            stage=stage["name"],
                            numerical_override=correction["numerical_override"],
                            records=evidence,
                            low_compile_count=low_evaluator.compile_count,
                            low_compile_seconds=low_evaluator.compile_seconds,
                            low_evaluation_count=low_evaluator.evaluation_count,
                            acceptance="exact_checked_warm_start_only",
                        )
                    )
                    low_evaluator.clear_responses()
                    warm_starts = np.vstack([warm, warm_starts])
                    save()
                result = fit_native_parameters(
                    lambda v, n=n: predict(v, n),
                    target_observations,
                    parameters,
                    warm_starts,
                    fixed_values=fixed,
                    previous_predictions=previous_predictions(n),
                    method=stage.get(
                        "method",
                        plan.get(
                            "method", "slsqp" if stage["enforce_historical_guards"] else "trf"
                        ),
                    ),
                    maximum_function_evaluations=stage.get(
                        "maximum_function_evaluations", plan.get("maximum_function_evaluations", 80)
                    ),
                    predict_many=lambda values, n=n: predict_many(values, n),
                    maximum_iterations=stage["maximum_iterations"],
                    enforce_historical_guards=stage["enforce_historical_guards"],
                    **options,
                )
                manifest["fits"].append(
                    dict(
                        N=n,
                        stage=stage["name"],
                        best_evaluated=_point_record(result.best_evaluated),
                        best_feasible=_point_record(result.best_feasible),
                        best_converged=_point_record(result.best_converged),
                        minimum_resolved=result.minimum_resolved,
                        runs=[_point_record(p) for p in result.runs],
                    )
                )
                chosen = native_fit_candidate(result)
                warm_starts = np.vstack([chosen.parameter_values, starts])
                arrays[f"fit_N{n}_{stage['name']}"] = chosen.prediction_count
                save()
            if stages:
                fitted.append((n, result))
        manifest["fitted_numerical_status"] = {}
        for n, result in fitted:
            label = f"fit_N{n}"
            manifest["fitted_numerical_status"][label] = "not_qualified"
            if checks and plan.get("qualify_fitted_candidates", True):
                point = native_fit_candidate(result)
                center = point.parameter_values
                directions = (
                    np.asarray(plan["qualification_candidates"])[1:]
                    - np.asarray(plan["qualification_candidates"])[0]
                )
                lower = np.array([p.lower for p in parameters])
                upper = np.array([p.upper for p in parameters])
                probes = [center]
                for direction in directions:
                    trial = np.clip(center + direction, lower, upper)
                    if np.linalg.norm(trial - center) < np.linalg.norm(direction) / 2:
                        trial = np.clip(center - direction, lower, upper)
                    probes.append(trial)
                manifest["fitted_numerical_status"][label] = run_checks(
                    label,
                    probes,
                    n,
                    point.scale,
                    target_observations,
                    stages[-1]["enforce_historical_guards"],
                )
        manifest["discrete_ranking_numerical_status"] = "not_qualified"
        if fitted and all(f"fit_N{n}" in center_errors for n, _ in fitted):
            offsets = np.array([center_errors[f"fit_N{n}"] for n, _ in fitted])
            arrays["discrete_objective_numerical_offsets"] = offsets
            arrays["discrete_objective_numerical_contrast_errors"] = offsets - offsets[0]
            if (
                np.max(np.ptp(offsets, axis=0))
                <= plan["numerical_tolerances"]["maximum_objective_contrast_error"]
            ):
                manifest["discrete_ranking_numerical_status"] = (
                    "empirical_agreement_at_fitted_candidates"
                )
        if fitted:
            manifest["numerical_status"] = (
                "empirical_agreement_at_declared_probes"
                if manifest["initial_numerical_status"] != "not_qualified"
                and all(v != "not_qualified" for v in manifest["fitted_numerical_status"].values())
                and manifest["discrete_ranking_numerical_status"] != "not_qualified"
                else "not_qualified"
            )
        eligible = [(n, native_fit_candidate(r)) for n, r in fitted]
        if eligible:
            selected_n, selected = min(eligible, key=lambda item: item[1].objective)
            manifest["optimizer_candidate"] = dict(N=selected_n, **_point_record(selected))
            manifest["selected"] = None
            manifest["all_choices_resolved"] = all(r.minimum_resolved for _, r in fitted)
            arrays["optimizer_candidate_prediction_count"] = selected.prediction_count
            if training is not None:
                validation = conditional_validation(
                    complete_target,
                    selected.prediction_count,
                    mask,
                    observations.valid & ~mask,
                    groups=validation_masks,
                )
                for name, value in validation.items():
                    if isinstance(value, np.ndarray):
                        arrays[f"validation_{name}"] = value
                manifest["validation"] = {
                    k: v for k, v in validation.items() if not isinstance(v, np.ndarray)
                }
                validation_checks = []
                for i, check in enumerate(checks):
                    key = f"fit_N{selected_n}_numerical_{i}_raw"
                    if key not in arrays:
                        continue
                    refined_raw = arrays[key][0]
                    refined_scale, _ = target_observations.profile_scale(refined_raw)
                    comparison = compare_conditional_predictions(
                        complete_target,
                        selected.prediction_count,
                        refined_scale * refined_raw,
                        mask,
                        observations.valid & ~mask,
                        maximum_whitened_rms=plan["numerical_tolerances"]["maximum_whitened_rms"],
                        maximum_objective_error=plan["numerical_tolerances"][
                            "maximum_objective_contrast_error"
                        ],
                    )
                    arrays[f"validation_numerical_{i}_whitened_difference"] = comparison.pop(
                        "whitened_difference"
                    )
                    validation_checks.append(dict(name=check["name"], **comparison))
                manifest["validation"]["numerical_checks"] = validation_checks
                manifest["validation"]["numerical_status"] = (
                    "empirical_agreement_at_candidate"
                    if checks
                    and len(validation_checks) == len(checks)
                    and all(c["empirical_agreement"] for c in validation_checks)
                    else "not_qualified"
                )
        else:
            selected_n, selected = repeats[0], None
            manifest["selected"] = None
        reference_values = starts[0] if selected is None else selected.parameter_values
        manifest["reported_candidate_inactive_parameters"] = evaluator.inactive_parameters(
            reference_values, selected_n
        )
        manifest["reported_candidate_stitch"] = evaluator.stitch_state(reference_values, selected_n)
        bound, reported_arguments, _, _ = evaluator.bind(reference_values, selected_n)
        manifest["reported_candidate_physics"] = dict(
            input_revision=bound.input_revision,
            source_revision=bound.source.revision,
            material_revision=bound.material.material_revision,
            instrument_revision=_instrument_revision(
                replace(bound.instrument, film_thickness_A=reported_arguments["film_thickness_A"])
            ),
            source_rule=bound.source_definition.kind,
            source_row_count=len(bound.source.mean_rays.wavelength_A),
            integration_parts=[
                dict(
                    rods=[(r.h, r.k, r.population) for r in p.rods],
                    source_revision=p.source.revision,
                    material_revision=p.material.material_revision,
                    divergence_order=p.source_definition.divergence_order,
                    source_row_count=len(p.source.mean_rays.wavelength_A),
                )
                for p in bound.integration_parts()
            ],
        )
        if plan.get("controls"):
            manifest["background_controls"] = []
            controls = load_native_background_controls(args.observations)
            cases = [("baseline", starts[0], repeats[0], baseline_scale)]
            if selected is not None:
                cases.append(("candidate", reference_values, selected_n, selected.scale))
            budget = plan.get("maximum_control_signal_measurement_sigma")
            if budget is not None and (not np.isfinite(budget) or budget <= 0):
                raise ValueError(
                    "control contamination budget must be declared positive and finite"
                )
            for label, values, n, scale in cases:
                bound, arguments, mosaic, stack = evaluator.bind(values, n)
                for i, control in enumerate(controls):
                    signal = np.zeros(control.projection.observation_count)
                    for part in bound.integration_parts():
                        detector = part.detector(mosaic=mosaic, **arguments)
                        detector = replace(detector, specular_stitch_stack=stack)
                        signal += scale * detector.integrate_native_regions(control.projection)
                    diagnostic = control.signal_diagnostic(signal)
                    arrays[f"control_{label}_{i}_signal_count"] = signal
                    arrays[f"control_{label}_{i}_split"] = control.split
                    for name, value in diagnostic.items():
                        if isinstance(value, np.ndarray):
                            arrays[f"control_{label}_{i}_{name}"] = value
                    manifest["background_controls"].append(
                        dict(
                            scope=label,
                            layout=i,
                            projection_revision=control.projection.projection_revision,
                            control_revision=control.revision,
                            split_summary=diagnostic["split_summary"],
                            status=(
                                "diagnostic_only"
                                if budget is None
                                else "within_declared_budget"
                                if np.max(diagnostic["signal_measurement_sigma"]) <= budget
                                else "fixed_background_validity_unresolved"
                            ),
                            numerical_status="not_qualified",
                        )
                    )
                    save()
                    print(
                        json.dumps(
                            dict(
                                control_scope=label,
                                layout=i,
                                maximum_signal_measurement_sigma=float(
                                    np.max(diagnostic["signal_measurement_sigma"])
                                ),
                            )
                        ),
                        flush=True,
                    )
        if plan.get("sensitivity"):
            sensitivity = native_sensitivity(
                lambda v: evaluator.predict(v, selected_n),
                target_observations,
                parameters,
                reference_values,
                relative_step=plan["finite_difference_step"],
            )
            for name, value in sensitivity.items():
                if isinstance(value, np.ndarray):
                    arrays[f"sensitivity_{name}"] = value
            manifest["sensitivity_stitches"] = [
                evaluator.stitch_state(v, selected_n) for v in sensitivity["probe_values"]
            ]
        manifest["profile_improved_candidates"] = []
        for specification in plan.get("profiles", []):
            predictors = {n: (lambda v, n=n: predict(v, n)) for n in repeats}
            profile = profile_native_parameter(
                predictors,
                target_observations,
                parameters,
                np.vstack([reference_values, starts]),
                name=specification["name"],
                grid=specification["grid"],
                method=specification.get("method", plan.get("method", "trf")),
                maximum_function_evaluations=specification.get(
                    "maximum_function_evaluations", plan.get("maximum_function_evaluations", 80)
                ),
                predict_many_by_choice={
                    n: (lambda values, n=n: predict_many(values, n)) for n in repeats
                },
                previous_predictions_by_choice={n: previous_predictions(n) for n in repeats},
                maximum_iterations=specification["maximum_iterations"],
                enforce_historical_guards=False,
                **options,
            )
            if fitted:
                fitted_by_choice = dict(fitted)
                for n, fits in profile["fits"].items():
                    fitted_minimum = fitted_by_choice[n].best_converged
                    if fitted_minimum is None:
                        continue
                    for i, fit in fits.items():
                        point = (
                            fit.best_feasible
                            if stages[-1]["enforce_historical_guards"]
                            else fit.best_evaluated
                        )
                        if (
                            point is not None
                            and point.objective
                            < fitted_minimum.objective
                            - 1e-7 * max(1.0, abs(fitted_minimum.objective))
                        ):
                            manifest["all_choices_resolved"] = False
                            manifest["profile_improved_candidates"].append(
                                dict(
                                    parameter=specification["name"],
                                    N=n,
                                    grid_index=i,
                                    **_point_record(point),
                                )
                            )
            for name in (
                "grid",
                "objective",
                "resolved",
                "envelope_objective",
                "envelope_resolved",
            ):
                arrays[f"profile_{specification['name']}_{name}"] = profile[name]
            profile_labels = []
            point_status = {}
            for n, fits in profile["fits"].items():
                label = f"profile_{specification['name']}_N{n}"
                profile_labels.append(label)
                points = [
                    reference_values,
                    *[
                        (
                            f.best_converged if f.best_converged is not None else f.best_evaluated
                        ).parameter_values
                        for f in (fits[i] for i in sorted(fits))
                    ],
                ]
                point_status[str(n)] = (
                    run_checks(
                        label,
                        points,
                        n,
                        target_observations.profile_scale(evaluator.predict(reference_values, n))[
                            0
                        ],
                        target_observations,
                        False,
                    )
                    if checks and plan.get("qualify_profile_candidates", True)
                    else "not_qualified"
                )
            profile_agreement = bool(checks) and all(
                v != "not_qualified" for v in point_status.values()
            )
            curve_errors = []
            for i, check in enumerate(checks):
                keys = [f"{label}_numerical_{i}_" for label in profile_labels]
                if any(key + "refined_objective" not in arrays for key in keys):
                    profile_agreement = False
                    continue
                offsets = np.concatenate(
                    [
                        arrays[key + "refined_objective"] - arrays[key + "reference_objective"]
                        for key in keys
                    ]
                )
                arrays[f"profile_{specification['name']}_numerical_{i}_objective_offsets"] = offsets
                error = float(np.ptp(offsets))
                curve_errors.append(
                    dict(name=check["name"], maximum_objective_contrast_error=error)
                )
                profile_agreement &= (
                    error <= plan["numerical_tolerances"]["maximum_objective_contrast_error"]
                )
            manifest["profiles"].append(
                dict(
                    parameter=specification["name"],
                    choices=profile["choices"],
                    interval_status=profile["interval_status"],
                    numerical_status="empirical_agreement_at_profile_candidates"
                    if profile_agreement
                    else "not_qualified",
                    numerical_status_by_choice=point_status,
                    numerical_grid_indices={
                        str(n): sorted(fits) for n, fits in profile["fits"].items()
                    },
                    numerical_curve_checks=curve_errors,
                    guard_conditioned=profile["guard_conditioned"],
                    observation_revision=target_observations.input_revision,
                    fits=[
                        dict(
                            N=n,
                            grid_index=i,
                            best_evaluated=_point_record(fit.best_evaluated),
                            best_feasible=_point_record(fit.best_feasible),
                            best_converged=_point_record(fit.best_converged),
                            minimum_resolved=fit.minimum_resolved,
                            runs=[_point_record(point) for point in fit.runs],
                        )
                        for n, fits in profile["fits"].items()
                        for i, fit in fits.items()
                    ],
                    candidate_stitches=[
                        dict(
                            N=n,
                            grid_index=i,
                            state=evaluator.stitch_state(
                                (
                                    fit.best_converged
                                    if fit.best_converged is not None
                                    else fit.best_evaluated
                                ).parameter_values,
                                n,
                            ),
                        )
                        for n, fits in profile["fits"].items()
                        for i, fit in fits.items()
                    ],
                    all_points_resolved=bool(profile["resolved"].all()),
                )
            )
        if manifest["profiles"]:
            manifest["identification_status"] = (
                "raw_profiles_available_without_confidence_intervals"
            )
            if any(p["numerical_status"] == "not_qualified" for p in manifest["profiles"]):
                manifest["numerical_status"] = "not_qualified"
        if manifest.get("validation", {}).get("numerical_status") == "not_qualified":
            manifest["numerical_status"] = "not_qualified"
        if (
            selected is not None
            and selected.optimizer_converged
            and manifest.get("all_choices_resolved", False)
            and manifest["numerical_status"] != "not_qualified"
            and all(p["all_points_resolved"] for p in manifest["profiles"])
            and not manifest["profile_improved_candidates"]
        ):
            manifest["selected"] = manifest["optimizer_candidate"]
            arrays["selected_prediction_count"] = selected.prediction_count
        manifest["selection_status"] = (
            "numerically_qualified_candidate"
            if manifest["selected"] is not None
            else "profile_improvement_requires_joint_refit"
            if manifest["profile_improved_candidates"]
            else "no_resolved_numerically_qualified_selection"
        )
        manifest["workflow_complete"] = True
        manifest["execution_status"] = "completed"
        save()
        print(
            json.dumps(
                dict(
                    output=str(args.output),
                    numerical_status=manifest["numerical_status"],
                    selected=manifest["selected"],
                    elapsed_seconds=manifest["elapsed_seconds"],
                )
            ),
            flush=True,
        )
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(
            execution_status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            failure_type=type(exc).__name__,
            failure_message=str(exc),
            workflow_complete=False,
        )
        save()
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
