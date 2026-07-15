"""Bounded solver for the exact frozen-caked geometry objective."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult

from ra_sim.fitting.caked_geometry_objective import (
    ACTIVE_PARAMETER_RUNGS,
    CakedGeometryObjectiveError,
    CakedGeometryProblem,
    ObjectiveEvaluation,
    branch_selection_evidence_to_jsonable,
    caked_geometry_parameter_numerics,
    coordinate_route_to_jsonable,
    evaluate_caked_geometry_objective,
    objective_evaluation_to_jsonable,
    trial_predictions_from_fresh_rows,
)
from ra_sim.fitting.geometry_fit_parameters import (
    build_geometry_fit_parameter_bounds,
    initial_geometry_fit_value,
)


RESULT_SCHEMA = "caked_geometry_first_rung_result_v2"
RESIDUAL_COMPONENT_ORDER = (
    "branch_0_delta_two_theta_deg",
    "branch_0_wrapped_delta_phi_deg",
    "branch_1_delta_two_theta_deg",
    "branch_1_wrapped_delta_phi_deg",
)
_FRESH_REBUILD_SOURCE = "geometry_manual_rebuild_source_rows_for_background"
_FRESH_REBUILD_STATUS = "rebuilt_for_trial_params"
_FRESH_ROWS_AUTHORITY = "pre_filter_fresh_rebuild_output"


def _emit_status(callback: Callable[[str], None] | None, message: str) -> None:
    if callable(callback):
        callback(str(message))


def _dataset_index(spec: Mapping[str, object]) -> int | None:
    value = spec.get("dataset_index", spec.get("background_index"))
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _single_bg0_dataset_spec(
    dataset_specs: Sequence[Mapping[str, object]] | None,
) -> Mapping[str, object]:
    specs = [spec for spec in (dataset_specs or ()) if isinstance(spec, Mapping)]
    if len(specs) != 1 or _dataset_index(specs[0]) != 0:
        raise CakedGeometryObjectiveError(
            "caked_geometry_fit_requires_one_bg0_dataset",
            dataset_count=len(specs),
        )
    return specs[0]


def _positive_float(value: object, *, name: str, default: float) -> float:
    if value is None:
        return float(default)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CakedGeometryObjectiveError(
            "invalid_caked_geometry_solver_setting",
            name=name,
            value=value,
        ) from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise CakedGeometryObjectiveError(
            "invalid_caked_geometry_solver_setting",
            name=name,
            value=value,
        )
    return numeric


def _solver_settings(
    refinement_config: Mapping[str, object] | None,
) -> dict[str, object]:
    config = refinement_config if isinstance(refinement_config, Mapping) else {}
    raw_optimizer = config.get("optimizer", {})
    if not isinstance(raw_optimizer, Mapping):
        raise CakedGeometryObjectiveError(
            "caked_geometry_optimizer_config_must_be_mapping"
        )
    try:
        max_nfev = int(raw_optimizer.get("max_nfev", 120))
    except (TypeError, ValueError) as exc:
        raise CakedGeometryObjectiveError(
            "invalid_caked_geometry_solver_setting",
            name="max_nfev",
            value=raw_optimizer.get("max_nfev"),
        ) from exc
    if max_nfev < 1:
        raise CakedGeometryObjectiveError(
            "invalid_caked_geometry_solver_setting",
            name="max_nfev",
            value=max_nfev,
        )
    return {
        "max_nfev": max_nfev,
        "ftol": _positive_float(
            raw_optimizer.get("ftol"), name="ftol", default=1.0e-4
        ),
        "xtol": _positive_float(
            raw_optimizer.get("xtol"), name="xtol", default=1.0e-4
        ),
        "gtol": _positive_float(
            raw_optimizer.get("gtol"), name="gtol", default=1.0e-4
        ),
    }


def _fresh_rows(payload: object) -> tuple[list[Mapping[str, object]], str]:
    if not isinstance(payload, Mapping):
        raise CakedGeometryObjectiveError(
            "fresh_prediction_builder_returned_invalid_payload",
            payload_type=type(payload).__name__,
        )
    source = str(payload.get("source") or "")
    if source != _FRESH_REBUILD_SOURCE:
        raise CakedGeometryObjectiveError(
            "fresh_prediction_source_not_authoritative",
            source=source,
        )
    if (
        payload.get("source_rows_rebuilt_or_reused") != _FRESH_REBUILD_STATUS
        or payload.get("rebuild_attempted") is not True
    ):
        raise CakedGeometryObjectiveError("fresh_prediction_rows_not_rebuilt")
    if payload.get("fresh_detector_native_rows_authority") != _FRESH_ROWS_AUTHORITY:
        raise CakedGeometryObjectiveError(
            "fresh_prediction_rows_authority_invalid"
        )
    if payload.get("caked_geometry_equivalence_identity_only") is not True:
        raise CakedGeometryObjectiveError(
            "fresh_prediction_equivalence_identity_not_exact"
        )
    raw_rows = payload.get("fresh_detector_native_rows")
    if not isinstance(raw_rows, Sequence) or isinstance(
        raw_rows, (str, bytes, bytearray)
    ):
        raise CakedGeometryObjectiveError("fresh_prediction_rows_missing")
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(rows) != len(raw_rows) or not rows:
        raise CakedGeometryObjectiveError("invalid_fresh_prediction_rows")
    declared_count = payload.get("fresh_detector_native_row_count")
    if declared_count != len(rows):
        raise CakedGeometryObjectiveError(
            "fresh_prediction_row_count_mismatch",
            declared_count=declared_count,
            actual_count=len(rows),
        )
    return rows, source


def _branch_identity_signature(
    evaluation: ObjectiveEvaluation,
) -> tuple[tuple[object, ...], tuple[dict[str, object], ...]]:
    signature: list[tuple[object, ...]] = []
    payloads: list[dict[str, object]] = []
    for expected_branch, prediction in enumerate(evaluation.predictions):
        evidence = prediction.selection_evidence
        payload = branch_selection_evidence_to_jsonable(evidence)
        if evidence is None or payload is None:
            raise CakedGeometryObjectiveError(
                "fresh_branch_selection_evidence_missing",
                branch=expected_branch,
            )
        if (
            prediction.branch_index != expected_branch
            or evidence.candidate_count < 1
            or evidence.candidate_count != prediction.candidate_count
            or evidence.selected_candidate_key not in evidence.candidate_keys
        ):
            raise CakedGeometryObjectiveError(
                "invalid_fresh_branch_selection_evidence",
                branch=expected_branch,
                evidence=payload,
            )
        signature.append(
            (
                expected_branch,
                evidence.selection_rule,
                evidence.selection_metric,
                evidence.candidate_count,
                evidence.candidate_identity_digest,
                evidence.selected_candidate_key,
                evidence.selected_candidate_digest,
            )
        )
        payloads.append(payload)
    return tuple(signature), tuple(payloads)


def _point_residual(evaluation: ObjectiveEvaluation) -> np.ndarray:
    residual = np.asarray(evaluation.residuals_deg, dtype=float)
    if residual.shape != (4,) or not np.all(np.isfinite(residual)):
        raise CakedGeometryObjectiveError(
            "invalid_caked_geometry_point_residual",
            shape=list(residual.shape),
        )
    return residual


def _wrapped_phi_delta_deg(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 and float(value) > 0.0 else wrapped


def _directed_line_evidence(
    measured_points: Sequence[Sequence[float]],
    predicted_points: Sequence[Sequence[float]],
) -> dict[str, object]:
    measured = np.asarray(measured_points, dtype=float)
    predicted = np.asarray(predicted_points, dtype=float)
    if (
        measured.shape != (2, 2)
        or predicted.shape != (2, 2)
        or not np.all(np.isfinite(measured))
        or not np.all(np.isfinite(predicted))
    ):
        raise CakedGeometryObjectiveError("branch_line_points_invalid")

    def directed_vector(points: np.ndarray, *, name: str) -> np.ndarray:
        phi_delta = _wrapped_phi_delta_deg(float(points[1, 1] - points[0, 1]))
        if abs(abs(phi_delta) - 180.0) <= 1.0e-9:
            raise CakedGeometryObjectiveError(
                "branch_line_antipodal_phi_ambiguous",
                point_set=name,
            )
        vector = np.asarray(
            [float(points[1, 0] - points[0, 0]), phi_delta],
            dtype=float,
        )
        if float(np.linalg.norm(vector)) <= np.finfo(float).eps:
            raise CakedGeometryObjectiveError(
                "branch_line_degenerate",
                point_set=name,
            )
        return vector

    measured_vector = directed_vector(measured, name="measured")
    predicted_vector = directed_vector(predicted, name="predicted")
    measured_angle = math.degrees(
        math.atan2(float(measured_vector[1]), float(measured_vector[0]))
    )
    predicted_angle = math.degrees(
        math.atan2(float(predicted_vector[1]), float(predicted_vector[0]))
    )
    angle_residual = _wrapped_phi_delta_deg(predicted_angle - measured_angle)
    measured_length = float(np.linalg.norm(measured_vector))
    predicted_length = float(np.linalg.norm(predicted_vector))
    return {
        "schema": "caked_geometry_directed_branch_line_v1",
        "branch_order": [0, 1],
        "measured_branch_vector_deg": measured_vector.tolist(),
        "predicted_branch_vector_deg": predicted_vector.tolist(),
        "measured_branch_length_deg": measured_length,
        "predicted_branch_length_deg": predicted_length,
        "measured_directed_angle_deg": measured_angle,
        "predicted_directed_angle_deg": predicted_angle,
        "angle_residual_deg": angle_residual,
    }


def _line_evidence(evaluation: ObjectiveEvaluation) -> dict[str, object]:
    rows = tuple(sorted(evaluation.rows, key=lambda row: row.branch_index))
    if tuple(row.branch_index for row in rows) != (0, 1):
        raise CakedGeometryObjectiveError("branch_line_requires_exact_branches_0_1")
    angular = _directed_line_evidence(
        [row.measured_caked_deg for row in rows],
        [row.predicted_caked_deg for row in rows],
    )
    measured_length = float(angular["measured_branch_length_deg"])
    angle_residual = float(angular["angle_residual_deg"])
    return {
        "schema": "caked_geometry_dependent_line_residual_v1",
        "residual_space": "caked_deg",
        "residual_units": "deg",
        "measured_line_length": measured_length,
        "predicted_line_length": float(angular["predicted_branch_length_deg"]),
        "directed_angle_residual_deg": angle_residual,
        "dependent_line_residual": float(
            measured_length * math.sin(math.radians(angle_residual) / 2.0)
        ),
        "angle_evidence": angular,
    }


def _joint_residual(evaluation: ObjectiveEvaluation) -> np.ndarray:
    residual = np.concatenate(
        (
            _point_residual(evaluation),
            [float(_line_evidence(evaluation)["dependent_line_residual"])],
        )
    )
    if residual.shape != (5,) or not np.all(np.isfinite(residual)):
        raise CakedGeometryObjectiveError(
            "invalid_caked_geometry_joint_residual",
            shape=list(residual.shape),
        )
    return residual


def _point_metrics(evaluation: ObjectiveEvaluation) -> dict[str, object]:
    components = _point_residual(evaluation)
    norms = np.asarray(
        [
            math.hypot(row.delta_two_theta_deg, row.wrapped_delta_phi_deg)
            for row in evaluation.rows
        ],
        dtype=float,
    )
    return {
        "component_rms_deg": float(np.sqrt(np.mean(components * components))),
        "point_rms_deg": float(np.sqrt(np.mean(norms * norms))),
        "point_max_deg": float(np.max(norms)),
    }


def _termination_condition(status: int) -> str:
    return {
        0: "max_nfev",
        1: "gtol",
        2: "ftol",
        3: "xtol",
        4: "ftol_and_xtol",
    }.get(int(status), "failure")


def _target_rows(problem: CakedGeometryProblem) -> list[dict[str, object]]:
    return [
        {
            "background_index": target.background_index,
            "q_group_key": list(target.q_group_key),
            "hkl": list(target.hkl),
            "hkl_equivalence_key": list(target.hkl_equivalence_key),
            "branch_index": target.branch_index,
            "measurement_origin": target.measurement_origin,
            "measured_detector_display_px": list(
                target.measured_detector_display_px
            ),
            "measured_caked_deg": list(target.measured_caked_deg),
            "measured_native_px": list(target.measured_native_px),
            "saved_caked_audit_deg": (
                list(target.saved_caked_audit_deg)
                if target.saved_caked_audit_deg is not None
                else None
            ),
            "coordinate_route": coordinate_route_to_jsonable(
                target.coordinate_route
            ),
            "pair_id": target.pair_id,
            "source_table_index": target.source_table_index,
            "source_row_index": target.source_row_index,
            "source_reflection_index": target.source_reflection_index,
        }
        for target in sorted(problem.targets, key=lambda item: item.branch_index)
    ]


def _diagnostic_rows(evaluation: ObjectiveEvaluation) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in sorted(evaluation.rows, key=lambda item: item.branch_index):
        native_dx = float(row.predicted_native_px[0] - row.measured_native_px[0])
        native_dy = float(row.predicted_native_px[1] - row.measured_native_px[1])
        output.append(
            {
                "dataset_index": 0,
                "background_index": 0,
                "pair_id": row.pair_id,
                "q_group_key": list(row.q_group_key),
                "hkl": list(row.hkl),
                "hkl_equivalence_key": list(row.hkl_equivalence_key),
                "source_branch_index": int(row.branch_index),
                "match_status": "matched",
                "measured_detector_native_px": list(row.measured_native_px),
                "fit_prediction_detector_native_px": list(row.predicted_native_px),
                "observed_caked_deg": list(row.measured_caked_deg),
                "predicted_caked_deg": list(row.predicted_caked_deg),
                "fit_residual_caked_deg": [
                    float(row.delta_two_theta_deg),
                    float(row.wrapped_delta_phi_deg),
                ],
                "angular_residual_norm_deg": float(
                    math.hypot(
                        row.delta_two_theta_deg,
                        row.wrapped_delta_phi_deg,
                    )
                ),
                "prediction_source": row.prediction_source,
                "prediction_candidate_count": int(
                    row.prediction_candidate_count
                ),
                "prediction_selection_rule": row.prediction_selection_rule,
                "dx_px": native_dx,
                "dy_px": native_dy,
                "distance_px": float(math.hypot(native_dx, native_dy)),
            }
        )
    return output


def _absolute_difference_jacobian(
    x: Sequence[float],
    *,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    steps: Sequence[float],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(x, dtype=float).reshape(-1)
    if vector.shape != lower_bounds.shape or not np.all(np.isfinite(vector)):
        raise CakedGeometryObjectiveError(
            "invalid_caked_geometry_jacobian_vector"
        )
    base: np.ndarray | None = None
    columns: list[np.ndarray] = []
    for index, raw_step in enumerate(steps):
        step = float(raw_step)
        plus_room = float(upper_bounds[index] - vector[index])
        minus_room = float(vector[index] - lower_bounds[index])
        if plus_room >= step and minus_room >= step:
            plus = vector.copy()
            minus = vector.copy()
            plus[index] += step
            minus[index] -= step
            columns.append((residual_fn(plus) - residual_fn(minus)) / (2.0 * step))
            continue
        if base is None:
            base = residual_fn(vector)
        if plus_room >= minus_room and plus_room > 0.0:
            local_step = min(step, plus_room)
            plus = vector.copy()
            plus[index] += local_step
            columns.append((residual_fn(plus) - base) / local_step)
        elif minus_room > 0.0:
            local_step = min(step, minus_room)
            minus = vector.copy()
            minus[index] -= local_step
            columns.append((base - residual_fn(minus)) / local_step)
        else:
            raise CakedGeometryObjectiveError(
                "caked_geometry_jacobian_has_no_step",
                parameter_index=index,
            )
    matrix = np.column_stack(columns)
    if matrix.shape != (5, vector.size) or not np.all(np.isfinite(matrix)):
        raise CakedGeometryObjectiveError(
            "invalid_caked_geometry_jacobian",
            shape=list(matrix.shape),
        )
    return matrix


def _decorate_result(
    result: OptimizeResult,
    *,
    problem: CakedGeometryProblem,
    active_names: tuple[str, ...],
    x0: np.ndarray,
    initial: ObjectiveEvaluation,
    final: ObjectiveEvaluation,
    bounds_table: list[dict[str, object]],
    settings: Mapping[str, object],
    x_scale: tuple[float, ...],
    evaluation_count: int,
    branch_selection: tuple[dict[str, object], ...],
) -> OptimizeResult:
    initial_metrics = _point_metrics(initial)
    final_metrics = _point_metrics(final)
    initial_joint = _joint_residual(initial)
    final_joint = _joint_residual(final)
    initial_cost = 0.5 * float(np.dot(initial_joint, initial_joint))
    final_cost = 0.5 * float(np.dot(final_joint, final_joint))
    result.fun = final_joint
    result.cost = final_cost
    result.x0 = x0.copy()
    result.initial_x = x0.copy()
    result.caked_geometry_first_rung = True
    result.final_metric_name = "dynamic_angular_point_match"
    result.final_metric_space = "caked_deg"
    result.final_metric_units = "deg"
    result.weighted_objective_rms = float(final_metrics["component_rms_deg"])
    result.weighted_objective_rms_units = "deg"
    result.rms_deg = float(final_metrics["point_rms_deg"])
    result.max_deg = float(final_metrics["point_max_deg"])
    result.initial_residual_rms = float(initial_metrics["point_rms_deg"])
    result.final_residual_rms = float(final_metrics["point_rms_deg"])
    result.initial_residual_norm = float(np.linalg.norm(initial_joint))
    result.final_residual_norm = float(np.linalg.norm(final_joint))
    result.initial_residual_count = int(initial_joint.size)
    result.final_residual_count = int(final_joint.size)
    result.initial_cost = initial_cost
    result.final_cost = final_cost
    result.cost_reduction = initial_cost - final_cost
    result.objective_eval_count = int(evaluation_count)
    result.optimization_residual_space = "caked_deg"
    result.optimization_residual_component_count = 5
    result.point_match_diagnostics = _diagnostic_rows(final)
    status = int(getattr(result, "status", 0) or 0)
    success = bool(getattr(result, "success", False))
    termination = _termination_condition(status)
    fit_result = {
        "x0": x0.tolist(),
        "x": np.asarray(result.x, dtype=float).tolist(),
        "bounds_table": [dict(entry) for entry in bounds_table],
        "success": success,
        "status": status,
        "message": str(getattr(result, "message", "")),
        "nfev": int(getattr(result, "nfev", 0) or 0),
        "njev": int(getattr(result, "njev", 0) or 0),
        "cost": final_cost,
        "optimality": float(getattr(result, "optimality", float("nan"))),
        "active_mask": np.asarray(
            getattr(result, "active_mask", np.zeros(len(active_names), dtype=int)),
            dtype=int,
        ).tolist(),
        "converged": bool(success and status in {1, 2, 3, 4}),
        "termination_condition": termination,
    }
    summary: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "objective_name": "frozen_caked_geometry",
        "objective_space": "caked_deg",
        "acceptance_metric_space": "caked_deg",
        "metric_name": "dynamic_angular_point_match",
        "metric_unit": "deg",
        "residual_units": "deg",
        "residual_component_order": list(RESIDUAL_COMPONENT_ORDER),
        "active_parameter_names": list(active_names),
        "active_parameter_count": len(active_names),
        "solver_settings": {
            "method": "trf",
            "loss": "linear",
            "x_scale": list(x_scale),
            **dict(settings),
        },
        "problem_digest": problem.problem_digest,
        "frame_digest": problem.frame.frame_digest,
        "projector_digest": problem.frame.projector_digest,
        "projector_kind": problem.frame.projector_kind,
        "projector_signature_json": problem.frame.projector_signature_json,
        "selected_rows": _target_rows(problem),
        "matched_pair_count": 2,
        "missing_pair_count": 0,
        "fixed_source_resolved_count": 2,
        "branch_selection": list(branch_selection),
        "fresh_prediction_evaluation_count": int(evaluation_count),
        "initial_evaluation": objective_evaluation_to_jsonable(initial),
        "final_evaluation": objective_evaluation_to_jsonable(final),
        "optimization_residual_contract": {
            "schema": "caked_geometry_joint_point_line_residual_v1",
            "independent_point_component_count": 4,
            "dependent_line_component_count": 1,
            "initial_point_residual": _point_residual(initial).tolist(),
            "final_point_residual": _point_residual(final).tolist(),
            "initial_line": _line_evidence(initial),
            "final_line": _line_evidence(final),
            "initial_joint_residual": initial_joint.tolist(),
            "final_joint_residual": final_joint.tolist(),
        },
        "initial_rms_deg": float(initial_metrics["point_rms_deg"]),
        "initial_component_rms_deg": float(
            initial_metrics["component_rms_deg"]
        ),
        "final_rms_deg": float(final_metrics["point_rms_deg"]),
        "final_max_deg": float(final_metrics["point_max_deg"]),
        "final_component_rms_deg": float(final_metrics["component_rms_deg"]),
        "raw_angular_rms_deg": float(final_metrics["point_rms_deg"]),
        "raw_angular_max_deg": float(final_metrics["point_max_deg"]),
        "bounded_solver_converged": fit_result["converged"],
        "bounded_solver_status": status,
        "bounded_solver_termination_condition": termination,
        "fit_result": fit_result,
        "worst_angular_residual_rows": sorted(
            [dict(row) for row in result.point_match_diagnostics],
            key=lambda row: float(row["angular_residual_norm_deg"]),
            reverse=True,
        ),
    }
    result.point_match_summary = summary
    result.geometry_fit_progress = {
        "evaluation_count": int(evaluation_count),
        "solver_converged": fit_result["converged"],
        "solver_status": status,
        "solver_termination_condition": termination,
        "start_x": x0.tolist(),
        "end_x": np.asarray(result.x, dtype=float).tolist(),
        "initial_point_rms_deg": float(initial_metrics["point_rms_deg"]),
        "final_point_rms_deg": float(final_metrics["point_rms_deg"]),
    }
    return result


def solve_caked_geometry_first_rung(
    *,
    problem: CakedGeometryProblem,
    params: Mapping[str, object],
    var_names: Sequence[str],
    dataset_specs: Sequence[Mapping[str, object]] | None,
    refinement_config: Mapping[str, object] | None,
    least_squares_fn: Callable[..., object],
    status_callback: Callable[[str], None] | None = None,
) -> OptimizeResult:
    """Solve one typed exact-caked geometry problem."""

    if not isinstance(problem, CakedGeometryProblem):
        raise CakedGeometryObjectiveError("invalid_caked_geometry_problem")
    active_names = tuple(str(name) for name in var_names)
    if (
        active_names not in ACTIVE_PARAMETER_RUNGS
        or active_names != tuple(problem.active_parameter_names)
    ):
        raise CakedGeometryObjectiveError(
            "active_parameters_must_match_supported_problem_rung",
            names=list(active_names),
            problem_names=list(problem.active_parameter_names),
        )
    dataset_spec = _single_bg0_dataset_spec(dataset_specs)
    fresh_builder = dataset_spec.get("caked_geometry_fresh_native_rows_builder")
    if not callable(fresh_builder):
        raise CakedGeometryObjectiveError(
            "missing_caked_geometry_fresh_native_prediction_builder"
        )

    x0 = np.asarray(
        [
            initial_geometry_fit_value(
                name,
                params,
            )
            for name in active_names
        ],
        dtype=float,
    )
    config = refinement_config if isinstance(refinement_config, Mapping) else {}
    bounds_cfg = config.get("bounds", {})
    if not isinstance(bounds_cfg, Mapping):
        raise CakedGeometryObjectiveError("caked_geometry_bounds_must_be_mapping")
    bounds = build_geometry_fit_parameter_bounds(
        var_names=active_names,
        x0=x0,
        bounds_cfg=bounds_cfg,
    )
    if np.any(x0 < bounds.lower_bounds) or np.any(x0 > bounds.upper_bounds):
        raise CakedGeometryObjectiveError(
            "caked_geometry_x0_outside_bounds",
            x0=x0.tolist(),
            lower=bounds.lower_bounds.tolist(),
            upper=bounds.upper_bounds.tolist(),
        )
    numerics = caked_geometry_parameter_numerics(active_names)
    x_scale = tuple(float(item.x_scale) for item in numerics)
    jacobian_steps = tuple(float(item.jacobian_step) for item in numerics)
    bounds_table = [dict(entry) for entry in bounds.bounds_table]
    for index, scale in enumerate(x_scale):
        bounds_table[index]["scale"] = scale
    settings = _solver_settings(refinement_config)
    evaluation_count = 0
    baseline_signature: tuple[tuple[object, ...], ...] | None = None
    baseline_selection: tuple[dict[str, object], ...] | None = None

    def evaluate_vector(x: Sequence[float]) -> ObjectiveEvaluation:
        nonlocal baseline_selection, baseline_signature, evaluation_count
        vector = np.asarray(x, dtype=float).reshape(-1)
        if vector.shape != x0.shape or not np.all(np.isfinite(vector)):
            raise CakedGeometryObjectiveError(
                "invalid_caked_geometry_trial_vector",
                shape=list(vector.shape),
            )

        def predict_native(
            trial_parameters: Mapping[str, float],
            _targets: object,
        ):
            local_params = dict(params)
            local_params.update(trial_parameters)
            local_params["_active_fit_param_names"] = list(active_names)
            try:
                payload = fresh_builder(local_params=local_params)
            except CakedGeometryObjectiveError:
                raise
            except Exception as exc:
                raise CakedGeometryObjectiveError(
                    "fresh_prediction_builder_failed",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                ) from exc
            rows, source = _fresh_rows(payload)
            return trial_predictions_from_fresh_rows(
                problem,
                rows,
                source=f"dynamic_trial_simulation:fresh_rebuild:{source}",
            )

        evaluation = evaluate_caked_geometry_objective(
            problem,
            {
                name: float(vector[index])
                for index, name in enumerate(active_names)
            },
            predict_native=predict_native,
        )
        signature, selection = _branch_identity_signature(evaluation)
        if baseline_signature is None:
            baseline_signature = signature
            baseline_selection = selection
        elif signature != baseline_signature:
            raise CakedGeometryObjectiveError(
                "caked_geometry_branch_identity_changed",
                baseline=baseline_selection,
                candidate=selection,
            )
        evaluation_count += 1
        return evaluation

    _emit_status(status_callback, "Geometry fit: evaluating exact caked objective")
    initial = evaluate_vector(x0)

    def residual_fn(x: np.ndarray) -> np.ndarray:
        return _joint_residual(evaluate_vector(x))

    def jacobian_fn(x: np.ndarray) -> np.ndarray:
        return _absolute_difference_jacobian(
            x,
            residual_fn=residual_fn,
            steps=jacobian_steps,
            lower_bounds=bounds.lower_bounds,
            upper_bounds=bounds.upper_bounds,
        )

    _emit_status(status_callback, "Geometry fit: running bounded least squares")
    raw_result = least_squares_fn(
        residual_fn,
        x0,
        bounds=(bounds.lower_bounds, bounds.upper_bounds),
        method="trf",
        jac=jacobian_fn,
        x_scale=np.asarray(x_scale, dtype=float),
        loss="linear",
        f_scale=1.0,
        max_nfev=int(settings["max_nfev"]),
        ftol=float(settings["ftol"]),
        xtol=float(settings["xtol"]),
        gtol=float(settings["gtol"]),
    )
    result = (
        raw_result
        if isinstance(raw_result, OptimizeResult)
        else OptimizeResult(raw_result)
    )
    if not hasattr(result, "x"):
        raise CakedGeometryObjectiveError(
            "caked_geometry_solver_returned_invalid_result",
            result_type=type(raw_result).__name__,
        )
    final = evaluate_vector(result.x)
    _emit_status(status_callback, "Geometry fit: exact caked solve complete")
    return _decorate_result(
        result,
        problem=problem,
        active_names=active_names,
        x0=x0,
        initial=initial,
        final=final,
        bounds_table=bounds_table,
        settings=settings,
        x_scale=x_scale,
        evaluation_count=evaluation_count,
        branch_selection=baseline_selection or (),
    )


__all__ = ["RESULT_SCHEMA", "solve_caked_geometry_first_rung"]
