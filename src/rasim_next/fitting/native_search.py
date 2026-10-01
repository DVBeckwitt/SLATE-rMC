"""Common bounded multistart, nuisance-refitted profiles and correlated validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from packaging.version import Version
from scipy import __version__ as scipy_version
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import OptimizeResult, least_squares, minimize

from rasim_next.core.contracts import canonical_revision_sha256
from rasim_next.fitting.native_observations import NativeFitObservations

if TYPE_CHECKING:
    from rasim_next.fitting.native_background import NativeBackgroundProblem


@dataclass(frozen=True, slots=True)
class FitParameter:
    """A search coordinate; bounds are not measurements or uncertainty intervals."""

    name: str
    unit: str
    owner: str
    lower: float
    upper: float
    sensitivity_scale: float
    lower_kind: str = "search"
    upper_kind: str = "search"

    def __post_init__(self):
        if (
            not self.name
            or not self.unit
            or not self.owner
            or self.lower_kind not in ("physical", "search")
            or self.upper_kind not in ("physical", "search")
            or not np.all(np.isfinite([self.lower, self.upper, self.sensitivity_scale]))
            or self.lower >= self.upper
            or self.sensitivity_scale <= 0
        ):
            raise ValueError("parameters require named, owned, finite ranges and physical scales")


@dataclass(frozen=True, slots=True)
class GaussianCalibration:
    """One measured Gaussian parameter block, counted once in the joint objective.

    The caller supplies independent measurement covariance, never a ring residual
    RMS or fit-search width. Assumed constraints must be labelled as assumptions.
    """

    indices: tuple[int, ...]
    mean: np.ndarray
    covariance: np.ndarray
    artifact_sha256: str
    acquisition_id: str
    evidence_kind: str
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    parameter_owners: tuple[str, ...]
    _factor: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        indices = tuple(self.indices)
        for name in ("parameter_names", "parameter_units", "parameter_owners"):
            value = tuple(getattr(self, name))
            if len(value) != len(indices) or any(not item for item in value):
                raise ValueError("calibration must bind each coordinate by name, unit and owner")
            object.__setattr__(self, name, value)
        if (
            not indices
            or any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != len(indices)
            or not self.acquisition_id
            or len(self.artifact_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.artifact_sha256)
            or self.evidence_kind not in ("independent_measurement", "assumption")
        ):
            raise ValueError(
                "calibration requires indices, artifact SHA, acquisition and evidence kind"
            )
        for name, shape in (
            ("mean", (len(indices),)),
            ("covariance", (len(indices), len(indices))),
        ):
            raw = np.asarray(getattr(self, name))
            if raw.shape != shape or np.iscomplexobj(raw) or np.any(~np.isfinite(raw)):
                raise ValueError("calibration mean and covariance must align and be finite real")
            value = np.array(raw, dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.allclose(self.covariance, self.covariance.T, rtol=1e-13, atol=0):
            raise ValueError("calibration covariance must be symmetric")
        factor = cholesky(self.covariance, lower=True)
        factor.setflags(write=False)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "_factor", factor)

    def residual(self, values):
        return solve_triangular(self._factor, values[list(self.indices)] - self.mean, lower=True)


def score_native_prediction(
    observations,
    parameters,
    values,
    raw,
    calibration=(),
    *,
    guarded=False,
    literal_scale=None,
    background_problem: NativeBackgroundProblem | None = None,
    mixture_parameter: str | None = None,
):
    """Score a full physical vector with conditional scale/background/mixture once.

    With mixture_parameter, raw has shape (2, observation), ordered Gaussian,
    Lorentzian. The fitted fraction is reconstructed in the full result vector.
    A zero exposure retains the supplied coordinate solely as provenance and
    reports it unidentified; no fitted fraction is fabricated.
    """
    values, raw = np.array(values, dtype=float, copy=True), np.asarray(raw)
    names = tuple(p.name for p in parameters)
    lower, upper = np.array([(p.lower, p.upper) for p in parameters]).T
    width = upper - lower
    root_count = np.sqrt(int(observations.valid.sum()))
    if guarded and not observations.allow_guard_constraints:
        raise ValueError("historical guards may only be diagnostic on a training split")
    background_fit, mixture_fit = None, None
    mixture_index, mixture_bounds = None, None
    if mixture_parameter is not None:
        if mixture_parameter not in names or guarded or literal_scale is not None:
            raise ValueError(
                "mixture profiling requires a declared fraction without guards/literal scale"
            )
        mixture_index = names.index(mixture_parameter)
        mixture_bounds = lower[mixture_index], upper[mixture_index]
        if any(mixture_index in block.indices for block in calibration):
            raise ValueError("a calibration on the profiled fraction needs a coupled inner solve")
    if background_problem is not None:
        if guarded or literal_scale is not None:
            raise ValueError("background profiling cannot use guards or literal exposure")
        background_fit = (
            background_problem.profile(observations, raw, mixture_bounds=mixture_bounds)
            if mixture_parameter is not None
            else background_problem.profile(observations, raw)
        )
        if not background_fit.success:
            from rasim_next.fitting.native_background import BackgroundProfileError

            raise BackgroundProfileError(background_fit)
        scale, residual = background_fit.scale, background_fit.residual
        mixture_fit = background_fit.get("mixture_fit")
    elif mixture_parameter is not None:
        from rasim_next.fitting.native_background import profile_mosaic_amplitudes

        mixture_fit = profile_mosaic_amplitudes(
            observations, raw, np.zeros(len(observations.net_count)), eta_bounds=mixture_bounds
        )
        scale, residual = mixture_fit.scale, mixture_fit.data_residual
    elif literal_scale is None:
        scale, residual = observations.profile_scale(raw, enforce_guards=guarded)
    else:
        scalar = np.asarray(literal_scale)
        if (
            observations.exposure_index is not None
            or scalar.shape != ()
            or scalar.dtype.kind not in "fiu"
            or not np.isfinite(scalar)
            or scalar < 0
        ):
            raise ValueError("literal scale must be a finite nonnegative scalar without exposures")
        scale = float(scalar)
    component_predictions = None
    if mixture_fit is not None:
        component_predictions = np.array(raw, dtype=float, copy=True)
        if mixture_fit.lorentzian_probability is not None:
            values[mixture_index] = mixture_fit.lorentzian_probability
        eta = values[mixture_index]
        raw = (1 - eta) * component_predictions[0] + eta * component_predictions[1]
    prediction = (
        mixture_fit.prediction_count
        if mixture_fit is not None and background_fit is None
        else observations.apply_scale(raw, scale)
        if background_fit is None
        else background_fit.prediction_count
    )
    if literal_scale is not None:
        residual = observations.objective_residual(prediction)
    scores = observations.scores(prediction)
    calibration_chi_square = assumption_chi_square = 0.0
    residual_blocks = [residual]
    for block in calibration:
        r = block.residual(values)
        residual_blocks.append(r)
        if block.evidence_kind == "independent_measurement":
            calibration_chi_square += float(r @ r)
        else:
            assumption_chi_square += float(r @ r)
    point = OptimizeResult(
        parameter_values=values,
        raw_prediction=np.array(raw, dtype=float, copy=True),
        optimization_residual=np.concatenate(residual_blocks) / root_count,
        scale=scale,
        prediction_count=prediction,
        data_chi_square=scores["gls_chi_square"],
        data_objective=float(residual @ residual),
        objective_kind=observations.objective_kind,
        calibration_chi_square=calibration_chi_square,
        assumption_chi_square=assumption_chi_square,
        objective=float(residual @ residual) + calibration_chi_square + assumption_chi_square,
        scores=scores,
        start_index=-1,
        optimizer_converged=False,
    )
    if mixture_fit is not None:
        point.mixture_fit = mixture_fit
        point.raw_component_predictions = component_predictions
        point.profiled_mixture_parameter = mixture_parameter
        point.profiled_mixture_identified = mixture_fit.mixture_identified
        point.signal_prediction_count = mixture_fit.signal_prediction_count
    if background_fit is not None:
        point.background_fit = background_fit
        point.signal_prediction_count = background_fit.signal_prediction_count
        point.background_prediction_count = background_fit.background_prediction_count
        point.data_objective = background_fit.data_objective
        point.background_penalty_objective = background_fit.penalty_objective
    for kind, attribute in (
        ("search", "search_bound_parameters"),
        ("physical", "physical_boundary_parameters"),
    ):
        point[attribute] = tuple(
            name
            for i, name in enumerate(names)
            if (
                (abs(values[i] - lower[i]) < 1e-6 * width[i] and parameters[i].lower_kind == kind)
                or (
                    abs(values[i] - upper[i]) < 1e-6 * width[i] and parameters[i].upper_kind == kind
                )
            )
        )
    return point


def validate_native_search_request(
    observations: NativeFitObservations,
    parameters: tuple[FitParameter, ...],
    starts,
    *,
    fixed_values: dict[str, float] | None = None,
    calibration: tuple[GaussianCalibration, ...] = (),
    method: str = "slsqp",
    maximum_iterations: int = 50,
    maximum_function_evaluations: int = 80,
    finite_difference_step: float = 1e-4,
    enforce_historical_guards: bool = False,
    batched: bool = False,
) -> None:
    """Validate a complete search request without calling a predictor."""
    if method not in ("slsqp", "trf"):
        raise ValueError("method must be slsqp or trf")
    if type(maximum_function_evaluations) is not int or maximum_function_evaluations < 1:
        raise ValueError("maximum_function_evaluations must be a positive integer")
    if method == "trf" and enforce_historical_guards:
        raise ValueError("TRF cannot enforce historical inequalities; use SLSQP")
    if batched and method == "slsqp" and Version(scipy_version) < Version("1.16"):
        raise RuntimeError("predict_many requires SciPy >= 1.16")
    names = tuple(p.name for p in parameters)
    if not names:
        raise ValueError("at least one declared parameter is required")
    if enforce_historical_guards and not observations.allow_guard_constraints:
        raise ValueError("historical guards may only be diagnostic on a training split")
    lower, upper = np.array([(p.lower, p.upper) for p in parameters]).T
    starts = np.asarray(starts)
    fixed_values = {} if fixed_values is None else dict(fixed_values)
    if (
        len(set(names)) != len(names)
        or set(fixed_values) - set(names)
        or starts.ndim != 2
        or starts.shape[1] != len(names)
        or not len(starts)
        or np.iscomplexobj(starts)
        or np.any(~np.isfinite(starts))
        or np.any(starts < lower)
        or np.any(starts > upper)
        or type(maximum_iterations) is not int
        or maximum_iterations < 1
        or not 0 < finite_difference_step < 1
    ):
        raise ValueError("invalid parameter roster, starts, fixed values or optimization budget")
    for name, value in fixed_values.items():
        i = names.index(name)
        if not np.isfinite(value) or not lower[i] <= value <= upper[i]:
            raise ValueError(f"fixed {name} lies outside the declared search range")
    if any(max(block.indices) >= len(names) for block in calibration):
        raise ValueError("calibration indices exceed the parameter roster")
    for block in calibration:
        for i, name, unit, owner in zip(
            block.indices,
            block.parameter_names,
            block.parameter_units,
            block.parameter_owners,
            strict=True,
        ):
            if (parameters[i].name, parameters[i].unit, parameters[i].owner) != (name, unit, owner):
                raise ValueError(
                    "calibration coordinate metadata differs from the parameter roster"
                )
    identifiers = [(b.artifact_sha256, b.acquisition_id) for b in calibration]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("the same calibration block must not be counted twice")


def fit_native_parameters(
    predict: Callable[[np.ndarray], np.ndarray],
    observations: NativeFitObservations,
    parameters: tuple[FitParameter, ...],
    starts,
    *,
    predict_many: Callable[[np.ndarray], np.ndarray] | None = None,
    fixed_values: dict[str, float] | None = None,
    calibration: tuple[GaussianCalibration, ...] = (),
    method: str = "slsqp",
    previous_predictions: tuple[np.ndarray, np.ndarray] | None = None,
    maximum_iterations: int = 50,
    maximum_function_evaluations: int = 80,
    finite_difference_step: float = 1e-4,
    enforce_historical_guards: bool = False,
    callback=None,
    background_problem: NativeBackgroundProblem | None = None,
    mixture_parameter: str | None = None,
):
    """Refit every unfixed coordinate and scale from each supplied start.

    Best evaluated, feasible and converged points remain separate. Optimization
    success cannot qualify quadrature, establish identification or promote a fit.
    Data-derived historical guards are optional promotion constraints, not priors.
    Optional predict_many receives (candidate, parameter) physical coordinates and
    returns (candidate, observation) raw predictions in the same order. Only these
    predictions may run concurrently; objective history and callbacks remain serial.
    The caller owns worker isolation. Batched SLSQP requires SciPy >= 1.16.
    TRF uses the public least-squares API and a shared bounded difference batch.
    Its function budget excludes derivative probes, which evaluation_count includes.
    Callbacks must not change the predictor state within a precomputed batch.
    """
    if background_problem is not None and (
        observations.objective_kind != "gls"
        or observations.exposure_index is not None
        or observations.allow_guard_constraints
        or enforce_historical_guards
        or background_problem.ownership.shape[0] != len(observations.net_count)
    ):
        raise ValueError("background profiling requires single-exposure raw GLS without guards")
    if predict_many is not None and not callable(predict_many):
        raise TypeError("predict_many must be callable")
    validate_native_search_request(
        observations,
        parameters,
        starts,
        fixed_values=fixed_values,
        calibration=calibration,
        method=method,
        maximum_iterations=maximum_iterations,
        maximum_function_evaluations=maximum_function_evaluations,
        finite_difference_step=finite_difference_step,
        enforce_historical_guards=enforce_historical_guards,
        batched=predict_many is not None,
    )
    names = tuple(p.name for p in parameters)
    lower, upper = np.array([(p.lower, p.upper) for p in parameters]).T
    starts = np.asarray(starts)
    fixed_values = {} if fixed_values is None else dict(fixed_values)
    if mixture_parameter is not None:
        if (
            mixture_parameter not in names
            or mixture_parameter in fixed_values
            or enforce_historical_guards
            or observations.objective_kind != "gls"
            or observations.exposure_index is not None
            or observations.allow_guard_constraints
        ):
            raise ValueError(
                "profiled mixture requires a free GLS fraction and no historical guards"
            )
        i = names.index(mixture_parameter)
        if not 0 <= lower[i] < upper[i] <= 1 or any(i in b.indices for b in calibration):
            raise ValueError(
                "profiled fraction needs probability bounds and no coupled calibration"
            )
    raw_shape = (
        (2, len(observations.net_count))
        if mixture_parameter is not None
        else (len(observations.net_count),)
    )
    active = np.array(
        [
            i
            for i, name in enumerate(names)
            if name not in fixed_values and name != mixture_parameter
        ],
        dtype=int,
    )
    width = upper - lower
    best, feasible, converged = None, None, None
    runs = []
    evaluation_count = 0
    effective_starts = np.array(starts, dtype=float, copy=True)
    for name, value in fixed_values.items():
        effective_starts[:, names.index(name)] = value
    if previous_predictions is not None:
        previous_values, previous_raw = map(np.asarray, previous_predictions)
        if (
            previous_values.ndim != 2
            or previous_values.shape[1] != len(names)
            or previous_raw.shape != (len(previous_values), *raw_shape)
            or any(
                np.iscomplexobj(a) or np.any(~np.isfinite(a))
                for a in (previous_values, previous_raw)
            )
        ):
            raise ValueError(
                "previous predictions must be aligned finite full vectors and raw rows"
            )
        for values, raw in zip(previous_values, previous_raw, strict=True):
            if (
                np.any(values < lower)
                or np.any(values > upper)
                or any(values[names.index(name)] != fixed for name, fixed in fixed_values.items())
            ):
                continue
            point = score_native_prediction(
                observations,
                parameters,
                values,
                raw,
                calibration,
                guarded=enforce_historical_guards,
                background_problem=background_problem,
                mixture_parameter=mixture_parameter,
            )
            if best is None or point.objective < best.objective:
                best = point
            if point.scores["guards_pass"] and (
                feasible is None or point.objective < feasible.objective
            ):
                feasible = point
        incumbent = feasible if enforce_historical_guards else best
        if incumbent is not None:
            effective_starts = np.vstack([incumbent.parameter_values, effective_starts])
    _, distinct = np.unique(effective_starts, axis=0, return_index=True)
    for start_index in sorted(distinct.tolist()):
        base = effective_starts[start_index].copy()
        initial = (base[active] - lower[active]) / width[active]
        last_x, last_point = None, None
        prediction_buffer = {}
        raw_cache = {}
        cache_limit = 3 * len(active) + 4
        scale_reference = None

        def remember_raw(values, raw, raw_cache=raw_cache, cache_limit=cache_limit):
            if not enforce_historical_guards:
                return raw
            owned = np.array(raw, copy=True)
            owned.setflags(write=False)
            key = values.tobytes()
            raw_cache.pop(key, None)
            raw_cache[key] = owned
            if len(raw_cache) > cache_limit:
                raw_cache.pop(next(iter(raw_cache)))
            return owned

        if enforce_historical_guards:
            initial_values = base.copy()
            initial_values[active] = lower[active] + width[active] * initial
            base_raw = remember_raw(initial_values, predict(initial_values))
            free_scale, _ = observations.profile_scale(base_raw)
            if free_scale > 0:
                scale_reference = free_scale
            else:
                shape, target = observations.scalar_scale_design(base_raw)
                scale_reference = (
                    float(np.linalg.norm(target) / np.linalg.norm(shape)) if np.any(target) else 1.0
                )
            if not np.isfinite(scale_reference) or scale_reference <= 0:
                raise ValueError("the initial scale reference must be finite and positive")
            interval = observations.guard_scale_interval(base_raw)
            initial_scale = float(np.clip(free_scale, *interval)) if interval else free_scale

        def physical_values(x, base=base):
            values = base.copy()
            values[active] = lower[active] + width[active] * x[: len(active)]
            return values

        def prediction_map(
            function, points, prediction_buffer=prediction_buffer, raw_cache=raw_cache
        ):
            points = list(points)
            if not points:
                return []
            physical = [physical_values(x) for x in points]
            if enforce_historical_guards:
                rows = dict(raw_cache)
                missing = {}
                for values in physical:
                    key = values.tobytes()
                    if key not in rows and key not in missing:
                        missing[key] = values
                if missing:
                    computed = np.asarray(predict_many(np.array(list(missing.values()))))
                    if (
                        computed.shape != (len(missing), *raw_shape)
                        or np.iscomplexobj(computed)
                        or np.any(~np.isfinite(computed))
                    ):
                        raise ValueError("predict_many must return aligned finite real predictions")
                    for (key, values), row in zip(missing.items(), computed, strict=True):
                        rows[key] = remember_raw(values, row)
                raw = [rows[values.tobytes()] for values in physical]
            else:
                raw = np.asarray(predict_many(np.array(physical)))
                if (
                    raw.shape != (len(points), *raw_shape)
                    or np.iscomplexobj(raw)
                    or np.any(~np.isfinite(raw))
                ):
                    raise ValueError("predict_many must return aligned finite real predictions")
            prediction_buffer.update((x.tobytes(), row) for x, row in zip(points, raw, strict=True))
            try:
                return [function(x) for x in points]
            finally:
                prediction_buffer.clear()

        def evaluate(
            x,
            prediction_buffer=prediction_buffer,
            raw_cache=raw_cache,
            scale_reference=scale_reference,
            start_index=start_index,
        ):
            nonlocal last_x, last_point, best, feasible, evaluation_count
            if last_x is not None and np.array_equal(x, last_x):
                return last_point
            values = physical_values(x)
            raw = prediction_buffer.get(x.tobytes())
            if raw is None:
                raw = raw_cache.get(values.tobytes()) if enforce_historical_guards else None
                if raw is None:
                    raw = predict(values)
                    raw = remember_raw(values, raw)
            point = score_native_prediction(
                observations,
                parameters,
                values,
                raw,
                calibration,
                guarded=enforce_historical_guards,
                background_problem=background_problem,
                mixture_parameter=mixture_parameter,
                literal_scale=scale_reference * x[-1] if enforce_historical_guards else None,
            )
            point.start_index = start_index
            if best is None or point.objective < best.objective:
                best = point
            if point.scores["guards_pass"] and (
                feasible is None or point.objective < feasible.objective
            ):
                feasible = point
            evaluation_count += 1
            last_x, last_point = x.copy(), point
            if callback is not None:
                callback(point)
            return point

        def residual_jacobian(x):
            center = evaluate(x)
            forward, backward = 1.0 - x, x
            step = finite_difference_step
            delta = np.where(
                forward >= np.minimum(step, backward),
                np.minimum(step, forward),
                -np.minimum(step, backward),
            )
            trials = np.broadcast_to(x, (len(active), len(active))).copy()
            trials[np.arange(len(active)), np.arange(len(active))] += delta
            physical = np.array([physical_values(t) for t in trials])
            actual_steps = (
                physical[np.arange(len(active)), active] - center.parameter_values[active]
            ) / width[active]
            if np.any(~np.isfinite(actual_steps)) or np.any(actual_steps == 0):
                raise ValueError("finite-difference step is not representable")
            if np.any(physical < lower) or np.any(physical > upper):
                raise ValueError("finite-difference stencil exceeds physical bounds")
            points = (
                prediction_map(evaluate, trials)
                if predict_many is not None
                else [evaluate(t) for t in trials]
            )
            residuals = np.array([p.optimization_residual for p in points])
            return ((residuals - center.optimization_residual) / actual_steps[:, None]).T

        constraints = ()
        if enforce_historical_guards:
            constraints = (
                {
                    "type": "ineq",
                    "fun": lambda x: (
                        1 - evaluate(x).scores["guard_scores"] / observations.guard_limit
                    ),
                },
            )
        if enforce_historical_guards:
            initial = np.append(initial, initial_scale / scale_reference)
        if len(active) and method == "trf":
            literal = score_native_prediction(
                observations,
                parameters,
                base,
                predict(base),
                calibration,
                guarded=False,
                background_problem=background_problem,
                mixture_parameter=mixture_parameter,
            )
            literal.start_index = start_index
            if best is None or literal.objective < best.objective:
                best = literal
            if literal.scores["guards_pass"] and (
                feasible is None or literal.objective < feasible.objective
            ):
                feasible = literal
            evaluation_count += 1
            if callback is not None:
                callback(literal)
            result = least_squares(
                lambda x: evaluate(x).optimization_residual,
                initial,
                jac=residual_jacobian,
                bounds=(0.0, 1.0),
                method="trf",
                loss="linear",
                tr_solver="exact",
                x_scale=np.array([parameters[i].sensitivity_scale for i in active]) / width[active],
                max_nfev=maximum_function_evaluations,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
            )
        elif len(active) or enforce_historical_guards:
            options = dict(maxiter=maximum_iterations, eps=finite_difference_step, ftol=1e-9)
            if predict_many is not None:
                options["workers"] = prediction_map
            result = minimize(
                lambda x: evaluate(x).objective / int(observations.valid.sum()),
                initial,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * len(active)
                + ([(0.0, None)] if enforce_historical_guards else []),
                constraints=constraints,
                options=options,
            )
        else:
            result = OptimizeResult(
                x=np.empty(0), success=True, message="all coordinates fixed", nit=0
            )
        point = OptimizeResult(evaluate(result.x))
        point.optimizer_converged = bool(result.success)
        point.optimizer_message = str(result.message)
        point.iterations = result.get("nit")
        point.function_evaluations = result.get("nfev")
        point.jacobian_evaluations = result.get("njev")
        point.optimality = result.get("optimality")
        point.method = method
        runs.append(point)
        if (
            point.optimizer_converged
            and (not enforce_historical_guards or point.scores["guards_pass"])
            and (converged is None or point.objective < converged.objective)
        ):
            converged = point
    return OptimizeResult(
        runs=tuple(runs),
        best_evaluated=best,
        best_feasible=feasible,
        best_converged=converged,
        evaluation_count=evaluation_count,
        requested_starts=np.array(starts, dtype=float, copy=True),
        effective_starts=effective_starts,
        method=method,
        numerical_status="not_qualified",
        identification_status="not_profiled",
        guard_conditioned=enforce_historical_guards,
        fixed_parameters=tuple(fixed_values),
        profiled_parameters=() if mixture_parameter is None else (mixture_parameter,),
        nonlinear_parameter_count=len(active),
        parameters=parameters,
        calibration=calibration,
        observation_revision=observations.input_revision,
        minimum_resolved=(
            converged is not None
            and (feasible if enforce_historical_guards else best).objective
            >= converged.objective - 1e-7 * max(1.0, abs(converged.objective))
        ),
    )


def refit_native_choices(predictors, observations, parameters, starts, **search_options):
    """Refit all continuous nuisance coordinates for every explicit discrete model/N."""
    if not predictors:
        raise ValueError("at least one explicit discrete choice is required")
    if len(predictors) > 1 and search_options.get("predict_many") is not None:
        raise ValueError("bind predict_many separately for each discrete choice")
    results = {
        choice: fit_native_parameters(predict, observations, parameters, starts, **search_options)
        for choice, predict in predictors.items()
    }
    eligible = {
        key: value.best_converged
        for key, value in results.items()
        if value.best_converged is not None
    }
    return dict(
        results=results,
        best_converged_choice=(
            min(eligible, key=lambda k: eligible[k].objective) if eligible else None
        ),
        all_choices_converged=len(eligible) == len(results),
    )


def profile_native_parameter(
    predictors,
    observations,
    parameters,
    starts,
    *,
    name,
    grid,
    predict_many_by_choice=None,
    previous_predictions_by_choice=None,
    **options,
):
    """Hold one coordinate and refit every nuisance, scale and discrete choice.

    Both sweep directions use warm neighbors plus all supplied starts. Failed
    optimizations remain unresolved. Raw objective curves have no automatic
    chi-square threshold or confidence-interval interpretation, especially at
    mixture boundaries or when conditioned on historical guards.
    """
    if len(predictors) > 1 and options.get("predict_many") is not None:
        raise ValueError("bind predict_many separately for each discrete choice")
    for mapping in (predict_many_by_choice, previous_predictions_by_choice):
        if mapping is not None and set(mapping) != set(predictors):
            raise ValueError("per-choice execution must bind every predictor key exactly")
    if predict_many_by_choice is not None and options.get("predict_many") is not None:
        raise ValueError("supply one batch binding per choice")
    names = tuple(p.name for p in parameters)
    if name not in names or "fixed_values" in options:
        raise ValueError("profile one declared coordinate without additional hidden fixed values")
    grid = np.asarray(grid, dtype=float)
    if (
        grid.ndim != 1
        or len(grid) == 0
        or np.any(~np.isfinite(grid))
        or len(np.unique(grid)) != len(grid)
    ):
        raise ValueError("profile grid must contain distinct finite values")
    curves = {key: {} for key in predictors}
    orders = (np.argsort(grid), np.argsort(grid)[::-1]) if len(grid) > 1 else (np.array([0]),)
    for choice, predict in predictors.items():
        choice_options = dict(options)
        if predict_many_by_choice is not None:
            choice_options["predict_many"] = predict_many_by_choice[choice]
        if previous_predictions_by_choice is not None:
            choice_options["previous_predictions"] = previous_predictions_by_choice[choice]
        for order in orders:
            warm = np.empty((0, len(parameters)))
            for i in order:
                result = fit_native_parameters(
                    predict,
                    observations,
                    parameters,
                    np.vstack([starts, warm]),
                    fixed_values={name: float(grid[i])},
                    **choice_options,
                )
                previous = curves[choice].get(int(i))
                point = result.best_converged
                if point is not None:
                    warm = point.parameter_values[None, :]
                if previous is not None:
                    # Retain lower unfinished evaluations from either direction;
                    # a later converged but worse run cannot erase that evidence.
                    for attribute in ("best_evaluated", "best_feasible", "best_converged"):
                        points = [
                            p for p in (result[attribute], previous[attribute]) if p is not None
                        ]
                        result[attribute] = (
                            min(points, key=lambda p: p.objective) if points else None
                        )
                    result.runs = (*previous.runs, *result.runs)
                    result.evaluation_count += previous.evaluation_count
                    minimum = result.best_converged
                    admissible = (
                        result.best_feasible if result.guard_conditioned else result.best_evaluated
                    )
                    result.minimum_resolved = (
                        minimum is not None
                        and admissible.objective
                        >= minimum.objective - 1e-7 * max(1.0, abs(minimum.objective))
                    )
                curves[choice][int(i)] = result
    objective = np.full((len(predictors), len(grid)), np.nan)
    for row, results in enumerate(curves.values()):
        for i, result in results.items():
            if result.best_converged is not None:
                objective[row, i] = result.best_converged.objective
    # Never subtract an unconverged reference or disguise a lower grid optimum.
    resolved = np.array(
        [[curves[key][i].minimum_resolved for i in range(len(grid))] for key in predictors]
    )
    envelope = np.min(np.where(np.isfinite(objective), objective, np.inf), axis=0)
    envelope[~np.isfinite(envelope)] = np.nan
    return dict(
        parameter=name,
        grid=grid,
        choices=tuple(predictors),
        objective=objective,
        resolved=resolved,
        envelope_objective=envelope,
        envelope_resolved=np.all(resolved, axis=0),
        numerical_status="not_qualified",
        fits=curves,
        interval_status="raw_profile_requires_threshold_calibration",
        guard_conditioned=options.get("enforce_historical_guards", False),
    )


def native_fit_candidate(result):
    """Use one candidate for qualification, reporting and rendering."""
    if result.best_converged is not None:
        return result.best_converged
    if result.guard_conditioned and result.best_feasible is not None:
        return result.best_feasible
    return result.best_evaluated


def training_observations(observations, training_mask):
    """Use only predeclared training rows for GLS; retain old promotion diagnostics."""
    if observations.objective_kind != "gls":
        raise ValueError(
            "training splits require GLS; the historical operator uses the full roster"
        )
    mask = np.asarray(training_mask)
    if mask.dtype.kind != "b" or mask.shape != observations.valid.shape:
        raise ValueError("training mask must be a boolean vector aligned with observations")
    if np.any(mask & ~observations.valid):
        raise ValueError("a training split cannot reactivate excluded observations")
    revision = canonical_revision_sha256(
        ("definition_id", "native_training_split.v1"),
        ("observations", observations.input_revision),
        ("training_mask", mask),
    )
    return replace(observations, valid=mask, allow_guard_constraints=False, input_revision=revision)


def conditional_validation(
    observations, prediction_count, training_mask, validation_mask, *, groups=None
):
    """Conditional held-out residuals with the original cross-row covariance.

    The prediction and scale must have been fitted on training data only.
    Parameter-estimation uncertainty is not included in this noise-only score.
    """
    train = np.asarray(training_mask)
    valid = np.asarray(validation_mask)
    for mask in (train, valid):
        if (
            mask.dtype.kind != "b"
            or mask.shape != observations.valid.shape
            or not np.any(mask)
            or np.any(mask & ~observations.valid)
        ):
            raise ValueError("validation splits require nonempty valid aligned boolean masks")
    if np.any(train & valid):
        raise ValueError("training and validation rows must be disjoint")
    prediction_count = np.asarray(prediction_count)
    observations.whiten(prediction_count)
    covariance = observations.covariance_count2
    train_factor = cholesky(covariance[np.ix_(train, train)], lower=True)
    cross = covariance[np.ix_(valid, train)]
    correction = cross @ cho_solve(
        (train_factor, True), observations.net_count[train] - prediction_count[train]
    )
    conditional_mean = prediction_count[valid] + correction
    conditional_covariance = covariance[np.ix_(valid, valid)] - cross @ cho_solve(
        (train_factor, True), cross.T
    )
    factor = cholesky((conditional_covariance + conditional_covariance.T) / 2, lower=True)
    residual = solve_triangular(
        factor, observations.net_count[valid] - conditional_mean, lower=True
    )
    group_scores = []
    if groups is not None:
        coverage = np.zeros_like(valid, dtype=int)
        for name, group in groups.items():
            group = np.asarray(group)
            if (
                not isinstance(name, str)
                or not name
                or group.dtype.kind != "b"
                or group.shape != valid.shape
                or not np.any(group)
                or np.any(group & ~valid)
            ):
                raise ValueError("validation groups require named nonempty held-out row masks")
            coverage += group
            local = group[valid]
            group_covariance = conditional_covariance[np.ix_(local, local)]
            group_residual = solve_triangular(
                cholesky((group_covariance + group_covariance.T) / 2, lower=True),
                (observations.net_count[valid] - conditional_mean)[local],
                lower=True,
            )
            group_scores.append(
                dict(
                    name=name,
                    row_count=int(local.sum()),
                    chi_square=float(group_residual @ group_residual),
                )
            )
        if not np.array_equal(coverage, valid.astype(int)):
            raise ValueError("validation groups must partition all held-out rows")
    return dict(
        conditional_mean_count=conditional_mean,
        covariance_count2=conditional_covariance,
        whitened_residual=residual,
        chi_square=float(residual @ residual),
        row_count=len(residual),
        uncertainty_scope="noise_conditional_on_fitted_parameters",
        groups=group_scores,
        group_score_scope="correlated marginal group scores; do not sum",
    )
