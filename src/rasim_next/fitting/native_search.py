"""Common bounded multistart, nuisance-refitted profiles and correlated validation."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from packaging.version import Version
from scipy import __version__ as scipy_version
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import OptimizeResult, minimize

from rasim_next.core.contracts import canonical_revision_sha256
from rasim_next.fitting.native_observations import NativeFitObservations


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


def fit_native_parameters(
    predict: Callable[[np.ndarray], np.ndarray],
    observations: NativeFitObservations,
    parameters: tuple[FitParameter, ...],
    starts,
    *,
    predict_many: Callable[[np.ndarray], np.ndarray] | None = None,
    fixed_values: dict[str, float] | None = None,
    calibration: tuple[GaussianCalibration, ...] = (),
    maximum_iterations: int = 50,
    finite_difference_step: float = 1e-4,
    enforce_historical_guards: bool = False,
    callback=None,
):
    """Refit every unfixed coordinate and scale from each supplied start.

    Best evaluated, feasible and converged points remain separate. Optimization
    success cannot qualify quadrature, establish identification or promote a fit.
    Data-derived historical guards are optional promotion constraints, not priors.
    Optional predict_many receives (candidate, parameter) physical coordinates and
    returns (candidate, observation) raw predictions in the same order. Only these
    predictions may run concurrently; objective history and callbacks remain serial.
    The caller owns worker isolation. This optional path requires SciPy >= 1.16.
    Callbacks must not change the predictor state within a precomputed batch.
    """
    if predict_many is not None:
        if not callable(predict_many):
            raise TypeError("predict_many must be callable")
        if Version(scipy_version) < Version("1.16"):
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
        not names
        or len(set(names)) != len(names)
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
    active = np.array([i for i, name in enumerate(names) if name not in fixed_values], dtype=int)
    width = upper - lower
    best, feasible, converged = None, None, None
    runs = []
    evaluation_count = 0
    effective_starts = np.array(starts, dtype=float, copy=True)
    for name, value in fixed_values.items():
        effective_starts[:, names.index(name)] = value
    _, distinct = np.unique(effective_starts, axis=0, return_index=True)
    for start_index in sorted(distinct.tolist()):
        base = effective_starts[start_index].copy()
        last_x, last_point = None, None
        prediction_buffer = {}

        def physical_values(x, base=base):
            values = base.copy()
            values[active] = lower[active] + width[active] * x
            return values

        def prediction_map(function, points, prediction_buffer=prediction_buffer):
            points = list(points)
            if not points:
                return []
            raw = np.asarray(predict_many(np.array([physical_values(x) for x in points])))
            if (
                raw.shape != (len(points), len(observations.net_count))
                or np.iscomplexobj(raw)
                or np.any(~np.isfinite(raw))
            ):
                raise ValueError("predict_many must return aligned finite real predictions")
            prediction_buffer.update((x.tobytes(), row) for x, row in zip(points, raw, strict=True))
            try:
                return [function(x) for x in points]
            finally:
                prediction_buffer.clear()

        def evaluate(x, prediction_buffer=prediction_buffer, start_index=start_index):
            nonlocal last_x, last_point, best, feasible, evaluation_count
            if last_x is not None and np.array_equal(x, last_x):
                return last_point
            values = physical_values(x)
            raw = prediction_buffer.get(x.tobytes())
            if raw is None:
                raw = predict(values)
            scale, residual = observations.profile_scale(
                raw, enforce_guards=enforce_historical_guards
            )
            prediction = scale * raw
            scores = observations.scores(prediction)
            calibration_chi_square = assumption_chi_square = 0.0
            for block in calibration:
                r = block.residual(values)
                if block.evidence_kind == "independent_measurement":
                    calibration_chi_square += float(r @ r)
                else:
                    assumption_chi_square += float(r @ r)
            point = OptimizeResult(
                parameter_values=values,
                scale=scale,
                prediction_count=prediction,
                data_chi_square=float(residual @ residual),
                calibration_chi_square=calibration_chi_square,
                assumption_chi_square=assumption_chi_square,
                objective=float(residual @ residual)
                + calibration_chi_square
                + assumption_chi_square,
                scores=scores,
                start_index=start_index,
                optimizer_converged=False,
            )
            for kind, attribute in (
                ("search", "search_bound_parameters"),
                ("physical", "physical_boundary_parameters"),
            ):
                point[attribute] = tuple(
                    name
                    for i, name in enumerate(names)
                    if (
                        (
                            abs(values[i] - lower[i]) < 1e-6 * width[i]
                            and parameters[i].lower_kind == kind
                        )
                        or (
                            abs(values[i] - upper[i]) < 1e-6 * width[i]
                            and parameters[i].upper_kind == kind
                        )
                    )
                )
            if best is None or point.objective < best.objective:
                best = point
            if scores["guards_pass"] and (feasible is None or point.objective < feasible.objective):
                feasible = point
            evaluation_count += 1
            last_x, last_point = x.copy(), point
            if callback is not None:
                callback(point)
            return point

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
        if len(active):
            options = dict(maxiter=maximum_iterations, eps=finite_difference_step, ftol=1e-9)
            if predict_many is not None:
                options["workers"] = prediction_map
            result = minimize(
                lambda x: evaluate(x).objective / int(observations.valid.sum()),
                (base[active] - lower[active]) / width[active],
                method="SLSQP",
                bounds=[(0.0, 1.0)] * len(active),
                constraints=constraints,
                options=options,
            )
        else:
            result = OptimizeResult(
                x=np.empty(0), success=True, message="all coordinates fixed", nit=0
            )
        point = OptimizeResult(evaluate(result.x))
        point.optimizer_converged = bool(result.success)
        point.optimizer_message, point.iterations = str(result.message), int(result.nit)
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
        numerical_status="not_qualified",
        identification_status="not_profiled",
        guard_conditioned=enforce_historical_guards,
        fixed_parameters=tuple(fixed_values),
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
    predictors, observations, parameters, starts, *, name, grid, **options
):
    """Hold one coordinate and refit every nuisance, scale and discrete choice.

    Both sweep directions use warm neighbors plus all supplied starts. Failed
    optimizations remain unresolved. Raw objective curves have no automatic
    chi-square threshold or confidence-interval interpretation, especially at
    mixture boundaries or when conditioned on historical guards.
    """
    if len(predictors) > 1 and options.get("predict_many") is not None:
        raise ValueError("bind predict_many separately for each discrete choice")
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
        for order in orders:
            warm = np.empty((0, len(parameters)))
            for i in order:
                result = fit_native_parameters(
                    predict,
                    observations,
                    parameters,
                    np.vstack([starts, warm]),
                    fixed_values={name: float(grid[i])},
                    **options,
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


def training_observations(observations, training_mask):
    """Use only predeclared training rows for GLS; retain old promotion diagnostics."""
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
