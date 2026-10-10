"""Positive native-pixel background masses and conditional exposure profiling."""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve_triangular
from scipy.optimize import OptimizeResult, least_squares, minimize, nnls
from scipy.sparse import csc_matrix

from rasim_next.fitting.native_observations import NativeFitObservations


def native_background_design(column_px, row_px):
    """The 44-column broad background basis at actual native pixel centers."""
    column, row = np.asarray(column_px), np.asarray(row_px)
    if (
        column.ndim != 1
        or column.shape != row.shape
        or any(np.iscomplexobj(a) or np.any(~np.isfinite(a)) for a in (column, row))
    ):
        raise ValueError("background coordinates must be aligned finite real vectors")
    dx, dy = (column - 1453.12) / 1500, (row - 1596.422) / 1500
    radius = np.minimum(np.hypot(column - 1453.12, row - 1596.422), 2500)
    knots = np.r_[np.zeros(3), np.linspace(0, 2500, 10), np.full(3, 2500)]
    pieces = [
        BSpline.design_matrix(radius, knots, 3).toarray(),
        dx[:, None],
        dy[:, None],
        (dx * dx - dy * dy)[:, None],
        (2 * dx * dy)[:, None],
    ]
    for cy in (400.0, 1000.0, 1600.0, 2400.0):
        for cx in (400.0, 800.0, 1200.0, 1600.0, 2000.0, 2400.0, 2800.0):
            pieces.append(np.exp(-0.5 * ((column - cx) ** 2 + (row - cy) ** 2) / 400.0**2)[:, None])
    return np.column_stack(pieces)


class BackgroundProfileError(RuntimeError):
    """An unresolved inner solve, retaining its result for explicit recovery."""

    def __init__(self, result):
        super().__init__(f"background profile unresolved: {result.message}")
        self.result = result


def _profile_exposure(signal, target):
    denominator = float(signal @ signal)
    if denominator <= np.finfo(float).tiny:
        raise ValueError("background profiling requires scale-identifying diffraction")
    exposure = max(0.0, float(signal @ target) / denominator)
    return exposure, exposure * signal - target, denominator


def profile_mosaic_amplitudes(observations, components, background, *, eta_bounds=(0.0, 1.0)):
    """Exact conditional GLS amplitudes for complete normalized G/L columns.

    The two extreme eta rays span the admissible nonnegative amplitude cone.
    NNLS on those rays also enforces narrower eta bounds without clipping a fit.
    No background is fitted here. A zero exposure has no fitted eta; singular
    component designs retain an explicit identification limitation.
    """
    raw, bounds = np.asarray(components), np.asarray(eta_bounds)
    if (
        observations.objective_kind != "gls"
        or observations.exposure_index is not None
        or observations.allow_guard_constraints
        or raw.shape != (2, len(observations.net_count))
        or np.iscomplexobj(raw)
        or np.any(~np.isfinite(raw))
        or np.any(raw < 0)
        or bounds.shape != (2,)
        or np.iscomplexobj(bounds)
        or np.any(~np.isfinite(bounds))
        or not 0 <= bounds[0] <= bounds[1] <= 1
    ):
        raise ValueError(
            "mixture profiling requires two nonnegative raw GLS columns and eta bounds"
        )
    background = np.asarray(background)
    if (
        background.shape != observations.net_count.shape
        or np.iscomplexobj(background)
        or np.any(~np.isfinite(background))
    ):
        raise ValueError("background must be an aligned finite real count vector")
    target = observations.whiten(observations.net_count - background)
    fractions = bounds[:1] if bounds[0] == bounds[1] else bounds
    rays = np.array([1 - fractions, fractions])
    whitened = solve_triangular(observations._cholesky, raw[:, observations.valid].T, lower=True)
    design = whitened @ rays
    norms = np.linalg.norm(design, axis=0)
    if np.any(~np.isfinite(norms)):
        raise ValueError("component design norms are nonfinite")
    normalization = np.where(norms > 0, norms, 1.0)
    scaled = design / normalization
    try:
        scaled_amplitudes, _ = nnls(scaled, target, maxiter=20)
    except RuntimeError as error:
        raise BackgroundProfileError(
            OptimizeResult(success=False, message=str(error), raw_component_predictions=raw.copy())
        ) from error
    ray_amplitudes = scaled_amplitudes / normalization
    amplitudes = rays @ ray_amplitudes
    exposure = float(amplitudes.sum())
    eta = float(amplitudes[1] / exposure) if exposure > 0 else None
    signal = amplitudes @ raw
    residual = scaled @ scaled_amplitudes - target
    gradient = scaled.T @ residual
    active = scaled_amplitudes > 0
    kkt_error = float(np.max(np.where(active, abs(gradient), np.maximum(-gradient, 0))))
    kkt_relative = kkt_error / max(1.0, float(np.linalg.norm(target)))
    rank = int(np.linalg.matrix_rank(whitened))
    _, singular_values, right = np.linalg.svd(
        scaled, full_matrices=scaled.shape[0] < scaled.shape[1]
    )
    rank_tolerance = np.finfo(float).eps * max(scaled.shape)
    admitted_rank = int(np.sum(singular_values > rank_tolerance * singular_values[0]))
    null_rays = right[admitted_rank:].T / normalization[:, None]
    # Only directions with a feasible sign at zero coefficients can preserve
    # the fitted signal in the admitted cone. Equal columns preserve exposure;
    # unequal proportional columns generally change it, despite positive rank.
    feasible = [
        d
        for d in null_rays.T
        if np.all(d[~active] >= -rank_tolerance * np.linalg.norm(d))
        or np.all(d[~active] <= rank_tolerance * np.linalg.norm(d))
    ]
    null_amplitudes = rays @ np.array(feasible).reshape(-1, len(fractions)).T
    null_norm = np.linalg.norm(null_amplitudes, axis=0)
    exposure_unique = np.all(abs(null_amplitudes.sum(axis=0)) <= rank_tolerance * null_norm)
    eta_change = null_amplitudes[1] - (eta or 0.0) * null_amplitudes.sum(axis=0)
    eta_unique = np.all(abs(eta_change) <= rank_tolerance * null_norm)
    result = OptimizeResult(
        scale=exposure,
        lorentzian_probability=eta,
        component_amplitudes=amplitudes,
        ray_amplitudes=ray_amplitudes,
        component_design_rank=rank,
        admitted_design_rank=admitted_rank,
        amplitude_null_directions=null_amplitudes,
        identification_kind="conditional amplitude cone only; no joint physical identification",
        exposure_identified=bool(exposure_unique and exposure > 0),
        mixture_identified=bool(eta_unique and exposure > 0 and bounds[0] != bounds[1]),
        inactive_components=tuple(np.flatnonzero(amplitudes == 0).tolist()),
        active_design=scaled[:, active],
        data_residual=residual,
        signal_prediction_count=signal,
        background_prediction_count=np.array(background, dtype=float, copy=True),
        prediction_count=signal + background,
        data_objective=float(residual @ residual),
        amplitude_kkt_relative_error=kkt_relative,
        success=kkt_relative <= 1e-10,
        message="conditional amplitude KKT satisfied"
        if kkt_relative <= 1e-10
        else "conditional amplitude KKT failed",
    )
    if not result.success:
        raise BackgroundProfileError(result)
    return result


@dataclass(frozen=True, slots=True)
class NativeLinearBackgroundProblem:
    """Nonnegative linear background masses with one jointly profiled exposure.

    Columns must already use the observation's exact native memberships. A
    monotone radial field is represented by cumulative piecewise-linear columns,
    whose nonnegative coefficients are successive density drops. This class owns
    the cone solve, not the caller's spatial basis or support assumptions.
    """

    design_count: np.ndarray
    maximum_iterations: int = 10000
    _observations: NativeFitObservations | None = field(default=None, init=False, repr=False)
    _white_background: np.ndarray | None = field(default=None, init=False, repr=False)
    _background_norms: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        raw = np.asarray(self.design_count)
        if (
            raw.ndim != 2
            or min(raw.shape) < 1
            or np.iscomplexobj(raw)
            or np.any(~np.isfinite(raw))
            or np.any(raw < 0)
            or type(self.maximum_iterations) is not int
            or self.maximum_iterations < 1
        ):
            raise ValueError("linear background needs a finite nonnegative mass design and budget")
        design = np.array(raw, dtype=float, copy=True)
        design.setflags(write=False)
        object.__setattr__(self, "design_count", design)

    @property
    def observation_count(self):
        return self.design_count.shape[0]

    def prepare(self, observations):
        """Bind fixed covariance work explicitly; candidate NNLS remains fresh."""
        if self._observations is observations:
            return self
        if (
            observations.objective_kind != "gls"
            or observations.exposure_index is not None
            or observations.allow_guard_constraints
            or len(observations.net_count) != self.observation_count
        ):
            raise ValueError("linear background requires single-exposure raw GLS without guards")
        white = solve_triangular(
            observations._cholesky, self.design_count[observations.valid], lower=True
        )
        norms = np.linalg.norm(white, axis=0)
        if np.any(~np.isfinite(white)) or np.any(~np.isfinite(norms)):
            raise ValueError("linear background whitening or column norms overflowed")
        norms = np.where(norms > np.finfo(float).tiny, norms, 1.0)
        white /= norms
        white.setflags(write=False)
        norms.setflags(write=False)
        prepared = replace(self)
        object.__setattr__(prepared, "_observations", observations)
        object.__setattr__(prepared, "_white_background", white)
        object.__setattr__(prepared, "_background_norms", norms)
        return prepared

    def profile(self, observations, raw, *, callback=None, mixture_bounds=None):
        if mixture_bounds is not None:
            raise ValueError("linear background does not profile mosaic mixtures")
        if self._observations is None:
            return self.prepare(observations).profile(observations, raw, callback=callback)
        if self._observations is not observations:
            raise ValueError("prepared background belongs to different observations")
        if (
            observations.objective_kind != "gls"
            or observations.exposure_index is not None
            or observations.allow_guard_constraints
            or len(observations.net_count) != self.observation_count
        ):
            raise ValueError("linear background requires single-exposure raw GLS without guards")
        raw = np.asarray(raw)
        if (
            raw.shape != (self.observation_count,)
            or np.iscomplexobj(raw)
            or np.any(~np.isfinite(raw))
            or np.any(raw < 0)
        ):
            raise ValueError("linear background requires an aligned nonnegative physical signal")
        white = observations.whiten(raw)
        norms = np.r_[np.linalg.norm(white), self._background_norms]
        if np.any(~np.isfinite(white)) or np.any(~np.isfinite(norms)):
            raise ValueError("linear background whitening or column norms overflowed")
        if norms[0] <= np.finfo(float).tiny:
            raise ValueError("linear background requires scale-identifying diffraction")
        norms = np.where(norms > np.finfo(float).tiny, norms, 1.0)
        normalized = np.column_stack((white / norms[0], self._white_background))
        target = observations._whitened_net
        try:
            coefficients, _ = nnls(
                normalized,
                target,
                maxiter=self.maximum_iterations,
            )
        except RuntimeError as exc:
            raise BackgroundProfileError(OptimizeResult(success=False, message=str(exc))) from exc
        gradient = normalized.T @ (normalized @ coefficients - target)
        active = coefficients > 0
        kkt_error = float(np.max(np.where(active, abs(gradient), np.maximum(-gradient, 0))))
        kkt_relative = kkt_error / max(1.0, float(np.linalg.norm(target)))
        rank = int(np.linalg.matrix_rank(normalized))
        coefficients /= norms
        signal = coefficients[0] * raw
        background = self.design_count @ coefficients[1:]
        residual = observations.whiten(signal + background - observations.net_count)
        objective = float(residual @ residual)
        if not np.isfinite(objective) or any(
            np.any(~np.isfinite(v)) for v in (coefficients, signal, background, residual)
        ):
            raise ValueError("linear background solve produced nonfinite results")
        result = OptimizeResult(
            scale=float(coefficients[0]),
            coefficients=coefficients[1:],
            signal_prediction_count=signal,
            background_prediction_count=background,
            prediction_count=signal + background,
            residual=residual,
            data_residual=residual,
            penalty_residual=np.empty(0),
            data_objective=objective,
            penalty_objective=0.0,
            objective=objective,
            success=True,
            message="nonnegative linear least-squares KKT satisfied"
            if kkt_relative <= 1e-10
            else "nonnegative linear least-squares KKT failed",
            amplitude_kkt_relative_error=kkt_relative,
            design_rank=rank,
            conditional_coefficients_unique=True if rank == normalized.shape[1] else None,
            identification_kind="conditional linear design only; no joint physical identification",
        )
        result.success = kkt_relative <= 1e-10
        if not result.success:
            raise BackgroundProfileError(result)
        if callback is not None:
            callback(result)
        return result


@dataclass(frozen=True, slots=True)
class NativeBackgroundProblem:
    """One fixed pixel design, ownership W, absolute penalty R and immutable start.

    Observations supplied to this path contain RAW measured counts and the declared
    count covariance, including any explicitly justified discrepancy covariance.
    Pixel ownership may overlap; its full covariance must already be propagated
    by the caller. No background uncertainty is added here.
    """

    pixel_design: np.ndarray
    ownership: csc_matrix
    beta0: np.ndarray
    regularization: np.ndarray
    maximum_function_evaluations: int = 150
    tolerance: float = 1e-8

    def __post_init__(self):
        for name in ("pixel_design", "beta0", "regularization"):
            raw = np.asarray(getattr(self, name))
            if np.iscomplexobj(raw) or np.any(~np.isfinite(raw)):
                raise ValueError("background arrays must be finite real")
            value = np.array(raw, dtype=float, copy=True, order="C")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.iscomplexobj(self.ownership):
            raise ValueError("background ownership must be finite real and nonnegative")
        operator = csc_matrix(self.ownership, dtype=float, copy=True)
        if np.any(~np.isfinite(operator.data)) or np.any(operator.data < 0):
            raise ValueError("background ownership must be finite real and nonnegative")
        operator.sum_duplicates()
        operator.sort_indices()
        if (
            self.pixel_design.ndim != 2
            or self.pixel_design.shape[1] != 44
            or self.beta0.shape != (44,)
            or self.regularization.ndim != 2
            or self.regularization.shape[1] != 44
            or operator.shape[1] != len(self.pixel_design)
            or not operator.shape[0]
            or not np.any(operator.data > 0)
            or np.any(np.abs(self.beta0) > 15)
            or type(self.maximum_function_evaluations) is not int
            or self.maximum_function_evaluations < 1
            or not np.isfinite(self.tolerance)
            or not np.finfo(float).eps < self.tolerance < 1
        ):
            raise ValueError("background design, ownership, penalty, start and budget must align")
        for array in (operator.data, operator.indices, operator.indptr):
            array.setflags(write=False)
        object.__setattr__(self, "ownership", operator)

    @property
    def observation_count(self):
        return self.ownership.shape[0]

    def mass_and_jacobian(self, beta):
        """Sum exp(X beta) over exact pixel footprints, never exponentiate averages."""
        beta = np.asarray(beta)
        if (
            beta.shape != (44,)
            or np.iscomplexobj(beta)
            or np.any(~np.isfinite(beta))
            or np.any(np.abs(beta) > 15)
        ):
            raise ValueError("background coefficients must be finite real inside [-15,15]")
        density = np.exp(self.pixel_design @ beta)
        if np.any(~np.isfinite(density)):
            raise ValueError("background pixel means are nonfinite")
        mass = np.asarray(self.ownership @ density)
        jacobian = np.zeros((self.ownership.shape[0], 44))
        # Bound temporary weighted-design storage independently of detector support.
        for first in range(0, len(density), 16384):
            last = first + 16384
            jacobian += self.ownership[:, first:last] @ (
                density[first:last, None] * self.pixel_design[first:last]
            )
        return mass, jacobian

    def at_beta(self, observations: NativeFitObservations, raw, beta, *, mixture_bounds=None):
        """Exact nonnegative exposure and the data/penalty residual and derivative."""
        if (
            observations.objective_kind != "gls"
            or observations.exposure_index is not None
            or observations.allow_guard_constraints
            or self.ownership.shape[0] != len(observations.net_count)
            or np.any(np.asarray(self.ownership.sum(axis=1)).ravel()[observations.valid] <= 0)
        ):
            raise ValueError("background profiling requires single-exposure raw GLS without guards")
        background, derivative = self.mass_and_jacobian(beta)
        data_jacobian = solve_triangular(
            observations._cholesky, derivative[observations.valid], lower=True
        )
        mixture = None
        if mixture_bounds is not None:
            mixture = profile_mosaic_amplitudes(
                observations, raw, background, eta_bounds=mixture_bounds
            )
            exposure = mixture.scale
            signal, data_residual = mixture.signal_prediction_count, mixture.data_residual
            active = mixture.active_design
            if active.shape[1]:
                data_jacobian -= active @ np.linalg.lstsq(active, data_jacobian, rcond=None)[0]
        else:
            u = observations.whiten(raw)
            v = observations.whiten(observations.net_count - background)
            exposure, data_residual, denominator = _profile_exposure(u, v)
            signal = exposure * np.asarray(raw)
            if exposure > 0:
                data_jacobian -= u[:, None] * ((u @ data_jacobian) / denominator)
        penalty_residual = self.regularization @ beta
        data_objective = float(data_residual @ data_residual)
        penalty_objective = float(penalty_residual @ penalty_residual)
        result = OptimizeResult(
            beta=np.array(beta, copy=True),
            scale=exposure,
            signal_prediction_count=signal,
            background_prediction_count=background,
            prediction_count=signal + background,
            data_residual=data_residual,
            penalty_residual=penalty_residual,
            residual=np.r_[data_residual, penalty_residual],
            jacobian=np.vstack([data_jacobian, self.regularization]),
            data_objective=data_objective,
            penalty_objective=penalty_objective,
            objective=data_objective + penalty_objective,
        )

        if mixture is not None:
            result.mixture_fit = mixture
        return result

    def profile(self, observations, raw, *, callback=None, mixture_bounds=None):
        """Deterministic bounded TRF from beta0; nonconvergence stays explicit.

        callback receives each distinct evaluated point for caller-owned admission
        and diagnostics. Scorers reject an unsuccessful result before outer fitting.
        """
        last = None
        evaluations = 0

        def evaluate(beta):
            nonlocal last, evaluations
            if last is None or not np.array_equal(beta, last.beta):
                last = self.at_beta(observations, raw, beta, mixture_bounds=mixture_bounds)
                evaluations += 1
                if callback is not None:
                    callback(last)
            return last

        result = least_squares(
            lambda beta: evaluate(beta).residual,
            self.beta0,
            jac=lambda beta: evaluate(beta).jacobian,
            method="trf",
            loss="linear",
            bounds=(-15.0, 15.0),
            max_nfev=self.maximum_function_evaluations,
            ftol=self.tolerance,
            xtol=self.tolerance,
            gtol=self.tolerance,
        )
        point = OptimizeResult(evaluate(result.x))
        point.success = bool(result.success)
        point.status = int(result.status)
        point.message = str(result.message)
        point.nfev, point.njev = result.nfev, result.njev
        point.evaluations = evaluations
        point.optimality = result.optimality
        point.active_mask = np.array(result.active_mask, copy=True)
        return point


class _HullCertificateConverged(Exception):
    """Internal termination at an accepted, feasible, certified optimizer iterate."""

    def __init__(self, point):
        super().__init__("convex objective-gap certificate satisfied")
        self.point = point


@dataclass(frozen=True, slots=True)
class NativeBackgroundHull:
    """Fixed count columns with one global simplex vector and no penalty.

    Each column must already integrate a declared native-pixel field over the
    observation footprints. Membership in this finite hull is a conditional
    model assumption, not calibrated background uncertainty. This solver admits
    one fixed scalar diffraction shape; it does not profile mosaic components.
    """

    columns_count: np.ndarray
    weights0: np.ndarray
    maximum_iterations: int = 300
    maximum_function_evaluations: int = 600
    tolerance: float = 1e-8

    def __post_init__(self):
        for name in ("columns_count", "weights0"):
            raw = np.asarray(getattr(self, name))
            if np.iscomplexobj(raw) or np.any(~np.isfinite(raw)) or np.any(raw < 0):
                raise ValueError("background hull arrays must be finite real and nonnegative")
            value = np.array(raw, dtype=float, copy=True, order="C")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if (
            self.columns_count.ndim != 2
            or not all(self.columns_count.shape)
            or self.weights0.shape != (self.columns_count.shape[1],)
            or abs(float(self.weights0.sum()) - 1) > 64 * np.finfo(float).eps
            or type(self.maximum_iterations) is not int
            or self.maximum_iterations < 1
            or type(self.maximum_function_evaluations) is not int
            or self.maximum_function_evaluations < 1
            or not np.isfinite(self.tolerance)
            or not np.finfo(float).eps < self.tolerance < 1
        ):
            raise ValueError("background hull columns, simplex start and finite budgets must align")

    def profile(self, observations, raw, *, callback=None):
        """Profile exposure and solve the convex simplex problem from weights0.

        No returned coefficients are clipped or normalized. An accepted iterate
        can terminate on primal feasibility and the normalized convex objective-gap
        certificate. Otherwise optimizer convergence and the same gates are required.
        A distinct-evaluation cap raises
        BackgroundProfileError with the last evaluated point. Rank diagnostics
        concern this fixed shape only; deficient rank does not prove nonuniqueness
        under the active inequalities.
        """
        raw = np.asarray(raw)
        if (
            observations.objective_kind != "gls"
            or observations.exposure_index is not None
            or observations.allow_guard_constraints
            or self.columns_count.shape[0] != len(observations.net_count)
            or raw.shape != observations.net_count.shape
            or np.iscomplexobj(raw)
            or np.any(~np.isfinite(raw))
            or np.any(raw < 0)
        ):
            raise ValueError("background hull requires single-exposure raw GLS without guards")
        signal = observations.whiten(raw)
        design = solve_triangular(
            observations._cholesky, self.columns_count[observations.valid], lower=True
        )
        target = observations.whiten(observations.net_count)
        if any(np.any(~np.isfinite(a)) for a in (signal, design, target)) or not np.isfinite(
            signal @ signal
        ):
            raise ValueError("whitened background hull design is nonfinite")
        _, initial_residual, _ = _profile_exposure(signal, target - design @ self.weights0)
        initial_objective = float(initial_residual @ initial_residual)
        if not np.isfinite(initial_objective):
            raise ValueError("background hull initial objective is nonfinite")
        objective_scale = max(1.0, initial_objective)
        last = None
        evaluations = 0

        def evaluate(weights):
            nonlocal last, evaluations
            if last is None or not np.array_equal(weights, last.weights):
                if evaluations == self.maximum_function_evaluations:
                    raise BackgroundProfileError(
                        OptimizeResult(
                            last,
                            success=False,
                            status=-1,
                            evaluations=evaluations,
                            message="background hull distinct-evaluation budget exhausted",
                        )
                    )
                exposure, residual, _ = _profile_exposure(signal, target - design @ weights)
                gradient = 2 * design.T @ residual
                background = self.columns_count @ weights
                predicted_signal = exposure * raw
                objective = float(residual @ residual)
                gap = float(gradient @ weights - gradient.min())
                if any(
                    np.any(~np.isfinite(a))
                    for a in (
                        weights,
                        exposure,
                        residual,
                        gradient,
                        background,
                        predicted_signal,
                        objective,
                        gap,
                    )
                ):
                    raise ValueError("background hull evaluated a nonfinite point")
                feasibility = max(abs(float(weights.sum()) - 1), -float(weights.min()), 0.0)
                last = OptimizeResult(
                    weights=np.array(weights, copy=True),
                    scale=exposure,
                    signal_prediction_count=predicted_signal,
                    background_prediction_count=background,
                    prediction_count=predicted_signal + background,
                    data_residual=residual,
                    penalty_residual=np.empty(0),
                    residual=residual,
                    data_objective=objective,
                    penalty_objective=0.0,
                    objective=objective,
                    gradient=gradient,
                    objective_scale=objective_scale,
                    simplex_feasibility_error=feasibility,
                    convex_objective_gap_bound=max(0.0, gap),
                    normalized_objective_gap=max(0.0, gap) / objective_scale,
                )
                evaluations += 1
                if callback is not None:
                    callback(last)
            return last

        objective_calls = gradient_calls = accepted_iterations = 0

        def objective(weights):
            nonlocal objective_calls
            objective_calls += 1
            return evaluate(weights).objective / objective_scale

        def jacobian(weights):
            nonlocal gradient_calls
            gradient_calls += 1
            return evaluate(weights).gradient / objective_scale

        def accepted_iterate(weights):
            nonlocal accepted_iterations
            accepted_iterations += 1
            point = evaluate(weights)
            if (
                point.simplex_feasibility_error <= self.tolerance
                and point.normalized_objective_gap <= self.tolerance
            ):
                raise _HullCertificateConverged(OptimizeResult(point))

        termination_kind = "optimizer"
        if len(self.weights0) == 1:
            termination_kind = "fixed_single_column"
            result = OptimizeResult(
                x=self.weights0,
                success=True,
                status=0,
                message="fixed single-column hull",
                nfev=1,
                njev=1,
                nit=0,
            )
        else:
            try:
                result = minimize(
                    objective,
                    self.weights0,
                    jac=jacobian,
                    callback=accepted_iterate,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * len(self.weights0),
                    constraints={
                        "type": "eq",
                        "fun": lambda weights: weights.sum() - 1,
                        "jac": lambda weights: np.ones_like(weights),
                    },
                    options={
                        "maxiter": self.maximum_iterations,
                        "ftol": max(np.finfo(float).eps, self.tolerance**2),
                    },
                )
            except _HullCertificateConverged as converged:
                termination_kind = "convex_certificate"
                result = OptimizeResult(
                    x=converged.point.weights,
                    success=True,
                    status=0,
                    message="convex objective-gap certificate at accepted SLSQP iterate",
                    nfev=objective_calls,
                    njev=gradient_calls,
                    nit=accepted_iterations,
                )
        point = OptimizeResult(evaluate(result.x))
        point.success = bool(
            result.success
            and point.simplex_feasibility_error <= self.tolerance
            and point.normalized_objective_gap <= self.tolerance
        )
        point.status, point.message = int(result.status), str(result.message)
        point.termination_kind = termination_kind
        point.optimizer_success = bool(result.success) if termination_kind == "optimizer" else None
        point.optimizer_status = int(result.status) if termination_kind == "optimizer" else None
        if result.success and not point.success:
            point.message = "background hull feasibility or convex objective-gap check failed"
        point.nfev, point.njev, point.nit = result.nfev, result.njev, result.nit
        point.evaluations = evaluations
        point.optimality = point.normalized_objective_gap
        point.active_mask = np.where(point.weights == 0, -1, 0)
        tangent = np.column_stack([signal, design[:, 1:] - design[:, :1]])
        norms = np.linalg.norm(tangent, axis=0)
        normalized = tangent / np.where(norms > 0, norms, 1.0)
        point.tangent_singular_values = np.linalg.svd(normalized, compute_uv=False)
        point.tangent_rank = int(np.linalg.matrix_rank(normalized))
        point.conditional_coefficients_unique = (
            True if point.tangent_rank == tangent.shape[1] else None
        )
        point.identification_kind = (
            "fixed-shape affine rank only; deficient rank needs feasible-null analysis; "
            "no joint physical identification"
        )
        return point
