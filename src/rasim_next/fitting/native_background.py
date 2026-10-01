"""Positive native-pixel background masses and conditional exposure profiling."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve_triangular
from scipy.optimize import OptimizeResult, least_squares
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

    def at_beta(self, observations: NativeFitObservations, raw, beta):
        """Exact nonnegative exposure and the data/penalty residual and derivative."""
        if (
            observations.objective_kind != "gls"
            or observations.exposure_index is not None
            or observations.allow_guard_constraints
            or self.ownership.shape[0] != len(observations.net_count)
            or np.any(np.asarray(self.ownership.sum(axis=1)).ravel()[observations.valid] <= 0)
        ):
            raise ValueError("background profiling requires single-exposure raw GLS without guards")
        u = observations.whiten(raw)
        denominator = float(u @ u)
        if denominator <= np.finfo(float).tiny:
            raise ValueError("background profiling requires scale-identifying diffraction")
        background, derivative = self.mass_and_jacobian(beta)
        v = observations.whiten(observations.net_count - background)
        exposure = max(0.0, float(u @ v) / denominator)
        data_residual = exposure * u - v
        data_jacobian = solve_triangular(
            observations._cholesky, derivative[observations.valid], lower=True
        )
        if exposure > 0:
            data_jacobian -= u[:, None] * ((u @ data_jacobian) / denominator)
        penalty_residual = self.regularization @ beta
        signal = exposure * np.asarray(raw)
        data_objective = float(data_residual @ data_residual)
        penalty_objective = float(penalty_residual @ penalty_residual)
        return OptimizeResult(
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

    def profile(self, observations, raw, *, callback=None):
        """Deterministic bounded TRF from beta0; nonconvergence stays explicit.

        callback receives each distinct evaluated point for caller-owned admission
        and diagnostics. Scorers reject an unsuccessful result before outer fitting.
        """
        last = None
        evaluations = 0

        def evaluate(beta):
            nonlocal last, evaluations
            if last is None or not np.array_equal(beta, last.beta):
                last = self.at_beta(observations, raw, beta)
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
