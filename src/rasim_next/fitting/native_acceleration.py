"""Reference-corrected proposals with mandatory current-engine qualification."""

from dataclasses import dataclass, replace

import numpy as np

from rasim_next.fitting.native_accuracy import compare_native_predictions
from rasim_next.fitting.native_search import fit_native_parameters, score_native_prediction


class ReferenceCorrectionRejected(ValueError):
    """A proposed numerical correction is outside its explicitly valid domain."""


@dataclass(frozen=True, slots=True)
class NativeReferenceCorrection:
    """An explicit, fixed raw-vector defect correction on a finite trust box.

    Only proposals use this object. Exact recovery, acceptance, qualification and
    final rendering must continue to call the high-resolution predictor.
    """

    reference_values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    high_reference: np.ndarray
    low_reference: np.ndarray

    def __post_init__(self):
        for name in ("reference_values", "lower", "upper", "high_reference", "low_reference"):
            raw = np.asarray(getattr(self, name))
            if np.iscomplexobj(raw) or raw.ndim != 1 or np.any(~np.isfinite(raw)):
                raise ValueError("reference correction requires finite real vectors")
            value = np.array(raw, dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if (
            self.lower.shape != self.reference_values.shape
            or self.upper.shape != self.lower.shape
            or np.any(self.lower >= self.upper)
            or np.any(self.reference_values < self.lower)
            or np.any(self.reference_values > self.upper)
            or self.high_reference.shape != self.low_reference.shape
            or np.any(self.high_reference < 0)
            or np.any(self.low_reference < 0)
        ):
            raise ValueError(
                "reference correction requires aligned nonnegative predictions and a containing trust box"
            )

    def predict(self, values, low_prediction):
        values, low = np.asarray(values), np.asarray(low_prediction)
        if (
            values.shape != self.reference_values.shape
            or np.iscomplexobj(values)
            or np.any(~np.isfinite(values))
            or np.any(values < self.lower)
            or np.any(values > self.upper)
        ):
            raise ReferenceCorrectionRejected(
                "reference-corrected candidate is outside its fixed trust box"
            )
        if (
            low.shape != self.low_reference.shape
            or np.iscomplexobj(low)
            or np.any(~np.isfinite(low))
            or np.any(low < 0)
        ):
            raise ValueError(
                "low-rule prediction must be finite, nonnegative and reference-aligned"
            )
        result = self.high_reference + (low - self.low_reference)
        if np.any(~np.isfinite(result)) or np.any(result < 0):
            raise ReferenceCorrectionRejected(
                "reference correction produced invalid or negative intensity"
            )
        result.setflags(write=False)
        return result


def reference_corrected_start(
    high_predict,
    low_predict,
    observations,
    parameters,
    start,
    *,
    trust_radii,
    numerical_tolerances,
    maximum_updates=2,
    fixed_values=None,
    calibration=(),
    method="trf",
    maximum_iterations=30,
    maximum_function_evaluations=30,
    finite_difference_step=1e-4,
    enforce_historical_guards=False,
):
    """Generate an exact-checked warm start; never report approximate convergence.

    Predictors are explicitly bound to one N and fixed high/low numerical rules.
    References stay fixed inside each public optimizer call. Only exact high-rule
    objective improvements passing the prediction/contrast gates are accepted.
    Rejected moves shrink the trust box. The caller must still run its ordinary
    final all-active high-rule fit and independent numerical qualification.
    """
    center = np.asarray(start)
    radius = np.asarray(trust_radii)
    lower = np.array([p.lower for p in parameters])
    upper = np.array([p.upper for p in parameters])
    if (
        center.shape != lower.shape
        or np.iscomplexobj(center)
        or np.any(~np.isfinite(center))
        or np.any(center < lower)
        or np.any(center > upper)
        or radius.shape != center.shape
        or np.iscomplexobj(radius)
        or np.any(~np.isfinite(radius))
        or np.any(radius <= 0)
        or type(maximum_updates) is not int
        or maximum_updates < 1
    ):
        raise ValueError(
            "reference search requires an admissible start, physical trust radii and a positive update budget"
        )
    center = np.array(center, dtype=float, copy=True)
    positions = {p.name: i for i, p in enumerate(parameters)}
    if any(
        name not in positions or value != center[positions[name]]
        for name, value in (fixed_values or {}).items()
    ):
        raise ValueError("reference start must preserve every declared fixed value")
    radius = np.array(radius, dtype=float, copy=True)
    high_center = high_predict(center)
    records = []
    for update in range(maximum_updates):
        low_center = low_predict(center)
        reference = NativeReferenceCorrection(
            center,
            np.maximum(lower, center - radius),
            np.minimum(upper, center + radius),
            high_center,
            low_center,
        )
        fallback_reasons = []

        def corrected(values, reference=reference, fallback_reasons=fallback_reasons):
            low = low_predict(values)
            try:
                return reference.predict(values, low)
            except ReferenceCorrectionRejected as error:
                fallback_reasons.append(str(error))
                return high_predict(values)

        local_parameters = tuple(
            replace(
                p,
                lower=float(lo),
                upper=float(hi),
                lower_kind=p.lower_kind if lo == p.lower else "search",
                upper_kind=p.upper_kind if hi == p.upper else "search",
            )
            for p, lo, hi in zip(parameters, reference.lower, reference.upper, strict=True)
        )
        result = fit_native_parameters(
            corrected,
            observations,
            local_parameters,
            center[None, :],
            fixed_values=fixed_values,
            calibration=calibration,
            method=method,
            maximum_iterations=maximum_iterations,
            maximum_function_evaluations=maximum_function_evaluations,
            finite_difference_step=finite_difference_step,
            enforce_historical_guards=enforce_historical_guards,
        )
        proposal = result.best_feasible if enforce_historical_guards else result.best_evaluated
        if proposal is None:
            records.append(
                dict(
                    update=update,
                    accepted=False,
                    reason="no feasible proposal",
                    fallback_reasons=fallback_reasons,
                )
            )
            radius *= 0.5
            continue
        candidate = proposal.parameter_values
        high_candidate = high_predict(candidate)
        before = score_native_prediction(
            observations,
            parameters,
            center,
            high_center,
            calibration,
            guarded=enforce_historical_guards,
        )
        after = score_native_prediction(
            observations,
            parameters,
            candidate,
            high_candidate,
            calibration,
            guarded=enforce_historical_guards,
        )
        comparison = compare_native_predictions(
            observations,
            np.array([high_center, corrected(candidate)]),
            np.array([high_center, high_candidate]),
            fixed_scale=before.scale,
            require_constrained_objective_agreement=enforce_historical_guards,
            **numerical_tolerances,
        )
        accepted = bool(
            comparison["empirical_agreement"]
            and after.objective < before.objective
            and (not enforce_historical_guards or after.scores["guards_pass"])
        )
        records.append(
            dict(
                update=update,
                reference_values=center.tolist(),
                candidate=candidate.tolist(),
                accepted=accepted,
                exact_objective_before=before.objective,
                exact_objective_after=after.objective,
                comparison=comparison,
                fallback_reasons=fallback_reasons,
            )
        )
        if accepted:
            center, high_center = np.array(candidate, copy=True), high_candidate
        else:
            radius *= 0.5
        if np.array_equal(candidate, reference.reference_values):
            break
    center.setflags(write=False)
    return center, records
