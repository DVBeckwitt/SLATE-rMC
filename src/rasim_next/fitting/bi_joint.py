"""Bounded Bi atomic/cell and morphology fitting on a frozen native observable."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from time import perf_counter

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from painted_ewald import MosaicParameters
from rasim_next.fitting.bi_native import (
    BI_CELL_SITE_PARAMETER_NAMES,
    BiCellSiteParameters,
    BiNativeStructureModel,
)
from rasim_next.fitting.native_observations import NativeFitObservations
from rasim_next.fitting.native_structure import native_stitch_records
from rasim_next.pipeline.conditional_detector import NativeMosaicCache
from rasim_next.pipeline.source_spatial import NativeSpatialRegionProjection

BI_JOINT_PARAMETER_NAMES = (
    *BI_CELL_SITE_PARAMETER_NAMES,
    "gaussian_sigma_rad",
    "lorentzian_half_width_rad",
    "lorentzian_probability",
    "surface_fraction_0",
    "surface_1_share_of_remainder",
    "extra_film_thickness_A",
    "top_roughness_A",
    "bottom_roughness_A",
)


@dataclass(frozen=True, slots=True)
class BiJointCandidate:
    """21 continuous coordinates and an independently selected integer repeat count.

    Film thickness = N*c + extra thickness. Surface fractions are
    (s0, (1-s0)*s1, (1-s0)*(1-s1)); these coordinates preserve physical bounds.
    """

    values: np.ndarray
    coherent_repeats: int

    def __post_init__(self) -> None:
        raw = np.asarray(self.values)
        if raw.shape != (21,) or np.iscomplexobj(raw) or np.any(~np.isfinite(raw)):
            raise ValueError("Bi joint candidates require 21 finite real coordinates")
        value = np.array(raw, dtype=float, copy=True)
        BiCellSiteParameters.from_array(value[:13])
        MosaicParameters(*value[13:16])
        if (
            type(self.coherent_repeats) is not int
            or self.coherent_repeats < 1
            or np.any(value[16:18] < 0)
            or np.any(value[16:18] > 1)
            or np.any(value[18:] < 0)
        ):
            raise ValueError("invalid repeat count, surface fractions, thickness or roughness")
        value.setflags(write=False)
        object.__setattr__(self, "values", value)

    @property
    def film_thickness_A(self) -> float:
        return float(self.coherent_repeats * self.values[1] + self.values[18])

    @property
    def surface_fractions(self) -> tuple[float, float, float]:
        first, second = self.values[16:18]
        return float(first), float((1 - first) * second), float((1 - first) * (1 - second))


@dataclass(frozen=True, slots=True)
class BiNativeFitEvaluator:
    """Explicit execution resource; holds at most two immutable physical responses.

    Only equal material AND reciprocal basis permit reuse. Proposal mosaic,
    source, mounting, rod roster, spatial integration and observations are fixed
    by this owner. Every uncached trial recomputes the signed SF and empirical stitch.
    """

    model: BiNativeStructureModel
    observations: NativeFitObservations
    proposal_mosaic: MosaicParameters
    worker_count: int = 1
    compile_count: int = field(default=0, init=False)
    evaluation_count: int = field(default=0, init=False)
    compile_seconds: float = field(default=0.0, init=False)
    _responses: OrderedDict[tuple[str, bytes], NativeMosaicCache] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _predictions: OrderedDict[tuple[int, bytes], np.ndarray] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _spatial_projection: NativeSpatialRegionProjection | None = field(
        default=None, init=False, repr=False
    )

    def predict(self, candidate: BiJointCandidate) -> np.ndarray:
        # SLSQP asks for the same finite-difference points separately for its
        # objective and guards. Retain only 64 small count vectors, not another
        # set of detector responses, to avoid repeating identical physical work.
        candidate_key = candidate.coherent_repeats, candidate.values.tobytes()
        if candidate_key in self._predictions:
            self._predictions.move_to_end(candidate_key)
            return self._predictions[candidate_key]
        physics = self.model.bind(BiCellSiteParameters.from_array(candidate.values[:13]))
        if len(physics.structure.crystals) != 3 or physics.specular_stitch_stack is None:
            raise ValueError(
                "joint Bi fitting requires three declared surfaces and the local composite"
            )
        stack = replace(
            physics.specular_stitch_stack,
            top_roughness_A=float(candidate.values[19]),
            bottom_roughness_A=float(candidate.values[20]),
        )
        detector = physics.detector(
            mosaic=self.proposal_mosaic,
            coherent_repeats=candidate.coherent_repeats,
            film_thickness_A=candidate.film_thickness_A,
            surface_fractions=candidate.surface_fractions,
            phase_fractions=(1.0,),
            fault_parameters={},
        )
        key = (physics.material.material_revision, physics.reciprocal_basis_Ainv.tobytes())
        if key in self._responses:
            mosaic_cache = self._responses.pop(key)
        else:
            start = perf_counter()
            if self._spatial_projection is None:
                object.__setattr__(
                    self,
                    "_spatial_projection",
                    NativeSpatialRegionProjection(self.observations.projection),
                )
            response = detector.compile_native_response(
                self.observations.projection,
                worker_count=self.worker_count,
                spatial_projection=self._spatial_projection,
            )
            mosaic_cache = NativeMosaicCache(response)
            object.__setattr__(
                self, "compile_seconds", self.compile_seconds + perf_counter() - start
            )
            object.__setattr__(self, "compile_count", self.compile_count + 1)
        self._responses[key] = mosaic_cache
        while len(self._responses) > 2:
            self._responses.popitem(last=False)
        object.__setattr__(self, "evaluation_count", self.evaluation_count + 1)
        prediction = mosaic_cache.response.evaluate(
            detector.strength_model,
            mosaic=MosaicParameters(*candidate.values[13:16]),
            thickness_A=candidate.film_thickness_A,
            specular_stitch_stack=stack,
            mosaic_cache=mosaic_cache,
        )
        prediction.setflags(write=False)
        self._predictions[candidate_key] = prediction
        while len(self._predictions) > 64:
            self._predictions.popitem(last=False)
        return prediction

    def clear_responses(self) -> None:
        """Release the explicitly retained native integration resources."""
        self._responses.clear()
        self._predictions.clear()
        object.__setattr__(self, "_spatial_projection", None)

    def stitch_state(self, candidate: BiJointCandidate) -> list[dict]:
        """Record the empirical normalization and selected interval for every surface/line."""
        physics = self.model.bind(BiCellSiteParameters.from_array(candidate.values[:13]))
        stack = replace(
            physics.specular_stitch_stack,
            top_roughness_A=float(candidate.values[19]),
            bottom_roughness_A=float(candidate.values[20]),
        )
        arguments = dict(
            coherent_repeats=candidate.coherent_repeats,
            film_thickness_A=candidate.film_thickness_A,
            surface_fractions=candidate.surface_fractions,
            phase_fractions=(1.0,),
            fault_parameters={},
        )
        return native_stitch_records(physics, arguments, stack)


def fit_bi_joint(
    evaluator: BiNativeFitEvaluator,
    initial: BiJointCandidate,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    active_indices: tuple[int, ...] = tuple(range(21)),
    maximum_iterations: int = 20,
    finite_difference_step: float = 1e-3,
    enforce_historical_guards: bool = True,
    callback=None,
) -> OptimizeResult:
    """Optimize scaled physical coordinates, profiling one nonnegative GLS scale.

    Bounds are mandatory declared search ranges, never inferred uncertainties.
    N is held exactly integer per call; run and compare explicitly chosen N
    values outside this optimizer. A returned candidate is not automatically
    accepted or numerically qualified.
    """
    if np.iscomplexobj(lower) or np.iscomplexobj(upper):
        raise ValueError("physical parameter bounds must be real")
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if any(type(index) is not int for index in active_indices):
        raise TypeError("active indices must be integers")
    active = np.asarray(active_indices, dtype=int)
    if (
        lower.shape != (21,)
        or upper.shape != (21,)
        or np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or np.any(upper <= lower)
        or np.any(initial.values < lower)
        or np.any(initial.values > upper)
        or len(active) == 0
        or len(set(active_indices)) != len(active)
        or np.any(active < 0)
        or np.any(active >= 21)
        or maximum_iterations < 1
        or not 0 < finite_difference_step < 1
    ):
        raise ValueError("invalid explicit joint fit ranges, active coordinates or search budget")
    # Validate coordinate-box extremes; orbit ordering must hold throughout it.
    BiJointCandidate(lower, initial.coherent_repeats)
    BiJointCandidate(upper, initial.coherent_repeats)
    for bi_z in (lower[2], upper[2]):
        for outer_z in (lower[3], upper[3]):
            BiCellSiteParameters.from_array(np.r_[lower[:2], bi_z, outer_z, lower[4:13]])
    width = upper - lower
    cached_x, cached = None, None

    def evaluate(x):
        nonlocal cached_x, cached
        if cached_x is None or not np.array_equal(x, cached_x):
            values = initial.values.copy()
            values[active] = lower[active] + width[active] * x
            candidate = BiJointCandidate(values, initial.coherent_repeats)
            raw = evaluator.predict(candidate)
            scale, residual = evaluator.observations.profile_scale(
                raw, enforce_guards=enforce_historical_guards
            )
            prediction = scale * raw
            scores = evaluator.observations.scores(prediction)
            cached = candidate, scale, residual, prediction, scores
            cached_x = x.copy()
            if callback is not None:
                callback(candidate, scale, residual, prediction, scores)
        return cached

    def objective(x):
        residual = evaluate(x)[2]
        return float(residual @ residual) / len(residual)

    constraints = ()
    if enforce_historical_guards:
        constraints = (
            {
                "type": "ineq",
                "fun": lambda x: (
                    1 - evaluate(x)[4]["guard_scores"] / evaluator.observations.guard_limit
                ),
            },
        )
    result = minimize(
        objective,
        (initial.values[active] - lower[active]) / width[active],
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(active),
        constraints=constraints,
        options=dict(maxiter=maximum_iterations, eps=finite_difference_step, ftol=1e-7),
    )
    candidate, scale, residual, prediction, scores = evaluate(result.x)
    result.candidate, result.scale, result.prediction_count = candidate, scale, prediction
    result.gls_chi_square, result.scores = float(residual @ residual), scores
    result.active_parameters = tuple(BI_JOINT_PARAMETER_NAMES[i] for i in active)
    result.numerical_status = "not_qualified"
    return result


def bi_joint_sensitivity(evaluator, candidate, *, lower, upper, step=1e-3, callback=None):
    """Profile-scale whitened Jacobian per search-range fraction, with bound-aware steps.

    This diagnoses local weak combinations; it is not a posterior covariance or
    evidence that quadrature errors are smaller than the inferred uncertainty.
    """
    lower, upper = np.asarray(lower), np.asarray(upper)
    if (
        lower.shape != (21,)
        or upper.shape != (21,)
        or np.iscomplexobj(lower)
        or np.iscomplexobj(upper)
        or np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or not 0 < step < 1
        or np.any(upper <= lower)
        or np.any(candidate.values < lower)
        or np.any(candidate.values > upper)
    ):
        raise ValueError("sensitivity requires containing ranges and a fractional step")
    raw = evaluator.predict(candidate)
    _, baseline = evaluator.observations.profile_scale(raw)
    result = np.empty((len(baseline), 21))
    if callback is not None:
        callback(None, candidate, evaluator.stitch_state(candidate))
    for index in range(21):
        width = upper[index] - lower[index]
        delta = step * width
        if candidate.values[index] + delta > upper[index]:
            delta = -min(delta, candidate.values[index] - lower[index])
            if -delta < upper[index] - candidate.values[index]:
                delta = upper[index] - candidate.values[index]
        values = candidate.values.copy()
        values[index] += delta
        perturbed = BiJointCandidate(values, candidate.coherent_repeats)
        _, residual = evaluator.observations.profile_scale(evaluator.predict(perturbed))
        if callback is not None:
            callback(index, perturbed, evaluator.stitch_state(perturbed))
        result[:, index] = (residual - baseline) / (delta / width)
    return result
