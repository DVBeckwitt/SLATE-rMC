"""Direct PbI2 stacking-profile compilation and population fitting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite, sqrt
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from rasim_next.core.contracts import (
    EventIntensityNormalization,
    LayerNormalQBatch,
    RodQueryBatch,
    canonical_revision_sha256,
)
from rasim_next.materials.crystal import CrystalStructure
from rasim_next.ordered.motifs import pbi2_layer_amplitudes
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.stacking.finite_intensity import finite_population_event_intensity
from rasim_next.stacking.parent_models import RichEpsilonModel, StackingPopulation
from rasim_next.stacking.transition import InitialPopulation, Parent, RegistryPhaseModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

STACKING_COMPONENT_IDS = ("2H", "4H+", "4H-", "6H+", "6H-")
STACKING_PHASE_IDS = ("2H", "4H", "6H")
_PBI2_PARENTS = (
    Parent.TWO_H,
    Parent.FOUR_H_PLUS,
    Parent.FOUR_H_MINUS,
    Parent.SIX_H_PLUS,
    Parent.SIX_H_MINUS,
)
_PBI2_EPSILON = 0.001
_PHASE_AGGREGATION = np.asarray(
    ((1.0, 0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0, 1.0))
)
_PHASE_AGGREGATION.setflags(write=False)


def _readonly_float(value: ArrayLike, shape: tuple[int | None, ...], name: str) -> FloatArray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    result = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
    if result.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(result.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has the wrong shape")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _readonly_int(value: ArrayLike, shape: tuple[int | None, ...], name: str) -> IntArray:
    supplied = np.asarray(value)
    if supplied.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    result = np.array(supplied, dtype=np.int64, copy=True, order="C")
    if result.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(result.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has the wrong shape")
    result.setflags(write=False)
    return result


def _positive_integer(value: int, name: str) -> int:
    try:
        result = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if isinstance(value, (bool, np.bool_)) or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledStackingResponse:
    """Five pointwise finite-parent strengths at explicit signed-rod/L queries."""

    component_response_A2: FloatArray
    signed_hk: IntArray
    l_coordinate: FloatArray
    wavelength_A: FloatArray
    fixed_model_revision: str
    component_ids: tuple[str, ...] = STACKING_COMPONENT_IDS
    normalization: EventIntensityNormalization = EventIntensityNormalization.FINITE_PER_LAYER
    sampling_revision: str = field(init=False)
    response_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if tuple(self.component_ids) != STACKING_COMPONENT_IDS:
            raise ValueError("component_ids must use the canonical five-parent order")
        response = _readonly_float(
            self.component_response_A2, (None, len(STACKING_COMPONENT_IDS)), "component_response_A2"
        )
        if response.shape[0] == 0 or np.any(response < 0.0):
            raise ValueError("component_response_A2 must be nonempty and nonnegative")
        signed_hk = _readonly_int(self.signed_hk, (response.shape[0], 2), "signed_hk")
        ell = _readonly_float(self.l_coordinate, (response.shape[0],), "l_coordinate")
        wavelength = _readonly_float(self.wavelength_A, (response.shape[0],), "wavelength_A")
        if np.any(wavelength <= 0.0):
            raise ValueError("wavelength_A must be positive")
        if not isinstance(self.fixed_model_revision, str) or not self.fixed_model_revision:
            raise ValueError("fixed_model_revision must be nonempty")
        normalization = EventIntensityNormalization(self.normalization)
        if normalization is not EventIntensityNormalization.FINITE_PER_LAYER:
            raise ValueError("stacking profiles require FINITE_PER_LAYER normalization")
        sampling_revision = canonical_revision_sha256(
            ("measure", "pointwise-intrinsic-signed-rod-L-strength-A2.v1"),
            ("signed_hk", signed_hk),
            ("l_coordinate", ell),
            ("wavelength_A", wavelength),
        )
        response_revision = canonical_revision_sha256(
            ("component_ids", STACKING_COMPONENT_IDS),
            ("component_response_A2", response),
            ("sampling_revision", sampling_revision),
            ("fixed_model_revision", self.fixed_model_revision),
            ("normalization", normalization.value),
        )
        object.__setattr__(self, "component_response_A2", response)
        object.__setattr__(self, "signed_hk", signed_hk)
        object.__setattr__(self, "l_coordinate", ell)
        object.__setattr__(self, "wavelength_A", wavelength)
        object.__setattr__(self, "component_ids", STACKING_COMPONENT_IDS)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "sampling_revision", sampling_revision)
        object.__setattr__(self, "response_revision", response_revision)


def _parent_populations() -> tuple[StackingPopulation, ...]:
    return tuple(
        StackingPopulation(
            population_id=parent.value,
            model=RichEpsilonModel(parent, _PBI2_EPSILON).transition_law(),
            initial=InitialPopulation.plus_only(),
        )
        for parent in _PBI2_PARENTS
    )


def compile_pbi2_stacking_profile_response(
    crystal: CrystalStructure,
    crystal_revision: str,
    *,
    signed_hk: ArrayLike,
    l_coordinate: ArrayLike,
    wavelength_A: ArrayLike,
    layers: int,
) -> CompiledStackingResponse:
    """Evaluate the five fixed PbI2 parents directly at signed-rod/L points."""

    if not isinstance(crystal, CrystalStructure):
        raise TypeError("crystal must be CrystalStructure")
    if (
        not isinstance(crystal_revision, str)
        or len(crystal_revision) != 64
        or any(character not in "0123456789abcdef" for character in crystal_revision)
    ):
        raise ValueError("crystal_revision must be a lowercase SHA-256 digest")
    layer_count = _positive_integer(layers, "layers")
    hk = _readonly_int(signed_hk, (None, 2), "signed_hk")
    if hk.shape[0] == 0:
        raise ValueError("signed_hk must be nonempty")
    int32 = np.iinfo(np.int32)
    if np.any((hk < int32.min) | (hk > int32.max)):
        raise ValueError("signed_hk values must fit in int32")
    ell = _readonly_float(l_coordinate, (hk.shape[0],), "l_coordinate")
    supplied_wavelength = np.asarray(wavelength_A)
    try:
        wavelength = _readonly_float(
            np.broadcast_to(supplied_wavelength, (hk.shape[0],)),
            (hk.shape[0],),
            "wavelength_A",
        )
    except ValueError as error:
        raise ValueError("wavelength_A must be scalar or align with signed_hk") from error
    if np.any(wavelength <= 0.0):
        raise ValueError("wavelength_A must be positive")
    reciprocal = ReciprocalLattice.from_crystal(crystal)
    hkl = np.column_stack((hk, ell))
    layer_normal = np.cross(crystal.direct_basis_A[:, 0], crystal.direct_basis_A[:, 1])
    layer_normal /= np.linalg.norm(layer_normal)
    layer_q = reciprocal.q_cartesian_Ainv(hkl) @ layer_normal
    _, rod_id = np.unique(hk, axis=0, return_inverse=True)
    event_id = np.arange(hk.shape[0], dtype=np.int64)
    query = RodQueryBatch(
        event_id=event_id,
        rod_id=rod_id,
        phase_id=(crystal.phase_id,) * hk.shape[0],
        h=hk[:, 0].astype(np.int32),
        k=hk[:, 1].astype(np.int32),
        q_sample_normal_Ainv=layer_q,
        l_coordinate=ell,
        wavelength_A=wavelength,
    )
    amplitudes = pbi2_layer_amplitudes(crystal, query, unknown_u_iso_A2=0.0)
    components = finite_population_event_intensity(
        query,
        amplitudes,
        _parent_populations(),
        layer_normal_q=LayerNormalQBatch(
            event_id=event_id,
            rod_id=query.rod_id,
            phase_id=query.phase_id,
            layer_normal_q_Ainv=layer_q,
            gauge_id=amplitudes.gauge_id,
        ),
        layers=layer_count,
        population_group_id="pbi2-stacking-parents",
        normalization=EventIntensityNormalization.FINITE_PER_LAYER,
        phase_model=RegistryPhaseModel.FORWARD_H_PLUS_2K,
    )
    if tuple(value.model_component_id for value in components) != STACKING_COMPONENT_IDS:
        raise RuntimeError("finite stacking components did not preserve canonical order")
    fixed_model_revision = canonical_revision_sha256(
        ("model", "pbi2-five-parent-finite-profile.v1"),
        ("crystal_revision", crystal_revision),
        ("layers", layer_count),
        ("epsilon", _PBI2_EPSILON),
        ("initial_population", "plus_only"),
        ("normalization", EventIntensityNormalization.FINITE_PER_LAYER.value),
        ("unknown_u_iso_A2", 0.0),
        ("registry_phase_model", RegistryPhaseModel.FORWARD_H_PLUS_2K.value),
    )
    return CompiledStackingResponse(
        component_response_A2=np.column_stack(
            tuple(value.scattering_strength_A2 for value in components)
        ),
        signed_hk=hk,
        l_coordinate=ell,
        wavelength_A=wavelength,
        fixed_model_revision=fixed_model_revision,
    )


class StackingPopulationIdentifiabilityError(ValueError):
    """The sampled profiles cannot distinguish all three phase totals."""

    def __init__(
        self,
        message: str,
        *,
        phase_contrast_rank: int,
        phase_contrast_singular_values: FloatArray,
        phase_contrast_condition: float,
    ) -> None:
        self.phase_contrast_rank = int(phase_contrast_rank)
        singular = np.array(phase_contrast_singular_values, dtype=np.float64, copy=True)
        singular.setflags(write=False)
        self.phase_contrast_singular_values = singular
        self.phase_contrast_condition = float(phase_contrast_condition)
        super().__init__(message)


@dataclass(frozen=True, slots=True, kw_only=True)
class StackingPopulationFitResult:
    """Fit and diagnostics; rank-deficient domain fractions are one NNLS representative."""

    domain_amount: FloatArray
    domain_fraction: FloatArray
    phase_fraction: FloatArray
    global_scale: float
    predicted_strength_A2: FloatArray
    weighted_residual: FloatArray
    chi_square: float
    active_component_ids: tuple[str, ...]
    domain_response_full_rank: bool
    response_singular_values: FloatArray
    response_rank: int
    response_condition: float
    phase_contrast_singular_values: FloatArray
    phase_contrast_rank: int
    phase_contrast_condition: float
    phase_profile_bounds: FloatArray
    phase_estimate_on_boundary: BoolArray
    profile_delta_chi_square: float
    response_revision: str
    component_ids: tuple[str, ...] = STACKING_COMPONENT_IDS
    phase_ids: tuple[str, ...] = STACKING_PHASE_IDS

    def __post_init__(self) -> None:
        if (
            tuple(self.component_ids) != STACKING_COMPONENT_IDS
            or tuple(self.phase_ids) != STACKING_PHASE_IDS
        ):
            raise ValueError("component_ids and phase_ids must use canonical order")
        arrays = {
            "domain_amount": _readonly_float(self.domain_amount, (5,), "domain_amount"),
            "domain_fraction": _readonly_float(self.domain_fraction, (5,), "domain_fraction"),
            "phase_fraction": _readonly_float(self.phase_fraction, (3,), "phase_fraction"),
            "predicted_strength_A2": _readonly_float(
                self.predicted_strength_A2, (None,), "predicted_strength_A2"
            ),
            "weighted_residual": _readonly_float(
                self.weighted_residual, (None,), "weighted_residual"
            ),
            "response_singular_values": _readonly_float(
                self.response_singular_values, (None,), "response_singular_values"
            ),
            "phase_contrast_singular_values": _readonly_float(
                self.phase_contrast_singular_values, (2,), "phase_contrast_singular_values"
            ),
            "phase_profile_bounds": _readonly_float(
                self.phase_profile_bounds, (3, 2), "phase_profile_bounds"
            ),
        }
        boundary = np.array(self.phase_estimate_on_boundary, dtype=np.bool_, copy=True)
        if boundary.shape != (3,):
            raise ValueError("phase_estimate_on_boundary must have shape (3,)")
        boundary.setflags(write=False)
        amount = arrays["domain_amount"]
        domain = arrays["domain_fraction"]
        phase = arrays["phase_fraction"]
        scale = float(self.global_scale)
        if np.any(amount < 0.0) or not isfinite(scale) or scale <= 0.0:
            raise ValueError("domain amounts and global_scale must be physically positive")
        if np.any(domain < 0.0) or not np.isclose(np.sum(domain), 1.0, rtol=0.0, atol=2.0e-12):
            raise ValueError("domain_fraction must lie on the probability simplex")
        if np.any(phase < 0.0) or not np.isclose(np.sum(phase), 1.0, rtol=0.0, atol=2.0e-12):
            raise ValueError("phase_fraction must lie on the probability simplex")
        if not np.allclose(amount, scale * domain, rtol=2.0e-12, atol=0.0):
            raise ValueError("global_scale must normalize domain_amount")
        if not np.allclose(phase, _PHASE_AGGREGATION @ domain, rtol=0.0, atol=2.0e-12):
            raise ValueError("phase fractions must aggregate domain fractions")
        bounds = arrays["phase_profile_bounds"]
        if np.any((bounds < 0.0) | (bounds > 1.0)) or np.any(
            (bounds[:, 0] > phase) | (bounds[:, 1] < phase)
        ):
            raise ValueError("phase_profile_bounds must contain each phase estimate")
        if arrays["predicted_strength_A2"].shape != arrays["weighted_residual"].shape:
            raise ValueError("prediction and residual arrays must align")
        if (
            not isfinite(self.chi_square)
            or self.chi_square < 0.0
            or not np.isclose(
                self.chi_square,
                arrays["weighted_residual"] @ arrays["weighted_residual"],
                rtol=2.0e-12,
                atol=1.0e-14,
            )
        ):
            raise ValueError("chi_square must be the squared weighted-residual norm")
        if set(self.active_component_ids) - set(STACKING_COMPONENT_IDS):
            raise ValueError("active_component_ids must identify canonical components")
        if not isfinite(self.profile_delta_chi_square) or self.profile_delta_chi_square <= 0.0:
            raise ValueError("profile_delta_chi_square must be positive")
        if not isinstance(self.response_revision, str) or not self.response_revision:
            raise ValueError("response_revision must be nonempty")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "global_scale", scale)
        object.__setattr__(self, "phase_estimate_on_boundary", boundary)
        object.__setattr__(self, "component_ids", STACKING_COMPONENT_IDS)
        object.__setattr__(self, "phase_ids", STACKING_PHASE_IDS)


@dataclass(frozen=True, slots=True)
class _LeastSquaresSolution:
    amount: FloatArray
    objective: float
    support: tuple[int, ...]


def _rank(singular_values: FloatArray, reference_scale: float) -> int:
    if singular_values.size == 0 or reference_scale == 0.0:
        return 0
    return int(np.count_nonzero(singular_values > sqrt(np.finfo(np.float64).eps) * reference_scale))


def _condition(singular_values: FloatArray, rank: int, required_rank: int) -> float:
    return (
        float(singular_values[0] / singular_values[required_rank - 1])
        if rank == required_rank and singular_values[required_rank - 1] > 0.0
        else float("inf")
    )


def _response_diagnostics(
    response: FloatArray,
    domain_fraction: FloatArray,
) -> tuple[FloatArray, int, float, FloatArray, int, float]:
    singular = np.linalg.svd(response, compute_uv=False)
    reference_scale = float(singular[0]) if singular.size else 0.0
    rank = _rank(singular, reference_scale)
    response_condition = _condition(singular, rank, response.shape[1])
    two_h = response[:, 0]
    four_h_mean = 0.5 * (response[:, 1] + response[:, 2])
    six_h_mean = 0.5 * (response[:, 3] + response[:, 4])
    phase_contrast = np.column_stack((four_h_mean - two_h, six_h_mean - two_h))
    nuisance = np.column_stack(
        (
            response @ domain_fraction,
            0.5 * (response[:, 1] - response[:, 2]),
            0.5 * (response[:, 3] - response[:, 4]),
        )
    )
    nuisance_left, nuisance_singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    nuisance_rank = _rank(nuisance_singular, reference_scale)
    if nuisance_rank:
        nuisance_basis = nuisance_left[:, :nuisance_rank]
        phase_contrast -= nuisance_basis @ (nuisance_basis.T @ phase_contrast)
    raw_phase_singular = np.linalg.svd(phase_contrast, compute_uv=False)
    phase_singular = np.zeros(2, dtype=np.float64)
    phase_singular[: raw_phase_singular.size] = raw_phase_singular
    phase_rank = _rank(phase_singular, reference_scale)
    phase_condition = _condition(phase_singular, phase_rank, 2)
    return (
        singular,
        rank,
        response_condition,
        phase_singular,
        phase_rank,
        phase_condition,
    )


def _nonnegative_least_squares(
    response: FloatArray,
    signal: FloatArray,
    equality: FloatArray | None = None,
) -> _LeastSquaresSolution:
    signal_norm_squared = float(signal @ signal)
    best = _LeastSquaresSolution(np.zeros(5), signal_norm_squared, ())
    rcond = sqrt(np.finfo(np.float64).eps)
    for mask in range(1, 1 << 5):
        support = tuple(component for component in range(5) if mask & (1 << component))
        selected = response[:, support]
        scale = np.linalg.norm(selected, axis=0)
        safe_scale = np.where(scale > 0.0, scale, 1.0)
        scaled = selected / safe_scale
        if equality is None:
            scaled_amount = np.linalg.lstsq(scaled, signal, rcond=rcond)[0]
        else:
            constraint = equality[list(support)] / safe_scale
            _, constraint_singular, constraint_right = np.linalg.svd(
                constraint[None, :], full_matrices=True
            )
            constraint_rank = _rank(constraint_singular, float(np.linalg.norm(constraint)))
            null_basis = constraint_right[constraint_rank:].T
            if null_basis.shape[1] == 0:
                continue
            scaled_amount = (
                null_basis @ np.linalg.lstsq(scaled @ null_basis, signal, rcond=rcond)[0]
            )
        feasibility_scale = max(
            float(np.linalg.norm(scaled_amount, ord=np.inf)),
            float(np.linalg.norm(signal)),
            np.finfo(np.float64).tiny,
        )
        positivity_tolerance = 8192.0 * np.finfo(np.float64).eps * feasibility_scale
        if np.any(scaled_amount < -positivity_tolerance):
            continue
        amount = np.zeros(5, dtype=np.float64)
        amount[list(support)] = np.maximum(scaled_amount, 0.0) / safe_scale
        if equality is not None:
            equality_tolerance = (
                8192.0
                * np.finfo(np.float64).eps
                * max(
                    float(np.linalg.norm(equality) * np.linalg.norm(amount)),
                    np.finfo(np.float64).tiny,
                )
            )
            if abs(float(equality @ amount)) > equality_tolerance:
                continue
        residual = response @ amount - signal
        objective = float(residual @ residual)
        effective_support = tuple(np.flatnonzero(amount > 0.0).tolist())
        tolerance = (
            64.0
            * np.finfo(np.float64).eps
            * max(objective, best.objective, np.finfo(np.float64).tiny)
        )
        if objective < best.objective - tolerance or (
            abs(objective - best.objective) <= tolerance
            and (len(effective_support), effective_support) < (len(best.support), best.support)
        ):
            best = _LeastSquaresSolution(amount, objective, effective_support)
    return best


def _profile_delta_function(
    response: FloatArray,
    signal: FloatArray,
    best_objective: float,
    phase_index: int,
    delta_chi_square: float,
) -> Callable[[float], float]:
    cache: dict[float, float] = {}

    def profile_delta(value: float) -> float:
        key = float(value)
        if key not in cache:
            equality = _PHASE_AGGREGATION[phase_index] - key
            constrained = _nonnegative_least_squares(response, signal, equality)
            cache[key] = constrained.objective - best_objective - delta_chi_square
        return cache[key]

    return profile_delta


def _profile_bounds(
    response: FloatArray,
    signal: FloatArray,
    best: _LeastSquaresSolution,
    phase_fraction: FloatArray,
    delta_chi_square: float,
) -> tuple[FloatArray, BoolArray]:
    bounds = np.zeros((3, 2), dtype=np.float64)
    for phase_index, optimum in enumerate(phase_fraction):
        profile_delta = _profile_delta_function(
            response, signal, best.objective, phase_index, delta_chi_square
        )
        tolerance = 64.0 * np.finfo(np.float64).eps * max(best.objective, 1.0)
        if profile_delta(float(optimum)) > tolerance:
            raise FloatingPointError("best phase fraction is outside its own profile bounds")
        bounds[phase_index, 0] = (
            0.0
            if profile_delta(0.0) <= tolerance
            else brentq(profile_delta, 0.0, float(optimum), xtol=2.0e-13)
        )
        bounds[phase_index, 1] = (
            1.0
            if profile_delta(1.0) <= tolerance
            else brentq(profile_delta, float(optimum), 1.0, xtol=2.0e-13)
        )
    boundary = (phase_fraction <= 4096.0 * np.finfo(np.float64).eps) | (
        phase_fraction >= 1.0 - 4096.0 * np.finfo(np.float64).eps
    )
    return bounds, boundary


def _broadcast_row(value: ArrayLike, size: int, name: str) -> FloatArray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(np.broadcast_to(supplied.real, (size,)), dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"{name} must be scalar or have shape ({size},)") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def fit_stacking_phase_totals(
    response: CompiledStackingResponse,
    observed_strength_A2: ArrayLike,
    variance_A4: ArrayLike,
    *,
    profile_delta_chi_square: float = 1.0,
) -> StackingPopulationFitResult:
    """Fit one global nonnegative amount vector, then normalize to phase fractions."""

    if not isinstance(response, CompiledStackingResponse):
        raise TypeError("response must be CompiledStackingResponse")
    row_count = response.component_response_A2.shape[0]
    observed = _broadcast_row(observed_strength_A2, row_count, "observed_strength_A2")
    variance = _broadcast_row(variance_A4, row_count, "variance_A4")
    if np.any(variance <= 0.0):
        raise ValueError("variance must be positive")
    profile_delta = float(profile_delta_chi_square)
    if not isfinite(profile_delta) or profile_delta <= 0.0:
        raise ValueError("profile_delta_chi_square must be finite and positive")
    inverse_sigma = 1.0 / np.sqrt(variance)
    weighted_response = response.component_response_A2 * inverse_sigma[:, None]
    weighted_signal = observed * inverse_sigma
    solution = _nonnegative_least_squares(weighted_response, weighted_signal)
    global_scale = float(np.sum(solution.amount))
    if not solution.support or global_scale <= 0.0:
        raise ValueError("fitted stacking scale is zero; phase fractions are undefined")
    domain_fraction = solution.amount / global_scale
    phase_fraction = _PHASE_AGGREGATION @ domain_fraction
    (
        response_singular,
        response_rank,
        response_condition,
        phase_singular,
        phase_rank,
        phase_condition,
    ) = _response_diagnostics(weighted_response, domain_fraction)
    if phase_rank < 2:
        raise StackingPopulationIdentifiabilityError(
            "sampled profiles cannot separate 2H, 4H, and 6H phase totals",
            phase_contrast_rank=phase_rank,
            phase_contrast_singular_values=phase_singular,
            phase_contrast_condition=phase_condition,
        )
    predicted = response.component_response_A2 @ solution.amount
    weighted_residual = (observed - predicted) * inverse_sigma
    profile_bounds, boundary = _profile_bounds(
        weighted_response, weighted_signal, solution, phase_fraction, profile_delta
    )
    return StackingPopulationFitResult(
        domain_amount=solution.amount,
        domain_fraction=domain_fraction,
        phase_fraction=phase_fraction,
        global_scale=global_scale,
        predicted_strength_A2=predicted,
        weighted_residual=weighted_residual,
        chi_square=float(weighted_residual @ weighted_residual),
        active_component_ids=tuple(STACKING_COMPONENT_IDS[value] for value in solution.support),
        domain_response_full_rank=response_rank == len(STACKING_COMPONENT_IDS),
        response_singular_values=response_singular,
        response_rank=response_rank,
        response_condition=response_condition,
        phase_contrast_singular_values=phase_singular,
        phase_contrast_rank=phase_rank,
        phase_contrast_condition=phase_condition,
        phase_profile_bounds=profile_bounds,
        phase_estimate_on_boundary=boundary,
        profile_delta_chi_square=profile_delta,
        response_revision=response.response_revision,
    )


__all__ = [
    "STACKING_COMPONENT_IDS",
    "STACKING_PHASE_IDS",
    "CompiledStackingResponse",
    "StackingPopulationFitResult",
    "StackingPopulationIdentifiabilityError",
    "compile_pbi2_stacking_profile_response",
    "fit_stacking_phase_totals",
]
