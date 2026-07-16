"""Named pure and manuscript-composite specular outputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.ordered.finite_stack import Bi2Se3WholeCellCompatResult
from rasim_next.reflectivity.parratt import ParrattResult

KinematicEvaluator = Callable[[NDArray[np.float64]], ArrayLike]


@dataclass(frozen=True, slots=True)
class SpecularResult:
    """Pure inputs and named composite retained as separate observables."""

    qz_Ainv: NDArray[np.float64]
    phase_l_coordinate: NDArray[np.float64]
    raw_kinematic_e2: NDArray[np.float64]
    parratt_reflectivity: NDArray[np.float64]
    scaled_high_branch: NDArray[np.float64]
    composite_reflectivity: NDArray[np.float64]
    scale_factor: float
    blend_bounds_q_over_qc: tuple[float, float]
    blend_selection: str
    raw_kinematic_normalization: str = "raw finite-stack electron2"
    parratt_normalization: str = "dimensionless pure Parratt reflectivity"
    composite_normalization: str = "dimensionless manuscript specular composite"

    def __post_init__(self) -> None:
        arrays = tuple(
            np.array(value, dtype=np.float64, copy=True, order="C")
            for value in (
                self.qz_Ainv,
                self.phase_l_coordinate,
                self.raw_kinematic_e2,
                self.parratt_reflectivity,
                self.scaled_high_branch,
                self.composite_reflectivity,
            )
        )
        if arrays[0].ndim != 1 or any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError("specular outputs must share one one-dimensional qz grid")
        if not all(np.all(np.isfinite(array)) and np.all(array >= 0.0) for array in arrays):
            raise ValueError("specular coordinates and outputs must be finite and nonnegative")
        bounds = tuple(float(value) for value in self.blend_bounds_q_over_qc)
        if (
            len(bounds) != 2
            or not np.all(np.isfinite(bounds))
            or not 0.0 <= bounds[0] < bounds[1]
            or not np.isfinite(self.scale_factor)
            or self.scale_factor <= 0.0
            or self.blend_selection not in {"automatic", "fallback"}
            or not all(
                (
                    self.raw_kinematic_normalization,
                    self.parratt_normalization,
                    self.composite_normalization,
                )
            )
        ):
            raise ValueError("specular scale, blend, and normalization metadata are invalid")
        for array in arrays:
            array.setflags(write=False)
        (
            qz,
            phase_l,
            raw_kinematic,
            parratt,
            high,
            composite,
        ) = arrays
        object.__setattr__(self, "qz_Ainv", qz)
        object.__setattr__(self, "phase_l_coordinate", phase_l)
        object.__setattr__(self, "raw_kinematic_e2", raw_kinematic)
        object.__setattr__(self, "parratt_reflectivity", parratt)
        object.__setattr__(self, "scaled_high_branch", high)
        object.__setattr__(self, "composite_reflectivity", composite)
        object.__setattr__(self, "scale_factor", float(self.scale_factor))
        object.__setattr__(self, "blend_bounds_q_over_qc", bounds)


@dataclass(frozen=True, slots=True)
class Bi2Se3WholeCellCompatSpecularResult:
    """Exact-coordinate stages of the frozen Bi2Se3 legacy specular stitch."""

    qz_Ainv: NDArray[np.float64]
    external_l_coordinate: NDArray[np.float64]
    phase_l_coordinate: NDArray[np.float64]
    raw_finite_stack_e2: NDArray[np.float64]
    external_legacy_kinematic: NDArray[np.float64]
    phase_legacy_kinematic: NDArray[np.float64]
    ht_over_qz2: NDArray[np.float64]
    parratt_reflectivity: NDArray[np.float64]
    kinematic_parratt_over_qz2: NDArray[np.float64]
    kinematic_stitched_over_qz2: NDArray[np.float64]
    composite_reflectivity: NDArray[np.float64]
    legacy_stitched_intensity: NDArray[np.float64]
    parratt_to_kinematic_scale: float
    kinematic_to_reflectivity_scale: float
    blend_bounds_q_over_qc: tuple[float, float]
    blend_selection: str
    cache_identity: str
    provenance: str
    scale_direction: str = "parratt_to_kinematic"
    raw_finite_stack_normalization: str = "raw finite-stack electron2"
    legacy_kinematic_normalization: str = "legacy AREA*(pole-clamped pair sum/17)*|F|^2"
    composite_normalization: str = "dimensionless compatibility specular composite"

    def __post_init__(self) -> None:
        arrays = tuple(
            np.array(value, dtype=np.float64, copy=True, order="C")
            for value in (
                self.qz_Ainv,
                self.external_l_coordinate,
                self.phase_l_coordinate,
                self.raw_finite_stack_e2,
                self.external_legacy_kinematic,
                self.phase_legacy_kinematic,
                self.ht_over_qz2,
                self.parratt_reflectivity,
                self.kinematic_parratt_over_qz2,
                self.kinematic_stitched_over_qz2,
                self.composite_reflectivity,
                self.legacy_stitched_intensity,
            )
        )
        if arrays[0].ndim != 1 or any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError("compatibility specular stages must share one qz grid")
        if not all(np.all(np.isfinite(array)) and np.all(array >= 0.0) for array in arrays):
            raise ValueError("compatibility specular stages must be finite and nonnegative")
        bounds = tuple(float(value) for value in self.blend_bounds_q_over_qc)
        forward_scale = float(self.parratt_to_kinematic_scale)
        inverse_scale = float(self.kinematic_to_reflectivity_scale)
        if (
            len(bounds) != 2
            or not np.all(np.isfinite(bounds))
            or not 0.0 <= bounds[0] < bounds[1]
            or not np.isfinite(forward_scale)
            or forward_scale <= 0.0
            or not np.isfinite(inverse_scale)
            or inverse_scale <= 0.0
            or not np.isclose(forward_scale * inverse_scale, 1.0, rtol=2e-15, atol=0.0)
            or self.blend_selection not in {"automatic", "fallback"}
            or self.scale_direction != "parratt_to_kinematic"
            or not self.cache_identity
            or not self.provenance
            or not self.raw_finite_stack_normalization
            or not self.legacy_kinematic_normalization
            or not self.composite_normalization
        ):
            raise ValueError("compatibility specular metadata are invalid")
        for array in arrays:
            array.setflags(write=False)
        (
            qz,
            external_l,
            phase_l,
            raw_finite,
            external_legacy,
            phase_legacy,
            ht_over_qz2,
            parratt,
            kinematic_parratt,
            stitched,
            composite,
            legacy_stitched,
        ) = arrays
        object.__setattr__(self, "qz_Ainv", qz)
        object.__setattr__(self, "external_l_coordinate", external_l)
        object.__setattr__(self, "phase_l_coordinate", phase_l)
        object.__setattr__(self, "raw_finite_stack_e2", raw_finite)
        object.__setattr__(self, "external_legacy_kinematic", external_legacy)
        object.__setattr__(self, "phase_legacy_kinematic", phase_legacy)
        object.__setattr__(self, "ht_over_qz2", ht_over_qz2)
        object.__setattr__(self, "parratt_reflectivity", parratt)
        object.__setattr__(self, "kinematic_parratt_over_qz2", kinematic_parratt)
        object.__setattr__(self, "kinematic_stitched_over_qz2", stitched)
        object.__setattr__(self, "composite_reflectivity", composite)
        object.__setattr__(self, "legacy_stitched_intensity", legacy_stitched)
        object.__setattr__(self, "parratt_to_kinematic_scale", forward_scale)
        object.__setattr__(self, "kinematic_to_reflectivity_scale", inverse_scale)
        object.__setattr__(self, "blend_bounds_q_over_qc", bounds)


def _evaluate_kinematic(
    evaluator: KinematicEvaluator, layer_coordinate: NDArray[np.float64], name: str
) -> NDArray[np.float64]:
    result = np.asarray(evaluator(layer_coordinate), dtype=np.float64)
    if result.shape != layer_coordinate.shape:
        raise ValueError(f"{name} kinematic evaluator result must align with layer coordinates")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} kinematic intensity must be finite and nonnegative")
    return result


def _blend_bounds(
    q_over_qc: NDArray[np.float64],
    valid: NDArray[np.bool_],
    mismatch: NDArray[np.float64],
) -> tuple[tuple[float, float], str]:
    indices = np.flatnonzero(valid)
    if indices.size:
        runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        eligible: list[tuple[float, float, float, float]] = []
        for run in runs:
            width = float(q_over_qc[run[-1]] - q_over_qc[run[0]])
            if width + 1e-12 >= 1.0:
                eligible.append(
                    (
                        -width,
                        float(np.median(mismatch[run])),
                        float(q_over_qc[run[0]]),
                        float(q_over_qc[run[-1]]),
                    )
                )
        if eligible:
            _, _, start, end = min(eligible)
            return (start, end), "automatic"
    return (3.0, 6.0), "fallback"


def manuscript_specular_composite(
    parratt: ParrattResult,
    kinematic_at_l: KinematicEvaluator,
    *,
    c_A: float,
    qc_Ainv: float,
    film_layer_index: int,
    fit_mask: ArrayLike | None = None,
) -> SpecularResult:
    """Build the named dimensionless handoff without changing either pure input."""

    qz = np.asarray(parratt.qz_Ainv)
    if qz.ndim != 1 or qz.size < 2 or np.any(qz <= 0.0) or np.any(np.diff(qz) <= 0.0):
        raise ValueError("composite qz_Ainv must be a positive strictly increasing grid")
    c_value = float(c_A)
    qc_value = float(qc_Ainv)
    if not np.isfinite(c_value) or c_value <= 0.0 or not np.isfinite(qc_value) or qc_value <= 0.0:
        raise ValueError("c_A and qc_Ainv must be finite and positive")
    try:
        film_index = index(film_layer_index)
    except TypeError as error:
        raise ValueError("film_layer_index must identify an interior Parratt layer") from error
    if isinstance(film_layer_index, bool) or not 0 < film_index < parratt.kz_Ainv.shape[-1] - 1:
        raise ValueError("film_layer_index must identify an interior Parratt layer")

    if fit_mask is None:
        fit = np.ones(qz.shape, dtype=np.bool_)
    else:
        fit = np.asarray(fit_mask, dtype=np.bool_)
        if fit.shape != qz.shape:
            raise ValueError("fit_mask must align with the Parratt qz grid")
    external_l = qz * c_value / (2.0 * np.pi)
    phase_qz = 2.0 * np.maximum(parratt.kz_Ainv[:, film_index].real, 0.0)
    phase_l = phase_qz * c_value / (2.0 * np.pi)
    raw_kinematic = _evaluate_kinematic(kinematic_at_l, external_l, "external-phase")
    phase_kinematic = _evaluate_kinematic(kinematic_at_l, phase_l, "internal-phase")
    zero = _evaluate_kinematic(kinematic_at_l, np.zeros(1, dtype=np.float64), "zero-phase")[0]
    if zero <= 0.0:
        raise ValueError("zero-phase kinematic intensity must be positive")

    q_over_qc = qz / qc_value
    shape_term = (phase_kinematic / zero) / qz**2
    pure_parratt = np.asarray(parratt.reflectivity)
    scale_points = (
        fit & (q_over_qc > 5.0) & (q_over_qc < 10.0) & (pure_parratt > 0.0) & (shape_term > 0.0)
    )
    if not np.any(scale_points):
        raise ValueError("no positive finite points exist in the declared 5<Qz/Qc<10 fit mask")
    log_scale = np.median(np.log(pure_parratt[scale_points]) - np.log(shape_term[scale_points]))
    scale_factor = float(np.exp(log_scale))
    high_branch = scale_factor * shape_term
    positive = (pure_parratt > 0.0) & (high_branch > 0.0)
    mismatch = np.full(qz.shape, np.inf, dtype=np.float64)
    mismatch[positive] = np.abs(np.log10(high_branch[positive] / pure_parratt[positive]))
    handoff_points = fit & positive & (q_over_qc >= 3.0) & (q_over_qc <= 10.0) & (mismatch <= 0.10)
    bounds, selection = _blend_bounds(q_over_qc, handoff_points, mismatch)
    lower, upper = bounds
    composite = np.array(pure_parratt, copy=True)
    above = q_over_qc >= upper
    composite[above] = high_branch[above]
    interior = (q_over_qc > lower) & (q_over_qc < upper)
    if np.any(interior):
        coordinate = np.clip((q_over_qc[interior] - lower) / (upper - lower), 0.0, 1.0)
        weight = 6.0 * coordinate**5 - 15.0 * coordinate**4 + 10.0 * coordinate**3
        floor = np.finfo(np.float64).tiny
        composite[interior] = 10.0 ** (
            (1.0 - weight) * np.log10(np.maximum(pure_parratt[interior], floor))
            + weight * np.log10(np.maximum(high_branch[interior], floor))
        )
    return SpecularResult(
        qz_Ainv=qz,
        phase_l_coordinate=phase_l,
        raw_kinematic_e2=raw_kinematic,
        parratt_reflectivity=pure_parratt,
        scaled_high_branch=high_branch,
        composite_reflectivity=composite,
        scale_factor=scale_factor,
        blend_bounds_q_over_qc=bounds,
        blend_selection=selection,
    )


def bi2se3_whole_cell_compat_specular(
    parratt: ParrattResult,
    external_curve: Bi2Se3WholeCellCompatResult,
    phase_curve: Bi2Se3WholeCellCompatResult,
    *,
    c_A: float,
    qc_Ainv: float,
    film_layer_index: int,
    fit_mask: ArrayLike | None = None,
) -> Bi2Se3WholeCellCompatSpecularResult:
    """Stitch exact external- and internal-phase compatibility curves without interpolation."""

    qz = np.asarray(parratt.qz_Ainv)
    if qz.ndim != 1 or qz.size < 2 or np.any(qz <= 0.0) or np.any(np.diff(qz) <= 0.0):
        raise ValueError("compatibility qz_Ainv must be a positive strictly increasing grid")
    c_value = float(c_A)
    qc_value = float(qc_Ainv)
    if not np.isclose(c_value, 28.636, rtol=0.0, atol=1e-12):
        raise ValueError("Bi2Se3 whole-cell compatibility requires c_A=28.636")
    if not np.isfinite(qc_value) or qc_value <= 0.0:
        raise ValueError("qc_Ainv must be finite and positive")
    try:
        film_index = index(film_layer_index)
    except TypeError as error:
        raise ValueError("film_layer_index must identify an interior Parratt layer") from error
    if isinstance(film_layer_index, bool) or not 0 < film_index < parratt.kz_Ainv.shape[-1] - 1:
        raise ValueError("film_layer_index must identify an interior Parratt layer")
    if (
        external_curve.event_id.shape != qz.shape
        or phase_curve.event_id.shape != qz.shape
        or not np.array_equal(external_curve.event_id, phase_curve.event_id)
    ):
        raise ValueError("compatibility curves must share event identity with the qz grid")

    external_l = qz * c_value / (2.0 * np.pi)
    if not np.array_equal(external_curve.external_l_coordinate, external_l):
        raise ValueError("external compatibility curve does not match exact external L")
    phase_qz = 2.0 * np.maximum(parratt.kz_Ainv[:, film_index].real, 0.0)
    phase_l = phase_qz * c_value / (2.0 * np.pi)
    if not np.array_equal(phase_curve.external_l_coordinate, phase_l):
        raise ValueError("phase compatibility curve does not match exact phase L")

    if fit_mask is None:
        fit = np.ones(qz.shape, dtype=np.bool_)
    else:
        fit = np.asarray(fit_mask, dtype=np.bool_)
        if fit.shape != qz.shape:
            raise ValueError("fit_mask must align with the Parratt qz grid")
    pure_parratt = np.asarray(parratt.reflectivity)
    ht_over_qz2 = phase_curve.legacy_intensity / qz**2
    q_over_qc = qz / qc_value
    scale_points = (
        fit & (q_over_qc >= 6.0) & (q_over_qc <= 10.0) & (pure_parratt > 0.0) & (ht_over_qz2 > 0.0)
    )
    if not np.any(scale_points):
        raise ValueError("no positive finite points exist in the declared 6<=Qz/Qc<=10 fit mask")
    log_scale = np.median(np.log(ht_over_qz2[scale_points]) - np.log(pure_parratt[scale_points]))
    parratt_to_kinematic_scale = float(np.exp(log_scale))
    kinematic_to_reflectivity_scale = 1.0 / parratt_to_kinematic_scale
    kinematic_parratt = pure_parratt * parratt_to_kinematic_scale

    positive = (kinematic_parratt > 0.0) & (ht_over_qz2 > 0.0)
    mismatch = np.full(qz.shape, np.inf, dtype=np.float64)
    mismatch[positive] = np.abs(np.log10(ht_over_qz2[positive] / kinematic_parratt[positive]))
    handoff_points = fit & positive & (q_over_qc >= 3.0) & (q_over_qc <= 10.0) & (mismatch <= 0.10)
    bounds, selection = _blend_bounds(q_over_qc, handoff_points, mismatch)
    lower, upper = bounds
    stitched = np.array(kinematic_parratt, copy=True)
    above = q_over_qc >= upper
    stitched[above] = ht_over_qz2[above]
    interior = (q_over_qc > lower) & (q_over_qc < upper)
    if np.any(interior):
        coordinate = np.clip((q_over_qc[interior] - lower) / (upper - lower), 0.0, 1.0)
        weight = 6.0 * coordinate**5 - 15.0 * coordinate**4 + 10.0 * coordinate**3
        floor = np.finfo(np.float64).tiny
        stitched[interior] = 10.0 ** (
            (1.0 - weight) * np.log10(np.maximum(kinematic_parratt[interior], floor))
            + weight * np.log10(np.maximum(ht_over_qz2[interior], floor))
        )
    composite = stitched * kinematic_to_reflectivity_scale
    legacy_stitched = stitched * qz**2
    identity = (
        f"{external_curve.cache_identity};phase={phase_curve.cache_identity};"
        f"qc_Ainv={qc_value:.17g};film_layer_index={film_index};"
        "scale_fit=6<=Qz/Qc<=10;scale_direction=parratt_to_kinematic"
    )
    return Bi2Se3WholeCellCompatSpecularResult(
        qz_Ainv=qz,
        external_l_coordinate=external_curve.external_l_coordinate,
        phase_l_coordinate=phase_curve.external_l_coordinate,
        raw_finite_stack_e2=external_curve.finite_stack.intensity.intensity_per_sr,
        external_legacy_kinematic=external_curve.legacy_intensity,
        phase_legacy_kinematic=phase_curve.legacy_intensity,
        ht_over_qz2=ht_over_qz2,
        parratt_reflectivity=pure_parratt,
        kinematic_parratt_over_qz2=kinematic_parratt,
        kinematic_stitched_over_qz2=stitched,
        composite_reflectivity=composite,
        legacy_stitched_intensity=legacy_stitched,
        parratt_to_kinematic_scale=parratt_to_kinematic_scale,
        kinematic_to_reflectivity_scale=kinematic_to_reflectivity_scale,
        blend_bounds_q_over_qc=bounds,
        blend_selection=selection,
        cache_identity=identity,
        provenance=identity,
    )
