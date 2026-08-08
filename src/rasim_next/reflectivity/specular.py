"""Named pure and manuscript-composite specular outputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.reflectivity.parratt import ParrattResult, parratt_reflectivity

KinematicEvaluator = Callable[[NDArray[np.float64]], ArrayLike]
LOCAL_LAMELLA_INTERFACE = "local_lamella_follows_mosaic.v1"
FIXED_EXTERNAL_QZ_INTERFACE = "fixed_external_qz_m0_strength.v1"


def parratt_stitch_interface_code(interface_assumption: str) -> int:
    """Return the compiled code for one explicit m=0 interface convention."""

    if interface_assumption == LOCAL_LAMELLA_INTERFACE:
        return 1
    if interface_assumption == FIXED_EXTERNAL_QZ_INTERFACE:
        return 2
    raise ValueError("unsupported Parratt stitch interface assumption")


def parratt_stitch_interface_assumption(interface_code: int) -> str:
    """Return the declared convention for a compiled nonzero stitch code."""

    if interface_code == 1:
        return LOCAL_LAMELLA_INTERFACE
    if interface_code == 2:
        return FIXED_EXTERNAL_QZ_INTERFACE
    raise ValueError("unsupported compiled Parratt stitch interface code")


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
class KinematicScaleSpecularResult:
    """Named empirical handoff expressed in finite-stack strength units."""

    qz_Ainv: NDArray[np.float64]
    phase_l_coordinate: NDArray[np.float64]
    phase_kinematic_strength_A2: NDArray[np.float64]
    scaled_parratt_strength_A2: NDArray[np.float64]
    composite_strength_A2: NDArray[np.float64]
    strength_ratio: NDArray[np.float64]
    dimensionless_scale_factor: float
    blend_bounds_q_over_qc: tuple[float, float]
    blend_selection: str
    parratt_normalization: str = "dimensionless pure Parratt reflectivity"
    composite_normalization: str = "kinematic finite-stack strength A2"

    def __post_init__(self) -> None:
        arrays = tuple(
            np.array(value, dtype=np.float64, copy=True, order="C")
            for value in (
                self.qz_Ainv,
                self.phase_l_coordinate,
                self.phase_kinematic_strength_A2,
                self.scaled_parratt_strength_A2,
                self.composite_strength_A2,
                self.strength_ratio,
            )
        )
        if arrays[0].ndim != 1 or any(value.shape != arrays[0].shape for value in arrays[1:]):
            raise ValueError("kinematic-scale specular outputs must align on one qz grid")
        if not all(np.all(np.isfinite(value)) and np.all(value >= 0.0) for value in arrays):
            raise ValueError("kinematic-scale specular outputs must be finite and nonnegative")
        scale = float(self.dimensionless_scale_factor)
        bounds = tuple(float(value) for value in self.blend_bounds_q_over_qc)
        if (
            not np.isfinite(scale)
            or scale <= 0.0
            or len(bounds) != 2
            or not 0.0 <= bounds[0] < bounds[1]
            or self.blend_selection not in {"automatic", "fallback"}
            or not self.parratt_normalization
            or not self.composite_normalization
        ):
            raise ValueError("kinematic-scale stitch metadata are invalid")
        for value in arrays:
            value.setflags(write=False)
        (
            qz,
            phase_l,
            phase_strength,
            scaled_parratt,
            composite,
            ratio,
        ) = arrays
        object.__setattr__(self, "qz_Ainv", qz)
        object.__setattr__(self, "phase_l_coordinate", phase_l)
        object.__setattr__(self, "phase_kinematic_strength_A2", phase_strength)
        object.__setattr__(self, "scaled_parratt_strength_A2", scaled_parratt)
        object.__setattr__(self, "composite_strength_A2", composite)
        object.__setattr__(self, "strength_ratio", ratio)
        object.__setattr__(self, "dimensionless_scale_factor", scale)
        object.__setattr__(self, "blend_bounds_q_over_qc", bounds)


@dataclass(frozen=True, slots=True)
class ParrattStitchStack:
    """Fixed substrate and interface inputs for the declared `m=0` handoff."""

    substrate_refractive_index: complex
    top_roughness_A: float = 0.0
    bottom_roughness_A: float = 0.0
    model_id: str = "empirical_parratt_kinematic_strength.v1"
    interface_assumption: str = LOCAL_LAMELLA_INTERFACE

    def __post_init__(self) -> None:
        index_value = complex(self.substrate_refractive_index)
        roughness = (float(self.top_roughness_A), float(self.bottom_roughness_A))
        if (
            not np.isfinite(index_value.real)
            or not np.isfinite(index_value.imag)
            or index_value.real <= 0.0
            or index_value.imag < 0.0
        ):
            raise ValueError("substrate_refractive_index must be finite, positive, and passive")
        if not all(np.isfinite(value) and value >= 0.0 for value in roughness):
            raise ValueError("interface roughnesses must be finite and nonnegative")
        if self.model_id != "empirical_parratt_kinematic_strength.v1":
            raise ValueError("unsupported Parratt stitch model_id")
        parratt_stitch_interface_code(self.interface_assumption)
        object.__setattr__(self, "substrate_refractive_index", index_value)
        object.__setattr__(self, "top_roughness_A", roughness[0])
        object.__setattr__(self, "bottom_roughness_A", roughness[1])


@dataclass(frozen=True, slots=True)
class CompiledParrattStitch:
    """Source-wavelength scalar state consumed by detector evaluators."""

    film_refractive_index: complex
    substrate_refractive_index: complex
    film_thickness_A: float
    top_roughness_A: float
    bottom_roughness_A: float
    qc_Ainv: float
    zero_strength_A2: float
    dimensionless_scale_factor: float
    blend_bounds_q_over_qc: tuple[float, float]
    blend_selection: str
    model_id: str = "empirical_parratt_kinematic_strength.v1"
    interface_assumption: str = LOCAL_LAMELLA_INTERFACE

    def __post_init__(self) -> None:
        film = complex(self.film_refractive_index)
        substrate = complex(self.substrate_refractive_index)
        scalars = tuple(
            float(value)
            for value in (
                self.film_thickness_A,
                self.top_roughness_A,
                self.bottom_roughness_A,
                self.qc_Ainv,
                self.zero_strength_A2,
                self.dimensionless_scale_factor,
            )
        )
        bounds = tuple(float(value) for value in self.blend_bounds_q_over_qc)
        if not all(
            np.isfinite(value.real)
            and np.isfinite(value.imag)
            and value.real > 0.0
            and value.imag >= 0.0
            for value in (film, substrate)
        ):
            raise ValueError(
                "compiled stitch refractive indices must be finite, positive, and passive"
            )
        if (
            not all(np.isfinite(value) and value >= 0.0 for value in scalars)
            or scalars[3] == 0.0
            or scalars[4] == 0.0
            or scalars[5] == 0.0
            or len(bounds) != 2
            or not 0.0 <= bounds[0] < bounds[1]
            or self.blend_selection not in {"automatic", "fallback"}
            or self.model_id != "empirical_parratt_kinematic_strength.v1"
        ):
            raise ValueError("compiled Parratt stitch state is invalid")
        parratt_stitch_interface_code(self.interface_assumption)
        object.__setattr__(self, "film_refractive_index", film)
        object.__setattr__(self, "substrate_refractive_index", substrate)
        (
            thickness,
            top,
            bottom,
            qc,
            zero,
            scale,
        ) = scalars
        object.__setattr__(self, "film_thickness_A", thickness)
        object.__setattr__(self, "top_roughness_A", top)
        object.__setattr__(self, "bottom_roughness_A", bottom)
        object.__setattr__(self, "qc_Ainv", qc)
        object.__setattr__(self, "zero_strength_A2", zero)
        object.__setattr__(self, "dimensionless_scale_factor", scale)
        object.__setattr__(self, "blend_bounds_q_over_qc", bounds)

    @property
    def interface_code(self) -> int:
        """Numeric convention code consumed by compiled detector kernels."""

        return parratt_stitch_interface_code(self.interface_assumption)


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


def kinematic_scale_specular_stitch(
    parratt: ParrattResult,
    kinematic_at_l: KinematicEvaluator,
    *,
    c_A: float,
    qc_Ainv: float,
    film_layer_index: int,
    fit_mask: ArrayLike | None = None,
) -> KinematicScaleSpecularResult:
    """Convert the named dimensionless handoff back to finite-stack strength units.

    The conversion is the corrected RA-SIM/manuscript compatibility observable.  It replaces the
    low-q ``m=0`` strength; it is not an additive reflectivity channel.  Above the selected handoff
    the result is assigned directly from the continuous internal-phase kinematic evaluator, making
    recovery exact rather than merely asymptotic.
    """

    dimensionless = manuscript_specular_composite(
        parratt,
        kinematic_at_l,
        c_A=c_A,
        qc_Ainv=qc_Ainv,
        film_layer_index=film_layer_index,
        fit_mask=fit_mask,
    )
    qz = np.asarray(dimensionless.qz_Ainv)
    phase_strength = _evaluate_kinematic(
        kinematic_at_l,
        np.asarray(dimensionless.phase_l_coordinate),
        "internal-phase",
    )
    zero_strength = float(
        _evaluate_kinematic(
            kinematic_at_l,
            np.zeros(1, dtype=np.float64),
            "zero-phase",
        )[0]
    )
    if zero_strength <= 0.0:
        raise ValueError("zero-phase kinematic intensity must be positive")
    conversion = qz**2 * zero_strength / dimensionless.scale_factor
    scaled_parratt = conversion * dimensionless.parratt_reflectivity
    composite = conversion * dimensionless.composite_reflectivity
    _, upper = dimensionless.blend_bounds_q_over_qc
    above = qz / float(qc_Ainv) >= upper
    composite[above] = phase_strength[above]
    ratio = np.ones(qz.shape, dtype=np.float64)
    positive = phase_strength > 0.0
    ratio[positive] = composite[positive] / phase_strength[positive]
    ratio[above] = 1.0
    if np.any(~positive & (composite > 0.0)):
        raise ValueError("positive stitched strength cannot replace a zero kinematic branch")
    return KinematicScaleSpecularResult(
        qz_Ainv=qz,
        phase_l_coordinate=dimensionless.phase_l_coordinate,
        phase_kinematic_strength_A2=phase_strength,
        scaled_parratt_strength_A2=scaled_parratt,
        composite_strength_A2=composite,
        strength_ratio=ratio,
        dimensionless_scale_factor=dimensionless.scale_factor,
        blend_bounds_q_over_qc=dimensionless.blend_bounds_q_over_qc,
        blend_selection=dimensionless.blend_selection,
    )


def compile_parratt_stitch(
    stack: ParrattStitchStack,
    kinematic_at_l: KinematicEvaluator,
    *,
    wavelength_A: float,
    film_refractive_index: complex,
    film_thickness_A: float,
    c_A: float,
    grid_size: int = 513,
) -> CompiledParrattStitch:
    """Compile the overlap scale and handoff once for one incoherent source wavelength."""

    if not isinstance(stack, ParrattStitchStack):
        raise TypeError("stack must be ParrattStitchStack")
    wavelength = float(wavelength_A)
    film_index = complex(film_refractive_index)
    thickness = float(film_thickness_A)
    c_value = float(c_A)
    if (
        not np.isfinite(wavelength)
        or wavelength <= 0.0
        or not np.isfinite(film_index.real)
        or not np.isfinite(film_index.imag)
        or film_index.real <= 0.0
        or film_index.imag < 0.0
        or not np.isfinite(thickness)
        or thickness < 0.0
        or not np.isfinite(c_value)
        or c_value <= 0.0
    ):
        raise ValueError("compiled detector stitching requires passive film optics and geometry")
    if isinstance(grid_size, bool) or int(grid_size) != grid_size or int(grid_size) < 257:
        raise ValueError("grid_size must be an integer of at least 257")
    k0 = 2.0 * np.pi / wavelength
    critical_radicand = max(1.0 - film_index.real**2, 0.0)
    qc = 2.0 * k0 * np.sqrt(critical_radicand)
    if qc <= 0.0:
        raise ValueError("film index does not define a positive external critical wavevector")
    qz = np.linspace(0.05, 10.25, int(grid_size), dtype=np.float64) * qc
    pure = parratt_reflectivity(
        qz,
        wavelength,
        refractive_index=(1.0 + 0.0j, film_index, stack.substrate_refractive_index),
        thickness_A=(None, thickness, None),
        roughness_A=(stack.top_roughness_A, stack.bottom_roughness_A),
    )
    stitched = kinematic_scale_specular_stitch(
        pure,
        kinematic_at_l,
        c_A=c_value,
        qc_Ainv=qc,
        film_layer_index=1,
    )
    return CompiledParrattStitch(
        film_refractive_index=film_index,
        substrate_refractive_index=stack.substrate_refractive_index,
        film_thickness_A=thickness,
        top_roughness_A=stack.top_roughness_A,
        bottom_roughness_A=stack.bottom_roughness_A,
        qc_Ainv=qc,
        zero_strength_A2=float(
            _evaluate_kinematic(
                kinematic_at_l,
                np.zeros(1, dtype=np.float64),
                "zero-phase",
            )[0]
        ),
        dimensionless_scale_factor=stitched.dimensionless_scale_factor,
        blend_bounds_q_over_qc=stitched.blend_bounds_q_over_qc,
        blend_selection=stitched.blend_selection,
        interface_assumption=stack.interface_assumption,
    )
