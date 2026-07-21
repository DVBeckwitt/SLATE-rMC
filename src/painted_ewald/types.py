"""Immutable records for rods, mosaic probability, and analytic Ewald roots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from painted_ewald.validation import finite_scalar, integer, positive_integer, readonly_float_array

FloatArray = NDArray[np.float64]


class RootStatus(StrEnum):
    """Classification of one infinite reciprocal-rod/Ewald intersection."""

    NO_ROOT = "no_root"
    TANGENT = "tangent"
    COLLAPSED_DIRECT = "collapsed_direct"
    REGULAR = "regular"


@dataclass(frozen=True, slots=True)
class EwaldRoot:
    """One retained analytic intersection of an infinite rod and Ewald sphere."""

    u_Ainv: float
    L: float
    q_sample_Ainv: FloatArray
    kf_sample_Ainv: FloatArray
    ewald_residual_Ainv: float
    coarea_jacobian: float
    branch: int

    def __post_init__(self) -> None:
        u_Ainv = finite_scalar(self.u_Ainv, "u_Ainv")
        l_coordinate = finite_scalar(self.L, "L")
        q_sample = readonly_float_array(self.q_sample_Ainv, (3,), "q_sample_Ainv")
        kf_sample = readonly_float_array(self.kf_sample_Ainv, (3,), "kf_sample_Ainv")
        residual = finite_scalar(self.ewald_residual_Ainv, "ewald_residual_Ainv")
        jacobian = finite_scalar(self.coarea_jacobian, "coarea_jacobian")
        branch = integer(self.branch, "branch")
        if residual < 0.0:
            raise ValueError("ewald_residual_Ainv must be nonnegative")
        if jacobian <= 0.0:
            raise ValueError("coarea_jacobian must be positive")
        if branch not in {0, 1, 2}:
            raise ValueError("branch must be 0, 1, or 2")
        object.__setattr__(self, "u_Ainv", u_Ainv)
        object.__setattr__(self, "L", l_coordinate)
        object.__setattr__(self, "q_sample_Ainv", q_sample)
        object.__setattr__(self, "kf_sample_Ainv", kf_sample)
        object.__setattr__(self, "ewald_residual_Ainv", residual)
        object.__setattr__(self, "coarea_jacobian", jacobian)
        object.__setattr__(self, "branch", branch)


@dataclass(frozen=True, slots=True)
class EwaldRootResult:
    """Roots and diagnostics for one complete infinite rod."""

    status: RootStatus
    emittable_roots: tuple[EwaldRoot, ...]
    direct_root_count: int

    def __post_init__(self) -> None:
        status = RootStatus(self.status)
        roots = tuple(self.emittable_roots)
        direct_count = integer(self.direct_root_count, "direct_root_count")
        if direct_count not in {0, 1}:
            raise ValueError("direct_root_count must be zero or one")
        if status is RootStatus.REGULAR:
            if direct_count == 1 and (len(roots) != 1 or roots[0].branch != 0):
                raise ValueError("regular m=0 results require one branch-0 root")
            if direct_count == 0 and (
                len(roots) != 2 or tuple(root.branch for root in roots) != (1, 2)
            ):
                raise ValueError("regular nonzero results require branches 1 and 2")
        elif roots:
            raise ValueError("non-regular results cannot emit roots")
        if status is RootStatus.COLLAPSED_DIRECT and direct_count != 1:
            raise ValueError("collapsed-direct results require one suppressed direct root")
        if status in {RootStatus.NO_ROOT, RootStatus.TANGENT} and direct_count != 0:
            raise ValueError("non-m0 classifications cannot suppress a direct root")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "emittable_roots", roots)
        object.__setattr__(self, "direct_root_count", direct_count)


@dataclass(frozen=True, slots=True)
class Rod:
    """One physical affine reciprocal rod identified by its in-plane indices."""

    h: int
    k: int
    population: float = 1.0

    def __post_init__(self) -> None:
        population = finite_scalar(self.population, "population")
        if population < 0.0:
            raise ValueError("population must be nonnegative")
        object.__setattr__(self, "h", integer(self.h, "h"))
        object.__setattr__(self, "k", integer(self.k, "k"))
        object.__setattr__(self, "population", population)

    @property
    def family_m(self) -> int:
        return self.h * self.h + self.h * self.k + self.k * self.k


class StrengthModel(Protocol):
    """Evaluate a finite nonnegative rod strength at one exact intersection."""

    def evaluate(self, *, rod: Rod, L: float, k_norm_Ainv: float) -> float: ...


class BasisBoundStrengthModel(StrengthModel, Protocol):
    """Strength model tied to the reciprocal basis used by its structure."""

    @property
    def reciprocal_basis_Ainv(self) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class MosaicParameters:
    """Wrapped Gaussian/Cauchy probability and deterministic quadrature controls."""

    gaussian_sigma_rad: float
    lorentzian_half_width_rad: float
    lorentzian_probability: float
    alpha_panel_count: int = 64
    alpha_gauss_order: int = 12
    azimuth_count: int = 256
    azimuth_phase_rad: float = 0.371

    def __post_init__(self) -> None:
        gaussian_sigma = finite_scalar(self.gaussian_sigma_rad, "gaussian_sigma_rad")
        lorentzian_half_width = finite_scalar(
            self.lorentzian_half_width_rad, "lorentzian_half_width_rad"
        )
        lorentzian_probability = finite_scalar(
            self.lorentzian_probability, "lorentzian_probability"
        )
        if gaussian_sigma < 0.0 or lorentzian_half_width < 0.0:
            raise ValueError("mosaic widths must be nonnegative")
        if not 0.0 <= lorentzian_probability <= 1.0:
            raise ValueError("lorentzian_probability must be between zero and one")
        object.__setattr__(self, "gaussian_sigma_rad", gaussian_sigma)
        object.__setattr__(self, "lorentzian_half_width_rad", lorentzian_half_width)
        object.__setattr__(self, "lorentzian_probability", lorentzian_probability)
        object.__setattr__(
            self, "alpha_panel_count", positive_integer(self.alpha_panel_count, "alpha_panel_count")
        )
        object.__setattr__(
            self, "alpha_gauss_order", positive_integer(self.alpha_gauss_order, "alpha_gauss_order")
        )
        object.__setattr__(
            self, "azimuth_count", positive_integer(self.azimuth_count, "azimuth_count")
        )
        object.__setattr__(
            self, "azimuth_phase_rad", finite_scalar(self.azimuth_phase_rad, "azimuth_phase_rad")
        )

    @property
    def zero_tilt_probability_mass(self) -> float:
        gaussian_mass = 1.0 - self.lorentzian_probability
        lorentzian_mass = self.lorentzian_probability
        return (gaussian_mass if self.gaussian_sigma_rad == 0.0 else 0.0) + (
            lorentzian_mass if self.lorentzian_half_width_rad == 0.0 else 0.0
        )


@dataclass(frozen=True, slots=True)
class MosaicSlice:
    """One normalized fixed-axial-coordinate cap or ring before Ewald conditioning."""

    rod: Rod
    u_Ainv: float
    q_sample_Ainv: FloatArray
    probability_mass: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.rod, Rod):
            raise TypeError("rod must be a Rod")
        u_Ainv = finite_scalar(self.u_Ainv, "u_Ainv")
        points = readonly_float_array(self.q_sample_Ainv, (None, 3), "q_sample_Ainv")
        mass = readonly_float_array(self.probability_mass, (points.shape[0],), "probability_mass")
        if np.any(mass < 0.0) or not np.isclose(
            np.sum(mass, dtype=np.float64), 1.0, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("probability_mass must be nonnegative and sum to one")
        object.__setattr__(self, "u_Ainv", u_Ainv)
        object.__setattr__(self, "q_sample_Ainv", points)
        object.__setattr__(self, "probability_mass", mass)
