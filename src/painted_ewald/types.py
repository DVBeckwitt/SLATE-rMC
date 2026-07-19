"""Immutable public records for the mosaic-space milestone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from painted_ewald.validation import (
    finite_scalar,
    integer,
    positive_integer,
    readonly_float_array,
)

FloatArray = NDArray[np.float64]


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

    @property
    def gaussian_fwhm_rad(self) -> float:
        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * self.gaussian_sigma_rad

    @property
    def lorentzian_fwhm_rad(self) -> float:
        return 2.0 * self.lorentzian_half_width_rad


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
