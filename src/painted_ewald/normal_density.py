"""One spherical-area mosaic law for directed rods and unoriented m=0 planes."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald.mosaic import (
    _component_quadrature,
    _wrapped_gaussian_density,
    _wrapped_lorentzian_density,
)
from painted_ewald.types import MosaicParameters
from painted_ewald.validation import reject_complex


@dataclass(frozen=True, slots=True)
class SphericalMosaicDensity:
    """Normalized restricted orientation law, not a complete SO(3) ODF.

    Directed tilt lies in [0, pi], with uniform azimuth. Each wrapped component
    is normalized separately against spherical area, so eta is its mass fraction.
    Rods retain directed tilt and the existing tied crystal rotation. Only m=0
    folds antipodal plane normals; its two signed structure strengths stay separate.
    """

    parameters: MosaicParameters
    gaussian_normalization: float = field(init=False)
    lorentzian_normalization: float = field(init=False)
    model_id: str = field(default="spherical_wrapped_tied_orientation.v1", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, MosaicParameters):
            raise TypeError("parameters must be MosaicParameters")
        if self.parameters.zero_tilt_probability_mass > 0.0:
            raise ValueError("finite spherical density cannot contain a zero-width atom")
        for name, width, density in (
            (
                "gaussian_normalization",
                self.parameters.gaussian_sigma_rad,
                _wrapped_gaussian_density,
            ),
            (
                "lorentzian_normalization",
                self.parameters.lorentzian_half_width_rad,
                _wrapped_lorentzian_density,
            ),
        ):
            normalization = 1.0
            if width > 0.0:
                alpha, mass = _component_quadrature(
                    width_rad=width,
                    probability=1.0,
                    panel_count=24,
                    gauss_order=24,
                    density=density,
                )
                normalization = float(np.sum(mass * np.sin(alpha)))
                if not np.isfinite(normalization) or normalization <= 0.0:
                    raise ValueError("spherical component normalization must be positive")
            object.__setattr__(self, name, normalization)

    def directed_density_sr_inv(self, alpha_rad: ArrayLike) -> NDArray[np.float64]:
        """Density per spherical area on the full directed-normal sphere."""
        reject_complex(alpha_rad, "alpha_rad")
        alpha = np.asarray(alpha_rad, dtype=np.float64)
        if not np.all(np.isfinite(alpha)) or np.any((alpha < 0) | (alpha > np.pi)):
            raise ValueError("directed tilt must lie in [0, pi]")
        result = np.zeros_like(alpha)
        eta = self.parameters.lorentzian_probability
        for probability, width, normalization, density in (
            (
                1 - eta,
                self.parameters.gaussian_sigma_rad,
                self.gaussian_normalization,
                _wrapped_gaussian_density,
            ),
            (
                eta,
                self.parameters.lorentzian_half_width_rad,
                self.lorentzian_normalization,
                _wrapped_lorentzian_density,
            ),
        ):
            if probability > 0:
                result += probability * density(alpha, width) / (np.pi * normalization)
        return result

    def latent_density_rad_inv2(self, alpha_rad: ArrayLike) -> NDArray[np.float64]:
        """Probability density with respect to d(alpha) d(beta), before pushforward."""
        return self.directed_density_sr_inv(alpha_rad) * np.sin(np.asarray(alpha_rad))

    def plane_density_sr_inv(self, alpha_rad: ArrayLike) -> NDArray[np.float64]:
        """Antipodal sum per spherical area on the unoriented-normal hemisphere."""
        reject_complex(alpha_rad, "alpha_rad")
        alpha = np.asarray(alpha_rad, dtype=np.float64)
        if np.any((alpha < 0) | (alpha > np.pi / 2)):
            raise ValueError("unoriented-normal tilt must lie in [0, pi/2]")
        return self.directed_density_sr_inv(alpha) + self.directed_density_sr_inv(np.pi - alpha)
