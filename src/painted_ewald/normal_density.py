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

    def cone_average_sr_inv(
        self,
        polar_angle_rad: ArrayLike,
        cone_angle_rad: ArrayLike,
        *,
        quadrature_order: int = 16,
    ) -> NDArray[np.float64]:
        """Average the directed normal law over a unit-mass orientation circle.

        The circle is centred on Q-hat, at polar_angle_rad from the mean normal,
        and has opening cone_angle_rad. Its normals obey
        n dot mean_axis = cos(polar)*cos(cone) + sin(polar)*sin(cone)*cos(psi).
        This integrates the independent uniform crystal azimuth, not an image
        convolution. Lorentzian averaging is analytic; the Gaussian uses complete
        scale-resolved quadrature on [0, pi], with no angular tail truncation.
        """
        reject_complex(polar_angle_rad, "polar_angle_rad")
        reject_complex(cone_angle_rad, "cone_angle_rad")
        polar, cone = np.broadcast_arrays(
            np.asarray(polar_angle_rad, dtype=np.float64),
            np.asarray(cone_angle_rad, dtype=np.float64),
        )
        if any(np.any(~np.isfinite(x)) or np.any((x < 0) | (x > np.pi)) for x in (polar, cone)):
            raise ValueError("polar and cone angles must be finite and lie in [0, pi]")
        if type(quadrature_order) is not int or quadrature_order < 4:
            raise ValueError("quadrature_order must be an integer of at least four")
        shape = polar.shape
        polar, cone = polar.ravel(), cone.ravel()
        result = np.empty(polar.size)
        degenerate = (polar == 0) | (polar == np.pi) | (cone == 0) | (cone == np.pi)
        if np.any(degenerate):
            result[degenerate] = self.directed_density_sr_inv(
                abs(polar[degenerate] - cone[degenerate])
            )
        if np.all(degenerate):
            return result.reshape(shape)
        polar, cone = polar[~degenerate], cone[~degenerate]
        delta = abs(polar - cone)
        b = np.sin(polar) * np.sin(cone)
        out = np.zeros(polar.size)
        eta = self.parameters.lorentzian_probability
        if eta > 0:
            gamma = self.parameters.lorentzian_half_width_rad
            rho, h, numerator = np.exp(-gamma), -np.expm1(-gamma), -np.expm1(-2 * gamma)
            # Factor the denominator at both circle endpoints. Subtracting A²-B²
            # would lose the narrow peak when the cone touches the mean normal.
            low = h * h + 4 * rho * np.sin(delta / 2) ** 2
            high = h * h + 4 * rho * np.sin((polar + cone) / 2) ** 2
            out += (
                eta
                * numerator
                / (2 * np.pi**2 * self.lorentzian_normalization * np.sqrt(low) * np.sqrt(high))
            )
        if eta < 1:
            sigma = self.parameters.gaussian_sigma_rad
            node, weight = np.polynomial.legendre.leggauss(quadrature_order)
            # The minimum cone tilt is delta. Beyond 40 sigma, every wrapped
            # Gaussian image underflows (exp(-800)); there is no Gaussian work.
            # Preserve the original angle arithmetic for subnormal squared widths.
            gaussian_indices = (
                np.arange(polar.size)
                if sigma < np.sqrt(np.finfo(float).tiny)
                else np.flatnonzero(delta < 40 * sigma)
            )
            # Bounded batches keep the angle workspace independent of image size.
            for start in range(0, len(gaussian_indices), 2048):
                selected = gaussian_indices[start : start + 2048]
                bb, dd = b[selected], delta[selected]
                curvature = bb / np.sinc(dd / np.pi)
                scale = np.minimum(
                    np.pi, sigma / np.sqrt(np.maximum(curvature, np.finfo(float).tiny))
                )
                lower = np.zeros(len(selected))
                upper = scale.copy()
                total = np.zeros(len(selected))
                while np.any(lower < np.pi):
                    half = (upper - lower) / 2
                    psi = (upper + lower)[:, None] / 2 + half[:, None] * node
                    sine_squared = np.sin(dd[:, None] / 2) ** 2 + bb[:, None] * np.sin(psi / 2) ** 2
                    sine_squared = np.clip(sine_squared, 0, 1)
                    angle = 2 * np.arctan2(np.sqrt(sine_squared), np.sqrt(1 - sine_squared))
                    total += half * (_wrapped_gaussian_density(angle, sigma) @ weight)
                    lower = upper
                    upper = np.minimum(np.pi, 2 * upper)
                out[selected] += (1 - eta) * total / (np.pi**2 * self.gaussian_normalization)
        result[~degenerate] = out
        return result.reshape(shape)

    def plane_density_sr_inv(self, alpha_rad: ArrayLike) -> NDArray[np.float64]:
        """Antipodal sum per spherical area on the unoriented-normal hemisphere."""
        reject_complex(alpha_rad, "alpha_rad")
        alpha = np.asarray(alpha_rad, dtype=np.float64)
        if np.any((alpha < 0) | (alpha > np.pi / 2)):
            raise ValueError("unoriented-normal tilt must lie in [0, pi/2]")
        return self.directed_density_sr_inv(alpha) + self.directed_density_sr_inv(np.pi - alpha)
