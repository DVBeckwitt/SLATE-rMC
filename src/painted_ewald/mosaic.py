"""Normalized folded-alpha/full-azimuth mosaic space and fixed-u pushforward."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import fsum

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from painted_ewald.rotations import axis_angle_rotation_batch, mosaic_axes
from painted_ewald.types import MosaicParameters, MosaicSlice, Rod
from painted_ewald.validation import (
    finite_scalar,
    proper_rotation,
    readonly_float_array,
    reciprocal_basis,
    reject_complex,
)

FloatArray = NDArray[np.float64]
_DENSITY_RELATIVE_TOLERANCE = 1.0e-15
_DENSITY_MAX_TERMS = 100_000


def _wrapped_gaussian_density(angle_rad: FloatArray, sigma_rad: float) -> FloatArray:
    if sigma_rad >= 1.0:
        density = np.ones_like(angle_rad)
        for harmonic in range(1, _DENSITY_MAX_TERMS + 1):
            amplitude = 2.0 * np.exp(-0.5 * (harmonic * sigma_rad) ** 2)
            density += amplitude * np.cos(harmonic * angle_rad)
            next_amplitude = 2.0 * np.exp(-0.5 * ((harmonic + 1) * sigma_rad) ** 2)
            if next_amplitude <= _DENSITY_RELATIVE_TOLERANCE * np.min(density):
                return density / (2.0 * np.pi)
        raise RuntimeError("wrapped Gaussian Fourier sum did not converge")

    scaled_density = np.exp(-0.5 * (angle_rad / sigma_rad) ** 2)
    for image in range(1, _DENSITY_MAX_TERMS + 1):
        offset = 2.0 * np.pi * image
        scaled_density += np.exp(-0.5 * ((angle_rad + offset) / sigma_rad) ** 2)
        scaled_density += np.exp(-0.5 * ((angle_rad - offset) / sigma_rad) ** 2)
        next_offset = 2.0 * np.pi * (image + 1)
        next_pair = np.exp(-0.5 * ((angle_rad + next_offset) / sigma_rad) ** 2)
        next_pair += np.exp(-0.5 * ((angle_rad - next_offset) / sigma_rad) ** 2)
        if np.all(next_pair <= _DENSITY_RELATIVE_TOLERANCE * scaled_density):
            return scaled_density / (np.sqrt(2.0 * np.pi) * sigma_rad)
    raise RuntimeError("wrapped Gaussian image sum did not converge")


def _wrapped_lorentzian_density(angle_rad: FloatArray, half_width_rad: float) -> FloatArray:
    rho = np.exp(-half_width_rad)
    one_minus_rho = -np.expm1(-half_width_rad)
    numerator = -np.expm1(-2.0 * half_width_rad)
    denominator = 2.0 * np.pi * (one_minus_rho**2 + 4.0 * rho * np.sin(0.5 * angle_rad) ** 2)
    return numerator / denominator


def wrapped_mosaic_line_density_rad_inv(
    theta_rad: ArrayLike, parameters: MosaicParameters
) -> FloatArray:
    """Evaluate the continuous signed wrapped density; atoms remain discrete."""

    reject_complex(theta_rad, "theta_rad")
    theta = np.asarray(theta_rad, dtype=np.float64)
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta_rad must be finite")
    wrapped = np.remainder(theta + np.pi, 2.0 * np.pi) - np.pi
    density = np.zeros_like(wrapped)
    eta = parameters.lorentzian_probability
    if eta < 1.0 and parameters.gaussian_sigma_rad > 0.0:
        density += (1.0 - eta) * _wrapped_gaussian_density(wrapped, parameters.gaussian_sigma_rad)
    if eta > 0.0 and parameters.lorentzian_half_width_rad > 0.0:
        density += eta * _wrapped_lorentzian_density(wrapped, parameters.lorentzian_half_width_rad)
    return np.asarray(density, dtype=np.float64)


def _scale_resolved_panels(width_rad: float, minimum_count: int) -> tuple[FloatArray, FloatArray]:
    segment_edges = [0.0]
    upper = min(width_rad, np.pi)
    while upper < np.pi:
        segment_edges.append(upper)
        upper = min(2.0 * upper, np.pi)
    segment_edges.append(np.pi)

    segment_count = len(segment_edges) - 1
    panel_count = max(minimum_count, segment_count)
    quotient, remainder = divmod(panel_count, segment_count)
    subdivisions = np.full(segment_count, quotient, dtype=np.int64)
    subdivisions[:remainder] += 1

    lower: list[float] = []
    upper_edges: list[float] = []
    for start, stop, count in zip(segment_edges[:-1], segment_edges[1:], subdivisions, strict=True):
        edges = np.linspace(start, stop, int(count) + 1)
        lower.extend(edges[:-1])
        upper_edges.extend(edges[1:])
    return np.asarray(lower), np.asarray(upper_edges)


def _component_quadrature(
    *,
    width_rad: float,
    probability: float,
    panel_count: int,
    gauss_order: int,
    density: Callable[[FloatArray, float], FloatArray],
) -> tuple[FloatArray, FloatArray]:
    lower, upper = _scale_resolved_panels(width_rad, panel_count)
    gauss_node, gauss_weight = leggauss(gauss_order)
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    alpha = midpoint[:, None] + half_width[:, None] * gauss_node
    mass = probability * 2.0 * density(alpha, width_rad) * half_width[:, None] * gauss_weight
    positive = mass > 0.0
    return alpha[positive], mass[positive]


def _midpoint_azimuth_nodes(count: int, phase_rad: float) -> FloatArray:
    return np.remainder(
        phase_rad + 2.0 * np.pi * (np.arange(count, dtype=np.float64) + 0.5) / count,
        2.0 * np.pi,
    )


def _nested_azimuth_nodes(count: int, phase_rad: float) -> FloatArray:
    turns = np.empty(count, dtype=np.float64)
    turns[0] = 0.0
    filled = 1
    denominator = 2.0
    while filled < count:
        added = min(filled, count - filled)
        turns[filled : filled + added] = (2.0 * np.arange(added) + 1.0) / denominator
        filled += added
        denominator *= 2.0
    return np.remainder(phase_rad + 2.0 * np.pi * turns, 2.0 * np.pi)


@dataclass(frozen=True, slots=True)
class MosaicSpace:
    """Immutable deterministic quadrature of the crystallite orientation measure."""

    reciprocal_basis_Ainv: FloatArray
    crystal_to_sample: FloatArray
    parameters: MosaicParameters
    orientation_id: NDArray[np.int64]
    alpha_rad: FloatArray
    beta_rad: FloatArray
    rotation_crystal: FloatArray
    probability_mass: FloatArray
    quadrature_rule_id: str = "midpoint_uniform_beta.v1"
    model_id: str = "painted_ewald.folded_alpha_full_beta.v1"

    def __post_init__(self) -> None:
        basis = reciprocal_basis(self.reciprocal_basis_Ainv)
        crystal_to_sample = proper_rotation(self.crystal_to_sample)
        if not isinstance(self.parameters, MosaicParameters):
            raise TypeError("parameters must be MosaicParameters")
        ids = np.array(self.orientation_id, dtype=np.int64, copy=True, order="C")
        if ids.ndim != 1 or np.any(ids < 0) or np.unique(ids).size != ids.size:
            raise ValueError("orientation_id must be a one-dimensional unique nonnegative key")
        ids.setflags(write=False)
        size = ids.size
        alpha = readonly_float_array(self.alpha_rad, (size,), "alpha_rad")
        beta = readonly_float_array(self.beta_rad, (size,), "beta_rad")
        rotations = readonly_float_array(self.rotation_crystal, (size, 3, 3), "rotation_crystal")
        mass = readonly_float_array(self.probability_mass, (size,), "probability_mass")
        if np.any((alpha < 0.0) | (alpha > np.pi)):
            raise ValueError("alpha_rad must lie in [0, pi]")
        if np.any((beta < 0.0) | (beta >= 2.0 * np.pi)):
            raise ValueError("beta_rad must lie in [0, 2*pi)")
        if np.any(mass < 0.0) or abs(fsum(mass) - 1.0) > 1.0e-12:
            raise ValueError("probability_mass must be nonnegative and sum to one")
        tolerance = 256.0 * np.finfo(np.float64).eps
        if not np.allclose(
            rotations @ np.swapaxes(rotations, 1, 2), np.eye(3), rtol=0.0, atol=tolerance
        ) or not np.allclose(np.linalg.det(rotations), 1.0, rtol=0.0, atol=tolerance):
            raise ValueError("rotation_crystal must contain proper rotations")
        if self.model_id != "painted_ewald.folded_alpha_full_beta.v1":
            raise ValueError("unsupported mosaic-space model")
        if self.quadrature_rule_id not in {
            "midpoint_uniform_beta.v1",
            "nested_uniform_beta.v1",
        }:
            raise ValueError("unsupported mosaic quadrature rule")
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)
        object.__setattr__(self, "crystal_to_sample", crystal_to_sample)
        object.__setattr__(self, "orientation_id", ids)
        object.__setattr__(self, "alpha_rad", alpha)
        object.__setattr__(self, "beta_rad", beta)
        object.__setattr__(self, "rotation_crystal", rotations)
        object.__setattr__(self, "probability_mass", mass)

    def mosaic_slice(self, *, rod: Rod, u_Ainv: float) -> MosaicSlice:
        """Push one complete fixed-u affine-rod point through every orientation."""

        if not isinstance(rod, Rod):
            raise TypeError("rod must be a Rod")
        u_value = finite_scalar(u_Ainv, "u_Ainv")
        mean_axis, _ = mosaic_axes(self.reciprocal_basis_Ainv)
        q_parallel = (
            rod.h * self.reciprocal_basis_Ainv[:, 0] + rod.k * self.reciprocal_basis_Ainv[:, 1]
        )
        q_crystal = q_parallel + u_value * mean_axis
        rotated_crystal = np.einsum("nij,j->ni", self.rotation_crystal, q_crystal, optimize=True)
        q_sample = rotated_crystal @ self.crystal_to_sample.T
        return MosaicSlice(
            rod=rod,
            u_Ainv=u_value,
            q_sample_Ainv=q_sample,
            probability_mass=self.probability_mass,
        )


def _build_mosaic_space(
    *,
    reciprocal_basis_Ainv: ArrayLike,
    crystal_to_sample: ArrayLike,
    parameters: MosaicParameters,
    azimuth_rule: str,
) -> MosaicSpace:
    if not isinstance(parameters, MosaicParameters):
        raise TypeError("parameters must be MosaicParameters")
    basis = reciprocal_basis(reciprocal_basis_Ainv)
    crystal_rotation = proper_rotation(crystal_to_sample)
    mean_axis, tilt_axis = mosaic_axes(basis)

    alpha_components: list[FloatArray] = []
    mass_components: list[FloatArray] = []
    eta = parameters.lorentzian_probability
    if parameters.gaussian_sigma_rad > 0.0 and eta < 1.0:
        alpha, mass = _component_quadrature(
            width_rad=parameters.gaussian_sigma_rad,
            probability=1.0 - eta,
            panel_count=parameters.alpha_panel_count,
            gauss_order=parameters.alpha_gauss_order,
            density=_wrapped_gaussian_density,
        )
        alpha_components.append(alpha)
        mass_components.append(mass)
    if parameters.lorentzian_half_width_rad > 0.0 and eta > 0.0:
        alpha, mass = _component_quadrature(
            width_rad=parameters.lorentzian_half_width_rad,
            probability=eta,
            panel_count=parameters.alpha_panel_count,
            gauss_order=parameters.alpha_gauss_order,
            density=_wrapped_lorentzian_density,
        )
        alpha_components.append(alpha)
        mass_components.append(mass)
    atom_mass = parameters.zero_tilt_probability_mass
    if atom_mass > 0.0:
        alpha_components.insert(0, np.zeros(1))
        mass_components.insert(0, np.array([atom_mass]))
    if not alpha_components:
        raise ValueError("mosaic probability has no active component")

    alpha_nodes = np.concatenate(alpha_components)
    alpha_mass = np.concatenate(mass_components)
    mass_sum = fsum(alpha_mass)
    if abs(mass_sum - 1.0) > 1.0e-10:
        raise ValueError("integrated folded tilt mass does not sum to one")
    alpha_mass = alpha_mass / mass_sum

    phase_rad = float(np.remainder(parameters.azimuth_phase_rad, 2.0 * np.pi))
    if azimuth_rule == "midpoint":
        beta_nodes = _midpoint_azimuth_nodes(parameters.azimuth_count, phase_rad)
        quadrature_rule_id = "midpoint_uniform_beta.v1"
    elif azimuth_rule == "nested":
        beta_nodes = _nested_azimuth_nodes(parameters.azimuth_count, phase_rad)
        quadrature_rule_id = "nested_uniform_beta.v1"
    else:
        raise ValueError("unsupported azimuth quadrature rule")

    tilt_rotations = axis_angle_rotation_batch(tilt_axis, alpha_nodes)
    beta_rotations = axis_angle_rotation_batch(mean_axis, beta_nodes)
    rotations = np.einsum("bij,ajk->abik", beta_rotations, tilt_rotations, optimize=True).reshape(
        -1, 3, 3
    )
    alpha = np.repeat(alpha_nodes, parameters.azimuth_count)
    beta = np.tile(beta_nodes, alpha_nodes.size)
    probability_mass = np.repeat(alpha_mass / parameters.azimuth_count, parameters.azimuth_count)
    if abs(fsum(probability_mass) - 1.0) > 1.0e-12:
        raise ValueError("orientation probability mass does not sum to one")
    return MosaicSpace(
        reciprocal_basis_Ainv=basis,
        crystal_to_sample=crystal_rotation,
        parameters=parameters,
        orientation_id=np.arange(alpha.size, dtype=np.int64),
        alpha_rad=alpha,
        beta_rad=beta,
        rotation_crystal=rotations,
        probability_mass=probability_mass,
        quadrature_rule_id=quadrature_rule_id,
    )


def build_mosaic_space(
    *,
    reciprocal_basis_Ainv: ArrayLike,
    crystal_to_sample: ArrayLike,
    parameters: MosaicParameters,
) -> MosaicSpace:
    """Build the specification's midpoint-azimuth deterministic mosaic quadrature."""

    return _build_mosaic_space(
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        crystal_to_sample=crystal_to_sample,
        parameters=parameters,
        azimuth_rule="midpoint",
    )
