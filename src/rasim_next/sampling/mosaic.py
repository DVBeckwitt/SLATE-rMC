"""Wrapped axisymmetric mosaic probability under the manuscript measure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _reject_complex(value: ArrayLike, name: str) -> None:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) or (
        supplied.dtype.kind == "O" and any(np.iscomplexobj(item) for item in supplied.flat)
    ):
        raise ValueError(f"{name} must be real")


@dataclass(frozen=True, slots=True)
class WrappedMosaicParameters:
    """Independent wrapped-core width, wrapped-tail width, and tail probability."""

    gaussian_sigma_rad: float
    lorentzian_half_width_rad: float
    lorentzian_probability: float

    def __post_init__(self) -> None:
        for name in (
            "gaussian_sigma_rad",
            "lorentzian_half_width_rad",
            "lorentzian_probability",
        ):
            _reject_complex(getattr(self, name), name)
        values = (
            float(self.gaussian_sigma_rad),
            float(self.lorentzian_half_width_rad),
            float(self.lorentzian_probability),
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("mosaic parameters must be finite")
        if values[0] < 0.0 or values[1] < 0.0:
            raise ValueError("mosaic widths must be nonnegative")
        if not 0.0 <= values[2] <= 1.0:
            raise ValueError("lorentzian_probability must be between zero and one")
        object.__setattr__(self, "gaussian_sigma_rad", values[0])
        object.__setattr__(self, "lorentzian_half_width_rad", values[1])
        object.__setattr__(self, "lorentzian_probability", values[2])

    @property
    def gaussian_fwhm_rad(self) -> float:
        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * self.gaussian_sigma_rad

    @property
    def lorentzian_fwhm_rad(self) -> float:
        return 2.0 * self.lorentzian_half_width_rad

    @property
    def zero_tilt_probability_mass(self) -> float:
        gaussian_mass = 1.0 - self.lorentzian_probability
        lorentzian_mass = self.lorentzian_probability
        return (gaussian_mass if self.gaussian_sigma_rad == 0.0 else 0.0) + (
            lorentzian_mass if self.lorentzian_half_width_rad == 0.0 else 0.0
        )


def _readonly_array(
    value: ArrayLike, dtype: np.dtype, shape: tuple[int | None, ...], name: str
) -> np.ndarray:
    _reject_complex(value, name)
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has invalid shape {array.shape}")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MosaicOrientationBatch:
    """Integrated axisymmetric orientation masses and active crystal rotations."""

    orientation_id: IntArray
    alpha_rad: FloatArray
    azimuth_rad: FloatArray
    rotation_crystal: FloatArray
    probability_mass: FloatArray
    reciprocal_basis_Ainv: FloatArray
    model_id: str

    def __post_init__(self) -> None:
        if np.asarray(self.orientation_id).dtype.kind not in "iu":
            raise ValueError("orientation_id must contain integers")
        ids = _readonly_array(self.orientation_id, np.dtype(np.int64), (None,), "orientation_id")
        size = ids.size
        if np.any(ids < 0) or np.unique(ids).size != size:
            raise ValueError("orientation_id must be nonnegative and unique")
        object.__setattr__(self, "orientation_id", ids)
        for name, shape in (
            ("alpha_rad", (size,)),
            ("azimuth_rad", (size,)),
            ("rotation_crystal", (size, 3, 3)),
            ("probability_mass", (size,)),
            ("reciprocal_basis_Ainv", (3, 3)),
        ):
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), np.dtype(np.float64), shape, name),
            )
        if np.any(self.probability_mass < 0.0) or not np.isclose(
            self.probability_mass.sum(), 1.0, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError("probability_mass must be nonnegative and sum to one")
        if np.any((self.alpha_rad < 0.0) | (self.alpha_rad > np.pi)):
            raise ValueError("alpha_rad must lie in [0, pi]")
        if np.any((self.azimuth_rad < 0.0) | (self.azimuth_rad >= 2.0 * np.pi)):
            raise ValueError("azimuth_rad must lie in [0, 2*pi)")
        if not np.allclose(
            self.rotation_crystal @ np.swapaxes(self.rotation_crystal, 1, 2),
            np.eye(3),
            rtol=0.0,
            atol=1.0e-12,
        ) or not np.allclose(np.linalg.det(self.rotation_crystal), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("rotation_crystal must contain proper rotations")
        if np.isclose(np.linalg.det(self.reciprocal_basis_Ainv), 0.0):
            raise ValueError("reciprocal_basis_Ainv must be nonsingular")
        if self.model_id != "manuscript_axisymmetric_v1":
            raise ValueError("unsupported mosaic orientation model")


def _wrapped_gaussian_density(
    angle_rad: NDArray[np.float64], sigma_rad: float
) -> NDArray[np.float64]:
    if sigma_rad < 1.0:
        image_count = int(np.ceil(8.0 * sigma_rad / (2.0 * np.pi))) + 2
        density = np.zeros_like(angle_rad)
        scale = 1.0 / (np.sqrt(2.0 * np.pi) * sigma_rad)
        for image in range(-image_count, image_count + 1):
            shifted = angle_rad + 2.0 * np.pi * image
            density += scale * np.exp(-0.5 * (shifted / sigma_rad) ** 2)
        return density
    cutoff = np.sqrt(-2.0 * np.log(np.finfo(np.float64).eps / 4.0))
    harmonic_count = max(1, int(np.ceil(cutoff / sigma_rad)))
    density = np.ones_like(angle_rad)
    for harmonic in range(1, harmonic_count + 1):
        density += 2.0 * np.exp(-0.5 * (harmonic * sigma_rad) ** 2) * np.cos(harmonic * angle_rad)
    return density / (2.0 * np.pi)


def _wrapped_lorentzian_density(
    angle_rad: NDArray[np.float64], half_width_rad: float
) -> NDArray[np.float64]:
    if half_width_rad < 50.0:
        sinh_half = np.sinh(0.5 * half_width_rad)
        denominator = sinh_half**2 + np.sin(0.5 * angle_rad) ** 2
        return sinh_half * np.cosh(0.5 * half_width_rad) / (2.0 * np.pi * denominator)
    rho = np.exp(-half_width_rad)
    return (1.0 - rho**2) / (2.0 * np.pi * (1.0 + rho**2 - 2.0 * rho * np.cos(angle_rad)))


def wrapped_mosaic_line_density_rad_inv(
    delta_theta_rad: ArrayLike, parameters: WrappedMosaicParameters
) -> NDArray[np.float64]:
    """Evaluate the continuous signed line density; zero-width mass stays discrete."""

    if parameters.zero_tilt_probability_mass > 0.0:
        raise ValueError("active zero width is a discrete zero-tilt mass, not a point density")
    _reject_complex(delta_theta_rad, "delta_theta_rad")
    angle = np.asarray(delta_theta_rad, dtype=np.float64)
    if not np.all(np.isfinite(angle)):
        raise ValueError("delta_theta_rad must be finite")
    wrapped = np.remainder(angle + np.pi, 2.0 * np.pi) - np.pi
    return _continuous_wrapped_density(wrapped, parameters)


def _continuous_wrapped_density(
    wrapped_angle_rad: NDArray[np.float64], parameters: WrappedMosaicParameters
) -> NDArray[np.float64]:
    tail_probability = parameters.lorentzian_probability
    density = np.zeros_like(wrapped_angle_rad)
    if tail_probability < 1.0 and parameters.gaussian_sigma_rad > 0.0:
        density += (1.0 - tail_probability) * _wrapped_gaussian_density(
            wrapped_angle_rad, parameters.gaussian_sigma_rad
        )
    if tail_probability > 0.0 and parameters.lorentzian_half_width_rad > 0.0:
        density += tail_probability * _wrapped_lorentzian_density(
            wrapped_angle_rad, parameters.lorentzian_half_width_rad
        )
    return density


def _axis_rotation(axis: FloatArray, angle_rad: float) -> FloatArray:
    x, y, z = axis
    cross_matrix = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    cosine = np.cos(angle_rad)
    return (
        cosine * np.eye(3)
        + (1.0 - cosine) * np.outer(axis, axis)
        + np.sin(angle_rad) * cross_matrix
    )


def _positive_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _scale_resolved_panels(width_rad: float, requested_count: int) -> tuple[FloatArray, FloatArray]:
    """Return direct-alpha panels, resolving every dyadic interval from width to pi."""

    segment_edges = [0.0]
    upper = min(width_rad, np.pi)
    while upper < np.pi:
        segment_edges.append(upper)
        upper = min(2.0 * upper, np.pi)
    segment_edges.append(np.pi)

    segment_count = len(segment_edges) - 1
    panel_count = max(requested_count, segment_count)
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


def _direct_alpha_component_quadrature(
    width_rad: float,
    probability: float,
    requested_count: int,
    density: Callable[[FloatArray, float], FloatArray],
) -> tuple[FloatArray, FloatArray]:
    """Integrate one folded component on alpha panels without probability resampling."""

    lower, upper = _scale_resolved_panels(width_rad, requested_count)
    legendre_node, legendre_weight = leggauss(16)
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    alpha = midpoint[:, None] + half_width[:, None] * legendre_node
    mass = probability * 2.0 * density(alpha, width_rad) * half_width[:, None] * legendre_weight
    positive = mass > 0.0
    return alpha[positive], mass[positive]


def manuscript_axisymmetric_v1_orientation_quadrature(
    parameters: WrappedMosaicParameters,
    *,
    reciprocal_basis_Ainv: ArrayLike,
    alpha_cell_count: int,
    azimuth_cell_count: int,
) -> MosaicOrientationBatch:
    """Integrate folded tilt mass and uniform powder azimuth without pole density.

    ``alpha_cell_count`` is the requested minimum panel count per active continuous
    component. Narrow profiles retain at least one direct-alpha panel per dyadic
    width interval, so the actual work count can be larger.
    """

    alpha_count = _positive_count(alpha_cell_count, "alpha_cell_count")
    azimuth_count = _positive_count(azimuth_cell_count, "azimuth_cell_count")
    _reject_complex(reciprocal_basis_Ainv, "reciprocal_basis_Ainv")
    basis = np.array(reciprocal_basis_Ainv, dtype=np.float64, copy=True)
    if (
        basis.shape != (3, 3)
        or not np.all(np.isfinite(basis))
        or np.isclose(np.linalg.det(basis), 0.0)
    ):
        raise ValueError("reciprocal_basis_Ainv must be a finite nonsingular 3 by 3 matrix")
    mean_axis = basis[:, 2] / np.linalg.norm(basis[:, 2])
    reference_axis = basis[:, 0] - np.dot(basis[:, 0], mean_axis) * mean_axis
    reference_norm = np.linalg.norm(reference_axis)
    if np.isclose(reference_norm, 0.0):
        raise ValueError("first and third reciprocal basis vectors must not be parallel")
    reference_axis /= reference_norm
    tilt_axis = np.cross(mean_axis, reference_axis)

    alpha_components: list[FloatArray] = []
    mass_components: list[FloatArray] = []
    tail_probability = parameters.lorentzian_probability
    if parameters.gaussian_sigma_rad > 0.0 and tail_probability < 1.0:
        alpha_nodes, tilt_mass = _direct_alpha_component_quadrature(
            parameters.gaussian_sigma_rad,
            1.0 - tail_probability,
            alpha_count,
            _wrapped_gaussian_density,
        )
        alpha_components.append(alpha_nodes)
        mass_components.append(tilt_mass)
    if parameters.lorentzian_half_width_rad > 0.0 and tail_probability > 0.0:
        alpha_nodes, tilt_mass = _direct_alpha_component_quadrature(
            parameters.lorentzian_half_width_rad,
            tail_probability,
            alpha_count,
            _wrapped_lorentzian_density,
        )
        alpha_components.append(alpha_nodes)
        mass_components.append(tilt_mass)

    atom_mass = parameters.zero_tilt_probability_mass
    if atom_mass > 0.0:
        alpha_components.insert(0, np.zeros(1))
        mass_components.insert(0, np.array([atom_mass]))
    alpha_nodes = np.concatenate(alpha_components)
    tilt_mass = np.concatenate(mass_components)
    if not np.isclose(tilt_mass.sum(), 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("integrated folded tilt mass does not sum to one")

    azimuth_nodes = 2.0 * np.pi * (np.arange(azimuth_count) + 0.5) / azimuth_count
    alpha = np.repeat(alpha_nodes, azimuth_count)
    azimuth = np.tile(azimuth_nodes, alpha_nodes.size)
    probability_mass = np.repeat(tilt_mass / azimuth_count, azimuth_count)
    rotations = np.empty((alpha.size, 3, 3))
    tilt_rotations = tuple(
        _axis_rotation(tilt_axis, float(alpha_value)) for alpha_value in alpha_nodes
    )
    azimuth_rotations = tuple(
        _axis_rotation(mean_axis, float(azimuth_value)) for azimuth_value in azimuth_nodes
    )
    index = 0
    for tilt_rotation in tilt_rotations:
        for azimuth_rotation in azimuth_rotations:
            rotations[index] = azimuth_rotation @ tilt_rotation
            index += 1
    return MosaicOrientationBatch(
        orientation_id=np.arange(alpha.size, dtype=np.int64),
        alpha_rad=alpha,
        azimuth_rad=azimuth,
        rotation_crystal=rotations,
        probability_mass=probability_mass,
        reciprocal_basis_Ainv=basis,
        model_id="manuscript_axisymmetric_v1",
    )
