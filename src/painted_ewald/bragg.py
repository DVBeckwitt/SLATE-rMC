"""Continuous pre-Ewald mosaic pushforward of reciprocal rods and strengths."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald.mosaic import (
    MosaicSpace,
    build_mosaic_space,
    wrapped_mosaic_line_density_rad_inv,
)
from painted_ewald.rotations import axis_angle_rotation_batch, mosaic_axes
from painted_ewald.types import BasisBoundStrengthModel, MosaicParameters, Rod
from painted_ewald.validation import (
    finite_scalar,
    proper_rotation,
    readonly_float_array,
    reciprocal_basis,
    reject_complex,
)

FloatArray = NDArray[np.float64]


def _latent_arrays(
    alpha_rad: ArrayLike,
    beta_rad: ArrayLike,
    u_Ainv: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    for value, name in (
        (alpha_rad, "alpha_rad"),
        (beta_rad, "beta_rad"),
        (u_Ainv, "u_Ainv"),
    ):
        reject_complex(value, name)
    alpha, beta, axial = np.broadcast_arrays(
        np.asarray(alpha_rad, dtype=np.float64),
        np.asarray(beta_rad, dtype=np.float64),
        np.asarray(u_Ainv, dtype=np.float64),
    )
    if not all(np.all(np.isfinite(value)) for value in (alpha, beta, axial)):
        raise ValueError("latent Bragg coordinates must be finite")
    if np.any((alpha < 0.0) | (alpha > np.pi)):
        raise ValueError("alpha_rad must lie in [0, pi]")
    if np.any((beta < 0.0) | (beta >= 2.0 * np.pi)):
        raise ValueError("beta_rad must lie in [0, 2*pi)")
    return alpha, beta, axial


def _map_tied_rotation_arrays(
    rod: Rod,
    basis: FloatArray,
    crystal_to_sample: FloatArray,
    mean_axis: FloatArray,
    tilt_axis: FloatArray,
    alpha: FloatArray,
    beta: FloatArray,
    axial: FloatArray,
) -> FloatArray:
    shape = alpha.shape
    tilt = axis_angle_rotation_batch(tilt_axis, alpha.reshape(-1))
    azimuth = axis_angle_rotation_batch(mean_axis, beta.reshape(-1))
    rotations = np.einsum("nij,njk->nik", azimuth, tilt, optimize=True)
    q_parallel = rod.h * basis[:, 0] + rod.k * basis[:, 1]
    q_crystal = q_parallel + axial.reshape(-1, 1) * mean_axis
    rotated = np.einsum("nij,nj->ni", rotations, q_crystal, optimize=True)
    return np.asarray((rotated @ crystal_to_sample.T).reshape((*shape, 3)), dtype=np.float64)


def map_tied_rotation_latent(
    *,
    rod: Rod,
    reciprocal_basis_Ainv: ArrayLike,
    crystal_to_sample: ArrayLike,
    alpha_rad: ArrayLike,
    beta_rad: ArrayLike,
    u_Ainv: ArrayLike,
) -> FloatArray:
    """Map one reciprocal rod through the authoritative tied mosaic rotation."""

    if not isinstance(rod, Rod):
        raise TypeError("rod must be a Rod")
    basis = reciprocal_basis(reciprocal_basis_Ainv)
    sample_rotation = proper_rotation(crystal_to_sample)
    alpha, beta, axial = _latent_arrays(alpha_rad, beta_rad, u_Ainv)
    mean_axis, tilt_axis = mosaic_axes(basis)
    return _map_tied_rotation_arrays(
        rod,
        basis,
        sample_rotation,
        mean_axis,
        tilt_axis,
        alpha,
        beta,
        axial,
    )


@dataclass(frozen=True, slots=True)
class BraggSpaceConfig:
    """Complete elastic-reach domain for a continuous pre-Ewald rod measure."""

    reciprocal_basis_Ainv: FloatArray
    crystal_to_sample: FloatArray
    rods: tuple[Rod, ...]
    mosaic: MosaicParameters
    k_norm_Ainv: float

    def __post_init__(self) -> None:
        basis = reciprocal_basis(self.reciprocal_basis_Ainv)
        crystal_to_sample = proper_rotation(self.crystal_to_sample)
        rods = tuple(self.rods)
        if not rods or not all(isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one Rod")
        rod_keys = tuple((rod.h, rod.k) for rod in rods)
        if len(set(rod_keys)) != len(rod_keys):
            raise ValueError("rods must not repeat a physical (h, k) line")
        if not isinstance(self.mosaic, MosaicParameters):
            raise TypeError("mosaic must be MosaicParameters")
        k_norm = finite_scalar(self.k_norm_Ainv, "k_norm_Ainv")
        if k_norm <= 0.0:
            raise ValueError("k_norm_Ainv must be positive")

        mean_axis, _ = mosaic_axes(basis)
        q_norm_max = 2.0 * k_norm
        tolerance = 256.0 * np.finfo(np.float64).eps * max(q_norm_max, 1.0)
        for rod in rods:
            q_parallel = rod.h * basis[:, 0] + rod.k * basis[:, 1]
            perpendicular = q_parallel - np.dot(q_parallel, mean_axis) * mean_axis
            if np.linalg.norm(perpendicular) > q_norm_max + tolerance:
                raise ValueError(f"rod ({rod.h}, {rod.k}) does not enter the elastic-reach domain")
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)
        object.__setattr__(self, "crystal_to_sample", crystal_to_sample)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "k_norm_Ainv", k_norm)

    @property
    def q_norm_max_Ainv(self) -> float:
        """Maximum elastic momentum transfer, ``2 * |ki|``."""

        return 2.0 * self.k_norm_Ainv

    def rod_u_bounds_Ainv(self, rod: Rod) -> tuple[float, float]:
        """Return one configured rod's complete interval inside the elastic ball."""

        if not isinstance(rod, Rod):
            raise TypeError("rod must be a Rod")
        try:
            configured = next(
                candidate for candidate in self.rods if (candidate.h, candidate.k) == (rod.h, rod.k)
            )
        except StopIteration as error:
            raise ValueError(f"rod ({rod.h}, {rod.k}) is not configured") from error
        mean_axis, _ = mosaic_axes(self.reciprocal_basis_Ainv)
        q_parallel = (
            configured.h * self.reciprocal_basis_Ainv[:, 0]
            + configured.k * self.reciprocal_basis_Ainv[:, 1]
        )
        axial_offset = float(np.dot(q_parallel, mean_axis))
        perpendicular = q_parallel - axial_offset * mean_axis
        half_width = float(
            np.sqrt(
                max(
                    0.0,
                    self.q_norm_max_Ainv**2 - float(np.dot(perpendicular, perpendicular)),
                )
            )
        )
        return -axial_offset - half_width, -axial_offset + half_width


@dataclass(frozen=True, slots=True)
class LatentBraggIntensity:
    """Continuous rod measure evaluated at arbitrary ``(alpha, beta, u)``.

    ``intensity_density_A2_rad2_inv`` is the exact-u coefficient of
    ``dB_r = density * d alpha * d beta * du``. Thus ``B_r`` has units of
    angstrom after axial integration.
    Any zero-width mosaic component is a Dirac mass and is reported separately
    by :attr:`MosaicBraggSpace.zero_tilt_probability_mass`.
    """

    rod: Rod
    alpha_rad: FloatArray
    beta_rad: FloatArray
    L: FloatArray
    u_Ainv: FloatArray
    q_sample_Ainv: FloatArray
    mosaic_probability_density_rad2_inv: FloatArray
    rod_strength_A2: FloatArray
    intensity_density_A2_rad2_inv: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.rod, Rod):
            raise TypeError("rod must be a Rod")
        reject_complex(self.alpha_rad, "alpha_rad")
        alpha = np.array(self.alpha_rad, dtype=np.float64, copy=True, order="C")
        shape = alpha.shape
        arrays = {
            "beta_rad": self.beta_rad,
            "L": self.L,
            "u_Ainv": self.u_Ainv,
            "mosaic_probability_density_rad2_inv": self.mosaic_probability_density_rad2_inv,
            "rod_strength_A2": self.rod_strength_A2,
            "intensity_density_A2_rad2_inv": self.intensity_density_A2_rad2_inv,
        }
        validated: dict[str, FloatArray] = {}
        for name, value in arrays.items():
            reject_complex(value, name)
            array = np.array(value, dtype=np.float64, copy=True, order="C")
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            array.setflags(write=False)
            validated[name] = array
        reject_complex(self.q_sample_Ainv, "q_sample_Ainv")
        points = np.array(self.q_sample_Ainv, dtype=np.float64, copy=True, order="C")
        if points.shape != (*shape, 3) or not np.all(np.isfinite(points)):
            raise ValueError(f"q_sample_Ainv must be finite with shape {(*shape, 3)}")
        if not np.all(np.isfinite(alpha)) or np.any((alpha < 0.0) | (alpha > np.pi)):
            raise ValueError("alpha_rad must be finite and lie in [0, pi]")
        beta = validated["beta_rad"]
        if np.any((beta < 0.0) | (beta >= 2.0 * np.pi)):
            raise ValueError("beta_rad must lie in [0, 2*pi)")
        density = validated["mosaic_probability_density_rad2_inv"]
        strength = validated["rod_strength_A2"]
        intensity = validated["intensity_density_A2_rad2_inv"]
        if np.any(density < 0.0) or np.any(strength < 0.0) or np.any(intensity < 0.0):
            raise ValueError(
                "mosaic density, rod strength, and intensity density must be nonnegative"
            )
        expected = density * self.rod.population * strength
        scale = max(float(np.max(expected, initial=0.0)), 1.0)
        tolerance = 512.0 * np.finfo(np.float64).eps * scale
        if not np.allclose(intensity, expected, rtol=0.0, atol=tolerance):
            raise ValueError(
                "intensity density must equal mosaic density times population and strength"
            )
        alpha.setflags(write=False)
        points.setflags(write=False)
        object.__setattr__(self, "alpha_rad", alpha)
        object.__setattr__(self, "q_sample_Ainv", points)
        for name, value in validated.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class WeightedMosaicSlice:
    """One exact-L rod slice after multiplying by its scalar structure strength."""

    rod: Rod
    L: float
    u_Ainv: float
    q_sample_Ainv: FloatArray
    probability_mass: FloatArray
    rod_strength_A2: float
    intensity_weight_A2: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.rod, Rod):
            raise TypeError("rod must be a Rod")
        ell = finite_scalar(self.L, "L")
        u_value = finite_scalar(self.u_Ainv, "u_Ainv")
        points = readonly_float_array(self.q_sample_Ainv, (None, 3), "q_sample_Ainv")
        mass = readonly_float_array(self.probability_mass, (points.shape[0],), "probability_mass")
        strength = finite_scalar(self.rod_strength_A2, "rod_strength_A2")
        weight = readonly_float_array(
            self.intensity_weight_A2,
            (points.shape[0],),
            "intensity_weight_A2",
        )
        if strength < 0.0 or np.any(mass < 0.0) or np.any(weight < 0.0):
            raise ValueError("mosaic mass, rod strength, and intensity weight must be nonnegative")
        if abs(fsum(mass) - 1.0) > 1.0e-12:
            raise ValueError("probability_mass must sum to one")
        expected = mass * self.rod.population * strength
        scale = max(self.rod.population * strength, 1.0)
        tolerance = 512.0 * np.finfo(np.float64).eps * scale
        if not np.allclose(weight, expected, rtol=0.0, atol=tolerance):
            raise ValueError("intensity_weight_A2 must equal mosaic mass times rod strength")
        object.__setattr__(self, "L", ell)
        object.__setattr__(self, "u_Ainv", u_value)
        object.__setattr__(self, "q_sample_Ainv", points)
        object.__setattr__(self, "probability_mass", mass)
        object.__setattr__(self, "rod_strength_A2", strength)
        object.__setattr__(self, "intensity_weight_A2", weight)


@dataclass(frozen=True, slots=True)
class BraggFamilySlice:
    """Exact-m, exact-L sum retaining every independently weighted physical rod."""

    family_m: int
    rod_slices: tuple[WeightedMosaicSlice, ...]
    total_intensity_weight_A2: float

    def __post_init__(self) -> None:
        if isinstance(self.family_m, bool) or not isinstance(self.family_m, (int, np.integer)):
            raise ValueError("family_m must be an integer")
        family_m = int(self.family_m)
        slices = tuple(self.rod_slices)
        if family_m < 0 or not slices:
            raise ValueError("family_m must be nonnegative and contain at least one rod slice")
        if not all(isinstance(item, WeightedMosaicSlice) for item in slices):
            raise TypeError("rod_slices must contain WeightedMosaicSlice values")
        if any(item.rod.family_m != family_m for item in slices):
            raise ValueError("every rod slice must belong to family_m")
        if any(item.L != slices[0].L for item in slices[1:]):
            raise ValueError("every rod slice must share one exact L")
        keys = tuple((item.rod.h, item.rod.k) for item in slices)
        if len(set(keys)) != len(keys):
            raise ValueError("a family slice cannot repeat a physical rod")
        total = finite_scalar(self.total_intensity_weight_A2, "total_intensity_weight_A2")
        expected = fsum(fsum(item.intensity_weight_A2) for item in slices)
        tolerance = 512.0 * np.finfo(np.float64).eps * max(expected, 1.0)
        if total < 0.0 or abs(total - expected) > tolerance:
            raise ValueError("family total must equal the sum of its physical rod weights")
        object.__setattr__(self, "family_m", family_m)
        object.__setattr__(self, "rod_slices", slices)
        object.__setattr__(self, "total_intensity_weight_A2", total)


class MosaicBraggSpace:
    """Immutable parametric Bragg-space model before Ewald conditioning."""

    __slots__ = (
        "_b3_norm_Ainv",
        "_config",
        "_mean_axis",
        "_mosaic_space",
        "_strength_model",
        "_tilt_axis",
    )

    def __init__(
        self,
        config: BraggSpaceConfig,
        strength_model: BasisBoundStrengthModel,
    ) -> None:
        if not isinstance(config, BraggSpaceConfig):
            raise TypeError("config must be BraggSpaceConfig")
        if not callable(getattr(strength_model, "evaluate", None)):
            raise TypeError("strength_model must implement evaluate")
        strength_basis_value = getattr(strength_model, "reciprocal_basis_Ainv", None)
        if strength_basis_value is None:
            raise TypeError("strength_model must declare reciprocal_basis_Ainv")
        strength_basis = reciprocal_basis(strength_basis_value)
        scale = max(float(np.linalg.norm(config.reciprocal_basis_Ainv)), 1.0)
        tolerance = 256.0 * np.finfo(np.float64).eps * scale
        if not np.allclose(
            strength_basis,
            config.reciprocal_basis_Ainv,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("strength-model and Bragg-space reciprocal bases do not match")
        mosaic_space = build_mosaic_space(
            reciprocal_basis_Ainv=config.reciprocal_basis_Ainv,
            crystal_to_sample=config.crystal_to_sample,
            parameters=config.mosaic,
        )
        mean_axis, tilt_axis = mosaic_axes(config.reciprocal_basis_Ainv)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_strength_model", strength_model)
        object.__setattr__(self, "_mosaic_space", mosaic_space)
        object.__setattr__(self, "_mean_axis", mean_axis)
        object.__setattr__(self, "_tilt_axis", tilt_axis)
        object.__setattr__(
            self,
            "_b3_norm_Ainv",
            float(np.linalg.norm(config.reciprocal_basis_Ainv[:, 2])),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("MosaicBraggSpace is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("MosaicBraggSpace is immutable")

    @property
    def config(self) -> BraggSpaceConfig:
        return self._config

    @property
    def strength_model(self) -> BasisBoundStrengthModel:
        return self._strength_model

    @property
    def mosaic_space(self) -> MosaicSpace:
        return self._mosaic_space

    @property
    def zero_tilt_probability_mass(self) -> float:
        """Probability carried by a discrete alpha=0 Dirac component."""

        return self._config.mosaic.zero_tilt_probability_mass

    def _configured_rod(self, rod: Rod) -> Rod:
        if not isinstance(rod, Rod):
            raise TypeError("rod must be a Rod")
        try:
            return next(
                candidate
                for candidate in self._config.rods
                if (candidate.h, candidate.k) == (rod.h, rod.k)
            )
        except StopIteration as error:
            raise ValueError(f"rod ({rod.h}, {rod.k}) is not configured") from error

    def rod_u_bounds_Ainv(self, rod: Rod) -> tuple[float, float]:
        """Return the complete axial interval inside the elastic-reach ball."""

        return self._config.rod_u_bounds_Ainv(rod)

    def map_latent(
        self,
        *,
        rod: Rod,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
        u_Ainv: ArrayLike,
    ) -> FloatArray:
        """Evaluate the continuous tied-rotation map at arbitrary latent coordinates."""

        configured = self._configured_rod(rod)
        alpha, beta, axial = _latent_arrays(alpha_rad, beta_rad, u_Ainv)
        return _map_tied_rotation_arrays(
            configured,
            self._config.reciprocal_basis_Ainv,
            self._config.crystal_to_sample,
            self._mean_axis,
            self._tilt_axis,
            alpha,
            beta,
            axial,
        )

    def _strength_profile(self, rod: Rod, ell: FloatArray) -> FloatArray:
        evaluate_profile = getattr(self._strength_model, "evaluate_profile", None)
        if callable(evaluate_profile):
            raw_strength = evaluate_profile(
                rod=rod,
                L=ell,
                k_norm_Ainv=self._config.k_norm_Ainv,
            )
            reject_complex(raw_strength, "strength profile")
            strength = np.asarray(raw_strength, dtype=np.float64)
            if strength.shape != ell.shape:
                raise ValueError("strength profile must match the latent-coordinate shape")
        else:
            raw_strength = [
                self._strength_model.evaluate(
                    rod=rod,
                    L=float(value),
                    k_norm_Ainv=self._config.k_norm_Ainv,
                )
                for value in ell.flat
            ]
            reject_complex(raw_strength, "strength profile")
            strength = np.asarray(raw_strength, dtype=np.float64).reshape(ell.shape)
        if not np.all(np.isfinite(strength)):
            raise ValueError("strength profile must be finite")
        if np.any(strength < 0.0):
            raise ValueError("rod strength must be nonnegative")
        return strength

    def _latent_mosaic_density_arrays(
        self,
        *,
        rod: Rod,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
        u_Ainv: ArrayLike,
    ) -> tuple[Rod, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        configured = self._configured_rod(rod)
        alpha, beta, axial = _latent_arrays(alpha_rad, beta_rad, u_Ainv)
        lower, upper = self.rod_u_bounds_Ainv(configured)
        tolerance = 256.0 * np.finfo(np.float64).eps * max(abs(lower), abs(upper), 1.0)
        if np.any((axial < lower - tolerance) | (axial > upper + tolerance)):
            raise ValueError("u_Ainv lies outside the elastic-reach domain for this rod")
        ell = axial / self._b3_norm_Ainv
        mosaic_density = (
            2.0 * wrapped_mosaic_line_density_rad_inv(alpha, self._config.mosaic) / (2.0 * np.pi)
        )
        return configured, alpha, beta, axial, ell, mosaic_density

    def evaluate_latent_mosaic_density(
        self,
        *,
        rod: Rod,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
        u_Ainv: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Evaluate latent ``L`` and mosaic density without structure strength."""

        _, _, _, _, ell, mosaic_density = self._latent_mosaic_density_arrays(
            rod=rod,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
            u_Ainv=u_Ainv,
        )
        ell.setflags(write=False)
        mosaic_density.setflags(write=False)
        return ell, mosaic_density

    def evaluate_latent(
        self,
        *,
        rod: Rod,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
        u_Ainv: ArrayLike,
    ) -> LatentBraggIntensity:
        """Evaluate continuous CIF strength times mosaic density off any quadrature grid."""

        configured, alpha, beta, axial, ell, mosaic_density = self._latent_mosaic_density_arrays(
            rod=rod,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
            u_Ainv=u_Ainv,
        )
        points = self.map_latent(
            rod=configured,
            alpha_rad=alpha,
            beta_rad=beta,
            u_Ainv=axial,
        )
        strength = self._strength_profile(configured, ell)
        return LatentBraggIntensity(
            rod=configured,
            alpha_rad=alpha,
            beta_rad=beta,
            L=ell,
            u_Ainv=axial,
            q_sample_Ainv=points,
            mosaic_probability_density_rad2_inv=mosaic_density,
            rod_strength_A2=strength,
            intensity_density_A2_rad2_inv=(mosaic_density * configured.population * strength),
        )

    def weighted_mosaic_slice(self, *, rod: Rod, L: float) -> WeightedMosaicSlice:
        """Multiply one exact-L quadrature slice before any family sum."""

        configured = self._configured_rod(rod)
        ell = finite_scalar(L, "L")
        u_value = self._b3_norm_Ainv * ell
        lower, upper = self.rod_u_bounds_Ainv(configured)
        tolerance = 256.0 * np.finfo(np.float64).eps * max(abs(lower), abs(upper), 1.0)
        if u_value < lower - tolerance or u_value > upper + tolerance:
            raise ValueError("L lies outside the elastic-reach domain for this rod")
        strength = finite_scalar(
            self._strength_model.evaluate(
                rod=configured,
                L=ell,
                k_norm_Ainv=self._config.k_norm_Ainv,
            ),
            "rod strength",
        )
        if strength < 0.0:
            raise ValueError("rod strength must be nonnegative")
        mosaic_slice = self._mosaic_space.mosaic_slice(rod=configured, u_Ainv=u_value)
        return WeightedMosaicSlice(
            rod=configured,
            L=ell,
            u_Ainv=u_value,
            q_sample_Ainv=mosaic_slice.q_sample_Ainv,
            probability_mass=mosaic_slice.probability_mass,
            rod_strength_A2=strength,
            intensity_weight_A2=(mosaic_slice.probability_mass * configured.population * strength),
        )

    def weighted_family_slice(self, *, family_m: int, L: float) -> BraggFamilySlice:
        """Return an exact family sum after independent per-rod mosaic weighting."""

        if isinstance(family_m, bool) or not isinstance(family_m, (int, np.integer)):
            raise ValueError("family_m must be an integer")
        selected = tuple(rod for rod in self._config.rods if rod.family_m == int(family_m))
        if not selected:
            raise ValueError(f"family_m={family_m} is not configured")
        slices = tuple(self.weighted_mosaic_slice(rod=rod, L=L) for rod in selected)
        return BraggFamilySlice(
            family_m=int(family_m),
            rod_slices=slices,
            total_intensity_weight_A2=fsum(fsum(item.intensity_weight_A2) for item in slices),
        )


__all__ = [
    "BraggFamilySlice",
    "BraggSpaceConfig",
    "LatentBraggIntensity",
    "MosaicBraggSpace",
    "WeightedMosaicSlice",
    "map_tied_rotation_latent",
]
