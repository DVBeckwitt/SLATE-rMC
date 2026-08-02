"""Continuous latent-coordinate intensity on an analytic Ewald surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald.bragg import MosaicBraggSpace, map_tied_rotation_latent
from painted_ewald.ewald import _solve_batched_infinite_rod_ewald
from painted_ewald.types import Rod, RootStatus
from painted_ewald.validation import finite_scalar, readonly_float_array

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EwaldLatentGeometry:
    """One analytic rod/root map at arbitrary folded-alpha/full-beta coordinates."""

    rod: Rod
    branch: int
    alpha_rad: FloatArray
    beta_rad: FloatArray
    u_Ainv: FloatArray
    L: FloatArray
    q_sample_Ainv: FloatArray
    kf_sample_Ainv: FloatArray
    ewald_residual_Ainv: FloatArray
    status: NDArray[np.str_]
    valid: NDArray[np.bool_] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.rod, Rod):
            raise TypeError("rod must be a Rod")
        if self.branch not in {0, 1, 2}:
            raise ValueError("branch must be 0, 1, or 2")
        alpha = np.array(self.alpha_rad, dtype=np.float64, copy=True, order="C")
        shape = alpha.shape
        if not np.all(np.isfinite(alpha)) or np.any((alpha < 0.0) | (alpha > np.pi)):
            raise ValueError("alpha_rad must be finite and lie in [0, pi]")
        beta = readonly_float_array(self.beta_rad, shape, "beta_rad")
        if np.any((beta < 0.0) | (beta >= 2.0 * np.pi)):
            raise ValueError("beta_rad must lie in [0, 2*pi)")
        scalar_arrays = {
            "u_Ainv": self.u_Ainv,
            "L": self.L,
            "ewald_residual_Ainv": self.ewald_residual_Ainv,
        }
        validated = {
            name: readonly_float_array(value, shape, name) for name, value in scalar_arrays.items()
        }
        q_sample = readonly_float_array(self.q_sample_Ainv, (*shape, 3), "q_sample_Ainv")
        kf_sample = readonly_float_array(self.kf_sample_Ainv, (*shape, 3), "kf_sample_Ainv")
        supplied_status = np.asarray(self.status, dtype="U32")
        if supplied_status.shape != shape:
            raise ValueError(f"status must have shape {shape}")
        allowed_status = np.asarray(tuple(item.value for item in RootStatus), dtype="U32")
        recognized = np.isin(supplied_status, allowed_status)
        if not np.all(recognized):
            invalid = np.unique(supplied_status[~recognized]).tolist()
            raise ValueError(f"status contains unsupported values: {invalid}")
        status = np.array(supplied_status, dtype="U32", copy=True, order="C")
        valid = status == RootStatus.REGULAR
        if np.any(validated["ewald_residual_Ainv"] < 0.0):
            raise ValueError("ewald_residual_Ainv must be nonnegative")
        alpha.setflags(write=False)
        status.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "alpha_rad", alpha)
        object.__setattr__(self, "beta_rad", beta)
        for name, value in validated.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "q_sample_Ainv", q_sample)
        object.__setattr__(self, "kf_sample_Ainv", kf_sample)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True, slots=True)
class EwaldLatentIntensity:
    """Continuous Bragg strength after applying the Ewald coarea factor once."""

    geometry: EwaldLatentGeometry
    mosaic_probability_density_rad2_inv: FloatArray
    rod_strength_A2: FloatArray
    latent_intensity_density_A2_rad2_inv: FloatArray
    coarea_jacobian: FloatArray
    coating_intensity_density_A2_rad2_inv: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, EwaldLatentGeometry):
            raise TypeError("geometry must be EwaldLatentGeometry")
        shape = self.geometry.alpha_rad.shape
        arrays = {
            "mosaic_probability_density_rad2_inv": self.mosaic_probability_density_rad2_inv,
            "rod_strength_A2": self.rod_strength_A2,
            "latent_intensity_density_A2_rad2_inv": self.latent_intensity_density_A2_rad2_inv,
            "coarea_jacobian": self.coarea_jacobian,
            "coating_intensity_density_A2_rad2_inv": (self.coating_intensity_density_A2_rad2_inv),
        }
        validated = {
            name: readonly_float_array(value, shape, name) for name, value in arrays.items()
        }
        if any(np.any(value < 0.0) for value in validated.values()):
            raise ValueError("Ewald intensity factors must be nonnegative")
        valid = self.geometry.valid
        expected = validated["latent_intensity_density_A2_rad2_inv"] * validated["coarea_jacobian"]
        scale = max(
            float(np.max(expected, initial=0.0)),
            np.finfo(np.float64).tiny,
        )
        tolerance = 1024.0 * np.finfo(np.float64).eps * scale
        if not np.allclose(
            validated["coating_intensity_density_A2_rad2_inv"],
            expected,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("coating intensity must apply the Ewald coarea factor exactly once")
        if any(np.any(value[~valid] != 0.0) for value in validated.values()):
            raise ValueError("invalid latent coordinates cannot carry intensity")
        for name, value in validated.items():
            object.__setattr__(self, name, value)


def _evaluate_infinite_rod_geometry(
    *,
    rod: Rod,
    branch: int,
    reciprocal_basis_Ainv: ArrayLike,
    crystal_to_sample: ArrayLike,
    ki_sample_Ainv: ArrayLike,
    alpha_rad: ArrayLike,
    beta_rad: ArrayLike,
    root_tolerance_rel: float,
    residual_tolerance_rel: float,
) -> tuple[EwaldLatentGeometry, FloatArray]:
    if not isinstance(rod, Rod):
        raise TypeError("rod must be a Rod")
    if branch not in {0, 1, 2}:
        raise ValueError("branch must be 0, 1, or 2")
    incident = readonly_float_array(ki_sample_Ainv, (3,), "ki_sample_Ainv")
    incident_norm = sqrt(float(np.dot(incident, incident)))
    if incident_norm == 0.0:
        raise ValueError("ki_sample_Ainv must be nonzero")
    root_tolerance = finite_scalar(root_tolerance_rel, "root_tolerance_rel")
    residual_tolerance = finite_scalar(residual_tolerance_rel, "residual_tolerance_rel")
    if root_tolerance < 0.0 or residual_tolerance < 0.0:
        raise ValueError("root and residual tolerances must be nonnegative")
    alpha, beta = np.broadcast_arrays(
        np.asarray(alpha_rad, dtype=np.float64),
        np.asarray(beta_rad, dtype=np.float64),
    )
    q0 = map_tied_rotation_latent(
        rod=rod,
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        crystal_to_sample=crystal_to_sample,
        alpha_rad=alpha,
        beta_rad=beta,
        u_Ainv=0.0,
    )
    q1 = map_tied_rotation_latent(
        rod=rod,
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        crystal_to_sample=crystal_to_sample,
        alpha_rad=alpha,
        beta_rad=beta,
        u_Ainv=1.0,
    )
    basis = np.asarray(reciprocal_basis_Ainv, dtype=np.float64)
    b3_norm_Ainv = float(np.linalg.norm(basis[:, 2]))
    shape = alpha.shape
    result = _solve_batched_infinite_rod_ewald(
        incident=incident,
        incident_norm=incident_norm,
        q0=q0.reshape(-1, 3),
        q0_parallel_Ainv=None,
        direction=(q1 - q0).reshape(-1, 3),
        b3_norm_Ainv=b3_norm_Ainv,
        rod_is_m0=rod.family_m == 0,
        root_tolerance_rel=root_tolerance,
        residual_tolerance_rel=residual_tolerance,
    )
    selected = next((root for root in result.roots if root.branch == branch), None)
    if selected is None:
        raise ValueError(f"branch {branch} is not defined for rod ({rod.h}, {rod.k})")
    size = alpha.size
    status = np.full(size, RootStatus.REGULAR.value, dtype="U32")
    status[result.no_root] = RootStatus.NO_ROOT.value
    status[result.tangent] = RootStatus.TANGENT.value
    status[result.collapsed_direct] = RootStatus.COLLAPSED_DIRECT.value
    regular = selected.orientation_index
    u_Ainv = np.zeros(size, dtype=np.float64)
    ell = np.zeros(size, dtype=np.float64)
    q_sample = np.zeros((size, 3), dtype=np.float64)
    kf_sample = np.zeros((size, 3), dtype=np.float64)
    residual = np.zeros(size, dtype=np.float64)
    coarea = np.zeros(size, dtype=np.float64)
    u_Ainv[regular] = selected.u_Ainv
    ell[regular] = selected.L
    q_sample[regular] = selected.q_sample_Ainv
    kf_sample[regular] = selected.kf_sample_Ainv
    residual[regular] = selected.ewald_residual_Ainv
    coarea[regular] = selected.coarea_jacobian
    geometry = EwaldLatentGeometry(
        rod=rod,
        branch=branch,
        alpha_rad=alpha,
        beta_rad=beta,
        u_Ainv=u_Ainv.reshape(shape),
        L=ell.reshape(shape),
        q_sample_Ainv=q_sample.reshape((*shape, 3)),
        kf_sample_Ainv=kf_sample.reshape((*shape, 3)),
        ewald_residual_Ainv=residual.reshape(shape),
        status=status.reshape(shape),
    )
    coarea = coarea.reshape(shape)
    coarea.setflags(write=False)
    return geometry, coarea


def evaluate_infinite_rod_ewald_geometry(
    *,
    rod: Rod,
    branch: int,
    reciprocal_basis_Ainv: ArrayLike,
    crystal_to_sample: ArrayLike,
    ki_sample_Ainv: ArrayLike,
    alpha_rad: ArrayLike,
    beta_rad: ArrayLike,
    root_tolerance_rel: float = 0.0,
    residual_tolerance_rel: float = 512.0 * np.finfo(np.float64).eps,
) -> EwaldLatentGeometry:
    """Evaluate analytic rod/Ewald geometry without strength or mosaic state."""

    if not isinstance(rod, Rod):
        raise TypeError("rod must be a Rod")
    if rod.family_m == 0:
        if branch != 0:
            raise ValueError("an m=0 rod requires branch 0")
    elif branch not in {1, 2}:
        raise ValueError("a nonzero rod requires branch 1 or 2")
    geometry, _ = _evaluate_infinite_rod_geometry(
        rod=rod,
        branch=branch,
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        crystal_to_sample=crystal_to_sample,
        ki_sample_Ainv=ki_sample_Ainv,
        alpha_rad=alpha_rad,
        beta_rad=beta_rad,
        root_tolerance_rel=root_tolerance_rel,
        residual_tolerance_rel=residual_tolerance_rel,
    )
    return geometry


class ContinuousEwaldCoating:
    """Callable zero-width rod coating in continuous curvilinear coordinates.

    The Bragg-space configuration owns the structure-factor wavelength through
    ``k_norm_Ainv``. The independent ``ki_sample_Ainv`` supplied here owns the
    refracted internal Ewald geometry.
    """

    __slots__ = (
        "_ki_sample_Ainv",
        "_residual_tolerance_rel",
        "_root_tolerance_rel",
        "_space",
    )

    def __init__(
        self,
        bragg_space: MosaicBraggSpace,
        *,
        ki_sample_Ainv: ArrayLike,
        root_tolerance_rel: float = 0.0,
        residual_tolerance_rel: float = 512.0 * np.finfo(np.float64).eps,
    ) -> None:
        if not isinstance(bragg_space, MosaicBraggSpace):
            raise TypeError("bragg_space must be MosaicBraggSpace")
        if bragg_space.zero_tilt_probability_mass > 0.0:
            raise ValueError(
                "ContinuousEwaldCoating does not implement a zero-tilt Dirac component"
            )
        incident = readonly_float_array(ki_sample_Ainv, (3,), "ki_sample_Ainv")
        incident_norm = sqrt(float(np.dot(incident, incident)))
        if incident_norm == 0.0:
            raise ValueError("ki_sample_Ainv must be nonzero")
        root_tolerance = finite_scalar(root_tolerance_rel, "root_tolerance_rel")
        residual_tolerance = finite_scalar(residual_tolerance_rel, "residual_tolerance_rel")
        if root_tolerance < 0.0 or residual_tolerance < 0.0:
            raise ValueError("root and residual tolerances must be nonnegative")
        object.__setattr__(self, "_space", bragg_space)
        object.__setattr__(self, "_ki_sample_Ainv", incident)
        object.__setattr__(self, "_root_tolerance_rel", root_tolerance)
        object.__setattr__(self, "_residual_tolerance_rel", residual_tolerance)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ContinuousEwaldCoating is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ContinuousEwaldCoating is immutable")

    @property
    def bragg_space(self) -> MosaicBraggSpace:
        return self._space

    @property
    def ki_sample_Ainv(self) -> FloatArray:
        return self._ki_sample_Ainv

    @property
    def root_tolerance_rel(self) -> float:
        return self._root_tolerance_rel

    @property
    def residual_tolerance_rel(self) -> float:
        return self._residual_tolerance_rel

    def _configured_rod(self, rod: Rod) -> Rod:
        if not isinstance(rod, Rod):
            raise TypeError("rod must be a Rod")
        for configured in self._space.config.rods:
            if (configured.h, configured.k) == (rod.h, rod.k):
                return configured
        raise ValueError(f"rod ({rod.h}, {rod.k}) is not configured")

    def _evaluate_geometry(
        self,
        *,
        rod: Rod,
        branch: int,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
    ) -> tuple[EwaldLatentGeometry, FloatArray]:
        configured = self._configured_rod(rod)
        return _evaluate_infinite_rod_geometry(
            rod=configured,
            branch=branch,
            reciprocal_basis_Ainv=self._space.config.reciprocal_basis_Ainv,
            crystal_to_sample=self._space.config.crystal_to_sample,
            ki_sample_Ainv=self._ki_sample_Ainv,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
            root_tolerance_rel=self._root_tolerance_rel,
            residual_tolerance_rel=self._residual_tolerance_rel,
        )

    def evaluate_geometry(
        self,
        *,
        rod: Rod,
        branch: int,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
    ) -> EwaldLatentGeometry:
        """Evaluate only the analytic Ewald geometry for one non-specular rod/root."""

        configured = self._configured_rod(rod)
        if configured.family_m == 0:
            raise ValueError("use evaluate_specular_geometry for the m=0 rod")
        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2 for a nonzero rod")
        geometry, _ = self._evaluate_geometry(
            rod=configured,
            branch=branch,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
        )
        return geometry

    def evaluate_latent(
        self,
        *,
        rod: Rod,
        branch: int,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
    ) -> EwaldLatentIntensity:
        """Evaluate one non-specular rod/root at arbitrary latent coordinates."""

        configured = self._configured_rod(rod)
        if configured.family_m == 0:
            raise ValueError("m=0 intensity is excluded without physical direct-beam support")
        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2 for a nonzero rod")
        geometry, coarea = self._evaluate_geometry(
            rod=configured,
            branch=branch,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
        )
        shape = geometry.alpha_rad.shape
        mosaic_density = np.zeros(shape, dtype=np.float64)
        strength = np.zeros(shape, dtype=np.float64)
        latent_density = np.zeros(shape, dtype=np.float64)
        valid = geometry.valid
        if np.any(valid):
            latent = self._space.evaluate_latent(
                rod=configured,
                alpha_rad=geometry.alpha_rad[valid],
                beta_rad=geometry.beta_rad[valid],
                u_Ainv=geometry.u_Ainv[valid],
            )
            mosaic_density[valid] = latent.mosaic_probability_density_rad2_inv
            strength[valid] = latent.rod_strength_A2
            latent_density[valid] = latent.intensity_density_A2_rad2_inv
        return EwaldLatentIntensity(
            geometry=geometry,
            mosaic_probability_density_rad2_inv=mosaic_density,
            rod_strength_A2=strength,
            latent_intensity_density_A2_rad2_inv=latent_density,
            coarea_jacobian=coarea,
            coating_intensity_density_A2_rad2_inv=latent_density * coarea,
        )

    def evaluate_specular_geometry(
        self,
        *,
        rod: Rod,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
    ) -> EwaldLatentGeometry:
        """Return branch-0 geometry without assigning the divergent m=0 intensity."""

        configured = self._configured_rod(rod)
        if configured.family_m != 0:
            raise ValueError("specular geometry requires the (0, 0) rod")
        return self._evaluate_geometry(
            rod=configured,
            branch=0,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
        )[0]


__all__ = [
    "ContinuousEwaldCoating",
    "EwaldLatentGeometry",
    "EwaldLatentIntensity",
    "evaluate_infinite_rod_ewald_geometry",
]
