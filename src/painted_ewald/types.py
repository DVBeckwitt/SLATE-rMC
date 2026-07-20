"""Immutable public records for normalized mosaic and painted Ewald geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from math import ceil, fsum, sqrt
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from painted_ewald.validation import (
    finite_scalar,
    integer,
    positive_integer,
    proper_rotation,
    readonly_float_array,
    reciprocal_basis,
)

FloatArray = NDArray[np.float64]


def _hexagonal_family_indices(family_m: int) -> set[tuple[int, int]]:
    index_bound = 0 if family_m == 0 else ceil(2.0 * sqrt(family_m / 3.0)) + 1
    return {
        (h, k)
        for h in range(-index_bound, index_bound + 1)
        for k in range(-index_bound, index_bound + 1)
        if h * h + h * k + k * k == family_m
    }


class RootStatus(StrEnum):
    """Classification of one infinite reciprocal-rod/Ewald intersection."""

    NO_ROOT = "no_root"
    TANGENT = "tangent"
    COLLAPSED_DIRECT = "collapsed_direct"
    REGULAR = "regular"


class PaintMeasure(StrEnum):
    """Selected measure carried by each painted Ewald-sphere point."""

    MOSAIC_PUSHFORWARD = "mosaic_pushforward"
    COAREA_INTENSITY = "coarea_intensity"


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

    @property
    def gaussian_fwhm_rad(self) -> float:
        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * self.gaussian_sigma_rad

    @property
    def lorentzian_fwhm_rad(self) -> float:
        return 2.0 * self.lorentzian_half_width_rad


@dataclass(frozen=True, slots=True)
class RasterParameters:
    """Resolution of the uniform-mu, uniform-azimuth sphere texture."""

    mu_bin_count: int = 256
    phi_bin_count: int = 512

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mu_bin_count", positive_integer(self.mu_bin_count, "mu_bin_count")
        )
        object.__setattr__(
            self, "phi_bin_count", positive_integer(self.phi_bin_count, "phi_bin_count")
        )


@dataclass(frozen=True, slots=True)
class ForwardPolicy:
    """Physical exclusion around the direct beam for m=0 coarea intensity."""

    q_min_Ainv: float | None = None

    def __post_init__(self) -> None:
        if self.q_min_Ainv is None:
            return
        q_min = finite_scalar(self.q_min_Ainv, "q_min_Ainv")
        if q_min <= 0.0:
            raise ValueError("q_min_Ainv must be positive")
        object.__setattr__(self, "q_min_Ainv", q_min)


@dataclass(frozen=True, slots=True)
class PainterConfig:
    """Complete immutable geometry, mosaic, measure, and raster configuration."""

    reciprocal_basis_Ainv: FloatArray
    crystal_to_sample: FloatArray
    rods: tuple[Rod, ...]
    mosaic: MosaicParameters
    raster: RasterParameters = field(default_factory=RasterParameters)
    measure: PaintMeasure = PaintMeasure.MOSAIC_PUSHFORWARD
    forward_policy: ForwardPolicy = field(default_factory=ForwardPolicy)
    root_tolerance_rel: float = 256.0 * np.finfo(np.float64).eps
    residual_tolerance_rel: float = 512.0 * np.finfo(np.float64).eps

    def __post_init__(self) -> None:
        basis = reciprocal_basis(self.reciprocal_basis_Ainv)
        crystal_to_sample = proper_rotation(self.crystal_to_sample)
        try:
            rods = tuple(self.rods)
        except TypeError as error:
            raise ValueError("rods must be a nonempty sequence of Rod values") from error
        if not rods or not all(isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must be a nonempty sequence of Rod values")
        rod_keys = [(rod.h, rod.k) for rod in rods]
        if len(set(rod_keys)) != len(rod_keys):
            raise ValueError("rods must not repeat a physical (h, k) identity")
        if not isinstance(self.mosaic, MosaicParameters):
            raise TypeError("mosaic must be MosaicParameters")
        if not isinstance(self.raster, RasterParameters):
            raise TypeError("raster must be RasterParameters")
        measure = PaintMeasure(self.measure)
        if not isinstance(self.forward_policy, ForwardPolicy):
            raise TypeError("forward_policy must be ForwardPolicy")
        root_tolerance = finite_scalar(self.root_tolerance_rel, "root_tolerance_rel")
        residual_tolerance = finite_scalar(self.residual_tolerance_rel, "residual_tolerance_rel")
        if root_tolerance <= 0.0 or residual_tolerance <= 0.0:
            raise ValueError("root and residual tolerances must be positive")
        if (
            measure is PaintMeasure.COAREA_INTENSITY
            and any(rod.family_m == 0 for rod in rods)
            and self.forward_policy.q_min_Ainv is None
        ):
            raise ValueError("coarea_intensity with m=0 requires a physical q_min_Ainv")
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)
        object.__setattr__(self, "crystal_to_sample", crystal_to_sample)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "measure", measure)
        object.__setattr__(self, "root_tolerance_rel", root_tolerance)
        object.__setattr__(self, "residual_tolerance_rel", residual_tolerance)


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


@dataclass(frozen=True, slots=True)
class MosaicOrientation:
    """One identified node of the deterministic orientation quadrature."""

    orientation_id: int
    alpha_rad: float
    beta_rad: float
    rotation_crystal: FloatArray
    probability_mass: float

    def __post_init__(self) -> None:
        orientation_id = integer(self.orientation_id, "orientation_id")
        alpha = finite_scalar(self.alpha_rad, "alpha_rad")
        beta = finite_scalar(self.beta_rad, "beta_rad")
        rotation = proper_rotation(self.rotation_crystal, "rotation_crystal")
        mass = finite_scalar(self.probability_mass, "probability_mass")
        if orientation_id < 0:
            raise ValueError("orientation_id must be nonnegative")
        if not 0.0 <= alpha <= np.pi:
            raise ValueError("alpha_rad must lie in [0, pi]")
        if not 0.0 <= beta < 2.0 * np.pi:
            raise ValueError("beta_rad must lie in [0, 2*pi)")
        if mass < 0.0:
            raise ValueError("probability_mass must be nonnegative")
        object.__setattr__(self, "orientation_id", orientation_id)
        object.__setattr__(self, "alpha_rad", alpha)
        object.__setattr__(self, "beta_rad", beta)
        object.__setattr__(self, "rotation_crystal", rotation)
        object.__setattr__(self, "probability_mass", mass)


@dataclass(frozen=True, slots=True)
class PaintedPoint:
    """One identified, weighted rod intersection on the Ewald sphere."""

    rod_h: int
    rod_k: int
    family_m: int
    branch: int
    orientation_id: int
    alpha_rad: float
    beta_rad: float
    u_Ainv: float
    L: float
    q_sample_Ainv: FloatArray
    kf_sample_Ainv: FloatArray
    orientation_mass: float
    rod_population: float
    rod_strength: float
    base_weight: float
    coarea_jacobian: float | None
    weight: float
    ewald_residual_Ainv: float

    def __post_init__(self) -> None:
        rod_h = integer(self.rod_h, "rod_h")
        rod_k = integer(self.rod_k, "rod_k")
        family_m = integer(self.family_m, "family_m")
        branch = integer(self.branch, "branch")
        orientation_id = integer(self.orientation_id, "orientation_id")
        if family_m != rod_h * rod_h + rod_h * rod_k + rod_k * rod_k:
            raise ValueError("family_m must match rod_h and rod_k")
        if (family_m == 0 and branch != 0) or (family_m != 0 and branch not in {1, 2}):
            raise ValueError("branch is inconsistent with the rod family")
        if orientation_id < 0:
            raise ValueError("orientation_id must be nonnegative")
        alpha = finite_scalar(self.alpha_rad, "alpha_rad")
        beta = finite_scalar(self.beta_rad, "beta_rad")
        if not 0.0 <= alpha <= np.pi or not 0.0 <= beta < 2.0 * np.pi:
            raise ValueError("orientation angles are outside their canonical domains")
        u_Ainv = finite_scalar(self.u_Ainv, "u_Ainv")
        l_coordinate = finite_scalar(self.L, "L")
        q_sample = readonly_float_array(self.q_sample_Ainv, (3,), "q_sample_Ainv")
        kf_sample = readonly_float_array(self.kf_sample_Ainv, (3,), "kf_sample_Ainv")
        scalar_names = (
            "orientation_mass",
            "rod_population",
            "rod_strength",
            "base_weight",
            "weight",
            "ewald_residual_Ainv",
        )
        scalar_values = {name: finite_scalar(getattr(self, name), name) for name in scalar_names}
        if any(value < 0.0 for value in scalar_values.values()):
            raise ValueError("point masses, strengths, weights, and residual must be nonnegative")
        jacobian = self.coarea_jacobian
        if jacobian is not None:
            jacobian = finite_scalar(jacobian, "coarea_jacobian")
            if jacobian <= 0.0:
                raise ValueError("coarea_jacobian must be positive when present")
        object.__setattr__(self, "rod_h", rod_h)
        object.__setattr__(self, "rod_k", rod_k)
        object.__setattr__(self, "family_m", family_m)
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "orientation_id", orientation_id)
        object.__setattr__(self, "alpha_rad", alpha)
        object.__setattr__(self, "beta_rad", beta)
        object.__setattr__(self, "u_Ainv", u_Ainv)
        object.__setattr__(self, "L", l_coordinate)
        object.__setattr__(self, "q_sample_Ainv", q_sample)
        object.__setattr__(self, "kf_sample_Ainv", kf_sample)
        for name, value in scalar_values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "coarea_jacobian", jacobian)


@dataclass(frozen=True, slots=True)
class MassLedger:
    """Selected painted weight plus finite pre-coarea classifications.

    ``painted_weight`` has the configured measure. Forward exclusions store
    finite base weight before coarea; collapsed, tangent, and no-root fields
    store orientation-population mass. A suppressed algebraic direct root is
    counted diagnostically and receives no second copy of probability mass.
    """

    painted_weight: float
    collapsed_direct_base_mass: float
    tangent_base_mass: float
    no_root_base_mass: float
    forward_excluded_base_mass: float
    suppressed_algebraic_direct_root_count: int

    def __post_init__(self) -> None:
        for name in (
            "painted_weight",
            "collapsed_direct_base_mass",
            "tangent_base_mass",
            "no_root_base_mass",
            "forward_excluded_base_mass",
        ):
            value = finite_scalar(getattr(self, name), name)
            if value < 0.0:
                raise ValueError("mass-ledger values must be nonnegative")
            object.__setattr__(self, name, value)
        count = integer(
            self.suppressed_algebraic_direct_root_count,
            "suppressed_algebraic_direct_root_count",
        )
        if count < 0:
            raise ValueError("suppressed_algebraic_direct_root_count must be nonnegative")
        object.__setattr__(self, "suppressed_algebraic_direct_root_count", count)

    @classmethod
    def combine(cls, ledgers: tuple[MassLedger, ...]) -> MassLedger:
        """Combine independent ledgers without changing any declared measure."""

        if not all(isinstance(ledger, cls) for ledger in ledgers):
            raise TypeError("ledgers must contain only MassLedger values")
        return cls(
            painted_weight=fsum(ledger.painted_weight for ledger in ledgers),
            collapsed_direct_base_mass=fsum(
                ledger.collapsed_direct_base_mass for ledger in ledgers
            ),
            tangent_base_mass=fsum(ledger.tangent_base_mass for ledger in ledgers),
            no_root_base_mass=fsum(ledger.no_root_base_mass for ledger in ledgers),
            forward_excluded_base_mass=fsum(
                ledger.forward_excluded_base_mass for ledger in ledgers
            ),
            suppressed_algebraic_direct_root_count=sum(
                ledger.suppressed_algebraic_direct_root_count for ledger in ledgers
            ),
        )


def _mass_ledgers_agree(left: MassLedger, right: MassLedger) -> bool:
    float_fields = (
        "painted_weight",
        "collapsed_direct_base_mass",
        "tangent_base_mass",
        "no_root_base_mass",
        "forward_excluded_base_mass",
    )
    for name in float_fields:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        tolerance = (
            1024.0
            * np.finfo(np.float64).eps
            * max(
                left_value,
                right_value,
                1.0,
            )
        )
        if abs(left_value - right_value) > tolerance:
            return False
    return (
        left.suppressed_algebraic_direct_root_count == right.suppressed_algebraic_direct_root_count
    )


@dataclass(frozen=True, slots=True)
class SphereTexture:
    """Conservative equal-solid-angle mass texture in final-direction coordinates."""

    mu_edges: FloatArray
    phi_edges_rad: FloatArray
    mass: FloatArray
    density_per_sr: FloatArray

    def __post_init__(self) -> None:
        mu_edges = readonly_float_array(self.mu_edges, (None,), "mu_edges")
        phi_edges = readonly_float_array(self.phi_edges_rad, (None,), "phi_edges_rad")
        if mu_edges.size < 2 or phi_edges.size < 2:
            raise ValueError("sphere-texture edge arrays require at least two values")
        if np.any(np.diff(mu_edges) <= 0.0) or np.any(np.diff(phi_edges) <= 0.0):
            raise ValueError("sphere-texture edges must be strictly increasing")
        tolerance = 64.0 * np.finfo(np.float64).eps
        if not np.allclose(mu_edges[[0, -1]], [-1.0, 1.0], rtol=0.0, atol=tolerance):
            raise ValueError("mu_edges must span [-1, 1]")
        if not np.allclose(phi_edges[[0, -1]], [0.0, 2.0 * np.pi], rtol=0.0, atol=tolerance):
            raise ValueError("phi_edges_rad must span [0, 2*pi]")
        shape = (mu_edges.size - 1, phi_edges.size - 1)
        mass = readonly_float_array(self.mass, shape, "mass")
        density = readonly_float_array(self.density_per_sr, shape, "density_per_sr")
        if np.any(mass < 0.0) or np.any(density < 0.0):
            raise ValueError("sphere-texture mass and density must be nonnegative")
        solid_angle = np.diff(mu_edges)[:, None] * np.diff(phi_edges)[None, :]
        if not np.allclose(density * solid_angle, mass, rtol=tolerance, atol=tolerance):
            raise ValueError("density_per_sr must equal mass per bin solid angle")
        object.__setattr__(self, "mu_edges", mu_edges)
        object.__setattr__(self, "phi_edges_rad", phi_edges)
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "density_per_sr", density)


@dataclass(frozen=True, slots=True)
class PaintedEwaldSphere:
    """Authoritative weighted points and their conservative visualization texture."""

    ki_sample_Ainv: FloatArray
    center_q_sample_Ainv: FloatArray
    radius_Ainv: float
    points: tuple[PaintedPoint, ...]
    texture: SphereTexture
    ledger: MassLedger

    def __post_init__(self) -> None:
        incident = readonly_float_array(self.ki_sample_Ainv, (3,), "ki_sample_Ainv")
        center = readonly_float_array(self.center_q_sample_Ainv, (3,), "center_q_sample_Ainv")
        radius = finite_scalar(self.radius_Ainv, "radius_Ainv")
        if radius <= 0.0:
            raise ValueError("radius_Ainv must be positive")
        tolerance = 1024.0 * np.finfo(np.float64).eps * max(radius, 1.0)
        if abs(np.linalg.norm(incident) - radius) > tolerance:
            raise ValueError("radius_Ainv must equal the incident wavevector norm")
        if not np.allclose(center, -incident, rtol=0.0, atol=tolerance):
            raise ValueError("center_q_sample_Ainv must equal -ki_sample_Ainv")
        points = tuple(self.points)
        if not all(isinstance(point, PaintedPoint) for point in points):
            raise TypeError("points must contain only PaintedPoint values")
        if not isinstance(self.texture, SphereTexture):
            raise TypeError("texture must be SphereTexture")
        if not isinstance(self.ledger, MassLedger):
            raise TypeError("ledger must be MassLedger")
        point_total = fsum(point.weight for point in points)
        texture_total = fsum(self.texture.mass.ravel())
        mass_tolerance = 1024.0 * np.finfo(np.float64).eps * max(point_total, 1.0)
        if abs(point_total - self.ledger.painted_weight) > mass_tolerance:
            raise ValueError("ledger painted_weight does not equal the point-weight total")
        if abs(point_total - texture_total) > mass_tolerance:
            raise ValueError("sphere texture does not conserve point weight")
        object.__setattr__(self, "ki_sample_Ainv", incident)
        object.__setattr__(self, "center_q_sample_Ainv", center)
        object.__setattr__(self, "radius_Ainv", radius)
        object.__setattr__(self, "points", points)


@dataclass(frozen=True, slots=True)
class BranchCoatingSummary:
    """Retained count and selected painted weight for one analytic root branch."""

    branch: int
    retained_root_count: int
    painted_weight: float

    def __post_init__(self) -> None:
        branch = integer(self.branch, "branch")
        count = integer(self.retained_root_count, "retained_root_count")
        weight = finite_scalar(self.painted_weight, "painted_weight")
        if branch not in {0, 1, 2}:
            raise ValueError("branch must be 0, 1, or 2")
        if count < 0 or weight < 0.0:
            raise ValueError("coating branch count and weight must be nonnegative")
        if count == 0 and weight != 0.0:
            raise ValueError("a branch with zero retained roots must have zero painted weight")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "retained_root_count", count)
        object.__setattr__(self, "painted_weight", weight)


@dataclass(frozen=True, slots=True)
class RodCoatingSummary:
    """Streaming coating result for one physical ``(h, k)`` rod."""

    rod: Rod
    branches: tuple[BranchCoatingSummary, ...]
    ledger: MassLedger
    maximum_ewald_residual_Ainv: float

    def __post_init__(self) -> None:
        if not isinstance(self.rod, Rod):
            raise TypeError("rod must be Rod")
        branches = tuple(self.branches)
        expected = (0,) if self.rod.family_m == 0 else (1, 2)
        if tuple(branch.branch for branch in branches) != expected:
            raise ValueError("rod coating branches are inconsistent with the rod family")
        if not isinstance(self.ledger, MassLedger):
            raise TypeError("ledger must be MassLedger")
        residual = finite_scalar(
            self.maximum_ewald_residual_Ainv,
            "maximum_ewald_residual_Ainv",
        )
        if residual < 0.0:
            raise ValueError("maximum_ewald_residual_Ainv must be nonnegative")
        branch_weight = fsum(branch.painted_weight for branch in branches)
        tolerance = 1024.0 * np.finfo(np.float64).eps * max(branch_weight, 1.0)
        if abs(branch_weight - self.ledger.painted_weight) > tolerance:
            raise ValueError("rod branch weights must equal its painted ledger weight")
        retained_count = sum(branch.retained_root_count for branch in branches)
        if retained_count == 0 and residual != 0.0:
            raise ValueError("a rod with zero retained roots must have zero maximum residual")
        classification_mass = fsum(
            (
                self.ledger.collapsed_direct_base_mass,
                self.ledger.tangent_base_mass,
                self.ledger.no_root_base_mass,
                self.ledger.forward_excluded_base_mass,
            )
        )
        if self.rod.population == 0.0 and (
            classification_mass != 0.0 or self.ledger.painted_weight != 0.0
        ):
            raise ValueError("a zero-population rod cannot carry classification or painted mass")
        if self.rod.family_m == 0:
            if self.ledger.tangent_base_mass != 0.0 or self.ledger.no_root_base_mass != 0.0:
                raise ValueError("m=0 rods cannot carry tangent or no-root mass")
            if self.ledger.suppressed_algebraic_direct_root_count < retained_count:
                raise ValueError("m=0 suppressed direct-root count cannot be below retained roots")
        elif (
            self.ledger.collapsed_direct_base_mass != 0.0
            or self.ledger.forward_excluded_base_mass != 0.0
            or self.ledger.suppressed_algebraic_direct_root_count != 0
        ):
            raise ValueError(
                "nonzero rods cannot carry collapsed-direct, forward-excluded, or direct-root data"
            )
        elif branches[0].retained_root_count != branches[1].retained_root_count:
            raise ValueError("nonzero rod branches must have equal retained-root counts")
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "maximum_ewald_residual_Ainv", residual)

    @property
    def retained_root_count(self) -> int:
        return sum(branch.retained_root_count for branch in self.branches)


@dataclass(frozen=True, slots=True)
class FamilyCoatingSummary:
    """Configured ``m`` subtotal, with completeness explicit for hexagonal cells."""

    family_m: int
    rod_summaries: tuple[RodCoatingSummary, ...]
    branches: tuple[BranchCoatingSummary, ...]
    ledger: MassLedger
    is_complete_hexagonal_family: bool
    maximum_ewald_residual_Ainv: float

    def __post_init__(self) -> None:
        family_m = integer(self.family_m, "family_m")
        rods = tuple(self.rod_summaries)
        if family_m < 0 or not rods:
            raise ValueError("a nonnegative family_m and at least one rod summary are required")
        if any(summary.rod.family_m != family_m for summary in rods):
            raise ValueError("every rod summary must belong to family_m")
        rod_keys = tuple((summary.rod.h, summary.rod.k) for summary in rods)
        if len(set(rod_keys)) != len(rod_keys):
            raise ValueError("a family subtotal cannot repeat a physical rod")
        branches = tuple(self.branches)
        expected = (0,) if family_m == 0 else (1, 2)
        if tuple(branch.branch for branch in branches) != expected:
            raise ValueError("family coating branches are inconsistent with family_m")
        if not isinstance(self.ledger, MassLedger):
            raise TypeError("ledger must be MassLedger")
        if not isinstance(self.is_complete_hexagonal_family, (bool, np.bool_)):
            raise TypeError("is_complete_hexagonal_family must be boolean")
        if self.is_complete_hexagonal_family and set(rod_keys) != _hexagonal_family_indices(
            family_m
        ):
            raise ValueError("a complete hexagonal family must contain the full integer shell")
        residual = finite_scalar(
            self.maximum_ewald_residual_Ainv,
            "maximum_ewald_residual_Ainv",
        )
        if residual < 0.0:
            raise ValueError("maximum_ewald_residual_Ainv must be nonnegative")
        combined_ledger = MassLedger.combine(tuple(summary.ledger for summary in rods))
        if not _mass_ledgers_agree(combined_ledger, self.ledger):
            raise ValueError("family ledger must combine every contained rod ledger")
        for index, branch in enumerate(branches):
            expected_count = sum(summary.branches[index].retained_root_count for summary in rods)
            expected_weight = fsum(summary.branches[index].painted_weight for summary in rods)
            tolerance = 1024.0 * np.finfo(np.float64).eps * max(expected_weight, 1.0)
            if (
                branch.retained_root_count != expected_count
                or abs(branch.painted_weight - expected_weight) > tolerance
            ):
                raise ValueError("family branches must combine their contained rod branches")
        expected_residual = max(summary.maximum_ewald_residual_Ainv for summary in rods)
        if residual != expected_residual:
            raise ValueError("family maximum residual must equal its contained rod maximum")
        object.__setattr__(self, "family_m", family_m)
        object.__setattr__(self, "rod_summaries", rods)
        object.__setattr__(self, "branches", branches)
        object.__setattr__(
            self,
            "is_complete_hexagonal_family",
            bool(self.is_complete_hexagonal_family),
        )
        object.__setattr__(self, "maximum_ewald_residual_Ainv", residual)

    @property
    def retained_root_count(self) -> int:
        return sum(branch.retained_root_count for branch in self.branches)


@dataclass(frozen=True, slots=True)
class PaintedEwaldCoating:
    """Bounded-memory Ewald coating aggregated by exact hexagonal family."""

    ki_sample_Ainv: FloatArray
    center_q_sample_Ainv: FloatArray
    radius_Ainv: float
    measure: PaintMeasure
    forward_policy: ForwardPolicy
    texture: SphereTexture
    ledger: MassLedger
    family_summaries: tuple[FamilyCoatingSummary, ...]

    def __post_init__(self) -> None:
        incident = readonly_float_array(self.ki_sample_Ainv, (3,), "ki_sample_Ainv")
        center = readonly_float_array(self.center_q_sample_Ainv, (3,), "center_q_sample_Ainv")
        radius = finite_scalar(self.radius_Ainv, "radius_Ainv")
        if radius <= 0.0:
            raise ValueError("radius_Ainv must be positive")
        tolerance = 1024.0 * np.finfo(np.float64).eps * max(radius, 1.0)
        if abs(np.linalg.norm(incident) - radius) > tolerance:
            raise ValueError("radius_Ainv must equal the incident wavevector norm")
        if not np.allclose(center, -incident, rtol=0.0, atol=tolerance):
            raise ValueError("center_q_sample_Ainv must equal -ki_sample_Ainv")
        if not isinstance(self.texture, SphereTexture):
            raise TypeError("texture must be SphereTexture")
        if not isinstance(self.ledger, MassLedger):
            raise TypeError("ledger must be MassLedger")
        measure = PaintMeasure(self.measure)
        if not isinstance(self.forward_policy, ForwardPolicy):
            raise TypeError("forward_policy must be ForwardPolicy")
        families = tuple(self.family_summaries)
        if not families or any(
            left.family_m >= right.family_m for left, right in pairwise(families)
        ):
            raise ValueError("family_summaries must be nonempty and ordered by unique family_m")
        combined_ledger = MassLedger.combine(tuple(summary.ledger for summary in families))
        if not _mass_ledgers_agree(combined_ledger, self.ledger):
            raise ValueError("coating ledger must combine every family ledger")
        if (
            measure is PaintMeasure.COAREA_INTENSITY
            and any(family.family_m == 0 for family in families)
            and self.forward_policy.q_min_Ainv is None
        ):
            raise ValueError("coarea_intensity with m=0 requires a physical q_min_Ainv")
        if measure is PaintMeasure.MOSAIC_PUSHFORWARD and self.ledger.forward_excluded_base_mass:
            raise ValueError("a pushforward coating cannot carry forward-excluded mass")
        texture_weight = fsum(self.texture.mass.ravel())
        mass_tolerance = 1024.0 * np.finfo(np.float64).eps * max(self.ledger.painted_weight, 1.0)
        if abs(texture_weight - self.ledger.painted_weight) > mass_tolerance:
            raise ValueError("coating texture must conserve its painted ledger weight")
        object.__setattr__(self, "ki_sample_Ainv", incident)
        object.__setattr__(self, "center_q_sample_Ainv", center)
        object.__setattr__(self, "radius_Ainv", radius)
        object.__setattr__(self, "measure", measure)
        object.__setattr__(self, "family_summaries", families)

    @property
    def rod_summaries(self) -> tuple[RodCoatingSummary, ...]:
        return tuple(rod for family in self.family_summaries for rod in family.rod_summaries)

    @property
    def retained_root_count(self) -> int:
        return sum(family.retained_root_count for family in self.family_summaries)

    @property
    def maximum_ewald_residual_Ainv(self) -> float:
        return max(family.maximum_ewald_residual_Ainv for family in self.family_summaries)
