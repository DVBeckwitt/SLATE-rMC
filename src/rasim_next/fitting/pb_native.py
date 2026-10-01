"""PbI2 atomic, finite-surface and stacking coordinates on the canonical recurrence."""

from dataclasses import dataclass, field, replace

import numpy as np

from painted_ewald import MosaicParameters
from rasim_next.core.contracts import canonical_revision_sha256
from rasim_next.fitting.fixed_lattice import hexagonal_direct_basis
from rasim_next.fitting.native_input import NativeFitPhysics
from rasim_next.fitting.native_structure import rebind_native_structure
from rasim_next.materials.crystal import crystal_with_direct_basis
from rasim_next.ordered.motifs import SiteDisplacementProfile, TransverseIsotropicSiteDisplacement
from rasim_next.stacking import InitialPopulation

PB_CELL_SITE_PARAMETER_NAMES = (
    "a_A",
    "c_A",
    "iodine_fractional_z",
    "pb_occupancy",
    "iodine_occupancy",
    "pb_u_radial_A2",
    "pb_u_normal_A2",
    "iodine_u_radial_A2",
    "iodine_u_normal_A2",
)


def simplex_fractions(shares):
    """Stick-breaking shares in [0,1] map onto the closed probability simplex."""
    shares = np.asarray(shares)
    if (
        shares.ndim != 1
        or np.iscomplexobj(shares)
        or np.any(~np.isfinite(shares))
        or np.any(shares < 0)
        or np.any(shares > 1)
    ):
        raise ValueError("simplex shares must be a finite real vector in [0,1]")
    remainder = np.r_[1.0, np.cumprod(1 - shares)]
    return tuple(np.r_[shares * remainder[:-1], remainder[-1]])


def simplex_shares(fractions):
    """Invert a normalized simplex, choosing zero for unobservable empty remainders."""
    values = np.asarray(fractions)
    if np.iscomplexobj(values):
        raise ValueError("simplex fractions must be real")
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or len(values) == 0
        or np.any(~np.isfinite(values))
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("fractions must be normalized nonnegative probabilities")
    remainder = np.cumsum(values[::-1])[::-1][:-1]
    return np.divide(values[:-1], remainder, out=np.zeros_like(remainder), where=remainder > 0)


@dataclass(frozen=True, slots=True)
class PbCellSiteParameters:
    """Nine cell/site coordinates with Pb as the fixed crystallographic origin."""

    a_A: float
    c_A: float
    iodine_fractional_z: float
    pb_occupancy: float
    iodine_occupancy: float
    pb_u_radial_A2: float
    pb_u_normal_A2: float
    iodine_u_radial_A2: float
    iodine_u_normal_A2: float

    def __post_init__(self):
        if any(np.iscomplexobj(getattr(self, name)) for name in PB_CELL_SITE_PARAMETER_NAMES):
            raise ValueError("Pb cell/site coordinates must be real")
        values = self.as_array()
        if (
            np.any(~np.isfinite(values))
            or np.any(values[:2] <= 0)
            or not 0 < values[2] < 0.5
            or np.any(values[3:5] < 0)
            or np.any(values[3:5] > 1)
            or np.any(values[5:] < 0)
        ):
            raise ValueError("invalid Pb cell, signed iodine height, occupancy or displacement")

    def as_array(self):
        return np.array([getattr(self, name) for name in PB_CELL_SITE_PARAMETER_NAMES], dtype=float)

    @classmethod
    def from_array(cls, values):
        values = np.asarray(values)
        if values.shape != (9,) or np.iscomplexobj(values):
            raise ValueError("Pb cell/site parameters require nine real coordinates")
        return cls(*values)


@dataclass(frozen=True, slots=True)
class PbNativeStructureModel:
    """Preserve signed iodine partners and every finite-window integer lift."""

    reference: NativeFitPhysics
    reference_parameters: PbCellSiteParameters = field(init=False)
    _signs: tuple[float, ...] = field(init=False, repr=False)
    _labels: tuple[str, str] = field(init=False, repr=False)

    def __post_init__(self):
        recipe = self.reference.structure
        if not recipe.stacking_phases or recipe.site_displacement_profile is not None:
            raise ValueError("Pb reference requires the declared isotropic trilayer and stacking")
        sites = recipe.crystals[0].sites
        lead = [s for s in sites if s.element == "Pb"]
        iodine = [s for s in sites if s.element == "I"]
        if len(sites) != 3 or len(lead) != 1 or len(iodine) != 2:
            raise ValueError("Pb model requires one Pb and two iodine sites")
        if not np.allclose(lead[0].fractional, 0, rtol=0, atol=1e-12):
            raise ValueError("Pb must be the fixed origin of the reference trilayer")
        signed = np.array([s.fractional[2] - np.rint(s.fractional[2]) for s in iodine])
        if (
            not np.isclose(signed.sum(), 0, rtol=0, atol=1e-12)
            or not 0 < abs(signed[0]) < 0.5
            or iodine[0].source_label != iodine[1].source_label
            or iodine[0].occupancy != iodine[1].occupancy
        ):
            raise ValueError("iodine sites must form one equally occupied signed pair")
        displacements = []
        for orbit in (lead, iodine):
            u = {recipe.unknown_u_iso_A2 if s.u_iso_A2 is None else s.u_iso_A2 for s in orbit}
            if len(u) != 1 or None in u:
                raise ValueError("each Pb reference orbit requires an explicit isotropic ADP")
            displacements.extend([u.pop()] * 2)
        basis = recipe.crystals[0].direct_basis_A
        object.__setattr__(
            self,
            "reference_parameters",
            PbCellSiteParameters(
                np.linalg.norm(basis[:, 0]),
                np.linalg.norm(basis[:, 2]),
                abs(signed[0]),
                lead[0].occupancy,
                iodine[0].occupancy,
                *displacements,
            ),
        )
        object.__setattr__(
            self,
            "_signs",
            tuple(
                0.0 if s.element == "Pb" else np.sign(s.fractional[2] - np.rint(s.fractional[2]))
                for s in sites
            ),
        )
        object.__setattr__(self, "_labels", (lead[0].source_label, iodine[0].source_label))

    def bind(self, parameters: PbCellSiteParameters) -> NativeFitPhysics:
        if not isinstance(parameters, PbCellSiteParameters):
            raise TypeError("parameters must be PbCellSiteParameters")
        initial = self.reference_parameters
        cell = self.reference.structure.crystals[0]
        basis = hexagonal_direct_basis(
            cell.direct_basis_A,
            np.log([parameters.a_A / initial.a_A, parameters.c_A / initial.c_A]),
        )
        dz = parameters.iodine_fractional_z - initial.iodine_fractional_z
        sites = tuple(
            replace(
                site,
                occupancy=parameters.pb_occupancy
                if site.element == "Pb"
                else parameters.iodine_occupancy,
                fractional=(*site.fractional[:2], site.fractional[2] + sign * dz),
            )
            for site, sign in zip(cell.sites, self._signs, strict=True)
        )
        crystal = crystal_with_direct_basis(
            replace(cell, sites=sites), basis, provenance="Pb native cell/site refinement"
        )
        profile = SiteDisplacementProfile(
            tuple(
                TransverseIsotropicSiteDisplacement(label, *u)
                for label, u in zip(
                    self._labels, parameters.as_array()[5:].reshape(2, 2), strict=True
                )
            )
        )
        recipe = replace(
            self.reference.structure,
            crystals=(crystal,),
            unknown_u_iso_A2=None,
            site_displacement_profile=profile,
        )
        revision = canonical_revision_sha256(
            ("definition_id", "pb_native_cell_site_candidate.v1"),
            ("reference_input", self.reference.input_revision),
            ("parameters", parameters.as_array()),
        )
        return rebind_native_structure(self.reference, recipe, revision)


@dataclass(frozen=True, slots=True)
class PbJointModel:
    """Explicit coordinates for the reference phase roster, with all mixture shares free.

    Each phase has its own epsilon coordinate, including separate handed phases.
    Parent shares act on transition laws or intensities exactly as declared by
    the reference recipe. N counts individual c-axis layers, not polytype cells.
    """

    atomic: PbNativeStructureModel

    @property
    def parameter_units(self):
        return (
            ("angstrom",) * 2
            + ("1",) * 3
            + ("angstrom^2",) * 4
            + ("radian",) * 2
            + ("1",) * 3
            + ("angstrom",)
            + ("1",) * (len(self.parameter_names) - 15)
        )

    @property
    def parameter_names(self):
        phases = self.atomic.reference.structure.stacking_phases
        return (
            *PB_CELL_SITE_PARAMETER_NAMES,
            "gaussian_sigma_rad",
            "lorentzian_half_width_rad",
            "lorentzian_probability",
            "surface_fraction_0",
            "surface_1_share_of_remainder",
            "extra_film_thickness_A",
            "initial_plus_probability",
            *(f"phase_{i}_share_of_remainder" for i in range(len(phases) - 1)),
            *(f"phase_{i}_epsilon" for i in range(len(phases))),
            *(
                f"phase_{i}_parent_{j}_share_of_remainder"
                for i, phase in enumerate(phases)
                for j in range(len(phase.parents) - 1)
            ),
        )

    def initial_values(self, seed):
        phases = self.atomic.reference.structure.stacking_phases
        n = seed["coherent_repeats"]
        extra = seed["film_thickness_nm"] * 10 - n * self.atomic.reference_parameters.c_A
        if extra < -1e-10:
            raise ValueError("seed film thickness is smaller than its coherent stack")
        return np.r_[
            self.atomic.reference_parameters.as_array(),
            np.deg2rad(seed["gaussian_sigma_deg"]),
            np.deg2rad(seed["lorentzian_hwhm_deg"]),
            seed["eta"],
            simplex_shares(seed["surface_fractions"]),
            max(0, extra),
            self.atomic.reference.structure.initial_population.plus,
            simplex_shares(seed["phase_fractions"]),
            [seed["fault_parameters"][p.fault_parameter] for p in phases],
            np.concatenate([simplex_shares(p.weights) for p in phases]),
        ]

    def bind(self, values, coherent_repeats):
        values = np.asarray(values)
        if (
            values.shape != (len(self.parameter_names),)
            or np.iscomplexobj(values)
            or np.any(~np.isfinite(values))
            or values[14] < 0
            or not 0 <= values[15] <= 1
            or type(coherent_repeats) is not int
            or coherent_repeats < 1
        ):
            raise ValueError("invalid Pb joint coordinate vector")
        physics = self.atomic.bind(PbCellSiteParameters.from_array(values[:9]))
        MosaicParameters(*values[9:12])
        phases = physics.structure.stacking_phases
        count = len(phases)
        fractions = simplex_fractions(values[16 : 15 + count])
        epsilon = values[15 + count : 15 + 2 * count]
        if np.any(epsilon < 0) or np.any(epsilon > 1):
            raise ValueError("phase epsilon probabilities must lie in [0,1]")
        offset, rebound = 15 + 2 * count, []
        for i, phase in enumerate(phases):
            size = len(phase.parents) - 1
            shares = values[offset : offset + size]
            # Keep an unchanged declared closed simplex exactly. Its inverse shares
            # can lose a final bit (or a tiny nonzero remainder) on reconstruction.
            weights = (
                phase.weights
                if np.array_equal(shares, simplex_shares(phase.weights))
                else simplex_fractions(shares)
            )
            rebound.append(
                replace(
                    phase,
                    fault_parameter=f"phase_{i}",
                    weights=weights,
                )
            )
            offset += size
        physics = replace(
            physics,
            structure=replace(
                physics.structure,
                stacking_phases=tuple(rebound),
                initial_population=InitialPopulation(values[15], 1 - values[15]),
            ),
        )
        physics = replace(
            physics,
            input_revision=canonical_revision_sha256(
                ("definition_id", "pb_native_joint_candidate.v1"),
                ("atomic_input", physics.input_revision),
                ("values", values),
                ("coherent_repeats", coherent_repeats),
            ),
        )
        return (
            physics,
            dict(
                coherent_repeats=coherent_repeats,
                film_thickness_A=coherent_repeats * values[1] + values[14],
                surface_fractions=simplex_fractions(values[12:14]),
                phase_fractions=fractions,
                fault_parameters={f"phase_{i}": float(e) for i, e in enumerate(epsilon)},
            ),
            MosaicParameters(*values[9:12]),
            None,
        )

    def inactive_parameters(self, values):
        parameters = dict(zip(self.parameter_names, values, strict=True))
        inactive = {}
        phases = self.atomic.reference.structure.stacking_phases
        share_names = [f"phase_{i}_share_of_remainder" for i in range(len(phases) - 1)]
        fractions = simplex_fractions([parameters[name] for name in share_names])
        exhausted = False
        for name in share_names:
            if exhausted:
                inactive[name] = "zero phase remainder"
            exhausted = exhausted or parameters[name] == 1
        for i, (phase, fraction) in enumerate(zip(phases, fractions, strict=True)):
            if fraction == 0:
                inactive[f"phase_{i}_epsilon"] = "zero phase population"
            exhausted = False
            for j in range(len(phase.parents) - 1):
                name = f"phase_{i}_parent_{j}_share_of_remainder"
                if fraction == 0 or exhausted:
                    inactive[name] = "zero phase population or parent remainder"
                exhausted = exhausted or parameters[name] == 1
        return inactive
