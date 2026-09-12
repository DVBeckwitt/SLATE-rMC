"""Symmetry-preserving Bi2X3 cell/site candidates on the native finite-CIF model."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import canonical_revision_sha256
from rasim_next.fitting.fixed_lattice import hexagonal_direct_basis
from rasim_next.fitting.native_input import NativeFitPhysics
from rasim_next.fitting.native_structure import (
    rebind_native_structure,
    validate_native_rod_coverage,
)
from rasim_next.materials.crystal import crystal_with_direct_basis
from rasim_next.ordered.motifs import (
    Bi2X3QuintupleLayerParameters,
    SiteDisplacementProfile,
    TransverseIsotropicSiteDisplacement,
    quintuple_layer_site_labels,
)

BI_CELL_SITE_PARAMETER_NAMES = (
    "a_A",
    "c_A",
    "bi_fractional_z",
    "outer_x_fractional_z",
    "bi_occupancy",
    "central_x_occupancy",
    "outer_x_occupancy",
    "bi_u_radial_A2",
    "bi_u_normal_A2",
    "central_x_u_radial_A2",
    "central_x_u_normal_A2",
    "outer_x_u_radial_A2",
    "outer_x_u_normal_A2",
)


@dataclass(frozen=True, slots=True)
class BiCellSiteParameters:
    """Thirteen physical coordinates; z is fractional, lengths A, and ADPs A²."""

    a_A: float
    c_A: float
    bi_fractional_z: float
    outer_x_fractional_z: float
    bi_occupancy: float
    central_x_occupancy: float
    outer_x_occupancy: float
    bi_u_radial_A2: float
    bi_u_normal_A2: float
    central_x_u_radial_A2: float
    central_x_u_normal_A2: float
    outer_x_u_radial_A2: float
    outer_x_u_normal_A2: float

    def __post_init__(self) -> None:
        for name in BI_CELL_SITE_PARAMETER_NAMES:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.a_A <= 0 or self.c_A <= 0:
            raise ValueError("cell lengths must be positive")
        # Reuse the accepted orbit-ordering and occupancy-budget constraints.
        Bi2X3QuintupleLayerParameters(
            self.bi_fractional_z,
            self.outer_x_fractional_z,
            self.bi_occupancy,
            self.central_x_occupancy,
            self.outer_x_occupancy,
            0.0,
            0.0,
        )
        if min(self.as_array()[7:]) < 0:
            raise ValueError("site displacement components must be nonnegative")

    def as_array(self) -> NDArray[np.float64]:
        """Return parameters in the declared BI_CELL_SITE_PARAMETER_NAMES order."""
        return np.array([getattr(self, name) for name in BI_CELL_SITE_PARAMETER_NAMES])

    @classmethod
    def from_array(cls, values: ArrayLike) -> BiCellSiteParameters:
        supplied = np.asarray(values)
        if supplied.shape != (13,) or np.iscomplexobj(supplied):
            raise ValueError("Bi cell/site parameters require thirteen real values")
        return cls(*supplied)


@dataclass(frozen=True, slots=True)
class BiNativeStructureModel:
    """Bind all cell/sites while preserving the reference finite surface windows.

    The reference is an expanded R-3m conventional cell, with equally ordered
    termination variants differing only by fixed integer z lifts. Bulk optics
    are derived from the occupied candidate cell. Source, mounting and the
    explicitly declared physical rod roster remain fixed inputs.
    """

    reference: NativeFitPhysics
    reference_parameters: BiCellSiteParameters = field(init=False)
    _labels: tuple[str, str, str] = field(init=False, repr=False)
    _orbit_index: NDArray[np.int64] = field(init=False, repr=False)
    _z_sign: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        recipe = self.reference.structure
        if recipe.stacking_phases or recipe.site_displacement_tensors_A2 is not None:
            raise ValueError("Bi reference must contain expanded ordered CIF termination motifs")
        crystal = recipe.crystals[0]
        labels = quintuple_layer_site_labels(crystal)
        # The legacy orbit extractor also requires a common isotropic ADP.
        # Extract geometry on a displacement-free copy; retain each orbit's
        # actual reference displacement independently below.
        baseline = Bi2X3QuintupleLayerParameters.from_crystal(
            replace(crystal, sites=tuple(replace(site, u_iso_A2=0.0) for site in crystal.sites))
        )
        orbit = np.array(
            [labels.index(site.source_label) for site in crystal.sites], dtype=np.int64
        )
        if tuple(np.bincount(orbit, minlength=3)) != (6, 3, 6):
            raise ValueError("Bi reference requires complete 6c/3a/6c conventional-cell orbits")
        reference_z = (baseline.bi_fractional_z, 0.0, baseline.se2_fractional_z)
        signs = np.zeros(len(orbit))
        for i, site in enumerate(crystal.sites):
            if orbit[i] == 1:
                continue
            # Fixed R-centering translates each signed 6c coordinate by thirds.
            delta = np.array([-1.0, 1.0]) * reference_z[orbit[i]]
            residual = 3.0 * (site.fractional[2] - delta)
            matches = np.flatnonzero(np.abs(residual - np.rint(residual)) < 1e-12)
            if len(matches) != 1:
                raise ValueError("reference rows must have unambiguous signed 6c coordinates")
            signs[i] = (-1.0, 1.0)[matches[0]]
        for surface in recipe.crystals[1:]:
            if len(surface.sites) != len(crystal.sites):
                raise ValueError("termination variants must retain every expanded site")
            for base_site, site in zip(crystal.sites, surface.sites, strict=True):
                shift = np.subtract(site.fractional, base_site.fractional)
                if (
                    replace(site, fractional=base_site.fractional) != base_site
                    or not np.allclose(shift[:2], 0, rtol=0, atol=1e-14)
                    or not np.isclose(shift[2], np.rint(shift[2]), rtol=0, atol=1e-14)
                ):
                    raise ValueError("surface variants may differ only by fixed integer z lifts")
        displacements = []
        for label in labels:
            values = {site.u_iso_A2 for site in crystal.sites if site.source_label == label}
            if len(values) != 1 or None in values:
                raise ValueError("each reference orbit requires a known isotropic displacement")
            value = values.pop()
            displacements.extend((value, value))
        parameters = BiCellSiteParameters(
            np.linalg.norm(crystal.direct_basis_A[:, 0]),
            np.linalg.norm(crystal.direct_basis_A[:, 2]),
            baseline.bi_fractional_z,
            baseline.se2_fractional_z,
            baseline.bi_occupancy,
            baseline.se1_occupancy,
            baseline.se2_occupancy,
            *displacements,
        )
        orbit.setflags(write=False)
        signs.setflags(write=False)
        object.__setattr__(self, "reference_parameters", parameters)
        object.__setattr__(self, "_labels", labels)
        object.__setattr__(self, "_orbit_index", orbit)
        object.__setattr__(self, "_z_sign", signs)

    def bind(self, parameters: BiCellSiteParameters) -> NativeFitPhysics:
        """Build a complete physical candidate, never reuse a stale optical transfer."""
        if not isinstance(parameters, BiCellSiteParameters):
            raise TypeError("parameters must be BiCellSiteParameters")
        reference, initial = self.reference, self.reference_parameters
        cell = reference.structure.crystals[0]
        basis = hexagonal_direct_basis(
            cell.direct_basis_A,
            np.log([parameters.a_A / initial.a_A, parameters.c_A / initial.c_A]),
        )
        dz = np.array(
            [
                parameters.bi_fractional_z - initial.bi_fractional_z,
                0.0,
                parameters.outer_x_fractional_z - initial.outer_x_fractional_z,
            ]
        )
        occupancies = (
            parameters.bi_occupancy,
            parameters.central_x_occupancy,
            parameters.outer_x_occupancy,
        )
        crystals = []
        for surface in reference.structure.crystals:
            sites = tuple(
                replace(
                    site,
                    occupancy=occupancies[orbit],
                    fractional=(*site.fractional[:2], site.fractional[2] + sign * dz[orbit]),
                )
                for site, orbit, sign in zip(
                    surface.sites, self._orbit_index, self._z_sign, strict=True
                )
            )
            crystals.append(
                crystal_with_direct_basis(
                    replace(surface, sites=sites),
                    basis,
                    provenance="Bi native cell/site refinement",
                )
            )
        profile = SiteDisplacementProfile(
            tuple(
                TransverseIsotropicSiteDisplacement(label, radial, normal)
                for label, (radial, normal) in zip(
                    self._labels, parameters.as_array()[7:].reshape(3, 2), strict=True
                )
            )
        )
        tensors = profile.tensors_A2(
            tuple(site.source_label for site in cell.sites), np.cross(basis[:, 0], basis[:, 1])
        )
        recipe = replace(
            reference.structure,
            crystals=tuple(crystals),
            unknown_u_iso_A2=None,
            site_displacement_tensors_A2=tuple(tensors for _ in crystals),
        )
        revision = canonical_revision_sha256(
            ("definition_id", "bi_native_cell_site_candidate.v1"),
            ("reference_input", reference.input_revision),
            ("parameters", parameters.as_array()),
        )
        return rebind_native_structure(reference, recipe, revision)

    def validate_rod_coverage(self, *, a_bounds_A, c_bounds_A) -> dict[str, float | int]:
        """Prove fixed-roster coverage over the complete declared cell box."""
        return validate_native_rod_coverage(
            self.reference, a_bounds_A=a_bounds_A, c_bounds_A=c_bounds_A
        )
