"""Validated material motifs for registry-resolved layer amplitudes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product

import numpy as np

from rasim_next.core.contracts import LayerAmplitudeResult, RodQueryBatch
from rasim_next.materials.crystal import CrystalSite, CrystalStructure
from rasim_next.ordered.amplitudes import unit_cell_amplitude


@dataclass(frozen=True, slots=True)
class MotifAtom:
    """One source site expressed in a motif-centered crystallographic gauge."""

    site_index: int
    source_label: str
    species: str
    element: str
    charge: int
    occupancy: float
    fractional_offset: tuple[float, float, float]
    u_iso_A2: float | None


@dataclass(frozen=True, slots=True)
class PbI2Motif:
    """One completely assigned I-Pb-I trilayer from an expanded structure."""

    orientation: str
    atoms: tuple[MotifAtom, MotifAtom, MotifAtom]


@dataclass(frozen=True, slots=True)
class TransverseIsotropicSiteDisplacement:
    """Reference displacement components for one crystallographic site orbit."""

    source_label: str
    u_radial_A2: float
    u_normal_A2: float

    def __post_init__(self) -> None:
        if not isinstance(self.source_label, str) or not self.source_label:
            raise ValueError("source_label must be a nonempty string")
        for name in ("u_radial_A2", "u_normal_A2"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class SiteDisplacementProfile:
    """Fixed site-resolved ADP shape with one constrained common scale."""

    sites: tuple[TransverseIsotropicSiteDisplacement, ...]
    scale: float = 1.0
    provenance: str = "declared"

    def __post_init__(self) -> None:
        sites = tuple(self.sites)
        if not sites or any(
            not isinstance(site, TransverseIsotropicSiteDisplacement) for site in sites
        ):
            raise ValueError("sites must contain transverse-isotropic site displacements")
        labels = tuple(site.source_label for site in sites)
        if len(set(labels)) != len(labels):
            raise ValueError("site displacement source labels must be unique")
        scale = float(self.scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("site displacement scale must be finite and nonnegative")
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("site displacement provenance must be a nonempty string")
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "scale", scale)

    def components_A2(self, source_label: str) -> tuple[float, float]:
        """Return scaled radial and normal components for one source-site label."""

        for site in self.sites:
            if site.source_label == source_label:
                return (
                    self.scale * site.u_radial_A2,
                    self.scale * site.u_normal_A2,
                )
        raise ValueError(f"site displacement profile lacks source label {source_label!r}")


@dataclass(frozen=True, slots=True)
class Bi2X3QuintupleLayerParameters:
    """Symmetry-preserving Bi2-chalcogen3 quintuple-layer parameters.

    The historical ``se1`` and ``se2`` field names denote the central and outer
    chalcogen orbits respectively.  Their chemical element and CIF labels are
    resolved from the expanded crystal rather than assumed to be selenium.
    """

    bi_fractional_z: float
    se2_fractional_z: float
    bi_occupancy: float
    se1_occupancy: float
    se2_occupancy: float
    u_radial_A2: float
    u_normal_A2: float
    outer_bi_antisite_fraction: float = 0.0

    def __post_init__(self) -> None:
        scalar_names = (
            "bi_fractional_z",
            "se2_fractional_z",
            "bi_occupancy",
            "se1_occupancy",
            "se2_occupancy",
            "u_radial_A2",
            "u_normal_A2",
            "outer_bi_antisite_fraction",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 1.0 / 3.0 < self.bi_fractional_z < 0.5:
            raise ValueError("bi_fractional_z must preserve the Bi 6c orbit ordering")
        if not 1.0 / 6.0 < self.se2_fractional_z < 1.0 / 3.0:
            raise ValueError("se2_fractional_z must preserve the Se2 6c orbit ordering")
        if self.bi_fractional_z + self.se2_fractional_z >= 2.0 / 3.0:
            raise ValueError("Bi and Se2 coordinates cross within the quintuple layer")
        for name in ("bi_occupancy", "se1_occupancy", "se2_occupancy"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.u_radial_A2 < 0.0 or self.u_normal_A2 < 0.0:
            raise ValueError("directional displacement parameters must be nonnegative")
        if not 0.0 <= self.outer_bi_antisite_fraction <= 1.0:
            raise ValueError("outer_bi_antisite_fraction must lie in [0, 1]")

    @classmethod
    def from_crystal(cls, crystal: CrystalStructure) -> Bi2X3QuintupleLayerParameters:
        """Derive the accepted seven-parameter baseline from one expanded R-3m CIF."""

        atoms = _bi2se3_quintuple_layers(crystal)[0]
        bi_label, center_label, outer_label = quintuple_layer_site_labels(crystal)
        by_label = {
            label: tuple(atom for atom in atoms if atom.source_label == label)
            for label in (bi_label, center_label, outer_label)
        }
        if tuple(len(by_label[label]) for label in (bi_label, center_label, outer_label)) != (
            2,
            1,
            2,
        ):
            raise ValueError("quintuple layer must contain Bi2-X1-X2 sites")

        def shared_property(label: str, name: str) -> float:
            values = {getattr(atom, name) for atom in by_label[label]}
            if len(values) != 1:
                raise ValueError(f"quintuple-layer {label} atoms must share {name}")
            value = values.pop()
            if value is None:
                raise ValueError(
                    f"quintuple-layer {label} requires a declared isotropic displacement"
                )
            return float(value)

        def symmetric_offset(label: str) -> float:
            offsets = np.abs([atom.fractional_offset[2] for atom in by_label[label]])
            if not np.allclose(offsets, offsets[0], rtol=0.0, atol=2.0e-15):
                raise ValueError("quintuple-layer symmetry mates must have equal normal offsets")
            return float(np.mean(offsets))

        bi_offset = symmetric_offset(bi_label)
        se2_offset = symmetric_offset(outer_label)
        u_values = {atom.u_iso_A2 for atom in atoms}
        if len(u_values) != 1 or None in u_values:
            raise ValueError("quintuple-layer baseline requires one shared isotropic displacement")
        u_iso = float(u_values.pop())
        return cls(
            bi_fractional_z=1.0 / 3.0 + bi_offset,
            se2_fractional_z=1.0 / 3.0 - se2_offset,
            bi_occupancy=shared_property(bi_label, "occupancy"),
            se1_occupancy=shared_property(center_label, "occupancy"),
            se2_occupancy=shared_property(outer_label, "occupancy"),
            u_radial_A2=u_iso,
            u_normal_A2=u_iso,
        )


def _nearest_periodic_offset(
    direct_basis_A: np.ndarray,
    iodine_fractional: tuple[float, float, float],
    lead_fractional: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    base = np.asarray(iodine_fractional) - np.asarray(lead_fractional)
    candidates: list[tuple[float, tuple[int, int, int], np.ndarray]] = []
    for shift in product((-1, 0, 1), repeat=3):
        offset = base + shift
        distance_squared = float(np.dot(direct_basis_A @ offset, direct_basis_A @ offset))
        candidates.append((distance_squared, shift, offset))
    minimum = min(item[0] for item in candidates)
    tied = [item for item in candidates if np.isclose(item[0], minimum, rtol=0.0, atol=1e-12)]
    if len({round(float(item[2][2]), 12) for item in tied}) > 1:
        raise ValueError("ambiguous vertical periodic image in PbI2 motif")
    _, _, chosen = min(tied, key=lambda item: item[1])
    chosen = np.asarray(chosen, dtype=np.float64)
    chosen[:2] = np.mod(chosen[:2], 1.0)
    chosen[:2][np.isclose(chosen[:2], 1.0, rtol=0.0, atol=1e-12)] = 0.0
    return chosen, minimum


def _assign_iodine_pairs(costs: np.ndarray) -> tuple[tuple[int, int], ...]:
    lead_count, iodine_count = costs.shape
    best_cost = np.inf
    best_assignments: list[tuple[tuple[int, int], ...]] = []

    def visit(
        lead_row: int,
        remaining: tuple[int, ...],
        assignment: tuple[tuple[int, int], ...],
        total_cost: float,
    ) -> None:
        nonlocal best_cost
        if lead_row == lead_count:
            if total_cost < best_cost - 1e-10:
                best_cost = total_cost
                best_assignments.clear()
                best_assignments.append(assignment)
            elif np.isclose(total_cost, best_cost, rtol=0.0, atol=1e-10):
                best_assignments.append(assignment)
            return
        for pair in combinations(remaining, 2):
            next_cost = total_cost + float(costs[lead_row, pair[0]] + costs[lead_row, pair[1]])
            if next_cost > best_cost + 1e-10:
                continue
            pair_set = set(pair)
            visit(
                lead_row + 1,
                tuple(index for index in remaining if index not in pair_set),
                (*assignment, pair),
                next_cost,
            )

    visit(0, tuple(range(iodine_count)), (), 0.0)
    unique = set(best_assignments)
    if len(unique) != 1:
        raise ValueError("ambiguous iodine-to-Pb assignment in PbI2 motif")
    return next(iter(unique))


def _motif_atom(site_index: int, site: CrystalSite, offset: np.ndarray) -> MotifAtom:
    return MotifAtom(
        site_index=site_index,
        source_label=site.source_label,
        species=site.species,
        element=site.element,
        charge=site.charge,
        occupancy=site.occupancy,
        fractional_offset=tuple(float(value) for value in offset),
        u_iso_A2=site.u_iso_A2,
    )


def _periodic_xy_close(actual: tuple[float, float], target: tuple[float, float]) -> bool:
    difference = np.asarray(actual) - np.asarray(target)
    difference -= np.rint(difference)
    return bool(np.linalg.norm(difference, ord=np.inf) <= 3e-5)


def _orientation(iodine_atoms: tuple[MotifAtom, MotifAtom]) -> str:
    above = [atom for atom in iodine_atoms if atom.fractional_offset[2] > 1e-12]
    below = [atom for atom in iodine_atoms if atom.fractional_offset[2] < -1e-12]
    if len(above) != 1 or len(below) != 1:
        raise ValueError("PbI2 motif must contain one iodine above and below Pb")
    above_xy = above[0].fractional_offset[:2]
    below_xy = below[0].fractional_offset[:2]
    plus = _periodic_xy_close(above_xy, (2.0 / 3.0, 1.0 / 3.0)) and _periodic_xy_close(
        below_xy, (1.0 / 3.0, 2.0 / 3.0)
    )
    minus = _periodic_xy_close(above_xy, (1.0 / 3.0, 2.0 / 3.0)) and _periodic_xy_close(
        below_xy, (2.0 / 3.0, 1.0 / 3.0)
    )
    if plus == minus:
        raise ValueError("PbI2 motif does not match a unique manuscript orientation")
    return "plus" if plus else "minus"


def extract_pbi2_motifs(crystal: CrystalStructure) -> tuple[PbI2Motif, ...]:
    """Assign every expanded PbI2 site to one rigid Pb-centered trilayer."""

    if any(site.element not in {"Pb", "I"} for site in crystal.sites):
        raise ValueError("PbI2 motif extraction accepts only Pb and I sites")
    lead_indices = tuple(index for index, site in enumerate(crystal.sites) if site.element == "Pb")
    iodine_indices = tuple(index for index, site in enumerate(crystal.sites) if site.element == "I")
    if not lead_indices or len(iodine_indices) != 2 * len(lead_indices):
        raise ValueError("expanded PbI2 structure must contain exactly two I sites per Pb site")

    offsets = np.empty((len(lead_indices), len(iodine_indices), 3), dtype=np.float64)
    costs = np.empty((len(lead_indices), len(iodine_indices)), dtype=np.float64)
    for lead_row, lead_index in enumerate(lead_indices):
        lead = crystal.sites[lead_index]
        for iodine_row, iodine_index in enumerate(iodine_indices):
            offsets[lead_row, iodine_row], costs[lead_row, iodine_row] = _nearest_periodic_offset(
                crystal.direct_basis_A,
                crystal.sites[iodine_index].fractional,
                lead.fractional,
            )
    assignment = _assign_iodine_pairs(costs)
    motifs: list[PbI2Motif] = []
    for lead_row, (first_row, second_row) in enumerate(assignment):
        lead_index = lead_indices[lead_row]
        lead_atom = _motif_atom(lead_index, crystal.sites[lead_index], np.zeros(3))
        iodine_atoms = tuple(
            _motif_atom(
                iodine_indices[iodine_row],
                crystal.sites[iodine_indices[iodine_row]],
                offsets[lead_row, iodine_row],
            )
            for iodine_row in (first_row, second_row)
        )
        orientation = _orientation(iodine_atoms)
        motifs.append(PbI2Motif(orientation, (lead_atom, *iodine_atoms)))
    covered = sorted(atom.site_index for motif in motifs for atom in motif.atoms)
    if covered != list(range(len(crystal.sites))):
        raise ValueError("PbI2 motif assignment must cover every expanded site exactly once")
    _canonical_plus_atoms(crystal, tuple(motifs))
    return tuple(motifs)


def _as_plus_atoms(motif: PbI2Motif) -> tuple[MotifAtom, MotifAtom, MotifAtom]:
    if motif.orientation == "plus":
        return motif.atoms
    return _reflected_atoms(motif.atoms)


def _reflected_atoms(
    atoms: tuple[MotifAtom, ...],
) -> tuple[MotifAtom, ...]:
    return tuple(
        replace(
            atom,
            fractional_offset=(
                atom.fractional_offset[0],
                atom.fractional_offset[1],
                -atom.fractional_offset[2],
            ),
        )
        for atom in atoms
    )


def _atom_signature(atom: MotifAtom) -> tuple[object, ...]:
    return atom.species, atom.element, atom.charge, atom.occupancy, atom.u_iso_A2


def _ordered_atoms(atoms: tuple[MotifAtom, MotifAtom, MotifAtom]) -> tuple[MotifAtom, ...]:
    return tuple(sorted(atoms, key=lambda atom: (atom.element != "Pb", atom.fractional_offset[2])))


def _canonical_plus_atoms(
    crystal: CrystalStructure, motifs: tuple[PbI2Motif, ...]
) -> tuple[MotifAtom, MotifAtom, MotifAtom]:
    candidates = tuple(_ordered_atoms(_as_plus_atoms(motif)) for motif in motifs)
    reference = candidates[0]
    reference_positions_A = np.asarray(
        [crystal.direct_basis_A @ atom.fractional_offset for atom in reference]
    )
    for candidate in candidates[1:]:
        if tuple(_atom_signature(atom) for atom in candidate) != tuple(
            _atom_signature(atom) for atom in reference
        ):
            raise ValueError("PbI2 orientation mapping does not preserve species and occupancy")
        candidate_positions_A = np.asarray(
            [crystal.direct_basis_A @ atom.fractional_offset for atom in candidate]
        )
        if not np.allclose(candidate_positions_A, reference_positions_A, rtol=0.0, atol=1e-5):
            raise ValueError("PbI2 motifs are not one rigid orientation-related trilayer")
    return reference


def _motif_crystal(
    crystal: CrystalStructure,
    atoms: tuple[MotifAtom, ...],
    phase_id: str,
) -> CrystalStructure:
    sites = tuple(
        CrystalSite(
            source_label=atom.source_label,
            species=atom.species,
            element=atom.element,
            charge=atom.charge,
            occupancy=atom.occupancy,
            fractional=atom.fractional_offset,
            u_iso_A2=atom.u_iso_A2,
            source_multiplicity=1,
        )
        for atom in atoms
    )
    return CrystalStructure(
        phase_id=phase_id,
        spacegroup_hm="P 1",
        direct_basis_A=crystal.direct_basis_A,
        volume_A3=crystal.volume_A3,
        sites=sites,
        source_path=crystal.source_path,
        provenance=f"registry-free motif extracted from {crystal.provenance}",
        source_sha256=crystal.source_sha256,
    )


def pbi2_layer_amplitudes(
    crystal: CrystalStructure,
    query: RodQueryBatch,
    *,
    unknown_u_iso_A2: float | None = None,
) -> LayerAmplitudeResult:
    """Return registry-free manuscript F+ and F- in raw electron units."""

    if any(phase_id != crystal.phase_id for phase_id in query.phase_id):
        raise ValueError("query phase does not match the PbI2 crystal")
    motifs = extract_pbi2_motifs(crystal)
    if len(motifs) != 1:
        raise ValueError("layer amplitudes require exactly one PbI2 motif per unit cell")
    plus_atoms = _canonical_plus_atoms(crystal, motifs)
    minus_atoms = _reflected_atoms(plus_atoms)
    hkl = np.column_stack((query.h, query.k, query.l_coordinate))
    f_plus = unit_cell_amplitude(
        _motif_crystal(crystal, plus_atoms, f"{crystal.phase_id}:motif-plus"),
        hkl,
        query.wavelength_A,
        unknown_u_iso_A2=unknown_u_iso_A2,
    ).amplitude_e
    f_minus = unit_cell_amplitude(
        _motif_crystal(crystal, minus_atoms, f"{crystal.phase_id}:motif-minus"),
        hkl,
        query.wavelength_A,
        unknown_u_iso_A2=unknown_u_iso_A2,
    ).amplitude_e
    layer_normal = np.cross(crystal.direct_basis_A[:, 0], crystal.direct_basis_A[:, 1])
    layer_normal /= np.linalg.norm(layer_normal)
    return LayerAmplitudeResult(
        event_id=query.event_id,
        rod_id=query.rod_id,
        phase_id=query.phase_id,
        f_plus_e=f_plus,
        f_minus_e=f_minus,
        normalization="ONE_REGISTRY_FREE_LAYER",
        phase_sign="POSITIVE_Q_DOT_R",
        gauge_id="pbi2.pb_centered.v1",
        layer_normal_crystal=layer_normal,
        layer_repeat_A=float(np.dot(crystal.direct_basis_A[:, 2], layer_normal)),
    )


def quintuple_layer_site_labels(crystal: CrystalStructure) -> tuple[str, str, str]:
    """Resolve Bi, central-X, and outer-X source labels from R-3m multiplicities."""

    sites_by_label: dict[str, list[CrystalSite]] = {}
    for site in crystal.sites:
        sites_by_label.setdefault(site.source_label, []).append(site)
    bi = tuple(
        label
        for label, sites in sites_by_label.items()
        if {site.element for site in sites} == {"Bi"} and len(sites) == 6
    )
    other_elements = {site.element for site in crystal.sites if site.element != "Bi"}
    if len(bi) != 1 or len(other_elements) != 1:
        raise ValueError("quintuple-layer crystal must contain one Bi orbit and one chalcogen")
    chalcogen = next(iter(other_elements))
    center = tuple(
        label
        for label, sites in sites_by_label.items()
        if {site.element for site in sites} == {chalcogen} and len(sites) == 3
    )
    outer = tuple(
        label
        for label, sites in sites_by_label.items()
        if {site.element for site in sites} == {chalcogen} and len(sites) == 6
    )
    if len(center) != 1 or len(outer) != 1 or len(sites_by_label) != 3:
        raise ValueError(
            "quintuple-layer crystal must contain Bi6, central-X3, and outer-X6 orbits"
        )
    return bi[0], center[0], outer[0]


def _bi2se3_quintuple_layers(
    crystal: CrystalStructure,
) -> tuple[tuple[MotifAtom, ...], ...]:
    bi_label, center_label, outer_label = quintuple_layer_site_labels(crystal)
    chalcogen = next(site.element for site in crystal.sites if site.source_label == center_label)
    centers = tuple(
        (site_index, site)
        for site_index, site in enumerate(crystal.sites)
        if site.source_label == center_label and site.element == chalcogen
    )
    if len(centers) != 3 or len(crystal.sites) != 15:
        raise ValueError("expanded Bi2-chalcogen3 must contain three X1-centered layers")

    motifs: list[tuple[MotifAtom, ...]] = []
    covered: list[int] = []
    center_coordinates: list[np.ndarray] = []
    for center_index, center in centers:
        neighbors: list[tuple[float, int, CrystalSite]] = []
        for site_index, site in enumerate(crystal.sites):
            if site_index == center_index:
                continue
            delta_z = site.fractional[2] - center.fractional[2]
            relative_z = delta_z - np.floor(delta_z + 0.5)
            neighbors.append((float(relative_z), site_index, site))
        lower = sorted((row for row in neighbors if row[0] < 0.0), reverse=True)[:2]
        upper = sorted(row for row in neighbors if row[0] > 0.0)[:2]
        block = (*sorted(lower), (0.0, center_index, center), *upper)
        if tuple(site.element for _, _, site in block) != (
            chalcogen,
            "Bi",
            chalcogen,
            "Bi",
            chalcogen,
        ):
            raise ValueError("quintuple layer must have X-Bi-X-Bi-X order")
        if (
            block[0][2].source_label != outer_label
            or block[-1][2].source_label != outer_label
            or block[1][2].source_label != bi_label
            or block[3][2].source_label != bi_label
        ):
            raise ValueError("quintuple-layer sites do not match the resolved Bi/X orbits")

        center_xy = np.asarray(center.fractional[:2], dtype=np.float64)
        motif: list[MotifAtom] = []
        for relative_z, site_index, site in block:
            offset_xy = np.mod(np.asarray(site.fractional[:2]) - center_xy, 1.0)
            offset_xy[np.isclose(offset_xy, 1.0, rtol=0.0, atol=1.0e-12)] = 0.0
            motif.append(
                _motif_atom(
                    site_index,
                    site,
                    np.asarray((offset_xy[0], offset_xy[1], relative_z)),
                )
            )
            covered.append(site_index)
        motifs.append(tuple(motif))
        center_coordinates.append(np.asarray(center.fractional, dtype=np.float64))

    if sorted(covered) != list(range(len(crystal.sites))):
        raise ValueError("quintuple layers must cover every expanded site exactly once")
    reference = motifs[0]
    reference_signature = tuple(_atom_signature(atom) for atom in reference)
    reference_labels = tuple(atom.source_label for atom in reference)
    reference_offsets = np.asarray([atom.fractional_offset for atom in reference])
    for motif in motifs[1:]:
        if (
            tuple(_atom_signature(atom) for atom in motif) != reference_signature
            or tuple(atom.source_label for atom in motif) != reference_labels
            or not np.allclose(
                np.asarray([atom.fractional_offset for atom in motif]),
                reference_offsets,
                rtol=0.0,
                atol=2.0e-15,
            )
        ):
            raise ValueError("quintuple layers must be property-identical translations")

    ordered_centers = sorted(center_coordinates, key=lambda coordinate: coordinate[2])
    translations = tuple(
        np.mod(right - left, 1.0)
        for left, right in zip(
            ordered_centers,
            (*ordered_centers[1:], ordered_centers[0] + np.asarray((0.0, 0.0, 1.0))),
            strict=True,
        )
    )
    expected_translation = np.asarray((2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    if not all(
        np.allclose(translation, expected_translation, rtol=0.0, atol=2.0e-15)
        for translation in translations
    ):
        raise ValueError("quintuple centers must follow the CIF R-centering translation")
    return tuple(motifs)


def _parameterized_bi2se3_quintuple_layer(
    crystal: CrystalStructure,
    parameters: Bi2X3QuintupleLayerParameters,
) -> tuple[MotifAtom, ...]:
    """Apply only the symmetry-allowed coordinates and source-site occupancies."""

    if not isinstance(parameters, Bi2X3QuintupleLayerParameters):
        raise TypeError("parameters must be Bi2X3QuintupleLayerParameters")
    baseline = _bi2se3_quintuple_layers(crystal)[0]
    bi_label, center_label, outer_label = quintuple_layer_site_labels(crystal)
    occupancy_by_label = {
        bi_label: parameters.bi_occupancy,
        center_label: parameters.se1_occupancy,
        outer_label: parameters.se2_occupancy,
    }
    normal_distance_by_label = {
        bi_label: parameters.bi_fractional_z - 1.0 / 3.0,
        center_label: 0.0,
        outer_label: 1.0 / 3.0 - parameters.se2_fractional_z,
    }
    bi_template = next(atom for atom in baseline if atom.source_label == bi_label)
    atoms: list[MotifAtom] = []
    for atom in baseline:
        try:
            occupancy = occupancy_by_label[atom.source_label]
            normal_distance = normal_distance_by_label[atom.source_label]
        except KeyError as error:
            raise ValueError("quintuple-layer motif contains an unsupported source site") from error
        baseline_z = atom.fractional_offset[2]
        fractional_z = 0.0 if normal_distance == 0.0 else np.copysign(normal_distance, baseline_z)
        positioned = replace(
            atom,
            occupancy=(
                occupancy * (1.0 - parameters.outer_bi_antisite_fraction)
                if atom.source_label == outer_label
                else occupancy
            ),
            fractional_offset=(
                atom.fractional_offset[0],
                atom.fractional_offset[1],
                float(fractional_z),
            ),
        )
        atoms.append(positioned)
        if atom.source_label == outer_label:
            atoms.append(
                replace(
                    positioned,
                    species=bi_template.species,
                    element=bi_template.element,
                    charge=bi_template.charge,
                    occupancy=occupancy * parameters.outer_bi_antisite_fraction,
                )
            )
    return tuple(atoms)


def bi2x3_quintuple_layer_amplitudes(
    crystal: CrystalStructure,
    query: RodQueryBatch,
    *,
    structure_parameters: Bi2X3QuintupleLayerParameters | None = None,
    site_displacement_profile: SiteDisplacementProfile | None = None,
) -> LayerAmplitudeResult:
    """Return central-chalcogen-centered quintuple-layer F+ and F- in electron units.

    The source CIF establishes the five-atom motif. The returned gauge is one
    registry-free layer; selecting a stacking parent remains a separate step.
    """

    if any(phase_id != crystal.phase_id for phase_id in query.phase_id):
        raise ValueError("query phase does not match the quintuple-layer crystal")
    baseline_parameters = Bi2X3QuintupleLayerParameters.from_crystal(crystal)
    parameters = baseline_parameters if structure_parameters is None else structure_parameters
    baseline_unchanged = parameters == baseline_parameters
    plus_atoms = (
        _bi2se3_quintuple_layers(crystal)[0]
        if baseline_unchanged
        else _parameterized_bi2se3_quintuple_layer(crystal, parameters)
    )
    minus_atoms = _reflected_atoms(plus_atoms)
    hkl = np.column_stack((query.h, query.k, query.l_coordinate))
    layer_normal = np.cross(crystal.direct_basis_A[:, 0], crystal.direct_basis_A[:, 1])
    layer_normal /= np.linalg.norm(layer_normal)
    if np.dot(layer_normal, crystal.direct_basis_A[:, 2]) < 0.0:
        layer_normal = -layer_normal
    if site_displacement_profile is not None and not isinstance(
        site_displacement_profile, SiteDisplacementProfile
    ):
        raise TypeError("site_displacement_profile must be a SiteDisplacementProfile")
    if baseline_unchanged and site_displacement_profile is None:
        displacement_tensor = None
        site_displacement_tensors = None
    elif site_displacement_profile is not None:
        if parameters.u_radial_A2 != 0.0 or parameters.u_normal_A2 != 0.0:
            raise ValueError(
                "site-resolved and shared quintuple-layer displacements are mutually exclusive"
            )
        normal_projector = np.outer(layer_normal, layer_normal)
        radial_projector = np.eye(3) - normal_projector
        site_displacement_tensors = np.asarray(
            [
                (
                    site_displacement_profile.components_A2(atom.source_label)[0] * radial_projector
                    + site_displacement_profile.components_A2(atom.source_label)[1]
                    * normal_projector
                )
                for atom in plus_atoms
            ],
            dtype=np.float64,
        )
        displacement_tensor = None
    elif parameters.u_radial_A2 == parameters.u_normal_A2:
        displacement_tensor = parameters.u_radial_A2 * np.eye(3)
        site_displacement_tensors = None
    else:
        normal_projector = np.outer(layer_normal, layer_normal)
        displacement_tensor = (
            parameters.u_radial_A2 * (np.eye(3) - normal_projector)
            + parameters.u_normal_A2 * normal_projector
        )
        site_displacement_tensors = None
    f_plus = unit_cell_amplitude(
        _motif_crystal(crystal, plus_atoms, f"{crystal.phase_id}:ql-plus"),
        hkl,
        query.wavelength_A,
        shared_displacement_tensor_A2=displacement_tensor,
        site_displacement_tensors_A2=site_displacement_tensors,
    ).amplitude_e
    f_minus = unit_cell_amplitude(
        _motif_crystal(crystal, minus_atoms, f"{crystal.phase_id}:ql-minus"),
        hkl,
        query.wavelength_A,
        shared_displacement_tensor_A2=displacement_tensor,
        site_displacement_tensors_A2=site_displacement_tensors,
    ).amplitude_e
    return LayerAmplitudeResult(
        event_id=query.event_id,
        rod_id=query.rod_id,
        phase_id=query.phase_id,
        f_plus_e=f_plus,
        f_minus_e=f_minus,
        normalization="ONE_REGISTRY_FREE_LAYER",
        phase_sign="POSITIVE_Q_DOT_R",
        gauge_id="bi2x3.central_chalcogen_centered_ql.v1",
        layer_normal_crystal=layer_normal,
        layer_repeat_A=float(np.dot(crystal.direct_basis_A[:, 2], layer_normal) / 3.0),
    )
