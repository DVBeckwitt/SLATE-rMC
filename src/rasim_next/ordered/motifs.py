"""Validated Pb-centered PbI2 trilayer amplitudes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from rasim_next.core.contracts import LayerAmplitudeResult, RodQueryBatch
from rasim_next.materials.crystal import CrystalSite, CrystalStructure
from rasim_next.ordered.amplitudes import unit_cell_amplitude


@dataclass(frozen=True, slots=True)
class MotifAtom:
    """One source site expressed in a Pb-centered crystallographic gauge."""

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
    atoms: tuple[MotifAtom, MotifAtom, MotifAtom],
) -> tuple[MotifAtom, MotifAtom, MotifAtom]:
    return tuple(
        MotifAtom(
            site_index=atom.site_index,
            source_label=atom.source_label,
            species=atom.species,
            element=atom.element,
            charge=atom.charge,
            occupancy=atom.occupancy,
            fractional_offset=(
                atom.fractional_offset[0],
                atom.fractional_offset[1],
                -atom.fractional_offset[2],
            ),
            u_iso_A2=atom.u_iso_A2,
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
    atoms: tuple[MotifAtom, MotifAtom, MotifAtom],
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
        provenance=f"Pb-centered motif extracted from {crystal.provenance}",
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
    return LayerAmplitudeResult(event_id=query.event_id, f_plus=f_plus, f_minus=f_minus)
