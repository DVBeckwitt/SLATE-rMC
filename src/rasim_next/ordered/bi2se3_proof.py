"""Compact independent Bi2Se3 quintuple-layer reconstruction proof."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rasim_next.core.contracts import RodQueryBatch
from rasim_next.materials import CrystalSite, CrystalStructure, read_crystal
from rasim_next.ordered.amplitudes import ordered_event_result, unit_cell_amplitude
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.reciprocal.rods import build_rod_catalog

_CENTERING_TRANSLATIONS = (
    (0.0, 0.0, 0.0),
    (2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
)
_WAVELENGTH_A = 1.540592925


@dataclass(frozen=True, slots=True)
class _MotifSite:
    source: CrystalSite
    fractional_offset: tuple[float, float, float]


def _canonical_fractional(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.mod(values, 1.0)
    result[np.isclose(result, 1.0, rtol=0.0, atol=1e-12)] = 0.0
    return result


def _quintuple_layers(
    crystal: CrystalStructure,
) -> tuple[tuple[_MotifSite, ...], tuple[tuple[float, float, float], ...], tuple[int, ...]]:
    centers = [
        (site_index, site)
        for site_index, site in enumerate(crystal.sites)
        if site.source_label == "Se1" and site.element == "Se"
    ]
    if len(centers) != 3 or len(crystal.sites) != 15:
        raise ValueError("Bi2Se3 proof requires three Se1-centered quintuple layers")

    motifs: list[tuple[_MotifSite, ...]] = []
    center_coordinates: list[tuple[float, float, float]] = []
    covered: list[int] = []
    for center_index, center in centers:
        neighbors: list[tuple[float, int, CrystalSite]] = []
        for site_index, site in enumerate(crystal.sites):
            if site_index == center_index:
                continue
            delta_z = site.fractional[2] - center.fractional[2]
            relative_z = delta_z - math.floor(delta_z + 0.5)
            neighbors.append((relative_z, site_index, site))
        lower = sorted((row for row in neighbors if row[0] < 0.0), reverse=True)[:2]
        upper = sorted(row for row in neighbors if row[0] > 0.0)[:2]
        block = (*sorted(lower), (0.0, center_index, center), *upper)
        if tuple(site.element for _, _, site in block) != ("Se", "Bi", "Se", "Bi", "Se"):
            raise ValueError("Bi2Se3 quintuple layer must have Se-Bi-Se-Bi-Se order")
        if block[0][2].source_label != "Se2" or block[-1][2].source_label != "Se2":
            raise ValueError("Bi2Se3 quintuple-layer outer sites must be Se2")

        center_xy = np.asarray(center.fractional[:2])
        motif: list[_MotifSite] = []
        for relative_z, site_index, site in block:
            offset_xy = _canonical_fractional(np.asarray(site.fractional[:2]) - center_xy)
            motif.append(
                _MotifSite(
                    source=site,
                    fractional_offset=(
                        float(offset_xy[0]),
                        float(offset_xy[1]),
                        float(relative_z),
                    ),
                )
            )
            covered.append(site_index)
        motifs.append(tuple(motif))
        center_coordinates.append(tuple(float(value) for value in center.fractional))

    reference = motifs[0]
    reference_properties = tuple(
        (
            site.source.source_label,
            site.source.species,
            site.source.occupancy,
            site.source.u_iso_A2,
        )
        for site in reference
    )
    reference_offsets = np.asarray([site.fractional_offset for site in reference])
    for motif in motifs[1:]:
        properties = tuple(
            (
                site.source.source_label,
                site.source.species,
                site.source.occupancy,
                site.source.u_iso_A2,
            )
            for site in motif
        )
        offsets = np.asarray([site.fractional_offset for site in motif])
        if properties != reference_properties or not np.allclose(
            offsets, reference_offsets, rtol=0.0, atol=2e-15
        ):
            raise ValueError("Bi2Se3 quintuple layers must be property-identical translations")
    return reference, tuple(center_coordinates), tuple(sorted(covered))


def _reconstructed_crystal(
    crystal: CrystalStructure, motif: tuple[_MotifSite, ...]
) -> CrystalStructure:
    sites: list[CrystalSite] = []
    for translation in _CENTERING_TRANSLATIONS:
        for motif_site in motif:
            fractional = _canonical_fractional(
                np.asarray(motif_site.fractional_offset) + np.asarray(translation)
            )
            source = motif_site.source
            sites.append(
                CrystalSite(
                    source_label=source.source_label,
                    species=source.species,
                    element=source.element,
                    charge=source.charge,
                    occupancy=source.occupancy,
                    fractional=tuple(float(value) for value in fractional),
                    u_iso_A2=source.u_iso_A2,
                    source_multiplicity=1,
                )
            )
    return CrystalStructure(
        phase_id="bi2se3-single-ql-reconstruction",
        spacegroup_hm="P 1",
        direct_basis_A=crystal.direct_basis_A,
        volume_A3=crystal.volume_A3,
        sites=tuple(sites),
        source_path=crystal.source_path,
        provenance="one Se1-centered QL plus exact R-centering translations",
    )


def _coordinate_error(first: CrystalStructure, second: CrystalStructure) -> float:
    def rows(crystal: CrystalStructure) -> list[tuple[str, str, float, float | None, np.ndarray]]:
        return [
            (
                site.source_label,
                site.species,
                site.occupancy,
                site.u_iso_A2,
                np.asarray(site.fractional),
            )
            for site in crystal.sites
        ]

    remaining = rows(second)
    maximum = 0.0
    for label, species, occupancy, u_iso, coordinate in rows(first):
        candidates = [
            (row_index, row)
            for row_index, row in enumerate(remaining)
            if row[:4] == (label, species, occupancy, u_iso)
        ]
        if not candidates:
            return math.inf
        distances = []
        for row_index, row in candidates:
            delta = coordinate - row[4]
            delta -= np.rint(delta)
            distances.append((float(np.max(np.abs(delta))), row_index))
        distance, selected = min(distances)
        maximum = max(maximum, distance)
        remaining.pop(selected)
    return maximum if not remaining else math.inf


def run_bi2se3_ql_proof(root: str) -> dict[str, object]:
    """Prove one QL reconstructs the tracked cell without collapsing rod identity."""

    path = Path(root) / "examples" / "bi2se3" / "structures" / "Bi2Se3_vesta.cif"
    crystal = read_crystal(path, phase_id="bi2se3")
    motif, centers, covered = _quintuple_layers(crystal)
    reconstructed = _reconstructed_crystal(crystal, motif)

    hkl = np.asarray(((0.0, 0.0, 3.0), (1.0, 0.0, 1.0), (0.0, 1.0, 2.0), (1.0, -1.0, 0.37)))
    wavelength = np.full(hkl.shape[0], _WAVELENGTH_A)
    original_amplitude = unit_cell_amplitude(crystal, hkl, wavelength).amplitude_e
    reconstructed_amplitude = unit_cell_amplitude(reconstructed, hkl, wavelength).amplitude_e
    amplitude_error_e = float(np.max(np.abs(original_amplitude - reconstructed_amplitude)))
    coordinate_error = _coordinate_error(crystal, reconstructed)

    catalog = build_rod_catalog(crystal, h_bounds=(-1, 1), k_bounds=(-1, 1))
    selected_hk = ((1, 0), (0, 1), (1, -1))
    rows = [
        int(np.flatnonzero((catalog.h == h_value) & (catalog.k == k_value))[0])
        for h_value, k_value in selected_hk
    ]
    l_coordinate = np.asarray((0.37, 0.37, 0.37))
    lattice = ReciprocalLattice.from_crystal(crystal)
    query = RodQueryBatch(
        event_id=np.asarray((31, 7, 19)),
        rod_id=catalog.rod_id[rows],
        phase_id=(crystal.phase_id,) * 3,
        h=np.asarray([value[0] for value in selected_hk], dtype=np.int32),
        k=np.asarray([value[1] for value in selected_hk], dtype=np.int32),
        q_sample_normal_Ainv=l_coordinate * lattice.basis_Ainv[2, 2],
        l_coordinate=l_coordinate,
        wavelength_A=np.full(3, _WAVELENGTH_A),
    )
    ordered = ordered_event_result(crystal, catalog, query)
    rods_are_distinct = bool(
        np.unique(query.rod_id).size == 3
        and len({catalog.family_id[row] for row in rows}) == 1
        and tuple(zip(query.h.tolist(), query.k.tolist(), strict=True)) == selected_hk
        and np.array_equal(ordered.event_id, query.event_id)
    )

    passed = bool(
        covered == tuple(range(15))
        and coordinate_error <= 2e-15
        and amplitude_error_e <= 1e-10
        and rods_are_distinct
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "ql_count": len(centers),
        "atoms_per_ql": len(motif),
        "stoichiometry": {"Bi": 2, "Se": 3},
        "centering_translations": [list(value) for value in _CENTERING_TRANSLATIONS],
        "site_coverage_exact": covered == tuple(range(15)),
        "maximum_coordinate_error_fractional": coordinate_error,
        "maximum_amplitude_error_e": amplitude_error_e,
        "rod_hk": [list(value) for value in selected_hk],
        "rod_ids": query.rod_id.tolist(),
        "same_family_distinct_rods": rods_are_distinct,
    }
