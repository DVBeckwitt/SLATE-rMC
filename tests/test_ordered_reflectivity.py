from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xraydb

from rasim_next.core.contracts import RodQueryBatch
from rasim_next.materials import (
    AVOGADRO_PER_MOL,
    CLASSICAL_ELECTRON_RADIUS_A,
    CrystalSite,
    CrystalStructure,
    mass_density_g_cm3,
    material_optics,
    read_crystal,
)
from rasim_next.ordered.amplitudes import ordered_event_result, unit_cell_amplitude
from rasim_next.ordered.finite_stack import coherent_finite_stack, uniform_finite_stack
from rasim_next.ordered.motifs import extract_pbi2_motifs, pbi2_layer_amplitudes
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.reciprocal.rods import build_rod_catalog
from rasim_next.reflectivity.parratt import ParrattResult, parratt_reflectivity
from rasim_next.reflectivity.specular import manuscript_specular_composite

ROOT = Path(__file__).parents[1]
STRUCTURES = ROOT / "examples"


def _fractional_by_element(sites: tuple[CrystalSite, ...]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for site in sites:
        grouped.setdefault(site.element, [])
        fractional = np.mod(site.fractional, 1.0)
        fractional[np.isclose(fractional, 1.0, rtol=0.0, atol=1e-12)] = 0.0
        grouped[site.element].append(tuple(float(value) for value in fractional))
    return {
        element: np.asarray(sorted(coordinates), dtype=np.float64)
        for element, coordinates in grouped.items()
    }


def _write_cif(tmp_path: Path, atoms: str, extra: str = "") -> Path:
    path = tmp_path / "structure.cif"
    path.write_text(
        """data_test
_cell_length_a 4
_cell_length_b 5
_cell_length_c 6
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P -1'
_space_group_IT_number 2
loop_
_space_group_symop_operation_xyz
'x,y,z'
'-x,-y,-z'
loop_
_atom_site_label
_atom_site_occupancy
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
_atom_site_type_symbol
"""
        + atoms
        + extra,
        encoding="utf-8",
    )
    return path


def test_cif_expansion_occupancy_and_displacement_policy(tmp_path: Path) -> None:
    legacy = read_crystal(
        STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_legacy.cif",
        phase_id="bi2se3",
    )
    expanded = read_crystal(
        STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_expanded_P1.cif",
        phase_id="bi2se3-p1",
    )

    assert len(legacy.sites) == len(expanded.sites) == 15
    legacy_coordinates = _fractional_by_element(legacy.sites)
    expanded_coordinates = _fractional_by_element(expanded.sites)
    assert legacy_coordinates.keys() == expanded_coordinates.keys()
    for element in legacy_coordinates:
        np.testing.assert_allclose(
            legacy_coordinates[element],
            expanded_coordinates[element],
            rtol=0.0,
            atol=1e-9,
        )
    assert {site.source_label: site.source_multiplicity for site in legacy.sites} == {
        "Bi": 6,
        "Se1": 3,
        "Se2": 6,
    }
    assert all(site.u_iso_A2 == pytest.approx(0.019, rel=0.0, abs=1e-15) for site in legacy.sites)
    assert not legacy.direct_basis_A.flags.writeable

    expected_counts = {"PbI2_2H.cif": 3, "PbI2_4H.cif": 6, "PbI2_6H.cif": 9}
    for filename, count in expected_counts.items():
        crystal = read_crystal(STRUCTURES / "pbi2" / "structures" / filename)
        assert len(crystal.sites) == count
        assert all(site.u_iso_A2 is None for site in crystal.sites)

    special = read_crystal(_write_cif(tmp_path, "X1 0.4 0 0 0 0.02 C\n"))
    assert len(special.sites) == 1
    assert special.sites[0].occupancy == pytest.approx(0.4, rel=0.0, abs=1e-15)
    assert special.sites[0].source_multiplicity == 1
    supplied_fractional = np.array([0.1, 0.2, 0.3])
    immutable_site = CrystalSite("C1", "C", "C", 0, 1.0, supplied_fractional, 0.0, 1)
    supplied_fractional[0] = 0.9
    assert immutable_site.fractional == (0.1, 0.2, 0.3)
    with pytest.raises(ValueError, match="integer"):
        CrystalSite("C1", "C", "C", 0.5, 1.0, (0.0, 0.0, 0.0), 0.0, 1)

    with pytest.raises(ValueError, match="occupancy"):
        read_crystal(_write_cif(tmp_path, "X1 1.2 0 0 0 0.02 C\n"))
    with pytest.raises(ValueError, match="occupancy"):
        read_crystal(_write_cif(tmp_path, "X1 ? 0 0 0 0.02 C\n"))

    anisotropic = _write_cif(
        tmp_path,
        "X1 1 0 0 0 0.02 C\n",
        """loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_12
_atom_site_aniso_U_13
_atom_site_aniso_U_23
X1 0.01 0.02 0.03 0 0 0
""",
    )
    with pytest.raises(NotImplementedError, match="anisotropic"):
        read_crystal(anisotropic)

    anisotropic_b = _write_cif(
        tmp_path,
        "X1 1 0 0 0 0.02 C\n",
        """loop_
_atom_site_aniso_label
_atom_site_aniso_B_11
_atom_site_aniso_B_22
_atom_site_aniso_B_33
_atom_site_aniso_B_12
_atom_site_aniso_B_13
_atom_site_aniso_B_23
X1 0.1 0.2 0.3 0 0 0
""",
    )
    with pytest.raises(NotImplementedError, match="anisotropic"):
        read_crystal(anisotropic_b)

    incomplete_anisotropic = _write_cif(
        tmp_path,
        "X1 1 0 0 0 0.02 C\n",
        """loop_
_atom_site_aniso_label
X1
""",
    )
    with pytest.raises(NotImplementedError, match="anisotropic"):
        read_crystal(incomplete_anisotropic)

    conflicting = _write_cif(tmp_path, "X1 1 0.2 0.3 0.4 0.02 C\n")
    conflicting.write_text(
        conflicting.read_text(encoding="utf-8").replace("'P -1'", "'P 1'"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="space-group"):
        read_crystal(conflicting)


def test_general_reciprocal_basis_and_layer_coordinate() -> None:
    direct = np.array([[4.0, 0.8, 0.2], [0.0, 5.0, 0.4], [0.0, 0.0, 6.0]], dtype=np.float64)
    lattice = ReciprocalLattice.from_direct_basis(direct)
    expected = 2.0 * np.pi * np.linalg.inv(direct).T

    np.testing.assert_allclose(lattice.basis_Ainv, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        direct.T @ lattice.basis_Ainv, 2.0 * np.pi * np.eye(3), rtol=1e-14, atol=1e-14
    )
    hkl = np.array([[1.0, 2.0, 0.5], [-2.0, 1.0, 3.25]])
    np.testing.assert_allclose(
        lattice.q_cartesian_Ainv(hkl), hkl @ expected.T, rtol=1e-14, atol=1e-14
    )
    rod_axis = expected[:, 2] / np.linalg.norm(expected[:, 2])
    radial_projector = np.eye(3) - np.outer(rod_axis, rod_axis)
    inplane_q = hkl[:, :2] @ expected[:, :2].T
    np.testing.assert_allclose(
        lattice.qr_Ainv(hkl[:, :2]),
        np.linalg.norm(inplane_q @ radial_projector, axis=1),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        lattice.inplane_metric_Ainv2,
        expected[:, :2].T @ radial_projector @ expected[:, :2],
        rtol=1e-14,
        atol=1e-14,
    )
    with pytest.raises(ValueError, match="reciprocal basis"):
        ReciprocalLattice(np.eye(3), np.zeros((3, 3)), 1.0)

    crystal = read_crystal(STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_legacy.cif")
    layered = ReciprocalLattice.from_crystal(crystal)
    l_coordinate = np.array([-1.25, 0.0, 2.75])
    qz = l_coordinate * layered.basis_Ainv[2, 2]
    np.testing.assert_allclose(
        layered.validate_layer_qz(l_coordinate, qz), l_coordinate, rtol=0.0, atol=0.0
    )
    with pytest.raises(ValueError, match="inconsistent"):
        layered.validate_layer_qz(l_coordinate, qz + 1e-3)


def test_raw_structure_amplitude_matches_direct_atom_sum() -> None:
    direct = np.diag([4.0, 5.0, 6.0])
    sites = (
        CrystalSite("C1", "C", "C", 0, 0.4, (0.125, 0.2, 0.3), 0.02, 1),
        CrystalSite("C2", "C", "C", 0, 0.6, (0.625, 0.2, 0.3), 0.02, 1),
    )
    crystal = CrystalStructure(
        phase_id="synthetic",
        spacegroup_hm="P 1",
        direct_basis_A=direct,
        volume_A3=float(np.linalg.det(direct)),
        sites=sites,
        source_path=Path("synthetic.cif"),
        provenance="analytic fixture",
    )
    hkl = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.25], [2.0, 1.0, -0.5]])
    wavelength = np.array([1.1, 1.540592925, 1.8])
    result = unit_cell_amplitude(crystal, hkl, wavelength)

    lattice = ReciprocalLattice.from_direct_basis(direct)
    expected: list[complex] = []
    for indices, wavelength_A in zip(hkl, wavelength, strict=True):
        q_vector = lattice.q_cartesian_Ainv(indices)
        q_magnitude = float(np.linalg.norm(q_vector))
        energy_eV = 12398.419843320026 / wavelength_A
        factor = complex(
            float(np.asarray(xraydb.f0("C", q_magnitude / (4.0 * np.pi))).item())
            + xraydb.f1_chantler("C", energy_eV),
            xraydb.f2_chantler("C", energy_eV),
        )
        amplitude = 0.0j
        for site in sites:
            position_A = direct @ np.asarray(site.fractional)
            amplitude += (
                site.occupancy
                * factor
                * np.exp(1.0j * np.dot(q_vector, position_A))
                * np.exp(-0.5 * site.u_iso_A2 * q_magnitude**2)
            )
        expected.append(amplitude)

    np.testing.assert_allclose(result.amplitude_e, expected, rtol=1e-12, atol=1e-10)
    assert result.amplitude_e[0].imag > 0.0
    assert "q=|Q|/(4*pi)" in result.provenance

    cancelling = CrystalStructure(
        phase_id="absence",
        spacegroup_hm="P 1",
        direct_basis_A=np.eye(3),
        volume_A3=1.0,
        sites=(
            CrystalSite("C1", "C", "C", 0, 1.0, (0.0, 0.0, 0.0), 0.0, 1),
            CrystalSite("C2", "C", "C", 0, 1.0, (0.5, 0.0, 0.0), 0.0, 1),
        ),
        source_path=Path("absence.cif"),
        provenance="analytic fixture",
    )
    absent = unit_cell_amplitude(cancelling, [1.0, 0.0, 0.0], 1.0)
    assert abs(complex(absent.amplitude_e)) < 1e-13

    unknown_u = CrystalStructure(
        phase_id="unknown-u",
        spacegroup_hm="P 1",
        direct_basis_A=np.eye(3),
        volume_A3=1.0,
        sites=(CrystalSite("I1", "I-", "I", -1, 1.0, (0.0, 0.0, 0.0), None, 1),),
        source_path=Path("unknown.cif"),
        provenance="analytic fixture",
    )
    with pytest.raises(ValueError, match="unknown isotropic displacement"):
        unit_cell_amplitude(unknown_u, [0.0, 0.0, 0.0], 1.0)
    explicit_zero = unit_cell_amplitude(unknown_u, [0.0, 0.0, 0.0], 1.0, unknown_u_iso_A2=0.0)
    assert "I-->I1-" in explicit_zero.provenance
    assert "database=9.2" in explicit_zero.provenance
    assert "unknown_u_iso_A2=0" in explicit_zero.provenance
    iodine_factor = complex(
        float(np.asarray(xraydb.f0("I1-", 0.0)).item())
        + xraydb.f1_chantler("I", 12398.419843320026),
        xraydb.f2_chantler("I", 12398.419843320026),
    )
    np.testing.assert_allclose(explicit_zero.amplitude_e, iodine_factor, rtol=1e-12, atol=1e-10)

    fallback = CrystalStructure(
        phase_id="fallback",
        spacegroup_hm="P 1",
        direct_basis_A=np.eye(3),
        volume_A3=1.0,
        sites=(CrystalSite("Se1", "Se2-", "Se", -2, 1.0, (0.0, 0.0, 0.0), 0.0, 1),),
        source_path=Path("fallback.cif"),
        provenance="analytic fixture",
    )
    assert "Se2-->Se" in unit_cell_amplitude(fallback, [0.0, 0.0, 0.0], 1.0).provenance

    empty = unit_cell_amplitude(crystal, np.empty((0, 3)), np.empty(0))
    assert empty.amplitude_e.shape == (0,)
    with pytest.raises(ValueError, match="tabulated Chantler energy"):
        unit_cell_amplitude(crystal, [0.0, 0.0, 0.0], 1e-6)


def test_material_optics_matches_atomic_forward_sum() -> None:
    crystal = read_crystal(
        STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_expanded_P1.cif",
        phase_id="bi2se3",
    )
    wavelength_A = np.array([1.1, 1.540592925, 2.0])
    optics = material_optics(crystal, wavelength_A)

    expected_density = sum(
        site.occupancy * xraydb.atomic_mass(site.element) for site in crystal.sites
    ) / (AVOGADRO_PER_MOL * crystal.volume_A3 * 1e-24)
    assert mass_density_g_cm3(crystal) == pytest.approx(expected_density, rel=2e-14, abs=2e-14)
    forward_e = []
    for wavelength in wavelength_A:
        energy_eV = 12398.419843320026 / wavelength
        total = 0.0j
        for site in crystal.sites:
            total += site.occupancy * complex(
                float(np.asarray(xraydb.f0(site.element, 0.0)).item())
                + xraydb.f1_chantler(site.element, energy_eV),
                xraydb.f2_chantler(site.element, energy_eV),
            )
        forward_e.append(total)
    forward = np.asarray(forward_e)
    prefactor = CLASSICAL_ELECTRON_RADIUS_A * wavelength_A**2 / (2.0 * np.pi * crystal.volume_A3)

    np.testing.assert_allclose(optics.delta, prefactor * forward.real, rtol=2e-11, atol=2e-15)
    np.testing.assert_allclose(optics.beta, prefactor * forward.imag, rtol=2e-11, atol=2e-15)
    np.testing.assert_allclose(
        optics.n_complex, 1.0 - optics.delta + 1.0j * optics.beta, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        optics.mu_Ainv, 4.0 * np.pi * optics.beta / wavelength_A, rtol=2e-15, atol=0.0
    )
    assert np.all(optics.beta > 0.0)
    assert f"density_g_cm3={expected_density:.16g}" in optics.provenance
    assert not optics.n_complex.flags.writeable


def test_rod_catalog_preserves_rods_and_exact_families() -> None:
    crystal = read_crystal(
        STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_legacy.cif",
        phase_id="bi2se3",
    )
    catalog = build_rod_catalog(crystal, h_bounds=(-1, 1), k_bounds=(-1, 1))

    assert catalog.rod_id.size == 9
    assert np.unique(catalog.rod_id).size == 9
    assert np.issubdtype(catalog.h.dtype, np.integer)
    assert np.issubdtype(catalog.k.dtype, np.integer)
    selected = []
    for h, k in ((1, 0), (0, 1), (1, -1)):
        index = int(np.flatnonzero((catalog.h == h) & (catalog.k == k))[0])
        selected.append(index)
    assert len({catalog.family_id[index] for index in selected}) == 1
    assert {catalog.family_key[index] for index in selected} == {"hex:m=1"}
    np.testing.assert_allclose(
        catalog.qr_Ainv[selected], catalog.qr_Ainv[selected[0]], rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        catalog.reciprocal_basis_Ainv,
        ReciprocalLattice.from_crystal(crystal).basis_Ainv,
        rtol=0.0,
        atol=0.0,
    )
    assert len(set(zip(catalog.h.tolist(), catalog.k.tolist(), strict=True))) == 9

    general = CrystalStructure(
        phase_id="triclinic",
        spacegroup_hm="P 1",
        direct_basis_A=np.array([[4.0, 0.7, 0.3], [0.0, 5.0, 0.2], [0.0, 0.0, 6.0]]),
        volume_A3=120.0,
        sites=(CrystalSite("C1", "C", "C", 0, 1.0, (0.0, 0.0, 0.0), 0.0, 1),),
        source_path=Path("triclinic.cif"),
        provenance="analytic fixture",
    )
    general_catalog = build_rod_catalog(general, h_bounds=(-1, 1), k_bounds=(0, 0))
    assert len(set(general_catalog.family_key)) == 3
    assert all("crystal-frame" in item for item in general_catalog.symmetry_metadata)
    changed_cell = CrystalStructure(
        phase_id=general.phase_id,
        spacegroup_hm=general.spacegroup_hm,
        direct_basis_A=np.diag([5.0, 5.0, 6.0]),
        volume_A3=150.0,
        sites=general.sites,
        source_path=Path("changed-cell.cif"),
        provenance="analytic fixture",
    )
    changed_catalog = build_rod_catalog(changed_cell, h_bounds=(1, 1), k_bounds=(0, 0))
    general_row = int(np.flatnonzero(general_catalog.h == 1)[0])
    assert general_catalog.family_key[general_row] != changed_catalog.family_key[0]


def test_event_aligned_ordered_result_preserves_identity_and_scale() -> None:
    crystal = read_crystal(
        STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_legacy.cif",
        phase_id="bi2se3",
    )
    catalog = build_rod_catalog(crystal, h_bounds=(0, 1), k_bounds=(0, 1))
    hk = ((1, 0), (0, 0), (0, 1))
    rows = [int(np.flatnonzero((catalog.h == h) & (catalog.k == k))[0]) for h, k in hk]
    l_coordinate = np.array([0.5, 3.0, 1.25])
    lattice = ReciprocalLattice.from_crystal(crystal)
    query = RodQueryBatch(
        event_id=np.array([42, 7, 99]),
        rod_id=catalog.rod_id[rows],
        phase_id=("bi2se3", "bi2se3", "bi2se3"),
        h=np.array([item[0] for item in hk], dtype=np.int32),
        k=np.array([item[1] for item in hk], dtype=np.int32),
        qz_Ainv=l_coordinate * lattice.basis_Ainv[2, 2],
        l_coordinate=l_coordinate,
        wavelength_A=np.array([1.1, 1.540592925, 1.8]),
    )
    result = ordered_event_result(crystal, catalog, query)
    expected = unit_cell_amplitude(
        crystal,
        np.column_stack((query.h, query.k, query.l_coordinate)),
        query.wavelength_A,
    ).amplitude_e

    np.testing.assert_array_equal(result.event_id, query.event_id)
    np.testing.assert_allclose(result.amplitude_e, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.intensity.intensity_per_sr,
        np.abs(expected) ** 2,
        rtol=2e-15,
        atol=0.0,
    )
    assert result.intensity.normalization == "|F_e|^2; electron2; no external factors"
    assert not result.amplitude_e.flags.writeable

    with pytest.raises(ValueError, match="rod identity"):
        ordered_event_result(
            crystal,
            catalog,
            RodQueryBatch(
                event_id=query.event_id,
                rod_id=query.rod_id,
                phase_id=query.phase_id,
                h=np.array([0, 0, 0], dtype=np.int32),
                k=query.k,
                qz_Ainv=query.qz_Ainv,
                l_coordinate=query.l_coordinate,
                wavelength_A=query.wavelength_A,
            ),
        )


def test_pbi2_motifs_cover_sites_and_preserve_orientation() -> None:
    expected_orientations = {
        "PbI2_2H.cif": ("minus",),
        "PbI2_4H.cif": ("plus", "minus"),
        "PbI2_6H.cif": ("plus", "plus", "plus"),
    }
    for filename, orientations in expected_orientations.items():
        crystal = read_crystal(
            STRUCTURES / "pbi2" / "structures" / filename,
            phase_id=filename.removesuffix(".cif"),
        )
        motifs = extract_pbi2_motifs(crystal)
        assert tuple(motif.orientation for motif in motifs) == orientations
        assert sorted(atom.site_index for motif in motifs for atom in motif.atoms) == list(
            range(len(crystal.sites))
        )
        assert all(
            [atom.element for atom in motif.atoms].count("Pb") == 1
            and [atom.element for atom in motif.atoms].count("I") == 2
            for motif in motifs
        )

    crystal = read_crystal(
        STRUCTURES / "pbi2" / "structures" / "PbI2_2H.cif",
        phase_id="pbi2",
    )
    catalog = build_rod_catalog(crystal, h_bounds=(1, 1), k_bounds=(0, 0))
    l_coordinate = np.array([0.75, -0.75])
    query = RodQueryBatch(
        event_id=np.array([9, 2]),
        rod_id=np.repeat(catalog.rod_id, 2),
        phase_id=("pbi2", "pbi2"),
        h=np.array([1, 1], dtype=np.int32),
        k=np.array([0, 0], dtype=np.int32),
        qz_Ainv=l_coordinate * ReciprocalLattice.from_crystal(crystal).basis_Ainv[2, 2],
        l_coordinate=l_coordinate,
        wavelength_A=np.array([1.540592925, 1.540592925]),
    )
    result = pbi2_layer_amplitudes(crystal, query, unknown_u_iso_A2=0.0)
    np.testing.assert_array_equal(result.event_id, query.event_id)
    np.testing.assert_allclose(result.f_minus[0], result.f_plus[1], rtol=1e-12, atol=1e-10)
    expected_plus = []
    expected_minus = []
    for h, k, layer, wavelength in zip(
        query.h, query.k, query.l_coordinate, query.wavelength_A, strict=True
    ):
        q_vector = ReciprocalLattice.from_crystal(crystal).q_cartesian_Ainv((h, k, layer))
        q_xraydb = np.linalg.norm(q_vector) / (4.0 * np.pi)
        energy_eV = 12398.419843320026 / wavelength
        f_pb = complex(
            float(np.asarray(xraydb.f0("Pb", q_xraydb)).item())
            + xraydb.f1_chantler("Pb", energy_eV),
            xraydb.f2_chantler("Pb", energy_eV),
        )
        f_i = complex(
            float(np.asarray(xraydb.f0("I", q_xraydb)).item()) + xraydb.f1_chantler("I", energy_eV),
            xraydb.f2_chantler("I", energy_eV),
        )
        eta = 0.2675
        phase_a = 0.666667 * h + 0.333333 * k
        phase_b = 0.333333 * h + 0.666667 * k
        expected_plus.append(
            f_pb
            + f_i * np.exp(2.0j * np.pi * (phase_a + layer * eta))
            + f_i * np.exp(2.0j * np.pi * (phase_b - layer * eta))
        )
        expected_minus.append(
            f_pb
            + f_i * np.exp(2.0j * np.pi * (phase_b + layer * eta))
            + f_i * np.exp(2.0j * np.pi * (phase_a - layer * eta))
        )
    np.testing.assert_allclose(result.f_plus, expected_plus, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(result.f_minus, expected_minus, rtol=1e-12, atol=1e-10)
    assert result.f_plus.dtype == result.f_minus.dtype == np.complex128
    assert not result.f_plus.flags.writeable

    malformed = CrystalStructure(
        phase_id="malformed",
        spacegroup_hm="P 1",
        direct_basis_A=np.diag([4.0, 4.0, 6.0]),
        volume_A3=96.0,
        sites=(
            CrystalSite("Pb1", "Pb", "Pb", 0, 1.0, (0.0, 0.0, 0.0), 0.0, 1),
            CrystalSite("I1", "I", "I", 0, 1.0, (1 / 3, 2 / 3, 0.2), 0.0, 1),
            CrystalSite("I2", "I", "I", 0, 1.0, (2 / 3, 1 / 3, 0.3), 0.0, 1),
        ),
        source_path=Path("malformed.cif"),
        provenance="analytic fixture",
    )
    with pytest.raises(ValueError, match="one iodine above and below"):
        extract_pbi2_motifs(malformed)


def test_finite_stack_limits_and_motif_gauge() -> None:
    event_id = np.array([17, 3])
    qz_Ainv = np.array([0.0, 0.73])
    layer_amplitude_e = np.array(
        [[1.0 + 0.5j, 2.0 - 0.25j, -0.4 + 0.7j], [0.5j, 1.2 - 0.3j, 0.8 + 0.1j]]
    )
    layer_depth_A = np.array([0.0, 2.5, 7.25])
    direct = coherent_finite_stack(event_id, qz_Ainv, layer_amplitude_e, layer_depth_A)
    expected = np.sum(layer_amplitude_e * np.exp(1.0j * qz_Ainv[:, None] * layer_depth_A), axis=1)
    np.testing.assert_allclose(direct.amplitude_e, expected, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(direct.intensity.intensity_per_sr, np.abs(expected) ** 2)
    assert direct.intensity.normalization == "finite total |sum(F_e exp(i phase))|^2; electron2"

    repeat = np.array([1.2 + 0.4j, -0.2 + 0.7j])
    for count in (1, 5):
        uniform = uniform_finite_stack(
            event_id, qz_Ainv, repeat, repeat_spacing_A=2.5, repeat_count=count
        )
        enumerated = repeat * np.sum(
            np.exp(1.0j * qz_Ainv[:, None] * 2.5 * np.arange(count)), axis=1
        )
        np.testing.assert_allclose(uniform.amplitude_e, enumerated, rtol=1e-12, atol=1e-10)
    exact_bragg = uniform_finite_stack(
        event_id[:1], np.array([2.0 * np.pi / 2.5]), repeat[:1], 2.5, 7
    )
    np.testing.assert_allclose(exact_bragg.amplitude_e, 7 * repeat[:1], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        uniform_finite_stack(
            event_id[:1], np.array([0.0]), repeat[:1], 2.5, 7
        ).intensity.intensity_per_sr,
        49 * np.abs(repeat[:1]) ** 2,
    )

    origin_shift_A = np.array([0.4, -0.8, 1.1])
    shifted_amplitude = layer_amplitude_e * np.exp(-1.0j * qz_Ainv[:, None] * origin_shift_A)
    shifted_depth = layer_depth_A + origin_shift_A
    shifted = coherent_finite_stack(event_id, qz_Ainv, shifted_amplitude, shifted_depth)
    np.testing.assert_allclose(shifted.amplitude_e, direct.amplitude_e, rtol=1e-12, atol=1e-10)

    registry_phase = np.array([[0.0, 2.0 * np.pi / 3.0]])
    registered = coherent_finite_stack(
        event_id[:1],
        np.array([0.0]),
        np.ones((1, 2), dtype=np.complex128),
        np.zeros(2),
        registry_phase_rad=registry_phase,
    )
    np.testing.assert_allclose(registered.intensity.intensity_per_sr, 1.0, rtol=0.0, atol=1e-15)
    assert (
        coherent_finite_stack(
            event_id[:1],
            np.array([0.0]),
            np.ones((1, 2), dtype=np.complex128),
            np.zeros(2),
        ).intensity.intensity_per_sr[0]
        == 4.0
    )
    gauge_shift = np.array([[0.37, -0.82, 1.14], [-0.21, 0.44, -0.63]])
    gauge_amplitude = layer_amplitude_e * np.exp(-1.0j * gauge_shift)
    gauge = coherent_finite_stack(
        event_id,
        qz_Ainv,
        gauge_amplitude,
        layer_depth_A,
        registry_phase_rad=gauge_shift,
    )
    np.testing.assert_allclose(gauge.amplitude_e, direct.amplitude_e, rtol=1e-12, atol=1e-10)


def test_parratt_matches_analytic_limits() -> None:
    wavelength_A = 1.540592925
    qz_Ainv = np.array([0.025, 0.055, 0.11])
    indices = np.array([1.0 + 0.0j, 0.999979 + 3.2e-7j, 0.99999 + 1.0e-8j])
    result = parratt_reflectivity(
        qz_Ainv,
        wavelength_A,
        refractive_index=indices,
        thickness_A=(None, 500.0, None),
        roughness_A=(2.0, 3.0),
    )

    k0 = 2.0 * np.pi / wavelength_A
    alpha = np.arcsin(qz_Ainv / (2.0 * k0))
    k_parallel = k0 * np.cos(alpha)
    expected_kz = np.empty((qz_Ainv.size, indices.size), dtype=np.complex128)
    for event_row, parallel in enumerate(k_parallel):
        for layer_row, refractive_index in enumerate(indices):
            root = np.sqrt((refractive_index * k0) ** 2 - parallel**2 + 0.0j)
            if root.imag < 0.0 or (root.imag == 0.0 and root.real < 0.0):
                root = -root
            expected_kz[event_row, layer_row] = root
    fresnel = (expected_kz[:, :-1] - expected_kz[:, 1:]) / (
        expected_kz[:, :-1] + expected_kz[:, 1:]
    )
    roughness = np.exp(-2.0 * expected_kz[:, :-1] * expected_kz[:, 1:] * np.array([2.0, 3.0]) ** 2)
    expected_interfaces = fresnel * roughness
    film_phase = np.exp(2.0j * expected_kz[:, 1] * 500.0)
    expected_amplitude = (expected_interfaces[:, 0] + expected_interfaces[:, 1] * film_phase) / (
        1.0 + expected_interfaces[:, 0] * expected_interfaces[:, 1] * film_phase
    )

    np.testing.assert_allclose(result.kz_Ainv, expected_kz, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result.interface_amplitude, expected_interfaces, rtol=5e-11, atol=2e-11
    )
    np.testing.assert_allclose(result.amplitude, expected_amplitude, rtol=5e-11, atol=2e-11)
    np.testing.assert_allclose(
        result.reflectivity, np.abs(expected_amplitude) ** 2, rtol=1e-10, atol=5e-11
    )
    assert np.all(result.kz_Ainv[:, 1].imag > 0.0)
    assert result.normalization == "dimensionless pure Parratt reflectivity"

    single = parratt_reflectivity(
        qz_Ainv,
        wavelength_A,
        refractive_index=indices[[0, 2]],
        thickness_A=(None, None),
        roughness_A=(1.5,),
    )
    np.testing.assert_allclose(
        single.amplitude, single.interface_amplitude[:, 0], rtol=0.0, atol=0.0
    )
    equal_media = parratt_reflectivity(
        qz_Ainv,
        wavelength_A,
        refractive_index=np.ones(3, dtype=np.complex128),
        thickness_A=(None, 12.0, None),
        roughness_A=(0.0, 0.0),
    )
    np.testing.assert_allclose(equal_media.amplitude, 0.0, rtol=0.0, atol=3e-16)
    equal_zero = parratt_reflectivity(
        np.array([0.0, 1e-12]),
        wavelength_A,
        refractive_index=np.ones(2, dtype=np.complex128),
        thickness_A=(None, None),
        roughness_A=(0.0,),
    )
    np.testing.assert_array_equal(equal_zero.amplitude, np.zeros(2, dtype=np.complex128))
    np.testing.assert_allclose(equal_zero.kz_Ainv[:, 0], [0.0, 5e-13], rtol=1e-15, atol=0.0)
    collapsed = parratt_reflectivity(
        qz_Ainv,
        wavelength_A,
        refractive_index=indices,
        thickness_A=(None, 0.0, None),
        roughness_A=(0.0, 0.0),
    )
    bare = parratt_reflectivity(
        qz_Ainv,
        wavelength_A,
        refractive_index=indices[[0, 2]],
        thickness_A=(None, None),
        roughness_A=(0.0,),
    )
    np.testing.assert_allclose(collapsed.amplitude, bare.amplitude, rtol=1e-12, atol=1e-12)
    thick = parratt_reflectivity(
        qz_Ainv,
        wavelength_A,
        refractive_index=indices,
        thickness_A=(None, 1e8, None),
        roughness_A=(0.0, 0.0),
    )
    np.testing.assert_allclose(
        thick.amplitude,
        thick.interface_amplitude[:, 0],
        rtol=1e-12,
        atol=1e-12,
    )

    with pytest.raises(ValueError, match="active-gain"):
        parratt_reflectivity(
            qz_Ainv,
            wavelength_A,
            refractive_index=np.array([1.0 + 0.0j, 1.0 - 1e-6j]),
            thickness_A=(None, None),
            roughness_A=(0.0,),
        )
    with pytest.raises(ValueError, match="external specular range"):
        parratt_reflectivity(
            np.array([2.1 * k0]),
            wavelength_A,
            refractive_index=indices[[0, 2]],
            thickness_A=(None, None),
            roughness_A=(0.0,),
        )


def test_specular_outputs_and_handoff_are_distinct() -> None:
    qz_Ainv = np.linspace(2.0, 11.0, 181)
    normalized_kinematic = (1.0 + qz_Ainv**2) / qz_Ainv**2
    log_offset = np.where(
        qz_Ainv < 5.0,
        0.3,
        np.where(qz_Ainv <= 7.0, 0.05 * (qz_Ainv - 6.0), -0.3),
    )
    pure_parratt = normalized_kinematic * 10.0**log_offset
    kz_Ainv = np.repeat((qz_Ainv / 2.0)[:, None], 3, axis=1).astype(np.complex128)
    parratt = ParrattResult(
        qz_Ainv=qz_Ainv,
        kz_Ainv=kz_Ainv,
        interface_amplitude=np.zeros((qz_Ainv.size, 2), dtype=np.complex128),
        amplitude=np.sqrt(pure_parratt).astype(np.complex128),
        reflectivity=pure_parratt,
    )
    fit_mask = (qz_Ainv >= 5.0) & (qz_Ainv <= 7.0)
    result = manuscript_specular_composite(
        parratt,
        lambda layer: 1.0 + layer**2,
        c_A=2.0 * np.pi,
        qc_Ainv=1.0,
        film_layer_index=1,
        fit_mask=fit_mask,
    )

    np.testing.assert_array_equal(result.parratt_reflectivity, pure_parratt)
    np.testing.assert_allclose(result.raw_kinematic_e2, 1.0 + qz_Ainv**2, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(result.phase_l_coordinate, qz_Ainv, rtol=1e-15, atol=0.0)
    assert result.blend_bounds_q_over_qc == pytest.approx((5.0, 7.0))
    assert result.blend_selection == "automatic"
    below = qz_Ainv <= result.blend_bounds_q_over_qc[0]
    above = qz_Ainv >= result.blend_bounds_q_over_qc[1]
    np.testing.assert_array_equal(result.composite_reflectivity[below], pure_parratt[below])
    np.testing.assert_array_equal(
        result.composite_reflectivity[above], result.scaled_high_branch[above]
    )
    interior = ~(below | above)
    x1, x2 = result.blend_bounds_q_over_qc
    blend_coordinate = (qz_Ainv[interior] - x1) / (x2 - x1)
    weight = 6 * blend_coordinate**5 - 15 * blend_coordinate**4 + 10 * blend_coordinate**3
    expected_blend = 10.0 ** (
        (1.0 - weight) * np.log10(pure_parratt[interior])
        + weight * np.log10(result.scaled_high_branch[interior])
    )
    np.testing.assert_allclose(
        result.composite_reflectivity[interior], expected_blend, rtol=2e-15, atol=0.0
    )
    assert result.raw_kinematic_normalization == "raw finite-stack electron2"
    assert result.composite_normalization == "dimensionless manuscript specular composite"

    rescaled = manuscript_specular_composite(
        parratt,
        lambda layer: 7.0 * (1.0 + layer**2),
        c_A=2.0 * np.pi,
        qc_Ainv=1.0,
        film_layer_index=1,
        fit_mask=fit_mask,
    )
    np.testing.assert_allclose(rescaled.raw_kinematic_e2, 7.0 * result.raw_kinematic_e2)
    np.testing.assert_allclose(rescaled.scaled_high_branch, result.scaled_high_branch)
    np.testing.assert_allclose(rescaled.composite_reflectivity, result.composite_reflectivity)

    isolated_fit = qz_Ainv == 6.0
    fallback = manuscript_specular_composite(
        parratt,
        lambda layer: 1.0 + layer**2,
        c_A=2.0 * np.pi,
        qc_Ainv=1.0,
        film_layer_index=1,
        fit_mask=isolated_fit,
    )
    assert fallback.blend_bounds_q_over_qc == (3.0, 6.0)
    assert fallback.blend_selection == "fallback"
