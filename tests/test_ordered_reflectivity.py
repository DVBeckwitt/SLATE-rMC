from __future__ import annotations

import cmath
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xraydb
from numpy.typing import NDArray

from rasim_next.core.contracts import RodQueryBatch
from rasim_next.materials import (
    CLASSICAL_ELECTRON_RADIUS_A,
    CrystalStructure,
    material_optics,
    read_crystal,
)
from rasim_next.materials.optics import HC_EV_A
from rasim_next.ordered import (
    coherent_finite_stack,
    extract_pbi2_motifs,
    ordered_event_result,
    pbi2_layer_amplitudes,
    uniform_finite_stack,
    unit_cell_amplitude,
)
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.reciprocal.rods import build_rod_catalog
from rasim_next.reflectivity import manuscript_specular_composite, parratt_reflectivity

ROOT = Path(__file__).parents[1]
STRUCTURES = ROOT / "examples"
WAVELENGTH_A = 1.540592925


def _factor_e(
    species: str,
    element: str,
    charge: int,
    q_magnitude_Ainv: float,
    wavelength_A: float,
) -> complex:
    requested = species
    if charge:
        sign = "+" if charge > 0 else "-"
        ionic = f"{element}{abs(charge)}{sign}"
        if ionic in xraydb.f0_ions(element):
            requested = ionic
    f0 = float(np.asarray(xraydb.f0(requested, q_magnitude_Ainv / (4.0 * np.pi))).item())
    energy_eV = HC_EV_A / wavelength_A
    return complex(
        f0 + xraydb.f1_chantler(element, energy_eV),
        xraydb.f2_chantler(element, energy_eV),
    )


def _scalar_atom_sum(
    crystal: CrystalStructure,
    hkl: NDArray[np.float64],
    wavelength_A: NDArray[np.float64],
    *,
    unknown_u_iso_A2: float | None = None,
) -> NDArray[np.complex128]:
    reciprocal = 2.0 * np.pi * np.linalg.inv(crystal.direct_basis_A).T
    values: list[complex] = []
    for miller, wavelength in zip(hkl, wavelength_A, strict=True):
        q_vector = reciprocal @ miller
        q_magnitude = float(np.linalg.norm(q_vector))
        terms = []
        for site in crystal.sites:
            u_iso = unknown_u_iso_A2 if site.u_iso_A2 is None else site.u_iso_A2
            assert u_iso is not None
            terms.append(
                site.occupancy
                * _factor_e(
                    site.species,
                    site.element,
                    site.charge,
                    q_magnitude,
                    float(wavelength),
                )
                * math.exp(-0.5 * u_iso * q_magnitude**2)
                * cmath.exp(2.0j * np.pi * float(np.dot(miller, site.fractional)))
            )
        values.append(
            complex(
                math.fsum(value.real for value in terms),
                math.fsum(value.imag for value in terms),
            )
        )
    return np.asarray(values, dtype=np.complex128)


def test_cif_scalar_amplitude_and_raw_event_measure(tmp_path: Path) -> None:
    invalid_cif = tmp_path / "empty-loop.cif"
    invalid_cif.write_text("data_invalid\nloop_\n_tag\n", encoding="ascii")
    with pytest.raises(ValueError, match="empty loop"):
        read_crystal(invalid_cif)

    source_cif = STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_vesta.cif"
    anisotropic_cif = tmp_path / "anisotropic.cif"
    anisotropic_cif.write_text(
        source_cif.read_text(encoding="utf-8")
        + "\nloop_\n_atom_site_aniso_label\n_atom_site_aniso_U_11\nBi 0.01\n",
        encoding="utf-8",
    )
    with pytest.raises(NotImplementedError, match="anisotropic"):
        read_crystal(anisotropic_cif)

    crystal = read_crystal(source_cif, phase_id="bi2se3")
    assert crystal.spacegroup_hm == "R -3 m:H"
    assert len(crystal.sites) == 15
    assert {(site.source_label, site.source_multiplicity) for site in crystal.sites} == {
        ("Bi", 6),
        ("Se1", 3),
        ("Se2", 6),
    }
    assert {site.u_iso_A2 for site in crystal.sites} == {0.019}
    assert not np.isclose(
        np.dot(crystal.direct_basis_A[:, 0], crystal.direct_basis_A[:, 1]),
        0.0,
    )

    hkl = np.asarray(((0.0, 0.0, 3.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, -1.0, 0.37)))
    wavelength = np.asarray((WAVELENGTH_A, WAVELENGTH_A, 1.1, 1.8))
    production = unit_cell_amplitude(crystal, hkl, wavelength)
    expected = _scalar_atom_sum(crystal, hkl, wavelength)
    np.testing.assert_allclose(production.amplitude_e, expected, rtol=1e-12, atol=1e-10)
    assert abs(production.amplitude_e[1]) <= 1e-10
    assert not production.amplitude_e.flags.writeable

    for atom_count in (1, 2):
        small = replace(
            crystal,
            phase_id=f"bi2se3-{atom_count}-atom",
            sites=crystal.sites[:atom_count],
        )
        small_production = unit_cell_amplitude(small, hkl, wavelength).amplitude_e
        small_expected = _scalar_atom_sum(small, hkl, wavelength)
        np.testing.assert_allclose(small_production, small_expected, rtol=1e-12, atol=1e-10)

    partial = replace(
        crystal,
        phase_id="bi2se3-half-bi",
        sites=tuple(
            replace(site, occupancy=0.5) if site.source_label == "Bi" else site
            for site in crystal.sites
        ),
    )
    partial_production = unit_cell_amplitude(partial, hkl, wavelength).amplitude_e
    partial_expected = _scalar_atom_sum(partial, hkl, wavelength)
    np.testing.assert_allclose(partial_production, partial_expected, rtol=1e-12, atol=1e-10)
    assert np.max(np.abs(partial_production - production.amplitude_e)) > 1.0

    optical_wavelength_A = np.asarray((1.1, WAVELENGTH_A, 1.8))
    optics = material_optics(crystal, optical_wavelength_A)
    forward = unit_cell_amplitude(
        crystal,
        np.zeros((optical_wavelength_A.size, 3)),
        optical_wavelength_A,
    ).amplitude_e
    prefactor = (
        CLASSICAL_ELECTRON_RADIUS_A * optical_wavelength_A**2 / (2.0 * np.pi * crystal.volume_A3)
    )
    np.testing.assert_allclose(optics.delta, prefactor * forward.real, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(optics.beta, prefactor * forward.imag, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(optics.mu_Ainv, 4.0 * np.pi * optics.beta / optical_wavelength_A)
    np.testing.assert_allclose(optics.n_complex, 1.0 - optics.delta + 1.0j * optics.beta)
    assert np.all(optics.beta > 0.0)

    catalog = build_rod_catalog(crystal, h_bounds=(0, 1), k_bounds=(0, 0))
    rows = [int(np.flatnonzero((catalog.h == h_value) & (catalog.k == 0))[0]) for h_value in (0, 1)]
    l_coordinate = np.asarray((0.5, 1.25))
    query = RodQueryBatch(
        event_id=np.asarray((42, 7)),
        rod_id=catalog.rod_id[rows],
        phase_id=(crystal.phase_id,) * 2,
        h=np.asarray((0, 1), dtype=np.int32),
        k=np.zeros(2, dtype=np.int32),
        qz_Ainv=np.asarray((0.123, -0.456)),
        l_coordinate=l_coordinate,
        wavelength_A=np.full(2, WAVELENGTH_A),
    )
    result = ordered_event_result(crystal, catalog, query)
    expected_event = unit_cell_amplitude(
        crystal,
        np.column_stack((query.h, query.k, query.l_coordinate)),
        query.wavelength_A,
    ).amplitude_e
    np.testing.assert_array_equal(result.event_id, query.event_id)
    np.testing.assert_allclose(result.amplitude_e, expected_event, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.intensity.intensity_per_sr,
        np.abs(expected_event) ** 2,
        rtol=2e-15,
        atol=0.0,
    )
    assert result.intensity.normalization == "|F_e|^2; electron2; no external factors"
    assert not result.amplitude_e.flags.writeable


def test_bi2se3_same_family_rods_keep_distinct_identities() -> None:
    crystal = read_crystal(
        STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_vesta.cif",
        phase_id="bi2se3",
    )
    catalog = build_rod_catalog(crystal, h_bounds=(-1, 1), k_bounds=(-1, 1))
    expected_hk = {(h_value, k_value) for h_value in range(-1, 2) for k_value in range(-1, 2)}
    assert set(zip(catalog.h.tolist(), catalog.k.tolist(), strict=True)) == expected_hk
    assert np.unique(catalog.rod_id).size == len(expected_hk)
    selected_hk = ((1, 0), (0, 1), (1, -1))
    rows = [
        int(np.flatnonzero((catalog.h == h_value) & (catalog.k == k_value))[0])
        for h_value, k_value in selected_hk
    ]

    assert np.unique(catalog.rod_id[rows]).size == len(selected_hk)
    assert len({catalog.family_id[row] for row in rows}) == 1
    assert len({catalog.family_key[row] for row in rows}) == 1


def test_pbi2_motif_layer_boundary_matches_scalar_sum() -> None:
    crystal = read_crystal(
        STRUCTURES / "pbi2" / "structures" / "PbI2_2H.cif",
        phase_id="pbi2",
    )
    motifs = extract_pbi2_motifs(crystal)
    assert len(motifs) == 1
    assert motifs[0].orientation == "minus"
    assert sorted(atom.site_index for atom in motifs[0].atoms) == list(range(len(crystal.sites)))

    catalog = build_rod_catalog(crystal, h_bounds=(1, 1), k_bounds=(0, 0))
    l_coordinate = np.asarray((0.75, -0.75))
    lattice = ReciprocalLattice.from_crystal(crystal)
    query = RodQueryBatch(
        event_id=np.asarray((9, 2)),
        rod_id=np.repeat(catalog.rod_id, 2),
        phase_id=(crystal.phase_id,) * 2,
        h=np.ones(2, dtype=np.int32),
        k=np.zeros(2, dtype=np.int32),
        qz_Ainv=np.asarray((0.2, 0.3)),
        l_coordinate=l_coordinate,
        wavelength_A=np.full(2, WAVELENGTH_A),
    )
    result = pbi2_layer_amplitudes(crystal, query, unknown_u_iso_A2=0.0)

    plus_offsets = [
        (
            atom,
            (
                atom.fractional_offset[0],
                atom.fractional_offset[1],
                -atom.fractional_offset[2],
            ),
        )
        for atom in motifs[0].atoms
    ]
    expected_plus: list[complex] = []
    expected_minus: list[complex] = []
    for h_value, k_value, l_value, wavelength in zip(
        query.h,
        query.k,
        query.l_coordinate,
        query.wavelength_A,
        strict=True,
    ):
        q_vector = lattice.q_cartesian_Ainv((h_value, k_value, l_value))
        q_magnitude = float(np.linalg.norm(q_vector))
        plus_terms = []
        minus_terms = []
        for atom, offset in plus_offsets:
            factor = _factor_e(
                atom.species,
                atom.element,
                atom.charge,
                q_magnitude,
                float(wavelength),
            )
            common = 2.0 * np.pi * (float(h_value) * offset[0] + float(k_value) * offset[1])
            plus_terms.append(
                atom.occupancy
                * factor
                * cmath.exp(1.0j * (common + 2.0 * np.pi * l_value * offset[2]))
            )
            minus_terms.append(
                atom.occupancy
                * factor
                * cmath.exp(1.0j * (common - 2.0 * np.pi * l_value * offset[2]))
            )
        expected_plus.append(sum(plus_terms))
        expected_minus.append(sum(minus_terms))

    np.testing.assert_allclose(result.f_plus, expected_plus, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(result.f_minus, expected_minus, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(result.f_minus[0], result.f_plus[1], rtol=1e-12, atol=1e-10)
    assert not result.f_plus.flags.writeable


def test_finite_stack_matches_direct_sum_and_bragg_limit() -> None:
    event_id = np.asarray((17, 3))
    qz_Ainv = np.asarray((0.0, 0.73))
    layer_amplitude_e = np.asarray(
        (
            (1.0 + 0.5j, 2.0 - 0.25j, -0.4 + 0.7j),
            (0.5j, 1.2 - 0.3j, 0.8 + 0.1j),
        )
    )
    layer_depth_A = np.asarray((0.0, 2.5, 7.25))
    result = coherent_finite_stack(event_id, qz_Ainv, layer_amplitude_e, layer_depth_A)
    expected = np.sum(
        layer_amplitude_e * np.exp(1.0j * qz_Ainv[:, None] * layer_depth_A),
        axis=1,
    )
    np.testing.assert_allclose(result.amplitude_e, expected, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(result.intensity.intensity_per_sr, np.abs(expected) ** 2)
    assert result.intensity.normalization == "finite total |sum(F_e exp(i phase))|^2; electron2"
    assert not result.amplitude_e.flags.writeable

    repeat = np.asarray((1.2 + 0.4j,))
    off_bragg_qz = np.asarray((0.31,))
    off_bragg = uniform_finite_stack(
        event_id[:1],
        off_bragg_qz,
        repeat,
        2.5,
        5,
    )
    enumerated = repeat * np.sum(
        np.exp(1.0j * off_bragg_qz[:, None] * 2.5 * np.arange(5)),
        axis=1,
    )
    np.testing.assert_allclose(off_bragg.amplitude_e, enumerated, rtol=1e-12, atol=1e-10)
    bragg = uniform_finite_stack(
        event_id[:1],
        np.asarray((2.0 * np.pi / 2.5,)),
        repeat,
        2.5,
        7,
    )
    np.testing.assert_allclose(bragg.amplitude_e, 7.0 * repeat)
    np.testing.assert_allclose(
        bragg.intensity.intensity_per_sr,
        49.0 * np.abs(repeat) ** 2,
    )


def test_parratt_matches_small_scalar_recursion() -> None:
    wavelength_A = WAVELENGTH_A
    qz_Ainv = np.asarray((0.025, 0.055, 0.11))
    indices = np.asarray((1.0 + 0.0j, 0.999979 + 3.2e-7j, 0.99999 + 1.0e-8j))
    result = parratt_reflectivity(
        qz_Ainv,
        wavelength_A,
        refractive_index=indices,
        thickness_A=(None, 500.0, None),
        roughness_A=(2.0, 3.0),
    )

    k0 = 2.0 * np.pi / wavelength_A
    expected_kz = np.empty((qz_Ainv.size, indices.size), dtype=np.complex128)
    expected_interfaces = np.empty((qz_Ainv.size, 2), dtype=np.complex128)
    expected_amplitude = np.empty(qz_Ainv.size, dtype=np.complex128)
    for event_row, qz in enumerate(qz_Ainv):
        kz = [cmath.sqrt((index**2 - 1.0) * k0**2 + (0.5 * qz) ** 2) for index in indices]
        kz = [
            -value if value.imag < 0.0 or (value.imag == 0.0 and value.real < 0.0) else value
            for value in kz
        ]
        interfaces = [
            (kz[row] - kz[row + 1])
            / (kz[row] + kz[row + 1])
            * cmath.exp(-2.0 * kz[row] * kz[row + 1] * (2.0 + row) ** 2)
            for row in range(2)
        ]
        phase = cmath.exp(2.0j * kz[1] * 500.0)
        expected_kz[event_row] = kz
        expected_interfaces[event_row] = interfaces
        expected_amplitude[event_row] = (interfaces[0] + interfaces[1] * phase) / (
            1.0 + interfaces[0] * interfaces[1] * phase
        )

    np.testing.assert_allclose(result.kz_Ainv, expected_kz, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result.interface_amplitude, expected_interfaces, rtol=5e-11, atol=2e-11
    )
    np.testing.assert_allclose(result.amplitude, expected_amplitude, rtol=5e-11, atol=2e-11)
    np.testing.assert_allclose(result.reflectivity, np.abs(expected_amplitude) ** 2)
    assert result.normalization == "dimensionless pure Parratt reflectivity"

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
        thickness_A=(None, 1.0e7, None),
        roughness_A=(0.0, 0.0),
    )
    np.testing.assert_allclose(
        thick.amplitude,
        thick.interface_amplitude[:, 0],
        rtol=1e-12,
        atol=1e-12,
    )

    ambient_indices = np.asarray((1.2 + 0.0j, 1.5 + 0.0j))
    ambient_qz = np.asarray((2.2 * k0,))
    ambient = parratt_reflectivity(
        ambient_qz,
        wavelength_A,
        refractive_index=ambient_indices,
        thickness_A=(None, None),
        roughness_A=(0.0,),
    )
    ambient_kz = 0.5 * ambient_qz[0]
    substrate_kz = cmath.sqrt(
        (ambient_indices[1] ** 2 - ambient_indices[0] ** 2) * k0**2 + ambient_kz**2
    )
    expected_ambient_amplitude = (ambient_kz - substrate_kz) / (ambient_kz + substrate_kz)
    np.testing.assert_allclose(ambient.kz_Ainv[0], (ambient_kz, substrate_kz))
    np.testing.assert_allclose(ambient.amplitude[0], expected_ambient_amplitude)


def test_corrected_specular_keeps_named_outputs_separate() -> None:
    qc_Ainv = 0.0528619975
    qz_Ainv = np.linspace(2.0, 11.0, 19) * qc_Ainv
    indices = np.asarray((1.0 + 0.0j, 0.999979 + 3.2e-7j, 0.99999 + 1.0e-8j))
    pure = parratt_reflectivity(
        qz_Ainv,
        WAVELENGTH_A,
        refractive_index=indices,
        thickness_A=(None, 500.0, None),
        roughness_A=(2.0, 3.0),
    )
    result = manuscript_specular_composite(
        pure,
        lambda layer: 3.0 + 0.05 * layer**2,
        c_A=28.636,
        qc_Ainv=qc_Ainv,
        film_layer_index=1,
    )

    external_l = qz_Ainv * 28.636 / (2.0 * np.pi)
    phase_l = 2.0 * np.maximum(pure.kz_Ainv[:, 1].real, 0.0) * 28.636 / (2.0 * np.pi)
    expected_raw = 3.0 + 0.05 * external_l**2
    shape_term = ((3.0 + 0.05 * phase_l**2) / 3.0) / qz_Ainv**2
    scale_points = (qz_Ainv / qc_Ainv > 5.0) & (qz_Ainv / qc_Ainv < 10.0)
    expected_scale = np.exp(
        np.median(np.log(pure.reflectivity[scale_points]) - np.log(shape_term[scale_points]))
    )
    np.testing.assert_array_equal(result.parratt_reflectivity, pure.reflectivity)
    np.testing.assert_allclose(result.phase_l_coordinate, phase_l, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.raw_kinematic_e2, expected_raw, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.scaled_high_branch,
        expected_scale * shape_term,
        rtol=2e-15,
        atol=0.0,
    )
    assert np.all(np.isfinite(result.composite_reflectivity))
    assert result.raw_kinematic_normalization == "raw finite-stack electron2"
    assert result.parratt_normalization == "dimensionless pure Parratt reflectivity"
    assert result.composite_normalization == "dimensionless manuscript specular composite"
    assert not result.composite_reflectivity.flags.writeable
