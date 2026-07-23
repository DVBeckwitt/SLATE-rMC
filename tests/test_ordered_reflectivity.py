from __future__ import annotations

import cmath
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xraydb
from numpy.typing import NDArray

from painted_ewald import BraggSpaceConfig, MosaicBraggSpace, MosaicParameters, Rod
from rasim_next.core.contracts import (
    EventIntensityNormalization,
    LayerNormalQBatch,
    MaterialOptics,
    RodQueryBatch,
    canonical_revision_sha256,
)
from rasim_next.core.scattering import (
    CLASSICAL_ELECTRON_RADIUS_A,
    electron_squared_to_scattering_strength_A2,
)
from rasim_next.materials import (
    CrystalSite,
    CrystalStructure,
    material_optics,
    read_crystal,
)
from rasim_next.materials.optics import HC_EV_A
from rasim_next.ordered import (
    Bi2Se3QuintupleLayerParameters,
    bi2se3_ql_amplitudes,
    coherent_finite_stack,
    extract_pbi2_motifs,
    ordered_event_result,
    pbi2_layer_amplitudes,
    uniform_finite_stack,
    unit_cell_amplitude,
)
from rasim_next.pipeline.bragg_space import Bi2Se3TwoHStrength
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.reciprocal.rods import build_rod_catalog
from rasim_next.reflectivity import manuscript_specular_composite, parratt_reflectivity
from rasim_next.stacking import (
    InitialPopulation,
    Parent,
    RichEpsilonModel,
    finite_event_intensity,
)

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


def _bi2se3_ql_crystal(crystal: CrystalStructure, *, reflected: bool) -> CrystalStructure:
    source_by_label = {
        label: next(site for site in crystal.sites if site.source_label == label)
        for label in ("Bi", "Se1", "Se2")
    }
    rows = (
        ("Se2", (1.0 / 3.0, 2.0 / 3.0, -0.12163333333333337)),
        ("Bi", (2.0 / 3.0, 1.0 / 3.0, -0.06746666666666667)),
        ("Se1", (0.0, 0.0, 0.0)),
        ("Bi", (1.0 / 3.0, 2.0 / 3.0, 0.06746666666666656)),
        ("Se2", (2.0 / 3.0, 1.0 / 3.0, 0.12163333333333332)),
    )
    sites = []
    for label, fractional in rows:
        source = source_by_label[label]
        x, y, z = fractional
        sites.append(
            CrystalSite(
                source_label=source.source_label,
                species=source.species,
                element=source.element,
                charge=source.charge,
                occupancy=source.occupancy,
                fractional=(x, y, -z if reflected else z),
                u_iso_A2=source.u_iso_A2,
                source_multiplicity=1,
            )
        )
    return CrystalStructure(
        phase_id=f"bi2se3-ql-{'minus' if reflected else 'plus'}",
        spacegroup_hm="P 1",
        direct_basis_A=crystal.direct_basis_A,
        volume_A3=crystal.volume_A3,
        sites=tuple(sites),
        source_path=crystal.source_path,
        provenance="independent signed-coordinate Bi2Se3 QL fixture",
    )


def test_bi2se3_quintuple_layer_and_finite_two_h_strength_match_direct_sums() -> None:
    crystal = read_crystal(
        STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_vesta.cif",
        phase_id="bi2se3",
    )
    h = np.asarray((1, 1, 0), dtype=np.int32)
    k = np.asarray((0, -1, 0), dtype=np.int32)
    ell = np.asarray((0.37, 1.13, 2.41))
    query = RodQueryBatch(
        event_id=np.asarray((11, 12, 13), dtype=np.int64),
        rod_id=np.asarray((31, 32, 33), dtype=np.int64),
        phase_id=(crystal.phase_id,) * 3,
        h=h,
        k=k,
        q_sample_normal_Ainv=np.zeros(3),
        l_coordinate=ell,
        wavelength_A=np.full(3, WAVELENGTH_A),
    )
    amplitudes = bi2se3_ql_amplitudes(crystal, query)
    hkl = np.column_stack((h, k, ell))
    plus_oracle = _scalar_atom_sum(
        _bi2se3_ql_crystal(crystal, reflected=False),
        hkl,
        query.wavelength_A,
    )
    minus_oracle = _scalar_atom_sum(
        _bi2se3_ql_crystal(crystal, reflected=True),
        hkl,
        query.wavelength_A,
    )
    np.testing.assert_allclose(amplitudes.f_plus_e, plus_oracle, rtol=2.0e-13, atol=2.0e-12)
    np.testing.assert_allclose(amplitudes.f_minus_e, minus_oracle, rtol=2.0e-13, atol=2.0e-12)
    mirrored = bi2se3_ql_amplitudes(crystal, replace(query, l_coordinate=-ell))
    np.testing.assert_allclose(amplitudes.f_minus_e, mirrored.f_plus_e, rtol=2.0e-13, atol=2.0e-12)
    assert amplitudes.normalization.value == "ONE_REGISTRY_FREE_LAYER"
    assert amplitudes.phase_sign.value == "POSITIVE_Q_DOT_R"
    assert amplitudes.gauge_id == "bi2se3.se1_centered_ql.v1"
    assert amplitudes.layer_repeat_A == pytest.approx(28.636 / 3.0, abs=2.0e-15)
    assert abs(amplitudes.f_plus_e[0] - amplitudes.f_minus_e[0]) > 10.0

    layers = 7
    strength_model = Bi2Se3TwoHStrength(
        crystal=crystal,
        layers=layers,
        normalization=EventIntensityNormalization.FINITE_TOTAL,
    )
    profile_l = np.asarray((0.37, 1.13, 3.0))
    strength = strength_model.evaluate_profile(
        rod=Rod(1, 0),
        L=profile_l,
        k_norm_Ainv=2.0 * np.pi / WAVELENGTH_A,
    )
    profile_query = RodQueryBatch(
        event_id=np.arange(profile_l.size, dtype=np.int64),
        rod_id=np.zeros(profile_l.size, dtype=np.int64),
        phase_id=(crystal.phase_id,) * profile_l.size,
        h=np.ones(profile_l.size, dtype=np.int32),
        k=np.zeros(profile_l.size, dtype=np.int32),
        q_sample_normal_Ainv=np.zeros(profile_l.size),
        l_coordinate=profile_l,
        wavelength_A=np.full(profile_l.size, WAVELENGTH_A),
    )
    f_plus = _scalar_atom_sum(
        _bi2se3_ql_crystal(crystal, reflected=False),
        np.column_stack((profile_query.h, profile_query.k, profile_query.l_coordinate)),
        profile_query.wavelength_A,
    )
    vertical = np.exp(2.0j * np.pi * profile_l / 3.0)
    geometric = np.sum(vertical[:, None] ** np.arange(layers), axis=1)
    expected = electron_squared_to_scattering_strength_A2(np.abs(f_plus * geometric) ** 2)
    np.testing.assert_allclose(strength, expected, rtol=2.0e-12, atol=2.0e-23)
    per_layer = Bi2Se3TwoHStrength(
        crystal=crystal,
        layers=layers,
        normalization=EventIntensityNormalization.FINITE_PER_LAYER,
    ).evaluate_profile(
        rod=Rod(1, 0),
        L=profile_l,
        k_norm_Ainv=2.0 * np.pi / WAVELENGTH_A,
    )
    np.testing.assert_allclose(per_layer * layers, strength, rtol=2.0e-15, atol=2.0e-23)

    rods = (Rod(1, 0), Rod(0, 1))
    bragg_space = MosaicBraggSpace(
        BraggSpaceConfig(
            reciprocal_basis_Ainv=strength_model.reciprocal_basis_Ainv,
            crystal_to_sample=np.eye(3),
            rods=rods,
            mosaic=MosaicParameters(
                gaussian_sigma_rad=np.deg2rad(5.0),
                lorentzian_half_width_rad=np.deg2rad(2.0),
                lorentzian_probability=0.1,
                alpha_panel_count=12,
                alpha_gauss_order=6,
                azimuth_count=16,
            ),
            k_norm_Ainv=2.0 * np.pi / WAVELENGTH_A,
        ),
        strength_model,
    )
    family = bragg_space.weighted_family_slice(family_m=1, L=float(profile_l[0]))
    direct_rod_strength = tuple(
        strength_model.evaluate(
            rod=rod,
            L=float(profile_l[0]),
            k_norm_Ainv=2.0 * np.pi / WAVELENGTH_A,
        )
        for rod in rods
    )
    for item, expected_rod_strength in zip(family.rod_slices, direct_rod_strength, strict=True):
        assert np.sum(item.intensity_weight_A2) == pytest.approx(
            expected_rod_strength,
            rel=2.0e-15,
        )
    assert family.total_intensity_weight_A2 == pytest.approx(
        sum(direct_rod_strength),
        rel=2.0e-15,
    )


def test_bi2se3_structure_parameters_match_directional_direct_sum() -> None:
    crystal = read_crystal(
        STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_vesta.cif",
        phase_id="bi2se3",
    )
    parameters = Bi2Se3QuintupleLayerParameters(
        bi_fractional_z=0.405,
        se2_fractional_z=0.207,
        bi_occupancy=0.91,
        se1_occupancy=0.83,
        se2_occupancy=0.74,
        u_radial_A2=0.006,
        u_normal_A2=0.032,
    )
    h = np.asarray((0, 1, -1), dtype=np.int32)
    k = np.asarray((0, 0, 1), dtype=np.int32)
    ell = np.asarray((6.2, 4.3, 9.1))
    query = RodQueryBatch(
        event_id=np.arange(3, dtype=np.int64),
        rod_id=np.arange(3, dtype=np.int64),
        phase_id=(crystal.phase_id,) * 3,
        h=h,
        k=k,
        q_sample_normal_Ainv=np.zeros(3),
        l_coordinate=ell,
        wavelength_A=np.full(3, WAVELENGTH_A),
    )
    actual = bi2se3_ql_amplitudes(
        crystal,
        query,
        structure_parameters=parameters,
    )
    reciprocal = ReciprocalLattice.from_crystal(crystal)
    q = reciprocal.q_cartesian_Ainv(np.column_stack((h, k, ell)))
    normal = np.cross(crystal.direct_basis_A[:, 0], crystal.direct_basis_A[:, 1])
    normal /= np.linalg.norm(normal)
    if np.dot(normal, crystal.direct_basis_A[:, 2]) < 0.0:
        normal = -normal
    q_normal = q @ normal
    q_radial_squared = np.maximum(np.einsum("ij,ij->i", q, q) - q_normal**2, 0.0)
    damping = np.exp(
        -0.5 * (parameters.u_radial_A2 * q_radial_squared + parameters.u_normal_A2 * q_normal**2)
    )
    d_bi = parameters.bi_fractional_z - 1.0 / 3.0
    d_se2 = 1.0 / 3.0 - parameters.se2_fractional_z
    rows = (
        ("Se2", parameters.se2_occupancy, (1.0 / 3.0, 2.0 / 3.0, -d_se2)),
        ("Bi", parameters.bi_occupancy, (2.0 / 3.0, 1.0 / 3.0, -d_bi)),
        ("Se1", parameters.se1_occupancy, (0.0, 0.0, 0.0)),
        ("Bi", parameters.bi_occupancy, (1.0 / 3.0, 2.0 / 3.0, d_bi)),
        ("Se2", parameters.se2_occupancy, (2.0 / 3.0, 1.0 / 3.0, d_se2)),
    )
    source_by_label = {
        label: next(site for site in crystal.sites if site.source_label == label)
        for label in ("Bi", "Se1", "Se2")
    }

    def direct(reflected: bool) -> np.ndarray:
        result = np.zeros(ell.size, dtype=np.complex128)
        for label, occupancy, fractional in rows:
            source = source_by_label[label]
            position = np.asarray(fractional)
            if reflected:
                position[2] *= -1.0
            factor = np.asarray(
                [
                    _factor_e(
                        source.species,
                        source.element,
                        source.charge,
                        float(np.linalg.norm(q_row)),
                        WAVELENGTH_A,
                    )
                    for q_row in q
                ]
            )
            result += (
                occupancy
                * factor
                * np.exp(
                    2.0j * np.pi * np.einsum("ij,j->i", np.column_stack((h, k, ell)), position)
                )
            )
        return damping * result

    np.testing.assert_allclose(actual.f_plus_e, direct(False), rtol=3.0e-13, atol=3.0e-12)
    np.testing.assert_allclose(actual.f_minus_e, direct(True), rtol=3.0e-13, atol=3.0e-12)
    baseline = Bi2Se3QuintupleLayerParameters.from_crystal(crystal)
    explicit_baseline = bi2se3_ql_amplitudes(crystal, query, structure_parameters=baseline)
    implicit_baseline = bi2se3_ql_amplitudes(crystal, query)
    np.testing.assert_allclose(explicit_baseline.f_plus_e, implicit_baseline.f_plus_e, rtol=0.0)
    np.testing.assert_allclose(explicit_baseline.f_minus_e, implicit_baseline.f_minus_e, rtol=0.0)


@pytest.mark.parametrize(
    ("epsilon", "normalization"),
    (
        (0.0, EventIntensityNormalization.FINITE_TOTAL),
        (0.001, EventIntensityNormalization.FINITE_TOTAL),
        (0.001, EventIntensityNormalization.FINITE_PER_LAYER),
    ),
)
def test_fixed_position_occupancy_quadratic_matches_full_strength(
    epsilon: float,
    normalization: EventIntensityNormalization,
) -> None:
    crystal = read_crystal(
        STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_vesta.cif",
        phase_id="bi2se3",
    )
    baseline = Bi2Se3QuintupleLayerParameters.from_crystal(crystal)
    candidate = replace(
        baseline,
        bi_occupancy=0.91,
        se1_occupancy=0.83,
        se2_occupancy=0.74,
        u_radial_A2=0.006,
        u_normal_A2=0.032,
    )
    h = np.asarray((0, 1, -1, 3), dtype=np.int32)
    k = np.asarray((0, 0, 1, 0), dtype=np.int32)
    ell = np.asarray((6.2, 4.3, 9.1, -2.7))
    k_norm = 2.0 * np.pi / WAVELENGTH_A
    model = Bi2Se3TwoHStrength(
        crystal=crystal,
        layers=52,
        normalization=normalization,
        shared_disorder_epsilon=epsilon,
        structure_parameters=baseline,
    )
    quadratic = model.fixed_position_occupancy_quadratic(
        h=h,
        k=k,
        L=ell,
        k_norm_Ainv=k_norm,
    )
    occupancy_products = np.asarray(
        (
            candidate.bi_occupancy**2,
            candidate.se1_occupancy**2,
            candidate.se2_occupancy**2,
            candidate.bi_occupancy * candidate.se1_occupancy,
            candidate.bi_occupancy * candidate.se2_occupancy,
            candidate.se1_occupancy * candidate.se2_occupancy,
        )
    )
    reciprocal = ReciprocalLattice.from_crystal(crystal)
    q = reciprocal.q_cartesian_Ainv(np.column_stack((h, k, ell)))
    normal = np.cross(crystal.direct_basis_A[:, 0], crystal.direct_basis_A[:, 1])
    normal /= np.linalg.norm(normal)
    q_normal_squared = (q @ normal) ** 2
    q_radial_squared = np.maximum(
        np.einsum("ij,ij->i", q, q) - q_normal_squared,
        0.0,
    )
    expected = (quadratic @ occupancy_products) * np.exp(
        -candidate.u_radial_A2 * q_radial_squared - candidate.u_normal_A2 * q_normal_squared
    )
    actual = replace(model, structure_parameters=candidate).evaluate_hkl(
        h=h,
        k=k,
        L=ell,
        k_norm_Ainv=k_norm,
    )
    np.testing.assert_allclose(expected, actual, rtol=3.0e-12, atol=2.0e-22)


def test_occupancy_quadratic_retains_a_disorder_extinction_as_nonnegative() -> None:
    crystal = read_crystal(
        STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_vesta.cif",
        phase_id="bi2se3",
    )
    baseline = Bi2Se3QuintupleLayerParameters.from_crystal(crystal)
    model = Bi2Se3TwoHStrength(
        crystal=crystal,
        layers=52,
        normalization=EventIntensityNormalization.FINITE_PER_LAYER,
        shared_disorder_epsilon=0.001,
        structure_parameters=baseline,
    )
    quadratic = model.fixed_position_occupancy_quadratic(
        h=np.asarray((-3,), dtype=np.int32),
        k=np.asarray((-3,), dtype=np.int32),
        L=np.asarray((-11.25,)),
        k_norm_Ainv=2.0 * np.pi / WAVELENGTH_A,
    )[0]
    matrix = np.asarray(
        (
            (quadratic[0], 0.5 * quadratic[3], 0.5 * quadratic[4]),
            (0.5 * quadratic[3], quadratic[1], 0.5 * quadratic[5]),
            (0.5 * quadratic[4], 0.5 * quadratic[5], quadratic[2]),
        )
    )
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues[0] >= -64.0 * np.finfo(np.float64).eps * eigenvalues[-1]

    occupancy = np.asarray((0.94, 0.78, 0.86))
    cached = float(occupancy @ matrix @ occupancy)
    candidate = replace(
        model,
        structure_parameters=replace(
            baseline,
            bi_occupancy=float(occupancy[0]),
            se1_occupancy=float(occupancy[1]),
            se2_occupancy=float(occupancy[2]),
        ),
    )
    direct = float(
        candidate.evaluate_hkl(
            h=np.asarray((-3,), dtype=np.int32),
            k=np.asarray((-3,), dtype=np.int32),
            L=np.asarray((-11.25,)),
            k_norm_Ainv=2.0 * np.pi / WAVELENGTH_A,
        )[0]
    )
    numerical_null_A2 = 1.0e-30
    assert 0.0 <= cached < numerical_null_A2
    assert 0.0 <= direct < numerical_null_A2
    assert abs(cached - direct) <= 1.0e-32


def test_bi2se3_near_ideal_two_h_strength_matches_shared_disorder_oracle() -> None:
    crystal = read_crystal(
        STRUCTURES / "bi2se3" / "structures" / "Bi2Se3_vesta.cif",
        phase_id="bi2se3",
    )
    rod = Rod(1, 0)
    ell = np.asarray((-5.93, 0.37, 6.02))
    layers = 52
    epsilon = 0.001
    model = Bi2Se3TwoHStrength(
        crystal=crystal,
        layers=layers,
        normalization=EventIntensityNormalization.FINITE_TOTAL,
        shared_disorder_epsilon=epsilon,
    )

    actual = model.evaluate_profile(
        rod=rod,
        L=ell,
        k_norm_Ainv=2.0 * np.pi / WAVELENGTH_A,
    )
    reciprocal = ReciprocalLattice.from_crystal(crystal)
    q_crystal = reciprocal.q_cartesian_Ainv(
        np.column_stack((np.full(ell.size, rod.h), np.full(ell.size, rod.k), ell))
    )
    layer_normal = np.cross(crystal.direct_basis_A[:, 0], crystal.direct_basis_A[:, 1])
    layer_normal /= np.linalg.norm(layer_normal)
    if np.dot(layer_normal, crystal.direct_basis_A[:, 2]) < 0.0:
        layer_normal = -layer_normal
    layer_normal_q = q_crystal @ layer_normal
    event_id = np.arange(ell.size, dtype=np.int64)
    rod_id = np.zeros(ell.size, dtype=np.int64)
    query = RodQueryBatch(
        event_id=event_id,
        rod_id=rod_id,
        phase_id=(crystal.phase_id,) * ell.size,
        h=np.full(ell.size, rod.h, dtype=np.int32),
        k=np.full(ell.size, rod.k, dtype=np.int32),
        q_sample_normal_Ainv=layer_normal_q,
        l_coordinate=ell,
        wavelength_A=np.full(ell.size, WAVELENGTH_A),
    )
    amplitudes = bi2se3_ql_amplitudes(crystal, query)
    oracle = finite_event_intensity(
        query,
        amplitudes,
        RichEpsilonModel(Parent.TWO_H, epsilon).transition_law(),
        layer_normal_q=LayerNormalQBatch(
            event_id=event_id,
            rod_id=rod_id,
            phase_id=query.phase_id,
            layer_normal_q_Ainv=layer_normal_q,
            gauge_id=amplitudes.gauge_id,
        ),
        layers=layers,
        initial=InitialPopulation.plus_only(),
        model_component_id="near-ideal-2H",
        population_group_id=None,
        normalization=EventIntensityNormalization.FINITE_TOTAL,
    )

    assert model.shared_disorder_epsilon == epsilon
    np.testing.assert_allclose(actual, oracle.scattering_strength_A2, rtol=2.0e-13, atol=2.0e-23)


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
    unknown_displacement = replace(
        crystal,
        sites=tuple(replace(site, u_iso_A2=None) for site in crystal.sites),
    )
    tensor_result = unit_cell_amplitude(
        unknown_displacement,
        hkl,
        wavelength,
        shared_displacement_tensor_A2=np.diag((0.007, 0.007, 0.034)),
    )
    assert np.all(np.isfinite(tensor_result.amplitude_e))
    assert "shared_displacement_tensor_A2=declared" in tensor_result.provenance

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
    expected_delta = prefactor * forward.real
    expected_beta = prefactor * forward.imag
    np.testing.assert_allclose(optics.n_complex.real, 1.0 - expected_delta, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(optics.n_complex.imag, expected_beta, rtol=1e-12, atol=0.0)
    assert np.all(expected_beta > 0.0)

    repeated = material_optics(
        crystal,
        np.asarray((1.8, 1.1, WAVELENGTH_A, 1.8, 1.1, WAVELENGTH_A)),
    )
    np.testing.assert_array_equal(repeated.wavelength_A, optical_wavelength_A)
    np.testing.assert_allclose(repeated.n_complex, optics.n_complex, rtol=0.0, atol=0.0)

    expected_revision = canonical_revision_sha256(
        ("material_id", optics.material_id),
        ("material_optics_revision_schema", "material_optics_revision.v2"),
        ("n_complex", optics.n_complex),
        ("provenance", optics.provenance),
        ("wavelength_A", optics.wavelength_A),
    )
    assert optics.material_revision == expected_revision
    assert repeated.material_revision == expected_revision

    changed_wavelength = optics.wavelength_A * 1.01
    changed_materials = (
        replace(optics, material_id=f"{optics.material_id}-changed"),
        replace(optics, wavelength_A=changed_wavelength),
        replace(optics, n_complex=optics.n_complex + (-1.0e-9 + 2.0e-10j)),
        replace(optics, provenance=f"{optics.provenance}; changed"),
    )
    assert all(item.material_revision != expected_revision for item in changed_materials)

    with pytest.raises(ValueError, match="wavelength"):
        MaterialOptics(
            material_id="invalid-duplicate",
            wavelength_A=optics.wavelength_A[[0, 0]],
            n_complex=optics.n_complex[[0, 0]],
            provenance="invalid duplicate fixture",
        )
    with pytest.raises(ValueError, match="wavelength"):
        MaterialOptics(
            material_id="invalid-unsorted",
            wavelength_A=optics.wavelength_A[::-1],
            n_complex=optics.n_complex[::-1],
            provenance="invalid unsorted fixture",
        )
    with pytest.raises(ValueError, match="absorption"):
        MaterialOptics(
            material_id="invalid-negative-absorption",
            wavelength_A=np.array([WAVELENGTH_A]),
            n_complex=np.array([1.0 - 1.0e-12j]),
            provenance="invalid absorption fixture",
        )

    catalog = build_rod_catalog(crystal, h_bounds=(0, 1), k_bounds=(0, 0))
    rows = [int(np.flatnonzero((catalog.h == h_value) & (catalog.k == 0))[0]) for h_value in (0, 1)]
    l_coordinate = np.asarray((0.5, 1.25))
    query = RodQueryBatch(
        event_id=np.asarray((42, 7)),
        rod_id=catalog.rod_id[rows],
        phase_id=(crystal.phase_id,) * 2,
        h=np.asarray((0, 1), dtype=np.int32),
        k=np.zeros(2, dtype=np.int32),
        q_sample_normal_Ainv=np.asarray((0.123, -0.456)),
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
    np.testing.assert_allclose(
        result.scattering_strength_A2,
        CLASSICAL_ELECTRON_RADIUS_A**2 * np.abs(expected_event) ** 2,
        rtol=2e-15,
        atol=0.0,
    )


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
        q_sample_normal_Ainv=np.asarray((0.2, 0.3)),
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
        expected_plus.append(sum(plus_terms))

    np.testing.assert_allclose(result.f_plus_e, expected_plus, rtol=1e-12, atol=1e-10)
    assert (
        np.array_equal(result.event_id, query.event_id)
        and np.array_equal(result.rod_id, query.rod_id)
        and result.phase_id == query.phase_id
    )
    assert (result.normalization, result.phase_sign, result.gauge_id) == (
        "ONE_REGISTRY_FREE_LAYER",
        "POSITIVE_Q_DOT_R",
        "pbi2.pb_centered.v1",
    )
    expected_normal = np.cross(crystal.direct_basis_A[:, 0], crystal.direct_basis_A[:, 1])
    expected_normal /= np.linalg.norm(expected_normal)
    np.testing.assert_allclose(result.layer_normal_crystal, expected_normal)
    assert np.isclose(
        result.layer_repeat_A, np.dot(crystal.direct_basis_A[:, 2], result.layer_normal_crystal)
    )

    double = replace(
        crystal,
        phase_id="pbi2-double",
        direct_basis_A=crystal.direct_basis_A @ np.diag((1.0, 1.0, 2.0)),
        volume_A3=2.0 * crystal.volume_A3,
        sites=tuple(
            replace(site, fractional=(*site.fractional[:2], (site.fractional[2] + offset) / 2.0))
            for offset in (0.0, 1.0)
            for site in crystal.sites
        ),
    )
    with pytest.raises(ValueError, match="exactly one PbI2 motif"):
        pbi2_layer_amplitudes(
            double, replace(query, phase_id=(double.phase_id,) * 2), unknown_u_iso_A2=0.0
        )


def test_finite_stack_matches_direct_sum_and_bragg_limit() -> None:
    event_id = np.asarray((17, 3))
    q_sample_normal_Ainv = np.asarray((0.0, 0.73))
    layer_amplitude_e = np.asarray(
        (
            (1.0 + 0.5j, 2.0 - 0.25j, -0.4 + 0.7j),
            (0.5j, 1.2 - 0.3j, 0.8 + 0.1j),
        )
    )
    layer_depth_A = np.asarray((0.0, 2.5, 7.25))
    result = coherent_finite_stack(event_id, q_sample_normal_Ainv, layer_amplitude_e, layer_depth_A)
    expected = np.sum(
        layer_amplitude_e * np.exp(1.0j * q_sample_normal_Ainv[:, None] * layer_depth_A),
        axis=1,
    )
    np.testing.assert_allclose(
        result.scattering_strength_A2, CLASSICAL_ELECTRON_RADIUS_A**2 * np.abs(expected) ** 2
    )

    repeat = np.asarray((1.2 + 0.4j,))
    bragg = uniform_finite_stack(
        event_id[:1],
        np.asarray((2.0 * np.pi / 2.5,)),
        repeat,
        2.5,
        7,
    )
    np.testing.assert_allclose(
        bragg.scattering_strength_A2,
        CLASSICAL_ELECTRON_RADIUS_A**2 * 49.0 * np.abs(repeat) ** 2,
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
