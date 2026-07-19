from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr

from painted_ewald import MosaicParameters, Rod, build_mosaic_space
from rasim_next.core.contracts import (
    IncidentSampleBatch,
    IncidentStateBatch,
    RodCatalog,
    canonical_revision_sha256,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.validity import ValidityCode
from rasim_next.reciprocal.events import build_scattering_events
from rasim_next.reciprocal.ewald import EwaldRootStatus, solve_continuous_rod_ewald
from rasim_next.sampling.mosaic import (
    MosaicOrientationBatch,
    WrappedMosaicParameters,
    manuscript_axisymmetric_v1_orientation_quadrature,
    wrapped_mosaic_line_density_rad_inv,
)
from rasim_next.sampling.source import sample_gaussian_source_rays

BI2SE3_RECIPROCAL_BASIS_AINV = np.array(
    [
        [1.516578640400576, 0.0, 0.0],
        [0.8755970862825088, 1.7511941725650182, 0.0],
        [0.0, 0.0, 0.2194156064806393],
    ]
)


@pytest.mark.parametrize("count", (6, 7))
def test_seeded_gaussian_source_has_exact_antithetic_lhs_strata(count: int) -> None:
    mean_origin = np.array([0.1, -0.2, 0.3])
    mean_direction = np.array([1.0, 0.0, 0.0])
    axes = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    spatial_sigma = np.array([1.0e-4, 2.0e-4])
    divergence_sigma = np.array([0.01, 0.02])
    arguments = {
        "mean_origin_lab_m": mean_origin,
        "mean_direction_lab": mean_direction,
        "transverse_axes_lab": axes,
        "spatial_sigma_m": spatial_sigma,
        "divergence_sigma_rad": divergence_sigma,
        "mean_wavelength_A": 1.24,
        "wavelength_sigma_A": 0.01,
        "sample_count": count,
        "polarization_state_id": "tabulated_state_7",
    }
    sampled = sample_gaussian_source_rays(**arguments, seed=1729)
    repeated = sample_gaussian_source_rays(**arguments, seed=1729)
    changed = sample_gaussian_source_rays(**arguments, seed=2718)
    for name in (
        "incident_sample_id",
        "origin_lab_m",
        "direction_lab",
        "wavelength_A",
        "source_weight",
    ):
        np.testing.assert_array_equal(getattr(sampled, name), getattr(repeated, name))
    assert not np.array_equal(sampled.origin_lab_m, changed.origin_lab_m)
    np.testing.assert_array_equal(sampled.source_weight, np.full(count, 1.0 / count))
    assert sampled.polarization_state_id == ("tabulated_state_7",) * count
    assert sampled.source_sampling_model_id == "independent_gaussian_antithetic_lhs.v2"
    assert sampled.source_rng_model_id == "numpy_pcg64.v1"
    assert sampled.source_seed == 1729
    assert sampled.source_parameter_revision == repeated.source_parameter_revision
    assert sampled.source_revision == repeated.source_revision
    assert changed.source_parameter_revision == sampled.source_parameter_revision
    assert changed.source_revision != sampled.source_revision
    assert '"mean_origin_lab_m":"m"' in sampled.source_parameter_provenance
    assert {
        "source_parameter_revision",
        "source_revision",
    }.isdisjoint(inspect.signature(IncidentSampleBatch).parameters)
    changed_provenance = replace(
        sampled,
        source_parameter_provenance=sampled.source_parameter_provenance + " ",
    )
    assert changed_provenance.source_parameter_revision != sampled.source_parameter_revision
    assert changed_provenance.source_revision != sampled.source_revision

    paired_stop = 2 * (count // 2)
    np.testing.assert_allclose(
        sampled.origin_lab_m[:paired_stop:2] + sampled.origin_lab_m[1:paired_stop:2],
        np.broadcast_to(2.0 * mean_origin, (count // 2, 3)),
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        sampled.direction_lab[:paired_stop:2] @ axes.T
        + sampled.direction_lab[1:paired_stop:2] @ axes.T,
        np.zeros((count // 2, 2)),
        rtol=0.0,
        atol=8.0e-16,
    )
    np.testing.assert_allclose(
        sampled.wavelength_A[:paired_stop:2] + sampled.wavelength_A[1:paired_stop:2],
        np.full(count // 2, 2.48),
        rtol=0.0,
        atol=5.0e-16,
    )
    if count % 2:
        np.testing.assert_array_equal(sampled.origin_lab_m[-1], mean_origin)
        np.testing.assert_array_equal(sampled.direction_lab[-1], mean_direction)
        assert sampled.wavelength_A[-1] == 1.24

    spatial = ((sampled.origin_lab_m - mean_origin) @ axes.T) / spatial_sigma
    cosine = np.clip(sampled.direction_lab @ mean_direction, -1.0, 1.0)
    radius = np.arccos(cosine)
    inverse_sine = np.divide(radius, np.sin(radius), out=np.ones_like(radius), where=radius != 0.0)
    angular = sampled.direction_lab @ axes.T * inverse_sine[:, None] / divergence_sigma
    wavelength = (sampled.wavelength_A - 1.24) / 0.01
    standardized = np.column_stack((spatial, angular, wavelength))
    unit = ndtr(standardized)
    strata = np.floor(unit * count).astype(np.int64)
    assert np.all(unit > strata / count)
    assert np.all(unit < (strata + 1) / count)
    np.testing.assert_array_equal(
        np.sort(strata, axis=0), np.broadcast_to(np.arange(count)[:, None], (count, 5))
    )
    np.testing.assert_allclose(
        unit[:paired_stop:2] + unit[1:paired_stop:2],
        np.ones((count // 2, 5)),
        rtol=0.0,
        atol=2.0e-15,
    )
    if count % 2:
        np.testing.assert_array_equal(unit[-1], np.full(5, 0.5))


def test_axisymmetric_mosaic_integrates_direct_alpha_probability_mass() -> None:
    parameters = WrappedMosaicParameters(0.2, 0.3, 0.35)
    angle = np.linspace(-np.pi, np.pi, 512, endpoint=False)
    density = wrapped_mosaic_line_density_rad_inv(angle, parameters)
    assert density.sum() * (2.0 * np.pi / angle.size) == pytest.approx(1.0, abs=1.0e-10)
    np.testing.assert_allclose(
        wrapped_mosaic_line_density_rad_inv(-angle, parameters),
        density,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        wrapped_mosaic_line_density_rad_inv(angle + 2.0 * np.pi, parameters),
        density,
        rtol=1.0e-12,
        atol=1.0e-14,
    )

    basis = np.diag([2.0, 3.0, 4.0])
    direct_parameters = WrappedMosaicParameters(4.0, 0.0, 0.0)
    direct = manuscript_axisymmetric_v1_orientation_quadrature(
        direct_parameters,
        reciprocal_basis_Ainv=basis,
        alpha_cell_count=1,
        azimuth_cell_count=1,
    )
    node, weight = leggauss(16)
    expected_alpha = 0.5 * np.pi * (node + 1.0)
    np.testing.assert_allclose(direct.alpha_rad, expected_alpha, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        direct.probability_mass,
        np.pi * weight * wrapped_mosaic_line_density_rad_inv(expected_alpha, direct_parameters),
        rtol=1.0e-14,
        atol=0.0,
    )
    rotated_axis = direct.rotation_crystal[0] @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        rotated_axis,
        [
            np.sin(direct.alpha_rad[0]) * np.cos(direct.azimuth_rad[0]),
            np.sin(direct.alpha_rad[0]) * np.sin(direct.azimuth_rad[0]),
            np.cos(direct.alpha_rad[0]),
        ],
        atol=1.0e-12,
    )

    mixed = manuscript_axisymmetric_v1_orientation_quadrature(
        WrappedMosaicParameters(0.0, 0.3, 0.25),
        reciprocal_basis_Ainv=basis,
        alpha_cell_count=4,
        azimuth_cell_count=4,
    )
    mixed_parameters = WrappedMosaicParameters(0.0, 0.3, 0.25)
    mixed_density = wrapped_mosaic_line_density_rad_inv(angle, mixed_parameters)
    continuous_mass = mixed_density.sum() * (2.0 * np.pi / angle.size)
    assert continuous_mass + mixed_parameters.zero_tilt_probability_mass == pytest.approx(
        1.0, abs=1.0e-10
    )
    assert mixed.probability_mass[mixed.alpha_rad == 0.0].sum() == pytest.approx(0.75)
    assert mixed.probability_mass[mixed.alpha_rad > 0.0].sum() == pytest.approx(0.25)
    assert mixed.probability_mass.sum() == pytest.approx(1.0, abs=1.0e-10)
    with pytest.raises(ValueError, match="real"):
        manuscript_axisymmetric_v1_orientation_quadrature(
            parameters,
            reciprocal_basis_Ainv=basis.astype(np.complex128),
            alpha_cell_count=4,
            azimuth_cell_count=4,
        )


def test_painted_ewald_mixed_mosaic_space_matches_analytic_probability() -> None:
    gaussian_sigma_rad = np.deg2rad(5.0)
    lorentzian_half_width_rad = np.deg2rad(2.0)
    lorentzian_probability = 0.1
    space = build_mosaic_space(
        reciprocal_basis_Ainv=BI2SE3_RECIPROCAL_BASIS_AINV,
        crystal_to_sample=np.eye(3),
        parameters=MosaicParameters(
            gaussian_sigma_rad=gaussian_sigma_rad,
            lorentzian_half_width_rad=lorentzian_half_width_rad,
            lorentzian_probability=lorentzian_probability,
            alpha_panel_count=8,
            alpha_gauss_order=12,
            azimuth_count=16,
            azimuth_phase_rad=0.371,
        ),
    )

    assert np.all(np.isfinite(space.probability_mass))
    assert np.all(space.probability_mass > 0.0)
    assert space.probability_mass.sum() == pytest.approx(1.0, abs=1.0e-12)
    for harmonic in (1, 2, 3):
        beta_moment = np.sum(space.probability_mass * np.exp(1j * harmonic * space.beta_rad))
        assert abs(beta_moment) <= 2.0e-15

    for harmonic in (1, 2):
        expected = (1.0 - lorentzian_probability) * np.exp(
            -0.5 * (harmonic * gaussian_sigma_rad) ** 2
        ) + lorentzian_probability * np.exp(-harmonic * lorentzian_half_width_rad)
        observed = np.sum(space.probability_mass * np.cos(harmonic * space.alpha_rad))
        assert observed == pytest.approx(expected, abs=2.0e-13)

    shifted = build_mosaic_space(
        reciprocal_basis_Ainv=BI2SE3_RECIPROCAL_BASIS_AINV,
        crystal_to_sample=np.eye(3),
        parameters=replace(space.parameters, azimuth_phase_rad=0.917),
    )
    assert not np.array_equal(space.beta_rad, shifted.beta_rad)
    np.testing.assert_array_equal(space.alpha_rad, shifted.alpha_rad)
    np.testing.assert_array_equal(space.probability_mass, shifted.probability_mass)


@pytest.mark.parametrize("half_width_rad", (1.0e-4, 1.0e-7))
def test_painted_ewald_narrow_lorentzian_remains_normalized(half_width_rad: float) -> None:
    space = build_mosaic_space(
        reciprocal_basis_Ainv=np.eye(3),
        crystal_to_sample=np.eye(3),
        parameters=MosaicParameters(
            gaussian_sigma_rad=0.0,
            lorentzian_half_width_rad=half_width_rad,
            lorentzian_probability=1.0,
            alpha_panel_count=8,
            alpha_gauss_order=12,
            azimuth_count=4,
        ),
    )

    assert space.probability_mass.sum() == pytest.approx(1.0, abs=1.0e-12)
    observed = np.sum(space.probability_mass * np.cos(space.alpha_rad))
    assert observed == pytest.approx(np.exp(-half_width_rad), abs=2.0e-13)


def test_painted_ewald_mosaic_space_rejects_invalid_public_inputs() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        MosaicParameters(-0.1, 0.2, 0.1)
    with pytest.raises(ValueError, match="between zero and one"):
        MosaicParameters(0.1, 0.2, 1.1)
    with pytest.raises(ValueError, match="positive integer"):
        MosaicParameters(0.1, 0.2, 0.1, alpha_gauss_order=0)
    with pytest.raises(ValueError, match="nonsingular"):
        build_mosaic_space(
            reciprocal_basis_Ainv=np.zeros((3, 3)),
            crystal_to_sample=np.eye(3),
            parameters=MosaicParameters(0.1, 0.2, 0.1),
        )
    with pytest.raises(ValueError, match="proper orthogonal rotation"):
        build_mosaic_space(
            reciprocal_basis_Ainv=BI2SE3_RECIPROCAL_BASIS_AINV,
            crystal_to_sample=np.diag([1.0, 1.0, 2.0]),
            parameters=MosaicParameters(0.1, 0.2, 0.1),
        )


def test_painted_ewald_mosaic_slice_conserves_caps_rings_and_rigid_rotation() -> None:
    sample_angle_rad = 0.23
    crystal_to_sample = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(sample_angle_rad), -np.sin(sample_angle_rad)],
            [0.0, np.sin(sample_angle_rad), np.cos(sample_angle_rad)],
        ]
    )
    space = build_mosaic_space(
        reciprocal_basis_Ainv=BI2SE3_RECIPROCAL_BASIS_AINV,
        crystal_to_sample=crystal_to_sample,
        parameters=MosaicParameters(
            gaussian_sigma_rad=np.deg2rad(5.0),
            lorentzian_half_width_rad=np.deg2rad(2.0),
            lorentzian_probability=0.1,
            alpha_panel_count=8,
            alpha_gauss_order=12,
            azimuth_count=16,
        ),
    )
    c_hat = BI2SE3_RECIPROCAL_BASIS_AINV[:, 2]
    c_hat = c_hat / np.linalg.norm(c_hat)

    for rod, u_Ainv in (
        (Rod(0, 0), 3.0 * np.linalg.norm(BI2SE3_RECIPROCAL_BASIS_AINV[:, 2])),
        (Rod(0, 0), -2.0 * np.linalg.norm(BI2SE3_RECIPROCAL_BASIS_AINV[:, 2])),
        (Rod(1, 0), 3.0 * np.linalg.norm(BI2SE3_RECIPROCAL_BASIS_AINV[:, 2])),
    ):
        mosaic_slice = space.mosaic_slice(rod=rod, u_Ainv=u_Ainv)
        q_parallel = (
            rod.h * BI2SE3_RECIPROCAL_BASIS_AINV[:, 0] + rod.k * BI2SE3_RECIPROCAL_BASIS_AINV[:, 1]
        )
        unrotated_q = q_parallel + u_Ainv * c_hat
        expected_q = np.einsum(
            "ij,njk,k->ni",
            crystal_to_sample,
            space.rotation_crystal,
            unrotated_q,
            optimize=True,
        )

        assert mosaic_slice.probability_mass.sum() == pytest.approx(1.0, abs=1.0e-12)
        np.testing.assert_allclose(
            mosaic_slice.q_sample_Ainv,
            expected_q,
            rtol=2.0e-15,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            np.linalg.norm(mosaic_slice.q_sample_Ainv, axis=1),
            np.linalg.norm(unrotated_q),
            rtol=2.0e-15,
            atol=2.0e-15,
        )

        if rod.family_m == 0:
            expected_cosine = 0.9 * np.exp(-0.5 * np.deg2rad(5.0) ** 2) + 0.1 * np.exp(
                -np.deg2rad(2.0)
            )
            weighted_mean = np.sum(
                mosaic_slice.probability_mass[:, None] * mosaic_slice.q_sample_Ainv,
                axis=0,
            )
            np.testing.assert_allclose(
                weighted_mean,
                crystal_to_sample @ (u_Ainv * expected_cosine * c_hat),
                rtol=0.0,
                atol=2.0e-13,
            )


def test_ewald_roots_preserve_line_geometry_and_unclipped_jacobian() -> None:
    incident = np.array([0.0, 0.0, 4.0])
    direction = np.array([0.0, 0.0, 1.0])
    regular = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([1.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    expected_u = np.array([-4.0 - np.sqrt(15.0), -4.0 + np.sqrt(15.0)])
    assert regular.status is EwaldRootStatus.TWO_ROOT
    np.testing.assert_allclose([root.u_Ainv for root in regular.emittable_roots], expected_u)
    for root in regular.emittable_roots:
        np.testing.assert_allclose(root.q_sample_Ainv, [1.0, 0.0, root.u_Ainv])
        np.testing.assert_allclose(root.kf_sample_Ainv, incident + root.q_sample_Ainv)
        assert root.l_coordinate == pytest.approx(root.u_Ainv / 2.0)
        assert root.coarea_jacobian == pytest.approx(4.0 / np.sqrt(15.0))
        assert root.ewald_residual_Ainv <= 64.0 * np.finfo(np.float64).eps * 4.0

    tangent = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([4.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    no_root = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([4.1, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    direct = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.zeros(3),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    direct_tangent = solve_continuous_rod_ewald(
        ki_sample_Ainv=np.array([4.0, 0.0, 0.0]),
        q0_sample_Ainv=np.zeros(3),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    assert tangent.status is EwaldRootStatus.TANGENT and tangent.emittable_roots == ()
    assert no_root.status is EwaldRootStatus.NO_ROOT and no_root.emittable_roots == ()
    assert direct.status is EwaldRootStatus.TWO_ROOT
    assert direct.direct_beam_root_count == 1 and len(direct.emittable_roots) == 1
    np.testing.assert_array_equal(direct.emittable_roots[0].q_sample_Ainv, [0.0, 0.0, -8.0])
    assert direct_tangent.status is EwaldRootStatus.TANGENT
    assert direct_tangent.direct_beam_root_count == 1 and direct_tangent.emittable_roots == ()

    qx = np.nextafter(4.0, 0.0)
    line_shift = 1050.0
    original = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([qx, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    shifted = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([qx, 0.0, line_shift]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    for original_root, shifted_root in zip(
        original.emittable_roots, shifted.emittable_roots, strict=True
    ):
        np.testing.assert_array_equal(shifted_root.q_sample_Ainv, original_root.q_sample_Ainv)
        np.testing.assert_array_equal(shifted_root.kf_sample_Ainv, original_root.kf_sample_Ainv)
        assert shifted_root.ewald_residual_Ainv == original_root.ewald_residual_Ainv == 0.0
        assert shifted_root.coarea_jacobian == original_root.coarea_jacobian == 2.0**26
        assert shifted_root.u_Ainv == pytest.approx(
            original_root.u_Ainv - line_shift,
            abs=2.0 * abs(np.spacing(line_shift)),
        )
    with pytest.raises(ValueError, match="real"):
        solve_continuous_rod_ewald(
            ki_sample_Ainv=incident.astype(np.complex128),
            q0_sample_Ainv=np.array([1.0, 0.0, 0.0]),
            d_hat_sample=direction,
            b3_norm_Ainv=2.0,
        )


def test_event_builder_preserves_sparse_order_frames_and_factor_boundary() -> None:
    sample_ids = np.array([10, 20, 30])
    origin_lab_m = np.zeros((3, 3))
    direction_lab = np.tile([0.0, 0.0, 1.0], (3, 1))
    wavelength_A = np.array([1.1, 1.3, 1.5])
    source_weight = np.full(3, 1.0 / 3.0)
    polarization_ids = ("linear_s", "circular_plus", "unused_invalid")
    provenance = "sparse event-builder permanent fixture.v1"
    samples = IncidentSampleBatch(
        incident_sample_id=sample_ids,
        origin_lab_m=origin_lab_m,
        direction_lab=direction_lab,
        wavelength_A=wavelength_A,
        source_weight=source_weight,
        polarization_state_id=polarization_ids,
        source_sampling_model_id="explicit_test_source.v1",
        source_rng_model_id="no_rng.v1",
        source_seed=0,
        source_parameter_provenance=provenance,
    )
    state_wavelength_A = np.array([1.3, 1.1, 1.5])
    k0_Ainv = 2.0 * np.pi / state_wavelength_A
    direction_sample = np.zeros((3, 3))
    direction_sample[:2, 0] = 4.0 / k0_Ainv[:2]
    direction_sample[:2, 2] = np.sqrt(1.0 - direction_sample[:2, 0] ** 2)
    k_air_sample_Ainv = k0_Ainv[:, None] * direction_sample
    k_film_phase_sample_Ainv = np.zeros((3, 3))
    k_film_phase_sample_Ainv[:2, 0] = 4.0
    states = IncidentStateBatch(
        incident_state_id=np.array([101, 100, 999]),
        incident_sample_id=np.array([20, 10, 30]),
        sample_intersection_lab_m=np.zeros((3, 3)),
        direction_sample=direction_sample,
        k_air_sample_Ainv=k_air_sample_Ainv,
        k_film_phase_sample_Ainv=k_film_phase_sample_Ainv,
        kz_film_Ainv=np.zeros(3, dtype=np.complex128),
        entrance_amplitude=np.array([2.0 + 0.0j, 3.0 + 0.0j, 0.0 + 0.0j]),
        footprint_acceptance=np.array([0.2, 0.9, 0.0]),
        source_weight=np.full(3, 1.0 / 3.0),
        wavelength_A=state_wavelength_A,
        polarization_state_id=("circular_plus", "linear_s", "unused_invalid"),
        status=(ValidityCode.VALID, ValidityCode.VALID, ValidityCode.OUTSIDE_SUPPORT),
        valid=np.array([True, True, False]),
        source_sampling_model_id=samples.source_sampling_model_id,
        source_rng_model_id=samples.source_rng_model_id,
        source_seed=samples.source_seed,
        source_parameter_provenance=samples.source_parameter_provenance,
        source_parameter_revision=samples.source_parameter_revision,
        source_revision=samples.source_revision,
        sample_geometry_revision=canonical_revision_sha256(
            ("sample_geometry", "sparse event-builder permanent fixture.v1")
        ),
        material_revision=canonical_revision_sha256(
            ("material", "sparse event-builder permanent fixture.v1")
        ),
        incident_model_id="one_transmitted_channel.v1",
    )
    basis = np.diag([1.0, 1.0, 2.0])
    h = np.array([1, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    rods = RodCatalog(
        rod_id=np.arange(30, 30 + h.size, dtype=np.int64),
        phase_id=("phase",) * h.size,
        h=h,
        k=np.zeros(h.size, dtype=np.int32),
        family_id=("family",) * h.size,
        family_key=("family",) * h.size,
        qr_Ainv=h.astype(np.float64),
        reciprocal_basis_Ainv=basis,
        symmetry_metadata=("none",) * h.size,
    )
    mosaic_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    orientations = MosaicOrientationBatch(
        orientation_id=np.array([40]),
        alpha_rad=np.array([0.0]),
        azimuth_rad=np.array([np.pi / 2.0]),
        rotation_crystal=mosaic_rotation[None, :, :],
        probability_mass=np.array([1.0]),
        reciprocal_basis_Ainv=basis,
        model_id="manuscript_axisymmetric_v1",
    )
    sample_rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    transform = RigidTransform(
        sample_rotation,
        np.array([10.0, 20.0, 30.0]),
        FrameId.CRYSTAL,
        FrameId.SAMPLE,
    )
    result = build_scattering_events(
        incident_states=states,
        rods=rods,
        orientations=orientations,
        sample_from_crystal=transform,
    )

    events = result.events
    np.testing.assert_array_equal(events.event_id, np.arange(4))
    np.testing.assert_array_equal(events.incident_state_id, [101, 101, 100, 100])
    np.testing.assert_array_equal(events.orientation_id, np.full(4, 40))
    np.testing.assert_array_equal(events.rod_id, [30, 30, 30, 30])
    np.testing.assert_array_equal(events.wavelength_A, [1.3, 1.3, 1.1, 1.1])
    assert (
        result.status.root_status
        == (
            EwaldRootStatus.TWO_ROOT,
            EwaldRootStatus.TANGENT,
            *(EwaldRootStatus.NO_ROOT,) * 6,
        )
        * 2
    )
    np.testing.assert_array_equal(result.status.attempt_id, np.arange(16))
    np.testing.assert_array_equal(result.status.incident_state_id, [101] * 8 + [100] * 8)
    np.testing.assert_array_equal(result.status.rod_id, np.tile(rods.rod_id, 2))
    np.testing.assert_array_equal(result.status.orientation_id, np.full(16, 40))
    np.testing.assert_array_equal(
        result.status.emitted_root_count, np.tile([2, 0, 0, 0, 0, 0, 0, 0], 2)
    )
    np.testing.assert_array_equal(result.status.direct_beam_root_count, np.zeros(16, dtype=np.int8))

    q_crystal = events.q_internal_sample_Ainv @ sample_rotation @ mosaic_rotation
    reconstructed = (
        np.column_stack(
            (np.ones(events.event_id.size), np.zeros(events.event_id.size), events.l_coordinate)
        )
        @ basis.T
    )
    np.testing.assert_allclose(q_crystal, reconstructed, atol=2.0e-15)
    np.testing.assert_array_equal(events.q_sample_normal_Ainv, events.q_internal_sample_Ainv[:, 2])
    incident_by_state = {
        101: states.k_film_phase_sample_Ainv[0],
        100: states.k_film_phase_sample_Ainv[1],
    }
    expected_kf = np.stack(
        [
            incident_by_state[int(state_id)] + q
            for state_id, q in zip(
                events.incident_state_id, events.q_internal_sample_Ainv, strict=True
            )
        ]
    )
    np.testing.assert_allclose(events.kf_film_phase_sample_Ainv, expected_kf, atol=2.0e-15)
    direction_sample = sample_rotation @ (mosaic_rotation @ np.array([0.0, 0.0, 1.0]))
    expected_weight = 1.0 / np.abs(
        (
            events.kf_film_phase_sample_Ainv
            / np.linalg.norm(events.kf_film_phase_sample_Ainv, axis=1)[:, None]
        )
        @ direction_sample
    )
    np.testing.assert_allclose(events.reciprocal_weight, expected_weight, rtol=1.0e-14)
    np.testing.assert_allclose(
        np.linalg.norm(events.kf_film_phase_sample_Ainv, axis=1), 4.0, atol=4.0e-15
    )
    np.testing.assert_allclose(
        events.ewald_residual_Ainv,
        np.abs(np.linalg.norm(events.kf_film_phase_sample_Ainv, axis=1) - 4.0),
        atol=0.0,
    )
    assert events.status == (ValidityCode.VALID,) * 4
    np.testing.assert_array_equal(events.valid, np.ones(4, dtype=np.bool_))
    with pytest.raises(ValueError, match="CRYSTAL to SAMPLE"):
        build_scattering_events(
            incident_states=states,
            rods=rods,
            orientations=orientations,
            sample_from_crystal=transform.inverse(),
        )
