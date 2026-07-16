from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from rasim_next.core.contracts import IncidentSampleBatch, IncidentStateBatch, RodCatalog
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.reciprocal.events import build_scattering_events
from rasim_next.reciprocal.ewald import EwaldRootStatus, solve_continuous_rod_ewald
from rasim_next.sampling.mosaic import (
    MosaicOrientationBatch,
    WrappedMosaicParameters,
    manuscript_axisymmetric_v1_orientation_quadrature,
    wrapped_mosaic_line_density_rad_inv,
)
from rasim_next.sampling.source import (
    compile_independent_gaussian_source_samples,
    compile_independent_source_samples,
    compile_joint_source_samples,
)


def test_source_sampling_preserves_mass_correlation_and_allocation_budget() -> None:
    origins = np.array([[0.0, 1.0e-4, 0.0], [0.0, -2.0e-4, 0.0]])
    directions = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]])
    wavelengths = np.array([1.20, 1.35])
    masses = np.array([0.25, 0.75])
    joint = compile_joint_source_samples(
        origin_lab_m=origins,
        direction_lab=directions,
        wavelength_A=wavelengths,
        probability_mass=masses,
        polarization_state_id=("unity_scalar", "unity_scalar"),
    )
    np.testing.assert_array_equal(joint.incident_sample_id, [0, 1])
    np.testing.assert_array_equal(joint.origin_lab_m, origins)
    np.testing.assert_array_equal(joint.direction_lab, directions)
    np.testing.assert_array_equal(joint.wavelength_A, wavelengths)
    np.testing.assert_array_equal(joint.source_weight, masses)
    assert joint.correlation_model == "explicit_joint"

    origin_mass = np.array([0.4, 0.6])
    direction_mass = np.array([0.25, 0.75])
    wavelength_mass = np.array([0.2, 0.8])
    independent = compile_independent_source_samples(
        origin_lab_m=origins,
        origin_probability_mass=origin_mass,
        direction_lab=directions,
        direction_probability_mass=direction_mass,
        wavelength_A=wavelengths,
        wavelength_probability_mass=wavelength_mass,
        polarization_state_id="unity_scalar",
    )
    expected = [
        (origin, direction, wavelength, origin_weight * direction_weight * wavelength_weight)
        for origin, origin_weight in zip(origins, origin_mass, strict=True)
        for direction, direction_weight in zip(directions, direction_mass, strict=True)
        for wavelength, wavelength_weight in zip(wavelengths, wavelength_mass, strict=True)
    ]
    np.testing.assert_array_equal(independent.origin_lab_m, [row[0] for row in expected])
    np.testing.assert_array_equal(independent.direction_lab, [row[1] for row in expected])
    np.testing.assert_array_equal(independent.wavelength_A, [row[2] for row in expected])
    np.testing.assert_allclose(
        independent.source_weight, [row[3] for row in expected], rtol=0.0, atol=1.0e-16
    )
    assert independent.correlation_model == "independent_product"
    assert independent.source_weight.sum() == pytest.approx(1.0, abs=1.0e-12)

    gaussian = compile_independent_gaussian_source_samples(
        mean_origin_lab_m=np.array([0.1, -0.2, 0.3]),
        mean_direction_lab=np.array([1.0, 0.0, 0.0]),
        transverse_axes_lab=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        spatial_sigma_m=np.array([1.0e-4, 2.0e-4]),
        divergence_sigma_rad=np.array([0.01, 0.02]),
        mean_wavelength_A=1.24,
        wavelength_sigma_A=0.01,
        spatial_order=3,
        direction_order=3,
        wavelength_order=3,
        polarization_state_id="unity_scalar",
    )
    assert gaussian.incident_sample_id.size == 3**5
    assert gaussian.source_weight.sum() == pytest.approx(1.0, abs=1.0e-12)
    np.testing.assert_allclose(
        np.average(gaussian.origin_lab_m, axis=0, weights=gaussian.source_weight),
        [0.1, -0.2, 0.3],
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(np.linalg.norm(gaussian.direction_lab, axis=1), 1.0, atol=1.0e-12)

    with pytest.raises(ValueError, match="real"):
        compile_joint_source_samples(
            origin_lab_m=origins.astype(np.complex128),
            direction_lab=directions,
            wavelength_A=wavelengths,
            probability_mass=masses,
            polarization_state_id=("unity_scalar", "unity_scalar"),
        )
    with pytest.raises(ValueError, match=r"262656.*262144"):
        compile_independent_source_samples(
            origin_lab_m=np.zeros((512, 3)),
            origin_probability_mass=np.full(512, 1.0 / 512.0),
            direction_lab=np.tile([1.0, 0.0, 0.0], (513, 1)),
            direction_probability_mass=np.full(513, 1.0 / 513.0),
            wavelength_A=np.ones(1),
            wavelength_probability_mass=np.ones(1),
            polarization_state_id="unity_scalar",
        )


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
        np.pi
        * weight
        * wrapped_mosaic_line_density_rad_inv(expected_alpha, direct_parameters),
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
    samples = IncidentSampleBatch(
        incident_sample_id=np.array([10, 20]),
        origin_lab_m=np.zeros((2, 3)),
        direction_lab=np.tile([0.0, 0.0, 1.0], (2, 1)),
        wavelength_A=np.array([1.1, 1.3]),
        source_weight=np.array([0.7, 0.3]),
        polarization_state_id=("unity_scalar", "unity_scalar"),
        correlation_model="explicit_joint",
    )
    states = IncidentStateBatch(
        incident_state_id=np.array([101, 100, 999]),
        incident_sample_id=np.array([20, 10, 10]),
        sample_intersection_lab_m=np.zeros((3, 3)),
        direction_sample=np.tile([1.0, 0.0, 0.0], (3, 1)),
        k_air_sample_Ainv=np.tile([7.0, 0.0, 0.0], (3, 1)),
        k_film_phase_sample_Ainv=np.tile([4.0, 0.0, 0.0], (3, 1)),
        kz_film_Ainv=np.full(3, 4.0 + 0.0j),
        entrance_amplitude=np.array([2.0 + 0.0j, 3.0 + 0.0j, 100.0 + 0.0j]),
        footprint_acceptance=np.array([0.2, 0.9, 1.0]),
        source_weight=np.array([0.3, 0.7, 1.0]),
        valid=np.array([True, True, False]),
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
        incident_samples=samples,
        incident_states=states,
        rods=rods,
        orientations=orientations,
        sample_from_crystal=transform,
    )

    events = result.events
    np.testing.assert_array_equal(events.event_id, np.arange(4))
    np.testing.assert_array_equal(events.incident_state_id, [101, 101, 100, 100])
    np.testing.assert_array_equal(events.rod_id, [30, 30, 30, 30])
    np.testing.assert_array_equal(events.wavelength_A, [1.3, 1.3, 1.1, 1.1])
    assert result.status.root_status == (
        EwaldRootStatus.TWO_ROOT,
        EwaldRootStatus.TANGENT,
        *(EwaldRootStatus.NO_ROOT,) * 6,
    ) * 2
    np.testing.assert_array_equal(result.status.attempt_id, np.arange(16))
    np.testing.assert_array_equal(result.status.incident_state_id, [101] * 8 + [100] * 8)
    np.testing.assert_array_equal(result.status.rod_id, np.tile(rods.rod_id, 2))
    np.testing.assert_array_equal(result.status.orientation_id, np.full(16, 40))
    np.testing.assert_array_equal(result.status.emitted_root_count, np.tile([2, 0, 0, 0, 0, 0, 0, 0], 2))

    q_crystal = events.q_internal_sample_Ainv @ sample_rotation @ mosaic_rotation
    reconstructed = np.column_stack(
        (np.ones(events.event_id.size), np.zeros(events.event_id.size), events.l_coordinate)
    ) @ basis.T
    np.testing.assert_allclose(q_crystal, reconstructed, atol=2.0e-15)
    direction_sample = sample_rotation @ (mosaic_rotation @ np.array([0.0, 0.0, 1.0]))
    expected_weight = 1.0 / np.abs(
        (events.kf_film_phase_sample_Ainv / np.linalg.norm(
            events.kf_film_phase_sample_Ainv, axis=1
        )[:, None])
        @ direction_sample
    )
    np.testing.assert_allclose(events.reciprocal_weight, expected_weight, rtol=1.0e-14)
    np.testing.assert_allclose(
        np.linalg.norm(events.kf_film_phase_sample_Ainv, axis=1), 4.0, atol=4.0e-15
    )
    np.testing.assert_array_equal(events.valid, np.ones(4, dtype=np.bool_))
    with pytest.raises(ValueError, match="CRYSTAL to SAMPLE"):
        build_scattering_events(
            incident_samples=samples,
            incident_states=states,
            rods=rods,
            orientations=orientations,
            sample_from_crystal=transform.inverse(),
        )
