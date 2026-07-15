from __future__ import annotations

from dataclasses import replace

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


def test_source_sampling_preserves_joint_and_independent_probability_mass() -> None:
    origins = np.array([[0.0, 1.0e-4, 0.0], [0.0, -2.0e-4, 0.0]])
    directions = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]])
    wavelengths = np.array([1.20, 1.35])
    masses = np.array([0.25, 0.75])
    polarization_ids = ("unity_scalar", "unity_scalar")

    first = compile_joint_source_samples(
        origin_lab_m=origins,
        direction_lab=directions,
        wavelength_A=wavelengths,
        probability_mass=masses,
        polarization_state_id=polarization_ids,
    )
    second = compile_joint_source_samples(
        origin_lab_m=origins,
        direction_lab=directions,
        wavelength_A=wavelengths,
        probability_mass=masses,
        polarization_state_id=polarization_ids,
    )

    np.testing.assert_array_equal(first.incident_sample_id, [0, 1])
    np.testing.assert_array_equal(first.origin_lab_m, origins)
    np.testing.assert_array_equal(first.direction_lab, directions)
    np.testing.assert_array_equal(first.wavelength_A, wavelengths)
    np.testing.assert_array_equal(first.source_weight, masses)
    assert first.polarization_state_id == polarization_ids
    assert first.correlation_model == "explicit_joint"
    for name in (
        "incident_sample_id",
        "origin_lab_m",
        "direction_lab",
        "wavelength_A",
        "source_weight",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    with pytest.raises(ValueError, match="nonnegative"):
        compile_joint_source_samples(
            origin_lab_m=origins,
            direction_lab=directions,
            wavelength_A=wavelengths,
            probability_mass=np.array([-0.25, 1.25]),
            polarization_state_id=polarization_ids,
        )
    with pytest.raises(ValueError, match="unity_scalar"):
        compile_joint_source_samples(
            origin_lab_m=origins,
            direction_lab=directions,
            wavelength_A=wavelengths,
            probability_mass=masses,
            polarization_state_id=("unity_scalar", ""),
        )

    arguments = {
        "mean_origin_lab_m": np.array([0.1, -0.2, 0.3]),
        "mean_direction_lab": np.array([1.0, 0.0, 0.0]),
        "transverse_axes_lab": np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        "spatial_sigma_m": np.array([1.0e-4, 2.0e-4]),
        "divergence_sigma_rad": np.array([0.01, 0.02]),
        "mean_wavelength_A": 1.24,
        "wavelength_sigma_A": 0.01,
        "spatial_order": 3,
        "direction_order": 3,
        "wavelength_order": 3,
        "polarization_state_id": "unity_scalar",
    }

    first = compile_independent_gaussian_source_samples(**arguments)
    second = compile_independent_gaussian_source_samples(**arguments)

    assert first.incident_sample_id.size == 3**5
    assert first.source_weight.sum() == pytest.approx(1.0, abs=1.0e-12)
    np.testing.assert_allclose(
        np.average(first.origin_lab_m, axis=0, weights=first.source_weight),
        arguments["mean_origin_lab_m"],
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.average(first.wavelength_A, weights=first.source_weight) == pytest.approx(
        arguments["mean_wavelength_A"], abs=1.0e-15
    )
    np.testing.assert_allclose(
        np.linalg.norm(first.direction_lab, axis=1), 1.0, rtol=0.0, atol=1.0e-12
    )
    for name in (
        "incident_sample_id",
        "origin_lab_m",
        "direction_lab",
        "wavelength_A",
        "source_weight",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    zero_width = compile_independent_gaussian_source_samples(
        **{
            **arguments,
            "spatial_sigma_m": np.zeros(2),
            "divergence_sigma_rad": np.zeros(2),
            "wavelength_sigma_A": 0.0,
        }
    )
    assert zero_width.incident_sample_id.size == 1
    np.testing.assert_array_equal(zero_width.origin_lab_m[0], arguments["mean_origin_lab_m"])
    np.testing.assert_array_equal(zero_width.direction_lab[0], arguments["mean_direction_lab"])
    assert zero_width.wavelength_A[0] == arguments["mean_wavelength_A"]
    assert zero_width.source_weight[0] == 1.0

    with pytest.raises(ValueError, match="positive"):
        compile_independent_gaussian_source_samples(
            **{
                **arguments,
                "mean_wavelength_A": 0.01,
                "wavelength_sigma_A": 1.0,
            }
        )

    origins = np.array([[0.0, 0.0, 0.0], [0.0, 1.0e-3, 0.0]])
    origin_mass = np.array([0.4, 0.6])
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    direction_mass = np.array([0.25, 0.75])
    wavelengths = np.array([1.0, 1.5])
    wavelength_mass = np.array([0.2, 0.8])

    samples = compile_independent_source_samples(
        origin_lab_m=origins,
        origin_probability_mass=origin_mass,
        direction_lab=directions,
        direction_probability_mass=direction_mass,
        wavelength_A=wavelengths,
        wavelength_probability_mass=wavelength_mass,
        polarization_state_id="unity_scalar",
    )

    expected_rows = [
        (origin, direction, wavelength, om * dm * wm)
        for origin, om in zip(origins, origin_mass, strict=True)
        for direction, dm in zip(directions, direction_mass, strict=True)
        for wavelength, wm in zip(wavelengths, wavelength_mass, strict=True)
    ]
    np.testing.assert_array_equal(samples.incident_sample_id, np.arange(8))
    np.testing.assert_array_equal(samples.origin_lab_m, [row[0] for row in expected_rows])
    np.testing.assert_array_equal(samples.direction_lab, [row[1] for row in expected_rows])
    np.testing.assert_array_equal(samples.wavelength_A, [row[2] for row in expected_rows])
    np.testing.assert_allclose(
        samples.source_weight, [row[3] for row in expected_rows], rtol=0.0, atol=1.0e-16
    )
    assert samples.correlation_model == "independent_product"
    assert samples.source_weight.sum() == pytest.approx(1.0, abs=1.0e-12)

    with pytest.raises(ValueError, match="real"):
        compile_joint_source_samples(
            origin_lab_m=origins.astype(np.complex128),
            direction_lab=directions,
            wavelength_A=wavelengths,
            probability_mass=origin_mass,
            polarization_state_id=polarization_ids,
        )
    with pytest.raises(ValueError, match="real"):
        compile_independent_source_samples(
            origin_lab_m=origins,
            origin_probability_mass=origin_mass,
            direction_lab=directions.astype(np.complex128),
            direction_probability_mass=direction_mass,
            wavelength_A=wavelengths,
            wavelength_probability_mass=wavelength_mass,
            polarization_state_id="unity_scalar",
        )
    with pytest.raises(ValueError, match="real"):
        compile_independent_gaussian_source_samples(
            **{
                **arguments,
                "mean_origin_lab_m": arguments["mean_origin_lab_m"].astype(np.complex128),
            }
        )

    oversized_origins = np.zeros((512, 3))
    oversized_directions = np.tile([1.0, 0.0, 0.0], (513, 1))
    with pytest.raises(ValueError, match=r"262656.*262144"):
        compile_independent_source_samples(
            origin_lab_m=oversized_origins,
            origin_probability_mass=np.full(512, 1.0 / 512.0),
            direction_lab=oversized_directions,
            direction_probability_mass=np.full(513, 1.0 / 513.0),
            wavelength_A=np.ones(1),
            wavelength_probability_mass=np.ones(1),
            polarization_state_id="unity_scalar",
        )


def test_axisymmetric_mosaic_conserves_wrapped_mass_at_poles() -> None:
    angle = np.linspace(-np.pi, np.pi, 512, endpoint=False)
    step = 2.0 * np.pi / angle.size
    parameter_sets = (
        WrappedMosaicParameters(0.2, 0.3, 0.0),
        WrappedMosaicParameters(0.2, 0.3, 1.0),
        WrappedMosaicParameters(0.2, 0.3, 0.35),
    )

    for parameters in parameter_sets:
        density = wrapped_mosaic_line_density_rad_inv(angle, parameters)
        assert np.all(np.isfinite(density))
        assert np.all(density >= 0.0)
        assert density.sum() * step == pytest.approx(1.0, abs=1.0e-10)
        np.testing.assert_allclose(
            wrapped_mosaic_line_density_rad_inv(angle + 2.0 * np.pi, parameters),
            density,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            wrapped_mosaic_line_density_rad_inv(-angle, parameters),
            density,
            rtol=1.0e-12,
            atol=1.0e-14,
        )

    parameters = parameter_sets[-1]
    assert parameters.gaussian_fwhm_rad == pytest.approx(2.0 * np.sqrt(2.0 * np.log(2.0)) * 0.2)
    assert parameters.lorentzian_fwhm_rad == pytest.approx(0.6)
    zero_core = WrappedMosaicParameters(0.0, 0.3, 0.25)
    assert zero_core.zero_tilt_probability_mass == pytest.approx(0.75)
    with pytest.raises(ValueError, match="discrete zero-tilt mass"):
        wrapped_mosaic_line_density_rad_inv(np.array([0.0]), zero_core)
    all_atom = WrappedMosaicParameters(0.0, 0.0, 0.4)
    assert all_atom.zero_tilt_probability_mass == 1.0
    pure_tail = wrapped_mosaic_line_density_rad_inv(angle, WrappedMosaicParameters(0.0, 0.3, 1.0))
    pure_core = wrapped_mosaic_line_density_rad_inv(angle, WrappedMosaicParameters(0.2, 0.0, 0.0))
    assert np.all(np.isfinite(pure_tail))
    assert np.all(np.isfinite(pure_core))
    assert pure_tail.sum() * step == pytest.approx(1.0, abs=1.0e-10)
    assert pure_core.sum() * step == pytest.approx(1.0, abs=1.0e-10)

    with pytest.raises(ValueError, match="nonnegative"):
        WrappedMosaicParameters(-0.1, 0.2, 0.5)
    with pytest.raises(ValueError, match="between zero and one"):
        WrappedMosaicParameters(0.1, 0.2, 1.1)

    basis = np.diag([2.0, 3.0, 4.0])
    parameters = WrappedMosaicParameters(0.12, 0.3, 0.25)
    first = manuscript_axisymmetric_v1_orientation_quadrature(
        parameters,
        reciprocal_basis_Ainv=basis,
        alpha_cell_count=8,
        azimuth_cell_count=4,
    )
    second = manuscript_axisymmetric_v1_orientation_quadrature(
        parameters,
        reciprocal_basis_Ainv=basis,
        alpha_cell_count=8,
        azimuth_cell_count=4,
    )

    assert first.model_id == "manuscript_axisymmetric_v1"
    assert first.probability_mass.sum() == pytest.approx(1.0, abs=1.0e-10)
    assert np.all(np.isfinite(first.probability_mass))
    assert np.all(first.probability_mass >= 0.0)
    np.testing.assert_allclose(
        first.rotation_crystal @ np.swapaxes(first.rotation_crystal, 1, 2),
        np.broadcast_to(np.eye(3), first.rotation_crystal.shape),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(np.linalg.det(first.rotation_crystal), 1.0, rtol=0.0, atol=1.0e-12)
    for name in (
        "orientation_id",
        "alpha_rad",
        "azimuth_rad",
        "rotation_crystal",
        "probability_mass",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    alpha = first.alpha_rad[0]
    azimuth = first.azimuth_rad[0]
    rotated_axis = first.rotation_crystal[0] @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        rotated_axis,
        [np.sin(alpha) * np.cos(azimuth), np.sin(alpha) * np.sin(azimuth), np.cos(alpha)],
        rtol=0.0,
        atol=1.0e-12,
    )

    pole = manuscript_axisymmetric_v1_orientation_quadrature(
        WrappedMosaicParameters(0.0, 0.0, 0.4),
        reciprocal_basis_Ainv=basis,
        alpha_cell_count=8,
        azimuth_cell_count=6,
    )
    np.testing.assert_array_equal(pole.alpha_rad, np.zeros(6))
    np.testing.assert_allclose(pole.probability_mass, np.full(6, 1.0 / 6.0))
    rotated_poles = pole.rotation_crystal @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(rotated_poles, np.tile([0.0, 0.0, 1.0], (6, 1)), atol=1e-12)

    narrow = manuscript_axisymmetric_v1_orientation_quadrature(
        WrappedMosaicParameters(1.0e-12, 0.0, 0.0),
        reciprocal_basis_Ainv=basis,
        alpha_cell_count=8,
        azimuth_cell_count=4,
    )
    assert narrow.probability_mass.sum() == pytest.approx(1.0, abs=1.0e-12)
    assert np.dot(narrow.alpha_rad, narrow.probability_mass) == pytest.approx(
        1.0e-12 * np.sqrt(2.0 / np.pi), rel=1.0e-8
    )
    assert narrow.probability_mass[narrow.alpha_rad > 1.0e-11].sum() < 1.0e-15

    direct = manuscript_axisymmetric_v1_orientation_quadrature(
        WrappedMosaicParameters(4.0, 0.0, 0.0),
        reciprocal_basis_Ainv=basis,
        alpha_cell_count=1,
        azimuth_cell_count=1,
    )
    direct_node, direct_weight = leggauss(16)
    expected_alpha = 0.5 * np.pi * (direct_node + 1.0)
    np.testing.assert_allclose(direct.alpha_rad, expected_alpha, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        direct.probability_mass,
        np.pi
        * direct_weight
        * wrapped_mosaic_line_density_rad_inv(
            expected_alpha, WrappedMosaicParameters(4.0, 0.0, 0.0)
        ),
        rtol=1.0e-14,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="integers"):
        replace(first, orientation_id=first.orientation_id.astype(np.float64))
    with pytest.raises(ValueError, match="alpha_rad"):
        replace(first, alpha_rad=np.full(first.alpha_rad.shape, -0.1))
    with pytest.raises(ValueError, match="azimuth_rad"):
        replace(first, azimuth_rad=np.full(first.azimuth_rad.shape, 2.0 * np.pi))
    with pytest.raises(ValueError, match="nonsingular"):
        replace(first, reciprocal_basis_Ainv=np.zeros((3, 3)))

    mixed = manuscript_axisymmetric_v1_orientation_quadrature(
        WrappedMosaicParameters(0.0, 0.3, 0.25),
        reciprocal_basis_Ainv=basis,
        alpha_cell_count=8,
        azimuth_cell_count=4,
    )
    assert mixed.probability_mass[mixed.alpha_rad == 0.0].sum() == pytest.approx(
        0.75, rel=0.0, abs=1.0e-12
    )
    assert mixed.probability_mass[mixed.alpha_rad > 0.0].sum() == pytest.approx(
        0.25, rel=0.0, abs=1.0e-12
    )

    with pytest.raises(ValueError, match="real"):
        wrapped_mosaic_line_density_rad_inv(angle.astype(np.complex128), parameters)
    with pytest.raises(ValueError, match="real"):
        manuscript_axisymmetric_v1_orientation_quadrature(
            parameters,
            reciprocal_basis_Ainv=basis.astype(np.complex128),
            alpha_cell_count=8,
            azimuth_cell_count=4,
        )
    with pytest.raises(ValueError, match="real"):
        replace(first, alpha_rad=first.alpha_rad.astype(np.complex128))


def test_ewald_roots_cover_statuses_direct_beam_and_unclipped_jacobian() -> None:
    incident = np.array([0.0, 0.0, 4.0])
    direction = np.array([0.0, 0.0, 1.0])
    two = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([1.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    assert two.status is EwaldRootStatus.TWO_ROOT
    assert len(two.emittable_roots) == 2
    assert [root.u_Ainv for root in two.emittable_roots] == sorted(
        root.u_Ainv for root in two.emittable_roots
    )
    for root in two.emittable_roots:
        np.testing.assert_allclose(
            root.q_sample_Ainv,
            np.array([1.0, 0.0, root.u_Ainv]),
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(root.kf_sample_Ainv, incident + root.q_sample_Ainv)
        assert root.l_coordinate == pytest.approx(root.u_Ainv / 2.0)
        assert root.ewald_residual_Ainv <= 64.0 * np.finfo(np.float64).eps * 4.0

    tangent = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([4.0, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    assert tangent.status is EwaldRootStatus.TANGENT
    assert tangent.emittable_roots == ()

    no_root = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([4.1, 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    assert no_root.status is EwaldRootStatus.NO_ROOT
    assert no_root.emittable_roots == ()

    specular = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.zeros(3),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    assert specular.status is EwaldRootStatus.TWO_ROOT
    assert specular.direct_beam_root_count == 1
    assert len(specular.emittable_roots) == 1
    np.testing.assert_array_equal(specular.emittable_roots[0].q_sample_Ainv, [0.0, 0.0, -8.0])

    tangent_direct = solve_continuous_rod_ewald(
        ki_sample_Ainv=np.array([4.0, 0.0, 0.0]),
        q0_sample_Ainv=np.zeros(3),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    assert tangent_direct.status is EwaldRootStatus.TANGENT
    assert tangent_direct.direct_beam_root_count == 1

    near_direct = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([1.0e-15, 0.0, 1.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    assert near_direct.direct_beam_root_count == 0
    assert len(near_direct.emittable_roots) == 2

    direct_near_tangent = solve_continuous_rod_ewald(
        ki_sample_Ainv=np.array(
            [-0.5284075151649418, 3.839076919837225, 0.9910973219065503]
        ),
        q0_sample_Ainv=np.zeros(3),
        d_hat_sample=np.array(
            [-0.9464149529866339, -0.04780873374614963, -0.31939483674739827]
        ),
        b3_norm_Ainv=1.0,
    )
    assert direct_near_tangent.status is not EwaldRootStatus.NO_ROOT
    assert direct_near_tangent.direct_beam_root_count == 1

    rounded_cross = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array(
            [-9.48242677221786, 0.5389383379354604, 1.8573945690457216]
        ),
        d_hat_sample=np.array(
            [-0.979828056869813, 0.055689004209250374, 0.19192632383523775]
        ),
        b3_norm_Ainv=1.0,
    )
    assert rounded_cross.direct_beam_root_count == 0
    assert len(rounded_cross.emittable_roots) == 2

    raw_direction = np.array(
        [-0.7495768210882333, -0.5975934847268614, -0.2846341797804071]
    )
    direct_from_raw_direction = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=raw_direction,
        d_hat_sample=raw_direction,
        b3_norm_Ainv=1.0,
    )
    assert direct_from_raw_direction.direct_beam_root_count == 1
    assert len(direct_from_raw_direction.emittable_roots) == 1

    corrected = solve_continuous_rod_ewald(
        ki_sample_Ainv=np.array([-0.19738778364534398, 0.04872338252052162, -0.6042577660404905]),
        q0_sample_Ainv=np.array([-2.4418544351606326, 9.418223922347948, 2.844707618400086]),
        d_hat_sample=np.array([0.24219174298077156, -0.9535915515058282, -0.17890308138386882]),
        b3_norm_Ainv=2.0,
    )
    corrected_limit = (
        64.0
        * np.finfo(np.float64).eps
        * max(np.linalg.norm([-0.19738778364534398, 0.04872338252052162, -0.6042577660404905]), 1.0)
    )
    assert max(root.ewald_residual_Ainv for root in corrected.emittable_roots) <= corrected_limit

    for qx_Ainv, line_shift_Ainv in (
        (np.nextafter(4.0, 0.0), 2.0),
        (1.0, 1050.0),
    ):
        original = solve_continuous_rod_ewald(
            ki_sample_Ainv=incident,
            q0_sample_Ainv=np.array([qx_Ainv, 0.0, 0.0]),
            d_hat_sample=direction,
            b3_norm_Ainv=2.0,
        )
        shifted = solve_continuous_rod_ewald(
            ki_sample_Ainv=incident,
            q0_sample_Ainv=np.array([qx_Ainv, 0.0, line_shift_Ainv]),
            d_hat_sample=direction,
            b3_norm_Ainv=2.0,
        )
        assert original.status is shifted.status is EwaldRootStatus.TWO_ROOT
        for original_root, shifted_root in zip(
            original.emittable_roots, shifted.emittable_roots, strict=True
        ):
            np.testing.assert_array_equal(
                shifted_root.q_sample_Ainv, original_root.q_sample_Ainv
            )
            np.testing.assert_array_equal(
                shifted_root.kf_sample_Ainv, original_root.kf_sample_Ainv
            )
            assert shifted_root.ewald_residual_Ainv == original_root.ewald_residual_Ainv
            assert shifted_root.coarea_jacobian == original_root.coarea_jacobian
            shift_tolerance = 2.0 * abs(np.spacing(line_shift_Ainv))
            assert shifted_root.u_Ainv == pytest.approx(
                original_root.u_Ainv - line_shift_Ainv,
                rel=0.0,
                abs=shift_tolerance,
            )
            assert shifted_root.l_coordinate == pytest.approx(
                (original_root.u_Ainv - line_shift_Ainv) / 2.0,
                rel=0.0,
                abs=shift_tolerance,
            )

    near_tangent = solve_continuous_rod_ewald(
        ki_sample_Ainv=incident,
        q0_sample_Ainv=np.array([np.nextafter(4.0, 0.0), 0.0, 0.0]),
        d_hat_sample=direction,
        b3_norm_Ainv=2.0,
    )
    np.testing.assert_array_equal(
        [root.coarea_jacobian for root in near_tangent.emittable_roots],
        np.full(2, 2.0**26),
    )

    with pytest.raises(ValueError, match="real"):
        solve_continuous_rod_ewald(
            ki_sample_Ainv=incident.astype(np.complex128),
            q0_sample_Ainv=np.array([1.0, 0.0, 0.0]),
            d_hat_sample=direction,
            b3_norm_Ainv=2.0,
        )

    def signed_residual(u_Ainv: float, qx_Ainv: float) -> float:
        q = np.array([qx_Ainv, 0.0, u_Ainv])
        return float(np.linalg.norm(incident + q) - np.linalg.norm(incident))

    largest_jacobian: list[float] = []
    for qx in (3.0, 3.9, 3.999):
        result = solve_continuous_rod_ewald(
            ki_sample_Ainv=incident,
            q0_sample_Ainv=np.array([qx, 0.0, 0.0]),
            d_hat_sample=direction,
            b3_norm_Ainv=2.0,
        )

        jacobians = []
        for root in result.emittable_roots:
            step = 1.0e-5
            derivative = abs(
                (signed_residual(root.u_Ainv + step, qx) - signed_residual(root.u_Ainv - step, qx))
                / (2.0 * step)
            )
            assert root.coarea_jacobian == pytest.approx(1.0 / derivative, rel=1.0e-8)
            jacobians.append(root.coarea_jacobian)
        largest_jacobian.append(max(jacobians))
    assert largest_jacobian[0] < largest_jacobian[1] < largest_jacobian[2]


def test_event_builder_joins_wavelength_and_applies_only_reciprocal_mass() -> None:
    samples = IncidentSampleBatch(
        incident_sample_id=np.array([10, 20]),
        origin_lab_m=np.zeros((2, 3)),
        direction_lab=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        wavelength_A=np.array([1.1, 1.3]),
        source_weight=np.array([0.7, 0.3]),
        polarization_state_id=("unity_scalar", "unity_scalar"),
        correlation_model="explicit_joint",
    )
    states = IncidentStateBatch(
        incident_state_id=np.array([101, 100, 999]),
        incident_sample_id=np.array([20, 10, 10]),
        sample_intersection_lab_m=np.zeros((3, 3)),
        direction_sample=np.tile([0.0, 0.0, 1.0], (3, 1)),
        k_air_sample_Ainv=np.tile([0.0, 0.0, 7.0], (3, 1)),
        k_film_phase_sample_Ainv=np.tile([0.0, 0.0, 4.0], (3, 1)),
        kz_film_Ainv=np.full(3, 4.0 + 0.0j),
        entrance_amplitude=np.array([2.0 + 0.0j, 3.0 + 0.0j, 100.0 + 0.0j]),
        footprint_acceptance=np.array([0.2, 0.9, 1.0]),
        source_weight=np.array([0.3, 0.7, 1.0]),
        valid=np.array([True, True, False]),
    )
    basis = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.5, 0.0, 4.0]])
    rods = RodCatalog(
        rod_id=np.array([30, 31]),
        phase_id=("phase", "phase"),
        h=np.array([1, 1], dtype=np.int32),
        k=np.array([0, 0], dtype=np.int32),
        family_id=("family", "family"),
        family_key=("family", "family"),
        qr_Ainv=np.full(2, np.sqrt(5.25)),
        reciprocal_basis_Ainv=basis,
        symmetry_metadata=("none", "none"),
    )
    orientation_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.sqrt(3.0) / 2.0, -0.5],
            [0.0, 0.5, np.sqrt(3.0) / 2.0],
        ]
    )
    orientations = MosaicOrientationBatch(
        orientation_id=np.array([40]),
        alpha_rad=np.array([np.pi / 6.0]),
        azimuth_rad=np.array([0.0]),
        rotation_crystal=orientation_rotation[None, :, :],
        probability_mass=np.array([1.0]),
        reciprocal_basis_Ainv=basis,
        model_id="manuscript_axisymmetric_v1",
    )
    sample_from_crystal = RigidTransform(
        rotation=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        translation_m=np.array([10.0, 20.0, 30.0]),
        source_frame=FrameId.CRYSTAL,
        target_frame=FrameId.SAMPLE,
    )

    result = build_scattering_events(
        incident_samples=samples,
        incident_states=states,
        rods=rods,
        orientations=orientations,
        sample_from_crystal=sample_from_crystal,
    )

    events = result.events
    np.testing.assert_array_equal(events.event_id, np.arange(8))
    np.testing.assert_array_equal(events.incident_state_id, [101] * 4 + [100] * 4)
    np.testing.assert_array_equal(events.rod_id, [30, 30, 31, 31] * 2)
    np.testing.assert_array_equal(events.wavelength_A, [1.3] * 4 + [1.1] * 4)
    q_crystal = (
        events.q_internal_sample_Ainv
        @ sample_from_crystal.rotation
        @ orientation_rotation
    )
    reconstructed = (
        np.column_stack(
            (
                np.ones(events.event_id.size),
                np.zeros(events.event_id.size),
                events.l_coordinate,
            )
        )
        @ basis.T
    )
    np.testing.assert_allclose(q_crystal, reconstructed, atol=2.0e-15)
    direction_sample = sample_from_crystal.rotation @ (
        orientation_rotation @ (basis[:, 2] / np.linalg.norm(basis[:, 2]))
    )
    expected_weight = 1.0 / np.abs(
        (events.kf_film_phase_sample_Ainv / np.linalg.norm(
            events.kf_film_phase_sample_Ainv, axis=1
        )[:, None])
        @ direction_sample
    )
    np.testing.assert_allclose(events.reciprocal_weight, expected_weight, rtol=1.0e-14)
    np.testing.assert_allclose(
        np.linalg.norm(events.kf_film_phase_sample_Ainv, axis=1), 4.0, rtol=0.0, atol=4.0e-15
    )
    np.testing.assert_allclose(events.ewald_residual_Ainv, 0.0, atol=4.0e-15)
    np.testing.assert_array_equal(events.valid, np.ones(8, dtype=bool))
    assert result.status.root_status == (EwaldRootStatus.TWO_ROOT,) * 4
    np.testing.assert_array_equal(result.status.emitted_root_count, np.full(4, 2))
    assert 999 not in result.status.incident_state_id

    with pytest.raises(ValueError, match="root counts"):
        replace(result.status, emitted_root_count=np.zeros(4, dtype=np.int8))

    with pytest.raises(ValueError, match="CRYSTAL to SAMPLE"):
        build_scattering_events(
            incident_samples=samples,
            incident_states=states,
            rods=rods,
            orientations=orientations,
            sample_from_crystal=sample_from_crystal.inverse(),
        )

    empty_basis = np.diag([4.0, 4.1, 2.0])
    empty_rods = RodCatalog(
        rod_id=np.array([50, 51]),
        phase_id=("phase", "phase"),
        h=np.array([1, 0], dtype=np.int32),
        k=np.array([0, 1], dtype=np.int32),
        family_id=("tangent", "none"),
        family_key=("tangent", "none"),
        qr_Ainv=np.array([4.0, 4.1]),
        reciprocal_basis_Ainv=empty_basis,
        symmetry_metadata=("none", "none"),
    )
    empty_orientations = MosaicOrientationBatch(
        orientation_id=np.array([0]),
        alpha_rad=np.array([0.0]),
        azimuth_rad=np.array([0.0]),
        rotation_crystal=np.eye(3)[None, :, :],
        probability_mass=np.array([1.0]),
        reciprocal_basis_Ainv=empty_basis,
        model_id="manuscript_axisymmetric_v1",
    )
    empty = build_scattering_events(
        incident_samples=samples,
        incident_states=states,
        rods=empty_rods,
        orientations=empty_orientations,
        sample_from_crystal=RigidTransform(
            np.eye(3), np.zeros(3), FrameId.CRYSTAL, FrameId.SAMPLE
        ),
    )
    assert empty.status.root_status == (
        EwaldRootStatus.TANGENT,
        EwaldRootStatus.NO_ROOT,
        EwaldRootStatus.TANGENT,
        EwaldRootStatus.NO_ROOT,
    )
    np.testing.assert_array_equal(empty.status.emitted_root_count, np.zeros(4, dtype=np.int8))
    np.testing.assert_array_equal(empty.status.direct_beam_root_count, np.zeros(4, dtype=np.int8))
    for name in (
        "event_id",
        "incident_state_id",
        "rod_id",
        "wavelength_A",
        "qz_Ainv",
        "l_coordinate",
        "reciprocal_weight",
        "ewald_residual_Ainv",
        "valid",
    ):
        assert getattr(empty.events, name).shape == (0,)
    assert empty.events.q_internal_sample_Ainv.shape == (0, 3)
    assert empty.events.kf_film_phase_sample_Ainv.shape == (0, 3)
