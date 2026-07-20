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


def test_painted_ewald_roots_match_reference_and_classification_contract() -> None:
    from painted_ewald.ewald import RootStatus, solve_infinite_rod_ewald

    incident = np.array([0.0, 0.0, -10.0])
    direction = np.array([0.0, 0.0, 1.0])
    solve_arguments = {
        "ki_sample_Ainv": incident,
        "d_hat_sample": direction,
        "b3_norm_Ainv": 0.2,
        "root_tolerance_rel": 256.0 * np.finfo(np.float64).eps,
        "residual_tolerance_rel": 512.0 * np.finfo(np.float64).eps,
    }

    direct_line = solve_infinite_rod_ewald(
        **solve_arguments,
        q0_sample_Ainv=np.zeros(3),
        rod_is_m0=True,
    )
    assert direct_line.status is RootStatus.REGULAR
    assert direct_line.direct_root_count == 1
    assert len(direct_line.emittable_roots) == 1
    direct_root = direct_line.emittable_roots[0]
    assert direct_root.branch == 0
    assert direct_root.u_Ainv == pytest.approx(20.0)
    assert pytest.approx(100.0) == direct_root.L
    assert direct_root.coarea_jacobian == pytest.approx(1.0)
    np.testing.assert_allclose(direct_root.q_sample_Ainv, [0.0, 0.0, 20.0])
    np.testing.assert_allclose(direct_root.kf_sample_Ainv, [0.0, 0.0, 10.0])

    for parallel_anchor, b3_norm in ((0.0, 0.2), (1.0e16, 0.2), (0.0, 1.0e16)):
        with pytest.raises(ValueError, match="must contain q=0"):
            solve_infinite_rod_ewald(
                **{**solve_arguments, "b3_norm_Ainv": b3_norm},
                q0_sample_Ainv=np.array([1.0, 0.0, parallel_anchor]),
                rod_is_m0=True,
            )
    with pytest.raises(ValueError, match="must contain q=0"):
        solve_infinite_rod_ewald(
            **{
                **solve_arguments,
                "ki_sample_Ainv": np.array([0.0, 0.0, -1.0e16]),
            },
            q0_sample_Ainv=np.array([1.0, 0.0, 0.0]),
            rod_is_m0=True,
        )
    oblique_direction = np.array([1.0, 2.0, 3.0])
    oblique_direction /= np.linalg.norm(oblique_direction)
    oblique_arguments = {**solve_arguments, "d_hat_sample": oblique_direction}
    centered_oblique = solve_infinite_rod_ewald(
        **oblique_arguments,
        q0_sample_Ainv=np.zeros(3),
        rod_is_m0=True,
    )
    shifted_oblique = solve_infinite_rod_ewald(
        **oblique_arguments,
        q0_sample_Ainv=1.0e4 * oblique_direction,
        rod_is_m0=True,
    )
    np.testing.assert_array_equal(
        shifted_oblique.emittable_roots[0].q_sample_Ainv,
        centered_oblique.emittable_roots[0].q_sample_Ainv,
    )
    np.testing.assert_array_equal(
        shifted_oblique.emittable_roots[0].kf_sample_Ainv,
        centered_oblique.emittable_roots[0].kf_sample_Ainv,
    )

    regular = solve_infinite_rod_ewald(
        **solve_arguments,
        q0_sample_Ainv=np.array([1.0, 0.0, 0.0]),
        rod_is_m0=False,
    )
    expected_u = np.array([10.0 - np.sqrt(99.0), 10.0 + np.sqrt(99.0)])
    assert regular.status is RootStatus.REGULAR
    assert regular.direct_root_count == 0
    np.testing.assert_allclose([root.u_Ainv for root in regular.emittable_roots], expected_u)
    np.testing.assert_allclose([root.L for root in regular.emittable_roots], expected_u / 0.2)
    assert [root.branch for root in regular.emittable_roots] == [1, 2]
    for root in regular.emittable_roots:
        assert root.coarea_jacobian == pytest.approx(10.0 / np.sqrt(99.0))
        assert root.ewald_residual_Ainv <= solve_arguments["residual_tolerance_rel"] * 10.0
        assert np.linalg.norm(root.q_sample_Ainv) <= 20.0

    anchor_shift = float(2**26)
    shifted = solve_infinite_rod_ewald(
        **solve_arguments,
        q0_sample_Ainv=np.array([1.0, 0.0, anchor_shift]),
        rod_is_m0=False,
    )
    assert shifted.status is RootStatus.REGULAR
    for original_root, shifted_root in zip(
        regular.emittable_roots, shifted.emittable_roots, strict=True
    ):
        assert shifted_root.branch == original_root.branch
        assert shifted_root.u_Ainv == pytest.approx(original_root.u_Ainv - anchor_shift)
        np.testing.assert_array_equal(shifted_root.q_sample_Ainv, original_root.q_sample_Ainv)
        np.testing.assert_array_equal(shifted_root.kf_sample_Ainv, original_root.kf_sample_Ainv)
        assert shifted_root.ewald_residual_Ainv == original_root.ewald_residual_Ainv
        assert shifted_root.coarea_jacobian == original_root.coarea_jacobian

    tangent = solve_infinite_rod_ewald(
        **solve_arguments,
        q0_sample_Ainv=np.array([10.0, 0.0, 0.0]),
        rod_is_m0=False,
    )
    no_root = solve_infinite_rod_ewald(
        **solve_arguments,
        q0_sample_Ainv=np.array([10.1, 0.0, 0.0]),
        rod_is_m0=False,
    )
    collapsed = solve_infinite_rod_ewald(
        **{**solve_arguments, "ki_sample_Ainv": np.array([10.0, 0.0, 0.0])},
        q0_sample_Ainv=np.zeros(3),
        rod_is_m0=True,
    )
    assert tangent.status is RootStatus.TANGENT and tangent.emittable_roots == ()
    assert no_root.status is RootStatus.NO_ROOT and no_root.emittable_roots == ()
    assert collapsed.status is RootStatus.COLLAPSED_DIRECT
    assert collapsed.direct_root_count == 1 and collapsed.emittable_roots == ()

    numerical_collapse = solve_infinite_rod_ewald(
        **{
            **solve_arguments,
            "ki_sample_Ainv": np.array([1.0e-15, 0.0, 10.0]),
            "d_hat_sample": np.array([1.0, 0.0, 0.0]),
        },
        q0_sample_Ainv=np.zeros(3),
        rod_is_m0=True,
    )
    assert numerical_collapse.status is RootStatus.COLLAPSED_DIRECT

    near_forward_direction = np.array([np.sqrt(1.0 - 1.0e-24), 0.0, 1.0e-12])
    near_forward = solve_infinite_rod_ewald(
        **{**solve_arguments, "d_hat_sample": near_forward_direction},
        q0_sample_Ainv=np.zeros(3),
        rod_is_m0=True,
    )
    assert near_forward.status is RootStatus.REGULAR
    near_forward_root = near_forward.emittable_roots[0]
    assert 0.0 < np.linalg.norm(near_forward_root.q_sample_Ainv) < 1.0e-9
    assert near_forward_root.coarea_jacobian > 1.0e10


def test_painted_ewald_painter_and_equal_solid_angle_texture_conserve_mass() -> None:
    from painted_ewald import (
        BranchCoatingSummary,
        EwaldSpherePainter,
        MassLedger,
        PainterConfig,
        RasterParameters,
        RodCoatingSummary,
    )

    config = PainterConfig(
        reciprocal_basis_Ainv=BI2SE3_RECIPROCAL_BASIS_AINV,
        crystal_to_sample=np.eye(3),
        rods=(Rod(0, 0), Rod(1, 0), Rod(0, 1)),
        mosaic=MosaicParameters(
            gaussian_sigma_rad=0.0,
            lorentzian_half_width_rad=0.0,
            lorentzian_probability=0.1,
            azimuth_count=1,
            azimuth_phase_rad=-np.pi,
        ),
        raster=RasterParameters(mu_bin_count=4, phi_bin_count=8),
    )
    painter = EwaldSpherePainter(config)
    incident = np.array([0.0, 0.0, -10.0])
    result = painter.paint(incident)

    np.testing.assert_array_equal(result.center_q_sample_Ainv, [0.0, 0.0, 10.0])
    assert result.radius_Ainv == pytest.approx(10.0)
    assert [point.branch for point in result.points] == [0, 1, 2, 1, 2]
    np.testing.assert_array_equal([point.weight for point in result.points], np.ones(5))
    assert all(point.coarea_jacobian is None for point in result.points)
    assert all(not point.q_sample_Ainv.flags.writeable for point in result.points)
    assert result.ledger.painted_weight == pytest.approx(5.0)
    assert result.ledger.collapsed_direct_base_mass == 0.0
    assert result.ledger.tangent_base_mass == 0.0
    assert result.ledger.no_root_base_mass == 0.0
    assert result.ledger.forward_excluded_base_mass == 0.0
    assert result.ledger.suppressed_algebraic_direct_root_count == 1

    texture = result.texture
    np.testing.assert_allclose(np.diff(texture.mu_edges), 0.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.diff(texture.phi_edges_rad), np.pi / 4.0)
    assert np.sum(texture.mass) == pytest.approx(5.0)
    solid_angle = np.diff(texture.mu_edges)[:, None] * np.diff(texture.phi_edges_rad)[None, :]
    np.testing.assert_allclose(texture.density_per_sr * solid_angle, texture.mass)
    for point in result.points:
        assert point.ewald_residual_Ainv <= config.residual_tolerance_rel * 10.0
        assert np.linalg.norm(point.q_sample_Ainv) <= 20.0

    coating = painter.paint_coating(incident)
    assert coating.retained_root_count == len(result.points)
    assert coating.maximum_ewald_residual_Ainv == max(
        point.ewald_residual_Ainv for point in result.points
    )
    assert coating.ledger == result.ledger
    np.testing.assert_array_equal(coating.texture.mass, result.texture.mass)
    np.testing.assert_array_equal(coating.texture.density_per_sr, result.texture.density_per_sr)
    assert [summary.rod for summary in coating.rod_summaries] == list(config.rods)
    assert [
        tuple(
            (branch.branch, branch.retained_root_count, branch.painted_weight)
            for branch in summary.branches
        )
        for summary in coating.rod_summaries
    ] == [
        ((0, 1, 1.0),),
        ((1, 1, 1.0), (2, 1, 1.0)),
        ((1, 1, 1.0), (2, 1, 1.0)),
    ]
    assert [summary.family_m for summary in coating.family_summaries] == [0, 1]
    assert [
        (branch.branch, branch.retained_root_count, branch.painted_weight)
        for branch in coating.family_summaries[1].branches
    ] == [(1, 2, 2.0), (2, 2, 2.0)]
    assert coating.family_summaries[1].ledger.painted_weight == 4.0
    assert coating.measure is config.measure
    assert coating.forward_policy == config.forward_policy
    assert coating.family_summaries[0].is_complete_hexagonal_family is True
    assert coating.family_summaries[1].is_complete_hexagonal_family is False
    with pytest.raises(ValueError, match="full integer shell"):
        replace(coating.family_summaries[1], is_complete_hexagonal_family=True)
    repeated_rod = coating.family_summaries[1].rod_summaries[0]
    with pytest.raises(ValueError, match="repeat a physical rod"):
        replace(coating.family_summaries[1], rod_summaries=(repeated_rod, repeated_rod))

    nonhexagonal_full_shell = PainterConfig(
        reciprocal_basis_Ainv=np.diag([1.0e-7, 1.0e-7, 0.2]),
        crystal_to_sample=np.eye(3),
        rods=tuple(Rod(h, k) for h, k in ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))),
        mosaic=config.mosaic,
        raster=config.raster,
    )
    nonhexagonal_coating = EwaldSpherePainter(nonhexagonal_full_shell).paint_coating(incident)
    assert nonhexagonal_coating.family_summaries[0].is_complete_hexagonal_family is False

    with pytest.raises(ValueError, match="zero retained roots"):
        BranchCoatingSummary(branch=1, retained_root_count=0, painted_weight=1.0)
    empty_branches = (
        BranchCoatingSummary(branch=1, retained_root_count=0, painted_weight=0.0),
        BranchCoatingSummary(branch=2, retained_root_count=0, painted_weight=0.0),
    )
    with pytest.raises(ValueError, match="nonzero rods cannot"):
        RodCoatingSummary(
            rod=Rod(1, 0),
            branches=empty_branches,
            ledger=MassLedger(0.0, 1.0, 0.0, 0.0, 1.0, 7),
            maximum_ewald_residual_Ainv=0.0,
        )
    with pytest.raises(ValueError, match="zero retained roots"):
        RodCoatingSummary(
            rod=Rod(1, 0),
            branches=empty_branches,
            ledger=MassLedger(0.0, 0.0, 0.0, 0.0, 0.0, 0),
            maximum_ewald_residual_Ainv=123.0,
        )
    with pytest.raises(ValueError, match="zero-population"):
        RodCoatingSummary(
            rod=Rod(1, 0, population=0.0),
            branches=(
                BranchCoatingSummary(branch=1, retained_root_count=1, painted_weight=2.0),
                BranchCoatingSummary(branch=2, retained_root_count=1, painted_weight=3.0),
            ),
            ledger=MassLedger(5.0, 0.0, 0.0, 0.0, 0.0, 0),
            maximum_ewald_residual_Ainv=0.0,
        )
    with pytest.raises(ValueError, match="equal retained-root counts"):
        RodCoatingSummary(
            rod=Rod(1, 0),
            branches=(
                BranchCoatingSummary(branch=1, retained_root_count=1, painted_weight=0.0),
                BranchCoatingSummary(branch=2, retained_root_count=2, painted_weight=0.0),
            ),
            ledger=MassLedger(0.0, 0.0, 0.0, 0.0, 0.0, 0),
            maximum_ewald_residual_Ainv=0.0,
        )
    with pytest.raises(ValueError, match="suppressed direct-root count"):
        RodCoatingSummary(
            rod=Rod(0, 0),
            branches=(BranchCoatingSummary(branch=0, retained_root_count=2, painted_weight=0.0),),
            ledger=MassLedger(0.0, 0.0, 0.0, 0.0, 0.0, 1),
            maximum_ewald_residual_Ainv=0.0,
        )
    with pytest.raises(ValueError, match="zero-population"):
        RodCoatingSummary(
            rod=Rod(1, 0, population=0.0),
            branches=empty_branches,
            ledger=MassLedger(0.0, 0.0, 0.0, 1.0, 0.0, 0),
            maximum_ewald_residual_Ainv=0.0,
        )

    with pytest.raises(ValueError, match="nonzero"):
        painter.paint(np.zeros(3))
    with pytest.raises(ValueError, match="real"):
        EwaldSpherePainter(config).paint(np.array([0.0, 0.0, -10.0 + 0.0j]))


def test_painted_ewald_measure_and_mass_ledger_are_explicit() -> None:
    from painted_ewald import (
        EwaldSpherePainter,
        ForwardPolicy,
        PainterConfig,
        PaintMeasure,
        RasterParameters,
    )

    class ConstantStrength:
        def __init__(self, value: float) -> None:
            self.value = value

        def evaluate(self, *, rod: Rod, L: float, k_norm_Ainv: float) -> float:
            return self.value

    zero_mosaic = MosaicParameters(0.0, 0.0, 0.1, azimuth_count=1, azimuth_phase_rad=-np.pi)
    regular_config = PainterConfig(
        reciprocal_basis_Ainv=np.diag([1.0, 1.0, 0.2]),
        crystal_to_sample=np.eye(3),
        rods=(Rod(1, 0, population=3.0),),
        mosaic=zero_mosaic,
        raster=RasterParameters(2, 4),
    )
    incident = np.array([0.0, 0.0, -10.0])
    pushforward = EwaldSpherePainter(regular_config, ConstantStrength(2.0)).paint(incident)
    np.testing.assert_allclose([point.weight for point in pushforward.points], [6.0, 6.0])
    assert all(point.coarea_jacobian is None for point in pushforward.points)
    pushforward_coating = EwaldSpherePainter(regular_config, ConstantStrength(2.0)).paint_coating(
        incident
    )
    assert pushforward_coating.ledger == pushforward.ledger
    np.testing.assert_array_equal(pushforward_coating.texture.mass, pushforward.texture.mass)

    coarea = EwaldSpherePainter(
        replace(regular_config, measure=PaintMeasure.COAREA_INTENSITY),
        ConstantStrength(2.0),
    ).paint(incident)
    expected_jacobian = 10.0 / np.sqrt(99.0)
    np.testing.assert_allclose(
        [point.weight for point in coarea.points],
        [6.0 * expected_jacobian, 6.0 * expected_jacobian],
    )
    np.testing.assert_allclose(
        [point.coarea_jacobian for point in coarea.points],
        [expected_jacobian, expected_jacobian],
    )
    coarea_coating = EwaldSpherePainter(
        replace(regular_config, measure=PaintMeasure.COAREA_INTENSITY),
        ConstantStrength(2.0),
    ).paint_coating(incident)
    assert coarea_coating.ledger.painted_weight == pytest.approx(
        coarea.ledger.painted_weight, abs=2.0e-15
    )
    np.testing.assert_allclose(
        coarea_coating.texture.mass,
        coarea.texture.mass,
        rtol=0.0,
        atol=2.0e-15,
    )

    with pytest.raises(ValueError, match="requires a physical q_min_Ainv"):
        PainterConfig(
            reciprocal_basis_Ainv=np.diag([1.0, 1.0, 0.2]),
            crystal_to_sample=np.eye(3),
            rods=(Rod(0, 0, population=3.0),),
            mosaic=zero_mosaic,
            measure=PaintMeasure.COAREA_INTENSITY,
        )
    cutoff_config = PainterConfig(
        reciprocal_basis_Ainv=np.diag([1.0, 1.0, 0.2]),
        crystal_to_sample=np.eye(3),
        rods=(Rod(0, 0, population=3.0),),
        mosaic=zero_mosaic,
        raster=RasterParameters(2, 4),
        measure=PaintMeasure.COAREA_INTENSITY,
        forward_policy=ForwardPolicy(q_min_Ainv=21.0),
    )
    excluded = EwaldSpherePainter(cutoff_config, ConstantStrength(2.0)).paint(incident)
    excluded_coating = EwaldSpherePainter(cutoff_config, ConstantStrength(2.0)).paint_coating(
        incident
    )
    assert excluded.points == ()
    assert excluded.ledger.painted_weight == 0.0
    assert excluded.ledger.forward_excluded_base_mass == pytest.approx(6.0)
    assert excluded.ledger.suppressed_algebraic_direct_root_count == 1
    assert excluded_coating.retained_root_count == 0
    assert excluded_coating.ledger == excluded.ledger
    with pytest.raises(ValueError, match="pushforward coating cannot"):
        replace(excluded_coating, measure=PaintMeasure.MOSAIC_PUSHFORWARD)

    retained_cutoff_config = replace(
        cutoff_config,
        forward_policy=ForwardPolicy(q_min_Ainv=19.0),
    )
    retained_cutoff = EwaldSpherePainter(retained_cutoff_config, ConstantStrength(2.0)).paint(
        incident
    )
    retained_cutoff_coating = EwaldSpherePainter(
        retained_cutoff_config, ConstantStrength(2.0)
    ).paint_coating(incident)
    assert len(retained_cutoff.points) == retained_cutoff_coating.retained_root_count == 1
    assert np.linalg.norm(retained_cutoff.points[0].q_sample_Ainv) >= 19.0
    assert retained_cutoff.points[0].coarea_jacobian == pytest.approx(1.0)
    assert retained_cutoff.ledger.forward_excluded_base_mass == 0.0
    assert retained_cutoff_coating.ledger == retained_cutoff.ledger
    np.testing.assert_array_equal(
        retained_cutoff_coating.texture.mass,
        retained_cutoff.texture.mass,
    )

    ledger_config = PainterConfig(
        reciprocal_basis_Ainv=np.diag([4.0, 1.0, 0.2]),
        crystal_to_sample=np.eye(3),
        rods=(Rod(0, 0, 2.0), Rod(-1, 0, 3.0), Rod(-2, 0, 5.0)),
        mosaic=zero_mosaic,
        raster=RasterParameters(2, 4),
    )
    ledger_result = EwaldSpherePainter(ledger_config).paint(np.array([2.0, 0.0, 0.0]))
    ledger_coating = EwaldSpherePainter(ledger_config).paint_coating(np.array([2.0, 0.0, 0.0]))
    assert ledger_result.points == ()
    assert ledger_result.ledger.collapsed_direct_base_mass == pytest.approx(2.0)
    assert ledger_result.ledger.tangent_base_mass == pytest.approx(3.0)
    assert ledger_result.ledger.no_root_base_mass == pytest.approx(5.0)
    assert ledger_result.ledger.suppressed_algebraic_direct_root_count == 1
    assert ledger_coating.ledger == ledger_result.ledger

    with pytest.raises(ValueError, match="nonnegative"):
        EwaldSpherePainter(regular_config, ConstantStrength(-1.0)).paint(incident)


def test_painted_ewald_streaming_matches_exact_rod_strengths_within_m() -> None:
    from math import cos, fsum, sin

    from painted_ewald import EwaldSpherePainter, PainterConfig, RasterParameters

    direction = np.array([1.0, 2.0, 3.0])
    direction /= np.linalg.norm(direction)
    first_perpendicular = np.array([2.0, -1.0, 0.0])
    first_perpendicular -= np.dot(first_perpendicular, direction) * direction
    first_perpendicular *= 1.5 / np.linalg.norm(first_perpendicular)
    second_perpendicular = cos(np.pi / 3.0) * first_perpendicular + sin(np.pi / 3.0) * np.cross(
        direction, first_perpendicular
    )
    reciprocal_basis = np.column_stack(
        (
            first_perpendicular + 1.0e4 * direction,
            second_perpendicular - 3.0e3 * direction,
            0.3 * direction,
        )
    )

    class ExactStrength:
        def evaluate(self, *, rod: Rod, L: float, k_norm_Ainv: float) -> float:
            return 1.0 + 0.03 * rod.h + 0.05 * rod.k + 1.0e-10 * L * L + 0.02 * k_norm_Ainv

    config = PainterConfig(
        reciprocal_basis_Ainv=reciprocal_basis,
        crystal_to_sample=np.eye(3),
        rods=(Rod(1, 0), Rod(0, 1)),
        mosaic=MosaicParameters(
            gaussian_sigma_rad=np.deg2rad(1.0),
            lorentzian_half_width_rad=np.deg2rad(0.5),
            lorentzian_probability=0.2,
            alpha_panel_count=8,
            alpha_gauss_order=8,
            azimuth_count=8,
        ),
        raster=RasterParameters(8, 16),
    )
    incident = np.array([0.2, 4.062900581047559, -0.3545543022596421])
    strength = ExactStrength()
    painter = EwaldSpherePainter(config, strength)
    scalar = painter.paint(incident)
    coating = painter.paint_coating(incident)

    assert coating.retained_root_count == len(scalar.points)
    assert coating.measure is config.measure
    assert coating.family_summaries[0].family_m == 1
    assert coating.family_summaries[0].is_complete_hexagonal_family is False
    np.testing.assert_allclose(coating.texture.mass, scalar.texture.mass, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(
        [
            coating.ledger.painted_weight,
            coating.ledger.no_root_base_mass,
            coating.ledger.tangent_base_mass,
        ],
        [
            scalar.ledger.painted_weight,
            scalar.ledger.no_root_base_mass,
            scalar.ledger.tangent_base_mass,
        ],
        rtol=0.0,
        atol=3.0e-14,
    )
    for point in scalar.points:
        expected = strength.evaluate(
            rod=Rod(point.rod_h, point.rod_k),
            L=point.L,
            k_norm_Ainv=np.linalg.norm(incident),
        )
        assert point.rod_strength == pytest.approx(expected, rel=0.0, abs=2.0e-15)
    scalar_groups = {
        key: (
            sum(
                point.rod_h == key[0] and point.rod_k == key[1] and point.branch == key[2]
                for point in scalar.points
            ),
            fsum(
                point.weight
                for point in scalar.points
                if (point.rod_h, point.rod_k, point.branch) == key
            ),
        )
        for key in ((1, 0, 1), (1, 0, 2), (0, 1, 1), (0, 1, 2))
    }
    for rod_summary in coating.rod_summaries:
        for branch in rod_summary.branches:
            count, weight = scalar_groups[(rod_summary.rod.h, rod_summary.rod.k, branch.branch)]
            assert branch.retained_root_count == count
            assert branch.painted_weight == pytest.approx(weight, rel=0.0, abs=3.0e-14)

    full_m1_shell = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
    complete_skew_coating = EwaldSpherePainter(
        replace(config, rods=tuple(Rod(h, k) for h, k in full_m1_shell)),
        strength,
    ).paint_coating(incident)
    assert complete_skew_coating.family_summaries[0].is_complete_hexagonal_family is True
    higher_shear_basis = np.column_stack(
        (
            first_perpendicular + 1.0e5 * direction,
            second_perpendicular - 3.0e4 * direction,
            0.3 * direction,
        )
    )
    higher_shear_coating = EwaldSpherePainter(
        replace(
            config,
            reciprocal_basis_Ainv=higher_shear_basis,
            rods=tuple(Rod(h, k) for h, k in full_m1_shell),
        ),
        strength,
    ).paint_coating(incident)
    assert higher_shear_coating.family_summaries[0].is_complete_hexagonal_family is True


def test_painted_ewald_bi2se3_catalog_includes_every_family_that_fits() -> None:
    from painted_ewald import (
        EwaldSpherePainter,
        PainterConfig,
        RasterParameters,
        enumerate_rods_within_ewald_sphere,
    )

    k_norm_Ainv = 4.078341560576728
    rods = enumerate_rods_within_ewald_sphere(
        reciprocal_basis_Ainv=BI2SE3_RECIPROCAL_BASIS_AINV,
        k_norm_Ainv=k_norm_Ainv,
    )

    c_hat = BI2SE3_RECIPROCAL_BASIS_AINV[:, 2]
    c_hat /= np.linalg.norm(c_hat)
    tolerance = 256.0 * np.finfo(np.float64).eps * max(2.0 * k_norm_Ainv, 1.0)
    expected: set[tuple[int, int]] = set()
    for h in range(-10, 11):
        for k in range(-10, 11):
            q_parallel = (
                h * BI2SE3_RECIPROCAL_BASIS_AINV[:, 0] + k * BI2SE3_RECIPROCAL_BASIS_AINV[:, 1]
            )
            line_distance = np.linalg.norm(q_parallel - np.dot(q_parallel, c_hat) * c_hat)
            if line_distance <= 2.0 * k_norm_Ainv + tolerance:
                expected.add((h, k))
    assert all(abs(h) < 10 and abs(k) < 10 for h, k in expected)
    assert {(rod.h, rod.k) for rod in rods} == expected
    assert len(rods) == 85
    assert sorted({rod.family_m for rod in rods}) == [0, 1, 3, 4, 7, 9, 12, 13, 16, 19, 21]
    assert [(rod.family_m, rod.h, rod.k) for rod in rods] == sorted(
        (rod.family_m, rod.h, rod.k) for rod in rods
    )

    config = PainterConfig(
        reciprocal_basis_Ainv=BI2SE3_RECIPROCAL_BASIS_AINV,
        crystal_to_sample=np.eye(3),
        rods=rods,
        mosaic=MosaicParameters(0.0, 0.0, 0.1, azimuth_count=1),
        raster=RasterParameters(2, 4),
    )
    coating = EwaldSpherePainter(config).paint_coating(np.array([0.0, k_norm_Ainv, 0.0]))
    assert all(family.is_complete_hexagonal_family for family in coating.family_summaries)


def test_legacy_ewald_adapter_preserves_authoritative_status_and_payload() -> None:
    from painted_ewald.ewald import RootStatus, solve_infinite_rod_ewald

    direction = np.array([0.0, 0.0, 1.0])
    cases = (
        (np.array([0.0, 0.0, 4.0]), np.array([1.0, 0.0, 0.0]), False),
        (np.array([0.0, 0.0, 4.0]), np.array([4.0, 0.0, 0.0]), False),
        (np.array([0.0, 0.0, 4.0]), np.array([4.1, 0.0, 0.0]), False),
        (np.array([0.0, 0.0, 4.0]), np.zeros(3), True),
        (np.array([4.0, 0.0, 0.0]), np.zeros(3), True),
    )
    status_map = {
        RootStatus.REGULAR: EwaldRootStatus.TWO_ROOT,
        RootStatus.TANGENT: EwaldRootStatus.TANGENT,
        RootStatus.NO_ROOT: EwaldRootStatus.NO_ROOT,
        RootStatus.COLLAPSED_DIRECT: EwaldRootStatus.TANGENT,
    }
    for incident, q0, rod_is_m0 in cases:
        authority = solve_infinite_rod_ewald(
            ki_sample_Ainv=incident,
            q0_sample_Ainv=q0,
            d_hat_sample=direction,
            b3_norm_Ainv=2.0,
            rod_is_m0=rod_is_m0,
            root_tolerance_rel=0.0,
            residual_tolerance_rel=64.0 * np.finfo(np.float64).eps,
        )
        adapter = solve_continuous_rod_ewald(
            ki_sample_Ainv=incident,
            q0_sample_Ainv=q0,
            d_hat_sample=direction,
            b3_norm_Ainv=2.0,
        )
        assert adapter.status is status_map[authority.status]
        assert adapter.direct_beam_root_count == authority.direct_root_count
        assert len(adapter.emittable_roots) == len(authority.emittable_roots)
        for legacy_root, root in zip(
            adapter.emittable_roots, authority.emittable_roots, strict=True
        ):
            assert legacy_root.u_Ainv == root.u_Ainv
            assert legacy_root.l_coordinate == root.L
            np.testing.assert_array_equal(legacy_root.q_sample_Ainv, root.q_sample_Ainv)
            np.testing.assert_array_equal(legacy_root.kf_sample_Ainv, root.kf_sample_Ainv)
            assert legacy_root.ewald_residual_Ainv == root.ewald_residual_Ainv
            assert legacy_root.coarea_jacobian == root.coarea_jacobian

    with pytest.raises(ValueError, match="real"):
        solve_continuous_rod_ewald(
            ki_sample_Ainv=cases[0][0].astype(np.complex128),
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
