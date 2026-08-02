from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr

from painted_ewald import (
    BraggSpaceConfig,
    ContinuousEwaldCoating,
    EwaldLatentGeometry,
    MosaicBraggSpace,
    MosaicParameters,
    Rod,
    build_mosaic_space,
    enumerate_rods_within_ewald_sphere,
    wrapped_mosaic_line_density_rad_inv,
)
from painted_ewald.ewald import RootStatus, solve_infinite_rod_ewald
from rasim_next.sampling.source import sample_gaussian_source_rays

BI2SE3_RECIPROCAL_BASIS_AINV = np.array(
    [
        [1.516578640400576, 0.0, 0.0],
        [0.8755970862825088, 1.7511941725650182, 0.0],
        [0.0, 0.0, 0.2194156064806393],
    ]
)


def test_latent_geometry_accepts_numpy_string_root_statuses() -> None:
    status = np.asarray(
        [RootStatus.REGULAR.value, RootStatus.NO_ROOT.value],
        dtype="U32",
    )
    geometry = EwaldLatentGeometry(
        rod=Rod(1, 0),
        branch=0,
        alpha_rad=np.array([0.1, 0.2]),
        beta_rad=np.array([0.3, 0.4]),
        u_Ainv=np.zeros(2),
        L=np.zeros(2),
        q_sample_Ainv=np.zeros((2, 3)),
        kf_sample_Ainv=np.zeros((2, 3)),
        ewald_residual_Ainv=np.zeros(2),
        status=status,
    )

    np.testing.assert_array_equal(geometry.status, status)
    np.testing.assert_array_equal(geometry.valid, [True, False])


class _UnequalSameFamilyStrength:
    reciprocal_basis_Ainv = BI2SE3_RECIPROCAL_BASIS_AINV

    def evaluate(self, *, rod: Rod, L: float, k_norm_Ainv: float) -> float:
        del k_norm_Ainv
        return float((2 if (rod.h, rod.k) == (1, 0) else 5) * (1.0 + 0.1 * L))

    def evaluate_profile(self, *, rod: Rod, L: np.ndarray, k_norm_Ainv: float) -> np.ndarray:
        del k_norm_Ainv
        multiplier = 2 if (rod.h, rod.k) == (1, 0) else 5
        return multiplier * (1.0 + 0.1 * np.asarray(L))


class _WavenumberTaggedStrength(_UnequalSameFamilyStrength):
    def evaluate_profile(self, *, rod: Rod, L: np.ndarray, k_norm_Ainv: float) -> np.ndarray:
        return np.full(np.shape(L), k_norm_Ainv + 0.01 * rod.h + 0.02 * rod.k)


def _mosaic() -> MosaicParameters:
    return MosaicParameters(
        gaussian_sigma_rad=np.deg2rad(5.0),
        lorentzian_half_width_rad=np.deg2rad(2.0),
        lorentzian_probability=0.1,
        alpha_panel_count=12,
        alpha_gauss_order=6,
        azimuth_count=32,
    )


def test_continuous_bragg_field_weights_physical_rods_before_family_sum() -> None:
    rods = (Rod(1, 0), Rod(0, 1))
    k_norm_Ainv = 2.0 * np.pi / 1.540592925
    space = MosaicBraggSpace(
        BraggSpaceConfig(
            reciprocal_basis_Ainv=BI2SE3_RECIPROCAL_BASIS_AINV,
            crystal_to_sample=np.eye(3),
            rods=rods,
            mosaic=_mosaic(),
            k_norm_Ainv=k_norm_Ainv,
        ),
        _UnequalSameFamilyStrength(),
    )
    ell = 0.25
    family = space.weighted_family_slice(family_m=1, L=ell)
    expected = (2.0 + 5.0) * (1.0 + 0.1 * ell)
    assert tuple(item.rod for item in family.rod_slices) == rods
    assert family.total_intensity_weight_A2 == pytest.approx(expected, abs=2.0e-14)
    assert family.total_intensity_weight_A2 == pytest.approx(
        sum(float(np.sum(item.intensity_weight_A2)) for item in family.rod_slices),
        abs=2.0e-14,
    )

    alpha = 0.123456789
    beta = 1.23456789
    u_Ainv = ell * np.linalg.norm(BI2SE3_RECIPROCAL_BASIS_AINV[:, 2])
    latent = space.evaluate_latent(
        rod=rods[0],
        alpha_rad=alpha,
        beta_rad=beta,
        u_Ainv=u_Ainv,
    )
    expected_density = (
        2.0 * float(wrapped_mosaic_line_density_rad_inv(alpha, space.config.mosaic)) / (2.0 * np.pi)
    )
    assert latent.mosaic_probability_density_rad2_inv == pytest.approx(
        expected_density, abs=2.0e-15
    )
    assert latent.intensity_density_A2_rad2_inv == pytest.approx(
        expected_density * 2.0 * (1.0 + 0.1 * ell), abs=2.0e-14
    )

    alpha_node, alpha_weight = leggauss(512)
    alpha_node = 0.5 * np.pi * (alpha_node + 1.0)
    alpha_weight *= 0.5 * np.pi
    integrated = space.evaluate_latent(
        rod=rods[0], alpha_rad=alpha_node, beta_rad=0.731, u_Ainv=u_Ainv
    )
    total = float(np.sum(integrated.intensity_density_A2_rad2_inv * alpha_weight) * 2.0 * np.pi)
    assert total == pytest.approx(2.0 * (1.0 + 0.1 * ell), rel=3.0e-11)

    with pytest.raises(ValueError, match="reciprocal bases do not match"):
        MosaicBraggSpace(
            replace(
                space.config,
                reciprocal_basis_Ainv=1.001 * BI2SE3_RECIPROCAL_BASIS_AINV,
            ),
            _UnequalSameFamilyStrength(),
        )


def test_continuous_ewald_coating_matches_scalar_root_and_latent_oracles() -> None:
    rods = tuple(Rod(h, k) for h, k in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)))
    k_norm_Ainv = 2.0 * np.pi / 1.540592925
    ki_sample_Ainv = np.array([0.0, 4.062900581047559, -0.3545543022596421])
    space = MosaicBraggSpace(
        BraggSpaceConfig(
            reciprocal_basis_Ainv=BI2SE3_RECIPROCAL_BASIS_AINV,
            crystal_to_sample=np.eye(3),
            rods=rods,
            mosaic=_mosaic(),
            k_norm_Ainv=k_norm_Ainv,
        ),
        _WavenumberTaggedStrength(),
    )
    coating = ContinuousEwaldCoating(space, ki_sample_Ainv=ki_sample_Ainv)
    alpha = np.deg2rad(np.array([1.0, 3.0, 8.0, 15.0]))
    beta = np.array([0.1, 0.2, 0.3, 0.4])
    rod = rods[0]
    evaluated = coating.evaluate_latent(rod=rod, branch=2, alpha_rad=alpha, beta_rad=beta)

    assert np.all(evaluated.geometry.valid)
    for index in range(alpha.size):
        q0 = space.map_latent(rod=rod, alpha_rad=alpha[index], beta_rad=beta[index], u_Ainv=0.0)
        q1 = space.map_latent(rod=rod, alpha_rad=alpha[index], beta_rad=beta[index], u_Ainv=1.0)
        root = solve_infinite_rod_ewald(
            ki_sample_Ainv=ki_sample_Ainv,
            q0_sample_Ainv=q0,
            d_hat_sample=q1 - q0,
            b3_norm_Ainv=np.linalg.norm(BI2SE3_RECIPROCAL_BASIS_AINV[:, 2]),
            rod_is_m0=False,
            root_tolerance_rel=coating.root_tolerance_rel,
            residual_tolerance_rel=coating.residual_tolerance_rel,
        ).emittable_roots[1]
        latent = space.evaluate_latent(
            rod=rod,
            alpha_rad=alpha[index],
            beta_rad=beta[index],
            u_Ainv=root.u_Ainv,
        )
        np.testing.assert_allclose(
            evaluated.geometry.q_sample_Ainv[index], root.q_sample_Ainv, atol=2.0e-15
        )
        np.testing.assert_allclose(
            evaluated.geometry.kf_sample_Ainv[index], root.kf_sample_Ainv, atol=2.0e-15
        )
        assert evaluated.latent_intensity_density_A2_rad2_inv[index] == pytest.approx(
            latent.intensity_density_A2_rad2_inv, abs=2.0e-14
        )
        assert evaluated.coating_intensity_density_A2_rad2_inv[index] == pytest.approx(
            latent.intensity_density_A2_rad2_inv * root.coarea_jacobian,
            abs=2.0e-14,
        )
        assert np.linalg.norm(root.q_sample_Ainv + ki_sample_Ainv) == pytest.approx(
            np.linalg.norm(ki_sample_Ainv), abs=3.0e-14
        )

    with pytest.raises(ValueError, match="zero-tilt Dirac"):
        ContinuousEwaldCoating(
            MosaicBraggSpace(
                replace(space.config, mosaic=replace(space.config.mosaic, gaussian_sigma_rad=0.0)),
                _WavenumberTaggedStrength(),
            ),
            ki_sample_Ainv=ki_sample_Ainv,
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
    np.testing.assert_array_equal(sampled.origin_lab_m, repeated.origin_lab_m)
    assert not np.array_equal(sampled.origin_lab_m, changed.origin_lab_m)
    np.testing.assert_array_equal(sampled.source_weight, np.full(count, 1.0 / count))

    paired_stop = 2 * (count // 2)
    np.testing.assert_allclose(
        sampled.origin_lab_m[:paired_stop:2] + sampled.origin_lab_m[1:paired_stop:2],
        np.broadcast_to(2.0 * mean_origin, (count // 2, 3)),
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        sampled.wavelength_A[:paired_stop:2] + sampled.wavelength_A[1:paired_stop:2],
        np.full(count // 2, 2.48),
        atol=5.0e-16,
    )

    spatial = ((sampled.origin_lab_m - mean_origin) @ axes.T) / spatial_sigma
    cosine = np.clip(sampled.direction_lab @ mean_direction, -1.0, 1.0)
    radius = np.arccos(cosine)
    inverse_sine = np.divide(radius, np.sin(radius), out=np.ones_like(radius), where=radius != 0.0)
    angular = sampled.direction_lab @ axes.T * inverse_sine[:, None] / divergence_sigma
    wavelength = (sampled.wavelength_A - 1.24) / 0.01
    unit = ndtr(np.column_stack((spatial, angular, wavelength)))
    strata = np.floor(unit * count).astype(np.int64)
    np.testing.assert_array_equal(
        np.sort(strata, axis=0), np.broadcast_to(np.arange(count)[:, None], (count, 5))
    )
    np.testing.assert_allclose(
        unit[:paired_stop:2] + unit[1:paired_stop:2], np.ones((count // 2, 5)), atol=2.0e-15
    )


def test_wrapped_mosaic_probability_is_normalized_without_signed_tilt_duplication() -> None:
    parameters = _mosaic()
    node, weight = leggauss(1024)
    theta = np.pi * node
    signed_mass = float(
        np.sum(np.pi * weight * wrapped_mosaic_line_density_rad_inv(theta, parameters))
    )
    assert signed_mass == pytest.approx(1.0, abs=5.0e-11)
    for harmonic in (1, 2):
        expected = 0.9 * np.exp(-0.5 * (harmonic * parameters.gaussian_sigma_rad) ** 2)
        expected += 0.1 * np.exp(-harmonic * parameters.lorentzian_half_width_rad)
        observed = float(
            np.sum(
                np.pi
                * weight
                * wrapped_mosaic_line_density_rad_inv(theta, parameters)
                * np.cos(harmonic * theta)
            )
        )
        assert observed == pytest.approx(expected, abs=5.0e-11)

    space = build_mosaic_space(
        reciprocal_basis_Ainv=BI2SE3_RECIPROCAL_BASIS_AINV,
        crystal_to_sample=np.eye(3),
        parameters=parameters,
    )
    assert np.sum(space.probability_mass) == pytest.approx(1.0, abs=1.0e-12)
    assert np.all(space.alpha_rad >= 0.0)
    for harmonic in (1, 2, 3):
        beta_moment = np.sum(space.probability_mass * np.exp(1j * harmonic * space.beta_rad))
        assert abs(beta_moment) <= 3.0e-15

    narrow = replace(
        parameters,
        gaussian_sigma_rad=0.0,
        lorentzian_half_width_rad=1.0e-7,
        lorentzian_probability=1.0,
        alpha_panel_count=8,
        alpha_gauss_order=12,
    )
    narrow_space = build_mosaic_space(
        reciprocal_basis_Ainv=np.eye(3),
        crystal_to_sample=np.eye(3),
        parameters=narrow,
    )
    assert np.sum(narrow_space.probability_mass) == pytest.approx(1.0, abs=1.0e-12)
    assert np.sum(narrow_space.probability_mass * np.cos(narrow_space.alpha_rad)) == pytest.approx(
        np.exp(-1.0e-7), abs=2.0e-13
    )


def test_mosaic_slice_conserves_mass_and_rigid_rotation() -> None:
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
        parameters=_mosaic(),
    )
    c_hat = BI2SE3_RECIPROCAL_BASIS_AINV[:, 2]
    c_hat /= np.linalg.norm(c_hat)
    for rod, u_Ainv in ((Rod(0, 0), 0.6), (Rod(1, 0), -0.4)):
        result = space.mosaic_slice(rod=rod, u_Ainv=u_Ainv)
        q_parallel = (
            rod.h * BI2SE3_RECIPROCAL_BASIS_AINV[:, 0] + rod.k * BI2SE3_RECIPROCAL_BASIS_AINV[:, 1]
        )
        q_crystal = q_parallel + u_Ainv * c_hat
        expected = np.einsum(
            "ij,njk,k->ni", crystal_to_sample, space.rotation_crystal, q_crystal, optimize=True
        )
        assert np.sum(result.probability_mass) == pytest.approx(1.0, abs=1.0e-12)
        np.testing.assert_allclose(result.q_sample_Ainv, expected, atol=2.0e-15)
        np.testing.assert_allclose(
            np.linalg.norm(result.q_sample_Ainv, axis=1), np.linalg.norm(q_crystal), atol=2.0e-15
        )


def test_analytic_ewald_roots_cover_regular_tangent_direct_and_no_root_cases() -> None:
    arguments = {
        "ki_sample_Ainv": np.array([0.0, 0.0, -10.0]),
        "d_hat_sample": np.array([0.0, 0.0, 1.0]),
        "b3_norm_Ainv": 0.2,
        "root_tolerance_rel": 256.0 * np.finfo(np.float64).eps,
        "residual_tolerance_rel": 512.0 * np.finfo(np.float64).eps,
    }
    direct = solve_infinite_rod_ewald(**arguments, q0_sample_Ainv=np.zeros(3), rod_is_m0=True)
    assert direct.status is RootStatus.REGULAR
    assert direct.direct_root_count == 1
    assert direct.emittable_roots[0].branch == 0
    assert direct.emittable_roots[0].u_Ainv == pytest.approx(20.0)

    regular = solve_infinite_rod_ewald(
        **arguments, q0_sample_Ainv=np.array([1.0, 0.0, 0.0]), rod_is_m0=False
    )
    expected_u = np.array([10.0 - np.sqrt(99.0), 10.0 + np.sqrt(99.0)])
    assert regular.status is RootStatus.REGULAR
    np.testing.assert_allclose([root.u_Ainv for root in regular.emittable_roots], expected_u)
    assert [root.branch for root in regular.emittable_roots] == [1, 2]
    for root in regular.emittable_roots:
        assert root.coarea_jacobian == pytest.approx(10.0 / np.sqrt(99.0))
        assert np.linalg.norm(root.kf_sample_Ainv) == pytest.approx(10.0, abs=2.0e-14)

    shifted = solve_infinite_rod_ewald(
        **arguments,
        q0_sample_Ainv=np.array([1.0, 0.0, float(2**26)]),
        rod_is_m0=False,
    )
    for original, translated in zip(regular.emittable_roots, shifted.emittable_roots, strict=True):
        np.testing.assert_array_equal(translated.q_sample_Ainv, original.q_sample_Ainv)
        np.testing.assert_array_equal(translated.kf_sample_Ainv, original.kf_sample_Ainv)

    tangent = solve_infinite_rod_ewald(
        **arguments, q0_sample_Ainv=np.array([10.0, 0.0, 0.0]), rod_is_m0=False
    )
    no_root = solve_infinite_rod_ewald(
        **arguments, q0_sample_Ainv=np.array([10.1, 0.0, 0.0]), rod_is_m0=False
    )
    collapsed = solve_infinite_rod_ewald(
        **{**arguments, "ki_sample_Ainv": np.array([10.0, 0.0, 0.0])},
        q0_sample_Ainv=np.zeros(3),
        rod_is_m0=True,
    )
    assert tangent.status is RootStatus.TANGENT and tangent.emittable_roots == ()
    assert no_root.status is RootStatus.NO_ROOT and no_root.emittable_roots == ()
    assert collapsed.status is RootStatus.COLLAPSED_DIRECT and collapsed.emittable_roots == ()


def test_bi2se3_catalog_contains_exactly_every_rod_that_can_reach_the_sphere() -> None:
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
    assert {(rod.h, rod.k) for rod in rods} == expected
    assert len(rods) == 85
    assert sorted({rod.family_m for rod in rods}) == [0, 1, 3, 4, 7, 9, 12, 13, 16, 19, 21]
    assert [(rod.family_m, rod.h, rod.k) for rod in rods] == sorted(
        (rod.family_m, rod.h, rod.k) for rod in rods
    )


def test_mosaic_public_inputs_reject_invalid_probability_and_geometry() -> None:
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
