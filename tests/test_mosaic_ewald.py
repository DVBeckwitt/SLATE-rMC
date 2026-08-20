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
from rasim_next.sampling.source import (
    NOMINAL_MEAN_GEOMETRY_REFERENCE_MODEL_ID,
    sample_discrete_gaussian_line_source_rays,
    sample_gaussian_source_rays,
    sample_nominal_mean_geometry_source_ray,
)

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
    explicit_zero = sample_gaussian_source_rays(
        **arguments,
        seed=1729,
        position_divergence_correlation=(0.0, 0.0),
    )
    changed = sample_gaussian_source_rays(**arguments, seed=2718)
    np.testing.assert_array_equal(sampled.origin_lab_m, repeated.origin_lab_m)
    np.testing.assert_array_equal(sampled.origin_lab_m, explicit_zero.origin_lab_m)
    np.testing.assert_array_equal(sampled.direction_lab, explicit_zero.direction_lab)
    np.testing.assert_array_equal(sampled.wavelength_A, explicit_zero.wavelength_A)
    np.testing.assert_array_equal(sampled.source_weight, explicit_zero.source_weight)
    assert sampled.source_parameter_provenance == explicit_zero.source_parameter_provenance
    assert sampled.source_revision == explicit_zero.source_revision
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
    if count == 7:
        assert (
            sampled.source_revision
            == "6f35c8f4e825542df9a4bda148e62a825c54455d0dbfc2c2d5ba3523ebd270de"
        )


@pytest.mark.parametrize(
    ("sample_count", "seed"),
    ((8, 1729), (8, 6085), (9, 7390)),
)
def test_source_position_angle_correlation_is_the_declared_gaussian_transform(
    sample_count: int,
    seed: int,
) -> None:
    mean_origin = np.asarray((0.0, -0.02, 0.0))
    mean_direction = np.asarray((0.0, 1.0, 0.0))
    axes = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    spatial_sigma = np.asarray((2.0e-5, 3.0e-5))
    divergence_sigma = np.asarray((4.0e-4, 5.0e-4))
    common = {
        "mean_origin_lab_m": mean_origin,
        "mean_direction_lab": mean_direction,
        "transverse_axes_lab": axes,
        "spatial_sigma_m": spatial_sigma,
        "divergence_sigma_rad": divergence_sigma,
        "mean_wavelength_A": 1.54,
        "wavelength_sigma_A": 0.001,
        "sample_count": sample_count,
        "seed": seed,
        "polarization_state_id": "THOMSON_UNPOLARIZED_UNANALYSED",
    }
    correlation = np.asarray((0.25, -0.4))
    correlated = sample_gaussian_source_rays(
        **common,
        position_divergence_correlation=correlation,
    )

    position_standard = ((correlated.origin_lab_m - mean_origin) @ axes.T) / spatial_sigma

    def standardized_direction(direction: np.ndarray) -> np.ndarray:
        cosine = np.clip(direction @ mean_direction, -1.0, 1.0)
        radius = np.arccos(cosine)
        scale = np.divide(radius, np.sin(radius), out=np.ones_like(radius), where=radius != 0.0)
        return ((direction @ axes.T) * scale[:, None]) / divergence_sigma

    correlated_direction = standardized_direction(correlated.direction_lab)
    weight = correlated.source_weight
    np.testing.assert_allclose(weight @ position_standard, np.zeros(2), atol=2.0e-16)
    np.testing.assert_allclose(weight @ correlated_direction, np.zeros(2), atol=2.0e-13)
    standardized = np.column_stack((position_standard, correlated_direction))
    covariance = standardized.T @ (weight[:, None] * standardized)
    expected_covariance = np.eye(4)
    expected_covariance[0, 2] = expected_covariance[2, 0] = correlation[0]
    expected_covariance[1, 3] = expected_covariance[3, 1] = correlation[1]
    np.testing.assert_allclose(covariance, expected_covariance, rtol=0.0, atol=3.0e-12)
    with pytest.raises(ValueError, match="strictly within"):
        sample_gaussian_source_rays(
            **common,
            position_divergence_correlation=(0.0, 1.0),
        )


def test_discrete_source_lines_retain_exact_probability_mass() -> None:
    lines_A = np.asarray((1.540592925, 1.544427))
    probabilities = np.asarray((1.0, 0.518), dtype=np.float64)
    probabilities /= probabilities.sum()
    sampled = sample_discrete_gaussian_line_source_rays(
        mean_origin_lab_m=np.asarray((0.0, -0.02, 0.0)),
        mean_direction_lab=np.asarray((0.0, 1.0, 0.0)),
        transverse_axes_lab=np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        spatial_sigma_m=np.asarray((2.0e-5, 3.0e-5)),
        divergence_sigma_rad=np.asarray((4.0e-4, 5.0e-4)),
        line_wavelength_A=lines_A,
        line_probability=probabilities,
        common_wavelength_sigma_A=0.0,
        sample_count=7,
        seed=1729,
        polarization_state_id="THOMSON_UNPOLARIZED_UNANALYSED",
    )

    assert set(sampled.wavelength_A.tolist()) == set(lines_A.tolist())
    assert np.sum(sampled.source_weight) == pytest.approx(1.0, rel=0.0, abs=2.0e-16)
    for line, probability in zip(lines_A, probabilities, strict=True):
        assert np.sum(sampled.source_weight[sampled.wavelength_A == line]) == pytest.approx(
            probability,
            rel=0.0,
            abs=2.0e-16,
        )
    assert np.sum(sampled.source_weight * sampled.wavelength_A) == pytest.approx(
        float(probabilities @ lines_A),
        rel=0.0,
        abs=5.0e-16,
    )
    assert "discrete_gaussian_lines.v1" in sampled.source_parameter_provenance
    assert sampled.source_sampling_model_id == "finite_unequal_line_geometry_quadrature.v1"

    with pytest.raises(ValueError, match="at least the number of source lines"):
        sample_discrete_gaussian_line_source_rays(
            mean_origin_lab_m=np.asarray((0.0, -0.02, 0.0)),
            mean_direction_lab=np.asarray((0.0, 1.0, 0.0)),
            transverse_axes_lab=np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
            spatial_sigma_m=np.asarray((2.0e-5, 3.0e-5)),
            divergence_sigma_rad=np.asarray((4.0e-4, 5.0e-4)),
            line_wavelength_A=lines_A,
            line_probability=probabilities,
            common_wavelength_sigma_A=0.0,
            sample_count=1,
            seed=1729,
            polarization_state_id="THOMSON_UNPOLARIZED_UNANALYSED",
        )

    reference = sample_nominal_mean_geometry_source_ray(
        mean_origin_lab_m=np.asarray((0.0, -0.02, 0.0)),
        mean_direction_lab=np.asarray((0.0, 1.0, 0.0)),
        transverse_axes_lab=np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        spatial_sigma_m=np.asarray((2.0e-5, 3.0e-5)),
        divergence_sigma_rad=np.asarray((4.0e-4, 5.0e-4)),
        reference_wavelength_A=1.5418,
        polarization_state_id="THOMSON_UNPOLARIZED_UNANALYSED",
    )
    np.testing.assert_array_equal(reference.wavelength_A, np.asarray((1.5418,)))
    np.testing.assert_array_equal(reference.source_weight, np.ones(1))
    assert reference.source_sampling_model_id == NOMINAL_MEAN_GEOMETRY_REFERENCE_MODEL_ID
    assert reference.source_rng_model_id == "no_rng.v1"
    assert reference.source_seed == 0
    assert "nominal_geometry_only_no_intensity" in reference.source_parameter_provenance

    finite_small_count = sample_discrete_gaussian_line_source_rays(
        mean_origin_lab_m=np.asarray((0.0, -0.02, 0.0)),
        mean_direction_lab=np.asarray((0.0, 1.0, 0.0)),
        transverse_axes_lab=np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        spatial_sigma_m=np.asarray((2.0e-5, 3.0e-5)),
        divergence_sigma_rad=np.asarray((4.0e-4, 5.0e-4)),
        line_wavelength_A=lines_A,
        line_probability=probabilities,
        common_wavelength_sigma_A=0.0,
        sample_count=6,
        seed=1729,
        polarization_state_id="THOMSON_UNPOLARIZED_UNANALYSED",
        position_divergence_correlation=(0.25, -0.4),
    )
    assert (
        finite_small_count.source_sampling_model_id
        == "gaussian_spatial_angular_stratified_lines_correlated_transform.v1"
    )
    assert np.all(np.isfinite(finite_small_count.origin_lab_m))
    assert np.all(np.isfinite(finite_small_count.direction_lab))

    invalid_common = {
        "mean_origin_lab_m": np.asarray((0.0, -0.02, 0.0)),
        "mean_direction_lab": np.asarray((0.0, 1.0, 0.0)),
        "transverse_axes_lab": np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        "spatial_sigma_m": np.asarray((2.0e-5, 3.0e-5)),
        "divergence_sigma_rad": np.asarray((4.0e-4, 5.0e-4)),
        "common_wavelength_sigma_A": 0.0,
        "sample_count": 4,
        "seed": 1729,
        "polarization_state_id": "THOMSON_UNPOLARIZED_UNANALYSED",
    }
    for invalid_lines, invalid_probability in (
        (np.asarray((1.54 + 0.1j, 1.55)), np.asarray((0.5, 0.5))),
        ((True, 2.0), (0.5, 0.5)),
        (np.asarray((True, 2.0), dtype=object), np.asarray((0.5, 0.5))),
        (np.asarray((1.54, 1.55)), np.asarray((0.5 + 0.1j, 0.5))),
        ((1.54, 1.55), (True, 0.0)),
        (np.asarray((1.54, 1.55)), np.asarray((True, False))),
        (np.asarray(("1.54", "1.55")), np.asarray((0.5, 0.5))),
        (np.asarray((1.54, 1.55)), np.asarray(("0.5", "0.5"))),
    ):
        with pytest.raises(ValueError, match="real numbers"):
            sample_discrete_gaussian_line_source_rays(
                **invalid_common,
                line_wavelength_A=invalid_lines,
                line_probability=invalid_probability,
            )
    with pytest.raises(ValueError, match="must be real"):
        sample_discrete_gaussian_line_source_rays(
            **{**invalid_common, "common_wavelength_sigma_A": np.bool_(False)},
            line_wavelength_A=np.asarray((1.54, 1.55)),
            line_probability=np.asarray((0.5, 0.5)),
        )
    with pytest.raises(ValueError, match="must be real"):
        sample_discrete_gaussian_line_source_rays(
            **invalid_common,
            line_wavelength_A=np.asarray((1.54, 1.55)),
            line_probability=np.asarray((0.5, 0.5)),
            position_divergence_correlation=(np.bool_(False), 0.0),
        )
    tiny = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="too small"):
        sample_discrete_gaussian_line_source_rays(
            **invalid_common,
            line_wavelength_A=np.asarray((1.54, 1.55)),
            line_probability=np.asarray((1.0, tiny)),
        )


def test_configured_count_discrete_source_matches_moments_without_line_geometry_coupling() -> None:
    lines_A = np.asarray((1.540592925, 1.544427))
    probabilities = np.asarray((0.6587615283267457, 0.3412384716732543))
    mean_origin = np.asarray((0.0, -0.02, 0.0))
    mean_direction = np.asarray((0.0, 1.0, 0.0))
    axes = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    spatial_sigma = np.asarray((1.4e-4, 1.2e-4))
    divergence_sigma = np.asarray((9.8e-4, 7.6e-4))
    correlation = np.asarray((-0.15, 0.39))
    sampled = sample_discrete_gaussian_line_source_rays(
        mean_origin_lab_m=mean_origin,
        mean_direction_lab=mean_direction,
        transverse_axes_lab=axes,
        spatial_sigma_m=spatial_sigma,
        divergence_sigma_rad=divergence_sigma,
        line_wavelength_A=lines_A,
        line_probability=probabilities,
        common_wavelength_sigma_A=0.0,
        sample_count=250,
        seed=20260809,
        polarization_state_id="THOMSON_UNPOLARIZED_UNANALYSED",
        position_divergence_correlation=correlation,
    )

    def standardized_direction(direction: np.ndarray) -> np.ndarray:
        cosine = np.clip(direction @ mean_direction, -1.0, 1.0)
        radius = np.arccos(cosine)
        scale = np.divide(radius, np.sin(radius), out=np.ones_like(radius), where=radius != 0.0)
        return ((direction @ axes.T) * scale[:, None]) / divergence_sigma

    position = ((sampled.origin_lab_m - mean_origin) @ axes.T) / spatial_sigma
    direction = standardized_direction(sampled.direction_lab)
    for line, probability in zip(lines_A, probabilities, strict=True):
        selected = sampled.wavelength_A == line
        line_weight = sampled.source_weight[selected]
        normalized = line_weight / np.sum(line_weight)
        assert np.sum(line_weight) == pytest.approx(probability, rel=0.0, abs=5.0e-16)
        np.testing.assert_allclose(normalized @ position[selected], np.zeros(2), atol=3.0e-16)
        np.testing.assert_allclose(normalized @ direction[selected], np.zeros(2), atol=2.0e-13)
        standardized = np.column_stack((position[selected], direction[selected]))
        covariance = standardized.T @ (normalized[:, None] * standardized)
        expected_covariance = np.eye(4)
        expected_covariance[0, 2] = expected_covariance[2, 0] = correlation[0]
        expected_covariance[1, 3] = expected_covariance[3, 1] = correlation[1]
        np.testing.assert_allclose(covariance, expected_covariance, rtol=0.0, atol=3.0e-12)


def test_public_configured_source_preserves_declared_line_covariance() -> None:
    from pathlib import Path

    from rasim_next.pipeline.configured_simulation import (
        load_simulation_config,
        sample_configured_source,
    )

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(
        root / "configs" / "bi2se3_simulation.yaml",
        repository_root=root,
    )
    source = replace(config.source, sample_count=16, seed=6085)
    sampled = sample_configured_source(source)
    axes = np.asarray(source.transverse_axes_lab)
    position = (
        (sampled.origin_lab_m - np.asarray(source.mean_origin_lab_m)) @ axes.T
    ) / np.asarray(source.spatial_sigma_m)
    mean_direction = np.asarray(source.mean_direction_lab)
    cosine = np.clip(sampled.direction_lab @ mean_direction, -1.0, 1.0)
    radius = np.arccos(cosine)
    inverse_sine = np.divide(
        radius,
        np.sin(radius),
        out=np.ones_like(radius),
        where=radius != 0.0,
    )
    direction = (sampled.direction_lab @ axes.T * inverse_sine[:, None]) / np.asarray(
        source.divergence_sigma_rad
    )
    expected = np.eye(4)
    expected[0, 2] = expected[2, 0] = source.position_divergence_correlation[0]
    expected[1, 3] = expected[3, 1] = source.position_divergence_correlation[1]
    for wavelength_A in source.line_wavelength_A:
        selected = sampled.wavelength_A == wavelength_A
        weight = sampled.source_weight[selected]
        weight /= np.sum(weight)
        standardized = np.column_stack((position[selected], direction[selected]))
        np.testing.assert_allclose(
            standardized.T @ (weight[:, None] * standardized),
            expected,
            rtol=0.0,
            atol=3.0e-12,
        )


def test_equal_line_grid_reuses_geometry_and_common_width_nodes() -> None:
    lines_A = np.asarray((1.53, 1.56))
    probabilities = np.asarray((0.4, 0.6))
    line_sigma_A = 1.0e-4
    sampled = sample_discrete_gaussian_line_source_rays(
        mean_origin_lab_m=np.asarray((0.0, -0.02, 0.0)),
        mean_direction_lab=np.asarray((0.0, 1.0, 0.0)),
        transverse_axes_lab=np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        spatial_sigma_m=np.asarray((2.0e-5, 3.0e-5)),
        divergence_sigma_rad=np.asarray((4.0e-4, 5.0e-4)),
        line_wavelength_A=lines_A,
        line_probability=probabilities,
        common_wavelength_sigma_A=line_sigma_A,
        sample_count=16,
        seed=1729,
        polarization_state_id="THOMSON_UNPOLARIZED_UNANALYSED",
    )

    nearest_line = np.argmin(abs(sampled.wavelength_A[:, None] - lines_A[None, :]), axis=1)
    first = nearest_line == 0
    second = nearest_line == 1
    assert np.count_nonzero(first) == np.count_nonzero(second) == 8
    assert np.sum(sampled.source_weight[first]) == pytest.approx(probabilities[0])
    assert np.sum(sampled.source_weight[second]) == pytest.approx(probabilities[1])
    np.testing.assert_array_equal(sampled.origin_lab_m[first], sampled.origin_lab_m[second])
    np.testing.assert_array_equal(sampled.direction_lab[first], sampled.direction_lab[second])
    first_offset = sampled.wavelength_A[first] - lines_A[0]
    second_offset = sampled.wavelength_A[second] - lines_A[1]
    np.testing.assert_allclose(first_offset, second_offset, rtol=0.0, atol=2.0e-16)
    assert np.any(first_offset != 0.0)


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
