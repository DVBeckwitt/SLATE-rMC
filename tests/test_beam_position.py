"""Conditional source probability, affine ray transport and finite pixel mass."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from rasim_next.geometry.detector import _intersect_detector_plane
from rasim_next.geometry.sample import _intersect_sample_rays
from rasim_next.pipeline.beam_position import ConditionalBeamPosition, gaussian_pixel_probability
from rasim_next.pipeline.configured_simulation import (
    build_configured_simulation_inputs,
    build_source_averaged_detector,
    load_simulation_config,
    sample_configured_source,
)


def _configuration():
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs/bi2se3_simulation.yaml", repository_root=root)
    return replace(
        config,
        source=replace(config.source, sample_count=16),
        instrument=replace(
            config.instrument,
            detector_shape_rc=(96, 128),
            detector_reference_coordinate_px=(63.2, 47.1),
            detector_column_pitch_m=0.002,
            detector_row_pitch_m=0.003,
        ),
    )


def test_conditional_source_preserves_angles_spectrum_and_spatial_moments():
    config = _configuration()
    source = replace(config.source, divergence_sigma_rad=(0.0, 0.001))
    sampled = sample_configured_source(source)
    means = sample_configured_source(source, conditional_position=True)
    np.testing.assert_array_equal(means.direction_lab, sampled.direction_lab)
    np.testing.assert_array_equal(means.wavelength_A, sampled.wavelength_A)
    np.testing.assert_array_equal(means.source_weight, sampled.source_weight)
    assert means.source_revision != sampled.source_revision
    axes = np.asarray(source.transverse_axes_lab)
    tangent = means.direction_lab @ axes.T
    angle = np.arctan2(
        np.linalg.norm(tangent, axis=1), means.direction_lab @ source.mean_direction_lab
    )
    z = np.sign(tangent[:, 1]) * angle / source.divergence_sigma_rad[1]
    offsets = (means.origin_lab_m - source.mean_origin_lab_m) @ axes.T
    np.testing.assert_array_equal(offsets[:, 0], 0.0)
    np.testing.assert_allclose(
        offsets[:, 1],
        source.position_divergence_correlation[1] * source.spatial_sigma_m[1] * z,
        atol=2e-18,
    )
    beam = ConditionalBeamPosition.from_source(source, source_revision=means.source_revision)
    variance = np.sum((axes @ beam.factor_lab_m) ** 2, axis=1)
    expected = np.square(source.spatial_sigma_m) * (
        1 - np.square([0.0, source.position_divergence_correlation[1]])
    )
    np.testing.assert_allclose(variance, expected, rtol=2e-15)
    for wavelength, probability in zip(
        source.line_wavelength_A, source.line_probability, strict=True
    ):
        assert means.source_weight[means.wavelength_A == wavelength].sum() == pytest.approx(
            probability, abs=2e-15
        )


def test_conditional_projection_matches_direct_rays_and_rejects_incomplete_physics():
    inputs = build_configured_simulation_inputs(_configuration(), conditional_source_position=True)
    instrument = inputs.instrument
    beam = ConditionalBeamPosition.from_source(
        inputs.config.source, source_revision=inputs.samples.source_revision
    )
    projection = beam.compile_projection(inputs.incident.states, instrument)
    index = 0
    normals = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.3, 0.7]])
    origins = inputs.samples.origin_lab_m[index] + normals @ beam.factor_lab_m.T
    direction = inputs.samples.direction_lab[index]
    sample = _intersect_sample_rays(
        origins,
        np.broadcast_to(direction, origins.shape),
        lab_from_sample=instrument.lab_from_sample,
        sample_from_lab=instrument.sample_from_lab,
        sample_support_model_id=instrument.sample_support_model_id,
        sample_width_m=instrument.sample_width_m,
        sample_length_m=instrument.sample_length_m,
    )
    detector_rotation = instrument.lab_from_detector.rotation
    for offset in (np.array([0.013, -0.009, 0.0]), np.array([-0.017, 0.021, 0.0])):
        outgoing = (
            instrument.lab_from_detector.translation_m
            + detector_rotation @ offset
            - sample.point_lab_m[0]
        )
        outgoing /= np.linalg.norm(outgoing)
        hits = _intersect_detector_plane(
            sample.point_lab_m, np.broadcast_to(outgoing, origins.shape), instrument
        )
        detector_direction = detector_rotation.T @ outgoing
        slopes = (
            detector_direction[:2]
            / detector_direction[2]
            / np.array([instrument.detector_column_pitch_m, instrument.detector_row_pitch_m])
        )
        factor = projection[index, :2] - slopes[:, None] * projection[index, 2]
        coordinates = np.column_stack((hits.column_px, hits.row_px))
        np.testing.assert_allclose(
            coordinates, coordinates[0] + normals @ factor.T, rtol=0, atol=4e-13
        )
    wrong = replace(beam, factor_lab_m=2 * beam.factor_lab_m)
    with pytest.raises(ValueError, match="declared source law"):
        wrong.compile_projection(inputs.incident.states, instrument)
    with pytest.raises(ValueError, match="absorption"):
        beam.compile_projection(
            inputs.incident.states,
            replace(
                instrument,
                detector_path_medium_id="test_absorbing.v1",
                detector_path_linear_attenuation_m_inv=1.0,
            ),
        )
    detector = build_source_averaged_detector(inputs)
    with pytest.raises(ValueError, match="residual"):
        detector.compile_monte_carlo_sampler(execution_backend="cpu", seed=7)
    with pytest.raises(ValueError, match="conditional source"):
        detector.evaluate_detector_density_all_roots(np.array([63.0]), np.array([47.0]))


def test_gaussian_pixel_boxes_match_independent_integral_and_keep_lost_mass():
    from rasim_next.pipeline._continuous_detector_kernel import _deposit_root_mass

    for sx, sy, rho, dc, dr in (
        (0.03, 0.2, 0.9, 0.49, -0.4),
        (1.2, 0.8, -0.8, 0.2, -0.7),
        (3.0, 2.0, 0.99999, 0.2, 0.7),
    ):
        residual = np.sqrt(1 - rho * rho)
        factor = (sx, 0.0, rho * sy, sy * residual)
        a, b = (dc - 0.5) / sx, (dc + 0.5) / sx
        c, d = (dr - 0.5) / sy, (dr + 0.5) / sy
        transitions = [x for x in (c / rho, d / rho, 0.0) if a < x < b]
        expected = quad(
            lambda x, c=c, d=d, rho=rho, residual=residual: (
                np.exp(-x * x / 2)
                / np.sqrt(2 * np.pi)
                * (ndtr((d - rho * x) / residual) - ndtr((c - rho * x) / residual))
            ),
            a,
            b,
            points=transitions,
            epsabs=1e-12,
            limit=300,
        )[0]
        assert gaussian_pixel_probability(dc, dr, *factor) == pytest.approx(expected, abs=5e-8)
    image = np.zeros(41 * 53)
    deposited = _deposit_root_mass(image, 26.2, 20.1, 2.0, 41, 53, 1.2, 0.0, 0.8, 1.1)
    assert image.sum() == pytest.approx(2.0, abs=4e-8)
    assert deposited == pytest.approx(image.sum(), abs=2e-14)
    image.fill(0)
    deposited = _deposit_root_mass(image, -1.0, 20.0, 2.0, 41, 53, 1.0, 0.0, 0.0, 1.0)
    assert deposited == pytest.approx(2 * ndtr(-0.5), abs=1e-8)
    assert 0 < image.sum() < 2.0
    image.fill(0)
    _deposit_root_mass(image, 52.5, 20.0, 2.0, 41, 53, 0.0, 0.0, 0.0, 1.0)
    assert image.reshape(41, 53)[:, -1].sum() == pytest.approx(2.0, abs=1e-8)


def test_smooth_sampler_prefix_rebind_and_zero_width_match_cpu_cuda():
    from numba import cuda

    from rasim_next.core.transforms import RigidTransform
    from rasim_next.pipeline.source_averaged_detector import MonteCarloSamplingCancelled

    config = _configuration()
    inputs = build_configured_simulation_inputs(config, conditional_source_position=True)
    detector = build_source_averaged_detector(inputs)
    beam = ConditionalBeamPosition.from_source(
        config.source, source_revision=inputs.samples.source_revision
    )
    expected = None
    for backend in ("cpu", "cuda") if cuda.is_available() else ("cpu",):
        sampler = detector.compile_monte_carlo_sampler(
            execution_backend=backend, seed=91, beam_position=beam
        )
        preview = sampler.advance_preview_to(1)
        assert preview.position_integration_model_id == "conditional_gaussian_pixel_integral.v1"
        result = sampler.advance_to(3)
        assert result.measure_id == "raw_detector_pixel_mass_conditional_position_estimate_A2.v1"
        if expected is None:
            expected = result.image_A2
        np.testing.assert_allclose(result.image_A2, expected, rtol=2e-8, atol=2e-17)
        sampler.reset()
        np.testing.assert_allclose(sampler.advance_to(3).image_A2, expected, rtol=2e-8, atol=2e-17)
        with pytest.raises(MonteCarloSamplingCancelled):
            sampler.advance_to(4, cancel_requested=lambda: True)
        pose = detector.instrument.lab_from_detector
        shifted = replace(
            detector.instrument,
            lab_from_detector=RigidTransform(
                pose.rotation,
                pose.translation_m + np.array([0.001, -0.002, 0.0]),
                pose.source_frame,
                pose.target_frame,
            ),
        )
        sampler.rebind_detector_pose(shifted)
        fresh = detector.rebind_geometry(
            incident=inputs.incident, instrument=shifted
        ).compile_monte_carlo_sampler(execution_backend=backend, seed=91, beam_position=beam)
        np.testing.assert_allclose(
            sampler.advance_to(3).image_A2, fresh.advance_to(3).image_A2, rtol=2e-8, atol=2e-17
        )
    point_config = replace(config, source=replace(config.source, spatial_sigma_m=(0.0, 0.0)))
    point_images = []
    for smooth in (False, True):
        point = build_configured_simulation_inputs(point_config, conditional_source_position=smooth)
        position = (
            ConditionalBeamPosition.from_source(
                point_config.source, source_revision=point.samples.source_revision
            )
            if smooth
            else None
        )
        point_images.append(
            build_source_averaged_detector(point)
            .compile_monte_carlo_sampler(execution_backend="cpu", seed=91, beam_position=position)
            .advance_to(3)
            .image_A2
        )
    np.testing.assert_array_equal(*point_images)
