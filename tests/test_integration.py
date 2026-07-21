from __future__ import annotations

import math
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull, QhullError

from rasim_next.core.contracts import EventIntensityNormalization
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.geometry import (
    AngleFrame,
    InstrumentConfiguration,
    build_incident_states,
    compile_instrument,
    detector_coordinates_to_angles,
    project_detector_rays,
)
from rasim_next.materials import material_optics
from rasim_next.measurement import (
    AngleBinGrid,
    compile_detector_angle_projector,
    project_normalized_angle_field,
    to_increasing_phi,
)
from rasim_next.pipeline.configured_simulation import (
    build_configured_simulation_inputs,
    load_simulation_config,
)


def _configured_inputs(*, sample_count: int, sample_angle_deg: float = 5.0) -> object:
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    rotations = (
        replace(config.instrument.axis_rotations[0], angle_deg=sample_angle_deg),
        *config.instrument.axis_rotations[1:],
    )
    return build_configured_simulation_inputs(
        replace(
            config,
            source=replace(config.source, sample_count=sample_count),
            instrument=replace(config.instrument, axis_rotations=rotations),
        )
    )


def _intrinsic_detector_tilt(
    base_rotation: np.ndarray,
    column_deg: float,
    row_deg: float,
) -> np.ndarray:
    column = math.radians(column_deg)
    row = math.radians(row_deg)
    about_column = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(column), -math.sin(column)),
            (0.0, math.sin(column), math.cos(column)),
        )
    )
    about_current_row = np.asarray(
        (
            (math.cos(row), 0.0, math.sin(row)),
            (0.0, 1.0, 0.0),
            (-math.sin(row), 0.0, math.cos(row)),
        )
    )
    return np.asarray(base_rotation) @ about_column @ about_current_row


def test_continuous_upper_m1_maps_through_canonical_exit_before_pixel_binning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from painted_ewald import (
        BraggSpaceConfig,
        ContinuousEwaldCoating,
        MosaicBraggSpace,
        MosaicParameters,
        Rod,
    )
    from rasim_next.geometry import project_detector_ray
    from rasim_next.optics.attenuation import (
        mode_decay_constant,
        scalar_optical_weight,
        uniform_depth_attenuation,
    )
    from rasim_next.optics.refraction import solve_exit_mode
    from rasim_next.pipeline.bragg_space import Bi2Se3TwoHStrength
    from rasim_next.pipeline.continuous_detector import (
        DetectorEwaldMeasure,
        DetectorQuadrature,
        IntensityStatus,
        PixelIntegrationMethod,
    )

    inputs = _configured_inputs(sample_count=1)
    samples = inputs.samples
    instrument = inputs.instrument
    crystal = inputs.crystal
    material = inputs.material
    incident = inputs.incident
    reciprocal = inputs.reciprocal
    rods = tuple(
        Rod(h, k)
        for h, k in (
            (0, 0),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
        )
    )
    air_k0_Ainv = 2.0 * np.pi / samples.wavelength_A[0]
    bragg = MosaicBraggSpace(
        BraggSpaceConfig(
            reciprocal_basis_Ainv=reciprocal.basis_Ainv,
            crystal_to_sample=instrument.sample_from_crystal.rotation,
            rods=rods,
            mosaic=MosaicParameters(
                gaussian_sigma_rad=math.radians(5.0),
                lorentzian_half_width_rad=math.radians(2.0),
                lorentzian_probability=0.1,
                alpha_panel_count=12,
                alpha_gauss_order=6,
                azimuth_count=32,
            ),
            k_norm_Ainv=air_k0_Ainv,
        ),
        Bi2Se3TwoHStrength(
            crystal=crystal,
            layers=7,
            normalization=EventIntensityNormalization.FINITE_TOTAL,
        ),
    )
    coating = ContinuousEwaldCoating(
        bragg,
        ki_sample_Ainv=incident.states.k_film_phase_sample_Ainv[0],
    )
    detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=instrument,
    )
    rod = Rod(-1, 1)
    alpha_rad = math.radians(15.0)
    beta_rad = 2.21704764194

    mapped = detector.map_latent(
        rod=rod,
        branch=2,
        alpha_rad=alpha_rad,
        beta_rad=beta_rad,
    )

    assert mapped.intensity_status is IntensityStatus.INCLUDED
    assert bool(mapped.geometry.valid)
    assert mapped.geometry.column_px == pytest.approx(1267.56918, abs=2.0e-5)
    assert mapped.geometry.row_px == pytest.approx(577.77153, abs=2.0e-5)
    latent = coating.evaluate_latent(
        rod=rod,
        branch=2,
        alpha_rad=alpha_rad,
        beta_rad=beta_rad,
    )
    exit_mode = solve_exit_mode(
        latent.geometry.kf_sample_Ainv,
        samples.wavelength_A[0],
        material,
    )
    incident_kappa = mode_decay_constant(
        incident.states.kz_film_Ainv[0],
        -1,
    )
    exit_kappa = mode_decay_constant(exit_mode.kz_film_Ainv, 1)
    attenuation = uniform_depth_attenuation(
        incident_kappa,
        exit_kappa,
        instrument.film_thickness_A,
    )
    optical = scalar_optical_weight(
        incident.states.entrance_amplitude[0],
        exit_mode.exit_amplitude,
        attenuation,
    )
    expected_kf_air_lab = instrument.lab_from_sample.apply_vector(exit_mode.k_air_phase_sample_Ainv)
    projection = project_detector_ray(
        incident.states.sample_intersection_lab_m[0],
        expected_kf_air_lab / air_k0_Ainv,
        instrument,
    )
    np.testing.assert_allclose(
        mapped.geometry.kf_air_lab_Ainv,
        expected_kf_air_lab,
        rtol=0.0,
        atol=3.0e-15,
    )
    assert mapped.geometry.column_px == pytest.approx(projection.column_px, abs=2.0e-12)
    assert mapped.geometry.row_px == pytest.approx(projection.row_px, abs=2.0e-12)
    assert mapped.attenuation_weight == pytest.approx(attenuation, abs=2.0e-15)
    assert mapped.optical_weight == pytest.approx(optical, abs=2.0e-15)
    assert mapped.postoptical_density_A2_rad2_inv == pytest.approx(
        latent.coating_intensity_density_A2_rad2_inv * optical,
        rel=0.0,
        abs=2.0e-20,
    )
    assert mapped.postoptical_density_A2_rad2_inv != pytest.approx(
        latent.coating_intensity_density_A2_rad2_inv
        * optical
        * mapped.geometry.pixel_solid_angle_sr,
        rel=1.0e-6,
        abs=0.0,
    )

    caustic_density = detector.evaluate_detector_coordinates(
        mapped.geometry.column_px,
        mapped.geometry.row_px,
        rods=(rod,),
    )
    assert bool(caustic_density.caustic)
    assert np.isinf(caustic_density.density_A2_per_px2)
    zero_weight_detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=instrument,
        polarization_weight=0.0,
    )
    zero_caustic_density = zero_weight_detector.evaluate_detector_coordinates(
        mapped.geometry.column_px,
        mapped.geometry.row_px,
        rods=(rod,),
    )
    assert bool(zero_caustic_density.caustic)
    assert zero_caustic_density.density_A2_per_px2 == 0.0
    tiny_weight_detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=instrument,
        polarization_weight=1.0e-300,
    )
    tiny_caustic_density = tiny_weight_detector.evaluate_detector_coordinates(
        mapped.geometry.column_px,
        mapped.geometry.row_px,
        rods=(rod,),
    )
    assert bool(tiny_caustic_density.caustic)
    assert np.isinf(tiny_caustic_density.density_A2_per_px2)

    regular_alpha = math.radians(2.0)
    regular_beta = math.radians(178.0)
    regular_mapped = detector.map_latent(
        rod=rod,
        branch=2,
        alpha_rad=regular_alpha,
        beta_rad=regular_beta,
    )
    assert bool(regular_mapped.geometry.valid)
    step = 1.0e-6
    alpha_pair = detector.map_latent(
        rod=rod,
        branch=2,
        alpha_rad=np.array([regular_alpha - step, regular_alpha + step]),
        beta_rad=regular_beta,
    )
    beta_pair = detector.map_latent(
        rod=rod,
        branch=2,
        alpha_rad=regular_alpha,
        beta_rad=np.array([regular_beta - step, regular_beta + step]),
    )
    d_column_d_alpha = np.diff(alpha_pair.geometry.column_px)[0] / (2.0 * step)
    d_row_d_alpha = np.diff(alpha_pair.geometry.row_px)[0] / (2.0 * step)
    d_column_d_beta = np.diff(beta_pair.geometry.column_px)[0] / (2.0 * step)
    d_row_d_beta = np.diff(beta_pair.geometry.row_px)[0] / (2.0 * step)
    detector_jacobian = abs(d_column_d_alpha * d_row_d_beta - d_column_d_beta * d_row_d_alpha)
    regular_density = detector.evaluate_detector_coordinates(
        regular_mapped.geometry.column_px,
        regular_mapped.geometry.row_px,
        rods=(rod,),
    )
    internal_k = np.linalg.norm(incident.states.k_film_phase_sample_Ainv[0])
    expected_surface_jacobian = (
        internal_k
        * air_k0_Ainv
        * regular_density.geometry.kf_air_sample_Ainv[2]
        / regular_density.geometry.kf_film_sample_Ainv[2]
        * regular_mapped.geometry.pixel_solid_angle_sr
    )
    assert regular_density.geometry.q_surface_jacobian_Ainv2_per_px2 == pytest.approx(
        expected_surface_jacobian,
        rel=2.0e-14,
    )
    assert np.linalg.norm(
        regular_density.geometry.q_sample_Ainv + incident.states.k_film_phase_sample_Ainv[0]
    ) == pytest.approx(internal_k, abs=3.0e-15)
    assert regular_density.per_rod_inverse_branch_count == 1
    assert not bool(regular_density.caustic)
    assert regular_density.density_A2_per_px2 == pytest.approx(
        regular_mapped.postoptical_density_A2_rad2_inv / detector_jacobian,
        rel=2.0e-8,
    )
    two_branch_seed = detector.map_latent(
        rod=rod,
        branch=2,
        alpha_rad=math.radians(10.0),
        beta_rad=2.0,
    )
    two_branch_density = detector.evaluate_detector_coordinates(
        two_branch_seed.geometry.column_px,
        two_branch_seed.geometry.row_px,
        rods=(rod,),
    )
    assert two_branch_density.per_rod_inverse_branch_count == 2
    oracle_contributions = []
    for inverse_alpha, inverse_beta in (
        (0.17453292519943342, 2.000000000000001),
        (0.34730205158064265, 2.3814947269767046),
    ):
        inverse_forward = detector.map_latent(
            rod=rod,
            branch=2,
            alpha_rad=inverse_alpha,
            beta_rad=inverse_beta,
        )
        assert inverse_forward.geometry.column_px == pytest.approx(
            two_branch_seed.geometry.column_px, abs=5.0e-10
        )
        assert inverse_forward.geometry.row_px == pytest.approx(
            two_branch_seed.geometry.row_px, abs=5.0e-10
        )
        alpha_pair = detector.map_latent(
            rod=rod,
            branch=2,
            alpha_rad=np.array([inverse_alpha - step, inverse_alpha + step]),
            beta_rad=inverse_beta,
        )
        beta_pair = detector.map_latent(
            rod=rod,
            branch=2,
            alpha_rad=inverse_alpha,
            beta_rad=np.remainder(
                np.array([inverse_beta - step, inverse_beta + step]),
                2.0 * np.pi,
            ),
        )
        dc_da = np.diff(alpha_pair.geometry.column_px)[0] / (2.0 * step)
        dr_da = np.diff(alpha_pair.geometry.row_px)[0] / (2.0 * step)
        dc_db = np.diff(beta_pair.geometry.column_px)[0] / (2.0 * step)
        dr_db = np.diff(beta_pair.geometry.row_px)[0] / (2.0 * step)
        forward_jacobian = abs(dc_da * dr_db - dc_db * dr_da)
        oracle_contributions.append(
            float(inverse_forward.postoptical_density_A2_rad2_inv / forward_jacobian)
        )
    assert two_branch_density.density_A2_per_px2 == pytest.approx(
        math.fsum(oracle_contributions),
        rel=5.0e-8,
        abs=0.0,
    )
    mixed_density = detector.evaluate_detector_coordinates(
        np.array([regular_mapped.geometry.column_px, 0.0]),
        np.array([regular_mapped.geometry.row_px, 0.0]),
        rods=(rod,),
    )
    np.testing.assert_array_equal(
        mixed_density.per_rod_inverse_branch_count[0],
        regular_density.per_rod_inverse_branch_count,
    )
    np.testing.assert_allclose(
        mixed_density.density_A2_per_px2[0],
        regular_density.density_A2_per_px2,
        rtol=0.0,
        atol=0.0,
    )

    proof_column = np.array(
        [
            regular_mapped.geometry.column_px,
            two_branch_seed.geometry.column_px,
            mapped.geometry.column_px,
            -1.0,
        ]
    )
    proof_row = np.array(
        [
            regular_mapped.geometry.row_px,
            two_branch_seed.geometry.row_px,
            mapped.geometry.row_px,
            -1.0,
        ]
    )
    m1_rods = tuple(candidate for candidate in rods if candidate.family_m == 1)
    numpy_proof = detector.evaluate_detector_coordinates(
        proof_column,
        proof_row,
        rods=m1_rods,
    )
    compiled_density, compiled_count, compiled_caustic = (
        detector._evaluate_compiled_coordinates_for_proof(
            proof_column,
            proof_row,
            rods=m1_rods,
            branch=2,
        )
    )
    np.testing.assert_allclose(
        compiled_density,
        numpy_proof.per_rod_density_A2_per_px2,
        rtol=3.0e-12,
        atol=2.0e-24,
    )
    np.testing.assert_array_equal(compiled_count, numpy_proof.per_rod_inverse_branch_count)
    np.testing.assert_array_equal(compiled_caustic, numpy_proof.caustic)

    # The production pixel kernel must integrate the same arbitrary continuous
    # detector-coordinate density as the independent NumPy point evaluator.
    detector_columns = detector.instrument.detector_shape_rc[1]
    fused_pixel_row = np.asarray(
        [
            round(float(regular_mapped.geometry.row_px)),
            round(float(two_branch_seed.geometry.row_px)),
        ],
        dtype=np.int64,
    )
    fused_pixel_column = np.asarray(
        [
            round(float(regular_mapped.geometry.column_px)),
            round(float(two_branch_seed.geometry.column_px)),
        ],
        dtype=np.int64,
    )
    fused_flat_index = fused_pixel_row * detector_columns + fused_pixel_column
    fused_offset = np.asarray((-0.25, 0.25), dtype=np.float64)
    fused_weight = np.asarray((0.5, 0.5), dtype=np.float64)
    fused = detector._compiled_evaluator(m1_rods).integrate_pixel_boxes(
        fused_flat_index,
        offset_px=fused_offset,
        one_dimensional_weight=fused_weight,
        branch=2,
        include_center_diagnostics=False,
    )
    oracle_column, oracle_row = np.broadcast_arrays(
        fused_pixel_column[:, None, None] + fused_offset[None, None, :],
        fused_pixel_row[:, None, None] + fused_offset[None, :, None],
    )
    oracle_density = detector.evaluate_detector_coordinates(
        oracle_column,
        oracle_row,
        rods=m1_rods,
        branch=2,
    ).per_rod_density_A2_per_px2
    oracle_mass = np.sum(
        oracle_density * (fused_weight[:, None] * fused_weight[None, :])[None, :, :, None],
        axis=(1, 2),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        fused.per_rod_mass_A2,
        oracle_mass,
        rtol=3.0e-12,
        atol=2.0e-24,
    )
    assert fused.center_per_rod_density_A2_per_px2.size == 0
    assert fused.per_rod_inverse_count_min.size == 0
    assert fused.per_rod_inverse_count_max.size == 0
    assert fused.per_rod_caustic.size == 0
    assert fused.valid_any.size == 0
    assert fused.valid_all.size == 0
    assert fused.center_valid.size == 0

    specular = detector.map_specular_geometry(
        rod=Rod(0, 0),
        alpha_rad=0.0,
        beta_rad=0.0,
    )
    assert specular.intensity_status is IntensityStatus.SPECULAR_INTENSITY_EXCLUDED
    assert bool(specular.geometry.valid)
    assert not hasattr(specular, "postoptical_density_A2_rad2_inv")

    tiny_instrument = replace(
        instrument,
        detector_shape_rc=(3, 3),
        detector_reference_coordinate_px=(
            instrument.detector_reference_coordinate_px[0]
            + 1.0
            - regular_mapped.geometry.column_px,
            instrument.detector_reference_coordinate_px[1] + 1.0 - regular_mapped.geometry.row_px,
        ),
    )
    tiny_detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=tiny_instrument,
    )
    quadrature = DetectorQuadrature(pixel_gauss_order=2, row_chunk_size=2)
    pixels = tiny_detector.integrate_native_pixels(
        rods=m1_rods,
        branch=2,
        quadrature=quadrature,
    )
    assert pixels.image_A2.shape == (3, 3)
    assert pixels.image_A2.flags.writeable is False
    assert pixels.rods == m1_rods
    assert pixels.branch == 2
    assert pixels.total_detector_mass_A2 == pytest.approx(
        np.sum(pixels.per_rod_detector_mass_A2),
        rel=0.0,
        abs=2.0e-20,
    )
    assert np.sum(pixels.image_A2, dtype=np.float64) == pytest.approx(
        pixels.total_detector_mass_A2,
        rel=0.0,
        abs=2.0e-20,
    )
    assert pixels.total_detector_mass_A2 > 0.0
    assert pixels.sampled_valid_pixel_center is None
    assert not hasattr(pixels, "alpha_rad")
    assert not hasattr(pixels, "beta_rad")

    accelerated_quadrature = DetectorQuadrature(
        method=PixelIntegrationMethod.ADAPTIVE_COMPILED,
        pixel_gauss_order=2,
        relative_tolerance=1.0e-6,
        max_depth=3,
        row_chunk_size=2,
        worker_count=1,
    )
    accelerated = tiny_detector.integrate_native_pixels(
        rods=m1_rods,
        branch=2,
        quadrature=accelerated_quadrature,
    )
    parallel = tiny_detector.integrate_native_pixels(
        rods=m1_rods,
        branch=2,
        quadrature=replace(accelerated_quadrature, worker_count=2),
    )
    np.testing.assert_allclose(
        accelerated.image_A2,
        pixels.image_A2,
        rtol=2.0e-5,
        atol=2.0e-20,
    )
    np.testing.assert_array_equal(parallel.image_A2, accelerated.image_A2)
    np.testing.assert_array_equal(
        parallel.per_rod_detector_mass_A2,
        accelerated.per_rod_detector_mass_A2,
    )
    center_column, center_row = np.meshgrid(
        np.arange(3, dtype=np.float64),
        np.arange(3, dtype=np.float64),
    )
    center_geometry = tiny_detector.evaluate_detector_geometry(
        center_column,
        center_row,
        include_surface_jacobian=False,
    )
    np.testing.assert_array_equal(center_geometry.q_surface_jacobian_Ainv2_per_px2, 0.0)
    expected_center_valid = center_geometry.valid
    assert accelerated.sampled_valid_pixel_center is not None
    assert accelerated.sampled_valid_pixel_center.flags.writeable is False
    np.testing.assert_array_equal(
        accelerated.sampled_valid_pixel_center,
        expected_center_valid,
    )
    np.testing.assert_array_equal(
        parallel.sampled_valid_pixel_center,
        accelerated.sampled_valid_pixel_center,
    )
    assert parallel.adaptive_refined_pixel_count == accelerated.adaptive_refined_pixel_count
    assert parallel.adaptive_unresolved_pixel_count == accelerated.adaptive_unresolved_pixel_count
    assert parallel.sampled_invalid_pixel_count == accelerated.sampled_invalid_pixel_count
    assert parallel.coordinate_evaluation_count == accelerated.coordinate_evaluation_count
    assert parallel.estimated_l1_error_A2 == accelerated.estimated_l1_error_A2
    assert accelerated.coordinate_evaluation_count > 0
    assert accelerated.execution_backend == "numba_nogil_thread_tiles.v1"

    from rasim_next.pipeline._continuous_detector_kernel import CompiledDetectorEvaluator

    def reject_node_field_materialization(*args: object, **kwargs: object) -> None:
        raise AssertionError("adaptive pixel integration materialized a node-scale field")

    monkeypatch.setattr(CompiledDetectorEvaluator, "evaluate", reject_node_field_materialization)
    fused_only = tiny_detector.integrate_native_pixels(
        rods=m1_rods,
        branch=2,
        quadrature=accelerated_quadrature,
    )
    np.testing.assert_array_equal(fused_only.image_A2, accelerated.image_A2)
    np.testing.assert_array_equal(
        fused_only.per_rod_detector_mass_A2,
        accelerated.per_rod_detector_mass_A2,
    )
    np.testing.assert_array_equal(
        fused_only.sampled_valid_pixel_center,
        accelerated.sampled_valid_pixel_center,
    )

    caustic_instrument = replace(
        instrument,
        detector_shape_rc=(3, 3),
        detector_reference_coordinate_px=(
            instrument.detector_reference_coordinate_px[0] + 1.0 - mapped.geometry.column_px,
            instrument.detector_reference_coordinate_px[1] + 1.0 - mapped.geometry.row_px,
        ),
    )
    caustic_detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=caustic_instrument,
    )
    caustic_pixels = caustic_detector.integrate_native_pixels(
        rods=(rod,),
        branch=2,
        quadrature=DetectorQuadrature(
            pixel_gauss_order=2,
            fold_gauss_order=4,
            fold_subdivision_count=8,
            row_chunk_size=2,
        ),
    )
    assert caustic_pixels.fold_refined_pixel_count > 0
    assert np.all(np.isfinite(caustic_pixels.image_A2))
    assert caustic_pixels.fold_refinement_l1_A2 / caustic_pixels.total_detector_mass_A2 < 2.0e-2
    assert caustic_pixels.fold_refinement_centroid_shift_px < 5.0e-2
    compiled_caustic_pixels = caustic_detector.integrate_native_pixels(
        rods=(rod,),
        branch=2,
        quadrature=replace(
            accelerated_quadrature,
            relative_tolerance=1.0e-3,
            worker_count=2,
        ),
    )
    assert np.all(np.isfinite(compiled_caustic_pixels.image_A2))
    assert compiled_caustic_pixels.adaptive_refined_pixel_count > 0
    assert compiled_caustic_pixels.adaptive_unresolved_pixel_count > 0
    assert not compiled_caustic_pixels.adaptive_tolerance_satisfied
    zero_caustic_detector = DetectorEwaldMeasure(
        coating=coating,
        incident=incident,
        material=material,
        instrument=caustic_instrument,
        polarization_weight=0.0,
    )
    zero_pixels = zero_caustic_detector.integrate_native_pixels(
        rods=(rod,),
        branch=2,
        quadrature=DetectorQuadrature(
            pixel_gauss_order=2,
            fold_gauss_order=4,
            fold_subdivision_count=8,
            row_chunk_size=2,
        ),
    )
    assert zero_pixels.total_detector_mass_A2 == 0.0
    np.testing.assert_array_equal(zero_pixels.per_rod_detector_mass_A2, 0.0)
    assert not np.any(zero_pixels.image_A2)
    assert zero_pixels.fold_refinement_l1_A2 == 0.0
    assert zero_pixels.fold_refinement_centroid_shift_px == 0.0


def _two_state_source_averaged_detector_fixture(
    *,
    detector_shape_rc: tuple[int, int] | None = None,
) -> tuple[object, tuple[object, ...]]:
    from painted_ewald import (
        BraggSpaceConfig,
        ContinuousEwaldCoating,
        MosaicBraggSpace,
        MosaicParameters,
        Rod,
    )
    from rasim_next.core.contracts import IncidentSampleBatch
    from rasim_next.pipeline.bragg_space import Bi2Se3TwoHStrength
    from rasim_next.pipeline.continuous_detector import (
        DetectorEwaldMeasure,
    )
    from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure

    inputs = _configured_inputs(sample_count=2)
    samples = inputs.samples
    instrument = inputs.instrument
    crystal = inputs.crystal
    material = inputs.material
    incident = inputs.incident
    reciprocal = inputs.reciprocal
    rods = tuple(Rod(h, k) for h, k in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)))
    mosaic = MosaicParameters(
        gaussian_sigma_rad=math.radians(5.0),
        lorentzian_half_width_rad=math.radians(2.0),
        lorentzian_probability=0.1,
    )
    strength = Bi2Se3TwoHStrength(
        crystal=crystal,
        layers=7,
        normalization=EventIntensityNormalization.FINITE_TOTAL,
    )
    singleton_incidents = []
    coatings = []
    for state_index in range(2):
        singleton = IncidentSampleBatch(
            incident_sample_id=np.asarray([state_index], dtype=np.int64),
            origin_lab_m=samples.origin_lab_m[state_index : state_index + 1],
            direction_lab=samples.direction_lab[state_index : state_index + 1],
            wavelength_A=samples.wavelength_A[state_index : state_index + 1],
            source_weight=np.asarray([1.0]),
            polarization_state_id=(samples.polarization_state_id[state_index],),
            source_sampling_model_id="explicit_external_source.v1",
            source_rng_model_id="no_rng.v1",
            source_seed=state_index,
            source_parameter_provenance=f"two-state scalar oracle row {state_index}",
        )
        singleton_incident = build_incident_states(singleton, material, instrument)
        singleton_incidents.append(singleton_incident)
        bragg = MosaicBraggSpace(
            BraggSpaceConfig(
                reciprocal_basis_Ainv=reciprocal.basis_Ainv,
                crystal_to_sample=instrument.sample_from_crystal.rotation,
                rods=rods,
                mosaic=mosaic,
                k_norm_Ainv=2.0 * np.pi / singleton.wavelength_A[0],
            ),
            strength,
        )
        coatings.append(
            ContinuousEwaldCoating(
                bragg,
                ki_sample_Ainv=singleton_incident.states.k_film_phase_sample_Ainv[0],
            )
        )
    if detector_shape_rc is not None:
        provisional = DetectorEwaldMeasure(
            coating=coatings[0],
            incident=singleton_incidents[0],
            material=material,
            instrument=instrument,
        )
        seed = provisional.map_latent(
            rod=rods[1],
            branch=2,
            alpha_rad=math.radians(2.0),
            beta_rad=math.radians(178.0),
        )
        rows, columns = detector_shape_rc
        instrument = replace(
            instrument,
            detector_shape_rc=detector_shape_rc,
            detector_reference_coordinate_px=(
                instrument.detector_reference_coordinate_px[0]
                + 0.5 * (columns - 1)
                - float(seed.geometry.column_px),
                instrument.detector_reference_coordinate_px[1]
                + 0.5 * (rows - 1)
                - float(seed.geometry.row_px),
            ),
        )
    averaged = SourceAveragedDetectorEwaldMeasure(
        reciprocal_basis_Ainv=reciprocal.basis_Ainv,
        crystal_to_sample=instrument.sample_from_crystal.rotation,
        rods=rods,
        mosaic=mosaic,
        strength_model=strength,
        incident=incident,
        material=material,
        instrument=instrument,
        worker_count=2,
    )
    scalar_detectors = tuple(
        DetectorEwaldMeasure(
            coating=coating,
            incident=singleton_incident,
            material=material,
            instrument=instrument,
        )
        for coating, singleton_incident in zip(coatings, singleton_incidents, strict=True)
    )
    return averaged, scalar_detectors


def test_source_averaged_detector_density_equals_independent_state_sum() -> None:
    averaged, scalar_detectors = _two_state_source_averaged_detector_fixture()
    rods = averaged.rods
    seed_rod = rods[1]
    mapped = tuple(
        detector.map_latent(
            rod=seed_rod,
            branch=2,
            alpha_rad=math.radians(2.0),
            beta_rad=math.radians(178.0),
        )
        for detector in scalar_detectors
    )
    column_px = np.asarray([item.geometry.column_px for item in mapped])
    row_px = np.asarray([item.geometry.row_px for item in mapped])

    result = averaged.evaluate_detector_coordinates(column_px, row_px, branch=2)
    scalar = tuple(
        detector.evaluate_detector_coordinates(column_px, row_px, rods=rods, branch=2)
        for detector in scalar_detectors
    )
    expected_per_rod = 0.5 * (
        scalar[0].per_rod_density_A2_per_px2 + scalar[1].per_rod_density_A2_per_px2
    )

    np.testing.assert_allclose(
        result.per_rod_density_A2_per_px2,
        expected_per_rod,
        rtol=3.0e-11,
        atol=2.0e-24,
    )
    np.testing.assert_allclose(
        result.density_A2_per_px2,
        np.sum(expected_per_rod, axis=-1),
        rtol=3.0e-11,
        atol=2.0e-24,
    )
    np.testing.assert_array_equal(result.caustic, scalar[0].caustic | scalar[1].caustic)
    np.testing.assert_array_equal(result.valid_source_count, np.asarray([2, 2]))
    assert result.source_revision == averaged.incident.states.source_revision
    assert result.measure_id == "raw_detector_coordinate_density_A2_per_px2.v1"
    for detector, scalar_result in zip(scalar_detectors, scalar, strict=True):
        internal_ki = detector.coating.ki_sample_Ainv
        valid = scalar_result.geometry.valid
        np.testing.assert_allclose(
            np.linalg.norm(scalar_result.geometry.q_sample_Ainv[valid] + internal_ki, axis=-1),
            np.linalg.norm(internal_ki),
            rtol=0.0,
            atol=4.0e-15,
        )


def test_source_averaged_pixel_integral_is_one_outer_integral_of_state_sum() -> None:
    from rasim_next.pipeline.continuous_detector import DetectorQuadrature

    detector_shape = (8, 8)
    averaged, scalar_detectors = _two_state_source_averaged_detector_fixture(
        detector_shape_rc=detector_shape
    )
    quadrature = DetectorQuadrature(
        pixel_gauss_order=2,
        fold_gauss_order=2,
        fold_subdivision_count=1,
        row_chunk_size=2,
    )

    result = averaged.integrate_native_pixels(branch=2, quadrature=quadrature)
    scalar = tuple(
        detector.integrate_native_pixels(
            rods=averaged.rods,
            branch=2,
            quadrature=quadrature,
        )
        for detector in scalar_detectors
    )
    expected_image = 0.5 * (scalar[0].image_A2 + scalar[1].image_A2)
    expected_per_rod = 0.5 * (
        scalar[0].per_rod_detector_mass_A2 + scalar[1].per_rod_detector_mass_A2
    )

    np.testing.assert_allclose(result.image_A2, expected_image, rtol=3.0e-11, atol=2.0e-24)
    np.testing.assert_allclose(
        result.per_rod_detector_mass_A2,
        expected_per_rod,
        rtol=3.0e-11,
        atol=2.0e-24,
    )
    assert result.coordinate_evaluation_count == detector_shape[0] * detector_shape[1] * 2**2
    assert result.execution_backend == "numba_source_averaged.v1"
    assert not result.adaptive_tolerance_satisfied


def test_source_average_all_roots_includes_detector_regularized_m0() -> None:
    from painted_ewald import (
        BraggSpaceConfig,
        ContinuousEwaldCoating,
        MosaicBraggSpace,
        Rod,
    )
    from rasim_next.pipeline.continuous_detector import DetectorEwaldMeasure, DetectorQuadrature
    from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure

    nonzero, scalar_detectors = _two_state_source_averaged_detector_fixture()
    rods = (Rod(0, 0), *nonzero.rods)
    strength = scalar_detectors[0].coating.bragg_space.strength_model
    material = material_optics(strength.crystal, nonzero.incident.states.wavelength_A)
    detector = SourceAveragedDetectorEwaldMeasure(
        reciprocal_basis_Ainv=scalar_detectors[0].coating.bragg_space.config.reciprocal_basis_Ainv,
        crystal_to_sample=nonzero.instrument.sample_from_crystal.rotation,
        rods=rods,
        mosaic=scalar_detectors[0].coating.bragg_space.config.mosaic,
        strength_model=strength,
        incident=nonzero.incident,
        material=material,
        instrument=nonzero.instrument,
        worker_count=2,
    )
    with pytest.raises(ValueError, match="pixel integration cannot include m=0"):
        detector.integrate_native_pixels(
            branch=2,
            quadrature=DetectorQuadrature(
                pixel_gauss_order=2,
                fold_gauss_order=2,
                fold_subdivision_count=1,
                row_chunk_size=2,
            ),
        )
    seed = scalar_detectors[0].map_latent(
        rod=nonzero.rods[1],
        branch=2,
        alpha_rad=math.radians(2.0),
        beta_rad=math.radians(178.0),
    )
    column_px = np.asarray([seed.geometry.column_px])
    row_px = np.asarray([seed.geometry.row_px])

    intrinsic_nonzero = scalar_detectors[0].map_detector_visible_coating(
        rod=nonzero.rods[1],
        branch=2,
        alpha_rad=np.asarray([math.radians(2.0)]),
        beta_rad=np.asarray([math.radians(178.0)]),
    )
    direct_nonzero = scalar_detectors[0].coating.evaluate_latent(
        rod=nonzero.rods[1],
        branch=2,
        alpha_rad=np.asarray([math.radians(2.0)]),
        beta_rad=np.asarray([math.radians(178.0)]),
    )
    assert intrinsic_nonzero.geometry.valid[0]
    np.testing.assert_allclose(
        intrinsic_nonzero.coating_intensity_density_A2_rad2_inv,
        direct_nonzero.coating_intensity_density_A2_rad2_inv,
        rtol=0.0,
        atol=0.0,
    )
    assert (
        intrinsic_nonzero.measure_id
        == "detector_visible_intrinsic_ewald_latent_density_A2_rad2_inv.v1"
    )
    zero_weight_detector = DetectorEwaldMeasure(
        coating=scalar_detectors[0].coating,
        incident=scalar_detectors[0].incident,
        material=material,
        instrument=nonzero.instrument,
        phase_population_weight=0.0,
        polarization_weight=0.0,
    )
    zero_weight_intrinsic = zero_weight_detector.map_detector_visible_coating(
        rod=nonzero.rods[1],
        branch=2,
        alpha_rad=np.asarray([math.radians(2.0)]),
        beta_rad=np.asarray([math.radians(178.0)]),
    )
    np.testing.assert_allclose(
        zero_weight_intrinsic.coating_intensity_density_A2_rad2_inv,
        intrinsic_nonzero.coating_intensity_density_A2_rad2_inv,
        rtol=0.0,
        atol=0.0,
    )

    all_roots = detector.evaluate_detector_coordinates_all_roots(column_px, row_px)
    explicit_cpu = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cpu",
    )
    np.testing.assert_array_equal(
        explicit_cpu.per_rod_density_A2_per_px2,
        all_roots.per_rod_density_A2_per_px2,
    )
    with pytest.raises(ValueError, match="identify exactly the CUDA backend"):
        replace(all_roots, execution_device="unexpected device")
    with pytest.raises(ValueError, match="identify exactly the CUDA backend"):
        replace(all_roots, execution_backend="numba_cuda_source_averaged.v1")
    with pytest.raises(ValueError, match="execution_backend"):
        detector.evaluate_detector_coordinates_all_roots(
            column_px,
            row_px,
            execution_backend="automatic",
        )
    lower = nonzero.evaluate_detector_coordinates(column_px, row_px, branch=1)
    upper = nonzero.evaluate_detector_coordinates(column_px, row_px, branch=2)

    assert all_roots.root_policy == "all_retained_roots.v1"
    assert all_roots.branch is None
    assert all_roots.detector_visible_m0_q_gap_Ainv > 0.3
    assert all_roots.rods[0] == Rod(0, 0)
    assert np.isfinite(all_roots.per_rod_density_A2_per_px2[..., 0]).all()
    assert np.all(all_roots.per_rod_density_A2_per_px2[..., 0] > 0.0)
    expected_m0 = np.zeros(column_px.shape, dtype=np.float64)
    first_m0_oracle = None
    for scalar_detector in scalar_detectors:
        scalar_bragg = MosaicBraggSpace(
            BraggSpaceConfig(
                reciprocal_basis_Ainv=(
                    scalar_detector.coating.bragg_space.config.reciprocal_basis_Ainv
                ),
                crystal_to_sample=nonzero.instrument.sample_from_crystal.rotation,
                rods=rods,
                mosaic=scalar_detector.coating.bragg_space.config.mosaic,
                k_norm_Ainv=scalar_detector.coating.bragg_space.config.k_norm_Ainv,
            ),
            strength,
        )
        scalar_oracle = DetectorEwaldMeasure(
            coating=ContinuousEwaldCoating(
                scalar_bragg,
                ki_sample_Ainv=scalar_detector.coating.ki_sample_Ainv,
            ),
            incident=scalar_detector.incident,
            material=material,
            instrument=nonzero.instrument,
        )
        if first_m0_oracle is None:
            first_m0_oracle = scalar_oracle
            intrinsic_m0 = scalar_oracle.map_detector_visible_coating(
                rod=rods[0],
                branch=0,
                alpha_rad=np.asarray([0.0]),
                beta_rad=np.asarray([0.0]),
            )
            assert intrinsic_m0.geometry.valid[0]
            assert intrinsic_m0.detector_visible_m0_q_gap_Ainv > 0.3
            m0_geometry = intrinsic_m0.geometry.ewald_geometry
            q_norm = float(np.linalg.norm(m0_geometry.q_sample_Ainv[0]))
            assert q_norm > intrinsic_m0.detector_visible_m0_q_gap_Ainv
            assert abs(float(m0_geometry.u_Ainv[0])) > 0.0
            assert m0_geometry.ewald_residual_Ainv[0] < 4.0e-15
            latent_m0 = scalar_bragg.evaluate_latent(
                rod=rods[0],
                alpha_rad=m0_geometry.alpha_rad,
                beta_rad=m0_geometry.beta_rad,
                u_Ainv=m0_geometry.u_Ainv,
            )
            incident_norm = float(np.linalg.norm(scalar_oracle.coating.ki_sample_Ainv))
            np.testing.assert_allclose(
                intrinsic_m0.coating_intensity_density_A2_rad2_inv,
                latent_m0.intensity_density_A2_rad2_inv * (2.0 * incident_norm / q_norm),
                rtol=3.0e-14,
                atol=0.0,
            )
            with pytest.raises(ValueError, match="branch 0"):
                scalar_oracle.map_detector_visible_coating(
                    rod=rods[0],
                    branch=1,
                    alpha_rad=0.0,
                    beta_rad=0.0,
                )
        geometry, optical = scalar_oracle._detector_coordinate_state(column_px, row_px)
        for branch in (1, 2):
            density, _, _ = scalar_oracle._inverse_rod_density(
                geometry=geometry,
                optical_weight=optical,
                rod=rods[0],
                branch=branch,
            )
            expected_m0 += 0.5 * density
    np.testing.assert_allclose(
        all_roots.per_rod_density_A2_per_px2[..., 0],
        expected_m0,
        rtol=4.0e-11,
        atol=3.0e-24,
    )

    assert first_m0_oracle is not None
    positive_k = first_m0_oracle.coating.ki_sample_Ainv.copy()
    positive_k[2] = abs(positive_k[2])
    positive_kz = first_m0_oracle.incident.states.kz_film_Ainv.copy()
    positive_kz.real[:] = np.abs(positive_kz.real)
    positive_states = replace(
        first_m0_oracle.incident.states,
        k_film_phase_sample_Ainv=positive_k[None, :],
        kz_film_Ainv=positive_kz,
    )
    positive_incident = replace(first_m0_oracle.incident, states=positive_states)
    with pytest.raises(ValueError, match="negative sample-normal half-space"):
        SourceAveragedDetectorEwaldMeasure(
            reciprocal_basis_Ainv=first_m0_oracle.coating.bragg_space.config.reciprocal_basis_Ainv,
            crystal_to_sample=nonzero.instrument.sample_from_crystal.rotation,
            rods=rods,
            mosaic=first_m0_oracle.coating.bragg_space.config.mosaic,
            strength_model=strength,
            incident=positive_incident,
            material=material,
            instrument=nonzero.instrument,
        )
    positive_detector = DetectorEwaldMeasure(
        coating=ContinuousEwaldCoating(
            first_m0_oracle.coating.bragg_space,
            ki_sample_Ainv=positive_k,
        ),
        incident=positive_incident,
        material=material,
        instrument=nonzero.instrument,
    )
    with pytest.raises(ValueError, match="negative incident sample-normal"):
        positive_detector.map_detector_visible_coating(
            rod=rods[0],
            branch=0,
            alpha_rad=0.0,
            beta_rad=0.0,
        )
    np.testing.assert_allclose(
        all_roots.per_rod_density_A2_per_px2[..., 1:],
        lower.per_rod_density_A2_per_px2 + upper.per_rod_density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )


def test_cuda_detector_backend_fails_closed_without_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rasim_next.pipeline import _continuous_detector_cuda as cuda_backend
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_source_averaged_detector,
        load_simulation_config,
    )

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(
        root / "configs" / "bi2se3_simulation.yaml",
        repository_root=root,
    )
    config = replace(config, source=replace(config.source, sample_count=1))
    detector = build_source_averaged_detector(build_configured_simulation_inputs(config))
    monkeypatch.setattr(cuda_backend.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device is available"):
        detector.evaluate_detector_coordinates_all_roots(
            np.asarray([0.0]),
            np.asarray([0.0]),
            execution_backend="cuda",
        )


def test_cuda_default_source_blocks_match_cpu_with_shared_disorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numba import cuda

    if not cuda.is_available():
        pytest.skip("requires a CUDA device")

    from rasim_next.pipeline import _continuous_detector_cuda as cuda_backend
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_source_averaged_detector,
        load_simulation_config,
    )
    from rasim_next.proof.tolerances import load_stage_tolerances

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(
        root / "configs" / "bi2se3_simulation.yaml",
        repository_root=root,
    )
    inputs = build_configured_simulation_inputs(config)
    detector = build_source_averaged_detector(inputs)
    assert inputs.strength.shared_disorder_epsilon == pytest.approx(0.001)
    unique_count, frequency = np.unique(
        detector.reachable_rod_count_per_source_state,
        return_counts=True,
    )
    assert dict(zip(unique_count.tolist(), frequency.tolist(), strict=True)) == {73: 9, 85: 991}

    panel_rows, panel_columns = detector.instrument.detector_shape_rc
    column_px = np.asarray(
        [1109.5, 1469.5, 2206.820508075689, -1.0, float(panel_columns)],
        dtype=np.float64,
    )
    row_px = np.asarray(
        [1349.5, 1469.5, 1272.1794919243112, 0.0, float(panel_rows)],
        dtype=np.float64,
    )
    cpu = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cpu",
    )
    gpu = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
    )
    repeated = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
    )
    monkeypatch.setattr(cuda_backend, "_MAX_COORDINATES_PER_CHUNK", 2)
    chunked = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
    )

    assert np.any(cpu.per_rod_density_A2_per_px2[0, 1:] > 0.0)
    cuda_compound_relative_tolerance = 6.0e-11
    assert (
        cuda_compound_relative_tolerance
        <= load_stage_tolerances()["stacking.finite_intensity"].rtol
    )
    np.testing.assert_allclose(
        gpu.per_rod_density_A2_per_px2,
        cpu.per_rod_density_A2_per_px2,
        rtol=cuda_compound_relative_tolerance,
        atol=3.0e-24,
    )
    np.testing.assert_allclose(
        gpu.density_A2_per_px2,
        cpu.density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(gpu.caustic, cpu.caustic)
    np.testing.assert_array_equal(gpu.valid_source_count, cpu.valid_source_count)
    np.testing.assert_array_equal(
        repeated.per_rod_density_A2_per_px2, gpu.per_rod_density_A2_per_px2
    )
    np.testing.assert_array_equal(repeated.caustic, gpu.caustic)
    np.testing.assert_array_equal(repeated.valid_source_count, gpu.valid_source_count)
    np.testing.assert_array_equal(
        chunked.per_rod_density_A2_per_px2, gpu.per_rod_density_A2_per_px2
    )
    np.testing.assert_array_equal(chunked.caustic, gpu.caustic)
    np.testing.assert_array_equal(chunked.valid_source_count, gpu.valid_source_count)
    assert gpu.execution_backend == "numba_cuda_source_averaged.v1"
    assert gpu.execution_device
    assert np.all(gpu.per_rod_density_A2_per_px2[3:] == 0.0)


def test_cuda_compound_detector_tilt_matches_cpu() -> None:
    from numba import cuda

    if not cuda.is_available():
        pytest.skip("requires a CUDA device")

    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_source_averaged_detector,
        load_simulation_config,
    )

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(
        root / "configs" / "bi2se3_simulation.yaml",
        repository_root=root,
    )
    base_detector = config.instrument.lab_from_detector
    tilted_rotation = _intrinsic_detector_tilt(
        np.asarray(base_detector.rotation),
        0.7,
        -1.1,
    )
    tilted_detector = replace(
        base_detector,
        rotation=tuple(tuple(float(entry) for entry in row) for row in tilted_rotation),
    )
    config = replace(
        config,
        source=replace(config.source, sample_count=1),
        instrument=replace(config.instrument, lab_from_detector=tilted_detector),
    )
    detector = build_source_averaged_detector(build_configured_simulation_inputs(config))
    column_px = np.asarray([1109.5, 1469.5, -1.0])
    row_px = np.asarray([1349.5, 1469.5, 0.0])
    cpu = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cpu",
    )
    gpu = detector.evaluate_detector_coordinates_all_roots(
        column_px,
        row_px,
        execution_backend="cuda",
    )

    assert np.any(cpu.per_rod_density_A2_per_px2[:2] > 0.0)
    np.testing.assert_allclose(
        gpu.per_rod_density_A2_per_px2,
        cpu.per_rod_density_A2_per_px2,
        rtol=6.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_allclose(
        gpu.density_A2_per_px2,
        cpu.density_A2_per_px2,
        rtol=4.0e-11,
        atol=3.0e-24,
    )
    np.testing.assert_array_equal(gpu.caustic, cpu.caustic)
    np.testing.assert_array_equal(gpu.valid_source_count, cpu.valid_source_count)
    assert np.all(gpu.per_rod_density_A2_per_px2[2] == 0.0)


def test_yaml_simulation_config_is_strict_and_plans_all_elastic_rods(
    tmp_path: Path,
) -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_source_averaged_detector,
        load_simulation_config,
    )

    root = Path(__file__).resolve().parents[1]
    default_path = root / "configs" / "bi2se3_simulation.yaml"
    config = load_simulation_config(default_path, repository_root=root)

    assert config.schema_version == "rasim-simulation-v2"
    assert config.enabled_artifact_names == (
        "reciprocal_space",
        "ewald_surface",
        "detector",
    )
    assert config.bragg.selection_model == "all_elastic_reachable.v1"
    assert config.bragg.include_detector_visible_m0
    assert config.source.sample_count == 1_000
    assert config.numerics.detector_execution_backend == "cpu"
    assert config.mosaic.gaussian_sigma_deg == pytest.approx(1.0)
    assert config.mosaic.lorentzian_probability == 0.0
    assert config.structure_factor.layers == 52
    assert config.structure_factor.shared_disorder_epsilon == pytest.approx(0.001)
    assert not config.output_directory.is_relative_to(root)

    inputs = build_configured_simulation_inputs(config)
    assert 2.0 * np.pi / inputs.bragg_space.config.k_norm_Ainv == pytest.approx(
        config.source.mean_wavelength_A,
        rel=0.0,
        abs=2.0e-15,
    )
    expected_multiplicity = {
        0: 1,
        1: 6,
        3: 6,
        4: 6,
        7: 12,
        9: 6,
        12: 6,
        13: 12,
        16: 6,
        19: 12,
        21: 12,
    }
    assert {
        family: sum(rod.family_m == family for rod in inputs.rods)
        for family in sorted({rod.family_m for rod in inputs.rods})
    } == expected_multiplicity
    detector = build_source_averaged_detector(inputs)
    unique_count, frequency = np.unique(
        detector.reachable_rod_count_per_source_state,
        return_counts=True,
    )
    assert dict(zip(unique_count.tolist(), frequency.tolist(), strict=True)) == {73: 9, 85: 991}
    assert not detector.reachable_rod_count_per_source_state.flags.writeable

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "schema_version: rasim-simulation-v2\nschema_version: duplicate\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"duplicate key.*schema_version"):
        load_simulation_config(duplicate, repository_root=root)

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(
        default_path.read_text(encoding="utf-8") + "\nunknown_parameter: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"unknown key.*unknown_parameter"):
        load_simulation_config(unknown, repository_root=root)

    duplicate_filename = tmp_path / "duplicate-filename.yaml"
    portable_default = default_path.read_text(encoding="utf-8").replace(
        "../examples/bi2se3/structures/Bi2Se3_vesta.cif",
        (root / "examples/bi2se3/structures/Bi2Se3_vesta.cif").as_posix(),
    )
    duplicate_filename.write_text(
        portable_default.replace(
            "filename: ewald-surface.png",
            "filename: reciprocal-space.png",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="output filenames must be unique"):
        load_simulation_config(duplicate_filename, repository_root=root)

    unsupported_backend = tmp_path / "unsupported-backend.yaml"
    unsupported_backend.write_text(
        portable_default.replace(
            "detector_execution_backend: cpu",
            "detector_execution_backend: automatic",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"detector_execution_backend.*cpu or cuda"):
        load_simulation_config(unsupported_backend, repository_root=root)

    missing_backend = tmp_path / "missing-backend.yaml"
    missing_backend.write_text(
        portable_default.replace("  detector_execution_backend: cpu\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"numerics.*missing.*detector_execution_backend"):
        load_simulation_config(missing_backend, repository_root=root)

    negative_source_sigma = tmp_path / "negative-source-sigma.yaml"
    negative_source_sigma.write_text(
        portable_default.replace(
            "spatial_sigma_m: [2.123304500720048e-05, 2.123304500720048e-05]",
            "spatial_sigma_m: [-1.0, 2.123304500720048e-05]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"source\.spatial_sigma_m.*nonnegative"):
        load_simulation_config(negative_source_sigma, repository_root=root)

    excessive_reciprocal_tilt = tmp_path / "excessive-reciprocal-tilt.yaml"
    excessive_reciprocal_tilt.write_text(
        portable_default.replace(
            "reciprocal_alpha_max_deg: 5.0",
            "reciprocal_alpha_max_deg: 181.0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"reciprocal_alpha_max_deg.*180"):
        load_simulation_config(excessive_reciprocal_tilt, repository_root=root)

    all_disabled = tmp_path / "all-disabled.yaml"
    all_disabled.write_text(
        portable_default.replace("enabled: true", "enabled: false"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one output"):
        load_simulation_config(all_disabled, repository_root=root)


def test_nominal_ewald_gap_is_attached_only_to_visible_m0_support() -> None:
    from painted_ewald import MosaicBraggSpace
    from rasim_next.pipeline.configured_simulation import (
        build_nominal_ewald_context,
        evaluate_nominal_ewald_surface,
    )

    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    config = replace(
        config,
        source=replace(config.source, sample_count=1),
        instrument=replace(
            config.instrument,
            detector_reference_coordinate_px=(-1000.0, 1500.0),
        ),
    )
    inputs = build_configured_simulation_inputs(config)
    stale_bragg = MosaicBraggSpace(
        replace(
            inputs.bragg_space.config,
            k_norm_Ainv=inputs.bragg_space.config.k_norm_Ainv * 1.001,
        ),
        inputs.strength,
    )
    with pytest.raises(ValueError, match="wavelength does not match"):
        build_nominal_ewald_context(replace(inputs, bragg_space=stale_bragg))
    display = evaluate_nominal_ewald_surface(
        build_nominal_ewald_context(inputs),
        alpha_count=16,
        beta_count=72,
        alpha_max_deg=5.0,
    )

    assert np.any(display.family_m != 0)
    assert not np.any(display.family_m == 0)
    assert display.detector_visible_m0_q_gap_Ainv is None


def test_yaml_detector_two_axis_tilt_folds_into_canonical_pose(tmp_path: Path) -> None:
    from rasim_next.core.validity import ValidityCode
    from rasim_next.geometry import detector_coordinate_to_ray, project_detector_ray
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        load_simulation_config,
    )

    root = Path(__file__).resolve().parents[1]
    default_path = root / "configs" / "bi2se3_simulation.yaml"
    portable_default = default_path.read_text(encoding="utf-8").replace(
        "../examples/bi2se3/structures/Bi2Se3_vesta.cif",
        (root / "examples/bi2se3/structures/Bi2Se3_vesta.cif").as_posix(),
    )
    zero_tilt = "  detector_tilt:\n    about_column_axis_deg: 0.0\n    about_row_axis_deg: 0.0\n"
    base_path = tmp_path / "base.yaml"
    zero_path = tmp_path / "zero.yaml"
    tilted_path = tmp_path / "tilted.yaml"
    base_path.write_text(portable_default.replace(zero_tilt, "", 1), encoding="utf-8")
    zero_path.write_text(portable_default, encoding="utf-8")
    tilted_path.write_text(
        portable_default.replace(
            zero_tilt,
            "  detector_tilt:\n    about_column_axis_deg: 7.0\n    about_row_axis_deg: -11.0\n",
            1,
        ),
        encoding="utf-8",
    )
    base = load_simulation_config(base_path, repository_root=root)
    zero = load_simulation_config(zero_path, repository_root=root)
    tilted = load_simulation_config(tilted_path, repository_root=root)

    assert zero.instrument.lab_from_detector == base.instrument.lab_from_detector
    assert zero.physics_revision == base.physics_revision
    base_rotation = np.asarray(base.instrument.lab_from_detector.rotation)
    expected_rotation = _intrinsic_detector_tilt(base_rotation, 7.0, -11.0)
    np.testing.assert_allclose(
        tilted.instrument.lab_from_detector.rotation,
        expected_rotation,
        rtol=0.0,
        atol=3.0e-16,
    )
    assert (
        tilted.instrument.lab_from_detector.translation_m
        == base.instrument.lab_from_detector.translation_m
    )
    assert tilted.physics_revision != base.physics_revision

    one_sample = replace(tilted, source=replace(tilted.source, sample_count=1))
    instrument = build_configured_simulation_inputs(one_sample).instrument
    np.testing.assert_allclose(
        instrument.lab_from_detector.rotation,
        expected_rotation,
        rtol=0.0,
        atol=3.0e-16,
    )
    column_px, row_px = 1200.25, 1500.75
    origin_lab_m = np.zeros(3)
    ray = detector_coordinate_to_ray(
        column_px,
        row_px,
        origin_lab_m=origin_lab_m,
        instrument=instrument,
    )
    assert ray.status is ValidityCode.VALID
    projection = project_detector_ray(origin_lab_m, ray.direction_lab, instrument)
    assert projection.status is ValidityCode.VALID
    assert projection.column_px == pytest.approx(column_px, rel=0.0, abs=2.0e-10)
    assert projection.row_px == pytest.approx(row_px, rel=0.0, abs=2.0e-10)
    pixel_area_m2 = instrument.detector_column_pitch_m * instrument.detector_row_pitch_m
    expected_solid_angle_sr = (
        pixel_area_m2
        * abs(float(expected_rotation[:, 2] @ ray.direction_lab))
        / ray.ray_distance_m**2
    )
    assert projection.pixel_solid_angle_sr == pytest.approx(
        expected_solid_angle_sr,
        rel=2.0e-15,
        abs=0.0,
    )


def test_detector_macrobin_preview_applies_the_fixed_quadrature_area_once() -> None:
    from types import SimpleNamespace

    from rasim_next.pipeline.configured_simulation import integrate_detector_macrobins

    class ConstantDetector:
        instrument = SimpleNamespace(detector_shape_rc=(4, 6))

        @staticmethod
        def evaluate_detector_coordinates_all_roots(
            column_px: np.ndarray,
            row_px: np.ndarray,
        ) -> object:
            shape = np.broadcast_shapes(column_px.shape, row_px.shape)
            per_rod = np.broadcast_to(np.asarray([2.0, 3.0]), (*shape, 2))
            return SimpleNamespace(
                per_rod_density_A2_per_px2=per_rod,
                density_A2_per_px2=np.sum(per_rod, axis=-1),
                caustic=np.zeros((*shape, 2), dtype=np.bool_),
                valid_source_count=np.ones(shape, dtype=np.int64),
            )

    result = integrate_detector_macrobins(
        ConstantDetector(),
        bin_size_px=2,
        gauss_order=2,
    )
    np.testing.assert_allclose(result.image_A2, 20.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(result.per_rod_image_A2[..., 0], 8.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.per_rod_image_A2[..., 1], 12.0, rtol=0.0, atol=1.0e-14)
    assert result.coordinate_evaluation_count == 24
    assert result.measure_id == "raw_detector_macrobin_fixed_quadrature_estimate_A2.v1"
    with pytest.raises(ValueError, match="identify exactly the CUDA backend"):
        replace(result, execution_device="unexpected device")
    with pytest.raises(ValueError, match="identify exactly the CUDA backend"):
        replace(result, execution_backend="numba_cuda_source_averaged.v1")


def test_continuous_detector_cli_exposes_mosaic_parameters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    namespace = runpy.run_path(str(root / "scripts" / "generate_bi2se3_continuous_detector.py"))

    with pytest.raises(SystemExit) as exit_info:
        namespace["main"](["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--gaussian-sigma-deg" in help_text
    assert "--lorentzian-hwhm-deg" in help_text
    assert "--eta" in help_text
    assert "--layers" in help_text
    assert "--stacking-epsilon" in help_text

    with pytest.raises(SystemExit) as invalid_exit:
        namespace["main"](
            [
                "--numeric-only",
                "--gaussian-sigma-deg",
                "0",
                "--lorentzian-hwhm-deg",
                "0",
                "--eta",
                "0",
            ]
        )

    assert invalid_exit.value.code == 2
    assert "Gaussian sigma must be positive" in capsys.readouterr().err


def _instrument(
    *,
    shape_rc: tuple[int, int] = (3, 4),
    reference_cr: tuple[float, float] = (1.5, 1.0),
    detector_rotation: np.ndarray | None = None,
) -> object:
    identity = np.eye(3)
    zero = np.zeros(3)
    rotation = identity if detector_rotation is None else detector_rotation
    configuration = InstrumentConfiguration(
        axis_rotations=(),
        lab_from_goniometer_zero=RigidTransform(identity, zero, FrameId.GONIOMETER, FrameId.LAB),
        goniometer_from_sample=RigidTransform(identity, zero, FrameId.SAMPLE, FrameId.GONIOMETER),
        sample_from_crystal=RigidTransform(identity, zero, FrameId.CRYSTAL, FrameId.SAMPLE),
        lab_from_detector=RigidTransform(
            rotation,
            [1.1e-3, -0.7e-3, 0.82],
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        detector_shape_rc=shape_rc,
        detector_row_pitch_m=3.1e-4,
        detector_column_pitch_m=1.7e-4,
        detector_reference_coordinate_px=reference_cr,
        sample_support_model_id="finite_rectangle.v1",
        sample_width_m=4.0e-4,
        sample_length_m=6.0e-4,
        film_thickness_A=500.0,
    )
    return compile_instrument(configuration)


def test_configured_initial_beam_maps_exactly_to_the_untilted_detector() -> None:
    inputs = _configured_inputs(sample_count=6)
    samples = inputs.samples
    instrument = inputs.instrument
    sample_angle_rad = math.radians(5.0)
    cosine = math.cos(sample_angle_rad)
    sine = math.sin(sample_angle_rad)
    expected_lab_from_sample_rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    )
    np.testing.assert_allclose(
        instrument.lab_from_sample.rotation,
        expected_lab_from_sample_rotation,
        rtol=0.0,
        atol=2.0e-16,
    )
    incident = inputs.incident.states
    sample_normal_lab = expected_lab_from_sample_rotation[:, 2]
    source_to_sample_m = -(
        (samples.origin_lab_m - instrument.lab_from_sample.translation_m) @ sample_normal_lab
    ) / (samples.direction_lab @ sample_normal_lab)
    expected_sample_intersection_lab_m = (
        samples.origin_lab_m + source_to_sample_m[:, None] * samples.direction_lab
    )
    np.testing.assert_allclose(
        incident.sample_intersection_lab_m,
        expected_sample_intersection_lab_m,
        rtol=0.0,
        atol=2.0e-16,
    )
    k_air_norm_Ainv = 2.0 * np.pi / incident.wavelength_A
    direction_lab = instrument.lab_from_sample.apply_vector(incident.k_air_sample_Ainv)
    direction_lab /= k_air_norm_Ainv[:, None]
    np.testing.assert_allclose(direction_lab, samples.direction_lab, rtol=0.0, atol=5.0e-16)
    projection = project_detector_rays(
        incident.sample_intersection_lab_m,
        direction_lab,
        instrument,
    )

    assert samples.incident_sample_id.size == 6
    np.testing.assert_array_equal(samples.source_weight, np.full(6, 1.0 / 6.0))
    assert np.all(incident.valid)
    assert np.all(projection.valid)
    detector_normal_lab = instrument.lab_from_detector.apply_vector([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(detector_normal_lab, np.array([0.0, 1.0, 0.0]))

    weights = incident.source_weight
    center_column, center_row = instrument.detector_reference_coordinate_px
    centroid = np.asarray(
        (
            np.sum(weights * projection.column_px),
            np.sum(weights * projection.row_px),
        )
    )
    np.testing.assert_allclose(
        centroid,
        np.asarray((center_column, center_row)),
        rtol=0.0,
        atol=3.0e-12,
    )
    detector_plane_y_m = 0.075
    plane_distance_m = (detector_plane_y_m - samples.origin_lab_m[:, 1]) / samples.direction_lab[
        :, 1
    ]
    expected_point_lab_m = samples.origin_lab_m + plane_distance_m[:, None] * samples.direction_lab
    np.testing.assert_allclose(
        projection.column_px,
        center_column + expected_point_lab_m[:, 0] / 1.0e-4,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        projection.row_px,
        center_row - expected_point_lab_m[:, 2] / 1.0e-4,
        rtol=0.0,
        atol=2.0e-12,
    )
    wavelength_mean_A = float(np.sum(weights * incident.wavelength_A))
    assert wavelength_mean_A == pytest.approx(1.540592925, abs=2.0e-15)
    assert np.sum(weights[projection.valid]) == pytest.approx(1.0, abs=2.0e-15)


def _frame(origin_lab_m: np.ndarray | list[float] | None = None) -> AngleFrame:
    return AngleFrame(
        origin_lab_m=np.zeros(3) if origin_lab_m is None else origin_lab_m,
        row_down_lab=np.array([0.0, 1.0, 0.0]),
        column_right_lab=np.array([1.0, 0.0, 0.0]),
        direct_beam_lab=np.array([0.0, 0.0, 1.0]),
        revision="integration-angle-frame.v1",
    )


def _full_grid(instrument: object, frame: AngleFrame, *, radial_bins: int = 5) -> AngleBinGrid:
    rows, columns = instrument.detector_shape_rc
    corner_column, corner_row = np.meshgrid(
        np.arange(columns + 1, dtype=np.float64) - 0.5,
        np.arange(rows + 1, dtype=np.float64) - 0.5,
    )
    angles = detector_coordinates_to_angles(
        corner_column,
        corner_row,
        instrument=instrument,
        angle_frame=frame,
    )
    assert np.all(angles.valid)
    theta_max = float(np.max(angles.two_theta_rad))
    return AngleBinGrid(
        two_theta_edges_rad=np.linspace(0.0, np.nextafter(theta_max, np.inf), radial_bins + 1),
        chi_raw_edges_rad=np.linspace(-np.pi, np.pi, 17),
        revision="integration-grid.v1",
    )


def _unwrap(raw_chi: np.ndarray) -> np.ndarray:
    result = np.array(raw_chi, dtype=np.float64, copy=True)
    for index in range(1, result.size):
        delta = (raw_chi[index] - raw_chi[index - 1] + np.pi) % (2.0 * np.pi) - np.pi
        result[index] = result[index - 1] + delta
    return result


def _cross(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _inside_convex(point: np.ndarray, polygon: np.ndarray) -> bool:
    edge = np.roll(polygon, -1, axis=0) - polygon
    offset = point - polygon
    crosses = edge[:, 0] * offset[:, 1] - edge[:, 1] * offset[:, 0]
    return bool(np.all(crosses >= -2e-14) or np.all(crosses <= 2e-14))


def _segment_intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray | None:
    first_delta = first_end - first_start
    second_delta = second_end - second_start
    denominator = _cross(first_delta, second_delta)
    if abs(denominator) <= 1e-18:
        return None
    offset = second_start - first_start
    first_fraction = _cross(offset, second_delta) / denominator
    second_fraction = _cross(offset, first_delta) / denominator
    if -2e-14 <= first_fraction <= 1.0 + 2e-14 and -2e-14 <= second_fraction <= 1.0 + 2e-14:
        return first_start + first_fraction * first_delta
    return None


def _independent_intersection_area(polygon: np.ndarray, rectangle: np.ndarray) -> float:
    lower = rectangle[0]
    upper = rectangle[2]
    points: list[np.ndarray] = []
    for point in polygon:
        if np.all(point >= lower - 2e-14) and np.all(point <= upper + 2e-14):
            points.append(point)
    for point in rectangle:
        if _inside_convex(point, polygon):
            points.append(point)
    for subject_index in range(polygon.shape[0]):
        subject_start = polygon[subject_index]
        subject_end = polygon[(subject_index + 1) % polygon.shape[0]]
        for clip_index in range(4):
            intersection = _segment_intersection(
                subject_start,
                subject_end,
                rectangle[clip_index],
                rectangle[(clip_index + 1) % 4],
            )
            if intersection is not None:
                points.append(intersection)
    unique = [
        point
        for index, point in enumerate(points)
        if not any(np.linalg.norm(point - earlier) <= 2e-13 for earlier in points[:index])
    ]
    if len(unique) < 3:
        return 0.0
    try:
        return float(ConvexHull(np.asarray(unique)).volume)
    except QhullError:
        return 0.0


def _independent_pixel_weights(
    column: int,
    row: int,
    *,
    instrument: object,
    frame: AngleFrame,
    grid: AngleBinGrid,
    pole_cr: tuple[float, float] | None = None,
) -> np.ndarray:
    corner_column = np.array([column - 0.5, column + 0.5, column + 0.5, column - 0.5])
    corner_row = np.array([row - 0.5, row - 0.5, row + 0.5, row + 0.5])
    angles = detector_coordinates_to_angles(
        corner_column,
        corner_row,
        instrument=instrument,
        angle_frame=frame,
    )
    physical_corners = np.column_stack((corner_column, corner_row))
    contains_pole = pole_cr is not None and (
        column - 0.5 <= pole_cr[0] <= column + 0.5 and row - 0.5 <= pole_cr[1] <= row + 0.5
    )
    pieces = []
    if contains_pole:
        pole = np.asarray(pole_cr)
        for first in range(4):
            second = (first + 1) % 4
            if (
                abs(
                    _cross(
                        physical_corners[first] - pole,
                        physical_corners[second] - pole,
                    )
                )
                <= 1e-15
            ):
                continue
            assert angles.azimuth_valid[first] and angles.azimuth_valid[second]
            chi = _unwrap(angles.chi_raw_rad[[first, second]])
            pieces.append(
                np.array(
                    [
                        [0.0, chi[0]],
                        [angles.two_theta_rad[first], chi[0]],
                        [angles.two_theta_rad[second], chi[1]],
                        [0.0, chi[1]],
                    ]
                )
            )
    else:
        assert np.all(angles.azimuth_valid)
        for indices in ((0, 1, 2), (0, 2, 3)):
            index = np.asarray(indices)
            pieces.append(
                np.column_stack((angles.two_theta_rad[index], _unwrap(angles.chi_raw_rad[index])))
            )
    piece_area = math.fsum(float(ConvexHull(piece).volume) for piece in pieces)
    expected = np.zeros(grid.shape, dtype=np.float64)
    period = 2.0 * np.pi
    for chi_bin in range(grid.shape[0]):
        for theta_bin in range(grid.shape[1]):
            theta_lower = grid.two_theta_edges_rad[theta_bin]
            theta_upper = grid.two_theta_edges_rad[theta_bin + 1]
            for shift in range(-2, 3):
                chi_lower = grid.chi_raw_edges_rad[chi_bin] + shift * period
                chi_upper = grid.chi_raw_edges_rad[chi_bin + 1] + shift * period
                rectangle = np.array(
                    [
                        [theta_lower, chi_lower],
                        [theta_upper, chi_lower],
                        [theta_upper, chi_upper],
                        [theta_lower, chi_upper],
                    ]
                )
                expected[chi_bin, theta_bin] += math.fsum(
                    _independent_intersection_area(piece, rectangle) for piece in pieces
                )
    return expected / piece_area


def _dense_projector(projector: object) -> np.ndarray:
    bins = int(np.prod(projector.grid.shape))
    pixels = int(np.prod(projector.instrument.detector_shape_rc))
    dense = np.zeros((bins, pixels), dtype=np.float64)
    np.add.at(
        dense, (projector.coverage_bin_index, projector.coverage_pixel_index), projector.weight
    )
    return dense


def test_angle_projector_fingerprint_tracks_only_detector_geometry() -> None:
    instrument = _instrument(shape_rc=(2, 3))
    frame = _frame([4.0e-3, -0.7e-3, 0.0])
    grid = _full_grid(instrument, frame, radial_bins=4)
    baseline = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=grid,
    )
    assert baseline.instrument_fingerprint.endswith(".v2")

    detector = instrument.lab_from_detector
    angle_rad = np.deg2rad(3.0)
    detector_rotation = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [np.sin(angle_rad), np.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    detector_causal_instruments = (
        replace(
            instrument,
            lab_from_detector=RigidTransform(
                detector_rotation,
                detector.translation_m,
                FrameId.DETECTOR,
                FrameId.LAB,
            ),
        ),
        replace(
            instrument,
            lab_from_detector=RigidTransform(
                detector.rotation,
                detector.translation_m + np.array([1.0e-4, 0.0, 0.0]),
                FrameId.DETECTOR,
                FrameId.LAB,
            ),
        ),
        replace(instrument, detector_shape_rc=(3, 3)),
        replace(instrument, detector_row_pitch_m=instrument.detector_row_pitch_m * 1.01),
        replace(
            instrument,
            detector_column_pitch_m=instrument.detector_column_pitch_m * 1.01,
        ),
        replace(
            instrument,
            detector_reference_coordinate_px=(
                instrument.detector_reference_coordinate_px[0] + 0.25,
                instrument.detector_reference_coordinate_px[1],
            ),
        ),
    )
    causal_projectors = tuple(
        compile_detector_angle_projector(
            instrument=item,
            angle_frame=frame,
            grid=grid,
        )
        for item in detector_causal_instruments
    )
    causal_fingerprints = {item.instrument_fingerprint for item in causal_projectors}
    assert baseline.instrument_fingerprint not in causal_fingerprints
    assert len(causal_fingerprints) == len(causal_projectors)
    assert all(item.cache_key != baseline.cache_key for item in causal_projectors)

    sample_rotation = np.array(
        [
            [np.cos(angle_rad), 0.0, np.sin(angle_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle_rad), 0.0, np.cos(angle_rad)],
        ]
    )
    excluded_instruments = (
        replace(
            instrument,
            lab_from_sample=RigidTransform(
                sample_rotation,
                np.array([1.0e-4, -2.0e-4, 3.0e-4]),
                FrameId.SAMPLE,
                FrameId.LAB,
            ),
        ),
        replace(
            instrument,
            sample_from_crystal=RigidTransform(
                sample_rotation,
                np.array([2.0e-4, 0.0, 0.0]),
                FrameId.CRYSTAL,
                FrameId.SAMPLE,
            ),
        ),
        replace(instrument, sample_width_m=instrument.sample_width_m * 1.1),
        replace(
            instrument,
            sample_support_model_id="unbounded_plane.v1",
            sample_width_m=None,
            sample_length_m=None,
        ),
        replace(instrument, film_thickness_A=instrument.film_thickness_A * 1.2),
    )
    for excluded_instrument in excluded_instruments:
        candidate = compile_detector_angle_projector(
            instrument=excluded_instrument,
            angle_frame=frame,
            grid=grid,
        )
        assert candidate.instrument_fingerprint == baseline.instrument_fingerprint
        assert candidate.cache_key == baseline.cache_key
        for name in (
            "detector_valid_mask",
            "angle_bin_valid_mask",
            "coverage_pixel_index",
            "coverage_bin_index",
            "weight",
            "lost_support",
        ):
            np.testing.assert_array_equal(getattr(candidate, name), getattr(baseline, name))

    changed_frame = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=replace(frame, revision="separate-angle-frame-owner.v2"),
        grid=grid,
    )
    assert changed_frame.instrument_fingerprint == baseline.instrument_fingerprint
    assert changed_frame.cache_key != baseline.cache_key


def test_sparse_projector_matches_independent_polygon_oracle_across_seam() -> None:
    angle = np.deg2rad(7.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    instrument = _instrument(shape_rc=(2, 3), detector_rotation=rotation)
    frame = _frame([4.0e-3, -0.7e-3, 0.0])
    grid = _full_grid(instrument, frame, radial_bins=4)
    near_canonical_theta = grid.two_theta_edges_rad.copy()
    near_canonical_chi = grid.chi_raw_edges_rad.copy()
    near_canonical_theta[0] = np.nextafter(0.0, np.inf)
    near_canonical_chi[[0, -1]] = np.nextafter([-np.pi, np.pi], 0.0)
    canonicalized = AngleBinGrid(near_canonical_theta, near_canonical_chi, "canonicalized.v1")
    assert canonicalized.two_theta_edges_rad[0] == 0.0
    np.testing.assert_array_equal(canonicalized.chi_raw_edges_rad[[0, -1]], [-np.pi, np.pi])
    projector = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=grid,
    )
    dense = _dense_projector(projector)

    seam_pixel: int | None = None

    for row in range(2):
        for column in range(3):
            pixel = row * 3 + column
            pixel_angles = detector_coordinates_to_angles(
                [column - 0.5, column + 0.5, column + 0.5, column - 0.5],
                [row - 0.5, row - 0.5, row + 0.5, row + 0.5],
                instrument=instrument,
                angle_frame=frame,
            )
            if np.ptp(pixel_angles.chi_raw_rad) > np.pi:
                seam_pixel = pixel
            expected = _independent_pixel_weights(
                column,
                row,
                instrument=instrument,
                frame=frame,
                grid=grid,
            )
            np.testing.assert_allclose(
                dense[:, pixel].reshape(grid.shape),
                expected,
                rtol=3e-11,
                atol=3e-13,
            )
    assert seam_pixel is not None
    seam_column = dense[:, seam_pixel].reshape(grid.shape)
    assert np.any(seam_column[0] > 0.0)
    assert np.any(seam_column[-1] > 0.0)
    np.testing.assert_allclose(np.sum(dense, axis=0), 1.0, rtol=0.0, atol=3e-12)
    np.testing.assert_allclose(projector.lost_support, 0.0, rtol=0.0, atol=3e-12)
    pair_key = projector.coverage_pixel_index * math.prod(grid.shape) + projector.coverage_bin_index
    assert np.all(np.diff(pair_key) > 0)
    assert np.all(projector.weight > 0.0)
    assert not projector.weight.flags.writeable
    assert not projector.coverage_bin_index.flags.writeable
    assert not projector.coverage_pixel_index.flags.writeable

    with pytest.raises(ValueError, match="cache_key"):
        replace(projector, cache_key="stale.v1")
    split_revision_a = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=replace(frame, revision="ab"),
        grid=replace(grid, revision="c"),
    )
    split_revision_b = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=replace(frame, revision="a"),
        grid=replace(grid, revision="bc"),
    )
    changed_mask = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=grid,
        detector_valid_mask=np.array([[False, True, True], [True, True, True]]),
    )
    assert (
        len(
            {
                projector.cache_key,
                split_revision_a.cache_key,
                split_revision_b.cache_key,
                changed_mask.cache_key,
            }
        )
        == 4
    )


@pytest.mark.parametrize("reference_cr", [(1.0, 1.0), (1.5, 1.0), (1.5, 1.5)])
def test_direct_beam_pole_ties_cover_full_detector_support(
    reference_cr: tuple[float, float],
) -> None:
    base = _instrument(shape_rc=(3, 3), reference_cr=reference_cr)
    instrument = replace(
        base,
        lab_from_detector=RigidTransform(
            np.eye(3), [0.0, 0.0, 0.82], FrameId.DETECTOR, FrameId.LAB
        ),
    )
    frame = _frame()
    projector = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=_full_grid(instrument, frame),
    )
    dense = _dense_projector(projector)
    np.testing.assert_allclose(projector.lost_support, 0.0, rtol=0.0, atol=4e-12)
    np.testing.assert_allclose(
        np.sum(dense, axis=0) + projector.lost_support.ravel(),
        1.0,
        rtol=0.0,
        atol=4e-12,
    )
    touching = [
        row * 3 + column
        for row in range(3)
        for column in range(3)
        if column - 0.5 <= reference_cr[0] <= column + 0.5
        and row - 0.5 <= reference_cr[1] <= row + 0.5
    ]
    assert len(touching) in (1, 2, 4)
    supports = []
    for pixel in touching:
        row, column = divmod(pixel, 3)
        expected = _independent_pixel_weights(
            column,
            row,
            instrument=instrument,
            frame=frame,
            grid=projector.grid,
            pole_cr=reference_cr,
        )
        actual = dense[:, pixel].reshape(projector.grid.shape)
        np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=4e-13)
        supports.append(np.flatnonzero(np.any(actual > 0.0, axis=1)))
    assert all(support.size == projector.grid.shape[0] // len(touching) for support in supports)
    np.testing.assert_array_equal(
        np.unique(np.concatenate(supports)),
        np.arange(projector.grid.shape[0]),
    )

    if reference_cr == (1.5, 1.0):
        for shifted_column in (np.nextafter(1.5, -np.inf), np.nextafter(1.5, np.inf)):
            shifted = replace(
                instrument,
                detector_reference_coordinate_px=(shifted_column, 1.0),
            )
            shifted_projector = compile_detector_angle_projector(
                instrument=shifted,
                angle_frame=frame,
                grid=projector.grid,
            )
            np.testing.assert_allclose(
                _dense_projector(shifted_projector),
                dense,
                rtol=0.0,
                atol=4e-12,
            )


def test_normalized_field_freezes_masks_losses_divide_order_and_phi_permutation() -> None:
    instrument = _instrument(shape_rc=(2, 3))
    frame = _frame([4.0e-3, -0.7e-3, 0.0])
    full_grid = _full_grid(instrument, frame, radial_bins=2)
    detector_mask = np.array([[True, True, False], [True, True, True]])
    angle_mask = np.ones(full_grid.shape, dtype=np.bool_)
    angle_mask[0, 0] = False
    projector = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=full_grid,
        detector_valid_mask=detector_mask,
        angle_bin_valid_mask=angle_mask,
    )
    signal = np.array([[2.0, 9.0, 7.0], [5.0, 4.0, 11.0]])
    normalization = np.array([[1.0, 3.0, 2.0], [5.0, 2.0, 4.0]])
    field = project_normalized_angle_field(projector, signal, normalization)
    dense = _dense_projector(projector)
    expected_signal = (dense @ np.where(detector_mask, signal, 0.0).ravel()).reshape(
        full_grid.shape
    )
    expected_normalization = (dense @ np.where(detector_mask, normalization, 0.0).ravel()).reshape(
        full_grid.shape
    )
    angle_excluded_signal = float(expected_signal[~angle_mask].sum())
    angle_excluded_normalization = float(expected_normalization[~angle_mask].sum())
    expected_signal[~angle_mask] = 0.0
    expected_normalization[~angle_mask] = 0.0
    expected_valid = angle_mask & (expected_normalization > 0.0)
    expected_intensity = np.zeros(full_grid.shape)
    np.divide(
        expected_signal,
        expected_normalization,
        out=expected_intensity,
        where=expected_valid,
    )

    np.testing.assert_allclose(field.S, expected_signal, rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(field.N, expected_normalization, rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(field.I, expected_intensity, rtol=0.0, atol=3e-14)
    np.testing.assert_array_equal(field.valid, expected_valid)
    assert field.detector_mask_excluded_signal == pytest.approx(7.0)
    assert field.detector_mask_excluded_normalization == pytest.approx(2.0)
    assert field.angle_mask_excluded_signal == pytest.approx(angle_excluded_signal)
    assert field.angle_mask_excluded_normalization == pytest.approx(angle_excluded_normalization)
    assert (
        field.S.sum() + field.angle_mask_excluded_signal + field.angular_lost_signal
        == pytest.approx(signal[detector_mask].sum(), abs=2e-12)
    )
    assert (
        field.N.sum() + field.angle_mask_excluded_normalization + field.angular_lost_normalization
        == pytest.approx(normalization[detector_mask].sum(), abs=2e-12)
    )

    ratio_first = np.zeros_like(signal)
    np.divide(signal, normalization, out=ratio_first, where=normalization > 0.0)
    wrong = (dense @ ratio_first.ravel()).reshape(full_grid.shape)
    assert np.max(np.abs(wrong[expected_valid] - field.I[expected_valid])) > 1e-3

    phi = to_increasing_phi(field)
    assert np.any(full_grid.chi_raw_edges_rad == np.pi / 2.0)
    mapped_phi = (-np.pi / 2.0 - full_grid.chi_raw_centers_rad + np.pi) % (2.0 * np.pi) - np.pi
    permutation = np.argsort(mapped_phi, kind="stable")
    assert np.all(np.diff(phi.grid.phi_centers_rad) > 0.0)
    np.testing.assert_allclose(
        phi.grid.phi_centers_rad,
        mapped_phi[permutation],
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_array_equal(phi.S, field.S[permutation])
    np.testing.assert_array_equal(phi.N, field.N[permutation])
    np.testing.assert_array_equal(phi.I, field.I[permutation])
    np.testing.assert_array_equal(phi.valid, field.valid[permutation])
    np.testing.assert_array_equal(
        phi.angle_bin_valid_mask,
        field.angle_bin_valid_mask[permutation],
    )
    assert phi.projector_cache_key == field.projector_cache_key == projector.cache_key
    assert phi.angular_lost_signal == field.angular_lost_signal
    for array_value in (field.S, phi.valid, projector.weight):
        assert not array_value.flags.writeable

    zero = project_normalized_angle_field(projector, signal, np.zeros_like(normalization))
    assert not np.any(zero.valid)
    assert not np.any(zero.I)
    np.testing.assert_array_equal(zero.S, field.S)
    np.testing.assert_array_equal(zero.N, np.zeros_like(field.N))
    with pytest.raises(ValueError, match="nonnegative"):
        project_normalized_angle_field(projector, -signal, normalization)

    clipped_grid = AngleBinGrid(
        two_theta_edges_rad=np.linspace(
            0.0,
            0.55 * full_grid.two_theta_edges_rad[-1],
            3,
        ),
        chi_raw_edges_rad=full_grid.chi_raw_edges_rad,
        revision="clipped-grid.v1",
    )
    clipped = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=clipped_grid,
    )
    clipped_field = project_normalized_angle_field(clipped, signal, normalization)
    assert np.any(clipped.lost_support > 0.0)
    clipped_dense = _dense_projector(clipped)
    oracle_column = _independent_pixel_weights(
        0,
        0,
        instrument=instrument,
        frame=frame,
        grid=clipped_grid,
    )
    np.testing.assert_allclose(
        clipped_dense[:, 0].reshape(clipped_grid.shape),
        oracle_column,
        rtol=3e-11,
        atol=3e-13,
    )
    assert clipped.lost_support[0, 0] == pytest.approx(
        1.0 - float(oracle_column.sum()),
        abs=3e-12,
    )
    assert float(clipped_dense[:, 0].sum()) < 1.0
    assert clipped_field.S.sum() + clipped_field.angular_lost_signal == pytest.approx(
        signal.sum(), abs=2e-12
    )
    assert clipped_field.N.sum() + clipped_field.angular_lost_normalization == pytest.approx(
        normalization.sum(), abs=2e-12
    )
