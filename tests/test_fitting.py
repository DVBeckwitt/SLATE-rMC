from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from painted_ewald.rotations import mosaic_axes
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.fitting import (
    ContinuousDetectorFunction,
    ContinuousDetectorGeometryModel,
    GeometryCorrectionBounds,
    GeometryCorrections,
    GeometryPredictionError,
    GeometryRankError,
    IntegerLMarkerKey,
    IntegerLMarkerObservations,
    IntegerLMarkerPrediction,
    M0IntegerLObservations,
    M0IntegerLPrediction,
    audit_integer_l_marker_selection,
    evaluate_tagged_geometry_objective_residual,
    fit_tagged_detector_function_geometry,
)
from rasim_next.geometry import build_incident_states
from rasim_next.pipeline.configured_simulation import (
    build_configured_simulation_inputs,
    build_nominal_ewald_context,
    build_source_averaged_detector,
    evaluate_nominal_integer_l_markers,
    load_simulation_config,
    sample_configured_source,
)
from rasim_next.pipeline.continuous_detector import DetectorEwaldMeasure
from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure


def _rotation_x(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))


def _rotation_y(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))


def _truth_inputs(
    base_inputs: object,
    truth: GeometryCorrections,
    *,
    sample_correction_pivot_lab_m: object,
) -> object:
    """Construct hidden geometry independently of the fit model."""

    instrument = base_inputs.instrument
    detector = instrument.lab_from_detector
    sample = instrument.lab_from_sample
    detector_rotation = (
        detector.rotation
        @ _rotation_x(truth.detector_column_tilt_rad)
        @ _rotation_y(truth.detector_row_tilt_rad)
    )
    sample_rotation = (
        sample.rotation
        @ _rotation_x(truth.sample_normal_x_tilt_rad)
        @ _rotation_y(truth.sample_normal_y_tilt_rad)
    )
    sample_pivot_lab_m = np.asarray(sample_correction_pivot_lab_m, dtype=np.float64)
    sample_delta_lab = sample_rotation @ sample.rotation.T
    sample_translation_m = sample_pivot_lab_m + sample_delta_lab @ (
        sample.translation_m - sample_pivot_lab_m
    )
    truth_instrument = replace(
        instrument,
        lab_from_detector=RigidTransform(
            detector_rotation,
            detector.translation_m,
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        lab_from_sample=RigidTransform(
            sample_rotation,
            sample_translation_m,
            FrameId.SAMPLE,
            FrameId.LAB,
        ),
    )
    return replace(base_inputs, instrument=truth_instrument)


def _truth_field_inputs(
    base_inputs: object,
    truth: GeometryCorrections,
    *,
    sample_correction_pivot_lab_m: object,
) -> object:
    """Construct hidden geometry and its incident transport independently of the fit model."""

    transformed = _truth_inputs(
        base_inputs,
        truth,
        sample_correction_pivot_lab_m=sample_correction_pivot_lab_m,
    )
    return replace(
        transformed,
        incident=build_incident_states(
            base_inputs.samples,
            base_inputs.material,
            transformed.instrument,
        ),
    )


def _raise_if_pixelized(*args: object, **kwargs: object) -> None:
    raise AssertionError("continuous-field fitting must not call a pixel integrator")


def test_continuous_detector_geometry_prediction_matches_fresh_nonpixel_oracle() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    shared_pivot_lab_m = (0.003, -0.001, 0.002)
    sample_mount = config.instrument.goniometer_from_sample
    config = replace(
        config,
        source=replace(config.source, sample_count=8),
        numerics=replace(config.numerics, worker_count=4),
        instrument=replace(
            config.instrument,
            axis_rotations=tuple(
                replace(rotation, pivot_lab_m=shared_pivot_lab_m)
                for rotation in config.instrument.axis_rotations
            ),
            goniometer_from_sample=replace(
                sample_mount,
                translation_m=(0.001, 0.0004, -0.0008),
            ),
        ),
    )
    base_inputs = build_configured_simulation_inputs(config)
    truth = GeometryCorrections.from_array(np.radians((0.17, -0.23, 0.11, -0.14)))
    truth_inputs = _truth_field_inputs(
        base_inputs,
        truth,
        sample_correction_pivot_lab_m=shared_pivot_lab_m,
    )
    assert not np.allclose(
        truth_inputs.instrument.lab_from_sample.translation_m,
        base_inputs.instrument.lab_from_sample.translation_m,
        rtol=0.0,
        atol=1.0e-12,
    )
    direct_detector = build_source_averaged_detector(truth_inputs)
    column_px = np.asarray((131.137, 722.283, 1104.417, 1818.639, 2387.811))
    row_px = np.asarray((83.219, 621.137, 1208.319, 1711.773, 1996.427))
    expected = direct_detector.evaluate_detector_coordinates_all_roots(column_px, row_px)
    mismatched_pose = _truth_inputs(
        base_inputs,
        GeometryCorrections.from_array(np.radians((-0.31, 0.19, -0.08, 0.16))),
        sample_correction_pivot_lab_m=shared_pivot_lab_m,
    )
    with pytest.raises(ValueError, match="share one pose"):
        direct_detector.rebind_geometry(
            incident=truth_inputs.incident,
            instrument=mismatched_pose.instrument,
        )

    model = ContinuousDetectorGeometryModel(base_inputs)
    reference = model.bind(truth)
    actual = reference.evaluate_detector_coordinates(column_px, row_px)

    np.testing.assert_allclose(
        actual.per_rod_density_A2_per_px2,
        expected.per_rod_density_A2_per_px2,
        rtol=2.0e-13,
        atol=0.0,
    )
    np.testing.assert_array_equal(actual.caustic, expected.caustic)
    np.testing.assert_array_equal(actual.valid_source_count, expected.valid_source_count)
    truth_observations = IntegerLMarkerObservations.from_markers(
        evaluate_nominal_integer_l_markers(build_nominal_ewald_context(truth_inputs))
    )
    truth_tag_prediction = reference.predict_integer_l_tags(truth_observations.keys)
    np.testing.assert_allclose(
        truth_tag_prediction.coordinates_px,
        truth_observations.coordinates_px,
        rtol=0.0,
        atol=5.0e-11,
    )

    no_axis_inputs = build_configured_simulation_inputs(
        replace(
            config,
            instrument=replace(config.instrument, axis_rotations=()),
        )
    )
    with pytest.raises(ValueError, match="common configured goniometer pivot"):
        ContinuousDetectorGeometryModel(no_axis_inputs)

    distinct_pivot_axis = replace(
        config.instrument.axis_rotations[0],
        angle_deg=0.0,
        pivot_lab_m=(0.004, -0.001, 0.002),
    )
    ambiguous_inputs = build_configured_simulation_inputs(
        replace(
            config,
            source=replace(config.source, sample_count=1),
            instrument=replace(
                config.instrument,
                axis_rotations=(*config.instrument.axis_rotations, distinct_pivot_axis),
            ),
        )
    )
    with pytest.raises(ValueError, match="common configured goniometer pivot"):
        ContinuousDetectorGeometryModel(ambiguous_inputs)
    explicit_pivot_model = ContinuousDetectorGeometryModel(
        ambiguous_inputs,
        sample_correction_pivot_lab_m=shared_pivot_lab_m,
    )
    np.testing.assert_array_equal(
        explicit_pivot_model.sample_correction_pivot_lab_m,
        shared_pivot_lab_m,
    )
    assert not explicit_pivot_model.sample_correction_pivot_lab_m.flags.writeable
    explicit_expected = build_source_averaged_detector(
        _truth_field_inputs(
            ambiguous_inputs,
            truth,
            sample_correction_pivot_lab_m=shared_pivot_lab_m,
        )
    ).evaluate_detector_coordinates_all_roots(column_px, row_px)
    explicit_actual = explicit_pivot_model.bind(truth).evaluate_detector_coordinates(
        column_px,
        row_px,
    )
    np.testing.assert_allclose(
        explicit_actual.per_rod_density_A2_per_px2,
        explicit_expected.per_rod_density_A2_per_px2,
        rtol=2.0e-12,
        atol=0.0,
    )


def test_tag_identity_rejects_duplicates_and_mismatched_pairs() -> None:
    with pytest.raises(ValueError, match="invalid non-specular integer-L marker identity"):
        IntegerLMarkerKey(1, 2, 2, 0, (1, 0))
    negative = IntegerLMarkerKey(1, 2, 2, -1, (1, 0))
    positive = IntegerLMarkerKey(1, 2, 2, 1, (1, 0))
    duplicate_user_identity = replace(negative, representative_rod_hk=(0, 1))
    with pytest.raises(ValueError, match=r"\(m,L,tag_branch\)"):
        IntegerLMarkerPrediction(
            keys=(negative, duplicate_user_identity),
            coordinates_px=np.zeros((2, 2)),
            detector_status=np.asarray(("VALID", "VALID")),
            ewald_residual_Ainv=np.zeros(2),
        )
    for changed, message in (
        (replace(positive, representative_rod_hk=(0, 1)), "physical rod"),
        (replace(positive, branch=1), "Ewald branch"),
    ):
        with pytest.raises(ValueError, match=message):
            IntegerLMarkerObservations(
                keys=(negative, changed),
                coordinates_px=np.zeros((2, 2)),
                covariance_px2=np.broadcast_to(np.eye(2), (2, 2, 2)),
                reference_wavelength_A=1.54,
            )


def test_nominal_tag_companion_is_single_and_source_count_invariant() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    assert config.source.sample_count == 1000
    inputs_1000 = build_configured_simulation_inputs(config)
    inputs_1 = build_configured_simulation_inputs(
        replace(config, source=replace(config.source, sample_count=1))
    )
    nominal_samples = sample_configured_source(config.source, sample_count=1)
    np.testing.assert_array_equal(
        nominal_samples.origin_lab_m,
        np.asarray((config.source.mean_origin_lab_m,)),
    )
    np.testing.assert_array_equal(
        nominal_samples.direction_lab,
        np.asarray((config.source.mean_direction_lab,)),
    )
    np.testing.assert_array_equal(
        nominal_samples.wavelength_A,
        np.asarray((config.source.mean_wavelength_A,)),
    )
    empirical_nominal = (
        np.all(inputs_1000.samples.origin_lab_m == nominal_samples.origin_lab_m[0], axis=1)
        & np.all(
            inputs_1000.samples.direction_lab == nominal_samples.direction_lab[0],
            axis=1,
        )
        & (inputs_1000.samples.wavelength_A == nominal_samples.wavelength_A[0])
    )
    assert not np.any(empirical_nominal)

    context_1000 = build_nominal_ewald_context(inputs_1000)
    context_1 = build_nominal_ewald_context(inputs_1)
    assert bool(context_1000.incident.states.valid[0])
    markers_1000 = evaluate_nominal_integer_l_markers(context_1000)
    markers_1 = evaluate_nominal_integer_l_markers(context_1)
    for name in (
        "family_m",
        "integer_L",
        "branch",
        "root_sign",
        "column_px",
        "row_px",
        "q_sample_Ainv",
        "ewald_residual_Ainv",
        "family_strength_weight_A2",
    ):
        np.testing.assert_array_equal(getattr(markers_1000, name), getattr(markers_1, name))
    for name in (
        "contributing_rod_hk",
        "contributing_beta_rad",
        "per_rod_strength_weight_A2",
        "reference_wavelength_A",
        "definition_id",
        "source_state_policy",
    ):
        assert getattr(markers_1000, name) == getattr(markers_1, name)
    observations = IntegerLMarkerObservations.from_markers(markers_1000)
    detector_function = ContinuousDetectorGeometryModel(inputs_1000).bind(
        GeometryCorrections.zero()
    )
    predicted = detector_function.predict_integer_l_tags(observations.keys)
    np.testing.assert_allclose(
        predicted.coordinates_px,
        observations.coordinates_px,
        rtol=0.0,
        atol=5.0e-11,
    )
    assert np.all(predicted.active_panel)
    assert detector_function.source_state_count == 1000
    assert (
        detector_function.tag_incident_state_policy
        == "nominal_source_center.zero_divergence.mean_wavelength.v1"
    )
    assert not detector_function.tag_incident_state_contributes_to_intensity
    assert {key.tag_branch for key in observations.keys} == {1, 2}
    assert len({(key.family_m, key.integer_L, key.tag_branch) for key in observations.keys}) == len(
        observations.keys
    )
    m0 = detector_function.predict_m0_minimum_tilt_exact_l_landmarks((2,))
    assert m0.tag_branch == 0


def test_tagged_detector_objective_retains_independent_line_angle_terms() -> None:
    keys = (
        IntegerLMarkerKey(1, 2, 2, -1, (1, 0)),
        IntegerLMarkerKey(1, 2, 2, 1, (1, 0)),
    )
    observations = IntegerLMarkerObservations(
        keys=keys,
        coordinates_px=np.asarray(((0.0, 0.0), (4.0, 0.0))),
        covariance_px2=np.broadcast_to(np.eye(2), (2, 2, 2)),
        reference_wavelength_A=1.54,
    )
    chord_angle = math.radians(60.0)
    prediction = IntegerLMarkerPrediction(
        keys=keys,
        coordinates_px=np.asarray(
            ((0.0, 0.0), (4.0 * math.cos(chord_angle), 4.0 * math.sin(chord_angle)))
        ),
        detector_status=np.asarray(("VALID", "VALID")),
        ewald_residual_Ainv=np.zeros(2),
    )

    integer_l = (2, 3, 4)
    m0_target = np.asarray(((0.0, -2.0), (0.0, 0.0), (0.0, 2.0)))
    m0_observations = M0IntegerLObservations(
        integer_L=integer_l,
        coordinates_px=m0_target,
        covariance_px2=np.broadcast_to(np.eye(2), (3, 2, 2)),
        reference_wavelength_A=1.54,
    )
    m0_angle = math.radians(30.0)
    increasing_l_direction = np.asarray((-math.sin(m0_angle), math.cos(m0_angle)))
    m0_trial = np.asarray((-2.0, 0.0, 2.0))[:, None] * increasing_l_direction
    m0_prediction = M0IntegerLPrediction(
        integer_L=integer_l,
        coordinates_px=m0_trial,
        alpha_rad=np.zeros(3),
        beta_rad=np.zeros(3),
        detector_status=np.asarray(("VALID", "VALID", "VALID")),
        ewald_residual_Ainv=np.zeros(3),
        reference_wavelength_A=1.54,
    )

    residual = evaluate_tagged_geometry_objective_residual(
        observations,
        prediction,
        m0_observations=m0_observations,
        m0_prediction=m0_prediction,
    )

    assert residual.shape == (12,)
    assert residual[4] == pytest.approx(4.0 * math.sin(0.5 * chord_angle))
    assert residual[-1] == pytest.approx(4.0 * math.sin(0.5 * m0_angle))
    with pytest.raises(ValueError, match="wavelength"):
        evaluate_tagged_geometry_objective_residual(
            observations,
            prediction,
            m0_observations=m0_observations,
            m0_prediction=replace(m0_prediction, reference_wavelength_A=1.55),
        )


def test_m0_minimum_tilt_landmarks_obey_independent_ewald_oracle() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    config = replace(config, source=replace(config.source, sample_count=1))
    inputs = build_configured_simulation_inputs(config)
    context = build_nominal_ewald_context(inputs)
    prediction = (
        ContinuousDetectorGeometryModel(inputs)
        .bind(GeometryCorrections.zero())
        .predict_m0_minimum_tilt_exact_l_landmarks(tuple(range(1, 21)))
    )

    assert prediction.tag_branch == 0
    np.testing.assert_array_equal(
        prediction.active_panel,
        np.asarray((False, *(True for _ in range(18)), False)),
    )
    assert prediction.detector_status[0] == "BACKWARD"
    assert prediction.detector_status[-1] == "OUTSIDE_SUPPORT"
    basis = inputs.reciprocal.basis_Ainv
    mean_axis_crystal, tilt_axis_crystal = mosaic_axes(basis)
    reference_axis_crystal = np.cross(tilt_axis_crystal, mean_axis_crystal)
    alpha = prediction.alpha_rad[1:19]
    beta = prediction.beta_rad[1:19]
    direction_crystal = np.cos(alpha)[:, None] * mean_axis_crystal + np.sin(alpha)[:, None] * (
        np.cos(beta)[:, None] * reference_axis_crystal + np.sin(beta)[:, None] * tilt_axis_crystal
    )
    crystal_to_sample = inputs.instrument.sample_from_crystal.rotation
    direction_sample = direction_crystal @ crystal_to_sample.T
    ki_sample = context.geometry.coating.ki_sample_Ainv
    k_norm = float(np.linalg.norm(ki_sample))
    incident_direction = ki_sample / k_norm
    integer_l = np.arange(2.0, 20.0)
    u_Ainv = integer_l * float(np.linalg.norm(basis[:, 2]))
    q_sample = u_Ainv[:, None] * direction_sample
    np.testing.assert_allclose(
        np.linalg.norm(ki_sample + q_sample, axis=1),
        k_norm,
        rtol=0.0,
        atol=2.0e-13,
    )

    z = -u_Ainv / (2.0 * k_norm)
    mean_axis_sample = crystal_to_sample @ mean_axis_crystal
    perpendicular = mean_axis_sample - (mean_axis_sample @ incident_direction) * incident_direction
    maximum_alignment = z * float(mean_axis_sample @ incident_direction) + np.sqrt(
        1.0 - z**2
    ) * float(np.linalg.norm(perpendicular))
    np.testing.assert_allclose(
        direction_sample @ mean_axis_sample,
        maximum_alignment,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_blind_integer_l_geometry_fit_recovers_ra_sim_bounded_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    config = replace(config, source=replace(config.source, sample_count=1))
    base_inputs = build_configured_simulation_inputs(config)
    truth = GeometryCorrections(
        detector_column_tilt_rad=math.radians(0.25),
        detector_row_tilt_rad=math.radians(-0.50),
        sample_normal_x_tilt_rad=math.radians(0.20),
        sample_normal_y_tilt_rad=math.radians(-0.30),
    )
    truth_markers = evaluate_nominal_integer_l_markers(
        build_nominal_ewald_context(
            _truth_inputs(
                base_inputs,
                truth,
                sample_correction_pivot_lab_m=config.instrument.axis_rotations[0].pivot_lab_m,
            )
        )
    )
    observations = IntegerLMarkerObservations.from_markers(truth_markers, sigma_px=0.25)

    heldout_labels = {
        (1, 2),
        (1, 9),
        (1, 16),
        (3, 2),
        (3, 8),
        (3, 15),
        (4, 2),
        (4, 8),
        (4, 14),
    }
    heldout_mask = np.asarray(
        [(key.family_m, key.integer_L) in heldout_labels for key in observations.keys]
    )
    training = observations.subset(~heldout_mask)
    heldout = observations.subset(heldout_mask)
    assert len(training.keys) == 66
    assert len(heldout.keys) == 18

    bounds = GeometryCorrectionBounds.rasim_reduced_pose()
    np.testing.assert_allclose(
        np.degrees(bounds.lower.as_array()),
        (-10.0, -10.0, -5.0, -5.0),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.degrees(bounds.upper.as_array()),
        (10.0, 10.0, 5.0, 5.0),
        rtol=0.0,
        atol=1.0e-14,
    )

    monkeypatch.setattr(
        DetectorEwaldMeasure,
        "integrate_native_pixels",
        _raise_if_pixelized,
    )
    monkeypatch.setattr(
        SourceAveragedDetectorEwaldMeasure,
        "integrate_native_pixels",
        _raise_if_pixelized,
    )
    field_model = ContinuousDetectorGeometryModel(base_inputs)
    reference_function = field_model.bind(truth)
    truth_prediction = reference_function.predict_integer_l_tags(observations.keys)
    tagged_field_value = reference_function(
        truth_prediction.coordinates_px[:, 0],
        truth_prediction.coordinates_px[:, 1],
    )
    rod_index = {(rod.h, rod.k): index for index, rod in enumerate(tagged_field_value.rods)}
    representative_density = np.asarray(
        [
            tagged_field_value.per_rod_density_A2_per_px2[
                index,
                rod_index[key.representative_rod_hk],
            ]
            for index, key in enumerate(observations.keys)
        ]
    )
    assert np.all(tagged_field_value.valid_source_count == 1)
    assert np.all(representative_density > 0.0)
    truth_m0_prediction = reference_function.predict_m0_minimum_tilt_exact_l_landmarks(
        tuple(range(2, 20)),
    )
    m0_field_value = reference_function(
        truth_m0_prediction.coordinates_px[:, 0],
        truth_m0_prediction.coordinates_px[:, 1],
    )
    m0_index = next(index for index, rod in enumerate(m0_field_value.rods) if rod.family_m == 0)
    assert np.all(m0_field_value.per_rod_density_A2_per_px2[:, m0_index] > 0.0)
    initial = GeometryCorrections.zero()

    outside_start = None
    for candidate in (
        GeometryCorrections(math.radians(10.0), 0.0, 0.0, 0.0),
        GeometryCorrections(math.radians(-10.0), 0.0, 0.0, 0.0),
        GeometryCorrections(0.0, math.radians(10.0), 0.0, 0.0),
        GeometryCorrections(0.0, math.radians(-10.0), 0.0, 0.0),
    ):
        boundary_prediction = field_model.bind(candidate).predict_integer_l_tags(training.keys)
        if np.any(boundary_prediction.detector_status == "OUTSIDE_SUPPORT"):
            outside_start = candidate
            assert not np.all(boundary_prediction.active_panel)
            break
    assert outside_start is not None
    with pytest.raises(GeometryPredictionError, match="topology changed"):
        fit_tagged_detector_function_geometry(
            field_model,
            reference_function,
            nonzero_keys=training.keys,
            m0_integer_L=tuple(range(2, 20)),
            initial=outside_start,
            bounds=bounds,
        )

    original_bind = ContinuousDetectorGeometryModel.bind
    bound_trial_corrections: list[GeometryCorrections] = []

    def bind_trial_function(
        self: ContinuousDetectorGeometryModel,
        corrections: GeometryCorrections,
    ) -> ContinuousDetectorFunction:
        bound_trial_corrections.append(corrections)
        return original_bind(self, corrections)

    monkeypatch.setattr(ContinuousDetectorGeometryModel, "bind", bind_trial_function)
    result = fit_tagged_detector_function_geometry(
        field_model,
        reference_function,
        nonzero_keys=training.keys,
        m0_integer_L=tuple(range(2, 20)),
        sigma_px=0.25,
        initial=initial,
        bounds=bounds,
    )
    assert len(bound_trial_corrections) == result.model_evaluation_count
    assert result.success, result.message
    assert result.parameterization_id == "detector_xy_plus_pivoted_effective_sample_normal_xy.v2"
    assert result.jacobian_rank == 4
    assert result.jacobian_condition < 100.0
    np.testing.assert_allclose(
        result.corrections.as_array(),
        truth.as_array(),
        rtol=0.0,
        atol=5.0e-7,
    )
    assert result.training_site_rms_px < 1.0e-4
    assert result.training_site_max_px < 5.0e-4
    assert result.training_chord_angle_rms_rad < 1.0e-8
    assert result.training_m0_line_angle_rad < 1.0e-8
    assert result.chord_count == 33
    assert result.m0_landmark_count == 18

    fitted_function = field_model.bind(result.corrections)
    heldout_prediction = fitted_function.predict_integer_l_tags(heldout.keys)
    heldout_error = heldout_prediction.coordinates_px - heldout.coordinates_px
    assert float(np.max(np.abs(heldout_error))) < 5.0e-3

    audit = audit_integer_l_marker_selection(
        fitted_function,
        observations.keys,
    )
    assert audit.classification == "SAME"
    assert audit.expected_count == audit.enumerated_count == 84
    with pytest.raises(GeometryRankError, match="rank"):
        fit_tagged_detector_function_geometry(
            field_model,
            reference_function,
            nonzero_keys=(training.keys[0],),
            initial=initial,
            bounds=bounds,
        )
