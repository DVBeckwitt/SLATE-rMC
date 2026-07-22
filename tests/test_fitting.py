from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import rasim_next.fitting.geometry as fitting_geometry_module
import rasim_next.selection.blind as blind_module
from painted_ewald import MosaicBraggSpace
from painted_ewald.rotations import mosaic_axes
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.fitting import (
    SHARED_GEOMETRY_PARAMETER_NAMES,
    ContinuousDetectorFunction,
    ContinuousDetectorGeometryModel,
    ExactTagGeometryModel,
    GeometryCorrectionBounds,
    GeometryCorrections,
    GeometryPredictionError,
    GeometryRankError,
    IndexedGeometryImage,
    IntegerLMarkerKey,
    IntegerLMarkerObservations,
    IntegerLMarkerPrediction,
    M0IntegerLObservations,
    M0IntegerLPrediction,
    SharedGeometryCorrectionBounds,
    SharedGeometryCorrections,
    apply_shared_geometry_corrections,
    audit_indexed_geometry_series_roots,
    audit_integer_l_marker_selection,
    evaluate_indexed_geometry_series_residual,
    evaluate_tagged_geometry_objective_residual,
    fit_indexed_geometry_series,
    fit_tagged_detector_function_geometry,
)
from rasim_next.geometry import AngleFrame, build_incident_states, detector_coordinates_to_angles
from rasim_next.pipeline.bragg_space import Bi2Se3TwoHStrength
from rasim_next.pipeline.configured_simulation import (
    build_configured_geometry_inputs,
    build_configured_simulation_inputs,
    build_geometry_only_ewald_context,
    build_nominal_ewald_context,
    build_source_averaged_detector,
    evaluate_nominal_integer_l_markers,
    load_simulation_config,
    rebind_configured_geometry_instrument,
    sample_configured_source,
    solve_integer_l_ewald_roots,
)
from rasim_next.pipeline.continuous_detector import DetectorEwaldMeasure
from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure
from rasim_next.selection import (
    BlindIndexingPolicy,
    DiscoveredCakePeak,
    MeasuredPeakDiscovery,
    build_osc_angle_frame,
    index_discovered_integer_l_peaks,
    load_osc_geometry_series,
    reindex_frozen_discovery_coordinates,
    simulation_config_for_osc_image,
)
from rasim_next.selection.blind import _discovery_geometry_hash


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


def _axis_from_pitch_yaw(pitch_rad: float, yaw_rad: float) -> np.ndarray:
    cosine_pitch = math.cos(pitch_rad)
    return np.asarray(
        (
            math.cos(yaw_rad) * cosine_pitch,
            -math.sin(yaw_rad) * cosine_pitch,
            math.sin(pitch_rad),
        )
    )


def _axis_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Independent Rodrigues oracle used only to construct hidden truth."""

    unit = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    cross = np.asarray(
        (
            (0.0, -unit[2], unit[1]),
            (unit[2], 0.0, -unit[0]),
            (-unit[1], unit[0], 0.0),
        )
    )
    return np.eye(3) + math.sin(angle_rad) * cross + (1.0 - math.cos(angle_rad)) * (cross @ cross)


def _shared_truth_instrument(
    geometry_inputs: object,
    truth: SharedGeometryCorrections,
) -> object:
    """Construct the nine-coordinate hidden pose independently of production fitting code."""

    configured = geometry_inputs.config.instrument.axis_rotations
    assert len(configured) == 1
    axis = configured[0]
    base_axis = np.asarray(axis.axis_lab, dtype=np.float64)
    horizontal = math.hypot(base_axis[0], base_axis[1])
    base_pitch = math.atan2(base_axis[2], horizontal)
    base_yaw = math.atan2(-base_axis[1], base_axis[0])
    pivot = np.asarray(axis.pivot_lab_m, dtype=np.float64)
    angle = math.radians(axis.angle_deg)
    corrected_axis = _axis_from_pitch_yaw(
        base_pitch + truth.goniometer_axis_pitch_rad,
        base_yaw + truth.goniometer_axis_yaw_rad,
    )
    lab_up = np.asarray((0.0, 0.0, 1.0))
    pivot_yaw_tangent = -np.cross(lab_up, corrected_axis)
    pivot_yaw_tangent /= np.linalg.norm(pivot_yaw_tangent)
    pivot_pitch_tangent = np.cross(pivot_yaw_tangent, corrected_axis)
    corrected_pivot = (
        pivot
        + truth.goniometer_pivot_pitch_offset_m * pivot_pitch_tangent
        + truth.goniometer_pivot_yaw_offset_m * pivot_yaw_tangent
    )
    assert float(corrected_axis @ (corrected_pivot - pivot)) == pytest.approx(0.0, abs=1e-18)
    base = geometry_inputs.instrument
    base_motion_rotation = _axis_rotation_matrix(base_axis, angle)
    base_motion_translation = pivot - base_motion_rotation @ pivot
    zero_sample_rotation = base_motion_rotation.T @ base.lab_from_sample.rotation
    zero_sample_translation = base_motion_rotation.T @ (
        base.lab_from_sample.translation_m - base_motion_translation
    )
    corrected_motion_rotation = _axis_rotation_matrix(corrected_axis, angle)
    corrected_motion_translation = corrected_pivot - corrected_motion_rotation @ corrected_pivot
    sample_after_axis_rotation = corrected_motion_rotation @ zero_sample_rotation
    sample_after_axis_translation = (
        corrected_motion_rotation @ zero_sample_translation + corrected_motion_translation
    )
    sample_rotation = (
        sample_after_axis_rotation
        @ _rotation_x(truth.sample_normal_x_tilt_rad)
        @ _rotation_y(truth.sample_normal_y_tilt_rad)
    )
    sample_delta_lab = sample_rotation @ sample_after_axis_rotation.T
    sample_translation = corrected_pivot + sample_delta_lab @ (
        sample_after_axis_translation - corrected_pivot
    )
    sample_translation = (
        sample_translation + truth.sample_plane_normal_offset_m * sample_rotation[:, 2]
    )
    detector = base.lab_from_detector
    detector_rotation = (
        detector.rotation
        @ _rotation_x(truth.detector_column_tilt_rad)
        @ _rotation_y(truth.detector_row_tilt_rad)
    )
    return replace(
        base,
        lab_from_detector=RigidTransform(
            detector_rotation,
            detector.translation_m,
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        lab_from_sample=RigidTransform(
            sample_rotation,
            sample_translation,
            FrameId.SAMPLE,
            FrameId.LAB,
        ),
    )


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
    np.testing.assert_allclose(
        reference.instrument.lab_from_detector.rotation,
        truth_inputs.instrument.lab_from_detector.rotation,
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_allclose(
        reference.instrument.lab_from_sample.rotation,
        truth_inputs.instrument.lab_from_sample.rotation,
        rtol=0.0,
        atol=3.0e-16,
    )
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
    geometry_inputs = build_configured_geometry_inputs(config)
    with pytest.raises(ValueError, match="source-center"):
        replace(
            geometry_inputs,
            samples=replace(
                nominal_samples,
                origin_lab_m=nominal_samples.origin_lab_m + np.asarray(((1.0e-6, 0.0, 0.0),)),
            ),
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


def test_three_incidence_hidden_shared_geometry_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    base_config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    truth = SharedGeometryCorrections(
        detector_column_tilt_rad=math.radians(0.25),
        detector_row_tilt_rad=math.radians(-0.45),
        sample_normal_x_tilt_rad=math.radians(0.18),
        sample_normal_y_tilt_rad=math.radians(-0.27),
        goniometer_axis_pitch_rad=math.radians(0.15),
        goniometer_axis_yaw_rad=math.radians(-0.22),
        sample_plane_normal_offset_m=2.0e-5,
        goniometer_pivot_pitch_offset_m=3.0e-5,
        goniometer_pivot_yaw_offset_m=-2.5e-5,
    )
    ell_by_angle = {
        5.0: {4, 5, 8, 10, 11},
        10.0: {4, 5, 8, 10},
        15.0: {5, 8, 10, 11},
    }
    images: list[IndexedGeometryImage] = []
    heldout: dict[str, IntegerLMarkerObservations] = {}
    for angle_deg in (5.0, 10.0, 15.0):
        axis = base_config.instrument.axis_rotations[0]
        config = replace(
            base_config,
            instrument=replace(
                base_config.instrument,
                axis_rotations=(replace(axis, angle_deg=angle_deg),),
            ),
        )
        geometry_inputs = build_configured_geometry_inputs(config)
        rod = next(
            rod for rod in geometry_inputs.rods if rod.family_m == 1 and (rod.h, rod.k) == (-1, 0)
        )
        incident = build_incident_states(
            geometry_inputs.samples,
            geometry_inputs.material,
            geometry_inputs.instrument,
        )
        keys = []
        for integer_l in sorted(ell_by_angle[angle_deg]):
            roots = solve_integer_l_ewald_roots(
                rod=rod,
                integer_l=integer_l,
                reciprocal_basis_Ainv=geometry_inputs.reciprocal.basis_Ainv,
                crystal_to_sample=geometry_inputs.instrument.sample_from_crystal.rotation,
                ki_sample_Ainv=incident.states.k_film_phase_sample_Ainv[0],
            )
            assert roots is not None and roots.branch == 2 and roots.root_sign == (-1, 1)
            keys.extend(
                IntegerLMarkerKey(
                    family_m=1,
                    integer_L=integer_l,
                    branch=roots.branch,
                    root_sign=root_sign,
                    representative_rod_hk=(-1, 0),
                )
                for root_sign in roots.root_sign
            )
        keys = tuple(keys)
        assert len(keys) == 2 * len(ell_by_angle[angle_deg])

        model = ExactTagGeometryModel(geometry_inputs)
        truth_prediction = model.predict_integer_l_tags(
            keys,
            instrument=_shared_truth_instrument(geometry_inputs, truth),
        )
        observations = IntegerLMarkerObservations.from_prediction(
            truth_prediction,
            reference_wavelength_A=model.reference_wavelength_A,
            sigma_px=0.25,
        )
        is_heldout = np.asarray(
            [key.integer_L in {4, 11} for key in observations.keys],
            dtype=np.bool_,
        )
        image_id = f"osc-{int(angle_deg):02d}"
        images.append(
            IndexedGeometryImage(
                image_id=image_id,
                commanded_angle_rad=math.radians(angle_deg),
                model=model,
                observations=observations.subset(~is_heldout),
            )
        )
        heldout[image_id] = observations.subset(is_heldout)

    with pytest.raises(ValueError, match="wavelength"):
        replace(
            images[0],
            observations=replace(
                images[0].observations,
                reference_wavelength_A=images[0].model.reference_wavelength_A + 0.01,
            ),
        )
    mutated_inputs = replace(
        images[0].model.inputs,
        instrument=replace(
            images[0].model.instrument,
            detector_column_pitch_m=1.01 * images[0].model.instrument.detector_column_pitch_m,
        ),
    )
    with pytest.raises(ValueError, match="instrument does not match its declared config"):
        replace(images[0], model=ExactTagGeometryModel(mutated_inputs))

    bounds = SharedGeometryCorrectionBounds.rasim_multi_angle_pose()
    np.testing.assert_allclose(
        bounds.half_span,
        np.asarray(
            (
                math.radians(10.0),
                math.radians(10.0),
                math.radians(5.0),
                math.radians(5.0),
                math.radians(5.0),
                math.radians(5.0),
                1.0e-4,
                1.0e-4,
                1.0e-4,
            )
        ),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(GeometryRankError, match=r"rank=5/9"):
        fit_indexed_geometry_series(
            tuple(images[:1]),
            initial=SharedGeometryCorrections.zero(),
            bounds=bounds,
        )
    with pytest.raises(GeometryRankError, match=r"rank=7/9"):
        fit_indexed_geometry_series(
            tuple(images[:2]),
            initial=SharedGeometryCorrections.zero(),
            bounds=bounds,
        )

    result = fit_indexed_geometry_series(
        tuple(reversed(images)),
        initial=SharedGeometryCorrections.zero(),
        bounds=bounds,
    )
    assert result.success, result.message
    np.testing.assert_array_equal(
        evaluate_indexed_geometry_series_residual(
            tuple(reversed(images)),
            SharedGeometryCorrections.zero(),
        ),
        evaluate_indexed_geometry_series_residual(
            tuple(images),
            SharedGeometryCorrections.zero(),
        ),
    )
    assert result.image_ids == ("osc-05", "osc-10", "osc-15")
    assert result.jacobian_rank == 9
    assert result.jacobian_condition < 15_000.0
    assert not np.any(result.active_bounds)
    np.testing.assert_array_less(
        np.abs(result.corrections.as_array() - truth.as_array()) / bounds.half_span,
        np.full(9, 2.5e-5),
    )
    assert result.training_site_rms_px < 1.0e-3
    assert result.training_site_max_px < 5.0e-3
    assert result.training_chord_angle_rms_rad < 1.0e-7

    fitted_without_detector_tilts = (
        "sample_normal_x_tilt_rad",
        "sample_normal_y_tilt_rad",
        "goniometer_axis_pitch_rad",
        "goniometer_axis_yaw_rad",
        "sample_plane_normal_offset_m",
        "goniometer_pivot_pitch_offset_m",
        "goniometer_pivot_yaw_offset_m",
    )
    fixed_detector_tilts = replace(
        SharedGeometryCorrections.zero(),
        detector_column_tilt_rad=truth.detector_column_tilt_rad,
        detector_row_tilt_rad=truth.detector_row_tilt_rad,
    )
    constrained = fit_indexed_geometry_series(
        tuple(images),
        initial=fixed_detector_tilts,
        bounds=bounds,
        fitted_parameter_names=fitted_without_detector_tilts,
    )
    assert constrained.success, constrained.message
    assert constrained.fitted_parameter_names == fitted_without_detector_tilts
    assert constrained.fixed_parameter_names == (
        "detector_column_tilt_rad",
        "detector_row_tilt_rad",
    )
    assert constrained.jacobian_rank == 7
    assert constrained.scaled_jacobian_singular_values.shape == (7,)
    assert constrained.scaled_jacobian_weakest_direction.shape == (7,)
    assert constrained.active_bounds.shape == (7,)
    assert constrained.corrections.detector_column_tilt_rad == (
        fixed_detector_tilts.detector_column_tilt_rad
    )
    assert constrained.corrections.detector_row_tilt_rad == (
        fixed_detector_tilts.detector_row_tilt_rad
    )
    np.testing.assert_array_less(
        np.abs(constrained.corrections.as_array()[2:] - truth.as_array()[2:])
        / bounds.half_span[2:],
        np.full(7, 2.5e-5),
    )
    with pytest.raises(ValueError, match="unknown shared geometry parameter"):
        fit_indexed_geometry_series(
            tuple(images),
            initial=fixed_detector_tilts,
            bounds=bounds,
            fitted_parameter_names=("not_a_parameter",),
        )

    requested_sparse_names = (
        "goniometer_pivot_yaw_offset_m",
        "detector_column_tilt_rad",
        "sample_plane_normal_offset_m",
    )
    expected_sparse_names = (
        "detector_column_tilt_rad",
        "sample_plane_normal_offset_m",
        "goniometer_pivot_yaw_offset_m",
    )
    sparse_indices = np.asarray(
        [SHARED_GEOMETRY_PARAMETER_NAMES.index(name) for name in expected_sparse_names]
    )
    sparse_initial_values = truth.as_array().copy()
    sparse_initial_values[sparse_indices] = 0.0
    sparse_initial = SharedGeometryCorrections.from_array(sparse_initial_values)
    sparse = fit_indexed_geometry_series(
        tuple(images),
        initial=sparse_initial,
        bounds=bounds,
        fitted_parameter_names=requested_sparse_names,
    )
    assert sparse.fitted_parameter_names == expected_sparse_names
    fixed_indices = np.asarray(
        [
            index
            for index, name in enumerate(SHARED_GEOMETRY_PARAMETER_NAMES)
            if name not in expected_sparse_names
        ]
    )
    np.testing.assert_array_equal(
        sparse.corrections.as_array()[fixed_indices],
        sparse_initial.as_array()[fixed_indices],
    )
    np.testing.assert_array_less(
        np.abs(sparse.corrections.as_array()[sparse_indices] - truth.as_array()[sparse_indices])
        / bounds.half_span[sparse_indices],
        np.full(3, 2.5e-5),
    )
    with pytest.raises(ValueError, match="at least one parameter"):
        fit_indexed_geometry_series(
            tuple(images),
            initial=sparse_initial,
            bounds=bounds,
            fitted_parameter_names=(),
        )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        fit_indexed_geometry_series(
            tuple(images),
            initial=sparse_initial,
            bounds=bounds,
            fitted_parameter_names=(
                "detector_column_tilt_rad",
                "detector_column_tilt_rad",
            ),
        )
    for image, metrics in zip(images, result.per_image, strict=True):
        assert metrics.image_id == image.image_id
        prediction = image.predict_integer_l_tags(
            heldout[image.image_id].keys,
            result.corrections,
        )
        error = prediction.coordinates_px - heldout[image.image_id].coordinates_px
        assert float(np.max(np.linalg.norm(error, axis=1), initial=0.0)) < 1.0e-2

    for image in images:
        fitted = image.corrected_instrument(result.corrections)
        normal = fitted.lab_from_sample.rotation[:, 2]
        zero_offset = image.corrected_instrument(
            replace(result.corrections, sample_plane_normal_offset_m=0.0)
        )
        translation_delta = (
            fitted.lab_from_sample.translation_m - zero_offset.lab_from_sample.translation_m
        )
        assert float(normal @ translation_delta) == pytest.approx(
            result.corrections.sample_plane_normal_offset_m,
            abs=2.0e-18,
        )
        np.testing.assert_allclose(
            translation_delta - normal * float(normal @ translation_delta),
            0.0,
            rtol=0.0,
            atol=2.0e-18,
        )
    root_audit = audit_indexed_geometry_series_roots(tuple(images), result.corrections)
    assert root_audit.classification == "SAME"
    assert all(
        item.audit.expected_count == item.audit.enumerated_count for item in root_audit.images
    )

    real_solver = fitting_geometry_module.solve_integer_l_ewald_roots

    def swapped_beta_solver(**kwargs: object) -> object:
        roots = real_solver(**kwargs)
        if roots is None or len(roots.beta_rad) != 2:
            return roots
        return replace(roots, beta_rad=tuple(reversed(roots.beta_rad)))

    monkeypatch.setattr(
        fitting_geometry_module,
        "solve_integer_l_ewald_roots",
        swapped_beta_solver,
    )
    swapped_audit = audit_indexed_geometry_series_roots(tuple(images), result.corrections)
    assert swapped_audit.classification == "CHANGED"
    assert any(item.audit.classification == "CHANGED" for item in swapped_audit.images)


def test_exact_tag_geometry_context_is_material_generic_and_mosaic_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    template = (root / "configs" / "bi2se3_simulation.yaml").read_text(encoding="utf-8")
    generic_config_path = tmp_path / "pbi2_geometry.yaml"
    generic_config_path.write_text(
        template.replace(
            "../examples/bi2se3/structures/Bi2Se3_vesta.cif",
            (root / "examples" / "pbi2" / "structures" / "PbI2_2H.cif").as_posix(),
        )
        .replace("phase_id: bi2se3", "phase_id: pbi2")
        .replace("model_id: bi2se3_finite_2h.v1", "model_id: geometry_only.unused.v1")
        .replace("normalization: FINITE_TOTAL", "normalization: UNUSED_BY_GEOMETRY")
        .replace("shared_disorder_epsilon: 0.001", "shared_disorder_epsilon: -1.0")
        .replace("gaussian_sigma_deg: 1.0", "gaussian_sigma_deg: 0.0")
        .replace("alpha_panel_count: 8", "alpha_panel_count: 1")
        .replace("alpha_gauss_order: 12", "alpha_gauss_order: 1")
        .replace("azimuth_count: 32", "azimuth_count: 1"),
        encoding="utf-8",
    )
    series_manifest = tmp_path / "pbi2_series.yaml"
    series_manifest.write_text(
        "\n".join(
            (
                "schema_version: rasim-osc-geometry-fit-v1",
                "simulation_config: pbi2_geometry.yaml",
                "incidence_axis_index: 0",
                "images:",
                "  - image_id: pbi2-example",
                "    osc_path: "
                f'"{(root / "examples" / "bi2se3" / "osc" / "Bi2Se3_5m_5d.osc.gz").as_posix()}"',
                "    axis_rotation_angles_deg: [5.0]",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    series = load_osc_geometry_series(series_manifest)
    config = load_simulation_config(series.config_path)
    assert config.structure_factor.model_id == "geometry_only.unused.v1"
    assert config.mosaic.gaussian_sigma_deg == 0.0
    with pytest.raises(ValueError, match=r"model_id.*for intensity"):
        build_configured_simulation_inputs(config)

    def reject_intensity_or_mosaic(*args: object, **kwargs: object) -> None:
        raise AssertionError("exact-tag geometry must not build strength or mosaic space")

    monkeypatch.setattr(Bi2Se3TwoHStrength, "__init__", reject_intensity_or_mosaic)
    monkeypatch.setattr(MosaicBraggSpace, "__init__", reject_intensity_or_mosaic)
    inputs = build_configured_geometry_inputs(config)
    assert not hasattr(inputs, "strength")
    assert not hasattr(inputs, "mosaic")
    assert not hasattr(inputs, "bragg_space")
    changed_axis = replace(config.instrument.axis_rotations[0], angle_deg=10.0)
    rebound = rebind_configured_geometry_instrument(
        inputs,
        replace(config, instrument=replace(config.instrument, axis_rotations=(changed_axis,))),
    )
    assert rebound.crystal is inputs.crystal
    assert rebound.material is inputs.material
    assert rebound.reciprocal is inputs.reciprocal
    assert rebound.rods is inputs.rods
    assert rebound.instrument.sample_geometry_revision != inputs.instrument.sample_geometry_revision
    model = ExactTagGeometryModel(inputs)
    incident = build_incident_states(inputs.samples, inputs.material, inputs.instrument)
    selected_keys: tuple[IntegerLMarkerKey, ...] | None = None
    for rod in inputs.rods:
        if rod.family_m == 0:
            continue
        for integer_l in range(-20, 21):
            roots = solve_integer_l_ewald_roots(
                rod=rod,
                integer_l=integer_l,
                reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
                crystal_to_sample=inputs.instrument.sample_from_crystal.rotation,
                ki_sample_Ainv=incident.states.k_film_phase_sample_Ainv[0],
            )
            if roots is None or roots.root_sign != (-1, 1):
                continue
            keys = tuple(
                IntegerLMarkerKey(
                    family_m=rod.family_m,
                    integer_L=integer_l,
                    branch=roots.branch,
                    root_sign=root_sign,
                    representative_rod_hk=(rod.h, rod.k),
                )
                for root_sign in roots.root_sign
            )
            if np.all(model.predict_integer_l_tags(keys).active_panel):
                selected_keys = keys
                break
        if selected_keys is not None:
            break
    assert selected_keys is not None
    prediction = model.predict_integer_l_tags(selected_keys)
    assert np.all(prediction.active_panel)
    assert {key.family_m for key in prediction.keys} != {0}

    context = build_geometry_only_ewald_context(inputs)
    direct_beam = np.asarray(config.source.mean_direction_lab, dtype=np.float64)
    direct_beam /= np.linalg.norm(direct_beam)
    detector_column_lab = inputs.instrument.lab_from_detector.apply_vector(
        np.asarray((1.0, 0.0, 0.0))
    )
    column_right = detector_column_lab - float(detector_column_lab @ direct_beam) * direct_beam
    column_right /= np.linalg.norm(column_right)
    frame = AngleFrame(
        origin_lab_m=context.incident.states.sample_intersection_lab_m[0],
        row_down_lab=np.cross(direct_beam, column_right),
        column_right_lab=column_right,
        direct_beam_lab=direct_beam,
        revision="pbi2-geometry-only-indexing.v1",
    )
    coordinate = prediction.coordinates_px[0]
    angles = detector_coordinates_to_angles(
        coordinate[0],
        coordinate[1],
        instrument=inputs.instrument,
        angle_frame=frame,
    )
    discovery = MeasuredPeakDiscovery(
        image_id="pbi2-geometry-only",
        detector_shape_rc=inputs.instrument.detector_shape_rc,
        peaks=(
            DiscoveredCakePeak(
                column_px=float(coordinate[0]),
                row_px=float(coordinate[1]),
                two_theta_rad=float(angles.two_theta_rad),
                phi_rad=float(angles.phi_rad),
                covariance_px2=((0.04, 0.0), (0.0, 0.04)),
                localization_covariance_px2=((0.01, 0.0), (0.0, 0.01)),
                z_score=20.0,
            ),
        ),
        detector_data_hash="sha256-" + "1" * 64,
        detector_mask_hash="sha256-" + "2" * 64,
        detector_mask_revision="synthetic-all-valid.v1",
        geometry_context_hash=_discovery_geometry_hash(inputs.instrument, frame),
        policy=BlindIndexingPolicy(),
    )
    indexed = index_discovered_integer_l_peaks(
        discovery,
        ewald_context=context,
        angle_frame=frame,
        incidence_angle_rad=math.radians(config.instrument.axis_rotations[0].angle_deg),
    )
    assert len(indexed.marker_decisions) == 1
    assert indexed.marker_decisions[0].status.value == "VISIBLE_CONFIDENT"

    frozen_observations = IntegerLMarkerObservations(
        keys=(indexed.marker_decisions[0].key,),
        coordinates_px=np.asarray((coordinate,)),
        covariance_px2=np.asarray((((0.04, 0.0), (0.0, 0.04)),)),
        reference_wavelength_A=model.reference_wavelength_A,
    )

    def reject_global_discovery(*args: object, **kwargs: object) -> None:
        raise AssertionError("frozen-coordinate reindexing must not rerun global discovery")

    monkeypatch.setattr(
        blind_module,
        "discover_measured_cake_peaks",
        reject_global_discovery,
    )
    reindexed = reindex_frozen_discovery_coordinates(
        discovery,
        frozen_observations=frozen_observations,
        ewald_context=context,
        angle_frame=frame,
        incidence_angle_rad=math.radians(config.instrument.axis_rotations[0].angle_deg),
    )
    assert tuple(item.key for item in reindexed.marker_decisions) == (
        indexed.marker_decisions[0].key,
    )
    np.testing.assert_array_equal(
        np.asarray(
            (
                reindexed.marker_decisions[0].observed_column_px,
                reindexed.marker_decisions[0].observed_row_px,
            )
        ),
        coordinate,
    )
    assert reindexed.context_hash != indexed.context_hash

    changed_mosaic = replace(
        config,
        mosaic=replace(
            config.mosaic,
            gaussian_sigma_deg=0.5,
            alpha_panel_count=1,
            alpha_gauss_order=1,
            azimuth_count=1,
        ),
    )
    changed_prediction = ExactTagGeometryModel(
        build_configured_geometry_inputs(changed_mosaic)
    ).predict_integer_l_tags(selected_keys)
    np.testing.assert_array_equal(
        changed_prediction.coordinates_px,
        prediction.coordinates_px,
    )


def test_osc_geometry_series_manifest_owns_ids_paths_and_commanded_angles() -> None:
    root = Path(__file__).resolve().parents[1]
    series = load_osc_geometry_series(root / "configs" / "bi2se3_osc_geometry_fit.yaml")
    assert tuple(image.image_id for image in series.images) == (
        "Bi2Se3_5m_5d",
        "Bi2Se3_10d_5m",
        "Bi2Se3_15d_5m",
    )
    assert tuple(image.axis_rotation_angles_deg for image in series.images) == (
        (5.0,),
        (10.0,),
        (15.0,),
    )
    assert all(image.osc_path.is_file() for image in series.images)
    with pytest.raises(ValueError, match="numeric scalars"):
        replace(series.images[0], axis_rotation_angles_deg=(True,))
    with pytest.raises(ValueError, match="numeric scalars"):
        replace(series.images[0], axis_rotation_angles_deg=("5.0",))
    repeated = replace(
        series,
        images=(
            series.images[0],
            replace(series.images[1], axis_rotation_angles_deg=(5.0,)),
            series.images[2],
        ),
    )
    assert (
        repeated.images[0].axis_rotation_angles_deg == repeated.images[1].axis_rotation_angles_deg
    )
    with pytest.raises(ValueError, match="IDs and paths must be unique"):
        replace(
            series,
            images=(
                series.images[0],
                replace(series.images[1], image_id=series.images[0].image_id),
                series.images[2],
            ),
        )

    base = load_simulation_config(series.config_path)
    image_config = simulation_config_for_osc_image(base, series.images[0])
    geometry_inputs = build_configured_geometry_inputs(image_config)
    correction = replace(
        SharedGeometryCorrections.zero(),
        detector_column_tilt_rad=math.radians(1.0),
        detector_row_tilt_rad=math.radians(-0.5),
        sample_plane_normal_offset_m=5.0e-5,
    )
    corrected = apply_shared_geometry_corrections(
        geometry_inputs.instrument,
        image_config.instrument.axis_rotations,
        correction,
    )
    context = build_geometry_only_ewald_context(geometry_inputs, instrument=corrected)
    frame = build_osc_angle_frame(
        mean_direction_lab=geometry_inputs.config.source.mean_direction_lab,
        instrument=context.instrument,
        sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
        revision="osc-geometry-angle-frame.offset-origin-proof.v1",
    )
    np.testing.assert_array_equal(
        frame.origin_lab_m,
        context.incident.states.sample_intersection_lab_m[0],
    )
    assert not np.array_equal(frame.origin_lab_m, corrected.lab_from_sample.translation_m)
    direct_beam = np.asarray(image_config.source.mean_direction_lab, dtype=np.float64)
    direct_beam /= np.linalg.norm(direct_beam)
    expected_column = corrected.lab_from_detector.apply_vector(np.asarray((1.0, 0.0, 0.0)))
    expected_column -= float(expected_column @ direct_beam) * direct_beam
    expected_column /= np.linalg.norm(expected_column)
    np.testing.assert_allclose(frame.column_right_lab, expected_column, rtol=0.0, atol=1.0e-15)
