from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.fitting import (
    GeometryCorrectionBounds,
    GeometryCorrections,
    GeometryPredictionError,
    GeometryRankError,
    IntegerLGeometryModel,
    IntegerLMarkerKey,
    IntegerLMarkerObservations,
    IntegerLMarkerPrediction,
    IntegerLSelectionAudit,
    audit_integer_l_marker_selection,
    fit_integer_l_marker_geometry,
)
from rasim_next.pipeline.configured_simulation import (
    build_configured_simulation_inputs,
    build_nominal_ewald_context,
    evaluate_nominal_integer_l_markers,
    load_simulation_config,
)


def _rotation_x(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))


def _rotation_y(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))


def _truth_inputs(base_inputs: object, truth: GeometryCorrections) -> object:
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
            sample.translation_m,
            FrameId.SAMPLE,
            FrameId.LAB,
        ),
    )
    return replace(base_inputs, instrument=truth_instrument)


def test_geometry_fit_public_records_reject_impossible_states() -> None:
    key = IntegerLMarkerKey(1, 2, 2, 1, (1, 0))
    with pytest.raises(ValueError, match="unique"):
        IntegerLMarkerPrediction(
            keys=(key, key),
            coordinates_px=np.zeros((2, 2)),
            detector_status=np.asarray(("VALID", "VALID")),
            ewald_residual_Ainv=np.zeros(2),
        )
    with pytest.raises(ValueError, match="unsupported detector prediction status"):
        IntegerLMarkerPrediction(
            keys=(key,),
            coordinates_px=np.zeros((1, 2)),
            detector_status=np.asarray(("FABRICATED",)),
            ewald_residual_Ainv=np.zeros(1),
        )
    with pytest.raises(ValueError, match="SAME audit"):
        IntegerLSelectionAudit(
            classification="SAME",
            missing_keys=(key,),
            unexpected_keys=(),
            expected_count=1,
            enumerated_count=0,
        )


def test_blind_integer_l_geometry_fit_recovers_ra_sim_bounded_pose() -> None:
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
        build_nominal_ewald_context(_truth_inputs(base_inputs, truth))
    )
    tangent_sign = truth_markers.root_sign.copy()
    tangent_sign[0] = 0
    tangent_markers = replace(truth_markers, root_sign=tangent_sign)
    with pytest.raises(ValueError, match="invalid non-specular integer-L marker identity"):
        IntegerLMarkerObservations.from_markers(tangent_markers, selection=np.asarray([0]))
    observations = IntegerLMarkerObservations.from_markers(truth_markers, sigma_px=0.25)

    assert len(observations.keys) == 84
    assert len(set(observations.keys)) == 84
    assert {key.root_sign for key in observations.keys} == {-1, 1}
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

    model = IntegerLGeometryModel(base_inputs)
    initial = GeometryCorrections.zero()
    initial_prediction = model.predict(training.keys, initial)
    initial_error = initial_prediction.coordinates_px - training.coordinates_px
    assert float(np.sqrt(np.mean(initial_error**2))) > 5.0

    outside_start = None
    for candidate in (
        GeometryCorrections(math.radians(10.0), 0.0, 0.0, 0.0),
        GeometryCorrections(math.radians(-10.0), 0.0, 0.0, 0.0),
        GeometryCorrections(0.0, math.radians(10.0), 0.0, 0.0),
        GeometryCorrections(0.0, math.radians(-10.0), 0.0, 0.0),
    ):
        boundary_prediction = model.predict(training.keys, candidate)
        if np.any(boundary_prediction.detector_status == "OUTSIDE_SUPPORT"):
            outside_start = candidate
            assert not np.all(boundary_prediction.active_panel)
            break
    assert outside_start is not None
    with pytest.raises(GeometryPredictionError, match="topology changed"):
        fit_integer_l_marker_geometry(
            model,
            training,
            initial=outside_start,
            bounds=bounds,
        )

    result = fit_integer_l_marker_geometry(
        model,
        training,
        initial=initial,
        bounds=bounds,
    )
    assert result.success, result.message
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

    heldout_prediction = model.predict(heldout.keys, result.corrections)
    heldout_error = heldout_prediction.coordinates_px - heldout.coordinates_px
    assert float(np.sqrt(np.mean(heldout_error**2))) < 1.0e-3
    assert float(np.max(np.abs(heldout_error))) < 5.0e-3

    audit = audit_integer_l_marker_selection(
        model,
        result.corrections,
        observations.keys,
    )
    assert audit.classification == "SAME"
    assert audit.missing_keys == ()
    assert audit.unexpected_keys == ()

    first_key = training.keys[0]
    degenerate_keys = tuple(
        IntegerLMarkerKey(
            family_m=first_key.family_m,
            integer_L=first_key.integer_L,
            branch=first_key.branch,
            root_sign=first_key.root_sign,
            representative_rod_hk=rod_hk,
        )
        for rod_hk in truth_markers.contributing_rod_hk[0][:4]
    )
    degenerate_observations = IntegerLMarkerObservations(
        keys=degenerate_keys,
        coordinates_px=np.broadcast_to(training.coordinates_px[0], (4, 2)),
        covariance_px2=np.broadcast_to(np.eye(2), (4, 2, 2)),
        reference_wavelength_A=training.reference_wavelength_A,
    )
    with pytest.raises(GeometryRankError, match="rank"):
        fit_integer_l_marker_geometry(
            model,
            degenerate_observations,
            initial=initial,
            bounds=bounds,
        )
