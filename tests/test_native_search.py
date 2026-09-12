"""Distinct statistical contracts: nuisance refits, calibration ownership and held-out noise."""

from dataclasses import replace

import numpy as np
import pytest

from rasim_next.fitting.native_observations import NativeFitObservations
from rasim_next.fitting.native_search import (
    FitParameter,
    GaussianCalibration,
    conditional_validation,
    fit_native_parameters,
    profile_native_parameter,
    training_observations,
)
from rasim_next.measurement.continuous_regions import NativePixelRegionProjection


def observations(counts, covariance=None):
    counts = np.array(counts, dtype=float)
    n = len(counts)
    projection = NativePixelRegionProjection(
        (1, n), np.arange(n), np.arange(n), np.arange(n), np.ones(n), n, "statistical-proof"
    )
    return NativeFitObservations(
        projection,
        counts,
        np.ones(n, dtype=bool),
        np.eye(n) if covariance is None else covariance,
        np.eye(n),
        counts,
        np.eye(n),
        np.array([0, n]),
        np.array([1e6]),
        "known-observations",
    )


def test_numerical_gates_detect_profile_offsets_and_held_out_only_error():
    from rasim_next.fitting.native_accuracy import (
        compare_conditional_predictions,
        compare_native_predictions,
    )

    obs = observations([2, 3, 5, 7], np.eye(4) + np.full((4, 4), 0.2))
    train = np.array([True, True, False, False])
    target = training_observations(obs, train)
    low = np.array([[1, 2, 3, 4], [1, 2.1, 3, 4]], dtype=float)
    high = low.copy()
    high[:, 2:] += 1
    limits = dict(
        fixed_scale=1.0,
        maximum_whitened_rms=0.1,
        maximum_contrast_rms=0.05,
        maximum_objective_contrast_error=0.5,
    )
    assert compare_native_predictions(target, low, high, **limits)["empirical_agreement"]
    scale = target.profile_scale(low[0])[0]
    validation = compare_conditional_predictions(
        obs,
        scale * low[0],
        scale * high[0],
        train,
        ~train,
        maximum_whitened_rms=0.1,
        maximum_objective_error=0.5,
    )
    assert not validation["empirical_agreement"]
    assert validation["whitened_rms"] > 0.1
    # A bias at only one refitted profile center cannot inherit a baseline pass.
    high = low.copy()
    high[1, 0] += 4
    profile = compare_native_predictions(target, low, high, **limits)
    assert not profile["empirical_agreement"]
    assert abs(profile["objective_contrast_error"][0]) > 0.5


def test_profile_refits_scale_and_every_nuisance_to_analytic_optimum():
    obs = observations([1, 3, 1])
    parameters = (
        FitParameter("theta", "1", "sample", 0.5, 3, 1),
        FitParameter("nu", "1", "sample", 0, 3, 1, "physical"),
    )
    seen = []

    def predict(values):
        seen.append(values.copy())
        theta, nu = values
        return np.array([1, theta + nu, theta - nu])

    grid = [1.0, 1.7, 2.0, 2.5]
    result = profile_native_parameter(
        {3: predict},
        obs,
        parameters,
        [[1.0, 0.2], [2.5, 2.0]],
        name="theta",
        grid=grid,
        maximum_iterations=100,
        finite_difference_step=1e-6,
    )
    assert set(np.array(seen)[:, 0]) == set(grid)
    assert result["resolved"].all()
    for i, theta in enumerate(grid):
        scale = (1 + 4 * theta) / (1 + 2 * theta**2)
        nuisance = 1 / scale
        expected = scale * np.array([1, theta + nuisance, theta - nuisance]) - obs.net_count
        point = result["fits"][3][i].best_converged
        assert point.scale == pytest.approx(scale, abs=2e-5)
        assert point.parameter_values[1] == pytest.approx(nuisance, abs=2e-5)
        assert result["objective"][0, i] == pytest.approx(expected @ expected, abs=2e-8)
    assert result["interval_status"] == "raw_profile_requires_threshold_calibration"


def test_training_guards_cannot_leak_and_correlated_validation_is_conditional():
    obs = observations([1, 100])
    obs = replace(
        obs,
        guard_operator=np.array([[0.0, 1.0]]),
        guard_pointer=np.array([0, 1]),
        guard_limit=np.array([0.1]),
    )
    train = training_observations(obs, np.array([True, False]))
    assert train.input_revision != obs.input_revision
    assert train.profile_scale(np.ones(2))[0] == pytest.approx(1)
    with pytest.raises(ValueError, match="diagnostic"):
        train.profile_scale(np.ones(2), enforce_guards=True)
    with pytest.raises(ValueError, match="diagnostic"):
        fit_native_parameters(
            lambda x: np.ones(2),
            train,
            (FitParameter("x", "1", "sample", 0, 2, 1),),
            [[1]],
            enforce_historical_guards=True,
        )
    covariance = np.array([[1.0, 0.5], [0.5, 2.0]])
    obs = observations([2, 3], covariance)
    result = conditional_validation(
        obs,
        np.zeros(2),
        np.array([True, False]),
        np.array([False, True]),
        groups={"held_out": np.array([False, True])},
    )
    assert result["conditional_mean_count"][0] == pytest.approx(1)
    assert result["covariance_count2"][0, 0] == pytest.approx(7 / 4)
    assert result["chi_square"] == pytest.approx(16 / 7)
    assert result["groups"][0]["chi_square"] == pytest.approx(16 / 7)
    with pytest.raises(ValueError, match="partition"):
        conditional_validation(
            obs,
            np.zeros(2),
            np.array([True, False]),
            np.array([False, True]),
            groups={"a": np.array([False, True]), "b": np.array([False, True])},
        )


def test_calibration_binding_boundary_kinds_and_incomplete_minima_are_explicit():
    obs = observations([1, 2])
    parameters = (FitParameter("x", "m", "beam-session", 0, 2, 1, "physical", "search"),)
    block = GaussianCalibration(
        (0,),
        np.array([1.0]),
        np.eye(1),
        "a" * 64,
        "beam-session",
        "independent_measurement",
        ("x",),
        ("m",),
        ("beam-session",),
    )
    with pytest.raises(ValueError, match="counted twice"):
        fit_native_parameters(
            lambda x: np.ones(2), obs, parameters, [[1]], calibration=(block, block)
        )
    with pytest.raises(ValueError, match="metadata"):
        fit_native_parameters(
            lambda x: np.ones(2),
            obs,
            (replace(parameters[0], name="different"),),
            [[1]],
            calibration=(block,),
        )
    result = fit_native_parameters(
        lambda x: np.array([1, 2 + x[0]]),
        obs,
        parameters,
        [[1]],
        maximum_iterations=100,
        finite_difference_step=1e-6,
    )
    point = result.best_converged
    assert point.parameter_values[0] == pytest.approx(0, abs=1e-5)
    assert point.physical_boundary_parameters == ("x",)
    assert result.best_evaluated.physical_boundary_parameters == ("x",)
    assert not point.search_bound_parameters
    assert result.numerical_status == "not_qualified"
    fixed = fit_native_parameters(
        lambda x: np.ones(2),
        obs,
        parameters,
        [[1]],
        fixed_values={"x": 0},
        calibration=(replace(block, evidence_kind="assumption"),),
    )
    assert fixed.best_converged.calibration_chi_square == 0
    assert fixed.best_converged.assumption_chi_square == pytest.approx(1)
    assert fixed.best_evaluated.physical_boundary_parameters == ("x",)
    assert fixed.best_converged.physical_boundary_parameters == ("x",)
    # One start terminates at a stationary maximum; another finds a lower value
    # before a one-iteration limit. A successful status alone cannot resolve it.
    ambiguous = fit_native_parameters(
        lambda x: np.array([1, 3 - x[0] ** 4]),
        obs,
        parameters,
        [[0], [0.5]],
        maximum_iterations=1,
        finite_difference_step=1e-6,
    )
    assert ambiguous.best_converged is not None
    assert ambiguous.best_evaluated.objective < ambiguous.best_converged.objective
    assert not ambiguous.minimum_resolved
    guarded_obs = replace(
        obs, guard_pointer=np.array([0, 1, 2]), guard_limit=np.array([0.01, 0.01])
    )
    guarded = fit_native_parameters(
        lambda x: np.array([1.0, x[0]]),
        guarded_obs,
        parameters,
        [[1], [2]],
        calibration=(replace(block, covariance=np.array([[0.01]]), evidence_kind="assumption"),),
        enforce_historical_guards=True,
        maximum_iterations=100,
        finite_difference_step=1e-6,
    )
    assert guarded.best_evaluated.objective < guarded.best_converged.objective
    assert not guarded.best_evaluated.scores["guards_pass"]
    assert guarded.minimum_resolved


def test_numerical_gate_detects_changed_guard_feasibility_and_control_noise_uses_measurement():
    from rasim_next.fitting.native_accuracy import compare_native_predictions, native_sensitivity
    from rasim_next.fitting.native_observations import NativeBackgroundControl

    obs = replace(
        observations([1, 1]), guard_pointer=np.array([0, 1, 2]), guard_limit=np.array([0.1, 0.1])
    )
    low, high = np.array([1.0, 1.22]), np.array([1.0, 1.23])
    assert not obs.scores(obs.profile_scale(low)[0] * low)["guards_pass"]
    assert not obs.scores(obs.profile_scale(high)[0] * high)["guards_pass"]
    assert obs.guard_scale_interval(low) is not None
    assert obs.guard_scale_interval(high) is None
    result = compare_native_predictions(
        obs,
        np.array([low, 2 * low]),
        np.array([high, 2 * high]),
        fixed_scale=1.0,
        maximum_whitened_rms=0.02,
        maximum_contrast_rms=0.01,
        maximum_objective_contrast_error=1e-10,
    )
    assert not result["empirical_agreement"]
    assert result["guard_scale_feasibility_changed"].all()
    training = training_observations(obs, np.array([True, False]))
    training_result = compare_native_predictions(
        training,
        np.array([low, 2 * low]),
        np.array([high, 2 * high]),
        fixed_scale=1.0,
        maximum_whitened_rms=0.02,
        maximum_contrast_rms=0.01,
        maximum_objective_contrast_error=1e-10,
    )
    assert training_result["empirical_agreement"]
    assert not training_result["guard_diagnostics_included"]
    assert training_result["objective_observation_revision"] == training.input_revision
    narrow = native_sensitivity(
        lambda v: np.array([1.0, 1.0 + v[0]]),
        obs,
        (FitParameter("x", "1", "specimen", 0, 1e-4, 1),),
        np.array([0.0]),
    )
    assert np.isfinite(narrow["jacobian"]).all()
    assert np.linalg.norm(narrow["jacobian"]) > 0
    controls = NativeBackgroundControl(
        obs.projection, np.array([1.0, 4.0]), np.array([0, 2]), "control"
    )
    diagnostic = controls.signal_diagnostic(np.array([2.0, 4.0]))
    np.testing.assert_array_equal(diagnostic["signal_measurement_sigma"], [2.0, 2.0])
    assert diagnostic["split_summary"][2]["squared_signal_norm"] == 4.0
