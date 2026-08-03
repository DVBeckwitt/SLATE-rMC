from __future__ import annotations

import math

import numpy as np
import pytest

from rasim_next.fitting import (
    FixedMatchedRegionBackground,
    IntegratedPeakAreaProjection,
    MatchedRegionObservations,
    RadialBackgroundProfiles,
    RadialBackgroundState,
    condition_matched_region_background_from_anchors,
    fit_matched_regions,
    fit_shared_radial_background,
    profile_matched_region_nuisance,
)


def _synthetic_observations() -> tuple[
    MatchedRegionObservations,
    FixedMatchedRegionBackground,
    np.ndarray,
    np.ndarray,
]:
    families = (0, 1, 3, 4)
    dataset_ids = ("first", "second", "third")
    dataset: list[int] = []
    block: list[int] = []
    family: list[int] = []
    background: list[bool] = []
    support: list[float] = []
    coordinate: list[float] = []
    features: list[tuple[float, float, float, float]] = []
    block_id = 0
    for dataset_index in range(3):
        for family_index, family_m in enumerate(families):
            for axial_index in range(2):
                signal_feature = (
                    float(family_m == 1),
                    float(family_m == 3),
                    float(family_m == 4),
                    (family_index + 1.0) * (axial_index + 1.0) / 8.0,
                )
                for is_background, position in ((False, 0.0), (True, -1.0), (True, 1.0)):
                    dataset.append(dataset_index)
                    block.append(block_id)
                    family.append(-1 if is_background else family_m)
                    background.append(is_background)
                    support.append(2.0)
                    coordinate.append(position)
                    features.append((0.0, 0.0, 0.0, 0.0) if is_background else signal_feature)
                block_id += 1
    feature_array = np.asarray(features, dtype=np.float64)
    truth = np.asarray((0.18, -0.12, 0.09, 0.24), dtype=np.float64)
    unscaled_model = np.exp(feature_array @ truth)
    unscaled_model[np.asarray(background)] = 0.0
    scales = np.asarray((2.0, 3.0, 4.0))
    count = np.empty(len(dataset), dtype=np.float64)
    for index in range(count.size):
        intercept = 20.0 + 0.05 * block[index]
        slope = 0.3 - 0.02 * dataset[index]
        count[index] = scales[dataset[index]] * unscaled_model[index] + support[index] * (
            intercept + slope * coordinate[index]
        )
    observations = MatchedRegionObservations(
        dataset_ids=dataset_ids,
        dataset_index=dataset,
        block_index=block,
        signal_family=family,
        is_background=background,
        count_mass=count,
        support_px2=support,
        background_coordinate=coordinate,
        required_signal_families=families,
    )
    fixed_background = FixedMatchedRegionBackground(
        count_mass=count - scales[np.asarray(dataset)] * unscaled_model,
        covariance_count2=np.zeros((count.size, count.size)),
        revision="synthetic-fixed-background.v1",
    )
    return observations, fixed_background, feature_array, truth


def test_fixed_background_preserves_signed_dark_correction() -> None:
    correction = np.asarray((-2.0, 1.0, -0.5))
    background = FixedMatchedRegionBackground(
        count_mass=correction,
        covariance_count2=np.eye(3),
        revision="signed-dark-corrected-background.v1",
    )

    np.testing.assert_array_equal(background.count_mass, correction)


def test_joint_matched_region_fit_recovers_one_vector_with_dataset_scales() -> None:
    observations, fixed_background, features, truth = _synthetic_observations()

    result = fit_matched_regions(
        observations,
        lambda parameters: (
            np.exp(features @ parameters) * (~np.asarray(observations.is_background))
        ),
        fixed_background=fixed_background,
        parameter_names=("p0", "p1", "p2", "p3"),
        initial_parameters=(np.zeros(4),),
        lower_bounds=np.full(4, -1.0),
        upper_bounds=np.full(4, 1.0),
    )

    np.testing.assert_allclose(result.parameters, truth, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(result.dataset_scales, (2.0, 3.0, 4.0), rtol=0.0, atol=2.0e-8)
    assert result.sensitivity_rank == 4
    assert result.success
    np.testing.assert_array_equal(result.fitted_signal_row, ~observations.is_background)
    np.testing.assert_allclose(
        result.weighted_residual[observations.is_background],
        0.0,
        rtol=0.0,
        atol=0.0,
    )


def test_integrated_peak_area_fit_sums_conditioned_bins_with_one_scale_per_dataset() -> None:
    observations, fixed_background, features, truth = _synthetic_observations()
    signal = ~np.asarray(observations.is_background)
    signal_dataset = np.asarray(observations.dataset_index)[signal]
    signal_family = np.asarray(observations.signal_family)[signal]
    families = tuple(observations.required_signal_families)
    peak_index = np.asarray(
        [dataset * len(families) + families.index(int(family)) for dataset, family in zip(
            signal_dataset,
            signal_family,
            strict=True,
        )],
        dtype=np.int64,
    )
    projection = IntegratedPeakAreaProjection(
        peak_ids=tuple(f"d{dataset}:m{family}" for dataset in range(3) for family in families),
        peak_dataset_index=np.repeat(np.arange(3), len(families)),
        peak_signal_family=np.tile(families, 3),
        source_signal_peak_index=peak_index,
        revision="synthetic-integrated-peak-areas.v1",
    )

    result = fit_matched_regions(
        observations,
        lambda parameters: np.exp(features @ parameters) * signal,
        fixed_background=fixed_background,
        peak_area_projection=projection,
        parameter_names=("p0", "p1", "p2", "p3"),
        initial_parameters=(truth,),
        lower_bounds=np.full(4, -1.0),
        upper_bounds=np.full(4, 1.0),
    )

    np.testing.assert_allclose(result.parameters, truth, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(result.dataset_scales, (2.0, 3.0, 4.0), rtol=0.0, atol=2.0e-8)
    assert result.weighted_residual.shape == (12,)
    np.testing.assert_array_equal(result.fitted_signal_row, np.ones(12, dtype=np.bool_))
    np.testing.assert_array_equal(result.objective_dataset_index, projection.peak_dataset_index)
    np.testing.assert_array_equal(result.objective_signal_family, projection.peak_signal_family)


def test_integrated_peak_covariance_and_residual_are_partition_invariant() -> None:
    signal_covariance = np.asarray(
        (
            (4.0, 0.8, 0.3, 0.1),
            (0.8, 3.0, 0.2, 0.4),
            (0.3, 0.2, 5.0, 1.1),
            (0.1, 0.4, 1.1, 4.0),
        )
    )
    split_covariance = np.zeros((6, 6), dtype=np.float64)
    split_covariance[:4, :4] = signal_covariance
    split_covariance[4:, 4:] = np.diag((7.0, 8.0))
    split_count = np.asarray((8.0, 5.0, 10.0, 4.0, 20.0, 21.0))
    split_model = np.asarray((1.0, 2.0, 2.0, 1.0, 0.0, 0.0))
    split = MatchedRegionObservations(
        dataset_ids=("image",),
        dataset_index=np.zeros(6, dtype=np.int64),
        block_index=np.zeros(6, dtype=np.int64),
        signal_family=(0, 0, 1, 1, -1, -1),
        is_background=(False, False, False, False, True, True),
        count_mass=split_count,
        support_px2=np.ones(6),
        background_coordinate=(0.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        required_signal_families=(0, 1),
        count_covariance_count2=split_covariance,
    )
    background = FixedMatchedRegionBackground(
        count_mass=np.zeros(6),
        covariance_count2=np.zeros((6, 6)),
        revision="zero-background.v1",
    )
    projection = IntegratedPeakAreaProjection(
        peak_ids=("m0", "m1"),
        peak_dataset_index=(0, 0),
        peak_signal_family=(0, 1),
        source_signal_peak_index=(0, 0, 1, 1),
        revision="partition-invariance.v1",
    )
    aggregation = projection.aggregation_matrix(split)
    expected_count = aggregation @ split_count[:4]
    expected_model = aggregation @ split_model[:4]
    expected_covariance = aggregation @ signal_covariance @ aggregation.T
    root = np.linalg.cholesky(expected_covariance)
    whitened_count = np.linalg.solve(root, expected_count)
    whitened_model = np.linalg.solve(root, expected_model)
    expected_scale = float(whitened_model @ whitened_count) / float(
        whitened_model @ whitened_model
    )
    expected_residual = whitened_count - expected_scale * whitened_model

    split_residual, split_scale, _ = profile_matched_region_nuisance(
        split_model,
        split,
        background,
        peak_area_projection=projection,
    )
    assert np.linalg.norm(expected_residual) > 1.0e-3
    np.testing.assert_allclose(split_scale, (expected_scale,), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(split_residual, expected_residual, rtol=0.0, atol=1.0e-14)

    unsplit_covariance = np.zeros((4, 4), dtype=np.float64)
    unsplit_covariance[:2, :2] = expected_covariance
    unsplit_covariance[2:, 2:] = np.diag((7.0, 8.0))
    unsplit = MatchedRegionObservations(
        dataset_ids=("image",),
        dataset_index=np.zeros(4, dtype=np.int64),
        block_index=np.zeros(4, dtype=np.int64),
        signal_family=(0, 1, -1, -1),
        is_background=(False, False, True, True),
        count_mass=(*expected_count, 20.0, 21.0),
        support_px2=(2.0, 2.0, 1.0, 1.0),
        background_coordinate=(0.0, 0.0, -1.0, 1.0),
        required_signal_families=(0, 1),
        count_covariance_count2=unsplit_covariance,
    )
    unsplit_background = FixedMatchedRegionBackground(
        count_mass=np.zeros(4),
        covariance_count2=np.zeros((4, 4)),
        revision="zero-background-unsplit.v1",
    )
    unsplit_residual, unsplit_scale, _ = profile_matched_region_nuisance(
        np.asarray((*expected_model, 0.0, 0.0)),
        unsplit,
        unsplit_background,
    )
    np.testing.assert_allclose(unsplit_scale, split_scale, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        unsplit_residual[:2],
        split_residual,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_matched_region_observations_reject_incomplete_or_missing_family_data() -> None:
    observations, _, _, _ = _synthetic_observations()
    values = {
        "dataset_ids": observations.dataset_ids,
        "dataset_index": observations.dataset_index,
        "block_index": observations.block_index,
        "signal_family": observations.signal_family,
        "is_background": observations.is_background,
        "count_mass": observations.count_mass,
        "support_px2": observations.support_px2,
        "background_coordinate": observations.background_coordinate,
        "required_signal_families": observations.required_signal_families,
    }
    keep = np.arange(observations.count_mass.size) != 1
    with pytest.raises(ValueError, match="exactly two background anchors"):
        MatchedRegionObservations(
            **{
                name: value if name in {"dataset_ids", "required_signal_families"} else value[keep]
                for name, value in values.items()
            }
        )

    family = np.array(observations.signal_family, copy=True)
    family[family == 4] = 3
    with pytest.raises(ValueError, match="lack signal families"):
        MatchedRegionObservations(**{**values, "signal_family": family})


def test_gaussian_regularization_is_reported_separately_and_cannot_create_data_rank() -> None:
    observations, fixed_background, features, _ = _synthetic_observations()
    rank_deficient_features = features.copy()
    rank_deficient_features[:, 1] = rank_deficient_features[:, 0]

    result = fit_matched_regions(
        observations,
        lambda parameters: (
            np.exp(rank_deficient_features @ parameters) * (~np.asarray(observations.is_background))
        ),
        fixed_background=fixed_background,
        parameter_names=("p0", "p1", "p2", "p3"),
        initial_parameters=(np.zeros(4),),
        lower_bounds=np.full(4, -1.0),
        upper_bounds=np.full(4, 1.0),
        parameter_scales=np.asarray((0.2, 0.2, 0.3, 0.4)),
        prior_residual=lambda parameters: parameters / 0.5,
    )

    assert result.sensitivity_rank == 3
    assert result.penalized_sensitivity_rank == 4
    assert result.objective_half_chi_squared == pytest.approx(
        result.data_objective_half_chi_squared + result.prior_objective_half_chi_squared
    )
    assert result.prior_weighted_residual.shape == (4,)
    assert result.parameter_correlation.shape == (4, 4)
    assert np.all(np.isnan(result.parameter_correlation))


def test_parameter_scaled_data_sensitivity_is_invariant_to_parameter_units() -> None:
    observations, fixed_background, features, _ = _synthetic_observations()
    reference = fit_matched_regions(
        observations,
        lambda parameters: (
            np.exp(features @ parameters) * (~np.asarray(observations.is_background))
        ),
        fixed_background=fixed_background,
        parameter_names=("p0", "p1", "p2", "p3"),
        initial_parameters=(np.zeros(4),),
        lower_bounds=np.full(4, -1.0),
        upper_bounds=np.full(4, 1.0),
        parameter_scales=np.ones(4),
    )
    rescaled_features = np.array(features, copy=True)
    rescaled_features[:, 0] *= 1.0e3
    lower = np.full(4, -1.0)
    upper = np.full(4, 1.0)
    lower[0] = -1.0e-3
    upper[0] = 1.0e-3
    rescaled = fit_matched_regions(
        observations,
        lambda parameters: (
            np.exp(rescaled_features @ parameters) * (~np.asarray(observations.is_background))
        ),
        fixed_background=fixed_background,
        parameter_names=("p0_milli", "p1", "p2", "p3"),
        initial_parameters=(np.zeros(4),),
        lower_bounds=lower,
        upper_bounds=upper,
        parameter_scales=np.asarray((1.0e-3, 1.0, 1.0, 1.0)),
    )

    np.testing.assert_allclose(
        rescaled.jacobian_singular_values,
        reference.jacobian_singular_values,
        rtol=2.0e-6,
        atol=1.0e-10,
    )
    assert rescaled.sensitivity_condition == pytest.approx(
        reference.sensitivity_condition,
        rel=2.0e-6,
    )


def test_practical_relative_tolerance_rejects_nearly_degenerate_data_direction() -> None:
    observations, fixed_background, features, _ = _synthetic_observations()
    nearly_degenerate = np.array(features, copy=True)
    nearly_degenerate[:, 1] = features[:, 0] + 1.0e-10 * features[:, 1]

    result = fit_matched_regions(
        observations,
        lambda parameters: (
            np.exp(nearly_degenerate @ parameters) * (~np.asarray(observations.is_background))
        ),
        fixed_background=fixed_background,
        parameter_names=("p0", "p1", "p2", "p3"),
        initial_parameters=(np.zeros(4),),
        lower_bounds=np.full(4, -1.0),
        upper_bounds=np.full(4, 1.0),
        parameter_scales=np.ones(4),
        sensitivity_relative_tolerance=1.0e-8,
    )

    assert result.sensitivity_rank == 3
    assert math.isinf(result.sensitivity_condition)
    assert result.sensitivity_relative_tolerance == 1.0e-8


def test_adjacent_anchors_condition_the_fixed_background_and_propagate_uncertainty() -> None:
    support = np.asarray((2.0, 4.0, 5.0, 2.0))
    coordinate = np.asarray((0.0, 1.0, 2.0, 3.0))
    is_background = np.asarray((True, False, False, True))
    baseline_density = 10.0
    residual_density = 2.0 + 3.0 * coordinate
    signal_density = np.asarray((0.0, 20.0, 7.0, 0.0))
    count = support * (baseline_density + residual_density + signal_density)
    count_covariance = np.diag(count)
    count_covariance[0, 1] = count_covariance[1, 0] = 3.0
    count_covariance[1, 2] = count_covariance[2, 1] = 2.0
    count_covariance[2, 3] = count_covariance[3, 2] = 4.0
    observations = MatchedRegionObservations(
        dataset_ids=("image",),
        dataset_index=np.zeros(4, dtype=np.int64),
        block_index=np.zeros(4, dtype=np.int64),
        signal_family=(-1, 3, 4, -1),
        is_background=is_background,
        count_mass=count,
        support_px2=support,
        background_coordinate=coordinate,
        required_signal_families=(3, 4),
        count_covariance_count2=count_covariance,
    )
    baseline = FixedMatchedRegionBackground(
        count_mass=support * baseline_density,
        covariance_count2=np.diag((1.0, 4.0, 9.0, 16.0)),
        revision="radial.v1",
    )

    conditioned = condition_matched_region_background_from_anchors(
        observations,
        baseline,
    )

    np.testing.assert_allclose(
        conditioned.count_mass,
        support * (baseline_density + residual_density),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        conditioned.covariance_count2,
        (
            lambda interpolation, selector: (
                (np.eye(4) - interpolation @ selector)
                @ baseline.covariance_count2
                @ (np.eye(4) - interpolation @ selector).T
                + (interpolation @ selector) @ count_covariance @ (interpolation @ selector).T
            )
        )(
            np.linalg.solve(
                np.column_stack((support, support * coordinate))[[0, 3]].T,
                np.column_stack((support, support * coordinate)).T,
            ).T,
            np.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
        ),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.min(np.linalg.eigvalsh(conditioned.covariance_count2)) >= -1.0e-10
    assert conditioned.covariance_count2[1, 2] != 0.0
    assert conditioned.revision.startswith("sha256-")

    model_mass = np.asarray((1.0, 30.0, 10.0, 2.0))
    signal = ~is_background
    signal_index = np.flatnonzero(signal)
    conditioned_model = (
        (np.eye(count.size) - np.asarray(conditioned.anchor_projection)) @ model_mass
    )[signal]
    residual_transform = np.eye(count.size) - np.asarray(conditioned.anchor_projection)
    signal_covariance = (
        residual_transform @ (count_covariance + baseline.covariance_count2) @ residual_transform.T
    )[np.ix_(signal_index, signal_index)]
    covariance_root = np.linalg.cholesky(signal_covariance)
    whitened_count = np.linalg.solve(
        covariance_root,
        (count - conditioned.count_mass)[signal],
    )
    whitened_model = np.linalg.solve(covariance_root, conditioned_model)
    expected_scale = max(
        float(whitened_model @ whitened_count) / float(whitened_model @ whitened_model),
        0.0,
    )
    expected_residual = whitened_count - expected_scale * whitened_model

    residual, scales, background = profile_matched_region_nuisance(
        model_mass,
        observations,
        conditioned,
    )

    assert signal_covariance[0, 1] != 0.0
    assert np.linalg.norm(expected_residual) > 1.0e-3
    np.testing.assert_allclose(scales, (expected_scale,), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        residual[signal],
        expected_residual,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(residual[is_background], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(background, conditioned.count_mass)

    raw_residual = (count - conditioned.count_mass)[signal] - expected_scale * conditioned_model
    assert float(expected_residual @ expected_residual) == pytest.approx(
        float(raw_residual @ np.linalg.solve(signal_covariance, raw_residual)),
        rel=1.0e-14,
        abs=1.0e-14,
    )

    extrapolated = MatchedRegionObservations(
        dataset_ids=("image",),
        dataset_index=np.zeros(4, dtype=np.int64),
        block_index=np.zeros(4, dtype=np.int64),
        signal_family=(-1, 3, 4, -1),
        is_background=is_background,
        count_mass=count,
        support_px2=support,
        background_coordinate=(0.0, 1.0, 4.0, 3.0),
        required_signal_families=(3, 4),
    )
    with pytest.raises(ValueError, match="bracket"):
        condition_matched_region_background_from_anchors(extrapolated, baseline)


def test_conditioned_background_removes_the_same_anchor_tail_from_the_model() -> None:
    support = np.asarray((2.0, 3.0, 4.0, 5.0))
    coordinate = np.asarray((-1.0, -0.25, 0.5, 1.0))
    is_background = np.asarray((True, False, False, True))
    feature = np.asarray((0.2, 1.0, -0.5, 0.7))
    truth = np.asarray((0.3,))
    true_scale = 3.2
    true_model = np.exp(feature * truth[0])
    true_background = support * (20.0 + 2.0 * coordinate)
    observations = MatchedRegionObservations(
        dataset_ids=("image",),
        dataset_index=np.zeros(4, dtype=np.int64),
        block_index=np.zeros(4, dtype=np.int64),
        signal_family=(-1, 0, 1, -1),
        is_background=is_background,
        count_mass=true_scale * true_model + true_background,
        support_px2=support,
        background_coordinate=coordinate,
        required_signal_families=(0, 1),
    )
    conditioned = condition_matched_region_background_from_anchors(
        observations,
        FixedMatchedRegionBackground(
            count_mass=true_background,
            covariance_count2=np.zeros((4, 4)),
            revision="known-affine-background.v1",
        ),
    )

    result = fit_matched_regions(
        observations,
        lambda parameters: np.exp(feature * parameters[0]),
        parameter_names=("shape",),
        initial_parameters=(np.zeros(1),),
        lower_bounds=np.asarray((-1.0,)),
        upper_bounds=np.asarray((1.0,)),
        fixed_background=conditioned,
    )

    np.testing.assert_allclose(result.parameters, truth, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(result.dataset_scales, (true_scale,), rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(
        observations.count_mass
        - result.fitted_background_mass
        - result.fitted_objective_model_mass,
        0.0,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        result.fitted_model_mass,
        true_scale * true_model,
        rtol=0.0,
        atol=1.0e-8,
    )


def test_shared_rise_decay_radial_background_recovers_common_shape() -> None:
    dataset_ids = ("five", "ten", "fifteen")
    radius = np.tile(np.linspace(50.0, 950.0, 91), 12 * len(dataset_ids))
    sector = np.repeat(np.tile(np.arange(12), len(dataset_ids)), 91)
    dataset = np.repeat(np.arange(len(dataset_ids)), 12 * 91)
    training = sector % 4 != 0
    truth = RadialBackgroundState(
        dataset_ids=dataset_ids,
        inner_scale_px=103.0,
        inner_power=1.8,
        outer_scale_px=427.0,
        outer_power=1.5,
        amplitude_count_per_px=(39.0, 57.0, 41.0),
        pedestal_count_per_px=(13.4, 13.8, 13.3),
        parameter_covariance=np.zeros((10, 10)),
    )
    density = truth.count_density(dataset, radius)
    profiles = RadialBackgroundProfiles(
        dataset_ids=dataset_ids,
        dataset_index=dataset,
        radius_px=radius,
        azimuth_sector_index=sector,
        density_count_per_px=density,
        support_px2=np.full(radius.size, 100.0),
        is_training=training,
    )

    result = fit_shared_radial_background(profiles)

    assert result.success
    np.testing.assert_allclose(result.state.parameter_vector, truth.parameter_vector, rtol=2e-6)
    assert result.heldout_rmse_count_per_px < 1.0e-8


def test_radial_background_revision_binds_parameters_and_covariance() -> None:
    parameters = np.asarray((100.0, 1.5, 400.0, 1.2, 20.0, 5.0))
    covariance = np.eye(6)
    reference = RadialBackgroundState.from_parameter_vector(
        ("image",),
        parameters,
        parameter_covariance=covariance,
    )
    changed_parameters = RadialBackgroundState.from_parameter_vector(
        ("image",),
        parameters + np.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        parameter_covariance=covariance,
    )
    changed_covariance = RadialBackgroundState.from_parameter_vector(
        ("image",),
        parameters,
        parameter_covariance=2.0 * covariance,
    )

    assert reference.revision != changed_parameters.revision
    assert reference.revision != changed_covariance.revision
