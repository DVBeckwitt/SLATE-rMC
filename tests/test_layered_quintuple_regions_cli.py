from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import math
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rasim_next.fitting import (
    FixedMatchedRegionBackground,
    FixedPositionState,
    MatchedRegionObservations,
    RadialBackgroundState,
    SharedGeometryCorrections,
)
from rasim_next.measurement import ContinuousRegionQuadrature

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fit_layered_quintuple_regions.py"
SPEC = importlib.util.spec_from_file_location("fit_layered_quintuple_regions_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)

STRUCTURE_LOWER_BOUNDS = np.asarray((-0.01, -0.01, 0.0, 0.0, 0.0))
STRUCTURE_UPPER_BOUNDS = np.asarray((0.01, 0.01, 0.03, 0.02, 0.02))
STRUCTURE_PARAMETER_SCALES = np.asarray((0.001, 0.001, 0.01, 0.005, 0.005))
TRUSTED_RECIPE = {
    "dataset_ids": ["a", "b", "c"],
    "display_dataset_id": "a",
    "dark_correction": {
        "model_id": ADAPTER.DARK_CORRECTION_MODEL,
        "path": "darkImg.osc.gz",
        "scale": 1.0,
        "scale_basis": "matched_exposure_assumed.v1",
    },
    "model_cubature": {
        "fit_gauss_order": 3,
        "oracle_gauss_order": 5,
        "fold_fit_subdivisions": 4,
        "fold_oracle_subdivisions": 8,
        "maximum_oracle_relative_l2": 0.03,
    },
    "profile_cubature": {
        "fit_gauss_order": 4,
        "fold_fit_subdivisions": 6,
        "minimum_valid_bins_per_profile": 3,
    },
}


def test_specular_angle_chart_does_not_require_a_propagating_diffraction_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinates = SimpleNamespace(
        column_px=np.array([10.0]),
        row_px=np.array([20.0]),
        valid=np.array([True]),
    )
    monkeypatch.setattr(
        ADAPTER,
        "angles_to_detector_coordinate_area_measure",
        lambda *args, **kwargs: SimpleNamespace(
            coordinates=coordinates,
            detector_area_jacobian_px2_per_rad2=np.array([2.0]),
        ),
    )
    chart = ADAPTER._OscAngleDetectorAreaChart(
        instrument=object(),
        angle_frame=object(),
        nominal_context=SimpleNamespace(
            evaluate_detector_geometry=lambda *args, **kwargs: SimpleNamespace(
                kf_air_sample_Ainv=np.array([[1.0, 0.0, 0.01]]),
                valid=np.array([False]),
            )
        ),
        horizon_acceptance=ADAPTER.DetectorHorizonAcceptance(
            offspecular_air_exit_guard_rad=np.deg2rad(1.0),
        ),
        revision="external-air-specular-test.v1",
    )

    mapped = chart.map_detector_area(np.array([0.1]), np.array([0.0]))

    np.testing.assert_array_equal(mapped.valid, np.array([True]))
    np.testing.assert_allclose(mapped.detector_area_jacobian_px2_per_chart2, np.array([2.0]))


def _projection_refinement(dataset_ids: tuple[str, ...] = ("a", "b", "c")) -> dict[str, object]:
    return {
        "status": "COMPLETE",
        "converged": True,
        "support_masks_equal": True,
        "acceptance_measure": "integrated_peak_count_mass_and_support.v1",
        "covariance_refinement_converged": True,
        "covariance_policy": "refined_covariance_is_authoritative_for_objective_whitening",
        "maximum_relative_l2": 0.03,
        "by_dataset": {
            dataset_id: {name: 0.01 for name in ADAPTER.PROJECTION_CONVERGENCE_METRICS}
            for dataset_id in dataset_ids
        },
    }


def test_radial_background_cells_retain_valid_zero_counts() -> None:
    row, column = np.indices((100, 100))
    counts = ((row + column) % 2).astype(np.int32)
    _, _, density, _ = ADAPTER._robust_radial_cells(
        counts=counts,
        beam_center_column_row_px=(49.5, 49.5),
        excluded_flat_pixel_index=np.empty(0, dtype=np.int64),
        sample_stride=1,
        radial_bin_width_px=100.0,
        azimuth_sector_count=8,
        minimum_radius_px=0.0,
        maximum_radius_px=70.0,
        border_px=0,
    )

    assert density.size == 8
    assert float(np.mean(density)) == pytest.approx(0.5, abs=1.0e-12)
    assert float(np.max(density)) < 0.52

    _, _, signed_density, _ = ADAPTER._robust_radial_cells(
        counts=2.0 * counts - 1.0,
        beam_center_column_row_px=(49.5, 49.5),
        excluded_flat_pixel_index=np.empty(0, dtype=np.int64),
        sample_stride=1,
        radial_bin_width_px=100.0,
        azimuth_sector_count=8,
        minimum_radius_px=0.0,
        maximum_radius_px=70.0,
        border_px=0,
    )
    assert signed_density.size == 8
    assert float(np.mean(signed_density)) == pytest.approx(0.0, abs=1.0e-12)
    assert float(np.max(np.abs(signed_density))) < 0.04


def test_integrated_peak_projection_rejects_silent_declared_peak_pruning() -> None:
    rows: list[dict[str, object]] = []

    def add_block(group: str, family: int, signal_bands: tuple[str, ...]) -> None:
        block = len({int(row["block_index"]) for row in rows})
        for band in (*signal_bands, "background_0", "background_1"):
            background = band.startswith("background_")
            rows.append(
                {
                    "dataset_id": "image",
                    "dataset_index": 0,
                    "block_index": block,
                    "signal_family_m": -1 if background else family,
                    "is_background": background,
                    "count_sum": 10.0,
                    "support_px2": 1.0,
                    "coordinate_mean": float(len(rows)),
                    "group": group,
                    "band": band,
                    "bin_index": 0,
                }
            )

    add_block("m0", 0, ("signal",))
    for family in (1, 3, 4):
        add_block(f"m{family}", family, (f"m{family}_plus", f"m{family}_minus"))
    arrays = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
    manifest = {
        "dataset_ids": ["image"],
        "fixed_lattice": {"active_direct_basis_A": np.eye(3).tolist()},
        "m0_region": {"two_theta_bin_edges_rad": np.deg2rad((0.0, 1.0)).tolist()},
        "offspecular_layouts": [
            {"group": f"m{family}", "axial_bin_edges": [0.0, 1.0]} for family in (1, 3, 4)
        ],
        "fit_peak_catalog": [
            {
                "identity": "m0-with-unmapped-peak",
                "dataset_id": "image",
                "profile_identity": "m0",
                "family_m": 0,
                "coordinate_kind": "two_theta_deg",
                "centers": [0.5, 10.0],
                "half_width": 0.4,
            },
            *[
                {
                    "identity": f"m{family}",
                    "dataset_id": "image",
                    "profile_identity": f"m{family}",
                    "family_m": family,
                    "coordinate_kind": "L",
                    "center": 0.5,
                    "half_width": 0.4,
                }
                for family in (1, 3, 4)
            ],
        ],
    }

    with pytest.raises(ValueError, match="no retained signal rows"):
        ADAPTER._integrated_peak_area_projection(arrays, manifest)

    valid_manifest = copy.deepcopy(manifest)
    valid_manifest["fit_peak_catalog"][0]["centers"] = [0.5]
    projection, catalog = ADAPTER._integrated_peak_area_projection(arrays, valid_manifest)
    prepared_arrays = {
        **arrays,
        "source_signal_peak_index": np.asarray(projection.source_signal_peak_index),
    }
    prepared_manifest = {
        **valid_manifest,
        "integrated_peak_catalog": catalog,
        "peak_area_projection_revision": projection.revision,
        "source_signal_peak_index_sha256": ADAPTER._array_sha256(
            prepared_arrays["source_signal_peak_index"]
        ),
    }
    observations = MatchedRegionObservations(
        dataset_ids=("image",),
        dataset_index=arrays["dataset_index"],
        block_index=arrays["block_index"],
        signal_family=arrays["signal_family_m"],
        is_background=arrays["is_background"],
        count_mass=arrays["count_sum"],
        support_px2=arrays["support_px2"],
        background_coordinate=arrays["coordinate_mean"],
        required_signal_families=ADAPTER.FAMILIES,
    )
    ADAPTER._prepared_peak_area_projection(prepared_arrays, prepared_manifest, observations)
    mutated_arrays = {**prepared_arrays}
    mutated_mapping = np.array(prepared_arrays["source_signal_peak_index"], copy=True)
    plus = next(index for index, record in enumerate(catalog) if record["peak_id"] == "m1:plus")
    minus = next(index for index, record in enumerate(catalog) if record["peak_id"] == "m1:minus")
    mutated_mapping[mutated_mapping == plus] = -1
    mutated_mapping[mutated_mapping == minus] = plus
    mutated_mapping[mutated_mapping == -1] = minus
    mutated_arrays["source_signal_peak_index"] = mutated_mapping
    with pytest.raises(ValueError, match="mapping changed"):
        ADAPTER._prepared_peak_area_projection(
            mutated_arrays,
            prepared_manifest,
            observations,
        )


def test_optional_continuous_plan_can_become_empty_after_support_screening() -> None:
    quadrature = ContinuousRegionQuadrature(
        column_px=np.asarray((0.0, 2.0, 3.0)),
        row_px=np.asarray((0.0, 0.0, 4.0)),
        detector_area_weight_px2=np.asarray((0.25, 0.75, 1.5)),
        observation_row=np.asarray((0, 0, 1)),
        background_coordinate=np.asarray((0.0, 0.0, 1.0)),
        observation_count=2,
        chart_revision="continuous-background.test.v1",
    )
    plan = ADAPTER._DatasetContinuousRegionPlan(
        dataset_index=0,
        dataset_id="image",
        global_observation_row=np.asarray((0, 1)),
        quadrature=quadrature,
        fold_bands=(),
        rectangle_count=2,
    )
    assert ADAPTER._drop_optional_continuous_plan_observations(plan, {0, 1}) is None
    retained = ADAPTER._drop_optional_continuous_plan_observations(plan, {1})
    assert retained is not None
    np.testing.assert_array_equal(retained.quadrature.observation_row, np.asarray((0, 0)))


def test_radial_background_is_integrated_on_continuous_region_nodes() -> None:
    quadrature = ContinuousRegionQuadrature(
        column_px=np.asarray((0.0, 2.0, 3.0)),
        row_px=np.asarray((0.0, 0.0, 4.0)),
        detector_area_weight_px2=np.asarray((0.25, 0.75, 1.5)),
        observation_row=np.asarray((0, 0, 1)),
        background_coordinate=np.asarray((0.0, 0.0, 1.0)),
        observation_count=2,
        chart_revision="continuous-background.test.v1",
    )
    plan = ADAPTER._DatasetContinuousRegionPlan(
        dataset_index=0,
        dataset_id="image",
        global_observation_row=np.asarray((0, 1)),
        quadrature=quadrature,
        fold_bands=(),
        rectangle_count=2,
    )
    state = RadialBackgroundState(
        dataset_ids=("image",),
        inner_scale_px=2.0,
        inner_power=1.5,
        outer_scale_px=10.0,
        outer_power=1.2,
        amplitude_count_per_px=(7.0,),
        pedestal_count_per_px=(3.0,),
        parameter_covariance=np.diag(np.linspace(0.01, 0.06, 6)),
    )

    background = ADAPTER._fixed_background_from_continuous_plans(
        state=state,
        plans=(plan,),
        observation_count=2,
        beam_center_column_row_px=(0.0, 0.0),
    )

    radius = np.hypot(quadrature.column_px, quadrature.row_px)
    dataset = np.zeros(radius.size, dtype=np.int64)
    expected_mass = quadrature.integrate_density(state.count_density(dataset, radius))
    node_jacobian = state.count_density_parameter_jacobian(dataset, radius)
    expected_jacobian = np.column_stack(
        [
            np.bincount(
                quadrature.observation_row,
                weights=quadrature.detector_area_weight_px2 * node_jacobian[:, index],
                minlength=2,
            )
            for index in range(6)
        ]
    )
    expected_covariance = expected_jacobian @ state.parameter_covariance @ expected_jacobian.T
    np.testing.assert_allclose(background.count_mass, expected_mass)
    np.testing.assert_allclose(background.covariance_count2, expected_covariance)


def test_background_exclusion_contains_center_and_continuous_projection_pixels() -> None:
    excluded = ADAPTER._background_exclusion_pixel_index(
        np.asarray((1, 3, 7), dtype=np.int64),
        np.asarray((2, 3, 8), dtype=np.int64),
    )

    np.testing.assert_array_equal(excluded, (1, 2, 3, 7, 8))
    assert not excluded.flags.writeable


def test_native_pixel_center_statistics_preserve_overlap_and_shared_dark_covariance() -> None:
    arrays = {
        "dataset_index": np.asarray((0, 0, 1), dtype=np.int64),
        "count_sum": np.asarray((4.0, 5.0, 12.0)),
        "support_px2": np.asarray((1.0, 1.0, 2.0)),
        "selected_dataset_index": np.asarray((0, 0, 1, 1), dtype=np.int64),
        "selected_flat_pixel_index": np.asarray((0, 1, 0, 1), dtype=np.int64),
        "selected_observation_row": np.asarray((0, 1, 2, 2), dtype=np.int64),
    }
    counts_by_dataset = {
        "a": np.asarray(((4.0, 5.0, 0.0), (0.0, 0.0, 0.0))),
        "b": np.asarray(((7.0, 5.0, 0.0), (0.0, 0.0, 0.0))),
    }

    mass, covariance, support, plans = ADAPTER._native_pixel_center_count_statistics(
        arrays,
        dataset_ids=("a", "b"),
        counts_by_dataset=counts_by_dataset,
        dark_counts=np.asarray(((1.0, 2.0, 0.0), (0.0, 0.0, 0.0))),
        dark_scale=1.0,
    )

    np.testing.assert_array_equal(mass, (3.0, 3.0, 9.0))
    np.testing.assert_array_equal(support, (1.0, 1.0, 2.0))
    np.testing.assert_allclose(
        covariance,
        ((5.0, 0.0, 1.0), (0.0, 7.0, 2.0), (1.0, 2.0, 15.0)),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert len(plans) == 2

    zero_dark_mass, zero_dark_covariance, zero_dark_support, _ = (
        ADAPTER._native_pixel_center_count_statistics(
            arrays,
            dataset_ids=("a", "b"),
            counts_by_dataset=counts_by_dataset,
            dark_counts=np.asarray(((1.0, 2.0, 0.0), (0.0, 0.0, 0.0))),
            dark_scale=0.0,
        )
    )
    np.testing.assert_array_equal(zero_dark_mass, arrays["count_sum"])
    np.testing.assert_array_equal(zero_dark_support, arrays["support_px2"])
    np.testing.assert_allclose(
        zero_dark_covariance,
        np.diag((4.0, 5.0, 12.0)),
        rtol=3.0e-16,
        atol=0.0,
    )
    assert ADAPTER._dark_covariance_model(0.0) == "no_dark_contribution.v1"
    assert ADAPTER._dark_scale_basis_is_valid(0.0, "no_acquisition_matched_dark.v1")
    assert not ADAPTER._dark_scale_basis_is_valid(0.0, "matched_exposure_assumed.v1")
    assert not ADAPTER._dark_scale_basis_is_valid(1.0, "no_acquisition_matched_dark.v1")

    changed = copy.deepcopy(arrays)
    changed["count_sum"][0] += 1.0
    with pytest.raises(ValueError, match="prepared count sums"):
        ADAPTER._native_pixel_center_count_statistics(
            changed,
            dataset_ids=("a", "b"),
            counts_by_dataset=counts_by_dataset,
            dark_counts=np.asarray(((1.0, 2.0, 0.0), (0.0, 0.0, 0.0))),
            dark_scale=1.0,
        )

    for invalid_scale in (True, "1.0"):
        with pytest.raises(ValueError, match="finite nonnegative real"):
            ADAPTER._verified_dark_counts({"dark_correction": {"scale": invalid_scale}})


def test_continuous_observations_make_zero_dark_an_exact_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = np.asarray(((4.0, 5.0, 6.0), (7.0, 8.0, 9.0)))
    coordinate = np.asarray((0.0, 0.0, 0.0, 0.0, -1.0, 1.0))
    quadrature = ContinuousRegionQuadrature(
        column_px=np.asarray((0.0, 1.0, 2.0, 0.0, 1.0, 2.0)),
        row_px=np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
        detector_area_weight_px2=np.ones(6),
        observation_row=np.arange(6),
        background_coordinate=coordinate,
        observation_count=6,
        chart_revision="continuous-zero-dark.test.v1",
    )
    plan = ADAPTER._DatasetContinuousRegionPlan(
        dataset_index=0,
        dataset_id="a",
        global_observation_row=np.arange(6),
        quadrature=quadrature,
        fold_bands=(),
        rectangle_count=6,
    )
    arrays = {
        "count_sum": np.arange(6, dtype=np.float64),
        "dataset_index": np.zeros(6, dtype=np.int64),
        "block_index": np.zeros(6, dtype=np.int64),
        "signal_family_m": np.asarray((0, 1, 3, 4, -1, -1), dtype=np.int64),
        "is_background": np.asarray((False, False, False, False, True, True)),
    }
    monkeypatch.setattr(
        ADAPTER,
        "_verified_dark_counts",
        lambda _manifest: (np.full_like(counts, 1000.0), 0.0),
    )

    observations, _, _ = ADAPTER._continuous_matched_observations(
        arrays,
        {"dataset_ids": ["a"]},
        (plan,),
        {"a": counts},
    )
    projection = ADAPTER.compile_native_pixel_region_projection(quadrature, counts.shape)
    expected_mass, expected_covariance = projection.integrate_counts(counts)

    np.testing.assert_array_equal(observations.count_mass, expected_mass)
    np.testing.assert_array_equal(
        observations.count_covariance_count2,
        expected_covariance,
    )


def test_conditioned_cubature_gate_detects_anchor_cancellation() -> None:
    oracle = np.asarray((101.0, 101.0, 101.0, 101.0, 100.0, 100.0))
    candidate = np.asarray((101.0, 101.0, 101.0, 101.0, 99.0, 101.0))
    projection = np.zeros((6, 6), dtype=np.float64)
    projection[:4, 4:] = np.asarray(((0.5, 0.5), (0.25, 0.75), (0.75, 0.25), (0.5, 0.5)))
    projection[4:, 4:] = np.eye(2)
    background = FixedMatchedRegionBackground(
        count_mass=np.zeros(6),
        covariance_count2=np.zeros((6, 6)),
        anchor_projection=projection,
        revision="conditioned-cubature.test.v1",
    )

    overall, by_family, raw_anchor = ADAPTER._conditioned_model_cubature_errors(
        candidate,
        oracle,
        fixed_background=background,
        signal_family_m=np.asarray((0, 1, 3, 4, -1, -1)),
        is_background=np.asarray((False, False, False, False, True, True)),
    )

    assert raw_anchor == pytest.approx(0.01)
    assert overall == pytest.approx(math.sqrt(0.5) / 2.0)
    assert by_family["1"] == pytest.approx(0.5)
    assert by_family["3"] == pytest.approx(0.5)

    integrated, integrated_by_family, _ = ADAPTER._conditioned_model_cubature_errors(
        candidate,
        oracle,
        fixed_background=background,
        signal_family_m=np.asarray((0, 1, 3, 4, -1, -1)),
        is_background=np.asarray((False, False, False, False, True, True)),
        peak_aggregation=np.asarray(((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 1.0, 0.0))),
        peak_family_m=np.asarray((0, 1)),
    )
    assert integrated == pytest.approx(0.0)
    assert integrated_by_family["0"] == pytest.approx(0.0)
    assert integrated_by_family["1"] == pytest.approx(0.0)


def test_projection_convergence_catches_weak_rows_support_and_covariance() -> None:
    refined_mass = np.asarray((1000.0, 1.0, 500.0, 2.0))
    refined_support = np.ones(4)
    refined_covariance = np.diag((1000.0, 1.0, 500.0, 2.0))
    common = {
        "refined_count_mass": refined_mass,
        "refined_support_px2": refined_support,
        "refined_count_covariance": refined_covariance,
        "dataset_index": np.asarray((0, 0, 1, 1)),
        "dataset_ids": ("a", "b"),
        "maximum_relative_l2": 0.03,
    }

    weak_row = ADAPTER._native_projection_convergence(
        coarse_count_mass=refined_mass + np.asarray((0.0, 0.5, 0.0, 0.0)),
        coarse_support_px2=refined_support,
        coarse_count_covariance=refined_covariance,
        **common,
    )
    assert weak_row["by_dataset"]["a"]["mass_relative_l2"] < 0.03
    assert weak_row["by_dataset"]["a"]["maximum_row_mass_standardized_error"] == pytest.approx(0.5)
    assert weak_row["converged"] is False

    support = ADAPTER._native_projection_convergence(
        coarse_count_mass=refined_mass,
        coarse_support_px2=np.asarray((1.0, 1.1, 1.0, 1.0)),
        coarse_count_covariance=refined_covariance,
        **common,
    )
    covariance = ADAPTER._native_projection_convergence(
        coarse_count_mass=refined_mass,
        coarse_support_px2=refined_support,
        coarse_count_covariance=2.0 * refined_covariance,
        **common,
    )
    assert support["converged"] is False
    assert covariance["converged"] is False
    assert ADAPTER._integrated_area_projection_converged(covariance) is True
    assert ADAPTER._integrated_area_projection_converged(weak_row) is False
    assert ADAPTER._display_profile_projection_converged(weak_row) is True
    assert ADAPTER._display_profile_projection_converged(support) is False


def test_position_dataset_binding_preserves_order_and_nonroundtripping_angles() -> None:
    angle = float(np.nextafter(12.0, np.inf))
    commanded_deg = (5.0, 10.0, angle)
    position = FixedPositionState(
        artifact_revision=f"sha256-{'a' * 64}",
        corrections=SharedGeometryCorrections.zero(),
        incidence_angle_delta_rad=0.0,
        commanded_incidence_angles_rad=tuple(map(math.radians, commanded_deg)),
        beam_center_column_row_px=(1453.12, 1596.422),
        incidence_angle_image_ids=("five", "ten", "high"),
        incidence_angle_trim_rad=(0.0, 0.0, 0.0),
        incidence_angle_trim_contrast_rad=(0.0, 0.0),
        incidence_angle_trim_prior_sigma_rad=math.radians(0.25),
        incidence_angle_trim_contrast_half_span_rad=math.radians(0.5),
    )
    datasets = [
        {
            "dataset_id": image_id,
            "position_image_id": image_id,
            "incidence_angle_deg": incidence_deg,
            "osc": {"path": f"C:/{image_id}.osc.gz", "sha256": "unused"},
        }
        for image_id, incidence_deg in zip(
            position.incidence_angle_image_ids,
            commanded_deg,
            strict=True,
        )
    ]

    assert ADAPTER._validate_position_dataset_binding(position, datasets) == commanded_deg
    datasets[0]["position_image_id"], datasets[1]["position_image_id"] = (
        datasets[1]["position_image_id"],
        datasets[0]["position_image_id"],
    )
    with pytest.raises(ValueError, match="OSC order"):
        ADAPTER._validate_position_dataset_binding(position, datasets)


def test_implementation_identity_covers_both_numerical_packages() -> None:
    identity = ADAPTER._implementation_identity()
    expected_files = {
        SCRIPT.resolve(),
        *(
            path.resolve()
            for package_name in ("painted_ewald", "rasim_next")
            for path in (ROOT / "src" / package_name).rglob("*.py")
        ),
    }

    assert identity["file_count"] == len(expected_files)
    assert len(identity["sha256"]) == 64
    assert "painted_ewald and rasim_next" in identity["scope"]


def _accepted_fit_document() -> dict[str, object]:
    rod_roster = ((0, 0, 0, 1.0), (1, 0, 1, 1.0), (1, 1, 3, 1.0), (2, 0, 4, 1.0))
    rod_roster_sha256 = ADAPTER._rod_roster_sha256(rod_roster)
    peak_revision = f"sha256-{'d' * 64}.integrated-peak-area.v1"
    peak_mapping_sha256 = "e" * 64
    full_parameters = np.asarray((0.001, -0.001, 0.012, 0.004, 0.006))
    fit_start = {
        "kind": "explicit",
        "initial_parameters": full_parameters.tolist(),
        "initial_full_parameters": full_parameters.tolist(),
        "initial_parameters_sha256": ADAPTER._array_sha256(full_parameters),
        "maximum_function_evaluations": 50,
        "source_artifact": None,
        "semantics": "new optimizer run",
    }
    document = {
        "schema_version": ADAPTER.FIT_SCHEMA,
        "stage": "joint",
        "execution_policy": "seeded_joint_only.v1",
        "status": "FIT",
        "active_parameter_names": list(ADAPTER.STRUCTURE_PARAMETER_NAMES),
        "frozen_parameter_names": [],
        "optimizer": {"success": True, "fit_start": fit_start},
        "numerical_convergence": {
            "converged": True,
            "optimizer_converged": True,
            "identifiable": True,
            "cubature_converged_by_family": True,
        },
        "sensitivity": {
            "rank": 5,
            "numerical_rank": 5,
            "parameter_count": 5,
            "condition": 25.0,
            "relative_tolerance": 1.0e-5,
            "parameter_names": list(ADAPTER.STRUCTURE_PARAMETER_NAMES),
            "parameter_scales": STRUCTURE_PARAMETER_SCALES.tolist(),
            "data_only": True,
            "parameter_scaled": True,
        },
        "regularization": {
            "parameter_scales": STRUCTURE_PARAMETER_SCALES.tolist(),
            "lower_bounds": STRUCTURE_LOWER_BOUNDS.tolist(),
            "upper_bounds": STRUCTURE_UPPER_BOUNDS.tolist(),
            "bound_proximity_in_parameter_scales": 1.0e-6,
        },
        "parameters_on_bounds": [],
        "model_rod_scope": "fitted_families_m_0_1_3_4",
        "model_rod_count": len(rod_roster),
        "model_rod_roster_h_k_m_population": rod_roster,
        "model_rod_roster_sha256": rod_roster_sha256,
        "stacking_model": ADAPTER._fault_free_three_r_definition(),
        "cubature_oracle": {
            "performed": True,
            "status": "COMPLETE",
            "fit_gauss_order": 3,
            "oracle_gauss_order": 5,
            "fit_subdivision_count": 4,
            "oracle_subdivision_count": 8,
            "relative_l2": 0.01,
            "relative_l2_by_family_m": {"0": 0.01, "1": 0.01, "3": 0.01, "4": 0.01},
            "relative_l2_background_anchor_rows": 0.01,
            "maximum_relative_l2": 0.03,
        },
        "data_projection": {
            "method": ADAPTER.MEASURED_PROJECTION_METHOD,
            **_projection_refinement(),
            "fit_projection_revisions": ["a" * 64, "b" * 64, "c" * 64],
            "oracle_projection_revisions": ["d" * 64, "e" * 64, "f" * 64],
            "fit_gauss_order": 3,
            "oracle_gauss_order": 5,
            "fit_subdivision_count": 4,
            "oracle_subdivision_count": 8,
            "relative_l2_by_family_m": {"0": 0.01, "1": 0.01, "3": 0.01, "4": 0.01},
            "relative_l2_background_anchor_rows": 0.01,
            "maximum_relative_l2": 0.03,
            "smoothing_applied": False,
            "diffraction_model_pixelized": False,
        },
        "provenance": {
            "background": {"sha256": "0" * 64},
            "recipe": {"sha256": "recipe"},
            "fit_plan": {"sha256": "fit-plan"},
            "fit_adapter": {"sha256": "adapter"},
            "implementation": {"sha256": "implementation"},
            "execution_identity": {
                "diagnostic_sha256": "diagnostic",
                "recipe_sha256": "recipe",
                "fit_plan_sha256": "fit-plan",
                "fit_adapter_sha256": "adapter",
                "implementation_sha256": "implementation",
                "stage": "joint",
                "active_parameter_names": list(ADAPTER.STRUCTURE_PARAMETER_NAMES),
                "frozen_parameter_names": [],
                "rod_scope": "families_m_0_1_3_4",
                "rod_count": len(rod_roster),
                "rod_roster_sha256": rod_roster_sha256,
                "model_measure": "continuous_detector_chart_area",
                "fit_gauss_order": 3,
                "fit_subdivision_count": 4,
                "oracle_gauss_order": 5,
                "oracle_subdivision_count": 8,
                "offspecular_axial_refinement": 3,
                "offspecular_radial_transform": "squared_fold_coordinate.v1",
                "offspecular_signal_minimum_radial_nodes_per_side": 24,
                "background_artifact_sha256": "0" * 64,
                "radial_background_state_revision": "1" * 64,
                "radial_background_parameter_vector_sha256": "2" * 64,
                "radial_background_parameter_covariance_sha256": "3" * 64,
                "radial_background_mass_sha256": "4" * 64,
                "radial_background_covariance_sha256": "5" * 64,
                "background_excluded_flat_pixel_count_by_dataset": {
                    "a": 10,
                    "b": 11,
                    "c": 12,
                },
                "background_excluded_flat_pixel_sha256_by_dataset": {
                    "a": "a" * 64,
                    "b": "b" * 64,
                    "c": "c" * 64,
                },
                "conditioned_background_revision": f"sha256-{'9' * 64}.adjacent-affine.v1",
                "conditioned_background_mass_sha256": "6" * 64,
                "conditioned_background_covariance_sha256": "7" * 64,
                "conditioned_model_anchor_projection_sha256": "8" * 64,
                "objective_measure": ADAPTER.PEAK_AREA_OBJECTIVE,
                "peak_area_projection_revision": peak_revision,
                "source_signal_peak_index_sha256": peak_mapping_sha256,
            },
        },
        "background_model": {
            "artifact_sha256": "0" * 64,
            "state_revision": "1" * 64,
            "parameter_vector_sha256": "2" * 64,
            "parameter_covariance_sha256": "3" * 64,
            "radial_mass_sha256": "4" * 64,
            "radial_covariance_sha256": "5" * 64,
            "excluded_flat_pixel_count_by_dataset": {"a": 10, "b": 11, "c": 12},
            "excluded_flat_pixel_sha256_by_dataset": {
                "a": "a" * 64,
                "b": "b" * 64,
                "c": "c" * 64,
            },
            "conditioned_revision": f"sha256-{'9' * 64}.adjacent-affine.v1",
            "conditioned_mass_sha256": "6" * 64,
            "conditioned_covariance_sha256": "7" * 64,
            "anchor_projection_sha256": "8" * 64,
        },
        "structure_representative": {
            "bi_delta_z_fractional": 0.001,
            "outer_chalcogen_delta_z_fractional": -0.001,
            "bi_occupancy": 1.0,
            "central_chalcogen_occupancy": 1.0,
            "outer_chalcogen_occupancy": 0.988,
            "outer_chalcogen_vacancy_fraction": 0.012,
            "outer_bi_antisite_fraction": 0.0,
            "outer_chalcogen_fraction": 0.988,
            "occupancy_rule": "outer_site_chalcogen_plus_vacancy.v1",
            "intensity_envelope_u_radial_A2": 0.004,
            "intensity_envelope_u_normal_A2": 0.006,
            "displacement_gauge": {
                "model_id": "fixed_site_adp_plus_regularized_sample_q_envelope.v1",
                "site_adp_common_mode": "fixed_reference",
                "sample_q_envelope_mode": "fit_zero_centered_regularized",
            },
            "site_adp_scale": 1.0,
            "site_adp_refinement_status": "fixed_literature_reference",
            "site_displacement_profile": {
                "model_id": "transverse_isotropic_site_reference_fixed.v1",
                "provenance": "test fixed site displacement profile",
                "sites": [
                    {
                        "source_label": label,
                        "reference_u_radial_A2": radial,
                        "reference_u_normal_A2": normal,
                        "fitted_u_radial_A2": radial,
                        "fitted_u_normal_A2": normal,
                    }
                    for label, radial, normal in (
                        ("Bi", 0.0036, 0.0264),
                        ("Se1", 0.0046, 0.0485),
                        ("Se2", 0.0046, 0.0485),
                    )
                ],
            },
        },
        "full_parameter_vector": full_parameters.tolist(),
        "diagnostic_sha256": "diagnostic",
        "dataset_scales": {"a": 2.0, "b": 3.0, "c": 4.0},
        "fitted_model_count": [4.0, 9.0, 16.0],
        "objective_measure": ADAPTER.PEAK_AREA_OBJECTIVE,
        "integrated_peak_areas": {
            "projection_revision": peak_revision,
            "peak_ids": ["m0", "m1", "m3", "m4"],
            "dataset_index": [0, 0, 1, 2],
            "family_m": [0, 1, 3, 4],
            "observed_background_subtracted_count_mass": [10.0, 20.0, 30.0, 40.0],
            "fitted_count_mass": [9.0, 19.0, 31.0, 39.0],
            "source_signal_peak_index_sha256": peak_mapping_sha256,
        },
        "model_measure": "continuous_detector_chart_area",
        "model_pixelized": False,
        "smoothing_applied": False,
    }
    provenance = document["provenance"]
    assert isinstance(provenance, dict)
    provenance["fit_run_identity"] = {
        "model_execution": copy.deepcopy(provenance["execution_identity"]),
        "fit_start": copy.deepcopy(fit_start),
    }
    return document


def _set_fit_start(
    document: dict[str, object],
    *,
    execution_policy: str,
    kind: str,
    initial_full_parameters: np.ndarray | None = None,
    source_artifact: dict[str, str] | None = None,
) -> None:
    active = tuple(document["active_parameter_names"])
    active_index = np.asarray(
        [ADAPTER.STRUCTURE_PARAMETER_NAMES.index(name) for name in active],
        dtype=np.int64,
    )
    full = (
        np.asarray(document["full_parameter_vector"], dtype=np.float64)
        if initial_full_parameters is None
        else np.asarray(initial_full_parameters, dtype=np.float64)
    )
    initial = full[active_index]
    fit_start = {
        "kind": kind,
        "initial_parameters": initial.tolist(),
        "initial_full_parameters": full.tolist(),
        "initial_parameters_sha256": ADAPTER._array_sha256(initial),
        "maximum_function_evaluations": 50,
        "source_artifact": copy.deepcopy(source_artifact),
        "semantics": (
            "restart from a completely evaluated parameter vector; optimizer state is not continued"
            if kind == "progress_restart"
            else "new optimizer run"
        ),
    }
    document["execution_policy"] = execution_policy
    optimizer = document["optimizer"]
    assert isinstance(optimizer, dict)
    optimizer["fit_start"] = fit_start
    provenance = document["provenance"]
    assert isinstance(provenance, dict)
    provenance["fit_run_identity"] = {
        "model_execution": copy.deepcopy(provenance["execution_identity"]),
        "fit_start": copy.deepcopy(fit_start),
    }


def _set_document_structure_vector(
    document: dict[str, object],
    parameters: np.ndarray,
) -> None:
    values = np.asarray(parameters, dtype=np.float64)
    document["full_parameter_vector"] = values.tolist()
    representative = document["structure_representative"]
    assert isinstance(representative, dict)
    for name, value in zip(ADAPTER.STRUCTURE_PARAMETER_NAMES, values, strict=True):
        representative[name] = float(value)
    representative["outer_chalcogen_occupancy"] = float(1.0 - values[2])
    representative["outer_chalcogen_fraction"] = float(1.0 - values[2])


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("status",), "MODEL_LIMITED_FIT"),
        (("execution_policy",), "invented.v1"),
        (("optimizer", "success"), False),
        (("optimizer", "fit_start", "kind"), "default"),
        (("optimizer", "fit_start", "initial_parameters_sha256"), "0" * 64),
        (("provenance", "fit_run_identity"), {}),
        (("numerical_convergence", "converged"), False),
        (("numerical_convergence", "optimizer_converged"), False),
        (("numerical_convergence", "identifiable"), False),
        (("numerical_convergence", "cubature_converged_by_family"), False),
        (("cubature_oracle", "relative_l2_background_anchor_rows"), None),
        (("cubature_oracle", "relative_l2_background_anchor_rows"), float("nan")),
        (("cubature_oracle", "status"), "PARTIAL"),
        (("cubature_oracle", "maximum_relative_l2"), 1.0),
        (("cubature_oracle", "fit_gauss_order"), 2),
        (("data_projection", "converged"), False),
        (("data_projection", "maximum_relative_l2"), 1.0),
        (("data_projection", "oracle_subdivision_count"), 7),
        (("data_projection", "by_dataset", "a", "support_relative_l2"), 0.04),
        (("data_projection", "smoothing_applied"), True),
        (("data_projection", "diffraction_model_pixelized"), True),
        (("data_projection", "relative_l2_by_family_m", "0"), 0.04),
        (("dataset_scales", "b"), float("nan")),
        (("structure_representative", "occupancy_rule"), "substitution.v1"),
        (("structure_representative", "outer_bi_antisite_fraction"), 0.01),
        (("structure_representative", "outer_chalcogen_occupancy"), 1.0),
        (("model_rod_roster_sha256",), "9" * 64),
        (("sensitivity", "rank"), 4),
        (("parameters_on_bounds",), ["intensity_envelope_u_normal_A2"]),
        (("provenance", "recipe", "sha256"), "wrong"),
        (("provenance", "fit_plan", "sha256"), "wrong"),
        (("provenance", "fit_adapter", "sha256"), "wrong"),
        (("provenance", "implementation", "sha256"), "wrong"),
        (("model_measure",), "native_detector_pixels"),
        (("provenance", "execution_identity", "model_measure"), "native_detector_pixels"),
        (("provenance", "execution_identity", "oracle_gauss_order"), 4),
        (("background_model", "radial_mass_sha256"), "9" * 64),
        (("background_model", "state_revision"), ""),
        (("background_model", "excluded_flat_pixel_count_by_dataset", "b"), 0),
        (("sensitivity", "condition"), 1.0e6),
    ),
)
def test_fit_document_admissibility_fails_closed(path: tuple[str, ...], value: object) -> None:
    document = _accepted_fit_document()
    target: dict[str, object] = document
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value

    assert not ADAPTER.fit_document_is_admissible(
        document,
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        adapter_sha256="adapter",
        fit_plan_sha256="fit-plan",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
    )


def test_fit_document_admissibility_accepts_complete_evidence() -> None:
    assert ADAPTER.fit_document_is_admissible(
        _accepted_fit_document(),
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        adapter_sha256="adapter",
        fit_plan_sha256="fit-plan",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
    )

    wrong_policy = _accepted_fit_document()
    _set_fit_start(
        wrong_policy,
        execution_policy="staged_A_B_C_joint.v1",
        kind="default",
    )
    assert not ADAPTER.fit_document_is_admissible(
        wrong_policy,
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        adapter_sha256="adapter",
        fit_plan_sha256="fit-plan",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
    )


def test_fit_document_rejects_relaxed_or_changed_fixed_site_adp_gauge() -> None:
    changed_scale = _accepted_fit_document()
    changed_scale["structure_representative"]["site_adp_scale"] = 0.9
    changed_site = _accepted_fit_document()
    changed_site["structure_representative"]["site_displacement_profile"]["sites"][0][
        "fitted_u_normal_A2"
    ] += 0.001
    for document in (changed_scale, changed_site):
        assert not ADAPTER.fit_document_is_admissible(
            document,
            trusted_recipe=TRUSTED_RECIPE,
            recipe_sha256="recipe",
            adapter_sha256="adapter",
            fit_plan_sha256="fit-plan",
            implementation_sha256="implementation",
            lower_bounds=STRUCTURE_LOWER_BOUNDS,
            upper_bounds=STRUCTURE_UPPER_BOUNDS,
            parameter_scales=STRUCTURE_PARAMETER_SCALES,
            sensitivity_relative_tolerance=1.0e-5,
            bound_proximity_in_parameter_scales=1.0e-6,
            maximum_sensitivity_condition=1.0e5,
        )


def test_fit_document_cannot_raise_its_own_projection_or_cubature_tolerance() -> None:
    document = _accepted_fit_document()
    for section in (document["cubature_oracle"], document["data_projection"]):
        assert isinstance(section, dict)
        section["maximum_relative_l2"] = 1.0
        section["relative_l2_background_anchor_rows"] = 0.5
        section["relative_l2_by_family_m"] = {str(family): 0.5 for family in ADAPTER.FAMILIES}

    assert not ADAPTER.fit_document_is_admissible(
        document,
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        adapter_sha256="adapter",
        fit_plan_sha256="fit-plan",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
    )


def test_fit_document_admissibility_requires_anchor_cubature_evidence() -> None:
    document = _accepted_fit_document()
    del document["cubature_oracle"]["relative_l2_background_anchor_rows"]

    assert not ADAPTER.fit_document_is_admissible(
        document,
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        adapter_sha256="adapter",
        fit_plan_sha256="fit-plan",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
    )


def test_stage_predecessor_contract_fails_closed() -> None:
    document = _accepted_fit_document()
    document["stage"] = "A"
    document["status"] = "STAGE_CONDITIONED"
    document["active_parameter_names"] = list(ADAPTER.STRUCTURE_PARAMETER_NAMES[:2])
    document["frozen_parameter_names"] = list(ADAPTER.STRUCTURE_PARAMETER_NAMES[2:])
    document["numerical_convergence"]["cubature_converged_by_family"] = "NOT_RUN"
    document["cubature_oracle"] = {
        "performed": False,
        "status": "NOT_RUN_INITIALIZER",
        **ADAPTER._trusted_model_cubature(TRUSTED_RECIPE),
        "relative_l2": None,
        "relative_l2_by_family_m": {},
        "relative_l2_background_anchor_rows": None,
    }
    sensitivity = document["sensitivity"]
    assert isinstance(sensitivity, dict)
    sensitivity["rank"] = 2
    sensitivity["numerical_rank"] = 2
    sensitivity["parameter_count"] = 2
    sensitivity["parameter_names"] = list(ADAPTER.STRUCTURE_PARAMETER_NAMES[:2])
    sensitivity["parameter_scales"] = STRUCTURE_PARAMETER_SCALES[:2].tolist()
    execution = document["provenance"]["execution_identity"]
    assert isinstance(execution, dict)
    execution["stage"] = "A"
    execution["active_parameter_names"] = list(ADAPTER.STRUCTURE_PARAMETER_NAMES[:2])
    execution["frozen_parameter_names"] = list(ADAPTER.STRUCTURE_PARAMETER_NAMES[2:])
    _set_fit_start(
        document,
        execution_policy="staged_A_B_C_joint.v1",
        kind="default",
    )

    assert ADAPTER.stage_fit_document_is_admissible(
        document,
        expected_stage="A",
        expected_active_parameter_names=ADAPTER.STRUCTURE_PARAMETER_NAMES[:2],
        diagnostic_sha256="diagnostic",
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        fit_plan_sha256="fit-plan",
        adapter_sha256="adapter",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
    )

    reordered = copy.deepcopy(document)
    reordered["dataset_scales"] = {"c": 4.0, "a": 2.0, "b": 3.0}
    assert ADAPTER.stage_fit_document_is_admissible(
        reordered,
        expected_stage="A",
        expected_active_parameter_names=ADAPTER.STRUCTURE_PARAMETER_NAMES[:2],
        diagnostic_sha256="diagnostic",
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        fit_plan_sha256="fit-plan",
        adapter_sha256="adapter",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
    )

    refined_covariance = copy.deepcopy(document)
    projection = refined_covariance["data_projection"]
    assert isinstance(projection, dict)
    projection["covariance_refinement_converged"] = False
    projection["relative_l2_background_anchor_rows"] = 1.0
    for metrics in projection["by_dataset"].values():
        metrics["covariance_relative_frobenius"] = 1.0
        metrics["maximum_covariance_row_relative_l2"] = 2.0
    assert ADAPTER.stage_fit_document_is_admissible(
        refined_covariance,
        expected_stage="A",
        expected_active_parameter_names=ADAPTER.STRUCTURE_PARAMETER_NAMES[:2],
        diagnostic_sha256="diagnostic",
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        fit_plan_sha256="fit-plan",
        adapter_sha256="adapter",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
    )

    for path, value in (
        (("status",), "MODEL_LIMITED_FIT"),
        (("optimizer", "success"), False),
        (("sensitivity", "condition"), 1.0e6),
        (("provenance", "fit_adapter", "sha256"), "wrong"),
        (("provenance", "implementation", "sha256"), "wrong"),
        (("active_parameter_names",), ["outer_bi_antisite_fraction"]),
        (("full_parameter_vector",), [0.01, -0.001, 0.012, 0.1, 0.08]),
        (("cubature_oracle", "relative_l2_background_anchor_rows"), 0.0),
        (("cubature_oracle", "oracle_gauss_order"), 4),
    ):
        invalid = copy.deepcopy(document)
        target = invalid
        for key in path[:-1]:
            nested = target[key]
            assert isinstance(nested, dict)
            target = nested
        target[path[-1]] = value
        assert not ADAPTER.stage_fit_document_is_admissible(
            invalid,
            expected_stage="A",
            expected_active_parameter_names=ADAPTER.STRUCTURE_PARAMETER_NAMES[:2],
            diagnostic_sha256="diagnostic",
            trusted_recipe=TRUSTED_RECIPE,
            recipe_sha256="recipe",
            fit_plan_sha256="fit-plan",
            adapter_sha256="adapter",
            implementation_sha256="implementation",
            lower_bounds=STRUCTURE_LOWER_BOUNDS,
            upper_bounds=STRUCTURE_UPPER_BOUNDS,
            parameter_scales=STRUCTURE_PARAMETER_SCALES,
            sensitivity_relative_tolerance=1.0e-5,
            bound_proximity_in_parameter_scales=1.0e-6,
            maximum_sensitivity_condition=1.0e5,
        )

    missing_anchor = copy.deepcopy(document)
    del missing_anchor["cubature_oracle"]["relative_l2_background_anchor_rows"]
    assert not ADAPTER.stage_fit_document_is_admissible(
        missing_anchor,
        expected_stage="A",
        expected_active_parameter_names=ADAPTER.STRUCTURE_PARAMETER_NAMES[:2],
        diagnostic_sha256="diagnostic",
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        fit_plan_sha256="fit-plan",
        adapter_sha256="adapter",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
    )


def test_bound_limited_intermediate_stage_can_initialize_successor() -> None:
    document = _accepted_fit_document()
    active = ADAPTER.STRUCTURE_PARAMETER_NAMES[:2]
    parameters = np.asarray(document["full_parameter_vector"], dtype=np.float64)
    parameters[0] = STRUCTURE_LOWER_BOUNDS[0]
    document["full_parameter_vector"] = parameters.tolist()
    structure = document["structure_representative"]
    assert isinstance(structure, dict)
    structure[active[0]] = parameters[0]
    document["stage"] = "A"
    document["status"] = "MODEL_LIMITED_FIT"
    document["active_parameter_names"] = list(active)
    document["frozen_parameter_names"] = list(ADAPTER.STRUCTURE_PARAMETER_NAMES[2:])
    document["parameters_on_bounds"] = [active[0]]
    numerical = document["numerical_convergence"]
    assert isinstance(numerical, dict)
    numerical["converged"] = False
    numerical["cubature_converged_by_family"] = "NOT_RUN"
    document["cubature_oracle"] = {
        "performed": False,
        "status": "NOT_RUN_INITIALIZER",
        **ADAPTER._trusted_model_cubature(TRUSTED_RECIPE),
        "relative_l2": None,
        "relative_l2_by_family_m": {},
        "relative_l2_background_anchor_rows": None,
    }
    sensitivity = document["sensitivity"]
    assert isinstance(sensitivity, dict)
    sensitivity["rank"] = 2
    sensitivity["numerical_rank"] = 2
    sensitivity["parameter_count"] = 2
    sensitivity["parameter_names"] = list(active)
    sensitivity["parameter_scales"] = STRUCTURE_PARAMETER_SCALES[:2].tolist()
    execution = document["provenance"]["execution_identity"]
    assert isinstance(execution, dict)
    execution["stage"] = "A"
    execution["active_parameter_names"] = list(active)
    execution["frozen_parameter_names"] = list(ADAPTER.STRUCTURE_PARAMETER_NAMES[2:])
    _set_fit_start(
        document,
        execution_policy="staged_A_B_C_joint.v1",
        kind="default",
    )

    assert ADAPTER.stage_fit_document_is_admissible(
        document,
        expected_stage="A",
        expected_active_parameter_names=active,
        diagnostic_sha256="diagnostic",
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        fit_plan_sha256="fit-plan",
        adapter_sha256="adapter",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
    )


def _in_memory_stage_chain() -> tuple[
    dict[str, object],
    dict[str, tuple[dict[str, object], dict[str, str]]],
]:
    plan_path = ROOT / "examples" / "bi2se3" / "experiment" / "figure7_structure_fit.toml"
    plan = ADAPTER._validated_fit_plan(tomllib.loads(plan_path.read_text(encoding="utf-8")))
    predecessor_identity = None
    predecessor_chain: list[dict[str, str]] = []
    predecessor_parameters = np.asarray(plan["baseline_parameters"], dtype=np.float64)
    records = {}
    for stage in ("A", "B", "C", "joint"):
        document = _accepted_fit_document()
        active = tuple(plan["stage"][stage]["active_parameters"])
        active_index = np.asarray(
            [ADAPTER.STRUCTURE_PARAMETER_NAMES.index(name) for name in active], dtype=np.int64
        )
        final_parameters = np.array(predecessor_parameters, copy=True)
        candidate_parameters = np.asarray(document["full_parameter_vector"], dtype=np.float64)
        final_parameters[active_index] = candidate_parameters[active_index]
        _set_document_structure_vector(document, final_parameters)
        frozen = tuple(name for name in ADAPTER.STRUCTURE_PARAMETER_NAMES if name not in active)
        document["stage"] = stage
        document["status"] = "FIT" if stage == "joint" else "STAGE_CONDITIONED"
        document["active_parameter_names"] = list(active)
        document["frozen_parameter_names"] = list(frozen)
        if stage != "joint":
            document["numerical_convergence"]["cubature_converged_by_family"] = "NOT_RUN"
            document["cubature_oracle"] = {
                "performed": False,
                "status": "NOT_RUN_INITIALIZER",
                **ADAPTER._trusted_model_cubature(TRUSTED_RECIPE),
                "relative_l2": None,
                "relative_l2_by_family_m": {},
                "relative_l2_background_anchor_rows": None,
            }
        sensitivity = document["sensitivity"]
        assert isinstance(sensitivity, dict)
        sensitivity["rank"] = len(active)
        sensitivity["numerical_rank"] = len(active)
        sensitivity["parameter_count"] = len(active)
        sensitivity["parameter_names"] = list(active)
        sensitivity["parameter_scales"] = [
            STRUCTURE_PARAMETER_SCALES[ADAPTER.STRUCTURE_PARAMETER_NAMES.index(name)]
            for name in active
        ]
        optimizer = document["optimizer"]
        assert isinstance(optimizer, dict)
        optimizer["fit_start"] = {
            "initial_full_parameters": predecessor_parameters.tolist(),
        }
        provenance = document["provenance"]
        assert isinstance(provenance, dict)
        provenance["predecessor"] = copy.deepcopy(predecessor_identity)
        provenance["predecessor_chain"] = copy.deepcopy(predecessor_chain)
        execution = provenance["execution_identity"]
        assert isinstance(execution, dict)
        execution["stage"] = stage
        execution["active_parameter_names"] = list(active)
        execution["frozen_parameter_names"] = list(frozen)
        execution["predecessor_sha256"] = (
            None if predecessor_identity is None else predecessor_identity["sha256"]
        )
        execution["predecessor_chain_sha256"] = [
            identity["sha256"] for identity in predecessor_chain
        ]
        _set_fit_start(
            document,
            execution_policy="staged_A_B_C_joint.v1",
            kind="default",
            initial_full_parameters=predecessor_parameters,
        )
        identity = {"path": stage, "sha256": f"sha-{stage}"}
        records[stage] = (document, identity)
        predecessor_identity = identity
        predecessor_chain = [identity, *predecessor_chain]
        predecessor_parameters = np.asarray(document["full_parameter_vector"], dtype=np.float64)
    return plan, records


def _qualify_in_memory_chain(
    plan: dict[str, object],
    records: dict[str, tuple[dict[str, object], dict[str, str]]],
    restart_records: dict[str, tuple[dict[str, object], dict[str, str]]] | None = None,
) -> tuple[tuple[dict[str, object], dict[str, str]], ...]:
    def load_predecessor(
        recorded_identity: object,
        stage: str,
    ) -> tuple[dict[str, object], dict[str, str]]:
        document, identity = records[stage]
        if recorded_identity != identity:
            raise ValueError("stale recorded identity")
        return document, identity

    def load_restart(
        recorded_identity: object,
        stage: str,
    ) -> tuple[dict[str, object], dict[str, str]]:
        if restart_records is None or stage not in restart_records:
            raise ValueError("missing restart record")
        document, identity = restart_records[stage]
        if recorded_identity != identity:
            raise ValueError("stale restart identity")
        return document, identity

    joint, joint_identity = records["joint"]
    return ADAPTER._qualify_stage_chain_documents(
        joint,
        joint_identity,
        expected_stage="joint",
        fit_plan=plan,
        diagnostic_sha256="diagnostic",
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        fit_plan_sha256="fit-plan",
        adapter_sha256="adapter",
        implementation_sha256="implementation",
        load_predecessor=load_predecessor,
        load_restart=load_restart,
    )


def test_stage_chain_requires_exact_ordered_predecessors() -> None:
    plan, records = _in_memory_stage_chain()

    chain = _qualify_in_memory_chain(plan, records)

    assert tuple(document["stage"] for document, _ in chain) == ("joint", "C", "B", "A")


def test_stage_chain_rejects_unversioned_active_warm_start() -> None:
    plan, records = _in_memory_stage_chain()
    stage_b = records["B"][0]
    warm = np.asarray(stage_b["optimizer"]["fit_start"]["initial_full_parameters"])
    warm[2] += 0.001
    _set_fit_start(
        stage_b,
        execution_policy="staged_A_B_C_joint.v1",
        kind="default",
        initial_full_parameters=warm,
    )

    with pytest.raises(ValueError, match="did not start from its predecessor"):
        _qualify_in_memory_chain(plan, records)


def test_stage_chain_accepts_same_stage_progress_restart() -> None:
    plan, records = _in_memory_stage_chain()
    stage_b = records["B"][0]
    warm = np.asarray(stage_b["optimizer"]["fit_start"]["initial_full_parameters"])
    warm[2] += 0.001
    restart_identity = {"path": "B.progress.json", "sha256": "a" * 64}
    _set_fit_start(
        stage_b,
        execution_policy="staged_A_B_C_joint.v1",
        kind="progress_restart",
        initial_full_parameters=warm,
        source_artifact=restart_identity,
    )
    restart_document = {
        "schema_version": ADAPTER.FIT_PROGRESS_SCHEMA,
        "execution_identity": copy.deepcopy(stage_b["provenance"]["execution_identity"]),
        "fit_run_identity": {
            "model_execution": copy.deepcopy(stage_b["provenance"]["execution_identity"]),
            "fit_start": {"kind": "default"},
        },
        "active_parameters": [warm[2]],
        "full_parameters": warm.tolist(),
        "completed_model_evaluations": 1,
        "devices": ["cpu"],
        "evaluated_backends": ["numba_cpu"],
    }

    chain = _qualify_in_memory_chain(
        plan,
        records,
        restart_records={"B": (restart_document, restart_identity)},
    )
    assert tuple(document["stage"] for document, _ in chain) == ("joint", "C", "B", "A")

    restart_document["active_parameters"] = [warm[2] + 0.001]
    with pytest.raises(ValueError, match="restart source is not qualified"):
        _qualify_in_memory_chain(
            plan,
            records,
            restart_records={"B": (restart_document, restart_identity)},
        )


def test_stage_chain_rejects_changed_child_start_and_stale_ancestor() -> None:
    plan, records = _in_memory_stage_chain()
    joint = records["joint"][0]
    changed_start = np.asarray(
        joint["optimizer"]["fit_start"]["initial_full_parameters"],
        dtype=np.float64,
    )
    changed_start[0] += 1.0e-4
    _set_fit_start(
        joint,
        execution_policy="staged_A_B_C_joint.v1",
        kind="default",
        initial_full_parameters=changed_start,
    )
    with pytest.raises(ValueError, match="did not start"):
        _qualify_in_memory_chain(plan, records)

    plan, records = _in_memory_stage_chain()
    records["B"][0]["provenance"]["predecessor"]["sha256"] = "stale"
    with pytest.raises(ValueError, match="stale recorded identity"):
        _qualify_in_memory_chain(plan, records)


def test_stage_chain_rejects_child_that_changes_a_frozen_parameter() -> None:
    plan, records = _in_memory_stage_chain()
    records["C"][0]["full_parameter_vector"][0] += 1.0e-4
    records["C"][0]["structure_representative"]["bi_delta_z_fractional"] += 1.0e-4
    with pytest.raises(ValueError, match="stage C is not qualified"):
        _qualify_in_memory_chain(plan, records)


def test_stage_chain_rejects_root_stage_that_changes_a_frozen_parameter() -> None:
    plan, records = _in_memory_stage_chain()
    stage_a = records["A"][0]
    stage_a["full_parameter_vector"][2] += 0.001
    stage_a["structure_representative"]["outer_chalcogen_vacancy_fraction"] += 0.001
    stage_a["structure_representative"]["outer_chalcogen_occupancy"] -= 0.001
    stage_a["structure_representative"]["outer_chalcogen_fraction"] -= 0.001

    with pytest.raises(ValueError, match="stage A is not qualified"):
        _qualify_in_memory_chain(plan, records)


def test_stage_chain_rejects_internally_stale_lineage() -> None:
    plan, records = _in_memory_stage_chain()
    records["joint"][0]["provenance"]["execution_identity"]["predecessor_chain_sha256"][-1] = (
        "stale"
    )
    records["joint"][0]["provenance"]["fit_run_identity"]["model_execution"] = copy.deepcopy(
        records["joint"][0]["provenance"]["execution_identity"]
    )

    with pytest.raises(ValueError, match="lineage is internally inconsistent"):
        _qualify_in_memory_chain(plan, records)


def test_stage_chain_rejects_a_changed_frozen_observation_objective() -> None:
    plan, records = _in_memory_stage_chain()
    records["B"][0]["data_projection"]["fit_projection_revisions"][0] = "9" * 64

    with pytest.raises(ValueError, match="frozen observation objective"):
        _qualify_in_memory_chain(plan, records)


def test_recipe_peak_coordinates_and_horizon_catalog_are_consistent() -> None:
    path = ROOT / "examples" / "bi2se3" / "experiment" / "figure7_matched_regions.toml"
    recipe = tomllib.loads(path.read_text(encoding="utf-8"))

    validated = ADAPTER._validated_recipe(recipe)
    horizon = ADAPTER._detector_horizon_acceptance(validated["horizon_gate"])
    assert math.degrees(horizon.offspecular_air_exit_guard_rad) == pytest.approx(1.0)
    assert validated["horizon_gate"]["diffraction_peak_air_exit_guard_deg"] == 1.0
    assert validated["parratt_stitch"]["top_roughness_A"] == pytest.approx(5.23725139)
    assert validated["parratt_stitch"]["bottom_roughness_A"] == pytest.approx(10.0)
    exclusions = {
        (item["dataset_id"], item["identity"]): item["reason"]
        for item in validated["excluded_peak"]
    }
    assert exclusions[("Bi2Se3-15deg", "m0-003")] == "below exit horizon"
    assert exclusions[("Bi2Se3-15deg", "m0-006")] == "horizon-clearance gate"

    invalid = copy.deepcopy(recipe)
    invalid["fit_peak"][0]["coordinate_kind"] = "L"
    with pytest.raises(ValueError, match="coordinate_kind"):
        ADAPTER._validated_recipe(invalid)

    invalid = copy.deepcopy(recipe)
    invalid["fit_peak"][0]["dataset_id"] = "unknown"
    with pytest.raises(ValueError, match="dataset_id"):
        ADAPTER._validated_recipe(invalid)

    for invalid_scale in (True, "1.0"):
        invalid = copy.deepcopy(recipe)
        invalid["dark_correction"]["scale"] = invalid_scale
        with pytest.raises(ValueError, match="explicit no-clip dark correction"):
            ADAPTER._validated_recipe(invalid)

    external = copy.deepcopy(recipe)
    external["parratt_stitch"]["interface_assumption"] = "fixed_external_qz_m0_strength.v1"
    assert (
        ADAPTER._validated_recipe(external)["parratt_stitch"]["interface_assumption"]
        == "fixed_external_qz_m0_strength.v1"
    )
    external["parratt_stitch"]["interface_assumption"] = "unknown"
    with pytest.raises(ValueError, match="interface assumption"):
        ADAPTER._validated_recipe(external)


def test_profile_selection_coordinate_contract_is_two_theta_then_L() -> None:
    arrays = {
        "profile_identity": np.asarray(
            ("m0", "m1_minus", "m1_plus", "m3_minus", "m3_plus", "m4_minus", "m4_plus")
        ),
        "profile_selection_coordinate": np.asarray((0.05, 2.3, 3.1, 6.7, 7.1, 12.9, np.nan)),
        "profile_selection_coordinate_kind": np.asarray(
            ("two_theta_deg", "L", "L", "L", "L", "L", "L")
        ),
        "profile_valid": np.asarray((True, True, True, True, True, True, False)),
    }

    ADAPTER._validate_profile_selection_coordinates(arrays)
    invalid = copy.deepcopy(arrays)
    invalid["profile_selection_coordinate_kind"][2] = "Qz_Ainv"
    with pytest.raises(ValueError, match="profile selection coordinate"):
        ADAPTER._validate_profile_selection_coordinates(invalid)

    invalid = copy.deepcopy(arrays)
    invalid["profile_selection_coordinate"][0] = np.nan
    with pytest.raises(ValueError, match="profile selection coordinate"):
        ADAPTER._validate_profile_selection_coordinates(invalid)


def test_fit_conditioned_profile_policy_accepts_complete_fit_evidence() -> None:
    policy = ADAPTER._profile_evidence_policy(
        fit_document=_accepted_fit_document(),
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        adapter_sha256="adapter",
        fit_plan_sha256="fit-plan",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
        expected_dataset_ids=("a", "b", "c"),
        predecessor_chain_complete=True,
    )

    assert policy == {
        "evidence_level": "FIT_CONDITIONED",
        "publication_ready": False,
        "all_rod_validation": "NOT_REQUIRED_FOR_RENDER",
        "model_rod_scope": "fitted_families_m_0_1_3_4",
        "pass_names": ("fit",),
        "result_pass_name": "fit",
    }


def test_fit_conditioned_profile_policy_accepts_declared_bound_limited_joint() -> None:
    document = _accepted_fit_document()
    document["status"] = "MODEL_LIMITED_FIT"
    document["parameters_on_bounds"] = ["outer_chalcogen_vacancy_fraction"]
    document["full_parameter_vector"][2] = 0.03
    document["structure_representative"]["outer_chalcogen_vacancy_fraction"] = 0.03
    document["structure_representative"]["outer_chalcogen_occupancy"] = 0.97
    document["structure_representative"]["outer_chalcogen_fraction"] = 0.97
    document["numerical_convergence"]["converged"] = False

    policy = ADAPTER._profile_evidence_policy(
        fit_document=document,
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        adapter_sha256="adapter",
        fit_plan_sha256="fit-plan",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
        expected_dataset_ids=("a", "b", "c"),
        predecessor_chain_complete=True,
    )

    assert policy["evidence_level"] == "FIT_CONDITIONED"
    assert policy["publication_ready"] is False

    document["numerical_convergence"]["cubature_converged_by_family"] = False
    document["cubature_oracle"]["relative_l2_by_family_m"]["3"] = 0.04
    with pytest.raises(ValueError, match="FIT_CONDITIONED"):
        ADAPTER._profile_evidence_policy(
            fit_document=document,
            trusted_recipe=TRUSTED_RECIPE,
            recipe_sha256="recipe",
            adapter_sha256="adapter",
            fit_plan_sha256="fit-plan",
            implementation_sha256="implementation",
            lower_bounds=STRUCTURE_LOWER_BOUNDS,
            upper_bounds=STRUCTURE_UPPER_BOUNDS,
            parameter_scales=STRUCTURE_PARAMETER_SCALES,
            sensitivity_relative_tolerance=1.0e-5,
            bound_proximity_in_parameter_scales=1.0e-6,
            maximum_sensitivity_condition=1.0e5,
            expected_dataset_ids=("a", "b", "c"),
            predecessor_chain_complete=True,
        )

    document["cubature_oracle"]["relative_l2_by_family_m"]["3"] = 0.01
    document["numerical_convergence"]["cubature_converged_by_family"] = True
    document["cubature_oracle"]["relative_l2_background_anchor_rows"] = 0.04
    policy = ADAPTER._profile_evidence_policy(
        fit_document=document,
        trusted_recipe=TRUSTED_RECIPE,
        recipe_sha256="recipe",
        adapter_sha256="adapter",
        fit_plan_sha256="fit-plan",
        implementation_sha256="implementation",
        lower_bounds=STRUCTURE_LOWER_BOUNDS,
        upper_bounds=STRUCTURE_UPPER_BOUNDS,
        parameter_scales=STRUCTURE_PARAMETER_SCALES,
        sensitivity_relative_tolerance=1.0e-5,
        bound_proximity_in_parameter_scales=1.0e-6,
        maximum_sensitivity_condition=1.0e5,
        expected_dataset_ids=("a", "b", "c"),
        predecessor_chain_complete=True,
    )
    assert policy["evidence_level"] == "FIT_CONDITIONED"


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("status",), "MODEL_LIMITED_FIT"),
        (("model_pixelized",), True),
        (("smoothing_applied",), True),
        (("provenance", "execution_identity", "rod_scope"), "all_configured_rods"),
        (("provenance", "execution_identity", "model_measure"), "native_detector_pixels"),
        (("structure_representative", "intensity_envelope_u_normal_A2"), float("nan")),
        (("dataset_scales", "b"), 0.0),
    ),
)
def test_fit_conditioned_profile_policy_fails_closed(path: tuple[str, ...], value: object) -> None:
    document = _accepted_fit_document()
    target: dict[str, object] = document
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value

    with pytest.raises(ValueError, match="FIT_CONDITIONED"):
        ADAPTER._profile_evidence_policy(
            fit_document=document,
            trusted_recipe=TRUSTED_RECIPE,
            recipe_sha256="recipe",
            adapter_sha256="adapter",
            fit_plan_sha256="fit-plan",
            implementation_sha256="implementation",
            lower_bounds=STRUCTURE_LOWER_BOUNDS,
            upper_bounds=STRUCTURE_UPPER_BOUNDS,
            parameter_scales=STRUCTURE_PARAMETER_SCALES,
            sensitivity_relative_tolerance=1.0e-5,
            bound_proximity_in_parameter_scales=1.0e-6,
            maximum_sensitivity_condition=1.0e5,
            expected_dataset_ids=("a", "b", "c"),
            predecessor_chain_complete=True,
        )


def _accepted_profile_manifest() -> dict[str, object]:
    fit_document = _accepted_fit_document()
    roster = ((0, 0, 0, 1.0), (1, 0, 1, 1.0), (1, 1, 3, 1.0), (2, 0, 4, 1.0))
    roster_sha256 = hashlib.sha256(
        json.dumps(roster, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()
    background_model = copy.deepcopy(fit_document["background_model"])
    execution = copy.deepcopy(fit_document["provenance"]["execution_identity"])
    execution["fit_gauss_order"] = TRUSTED_RECIPE["profile_cubature"]["fit_gauss_order"]
    execution["fit_subdivision_count"] = TRUSTED_RECIPE["profile_cubature"]["fold_fit_subdivisions"]
    execution["m0_model_phi_subdivision_count"] = TRUSTED_RECIPE["profile_cubature"][
        "fold_fit_subdivisions"
    ]
    execution.update(
        {
            "rod_scope": "fitted_families_m_0_1_3_4",
            "rod_count": len(roster),
            "rod_roster_sha256": roster_sha256,
            "diagnostic_sha256": "d" * 64,
            "fit_sha256": "f" * 64,
            "recipe_sha256": "e" * 64,
            "fit_plan_sha256": "1" * 64,
            "implementation_sha256": "b" * 64,
            "osc_sha256": "c" * 64,
            "dark_osc_sha256": "9" * 64,
            "dark_scale": 1.0,
            "fit_chain_sha256": ["7" * 64],
            "m0_signal_only_projection_revision": "a" * 64,
            "m0_signal_only_measured_quadrature_revision": "b" * 64,
            "m0_signal_only_model_quadrature_revision": "0" * 64,
            "m0_signal_only_bin_count": 2,
            "m0_signal_only_count_mass_sha256": "c" * 64,
            "m0_signal_only_count_covariance_sha256": "d" * 64,
            "m0_signal_only_radial_background_mass_sha256": "e" * 64,
            "m0_signal_only_model_mass_sha256": "f" * 64,
        }
    )
    return {
        "computationally_valid": True,
        "model_pixelized": False,
        "smoothing_applied": False,
        "model_rod_count": len(roster),
        "model_rod_roster_h_k_m_population": roster,
        "model_rod_roster_sha256": roster_sha256,
        "fit_model_rod_roster_sha256": roster_sha256,
        "evidence_level": "FIT_CONDITIONED",
        "publication_ready": False,
        "fit_status": "FIT",
        "all_rod_validation": "NOT_REQUIRED_FOR_RENDER",
        "rod_scope_validation_status": "NOT_RUN",
        "model_rod_scope": "fitted_families_m_0_1_3_4",
        "structure_representative": copy.deepcopy(fit_document["structure_representative"]),
        "stacking_model": ADAPTER._fault_free_three_r_definition(),
        "dark_correction": {
            "model_id": ADAPTER.DARK_CORRECTION_MODEL,
            "file_sha256": "9" * 64,
            "detector_native_bytes_sha256": "8" * 64,
            "detector_native_shape_rc": [3000, 3000],
            "detector_native_dtype": "int32",
            "scale": 1.0,
            "scale_basis": "matched_exposure_assumed.v1",
            "negative_values_clipped": False,
            "smoothing_applied": False,
            "covariance_model": "shared_independent_poisson_dark_across_datasets.v1",
        },
        "figure_recipe": copy.deepcopy(TRUSTED_RECIPE),
        "m0_signal_only_display": {
            "status": "DISPLAY_ONLY_INCOMPLETE_SIDEBAND_SUPPLEMENT",
            "supplemental_bin_count": 2,
            "fit_role": "not_used_in_fit_objective_or_parameter_estimation",
            "background_conditioning": "fixed_radial_only_no_adjacent_sideband_conditioning",
            "projection_method": ADAPTER.MEASURED_PROJECTION_METHOD,
            "projection_revision": "a" * 64,
            "measured_quadrature_revision": "b" * 64,
            "model_quadrature_revision": "0" * 64,
            "gauss_order": TRUSTED_RECIPE["model_cubature"]["oracle_gauss_order"],
            "subdivision_count": TRUSTED_RECIPE["model_cubature"]["fold_oracle_subdivisions"],
            "measured_phi_subdivision_count": TRUSTED_RECIPE["model_cubature"][
                "fold_oracle_subdivisions"
            ],
            "model_phi_subdivision_count": TRUSTED_RECIPE["model_cubature"][
                "fold_oracle_subdivisions"
            ],
            "measured_continuous_node_count": 32,
            "model_continuous_node_count": 32,
            "count_mass_sha256": "c" * 64,
            "count_covariance_sha256": "d" * 64,
            "radial_background_mass_sha256": "e" * 64,
            "model_mass_sha256": "f" * 64,
            "detector_overlay_flat_pixel_count": 10,
            "detector_overlay_flat_pixel_sha256": "7" * 64,
            "execution": {"model_measure": "continuous_detector_chart_area"},
            "smoothing_applied": False,
            "model_pixelized": False,
        },
        "background_model": background_model,
        "profile_cubature": {
            "settings": copy.deepcopy(TRUSTED_RECIPE["profile_cubature"]),
            "m0_model_phi_subdivision_count": TRUSTED_RECIPE["profile_cubature"][
                "fold_fit_subdivisions"
            ],
            "evaluated_passes": ["fit"],
            "full_profile_oracle_performed": False,
            "fit_artifact_fitted_region_oracle": copy.deepcopy(fit_document["cubature_oracle"]),
            "fit_artifact_data_projection": copy.deepcopy(fit_document["data_projection"]),
        },
        "measured_data_projection": {
            "method": ADAPTER.MEASURED_PROJECTION_METHOD,
            "coarse_projection_revision": "1" * 64,
            "projection_revision": "2" * 64,
            "display_fit_projection_revision": "3" * 64,
            "quadrature_revision": "4" * 64,
            "count_mass_sha256": "5" * 64,
            "count_covariance_sha256": "6" * 64,
            "gauss_order": TRUSTED_RECIPE["model_cubature"]["oracle_gauss_order"],
            "subdivision_count": TRUSTED_RECIPE["model_cubature"]["fold_oracle_subdivisions"],
            "m0_phi_subdivision_count": TRUSTED_RECIPE["model_cubature"][
                "fold_oracle_subdivisions"
            ],
            "refinement_oracle": {
                **_projection_refinement(("a",)),
                "acceptance_measure": "display_profile_pooled_count_mass_and_support.v1",
                "covariance_policy": "refined_covariance_is_authoritative_for_display",
            },
            "smoothing_applied": False,
        },
        "provenance": {
            "background_sha256": "0" * 64,
            "fit_diagnostic_sha256": "d" * 64,
            "fit_sha256": "f" * 64,
            "recipe_sha256": "e" * 64,
            "fit_plan": {"sha256": "1" * 64},
            "osc_sha256": "c" * 64,
            "dark_osc_sha256": "9" * 64,
            "fit_chain": [{"sha256": "7" * 64}],
            "fit_origin_adapter_sha256": "a" * 64,
            "fit_origin_implementation_sha256": "b" * 64,
            "profile_adapter_sha256": "a" * 64,
            "implementation": {"sha256": "b" * 64},
            "execution_identity": execution,
        },
    }


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("figure_recipe", "display_dataset_id"), "b"),
        (("profile_cubature", "settings", "fit_gauss_order"), 3),
        (("profile_cubature", "fit_artifact_fitted_region_oracle", "status"), "PARTIAL"),
        (("profile_cubature", "fit_artifact_data_projection", "maximum_relative_l2"), 1.0),
        (("measured_data_projection", "refinement_oracle", "converged"), False),
        (
            (
                "measured_data_projection",
                "refinement_oracle",
                "by_dataset",
                "a",
                "mass_relative_l2",
            ),
            0.04,
        ),
        (("background_model", "conditioned_covariance_sha256"), "9" * 64),
        (("fit_model_rod_roster_sha256",), "9" * 64),
        (("background_model", "conditioned_revision"), ""),
        (("dark_correction", "file_sha256"), "7" * 64),
        (("dark_correction", "path"), "C:/wrong-dark.osc.gz"),
        (("dark_correction", "detector_native_bytes_sha256"), "6" * 64),
        (("dark_correction", "detector_native_shape_rc"), [1, 1]),
        (("dark_correction", "detector_native_dtype"), "float32"),
        (("dark_correction", "covariance_model"), "independent_per_dataset"),
        (("dark_correction", "scale_basis"), "invented_basis.v1"),
        (("m0_signal_only_display", "background_conditioning"), "sideband-subtracted"),
    ),
)
def test_profile_manifest_admission_is_bound_to_trusted_evidence(
    path: tuple[str, ...],
    value: object,
) -> None:
    accepted = _accepted_profile_manifest()
    prepared_dark = copy.deepcopy(accepted["dark_correction"])
    assert ADAPTER._profile_manifest_is_admissible(
        accepted,
        trusted_recipe=TRUSTED_RECIPE,
        prepared_dark_correction=prepared_dark,
    )
    target: dict[str, object] = accepted
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value

    assert not ADAPTER._profile_manifest_is_admissible(
        accepted,
        trusted_recipe=TRUSTED_RECIPE,
        prepared_dark_correction=prepared_dark,
    )


def test_profile_manifest_retains_covariance_refinement_as_diagnostic() -> None:
    accepted = _accepted_profile_manifest()
    prepared_dark = copy.deepcopy(accepted["dark_correction"])
    refinement = accepted["measured_data_projection"]["refinement_oracle"]
    refinement["covariance_refinement_converged"] = False
    refinement["by_dataset"]["a"]["maximum_covariance_row_relative_l2"] = 8.0

    assert ADAPTER._profile_manifest_is_admissible(
        accepted,
        trusted_recipe=TRUSTED_RECIPE,
        prepared_dark_correction=prepared_dark,
    )


def test_profile_manifest_admits_only_the_declared_zero_dark_contract() -> None:
    trusted_recipe = copy.deepcopy(TRUSTED_RECIPE)
    trusted_recipe["dark_correction"].update(
        {
            "scale": 0.0,
            "scale_basis": "no_acquisition_matched_dark.v1",
        }
    )
    accepted = _accepted_profile_manifest()
    accepted["figure_recipe"] = copy.deepcopy(trusted_recipe)
    accepted["dark_correction"].update(
        {
            "scale": 0.0,
            "scale_basis": "no_acquisition_matched_dark.v1",
            "covariance_model": "no_dark_contribution.v1",
        }
    )
    accepted["provenance"]["execution_identity"]["dark_scale"] = 0.0
    prepared_dark = copy.deepcopy(accepted["dark_correction"])

    assert ADAPTER._profile_manifest_is_admissible(
        accepted,
        trusted_recipe=trusted_recipe,
        prepared_dark_correction=prepared_dark,
    )
    for key, wrong_value in (
        ("scale_basis", "matched_exposure_assumed.v1"),
        ("covariance_model", "shared_independent_poisson_dark_across_datasets.v1"),
    ):
        tampered = copy.deepcopy(accepted)
        tampered["dark_correction"][key] = wrong_value
        assert not ADAPTER._profile_manifest_is_admissible(
            tampered,
            trusted_recipe=trusted_recipe,
            prepared_dark_correction=prepared_dark,
        )


def test_profile_manifest_accepts_an_empty_m0_signal_only_supplement() -> None:
    accepted = _accepted_profile_manifest()
    prepared_dark = copy.deepcopy(accepted["dark_correction"])
    supplement = accepted["m0_signal_only_display"]
    supplement.update(
        {
            "status": "NOT_REQUIRED_ALL_M0_BINS_CONDITIONED",
            "supplemental_bin_count": 0,
            "projection_method": None,
            "projection_revision": None,
            "measured_quadrature_revision": None,
            "model_quadrature_revision": None,
            "measured_continuous_node_count": 0,
            "model_continuous_node_count": 0,
            "detector_overlay_flat_pixel_count": 0,
            "execution": None,
        }
    )
    execution = accepted["provenance"]["execution_identity"]
    execution.update(
        {
            "m0_signal_only_bin_count": 0,
            "m0_signal_only_projection_revision": None,
            "m0_signal_only_measured_quadrature_revision": None,
            "m0_signal_only_model_quadrature_revision": None,
        }
    )

    assert ADAPTER._profile_manifest_is_admissible(
        accepted,
        trusted_recipe=TRUSTED_RECIPE,
        prepared_dark_correction=prepared_dark,
    )
    supplement["execution"] = {}
    assert not ADAPTER._profile_manifest_is_admissible(
        accepted,
        trusted_recipe=TRUSTED_RECIPE,
        prepared_dark_correction=prepared_dark,
    )


def test_profile_manifest_accepts_only_declared_frozen_parameter_replay() -> None:
    trusted_recipe = copy.deepcopy(TRUSTED_RECIPE)
    trusted_recipe["parratt_stitch"] = {
        "interface_assumption": ADAPTER.FIXED_EXTERNAL_QZ_INTERFACE,
    }
    accepted = _accepted_profile_manifest()
    prepared_dark = copy.deepcopy(accepted["dark_correction"])
    execution = accepted["provenance"]["execution_identity"]
    structure_parameters = np.asarray((0.001, -0.001, 0.012, 0.004, 0.006))
    dataset_scales = {"a": 2.0, "b": 3.0, "c": 4.0}
    replay = {
        "method": ADAPTER.PROFILE_PARAMETER_REPLAY_METHOD,
        "status": "COMPLETE",
        "fit_role": "frozen_parameter_and_dataset_scale_source_only",
        "profile_role": "continuous_profile_recalculation_without_optimization",
        "scope": "m0_specular_interface_assumption_only",
        "fit_parameters_reused": True,
        "optimizer_executed": False,
        "fit_reexecuted": False,
        "profile_reexecuted": True,
        "objective_requalified_under_profile_model": False,
        "dataset_scales_requalified": False,
        "source_fit_sha256": "f" * 64,
        "source_fit_status": "FIT",
        "structure_parameter_vector_sha256": ADAPTER._array_sha256(structure_parameters),
        "dataset_scales": dataset_scales,
        "dataset_scale_vector_sha256": ADAPTER._array_sha256(
            np.asarray(tuple(dataset_scales.values()))
        ),
        "fit_interface_assumption": ADAPTER.FIXED_EXTERNAL_QZ_INTERFACE,
        "profile_interface_assumption": ADAPTER.LOCAL_LAMELLA_INTERFACE,
    }
    accepted.update(
        evidence_level=ADAPTER.PROFILE_PARAMETER_REPLAY_EVIDENCE,
        dataset_scale=2.0,
        fit_parameter_replay=replay,
        figure_recipe=ADAPTER._profile_recipe_with_specular_interface(
            trusted_recipe,
            ADAPTER.LOCAL_LAMELLA_INTERFACE,
        ),
    )
    accepted["provenance"].update(
        profile_adapter_sha256="c" * 64,
        implementation={"sha256": "d" * 64},
    )
    execution.update(
        structure_parameters=structure_parameters.tolist(),
        fit_parameter_replay=replay,
        profile_adapter_sha256="c" * 64,
        fit_origin_adapter_sha256="a" * 64,
        fit_origin_implementation_sha256="b" * 64,
        implementation_sha256="d" * 64,
    )

    assert ADAPTER._profile_manifest_is_admissible(
        accepted,
        trusted_recipe=trusted_recipe,
        prepared_dark_correction=prepared_dark,
    )
    replay["optimizer_executed"] = True
    assert not ADAPTER._profile_manifest_is_admissible(
        accepted,
        trusted_recipe=trusted_recipe,
        prepared_dark_correction=prepared_dark,
    )


def test_profiles_cli_supports_explicit_frozen_parameter_replay() -> None:
    arguments = ADAPTER._parser().parse_args(
        [
            "profiles",
            "--diagnostic",
            "prepared.ra_diag.npz",
            "--background",
            "background.ra_diag.npz",
            "--fit",
            "fit.json",
            "--destination",
            "profiles.ra_diag.npz",
            "--specular-interface-assumption",
            ADAPTER.LOCAL_LAMELLA_INTERFACE,
        ]
    )

    assert arguments.command == "profiles"
    assert arguments.destination == Path("profiles.ra_diag.npz")
    assert arguments.specular_interface_assumption == ADAPTER.LOCAL_LAMELLA_INTERFACE
    assert not hasattr(arguments, "reuse_fit_parameters")
    assert not hasattr(arguments, "reuse_profile_diagnostic")


def test_profile_specular_replay_changes_only_the_interface_assumption() -> None:
    recipe = {
        "dataset_ids": ["a"],
        "parratt_stitch": {
            "model_id": "empirical_parratt_kinematic_strength.v1",
            "interface_assumption": ADAPTER.FIXED_EXTERNAL_QZ_INTERFACE,
            "top_roughness_A": 5.0,
        },
    }

    replay = ADAPTER._profile_recipe_with_specular_interface(
        recipe,
        ADAPTER.LOCAL_LAMELLA_INTERFACE,
    )

    assert recipe["parratt_stitch"]["interface_assumption"] == (ADAPTER.FIXED_EXTERNAL_QZ_INTERFACE)
    assert replay == {
        **recipe,
        "parratt_stitch": {
            **recipe["parratt_stitch"],
            "interface_assumption": ADAPTER.LOCAL_LAMELLA_INTERFACE,
        },
    }


def test_background_cli_requires_the_exact_structure_fit_plan() -> None:
    arguments = ADAPTER._parser().parse_args(
        [
            "background",
            "--diagnostic",
            "prepared.ra_diag.npz",
            "--fit-plan",
            "structure.toml",
            "--destination",
            "background.ra_diag.npz",
        ]
    )

    assert arguments.fit_plan == Path("structure.toml")


def test_pixel_cell_boundary_segments_follow_exact_native_pixel_edges() -> None:
    mask = np.asarray(((True, True), (False, True)), dtype=np.bool_)

    segments = ADAPTER._pixel_cell_boundary_segments(mask)
    actual = {
        tuple(tuple(float(coordinate) for coordinate in point) for point in segment)
        for segment in segments
    }
    expected = {
        ((-0.5, -0.5), (1.5, -0.5)),
        ((-0.5, 0.5), (0.5, 0.5)),
        ((0.5, 1.5), (1.5, 1.5)),
        ((-0.5, -0.5), (-0.5, 0.5)),
        ((0.5, 0.5), (0.5, 1.5)),
        ((1.5, -0.5), (1.5, 1.5)),
    }

    assert actual == expected
    assert ADAPTER._pixel_cell_boundary_segments(np.zeros((2, 3), dtype=np.bool_)).shape == (
        0,
        2,
        2,
    )


def test_region_display_row_spans_fill_each_detector_branch_independently() -> None:
    mask = np.zeros((2, 10), dtype=np.bool_)
    mask[0, (1, 3, 6, 8)] = True
    mask[1, 2:5] = True

    split = ADAPTER._filled_region_row_spans(mask, split_column=5)
    unsplit = ADAPTER._filled_region_row_spans(mask, split_column=None)

    assert np.array_equal(np.flatnonzero(split[0]), np.asarray((1, 2, 3, 6, 7, 8)))
    assert np.array_equal(np.flatnonzero(split[1]), np.asarray((2, 3, 4)))
    assert np.array_equal(np.flatnonzero(unsplit[0]), np.arange(1, 9))


def test_m0_is_rendered_only_from_unified_profile_field() -> None:
    parser = ADAPTER._parser()
    subparser_action = next(
        action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
    )
    assert "m0-low-angle" not in subparser_action.choices
    render_options = {
        option
        for action in subparser_action.choices["render"]._actions
        for option in action.option_strings
    }
    assert "--m0-count-calibration" not in render_options
    assert "--m0-low-angle-comparison" not in render_options
    assert tuple(inspect.signature(ADAPTER.render).parameters) == (
        "profile_diagnostic_path",
        "output_directory",
    )


def test_m0_signal_only_display_fills_only_conditioning_gaps() -> None:
    selected = ADAPTER._m0_signal_only_display_mask(
        np.asarray((0, 1, 2, 3, 4)),
        np.asarray((True, True, True, False, True)),
        np.asarray(("m0", "m0", "m1_plus")),
        np.asarray((1, 2, 4)),
        np.asarray((True, False, True)),
    )

    np.testing.assert_array_equal(selected, np.asarray((True, False, True, False, True)))
    empty = ADAPTER._m0_signal_only_display_mask(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.bool_),
        np.asarray(("m0",)),
        np.asarray((0,), dtype=np.int64),
        np.asarray((True,), dtype=np.bool_),
    )
    assert empty.dtype == np.bool_
    assert empty.shape == (0,)


def test_five_coordinate_structure_adapter_separates_site_adps_and_intensity_envelope() -> None:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        load_simulation_config,
    )

    configured = build_configured_simulation_inputs(
        load_simulation_config(ROOT / "configs" / "bi2se3_r3_simulation.yaml")
    )
    baseline = configured.strength.structure_parameters
    plan = ADAPTER._load_fit_plan(
        ROOT / "examples" / "bi2se3" / "experiment" / "figure7_structure_fit.toml"
    )
    candidate_strength = ADAPTER._candidate_strength(
        configured.strength,
        np.asarray((0.001, -0.002, 0.012, 0.007, 0.011)),
        fit_plan=plan,
    )
    candidate_envelope = ADAPTER._candidate_intensity_envelope(
        np.asarray((0.001, -0.002, 0.012, 0.007, 0.011))
    )
    candidate = candidate_strength.structure_parameters

    assert candidate.bi_fractional_z == pytest.approx(baseline.bi_fractional_z + 0.001)
    assert candidate.se2_fractional_z == pytest.approx(baseline.se2_fractional_z - 0.002)
    assert candidate.bi_occupancy == 1.0
    assert candidate.se1_occupancy == 1.0
    assert candidate.se2_occupancy == pytest.approx(0.988)
    assert candidate.outer_bi_antisite_fraction == 0.0
    assert candidate.u_radial_A2 == 0.0
    assert candidate.u_normal_A2 == 0.0
    profile = candidate_strength.site_displacement_profile
    assert profile.scale == 1.0
    assert profile.components_A2("Bi") == pytest.approx((0.0036, 0.0264))
    assert profile.components_A2("Se1") == pytest.approx((0.0046, 0.0485))
    assert profile.components_A2("Se2") == pytest.approx((0.0046, 0.0485))
    assert candidate_envelope.u_radial_A2 == pytest.approx(0.007)
    assert candidate_envelope.u_normal_A2 == pytest.approx(0.011)


def test_structure_fit_plan_declares_stages_priors_and_bounds() -> None:
    plan_path = ROOT / "examples" / "bi2se3" / "experiment" / "figure7_structure_fit.toml"
    plan = ADAPTER._validated_fit_plan(tomllib.loads(plan_path.read_text(encoding="utf-8")))

    assert plan["material_id"] == "Bi2Se3"
    assert tuple(plan["parameter_names"]) == ADAPTER.STRUCTURE_PARAMETER_NAMES
    assert tuple(plan["stage"]) == ("A", "B", "C", "joint")
    assert tuple(plan["stage"]["A"]["active_parameters"]) == (
        "bi_delta_z_fractional",
        "outer_chalcogen_delta_z_fractional",
    )
    assert all(float(value) > 0.0 for value in plan["prior_sigma"])
    assert float(plan["sensitivity_relative_tolerance"]) == 1.0e-5
    assert float(plan["maximum_sensitivity_condition"]) == 1.0e5
    assert float(plan["bound_proximity_in_parameter_scales"]) == 1.0e-6
    assert plan["continuous_quadrature"] == {
        "offspecular_axial_refinement": 3,
        "offspecular_radial_transform": "squared_fold_coordinate.v1",
        "offspecular_signal_minimum_radial_nodes_per_side": 24,
    }
    bi2te3_plan_path = ROOT / "examples" / "bi2te3" / "experiment" / "figure7_structure_fit.toml"
    bi2te3_plan = ADAPTER._validated_fit_plan(
        tomllib.loads(bi2te3_plan_path.read_text(encoding="utf-8"))
    )
    assert bi2te3_plan["material_id"] == "Bi2Te3"
    assert tuple(bi2te3_plan["parameter_names"]) == ADAPTER.STRUCTURE_PARAMETER_NAMES

    seeded_plan_path = (
        ROOT / "examples" / "bi2te3" / "experiment" / "figure7_fast_seeded_joint_fit.toml"
    )
    seeded_plan = ADAPTER._validated_fit_plan(
        tomllib.loads(seeded_plan_path.read_text(encoding="utf-8"))
    )
    assert seeded_plan["execution_policy"] == "seeded_joint_only.v1"
    assert tuple(seeded_plan["stage"]) == ("joint",)
    assert seeded_plan["stage"]["joint"].get("predecessor") is None
    invalid_seeded_plan = copy.deepcopy(seeded_plan)
    invalid_seeded_plan["execution_policy"] = "staged_A_B_C_joint.v1"
    with pytest.raises(ValueError, match="seeded joint-only"):
        ADAPTER._validated_fit_plan(invalid_seeded_plan)
    with pytest.raises(ValueError, match="explicit initial parameters or a restart"):
        ADAPTER._require_declared_fit_start(
            seeded_plan["execution_policy"],
            initial_parameters_override=None,
            resume_path=None,
        )
    ADAPTER._require_declared_fit_start(
        seeded_plan["execution_policy"],
        initial_parameters_override=seeded_plan["baseline_parameters"],
        resume_path=None,
    )
    with pytest.raises(ValueError, match="fixed by the baseline or predecessor"):
        ADAPTER._require_declared_fit_start(
            plan["execution_policy"],
            initial_parameters_override=plan["baseline_parameters"],
            resume_path=None,
        )

    unbound = copy.deepcopy(plan)
    del unbound["material_id"]
    with pytest.raises(ValueError, match="material_id"):
        ADAPTER._validated_fit_plan(unbound)


def test_prepared_lattice_handoff_requires_explicit_state() -> None:
    from rasim_next.fitting import FixedLatticeState

    reference = np.diag((4.0, 4.0, 28.0))
    implicit = FixedLatticeState.implicit_cif(reference)
    accepted = FixedLatticeState.from_lattice_artifact(
        decision="ACCEPT_FITTED_LATTICE",
        reference_direct_basis_A=reference,
        active_direct_basis_A=reference @ np.diag((1.001, 1.001, 0.999)),
        artifact_path="C:/evidence/lattice.json",
        artifact_sha256="a" * 64,
        position_artifact_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="rebuilt lattice"):
        ADAPTER._qualified_prepared_lattice({}, implicit, "rods")
    with pytest.raises(ValueError, match="rebuilt lattice"):
        ADAPTER._qualified_prepared_lattice({}, accepted, "changed-rods")

    implicit_manifest = {
        "fixed_lattice": implicit.to_record(),
        "rod_catalog_revision": "rods",
    }
    assert ADAPTER._qualified_prepared_lattice(implicit_manifest, implicit, "rods") == (
        implicit.to_record()
    )

    manifest = {
        "fixed_lattice": accepted.to_record(),
        "rod_catalog_revision": "changed-rods",
    }
    assert (
        ADAPTER._qualified_prepared_lattice(manifest, accepted, "changed-rods")
        == accepted.to_record()
    )
    manifest["rod_catalog_revision"] = "stale"
    with pytest.raises(ValueError, match="rod catalog"):
        ADAPTER._qualified_prepared_lattice(manifest, accepted, "changed-rods")
