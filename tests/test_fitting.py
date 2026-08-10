from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

import rasim_next.fitting.geometry as fitting_geometry_module
import rasim_next.fitting.stacking_intensity as stacking_intensity_module
import rasim_next.ordered.motifs as ordered_motifs_module
import rasim_next.pipeline.configured_simulation as configured_simulation_module
import rasim_next.selection.blind as blind_module
import rasim_next.stacking.finite_intensity as finite_intensity_module
from painted_ewald import MosaicBraggSpace, Rod
from painted_ewald.rotations import mosaic_axes
from rasim_next.core.frames import FrameId
from rasim_next.core.layer_order import CommensurateLayerOrder
from rasim_next.core.transforms import RigidTransform
from rasim_next.fitting import (
    PBI2_IDEAL_PARENTS,
    SHARED_GEOMETRY_PARAMETER_NAMES,
    STACKING_COMPONENT_IDS,
    STACKING_PHASE_IDS,
    CompiledStackingResponse,
    ContinuousDetectorFunction,
    ContinuousDetectorGeometryModel,
    ExactTagGeometryModel,
    GeometryCorrectionBounds,
    GeometryCorrections,
    GeometryPredictionError,
    GeometryRankError,
    IncidenceAngleDeltaBounds,
    IndexedGeometryImage,
    IntegerLMarkerKey,
    IntegerLMarkerObservations,
    IntegerLMarkerPrediction,
    LayerLMarkerDefinition,
    LayerLMarkerKey,
    LayerLMarkerObservations,
    M0IntegerLObservations,
    M0IntegerLPrediction,
    MosaicComponentProfile,
    MosaicComponentProfileBank,
    MosaicIdentifiabilityError,
    MosaicProfileDefinition,
    MosaicProfileIdentity,
    MosaicProfileNuisanceBasis,
    MosaicProfileSet,
    MosaicReflectionGroupKey,
    SharedGeometryCorrectionBounds,
    SharedGeometryCorrections,
    StackingPopulationIdentifiabilityError,
    apply_shared_geometry_corrections,
    audit_indexed_geometry_series_roots,
    audit_integer_l_marker_selection,
    build_ideal_pbi2_polytype_landmark_catalogue,
    compile_pbi2_stacking_profile_response,
    evaluate_continuous_mosaic_profiles,
    evaluate_indexed_geometry_series_residual,
    evaluate_layer_l_geometry_objective_residual,
    evaluate_tagged_geometry_objective_residual,
    fit_indexed_geometry_series,
    fit_mosaic_component_profiles,
    fit_refined_mosaic_component_profiles,
    fit_stacking_phase_totals,
    fit_tagged_detector_function_geometry,
    merge_layer_l_marker_observations,
    pbi2_ideal_parent_landmark_contributions,
)
from rasim_next.geometry import (
    AngleFrame,
    angles_to_detector_coordinate_area_measure,
    build_incident_states,
    detector_coordinates_to_angles,
)
from rasim_next.pipeline.bragg_space import Bi2X3FiniteStackStrength
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
    solve_layer_l_ewald_roots,
)
from rasim_next.pipeline.continuous_detector import (
    DetectorEwaldMeasure,
)
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
from rasim_next.stacking import Parent


def test_reduced_layer_orders_and_fixed_l_roots_match_analytic_oracle() -> None:
    half = CommensurateLayerOrder(2, 4)
    assert half == CommensurateLayerOrder(-2, -4) == CommensurateLayerOrder(1, 2)
    assert CommensurateLayerOrder(2, -4) == CommensurateLayerOrder(-1, 2)
    assert CommensurateLayerOrder(0, 9) == CommensurateLayerOrder(0, 1)
    assert hash(half) == hash(CommensurateLayerOrder(1, 2))
    assert sorted(
        (
            CommensurateLayerOrder(2, 3),
            CommensurateLayerOrder(1, 3),
            CommensurateLayerOrder(1, 2),
        )
    ) == [
        CommensurateLayerOrder(1, 3),
        CommensurateLayerOrder(1, 2),
        CommensurateLayerOrder(2, 3),
    ]
    with pytest.raises(ValueError, match="denominator"):
        CommensurateLayerOrder(1, 0)
    with pytest.raises(TypeError, match="integer"):
        CommensurateLayerOrder(True, 1)

    basis = np.eye(3, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    ki_sample = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    rod = Rod(1, 0, 1.0)
    expected_regular_beta = (4.037257447447658, 2.2459278597319283)
    for layer_order, expected_branch in (
        (CommensurateLayerOrder(-1, 2), 1),
        (CommensurateLayerOrder(1, 2), 2),
    ):
        roots = solve_layer_l_ewald_roots(
            rod=rod,
            layer_order=layer_order,
            reciprocal_basis_Ainv=basis,
            crystal_to_sample=rotation,
            ki_sample_Ainv=ki_sample,
        )
        assert roots is not None
        assert roots.branch == expected_branch
        assert roots.root_sign == (-1, 1)
        np.testing.assert_allclose(roots.beta_rad, expected_regular_beta, rtol=0.0, atol=1.0e-14)
        layer_l = layer_order.as_float()
        for beta in roots.beta_rad:
            q = np.asarray((math.cos(beta), math.sin(beta), layer_l))
            assert float(q @ (q + 2.0 * ki_sample)) == pytest.approx(0.0, abs=5.0e-15)

    tangent = solve_layer_l_ewald_roots(
        rod=rod,
        layer_order=CommensurateLayerOrder(1),
        reciprocal_basis_Ainv=basis,
        crystal_to_sample=rotation,
        ki_sample_Ainv=ki_sample,
    )
    assert tangent is not None
    assert tangent.beta_rad == pytest.approx((math.pi,), abs=1.0e-15)
    assert tangent.root_sign == (0,)
    assert tangent.branch == 2
    assert (
        solve_layer_l_ewald_roots(
            rod=rod,
            layer_order=CommensurateLayerOrder(3, 2),
            reciprocal_basis_Ainv=basis,
            crystal_to_sample=rotation,
            ki_sample_Ainv=ki_sample,
        )
        is None
    )


def test_denominator_one_layer_prediction_is_bit_exact_with_integer_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    inputs = build_configured_geometry_inputs(config)
    model = ExactTagGeometryModel(inputs)
    rod = next(rod for rod in inputs.rods if rod.family_m == 1 and (rod.h, rod.k) == (-1, 0))
    incident = build_incident_states(inputs.samples, inputs.material, inputs.instrument)
    integer_keys: list[IntegerLMarkerKey] = []
    layer_definitions: list[LayerLMarkerDefinition] = []
    for integer_l in (4, 5):
        integer_roots = solve_integer_l_ewald_roots(
            rod=rod,
            integer_l=integer_l,
            reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
            crystal_to_sample=inputs.instrument.sample_from_crystal.rotation,
            ki_sample_Ainv=incident.states.k_film_phase_sample_Ainv[0],
        )
        layer_roots = solve_layer_l_ewald_roots(
            rod=rod,
            layer_order=CommensurateLayerOrder(integer_l),
            reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
            crystal_to_sample=inputs.instrument.sample_from_crystal.rotation,
            ki_sample_Ainv=incident.states.k_film_phase_sample_Ainv[0],
        )
        assert integer_roots is not None and layer_roots is not None
        assert layer_roots.beta_rad == integer_roots.beta_rad
        assert layer_roots.root_sign == integer_roots.root_sign
        assert layer_roots.branch == integer_roots.branch
        for root_sign in integer_roots.root_sign:
            integer_keys.append(
                IntegerLMarkerKey(
                    family_m=rod.family_m,
                    integer_L=integer_l,
                    branch=integer_roots.branch,
                    root_sign=root_sign,
                    representative_rod_hk=(rod.h, rod.k),
                )
            )
            layer_definitions.append(
                LayerLMarkerDefinition(
                    key=LayerLMarkerKey(
                        family_m=rod.family_m,
                        layer_order=CommensurateLayerOrder(integer_l),
                        branch=integer_roots.branch,
                        root_sign=root_sign,
                        reciprocal_basis_revision=model.reciprocal_basis_revision,
                    ),
                    contributing_rod_hk=((rod.h, rod.k),),
                )
            )

    integer_prediction = model.predict_integer_l_tags(tuple(integer_keys))
    layer_prediction = model.predict_layer_l_tags(tuple(layer_definitions))
    np.testing.assert_array_equal(
        layer_prediction.coordinates_px, integer_prediction.coordinates_px
    )
    np.testing.assert_array_equal(
        layer_prediction.ewald_residual_Ainv,
        integer_prediction.ewald_residual_Ainv,
    )
    np.testing.assert_array_equal(
        layer_prediction.detector_status, integer_prediction.detector_status
    )


@pytest.mark.parametrize(
    ("signed_rod_hk", "layer_order", "expected_parents"),
    (
        (
            (1, 1),
            CommensurateLayerOrder(1),
            (
                Parent.TWO_H,
                Parent.FOUR_H_PLUS,
                Parent.FOUR_H_MINUS,
                Parent.SIX_H_PLUS,
                Parent.SIX_H_MINUS,
            ),
        ),
        (
            (1, 0),
            CommensurateLayerOrder(1),
            (Parent.TWO_H, Parent.FOUR_H_PLUS, Parent.FOUR_H_MINUS),
        ),
        (
            (1, 0),
            CommensurateLayerOrder(1, 2),
            (Parent.FOUR_H_PLUS, Parent.FOUR_H_MINUS),
        ),
        ((1, 0), CommensurateLayerOrder(1, 3), (Parent.SIX_H_MINUS,)),
        ((1, 0), CommensurateLayerOrder(2, 3), (Parent.SIX_H_PLUS,)),
        ((1, 0), CommensurateLayerOrder(-1, 3), (Parent.SIX_H_PLUS,)),
        ((-1, 0), CommensurateLayerOrder(1, 3), (Parent.SIX_H_PLUS,)),
        ((-1, 0), CommensurateLayerOrder(2, 3), (Parent.SIX_H_MINUS,)),
        ((1, 1), CommensurateLayerOrder(1, 2), ()),
    ),
)
def test_ideal_pbi2_parent_landmark_support_is_exact(
    signed_rod_hk: tuple[int, int],
    layer_order: CommensurateLayerOrder,
    expected_parents: tuple[Parent, ...],
) -> None:
    contributions = pbi2_ideal_parent_landmark_contributions(
        signed_rod_hk=signed_rod_hk,
        layer_order=layer_order,
    )
    assert tuple(item.parent for item in contributions) == expected_parents
    assert all(item.signed_rod_hk == signed_rod_hk for item in contributions)


def test_ideal_pbi2_polytype_catalogue_deduplicates_physical_overlaps() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    pbi2_2h = root / "examples" / "pbi2" / "structures" / "PbI2_2H.cif"
    config = replace(
        base,
        material=replace(base.material, cif_path=pbi2_2h, phase_id="pbi2"),
    )
    model = ExactTagGeometryModel(build_configured_geometry_inputs(config))
    layer_orders = tuple(
        CommensurateLayerOrder(numerator, denominator)
        for numerator, denominator in (
            (1, 3),
            (1, 2),
            (2, 3),
            (1, 1),
            (4, 3),
            (3, 2),
            (5, 3),
            (2, 1),
            (5, 2),
            (3, 1),
        )
    )
    catalogue = build_ideal_pbi2_polytype_landmark_catalogue(model, layer_orders)
    assert catalogue.source_cif_sha256 == config.cif_sha256
    assert catalogue.reciprocal_basis_revision == model.reciprocal_basis_revision
    assert catalogue.geometry_context_revision == model.geometry_context_revision
    assert catalogue.model_id == "pbi2.declared_single_trilayer_metric.landmarks.v1"
    assert len(catalogue.keys) == len(set(catalogue.keys))
    assert catalogue.prediction.active_panel.all()
    assert all(
        definition.contributing_rod_hk == tuple(sorted(set(definition.contributing_rod_hk)))
        for definition in catalogue.definitions
    )
    assert all(
        contribution.signed_rod_hk in definition.contributing_rod_hk
        for definition, contributions in zip(
            catalogue.definitions,
            catalogue.parent_contributions,
            strict=True,
        )
        for contribution in contributions
    )
    assert any(
        definition.key.layer_order.denominator == 2
        and {item.parent for item in contributions} == {Parent.FOUR_H_PLUS, Parent.FOUR_H_MINUS}
        for definition, contributions in zip(
            catalogue.definitions,
            catalogue.parent_contributions,
            strict=True,
        )
    )
    assert any(
        definition.key.layer_order.denominator == 3
        and {item.parent for item in contributions} <= {Parent.SIX_H_PLUS, Parent.SIX_H_MINUS}
        for definition, contributions in zip(
            catalogue.definitions,
            catalogue.parent_contributions,
            strict=True,
        )
    )
    assert any(
        definition.key.layer_order.is_integer
        and {item.parent for item in contributions} == set(PBI2_IDEAL_PARENTS)
        for definition, contributions in zip(
            catalogue.definitions,
            catalogue.parent_contributions,
            strict=True,
        )
    )
    with pytest.raises(ValueError, match="catalogue_revision"):
        replace(catalogue, catalogue_revision="0" * 64)

    restricted = build_ideal_pbi2_polytype_landmark_catalogue(
        model,
        layer_orders,
        parents=(Parent.SIX_H_PLUS,),
    )
    restricted_index, unsupported_rod_hk = next(
        (index, (rod.h, rod.k))
        for index, definition in enumerate(restricted.definitions)
        for rod in model.inputs.rods
        if rod.family_m == definition.key.family_m
        and (rod.h, rod.k) not in definition.contributing_rod_hk
        and not pbi2_ideal_parent_landmark_contributions(
            signed_rod_hk=(rod.h, rod.k),
            layer_order=definition.key.layer_order,
            parents=restricted.selected_parents,
        )
    )
    restricted_definition = restricted.definitions[restricted_index]
    paired_identity = (
        restricted_definition.key.family_m,
        restricted_definition.key.layer_order,
        restricted_definition.key.reciprocal_basis_revision,
    )
    forged_definitions = list(restricted.definitions)
    for index, definition in enumerate(restricted.definitions):
        if (
            definition.key.family_m,
            definition.key.layer_order,
            definition.key.reciprocal_basis_revision,
        ) == paired_identity:
            forged_definitions[index] = replace(
                definition,
                contributing_rod_hk=(*definition.contributing_rod_hk, unsupported_rod_hk),
            )
    with pytest.raises(ValueError, match="exact signed-rod support"):
        replace(
            restricted,
            prediction=replace(
                restricted.prediction,
                definitions=tuple(forged_definitions),
            ),
        )

    overlap_index = next(
        index
        for index, definition in enumerate(catalogue.definitions)
        if len(definition.contributing_rod_hk) > 1
    )
    overlap = LayerLMarkerObservations.from_prediction(
        catalogue.prediction.subset(np.asarray([overlap_index])),
        reference_wavelength_A=model.reference_wavelength_A,
        sigma_px=0.25,
    )
    overlap_definition = overlap.definitions[0]
    split = tuple(
        LayerLMarkerObservations(
            definitions=(
                LayerLMarkerDefinition(
                    key=overlap_definition.key,
                    contributing_rod_hk=rods,
                ),
            ),
            coordinates_px=overlap.coordinates_px,
            covariance_px2=overlap.covariance_px2,
            reference_wavelength_A=overlap.reference_wavelength_A,
        )
        for rods in (
            overlap_definition.contributing_rod_hk[:1],
            overlap_definition.contributing_rod_hk[1:],
        )
    )
    merged_overlap = merge_layer_l_marker_observations(*split)
    assert isinstance(merged_overlap, LayerLMarkerObservations)
    assert len(merged_overlap.keys) == 1
    assert (
        merged_overlap.definitions[0].contributing_rod_hk == overlap_definition.contributing_rod_hk
    )
    with pytest.raises(ValueError, match="conflict"):
        merge_layer_l_marker_observations(
            split[0],
            replace(
                split[1],
                coordinates_px=split[1].coordinates_px + np.asarray((0.1, 0.0)),
            ),
        )
    foreign_definition = LayerLMarkerDefinition(
        key=replace(overlap_definition.key, reciprocal_basis_revision="f" * 64),
        contributing_rod_hk=split[1].definitions[0].contributing_rod_hk,
    )
    foreign = replace(split[1], definitions=(foreign_definition,))
    with pytest.raises(ValueError, match="one reciprocal-basis revision"):
        merge_layer_l_marker_observations(split[0], foreign)
    with pytest.raises(ValueError, match="reciprocal-basis revision"):
        IndexedGeometryImage(
            image_id="foreign-basis",
            commanded_angle_rad=math.radians(config.instrument.axis_rotations[0].angle_deg),
            model=model,
            observations=foreign,
        )
    subset_prediction = model.predict_layer_l_tags(split[0].definitions)
    with pytest.raises(ValueError, match="definitions"):
        evaluate_layer_l_geometry_objective_residual(overlap, subset_prediction)

    relaxed_4h = root / "examples" / "pbi2" / "structures" / "PbI2_4H.cif"
    relaxed_model = ExactTagGeometryModel(
        build_configured_geometry_inputs(
            replace(config, material=replace(config.material, cif_path=relaxed_4h))
        )
    )
    with pytest.raises(ValueError, match="one PbI2 trilayer"):
        build_ideal_pbi2_polytype_landmark_catalogue(relaxed_model, layer_orders)


def test_optional_pbi2_polytype_landmarks_strengthen_shared_geometry_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    pbi2_2h = root / "examples" / "pbi2" / "structures" / "PbI2_2H.cif"
    base = replace(
        base,
        material=replace(base.material, cif_path=pbi2_2h, phase_id="pbi2"),
    )
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
    baseline_orders = tuple(CommensurateLayerOrder(value) for value in range(1, 5))
    augmented_orders = tuple(
        sorted(
            {
                *baseline_orders,
                *(CommensurateLayerOrder(value, 2) for value in range(1, 8, 2)),
                *(CommensurateLayerOrder(value, 3) for value in (2, 5, 8, 11)),
            }
        )
    )
    baseline_images: list[IndexedGeometryImage] = []
    augmented_images: list[IndexedGeometryImage] = []

    def reject_intensity_or_mosaic(*args: object, **kwargs: object) -> None:
        raise AssertionError("polytype geometry must not build intensity or mosaic state")

    monkeypatch.setattr(Bi2X3FiniteStackStrength, "__init__", reject_intensity_or_mosaic)
    monkeypatch.setattr(MosaicBraggSpace, "__init__", reject_intensity_or_mosaic)
    monkeypatch.setattr(
        ordered_motifs_module,
        "pbi2_layer_amplitudes",
        reject_intensity_or_mosaic,
    )
    monkeypatch.setattr(
        stacking_intensity_module,
        "compile_pbi2_stacking_profile_response",
        reject_intensity_or_mosaic,
    )
    monkeypatch.setattr(
        finite_intensity_module,
        "finite_population_event_intensity",
        reject_intensity_or_mosaic,
    )
    monkeypatch.setattr(
        configured_simulation_module,
        "integrate_detector_macrobins",
        reject_intensity_or_mosaic,
    )
    monkeypatch.setattr(
        configured_simulation_module,
        "sample_detector_pixel_center_density",
        reject_intensity_or_mosaic,
    )
    for angle_deg in (5.0, 10.0, 15.0):
        axis = base.instrument.axis_rotations[0]
        config = replace(
            base,
            instrument=replace(
                base.instrument,
                axis_rotations=(replace(axis, angle_deg=angle_deg),),
            ),
        )
        inputs = build_configured_geometry_inputs(config)
        model = ExactTagGeometryModel(inputs)
        truth_instrument = _shared_truth_instrument(inputs, truth)
        baseline_catalogue = build_ideal_pbi2_polytype_landmark_catalogue(
            model,
            baseline_orders,
            parents=(Parent.TWO_H,),
        )
        augmented_catalogue = build_ideal_pbi2_polytype_landmark_catalogue(
            model,
            augmented_orders,
            parents=(Parent.TWO_H, Parent.FOUR_H_PLUS, Parent.SIX_H_MINUS),
        )
        baseline_definitions = tuple(
            definition
            for definition in baseline_catalogue.definitions
            if definition.key.family_m == 1
            and definition.key.branch == 2
            and (-1, 0) in definition.contributing_rod_hk
        )
        augmented_definitions = tuple(
            definition
            for definition in augmented_catalogue.definitions
            if definition.key.family_m == 1
            and definition.key.branch == 2
            and (-1, 0) in definition.contributing_rod_hk
        )
        baseline_prediction = model.predict_layer_l_tags(
            baseline_definitions,
            instrument=truth_instrument,
        )
        augmented_prediction = model.predict_layer_l_tags(
            augmented_definitions,
            instrument=truth_instrument,
        )
        baseline_layer_prediction = baseline_prediction.subset(baseline_prediction.active_panel)
        baseline_integer_keys = tuple(
            IntegerLMarkerKey(
                family_m=definition.key.family_m,
                integer_L=definition.key.layer_order.numerator,
                branch=definition.key.branch,
                root_sign=definition.key.root_sign,
                representative_rod_hk=definition.representative_rod_hk,
            )
            for definition in baseline_layer_prediction.definitions
        )
        baseline_integer_prediction = model.predict_integer_l_tags(
            baseline_integer_keys,
            instrument=truth_instrument,
        )
        np.testing.assert_array_equal(
            baseline_layer_prediction.coordinates_px,
            baseline_integer_prediction.coordinates_px,
        )
        baseline_observations = IntegerLMarkerObservations.from_prediction(
            baseline_integer_prediction,
            reference_wavelength_A=model.reference_wavelength_A,
            sigma_px=0.25,
        )
        all_augmented_observations = LayerLMarkerObservations.from_prediction(
            augmented_prediction.subset(augmented_prediction.active_panel),
            reference_wavelength_A=model.reference_wavelength_A,
            sigma_px=0.25,
        )
        assert merge_layer_l_marker_observations(baseline_observations) is baseline_observations
        augmented_observations = merge_layer_l_marker_observations(
            baseline_observations,
            all_augmented_observations,
            reciprocal_basis_revision=model.reciprocal_basis_revision,
        )
        assert isinstance(augmented_observations, LayerLMarkerObservations)
        assert augmented_observations.keys == all_augmented_observations.keys
        assert augmented_observations.definitions == all_augmented_observations.definitions
        image_id = f"pbi2-{int(angle_deg):02d}"
        baseline_images.append(
            IndexedGeometryImage(
                image_id=image_id,
                commanded_angle_rad=math.radians(angle_deg),
                model=model,
                observations=baseline_observations,
            )
        )
        augmented_images.append(
            IndexedGeometryImage(
                image_id=image_id,
                commanded_angle_rad=math.radians(angle_deg),
                model=model,
                observations=augmented_observations,
            )
        )

    assert tuple(len(image.observations.keys) for image in baseline_images) == (8, 8, 6)
    assert tuple(len(image.observations.keys) for image in augmented_images) == (24, 20, 18)
    assert {
        key.layer_order.denominator
        for image in augmented_images
        if isinstance(image.observations, LayerLMarkerObservations)
        for key in image.observations.keys
    } == {1, 2, 3}
    assert all(
        len(augmented.observations.keys) > len(baseline.observations.keys)
        for baseline, augmented in zip(baseline_images, augmented_images, strict=True)
    )
    bounds = SharedGeometryCorrectionBounds.rasim_multi_angle_pose()
    baseline_fit = fit_indexed_geometry_series(
        tuple(baseline_images),
        initial=SharedGeometryCorrections.zero(),
        bounds=bounds,
    )
    augmented_fit = fit_indexed_geometry_series(
        tuple(augmented_images),
        initial=SharedGeometryCorrections.zero(),
        bounds=bounds,
    )
    for result in (baseline_fit, augmented_fit):
        assert result.success, result.message
        assert result.jacobian_rank == len(SHARED_GEOMETRY_PARAMETER_NAMES)
        assert not np.any(result.active_bounds)
        np.testing.assert_array_less(
            np.abs(result.corrections.as_array() - truth.as_array()) / bounds.half_span,
            np.full(9, 2.5e-5),
        )
        assert result.training_site_max_px < 5.0e-3
    assert (
        augmented_fit.scaled_jacobian_singular_values[-1]
        > 1.25 * baseline_fit.scaled_jacobian_singular_values[-1]
    )
    assert (
        audit_indexed_geometry_series_roots(
            tuple(baseline_images),
            baseline_fit.corrections,
        ).classification
        == "SAME"
    )
    assert (
        audit_indexed_geometry_series_roots(
            tuple(augmented_images),
            augmented_fit.corrections,
        ).classification
        == "SAME"
    )

    real_solver = fitting_geometry_module.solve_layer_l_ewald_roots

    def swapped_layer_root_solver(**kwargs: object) -> object:
        roots = real_solver(**kwargs)
        if roots is None or len(roots.beta_rad) != 2:
            return roots
        return replace(roots, beta_rad=tuple(reversed(roots.beta_rad)))

    monkeypatch.setattr(
        fitting_geometry_module,
        "solve_layer_l_ewald_roots",
        swapped_layer_root_solver,
    )
    assert (
        audit_indexed_geometry_series_roots(
            tuple(augmented_images),
            augmented_fit.corrections,
        ).classification
        == "CHANGED"
    )


def _analytic_mosaic_profile_bank() -> tuple[
    MosaicComponentProfileBank,
    dict[MosaicProfileIdentity, float],
]:
    from painted_ewald import MosaicParameters, wrapped_mosaic_line_density_rad_inv

    group_keys = (
        MosaicReflectionGroupKey(
            group_id="synthetic-00L-6",
            rod_catalog_revision="synthetic-rods.v1",
            member_rod_hk=((0, 0),),
            branch_mode="COLLAPSED_00L",
            layered_family_m=0,
            layered_integer_L=6,
        ),
        MosaicReflectionGroupKey(
            group_id="synthetic-m1-L8",
            rod_catalog_revision="synthetic-rods.v1",
            member_rod_hk=((1, 0),),
            branch_mode="EXPLICIT_NONZERO",
            layered_family_m=1,
            layered_integer_L=8,
        ),
        MosaicReflectionGroupKey(
            group_id="synthetic-m3-L10",
            rod_catalog_revision="synthetic-rods.v1",
            member_rod_hk=((1, 1),),
            branch_mode="EXPLICIT_NONZERO",
            layered_family_m=3,
            layered_integer_L=10,
        ),
    )
    identities: list[MosaicProfileIdentity] = []
    alpha_rows: list[np.ndarray] = []
    response_rows: list[np.ndarray] = []
    sample_axis = np.linspace(-1.0, 1.0, 41)
    for incidence_index, incidence_deg in enumerate((5.0, 10.0, 15.0)):
        for group_index, group_key in enumerate(group_keys):
            branches = (None,) if group_key.branch_mode == "COLLAPSED_00L" else (1, 2)
            for branch_id in branches:
                identities.append(
                    MosaicProfileIdentity(
                        dataset_id=f"synthetic-{incidence_deg:g}",
                        incidence_angle_rad=math.radians(incidence_deg),
                        group_key=group_key,
                        branch_id=branch_id,
                        analytic_branch_id=0 if branch_id is None else 1,
                    )
                )
                branch_shift = 0.0 if branch_id is None else (branch_id - 1.5) * 0.11
                center_deg = 0.35 + 0.55 * group_index + 0.42 * incidence_index + branch_shift
                alpha_rows.append(
                    np.radians(np.abs(center_deg + (0.8 + 0.1 * group_index) * sample_axis))
                )
                response_rows.append(
                    (1.0 + 0.12 * sample_axis + 0.03 * sample_axis**2)
                    * (1.0 + 0.2 * incidence_index + 0.07 * group_index)
                )

    alpha = np.asarray(alpha_rows)
    response = np.asarray(response_rows)
    normalization = np.broadcast_to(
        0.8 + 0.3 * (sample_axis + 1.0),
        alpha.shape,
    ).copy()
    valid = np.ones(alpha.shape, dtype=np.bool_)
    valid[:, (0, -1)] = False
    phi_bin_edges = np.broadcast_to(
        np.linspace(-1.05, 1.05, sample_axis.size + 1),
        (len(identities), sample_axis.size + 1),
    ).copy()
    two_theta_bounds = np.broadcast_to((0.2, 0.3), (len(identities), 2)).copy()
    frame_revisions = ("analytic-response-frame.v1",) * len(identities)
    gaussian_width_deg = np.asarray((0.65, 1.1, 2.0, 3.3))
    lorentzian_width_deg = np.asarray((0.12, 0.3, 0.5, 0.9, 1.8))

    def component_signal(width_deg: float, component: str) -> np.ndarray:
        if component == "gaussian":
            parameters = MosaicParameters(math.radians(width_deg), 1.0, 0.0)
        else:
            parameters = MosaicParameters(1.0, math.radians(width_deg), 1.0)
        density = wrapped_mosaic_line_density_rad_inv(alpha, parameters) / np.pi
        return density * response * normalization

    gaussian_signal = np.asarray(
        [component_signal(width, "gaussian") for width in gaussian_width_deg]
    )
    lorentzian_signal = np.asarray(
        [component_signal(width, "lorentzian") for width in lorentzian_width_deg]
    )
    gaussian_signal[:, ~valid] = 0.0
    lorentzian_signal[:, ~valid] = 0.0
    normalization[~valid] = 0.0
    truth_gaussian_index = int(np.flatnonzero(gaussian_width_deg == 2.0)[0])
    truth_lorentzian_index = int(np.flatnonzero(lorentzian_width_deg == 0.5)[0])
    eta = 0.1
    unscaled_truth = (1.0 - eta) * gaussian_signal[truth_gaussian_index] + eta * lorentzian_signal[
        truth_lorentzian_index
    ]
    planted_scale = {
        identity: float(scale)
        for identity, scale in zip(
            identities,
            np.geomspace(2.5e-4, 1.4e4, len(identities)),
            strict=True,
        )
    }
    observed_signal = np.asarray(
        [
            planted_scale[identity] * profile
            for identity, profile in zip(identities, unscaled_truth, strict=True)
        ]
    )
    observations = MosaicProfileSet(
        identities=tuple(identities),
        signal=observed_signal,
        normalization=normalization,
        valid=valid,
        profile_revision="analytic-response.v1",
        phi_bin_edges_rad=phi_bin_edges,
        two_theta_bounds_rad=two_theta_bounds,
        angle_frame_revisions=frame_revisions,
        source_revision="analytic-source.v1",
    )

    def component_profile(signal: np.ndarray) -> MosaicProfileSet:
        return MosaicProfileSet(
            identities=tuple(identities),
            signal=signal,
            normalization=normalization,
            valid=valid,
            profile_revision="analytic-response.v1",
            phi_bin_edges_rad=phi_bin_edges,
            two_theta_bounds_rad=two_theta_bounds,
            angle_frame_revisions=frame_revisions,
            source_revision="analytic-source.v1",
        )

    return (
        MosaicComponentProfileBank(
            observations=observations,
            gaussian_sigma_rad=np.radians(gaussian_width_deg),
            gaussian_profiles=tuple(
                MosaicComponentProfile(
                    "gaussian",
                    math.radians(float(width)),
                    component_profile(signal),
                )
                for width, signal in zip(gaussian_width_deg, gaussian_signal, strict=True)
            ),
            lorentzian_half_width_rad=np.radians(lorentzian_width_deg),
            lorentzian_profiles=tuple(
                MosaicComponentProfile(
                    "lorentzian",
                    math.radians(float(width)),
                    component_profile(signal),
                )
                for width, signal in zip(lorentzian_width_deg, lorentzian_signal, strict=True)
            ),
        ),
        planted_scale,
    )


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
    *,
    incidence_angle_delta_rad: float = 0.0,
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
    base_angle = math.radians(axis.angle_deg)
    corrected_angle = base_angle + incidence_angle_delta_rad
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
    base_motion_rotation = _axis_rotation_matrix(base_axis, base_angle)
    base_motion_translation = pivot - base_motion_rotation @ pivot
    zero_sample_rotation = base_motion_rotation.T @ base.lab_from_sample.rotation
    zero_sample_translation = base_motion_rotation.T @ (
        base.lab_from_sample.translation_m - base_motion_translation
    )
    corrected_motion_rotation = _axis_rotation_matrix(corrected_axis, corrected_angle)
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
        sample_normal_x_tilt_rad=0.0,
        sample_normal_y_tilt_rad=math.radians(-0.27),
        goniometer_axis_pitch_rad=math.radians(0.15),
        goniometer_axis_yaw_rad=math.radians(-0.22),
        sample_plane_normal_offset_m=2.0e-5,
        goniometer_pivot_pitch_offset_m=3.0e-5,
        goniometer_pivot_yaw_offset_m=-2.5e-5,
    )
    truth_incidence_angle_delta_rad = math.radians(0.18)
    fitted_with_incidence_delta = tuple(
        name for name in SHARED_GEOMETRY_PARAMETER_NAMES if name != "sample_normal_x_tilt_rad"
    )
    incidence_bounds = IncidenceAngleDeltaBounds.rasim_shared_offset()
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
            instrument=_shared_truth_instrument(
                geometry_inputs,
                truth,
                incidence_angle_delta_rad=truth_incidence_angle_delta_rad,
            ),
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
            fitted_parameter_names=fitted_with_incidence_delta,
            incidence_angle_delta_bounds=incidence_bounds,
        )
    with pytest.raises(GeometryRankError, match=r"rank=7/9"):
        fit_indexed_geometry_series(
            tuple(images[:2]),
            initial=SharedGeometryCorrections.zero(),
            bounds=bounds,
            fitted_parameter_names=fitted_with_incidence_delta,
            incidence_angle_delta_bounds=incidence_bounds,
        )

    with pytest.raises(ValueError, match="share the nominal incidence-axis gauge"):
        fit_indexed_geometry_series(
            tuple(images),
            initial=SharedGeometryCorrections.zero(),
            bounds=bounds,
            incidence_angle_delta_bounds=incidence_bounds,
        )

    result = fit_indexed_geometry_series(
        tuple(reversed(images)),
        initial=SharedGeometryCorrections.zero(),
        bounds=bounds,
        fitted_parameter_names=fitted_with_incidence_delta,
        initial_incidence_angle_delta_rad=math.radians(0.1),
        incidence_angle_delta_bounds=incidence_bounds,
    )
    assert result.success, result.message
    np.testing.assert_array_equal(
        evaluate_indexed_geometry_series_residual(
            tuple(reversed(images)),
            SharedGeometryCorrections.zero(),
            incidence_angle_delta_rad=0.0,
        ),
        evaluate_indexed_geometry_series_residual(
            tuple(images),
            SharedGeometryCorrections.zero(),
            incidence_angle_delta_rad=0.0,
        ),
    )
    assert result.image_ids == ("osc-05", "osc-10", "osc-15")
    assert result.fitted_parameter_names == fitted_with_incidence_delta
    assert result.fixed_parameter_names == ("sample_normal_x_tilt_rad",)
    assert result.jacobian_parameter_names == (
        *fitted_with_incidence_delta,
        "incidence_angle_delta_rad",
    )
    assert result.jacobian_rank == 9
    assert result.jacobian_condition < 15_000.0
    assert not np.any(result.active_bounds)
    np.testing.assert_array_less(
        np.abs(result.corrections.as_array() - truth.as_array()) / bounds.half_span,
        np.full(9, 2.5e-5),
    )
    assert result.incidence_angle_delta_rad == pytest.approx(
        truth_incidence_angle_delta_rad,
        abs=2.5e-5 * incidence_bounds.half_span_rad,
    )
    assert result.training_site_rms_px < 1.0e-3
    assert result.training_site_max_px < 5.0e-3
    assert result.training_chord_angle_rms_rad < 1.0e-7

    truth_trim_by_id = {
        "osc-05": math.radians(0.12),
        "osc-10": math.radians(-0.08),
        "osc-15": math.radians(-0.04),
    }
    trimmed_images = tuple(
        replace(
            image,
            observations=IntegerLMarkerObservations.from_prediction(
                image.model.predict_integer_l_tags(
                    image.observations.keys,
                    instrument=_shared_truth_instrument(
                        image.model.inputs,
                        truth,
                        incidence_angle_delta_rad=(
                            truth_incidence_angle_delta_rad + truth_trim_by_id[image.image_id]
                        ),
                    ),
                ),
                reference_wavelength_A=image.model.reference_wavelength_A,
                sigma_px=0.25,
            ),
        )
        for image in images
    )
    trimmed = fit_indexed_geometry_series(
        trimmed_images,
        initial=SharedGeometryCorrections.zero(),
        bounds=bounds,
        fitted_parameter_names=fitted_with_incidence_delta,
        initial_incidence_angle_delta_rad=math.radians(0.1),
        incidence_angle_delta_bounds=incidence_bounds,
        incidence_angle_trim_contrast_half_span_rad=math.radians(0.5),
        incidence_angle_trim_prior_sigma_rad=math.radians(0.25),
    )
    assert trimmed.success, trimmed.message
    assert trimmed.jacobian_rank == 11
    assert trimmed.posterior_jacobian_rank == 11
    assert trimmed.jacobian_parameter_names[-2:] == (
        "incidence_angle_trim_helmert_1_rad",
        "incidence_angle_trim_helmert_2_rad",
    )
    recovered_trim_by_id = dict(
        zip(
            trimmed.image_ids,
            trimmed.incidence_angle_trim_by_image_id_rad,
            strict=True,
        )
    )
    assert math.fsum(recovered_trim_by_id.values()) == pytest.approx(0.0, abs=1.0e-14)
    for image_id, expected_trim in truth_trim_by_id.items():
        assert recovered_trim_by_id[image_id] == pytest.approx(
            expected_trim,
            abs=math.radians(3.0e-3),
        )
    assert trimmed.incidence_angle_delta_rad == pytest.approx(
        truth_incidence_angle_delta_rad,
        abs=math.radians(3.0e-3),
    )
    assert (
        math.degrees(
            trimmed_images[2].commanded_angle_rad
            + trimmed.incidence_angle_delta_rad
            + recovered_trim_by_id[trimmed_images[2].image_id]
        )
        > 13.9
    )

    fitted_without_detector_tilts = (
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
        initial_incidence_angle_delta_rad=math.radians(0.1),
        incidence_angle_delta_bounds=incidence_bounds,
    )
    assert constrained.success, constrained.message
    assert constrained.fitted_parameter_names == fitted_without_detector_tilts
    assert constrained.fixed_parameter_names == (
        "detector_column_tilt_rad",
        "detector_row_tilt_rad",
        "sample_normal_x_tilt_rad",
    )
    assert constrained.jacobian_parameter_names == (
        *fitted_without_detector_tilts,
        "incidence_angle_delta_rad",
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
    assert constrained.incidence_angle_delta_rad == pytest.approx(
        truth_incidence_angle_delta_rad,
        abs=2.5e-5 * incidence_bounds.half_span_rad,
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
        initial_incidence_angle_delta_rad=truth_incidence_angle_delta_rad,
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
            incidence_angle_delta_rad=result.incidence_angle_delta_rad,
        )
        error = prediction.coordinates_px - heldout[image.image_id].coordinates_px
        assert float(np.max(np.linalg.norm(error, axis=1), initial=0.0)) < 1.0e-2

    for image in images:
        fitted = image.corrected_instrument(
            result.corrections,
            incidence_angle_delta_rad=result.incidence_angle_delta_rad,
        )
        normal = fitted.lab_from_sample.rotation[:, 2]
        zero_offset = image.corrected_instrument(
            replace(result.corrections, sample_plane_normal_offset_m=0.0),
            incidence_angle_delta_rad=result.incidence_angle_delta_rad,
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
    root_audit = audit_indexed_geometry_series_roots(
        tuple(images),
        result.corrections,
        incidence_angle_delta_rad=result.incidence_angle_delta_rad,
    )
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
    swapped_audit = audit_indexed_geometry_series_roots(
        tuple(images),
        result.corrections,
        incidence_angle_delta_rad=result.incidence_angle_delta_rad,
    )
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
        .replace("model_id: r3m_quintuple_finite_2h.v1", "model_id: geometry_only.unused.v1")
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

    monkeypatch.setattr(Bi2X3FiniteStackStrength, "__init__", reject_intensity_or_mosaic)
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


def test_configured_build_rejects_cif_changed_after_its_revision_was_frozen(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    original = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    cif_copy = tmp_path / original.material.cif_path.name
    source_bytes = original.material.cif_path.read_bytes()
    cif_copy.write_bytes(source_bytes)
    frozen = replace(
        original,
        material=replace(original.material, cif_path=cif_copy),
    )
    cif_copy.write_bytes(source_bytes + b"\n")

    with pytest.raises(ValueError, match="CIF content changed after"):
        build_configured_geometry_inputs(frozen)


def test_configured_geometry_rebind_rejects_new_content_at_the_same_cif_path(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    original = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    cif_copy = tmp_path / original.material.cif_path.name
    source_bytes = original.material.cif_path.read_bytes()
    cif_copy.write_bytes(source_bytes)
    frozen = replace(original, material=replace(original.material, cif_path=cif_copy))
    inputs = build_configured_geometry_inputs(frozen)

    cif_copy.write_bytes(source_bytes + b"\n")
    refreshed = replace(frozen)
    assert refreshed.cif_sha256 != frozen.cif_sha256
    with pytest.raises(ValueError, match="CIF content"):
        rebind_configured_geometry_instrument(inputs, refreshed)


def test_mosaic_component_profiles_recover_widths_with_unknown_profile_scales() -> None:
    bank, planted_scale = _analytic_mosaic_profile_bank()
    result = fit_mosaic_component_profiles(bank)

    assert math.degrees(result.gaussian_sigma_rad) == pytest.approx(2.0, abs=1.0e-12)
    assert math.degrees(result.lorentzian_half_width_rad) == pytest.approx(0.5, abs=1.0e-12)
    assert result.lorentzian_probability == pytest.approx(0.1, abs=2.0e-9)
    assert result.objective <= 2.0e-20
    assert result.sensitivity_rank == 3
    assert np.isfinite(result.sensitivity_condition)
    assert result.objective == pytest.approx(
        float(result.profile_relative_l2_residual @ result.profile_relative_l2_residual),
        abs=2.0e-20,
    )
    for identity in result.profile_identities:
        assert result.scale_for_profile(identity) == pytest.approx(
            planted_scale[identity],
            rel=1.0e-9,
        )

    order = np.arange(len(bank.observations.identities))[::-1]
    reordered = fit_mosaic_component_profiles(bank.reorder_profiles(order))
    assert reordered.gaussian_sigma_rad == pytest.approx(result.gaussian_sigma_rad, abs=0.0)
    assert reordered.lorentzian_half_width_rad == pytest.approx(
        result.lorentzian_half_width_rad,
        abs=0.0,
    )
    assert reordered.lorentzian_probability == pytest.approx(
        result.lorentzian_probability,
        abs=2.0e-13,
    )
    assert reordered.objective == pytest.approx(result.objective, abs=1.0e-22)
    np.testing.assert_allclose(
        reordered.predicted_intensity,
        result.predicted_intensity[order],
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="source-profile provenance"):
        replace(
            bank,
            observations=replace(bank.observations, source_revision="stale-source.v1"),
        )
    with pytest.raises(ValueError, match="layered_family_m disagrees"):
        MosaicReflectionGroupKey(
            "contradictory-m0",
            "contradictory-rods.v1",
            ((0, 0),),
            "COLLAPSED_00L",
            layered_family_m=1,
            layered_integer_L=2,
        )
    shifted_layout = replace(
        bank.gaussian_profiles[0].profile,
        phi_bin_edges_rad=bank.gaussian_profiles[0].profile.phi_bin_edges_rad + 0.01,
    )
    with pytest.raises(ValueError, match="changed component or frozen profile provenance"):
        replace(
            bank,
            gaussian_profiles=(
                replace(bank.gaussian_profiles[0], profile=shifted_layout),
                *bank.gaussian_profiles[1:],
            ),
        )
    with pytest.raises(ValueError, match="changed component or frozen profile provenance"):
        replace(
            bank,
            gaussian_profiles=(
                replace(bank.gaussian_profiles[0], component_kind="lorentzian"),
                *bank.gaussian_profiles[1:],
            ),
        )

    explicit_nonzero = next(
        identity for identity in bank.observations.identities if identity.branch_id == 2
    )
    assert replace(explicit_nonzero, analytic_branch_id=2).analytic_branch_id == 2
    with pytest.raises(ValueError, match="explicit nonzero profiles"):
        replace(explicit_nonzero, analytic_branch_id=0)


def test_mosaic_profile_weighting_preserves_weak_profile_shape_leverage() -> None:
    bank, _ = _analytic_mosaic_profile_bank()
    profile_index = 0
    m0_identity = bank.observations.identities[profile_index]
    assert m0_identity.group_key.layered_family_m == 0
    assert m0_identity.branch_id is None
    assert m0_identity.analytic_branch_id == 0
    perturbation = 1.0 + 0.03 * np.linspace(-1.0, 1.0, bank.observations.signal.shape[1])
    observed_signal = np.array(bank.observations.signal, copy=True)
    observed_signal[profile_index] *= perturbation
    perturbed = replace(
        bank,
        observations=replace(bank.observations, signal=observed_signal),
    )

    weak_factor = 1.0e-6
    weak_observed_signal = np.array(observed_signal, copy=True)
    weak_observed_signal[profile_index] *= weak_factor

    def weaken_profile(component: MosaicComponentProfile) -> MosaicComponentProfile:
        signal = np.array(component.profile.signal, copy=True)
        signal[profile_index] *= weak_factor
        return replace(component, profile=replace(component.profile, signal=signal))

    weak = replace(
        perturbed,
        observations=replace(perturbed.observations, signal=weak_observed_signal),
        gaussian_profiles=tuple(weaken_profile(item) for item in perturbed.gaussian_profiles),
        lorentzian_profiles=tuple(weaken_profile(item) for item in perturbed.lorentzian_profiles),
    )

    baseline = fit_mosaic_component_profiles(bank)
    reference = fit_mosaic_component_profiles(perturbed)
    rescaled = fit_mosaic_component_profiles(weak)

    assert reference.weighting_id == (
        "profile_shape_profiled_scale_optional_additive_basis_angle_intensity_l2.v5"
    )
    assert reference.objective > baseline.objective + 1.0e-8
    assert reference.profile_relative_l2_residual[profile_index] > 0.0
    np.testing.assert_allclose(
        rescaled.width_pair_objective,
        reference.width_pair_objective,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        rescaled.width_pair_eta,
        reference.width_pair_eta,
        rtol=2.0e-12,
        atol=2.0e-13,
    )


def test_mosaic_fit_projects_affine_background_and_accepts_measured_provenance() -> None:
    bank, planted_scale = _analytic_mosaic_profile_bank()
    observations = bank.observations
    centers = 0.5 * (observations.phi_bin_edges_rad[:, :-1] + observations.phi_bin_edges_rad[:, 1:])
    coordinate = centers / np.max(np.abs(centers), axis=1, keepdims=True)
    background = 4.0 + 0.7 * coordinate
    background[~observations.valid] = 0.0
    measured_intensity = observations.intensity + background
    measured_normalization = np.array(observations.normalization, copy=True)
    measured_normalization[observations.valid] *= 1.0 + 0.2 * (coordinate[observations.valid] + 1.0)
    measured_signal = measured_intensity * measured_normalization
    measured = replace(
        observations,
        signal=measured_signal,
        normalization=measured_normalization,
        source_revision=None,
        observation_revision="measured-affine-background.v1",
    )
    basis_values = np.stack((np.ones(coordinate.shape), coordinate), axis=-1)
    basis_values[~measured.valid] = 0.0
    nuisance = MosaicProfileNuisanceBasis(
        measured.identities,
        basis_values,
        measured.valid,
        "constant-plus-linear.v1",
        measured.profile_revision,
    )
    measured_bank = replace(bank, observations=measured)
    result = fit_mosaic_component_profiles(measured_bank, nuisance_basis=nuisance)

    with pytest.raises(ValueError, match="layout or revision"):
        fit_mosaic_component_profiles(
            measured_bank,
            nuisance_basis=replace(nuisance, profile_revision="stale-profile-revision.v1"),
        )

    assert math.degrees(result.gaussian_sigma_rad) == pytest.approx(2.0, abs=1.0e-12)
    assert math.degrees(result.lorentzian_half_width_rad) == pytest.approx(0.5, abs=1.0e-12)
    assert result.lorentzian_probability == pytest.approx(0.1, abs=2.0e-9)
    assert result.objective <= 3.0e-20
    assert result.nuisance_basis_revision == nuisance.revision
    np.testing.assert_allclose(
        result.predicted_total_intensity[measured.valid],
        measured.intensity[measured.valid],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    for identity in result.profile_identities:
        assert result.scale_for_profile(identity) == pytest.approx(
            planted_scale[identity],
            rel=2.0e-9,
        )

    profile_multiplier = np.geomspace(1.0e-6, 1.0e6, len(measured.identities))
    rescaled_signal = measured.signal * profile_multiplier[:, None]
    rescaled = replace(
        measured_bank,
        observations=replace(measured, signal=rescaled_signal),
    )
    rescaled_result = fit_mosaic_component_profiles(rescaled, nuisance_basis=nuisance)
    np.testing.assert_allclose(
        rescaled_result.width_pair_objective,
        result.width_pair_objective,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    assert rescaled_result.lorentzian_probability == pytest.approx(
        result.lorentzian_probability,
        abs=3.0e-12,
    )

    affine_only = 8.0 + coordinate
    affine_only[~measured.valid] = 0.0
    with pytest.raises(ValueError, match="nuisance-projected observed energy"):
        fit_mosaic_component_profiles(
            replace(
                measured_bank,
                observations=replace(
                    measured,
                    signal=affine_only * measured.normalization,
                ),
            ),
            nuisance_basis=nuisance,
        )

    alternating = np.where(np.arange(coordinate.shape[1]) % 2 == 0, 1.0, -1.0)
    nearly_spanned_intensity = 0.01 * (1.0 + 0.2 * coordinate[0]) + 1.0e-16 * alternating

    def with_first_profile_intensity(
        component: MosaicComponentProfile,
        intensity: np.ndarray,
    ) -> MosaicComponentProfile:
        signal = np.array(component.profile.signal, copy=True)
        signal[0] = intensity * component.profile.normalization[0]
        return replace(component, profile=replace(component.profile, signal=signal))

    nearly_spanned_gaussian = with_first_profile_intensity(
        bank.gaussian_profiles[0],
        nearly_spanned_intensity,
    )
    nearly_spanned_lorentzian = with_first_profile_intensity(
        bank.lorentzian_profiles[0],
        nearly_spanned_intensity,
    )
    near_span_result = fit_mosaic_component_profiles(
        replace(
            measured_bank,
            gaussian_profiles=(nearly_spanned_gaussian, *bank.gaussian_profiles[1:]),
            lorentzian_profiles=(nearly_spanned_lorentzian, *bank.lorentzian_profiles[1:]),
        ),
        nuisance_basis=nuisance,
    )
    assert 1.0 <= near_span_result.width_pair_objective[0, 0] <= len(measured.identities)

    asymmetric_result = fit_mosaic_component_profiles(
        replace(
            measured_bank,
            gaussian_profiles=(nearly_spanned_gaussian, *bank.gaussian_profiles[1:]),
        ),
        nuisance_basis=nuisance,
    )
    np.testing.assert_array_equal(
        asymmetric_result.width_pair_objective[0],
        np.full(bank.lorentzian_half_width_rad.shape, len(measured.identities) + 1.0),
    )

    cancelling_gaussian = with_first_profile_intensity(
        bank.gaussian_profiles[0],
        2.0 + alternating,
    )
    cancelling_lorentzian = with_first_profile_intensity(
        bank.lorentzian_profiles[0],
        2.0 - alternating,
    )
    cancellation_result = fit_mosaic_component_profiles(
        replace(
            measured_bank,
            gaussian_profiles=(cancelling_gaussian, *bank.gaussian_profiles[1:]),
            lorentzian_profiles=(cancelling_lorentzian, *bank.lorentzian_profiles[1:]),
        ),
        nuisance_basis=nuisance,
    )
    assert math.isfinite(cancellation_result.objective)


@pytest.mark.parametrize(
    ("observation_scale", "component_scale"),
    (
        (1.0e150, 1.0e-150),
        (1.0e-150, 1.0e150),
    ),
)
def test_mosaic_normalization_is_stable_at_extreme_scales(
    observation_scale: float,
    component_scale: float,
) -> None:
    bank, planted_scale = _analytic_mosaic_profile_bank()

    def scaled_component(component: MosaicComponentProfile) -> MosaicComponentProfile:
        return replace(
            component,
            profile=replace(
                component.profile,
                signal=component.profile.signal * component_scale,
            ),
        )

    scaled_bank = replace(
        bank,
        observations=replace(
            bank.observations,
            signal=bank.observations.signal * observation_scale,
        ),
        gaussian_profiles=tuple(scaled_component(item) for item in bank.gaussian_profiles),
        lorentzian_profiles=tuple(scaled_component(item) for item in bank.lorentzian_profiles),
    )

    result = fit_mosaic_component_profiles(scaled_bank)
    assert math.degrees(result.gaussian_sigma_rad) == pytest.approx(2.0, abs=1.0e-12)
    assert math.degrees(result.lorentzian_half_width_rad) == pytest.approx(0.5, abs=1.0e-12)
    assert result.lorentzian_probability == pytest.approx(0.1, abs=2.0e-9)
    for identity in result.profile_identities:
        assert result.scale_for_profile(identity) == pytest.approx(
            planted_scale[identity] * observation_scale / component_scale,
            rel=1.0e-9,
        )


@pytest.mark.parametrize(
    ("component_name", "component_index", "expected_probability"),
    (("gaussian", 2, 0.0), ("lorentzian", 2, 1.0)),
)
def test_mosaic_component_profiles_report_identifiable_boundary_models(
    component_name: str,
    component_index: int,
    expected_probability: float,
) -> None:
    bank, planted_scale = _analytic_mosaic_profile_bank()
    profiles = bank.gaussian_profiles if component_name == "gaussian" else bank.lorentzian_profiles
    truth = profiles[component_index].profile
    active_widths = (
        bank.gaussian_sigma_rad if component_name == "gaussian" else bank.lorentzian_half_width_rad
    )
    active_intensity = np.asarray([item.profile.intensity for item in profiles])
    active_derivative = (active_intensity[3] - active_intensity[1]) / math.log(
        active_widths[3] / active_widths[1]
    )
    dependent_probe_intensity = truth.intensity + 0.01 * active_derivative
    dormant = bank.lorentzian_profiles if component_name == "gaussian" else bank.gaussian_profiles
    dependent_probe = replace(
        dormant[0],
        profile=replace(
            dormant[0].profile,
            signal=np.where(
                truth.valid,
                dependent_probe_intensity * truth.normalization,
                0.0,
            ),
        ),
    )
    if component_name == "gaussian":
        bank = replace(bank, lorentzian_profiles=(dependent_probe, *dormant[1:]))
    else:
        bank = replace(bank, gaussian_profiles=(dependent_probe, *dormant[1:]))
    observed_signal = np.asarray(
        [
            planted_scale[identity] * profile
            for identity, profile in zip(
                bank.observations.identities,
                truth.signal,
                strict=True,
            )
        ]
    )
    result = fit_mosaic_component_profiles(
        replace(bank, observations=replace(bank.observations, signal=observed_signal))
    )

    assert result.lorentzian_probability == expected_probability
    assert result.sensitivity_rank == 2
    assert np.isfinite(result.sensitivity_condition)
    if expected_probability == 0.0:
        assert result.gaussian_sigma_rad == pytest.approx(bank.gaussian_sigma_rad[component_index])
        assert result.lorentzian_half_width_rad is None
        assert result.active_parameter_names == (
            "log_gaussian_sigma",
            "lorentzian_probability_from_zero",
        )
    else:
        assert result.gaussian_sigma_rad is None
        assert result.lorentzian_half_width_rad == pytest.approx(
            bank.lorentzian_half_width_rad[component_index]
        )
        assert result.active_parameter_names == (
            "log_lorentzian_half_width",
            "gaussian_probability_from_one",
        )


def test_refined_mosaic_search_finds_an_interior_basin_hidden_between_face_widths() -> None:
    key = MosaicReflectionGroupKey(
        "hidden-interior",
        "hidden-interior-rods.v1",
        ((0, 0),),
        "COLLAPSED_00L",
    )
    identities = (MosaicProfileIdentity("hidden-interior", 0.1, key, None),)
    base = np.full(8, 10.0)
    direction = np.asarray((1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    gaussian_shape = np.asarray((0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0))
    lorentzian_shape = np.asarray((0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0))
    normalization = np.ones((1, 8))
    valid = np.ones((1, 8), dtype=np.bool_)
    phi_bin_edges = np.linspace(-1.0, 1.0, 9)[None, :]
    two_theta_bounds = np.asarray(((0.2, 0.3),))

    def profile(intensity: np.ndarray) -> MosaicProfileSet:
        return MosaicProfileSet(
            identities,
            np.asarray(intensity)[None, :],
            normalization,
            valid,
            "hidden-interior-response.v1",
            phi_bin_edges,
            two_theta_bounds,
            ("hidden-interior-frame.v1",),
            "analytic-source.v1",
        )

    def gaussian_field(width: float) -> np.ndarray:
        return base + 0.25 * math.log(width / 2.0) * gaussian_shape

    def lorentzian_field(width: float) -> np.ndarray:
        coordinate = math.log(width / 2.0, 2.0)
        return (
            base
            + (-1.0 + (16.0 / 3.0) * coordinate * (coordinate**2 - 1.0)) * direction
            + 0.2 * (coordinate + 0.5) * lorentzian_shape
        )

    gaussian_calls: list[float] = []
    lorentzian_calls: list[float] = []

    def gaussian_callback(width: float) -> MosaicProfileSet:
        gaussian_calls.append(width)
        return profile(gaussian_field(width))

    def lorentzian_callback(width: float) -> MosaicProfileSet:
        lorentzian_calls.append(width)
        return profile(lorentzian_field(width))

    search = fit_refined_mosaic_component_profiles(
        profile(base + 0.1 * direction),
        evaluate_gaussian_profile=gaussian_callback,
        evaluate_lorentzian_profile=lorentzian_callback,
        gaussian_sigma_bounds_rad=(1.0, 4.0),
        lorentzian_half_width_bounds_rad=(1.0, 4.0),
        coarse_width_count=3,
        refinement_width_count=5,
        refinement_levels=2,
        near_optimal_objective_delta=0.0,
    )

    assert search.fit.gaussian_sigma_rad == pytest.approx(2.0)
    assert search.fit.lorentzian_half_width_rad == pytest.approx(math.sqrt(2.0))
    assert search.fit.lorentzian_probability == pytest.approx(0.1, abs=2.0e-9)
    assert search.fit.objective < 1.0e-28
    assert math.sqrt(2.0) in lorentzian_calls
    assert search.bank.gaussian_sigma_rad.size == len(gaussian_calls)
    assert search.bank.lorentzian_half_width_rad.size == len(lorentzian_calls)
    assert len(gaussian_calls) == len(set(gaussian_calls))
    assert len(lorentzian_calls) == len(set(lorentzian_calls))


def test_continuous_mosaic_profiles_integrate_signal_and_normalization_before_division() -> None:
    from types import SimpleNamespace

    base = load_simulation_config(
        Path(__file__).resolve().parents[1] / "configs" / "bi2se3_simulation.yaml"
    )
    config = replace(base, source=replace(base.source, sample_count=1))
    inputs = build_configured_simulation_inputs(config)
    frame = build_osc_angle_frame(
        mean_direction_lab=config.source.mean_direction_lab,
        instrument=inputs.instrument,
        sample_intersection_lab_m=inputs.incident.states.sample_intersection_lab_m[0],
        revision="mosaic-profile-test.v1",
    )
    center = detector_coordinates_to_angles(
        config.instrument.detector_reference_coordinate_px[0],
        config.instrument.detector_reference_coordinate_px[1] - 100.0,
        instrument=inputs.instrument,
        angle_frame=frame,
    )
    assert bool(center.valid)
    test_rods = (Rod(0, 0), Rod(1, 0))
    reference_column = config.instrument.detector_reference_coordinate_px[0]
    reference_row = config.instrument.detector_reference_coordinate_px[1]

    def spatial_density(
        column_px: np.ndarray,
        row_px: np.ndarray,
    ) -> np.ndarray:
        return np.stack(
            (
                2.0
                + 2.0e-3 * (column_px - reference_column)
                + 1.0e-5 * (row_px - reference_row) ** 2,
                3.0
                + 1.0e-3 * (row_px - reference_row)
                + 2.0e-5 * (column_px - reference_column) ** 2,
            ),
            axis=-1,
        )

    class ConstantPerRodDetector:
        instrument = inputs.instrument
        rods = test_rods
        rod_catalog_revision = "test-rods.v1"
        returned_rods = test_rods
        evaluated_node_count = 0

        @classmethod
        def evaluate_detector_coordinates_all_roots(
            cls,
            column_px: np.ndarray,
            row_px: np.ndarray,
            *,
            execution_backend: str = "cpu",
        ) -> object:
            assert execution_backend == "cpu"
            shape = np.broadcast_shapes(column_px.shape, row_px.shape)
            cls.evaluated_node_count += int(np.prod(shape))
            column, row = np.broadcast_arrays(column_px, row_px)
            per_rod = spatial_density(column, row)
            return SimpleNamespace(
                column_px=np.asarray(column_px),
                row_px=np.asarray(row_px),
                rods=cls.returned_rods,
                rod_catalog_revision=cls.rod_catalog_revision,
                branch=None,
                per_rod_density_A2_per_px2=per_rod,
                caustic=np.zeros((*shape, 2), dtype=np.bool_),
                root_policy="all_retained_roots.v1",
                measure_id="raw_detector_coordinate_density_A2_per_px2.v1",
                source_revision="constant-source.v1",
                execution_backend="numba_cpu_source_averaged.v1",
                execution_device=None,
            )

    definitions = (
        MosaicProfileDefinition(
            identity=MosaicProfileIdentity(
                dataset_id="five-degree",
                incidence_angle_rad=math.radians(5.0),
                group_key=MosaicReflectionGroupKey(
                    group_id="test-00L-6",
                    rod_catalog_revision="test-rods.v1",
                    member_rod_hk=((0, 0),),
                    branch_mode="COLLAPSED_00L",
                    layered_family_m=0,
                    layered_integer_L=6,
                ),
                branch_id=None,
            ),
            center_two_theta_rad=float(center.two_theta_rad),
            center_phi_rad=float(center.phi_rad),
            two_theta_half_width_rad=math.radians(0.02),
            phi_half_width_rad=math.radians(0.2),
            phi_bin_count=7,
            two_theta_gauss_order=4,
            phi_gauss_order=4,
        ),
        MosaicProfileDefinition(
            identity=MosaicProfileIdentity(
                dataset_id="five-degree",
                incidence_angle_rad=math.radians(5.0),
                group_key=MosaicReflectionGroupKey(
                    group_id="test-m1-L8",
                    rod_catalog_revision="test-rods.v1",
                    member_rod_hk=((1, 0),),
                    branch_mode="EXPLICIT_NONZERO",
                    layered_family_m=1,
                    layered_integer_L=8,
                ),
                branch_id=1,
                analytic_branch_id=1,
            ),
            center_two_theta_rad=float(center.two_theta_rad),
            center_phi_rad=float(center.phi_rad),
            two_theta_half_width_rad=math.radians(0.02),
            phi_half_width_rad=math.radians(0.2),
            phi_bin_count=7,
            excluded_phi_bin_indices=(1, 5),
        ),
        MosaicProfileDefinition(
            identity=MosaicProfileIdentity(
                dataset_id="five-degree",
                incidence_angle_rad=math.radians(5.0),
                group_key=MosaicReflectionGroupKey(
                    group_id="test-m1-L8",
                    rod_catalog_revision="test-rods.v1",
                    member_rod_hk=((1, 0),),
                    branch_mode="EXPLICIT_NONZERO",
                    layered_family_m=1,
                    layered_integer_L=8,
                ),
                branch_id=2,
                analytic_branch_id=1,
            ),
            center_two_theta_rad=float(center.two_theta_rad),
            center_phi_rad=float(center.phi_rad) + math.radians(0.5),
            two_theta_half_width_rad=math.radians(0.02),
            phi_half_width_rad=math.radians(0.2),
            phi_bin_count=7,
        ),
    )
    profiles = evaluate_continuous_mosaic_profiles(
        ConstantPerRodDetector(),
        angle_frame=frame,
        definitions=definitions,
        profile_revision="constant-density.v1",
    )

    assert profiles.signal.shape == (3, 7)
    expected_node_count = 7 * 4 * 4 + (7 - 2) * 2 * 2 + 7 * 2 * 2
    assert ConstantPerRodDetector.evaluated_node_count == expected_node_count
    expected_valid = np.ones((3, 7), dtype=np.bool_)
    expected_valid[1, (1, 5)] = False
    np.testing.assert_array_equal(profiles.valid, expected_valid)
    assert profiles.source_revision == "constant-source.v1"
    assert profiles.source_revision != profiles.profile_revision
    np.testing.assert_allclose(
        profiles.signal,
        profiles.intensity * profiles.normalization,
        rtol=2.0e-15,
        atol=0.0,
    )

    expected_signal = np.zeros_like(profiles.signal)
    expected_normalization = np.zeros_like(profiles.normalization)
    pointwise_average = np.zeros_like(profiles.intensity)
    for profile_index, definition in enumerate(definitions):
        theta_node, theta_rule_weight = np.polynomial.legendre.leggauss(
            definition.two_theta_gauss_order
        )
        phi_node, phi_rule_weight = np.polynomial.legendre.leggauss(definition.phi_gauss_order)
        theta = definition.center_two_theta_rad + definition.two_theta_half_width_rad * theta_node
        theta_weight = definition.two_theta_half_width_rad * theta_rule_weight
        phi_edges = np.linspace(
            definition.center_phi_rad - definition.phi_half_width_rad,
            definition.center_phi_rad + definition.phi_half_width_rad,
            definition.phi_bin_count + 1,
        )
        for bin_index, (phi_lower, phi_upper) in enumerate(pairwise(phi_edges)):
            if bin_index in definition.excluded_phi_bin_indices:
                continue
            phi_half_width = 0.5 * (phi_upper - phi_lower)
            phi = 0.5 * (phi_lower + phi_upper) + phi_half_width * phi_node
            theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
            measure = angles_to_detector_coordinate_area_measure(
                theta_grid,
                phi_grid,
                instrument=inputs.instrument,
                angle_frame=frame,
            )
            area_weight = theta_weight[:, None] * (phi_half_width * phi_rule_weight)[None, :]
            density = spatial_density(
                measure.coordinates.column_px,
                measure.coordinates.row_px,
            )[..., 0 if profile_index == 0 else 1]
            jacobian = measure.detector_area_jacobian_px2_per_rad2
            expected_signal[profile_index, bin_index] = np.sum(density * jacobian * area_weight)
            expected_normalization[profile_index, bin_index] = np.sum(jacobian * area_weight)
            pointwise_average[profile_index, bin_index] = np.sum(density * area_weight) / np.sum(
                area_weight
            )
    np.testing.assert_allclose(profiles.signal, expected_signal, rtol=3.0e-15, atol=0.0)
    np.testing.assert_allclose(
        profiles.normalization,
        expected_normalization,
        rtol=3.0e-15,
        atol=0.0,
    )
    assert (
        np.max(np.abs(profiles.intensity[profiles.valid] - pointwise_average[profiles.valid]))
        > 1.0e-7
    )

    ConstantPerRodDetector.returned_rods = tuple(reversed(test_rods))
    with pytest.raises(ValueError, match="changed the physical rod axis or ordering"):
        evaluate_continuous_mosaic_profiles(
            ConstantPerRodDetector(),
            angle_frame=frame,
            definitions=definitions,
            profile_revision="constant-density.v1",
        )

    ConstantPerRodDetector.returned_rods = test_rods
    stale_group = replace(
        definitions[0].identity.group_key,
        rod_catalog_revision="stale-test-rods.v1",
    )
    stale_definition = replace(
        definitions[0],
        identity=replace(definitions[0].identity, group_key=stale_group),
    )
    with pytest.raises(ValueError, match="detector rod catalog revision"):
        evaluate_continuous_mosaic_profiles(
            ConstantPerRodDetector(),
            angle_frame=frame,
            definitions=(stale_definition,),
            profile_revision="constant-density.v1",
        )


def test_sparse_ordered_intensity_response_matches_direct_m0_and_nonzero_profiles() -> None:
    from rasim_next.fitting import (
        OrderedIntensityObservations,
        OrderedIntensityPeakCenterObservations,
        compile_ordered_intensity_response,
        compile_source_averaged_ordered_intensity_response,
        fit_ordered_intensity_series,
        probe_ordered_intensity_inverse_boundary_bins,
    )
    from rasim_next.ordered import Bi2X3QuintupleLayerParameters
    from rasim_next.pipeline.configured_simulation import configured_rod_catalog_revision

    base = load_simulation_config(
        Path(__file__).resolve().parents[1] / "configs" / "bi2se3_simulation.yaml"
    )
    config = replace(
        base,
        source=replace(
            base.source,
            spatial_sigma_m=(0.0, 0.0),
            divergence_sigma_rad=(0.0, 0.0),
            wavelength_sigma_A=0.0,
            sample_count=1,
        ),
        mosaic=replace(
            base.mosaic,
            gaussian_sigma_deg=2.0,
            lorentzian_hwhm_deg=0.5,
            lorentzian_probability=0.1,
        ),
    )
    inputs = build_configured_simulation_inputs(config)
    context = build_nominal_ewald_context(inputs)
    frame = build_osc_angle_frame(
        mean_direction_lab=config.source.mean_direction_lab,
        instrument=inputs.instrument,
        sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
        revision="ordered-response-test.v1",
    )
    markers = evaluate_nominal_integer_l_markers(context)
    m0_prediction = ExactTagGeometryModel(
        build_configured_geometry_inputs(config)
    ).predict_m0_minimum_tilt_exact_l_landmarks((6,))
    assert tuple(m0_prediction.detector_status) == ("VALID",)
    nonzero_index = int(np.flatnonzero((markers.family_m == 1) & (markers.root_sign != 0))[0])
    marker_angles = detector_coordinates_to_angles(
        np.asarray((m0_prediction.coordinates_px[0, 0], markers.column_px[nonzero_index])),
        np.asarray((m0_prediction.coordinates_px[0, 1], markers.row_px[nonzero_index])),
        instrument=inputs.instrument,
        angle_frame=frame,
    )
    assert np.all(marker_angles.valid & marker_angles.azimuth_valid)
    revision = configured_rod_catalog_revision(inputs)
    dataset_id = "five-degree-response-test"
    definitions: list[MosaicProfileDefinition] = []
    for profile_index in range(2):
        index = nonzero_index
        family = 0 if profile_index == 0 else int(markers.family_m[index])
        integer_l = 6 if profile_index == 0 else int(markers.integer_L[index])
        root_sign = 0 if profile_index == 0 else int(markers.root_sign[index])
        collapsed = family == 0
        definitions.append(
            MosaicProfileDefinition(
                identity=MosaicProfileIdentity(
                    dataset_id=dataset_id,
                    incidence_angle_rad=math.radians(5.0),
                    group_key=MosaicReflectionGroupKey(
                        group_id=f"response:m={family}:L={integer_l}",
                        rod_catalog_revision=revision,
                        member_rod_hk=(
                            ((0, 0),) if collapsed else markers.contributing_rod_hk[index]
                        ),
                        branch_mode="COLLAPSED_00L" if collapsed else "EXPLICIT_NONZERO",
                        layered_family_m=family,
                        layered_integer_L=integer_l,
                    ),
                    branch_id=None if collapsed else (1 if root_sign < 0 else 2),
                    analytic_branch_id=0 if collapsed else int(markers.branch[index]),
                ),
                center_two_theta_rad=float(marker_angles.two_theta_rad[profile_index]),
                center_phi_rad=float(marker_angles.phi_rad[profile_index]),
                two_theta_half_width_rad=math.radians(0.04),
                phi_half_width_rad=math.radians(5.0),
                phi_bin_count=21,
                two_theta_gauss_order=4,
                phi_gauss_order=4,
            )
        )
    frozen_definitions = tuple(definitions)
    audited_definitions = probe_ordered_intensity_inverse_boundary_bins(
        context.geometry,
        angle_frame=frame,
        definitions=frozen_definitions,
    )
    assert tuple(definition.excluded_phi_bin_indices for definition in audited_definitions) == (
        (),
        (6, 8, 9, 10, 11, 12, 13),
    )
    assert (
        probe_ordered_intensity_inverse_boundary_bins(
            context.geometry,
            angle_frame=frame,
            definitions=audited_definitions,
        )
        == audited_definitions
    )
    stale_identity = frozen_definitions[0].identity
    stale_definitions = (
        replace(
            frozen_definitions[0],
            identity=replace(
                stale_identity,
                group_key=replace(
                    stale_identity.group_key,
                    rod_catalog_revision="stale-ordered-response-rods.v1",
                ),
            ),
        ),
        *frozen_definitions[1:],
    )
    with pytest.raises(ValueError, match="rod catalog revision"):
        compile_ordered_intensity_response(
            context.geometry,
            angle_frame=frame,
            definitions=stale_definitions,
        )
    response = compile_ordered_intensity_response(
        context.geometry,
        angle_frame=frame,
        definitions=frozen_definitions,
    )
    assert response.excluded_phi_bin_indices == ((), (6, 8, 9, 10, 11, 12, 13))
    assert response.topology_probe_revision == "inverse_root_signature_grid_17x17.v1"

    def direct(strength: Bi2X3FiniteStackStrength) -> np.ndarray:
        candidate_inputs = replace(
            inputs,
            strength=strength,
            bragg_space=MosaicBraggSpace(inputs.bragg_space.config, strength),
        )
        profiles = evaluate_continuous_mosaic_profiles(
            build_source_averaged_detector(candidate_inputs),
            angle_frame=frame,
            definitions=audited_definitions,
            profile_revision="ordered-response-test.v1",
        )
        return np.sum(profiles.signal, axis=1, dtype=np.float64)

    baseline_predicted = response.predict_mass_A2(inputs.strength.structure_parameters)
    np.testing.assert_allclose(baseline_predicted, direct(inputs.strength), rtol=5.0e-11)
    baseline = Bi2X3QuintupleLayerParameters.from_crystal(inputs.crystal)
    candidate = replace(
        inputs.strength,
        structure_parameters=replace(
            baseline,
            bi_occupancy=0.93,
            se1_occupancy=0.81,
            se2_occupancy=0.72,
            u_radial_A2=0.007,
            u_normal_A2=0.034,
        ),
    )
    np.testing.assert_allclose(
        response.predict_mass_A2(candidate.structure_parameters),
        direct(candidate),
        rtol=5.0e-11,
    )
    np.testing.assert_allclose(
        response.predict_mass_direct_A2(candidate),
        direct(candidate),
        rtol=5.0e-11,
    )
    m0_term = response.term_observation_index == 0
    m0_rod = response.rods.index(next(rod for rod in response.rods if rod.family_m == 0))
    m0_term &= response.term_rod_index == m0_rod
    assert set(response.term_root_sign[m0_term]) == {-1, 1}

    distributed_config = replace(config, source=replace(base.source, sample_count=2))
    distributed_inputs = build_configured_simulation_inputs(distributed_config)
    assert configured_rod_catalog_revision(distributed_inputs) == revision
    distributed_detector = build_source_averaged_detector(distributed_inputs)
    distributed_response = compile_source_averaged_ordered_intensity_response(
        distributed_detector,
        angle_frame=frame,
        definitions=audited_definitions,
        execution_backend="cpu",
    )

    def distributed_direct(strength: Bi2X3FiniteStackStrength) -> np.ndarray:
        from rasim_next.measurement import evaluate_continuous_per_rod_angle_signal

        points = evaluate_continuous_per_rod_angle_signal(
            distributed_detector.rebind_physics(strength_model=strength),
            angle_frame=frame,
            two_theta_rad=np.asarray(
                [definition.center_two_theta_rad for definition in audited_definitions]
            ),
            phi_rad=np.asarray([definition.center_phi_rad for definition in audited_definitions]),
            execution_backend="cpu",
        )
        rod_lookup = {(rod.h, rod.k): index for index, rod in enumerate(points.rods)}
        return np.asarray(
            [
                np.sum(
                    points.per_rod_signal_density_A2_per_rad2[
                        profile_index,
                        [
                            rod_lookup[rod_hk]
                            for rod_hk in definition.identity.group_key.member_rod_hk
                        ],
                    ]
                )
                for profile_index, definition in enumerate(audited_definitions)
            ]
        )

    zero_u = replace(candidate.structure_parameters, u_radial_A2=0.0, u_normal_A2=0.0)
    zero_u_strength = replace(distributed_inputs.strength, structure_parameters=zero_u)
    distributed_candidate = replace(
        distributed_inputs.strength,
        structure_parameters=candidate.structure_parameters,
    )
    np.testing.assert_allclose(
        distributed_response.predict_signal_density_A2_per_rad2(zero_u),
        distributed_direct(zero_u_strength),
        rtol=8.0e-11,
    )
    np.testing.assert_allclose(
        distributed_response.predict_signal_density_A2_per_rad2(candidate.structure_parameters),
        distributed_direct(distributed_candidate),
        rtol=2.0e-4,
    )
    assert distributed_response.source_state_count == 2
    assert distributed_response.source_revision == distributed_inputs.samples.source_revision

    second_dataset_id = "second-five-degree-response-test"
    second_response = replace(
        distributed_response,
        dataset_id=second_dataset_id,
        identities=tuple(
            replace(identity, dataset_id=second_dataset_id)
            for identity in distributed_response.identities
        ),
        occupancy_quadratic_chebyshev_signal_density_A2_per_rad2=(
            distributed_response.occupancy_quadratic_chebyshev_signal_density_A2_per_rad2
            * np.asarray((1.0, 1.2))[:, None, None]
        ),
        observable_revision="second-point-observable.v1",
    )
    point_truth = candidate.structure_parameters
    point_observations = tuple(
        OrderedIntensityPeakCenterObservations(
            dataset_id=point_response.dataset_id,
            observable_revision=point_response.observable_revision,
            signal_density_A2_per_rad2=point_response.predict_signal_density_A2_per_rad2(
                point_truth
            ),
        )
        for point_response in (distributed_response, second_response)
    )
    point_fit = fit_ordered_intensity_series(
        (distributed_response, second_response),
        point_observations,
        base_strength=distributed_inputs.strength,
        active_parameter_names=("u_normal_A2",),
        initial_parameters=replace(point_truth, u_normal_A2=0.02),
        relative_scale_mode=False,
        required_source_state_count=2,
        required_source_revision=distributed_inputs.samples.source_revision,
    )
    assert point_fit.predicted_mass_A2 == ()
    assert len(point_fit.predicted_signal_density_A2_per_rad2) == 2
    assert point_fit.observable_measure_id == (
        "selected_group_angular_signal_density_A2_per_rad2.v1"
    )
    assert point_fit.structure_representative.u_normal_A2 == pytest.approx(
        point_truth.u_normal_A2,
        abs=2.0e-6,
    )
    with pytest.raises(ValueError, match="peak-center responses require"):
        fit_ordered_intensity_series(
            (distributed_response, second_response),
            (
                OrderedIntensityObservations(
                    dataset_id=distributed_response.dataset_id,
                    observable_revision=distributed_response.observable_revision,
                    mass_A2=point_observations[0].signal_density_A2_per_rad2,
                ),
                point_observations[1],
            ),
            base_strength=distributed_inputs.strength,
            active_parameter_names=(),
        )
    with pytest.raises(ValueError, match="required source-state count"):
        fit_ordered_intensity_series(
            (distributed_response, second_response),
            point_observations,
            base_strength=distributed_inputs.strength,
            active_parameter_names=(),
            required_source_state_count=250,
        )
    with pytest.raises(ValueError, match="required source revision"):
        fit_ordered_intensity_series(
            (distributed_response, second_response),
            point_observations,
            base_strength=distributed_inputs.strength,
            active_parameter_names=(),
            required_source_revision="stale-source-revision",
        )


def test_ordered_intensity_fit_enforces_freeze_gauge_revision_and_rank_contracts() -> None:
    from rasim_next.fitting import (
        OrderedIntensityDatasetResponse,
        OrderedIntensityIdentifiabilityError,
        OrderedIntensityObservations,
        fit_ordered_intensity_series,
        ordered_intensity_structure_model_revision,
    )
    from rasim_next.ordered import Bi2X3QuintupleLayerParameters

    config = load_simulation_config(
        Path(__file__).resolve().parents[1] / "configs" / "bi2se3_simulation.yaml"
    )
    inputs = build_configured_simulation_inputs(config)
    strength = inputs.strength
    fixed = Bi2X3QuintupleLayerParameters.from_crystal(inputs.crystal)
    rods = (Rod(-1, 0),)
    term_l = np.arange(1.0, 9.0)
    amplitude_basis = np.asarray(
        (
            (1.0, 0.2, 0.1),
            (0.2, 1.0, 0.3),
            (0.1, 0.4, 1.0),
            (1.0, 1.0, 0.2),
            (0.3, 1.0, 1.0),
            (1.0, 0.3, 1.0),
            (0.7, 0.5, 1.2),
            (1.1, 0.8, 0.4),
        )
    )
    quadratic = np.column_stack(
        (
            amplitude_basis[:, 0] ** 2,
            amplitude_basis[:, 1] ** 2,
            amplitude_basis[:, 2] ** 2,
            2.0 * amplitude_basis[:, 0] * amplitude_basis[:, 1],
            2.0 * amplitude_basis[:, 0] * amplitude_basis[:, 2],
            2.0 * amplitude_basis[:, 1] * amplitude_basis[:, 2],
        )
    )
    dataset_id = "ordered-intensity-contract"
    catalog_revision = "ordered-intensity-contract-rods.v1"
    identities = tuple(
        MosaicProfileIdentity(
            dataset_id=dataset_id,
            incidence_angle_rad=math.radians(10.0),
            group_key=MosaicReflectionGroupKey(
                group_id=f"contract:L={int(ell)}",
                rod_catalog_revision=catalog_revision,
                member_rod_hk=((-1, 0),),
                branch_mode="EXPLICIT_NONZERO",
                layered_family_m=1,
                layered_integer_L=int(ell),
            ),
            branch_id=2,
            analytic_branch_id=2,
        )
        for ell in term_l
    )
    response = OrderedIntensityDatasetResponse(
        dataset_id=dataset_id,
        incidence_angle_rad=math.radians(10.0),
        identities=identities,
        rods=rods,
        term_observation_index=np.arange(term_l.size),
        term_rod_index=np.zeros(term_l.size, dtype=np.int64),
        term_L=term_l,
        term_fixed_mass_per_strength=np.linspace(0.8, 1.2, term_l.size),
        term_root_sign=np.ones(term_l.size, dtype=np.int8),
        term_occupancy_quadratic_strength_A2=quadratic,
        term_q_radial_squared_Ainv2=np.asarray((0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)),
        term_q_normal_squared_Ainv2=np.asarray((3.2, 2.8, 2.1, 1.7, 1.2, 0.8, 0.4, 3.7)),
        normalization_mass_px2=np.ones(term_l.size),
        reciprocal_basis_Ainv=strength.reciprocal_basis_Ainv,
        k_norm_Ainv=inputs.bragg_space.config.k_norm_Ainv,
        fixed_structure_parameters=fixed,
        rod_catalog_revision=catalog_revision,
        structure_model_revision=ordered_intensity_structure_model_revision(strength),
        mosaic_model_revision="ordered-intensity-contract-mosaic.v1",
        angle_frame_revision="ordered-intensity-contract-frame.v1",
        source_revision="ordered-intensity-contract-source.v1",
        sample_geometry_revision="ordered-intensity-contract-sample.v1",
        material_revision="ordered-intensity-contract-material.v1",
        observable_revision="ordered-intensity-contract-observable.v1",
        excluded_phi_bin_indices=((),) * len(identities),
        topology_probe_revision="ordered-intensity-contract-topology.v1",
    )
    truth = replace(
        fixed,
        bi_occupancy=0.7,
        se1_occupancy=0.9,
        se2_occupancy=0.8,
        u_radial_A2=0.007,
        u_normal_A2=0.034,
    )
    observation = OrderedIntensityObservations(
        dataset_id=dataset_id,
        observable_revision=response.observable_revision,
        mass_A2=response.predict_mass_A2(truth),
    )
    relative_initial = replace(fixed, bi_occupancy=0.7)
    relative = fit_ordered_intensity_series(
        (response,),
        (observation,),
        base_strength=strength,
        active_parameter_names=(
            "se1_occupancy",
            "se2_occupancy",
            "u_radial_A2",
            "u_normal_A2",
        ),
        initial_parameters=relative_initial,
        relative_scale_mode=True,
    )
    np.testing.assert_allclose(
        relative.occupancy_ratios,
        (1.0, 0.9 / 0.7, 0.8 / 0.7),
        rtol=0.0,
        atol=2.0e-10,
    )
    assert relative.occupancy_ratio_reference == "bi_occupancy"
    np.testing.assert_allclose(
        (
            relative.structure_representative.bi_occupancy,
            relative.structure_representative.se1_occupancy,
            relative.structure_representative.se2_occupancy,
        ),
        (0.7 / 0.9, 1.0, 0.8 / 0.9),
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        relative.dataset_scales[0] * relative.predicted_mass_A2[0],
        observation.mass_A2,
        rtol=2.0e-10,
    )

    bounded_relative = fit_ordered_intensity_series(
        (response,),
        (observation,),
        base_strength=strength,
        active_parameter_names=(
            "se1_occupancy",
            "se2_occupancy",
            "u_radial_A2",
            "u_normal_A2",
        ),
        initial_parameters=replace(
            relative_initial,
            se1_occupancy=0.7,
            se2_occupancy=0.7,
        ),
        relative_scale_mode=True,
        active_parameter_bounds={
            "se1_occupancy": (0.0, 1.0),
            "se2_occupancy": (0.0, 1.0),
        },
    )
    assert bounded_relative.occupancy_ratios[1] <= 1.0
    assert bounded_relative.active_bounds[0]

    with pytest.raises(ValueError, match="active parameter"):
        fit_ordered_intensity_series(
            (response,),
            (observation,),
            base_strength=strength,
            active_parameter_names=("se1_occupancy",),
            active_parameter_bounds={"se2_occupancy": (0.0, 1.0)},
        )

    with pytest.raises(ValueError, match="narrow the canonical"):
        fit_ordered_intensity_series(
            (response,),
            (observation,),
            base_strength=strength,
            active_parameter_names=("u_radial_A2",),
            relative_scale_mode=False,
            active_parameter_bounds={"u_radial_A2": (-0.01, 0.1)},
        )

    inactive_outside_optimizer_bounds = replace(fixed, u_radial_A2=0.2)
    fully_frozen = fit_ordered_intensity_series(
        (response,),
        (observation,),
        base_strength=strength,
        active_parameter_names=(),
        initial_parameters=inactive_outside_optimizer_bounds,
        relative_scale_mode=False,
    )
    assert fully_frozen.structure_representative == inactive_outside_optimizer_bounds
    assert fully_frozen.active_parameter_names == ()

    with pytest.raises(OrderedIntensityIdentifiabilityError, match="common-scale gauge"):
        fit_ordered_intensity_series(
            (response,),
            (observation,),
            base_strength=strength,
            active_parameter_names=(
                "bi_occupancy",
                "se1_occupancy",
                "se2_occupancy",
                "u_radial_A2",
                "u_normal_A2",
            ),
        )
    with pytest.raises(ValueError, match="atomic positions are fixed"):
        fit_ordered_intensity_series(
            (response,),
            (observation,),
            base_strength=strength,
            active_parameter_names=("bi_delta_z_fractional",),
            relative_scale_mode=False,
        )
    with pytest.raises(ValueError, match="observable revision"):
        fit_ordered_intensity_series(
            (response,),
            (replace(observation, observable_revision="stale-observable.v1"),),
            base_strength=strength,
            active_parameter_names=(),
        )
    with pytest.raises(RuntimeError, match="optimization failed"):
        fit_ordered_intensity_series(
            (response,),
            (observation,),
            base_strength=strength,
            active_parameter_names=(
                "bi_occupancy",
                "se1_occupancy",
                "se2_occupancy",
                "u_radial_A2",
                "u_normal_A2",
            ),
            initial_parameters=fixed,
            relative_scale_mode=False,
            maximum_function_evaluations=1,
        )

    deficient = replace(
        response,
        identities=response.identities[:3],
        term_observation_index=np.arange(3),
        term_rod_index=response.term_rod_index[:3],
        term_L=response.term_L[:3],
        term_fixed_mass_per_strength=response.term_fixed_mass_per_strength[:3],
        term_root_sign=response.term_root_sign[:3],
        term_occupancy_quadratic_strength_A2=response.term_occupancy_quadratic_strength_A2[:3],
        term_q_radial_squared_Ainv2=response.term_q_radial_squared_Ainv2[:3],
        term_q_normal_squared_Ainv2=response.term_q_normal_squared_Ainv2[:3],
        normalization_mass_px2=response.normalization_mass_px2[:3],
        excluded_phi_bin_indices=response.excluded_phi_bin_indices[:3],
    )
    deficient_observation = OrderedIntensityObservations(
        dataset_id=dataset_id,
        observable_revision=deficient.observable_revision,
        mass_A2=deficient.predict_mass_A2(truth),
    )
    with pytest.raises(OrderedIntensityIdentifiabilityError, match="rank 3/5"):
        fit_ordered_intensity_series(
            (deficient,),
            (deficient_observation,),
            base_strength=strength,
            active_parameter_names=(
                "bi_occupancy",
                "se1_occupancy",
                "se2_occupancy",
                "u_radial_A2",
                "u_normal_A2",
            ),
            initial_parameters=fixed,
            relative_scale_mode=False,
        )


def test_mosaic_component_profiles_refuse_nuisance_projected_rank_deficiency() -> None:
    bank, planted_scale = _analytic_mosaic_profile_bank()
    reference = bank.gaussian_profiles[0].profile
    observed_signal = np.asarray(
        [
            planted_scale[identity] * profile
            for identity, profile in zip(
                bank.observations.identities,
                reference.signal,
                strict=True,
            )
        ]
    )
    degenerate = replace(
        bank,
        observations=replace(bank.observations, signal=observed_signal),
        lorentzian_profiles=tuple(
            replace(component, profile=reference) for component in bank.lorentzian_profiles
        ),
    )

    with pytest.raises(MosaicIdentifiabilityError) as caught:
        fit_mosaic_component_profiles(degenerate)
    assert caught.value.rank == 1
    assert caught.value.active_parameter_names == (
        "log_gaussian_sigma",
        "lorentzian_probability_from_zero",
    )
    assert caught.value.singular_values.shape == (len(caught.value.active_parameter_names),)
    assert not np.isfinite(caught.value.condition) or caught.value.condition > 1.0e8


def test_mosaic_component_profiles_reject_two_interior_eta_minima_for_one_width_pair() -> None:
    key = MosaicReflectionGroupKey(
        "eta-alias",
        "eta-alias-rods.v1",
        ((0, 0),),
        "COLLAPSED_00L",
    )
    identities = (
        MosaicProfileIdentity("eta-alias-a", 0.1, key, None),
        MosaicProfileIdentity("eta-alias-b", 0.2, key, None),
    )
    normalization = np.ones((2, 4))
    valid = np.ones((2, 4), dtype=np.bool_)
    phi_bin_edges = np.broadcast_to(np.linspace(-1.0, 1.0, 5), (2, 5)).copy()
    two_theta_bounds = np.asarray(((0.2, 0.3), (0.2, 0.3)))

    def profile(intensity: np.ndarray) -> MosaicProfileSet:
        return MosaicProfileSet(
            identities,
            intensity,
            normalization,
            valid,
            "eta-alias-response.v1",
            phi_bin_edges,
            two_theta_bounds,
            ("eta-alias-frame.v1",) * 2,
            "analytic-source.v1",
        )

    widths = np.asarray((1.0, 2.0))
    observed = np.asarray(((7.0, 22.0, 28.0, 25.0), (7.0, 22.0, 28.0, 25.0)))
    gaussian = (
        np.asarray(((23.0, 16.0, 30.0, 10.0), (2.0, 1.0, 3.0, 6.0))),
        np.asarray(((21.0, 5.0, 29.0, 7.0), (28.0, 19.0, 8.0, 6.0))),
    )
    lorentzian = (
        np.asarray(((2.0, 1.0, 3.0, 6.0), (23.0, 16.0, 30.0, 10.0))),
        np.asarray(((13.0, 23.0, 24.0, 1.0), (12.0, 21.0, 9.0, 28.0))),
    )
    bank = MosaicComponentProfileBank(
        observations=profile(observed),
        gaussian_sigma_rad=widths,
        gaussian_profiles=tuple(
            MosaicComponentProfile("gaussian", width, profile(component))
            for width, component in zip(widths, gaussian, strict=True)
        ),
        lorentzian_half_width_rad=widths,
        lorentzian_profiles=tuple(
            MosaicComponentProfile("lorentzian", width, profile(component))
            for width, component in zip(widths, lorentzian, strict=True)
        ),
    )

    with pytest.raises(MosaicIdentifiabilityError) as caught:
        fit_mosaic_component_profiles(bank)
    assert caught.value.reason == "global_alias"
    candidate = caught.value.candidate_result
    assert candidate is not None
    assert candidate.sensitivity_rank == caught.value.rank
    assert candidate.sensitivity_condition == caught.value.condition
    assert candidate.objective >= 0.0
    assert caught.value.rank == len(caught.value.active_parameter_names)
    assert np.isfinite(caught.value.condition)
    competing = caught.value.competing_solution_keys
    assert len(competing) == 2
    assert {
        (kind, gaussian_index, lorentzian_index)
        for kind, gaussian_index, lorentzian_index, _ in competing
    } == {("GL", 0, 0)}
    eta = sorted(item[3] for item in competing)
    assert 0.0 < eta[0] < 0.25
    assert 0.75 < eta[1] < 1.0
    assert sum(eta) == pytest.approx(1.0, abs=2.0e-12)
    competing_parameters = caught.value.competing_parameter_sets
    assert len(competing_parameters) == 2
    assert {
        (item.gaussian_sigma_rad, item.lorentzian_half_width_rad) for item in competing_parameters
    } == {(1.0, 1.0)}
    assert sorted(item.lorentzian_probability for item in competing_parameters) == pytest.approx(
        eta
    )
    assert all(
        item.objective == pytest.approx(candidate.objective) for item in competing_parameters
    )


def test_mosaic_component_profiles_reject_eta_alias_after_constant_projection() -> None:
    key = MosaicReflectionGroupKey(
        "projected-eta-alias",
        "projected-eta-alias-rods.v1",
        ((0, 0),),
        "COLLAPSED_00L",
    )
    identities = (
        MosaicProfileIdentity("projected-eta-a", 0.1, key, None),
        MosaicProfileIdentity("projected-eta-b", 0.2, key, None),
    )
    gaussian_row = np.asarray(
        (
            -0.5002908599651601,
            0.662805580105935,
            -0.45770187046739114,
            -0.05267588575647031,
            0.3478630360830864,
        )
    )
    lorentzian_row = np.asarray(
        (
            1.3715226089283317,
            -0.36140526514948973,
            0.2528490373494388,
            -0.7685581033334808,
            -0.49440827779479984,
        )
    )
    observed_row = np.asarray(
        (
            -0.6042013846407788,
            -0.062061226398191977,
            0.8628575607932455,
            -1.0103157251245258,
            0.813720775370251,
        )
    )
    normalization = np.ones((2, 5))
    valid = np.ones((2, 5), dtype=np.bool_)
    phi_bin_edges = np.broadcast_to(np.linspace(-1.0, 1.0, 6), (2, 6)).copy()
    two_theta_bounds = np.asarray(((0.2, 0.3), (0.2, 0.3)))

    def profile(intensity: np.ndarray, *, measured: bool = False) -> MosaicProfileSet:
        return MosaicProfileSet(
            identities=identities,
            signal=intensity,
            normalization=normalization,
            valid=valid,
            profile_revision="projected-eta-alias-response.v1",
            phi_bin_edges_rad=phi_bin_edges,
            two_theta_bounds_rad=two_theta_bounds,
            angle_frame_revisions=("projected-eta-alias-frame.v1",) * 2,
            source_revision=None if measured else "analytic-source.v1",
            observation_revision="projected-eta-alias-observation.v1" if measured else None,
        )

    offset = 2.0
    observed = profile(
        np.asarray((observed_row + offset, observed_row + offset)),
        measured=True,
    )
    gaussian = profile(
        np.asarray((gaussian_row + offset, lorentzian_row + offset)),
    )
    lorentzian = profile(
        np.asarray((lorentzian_row + offset, gaussian_row + offset)),
    )
    widths = np.asarray((1.0, 2.0))
    bank = MosaicComponentProfileBank(
        observations=observed,
        gaussian_sigma_rad=widths,
        gaussian_profiles=tuple(
            MosaicComponentProfile("gaussian", width, gaussian) for width in widths
        ),
        lorentzian_half_width_rad=widths,
        lorentzian_profiles=tuple(
            MosaicComponentProfile("lorentzian", width, lorentzian) for width in widths
        ),
    )
    constant = np.ones((2, 5, 1))
    nuisance = MosaicProfileNuisanceBasis(
        identities,
        constant,
        valid,
        "constant-projection.v1",
        observed.profile_revision,
    )

    with pytest.raises(MosaicIdentifiabilityError) as caught:
        fit_mosaic_component_profiles(bank, nuisance_basis=nuisance)
    assert caught.value.reason == "global_alias"
    eta = sorted(
        key[3]
        for key in caught.value.competing_solution_keys
        if key[0] == "GL" and key[1:3] == (0, 0)
    )
    assert len(eta) == 2
    assert 0.0 < eta[0] < 0.25
    assert 0.75 < eta[1] < 1.0
    assert sum(eta) == pytest.approx(1.0, abs=2.0e-12)


def test_mosaic_component_profiles_report_crossed_zero_energy_nonattainment() -> None:
    key = MosaicReflectionGroupKey(
        "crossed-zero-energy",
        "crossed-zero-energy-rods.v1",
        ((0, 0),),
        "COLLAPSED_00L",
    )
    identities = (
        MosaicProfileIdentity("crossed-a", 0.1, key, None),
        MosaicProfileIdentity("crossed-b", 0.2, key, None),
    )
    normalization = np.ones((2, 3))
    valid = np.ones((2, 3), dtype=np.bool_)
    phi_edges = np.broadcast_to(np.linspace(-1.0, 1.0, 4), (2, 4)).copy()
    theta_bounds = np.asarray(((0.2, 0.3), (0.2, 0.3)))
    gaussian_signal = np.asarray(((3.0, 3.0, 3.0), (4.0, 2.0, 3.0)))
    lorentzian_signal = np.asarray(((4.0, 4.0, 1.0), (3.0, 3.0, 3.0)))

    def profile(signal: np.ndarray, *, measured: bool = False) -> MosaicProfileSet:
        return MosaicProfileSet(
            identities=identities,
            signal=signal,
            normalization=normalization,
            valid=valid,
            profile_revision="crossed-zero-energy-response.v1",
            phi_bin_edges_rad=phi_edges,
            two_theta_bounds_rad=theta_bounds,
            angle_frame_revisions=("crossed-zero-energy-frame.v1",) * 2,
            source_revision=None if measured else "analytic-source.v1",
            observation_revision="crossed-zero-energy-observation.v1" if measured else None,
        )

    widths = np.asarray((1.0, 2.0))
    bank = MosaicComponentProfileBank(
        observations=profile(
            np.asarray(((4.0, 4.0, 1.0), (4.0, 2.0, 3.0))),
            measured=True,
        ),
        gaussian_sigma_rad=widths,
        gaussian_profiles=tuple(
            MosaicComponentProfile("gaussian", width, profile(gaussian_signal)) for width in widths
        ),
        lorentzian_half_width_rad=widths,
        lorentzian_profiles=tuple(
            MosaicComponentProfile("lorentzian", width, profile(lorentzian_signal))
            for width in widths
        ),
    )
    nuisance = MosaicProfileNuisanceBasis(
        identities,
        np.ones((2, 3, 1)),
        valid,
        "crossed-zero-energy-constant.v1",
        bank.observations.profile_revision,
    )

    with pytest.raises(MosaicIdentifiabilityError) as caught:
        fit_mosaic_component_profiles(bank, nuisance_basis=nuisance)
    assert caught.value.reason == "nonattained_boundary"


def test_bi2te3_config_builds_material_generic_quintuple_layer_strength() -> None:
    config = load_simulation_config(
        Path(__file__).resolve().parents[1] / "configs" / "bi2te3_simulation.yaml"
    )
    inputs = build_configured_simulation_inputs(
        replace(config, source=replace(config.source, sample_count=1))
    )

    assert config.structure_factor.model_id == "r3m_quintuple_finite_2h.v1"
    assert {site.element for site in inputs.crystal.sites} == {"Bi", "Te"}
    assert len(inputs.crystal.sites) == 15
    assert inputs.strength.site_labels == ("Bi", "Te1", "Te2")
    detector = build_source_averaged_detector(inputs)
    density = detector.evaluate_detector_density_all_roots(
        np.asarray([1453.0]),
        np.asarray([1360.0]),
        execution_backend="cpu",
    )
    assert density.source_state_count == 1
    assert np.isfinite(density.density_A2_per_px2[0])
    assert density.density_A2_per_px2[0] >= 0.0


def test_bi2te3_figure7_config_selects_fault_free_three_r_parent() -> None:
    from rasim_next.stacking import Parent

    config = load_simulation_config(
        Path(__file__).resolve().parents[1] / "configs" / "bi2te3_r3_simulation.yaml"
    )
    inputs = build_configured_simulation_inputs(
        replace(config, source=replace(config.source, sample_count=1))
    )

    assert config.structure_factor.model_id == "r3m_quintuple_finite_3r.v1"
    assert inputs.strength.parent is Parent.THREE_R
    assert inputs.strength.shared_disorder_epsilon == 0.0


def _synthetic_stacking_response(matrix: np.ndarray) -> CompiledStackingResponse:
    row_count = matrix.shape[0]
    return CompiledStackingResponse(
        component_response_A2=matrix,
        signed_hk=np.column_stack((np.arange(row_count), -np.arange(row_count))),
        l_coordinate=np.linspace(0.0, 1.0, row_count),
        wavelength_A=np.full(row_count, 1.540592925),
        fixed_model_revision="synthetic-fixed-model.v1",
    )


def test_stacking_population_fit_recovers_global_scale_and_profiles_phase_totals() -> None:
    rng = np.random.default_rng(20260728)
    matrix = rng.lognormal(mean=0.0, sigma=0.7, size=(40, 5))
    matrix += np.linspace(0.0, 1.0, 40)[:, None] ** np.arange(1, 6)[None, :]
    response = _synthetic_stacking_response(matrix)
    planted_domain = np.asarray((0.60, 0.16, 0.09, 0.10, 0.05))
    planted_phase = np.asarray((0.60, 0.25, 0.15))
    planted_scale = 7.25
    signal = matrix @ (planted_scale * planted_domain)
    variance = np.full(signal.size, 0.04)

    result = fit_stacking_phase_totals(
        response,
        signal,
        variance,
        profile_delta_chi_square=1.0,
    )

    assert result.component_ids == STACKING_COMPONENT_IDS
    assert result.phase_ids == STACKING_PHASE_IDS
    np.testing.assert_allclose(result.domain_amount, planted_scale * planted_domain, atol=2.0e-13)
    np.testing.assert_allclose(result.domain_fraction, planted_domain, atol=3.0e-14)
    np.testing.assert_allclose(result.phase_fraction, planted_phase, atol=3.0e-14)
    np.testing.assert_allclose(result.predicted_strength_A2, signal, atol=2.0e-13)
    assert result.global_scale == pytest.approx(planted_scale, abs=2.0e-13)
    assert result.chi_square < 1.0e-24
    assert result.response_rank == 5
    assert result.phase_contrast_rank == 2
    assert result.domain_response_full_rank
    assert result.active_component_ids == STACKING_COMPONENT_IDS
    assert np.all(result.phase_profile_bounds[:, 0] < planted_phase)
    assert np.all(result.phase_profile_bounds[:, 1] > planted_phase)
    assert not result.domain_fraction.flags.writeable
    assert not result.phase_profile_bounds.flags.writeable

    scale_factor = 1.0e-15 / planted_scale
    scaled = fit_stacking_phase_totals(
        response,
        scale_factor * signal,
        scale_factor**2 * variance,
    )
    assert scaled.global_scale == pytest.approx(1.0e-15, rel=2.0e-12)
    np.testing.assert_allclose(scaled.phase_fraction, planted_phase, atol=3.0e-14)
    np.testing.assert_allclose(scaled.phase_profile_bounds, result.phase_profile_bounds, atol=2e-13)

    pure = fit_stacking_phase_totals(response, planted_scale * matrix[:, 0], variance)
    np.testing.assert_allclose(pure.phase_fraction, (1.0, 0.0, 0.0), atol=2.0e-16)
    np.testing.assert_array_equal(pure.phase_estimate_on_boundary, (True, True, True))
    np.testing.assert_array_equal(
        (pure.phase_profile_bounds[0, 1], *pure.phase_profile_bounds[1:, 0]),
        (1.0, 0.0, 0.0),
    )


def test_stacking_population_fit_keeps_identifiable_phase_totals_when_hands_alias() -> None:
    coordinate = np.linspace(-1.0, 1.0, 31)
    two_h = 1.0 + coordinate**2
    four_h = 1.4 + 0.3 * coordinate + coordinate**4
    six_h = 0.8 - 0.2 * coordinate + np.exp(0.4 * coordinate)
    matrix = np.column_stack((two_h, four_h, four_h, six_h, six_h))
    response = _synthetic_stacking_response(matrix)
    planted_domain = np.asarray((0.55, 0.20, 0.10, 0.09, 0.06))
    signal = matrix @ (3.0 * planted_domain)

    result = fit_stacking_phase_totals(response, signal, np.ones(signal.size))

    np.testing.assert_allclose(result.phase_fraction, (0.55, 0.30, 0.15), atol=2.0e-13)
    assert result.response_rank == 3
    assert result.phase_contrast_rank == 2
    assert not result.domain_response_full_rank


def test_pbi2_stacking_profile_response_matches_direct_enumeration_and_rejects_m3_only() -> None:
    from rasim_next.core.contracts import RodQueryBatch
    from rasim_next.core.scattering import electron_squared_to_scattering_strength_A2
    from rasim_next.materials import read_crystal
    from rasim_next.ordered import pbi2_layer_amplitudes
    from rasim_next.reciprocal.lattice import ReciprocalLattice
    from rasim_next.stacking import (
        InitialPopulation,
        Parent,
        RegistryPhaseModel,
        RichEpsilonModel,
        registry_phase,
    )
    from rasim_next.stacking.enumeration import finite_intensity_by_enumeration

    root = Path(__file__).resolve().parents[1]
    crystal_revision = "7cf2a5e1957ea63d277c704cff390724175f96e6d26f982287490eedc24afbf9"
    crystal = read_crystal(
        root / "examples" / "pbi2" / "structures" / "PbI2_2H.cif",
        phase_id="pbi2-2h",
        expected_sha256=crystal_revision,
    )
    reciprocal = ReciprocalLattice.from_crystal(crystal)
    signed_hk = np.asarray(((-1, 0), (-1, 0), (0, 1), (1, -1), (-2, 0), (-2, 0), (0, 2), (2, -2)))
    ell = np.asarray((1.1, 1.7, 2.2, 2.8, 3.1, 3.6, 4.2, 4.7))
    wavelength = np.linspace(1.53, 1.55, ell.size)
    layers = 7
    epsilon = 0.001
    response = compile_pbi2_stacking_profile_response(
        crystal=crystal,
        crystal_revision=crystal_revision,
        signed_hk=signed_hk,
        l_coordinate=ell,
        wavelength_A=wavelength,
        layers=layers,
    )
    event_id = np.arange(ell.size, dtype=np.int64)
    _, rod_id = np.unique(signed_hk, axis=0, return_inverse=True)
    hkl = np.column_stack((signed_hk, ell))
    layer_normal = np.cross(crystal.direct_basis_A[:, 0], crystal.direct_basis_A[:, 1])
    layer_normal /= np.linalg.norm(layer_normal)
    layer_q = reciprocal.q_cartesian_Ainv(hkl) @ layer_normal
    query = RodQueryBatch(
        event_id=event_id,
        rod_id=rod_id,
        phase_id=(crystal.phase_id,) * ell.size,
        h=signed_hk[:, 0].astype(np.int32),
        k=signed_hk[:, 1].astype(np.int32),
        q_sample_normal_Ainv=layer_q,
        l_coordinate=ell,
        wavelength_A=wavelength,
    )
    amplitudes = pbi2_layer_amplitudes(crystal, query, unknown_u_iso_A2=0.0)
    omega = np.asarray(registry_phase(query.h, query.k, RegistryPhaseModel.FORWARD_H_PLUS_2K))
    vertical_phase = np.exp(1j * layer_q * amplitudes.layer_repeat_A)
    expected = electron_squared_to_scattering_strength_A2(
        np.column_stack(
            tuple(
                np.asarray(
                    [
                        finite_intensity_by_enumeration(
                            layers,
                            amplitudes.f_plus_e[event],
                            amplitudes.f_minus_e[event],
                            omega[event],
                            vertical_phase[event],
                            RichEpsilonModel(parent, epsilon).transition_law(),
                            InitialPopulation.plus_only(),
                        )
                        for event in range(ell.size)
                    ]
                )
                / layers
                for parent in (
                    Parent.TWO_H,
                    Parent.FOUR_H_PLUS,
                    Parent.FOUR_H_MINUS,
                    Parent.SIX_H_PLUS,
                    Parent.SIX_H_MINUS,
                )
            )
        )
    )

    assert response.component_ids == STACKING_COMPONENT_IDS
    np.testing.assert_array_equal(response.signed_hk, signed_hk)
    np.testing.assert_allclose(response.component_response_A2, expected, rtol=5.0e-13, atol=1.0e-18)
    assert not response.component_response_A2.flags.writeable

    planted_domain = np.asarray((0.60, 0.16, 0.09, 0.10, 0.05))
    independent_signal = expected @ (1.0e6 * planted_domain)
    independent_fit = fit_stacking_phase_totals(
        response, independent_signal, np.ones(independent_signal.size)
    )
    np.testing.assert_allclose(
        independent_fit.phase_fraction, (0.60, 0.25, 0.15), rtol=0.0, atol=2.0e-12
    )

    control_hk = np.tile((-2, 1), (17, 1))
    control = compile_pbi2_stacking_profile_response(
        crystal,
        crystal_revision,
        signed_hk=control_hk,
        l_coordinate=np.linspace(0.4, 5.6, control_hk.shape[0]),
        wavelength_A=1.540592925,
        layers=layers,
    )
    control_signal = control.component_response_A2 @ np.asarray((0.6, 0.15, 0.1, 0.1, 0.05))
    with pytest.raises(StackingPopulationIdentifiabilityError, match="cannot separate"):
        fit_stacking_phase_totals(control, control_signal, np.ones(control_signal.size))
