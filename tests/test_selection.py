from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import rasim_next.selection.blind as blind_module
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.fitting import (
    ExactTagGeometryModel,
    IndexedGeometryImage,
    IntegerLMarkerKey,
    IntegerLMarkerPrediction,
)
from rasim_next.geometry import (
    AngleFrame,
    CompiledInstrument,
    angles_to_detector_coordinates,
    detector_coordinates_to_angles,
)
from rasim_next.pipeline.configured_simulation import (
    IntegerLEwaldRoots,
    NominalEwaldContext,
    build_configured_geometry_inputs,
    build_configured_simulation_inputs,
    build_geometry_only_ewald_context,
    build_nominal_ewald_context,
    evaluate_nominal_integer_l_markers,
    load_simulation_config,
    rebind_configured_geometry_instrument,
)
from rasim_next.pipeline.continuous_detector import DetectorEwaldMeasure
from rasim_next.selection import (
    BlindIndexingPolicy,
    DiscoveredCakePeak,
    ExpectedM0Peak,
    FrozenOscGeometryReindexing,
    M0PeakEvidencePolicy,
    MarkerIndexingDecision,
    MarkerIndexingStatus,
    MeasuredImageIndexingResult,
    MeasuredIndexingResult,
    MeasuredPeakDiscovery,
    OscGeometryIndexingRun,
    PeakIndexingPolicy,
    audit_frozen_marker_visibility,
    audit_frozen_osc_geometry_reindexing,
    build_osc_angle_frame,
    discover_measured_cake_peaks,
    evaluate_expected_m0_peak_evidence,
    index_discovered_integer_l_peaks,
    index_measured_integer_l_branches,
    load_osc_geometry_series,
    reindex_frozen_osc_geometry_series,
    select_confident_branch_tracks,
    simulation_config_for_osc_image,
)
from rasim_next.selection.blind import _discovery_geometry_hash, _indexing_context_hash


def _instrument() -> CompiledInstrument:
    identity = np.eye(3)
    zero = np.zeros(3)
    angle = np.deg2rad(11.0)
    detector_rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    return CompiledInstrument(
        lab_from_sample=RigidTransform(identity, zero, FrameId.SAMPLE, FrameId.LAB),
        sample_from_crystal=RigidTransform(identity, zero, FrameId.CRYSTAL, FrameId.SAMPLE),
        lab_from_detector=RigidTransform(
            detector_rotation,
            np.asarray((0.0, 0.0, 0.20)),
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        detector_shape_rc=(180, 190),
        detector_row_pitch_m=1.0e-3,
        detector_column_pitch_m=1.0e-3,
        detector_reference_coordinate_px=(90.0, 90.0),
        sample_support_model_id="unbounded_plane.v1",
        sample_width_m=None,
        sample_length_m=None,
        film_thickness_A=500.0,
    )


def _angle_frame() -> AngleFrame:
    return AngleFrame(
        origin_lab_m=np.zeros(3),
        row_down_lab=np.asarray((0.0, 1.0, 0.0)),
        column_right_lab=np.asarray((1.0, 0.0, 0.0)),
        direct_beam_lab=np.asarray((0.0, 0.0, 1.0)),
        revision="selection-test-angle-frame.v1",
    )


def _key(family_m: int, integer_l: int, root_sign: int) -> IntegerLMarkerKey:
    return IntegerLMarkerKey(family_m, integer_l, 2, root_sign, (family_m, 0))


def _prediction(
    entries: tuple[tuple[IntegerLMarkerKey, tuple[float, float]], ...],
) -> IntegerLMarkerPrediction:
    return IntegerLMarkerPrediction(
        keys=tuple(key for key, _ in entries),
        coordinates_px=np.asarray(tuple(coordinate for _, coordinate in entries)),
        detector_status=np.full(len(entries), "VALID"),
        ewald_residual_Ainv=np.zeros(len(entries)),
    )


def _image(
    centers: tuple[tuple[float, float, float], ...],
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    row, column = np.indices((180, 190), dtype=np.float64)
    counts = 18.0 + rng.normal(0.0, 0.7, (180, 190))
    for center_column, center_row, amplitude in centers:
        counts += amplitude * np.exp(
            -0.5 * (((column - center_column) / 2.2) ** 2 + ((row - center_row) / 5.5) ** 2)
        )
    return np.maximum(counts, 0.0)


def _policy() -> PeakIndexingPolicy:
    return PeakIndexingPolicy(
        search_radius_px=9.0,
        cake_step_px=0.75,
        minimum_candidate_z=4.0,
        minimum_site_z=7.0,
        minimum_assignment_margin=0.35,
        minimum_track_sites=3,
        minimum_track_images=2,
        maximum_track_rms_px=2.5,
    )


def _synthetic_image_result(
    image_id: str,
    incidence_angle_deg: float,
    integer_l_values: tuple[int, ...],
    *,
    geometry_offset_px: float,
    detector_hash_digit: str,
    context_hash_digit: str,
    wavelength_A: float = 1.5406,
) -> MeasuredImageIndexingResult:
    decisions = tuple(
        MarkerIndexingDecision(
            key=_key(1, integer_l, -1),
            predicted_column_px=50.0 + geometry_offset_px + integer_l,
            predicted_row_px=30.0 + 2.0 * integer_l,
            predicted_two_theta_rad=0.01 * integer_l + 1.0e-4 * geometry_offset_px,
            predicted_phi_rad=0.2,
            status=MarkerIndexingStatus.VISIBLE_CONFIDENT,
            reason="synthetic coherent track site",
            observed_column_px=50.25 + geometry_offset_px + integer_l,
            observed_row_px=29.85 + 2.0 * integer_l,
            observed_two_theta_rad=0.01 * integer_l + 1.0e-4 * geometry_offset_px,
            observed_phi_rad=0.2,
            covariance_px2=((0.25, 0.0), (0.0, 0.25)),
            z_score=20.0,
            assignment_cost=0.0,
            assignment_margin=1.0,
        )
        for integer_l in integer_l_values
    )
    return MeasuredImageIndexingResult(
        image_id=image_id,
        incidence_angle_rad=np.deg2rad(incidence_angle_deg),
        reference_wavelength_A=wavelength_A,
        marker_decisions=decisions,
        detector_data_hash="sha256-" + detector_hash_digit * 64,
        detector_mask_hash="sha256-" + "0" * 64,
        detector_mask_revision="synthetic-all-valid.v1",
        context_hash="sha256-" + context_hash_digit * 64,
        policy=_policy(),
    )


def test_selection_policies_preserve_mandatory_evidence_and_alias_gates() -> None:
    for field_name, weakened_value in (
        ("minimum_track_sites", 2),
        ("minimum_common_l_sites", 1),
        ("minimum_track_images", 1),
    ):
        with pytest.raises(ValueError, match=field_name):
            replace(_policy(), **{field_name: weakened_value})

    blind_policy = BlindIndexingPolicy(track_policy=_policy())
    for field_name, weakened_value in (
        ("maximum_anchor_distance_px", np.nextafter(32.0, np.inf)),
        ("separation_fraction", np.nextafter(0.25, np.inf)),
    ):
        with pytest.raises(ValueError, match=field_name):
            replace(blind_policy, **{field_name: weakened_value})


def test_provenance_identifiers_require_canonical_sha256() -> None:
    valid_hash = "sha256-" + "0" * 64
    with pytest.raises(ValueError, match="detector_data_hash"):
        MeasuredImageIndexingResult(
            image_id="malformed-image-hash",
            incidence_angle_rad=0.1,
            reference_wavelength_A=1.5406,
            marker_decisions=(),
            detector_data_hash="sha256-a",
            detector_mask_hash=valid_hash,
            detector_mask_revision="synthetic-all-valid.v1",
            context_hash=valid_hash,
            policy=_policy(),
        )
    with pytest.raises(ValueError, match="geometry_context_hash"):
        MeasuredPeakDiscovery(
            image_id="malformed-discovery-hash",
            detector_shape_rc=(1, 1),
            peaks=(),
            detector_data_hash=valid_hash,
            detector_mask_hash=valid_hash,
            detector_mask_revision="synthetic-all-valid.v1",
            geometry_context_hash="sha256-" + "G" * 64,
            policy=BlindIndexingPolicy(track_policy=_policy()),
        )


def test_overlapping_replication_chains_retain_all_qualified_images_and_sites() -> None:
    first = _synthetic_image_result(
        "A", 5.0, (2, 4, 6), geometry_offset_px=0.0, detector_hash_digit="1", context_hash_digit="a"
    )
    second = _synthetic_image_result(
        "B",
        10.0,
        (2, 4, 6, 8),
        geometry_offset_px=1.0,
        detector_hash_digit="2",
        context_hash_digit="b",
    )
    third = _synthetic_image_result(
        "C",
        15.0,
        (6, 8, 10),
        geometry_offset_px=2.0,
        detector_hash_digit="3",
        context_hash_digit="c",
    )

    manifest = select_confident_branch_tracks((first, second, third), policy=_policy())
    track = next(item for item in manifest.branch_tracks if item.accepted)

    assert track.image_ids == ("A", "B", "C")
    assert {key.integer_L for key in manifest.observations_for("B").keys} == {2, 4, 6, 8}
    assert {key.integer_L for key in manifest.observations_for("C").keys} == {6, 8}


def test_frozen_visibility_audit_ignores_new_candidate_track_censoring() -> None:
    original_results = tuple(
        _synthetic_image_result(
            image_id,
            incidence,
            (2, 4, 6),
            geometry_offset_px=offset,
            detector_hash_digit=detector_digit,
            context_hash_digit=context_digit,
        )
        for image_id, incidence, offset, detector_digit, context_digit in (
            ("A", 5.0, 0.0, "1", "a"),
            ("B", 10.0, 1.0, "2", "b"),
            ("C", 15.0, 2.0, "3", "c"),
        )
    )
    frozen = select_confident_branch_tracks(original_results, policy=_policy())
    second = original_results[1]
    extra = MarkerIndexingDecision(
        key=_key(1, 8, -1),
        predicted_column_px=59.0,
        predicted_row_px=46.0,
        predicted_two_theta_rad=0.081,
        predicted_phi_rad=0.2,
        status=MarkerIndexingStatus.VISIBLE_CONFIDENT,
        reason="new post-fit candidate",
        observed_column_px=79.0,
        observed_row_px=46.0,
        observed_two_theta_rad=0.081,
        observed_phi_rad=0.2,
        covariance_px2=((0.25, 0.0), (0.0, 0.25)),
        z_score=20.0,
        assignment_cost=18.0,
        assignment_margin=1.0,
    )
    corrected_second = MeasuredImageIndexingResult(
        image_id=second.image_id,
        incidence_angle_rad=second.incidence_angle_rad,
        reference_wavelength_A=second.reference_wavelength_A,
        marker_decisions=(*second.marker_decisions, extra),
        detector_data_hash=second.detector_data_hash,
        detector_mask_hash=second.detector_mask_hash,
        detector_mask_revision=second.detector_mask_revision,
        context_hash="sha256-" + "d" * 64,
        policy=second.policy,
    )
    reindexed = select_confident_branch_tracks(
        (original_results[0], corrected_second, original_results[2]),
        policy=_policy(),
    )
    assert not next(track for track in corrected_second.branch_tracks).accepted
    with pytest.raises(ValueError, match="no accepted visible"):
        reindexed.observations_for("B")

    audit = audit_frozen_marker_visibility(frozen, reindexed)
    assert audit.classification == "SAME"
    second_audit = next(item for item in audit.images if item.image_id == "B")
    assert second_audit.newly_visible_keys == (_key(1, 8, -1),)
    assert second_audit.frozen_subset_track_coherent

    missing_second = replace(
        corrected_second,
        marker_decisions=tuple(
            item for item in corrected_second.marker_decisions if item.key != _key(1, 4, -1)
        ),
    )
    changed = select_confident_branch_tracks(
        (original_results[0], missing_second, original_results[2]),
        policy=_policy(),
    )
    changed_audit = audit_frozen_marker_visibility(frozen, changed)
    assert changed_audit.classification == "CHANGED"
    assert next(item for item in changed_audit.images if item.image_id == "B").missing_keys == (
        _key(1, 4, -1),
    )
    with pytest.raises(ValueError, match="incomparable measured provenance"):
        audit_frozen_marker_visibility(
            frozen,
            select_confident_branch_tracks(
                (
                    original_results[0],
                    replace(
                        corrected_second,
                        detector_data_hash="sha256-" + "9" * 64,
                    ),
                    original_results[2],
                ),
                policy=_policy(),
            ),
        )


def test_osc_indexing_run_rejects_mismatched_series_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    series = load_osc_geometry_series(root / "configs" / "bi2se3_osc_geometry_fit.yaml")
    base = load_simulation_config(series.config_path)
    inputs_by_image = []
    contexts = []
    models = []
    results = []
    discoveries = []
    shared_inputs = None
    for index, image in enumerate(series.images):
        config = simulation_config_for_osc_image(base, image)
        inputs = (
            build_configured_geometry_inputs(config)
            if shared_inputs is None
            else rebind_configured_geometry_instrument(shared_inputs, config)
        )
        if shared_inputs is None:
            shared_inputs = inputs
        context = build_geometry_only_ewald_context(inputs)
        frame = build_osc_angle_frame(
            mean_direction_lab=inputs.config.source.mean_direction_lab,
            instrument=context.instrument,
            sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
            revision=f"osc-geometry-angle-frame.{image.image_id}.v1",
        )
        result = _synthetic_image_result(
            image.image_id,
            image.axis_rotation_angles_deg[0],
            (2, 4, 6),
            geometry_offset_px=float(index),
            detector_hash_digit=str(index + 1),
            context_hash_digit="abcdef"[index],
            wavelength_A=1.540592925,
        )
        discovery = MeasuredPeakDiscovery(
            image_id=result.image_id,
            detector_shape_rc=context.instrument.detector_shape_rc,
            peaks=tuple(
                DiscoveredCakePeak(
                    column_px=float(decision.observed_column_px),
                    row_px=float(decision.observed_row_px),
                    two_theta_rad=float(decision.observed_two_theta_rad),
                    phi_rad=float(decision.observed_phi_rad),
                    covariance_px2=decision.covariance_px2,
                    localization_covariance_px2=decision.covariance_px2,
                    z_score=float(decision.z_score),
                )
                for decision in result.marker_decisions
            ),
            detector_data_hash=result.detector_data_hash,
            detector_mask_hash=result.detector_mask_hash,
            detector_mask_revision=result.detector_mask_revision,
            geometry_context_hash=_discovery_geometry_hash(context.instrument, frame),
            policy=BlindIndexingPolicy(track_policy=_policy()),
        )
        results.append(
            replace(
                result,
                context_hash=_indexing_context_hash(discovery, context, frame),
            )
        )
        discoveries.append(discovery)
        inputs_by_image.append(inputs)
        contexts.append(context)
        models.append(ExactTagGeometryModel(inputs))
    results = tuple(results)
    discoveries = tuple(discoveries)
    selection = select_confident_branch_tracks(tuple(reversed(results)), policy=_policy())
    indexed_images = tuple(
        IndexedGeometryImage(
            image_id=image.image_id,
            commanded_angle_rad=math.radians(image.axis_rotation_angles_deg[0]),
            model=model,
            observations=selection.observations_for(image.image_id),
        )
        for image, model in zip(series.images, models, strict=True)
    )
    run = OscGeometryIndexingRun(
        series=series,
        selection=selection,
        discoveries=discoveries,
        geometry_inputs=tuple(inputs_by_image),
        geometry_contexts=tuple(contexts),
        indexed_images=indexed_images,
        geometry_setup_seconds=0.0,
        elapsed_seconds=0.0,
    )
    instruments = {
        image.image_id: model.instrument for image, model in zip(series.images, models, strict=True)
    }
    expected_reindexing = reindex_frozen_osc_geometry_series(
        run,
        instrument_by_image_id=instruments,
    )
    first_reindexed = expected_reindexing.selection.image_results[0]
    fabricated_selection = MeasuredIndexingResult(
        image_results=(
            replace(
                first_reindexed,
                context_hash="sha256-ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            ),
            *expected_reindexing.selection.image_results[1:],
        ),
        policy=expected_reindexing.selection.policy,
    )
    fabricated_reindexing = FrozenOscGeometryReindexing(
        selection=fabricated_selection,
        source_manifest_hash=expected_reindexing.source_manifest_hash,
        source_discovery_hashes=expected_reindexing.source_discovery_hashes,
    )
    with pytest.raises(ValueError, match="not the exact source-coordinate relabeling"):
        audit_frozen_osc_geometry_reindexing(
            run,
            fabricated_reindexing,
            instrument_by_image_id=instruments,
        )
    with pytest.raises(ValueError, match="selection image IDs"):
        OscGeometryIndexingRun(
            series=series,
            selection=select_confident_branch_tracks(results[:2], policy=_policy()),
            discoveries=discoveries,
            geometry_inputs=tuple(inputs_by_image),
            geometry_contexts=tuple(contexts),
            indexed_images=indexed_images,
            geometry_setup_seconds=0.0,
            elapsed_seconds=0.0,
        )
    changed_angle = replace(results[2], incidence_angle_rad=np.deg2rad(12.0))
    with pytest.raises(ValueError, match="selection incidence"):
        OscGeometryIndexingRun(
            series=series,
            selection=select_confident_branch_tracks(
                (*results[:2], changed_angle), policy=_policy()
            ),
            discoveries=discoveries,
            geometry_inputs=tuple(inputs_by_image),
            geometry_contexts=tuple(contexts),
            indexed_images=indexed_images,
            geometry_setup_seconds=0.0,
            elapsed_seconds=0.0,
        )

    detector = models[0].instrument.lab_from_detector
    angle = math.radians(3.0)
    tilt = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(angle), -math.sin(angle)),
            (0.0, math.sin(angle), math.cos(angle)),
        )
    )
    wrong_inputs = replace(
        inputs_by_image[0],
        instrument=replace(
            models[0].instrument,
            lab_from_detector=RigidTransform(
                detector.rotation @ tilt,
                detector.translation_m,
                FrameId.DETECTOR,
                FrameId.LAB,
            ),
        ),
    )
    with pytest.raises(ValueError, match="instrument does not match its declared config"):
        replace(
            indexed_images[0],
            model=ExactTagGeometryModel(wrong_inputs),
        )


def test_changed_wavelength_does_not_make_repeated_detector_geometry_distinct() -> None:
    first = _synthetic_image_result(
        "A", 5.0, (2, 4, 6), geometry_offset_px=0.0, detector_hash_digit="1", context_hash_digit="a"
    )
    repeated = replace(
        first,
        image_id="B",
        incidence_angle_rad=np.deg2rad(10.0),
        reference_wavelength_A=1.6,
        detector_data_hash="sha256-" + "2" * 64,
        context_hash="sha256-" + "b" * 64,
    )

    manifest = select_confident_branch_tracks((first, repeated), policy=_policy())

    assert not any(track.accepted for track in manifest.branch_tracks)


def test_measured_indexing_exports_only_visible_replicated_branch_tracks() -> None:
    entries = (
        *tuple(
            (_key(1, ell, -1), (52.0, row))
            for ell, row in zip((2, 4, 6, 8), (38, 70, 102, 134), strict=True)
        ),
        *tuple(
            (_key(1, ell, 1), (128.0, row))
            for ell, row in zip((2, 4, 6, 8), (38, 70, 102, 134), strict=True)
        ),
        (_key(3, 4, -1), (78.0, 55.0)),
        (_key(3, 7, -1), (78.0, 117.0)),
    )
    prediction = _prediction(entries)
    second_prediction = _prediction(
        tuple((key, (column + 0.4, row - 0.3)) for key, (column, row) in entries)
    )
    third_prediction = _prediction(
        tuple((key, (column + 0.8, row - 0.6)) for key, (column, row) in entries)
    )
    strong = tuple(
        (column + 1.8, row - 1.2, 95.0)
        for key, (column, row) in entries
        if key.family_m == 1 and not (key.root_sign == 1 and key.integer_L == 4)
    )
    second_strong = tuple(
        (column + 1.8, row - 1.2, 95.0)
        for key, (column, row) in entries
        if key.family_m == 1
        and not (key.root_sign == 1 and key.integer_L == 4)
        and not (key.root_sign == -1 and key.integer_L == 8)
    )
    sparse_track = ((79.5, 53.8, 100.0), (79.5, 115.8, 100.0))
    instrument = _instrument()
    frame = _angle_frame()
    second_frame = replace(frame, revision="selection-test-angle-frame-10deg.v1")
    third_frame = replace(frame, revision="selection-test-angle-frame-15deg.v1")
    first = index_measured_integer_l_branches(
        _image((*strong, *sparse_track), seed=7),
        prediction,
        instrument=instrument,
        angle_frame=frame,
        image_id="synthetic-5deg",
        incidence_angle_rad=np.deg2rad(5.0),
        reference_wavelength_A=1.5406,
        policy=_policy(),
    )
    second = index_measured_integer_l_branches(
        _image((*second_strong, *sparse_track), seed=11),
        second_prediction,
        instrument=instrument,
        angle_frame=second_frame,
        image_id="synthetic-10deg",
        incidence_angle_rad=np.deg2rad(10.0),
        reference_wavelength_A=1.5406,
        policy=_policy(),
    )
    third_strong = tuple(
        (column + 1.8, row - 1.2, 95.0)
        for key, (column, row) in entries
        if key.family_m == 1
        and (
            (key.root_sign == 1 and key.integer_L != 4)
            or (key.root_sign == -1 and key.integer_L in {2, 6})
        )
    )
    third = index_measured_integer_l_branches(
        _image(third_strong, seed=13),
        third_prediction,
        instrument=instrument,
        angle_frame=third_frame,
        image_id="synthetic-15deg",
        incidence_angle_rad=np.deg2rad(15.0),
        reference_wavelength_A=1.5406,
        policy=_policy(),
    )
    manifest = select_confident_branch_tracks((first, second, third), policy=_policy())

    accepted_tracks = tuple(track for track in manifest.branch_tracks if track.accepted)
    assert {(track.family_m, track.root_sign) for track in accepted_tracks} == {(1, -1), (1, 1)}
    assert all(track.family_m != 3 or not track.accepted for track in manifest.branch_tracks)
    weak = next(decision for decision in first.marker_decisions if decision.key == _key(1, 4, 1))
    assert weak.status == MarkerIndexingStatus.BELOW_DETECTION
    assert "extinct" not in weak.reason.casefold()

    observations = manifest.observations_for("synthetic-5deg")
    assert len(observations.keys) == 6
    assert all(key.family_m == 1 for key in observations.keys)
    assert _key(1, 4, 1) not in observations.keys
    assert _key(1, 8, -1) not in observations.keys
    covariance_eigenvalues = np.linalg.eigvalsh(observations.covariance_px2)
    assert np.all(covariance_eigenvalues > 0.0)
    assert np.max(covariance_eigenvalues[:, 1] / covariance_eigenvalues[:, 0]) > 3.0
    assert not observations.coordinates_px.flags.writeable
    assert manifest.manifest_hash.startswith("sha256-")
    probe = observations.coordinates_px[0]
    probe_angles = detector_coordinates_to_angles(
        np.asarray((probe[0],)),
        np.asarray((probe[1],)),
        instrument=instrument,
        angle_frame=frame,
    )
    roundtrip = angles_to_detector_coordinates(
        probe_angles.two_theta_rad,
        probe_angles.phi_rad,
        instrument=instrument,
        angle_frame=frame,
    )
    np.testing.assert_allclose(
        (roundtrip.column_px[0], roundtrip.row_px[0]),
        probe,
        rtol=0.0,
        atol=1.0e-10,
    )

    third_observations = manifest.observations_for("synthetic-15deg")
    assert len(third_observations.keys) == 3
    assert all(key.root_sign == 1 for key in third_observations.keys)

    duplicate_angle = replace(second, incidence_angle_rad=first.incidence_angle_rad)
    duplicate_manifest = select_confident_branch_tracks((first, duplicate_angle), policy=_policy())
    assert not any(track.accepted for track in duplicate_manifest.branch_tracks)
    spoofed_duplicate = replace(
        first,
        image_id="same-detector-data-spoofed-incidence",
        incidence_angle_rad=np.deg2rad(10.0),
    )
    spoofed_manifest = select_confident_branch_tracks(
        (first, spoofed_duplicate),
        policy=_policy(),
    )
    assert not any(track.accepted for track in spoofed_manifest.branch_tracks)

    disjoint_entries = tuple(
        (_key(1, ell, -1), (92.0, row))
        for ell, row in zip(
            (2, 4, 6, 8, 10, 12),
            (26, 50, 74, 98, 122, 146),
            strict=True,
        )
    )
    disjoint_prediction = _prediction(disjoint_entries)
    disjoint_policy = _policy()
    early = index_measured_integer_l_branches(
        _image(
            ((93.5, 25.0, 100.0), (93.5, 49.0, 100.0), (93.5, 73.0, 100.0)),
            seed=23,
        ),
        disjoint_prediction,
        instrument=instrument,
        angle_frame=replace(frame, revision="selection-test-disjoint-5deg.v1"),
        image_id="early-L",
        incidence_angle_rad=np.deg2rad(5.0),
        reference_wavelength_A=1.5406,
        policy=disjoint_policy,
    )
    late = index_measured_integer_l_branches(
        _image(
            ((93.5, 97.0, 100.0), (93.5, 121.0, 100.0), (93.5, 145.0, 100.0)),
            seed=29,
        ),
        disjoint_prediction,
        instrument=instrument,
        angle_frame=replace(frame, revision="selection-test-disjoint-10deg.v1"),
        image_id="late-L",
        incidence_angle_rad=np.deg2rad(10.0),
        reference_wavelength_A=1.5406,
        policy=disjoint_policy,
    )
    disjoint_manifest = select_confident_branch_tracks(
        (early, late),
        policy=disjoint_policy,
    )
    assert not any(track.accepted for track in disjoint_manifest.branch_tracks)


def test_indexing_rejects_ambiguous_ownership_and_hashes_canonical_decisions() -> None:
    entries = (
        (_key(1, 2, -1), (70.0, 42.0)),
        (_key(1, 3, -1), (70.0, 74.0)),
        (_key(1, 4, -1), (70.0, 106.0)),
        (_key(3, 2, -1), (74.0, 42.0)),
        (_key(3, 3, -1), (74.0, 74.0)),
        (_key(3, 4, -1), (74.0, 106.0)),
    )
    centers = ((72.0, 42.0, 100.0), (72.0, 74.0, 100.0), (72.0, 106.0, 100.0))
    prediction = _prediction(entries)
    instrument = _instrument()
    frame = _angle_frame()
    image = _image(centers, seed=19)
    result = index_measured_integer_l_branches(
        image,
        prediction,
        instrument=instrument,
        angle_frame=frame,
        image_id="ambiguous",
        incidence_angle_rad=np.deg2rad(5.0),
        reference_wavelength_A=1.5406,
        policy=_policy(),
    )
    permuted = index_measured_integer_l_branches(
        image,
        _prediction(tuple(reversed(entries))),
        instrument=instrument,
        angle_frame=frame,
        image_id="ambiguous",
        incidence_angle_rad=np.deg2rad(5.0),
        reference_wavelength_A=1.5406,
        policy=_policy(),
    )

    assert all(
        decision.status == MarkerIndexingStatus.AMBIGUOUS_BLEND
        for decision in result.marker_decisions
    )
    assert result.result_hash == permuted.result_hash
    changed_mask = np.ones(image.shape, dtype=np.bool_)
    changed_mask[0, 0] = False
    remasked = index_measured_integer_l_branches(
        image,
        prediction,
        instrument=instrument,
        angle_frame=frame,
        image_id="ambiguous",
        incidence_angle_rad=np.deg2rad(5.0),
        reference_wavelength_A=1.5406,
        detector_valid_mask=changed_mask,
        policy=_policy(),
    )
    assert remasked.detector_mask_hash != result.detector_mask_hash
    assert remasked.result_hash != result.result_hash
    changed_residual = replace(
        prediction,
        ewald_residual_Ainv=np.ones(len(prediction.keys)),
    )
    recontextualized = index_measured_integer_l_branches(
        image,
        changed_residual,
        instrument=instrument,
        angle_frame=frame,
        image_id="ambiguous",
        incidence_angle_rad=np.deg2rad(5.0),
        reference_wavelength_A=1.5406,
        policy=_policy(),
    )
    assert recontextualized.context_hash != result.context_hash
    assert recontextualized.result_hash != result.result_hash
    changed = replace(result.marker_decisions[0], predicted_column_px=69.5)
    assert (
        replace(result, marker_decisions=(changed, *result.marker_decisions[1:])).result_hash
        != result.result_hash
    )


def test_global_cake_discovery_finds_peaks_without_marker_position_input() -> None:
    centers = (
        (48.0, 42.0, 110.0),
        (72.0, 66.0, 95.0),
        (119.0, 49.0, 105.0),
        (139.0, 111.0, 100.0),
        (91.0, 139.0, 90.0),
    )
    policy = BlindIndexingPolicy(
        track_policy=replace(
            _policy(),
            cake_step_px=1.5,
            minimum_candidate_z=3.5,
            minimum_site_z=6.0,
        ),
        maximum_candidate_count=64,
        minimum_scattering_radius_px=8.0,
    )
    image = _image(centers, seed=37)
    discovery = discover_measured_cake_peaks(
        image,
        instrument=_instrument(),
        angle_frame=_angle_frame(),
        image_id="position-free-synthetic",
        policy=policy,
    )

    observed = np.asarray(
        tuple((peak.column_px, peak.row_px) for peak in discovery.peaks),
        dtype=np.float64,
    )
    assert observed.shape[0] >= len(centers)
    for expected_column, expected_row, _ in centers:
        distance = np.linalg.norm(observed - (expected_column, expected_row), axis=1)
        assert float(np.min(distance)) < 2.5
    assert discovery.discovery_hash.startswith("sha256-")
    for scale in (1.0e-9, 1.0e38, 1.0e-40):
        scaled = discover_measured_cake_peaks(
            image * scale,
            instrument=_instrument(),
            angle_frame=_angle_frame(),
            image_id="position-free-synthetic",
            policy=policy,
        )
        np.testing.assert_allclose(
            tuple((peak.column_px, peak.row_px) for peak in scaled.peaks),
            tuple((peak.column_px, peak.row_px) for peak in discovery.peaks),
            rtol=0.0,
            atol=1.0e-10,
        )


def test_q_space_label_algebra_matches_exact_bi2se3_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    inputs = build_configured_simulation_inputs(
        replace(config, source=replace(config.source, sample_count=1))
    )
    context = build_nominal_ewald_context(inputs)
    markers = evaluate_nominal_integer_l_markers(context)
    selection = np.flatnonzero(
        (markers.family_m == 1)
        & (markers.branch == 2)
        & np.isin(markers.integer_L, (2, 4, 6, 8))
        & np.isin(markers.root_sign, (-1, 1))
    )
    frame = _configured_angle_frame(inputs)
    angles = detector_coordinates_to_angles(
        markers.column_px[selection],
        markers.row_px[selection],
        instrument=inputs.instrument,
        angle_frame=frame,
    )
    peaks = tuple(
        DiscoveredCakePeak(
            column_px=float(markers.column_px[index]),
            row_px=float(markers.row_px[index]),
            two_theta_rad=float(theta),
            phi_rad=float(phi),
            covariance_px2=((0.25, 0.0), (0.0, 0.25)),
            localization_covariance_px2=((0.25, 0.0), (0.0, 0.25)),
            z_score=30.0,
        )
        for index, theta, phi in zip(
            selection,
            angles.two_theta_rad,
            angles.phi_rad,
            strict=True,
        )
    )
    policy = BlindIndexingPolicy(track_policy=_policy())
    discovery = MeasuredPeakDiscovery(
        image_id="bi2se3-exact-coordinate-probe",
        detector_shape_rc=inputs.instrument.detector_shape_rc,
        peaks=peaks,
        detector_data_hash="sha256-" + "1" * 64,
        detector_mask_hash="sha256-" + "2" * 64,
        detector_mask_revision="all-valid-test-mask.v1",
        geometry_context_hash=_discovery_geometry_hash(inputs.instrument, frame),
        policy=policy,
    )

    indexed = index_discovered_integer_l_peaks(
        discovery,
        ewald_context=context,
        angle_frame=frame,
        incidence_angle_rad=np.deg2rad(5.0),
    )
    inconsistent = replace(
        discovery,
        peaks=(
            replace(discovery.peaks[0], phi_rad=discovery.peaks[0].phi_rad + 0.5),
            *discovery.peaks[1:],
        ),
    )
    with pytest.raises(ValueError, match="detector and angle coordinates disagree"):
        index_discovered_integer_l_peaks(
            inconsistent,
            ewald_context=context,
            angle_frame=frame,
            incidence_angle_rad=np.deg2rad(5.0),
        )
    expected = {
        (
            int(markers.family_m[index]),
            int(markers.integer_L[index]),
            int(markers.branch[index]),
            int(markers.root_sign[index]),
        )
        for index in selection
    }
    actual = {
        (
            decision.key.family_m,
            decision.key.integer_L,
            decision.key.branch,
            decision.key.root_sign,
        )
        for decision in indexed.marker_decisions
    }
    assert actual == expected
    changed_discovery_policy = replace(
        discovery,
        policy=replace(
            policy,
            maximum_anchor_distance_px=policy.maximum_anchor_distance_px - 1.0,
        ),
    )
    changed_policy_result = index_discovered_integer_l_peaks(
        changed_discovery_policy,
        ewald_context=context,
        angle_frame=frame,
        incidence_angle_rad=np.deg2rad(5.0),
    )
    assert changed_policy_result.marker_decisions == indexed.marker_decisions
    assert changed_discovery_policy.discovery_hash != discovery.discovery_hash
    assert changed_policy_result.context_hash != indexed.context_hash
    assert changed_policy_result.result_hash != indexed.result_hash
    np.testing.assert_allclose(
        tuple(
            (decision.predicted_column_px, decision.predicted_row_px)
            for decision in indexed.marker_decisions
        ),
        tuple(
            (
                markers.column_px[index],
                markers.row_px[index],
            )
            for index in sorted(
                selection,
                key=lambda item: (
                    int(markers.family_m[item]),
                    int(markers.integer_L[item]),
                    int(markers.branch[item]),
                    int(markers.root_sign[item]),
                    markers.contributing_rod_hk[item][0],
                ),
            )
        ),
        rtol=0.0,
        atol=1.0e-8,
    )
    with pytest.raises(ValueError, match="discovery geometry does not match"):
        index_discovered_integer_l_peaks(
            discovery,
            ewald_context=context,
            angle_frame=replace(frame, revision="wrong-angle-frame.v1"),
            incidence_angle_rad=np.deg2rad(5.0),
        )

    def indexed_with_active_authorities() -> tuple[object, ...]:
        return index_discovered_integer_l_peaks(
            discovery,
            ewald_context=context,
            angle_frame=frame,
            incidence_angle_rad=np.deg2rad(5.0),
        ).marker_decisions

    with monkeypatch.context() as patch:
        patch.setattr(
            blind_module,
            "solve_integer_l_ewald_roots",
            lambda **_: IntegerLEwaldRoots(beta_rad=(0.0,), root_sign=(0,), branch=2),
        )
        assert indexed_with_active_authorities() == ()

    authoritative_solver = blind_module.solve_integer_l_ewald_roots
    first_m1_rod = next(
        rod for rod in context.geometry.coating.bragg_space.config.rods if rod.family_m == 1
    )
    first_m1_key = (first_m1_rod.h, first_m1_rod.k)

    def conflicting_solver(**arguments: object) -> IntegerLEwaldRoots | None:
        roots = authoritative_solver(**arguments)
        rod = arguments["rod"]
        if roots is None or (rod.h, rod.k) != first_m1_key:
            return roots
        return replace(roots, branch=1 if roots.branch == 2 else 2)

    with monkeypatch.context() as patch:
        patch.setattr(blind_module, "solve_integer_l_ewald_roots", conflicting_solver)
        assert indexed_with_active_authorities() == ()

    authoritative_mapper = DetectorEwaldMeasure.map_latent_geometry

    def noncoincident_mapper(
        self: DetectorEwaldMeasure,
        **arguments: object,
    ) -> object:
        mapped = authoritative_mapper(self, **arguments)
        rod = arguments["rod"]
        if (rod.h, rod.k) != first_m1_key:
            return mapped
        return replace(mapped, column_px=np.asarray(mapped.column_px) + 1.0)

    with monkeypatch.context() as patch:
        patch.setattr(DetectorEwaldMeasure, "map_latent_geometry", noncoincident_mapper)
        assert indexed_with_active_authorities() == ()


def test_withheld_raster_peaks_are_discovered_and_q_indexed_with_coarse_geometry() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
    inputs = build_configured_simulation_inputs(
        replace(config, source=replace(config.source, sample_count=1))
    )
    nominal = build_nominal_ewald_context(inputs)
    shape_rc = (300, 850)
    column_offset = 1000
    row_offset = 1100
    reference_column, reference_row = inputs.instrument.detector_reference_coordinate_px
    true_instrument = replace(
        inputs.instrument,
        detector_shape_rc=shape_rc,
        detector_reference_coordinate_px=(
            reference_column - column_offset,
            reference_row - row_offset,
        ),
    )
    true_context = NominalEwaldContext(
        geometry=DetectorEwaldMeasure(
            coating=nominal.geometry.coating,
            incident=nominal.incident,
            material=inputs.material,
            instrument=true_instrument,
        ),
        incident=nominal.incident,
    )
    truth = evaluate_nominal_integer_l_markers(true_context)
    selected = np.flatnonzero(
        (truth.family_m == 1)
        & (truth.branch == 2)
        & np.isin(truth.integer_L, (5, 7, 9))
        & np.isin(truth.root_sign, (-1, 1))
    )
    assert selected.size == 6

    rng = np.random.default_rng(20260721)
    row, column = np.indices(shape_rc, dtype=np.float64)
    background = 15.0 + 0.004 * row + 0.002 * column + rng.normal(0.0, 0.55, shape_rc)
    measured_shift = np.asarray((1.2, -0.8))
    image = background.copy()
    expected_coordinate_by_key: dict[tuple[int, int, int, int], np.ndarray] = {}
    for marker_index in selected:
        center = (
            np.asarray((truth.column_px[marker_index], truth.row_px[marker_index])) + measured_shift
        )
        image += 90.0 * np.exp(
            -0.5 * (((column - center[0]) / 2.2) ** 2 + ((row - center[1]) / 4.8) ** 2)
        )
        expected_coordinate_by_key[
            (
                int(truth.family_m[marker_index]),
                int(truth.integer_L[marker_index]),
                int(truth.branch[marker_index]),
                int(truth.root_sign[marker_index]),
            )
        ] = center
    image = np.maximum(image, 0.0)

    coarse_instrument = replace(
        true_instrument,
        detector_reference_coordinate_px=(
            true_instrument.detector_reference_coordinate_px[0] + 3.0,
            true_instrument.detector_reference_coordinate_px[1] - 2.0,
        ),
    )
    coarse_context = NominalEwaldContext(
        geometry=DetectorEwaldMeasure(
            coating=nominal.geometry.coating,
            incident=nominal.incident,
            material=inputs.material,
            instrument=coarse_instrument,
        ),
        incident=nominal.incident,
    )
    frame = _configured_angle_frame(inputs)
    policy = BlindIndexingPolicy(
        track_policy=replace(
            _policy(),
            cake_step_px=2.0,
            minimum_candidate_z=3.5,
            minimum_site_z=6.0,
        ),
        maximum_candidate_count=64,
        minimum_scattering_radius_px=8.0,
    )
    discovery = discover_measured_cake_peaks(
        image,
        instrument=coarse_instrument,
        angle_frame=frame,
        image_id="withheld-raster-centers",
        policy=policy,
    )
    indexed = index_discovered_integer_l_peaks(
        discovery,
        ewald_context=coarse_context,
        angle_frame=frame,
        incidence_angle_rad=np.deg2rad(5.0),
    )
    decision_by_key = {
        (
            decision.key.family_m,
            decision.key.integer_L,
            decision.key.branch,
            decision.key.root_sign,
        ): decision
        for decision in indexed.marker_decisions
    }
    assert decision_by_key.keys() == expected_coordinate_by_key.keys()
    truth_coordinate_by_key = {
        (
            int(truth.family_m[index]),
            int(truth.integer_L[index]),
            int(truth.branch[index]),
            int(truth.root_sign[index]),
        ): np.asarray((truth.column_px[index], truth.row_px[index]))
        for index in selected
    }
    for key, decision in decision_by_key.items():
        observed = np.asarray((decision.observed_column_px, decision.observed_row_px))
        predicted = np.asarray((decision.predicted_column_px, decision.predicted_row_px))
        assert np.linalg.norm(observed - expected_coordinate_by_key[key]) < 0.5
        assert 3.0 < np.linalg.norm(predicted - truth_coordinate_by_key[key]) < 4.0
        assert np.linalg.norm(predicted - observed) > 1.5

    tight_discovery = replace(
        discovery,
        policy=replace(policy, maximum_anchor_distance_px=1.0),
    )
    tight_indexed = index_discovered_integer_l_peaks(
        tight_discovery,
        ewald_context=coarse_context,
        angle_frame=frame,
        incidence_angle_rad=np.deg2rad(5.0),
    )
    assert tight_indexed.marker_decisions == ()

    valid_mask = np.ones(shape_rc, dtype=np.bool_)
    valid_mask[:20] = False
    valid_mask[-20:] = False
    valid_mask[:, :20] = False
    valid_mask[:, -20:] = False
    noise_only = background.copy()
    noise_only[3:16, 120:200] += 300.0
    noise_only[285:298, 400:520] += 400.0
    noise_discovery = discover_measured_cake_peaks(
        np.maximum(noise_only, 0.0),
        instrument=coarse_instrument,
        angle_frame=frame,
        image_id="noise-and-masked-border",
        detector_valid_mask=valid_mask,
        detector_mask_revision="masked-border-test.v1",
        policy=policy,
    )
    noise_indexed = index_discovered_integer_l_peaks(
        noise_discovery,
        ewald_context=coarse_context,
        angle_frame=frame,
        incidence_angle_rad=np.deg2rad(5.0),
    )
    assert noise_indexed.marker_decisions == ()


def _configured_angle_frame(inputs: object) -> AngleFrame:
    direct_beam = np.asarray(inputs.config.source.mean_direction_lab, dtype=np.float64)
    direct_beam /= np.linalg.norm(direct_beam)
    detector_column_lab = inputs.instrument.lab_from_detector.apply_vector(
        np.asarray((1.0, 0.0, 0.0))
    )
    column_right = detector_column_lab - float(detector_column_lab @ direct_beam) * direct_beam
    column_right /= np.linalg.norm(column_right)
    return AngleFrame(
        origin_lab_m=inputs.instrument.lab_from_sample.translation_m,
        row_down_lab=np.cross(direct_beam, column_right),
        column_right_lab=column_right,
        direct_beam_lab=direct_beam,
        revision="configured-selection-test-angle-frame.v1",
    )


def test_expected_m0_peak_evidence_requires_raw_local_significance() -> None:
    row, column = np.indices((81, 81))
    background = 20.0 + ((3 * row + 5 * column) % 7)
    counts = np.asarray(background, dtype=np.float64)
    accepted_center = (20.25, 30.5)
    accepted_radius = np.hypot(column - accepted_center[0], row - accepted_center[1]) <= 2.0
    counts[accepted_radius] += 80.0
    rejected_center = (55.0, 55.0)
    rejected_radius = np.hypot(column - rejected_center[0], row - rejected_center[1]) <= 2.0
    counts[rejected_radius] += 2.0
    candidates = (
        ExpectedM0Peak(integer_L=3, column_px=accepted_center[0], row_px=accepted_center[1]),
        ExpectedM0Peak(integer_L=6, column_px=rejected_center[0], row_px=rejected_center[1]),
        ExpectedM0Peak(integer_L=9, column_px=-20.0, row_px=40.0),
        ExpectedM0Peak(integer_L=12, column_px=0.0, row_px=40.0),
    )
    policy = M0PeakEvidencePolicy(
        core_radius_px=2.0,
        background_inner_radius_px=4.0,
        background_outer_radius_px=8.0,
        minimum_peak_z=5.0,
        minimum_integrated_z=5.0,
    )

    evidence = evaluate_expected_m0_peak_evidence(
        counts,
        detector_valid_mask=np.ones(counts.shape, dtype=np.bool_),
        candidates=candidates,
        policy=policy,
    )
    scaled = evaluate_expected_m0_peak_evidence(
        13.0 * counts,
        detector_valid_mask=np.ones(counts.shape, dtype=np.bool_),
        candidates=candidates,
        policy=policy,
    )

    assert tuple(item.integer_L for item in evidence) == (3, 6, 9, 12)
    assert tuple(item.classification for item in evidence) == (
        "LOCAL_SIGNAL_SUPPORTED",
        "LOCAL_SIGNAL_BELOW_GATE",
        "OUTSIDE_PANEL",
        "INSUFFICIENT_VALID_SUPPORT",
    )
    assert np.allclose(
        [item.peak_z for item in evidence[:2]],
        [item.peak_z for item in scaled[:2]],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert np.allclose(
        [item.integrated_z for item in evidence[:2]],
        [item.integrated_z for item in scaled[:2]],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert evaluate_expected_m0_peak_evidence(counts, candidates=()) == ()

    hot_pixel_counts = np.full((81, 81), 20.0)
    hot_pixel_counts[40, 40] = 200.0
    hot_pixel = evaluate_expected_m0_peak_evidence(
        hot_pixel_counts,
        candidates=(ExpectedM0Peak(integer_L=3, column_px=40.0, row_px=40.0),),
    )
    assert hot_pixel[0].classification == "LOCAL_SIGNAL_BELOW_GATE"

    empty_support = evaluate_expected_m0_peak_evidence(
        counts,
        candidates=(ExpectedM0Peak(integer_L=3, column_px=40.5, row_px=40.5),),
        policy=M0PeakEvidencePolicy(
            core_radius_px=0.1,
            background_inner_radius_px=0.2,
            background_outer_radius_px=0.3,
        ),
    )
    assert empty_support[0].classification == "INSUFFICIENT_VALID_SUPPORT"
