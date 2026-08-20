"""Position-free indexing for tracked or simulated Bi2Se3 detector images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from rasim_next.io.osc import read_osc
from rasim_next.pipeline.configured_simulation import (
    ConfiguredSimulationInputs,
    SimulationConfiguration,
    build_configured_simulation_inputs,
    build_nominal_ewald_context,
    evaluate_nominal_integer_l_markers,
    load_simulation_config,
)
from rasim_next.selection import (
    BlindIndexingPolicy,
    BranchTrackDecision,
    MarkerIndexingDecision,
    MarkerIndexingStatus,
    MeasuredIndexingResult,
    MeasuredPeakDiscovery,
    PeakIndexingPolicy,
    build_osc_angle_frame,
    detector_valid_mask_from_counts,
    discover_measured_cake_peaks,
    index_discovered_integer_l_peaks,
    select_confident_branch_tracks,
)

ROOT = Path(__file__).resolve().parents[1]
OSC_FILES = (
    "Bi2Se3_5m_5d.osc.gz",
    "Bi2Se3_10d_5m.osc.gz",
    "Bi2Se3_15d_5m.osc.gz",
)
DEFAULT_INCIDENCE_DEG = (5.0, 10.0, 15.0)
M1_ROD_KEYS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))


def _inputs_for_incidence(
    config: SimulationConfiguration,
    incidence_deg: float,
) -> ConfiguredSimulationInputs:
    if not config.instrument.axis_rotations:
        raise ValueError("the Bi2Se3 configuration must declare an incidence rotation")
    rotation = replace(config.instrument.axis_rotations[0], angle_deg=incidence_deg)
    minimum_physical_source_count = config.source.minimum_physical_sample_count
    configured = replace(
        config,
        source=replace(config.source, sample_count=minimum_physical_source_count),
        instrument=replace(
            config.instrument,
            axis_rotations=(rotation, *config.instrument.axis_rotations[1:]),
        ),
    )
    return build_configured_simulation_inputs(configured)


def _track_record(track: BranchTrackDecision) -> dict[str, object]:
    return {
        "family_m": track.family_m,
        "analytic_ewald_branch": track.branch,
        "root_sign": track.root_sign,
        "tag_branch": 1 if track.root_sign < 0 else 2,
        "representative_rod_hk": track.representative_rod_hk,
        "accepted": track.accepted,
        "image_ids": track.image_ids,
        "confident_site_count": track.confident_site_count,
        "distinct_integer_L": track.distinct_integer_L,
        "maximum_evidence_rms_px": track.rms_px,
        "reason": track.reason,
    }


def _decision_record(decision: MarkerIndexingDecision, used: bool) -> dict[str, object]:
    record = {
        "family_m": decision.key.family_m,
        "integer_L": decision.key.integer_L,
        "analytic_ewald_branch": decision.key.branch,
        "root_sign": decision.key.root_sign,
        "tag_branch": decision.key.tag_branch,
        "representative_rod_hk": decision.key.representative_rod_hk,
        "status": decision.status.value,
        "used_for_geometry": used,
        "reason": decision.reason,
        "predicted_native_px": (
            decision.predicted_column_px,
            decision.predicted_row_px,
        ),
        "predicted_cake_rad": (
            decision.predicted_two_theta_rad,
            decision.predicted_phi_rad,
        ),
    }
    if decision.observed_column_px is not None:
        record.update(
            {
                "observed_native_px": (
                    decision.observed_column_px,
                    decision.observed_row_px,
                ),
                "observed_cake_rad": (
                    decision.observed_two_theta_rad,
                    decision.observed_phi_rad,
                ),
                "covariance_px2": decision.covariance_px2,
                "z_score": decision.z_score,
                "assignment_cost": decision.assignment_cost,
                "assignment_margin": decision.assignment_margin,
            }
        )
    return record


def _payload(
    manifest: MeasuredIndexingResult,
    discoveries: tuple[MeasuredPeakDiscovery, ...],
    elapsed_seconds: float,
) -> dict[str, object]:
    discovery_by_image = {item.image_id: item for item in discoveries}
    images = []
    for result in manifest.image_results:
        discovery = discovery_by_image[result.image_id]
        try:
            used_keys = set(manifest.observations_for(result.image_id).keys)
        except ValueError:
            used_keys = set()
        decisions = tuple(
            _decision_record(
                decision,
                decision.key in used_keys,
            )
            for decision in result.marker_decisions
        )
        indexed_decision_status_counts = {
            status.value: sum(decision.status == status for decision in result.marker_decisions)
            for status in MarkerIndexingStatus
        }
        images.append(
            {
                "image_id": result.image_id,
                "incidence_angle_deg": math.degrees(result.incidence_angle_rad),
                "result_hash": result.result_hash,
                "detector_data_hash": result.detector_data_hash,
                "detector_mask_hash": result.detector_mask_hash,
                "detector_mask_revision": result.detector_mask_revision,
                "position_free_discovery_hash": discovery.discovery_hash,
                "globally_discovered_peak_count": len(discovery.peaks),
                "uniquely_q_indexed_peak_count": len(result.marker_decisions),
                "indexed_decision_status_counts": indexed_decision_status_counts,
                "used_tag_count": sum(record["used_for_geometry"] for record in decisions),
                "marker_decisions": decisions,
            }
        )
    return {
        "schema": "bi2se3-position-free-measured-indexing-run.v2",
        "manifest_hash": manifest.manifest_hash,
        "elapsed_seconds": elapsed_seconds,
        "policy": asdict(manifest.policy),
        "coordinate_route": {
            "osc_orientation": "one clockwise rotation at read_osc boundary",
            "detection_chart": "global uniform (two_theta, phi) search cake",
            "marker_position_input": "none",
            "label_route": "detector point -> q_sample -> (m, L, analytic branch, root sign)",
            "exact_anchor_timing": "generated only after the discrete q-space label is frozen",
            "final_observation": "detector-native (column_px, row_px)",
            "quantitative_caker_available": True,
            "quantitative_caker_api": (
                "compile_detector_angle_projector + project_normalized_angle_field"
            ),
        },
        "metadata_note": (
            "The third tracked TOML/legacy label says 12 degrees; the filename, measured "
            "association, and user-authoritative value used here are 15 degrees."
        ),
        "branch_tracks": tuple(_track_record(track) for track in manifest.branch_tracks),
        "images": tuple(images),
    }


def _require_close(name: str, actual: object, expected: object) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    scale = max(
        float(np.max(np.abs(expected_array), initial=0.0)),
        1.0,
    )
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=512.0 * np.finfo(np.float64).eps * scale,
    ):
        raise ValueError(f"simulation diagnostic {name} does not match the requested context")


def _simulation_qualification_payload(
    diagnostic_path: Path,
    *,
    incidence_deg: float,
    allow_unresolved_diagnostic: bool,
) -> dict[str, object]:
    with np.load(diagnostic_path, allow_pickle=False) as data:
        required = {
            "image_A2",
            "valid_pixel_center",
            "ki_sample_Ainv",
            "per_rod_detector_mass_A2",
            "manifest_json",
        }
        if not required.issubset(data.files):
            raise ValueError("simulation diagnostic lacks required image, mask, or manifest arrays")
        detector_image = np.array(data["image_A2"], dtype=np.float64, copy=True)
        detector_mask = np.array(data["valid_pixel_center"], dtype=np.bool_, copy=True)
        diagnostic_ki = np.array(data["ki_sample_Ainv"], dtype=np.float64, copy=True)
        diagnostic_per_rod = np.array(
            data["per_rod_detector_mass_A2"],
            dtype=np.float64,
            copy=True,
        )
        manifest_array = np.asarray(data["manifest_json"])
    if manifest_array.dtype != np.uint8 or manifest_array.ndim != 1:
        raise ValueError("simulation diagnostic manifest_json must be one uint8 array")
    try:
        source_manifest = json.loads(manifest_array.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("simulation diagnostic manifest_json is invalid") from error
    if not isinstance(source_manifest, dict):
        raise ValueError("simulation diagnostic manifest must be a JSON object")
    required_manifest_fields = {
        "adaptive_tolerance_satisfied",
        "adaptive_unresolved_pixel_count",
        "artifact_status",
        "branch",
        "detector_column_pitch_m",
        "detector_distance_m",
        "detector_reference_column_row_px",
        "detector_row_pitch_m",
        "detector_shape_rc",
        "image_sha256",
        "ki_sample_Ainv",
        "m0_intensity_status",
        "m0_specular_column_row_px",
        "m1_rod_keys",
        "mosaic_gaussian_sigma_deg",
        "mosaic_lorentzian_hwhm_deg",
        "mosaic_lorentzian_probability",
        "nonzero_pixel_count",
        "per_rod_detector_mass_A2",
        "strength_layer_count",
        "strength_normalization",
        "strength_shared_disorder_epsilon",
        "total_detector_mass_A2",
        "wavelength_A",
    }
    missing_fields = sorted(required_manifest_fields.difference(source_manifest))
    if missing_fields:
        raise ValueError(f"simulation diagnostic manifest lacks fields: {missing_fields}")
    unresolved = int(source_manifest["adaptive_unresolved_pixel_count"])
    source_status = source_manifest["artifact_status"]
    tolerance_satisfied = source_manifest["adaptive_tolerance_satisfied"]
    if source_status == "ADAPTIVE_CONVERGED":
        if tolerance_satisfied is not True or unresolved != 0:
            raise ValueError("converged simulation diagnostic has inconsistent adaptive status")
        qualification_status = "QUALIFIED_SINGLE_INCIDENCE"
    elif source_status == "ADAPTIVE_UNRESOLVED_DIAGNOSTIC":
        if tolerance_satisfied is not False or unresolved <= 0:
            raise ValueError("unresolved simulation diagnostic has inconsistent adaptive status")
        if not allow_unresolved_diagnostic:
            raise RuntimeError(
                "simulation diagnostic has unresolved adaptive pixels; pass "
                "--allow-unresolved-diagnostic only for an explicitly non-accepted audit"
            )
        qualification_status = "NON_ACCEPTED_UNRESOLVED_DIAGNOSTIC"
    else:
        raise ValueError("simulation diagnostic is neither converged nor an unresolved audit")
    config = load_simulation_config(ROOT / "configs" / "bi2se3_simulation.yaml")
    inputs = _inputs_for_incidence(config, incidence_deg)
    context = build_nominal_ewald_context(inputs)
    frame = build_osc_angle_frame(
        mean_direction_lab=inputs.config.source.mean_direction_lab,
        instrument=inputs.instrument,
        sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
        revision=(f"bi2se3-detector-image-angle-frame.simulated-{incidence_deg:g}deg.v1"),
    )
    if (
        detector_image.shape != inputs.instrument.detector_shape_rc
        or detector_mask.shape != detector_image.shape
        or not np.all(np.isfinite(detector_image))
        or np.any(detector_image < 0.0)
        or not np.any(detector_mask)
    ):
        raise ValueError("simulation diagnostic image or validity mask is invalid")
    if tuple(source_manifest["detector_shape_rc"]) != detector_image.shape:
        raise ValueError("simulation diagnostic detector shape provenance is inconsistent")
    expected_image_hash = hashlib.sha256(
        memoryview(np.ascontiguousarray(detector_image)).cast("B")
    ).hexdigest()
    if source_manifest["image_sha256"] != expected_image_hash:
        raise ValueError("simulation diagnostic image hash is invalid")
    if int(source_manifest["nonzero_pixel_count"]) != int(np.count_nonzero(detector_image)):
        raise ValueError("simulation diagnostic nonzero-pixel count is invalid")
    _require_close("NPZ/manifest ki_sample_Ainv", diagnostic_ki, source_manifest["ki_sample_Ainv"])
    _require_close(
        "NPZ/manifest per_rod_detector_mass_A2",
        diagnostic_per_rod,
        source_manifest["per_rod_detector_mass_A2"],
    )
    _require_close(
        "ki_sample_Ainv",
        diagnostic_ki,
        context.geometry.coating.ki_sample_Ainv,
    )
    _require_close(
        "wavelength_A",
        source_manifest["wavelength_A"],
        context.geometry.incident.states.wavelength_A[0],
    )
    instrument = inputs.instrument
    _require_close(
        "detector_row_pitch_m",
        source_manifest["detector_row_pitch_m"],
        instrument.detector_row_pitch_m,
    )
    _require_close(
        "detector_column_pitch_m",
        source_manifest["detector_column_pitch_m"],
        instrument.detector_column_pitch_m,
    )
    _require_close(
        "detector_reference_column_row_px",
        source_manifest["detector_reference_column_row_px"],
        instrument.detector_reference_coordinate_px,
    )
    detector_distance_m = float(
        np.linalg.norm(
            instrument.lab_from_detector.translation_m - instrument.lab_from_sample.translation_m
        )
    )
    _require_close(
        "detector_distance_m",
        source_manifest["detector_distance_m"],
        detector_distance_m,
    )
    expected_m1_rods = tuple(
        (rod.h, rod.k)
        for rod in context.geometry.coating.bragg_space.config.rods
        if rod.family_m == 1
    )
    if (
        expected_m1_rods != M1_ROD_KEYS
        or tuple(tuple(int(value) for value in rod) for rod in source_manifest["m1_rod_keys"])
        != expected_m1_rods
    ):
        raise ValueError("simulation diagnostic m=1 rod scope is inconsistent")
    if int(source_manifest["branch"]) != 2:
        raise ValueError("simulation diagnostic must contain analytic branch 2")
    mosaic = inputs.mosaic
    _require_close(
        "mosaic_gaussian_sigma_deg",
        source_manifest["mosaic_gaussian_sigma_deg"],
        math.degrees(mosaic.gaussian_sigma_rad),
    )
    _require_close(
        "mosaic_lorentzian_hwhm_deg",
        source_manifest["mosaic_lorentzian_hwhm_deg"],
        math.degrees(mosaic.lorentzian_half_width_rad),
    )
    _require_close(
        "mosaic_lorentzian_probability",
        source_manifest["mosaic_lorentzian_probability"],
        mosaic.lorentzian_probability,
    )
    if (
        int(source_manifest["strength_layer_count"]) != inputs.strength.layers
        or source_manifest["strength_normalization"] != inputs.strength.normalization.value
    ):
        raise ValueError("simulation diagnostic finite-stack strength provenance is inconsistent")
    _require_close(
        "strength_shared_disorder_epsilon",
        source_manifest["strength_shared_disorder_epsilon"],
        inputs.strength.shared_disorder_epsilon,
    )
    m0_rod = next(
        rod for rod in context.geometry.coating.bragg_space.config.rods if rod.family_m == 0
    )
    specular = context.geometry.map_specular_geometry(
        rod=m0_rod,
        alpha_rad=0.0,
        beta_rad=0.0,
    )
    _require_close(
        "m0_specular_column_row_px",
        source_manifest["m0_specular_column_row_px"],
        (float(specular.geometry.column_px), float(specular.geometry.row_px)),
    )
    if source_manifest["m0_intensity_status"] != "SPECULAR_INTENSITY_EXCLUDED":
        raise ValueError("simulation diagnostic specular intensity status is invalid")
    declared_total = float(source_manifest["total_detector_mass_A2"])
    _require_close("per-rod total mass", np.sum(diagnostic_per_rod), declared_total)
    _require_close("image total mass", np.sum(detector_image, dtype=np.float64), declared_total)
    policy = BlindIndexingPolicy(track_policy=PeakIndexingPolicy(cake_step_px=1.0))

    # This stage receives only the raster, mask, calibration, lattice, and incident state.
    discovery = discover_measured_cake_peaks(
        detector_image,
        instrument=inputs.instrument,
        angle_frame=frame,
        image_id=f"simulated-{incidence_deg:g}deg",
        detector_valid_mask=detector_mask,
        detector_mask_revision="simulation-valid-pixel-center.v1",
        policy=policy,
    )
    indexed = index_discovered_integer_l_peaks(
        discovery,
        ewald_context=context,
        angle_frame=frame,
        incidence_angle_rad=math.radians(incidence_deg),
    )
    frozen_before_posthoc_oracle = {
        "discovery_hash": discovery.discovery_hash,
        "result_hash": indexed.result_hash,
        "discovered_peak_count": len(discovery.peaks),
        "indexed_site_count": len(indexed.marker_decisions),
    }

    # Same-context anchors are generated only after discovery and indexing are frozen.
    truth = evaluate_nominal_integer_l_markers(context)
    truth_coordinates = {
        (
            int(truth.family_m[index]),
            int(truth.integer_L[index]),
            int(truth.branch[index]),
            int(truth.root_sign[index]),
        ): np.asarray((truth.column_px[index], truth.row_px[index]), dtype=np.float64)
        for index in range(truth.family_m.size)
    }
    accepted_tracks = {track.track_key for track in indexed.branch_tracks if track.accepted}
    accepted_decisions = tuple(
        decision for decision in indexed.marker_decisions if decision.track_key in accepted_tracks
    )
    site_audit = []
    for decision in accepted_decisions:
        key = (
            decision.key.family_m,
            decision.key.integer_L,
            decision.key.branch,
            decision.key.root_sign,
        )
        truth_coordinate = truth_coordinates.get(key)
        record = _decision_record(decision, used=True)
        record["same_context_label_present"] = truth_coordinate is not None
        if truth_coordinate is not None:
            predicted = np.asarray(
                (decision.predicted_column_px, decision.predicted_row_px), dtype=np.float64
            )
            observed = np.asarray(
                (decision.observed_column_px, decision.observed_row_px), dtype=np.float64
            )
            record["same_context_anchor_roundtrip_error_px"] = float(
                np.linalg.norm(predicted - truth_coordinate)
            )
            record["observed_to_same_context_alpha0_anchor_error_px"] = float(
                np.linalg.norm(observed - truth_coordinate)
            )
        site_audit.append(record)
    matched_count = sum(record["same_context_label_present"] for record in site_audit)
    if not accepted_decisions:
        raise RuntimeError("simulation indexing recovered no locally confident branch sites")
    if matched_count != len(accepted_decisions):
        raise RuntimeError(
            "simulation indexing produced a label absent from the same-context oracle"
        )
    roundtrip_error = max(
        float(record["same_context_anchor_roundtrip_error_px"]) for record in site_audit
    )
    observed_anchor_error = max(
        float(record["observed_to_same_context_alpha0_anchor_error_px"]) for record in site_audit
    )
    if roundtrip_error > 1.0e-8:
        raise RuntimeError("same-context exact-anchor round trip exceeded numerical tolerance")
    if observed_anchor_error > policy.track_policy.maximum_track_rms_px:
        raise RuntimeError("recovered simulation positions are too far from alpha-zero anchors")
    return {
        "schema": "bi2se3-position-free-simulation-qualification.v2",
        "diagnostic_path": str(diagnostic_path.resolve()),
        "qualification_status": qualification_status,
        "source_artifact_status": source_status,
        "adaptive_unresolved_pixel_count": unresolved,
        "frozen_before_posthoc_oracle": frozen_before_posthoc_oracle,
        "policy": asdict(policy),
        "validation_scope": "local_single_incidence_m1_branch2",
        "cross_incidence_confidence_established": False,
        "marker_coordinates_supplied_to_discovery": False,
        "same_context_oracle_scope": (
            "algebraic label and alpha-zero-anchor consistency; not an independent physics oracle"
        ),
        "detectable_site_recall": None,
        "detectable_site_recall_reason": (
            "the continuous mosaic raster has no independent binary visible-integer-L catalogue"
        ),
        "local_branch_tracks": tuple(_track_record(track) for track in indexed.branch_tracks),
        "locally_accepted_site_count": len(accepted_decisions),
        "same_context_oracle_key_match_count": matched_count,
        "same_context_oracle_key_fraction": matched_count / len(accepted_decisions),
        "maximum_observed_to_same_context_alpha0_anchor_error_px": observed_anchor_error,
        "site_audit": tuple(site_audit),
        "position_observables": {
            "observed_native_px": "measured raster-feature location",
            "predicted_native_px": "exact alpha-zero integer-L anchor generated after labeling",
            "covariance_px2": "conservative ridge-support weight, not calibrated coverage",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incidence-deg",
        type=float,
        nargs=3,
        default=DEFAULT_INCIDENCE_DEG,
        metavar=("FIRST", "SECOND", "THIRD"),
        help="incidence angles for the three tracked files (default: 5 10 15)",
    )
    parser.add_argument(
        "--simulation-diagnostic",
        type=Path,
        help="index one external 3000x3000 .ra_diag.npz simulation qualification image",
    )
    parser.add_argument(
        "--simulation-incidence-deg",
        type=float,
        default=5.0,
        help="incidence angle for --simulation-diagnostic (default: 5)",
    )
    parser.add_argument(
        "--allow-unresolved-diagnostic",
        action="store_true",
        help="permit an explicitly labelled nonconverged simulation diagnostic",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="write the complete immutable decision manifest to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if any(not math.isfinite(value) for value in arguments.incidence_deg):
        raise ValueError("incidence angles must be finite")
    if not math.isfinite(arguments.simulation_incidence_deg):
        raise ValueError("simulation incidence angle must be finite")
    if arguments.simulation_diagnostic is not None:
        payload = _simulation_qualification_payload(
            arguments.simulation_diagnostic,
            incidence_deg=arguments.simulation_incidence_deg,
            allow_unresolved_diagnostic=arguments.allow_unresolved_diagnostic,
        )
        if arguments.json:
            json.dump(payload, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        frozen = payload["frozen_before_posthoc_oracle"]
        print(f"discovery {frozen['discovery_hash']}")
        print(f"indexed {frozen['result_hash']}")
        print(
            f"discovered={frozen['discovered_peak_count']} "
            f"indexed={frozen['indexed_site_count']} "
            f"locally_accepted={payload['locally_accepted_site_count']} "
            f"same_context_matches={payload['same_context_oracle_key_match_count']}"
        )
        for record in payload["site_audit"]:
            print(
                f"  m={record['family_m']} L={record['integer_L']} "
                f"analytic_branch={record['analytic_ewald_branch']} "
                f"tag_branch={record['tag_branch']} "
                f"observed={record['observed_native_px']} "
                f"anchor={record['predicted_native_px']}"
            )
        print(
            f"source_artifact_status={payload['source_artifact_status']} "
            f"qualification_status={payload['qualification_status']} "
            f"unresolved={payload['adaptive_unresolved_pixel_count']}"
        )
        return 0
    config = load_simulation_config(ROOT / "configs" / "bi2se3_simulation.yaml")
    osc_directory = ROOT / "examples" / "bi2se3" / "osc"
    blind_policy = BlindIndexingPolicy(track_policy=PeakIndexingPolicy(cake_step_px=1.0))
    policy: PeakIndexingPolicy = blind_policy.track_policy
    started = perf_counter()
    image_results = []
    discoveries = []
    for incidence_deg, filename in zip(
        arguments.incidence_deg,
        OSC_FILES,
        strict=True,
    ):
        inputs = _inputs_for_incidence(config, incidence_deg)
        context = build_nominal_ewald_context(inputs)
        osc = read_osc(osc_directory / filename)
        image_id = filename.removesuffix(".gz").removesuffix(".osc")
        frame = build_osc_angle_frame(
            mean_direction_lab=inputs.config.source.mean_direction_lab,
            instrument=inputs.instrument,
            sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
            revision=f"bi2se3-detector-image-angle-frame.{image_id}.v1",
        )
        discovery = discover_measured_cake_peaks(
            osc.detector_native_counts,
            instrument=inputs.instrument,
            angle_frame=frame,
            image_id=image_id,
            detector_valid_mask=detector_valid_mask_from_counts(osc.detector_native_counts),
            detector_mask_revision="osc-all-zero-edge-mask.v1",
            policy=blind_policy,
        )
        discoveries.append(discovery)
        image_results.append(
            index_discovered_integer_l_peaks(
                discovery,
                ewald_context=context,
                angle_frame=frame,
                incidence_angle_rad=math.radians(incidence_deg),
            )
        )
    manifest = select_confident_branch_tracks(tuple(image_results), policy=policy)
    payload = _payload(manifest, tuple(discoveries), perf_counter() - started)
    if arguments.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    accepted = tuple(track for track in manifest.branch_tracks if track.accepted)
    print(f"manifest {manifest.manifest_hash}")
    print(f"accepted branch tracks: {len(accepted)}/{len(manifest.branch_tracks)}")
    for track in accepted:
        print(
            f"  m={track.family_m} analytic_branch={track.branch} "
            f"tag_branch={1 if track.root_sign < 0 else 2} "
            f"L={track.distinct_integer_L} images={track.image_ids}"
        )
    for image in payload["images"]:
        print(
            f"{image['image_id']}: used={image['used_tag_count']} "
            f"status={image['indexed_decision_status_counts']}"
        )
    print(payload["metadata_note"])
    print(f"elapsed_seconds={payload['elapsed_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
