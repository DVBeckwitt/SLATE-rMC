"""Fit one shared detector-native geometry correction to an indexed OSC series."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tracemalloc
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np

from rasim_next.fitting import (
    IndexedGeometryImage,
    SharedGeometryCorrectionBounds,
    SharedGeometryCorrections,
    audit_indexed_geometry_series_roots,
    evaluate_indexed_geometry_series_metrics,
    evaluate_indexed_geometry_series_residual,
    fit_indexed_geometry_series,
)
from rasim_next.selection import (
    MeasuredIndexingResult,
    audit_frozen_marker_visibility,
    audit_frozen_osc_geometry_reindexing,
    index_osc_geometry_series,
    load_osc_geometry_series,
    reindex_frozen_osc_geometry_series,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "bi2se3_osc_geometry_fit.yaml"
_MAXIMUM_POOLED_RMS_PX = 3.0
_MAXIMUM_PER_IMAGE_RMS_PX = 4.0
_MAXIMUM_SITE_ERROR_PX = 8.0
_MAXIMUM_HELDOUT_RMS_PX = 5.0
_MAXIMUM_HELDOUT_ERROR_PX = 10.0
_MAXIMUM_MULTISTART_SEPARATION_PX = 0.25
_BI2SE3_QUALIFICATION_PROFILE = "bi2se3-osc-5-10-15.v1"
_BI2SE3_INDEXED_MANIFEST_HASH = (
    "sha256-1de21e03a801fa38390ef5280133666474bfd969377024ef6dd4fb34e40f3132"
)


def _subset_images(
    images: tuple[IndexedGeometryImage, ...],
    integer_l: set[int],
    *,
    selected: bool,
) -> tuple[IndexedGeometryImage, ...]:
    result = []
    for image in images:
        mask = np.asarray(
            [((key.integer_L in integer_l) is selected) for key in image.observations.keys],
            dtype=np.bool_,
        )
        if np.any(mask):
            result.append(
                IndexedGeometryImage(
                    image_id=image.image_id,
                    commanded_angle_rad=image.commanded_angle_rad,
                    model=image.model,
                    observations=image.observations.subset(mask),
                )
            )
    return tuple(result)


def _metrics_payload(metrics: object) -> dict[str, object]:
    return {
        "site_rms_px": metrics.site_rms_px,
        "site_max_px": metrics.site_max_px,
        "chord_angle_rms_rad": metrics.chord_angle_rms_rad,
        "per_image": tuple(asdict(item) for item in metrics.per_image),
    }


def _fit_payload(result: object) -> dict[str, object]:
    return {
        "success": result.success,
        "message": result.message,
        "parameterization_id": result.parameterization_id,
        "corrections": asdict(result.corrections),
        "jacobian_rank": result.jacobian_rank,
        "jacobian_condition": result.jacobian_condition,
        "scaled_jacobian_singular_values": result.scaled_jacobian_singular_values.tolist(),
        "scaled_jacobian_weakest_direction": (result.scaled_jacobian_weakest_direction.tolist()),
        "active_bounds": result.active_bounds.tolist(),
        "training_site_rms_px": result.training_site_rms_px,
        "training_site_max_px": result.training_site_max_px,
        "training_chord_angle_rms_rad": result.training_chord_angle_rms_rad,
        "per_image": tuple(asdict(item) for item in result.per_image),
        "model_evaluation_count": result.model_evaluation_count,
        "optimizer_function_evaluation_count": result.optimizer_function_evaluation_count,
        "optimizer_jacobian_evaluation_count": result.optimizer_jacobian_evaluation_count,
    }


def _key_payload(key: object) -> dict[str, object]:
    return {
        "family_m": key.family_m,
        "integer_L": key.integer_L,
        "analytic_ewald_branch": key.branch,
        "root_sign": key.root_sign,
        "representative_rod_hk": key.representative_rod_hk,
    }


def _prediction_payload(
    images: tuple[IndexedGeometryImage, ...],
    corrections: SharedGeometryCorrections,
) -> tuple[dict[str, object], ...]:
    payload = []
    for image in sorted(images, key=lambda item: item.image_id):
        prediction = image.predict_integer_l_tags(image.observations.keys, corrections)
        entries = []
        for index, key in enumerate(image.observations.keys):
            error = prediction.coordinates_px[index] - image.observations.coordinates_px[index]
            entries.append(
                {
                    "key": _key_payload(key),
                    "observed_coordinate_px": image.observations.coordinates_px[index].tolist(),
                    "predicted_coordinate_px": prediction.coordinates_px[index].tolist(),
                    "detector_status": str(prediction.detector_status[index]),
                    "error_norm_px": float(np.linalg.norm(error)),
                }
            )
        payload.append({"image_id": image.image_id, "sites": tuple(entries)})
    return tuple(payload)


def _deterministic_starts(
    bounds: SharedGeometryCorrectionBounds,
) -> tuple[SharedGeometryCorrections, ...]:
    pattern = np.asarray((1.0, -0.8, 0.6, -0.4, 0.7, -0.5, 0.3, 0.9, -0.7))
    offset = 0.08 * bounds.half_span * pattern
    return (
        SharedGeometryCorrections.zero(),
        SharedGeometryCorrections.from_array(offset),
        SharedGeometryCorrections.from_array(-offset),
    )


def _maximum_prediction_separation_px(
    images: tuple[IndexedGeometryImage, ...],
    first: SharedGeometryCorrections,
    second: SharedGeometryCorrections,
) -> float:
    maximum = 0.0
    for image in images:
        left = image.predict_integer_l_tags(image.observations.keys, first)
        right = image.predict_integer_l_tags(image.observations.keys, second)
        maximum = max(
            maximum,
            float(np.max(np.linalg.norm(left.coordinates_px - right.coordinates_px, axis=1))),
        )
    return maximum


def _track_export_counts(
    selection: MeasuredIndexingResult,
    image_ids: tuple[str, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for image_id in image_ids:
        try:
            counts[image_id] = len(selection.observations_for(image_id).keys)
        except ValueError:
            counts[image_id] = 0
    return counts


def fit_osc_geometry_series(
    manifest_path: str | Path,
    *,
    heldout_integer_l: tuple[int, ...] = (),
    benchmark: bool = False,
) -> dict[str, object]:
    """Index once, fit frozen observations jointly, and audit without reassignment."""

    series = load_osc_geometry_series(manifest_path)
    if series.qualification_profile not in {None, _BI2SE3_QUALIFICATION_PROFILE}:
        raise ValueError(f"unsupported qualification profile {series.qualification_profile!r}")
    indexing = index_osc_geometry_series(series)
    zero = SharedGeometryCorrections.zero()
    bounds = SharedGeometryCorrectionBounds.rasim_multi_angle_pose()
    if indexing.indexed_images is None:
        raise RuntimeError("the initial indexing run did not retain fit-ready geometry models")
    images = indexing.indexed_images
    baseline = evaluate_indexed_geometry_series_metrics(images, zero)

    starts = _deterministic_starts(bounds)
    fit_results = []
    fit_wall_times = []
    for initial in starts:
        started = perf_counter()
        fit_results.append(fit_indexed_geometry_series(images, initial=initial, bounds=bounds))
        fit_wall_times.append(perf_counter() - started)
    fit_objective_sums = []
    for candidate in fit_results:
        residual = evaluate_indexed_geometry_series_residual(images, candidate.corrections)
        fit_objective_sums.append(float(np.dot(residual, residual)))
    selected_start_index = min(
        range(len(fit_results)),
        key=lambda index: (not fit_results[index].success, fit_objective_sums[index]),
    )
    result = fit_results[selected_start_index]
    post_fit = evaluate_indexed_geometry_series_metrics(images, result.corrections)
    multi_start = tuple(
        {
            "initial": asdict(initial),
            "fit": _fit_payload(candidate),
            "normalized_correction_separation_from_selected_start": float(
                np.max(
                    np.abs(candidate.corrections.as_array() - result.corrections.as_array())
                    / bounds.half_span
                )
            ),
            "maximum_prediction_separation_from_selected_start_px": (
                _maximum_prediction_separation_px(
                    images,
                    result.corrections,
                    candidate.corrections,
                )
            ),
            "objective_sum_squares": objective_sum,
            "wall_time_seconds": elapsed,
        }
        for initial, candidate, objective_sum, elapsed in zip(
            starts, fit_results, fit_objective_sums, fit_wall_times, strict=True
        )
    )

    root_audit = audit_indexed_geometry_series_roots(images, result.corrections)
    if root_audit.classification != "SAME":
        raise RuntimeError("the independent fitted-root audit changed a frozen marker")

    indexing_wall_times = [indexing.elapsed_seconds]
    corrected = {image.image_id: image.corrected_instrument(result.corrections) for image in images}
    frozen_reindex_started = perf_counter()
    frozen_reindexed = reindex_frozen_osc_geometry_series(
        indexing,
        instrument_by_image_id=corrected,
    )
    frozen_reindex_seconds = perf_counter() - frozen_reindex_started
    visibility = audit_frozen_osc_geometry_reindexing(
        indexing,
        frozen_reindexed,
        instrument_by_image_id=corrected,
    )
    global_rediscovery = index_osc_geometry_series(
        series,
        instrument_by_image_id=corrected,
    )
    indexing_wall_times.append(global_rediscovery.elapsed_seconds)
    global_visibility = audit_frozen_marker_visibility(
        indexing.selection,
        global_rediscovery.selection,
    )
    image_ids = tuple(image.image_id for image in series.images)
    outer_payload = {
        "classification": visibility.classification,
        "basis": (
            "original position-free native candidates retain their full reciprocal keys "
            "and coherent tracks after corrected-geometry relabeling"
        ),
        "frozen_manifest_hash": indexing.selection.manifest_hash,
        "frozen_reindex_manifest_hash": frozen_reindexed.selection.manifest_hash,
        "frozen_reindex_track_export_counts": _track_export_counts(
            frozen_reindexed.selection,
            image_ids,
        ),
        "frozen_reindex_wall_time_seconds": frozen_reindex_seconds,
        "frozen_visibility": asdict(visibility),
        "global_rediscovery": {
            "classification": global_visibility.classification,
            "acceptance_critical": False,
            "basis": (
                "diagnostic only because cake sampling and candidate generation change "
                "with the corrected geometry"
            ),
            "manifest_hash": global_rediscovery.selection.manifest_hash,
            "track_export_counts": _track_export_counts(
                global_rediscovery.selection,
                image_ids,
            ),
            "frozen_visibility": asdict(global_visibility),
            "wall_time_seconds": global_rediscovery.elapsed_seconds,
        },
    }

    cross_validation: dict[str, object] | None = None
    cross_validation_fit_succeeded = True
    heldout_set = set(heldout_integer_l)
    if heldout_set:
        available_integer_l = {key.integer_L for image in images for key in image.observations.keys}
        missing_integer_l = heldout_set - available_integer_l
        if missing_integer_l:
            raise ValueError(
                "held-out integer L values are absent from the frozen observations: "
                f"{sorted(missing_integer_l)}"
            )
        training_images = _subset_images(images, heldout_set, selected=False)
        heldout_images = _subset_images(images, heldout_set, selected=True)
        if len(training_images) != len(images) or not heldout_images:
            raise ValueError("held-out integer L values must leave training sites in every image")
        started = perf_counter()
        cross_fit = fit_indexed_geometry_series(
            training_images,
            initial=zero,
            bounds=bounds,
        )
        cross_validation_fit_succeeded = cross_fit.success
        cross_validation = {
            "heldout_integer_L": tuple(sorted(heldout_set)),
            "fit": _fit_payload(cross_fit),
            "heldout": _metrics_payload(
                evaluate_indexed_geometry_series_metrics(
                    heldout_images,
                    cross_fit.corrections,
                )
            ),
            "heldout_predictions": _prediction_payload(
                heldout_images,
                cross_fit.corrections,
            ),
            "wall_time_seconds": perf_counter() - started,
        }

    warm_residual_median_seconds: float | None = None
    fit_peak_memory_bytes: int | None = None
    if benchmark:
        residual_times = []
        for _ in range(25):
            started = perf_counter()
            evaluate_indexed_geometry_series_residual(images, result.corrections)
            residual_times.append(perf_counter() - started)
        warm_residual_median_seconds = float(np.median(residual_times))
        tracemalloc.start()
        fit_indexed_geometry_series(images, initial=zero, bounds=bounds)
        _, fit_peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    baseline_by_id = {item.image_id: item for item in baseline.per_image}
    every_image_improved = all(
        item.site_rms_px < baseline_by_id[item.image_id].site_rms_px for item in post_fit.per_image
    )
    pairwise_multistart_prediction_separation = tuple(
        {
            "left_start_index": left,
            "right_start_index": right,
            "maximum_prediction_separation_px": _maximum_prediction_separation_px(
                images,
                fit_results[left].corrections,
                fit_results[right].corrections,
            ),
        }
        for left, right in combinations(range(len(fit_results)), 2)
    )
    maximum_multistart_prediction_separation = max(
        float(item["maximum_prediction_separation_px"])
        for item in pairwise_multistart_prediction_separation
    )
    fit_metrics_pass = (
        post_fit.site_rms_px <= _MAXIMUM_POOLED_RMS_PX
        and post_fit.site_max_px <= _MAXIMUM_SITE_ERROR_PX
        and all(item.site_rms_px <= _MAXIMUM_PER_IMAGE_RMS_PX for item in post_fit.per_image)
    )
    minimum_improvement_pass = (
        post_fit.site_rms_px <= baseline.site_rms_px
        if baseline.site_rms_px <= _MAXIMUM_POOLED_RMS_PX
        else post_fit.site_rms_px <= 0.75 * baseline.site_rms_px
    )
    heldout_metrics_pass = cross_validation is None or (
        cross_validation["heldout"]["site_rms_px"] <= _MAXIMUM_HELDOUT_RMS_PX
        and cross_validation["heldout"]["site_max_px"] <= _MAXIMUM_HELDOUT_ERROR_PX
    )
    all_fits_succeeded = all(item.success for item in fit_results)
    no_active_bounds = not any(np.any(item.active_bounds) for item in fit_results)
    multistart_stable = (
        maximum_multistart_prediction_separation <= _MAXIMUM_MULTISTART_SEPARATION_PX
    )
    root_audit_same = root_audit.classification == "SAME"
    outer_audit_same = outer_payload["classification"] == "SAME"
    run_completed = all(
        (
            all_fits_succeeded,
            multistart_stable,
            root_audit_same,
            outer_audit_same,
        )
    )
    qualification_requested = series.qualification_profile is not None
    qualification_heldout_matches = heldout_set == {4, 11}
    qualification_evidence_complete = (
        cross_validation is not None and benchmark and qualification_heldout_matches
    )
    qualification_manifest_matches = (
        indexing.selection.manifest_hash == _BI2SE3_INDEXED_MANIFEST_HASH
    )
    accepted = all(
        (
            run_completed,
            fit_metrics_pass,
            minimum_improvement_pass,
            cross_validation_fit_succeeded,
            heldout_metrics_pass,
            qualification_requested,
            qualification_evidence_complete,
            qualification_manifest_matches,
        )
    )

    return {
        "schema": "rasim-osc-geometry-fit-result-v3",
        "manifest_path": str(Path(manifest_path).resolve()),
        "indexed_manifest_hash": indexing.selection.manifest_hash,
        "run_completed": run_completed,
        "indexing_wall_time_seconds": math.fsum(indexing_wall_times),
        "indexing_pass_wall_times_seconds": tuple(indexing_wall_times),
        "geometry_setup_wall_time_seconds": indexing.geometry_setup_seconds,
        "source_state_policy": "nominal_source_center.zero_divergence.mean_wavelength.v1",
        "source_state_count_per_image": 1,
        "indexing_mosaic_work": "none",
        "indexing_intensity_work": "none",
        "fit_mosaic_work": "none",
        "fit_intensity_work": "none",
        "fit_pixel_work": "none",
        "image_site_counts": {image.image_id: len(image.observations.keys) for image in images},
        "baseline": _metrics_payload(baseline),
        "cross_validation": cross_validation,
        "fit": _fit_payload(result),
        "post_fit": _metrics_payload(post_fit),
        "selected_start_index": selected_start_index,
        "primary_fit_wall_time_seconds": fit_wall_times[selected_start_index],
        "zero_start_fit_wall_time_seconds": fit_wall_times[0],
        "multi_start": multi_start,
        "pairwise_multistart_prediction_separation": (pairwise_multistart_prediction_separation),
        "root_audit": asdict(root_audit),
        "outer_audit": outer_payload,
        "benchmark": {
            "warm_residual_median_seconds": warm_residual_median_seconds,
            "fit_peak_memory_bytes": fit_peak_memory_bytes,
        },
        "qualification": {
            "accepted": accepted,
            "requested": qualification_requested,
            "profile_id": series.qualification_profile,
            "evidence_complete": qualification_evidence_complete,
            "heldout_integer_l_matches_profile": qualification_heldout_matches,
            "indexed_manifest_matches_profile": qualification_manifest_matches,
            "all_fits_succeeded": all_fits_succeeded,
            "every_image_improved": every_image_improved,
            "fit_metrics_pass": fit_metrics_pass,
            "minimum_improvement_pass": minimum_improvement_pass,
            "cross_validation_fit_succeeded": cross_validation_fit_succeeded,
            "heldout_metrics_pass": heldout_metrics_pass,
            "multistart_stable": multistart_stable,
            "maximum_pooled_rms_px": _MAXIMUM_POOLED_RMS_PX,
            "maximum_per_image_rms_px": _MAXIMUM_PER_IMAGE_RMS_PX,
            "maximum_site_error_px": _MAXIMUM_SITE_ERROR_PX,
            "maximum_heldout_rms_px": _MAXIMUM_HELDOUT_RMS_PX,
            "maximum_heldout_error_px": _MAXIMUM_HELDOUT_ERROR_PX,
            "maximum_multistart_separation_px": _MAXIMUM_MULTISTART_SEPARATION_PX,
            "no_active_bounds": no_active_bounds,
            "parameter_precision_qualified": no_active_bounds,
            "maximum_multistart_prediction_separation_px": (
                maximum_multistart_prediction_separation
            ),
            "root_audit_same": root_audit_same,
            "outer_audit_same": outer_audit_same,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--heldout-integer-l",
        type=int,
        nargs="+",
        default=(),
        help="integer-L values excluded from a separate cross-validation fit",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="also run separate warm-residual and traced peak-memory measurements",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        payload = fit_osc_geometry_series(
            arguments.manifest,
            heldout_integer_l=tuple(arguments.heldout_integer_l),
            benchmark=arguments.benchmark,
        )
    except (OSError, ValueError, RuntimeError) as error:
        if arguments.json:
            json.dump(
                {
                    "schema": "rasim-osc-geometry-fit-rejection-v1",
                    "qualification": {"accepted": False},
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
                sys.stdout,
                indent=2,
                sort_keys=True,
            )
            sys.stdout.write("\n")
        else:
            print(f"geometry fit rejected: {error}", file=sys.stderr)
        return 1
    if arguments.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        succeeded = (
            payload["qualification"]["accepted"]
            if payload["qualification"]["requested"]
            else payload["run_completed"]
        )
        return 0 if succeeded else 1
    print(f"manifest {payload['indexed_manifest_hash']}")
    print(f"sites {payload['image_site_counts']}")
    print(f"baseline_rms_px={payload['baseline']['site_rms_px']:.6g}")
    print(
        f"fitted_rms_px={payload['post_fit']['site_rms_px']:.6g} "
        f"fitted_max_px={payload['post_fit']['site_max_px']:.6g} "
        f"chord_rms_rad={payload['post_fit']['chord_angle_rms_rad']:.6g}"
    )
    print(f"corrections={payload['fit']['corrections']}")
    print(
        f"rank={payload['fit']['jacobian_rank']} "
        f"condition={payload['fit']['jacobian_condition']:.6g}"
    )
    print(f"active_bounds={payload['fit']['active_bounds']}")
    print(f"scaled_singular_values={payload['fit']['scaled_jacobian_singular_values']}")
    print(f"weakest_scaled_direction={payload['fit']['scaled_jacobian_weakest_direction']}")
    baseline_by_id = {item["image_id"]: item for item in payload["baseline"]["per_image"]}
    for item in payload["post_fit"]["per_image"]:
        before = baseline_by_id[item["image_id"]]
        print(
            f"{item['image_id']}: rms_px={before['site_rms_px']:.6g}"
            f"->{item['site_rms_px']:.6g} max_px={item['site_max_px']:.6g}"
        )
    print(
        "maximum_multistart_prediction_separation_px="
        f"{payload['qualification']['maximum_multistart_prediction_separation_px']:.6g}"
    )
    print(
        f"selected_start_index={payload['selected_start_index']} "
        f"nfev={payload['fit']['optimizer_function_evaluation_count']} "
        f"njev={payload['fit']['optimizer_jacobian_evaluation_count']} "
        f"model_evaluations={payload['fit']['model_evaluation_count']}"
    )
    for index, item in enumerate(payload["multi_start"]):
        print(
            f"start[{index}]: success={item['fit']['success']} "
            f"objective_sum_squares={item['objective_sum_squares']:.6g} "
            "prediction_separation_px="
            f"{item['maximum_prediction_separation_from_selected_start_px']:.6g} "
            f"wall_seconds={item['wall_time_seconds']:.6g}"
        )
    if payload["cross_validation"] is not None:
        heldout = payload["cross_validation"]["heldout"]
        print(
            f"heldout_rms_px={heldout['site_rms_px']:.6g} "
            f"heldout_max_px={heldout['site_max_px']:.6g}"
        )
    print(
        f"outer_audit={payload['outer_audit']['classification']} "
        "frozen_reindex_manifest="
        f"{payload['outer_audit']['frozen_reindex_manifest_hash']} "
        f"track_exports={payload['outer_audit']['frozen_reindex_track_export_counts']}"
    )
    print(f"frozen_visibility={payload['outer_audit']['frozen_visibility']}")
    print(
        "global_rediscovery="
        f"{payload['outer_audit']['global_rediscovery']['classification']} "
        "manifest="
        f"{payload['outer_audit']['global_rediscovery']['manifest_hash']} "
        "track_exports="
        f"{payload['outer_audit']['global_rediscovery']['track_export_counts']}"
    )
    if payload["benchmark"]["warm_residual_median_seconds"] is not None:
        print(
            "warm_residual_median_seconds="
            f"{payload['benchmark']['warm_residual_median_seconds']:.6g} "
            "fit_peak_memory_bytes="
            f"{payload['benchmark']['fit_peak_memory_bytes']}"
        )
    print(f"primary_fit_wall_time_seconds={payload['primary_fit_wall_time_seconds']:.6g}")
    print(f"indexing_pass_wall_times_seconds={payload['indexing_pass_wall_times_seconds']}")
    print(f"root_audit={payload['root_audit']}")
    print(
        f"run_completed={payload['run_completed']} "
        f"qualification_accepted={payload['qualification']['accepted']}"
    )
    succeeded = (
        payload["qualification"]["accepted"]
        if payload["qualification"]["requested"]
        else payload["run_completed"]
    )
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
