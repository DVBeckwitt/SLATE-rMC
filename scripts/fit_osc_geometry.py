"""Fit one shared detector-native geometry correction to an indexed OSC series."""

from __future__ import annotations

import argparse
import hashlib
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
    DETECTOR_CALIBRATION_PARAMETER_NAMES,
    INCIDENCE_ANGLE_DELTA_PARAMETER_NAME,
    SHARED_GEOMETRY_PARAMETER_NAMES,
    DetectorCalibrationCorrectionBounds,
    DetectorCalibrationCorrections,
    FixedPositionState,
    IncidenceAngleDeltaBounds,
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
_BI2SE3_FITTED_SHARED_PARAMETER_NAMES = tuple(
    name for name in SHARED_GEOMETRY_PARAMETER_NAMES if name != "sample_normal_x_tilt_rad"
)
_BI2SE3_INDEXED_MANIFEST_HASH = (
    "sha256-1de21e03a801fa38390ef5280133666474bfd969377024ef6dd4fb34e40f3132"
)


def _write_external_json(destination: Path, payload: dict[str, object]) -> Path:
    path = destination.resolve()
    if path == ROOT or path.is_relative_to(ROOT):
        raise ValueError("position artifact must be written outside the repository")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


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


def _trim_by_image_id(result: object) -> dict[str, float]:
    return {
        image_id: float(value)
        for image_id, value in zip(
            result.image_ids,
            result.incidence_angle_trim_by_image_id_rad,
            strict=True,
        )
    }


def _detector_calibration_active(result: object) -> bool:
    return bool(result.fitted_detector_calibration_parameter_names) or bool(
        np.any(result.detector_calibration_corrections.as_array() != 0.0)
    )


def serialize_geometry_fit_result(result: object) -> dict[str, object]:
    """Serialize one geometry-fit result without changing the legacy null-calibration record."""
    payload = {
        "success": result.success,
        "message": result.message,
        "parameterization_id": result.parameterization_id,
        "fitted_parameter_names": result.fitted_parameter_names,
        "fixed_parameter_names": result.fixed_parameter_names,
        "jacobian_parameter_names": result.jacobian_parameter_names,
        "corrections": asdict(result.corrections),
        "incidence_angle_delta_rad": result.incidence_angle_delta_rad,
        "incidence_angle_delta_fitted": result.incidence_angle_delta_fitted,
        "incidence_angle_trim_fitted": result.incidence_angle_trim_fitted,
        "incidence_angle_trim_contrast_rad": (result.incidence_angle_trim_contrast_rad.tolist()),
        "incidence_angle_trim_by_image_id_rad": {
            image_id: float(value)
            for image_id, value in zip(
                result.image_ids,
                result.incidence_angle_trim_by_image_id_rad,
                strict=True,
            )
        },
        "incidence_angle_trim_prior_sigma_rad": result.incidence_angle_trim_prior_sigma_rad,
        "incidence_angle_trim_contrast_half_span_rad": (
            result.incidence_angle_trim_contrast_half_span_rad
        ),
        "jacobian_rank": result.jacobian_rank,
        "jacobian_condition": result.jacobian_condition,
        "posterior_jacobian_rank": result.posterior_jacobian_rank,
        "posterior_jacobian_condition": result.posterior_jacobian_condition,
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
    if _detector_calibration_active(result):
        payload.update(
            {
                "fitted_detector_calibration_parameter_names": (
                    result.fitted_detector_calibration_parameter_names
                ),
                "fixed_detector_calibration_parameter_names": (
                    result.fixed_detector_calibration_parameter_names
                ),
                "detector_calibration_corrections": asdict(result.detector_calibration_corrections),
            }
        )
    return payload


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
    incidence_angle_delta_rad: float,
    incidence_angle_trim_by_image_id_rad: dict[str, float] | None = None,
    detector_calibration_corrections: DetectorCalibrationCorrections | None = None,
) -> tuple[dict[str, object], ...]:
    payload = []
    for image in sorted(images, key=lambda item: item.image_id):
        prediction = image.predict_integer_l_tags(
            image.observations.keys,
            corrections,
            detector_calibration_corrections=detector_calibration_corrections,
            incidence_angle_delta_rad=incidence_angle_delta_rad,
            incidence_angle_trim_rad=(incidence_angle_trim_by_image_id_rad or {}).get(
                image.image_id, 0.0
            ),
        )
        entries = []
        for index, key in enumerate(image.observations.keys):
            error = prediction.coordinates_px[index] - image.observations.coordinates_px[index]
            whitened_error = image.observations.whitening_matrix_px_inv[index] @ error
            entries.append(
                {
                    "key": _key_payload(key),
                    "observed_coordinate_px": image.observations.coordinates_px[index].tolist(),
                    "predicted_coordinate_px": prediction.coordinates_px[index].tolist(),
                    "covariance_px2": image.observations.covariance_px2[index].tolist(),
                    "detector_status": str(prediction.detector_status[index]),
                    "error_column_px": float(error[0]),
                    "error_row_px": float(error[1]),
                    "error_norm_px": float(np.linalg.norm(error)),
                    "whitened_error_norm": float(np.linalg.norm(whitened_error)),
                }
            )
        payload.append({"image_id": image.image_id, "sites": tuple(entries)})
    return tuple(payload)


def _incidence_angle_payload(
    images: tuple[IndexedGeometryImage, ...],
    incidence_angle_delta_rad: float,
    incidence_angle_trim_by_image_id_rad: dict[str, float],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "image_id": image.image_id,
            "commanded_angle_rad": image.commanded_angle_rad,
            "trim_rad": incidence_angle_trim_by_image_id_rad[image.image_id],
            "effective_angle_rad": (
                image.commanded_angle_rad
                + incidence_angle_delta_rad
                + incidence_angle_trim_by_image_id_rad[image.image_id]
            ),
        }
        for image in sorted(images, key=lambda item: (item.commanded_angle_rad, item.image_id))
    )


def _deterministic_starts(
    bounds: SharedGeometryCorrectionBounds,
    fitted_parameter_names: tuple[str, ...],
    *,
    initial: SharedGeometryCorrections,
    detector_calibration_bounds: DetectorCalibrationCorrectionBounds | None,
    fitted_detector_calibration_parameter_names: tuple[str, ...],
    initial_detector_calibration: DetectorCalibrationCorrections,
    initial_incidence_angle_delta_rad: float,
    incidence_angle_delta_bounds: IncidenceAngleDeltaBounds | None,
) -> tuple[tuple[SharedGeometryCorrections, DetectorCalibrationCorrections, float], ...]:
    pattern = np.asarray((1.0, -0.8, 0.6, -0.4, 0.7, -0.5, 0.3, 0.9, -0.7))
    offset = 0.08 * bounds.half_span * pattern
    fitted = set(fitted_parameter_names)
    offset[[name not in fitted for name in SHARED_GEOMETRY_PARAMETER_NAMES]] = 0.0
    if incidence_angle_delta_bounds is None:
        positive_incidence_start = initial_incidence_angle_delta_rad
        negative_incidence_start = initial_incidence_angle_delta_rad
    else:
        incidence_offset = 0.08 * incidence_angle_delta_bounds.half_span_rad
        positive_incidence_start = incidence_offset
        negative_incidence_start = -incidence_offset
    positive = np.array(offset, copy=True)
    negative = -offset
    initial_values = initial.as_array()
    for index, name in enumerate(SHARED_GEOMETRY_PARAMETER_NAMES):
        if name not in fitted:
            positive[index] = initial_values[index]
            negative[index] = initial_values[index]
    calibration_initial = initial_detector_calibration.as_array()
    calibration_positive = np.array(calibration_initial, copy=True)
    calibration_negative = np.array(calibration_initial, copy=True)
    if detector_calibration_bounds is not None:
        calibration_pattern = np.asarray((0.5, -0.7, 0.6), dtype=np.float64)
        calibration_offset = 0.08 * detector_calibration_bounds.half_span * calibration_pattern
        fitted_calibration = set(fitted_detector_calibration_parameter_names)
        for index, name in enumerate(DETECTOR_CALIBRATION_PARAMETER_NAMES):
            if name in fitted_calibration:
                calibration_positive[index] += calibration_offset[index]
                calibration_negative[index] -= calibration_offset[index]
    return (
        (initial, initial_detector_calibration, initial_incidence_angle_delta_rad),
        (
            SharedGeometryCorrections.from_array(positive),
            DetectorCalibrationCorrections.from_array(calibration_positive),
            positive_incidence_start,
        ),
        (
            SharedGeometryCorrections.from_array(negative),
            DetectorCalibrationCorrections.from_array(calibration_negative),
            negative_incidence_start,
        ),
    )


def _maximum_prediction_separation_px(
    images: tuple[IndexedGeometryImage, ...],
    first: SharedGeometryCorrections,
    first_detector_calibration: DetectorCalibrationCorrections,
    first_incidence_angle_delta_rad: float,
    first_incidence_angle_trim_by_image_id_rad: dict[str, float],
    second: SharedGeometryCorrections,
    second_detector_calibration: DetectorCalibrationCorrections,
    second_incidence_angle_delta_rad: float,
    second_incidence_angle_trim_by_image_id_rad: dict[str, float],
) -> float:
    maximum = 0.0
    for image in images:
        left = image.predict_integer_l_tags(
            image.observations.keys,
            first,
            detector_calibration_corrections=first_detector_calibration,
            incidence_angle_delta_rad=first_incidence_angle_delta_rad,
            incidence_angle_trim_rad=first_incidence_angle_trim_by_image_id_rad[image.image_id],
        )
        right = image.predict_integer_l_tags(
            image.observations.keys,
            second,
            detector_calibration_corrections=second_detector_calibration,
            incidence_angle_delta_rad=second_incidence_angle_delta_rad,
            incidence_angle_trim_rad=second_incidence_angle_trim_by_image_id_rad[image.image_id],
        )
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
    fitted_parameter_names: tuple[str, ...] = SHARED_GEOMETRY_PARAMETER_NAMES,
    fitted_detector_calibration_parameter_names: tuple[str, ...] = (),
    detector_center_half_span_px: float = 10.0,
    detector_distance_half_span_m: float = 1.0e-2,
    fit_incidence_angle_delta: bool = False,
    incidence_angle_delta_half_span_deg: float = 0.5,
    fit_incidence_angle_trim: bool = False,
    incidence_angle_trim_contrast_half_span_deg: float = 0.5,
    incidence_angle_trim_prior_sigma_deg: float = 0.25,
    initial: SharedGeometryCorrections | None = None,
    initial_detector_calibration: DetectorCalibrationCorrections | None = None,
    initial_incidence_angle_delta_rad: float = 0.0,
) -> dict[str, object]:
    """Index once, fit frozen observations jointly, and audit without reassignment."""

    if (
        not fitted_parameter_names
        and not fitted_detector_calibration_parameter_names
        and not fit_incidence_angle_delta
        and not fit_incidence_angle_trim
    ):
        raise ValueError("at least one geometry parameter must remain fitted")
    if fit_incidence_angle_delta and "sample_normal_x_tilt_rad" in fitted_parameter_names:
        raise ValueError("fit_incidence_angle_delta requires sample_normal_x_tilt_rad to be frozen")
    center_half_span_px = float(detector_center_half_span_px)
    distance_half_span_m = float(detector_distance_half_span_m)
    if (
        not math.isfinite(center_half_span_px)
        or center_half_span_px <= 0.0
        or not math.isfinite(distance_half_span_m)
        or distance_half_span_m <= 0.0
    ):
        raise ValueError("detector calibration half-spans must be positive and finite")
    delta_half_span_deg = float(incidence_angle_delta_half_span_deg)
    if not math.isfinite(delta_half_span_deg) or delta_half_span_deg <= 0.0:
        raise ValueError("incidence_angle_delta_half_span_deg must be positive and finite")
    trim_half_span_deg = float(incidence_angle_trim_contrast_half_span_deg)
    trim_prior_sigma_deg = float(incidence_angle_trim_prior_sigma_deg)
    if fit_incidence_angle_trim and (
        not math.isfinite(trim_half_span_deg)
        or trim_half_span_deg <= 0.0
        or not math.isfinite(trim_prior_sigma_deg)
        or trim_prior_sigma_deg <= 0.0
    ):
        raise ValueError("incidence-angle trim half-span and prior sigma must be positive")
    series = load_osc_geometry_series(manifest_path)
    if series.qualification_profile not in {None, _BI2SE3_QUALIFICATION_PROFILE}:
        raise ValueError(f"unsupported qualification profile {series.qualification_profile!r}")
    zero = SharedGeometryCorrections.zero()
    initial_corrections = zero if initial is None else initial
    if not isinstance(initial_corrections, SharedGeometryCorrections):
        raise TypeError("initial must be SharedGeometryCorrections or None")
    initial_calibration = (
        DetectorCalibrationCorrections.zero()
        if initial_detector_calibration is None
        else initial_detector_calibration
    )
    if not isinstance(initial_calibration, DetectorCalibrationCorrections):
        raise TypeError(
            "initial_detector_calibration must be DetectorCalibrationCorrections or None"
        )
    fitted_calibration_names = tuple(
        name
        for name in DETECTOR_CALIBRATION_PARAMETER_NAMES
        if name in set(fitted_detector_calibration_parameter_names)
    )
    if len(fitted_calibration_names) != len(fitted_detector_calibration_parameter_names):
        raise ValueError("detector calibration parameter names are unknown or duplicated")
    if series.qualification_profile == _BI2SE3_QUALIFICATION_PROFILE and fit_incidence_angle_trim:
        raise ValueError("the Bi2Se3 qualification does not include incidence-angle trims")
    if series.qualification_profile == _BI2SE3_QUALIFICATION_PROFILE and (
        fitted_calibration_names or np.any(initial_calibration.as_array() != 0.0)
    ):
        raise ValueError("the Bi2Se3 qualification does not include detector calibration polish")
    if series.qualification_profile == _BI2SE3_QUALIFICATION_PROFILE and (
        initial_corrections.sample_normal_x_tilt_rad != 0.0
        or not fit_incidence_angle_delta
        or delta_half_span_deg != 0.5
    ):
        raise ValueError(
            "the Bi2Se3 qualification requires sample_normal_x_tilt_rad fixed at zero "
            "with one common delta using the qualified bounds/prior"
        )
    indexing = index_osc_geometry_series(series)
    bounds = SharedGeometryCorrectionBounds.rasim_multi_angle_pose()
    calibration_half_span = DetectorCalibrationCorrections(
        center_half_span_px,
        center_half_span_px,
        distance_half_span_m,
    )
    calibration_bounds = (
        DetectorCalibrationCorrectionBounds(
            lower=DetectorCalibrationCorrections.from_array(
                initial_calibration.as_array() - calibration_half_span.as_array()
            ),
            upper=DetectorCalibrationCorrections.from_array(
                initial_calibration.as_array() + calibration_half_span.as_array()
            ),
        )
        if fitted_calibration_names
        else None
    )
    incidence_bounds = (
        IncidenceAngleDeltaBounds(
            lower_rad=-math.radians(delta_half_span_deg),
            upper_rad=math.radians(delta_half_span_deg),
        )
        if fit_incidence_angle_delta
        else None
    )
    trim_half_span_rad = math.radians(trim_half_span_deg) if fit_incidence_angle_trim else None
    trim_prior_sigma_rad = math.radians(trim_prior_sigma_deg) if fit_incidence_angle_trim else None
    if indexing.indexed_images is None:
        raise RuntimeError("the initial indexing run did not retain fit-ready geometry models")
    images = indexing.indexed_images
    baseline = evaluate_indexed_geometry_series_metrics(images, zero)

    starts = _deterministic_starts(
        bounds,
        fitted_parameter_names,
        initial=initial_corrections,
        detector_calibration_bounds=calibration_bounds,
        fitted_detector_calibration_parameter_names=fitted_calibration_names,
        initial_detector_calibration=initial_calibration,
        initial_incidence_angle_delta_rad=initial_incidence_angle_delta_rad,
        incidence_angle_delta_bounds=incidence_bounds,
    )
    fit_results = []
    fit_wall_times = []
    for initial_correction, initial_detector, initial_delta in starts:
        started = perf_counter()
        fit_results.append(
            fit_indexed_geometry_series(
                images,
                initial=initial_correction,
                bounds=bounds,
                fitted_parameter_names=fitted_parameter_names,
                initial_detector_calibration_corrections=initial_detector,
                detector_calibration_correction_bounds=calibration_bounds,
                fitted_detector_calibration_parameter_names=fitted_calibration_names,
                initial_incidence_angle_delta_rad=initial_delta,
                incidence_angle_delta_bounds=incidence_bounds,
                incidence_angle_trim_contrast_half_span_rad=trim_half_span_rad,
                incidence_angle_trim_prior_sigma_rad=trim_prior_sigma_rad,
            )
        )
        fit_wall_times.append(perf_counter() - started)
    fit_objective_sums = []
    for candidate in fit_results:
        candidate_trims = _trim_by_image_id(candidate)
        residual = evaluate_indexed_geometry_series_residual(
            images,
            candidate.corrections,
            detector_calibration_corrections=(candidate.detector_calibration_corrections),
            incidence_angle_delta_rad=candidate.incidence_angle_delta_rad,
            incidence_angle_trim_by_image_id_rad=candidate_trims,
        )
        fit_objective_sums.append(float(np.dot(residual, residual)))
    selected_start_index = min(
        range(len(fit_results)),
        key=lambda index: (not fit_results[index].success, fit_objective_sums[index]),
    )
    result = fit_results[selected_start_index]
    result_trims = _trim_by_image_id(result)
    post_fit = evaluate_indexed_geometry_series_metrics(
        images,
        result.corrections,
        detector_calibration_corrections=result.detector_calibration_corrections,
        incidence_angle_delta_rad=result.incidence_angle_delta_rad,
        incidence_angle_trim_by_image_id_rad=result_trims,
    )
    multi_start = tuple(
        {
            "initial": {
                "corrections": asdict(initial_correction),
                "detector_calibration_corrections": asdict(initial_detector),
                "incidence_angle_delta_rad": initial_delta,
            },
            "fit": serialize_geometry_fit_result(candidate),
            "normalized_correction_separation_from_selected_start": float(
                np.max(
                    np.abs(candidate.corrections.as_array() - result.corrections.as_array())
                    / bounds.half_span
                )
            ),
            "absolute_incidence_angle_delta_separation_from_selected_start_rad": abs(
                candidate.incidence_angle_delta_rad - result.incidence_angle_delta_rad
            ),
            "normalized_detector_calibration_separation_from_selected_start": (
                0.0
                if calibration_bounds is None
                else float(
                    np.max(
                        np.abs(
                            candidate.detector_calibration_corrections.as_array()
                            - result.detector_calibration_corrections.as_array()
                        )
                        / calibration_bounds.half_span
                    )
                )
            ),
            "maximum_prediction_separation_from_selected_start_px": (
                _maximum_prediction_separation_px(
                    images,
                    result.corrections,
                    result.detector_calibration_corrections,
                    result.incidence_angle_delta_rad,
                    result_trims,
                    candidate.corrections,
                    candidate.detector_calibration_corrections,
                    candidate.incidence_angle_delta_rad,
                    _trim_by_image_id(candidate),
                )
            ),
            "objective_sum_squares": objective_sum,
            "wall_time_seconds": elapsed,
        }
        for (
            initial_correction,
            initial_detector,
            initial_delta,
        ), candidate, objective_sum, elapsed in zip(
            starts, fit_results, fit_objective_sums, fit_wall_times, strict=True
        )
    )

    root_audit = audit_indexed_geometry_series_roots(
        images,
        result.corrections,
        detector_calibration_corrections=result.detector_calibration_corrections,
        incidence_angle_delta_rad=result.incidence_angle_delta_rad,
        incidence_angle_trim_by_image_id_rad=result_trims,
    )
    if root_audit.classification != "SAME":
        raise RuntimeError("the independent fitted-root audit changed a frozen marker")

    indexing_wall_times = [indexing.elapsed_seconds]
    corrected = {
        image.image_id: image.corrected_instrument(
            result.corrections,
            detector_calibration_corrections=result.detector_calibration_corrections,
            incidence_angle_delta_rad=result.incidence_angle_delta_rad,
            incidence_angle_trim_rad=result_trims[image.image_id],
        )
        for image in images
    }
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
            initial=initial_corrections,
            bounds=bounds,
            fitted_parameter_names=fitted_parameter_names,
            initial_detector_calibration_corrections=initial_calibration,
            detector_calibration_correction_bounds=calibration_bounds,
            fitted_detector_calibration_parameter_names=fitted_calibration_names,
            initial_incidence_angle_delta_rad=initial_incidence_angle_delta_rad,
            incidence_angle_delta_bounds=incidence_bounds,
            incidence_angle_trim_contrast_half_span_rad=trim_half_span_rad,
            incidence_angle_trim_prior_sigma_rad=trim_prior_sigma_rad,
        )
        cross_validation_fit_succeeded = cross_fit.success
        cross_validation = {
            "heldout_integer_L": tuple(sorted(heldout_set)),
            "fit": serialize_geometry_fit_result(cross_fit),
            "heldout": _metrics_payload(
                evaluate_indexed_geometry_series_metrics(
                    heldout_images,
                    cross_fit.corrections,
                    detector_calibration_corrections=(cross_fit.detector_calibration_corrections),
                    incidence_angle_delta_rad=cross_fit.incidence_angle_delta_rad,
                    incidence_angle_trim_by_image_id_rad=_trim_by_image_id(cross_fit),
                )
            ),
            "heldout_predictions": _prediction_payload(
                heldout_images,
                cross_fit.corrections,
                cross_fit.incidence_angle_delta_rad,
                _trim_by_image_id(cross_fit),
                cross_fit.detector_calibration_corrections,
            ),
            "wall_time_seconds": perf_counter() - started,
        }

    warm_residual_median_seconds: float | None = None
    fit_peak_memory_bytes: int | None = None
    if benchmark:
        residual_times = []
        for _ in range(25):
            started = perf_counter()
            evaluate_indexed_geometry_series_residual(
                images,
                result.corrections,
                detector_calibration_corrections=(result.detector_calibration_corrections),
                incidence_angle_delta_rad=result.incidence_angle_delta_rad,
                incidence_angle_trim_by_image_id_rad=result_trims,
            )
            residual_times.append(perf_counter() - started)
        warm_residual_median_seconds = float(np.median(residual_times))
        tracemalloc.start()
        fit_indexed_geometry_series(
            images,
            initial=initial_corrections,
            bounds=bounds,
            fitted_parameter_names=fitted_parameter_names,
            initial_detector_calibration_corrections=initial_calibration,
            detector_calibration_correction_bounds=calibration_bounds,
            fitted_detector_calibration_parameter_names=fitted_calibration_names,
            initial_incidence_angle_delta_rad=initial_incidence_angle_delta_rad,
            incidence_angle_delta_bounds=incidence_bounds,
            incidence_angle_trim_contrast_half_span_rad=trim_half_span_rad,
            incidence_angle_trim_prior_sigma_rad=trim_prior_sigma_rad,
        )
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
                fit_results[left].detector_calibration_corrections,
                fit_results[left].incidence_angle_delta_rad,
                _trim_by_image_id(fit_results[left]),
                fit_results[right].corrections,
                fit_results[right].detector_calibration_corrections,
                fit_results[right].incidence_angle_delta_rad,
                _trim_by_image_id(fit_results[right]),
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
    qualification_parameterization_matches = (
        result.fitted_parameter_names == _BI2SE3_FITTED_SHARED_PARAMETER_NAMES
        and result.fixed_parameter_names == ("sample_normal_x_tilt_rad",)
        and result.corrections.sample_normal_x_tilt_rad == 0.0
        and not result.fitted_detector_calibration_parameter_names
        and np.all(result.detector_calibration_corrections.as_array() == 0.0)
        and result.incidence_angle_delta_fitted
        and incidence_bounds is not None
        and incidence_bounds.lower_rad == -math.radians(0.5)
        and incidence_bounds.upper_rad == math.radians(0.5)
        and not result.incidence_angle_trim_fitted
        and result.incidence_angle_trim_contrast_half_span_rad is None
        and result.incidence_angle_trim_prior_sigma_rad is None
        and len(result.incidence_angle_trim_contrast_rad) == 0
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
            qualification_parameterization_matches,
        )
    )
    coordinate_prediction_accepted = all(
        (
            run_completed,
            fit_metrics_pass,
            minimum_improvement_pass,
            cross_validation is not None,
            cross_validation_fit_succeeded,
            heldout_metrics_pass,
        )
    )
    parameter_precision_qualified = coordinate_prediction_accepted and no_active_bounds
    position_classification = (
        "POSITION_FIT"
        if parameter_precision_qualified
        else "POSITION_MODEL_LIMITED"
        if coordinate_prediction_accepted
        else "POSITION_UNQUALIFIED"
    )
    position_revision_payload = {
        "indexed_manifest_hash": indexing.selection.manifest_hash,
        "corrections": asdict(result.corrections),
        "incidence_angle_delta_rad": result.incidence_angle_delta_rad,
        "incidence_angle_image_ids": image_ids,
        "incidence_angle_trim_by_image_id_rad": result_trims,
    }
    calibration_active = _detector_calibration_active(result)
    if calibration_active:
        position_revision_payload.update(
            {
                "configured_detector_reference_coordinate_px": tuple(
                    float(value)
                    for value in images[0].model.inputs.instrument.detector_reference_coordinate_px
                ),
                "detector_calibration_corrections": asdict(result.detector_calibration_corrections),
            }
        )
    position_revision = (
        "sha256-"
        + hashlib.sha256(
            json.dumps(
                position_revision_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    fixed_position = FixedPositionState(
        artifact_revision=position_revision,
        corrections=result.corrections,
        incidence_angle_delta_rad=result.incidence_angle_delta_rad,
        commanded_incidence_angles_rad=tuple(image.commanded_angle_rad for image in images),
        beam_center_column_row_px=tuple(
            float(value + offset)
            for value, offset in zip(
                images[0].model.inputs.instrument.detector_reference_coordinate_px,
                result.detector_calibration_corrections.as_array()[:2],
                strict=True,
            )
        ),
        detector_calibration_active=calibration_active,
        detector_plane_normal_offset_m=(
            result.detector_calibration_corrections.detector_plane_normal_offset_m
        ),
        incidence_angle_image_ids=image_ids if result.incidence_angle_trim_fitted else (),
        incidence_angle_trim_rad=(
            tuple(result_trims[image_id] for image_id in image_ids)
            if result.incidence_angle_trim_fitted
            else ()
        ),
        incidence_angle_trim_contrast_rad=(
            tuple(float(value) for value in result.incidence_angle_trim_contrast_rad)
            if result.incidence_angle_trim_fitted
            else ()
        ),
        incidence_angle_trim_prior_sigma_rad=(
            result.incidence_angle_trim_prior_sigma_rad
            if result.incidence_angle_trim_fitted
            else None
        ),
        incidence_angle_trim_contrast_half_span_rad=(
            result.incidence_angle_trim_contrast_half_span_rad
            if result.incidence_angle_trim_fitted
            else None
        ),
    ).to_record()

    payload = {
        "schema": "rasim-osc-geometry-fit-result-v6",
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": hashlib.sha256(Path(manifest_path).resolve().read_bytes()).hexdigest(),
        "indexed_manifest_hash": indexing.selection.manifest_hash,
        "run_completed": run_completed,
        "fixed_position": fixed_position,
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
        "fit": serialize_geometry_fit_result(result),
        "predictions": _prediction_payload(
            images,
            result.corrections,
            result.incidence_angle_delta_rad,
            result_trims,
            result.detector_calibration_corrections,
        ),
        "incidence_angle_correction": {
            "model_id": "commanded_plus_common_delta_plus_zero_sum_trim.helmert.v1",
            "parameter_name": INCIDENCE_ANGLE_DELTA_PARAMETER_NAME,
            "delta_rad": result.incidence_angle_delta_rad,
            "trim_parameterization": "orthonormal_helmert_zero_sum.v1",
            "trim_contrast_rad": result.incidence_angle_trim_contrast_rad.tolist(),
            "trim_prior_sigma_rad": result.incidence_angle_trim_prior_sigma_rad,
            "trim_contrast_half_span_rad": (result.incidence_angle_trim_contrast_half_span_rad),
            "trim_sum_rad": math.fsum(result_trims.values()),
            "images": _incidence_angle_payload(
                images,
                result.incidence_angle_delta_rad,
                result_trims,
            ),
        },
        "post_fit": _metrics_payload(post_fit),
        "selected_start_index": selected_start_index,
        "primary_fit_wall_time_seconds": fit_wall_times[selected_start_index],
        "initial_start_fit_wall_time_seconds": fit_wall_times[0],
        "multi_start": multi_start,
        "pairwise_multistart_prediction_separation": (pairwise_multistart_prediction_separation),
        "root_audit": asdict(root_audit),
        "outer_audit": outer_payload,
        "benchmark": {
            "warm_residual_median_seconds": warm_residual_median_seconds,
            "fit_peak_memory_bytes": fit_peak_memory_bytes,
        },
        "assessment": {
            "classification": position_classification,
            "coordinate_prediction_accepted": coordinate_prediction_accepted,
            "dataset_qualified": accepted,
            "parameter_precision_qualified": parameter_precision_qualified,
            "heldout_validation_performed": cross_validation is not None,
        },
        "qualification": {
            "accepted": accepted,
            "requested": qualification_requested,
            "profile_id": series.qualification_profile,
            "evidence_complete": qualification_evidence_complete,
            "heldout_integer_l_matches_profile": qualification_heldout_matches,
            "indexed_manifest_matches_profile": qualification_manifest_matches,
            "parameterization_matches_profile": qualification_parameterization_matches,
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
            "parameter_precision_qualified": parameter_precision_qualified,
            "maximum_multistart_prediction_separation_px": (
                maximum_multistart_prediction_separation
            ),
            "root_audit_same": root_audit_same,
            "outer_audit_same": outer_audit_same,
        },
    }
    if calibration_active:
        payload["configured_detector_reference_coordinate_px"] = tuple(
            float(value)
            for value in images[0].model.inputs.instrument.detector_reference_coordinate_px
        )
    return payload


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
    parser.add_argument(
        "--freeze-parameter",
        action="append",
        choices=SHARED_GEOMETRY_PARAMETER_NAMES,
        default=None,
        help="shared geometry coordinate to hold at its configured/initial value; repeatable",
    )
    parser.add_argument(
        "--fit-detector-center",
        action="store_true",
        help="fit shared native detector reference column and row offsets",
    )
    parser.add_argument(
        "--detector-center-half-span-px",
        type=float,
        default=10.0,
        help="symmetric bound in pixels around the configured detector center",
    )
    parser.add_argument(
        "--fit-detector-distance",
        action="store_true",
        help=(
            "fit panel translation along its configured normal; requires freezing "
            "goniometer_pivot_yaw_offset_m"
        ),
    )
    parser.add_argument(
        "--detector-distance-half-span-mm",
        type=float,
        default=10.0,
        help="symmetric panel-normal distance bound in millimetres",
    )
    parser.add_argument(
        "--fit-incidence-angle-delta",
        action="store_true",
        help="fit one additive incidence-angle delta shared by every OSC image",
    )
    parser.add_argument(
        "--incidence-angle-delta-half-span-deg",
        type=float,
        default=0.5,
        help="symmetric bound in degrees for the one shared incidence-angle delta",
    )
    parser.add_argument(
        "--fit-incidence-angle-trim",
        action="store_true",
        help="fit zero-sum per-image incidence trims in an orthonormal Helmert basis",
    )
    parser.add_argument(
        "--incidence-angle-trim-contrast-half-span-deg",
        type=float,
        default=0.5,
        help="symmetric bound for each zero-sum Helmert contrast",
    )
    parser.add_argument(
        "--incidence-angle-trim-prior-sigma-deg",
        type=float,
        default=0.25,
        help="Gaussian prior sigma for each zero-sum Helmert contrast",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="persist the raw JSON position artifact outside the repository",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    frozen = set(arguments.freeze_parameter or ())
    fitted_parameter_names = tuple(
        name for name in SHARED_GEOMETRY_PARAMETER_NAMES if name not in frozen
    )
    fitted_detector_calibration_parameter_names = (
        (
            "detector_reference_column_offset_px",
            "detector_reference_row_offset_px",
        )
        if arguments.fit_detector_center
        else ()
    ) + (("detector_plane_normal_offset_m",) if arguments.fit_detector_distance else ())
    try:
        if (
            not fitted_parameter_names
            and not fitted_detector_calibration_parameter_names
            and not arguments.fit_incidence_angle_delta
            and not arguments.fit_incidence_angle_trim
        ):
            raise ValueError("at least one shared geometry parameter must remain fitted")
        payload = fit_osc_geometry_series(
            arguments.manifest,
            heldout_integer_l=tuple(arguments.heldout_integer_l),
            benchmark=arguments.benchmark,
            fitted_parameter_names=fitted_parameter_names,
            fitted_detector_calibration_parameter_names=(
                fitted_detector_calibration_parameter_names
            ),
            detector_center_half_span_px=arguments.detector_center_half_span_px,
            detector_distance_half_span_m=(arguments.detector_distance_half_span_mm * 1.0e-3),
            fit_incidence_angle_delta=arguments.fit_incidence_angle_delta,
            incidence_angle_delta_half_span_deg=(arguments.incidence_angle_delta_half_span_deg),
            fit_incidence_angle_trim=arguments.fit_incidence_angle_trim,
            incidence_angle_trim_contrast_half_span_deg=(
                arguments.incidence_angle_trim_contrast_half_span_deg
            ),
            incidence_angle_trim_prior_sigma_deg=(arguments.incidence_angle_trim_prior_sigma_deg),
        )
        if arguments.destination is not None:
            _write_external_json(arguments.destination, payload)
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
    if "detector_calibration_corrections" in payload["fit"]:
        print(
            f"detector_calibration_corrections={payload['fit']['detector_calibration_corrections']}"
        )
    print(f"fitted_parameters={payload['fit']['fitted_parameter_names']}")
    print(f"fixed_parameters={payload['fit']['fixed_parameter_names']}")
    incidence = payload["incidence_angle_correction"]
    print(
        "shared_incidence_angle_delta_rad="
        f"{incidence['delta_rad']:.12g} "
        f"shared_incidence_angle_delta_deg={math.degrees(incidence['delta_rad']):.12g}"
    )
    for item in incidence["images"]:
        print(
            f"{item['image_id']}: commanded_theta_i_deg="
            f"{math.degrees(item['commanded_angle_rad']):.12g} "
            "trim_theta_i_deg="
            f"{math.degrees(item['trim_rad']):.12g} "
            "effective_theta_i_deg="
            f"{math.degrees(item['effective_angle_rad']):.12g}"
        )
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
