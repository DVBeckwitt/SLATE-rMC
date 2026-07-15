"""Mosaic-profile helpers extracted from the monolithic optimization module."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ra_sim.fitting._numeric import normalize_tuple_key as _normalized_q_group_key
from ra_sim.fitting.geometry_fit_parameters import geometry_fit_pixel_size_m
from ra_sim.fitting.optimization_runtime import coerce_sequence_items as _coerce_sequence_items
from ra_sim.utils.calculations import resolve_canonical_branch


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if isinstance(value, (int, float, np.integer, np.floating, str)):
            return float(value)
        return float(str(value))
    except Exception:
        return float(default)


def _safe_int(value: object, default: int = -1) -> int:
    try:
        if isinstance(value, (int, np.integer, str)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return int(value)
        return int(str(value))
    except Exception:
        return int(default)


def _miller_key_from_row(row: Sequence[float]) -> Optional[Tuple[int, int, int]]:
    """Return an integer HKL tuple from a Miller-array row."""

    try:
        return (
            int(round(float(row[0]))),
            int(round(float(row[1]))),
            int(round(float(row[2]))),
        )
    except Exception:
        return None


def _normalized_hkl_key(
    value: object,
) -> Optional[Tuple[int, int, int]]:
    """Return one integer HKL tuple from a stored entry value."""

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (
            int(value[0]),
            int(value[1]),
            int(value[2]),
        )
    except Exception:
        return None












@dataclass
class MosaicProfileROI:
    """Prepared fixed-window angular profile extracted around one measured peak."""

    dataset_index: int
    dataset_label: str
    reflection_index: int
    hkl: Tuple[int, int, int]
    family: str
    axis_name: str
    center_row: float
    center_col: float
    row_bounds: Tuple[int, int]
    col_bounds: Tuple[int, int]
    flat_indices: np.ndarray
    axis_bin_indices: np.ndarray
    signal_mask: np.ndarray
    side_mask: np.ndarray
    signal_counts: np.ndarray
    side_counts: np.ndarray
    axis_bin_centers: np.ndarray
    axis_half_span_deg: float
    orthogonal_half_window_deg: float
    measured_two_theta_deg: float
    measured_phi_deg: float
    measured_profile: np.ndarray
    measured_area: float
    measured_shape_profile: np.ndarray


@dataclass
class MosaicProfileDatasetContext:
    """Prepared per-dataset inputs for profile-based mosaic fitting."""

    dataset_index: int
    label: str
    theta_initial: float
    experimental_image: np.ndarray
    miller: np.ndarray
    intensities: np.ndarray
    rois: List[MosaicProfileROI]
    measured_peak_count: int
    in_plane_roi_count: int
    specular_roi_count: int


def _mosaic_profile_group_key(
    entry: Mapping[str, object] | None,
    *,
    hkl: Optional[Tuple[int, int, int]] = None,
) -> Optional[Tuple[object, ...]]:
    """Return the stable in-plane grouping key used by focused mosaic fitting."""

    if isinstance(entry, Mapping):
        q_group_key = _normalized_q_group_key(entry.get("q_group_key"))
        if q_group_key is not None:
            return ("q_group",) + tuple(q_group_key)
    if hkl is None and isinstance(entry, Mapping):
        hkl = _normalized_hkl_key(entry.get("hkl", entry.get("label")))
    if hkl is None:
        return None
    return ("hkl", int(hkl[0]), int(hkl[1]), int(hkl[2]))


def _mosaic_profile_intensity_lookup(
    miller: np.ndarray,
    intensities: np.ndarray,
) -> Dict[Tuple[int, int, int], float]:
    """Return the strongest absolute intensity recorded for each HKL."""

    lookup: Dict[Tuple[int, int, int], float] = {}
    miller_arr = np.asarray(miller, dtype=np.float64)
    intensities_arr = np.asarray(intensities, dtype=np.float64).reshape(-1)
    row_count = min(
        int(miller_arr.shape[0]) if miller_arr.ndim == 2 else 0,
        int(intensities_arr.shape[0]),
    )
    for idx in range(max(row_count, 0)):
        hkl_key = _miller_key_from_row(miller_arr[idx])
        if hkl_key is None:
            continue
        try:
            intensity_value = float(abs(float(intensities_arr[idx])))
        except Exception:
            continue
        if not np.isfinite(intensity_value):
            continue
        lookup[hkl_key] = max(float(lookup.get(hkl_key, 0.0)), intensity_value)
    return lookup


def _mosaic_profile_peak_score(
    entry: Mapping[str, object] | None,
    *,
    miller: np.ndarray,
    intensities: np.ndarray,
    intensity_lookup: Mapping[Tuple[int, int, int], float],
) -> float:
    """Return the ranking score for one focused mosaic-profile peak candidate."""

    if not isinstance(entry, Mapping):
        return 0.0

    for key in ("weight", "background_intensity", "prominence_sigma", "confidence"):
        value = _safe_float(entry.get(key, np.nan))
        if np.isfinite(value) and value > 0.0:
            return float(value)

    hkl_key = _normalized_hkl_key(entry.get("hkl", entry.get("label")))
    miller_arr = np.asarray(miller, dtype=np.float64)
    intensities_arr = np.asarray(intensities, dtype=np.float64).reshape(-1)
    table_idx = _safe_int(entry.get("source_table_index"))
    if (
        hkl_key is not None
        and miller_arr.ndim == 2
        and 0 <= table_idx < miller_arr.shape[0]
        and table_idx < intensities_arr.shape[0]
    ):
        source_hkl = _miller_key_from_row(miller_arr[table_idx])
        if source_hkl == hkl_key:
            try:
                score = float(abs(float(intensities_arr[table_idx])))
            except Exception:
                score = float("nan")
            if np.isfinite(score) and score > 0.0:
                return float(score)

    if hkl_key is not None:
        try:
            score = float(intensity_lookup.get(hkl_key, 0.0))
        except Exception:
            score = 0.0
        if np.isfinite(score) and score > 0.0:
            return float(score)
    return 0.0


def _mosaic_profile_entry_priority(
    entry: Mapping[str, object] | None,
    *,
    miller: np.ndarray,
    intensities: np.ndarray,
    intensity_lookup: Mapping[Tuple[int, int, int], float],
) -> Tuple[object, ...]:
    """Return a stable ordering tuple for focused mosaic-profile candidates."""

    score = _mosaic_profile_peak_score(
        entry,
        miller=miller,
        intensities=intensities,
        intensity_lookup=intensity_lookup,
    )
    source_branch_index = math.inf
    if isinstance(entry, Mapping):
        branch_idx, _branch_source, _branch_reason = resolve_canonical_branch(
            entry,
        )
        if branch_idx in {0, 1}:
            source_branch_index = int(branch_idx)
    source_row_index = (
        _safe_int(entry.get("source_row_index")) if isinstance(entry, Mapping) else math.inf
    )
    hkl_key = (
        _normalized_hkl_key(entry.get("hkl", entry.get("label")))
        if isinstance(entry, Mapping)
        else None
    )
    group_key = (
        _normalized_q_group_key(entry.get("q_group_key")) if isinstance(entry, Mapping) else None
    )
    return (
        -float(score),
        int(source_branch_index) if np.isfinite(source_branch_index) else math.inf,
        int(source_row_index) if np.isfinite(source_row_index) else math.inf,
        str(group_key) if group_key is not None else "",
        str(hkl_key) if hkl_key is not None else "",
        str(entry.get("label", "")) if isinstance(entry, Mapping) else "",
    )


def focus_mosaic_profile_dataset_specs(
    dataset_specs: Sequence[Dict[str, object]],
    *,
    source_miller: np.ndarray,
    source_intensities: np.ndarray,
    reference_dataset_index: object = None,
    max_in_plane_groups: int = 3,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Reduce mosaic-fit datasets to all 00L peaks plus the top in-plane groups."""

    dataset_entries = [
        dict(spec) for spec in _coerce_sequence_items(dataset_specs) if isinstance(spec, Mapping)
    ]
    max_groups = max(int(max_in_plane_groups), 0)
    input_peak_count_by_dataset: Dict[str, int] = {}
    selected_peak_count_by_dataset: Dict[str, int] = {}
    summary: Dict[str, object] = {
        "mode": "focused",
        "max_in_plane_groups": int(max_groups),
        "reference_dataset_index": None,
        "reference_dataset_label": None,
        "selected_specular_hkls": [],
        "selected_in_plane_hkls": [],
        "selected_in_plane_groups": [],
        "input_peak_count_by_dataset": input_peak_count_by_dataset,
        "selected_peak_count_by_dataset": selected_peak_count_by_dataset,
    }
    if not dataset_entries:
        return [], summary

    source_miller_arr = np.asarray(source_miller, dtype=np.float64)
    source_intensities_arr = np.asarray(source_intensities, dtype=np.float64)
    intensity_lookup = _mosaic_profile_intensity_lookup(
        source_miller_arr,
        source_intensities_arr,
    )

    reference_pos = 0
    try:
        reference_idx = _safe_int(reference_dataset_index)
    except Exception:
        reference_idx = None
    if reference_idx is not None:
        for idx, spec in enumerate(dataset_entries):
            try:
                if int(spec.get("dataset_index", idx)) == reference_idx:
                    reference_pos = idx
                    break
            except Exception:
                continue

    reference_spec = dict(dataset_entries[reference_pos])
    summary["reference_dataset_index"] = reference_spec.get(
        "dataset_index",
        reference_pos,
    )
    summary["reference_dataset_label"] = reference_spec.get(
        "label",
        f"dataset_{reference_pos}",
    )

    specular_order: List[Tuple[int, int, int]] = []
    seen_specular: Set[Tuple[int, int, int]] = set()
    in_plane_groups: Dict[Tuple[object, ...], Dict[str, object]] = {}

    for raw_entry in _coerce_sequence_items(reference_spec.get("measured_peaks")):
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        hkl_key = _normalized_hkl_key(entry.get("hkl", entry.get("label")))
        family_info = _classify_mosaic_profile_family(hkl_key) if hkl_key is not None else None
        if family_info is None or hkl_key is None:
            continue
        family, _ = family_info
        if family == "specular":
            if hkl_key not in seen_specular:
                seen_specular.add(hkl_key)
                specular_order.append(hkl_key)
            continue

        group_key = _mosaic_profile_group_key(entry, hkl=hkl_key)
        if group_key is None:
            continue
        group_info = in_plane_groups.setdefault(
            group_key,
            {"entries": [], "score": 0.0},
        )
        group_info["entries"].append(dict(entry))
        group_info["score"] = _safe_float(
            group_info.get("score", 0.0), 0.0
        ) + _mosaic_profile_peak_score(
            entry,
            miller=source_miller_arr,
            intensities=source_intensities_arr,
            intensity_lookup=intensity_lookup,
        )

    ranked_group_items = sorted(
        in_plane_groups.items(),
        key=lambda item: (
            -_safe_float(item[1].get("score", 0.0), 0.0),
            min(
                _mosaic_profile_entry_priority(
                    entry,
                    miller=source_miller_arr,
                    intensities=source_intensities_arr,
                    intensity_lookup=intensity_lookup,
                )
                for entry in item[1].get("entries", [])
                if isinstance(entry, Mapping)
            )
            if item[1].get("entries")
            else (math.inf,),
        ),
    )

    selected_in_plane_group_order: List[Tuple[object, ...]] = []
    selected_in_plane_hkls: List[Tuple[int, int, int]] = []
    for group_key, group_info in ranked_group_items[:max_groups]:
        entries = [
            dict(entry) for entry in group_info.get("entries", []) if isinstance(entry, Mapping)
        ]
        if not entries:
            continue
        representative = min(
            entries,
            key=lambda entry: _mosaic_profile_entry_priority(
                entry,
                miller=source_miller_arr,
                intensities=source_intensities_arr,
                intensity_lookup=intensity_lookup,
            ),
        )
        representative_hkl = _normalized_hkl_key(
            representative.get("hkl", representative.get("label"))
        )
        if representative_hkl is None:
            continue
        selected_in_plane_group_order.append(group_key)
        selected_in_plane_hkls.append(representative_hkl)

    allowed_specular = set(specular_order)
    allowed_in_plane_groups = set(selected_in_plane_group_order)
    summary["selected_specular_hkls"] = [[int(v) for v in hkl] for hkl in specular_order]
    summary["selected_in_plane_hkls"] = [[int(v) for v in hkl] for hkl in selected_in_plane_hkls]
    summary["selected_in_plane_groups"] = [
        list(group_key) for group_key in selected_in_plane_group_order
    ]

    focused_specs: List[Dict[str, object]] = []
    for spec_idx, raw_spec in enumerate(dataset_entries):
        spec = dict(raw_spec)
        dataset_key = str(spec.get("dataset_index", spec_idx))
        raw_measured = _coerce_sequence_items(spec.get("measured_peaks"))
        input_peak_count_by_dataset[dataset_key] = int(len(raw_measured))

        best_specular_entries: Dict[Tuple[int, int, int], Dict[str, object]] = {}
        best_in_plane_entries: Dict[Tuple[object, ...], Dict[str, object]] = {}
        for raw_entry in raw_measured:
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            hkl_key = _normalized_hkl_key(entry.get("hkl", entry.get("label")))
            family_info = _classify_mosaic_profile_family(hkl_key) if hkl_key is not None else None
            if family_info is None or hkl_key is None:
                continue
            family, _ = family_info
            priority = _mosaic_profile_entry_priority(
                entry,
                miller=source_miller_arr,
                intensities=source_intensities_arr,
                intensity_lookup=intensity_lookup,
            )
            if family == "specular":
                if hkl_key not in allowed_specular:
                    continue
                previous = best_specular_entries.get(hkl_key)
                if previous is None or priority < _mosaic_profile_entry_priority(
                    previous,
                    miller=source_miller_arr,
                    intensities=source_intensities_arr,
                    intensity_lookup=intensity_lookup,
                ):
                    best_specular_entries[hkl_key] = dict(entry)
                continue

            group_key = _mosaic_profile_group_key(entry, hkl=hkl_key)
            if group_key is None or group_key not in allowed_in_plane_groups:
                continue
            previous = best_in_plane_entries.get(group_key)
            if previous is None or priority < _mosaic_profile_entry_priority(
                previous,
                miller=source_miller_arr,
                intensities=source_intensities_arr,
                intensity_lookup=intensity_lookup,
            ):
                best_in_plane_entries[group_key] = dict(entry)

        focused_measured: List[Dict[str, object]] = []
        for hkl_key in specular_order:
            selected_entry = best_specular_entries.get(hkl_key)
            if selected_entry is not None:
                focused_measured.append(dict(selected_entry))
        for group_key in selected_in_plane_group_order:
            selected_entry = best_in_plane_entries.get(group_key)
            if selected_entry is not None:
                focused_measured.append(dict(selected_entry))

        spec["measured_peaks"] = focused_measured
        selected_peak_count_by_dataset[dataset_key] = int(len(focused_measured))
        focused_specs.append(spec)

    return focused_specs, summary


def _angular_difference_array_deg(
    values: Sequence[float] | np.ndarray,
    reference: float,
) -> np.ndarray:
    """Return wrapped angular deltas relative to one reference angle."""

    arr = np.asarray(values, dtype=np.float64)
    return np.mod(arr - float(reference) + 180.0, 360.0) - 180.0


def _classify_mosaic_profile_family(
    hkl: Tuple[int, int, int],
) -> Optional[Tuple[str, str]]:
    """Classify one reflection into the supported mosaic-profile families."""

    h, k, _ = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
    if h == 0 and k == 0:
        return "specular", "two_theta"
    if h != 0 or k != 0:
        return "in_plane", "phi"
    return None


def _profile_half_span_deg(
    values: Sequence[float] | np.ndarray,
    *,
    min_half_span: float,
    max_half_span: float,
) -> float:
    """Choose one robust symmetric half-span for a fixed profile window."""

    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(min_half_span)
    try:
        span = float(np.percentile(finite, 98.0))
    except Exception:
        span = float(np.max(finite))
    if not np.isfinite(span) or span <= 0.0:
        span = float(np.max(finite))
    return float(np.clip(span, float(min_half_span), float(max_half_span)))


def _profile_bin_indices(
    values: Sequence[float] | np.ndarray,
    *,
    half_span_deg: float,
    bin_count: int,
) -> np.ndarray:
    """Return fixed symmetric profile-bin indices for one angular axis."""

    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, -1, dtype=np.int64)
    if int(bin_count) <= 0 or not np.isfinite(float(half_span_deg)) or float(half_span_deg) <= 0.0:
        return out
    half_span = float(half_span_deg)
    valid = np.isfinite(arr) & (arr >= -half_span) & (arr <= half_span)
    if not np.any(valid):
        return out
    scaled = (arr[valid] + half_span) / max(2.0 * half_span, 1.0e-12) * float(bin_count)
    scaled = np.minimum(scaled, float(bin_count) - 1.0e-12)
    out[valid] = np.floor(scaled).astype(np.int64)
    return out


def _extract_profile_from_flat_image(
    flat_image: np.ndarray,
    roi: MosaicProfileROI,
) -> np.ndarray:
    """Extract one background-subtracted 1D profile from a flat image buffer."""

    values = np.asarray(flat_image[roi.flat_indices], dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    profile = np.zeros_like(roi.signal_counts, dtype=np.float64)
    if np.any(roi.signal_mask):
        profile = np.bincount(
            roi.axis_bin_indices[roi.signal_mask],
            weights=values[roi.signal_mask],
            minlength=roi.signal_counts.size,
        ).astype(np.float64, copy=False)

    if np.any(roi.side_mask):
        side_sums = np.bincount(
            roi.axis_bin_indices[roi.side_mask],
            weights=values[roi.side_mask],
            minlength=roi.side_counts.size,
        ).astype(np.float64, copy=False)
        bg_level = np.full(
            roi.side_counts.shape,
            float(np.mean(values[roi.side_mask])),
            dtype=np.float64,
        )
        valid_side = roi.side_counts > 0
        bg_level[valid_side] = side_sums[valid_side] / roi.side_counts[valid_side]
    else:
        raise ValueError("mosaic profile ROI requires side-band pixels for background subtraction")

    corrected = profile - bg_level * roi.signal_counts.astype(np.float64)
    corrected = np.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(corrected, 0.0, None)


def _build_mosaic_profile_dataset_contexts(
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, object],
    dataset_specs: Sequence[Dict[str, object]],
    *,
    roi_half_width: int,
    build_geometry_fit_dataset_contexts: Callable[..., Sequence[object]],
    miller_key_from_row: Callable[[Sequence[float]], Optional[Tuple[int, int, int]]],
    normalized_hkl_key: Callable[[object], Optional[Tuple[int, int, int]]],
    measured_detector_anchor: Callable[[Mapping[str, object]], Tuple[object, object]],
    detector_pixels_to_fit_space: Callable[..., Tuple[np.ndarray, np.ndarray]],
) -> Tuple[List[MosaicProfileDatasetContext], List[Dict[str, object]]]:
    """Prepare fixed angular-profile windows for the sigma/gamma/eta fitter."""

    contexts = build_geometry_fit_dataset_contexts(
        miller,
        intensities,
        dataset_specs,
    )

    detector_distance = _safe_float(params.get("corto_detector", 0.0), 0.0)
    pixel_size = geometry_fit_pixel_size_m(params)
    centre = params.get("center")
    image_size = int(image_size)
    roi_half_width = max(int(roi_half_width), 1)
    prepared: List[MosaicProfileDatasetContext] = []
    rejected_rois: List[Dict[str, object]] = []

    for dataset_ctx in contexts:
        experimental_image = (
            np.asarray(dataset_ctx.experimental_image, dtype=np.float64)
            if dataset_ctx.experimental_image is not None
            else None
        )
        if experimental_image is None or experimental_image.size == 0:
            rejected_rois.append(
                {
                    "dataset_index": int(dataset_ctx.dataset_index),
                    "dataset_label": str(dataset_ctx.label),
                    "stage": "dataset_prep",
                    "reason": "missing_experimental_image",
                }
            )
            continue

        subset_miller = np.asarray(dataset_ctx.subset.miller, dtype=np.float64)
        subset_intensities = np.asarray(dataset_ctx.subset.intensities, dtype=np.float64)
        measured_entries = list(dataset_ctx.subset.measured_entries or [])
        hkl_to_reflection_index = {
            miller_key_from_row(row): int(idx)
            for idx, row in enumerate(subset_miller)
            if miller_key_from_row(row) is not None
        }
        rois: List[MosaicProfileROI] = []
        in_plane_count = 0
        specular_count = 0
        flat_experimental = np.asarray(experimental_image, dtype=np.float64).ravel()

        for measured_entry in measured_entries:
            if not isinstance(measured_entry, Mapping):
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "stage": "profile_prep",
                        "reason": "invalid_measured_entry",
                    }
                )
                continue

            hkl_key = normalized_hkl_key(measured_entry.get("hkl"))
            if hkl_key is None:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "stage": "profile_prep",
                        "reason": "missing_hkl",
                    }
                )
                continue

            family_info = _classify_mosaic_profile_family(hkl_key)
            if family_info is None:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "unsupported_peak_family",
                    }
                )
                continue
            family, axis_name = family_info

            reflection_index = hkl_to_reflection_index.get(hkl_key)
            if reflection_index is None:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "hkl_not_in_subset",
                    }
                )
                continue

            anchor, anchor_reason = measured_detector_anchor(measured_entry)
            if anchor is None:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": anchor_reason,
                    }
                )
                continue

            if not isinstance(anchor, (list, tuple, np.ndarray)) or len(anchor) < 2:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "invalid_anchor_shape",
                    }
                )
                continue

            center_col = _safe_float(anchor[0], 0.0)
            center_row = _safe_float(anchor[1], 0.0)
            row0 = max(int(math.floor(center_row)) - roi_half_width, 0)
            row1 = min(int(math.floor(center_row)) + roi_half_width + 1, image_size)
            col0 = max(int(math.floor(center_col)) - roi_half_width, 0)
            col1 = min(int(math.floor(center_col)) + roi_half_width + 1, image_size)
            if row1 <= row0 or col1 <= col0:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "degenerate_roi",
                    }
                )
                continue

            rows = np.arange(row0, row1, dtype=np.int64)
            cols = np.arange(col0, col1, dtype=np.int64)
            yy, xx = np.meshgrid(rows, cols, indexing="ij")
            flat_indices = (yy * image_size + xx).reshape(-1)
            two_theta_deg, phi_deg = detector_pixels_to_fit_space(
                xx.reshape(-1),
                yy.reshape(-1),
                center=centre,
                detector_distance=detector_distance,
                pixel_size=pixel_size,
            )
            peak_two_theta, peak_phi = detector_pixels_to_fit_space(
                np.asarray([center_col], dtype=np.float64),
                np.asarray([center_row], dtype=np.float64),
                center=centre,
                detector_distance=detector_distance,
                pixel_size=pixel_size,
            )
            if (
                peak_two_theta.size == 0
                or peak_phi.size == 0
                or not np.isfinite(float(peak_two_theta[0]))
                or not np.isfinite(float(peak_phi[0]))
            ):
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "invalid_peak_angles",
                    }
                )
                continue

            measured_two_theta_deg = float(peak_two_theta[0])
            measured_phi_deg = float(peak_phi[0])
            if family == "in_plane":
                axis_delta = _angular_difference_array_deg(phi_deg, measured_phi_deg)
                orth_delta = two_theta_deg - measured_two_theta_deg
                axis_half_span_deg = _profile_half_span_deg(
                    axis_delta,
                    min_half_span=0.35,
                    max_half_span=20.0,
                )
                orth_half_span_deg = _profile_half_span_deg(
                    orth_delta,
                    min_half_span=0.03,
                    max_half_span=5.0,
                )
                orthogonal_half_window_deg = float(
                    np.clip(
                        0.35 * orth_half_span_deg,
                        0.04,
                        max(orth_half_span_deg, 0.15),
                    )
                )
            else:
                axis_delta = two_theta_deg - measured_two_theta_deg
                orth_delta = _angular_difference_array_deg(phi_deg, measured_phi_deg)
                axis_half_span_deg = _profile_half_span_deg(
                    axis_delta,
                    min_half_span=0.15,
                    max_half_span=15.0,
                )
                orth_half_span_deg = _profile_half_span_deg(
                    orth_delta,
                    min_half_span=0.15,
                    max_half_span=15.0,
                )
                orthogonal_half_window_deg = float(
                    np.clip(
                        0.35 * orth_half_span_deg,
                        0.15,
                        max(orth_half_span_deg, 0.75),
                    )
                )

            finite_orth = orth_delta[np.isfinite(orth_delta)]
            max_abs_orth = float(np.max(np.abs(finite_orth))) if finite_orth.size else 0.0
            if not np.isfinite(max_abs_orth) or max_abs_orth <= 0.0:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "invalid_orthogonal_span",
                    }
                )
                continue

            orthogonal_half_window_deg = min(
                float(orthogonal_half_window_deg),
                max_abs_orth,
            )
            side_inner = min(max_abs_orth, 1.5 * orthogonal_half_window_deg)
            side_outer = min(
                max_abs_orth,
                max(side_inner + 0.05, 3.0 * orthogonal_half_window_deg),
            )

            axis_bin_count = 41
            axis_bin_indices = _profile_bin_indices(
                axis_delta,
                half_span_deg=axis_half_span_deg,
                bin_count=axis_bin_count,
            )
            valid = np.isfinite(axis_delta) & np.isfinite(orth_delta) & (axis_bin_indices >= 0)
            signal_mask = valid & (np.abs(orth_delta) <= orthogonal_half_window_deg)
            side_mask = (
                valid & (np.abs(orth_delta) >= side_inner) & (np.abs(orth_delta) <= side_outer)
            )
            if int(np.count_nonzero(signal_mask)) < 5:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "insufficient_support_pixels",
                    }
                )
                continue

            signal_counts = np.bincount(
                axis_bin_indices[signal_mask],
                minlength=axis_bin_count,
            ).astype(np.float64, copy=False)
            side_counts = np.bincount(
                axis_bin_indices[side_mask],
                minlength=axis_bin_count,
            ).astype(np.float64, copy=False)
            axis_bin_centers = np.linspace(
                -axis_half_span_deg,
                axis_half_span_deg,
                axis_bin_count,
                dtype=np.float64,
            )

            roi = MosaicProfileROI(
                dataset_index=int(dataset_ctx.dataset_index),
                dataset_label=str(dataset_ctx.label),
                reflection_index=int(reflection_index),
                hkl=hkl_key,
                family=str(family),
                axis_name=str(axis_name),
                center_row=float(center_row),
                center_col=float(center_col),
                row_bounds=(int(row0), int(row1)),
                col_bounds=(int(col0), int(col1)),
                flat_indices=np.asarray(flat_indices, dtype=np.int64),
                axis_bin_indices=np.asarray(axis_bin_indices, dtype=np.int64),
                signal_mask=np.asarray(signal_mask, dtype=bool),
                side_mask=np.asarray(side_mask, dtype=bool),
                signal_counts=np.asarray(signal_counts, dtype=np.float64),
                side_counts=np.asarray(side_counts, dtype=np.float64),
                axis_bin_centers=np.asarray(axis_bin_centers, dtype=np.float64),
                axis_half_span_deg=float(axis_half_span_deg),
                orthogonal_half_window_deg=float(orthogonal_half_window_deg),
                measured_two_theta_deg=float(measured_two_theta_deg),
                measured_phi_deg=float(measured_phi_deg),
                measured_profile=np.zeros(axis_bin_count, dtype=np.float64),
                measured_area=0.0,
                measured_shape_profile=np.zeros(axis_bin_count, dtype=np.float64),
            )
            measured_profile = _extract_profile_from_flat_image(flat_experimental, roi)
            measured_area = float(np.sum(measured_profile))
            if not np.isfinite(measured_area) or measured_area <= 0.0:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "empty_measured_profile",
                    }
                )
                continue

            roi.measured_profile = np.asarray(measured_profile, dtype=np.float64)
            roi.measured_area = float(measured_area)
            roi.measured_shape_profile = (
                np.asarray(measured_profile, dtype=np.float64) / measured_area
            )
            rois.append(roi)
            if family == "in_plane":
                in_plane_count += 1
            else:
                specular_count += 1

        selected_reflection_indices = sorted({int(roi.reflection_index) for roi in rois})
        if selected_reflection_indices:
            selected_miller = np.asarray(
                subset_miller[selected_reflection_indices],
                dtype=np.float64,
            )
            selected_intensities = np.asarray(
                subset_intensities[selected_reflection_indices],
                dtype=np.float64,
            )
        else:
            selected_miller = np.empty((0, 3), dtype=np.float64)
            selected_intensities = np.empty((0,), dtype=np.float64)

        prepared.append(
            MosaicProfileDatasetContext(
                dataset_index=int(dataset_ctx.dataset_index),
                label=str(dataset_ctx.label),
                theta_initial=float(dataset_ctx.theta_initial),
                experimental_image=np.asarray(experimental_image, dtype=np.float64),
                miller=np.asarray(selected_miller, dtype=np.float64),
                intensities=np.asarray(selected_intensities, dtype=np.float64),
                rois=rois,
                measured_peak_count=int(len(measured_entries)),
                in_plane_roi_count=int(in_plane_count),
                specular_roi_count=int(specular_count),
            )
        )

    return prepared, rejected_rois

