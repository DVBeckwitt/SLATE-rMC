"""Workflow helpers for the geometry Q-group selector window."""

from __future__ import annotations

import json
import math
import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from ra_sim.fitting._numeric import (
    coerce_nonnegative_int as _nonnegative_identity_index,
    finite_field_pair as _finite_field_pair,
    finite_pair_from_sequence as _pair_from_sequence,
    positive_shape_2d as _positive_shape,
    safe_int as _coerce_int,
)
from ra_sim.gui.runtime_values import coerce_float as _coerce_float
from ra_sim.gui.geometry_fit_source_rows import GeometryFitSourceRowsHitTables
from ra_sim.simulation.intersection_cache_schema import (
    HIT_ROW_COL_DETECTOR_COL,
    HIT_ROW_COL_DETECTOR_ROW,
    HIT_ROW_COL_H,
    HIT_ROW_COL_INTENSITY,
    HIT_ROW_COL_K,
    HIT_ROW_COL_L,
    HIT_ROW_COL_PHI,
    HIT_ROW_COL_SOURCE_ROW_INDEX,
    HIT_ROW_COL_SOURCE_TABLE_INDEX,
    HIT_ROW_WITH_CONTEXT_WIDTH,
    HIT_ROW_WITH_PROVENANCE_WIDTH,
    extract_hit_row_provenance,
)
from ra_sim.utils import wrap_degrees as _wrap_caked_phi_deg
from ra_sim.utils.calculations import (
    SOURCE_BRANCH_PHI_ZERO_DEADBAND_RAD,
    resolve_canonical_branch,
    source_branch_index_from_phi_rad,
)
from ra_sim.utils.pbi2_ht_shift_cif import (
    DISORDERED_PHASE_DISPLAY_LABEL,
    DISORDERED_PHASE_SOURCE_LABEL,
)

from . import controllers as gui_controllers
from . import manual_geometry as gui_manual_geometry
from . import mosaic_top_selection as gui_mosaic_top
from . import geometry_overlay as gui_geometry_overlay
from . import overlays as gui_overlays
from . import views as gui_views
from .runtime_values import call_status_text as _set_status_text
from .runtime_values import resolve_runtime_value as _resolve_runtime_value


_GEOMETRY_FIT_CENTROID_RAY_DENSITY = 3
_GEOMETRY_FIT_CENTROID_EVENTS_PER_BEAM_PHASE = 30


def copy_geometry_fit_hit_tables(
    hit_tables: Sequence[object] | None,
) -> list[np.ndarray]:
    copied: list[np.ndarray] = []
    if not isinstance(hit_tables, Sequence) or isinstance(hit_tables, (str, bytes)):
        return copied
    for table in hit_tables:
        arr = np.asarray(table, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("Hit tables must be two-dimensional float arrays.")
        if arr.shape[1] not in {HIT_ROW_WITH_PROVENANCE_WIDTH, HIT_ROW_WITH_CONTEXT_WIDTH}:
            raise ValueError("Hit tables require the current 10- or 15-column layout.")
        copied.append(arr.copy())

    centroid_hit_tables = getattr(hit_tables, "centroid_hit_tables", None)
    if centroid_hit_tables is None:
        return copied
    return GeometryFitSourceRowsHitTables(
        copied,
        centroid_hit_tables=copy_geometry_fit_hit_tables(centroid_hit_tables),
    )


def filter_geometry_fit_hit_tables_for_required_branch_groups(
    hit_tables: Sequence[object] | None,
    *,
    required_branch_group_keys: Sequence[
        tuple[tuple[int, int, int], int | None, object | None]
    ]
    | None,
) -> list[np.ndarray]:
    """Keep current hit rows matching required HKL/Q-group branch identities."""

    table_list = copy_geometry_fit_hit_tables(hit_tables)
    required_keys: list[tuple[tuple[int, int, int], int | None, tuple[object, ...] | None]] = []
    for raw_key in required_branch_group_keys or ():
        if not isinstance(raw_key, (tuple, list)) or len(raw_key) != 3:
            raise ValueError("Required branch-group keys must be canonical three-item tuples.")
        hkl_raw, branch_raw, q_group_raw = raw_key
        if not isinstance(hkl_raw, (tuple, list)) or len(hkl_raw) != 3:
            raise ValueError("Required branch-group keys require a three-integer HKL.")
        hkl = tuple(int(value) for value in hkl_raw)
        branch = None if branch_raw is None else int(branch_raw)
        if branch not in {None, 0, 1}:
            raise ValueError("Required source_branch_index must be 0, 1, or null.")
        q_group_key = None if q_group_raw is None else tuple(q_group_raw)
        required_keys.append((hkl, branch, q_group_key))

    if not table_list or not required_keys:
        return table_list

    required_group_identities = {
        q_group_key for _hkl, _branch, q_group_key in required_keys if q_group_key is not None
    }

    def _matching_required_keys(
        hkl: tuple[int, int, int],
    ) -> list[tuple[tuple[int, int, int], int | None, tuple[object, ...] | None]]:
        exact_matches = [key for key in required_keys if key[0] == hkl]
        if exact_matches:
            return exact_matches
        q_group_key = _primary_reflection_q_group_identity(hkl)
        if q_group_key is None or q_group_key not in required_group_identities:
            return []
        return [key for key in required_keys if key[2] == q_group_key]

    def _filter_tables(tables: Sequence[object]) -> list[np.ndarray]:
        filtered_tables: list[np.ndarray] = []
        for table in tables:
            arr = np.asarray(table, dtype=np.float64)
            row_mask = np.zeros(arr.shape[0], dtype=bool)
            for row_index, row in enumerate(arr):
                hkl = (
                    int(np.rint(float(row[HIT_ROW_COL_H]))),
                    int(np.rint(float(row[HIT_ROW_COL_K]))),
                    int(np.rint(float(row[HIT_ROW_COL_L]))),
                )
                matches = _matching_required_keys(hkl)
                if not matches:
                    continue
                allowed_branches = {
                    int(branch) for _hkl, branch, _q_group_key in matches if branch is not None
                }
                if allowed_branches and source_branch_index_from_phi_rad(
                    row[HIT_ROW_COL_PHI]
                ) not in allowed_branches:
                    continue
                row_mask[row_index] = True
            if np.any(row_mask):
                filtered_tables.append(np.asarray(arr[row_mask], dtype=np.float64).copy())
        return filtered_tables

    filtered = _filter_tables(table_list)
    centroid_hit_tables = getattr(table_list, "centroid_hit_tables", None)
    if centroid_hit_tables is None:
        return filtered
    return GeometryFitSourceRowsHitTables(
        filtered,
        centroid_hit_tables=_filter_tables(centroid_hit_tables),
    )


@dataclass
class GeometryQGroupRuntimeBindings:
    """Runtime callbacks and shared state used by the geometry Q-group selector."""

    view_state: Any
    preview_state: Any
    q_group_state: Any
    fit_config: Mapping[str, object] | None
    current_geometry_fit_var_names_factory: object
    invalidate_geometry_manual_pick_cache: Callable[[], None]
    update_geometry_preview_exclude_button_label: Callable[[], None]
    live_geometry_preview_enabled: Callable[[], bool]
    refresh_live_geometry_preview: Callable[[], None]
    set_hkl_pick_mode: Callable[[bool], None] | None = None
    live_preview_match_key: (
        Callable[[dict[str, object] | None], tuple[object, ...] | None] | None
    ) = None
    live_preview_match_hkl: (
        Callable[[dict[str, object] | None], tuple[int, int, int] | None] | None
    ) = None
    render_live_geometry_preview_state: Callable[[], object] | None = None
    clear_geometry_preview_artists: Callable[[], None] | None = None
    preview_toggle_max_distance_px: float = 20.0
    update_running: object | None = None
    has_cached_hit_tables: object | None = None
    build_live_preview_simulated_peaks_from_cache: Callable[[], list[dict[str, object]]] | None = (
        None
    )
    filter_simulated_peaks: (
        Callable[
            [Sequence[dict[str, object]] | None],
            tuple[list[dict[str, object]], int, int],
        ]
        | None
    ) = None
    collapse_simulated_peaks: Callable[..., tuple[list[dict[str, object]], int]] | None = None
    excluded_q_group_count: Callable[[], int] | None = None
    caked_view_enabled: Callable[[], bool] | None = None
    background_visible: object | None = None
    current_background_display_factory: Callable[[], object] | None = None
    axis: object | None = None
    geometry_preview_artists: list[object] | None = None
    draw_idle: Callable[[], None] | None = None
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] | None = None
    live_preview_match_is_excluded: Callable[[dict[str, object] | None], bool] | None = None
    filter_live_preview_matches: (
        Callable[[Sequence[dict[str, object]] | None], tuple[list[dict[str, object]], int]] | None
    ) = None
    build_entries_snapshot: Callable[[], Sequence[dict[str, object]] | None] | None = None
    refresh_live_geometry_preview_quiet: Callable[[], None] | None = None
    clear_last_simulation_signature: Callable[[], None] | None = None
    schedule_update: Callable[[], None] | None = None
    set_status_text: Callable[[str], None] | None = None
    file_dialog_dir: object | None = None
    asksaveasfilename: Callable[..., object] | None = None
    askopenfilename: Callable[..., object] | None = None
    warm_detector_mode_qr_caked_cache: Callable[[], object] | None = None


@dataclass(frozen=True)
class GeometryQGroupRuntimeCallbacks:
    """Bound zero-arg callbacks for the runtime Qr/Qz selector workflow."""

    update_window_status: Callable[[Sequence[dict[str, object]] | None], None]
    refresh_window: Callable[[], bool]
    on_toggle: Callable[[tuple[object, ...] | None, object], bool]
    include_all: Callable[[], bool]
    exclude_all: Callable[[], bool]
    update_listed_peaks: Callable[[], None]
    save_selection: Callable[[], bool]
    load_selection: Callable[[], bool]
    close_window: Callable[[], None]
    open_window: Callable[[], bool]
    open_preview_exclusion_window: Callable[[], bool]
    set_preview_exclude_mode: Callable[..., bool]
    clear_preview_exclusions: Callable[[], None]
    toggle_preview_exclusion_at: Callable[[float, float], bool]
    toggle_live_preview: Callable[[], bool]
    live_preview_enabled: Callable[[], bool]
    render_live_preview_state: Callable[..., bool]


@dataclass(frozen=True)
class GeometryFitSimulationRuntimeCallbacks:
    """Bound callbacks for live geometry-fit hit-table and peak simulation."""

    simulate_hit_tables: Callable[..., tuple[list[object], dict[str, object]]]


@dataclass(frozen=True)
class GeometryQGroupRuntimeValueCallbacks:
    """Bound callbacks for live Qr/Qz selector values and peak snapshots."""

    build_live_preview_simulated_peaks_from_cache: Callable[[], list[dict[str, object]]]
    filter_simulated_peaks: Callable[
        [Sequence[dict[str, object]] | None],
        tuple[list[dict[str, object]], int, int],
    ]
    collapse_simulated_peaks: Callable[..., tuple[list[dict[str, object]], int]]
    build_entries_snapshot: Callable[[], list[dict[str, object]]]
    clone_entries: Callable[[Sequence[dict[str, object]] | None], list[dict[str, object]]]
    listed_entries: Callable[[], list[dict[str, object]]]
    listed_keys: Callable[[Sequence[dict[str, object]] | None], set[tuple[object, ...]]]
    key_from_jsonable: Callable[[object], tuple[object, ...] | None]
    export_rows: Callable[[Sequence[dict[str, object]] | None], list[dict[str, object]]]
    format_line: Callable[[dict[str, object]], str]
    current_min_matches: Callable[[], int]
    excluded_count: Callable[[Sequence[dict[str, object]] | None], int]
    build_window_status: Callable[[Sequence[dict[str, object]] | None], str]
    build_preview_exclude_button_label: Callable[
        [Sequence[dict[str, object]] | None],
        str,
    ]
    live_preview_match_key: Callable[[dict[str, object] | None], tuple[object, ...] | None]
    live_preview_match_hkl: Callable[[dict[str, object] | None], tuple[int, int, int] | None]
    live_preview_match_is_excluded: Callable[[dict[str, object] | None], bool]
    filter_live_preview_matches: Callable[
        [Sequence[dict[str, object]] | None],
        tuple[list[dict[str, object]], int],
    ]
    apply_live_preview_match_exclusions: Callable[
        [Sequence[dict[str, object]] | None, dict[str, object] | None],
        tuple[list[dict[str, object]], dict[str, object], int],
    ]
    last_live_preview_cache_metadata: Callable[[], dict[str, object]] | None = None


def _runtime_geometry_fit_var_names(
    bindings: GeometryQGroupRuntimeBindings,
) -> list[object]:
    raw_value = _resolve_runtime_value(bindings.current_geometry_fit_var_names_factory)
    if raw_value is None:
        return []
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        return list(raw_value)
    try:
        return list(raw_value)
    except Exception:
        return []


def _geometry_q_group_cache_scalar(value: object) -> object:
    if value is None or isinstance(value, (str, bytes, int, bool)):
        return value
    try:
        numeric = float(value)
    except Exception:
        return repr(value)
    if not np.isfinite(numeric):
        return repr(value)
    return round(float(numeric), 9)


def _geometry_q_group_safe_repr(value: object, *, limit: int = 160) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > int(limit):
        return text[: int(limit)] + "..."
    return text


def _geometry_q_group_signature_value(
    value: object,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> object:
    if _seen is None:
        _seen = set()
    if int(_depth) > 64:
        return ("max_depth", type(value).__name__, _geometry_q_group_safe_repr(value))
    if isinstance(value, np.ndarray):
        array_value = np.asarray(value)
        try:
            payload = hash(np.ascontiguousarray(array_value).tobytes())
        except Exception:
            payload = _geometry_q_group_safe_repr(array_value)
        return (
            "ndarray",
            tuple(int(size) for size in array_value.shape),
            str(array_value.dtype),
            payload,
        )

    try:
        is_mapping = isinstance(value, Mapping)
    except Exception as exc:
        return (
            "mapping_check_error",
            type(value).__name__,
            _geometry_q_group_safe_repr(exc),
            _geometry_q_group_cache_scalar(value),
        )
    if is_mapping:
        value_id = id(value)
        if value_id in _seen:
            return ("cycle", type(value).__name__)
        _seen.add(value_id)
        try:
            try:
                items = list(value.items())
            except Exception as exc:
                return (
                    "mapping_items_error",
                    type(value).__name__,
                    _geometry_q_group_safe_repr(exc),
                )
            return tuple(
                sorted(
                    (
                        _geometry_q_group_safe_repr(key),
                        _geometry_q_group_signature_value(item, _seen, int(_depth) + 1),
                    )
                    for key, item in items
                )
            )
        finally:
            _seen.discard(value_id)

    try:
        is_sequence = isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    except Exception as exc:
        return (
            "sequence_check_error",
            type(value).__name__,
            _geometry_q_group_safe_repr(exc),
            _geometry_q_group_cache_scalar(value),
        )
    if is_sequence:
        value_id = id(value)
        if value_id in _seen:
            return ("cycle", type(value).__name__)
        _seen.add(value_id)
        try:
            try:
                return tuple(
                    _geometry_q_group_signature_value(item, _seen, int(_depth) + 1)
                    for item in value
                )
            except Exception as exc:
                return (
                    "sequence_iter_error",
                    type(value).__name__,
                    _geometry_q_group_safe_repr(exc),
                )
        finally:
            _seen.discard(value_id)
    return _geometry_q_group_cache_scalar(value)


def _geometry_q_group_content_signature_from_hit_tables(
    hit_tables: Sequence[object] | None,
) -> object:
    if not isinstance(hit_tables, Sequence) or isinstance(hit_tables, (str, bytes)):
        return None
    table_signatures: list[tuple[object, ...]] = []
    for table_index, table in enumerate(hit_tables):
        row_signatures: list[tuple[object, ...]] = []
        for row_arr in geometry_reference_hit_rows(table):
            source_table_index, source_row_index, best_sample_index = extract_hit_row_provenance(
                row_arr
            )
            row_signatures.append(
                (
                    "hit_row",
                    _geometry_q_group_cache_scalar(row_arr[0]),
                    _geometry_q_group_cache_scalar(row_arr[1]),
                    _geometry_q_group_cache_scalar(row_arr[2]),
                    _geometry_q_group_cache_scalar(row_arr[4]),
                    _geometry_q_group_cache_scalar(row_arr[5]),
                    _geometry_q_group_cache_scalar(row_arr[6]),
                    source_table_index,
                    source_row_index,
                    best_sample_index,
                )
            )
        table_signatures.append(
            (
                "table",
                int(table_index),
                int(len(row_signatures)),
                tuple(row_signatures),
            )
        )
    return (
        "q_group_content",
        "hit_tables",
        int(len(table_signatures)),
        tuple(table_signatures),
    )


def _geometry_q_group_content_signature_from_source_rows(
    source_rows: Sequence[object] | None,
) -> object:
    if not isinstance(source_rows, Sequence) or isinstance(source_rows, (str, bytes)):
        return None
    row_signatures: list[tuple[object, ...]] = []
    for entry in source_rows:
        if not isinstance(entry, Mapping):
            continue
        hkl_value = entry.get("hkl_raw", entry.get("hkl"))
        intensity_value = entry.get("intensity", entry.get("weight"))
        row_signatures.append(
            (
                "source_row",
                _geometry_q_group_signature_value(entry.get("source_table_index")),
                _geometry_q_group_signature_value(entry.get("source_row_index")),
                _geometry_q_group_signature_value(entry.get("source_reflection_index")),
                _geometry_q_group_signature_value(str(entry.get("source_label", "primary"))),
                _geometry_q_group_signature_value(hkl_value),
                _geometry_q_group_signature_value(intensity_value),
                _geometry_q_group_signature_value(
                    entry.get(
                        "theta_initial",
                        entry.get("theta_initial_deg", entry.get("theta_i")),
                    )
                ),
                _geometry_q_group_signature_value(entry.get("native_col")),
                _geometry_q_group_signature_value(entry.get("native_row")),
            )
        )
    return (
        "q_group_content",
        "source_rows",
        int(len(row_signatures)),
        tuple(row_signatures),
    )


def _copy_simulation_diag_value(
    value: object,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> object:
    """Return one log-friendly deep copy of simulation diagnostics state."""

    max_items = 256
    max_array_items = 4096
    if _seen is None:
        _seen = set()
    if _depth > 64:
        return {"truncated": "max_depth", "type": type(value).__name__}
    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in _seen:
            return {"cycle": type(value).__name__}
        _seen.add(value_id)
        try:
            copied: dict[str, object] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= max_items:
                    copied["__truncated_items__"] = int(len(value) - max_items)
                    break
                copied[str(key)] = _copy_simulation_diag_value(item, _seen, _depth + 1)
            return copied
        finally:
            _seen.discard(value_id)
    if isinstance(value, np.ndarray):
        value_id = id(value)
        if value_id in _seen:
            return {"cycle": type(value).__name__}
        _seen.add(value_id)
        try:
            if int(value.size) > max_array_items:
                return {
                    "truncated": "max_array_items",
                    "type": type(value).__name__,
                    "dtype": str(value.dtype),
                    "shape": [int(dim) for dim in value.shape],
                    "size": int(value.size),
                }
            return _copy_simulation_diag_value(value.tolist(), _seen, _depth + 1)
        finally:
            _seen.discard(value_id)
    if isinstance(value, (list, tuple)):
        value_id = id(value)
        if value_id in _seen:
            return {"cycle": type(value).__name__}
        _seen.add(value_id)
        try:
            copied_list = [
                _copy_simulation_diag_value(item, _seen, _depth + 1) for item in value[:max_items]
            ]
            if len(value) > max_items:
                copied_list.append(
                    {
                        "truncated": "max_items",
                        "type": type(value).__name__,
                        "size": int(len(value)),
                    }
                )
            return copied_list
        finally:
            _seen.discard(value_id)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _array_shape_list(value: object) -> list[int]:
    """Return one array-like object's shape as plain integers."""

    try:
        return [int(v) for v in np.asarray(value).shape]
    except Exception:
        return []


def _array_size(value: object) -> int | None:
    """Return one array-like object's flattened size when available."""

    if value is None:
        return None
    try:
        return int(np.asarray(value).size)
    except Exception:
        try:
            return int(len(value))  # type: ignore[arg-type]
        except Exception:
            return None


def _array_row_count(value: object) -> int | None:
    """Return one array-like object's leading-dimension count when available."""

    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        arr = None
    if arr is not None:
        if arr.ndim == 0:
            return int(arr.size)
        return int(arr.shape[0])
    try:
        return int(len(value))  # type: ignore[arg-type]
    except Exception:
        return None


def _geometry_fit_exception_diagnostics(exc: Exception) -> dict[str, object]:
    """Return one stable exception payload for simulation diagnostics."""

    exception = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    return {
        "exception_type": str(exception["type"]),
        "exception_message": str(exception["message"]),
        "exception": dict(exception),
        "exceptions": [dict(exception)],
    }


def _geometry_fit_miller_hkl_inventory(miller_array: object) -> list[dict[str, object]]:
    """Return compact HKL counts for one Miller array diagnostic."""

    try:
        arr = np.asarray(miller_array, dtype=np.float64)
    except Exception:
        return []
    if arr.ndim != 2 or arr.shape[1] < 3:
        return []
    counts: dict[tuple[int, int, int], int] = {}
    for row in arr:
        try:
            hkl = (
                int(np.rint(float(row[0]))),
                int(np.rint(float(row[1]))),
                int(np.rint(float(row[2]))),
            )
        except Exception:
            continue
        counts[hkl] = counts.get(hkl, 0) + 1
    return [
        {"hkl": tuple(hkl), "count": int(count)}
        for hkl, count in sorted(counts.items(), key=lambda item: item[0])
    ]


def _geometry_fit_hit_table_hkl_inventory(
    hit_tables: Sequence[object] | None,
) -> list[dict[str, object]]:
    """Return compact HKL counts for hit-table rows."""

    counts: dict[tuple[int, int, int], int] = {}
    for table in hit_tables or ():
        try:
            arr = np.asarray(table, dtype=np.float64)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] < 7:
            continue
        for row in arr:
            try:
                hkl = (
                    int(np.rint(float(row[4]))),
                    int(np.rint(float(row[5]))),
                    int(np.rint(float(row[6]))),
                )
            except Exception:
                continue
            counts[hkl] = counts.get(hkl, 0) + 1
    return [
        {"hkl": tuple(hkl), "count": int(count)}
        for hkl, count in sorted(counts.items(), key=lambda item: item[0])
    ]


def _geometry_fit_hit_table_hkl_branch_inventory(
    hit_tables: Sequence[object] | None,
) -> list[dict[str, object]]:
    """Return compact HKL/branch counts for hit-table rows."""

    counts: dict[tuple[tuple[int, int, int], int | None], int] = {}
    for table in hit_tables or ():
        try:
            arr = np.asarray(table, dtype=np.float64)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] < 7:
            continue
        for row in arr:
            try:
                hkl = (
                    int(np.rint(float(row[4]))),
                    int(np.rint(float(row[5]))),
                    int(np.rint(float(row[6]))),
                )
            except Exception:
                continue
            branch_idx = source_branch_index_from_phi_rad(row[3])
            branch = int(branch_idx) if branch_idx in {0, 1} else None
            counts[(hkl, branch)] = counts.get((hkl, branch), 0) + 1
    return [
        {"hkl": tuple(hkl), "branch_index": branch, "count": int(count)}
        for (hkl, branch), count in sorted(
            counts.items(),
            key=lambda item: (item[0][0], -1 if item[0][1] is None else int(item[0][1])),
        )
    ]


def _geometry_fit_hit_table_source_index_inventory(
    hit_tables: Sequence[object] | None,
) -> list[dict[str, object]]:
    """Return compact source-table provenance counts for hit-table rows."""

    counts: dict[int | None, int] = {}
    for table in hit_tables or ():
        try:
            arr = np.asarray(table, dtype=np.float64)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[0] <= 0:
            continue
        for row in arr:
            source_table_index, _source_row_index, _best_sample_index = extract_hit_row_provenance(
                row
            )
            key = int(source_table_index) if source_table_index is not None else None
            counts[key] = counts.get(key, 0) + 1
    return [
        {"source_table_index": key, "count": int(count)}
        for key, count in sorted(
            counts.items(),
            key=lambda item: -1 if item[0] is None else int(item[0]),
        )
    ]


def _geometry_fit_attach_targeted_hit_table_provenance(
    hit_tables: Sequence[object],
    source_indices: Sequence[object],
) -> list[object]:
    """Attach original Miller/source indices after targeted Miller filtering."""

    table_list = list(hit_tables)
    source_index_list = list(source_indices)
    if len(table_list) != len(source_index_list):
        raise ValueError("targeted hit tables require one source index per table")
    rebuilt: list[object] = []
    for table, source_index_raw in zip(table_list, source_index_list, strict=True):
        try:
            arr = np.asarray(table, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("targeted hit table must be a numeric matrix") from exc
        if arr.ndim != 2:
            raise ValueError("targeted hit table must be two-dimensional")
        if arr.shape[1] not in {HIT_ROW_WITH_PROVENANCE_WIDTH, HIT_ROW_WITH_CONTEXT_WIDTH}:
            raise ValueError("targeted hit tables require the current 10- or 15-column layout")
        try:
            source_index = int(source_index_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("targeted hit-table source index must be an integer") from exc
        if source_index < 0:
            raise ValueError("targeted hit-table source index must be nonnegative")
        with_provenance = np.asarray(arr, dtype=np.float64).copy()
        if arr.shape[0] > 0:
            with_provenance[:, HIT_ROW_COL_SOURCE_TABLE_INDEX] = float(source_index)
            with_provenance[:, HIT_ROW_COL_SOURCE_ROW_INDEX] = np.arange(
                arr.shape[0],
                dtype=np.float64,
            )
        rebuilt.append(with_provenance)
    return rebuilt


def _finite_spread_or_default(value: object, default: float) -> float:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return float(default)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return float(default)
    spread = float(np.std(finite))
    if math.isfinite(spread) and spread > 0.0:
        return spread
    span = float(np.max(finite) - np.min(finite))
    return span / 2.0 if math.isfinite(span) and span > 0.0 else float(default)


def _geometry_fit_mosaic_peak_sample_index(mosaic: Mapping[str, object]) -> int:
    arrays: dict[str, np.ndarray] = {}
    for key in ("beam_x_array", "beam_y_array", "theta_array", "phi_array", "wavelength_array"):
        value = mosaic.get(key)
        if value is None and key == "wavelength_array":
            value = mosaic.get("wavelength_i_array")
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size:
            arrays[key] = arr
    if not arrays:
        return 0
    size = min(arr.size for arr in arrays.values())
    if size <= 1:
        return 0
    score = np.zeros(size, dtype=np.float64)
    for arr in arrays.values():
        data = np.asarray(arr[:size], dtype=np.float64)
        center = float(np.median(data))
        spread = float(np.std(data))
        if not math.isfinite(spread) or spread <= 1.0e-15:
            spread = max(float(np.max(np.abs(data - center))), 1.0e-15)
        score += np.square((data - center) / spread)
    return int(np.argmin(score))


def build_locked_detector_native_central_mosaic_params(
    source_mosaic_params: Mapping[str, object] | None,
    params: Mapping[str, object],
) -> dict[str, object]:
    """Return one detached central carrier shared by GUI and headless fits."""

    mosaic = copy.deepcopy(dict(source_mosaic_params or {}))
    try:
        wavelength_value = float(params["lambda"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("geometry-fit central carrier requires wavelength lambda") from exc
    if not np.isfinite(wavelength_value) or wavelength_value <= 0.0:
        raise ValueError("geometry-fit central carrier wavelength must be finite and positive")
    try:
        n2_value = complex(params["n2"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("geometry-fit central carrier requires complex n2") from exc
    if not (np.isfinite(n2_value.real) and np.isfinite(n2_value.imag)):
        raise ValueError("geometry-fit central carrier n2 must be finite")
    zero = np.zeros(1, dtype=np.float64)
    mosaic["beam_x_array"] = zero.copy()
    mosaic["beam_y_array"] = zero.copy()
    mosaic["theta_array"] = zero.copy()
    mosaic["phi_array"] = zero.copy()
    mosaic["wavelength_array"] = np.asarray([wavelength_value], dtype=np.float64)
    mosaic["wavelength_i_array"] = np.asarray([wavelength_value], dtype=np.float64)
    mosaic["sample_weights"] = None

    mosaic["n2_sample_array"] = np.asarray([n2_value], dtype=np.complex128)
    mosaic["_n2_sample_array_wavelength_snapshot"] = np.asarray(
        [wavelength_value],
        dtype=np.float64,
    )
    mosaic["_sampling_signature"] = (
        "geometry_fit_frozen_central_ray_v1",
        1,
        0,
        float(wavelength_value),
    )
    mosaic["_locked_detector_native_central_ray"] = True
    return mosaic


def _geometry_fit_mosaic_is_locked_zero_central_carrier(
    mosaic: Mapping[str, object] | None,
    *,
    expected_wavelength: object,
) -> bool:
    if not isinstance(mosaic, Mapping) or not bool(
        mosaic.get("_locked_detector_native_central_ray", False)
    ):
        return False
    for key in ("beam_x_array", "beam_y_array", "theta_array", "phi_array"):
        try:
            values = np.asarray(mosaic.get(key), dtype=np.float64).reshape(-1)
        except Exception:
            return False
        if values.size != 1 or not np.isfinite(values[0]) or float(values[0]) != 0.0:
            return False
    try:
        wavelengths = np.asarray(
            mosaic.get("wavelength_array", mosaic.get("wavelength_i_array")),
            dtype=np.float64,
        ).reshape(-1)
    except Exception:
        return False
    if wavelengths.size != 1 or not np.isfinite(wavelengths[0]) or float(wavelengths[0]) <= 0.0:
        return False
    try:
        expected_wavelength_value = float(expected_wavelength)
    except Exception:
        return False
    if (
        not np.isfinite(expected_wavelength_value)
        or float(wavelengths[0]) != expected_wavelength_value
    ):
        return False
    sample_weights = mosaic.get("sample_weights")
    if sample_weights is not None:
        try:
            weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        except Exception:
            return False
        if weights.size != 1 or not np.isfinite(weights[0]) or float(weights[0]) != 1.0:
            return False
    n2_samples = mosaic.get("n2_sample_array")
    if n2_samples is not None:
        try:
            n2_values = np.asarray(n2_samples, dtype=np.complex128).reshape(-1)
        except Exception:
            return False
        if n2_values.size != 1 or not np.isfinite(n2_values[0]):
            return False
    return True


def _geometry_fit_centroid_grid_params(
    params: Mapping[str, object],
    *,
    ray_density: int,
) -> dict[str, object]:
    local = copy.deepcopy(dict(params))
    mosaic = local.get("mosaic_params")
    if not isinstance(mosaic, Mapping):
        return local
    density = max(1, int(ray_density))
    if density % 2 == 0:
        density += 1
    reduced = copy.deepcopy(dict(mosaic))
    original_count = _array_size(reduced.get("theta_array")) or 0
    original_peak_index = _geometry_fit_mosaic_peak_sample_index(reduced)

    locked_central_carrier = bool(reduced.get("_locked_detector_native_central_ray", False))
    if locked_central_carrier:
        # The all-candidate centroid route enumerates every physical candidate
        # directly. Repeating the identical frozen central carrier only
        # duplicates each row with a correspondingly divided sample weight;
        # it cannot change the weighted moment and needlessly multiplies the
        # exact solve cost.
        density = 1
        theta_spread = 0.0
        phi_spread = 0.0
        carrier_strategy = "central_ray_single_candidate_moment"
    else:
        theta_spread = _finite_spread_or_default(reduced.get("theta_array"), 0.0)
        phi_spread = _finite_spread_or_default(reduced.get("phi_array"), theta_spread)
        if theta_spread <= 0.0:
            theta_spread = phi_spread
        if phi_spread <= 0.0:
            phi_spread = theta_spread
        carrier_strategy = "beam_divergence_grid"
    axis = np.linspace(-1.0, 1.0, density, dtype=np.float64)
    theta_values: list[float] = []
    phi_values: list[float] = []
    weights: list[float] = []
    for theta_unit in axis:
        for phi_unit in axis:
            theta_values.append(float(theta_unit) * float(theta_spread))
            phi_values.append(float(phi_unit) * float(phi_spread))
            weights.append(float(math.exp(-0.5 * (float(theta_unit) ** 2 + float(phi_unit) ** 2))))
    weight_arr = np.asarray(weights, dtype=np.float64)
    if float(np.sum(weight_arr)) > 0.0:
        weight_arr = weight_arr / float(np.sum(weight_arr))
    else:
        weight_arr = np.full(density * density, 1.0 / float(density * density), dtype=np.float64)

    sample_count = int(density * density)
    try:
        nominal_wavelength = float(local.get("lambda"))
    except Exception:
        nominal_wavelength = float("nan")
    if not math.isfinite(nominal_wavelength):
        nominal_wavelength = _finite_spread_or_default(reduced.get("wavelength_array"), 1.0)

    reduced["beam_x_array"] = np.zeros(sample_count, dtype=np.float64)
    reduced["beam_y_array"] = np.zeros(sample_count, dtype=np.float64)
    reduced["theta_array"] = np.asarray(theta_values, dtype=np.float64)
    reduced["phi_array"] = np.asarray(phi_values, dtype=np.float64)
    reduced["wavelength_array"] = np.full(
        sample_count,
        float(nominal_wavelength),
        dtype=np.float64,
    )
    reduced["wavelength_i_array"] = np.full(
        sample_count,
        float(nominal_wavelength),
        dtype=np.float64,
    )
    reduced["sample_weights"] = weight_arr
    if "n2_sample_array" in reduced:
        try:
            central_n2 = np.asarray(
                reduced.get("n2_sample_array"),
                dtype=np.complex128,
            ).reshape(-1)
            n2_value = (
                central_n2[int(original_peak_index)]
                if central_n2.size
                else complex(local.get("n2", 1.0))
            )
        except Exception:
            n2_value = complex(local.get("n2", 1.0))
        reduced["n2_sample_array"] = np.full(sample_count, n2_value, dtype=np.complex128)
    reduced["centroid_ray_density"] = int(density)
    reduced["centroid_grid_sample_count"] = int(sample_count)
    reduced["centroid_grid_original_sample_count"] = int(original_count)
    reduced["centroid_grid_nominal_wavelength"] = float(nominal_wavelength)
    reduced["centroid_grid_theta_spread"] = float(theta_spread)
    reduced["centroid_grid_phi_spread"] = float(phi_spread)
    reduced["centroid_grid_carrier_strategy"] = str(carrier_strategy)
    reduced["centroid_grid_unique_carrier_count"] = (
        1 if locked_central_carrier else int(sample_count)
    )
    reduced["centroid_grid_beam_divergence_enabled"] = bool(
        not locked_central_carrier and (theta_spread > 0.0 or phi_spread > 0.0)
    )
    reduced["centroid_grid_collapsed_identical_central_carrier"] = bool(locked_central_carrier)
    reduced["_sampling_signature"] = (
        "geometry_fit_centroid_candidate_moment_v2",
        int(sample_count),
        int(density),
        float(theta_spread),
        float(phi_spread),
        float(nominal_wavelength),
    )
    local["mosaic_params"] = reduced
    return local


def _simulate_geometry_fit_density_centroid_hit_tables(
    *,
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
    image_size: int,
    params_local: Mapping[str, object],
    process_peaks_parallel: Callable[..., object],
    default_solve_q_steps: int,
    default_solve_q_rel_tol: float,
    default_solve_q_mode: int,
    source_indices: Sequence[object] | None,
) -> tuple[list[object], dict[str, object]]:
    centroid_params = _geometry_fit_centroid_grid_params(
        params_local,
        ray_density=_GEOMETRY_FIT_CENTROID_RAY_DENSITY,
    )
    mosaic = (
        centroid_params.get("mosaic_params", {})
        if isinstance(centroid_params.get("mosaic_params"), Mapping)
        else {}
    )
    wavelength_array = mosaic.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic.get("wavelength_i_array")
    if wavelength_array is None:
        wavelength_array = np.full(
            int(np.asarray(mosaic.get("theta_array"), dtype=np.float64).reshape(-1).size),
            float(centroid_params.get("lambda", 1.0)),
            dtype=np.float64,
        )

    n2_override = mosaic.get("n2_sample_array")
    if n2_override is not None:
        try:
            n2_override = np.ascontiguousarray(
                np.asarray(n2_override, dtype=np.complex128).reshape(-1),
                dtype=np.complex128,
            )
        except Exception:
            n2_override = None

    sim_buffer = np.zeros((1, 1), dtype=np.float64)
    process_kwargs = {
            "save_flag": 0,
            "solve_q_steps": int(mosaic.get("solve_q_steps", default_solve_q_steps)),
            "solve_q_rel_tol": float(mosaic.get("solve_q_rel_tol", default_solve_q_rel_tol)),
            "solve_q_mode": int(mosaic.get("solve_q_mode", default_solve_q_mode)),
            "sample_weights": mosaic.get("sample_weights"),
            "n2_sample_array_override": n2_override,
            "collect_hit_tables": True,
            "hit_table_collection_mode": "all_weighted_candidates",
            "accumulate_image": False,
            "events_per_beam_phase": int(_GEOMETRY_FIT_CENTROID_EVENTS_PER_BEAM_PHASE),
            "numba_thread_count": 1,
    }
    _image, hit_tables, *_rest = process_peaks_parallel(
        np.asarray(miller_array, dtype=np.float64),
        np.asarray(intensity_array, dtype=np.float64),
        int(image_size),
        float(centroid_params["a"]),
        float(centroid_params["c"]),
        float(centroid_params.get("lambda", 1.0)),
        sim_buffer,
        float(centroid_params["corto_detector"]),
        float(centroid_params["gamma"]),
        float(centroid_params["Gamma"]),
        float(centroid_params["chi"]),
        float(centroid_params.get("psi", 0.0)),
        float(centroid_params.get("psi_z", 0.0)),
        float(centroid_params["zs"]),
        float(centroid_params["zb"]),
        centroid_params.get("n2", 1.0),
        np.asarray(mosaic["beam_x_array"], dtype=np.float64),
        np.asarray(mosaic["beam_y_array"], dtype=np.float64),
        np.asarray(mosaic["theta_array"], dtype=np.float64),
        np.asarray(mosaic["phi_array"], dtype=np.float64),
        float(mosaic["sigma_mosaic_deg"]),
        float(mosaic["gamma_mosaic_deg"]),
        float(mosaic["eta"]),
        np.asarray(wavelength_array, dtype=np.float64),
        float(centroid_params["debye_x"]),
        float(centroid_params["debye_y"]),
        [
            float(centroid_params.get("center", (0.0, 0.0))[0]),
            float(centroid_params.get("center", (0.0, 0.0))[1]),
        ],
        float(centroid_params["theta_initial"]),
        float(centroid_params.get("cor_angle", 0.0)),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        **process_kwargs,
    )

    centroid_tables = _geometry_fit_attach_targeted_hit_table_provenance(
        list(hit_tables or ()),
        source_indices,
    )
    row_counts = [int(len(geometry_reference_hit_rows(table))) for table in centroid_tables]
    diagnostics = {
        "centroid_source_strategy": ("weighted_candidate_moments_adaptive_grid_process_peaks"),
        "centroid_ray_density": int(
            mosaic.get("centroid_ray_density", _GEOMETRY_FIT_CENTROID_RAY_DENSITY)
        ),
        "centroid_requested_ray_density": int(_GEOMETRY_FIT_CENTROID_RAY_DENSITY),
        "centroid_events_per_beam_phase": int(_GEOMETRY_FIT_CENTROID_EVENTS_PER_BEAM_PHASE),
        "centroid_grid_sample_count": int(
            mosaic.get("centroid_grid_sample_count", _GEOMETRY_FIT_CENTROID_RAY_DENSITY**2)
        ),
        "centroid_process_peaks_call_count_for_trial_prediction": 1,
        "centroid_hit_table_count": int(len(centroid_tables)),
        "centroid_finite_hit_row_total": int(sum(row_counts)),
        "centroid_grid_carrier_strategy": str(
            mosaic.get("centroid_grid_carrier_strategy", "beam_divergence_grid")
        ),
        "centroid_grid_unique_carrier_count": int(
            mosaic.get("centroid_grid_unique_carrier_count", 0) or 0
        ),
        "centroid_grid_phase_count": int(mosaic.get("centroid_grid_sample_count", 0) or 0),
        "centroid_expected_event_count": 0,
        "centroid_quantile_event_budget_ignored": True,
        "centroid_candidate_weight_mode": "continuous_projected_candidate_mass",
        "centroid_grid_beam_divergence_enabled": bool(
            mosaic.get("centroid_grid_beam_divergence_enabled", False)
        ),
        "centroid_grid_collapsed_identical_central_carrier": bool(
            mosaic.get("centroid_grid_collapsed_identical_central_carrier", False)
        ),
        "centroid_mosaic_sigma_deg": float(mosaic.get("sigma_mosaic_deg", 0.0)),
        "centroid_mosaic_gamma_deg": float(mosaic.get("gamma_mosaic_deg", 0.0)),
        "centroid_mosaic_eta": float(mosaic.get("eta", 0.0)),
        "centroid_mosaic_event_density_retained": True,
        "centroid_sampling_signature": tuple(mosaic.get("_sampling_signature", ()) or ()),
    }
    return centroid_tables, diagnostics


def geometry_reference_hit_rows(table: object) -> list[np.ndarray]:
    """Return the finite propagated hit rows recorded for one beam sample."""

    try:
        tbl_arr = np.asarray(table, dtype=object)
    except Exception:
        return []
    if tbl_arr.ndim not in (1, 2) or tbl_arr.shape[0] == 0:
        return []

    rows: list[np.ndarray] = []
    for row in list(tbl_arr):
        try:
            row_arr = np.asarray(row, dtype=float)
        except Exception:
            continue
        if row_arr.ndim != 1 or row_arr.shape[0] < 7:
            continue
        if not (
            np.isfinite(row_arr[0])
            and np.isfinite(row_arr[1])
            and np.isfinite(row_arr[2])
            and np.isfinite(row_arr[4])
            and np.isfinite(row_arr[5])
            and np.isfinite(row_arr[6])
        ):
            continue
        rows.append(np.asarray(row_arr, dtype=float))
    return rows


def geometry_q_group_m_from_hk(h_value: object, k_value: object) -> int | float | None:
    """Return the stable hexagonal radial group component ``m`` for ``H,K``."""

    try:
        h_raw = float(h_value)
        k_raw = float(k_value)
    except Exception:
        return None
    if not (np.isfinite(h_raw) and np.isfinite(k_raw)):
        return None
    m_val = h_raw * h_raw + h_raw * k_raw + k_raw * k_raw
    if not np.isfinite(m_val):
        return None
    return gui_manual_geometry.q_group_key_component(float(m_val))


def geometry_q_group_ml_from_hkl(
    hkl_value: object,
    *,
    allow_nominal_hkl_indices: bool = False,
) -> tuple[int | float, int] | None:
    """Return the stable ``(m, L)`` Qr/Qz group identity for one HKL value."""

    if not isinstance(hkl_value, (list, tuple, np.ndarray)) or len(hkl_value) < 3:
        return None
    try:
        h_raw = float(hkl_value[0])
        k_raw = float(hkl_value[1])
        l_raw = float(hkl_value[2])
    except Exception:
        return None

    if allow_nominal_hkl_indices:
        hkl_group = gui_geometry_overlay.normalize_hkl_key(hkl_value)
        if hkl_group is None:
            return None
        h_value = hkl_group[0]
        k_value = hkl_group[1]
        l_int = int(hkl_group[2])
    else:
        h_value = h_raw
        k_value = k_raw
        l_int = gui_manual_geometry.integer_gz_index(l_raw)
        if l_int is None:
            return None

    m_component = geometry_q_group_m_from_hk(h_value, k_value)
    if m_component is None:
        return None
    return m_component, int(l_int)


def geometry_q_group_ml_from_key(
    key_or_row: object,
) -> tuple[int | float, int] | None:
    """Return the stable ``(m, L)`` identity from a serialized Q-group key."""

    candidate = key_or_row
    if isinstance(key_or_row, Mapping):
        candidate = key_or_row.get("q_group_key", key_or_row.get("key"))
    if not isinstance(candidate, (list, tuple)):
        return None

    if len(candidate) >= 4 and str(candidate[0]) == "q_group":
        m_value = candidate[2]
        l_value = candidate[3]
    elif len(candidate) >= 3 and isinstance(candidate[0], str):
        m_value = candidate[1]
        l_value = candidate[2]
    else:
        return None

    try:
        m_component = gui_manual_geometry.q_group_key_component(float(m_value))
    except Exception:
        return None
    l_int = gui_manual_geometry.integer_gz_index(l_value)
    if l_int is None:
        return None
    return m_component, int(l_int)


def reflection_q_group_metadata(
    hkl_value: object,
    *,
    source_label: object = "primary",
    a_value: object = np.nan,
    c_value: object = np.nan,
    qr_value: object = np.nan,
    allow_nominal_hkl_indices: bool = False,
) -> tuple[tuple[object, ...] | None, float, float]:
    """Return stable Qr/Qz grouping metadata for one simulated reflection."""

    components = geometry_q_group_ml_from_hkl(
        hkl_value,
        allow_nominal_hkl_indices=allow_nominal_hkl_indices,
    )
    if components is None:
        return None, float("nan"), float("nan")
    m_val, l_int = components

    try:
        qr_val = float(qr_value)
    except Exception:
        qr_val = float("nan")
    try:
        a_used = float(a_value)
    except Exception:
        a_used = float("nan")
    try:
        c_used = float(c_value)
    except Exception:
        c_used = float("nan")

    if not np.isfinite(qr_val):
        if np.isfinite(a_used) and a_used > 0.0 and m_val >= 0.0:
            qr_val = (2.0 * np.pi / a_used) * np.sqrt((4.0 / 3.0) * m_val)
        else:
            qr_val = float("nan")
    qz_val = (
        (2.0 * np.pi / c_used) * float(l_int)
        if np.isfinite(c_used) and c_used > 0.0
        else float("nan")
    )
    key = (
        "q_group",
        gui_controllers.normalize_bragg_qr_source_label(
            str(source_label) if source_label is not None else "primary"
        ),
        gui_manual_geometry.q_group_key_component(m_val),
        int(l_int),
    )
    return key, float(qr_val), float(qz_val)


def _primary_reflection_q_group_identity(
    hkl: tuple[int, int, int],
) -> tuple[object, ...] | None:
    try:
        q_group_key, _qr_val, _qz_val = reflection_q_group_metadata(
            hkl,
            source_label="primary",
            allow_nominal_hkl_indices=True,
        )
    except Exception:
        return None
    if q_group_key is None:
        return None
    try:
        return tuple(q_group_key)
    except Exception:
        return None


QR_QZ_DUPLICATE_ATOL = 1.0e-6


def _qr_qz_duplicate_identity(
    qr_value: object,
    qz_value: object,
    *,
    atol: float = QR_QZ_DUPLICATE_ATOL,
) -> tuple[int, int] | None:
    try:
        qr_val = float(qr_value)
        qz_val = float(qz_value)
    except Exception:
        return None
    if not (np.isfinite(qr_val) and np.isfinite(qz_val)):
        return None
    scale = max(float(atol), 1.0e-12)
    return (int(round(qr_val / scale)), int(round(qz_val / scale)))


def q_group_source_priority(source_label: object) -> int:
    label = gui_controllers.normalize_bragg_qr_source_label(
        str(source_label) if source_label is not None else "primary"
    )
    if label == "primary":
        return 0
    if label == DISORDERED_PHASE_SOURCE_LABEL:
        return 1
    if label == "secondary":
        return 2
    return 9


def _qr_qz_duplicate_group_key(
    entry: Mapping[str, object],
    identity: tuple[int, int],
    *,
    preserve_source_identity: bool,
) -> tuple[object, ...]:
    if preserve_source_identity:
        source_label = gui_controllers.normalize_bragg_qr_source_label(
            str(entry.get("source_label")) if entry.get("source_label") is not None else "primary"
        )
        return (source_label, identity)
    return identity


def canonicalize_qr_qz_duplicate_source_rows(
    rows: Sequence[Mapping[str, object]] | None,
    *,
    atol: float = QR_QZ_DUPLICATE_ATOL,
    preserve_source_identity: bool = False,
) -> list[dict[str, object]]:
    """Remap rows sharing one numeric Qr/Qz identity onto one picker key."""

    normalized = [dict(entry) for entry in (rows or ()) if isinstance(entry, Mapping)]
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for entry in normalized:
        identity = _qr_qz_duplicate_identity(
            entry.get("qr", np.nan),
            entry.get("qz", np.nan),
            atol=atol,
        )
        if identity is not None:
            identity_key = _qr_qz_duplicate_group_key(
                entry,
                identity,
                preserve_source_identity=preserve_source_identity,
            )
            grouped.setdefault(identity_key, []).append(entry)

    canonical_by_identity: dict[tuple[object, ...], tuple[object, ...]] = {}
    for identity_key, entries in grouped.items():
        ranked = sorted(
            entries,
            key=lambda entry: (
                q_group_source_priority(entry.get("source_label")),
                str(entry.get("source_label", "")),
            ),
        )
        raw_key = ranked[0].get("q_group_key")
        if isinstance(raw_key, list):
            raw_key = tuple(raw_key)
        if isinstance(raw_key, tuple):
            canonical_by_identity[identity_key] = raw_key

    result: list[dict[str, object]] = []
    for raw_entry in normalized:
        entry = dict(raw_entry)
        identity = _qr_qz_duplicate_identity(
            entry.get("qr", np.nan),
            entry.get("qz", np.nan),
            atol=atol,
        )
        identity_key = (
            _qr_qz_duplicate_group_key(
                entry,
                identity,
                preserve_source_identity=preserve_source_identity,
            )
            if identity is not None
            else None
        )
        canonical_key = (
            canonical_by_identity.get(identity_key) if identity_key is not None else None
        )
        original_key = entry.get("q_group_key")
        if isinstance(original_key, list):
            original_key = tuple(original_key)
        if canonical_key is not None:
            if isinstance(original_key, tuple) and tuple(original_key) != tuple(canonical_key):
                entry.setdefault("source_q_group_key", tuple(original_key))
            entry["q_group_key"] = tuple(canonical_key)
        result.append(entry)
    return result


def _finite_pair_value(value: object) -> tuple[float, float] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        return _pair_from_sequence(np.asarray(value, dtype=float).reshape(-1))
    except Exception:
        return None


def _caked_phi_delta_deg(value: float, center: float) -> float:
    return _wrap_caked_phi_deg(float(value) - float(center))


def _projected_source_row_caked_point(
    entry: Mapping[str, object],
) -> tuple[float, float] | None:
    for key in (
        "selected_branch_caked_centroid_deg",
        "process_peaks_centroid_caked_deg",
        "sim_refined_caked_deg",
        "sim_visual_caked_deg",
    ):
        point = _finite_pair_value(entry.get(key))
        if point is not None:
            return point
    for x_key, y_key in (
        ("caked_x", "caked_y"),
        ("raw_caked_x", "raw_caked_y"),
        ("two_theta_deg", "phi_deg"),
        ("simulated_two_theta_deg", "simulated_phi_deg"),
    ):
        point = _finite_field_pair(entry, x_key, y_key)
        if point is not None:
            return point
    return None


def _centroid_source_row_group_key(
    entry: Mapping[str, object],
) -> tuple[object, ...] | None:
    q_group = entry.get("q_group_key")
    if isinstance(q_group, list):
        q_group_key = tuple(q_group)
    elif isinstance(q_group, tuple):
        q_group_key = tuple(q_group)
    else:
        q_group_key = geometry_q_group_key_from_entry(entry)
    if q_group_key is None:
        return None
    hkl_value = entry.get("hkl")
    if not isinstance(hkl_value, (list, tuple, np.ndarray)) or len(hkl_value) < 3:
        return None
    try:
        hkl = tuple(int(np.rint(float(value))) for value in hkl_value[:3])
    except Exception:
        return None
    _branch_idx, branch_source, _branch_reason = resolve_canonical_branch(
        entry,
    )
    if branch_source == "00l_collapsed" or (
        int(hkl[0]) == 0 and int(hkl[1]) == 0 and int(hkl[2]) > 0
    ):
        branch_token: object = "00l_collapsed"
    else:
        branch_raw = entry.get("source_branch_index")
        try:
            branch_token = int(branch_raw)
        except Exception:
            return None
        if branch_token not in {0, 1}:
            return None
    source_index = entry.get("source_reflection_index")
    try:
        source_token: object = int(source_index)
    except Exception:
        source_token = None
    return (tuple(q_group_key), hkl, branch_token, source_token)


_FROZEN_ROI_GAUSSIAN_SIGMA_RADIUS_FRACTION = 0.5
_FROZEN_ROI_GAUSSIAN_MIN_EFFECTIVE_WEIGHT_FRACTION = 1.0e-12


def _frozen_roi_gaussian_caked_centroid_from_points(
    finite_points: Sequence[Mapping[str, float]],
    *,
    candidate_count: int,
    raw_finite_count: int,
    center_2theta_deg: float | None,
    center_phi_deg: float | None,
    roi_radius_deg: float | None,
    nearest_roi_distance_deg: float,
    nearest_roi_dtheta_deg: float,
    nearest_roi_dphi_deg: float,
) -> dict[str, object]:
    weighting_mode = "frozen_roi_gaussian_local_moment"
    try:
        center_theta = float(center_2theta_deg)
        center_phi = float(center_phi_deg)
        radius = float(roi_radius_deg)
    except Exception:
        center_theta = center_phi = radius = float("nan")
    sigma = float(radius * _FROZEN_ROI_GAUSSIAN_SIGMA_RADIUS_FRACTION)
    base_diagnostics: dict[str, object] = {
        "weighting_mode": weighting_mode,
        "sigma_deg": sigma,
        "effective_weight_sum": 0.0,
        "effective_weight_fraction": 0.0,
        "minimum_effective_weight_fraction": (_FROZEN_ROI_GAUSSIAN_MIN_EFFECTIVE_WEIGHT_FRACTION),
        "candidate_count": int(candidate_count),
        "raw_finite_count": int(raw_finite_count),
        "roi_rejected_count": 0,
        "roi_center_2theta_deg": center_2theta_deg,
        "roi_center_phi_deg": center_phi_deg,
        "roi_radius_deg": roi_radius_deg,
        "nearest_roi_distance_deg": nearest_roi_distance_deg,
        "nearest_roi_dtheta_deg": nearest_roi_dtheta_deg,
        "nearest_roi_dphi_deg": nearest_roi_dphi_deg,
        "component_selection_used": False,
    }
    if not (
        np.isfinite(center_theta)
        and np.isfinite(center_phi)
        and np.isfinite(radius)
        and radius > 0.0
        and np.isfinite(sigma)
        and sigma > 0.0
    ):
        return {
            "status": "rejected",
            "reject_reason": "invalid_frozen_roi_gaussian_configuration",
            **base_diagnostics,
        }
    if not finite_points:
        return {
            "status": "rejected",
            "reject_reason": "missing_peak_points",
            **base_diagnostics,
        }

    ordered_points = sorted(
        finite_points,
        key=lambda point: (
            float(point["two_theta_deg"]),
            float(point["phi_deg"]),
            float(point["weight"]),
        ),
    )
    theta_values = np.asarray(
        [float(point["two_theta_deg"]) for point in ordered_points],
        dtype=np.float64,
    )
    phi_deltas = np.asarray(
        [_caked_phi_delta_deg(float(point["phi_deg"]), center_phi) for point in ordered_points],
        dtype=np.float64,
    )
    physical_weights = np.asarray(
        [float(point["weight"]) for point in ordered_points],
        dtype=np.float64,
    )
    theta_deltas = theta_values - center_theta
    squared_distances = np.square(theta_deltas) + np.square(phi_deltas)
    kernel_weights = np.exp(-0.5 * squared_distances / float(sigma * sigma))
    max_physical_weight = float(np.max(physical_weights))
    if not np.isfinite(max_physical_weight) or max_physical_weight <= 0.0:
        return {
            "status": "rejected",
            "reject_reason": "nonfinite_frozen_roi_effective_mass",
            **base_diagnostics,
        }
    scaled_physical_weights = physical_weights / max_physical_weight
    scaled_effective_weights = scaled_physical_weights * kernel_weights
    scaled_physical_sum = float(np.sum(scaled_physical_weights, dtype=np.float64))
    scaled_effective_sum = float(np.sum(scaled_effective_weights, dtype=np.float64))
    physical_weight_sum = float(max_physical_weight * scaled_physical_sum)
    effective_weight_sum = float(max_physical_weight * scaled_effective_sum)
    effective_weight_fraction = (
        float(scaled_effective_sum / scaled_physical_sum) if scaled_physical_sum > 0.0 else 0.0
    )
    base_diagnostics.update(
        {
            "effective_weight_sum": effective_weight_sum,
            "effective_weight_fraction": effective_weight_fraction,
            "physical_weight_sum": physical_weight_sum,
        }
    )
    if not all(
        np.isfinite(value)
        for value in (
            scaled_physical_sum,
            scaled_effective_sum,
            physical_weight_sum,
            effective_weight_sum,
            effective_weight_fraction,
        )
    ):
        return {
            "status": "rejected",
            "reject_reason": "nonfinite_frozen_roi_effective_mass",
            **base_diagnostics,
        }
    if (
        scaled_effective_sum <= 0.0
        or effective_weight_fraction < _FROZEN_ROI_GAUSSIAN_MIN_EFFECTIVE_WEIGHT_FRACTION
    ):
        return {
            "status": "rejected",
            "reject_reason": "negligible_frozen_roi_effective_mass",
            **base_diagnostics,
        }

    centroid_two_theta = float(
        np.dot(scaled_effective_weights, theta_values) / scaled_effective_sum
    )
    centroid_local_phi = float(np.dot(scaled_effective_weights, phi_deltas) / scaled_effective_sum)
    centroid_phi = _wrap_caked_phi_deg(center_phi + centroid_local_phi)
    if not (np.isfinite(centroid_two_theta) and np.isfinite(centroid_phi)):
        return {
            "status": "rejected",
            "reject_reason": "nonfinite_frozen_roi_centroid",
            **base_diagnostics,
        }
    paired_detector_native: dict[str, object] = {}
    if all(
        np.isfinite(float(point.get("detector_native_x", np.nan)))
        and np.isfinite(float(point.get("detector_native_y", np.nan)))
        for point in ordered_points
    ):
        detector_native_x = np.asarray(
            [float(point["detector_native_x"]) for point in ordered_points],
            dtype=np.float64,
        )
        detector_native_y = np.asarray(
            [float(point["detector_native_y"]) for point in ordered_points],
            dtype=np.float64,
        )
        paired_detector_native["detector_native_px"] = [
            float(np.dot(scaled_effective_weights, detector_native_x) / scaled_effective_sum),
            float(np.dot(scaled_effective_weights, detector_native_y) / scaled_effective_sum),
        ]
    return {
        "status": "ok",
        "centroid_2theta_deg": centroid_two_theta,
        "centroid_phi_deg": centroid_phi,
        "total_weight": effective_weight_sum,
        "branch_total_weight": physical_weight_sum,
        "peak_height": float(max_physical_weight * float(np.max(scaled_effective_weights))),
        "sample_count": int(len(ordered_points)),
        "component_count": 0,
        "connect_radius_deg": None,
        "centroid_points_sampled": False,
        "prediction_mode": "selected_branch_caked_centroid",
        **paired_detector_native,
        **base_diagnostics,
    }


def _selected_branch_caked_centroid_from_points(
    points: Sequence[Mapping[str, object]],
    *,
    connect_radius_deg: float | None = 0.5,
    ambiguity_fraction: float = 0.25,
    center_2theta_deg: float | None = None,
    center_phi_deg: float | None = None,
    roi_radius_deg: float | None = None,
    frozen_roi_gaussian_local_moment: bool = False,
) -> dict[str, object]:
    finite_points: list[dict[str, float]] = []
    raw_finite_count = 0
    nearest_roi_distance = float("nan")
    nearest_roi_dtheta = float("nan")
    nearest_roi_dphi = float("nan")
    for point in points or ():
        caked = _projected_source_row_caked_point(point)
        if caked is None:
            continue
        try:
            weight = float(point.get("weight", point.get("intensity", 1.0)))
        except Exception:
            weight = 1.0
        if not (
            np.isfinite(caked[0]) and np.isfinite(caked[1]) and np.isfinite(weight) and weight > 0.0
        ):
            continue
        raw_finite_count += 1
        if (
            roi_radius_deg is not None
            and center_2theta_deg is not None
            and center_phi_deg is not None
        ):
            dtheta_roi = float(caked[0]) - float(center_2theta_deg)
            dphi_roi = _caked_phi_delta_deg(float(caked[1]), float(center_phi_deg))
            roi_distance = float(math.hypot(dtheta_roi, dphi_roi))
            if np.isfinite(roi_distance) and (
                not np.isfinite(nearest_roi_distance) or roi_distance < nearest_roi_distance
            ):
                nearest_roi_distance = roi_distance
                nearest_roi_dtheta = float(dtheta_roi)
                nearest_roi_dphi = float(dphi_roi)
            if not bool(frozen_roi_gaussian_local_moment) and (
                not np.isfinite(roi_distance) or roi_distance > float(roi_radius_deg)
            ):
                continue
        finite_point = {
            "two_theta_deg": float(caked[0]),
            "phi_deg": _wrap_caked_phi_deg(float(caked[1])),
            "weight": float(weight),
        }
        try:
            detector_native_x = float(point.get("detector_native_x", np.nan))
            detector_native_y = float(point.get("detector_native_y", np.nan))
        except Exception:
            detector_native_x = detector_native_y = float("nan")
        if np.isfinite(detector_native_x) and np.isfinite(detector_native_y):
            finite_point["detector_native_x"] = float(detector_native_x)
            finite_point["detector_native_y"] = float(detector_native_y)
        finite_points.append(finite_point)
    if bool(frozen_roi_gaussian_local_moment):
        return _frozen_roi_gaussian_caked_centroid_from_points(
            finite_points,
            candidate_count=int(len(points or ())),
            raw_finite_count=int(raw_finite_count),
            center_2theta_deg=center_2theta_deg,
            center_phi_deg=center_phi_deg,
            roi_radius_deg=roi_radius_deg,
            nearest_roi_distance_deg=nearest_roi_distance,
            nearest_roi_dtheta_deg=nearest_roi_dtheta,
            nearest_roi_dphi_deg=nearest_roi_dphi,
        )
    if not finite_points:
        return {
            "status": "rejected",
            "reject_reason": "peak_left_roi" if raw_finite_count else "missing_peak_points",
            "candidate_count": int(len(points or ())),
            "raw_finite_count": int(raw_finite_count),
            "roi_rejected_count": int(raw_finite_count),
            "roi_center_2theta_deg": center_2theta_deg,
            "roi_center_phi_deg": center_phi_deg,
            "roi_radius_deg": roi_radius_deg,
            "nearest_roi_distance_deg": nearest_roi_distance,
            "nearest_roi_dtheta_deg": nearest_roi_dtheta,
            "nearest_roi_dphi_deg": nearest_roi_dphi,
        }

    centroid_points = finite_points
    centroid_point_cap = 8000
    sampled = False
    if len(centroid_points) > centroid_point_cap:
        sample_indices = np.unique(
            np.linspace(0, len(centroid_points) - 1, centroid_point_cap, dtype=int)
        )
        centroid_points = [centroid_points[int(idx)] for idx in sample_indices.tolist()]
        sampled = True

    seed_index = max(
        range(len(centroid_points)),
        key=lambda idx: (
            centroid_points[idx]["weight"],
            -abs(float(centroid_points[idx]["two_theta_deg"])),
        ),
    )
    seed_phi = float(centroid_points[seed_index]["phi_deg"])
    coords = np.asarray(
        [
            [
                float(point["two_theta_deg"]),
                _caked_phi_delta_deg(float(point["phi_deg"]), seed_phi),
            ]
            for point in centroid_points
        ],
        dtype=np.float64,
    )
    weights = np.asarray(
        [float(point["weight"]) for point in centroid_points],
        dtype=np.float64,
    )
    if connect_radius_deg is None:
        if len(centroid_points) >= 3:
            radius_coords = coords
            if len(centroid_points) > 2500:
                sample_indices = np.unique(
                    np.linspace(0, len(centroid_points) - 1, 2500, dtype=int)
                )
                radius_coords = coords[sample_indices]
            deltas = radius_coords[:, None, :] - radius_coords[None, :, :]
            distances = np.sqrt(np.sum(np.square(deltas), axis=2))
            distances[distances <= 0.0] = np.nan
            with np.errstate(all="ignore"):
                nearest = np.nanmin(distances, axis=1)
            nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
            radius = (
                float(max(0.5, min(0.75, 1.6 * np.nanpercentile(nearest, 90.0))))
                if nearest.size
                else 0.5
            )
        else:
            radius = 0.5
    else:
        radius = max(float(connect_radius_deg), 1.0e-12)
    component_ids = np.full(len(centroid_points), -1, dtype=np.int64)
    cell_coords = np.floor(coords / radius).astype(np.int64)
    cells: dict[tuple[int, int], list[int]] = {}
    for idx, cell in enumerate(cell_coords):
        cells.setdefault((int(cell[0]), int(cell[1])), []).append(int(idx))
    components: list[list[int]] = []
    for start in range(len(centroid_points)):
        if component_ids[start] >= 0:
            continue
        component_index = len(components)
        queue = [int(start)]
        component_ids[start] = component_index
        members: list[int] = []
        while queue:
            current = queue.pop()
            members.append(int(current))
            current_cell = cell_coords[current]
            candidate_indices: list[int] = []
            for dtheta_cell in (-1, 0, 1):
                for dphi_cell in (-1, 0, 1):
                    candidate_indices.extend(
                        cells.get(
                            (
                                int(current_cell[0]) + dtheta_cell,
                                int(current_cell[1]) + dphi_cell,
                            ),
                            [],
                        )
                    )
            if not candidate_indices:
                continue
            candidates = np.asarray(candidate_indices, dtype=np.int64)
            candidates = candidates[component_ids[candidates] < 0]
            if candidates.size <= 0:
                continue
            distances = np.sqrt(np.sum(np.square(coords[candidates] - coords[current]), axis=1))
            neighbors = candidates[distances <= radius]
            for neighbor in neighbors.tolist():
                component_ids[int(neighbor)] = component_index
                queue.append(int(neighbor))
        components.append(members)

    seed_component_id = int(component_ids[seed_index])
    component_weights = [
        float(np.sum(weights[np.asarray(members, dtype=np.int64)])) for members in components
    ]
    seed_weight = float(component_weights[seed_component_id])
    competing_weight = max(
        (weight for idx, weight in enumerate(component_weights) if idx != seed_component_id),
        default=0.0,
    )
    if seed_weight <= 0.0 or competing_weight >= float(ambiguity_fraction) * seed_weight:
        return {
            "status": "rejected",
            "reject_reason": "ambiguous_multimodal_peak",
            "candidate_count": int(len(finite_points)),
            "component_count": int(len(components)),
            "seed_component_weight": seed_weight,
            "competing_component_weight": competing_weight,
            "connect_radius_deg": float(radius),
        }
    member_indices = np.asarray(components[seed_component_id], dtype=np.int64)
    member_weights = weights[member_indices]
    member_coords = coords[member_indices]
    centroid_two_theta = float(np.average(member_coords[:, 0], weights=member_weights))
    centroid_local_phi = float(np.average(member_coords[:, 1], weights=member_weights))
    centroid_phi = _wrap_caked_phi_deg(seed_phi + centroid_local_phi)
    paired_detector_native: dict[str, object] = {}
    member_points = [centroid_points[int(index)] for index in member_indices.tolist()]
    if all(
        np.isfinite(float(point.get("detector_native_x", np.nan)))
        and np.isfinite(float(point.get("detector_native_y", np.nan)))
        for point in member_points
    ):
        paired_detector_native["detector_native_px"] = [
            float(
                np.average(
                    np.asarray(
                        [float(point["detector_native_x"]) for point in member_points],
                        dtype=np.float64,
                    ),
                    weights=member_weights,
                )
            ),
            float(
                np.average(
                    np.asarray(
                        [float(point["detector_native_y"]) for point in member_points],
                        dtype=np.float64,
                    ),
                    weights=member_weights,
                )
            ),
        ]
    return {
        "status": "ok",
        "centroid_2theta_deg": centroid_two_theta,
        "centroid_phi_deg": centroid_phi,
        "total_weight": seed_weight,
        "branch_total_weight": float(np.sum(weights)),
        "peak_height": float(np.max(member_weights)),
        "sample_count": int(member_indices.size),
        "candidate_count": int(len(finite_points)),
        "component_count": int(len(components)),
        "connect_radius_deg": float(radius),
        "centroid_points_sampled": bool(sampled),
        "prediction_mode": "selected_branch_caked_centroid",
        **paired_detector_native,
        "raw_finite_count": int(raw_finite_count),
        "roi_rejected_count": int(max(0, raw_finite_count - len(finite_points))),
        "roi_center_2theta_deg": center_2theta_deg,
        "roi_center_phi_deg": center_phi_deg,
        "roi_radius_deg": roi_radius_deg,
        "nearest_roi_distance_deg": nearest_roi_distance,
        "nearest_roi_dtheta_deg": nearest_roi_dtheta,
        "nearest_roi_dphi_deg": nearest_roi_dphi,
    }


def _attach_selected_branch_caked_centroid_fields(
    entry: dict[str, object],
    centroid: Mapping[str, object],
) -> None:
    theta = float(centroid["centroid_2theta_deg"])
    phi = float(centroid["centroid_phi_deg"])
    entry["selected_branch_caked_centroid_deg"] = [theta, phi]
    entry["process_peaks_centroid_caked_deg"] = [theta, phi]
    entry["predicted_centroid_2theta_deg"] = theta
    entry["predicted_centroid_phi_deg"] = phi
    entry["selected_branch_caked_centroid_source"] = "fresh_process_peaks_branch_peak_centroid"
    entry["process_peaks_centroid_caked_source"] = "fresh_process_peaks_branch_peak_centroid"
    entry["selected_branch_caked_centroid_prediction_mode"] = "selected_branch_caked_centroid"
    entry["selected_branch_caked_centroid_point_frame"] = "process_peaks_centroid_caked"
    entry["selected_branch_caked_centroid_status"] = "ok"
    detector_native_px = _finite_pair_value(centroid.get("detector_native_px"))
    if detector_native_px is not None:
        entry["selected_branch_caked_centroid_detector_native_px"] = [
            float(detector_native_px[0]),
            float(detector_native_px[1]),
        ]
        entry["selected_branch_caked_centroid_detector_native_source"] = (
            "fresh_process_peaks_branch_peak_centroid_same_physical_hits"
        )
        entry["selected_branch_caked_centroid_detector_native_point_frame"] = (
            "background_detector_native"
        )
    for field in (
        "total_weight",
        "branch_total_weight",
        "peak_height",
        "sample_count",
        "candidate_count",
        "component_count",
        "connect_radius_deg",
        "centroid_points_sampled",
        "raw_finite_count",
        "roi_rejected_count",
        "roi_center_2theta_deg",
        "roi_center_phi_deg",
        "roi_radius_deg",
        "nearest_roi_distance_deg",
        "nearest_roi_dtheta_deg",
        "nearest_roi_dphi_deg",
        "weighting_mode",
        "sigma_deg",
        "effective_weight_sum",
        "effective_weight_fraction",
        "minimum_effective_weight_fraction",
        "physical_weight_sum",
        "component_selection_used",
    ):
        if field in centroid:
            entry[f"selected_branch_caked_centroid_{field}"] = centroid.get(field)


def _centroid_source_table_token(entry: Mapping[str, object]) -> int | None:
    try:
        token = int(entry["source_reflection_index"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return token if token >= 0 else None


def _centroid_hkl_token(entry: Mapping[str, object]) -> tuple[int, int, int] | None:
    value = entry.get("hkl")
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
        return None
    try:
        return tuple(int(np.rint(float(item))) for item in value[:3])
    except Exception:
        return None


def _centroid_branch_token(entry: Mapping[str, object]) -> object | None:
    hkl = _centroid_hkl_token(entry)
    if hkl is not None and int(hkl[0]) == 0 and int(hkl[1]) == 0 and int(hkl[2]) > 0:
        return "00l_collapsed"
    branch_raw = entry.get("source_branch_index")
    try:
        branch = int(branch_raw)
    except Exception:
        return None
    return int(branch) if branch in {0, 1} else None


def stamp_selected_branch_frozen_caked_centroid_roi_centers(
    source_rows: Sequence[Mapping[str, object]] | None,
    *,
    required_manual_fit_targets: Sequence[Mapping[str, object]] | None,
    frozen_roi_native_detector_coords_to_caked_display_coords: (
        Callable[[float, float], object] | None
    ),
) -> list[dict[str, object]]:
    """Anchor each branch centroid ROI to its fit-start simulated peak."""

    copied_rows = [dict(entry) for entry in (source_rows or ()) if isinstance(entry, Mapping)]
    if not copied_rows or not callable(frozen_roi_native_detector_coords_to_caked_display_coords):
        return copied_rows

    def _q_group_branch_key(
        entry: Mapping[str, object],
    ) -> tuple[tuple[object, ...], object] | None:
        raw_q_group = entry.get("q_group_key")
        if isinstance(raw_q_group, (list, tuple)):
            q_group_key = tuple(raw_q_group)
        else:
            q_group_key = geometry_q_group_key_from_entry(entry)
        branch = _centroid_branch_token(entry)
        if q_group_key is None or branch is None:
            return None
        return tuple(q_group_key), branch

    row_keys = {
        key for key in (_q_group_branch_key(entry) for entry in copied_rows) if key is not None
    }
    if not row_keys:
        return copied_rows

    frozen_centers_by_key: dict[tuple[tuple[object, ...], object], tuple[float, float]] = {}
    for target in required_manual_fit_targets or ():
        if not isinstance(target, Mapping):
            continue
        key = _q_group_branch_key(target)
        if key is None or key not in row_keys or key in frozen_centers_by_key:
            continue
        native_point = _finite_pair_value(target.get("sim_refined_detector_native_px"))
        if native_point is None:
            native_point = _finite_pair_value(
                (
                    target.get("refined_sim_native_x", np.nan),
                    target.get("refined_sim_native_y", np.nan),
                )
            )
        if native_point is None:
            continue
        try:
            projected = frozen_roi_native_detector_coords_to_caked_display_coords(
                float(native_point[0]),
                float(native_point[1]),
            )
            center = (float(projected[0]), float(projected[1]))  # type: ignore[index]
        except Exception:
            continue
        if not all(np.isfinite(value) for value in center):
            continue
        frozen_centers_by_key[key] = center

    for entry in copied_rows:
        key = _q_group_branch_key(entry)
        center = frozen_centers_by_key.get(key) if key is not None else None
        if center is None:
            continue
        entry["selected_branch_caked_centroid_frozen_roi_center_deg"] = [
            float(center[0]),
            float(center[1]),
        ]
        entry["selected_branch_caked_centroid_frozen_roi_center_source"] = (
            "fit_start_saved_simulated_native_via_frozen_frame_adapter"
        )
        entry["selected_branch_caked_centroid_frozen_roi_center_input_frame"] = (
            "background_detector_native"
        )
        entry["selected_branch_caked_centroid_frozen_roi_center_projector_input_frame"] = (
            "simulation_detector_native"
        )
    return copied_rows


def build_frozen_caked_centroid_native_roi_projector(
    *,
    centroid_hit_projector: Callable[[float, float], object] | None,
    native_detector_shape: object,
    bundle_detector_shape: object,
    hit_table_detector_shape: object,
    native_detector_coords_to_bundle_detector_coords: (Callable[[float, float], object] | None),
    bundle_detector_coords_to_hit_table_detector_coords: (Callable[[float, float], object] | None),
    hit_table_detector_coords_to_bundle_detector_coords: (Callable[[float, float], object] | None),
) -> Callable[[float, float], tuple[float, float]] | None:
    """Adapt saved detector-native ROI centers into a frozen hit projector."""

    native_shape = _positive_shape(native_detector_shape)
    bundle_shape = _positive_shape(bundle_detector_shape)
    hit_shape = _positive_shape(hit_table_detector_shape)
    if (
        native_shape is None
        or bundle_shape is None
        or hit_shape is None
        or not callable(centroid_hit_projector)
        or not callable(native_detector_coords_to_bundle_detector_coords)
        or not callable(bundle_detector_coords_to_hit_table_detector_coords)
        or not callable(hit_table_detector_coords_to_bundle_detector_coords)
    ):
        return None

    def _finite_projected_pair(
        callback: Callable[[float, float], object],
        col: float,
        row: float,
    ) -> tuple[float, float] | None:
        try:
            raw_point = callback(float(col), float(row))
            point = (float(raw_point[0]), float(raw_point[1]))  # type: ignore[index]
        except Exception:
            return None
        if not all(np.isfinite(value) for value in point):
            return None
        return point

    native_height, native_width = native_shape
    bundle_height, bundle_width = bundle_shape
    expected_bundle_corners = {
        (0.0, 0.0),
        (float(bundle_width - 1), 0.0),
        (0.0, float(bundle_height - 1)),
        (float(bundle_width - 1), float(bundle_height - 1)),
    }
    mapped_bundle_corners: set[tuple[float, float]] = set()
    for native_col, native_row in (
        (0.0, 0.0),
        (float(native_width - 1), 0.0),
        (0.0, float(native_height - 1)),
        (float(native_width - 1), float(native_height - 1)),
    ):
        bundle_point = _finite_projected_pair(
            native_detector_coords_to_bundle_detector_coords,
            native_col,
            native_row,
        )
        if bundle_point is None:
            return None
        mapped_bundle_corners.add(
            (round(float(bundle_point[0]), 9), round(float(bundle_point[1]), 9))
        )
        hit_point = _finite_projected_pair(
            bundle_detector_coords_to_hit_table_detector_coords,
            *bundle_point,
        )
        if hit_point is None:
            return None
        roundtrip_bundle_point = _finite_projected_pair(
            hit_table_detector_coords_to_bundle_detector_coords,
            *hit_point,
        )
        if roundtrip_bundle_point is None or not np.allclose(
            roundtrip_bundle_point,
            bundle_point,
            rtol=0.0,
            atol=1.0e-9,
        ):
            return None
    if mapped_bundle_corners != expected_bundle_corners:
        return None

    def _project(col: float, row: float) -> tuple[float, float]:
        bundle_point = _finite_projected_pair(
            native_detector_coords_to_bundle_detector_coords,
            float(col),
            float(row),
        )
        if bundle_point is None:
            raise ValueError("frozen centroid ROI native detector point is invalid")
        hit_point = _finite_projected_pair(
            bundle_detector_coords_to_hit_table_detector_coords,
            *bundle_point,
        )
        if hit_point is None:
            raise ValueError("frozen centroid ROI hit-table point is invalid")
        projected = centroid_hit_projector(float(hit_point[0]), float(hit_point[1]))
        try:
            caked_point = (float(projected[0]), float(projected[1]))  # type: ignore[index]
        except Exception as exc:
            raise ValueError("frozen centroid ROI caked point is invalid") from exc
        if not all(np.isfinite(value) for value in caked_point):
            raise ValueError("frozen centroid ROI caked point is nonfinite")
        return caked_point

    return _project


def stamp_selected_branch_caked_centroid_projector_provenance(
    rows: Sequence[Mapping[str, object]] | None,
    provenance: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    """Copy frozen-projector provenance onto rows with valid caked centroids."""

    copied_rows = [dict(row) for row in (rows or ()) if isinstance(row, Mapping)]
    if not isinstance(provenance, Mapping):
        return copied_rows
    field_names = {
        "schema": "selected_branch_caked_centroid_projector_schema",
        "projection_payload_digest": (
            "selected_branch_caked_centroid_projection_payload_digest"
        ),
        "native_detector_shape": (
            "selected_branch_caked_centroid_projector_native_detector_shape"
        ),
        "bundle_detector_shape": (
            "selected_branch_caked_centroid_projector_bundle_detector_shape"
        ),
        "hit_table_detector_shape": (
            "selected_branch_caked_centroid_projector_hit_table_detector_shape"
        ),
        "center_row_col": "selected_branch_caked_centroid_projector_center_row_col",
        "distance_m": "selected_branch_caked_centroid_projector_distance_m",
        "pixel_size_m": "selected_branch_caked_centroid_projector_pixel_size_m",
        "center_input_frame": (
            "selected_branch_caked_centroid_projector_center_input_frame"
        ),
        "hit_table_point_frame": (
            "selected_branch_caked_centroid_projector_hit_table_point_frame"
        ),
        "projector_point_frame": (
            "selected_branch_caked_centroid_projector_point_frame"
        ),
    }
    for row in copied_rows:
        if str(row.get("selected_branch_caked_centroid_status") or "") != "ok":
            continue
        for source_field, target_field in field_names.items():
            if source_field in provenance:
                row[target_field] = copy.deepcopy(provenance[source_field])
    return copied_rows


def _centroid_roi_center_from_target(
    entry: Mapping[str, object],
) -> tuple[float, float] | None:
    return _finite_pair_value(entry.get("selected_branch_caked_centroid_frozen_roi_center_deg"))


def _centroid_hit_table_source_token(table: object) -> int | None:
    try:
        arr = np.asarray(table, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("centroid hit table must be a numeric matrix") from exc
    if arr.ndim != 2:
        raise ValueError("centroid hit table must be two-dimensional")
    if arr.shape[0] == 0:
        return None
    source_tokens: set[int] = set()
    for row in arr:
        source_index, _source_row_index, _best_sample_index = extract_hit_row_provenance(row)
        if source_index is None:
            raise ValueError("centroid hit table rows require source-table provenance")
        source_tokens.add(int(source_index))
    if len(source_tokens) != 1:
        raise ValueError("centroid hit table mixes source-table provenance")
    return source_tokens.pop()


def _selected_branch_hit_table_mask(
    table: np.ndarray,
    selected: Mapping[str, object],
) -> np.ndarray:
    arr = np.asarray(table, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        return np.zeros(0, dtype=bool)
    required_width = (
        max(
            HIT_ROW_COL_INTENSITY,
            HIT_ROW_COL_DETECTOR_COL,
            HIT_ROW_COL_DETECTOR_ROW,
            HIT_ROW_COL_PHI,
            HIT_ROW_COL_H,
            HIT_ROW_COL_K,
            HIT_ROW_COL_L,
        )
        + 1
    )
    if arr.shape[1] < required_width:
        return np.zeros(arr.shape[0], dtype=bool)
    mask = (
        np.isfinite(arr[:, HIT_ROW_COL_INTENSITY])
        & (arr[:, HIT_ROW_COL_INTENSITY] > 0.0)
        & np.isfinite(arr[:, HIT_ROW_COL_DETECTOR_COL])
        & np.isfinite(arr[:, HIT_ROW_COL_DETECTOR_ROW])
    )
    hkl_finite = (
        np.isfinite(arr[:, HIT_ROW_COL_H])
        & np.isfinite(arr[:, HIT_ROW_COL_K])
        & np.isfinite(arr[:, HIT_ROW_COL_L])
    )
    row_h = np.zeros(arr.shape[0], dtype=np.int64)
    row_k = np.zeros(arr.shape[0], dtype=np.int64)
    row_l = np.zeros(arr.shape[0], dtype=np.int64)
    if np.any(hkl_finite):
        row_h[hkl_finite] = np.rint(arr[hkl_finite, HIT_ROW_COL_H]).astype(
            np.int64,
            copy=False,
        )
        row_k[hkl_finite] = np.rint(arr[hkl_finite, HIT_ROW_COL_K]).astype(
            np.int64,
            copy=False,
        )
        row_l[hkl_finite] = np.rint(arr[hkl_finite, HIT_ROW_COL_L]).astype(
            np.int64,
            copy=False,
        )
    hkl = _centroid_hkl_token(selected)
    if hkl is None:
        return np.zeros(arr.shape[0], dtype=bool)
    mask &= hkl_finite & (row_h == int(hkl[0])) & (row_k == int(hkl[1])) & (row_l == int(hkl[2]))
    branch = _centroid_branch_token(selected)
    if branch == "00l_collapsed":
        return mask
    try:
        expected_branch = int(branch)
    except Exception:
        return mask
    phi = arr[:, HIT_ROW_COL_PHI]
    wrapped = ((phi + np.pi) % (2.0 * np.pi)) - np.pi
    wrapped = np.where(wrapped <= -np.pi + 1.0e-12, np.pi, wrapped)
    branch_mask = np.isfinite(wrapped) & (
        np.abs(wrapped) > SOURCE_BRANCH_PHI_ZERO_DEADBAND_RAD
    )
    branch_mask &= wrapped < 0.0 if expected_branch == 0 else wrapped > 0.0
    return mask & branch_mask


def _centroid_caked_points_from_hit_table(
    selected: Mapping[str, object],
    table: object,
    native_detector_coords_to_caked_display_coords: Callable[[float, float], object],
    *,
    hit_table_detector_coords_to_native_detector_coords: (
        Callable[[float, float], object] | None
    ) = None,
) -> list[dict[str, float]]:
    try:
        arr = np.asarray(table, dtype=np.float64)
    except Exception:
        return []
    mask = _selected_branch_hit_table_mask(
        arr,
        selected,
    )
    if mask.size <= 0 or not np.any(mask):
        return []
    points: list[dict[str, float]] = []
    for row_arr in arr[mask]:
        try:
            weight = float(row_arr[HIT_ROW_COL_INTENSITY])
            detector_x = float(row_arr[HIT_ROW_COL_DETECTOR_COL])
            detector_y = float(row_arr[HIT_ROW_COL_DETECTOR_ROW])
        except Exception:
            continue
        if not (
            np.isfinite(weight)
            and weight > 0.0
            and np.isfinite(detector_x)
            and np.isfinite(detector_y)
        ):
            continue
        try:
            caked = native_detector_coords_to_caked_display_coords(
                float(detector_x),
                float(detector_y),
            )
        except Exception:
            continue
        point = _finite_pair_value(caked)
        if point is None:
            continue
        projected_point = {
            "two_theta_deg": float(point[0]),
            "phi_deg": float(point[1]),
            "weight": float(weight),
            "detector_x": float(detector_x),
            "detector_y": float(detector_y),
        }
        if callable(hit_table_detector_coords_to_native_detector_coords):
            try:
                detector_native = _finite_pair_value(
                    hit_table_detector_coords_to_native_detector_coords(
                        float(detector_x),
                        float(detector_y),
                    )
                )
            except Exception:
                detector_native = None
            if detector_native is not None:
                projected_point["detector_native_x"] = float(detector_native[0])
                projected_point["detector_native_y"] = float(detector_native[1])
        points.append(projected_point)
    return points


def _stamp_selected_branch_caked_centroid_rejection(
    entry: dict[str, object],
    reason: str,
    **fields: object,
) -> None:
    entry.setdefault("selected_branch_caked_centroid_status", "rejected")
    entry.setdefault("selected_branch_caked_centroid_reject_reason", str(reason))
    for key, value in fields.items():
        entry.setdefault(f"selected_branch_caked_centroid_{key}", value)


def attach_selected_branch_caked_centroids_from_hit_tables(
    target_rows: Sequence[Mapping[str, object]],
    centroid_hit_tables: Sequence[object],
    *,
    native_detector_coords_to_caked_display_coords: Callable[[float, float], object],
    hit_table_detector_coords_to_native_detector_coords: (
        Callable[[float, float], object] | None
    ) = None,
) -> list[dict[str, object]]:
    """Attach selected-branch centroids from provenance-matched hit tables."""

    if not callable(native_detector_coords_to_caked_display_coords):
        raise TypeError("native detector-to-caked projector must be callable")

    copied_targets = [dict(entry) for entry in target_rows]
    tables_by_source: dict[int, object] = {}
    for table in centroid_hit_tables:
        source_token = _centroid_hit_table_source_token(table)
        if source_token is None:
            continue
        if source_token in tables_by_source:
            raise ValueError(
                f"multiple centroid hit tables carry source_reflection_index={source_token}"
            )
        tables_by_source[source_token] = table

    rejection_fields = {
        "candidate_count",
        "raw_finite_count",
        "roi_rejected_count",
        "roi_center_2theta_deg",
        "roi_center_phi_deg",
        "roi_radius_deg",
        "nearest_roi_distance_deg",
        "nearest_roi_dtheta_deg",
        "nearest_roi_dphi_deg",
        "weighting_mode",
        "sigma_deg",
        "effective_weight_sum",
        "effective_weight_fraction",
        "minimum_effective_weight_fraction",
        "physical_weight_sum",
        "component_selection_used",
    }

    for entry in copied_targets:
        source_token = _centroid_source_table_token(entry)
        if source_token is None:
            _stamp_selected_branch_caked_centroid_rejection(
                entry,
                "missing_source_reflection_index",
            )
            continue
        if _centroid_hkl_token(entry) is None:
            _stamp_selected_branch_caked_centroid_rejection(
                entry,
                "missing_hkl",
                source_table_index=source_token,
            )
            continue
        if _centroid_branch_token(entry) is None:
            _stamp_selected_branch_caked_centroid_rejection(
                entry,
                "missing_source_branch_index",
                source_table_index=source_token,
            )
            continue
        center = _centroid_roi_center_from_target(entry)
        if center is None:
            _stamp_selected_branch_caked_centroid_rejection(
                entry,
                "missing_frozen_roi_center",
                source_table_index=source_token,
            )
            continue
        table = tables_by_source.get(source_token)
        if table is None:
            _stamp_selected_branch_caked_centroid_rejection(
                entry,
                "missing_source_hit_table",
                source_table_index=source_token,
            )
            continue

        points = _centroid_caked_points_from_hit_table(
            entry,
            table,
            native_detector_coords_to_caked_display_coords,
            hit_table_detector_coords_to_native_detector_coords=(
                hit_table_detector_coords_to_native_detector_coords
            ),
        )
        if not points:
            _stamp_selected_branch_caked_centroid_rejection(
                entry,
                "missing_selected_branch_hit_points",
                source_table_index=source_token,
            )
            continue

        centroid = _selected_branch_caked_centroid_from_points(
            points,
            connect_radius_deg=None,
            center_2theta_deg=center[0],
            center_phi_deg=center[1],
            roi_radius_deg=2.0,
            frozen_roi_gaussian_local_moment=True,
        )
        entry["selected_branch_caked_centroid_source_match_policy"] = "source_reflection_index"
        entry["selected_branch_caked_centroid_source_table_index"] = source_token
        if centroid.get("status") == "ok":
            _attach_selected_branch_caked_centroid_fields(entry, centroid)
            continue

        reject_reason = str(centroid.get("reject_reason") or "centroid_rejected")
        _stamp_selected_branch_caked_centroid_rejection(
            entry,
            reject_reason,
            source_table_index=source_token,
            **{str(key): value for key, value in centroid.items() if str(key) in rejection_fields},
        )
    return copied_targets


def geometry_q_group_key_from_entry(
    entry: Mapping[str, object] | None,
) -> tuple[object, ...] | None:
    """Return the stable Qr/Qz group key for one simulated peak record."""

    if not isinstance(entry, Mapping):
        return None
    allow_nominal_hkl_indices = bool(entry.get("q_group_nominal_hkl", False))
    hkl_value = entry.get("hkl_raw", entry.get("hkl"))
    key, _, _ = reflection_q_group_metadata(
        hkl_value,
        source_label=entry.get("source_label", "primary"),
        a_value=entry.get("av", np.nan),
        c_value=entry.get("cv", np.nan),
        qr_value=entry.get("qr", np.nan),
        allow_nominal_hkl_indices=allow_nominal_hkl_indices,
    )
    if key is None:
        key, _, _ = reflection_q_group_metadata(
            entry.get("hkl"),
            source_label=entry.get("source_label", "primary"),
            a_value=entry.get("av", np.nan),
            c_value=entry.get("cv", np.nan),
            qr_value=entry.get("qr", np.nan),
            allow_nominal_hkl_indices=True,
        )
    return key


def build_geometry_q_group_entries(
    max_positions_local: Sequence[object] | None,
    *,
    peak_table_lattice: Sequence[Sequence[object]] | None = None,
    peak_records: Sequence[Mapping[str, object]] | None = None,
    primary_a: object = np.nan,
    primary_c: object = np.nan,
    allow_nominal_hkl_indices: bool = False,
) -> list[dict[str, object]]:
    """Aggregate simulated hit tables or cached peak records into Qr/Qz rows."""

    try:
        default_primary_a = float(primary_a)
    except Exception:
        default_primary_a = float("nan")
    try:
        default_primary_c = float(primary_c)
    except Exception:
        default_primary_c = float("nan")

    entries_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    entries_by_identity: dict[tuple[object, ...], tuple[object, ...]] = {}

    def _source_metadata(
        source_label: object,
        *,
        phase_label: object | None = None,
        structure_role: object | None = None,
    ) -> tuple[str | None, str | None]:
        source_text = gui_controllers.normalize_bragg_qr_source_label(
            str(source_label) if source_label is not None else "primary"
        )
        phase_text = str(phase_label) if phase_label not in (None, "") else None
        role_text = str(structure_role) if structure_role not in (None, "") else None
        if source_text == DISORDERED_PHASE_SOURCE_LABEL:
            phase_text = phase_text or DISORDERED_PHASE_DISPLAY_LABEL
            role_text = role_text or "disordered"
        return phase_text, role_text

    def _entry_for_group(
        group_key: tuple[object, ...],
        *,
        source_label: object,
        qr_val: object,
        qz_val: object,
        phase_label: object | None = None,
        structure_role: object | None = None,
    ) -> dict[str, object]:
        identity = _qr_qz_duplicate_identity(qr_val, qz_val)
        phase_text, role_text = _source_metadata(
            source_label,
            phase_label=phase_label,
            structure_role=structure_role,
        )
        lookup_key = group_key
        if identity is not None:
            identity_key = (
                gui_controllers.normalize_bragg_qr_source_label(str(source_label)),
                identity,
            )
            existing_key = entries_by_identity.get(identity_key)
            if existing_key is not None:
                lookup_key = existing_key
                existing = entries_by_key.get(existing_key)
                if existing is not None and q_group_source_priority(
                    source_label
                ) < q_group_source_priority(existing.get("source_label")):
                    entries_by_key.pop(existing_key, None)
                    lookup_key = group_key
                    existing["key"] = group_key
                    existing["source_label"] = str(source_label)
                    entries_by_key[group_key] = existing
                    entries_by_identity[identity_key] = group_key
            else:
                entries_by_identity[identity_key] = group_key

        entry = entries_by_key.get(lookup_key)
        if entry is None:
            ml_components = geometry_q_group_ml_from_key(group_key)
            if ml_components is None:
                m_index = lookup_key[2]
                l_index = lookup_key[3]
            else:
                m_index, l_index = ml_components
            entry = {
                "key": lookup_key,
                "source_label": str(source_label),
                "source_labels": [str(source_label)],
                "overlap_identity": identity,
                "qr": float(qr_val),
                "qz": float(qz_val),
                "m_index": m_index,
                "l_index": int(l_index),
                "gz_index": int(lookup_key[3]),
                "total_intensity": 0.0,
                "peak_count": 0,
                "hkl_preview": [],
            }
            if phase_text is not None:
                entry["phase_label"] = phase_text
            if role_text is not None:
                entry["structure_role"] = role_text
            entries_by_key[lookup_key] = entry
        else:
            labels = entry.setdefault("source_labels", [])
            if isinstance(labels, list) and str(source_label) not in labels:
                labels.append(str(source_label))
                entry["source_label"] = "+".join(str(label) for label in labels)
            if identity is not None and entry.get("overlap_identity") is None:
                entry["overlap_identity"] = identity
            if phase_text is not None:
                phase_labels = entry.setdefault("phase_labels", [])
                if isinstance(phase_labels, list) and phase_text not in phase_labels:
                    phase_labels.append(phase_text)
                entry.setdefault("phase_label", phase_text)
            if role_text is not None:
                roles = entry.setdefault("structure_roles", [])
                if isinstance(roles, list) and role_text not in roles:
                    roles.append(role_text)
                entry.setdefault("structure_role", role_text)
        return entry

    cached_peak_records = list(peak_records or [])
    if cached_peak_records:
        for raw_record in cached_peak_records:
            if not isinstance(raw_record, Mapping):
                continue
            source_label = str(raw_record.get("source_label", "primary"))
            phase_label = raw_record.get("phase_label")
            structure_role = raw_record.get("structure_role")
            av_used = _coerce_float(raw_record.get("av", primary_a), default_primary_a)
            cv_used = _coerce_float(raw_record.get("cv", primary_c), default_primary_c)
            intensity = _coerce_float(
                raw_record.get("intensity", raw_record.get("weight", 0.0)),
                0.0,
            )
            hkl_raw = raw_record.get("hkl_raw", raw_record.get("hkl"))
            hkl_key = gui_geometry_overlay.normalize_hkl_key(hkl_raw)
            use_nominal_hkl_indices = bool(
                raw_record.get("q_group_nominal_hkl", allow_nominal_hkl_indices)
            )
            raw_group_key = raw_record.get("q_group_key")
            if isinstance(raw_group_key, list):
                group_key = tuple(raw_group_key)
            elif isinstance(raw_group_key, tuple):
                group_key = raw_group_key
            else:
                group_key = None
            resolved_group_key, qr_val, qz_val = reflection_q_group_metadata(
                hkl_raw,
                source_label=source_label,
                a_value=av_used,
                c_value=cv_used,
                qr_value=raw_record.get("qr", np.nan),
                allow_nominal_hkl_indices=use_nominal_hkl_indices,
            )
            if group_key is None:
                group_key = resolved_group_key
            if group_key is None:
                continue
            if hkl_key is None:
                qr_val = _coerce_float(raw_record.get("qr", qr_val), float("nan"))
                qz_val = _coerce_float(raw_record.get("qz", qz_val), float("nan"))
                try:
                    m_val = float(group_key[2])
                except Exception:
                    m_val = float("nan")
                try:
                    l_val = int(group_key[3])
                except Exception:
                    l_val = 0
                if not np.isfinite(qr_val) and np.isfinite(av_used) and av_used > 0.0:
                    qr_val = (2.0 * np.pi / av_used) * np.sqrt((4.0 / 3.0) * max(0.0, float(m_val)))
                if not np.isfinite(qz_val) and np.isfinite(cv_used) and cv_used > 0.0:
                    qz_val = (2.0 * np.pi / cv_used) * float(l_val)

            entry = _entry_for_group(
                group_key,
                source_label=source_label,
                qr_val=qr_val,
                qz_val=qz_val,
                phase_label=phase_label,
                structure_role=structure_role,
            )
            if not np.isfinite(_coerce_float(entry.get("qr", np.nan), np.nan)) and np.isfinite(
                qr_val
            ):
                entry["qr"] = float(qr_val)
            if not np.isfinite(_coerce_float(entry.get("qz", np.nan), np.nan)) and np.isfinite(
                qz_val
            ):
                entry["qz"] = float(qz_val)

            entry["total_intensity"] = float(entry["total_intensity"]) + float(abs(intensity))
            entry["peak_count"] = int(entry["peak_count"]) + 1
            if (
                hkl_key is not None
                and len(entry["hkl_preview"]) < 4
                and hkl_key not in entry["hkl_preview"]
            ):
                entry["hkl_preview"].append(hkl_key)
    elif max_positions_local is not None:
        peak_table_lattice_local = list(peak_table_lattice or [])
        if not peak_table_lattice_local or len(peak_table_lattice_local) != len(
            max_positions_local
        ):
            peak_table_lattice_local = [
                (default_primary_a, default_primary_c, "primary") for _ in max_positions_local
            ]

        for table_idx, tbl in enumerate(max_positions_local):
            rows = geometry_reference_hit_rows(tbl)
            if not rows:
                continue

            av_used = default_primary_a
            cv_used = default_primary_c
            source_label = "primary"
            phase_label = None
            structure_role = None
            if table_idx < len(peak_table_lattice_local):
                try:
                    av_used = float(peak_table_lattice_local[table_idx][0])
                    cv_used = float(peak_table_lattice_local[table_idx][1])
                    source_label = str(peak_table_lattice_local[table_idx][2])
                    if len(peak_table_lattice_local[table_idx]) >= 4:
                        phase_label = peak_table_lattice_local[table_idx][3]
                    if len(peak_table_lattice_local[table_idx]) >= 5:
                        structure_role = peak_table_lattice_local[table_idx][4]
                except Exception:
                    av_used = default_primary_a
                    cv_used = default_primary_c
                    source_label = "primary"
                    phase_label = None
                    structure_role = None

            for row in rows:
                intensity, _xpix, _ypix, _phi, h_val, k_val, l_val = row[:7]
                if not np.isfinite(intensity):
                    continue
                hkl_key = tuple(int(np.rint(v)) for v in (h_val, k_val, l_val))
                group_key, qr_val, qz_val = reflection_q_group_metadata(
                    (h_val, k_val, l_val),
                    source_label=source_label,
                    a_value=av_used,
                    c_value=cv_used,
                    allow_nominal_hkl_indices=allow_nominal_hkl_indices,
                )
                if group_key is None:
                    continue

                entry = _entry_for_group(
                    group_key,
                    source_label=source_label,
                    qr_val=qr_val,
                    qz_val=qz_val,
                    phase_label=phase_label,
                    structure_role=structure_role,
                )

                entry["total_intensity"] = float(entry["total_intensity"]) + float(abs(intensity))
                entry["peak_count"] = int(entry["peak_count"]) + 1
                if len(entry["hkl_preview"]) < 4 and hkl_key not in entry["hkl_preview"]:
                    entry["hkl_preview"].append(hkl_key)
    else:
        return []

    def _sort_value(value: object) -> float:
        numeric = _coerce_float(value, float("nan"))
        return numeric if np.isfinite(numeric) else float("inf")

    entries = list(entries_by_key.values())
    entries.sort(
        key=lambda entry: (
            _sort_value(entry.get("qr", np.nan)),
            _sort_value(entry.get("qz", np.nan)),
            q_group_source_priority(entry.get("source_label")),
            str(entry.get("source_label", "")),
        )
    )
    return entries


def build_geometry_fit_simulated_peaks(
    hit_tables: Sequence[object] | None,
    *,
    image_shape: tuple[int, int],
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ],
    peak_table_lattice: Sequence[Sequence[object]] | None = None,
    primary_a: object = np.nan,
    primary_c: object = np.nan,
    default_source_label: str | None = "primary",
    round_pixel_centers: bool = False,
    allow_nominal_hkl_indices: bool = False,
    profile_cache: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Build simulated-peak records from detector hit tables for geometry workflows."""

    if not hit_tables:
        return []

    try:
        default_primary_a = float(primary_a)
    except Exception:
        default_primary_a = float("nan")
    try:
        default_primary_c = float(primary_c)
    except Exception:
        default_primary_c = float("nan")

    simulated_peaks: list[dict[str, object]] = []
    peak_table_lattice_local = list(peak_table_lattice or [])
    for table_idx, tbl in enumerate(hit_tables):
        rows = geometry_reference_hit_rows(tbl)
        if not rows:
            continue

        source_label = (
            str(default_source_label) if default_source_label is not None else f"table_{table_idx}"
        )
        av_used = default_primary_a
        cv_used = default_primary_c
        if table_idx < len(peak_table_lattice_local):
            lattice_entry = peak_table_lattice_local[table_idx]
            if isinstance(lattice_entry, Sequence) and len(lattice_entry) >= 3:
                try:
                    av_used = float(lattice_entry[0])
                    cv_used = float(lattice_entry[1])
                    source_label = str(lattice_entry[2])
                except Exception:
                    av_used = default_primary_a
                    cv_used = default_primary_c
                    source_label = (
                        str(default_source_label)
                        if default_source_label is not None
                        else f"table_{table_idx}"
                    )

        for row in rows:
            intensity, xpix, ypix, phi_rad, h_val, k_val, l_val = row[:7]
            if not (np.isfinite(intensity) and np.isfinite(xpix) and np.isfinite(ypix)):
                continue

            source_table_index, source_row_index, best_sample_index = extract_hit_row_provenance(
                row
            )
            if source_table_index is None or source_row_index is None:
                raise ValueError("Current geometry hit tables require source provenance.")

            native_col = float(xpix)
            native_row = float(ypix)
            if round_pixel_centers:
                native_col = float(int(round(native_col)))
                native_row = float(int(round(native_row)))

            display_col, display_row = native_sim_to_display_coords(
                native_col,
                native_row,
                image_shape,
            )
            hkl = tuple(int(np.rint(val)) for val in (h_val, k_val, l_val))
            hkl_raw = (float(h_val), float(k_val), float(l_val))
            q_group_key, qr_val, qz_val = reflection_q_group_metadata(
                hkl_raw,
                source_label=source_label,
                a_value=av_used,
                c_value=cv_used,
                allow_nominal_hkl_indices=allow_nominal_hkl_indices,
            )
            if q_group_key is None:
                continue

            branch_index = source_branch_index_from_phi_rad(phi_rad)
            if branch_index not in {0, 1} and not (
                hkl[0] == 0 and hkl[1] == 0 and hkl[2] > 0
            ):
                continue

            peak_record = {
                "hkl": hkl,
                "native_col": float(native_col),
                "native_row": float(native_row),
                "coordinate_frame": "simulation_native",
                "sim_col": float(display_col),
                "sim_row": float(display_row),
                "sim_col_raw": float(display_col),
                "sim_row_raw": float(display_row),
                "display_col": float(display_col),
                "display_row": float(display_row),
                "detector_display_source": "native_sim_to_display",
                "weight": max(0.0, float(abs(intensity))),
                "source_label": str(source_label),
                "source_table_index": int(source_table_index),
                "source_reflection_index": int(source_table_index),
                "source_row_index": int(source_row_index),
                "hkl_raw": hkl_raw,
                "phi": float(phi_rad),
                "av": float(av_used),
                "cv": float(cv_used),
                "qr": float(qr_val),
                "qz": float(qz_val),
                "q_group_key": q_group_key,
            }
            if best_sample_index is not None:
                peak_record["best_sample_index"] = int(best_sample_index)
            if branch_index in {0, 1}:
                peak_record["source_branch_index"] = int(branch_index)
            peak_record = gui_manual_geometry.geometry_manual_canonicalize_live_source_entry(
                peak_record,
                normalize_hkl_key=(
                    gui_geometry_overlay.normalize_hkl_key
                    if callable(getattr(gui_geometry_overlay, "normalize_hkl_key", None))
                    else None
                )
                or (lambda value: hkl if value is not None else None),
            )
            if peak_record is None:
                continue
            if allow_nominal_hkl_indices:
                peak_record["q_group_nominal_hkl"] = True
            simulated_peaks.append(peak_record)

    simulated_peaks = canonicalize_qr_qz_duplicate_source_rows(
        simulated_peaks,
        preserve_source_identity=True,
    )
    simulated_peaks = [
        gui_mosaic_top.annotate_selection_metadata(
            entry,
            target_key=entry.get("q_group_key"),
            profile_cache=profile_cache,
        )
        for entry in simulated_peaks
    ]
    return simulated_peaks


def _runtime_peak_row_finite_point(
    source: Mapping[str, object],
    x_key: str,
    y_key: str,
) -> tuple[float, float] | None:
    try:
        col = float(source.get(x_key, np.nan))
        row = float(source.get(y_key, np.nan))
    except Exception:
        return None
    if not (np.isfinite(col) and np.isfinite(row)):
        return None
    return float(col), float(row)


def build_projected_geometry_fit_simulated_peaks(
    hit_tables: Sequence[object] | None,
    *,
    image_shape: tuple[int, int],
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ],
    peak_table_lattice: Sequence[Sequence[object]] | None = None,
    primary_a: object = np.nan,
    primary_c: object = np.nan,
    default_source_label: str | None = "primary",
    allow_nominal_hkl_indices: bool = False,
    project_peaks_to_current_view: Callable[
        [Sequence[dict[str, object]]],
        Sequence[Mapping[str, object]] | None,
    ]
    | None = None,
    caked_view_enabled: bool = False,
    profile_cache: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    records = build_geometry_fit_simulated_peaks(
        hit_tables,
        image_shape=image_shape,
        native_sim_to_display_coords=native_sim_to_display_coords,
        peak_table_lattice=peak_table_lattice,
        primary_a=primary_a,
        primary_c=primary_c,
        default_source_label=default_source_label,
        allow_nominal_hkl_indices=allow_nominal_hkl_indices,
        profile_cache=profile_cache,
    )
    if bool(caked_view_enabled) and records:
        if not callable(project_peaks_to_current_view):
            raise RuntimeError("Caked geometry preview requires the current caked projector.")
        records = [
            dict(entry)
            for entry in project_peaks_to_current_view(records)
            if isinstance(entry, Mapping)
        ]
    if caked_view_enabled:
        records = [
            dict(entry)
            for entry in records
            if _runtime_peak_row_finite_point(entry, "caked_x", "caked_y") is not None
        ]
    records = canonicalize_qr_qz_duplicate_source_rows(
        records,
        preserve_source_identity=True,
    )
    return records


def build_geometry_fit_full_order_source_rows(
    hit_tables: Sequence[object] | None,
    *,
    image_shape: tuple[int, int],
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ],
    primary_a: object = np.nan,
    primary_c: object = np.nan,
    default_source_label: str = "primary",
    round_pixel_centers: bool = False,
    allow_nominal_hkl_indices: bool = False,
) -> tuple[list[dict[str, object]], list[tuple[float, float, str]]]:
    """Build full-order source rows from authoritative hit-row provenance."""

    peak_table_lattice = [
        (float(primary_a), float(primary_c), str(default_source_label)) for _ in (hit_tables or ())
    ]
    source_rows = build_geometry_fit_simulated_peaks(
        hit_tables,
        image_shape=image_shape,
        native_sim_to_display_coords=native_sim_to_display_coords,
        peak_table_lattice=peak_table_lattice,
        primary_a=primary_a,
        primary_c=primary_c,
        default_source_label=default_source_label,
        round_pixel_centers=round_pixel_centers,
        allow_nominal_hkl_indices=allow_nominal_hkl_indices,
    )
    return list(source_rows), peak_table_lattice


def _signature_summary(signature: object) -> str | None:
    if signature is None:
        return None
    text = repr(signature)
    return text if len(text) <= 240 else text[:237] + "..."


def simulate_geometry_fit_hit_tables(
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
    image_size: int,
    param_set: Mapping[str, object] | dict[str, object],
    *,
    build_geometry_fit_central_mosaic_params: Callable[[Mapping[str, object]], Mapping[str, object]]
    | None = None,
    process_peaks_parallel: Callable[..., object],
    default_solve_q_steps: int,
    default_solve_q_rel_tol: float,
    default_solve_q_mode: int,
    required_branch_group_keys: Sequence[tuple[tuple[int, int, int], int | None, object | None]]
    | None = None,
    required_manual_fit_targets: Sequence[Mapping[str, object]] | None = None,
    hit_tables_only: bool = False,
) -> tuple[list[object], dict[str, object]]:
    """Simulate once and return raw hit tables with diagnostics."""

    params_local = dict(param_set)
    filtered_miller_array = np.asarray(miller_array, dtype=np.float64)
    filtered_intensity_array = np.asarray(intensity_array, dtype=np.float64)
    filtered_source_indices = (
        np.arange(int(filtered_miller_array.shape[0]), dtype=np.int64)
        if filtered_miller_array.ndim >= 1
        else np.empty((0,), dtype=np.int64)
    )
    required_branch_keys = list(required_branch_group_keys or ())
    required_targets = [
        dict(entry) for entry in (required_manual_fit_targets or ()) if isinstance(entry, Mapping)
    ]

    def _target_physical_hkl(target: Mapping[str, object]) -> tuple[int, int, int] | None:
        raw_hkl = target.get("hkl")
        try:
            physical_hkl = tuple(int(np.rint(float(value))) for value in raw_hkl[:3])
        except Exception:
            return None
        return physical_hkl if len(physical_hkl) == 3 else None

    def _trusted_full_reflection_identity(
        target: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        identity_candidates: list[Mapping[str, object]] = [target]
        for key in (
            "provider_selected_source_identity_canonical",
            "selected_source_identity_canonical",
            "manual_picker_selected_source_identity_canonical",
        ):
            identity = target.get(key)
            if isinstance(identity, Mapping):
                identity_candidates.append(identity)
        for identity in identity_candidates:
            try:
                source_index = int(identity.get("source_reflection_index"))
            except (TypeError, ValueError):
                continue
            if source_index < 0:
                continue
            return identity
        return None

    trusted_source_indices_by_hkl: dict[tuple[int, int, int], set[int]] = {}
    ignored_trusted_source_indices: list[dict[str, object]] = []
    for target in required_targets:
        physical_hkl = _target_physical_hkl(target)
        if physical_hkl is None:
            continue
        trusted_identity = _trusted_full_reflection_identity(target)
        if trusted_identity is not None:
            source_index = int(trusted_identity["source_reflection_index"])
            if source_index >= int(filtered_miller_array.shape[0]):
                ignored_trusted_source_indices.append(
                    {
                        "source_reflection_index": int(source_index),
                        "target_hkl": tuple(physical_hkl),
                        "reason": "index_out_of_range",
                    }
                )
            else:
                try:
                    indexed_hkl = tuple(
                        int(np.rint(float(value)))
                        for value in filtered_miller_array[source_index, :3]
                    )
                except Exception:
                    indexed_hkl = None
                if indexed_hkl == physical_hkl:
                    trusted_source_indices_by_hkl.setdefault(physical_hkl, set()).add(
                        int(source_index)
                    )
                else:
                    ignored_trusted_source_indices.append(
                        {
                            "source_reflection_index": int(source_index),
                            "target_hkl": tuple(physical_hkl),
                            "indexed_hkl": indexed_hkl,
                            "reason": "indexed_hkl_mismatch",
                        }
                    )
        raw_branch = target.get("source_branch_index")
        try:
            physical_branch = None if raw_branch is None else int(raw_branch)
        except Exception:
            physical_branch = None
        raw_q_group = target.get("q_group_key")
        physical_q_group = (
            tuple(raw_q_group) if isinstance(raw_q_group, (tuple, list)) and raw_q_group else None
        )
        physical_key = (physical_hkl, physical_branch, physical_q_group)
        if physical_key not in required_branch_keys:
            required_branch_keys.append(physical_key)
    diagnostics: dict[str, object] = {
        "stage": "simulate_hit_tables",
        "miller_shape": _array_shape_list(filtered_miller_array),
        "miller_count": _array_row_count(filtered_miller_array),
        "intensity_shape": _array_shape_list(filtered_intensity_array),
        "intensity_count": _array_row_count(filtered_intensity_array),
        "image_size": int(image_size),
        "targeted_simulation_supported": bool(required_branch_keys),
        "targeted_simulation_used": False,
        "process_peaks_called_for_trial_prediction": False,
        "process_peaks_call_count_for_trial_prediction": 0,
        "targeted_trusted_full_reflection_indices": sorted(
            {
                source_index
                for source_indices in trusted_source_indices_by_hkl.values()
                for source_index in source_indices
            }
        ),
        "targeted_ignored_trusted_full_reflection_indices": (ignored_trusted_source_indices),
    }

    def _raise_shared_centroid_validation(
        reason: str,
        **extra_diagnostics: object,
    ) -> NoReturn:
        exc = RuntimeError(str(reason))
        diagnostics.update(
            {
                "status": "shared_centroid_validation_exception",
                "centroid_source_authority": None,
                **extra_diagnostics,
                **_geometry_fit_exception_diagnostics(exc),
            }
        )
        raise exc

    def _shared_centroid_required_target_coverage(
        tables: Sequence[object],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        candidate_rows = [row for table in tables for row in geometry_reference_hit_rows(table)]
        covered: list[dict[str, object]] = []
        missing: list[dict[str, object]] = []
        for target in required_targets:
            physical_hkl = _target_physical_hkl(target)
            branch_token = _centroid_branch_token(target)
            required_branch = int(branch_token) if branch_token in {0, 1} else None
            if branch_token == "00l_collapsed":
                required_branch = None
            trusted_identity = _trusted_full_reflection_identity(target)
            raw_trusted_source_index = (
                int(trusted_identity["source_reflection_index"])
                if trusted_identity is not None
                else None
            )
            validated_source_indices = trusted_source_indices_by_hkl.get(
                physical_hkl,
                set(),
            )
            trusted_source_index = (
                raw_trusted_source_index
                if raw_trusted_source_index in validated_source_indices
                else None
            )
            target_identity = {
                "hkl": physical_hkl,
                "source_branch_index": required_branch,
                "source_reflection_index": trusted_source_index,
            }
            target_is_covered = False
            for row in candidate_rows:
                try:
                    candidate_mass = float(row[0])
                except Exception:
                    continue
                if not np.isfinite(candidate_mass) or candidate_mass <= 0.0:
                    continue
                try:
                    row_hkl = tuple(int(np.rint(float(value))) for value in row[4:7])
                except Exception:
                    continue
                if physical_hkl is None or row_hkl != physical_hkl:
                    continue
                if required_branch is not None:
                    row_branch = source_branch_index_from_phi_rad(row[3])
                    if row_branch != required_branch:
                        continue
                if trusted_source_index is not None:
                    source_index, _source_row_index, _sample_index = extract_hit_row_provenance(row)
                    if source_index != trusted_source_index:
                        continue
                target_is_covered = True
                break
            (covered if target_is_covered else missing).append(target_identity)
        return covered, missing

    use_miller_prefilter = bool(required_branch_keys)
    diagnostics["targeted_miller_hkl_inventory_before_filter"] = _geometry_fit_miller_hkl_inventory(
        filtered_miller_array
    )

    def _row_matches_required_hkl(row_hkl: Sequence[object]) -> bool:
        try:
            hkl = tuple(int(np.rint(float(v))) for v in row_hkl[:3])
        except Exception:
            return False
        return hkl in required_hkls

    def _row_allowed_by_trusted_source_identity(
        row_index: int,
        row_hkl: Sequence[object],
    ) -> bool:
        try:
            hkl = tuple(int(np.rint(float(v))) for v in row_hkl[:3])
        except Exception:
            return True
        trusted_indices = trusted_source_indices_by_hkl.get(hkl)
        return trusted_indices is None or int(row_index) in trusted_indices

    if (
        use_miller_prefilter
        and filtered_miller_array.ndim == 2
        and filtered_miller_array.shape[1] >= 3
    ):
        required_hkls = {tuple(key[0]) for key in required_branch_keys}
        diagnostics["targeted_required_hkl_count"] = int(len(required_hkls))
        diagnostics["targeted_required_branch_group_count"] = int(len(required_branch_keys))
        exact_hkl_mask = np.asarray(
            [_row_matches_required_hkl(row[:3]) for row in filtered_miller_array],
            dtype=bool,
        )
        trusted_source_mask = np.asarray(
            [
                _row_allowed_by_trusted_source_identity(
                    row_index,
                    filtered_miller_array[row_index, :3],
                )
                for row_index in range(filtered_miller_array.shape[0])
            ],
            dtype=bool,
        )
        if trusted_source_indices_by_hkl:
            diagnostics["targeted_miller_filter_policy"] = (
                "trusted_full_reflection_and_exact_hkl_before_simulation"
            )
            keep_mask = np.asarray(
                exact_hkl_mask & trusted_source_mask,
                dtype=bool,
            )
        else:
            diagnostics["targeted_miller_filter_policy"] = "exact_hkl_before_simulation"
            keep_mask = exact_hkl_mask
        if keep_mask.shape[0] != filtered_miller_array.shape[0]:
            raise RuntimeError("targeted geometry-fit HKL filter shape mismatch")
        if not np.any(keep_mask):
            raise RuntimeError("targeted geometry-fit HKL filter matched no reflections")
        filtered_miller_array = np.asarray(
            filtered_miller_array[keep_mask],
            dtype=np.float64,
        )
        filtered_intensity_array = np.asarray(
            filtered_intensity_array[keep_mask],
            dtype=np.float64,
        )
        filtered_source_indices = np.asarray(
            filtered_source_indices[keep_mask],
            dtype=np.int64,
        )
        diagnostics["targeted_simulation_used"] = True
        diagnostics["targeted_miller_count_after_filter"] = int(filtered_miller_array.shape[0])
    diagnostics["targeted_miller_hkl_inventory_after_filter"] = _geometry_fit_miller_hkl_inventory(
        filtered_miller_array
    )

    mosaic = dict(params_local.get("mosaic_params", {}))
    targeted_central_mosaic_used = False
    central_mosaic_build_error: str | None = None
    if use_miller_prefilter and callable(build_geometry_fit_central_mosaic_params):
        try:
            built_mosaic = build_geometry_fit_central_mosaic_params(params_local)
        except Exception as exc:
            built_mosaic = None
            central_mosaic_build_error = f"{type(exc).__name__}:{exc}"
        if not isinstance(built_mosaic, Mapping) or not built_mosaic:
            diagnostics["targeted_central_mosaic_used"] = False
            diagnostics["central_mosaic_build_error"] = (
                central_mosaic_build_error or "invalid_or_empty_central_mosaic"
            )
            diagnostics["status"] = "targeted_central_mosaic_build_failed"
            raise RuntimeError(
                "targeted geometry-fit central-ray mosaic build failed: "
                f"{diagnostics['central_mosaic_build_error']}"
            )
        mosaic = dict(built_mosaic)
        targeted_central_mosaic_used = True
    elif not mosaic and callable(build_geometry_fit_central_mosaic_params):
        try:
            built_mosaic = build_geometry_fit_central_mosaic_params(params_local)
        except Exception as exc:
            built_mosaic = None
            central_mosaic_build_error = f"{type(exc).__name__}:{exc}"
        if isinstance(built_mosaic, Mapping):
            mosaic = dict(built_mosaic)
    diagnostics["targeted_central_mosaic_used"] = bool(targeted_central_mosaic_used)
    diagnostics["central_mosaic_build_error"] = central_mosaic_build_error
    wavelength_array = mosaic.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic.get("wavelength_i_array")
    if wavelength_array is None:
        wavelength_array = np.full(
            int(np.size(mosaic.get("beam_x_array", []))),
            float(params_local.get("lambda", 1.0)),
            dtype=np.float64,
        )

    param_summary = {
        "a": _copy_simulation_diag_value(params_local.get("a")),
        "c": _copy_simulation_diag_value(params_local.get("c")),
        "lambda": _copy_simulation_diag_value(params_local.get("lambda")),
        "theta_initial": _copy_simulation_diag_value(params_local.get("theta_initial")),
        "gamma": _copy_simulation_diag_value(params_local.get("gamma")),
        "Gamma": _copy_simulation_diag_value(params_local.get("Gamma")),
        "center": _copy_simulation_diag_value(params_local.get("center")),
        "n2": _copy_simulation_diag_value(params_local.get("n2")),
    }
    mosaic_array_sizes = {
        "beam_x_array": _array_size(mosaic.get("beam_x_array")),
        "beam_y_array": _array_size(mosaic.get("beam_y_array")),
        "theta_array": _array_size(mosaic.get("theta_array")),
        "phi_array": _array_size(mosaic.get("phi_array")),
        "wavelength_array": _array_size(wavelength_array),
        "sample_weights": _array_size(mosaic.get("sample_weights")),
    }
    diagnostics.update(
        {
            "status": "ready",
            "param_summary": param_summary,
            "parameter_summary": _copy_simulation_diag_value(param_summary),
            "process_peaks_gamma": _copy_simulation_diag_value(params_local.get("gamma")),
            "process_peaks_Gamma": _copy_simulation_diag_value(params_local.get("Gamma")),
            "process_peaks_param_summary": _copy_simulation_diag_value(param_summary),
            "mosaic_array_sizes": mosaic_array_sizes,
        }
    )

    shared_representative_centroid_call = False
    try:
        beam_x_array = np.asarray(mosaic["beam_x_array"], dtype=np.float64)
        beam_y_array = np.asarray(mosaic["beam_y_array"], dtype=np.float64)
        theta_array = np.asarray(mosaic["theta_array"], dtype=np.float64)
        phi_array = np.asarray(mosaic["phi_array"], dtype=np.float64)
        wavelength_arg = np.asarray(wavelength_array, dtype=np.float64)
        beam_count = int(beam_x_array.reshape(-1).size)

        n2_override = mosaic.get("n2_sample_array")
        if n2_override is not None:
            try:
                n2_override_arr = np.ascontiguousarray(
                    np.asarray(n2_override, dtype=np.complex128).reshape(-1),
                    dtype=np.complex128,
                )
                if beam_count > 0 and n2_override_arr.size == beam_count:
                    n2_override = n2_override_arr
                else:
                    n2_override = None
            except Exception:
                n2_override = None

        targeted_hit_tables_only = bool(hit_tables_only and use_miller_prefilter)
        shared_representative_centroid_call = bool(
            targeted_hit_tables_only
            and required_targets
            and _geometry_fit_mosaic_is_locked_zero_central_carrier(
                mosaic,
                expected_wavelength=params_local.get("lambda"),
            )
        )
        sim_buffer = np.zeros(
            (1, 1) if targeted_hit_tables_only else (image_size, image_size),
            dtype=np.float64,
        )
        diagnostics["hit_tables_only"] = targeted_hit_tables_only
        diagnostics["process_peaks_called_for_trial_prediction"] = True
        diagnostics["process_peaks_call_count_for_trial_prediction"] = (
            int(diagnostics.get("process_peaks_call_count_for_trial_prediction") or 0) + 1
        )
        process_kwargs = {}
        collect_representatives = bool(use_miller_prefilter)
        representative_peak_indices = None
        if collect_representatives:
            representative_peak_indices = np.arange(
                int(filtered_miller_array.shape[0]),
                dtype=np.int64,
            )
        diagnostics["representative_collection_requested"] = bool(collect_representatives)
        process_args = (
            filtered_miller_array,
            filtered_intensity_array,
            image_size,
            float(params_local["a"]),
            float(params_local["c"]),
            wavelength_arg,
            sim_buffer,
            float(params_local["corto_detector"]),
            float(params_local["gamma"]),
            float(params_local["Gamma"]),
            float(params_local["chi"]),
            float(params_local.get("psi", 0.0)),
            float(params_local.get("psi_z", 0.0)),
            float(params_local["zs"]),
            float(params_local["zb"]),
            params_local["n2"],
            beam_x_array,
            beam_y_array,
            theta_array,
            phi_array,
            float(mosaic["sigma_mosaic_deg"]),
            float(mosaic["gamma_mosaic_deg"]),
            float(mosaic["eta"]),
            wavelength_arg,
            float(params_local["debye_x"]),
            float(params_local["debye_y"]),
            [float(params_local["center"][0]), float(params_local["center"][1])],
            float(params_local["theta_initial"]),
            float(params_local.get("cor_angle", 0.0)),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        process_kwargs.update(
            {
                "save_flag": 0,
                "solve_q_steps": int(mosaic.get("solve_q_steps", default_solve_q_steps)),
                "solve_q_rel_tol": float(mosaic.get("solve_q_rel_tol", default_solve_q_rel_tol)),
                "solve_q_mode": int(mosaic.get("solve_q_mode", default_solve_q_mode)),
                "sample_weights": mosaic.get("sample_weights"),
                "n2_sample_array_override": n2_override,
                "accumulate_image": not targeted_hit_tables_only,
                "collect_representative_hit_tables": collect_representatives,
                "representative_peak_indices": representative_peak_indices,
            }
        )
        if shared_representative_centroid_call:
            process_kwargs.update(
                {
                    "collect_hit_tables": True,
                    "hit_table_collection_mode": "all_weighted_candidates",
                    "events_per_beam_phase": int(_GEOMETRY_FIT_CENTROID_EVENTS_PER_BEAM_PHASE),
                    "numba_thread_count": 1,
                }
            )
        (
            _,
            hit_tables,
            _,
            _,
            _,
            _,
            returned_representative_hit_tables,
            weighted_event_stats,
        ) = process_peaks_parallel(
            *process_args,
            **process_kwargs,
        )
        diagnostics["process_peaks_weighted_event_stats"] = _copy_simulation_diag_value(
            weighted_event_stats
        )
        for key in (
            "parallel_backend",
            "n_solve_q_calls",
            "n_project_candidate_calls",
            "n_valid_candidates",
            "n_selected_events",
            "n_hit_table_rows",
            "n_nonempty_hit_tables",
            "n_representative_hit_tables",
            "n_raw_beam_phases",
            "n_effective_beam_phases",
            "pass1_total_mass",
            "phase_event_count_total",
        ):
            if key in weighted_event_stats:
                diagnostics[f"process_peaks_{key}"] = _copy_simulation_diag_value(
                    weighted_event_stats.get(key)
                )
        if shared_representative_centroid_call:
            if str(weighted_event_stats.get("hit_table_collection_mode") or "") != (
                "all_weighted_candidates"
            ):
                raise RuntimeError(
                    "geometry_fit_shared_centroid_call_did_not_return_all_candidates"
                )
    except Exception as exc:
        is_shared_validation_failure = bool(
            shared_representative_centroid_call
            and str(exc).startswith("geometry_fit_shared_centroid_call_")
        )
        diagnostics.update(
            {
                "status": (
                    "shared_centroid_validation_exception"
                    if is_shared_validation_failure
                    else "process_peaks_parallel_exception"
                ),
                **({"centroid_source_authority": None} if is_shared_validation_failure else {}),
                **_geometry_fit_exception_diagnostics(exc),
            }
        )
        raise

    full_hit_table_list = list(hit_tables or ())
    centroid_hit_table_list: list[object] | None = None
    representative_hit_tables = (
        returned_representative_hit_tables
        if bool(diagnostics.get("representative_collection_requested", False))
        else None
    )
    if representative_hit_tables is not None:
        representative_list = list(representative_hit_tables)
        if len(representative_list) != int(filtered_miller_array.shape[0]):
            raise RuntimeError(
                "geometry-fit representative hit-table count does not match the Miller set"
            )
        hit_table_list = representative_list
        centroid_hit_table_list = list(full_hit_table_list)
        diagnostics["representative_hit_tables_used"] = True
        diagnostics["representative_hit_table_count"] = int(len(representative_list))
    else:
        hit_table_list = list(full_hit_table_list)
        diagnostics["representative_hit_tables_used"] = False
        if bool(diagnostics.get("representative_collection_requested", False)):
            raise RuntimeError("geometry-fit representative hit tables are unavailable")
    if shared_representative_centroid_call:
        if not bool(diagnostics.get("representative_hit_tables_used", False)):
            _raise_shared_centroid_validation(
                "geometry_fit_shared_centroid_call_missing_representative_tables",
            )
        if not centroid_hit_table_list:
            _raise_shared_centroid_validation(
                "geometry_fit_shared_centroid_call_missing_candidate_tables",
            )
        centroid_finite_hit_row_total = int(
            sum(len(geometry_reference_hit_rows(table)) for table in centroid_hit_table_list)
        )
        if centroid_finite_hit_row_total <= 0:
            _raise_shared_centroid_validation(
                "geometry_fit_shared_centroid_call_missing_candidate_rows",
            )
        diagnostics.update(
            {
                "centroid_source_strategy": (
                    "weighted_candidate_moments_adaptive_grid_process_peaks"
                ),
                "centroid_source_authority": None,
                "centroid_ray_density": 1,
                "centroid_requested_ray_density": int(_GEOMETRY_FIT_CENTROID_RAY_DENSITY),
                "centroid_events_per_beam_phase": int(
                    process_kwargs.get(
                        "events_per_beam_phase",
                        _GEOMETRY_FIT_CENTROID_EVENTS_PER_BEAM_PHASE,
                    )
                ),
                "centroid_grid_sample_count": 1,
                "centroid_process_peaks_call_count_for_trial_prediction": 1,
                "additional_centroid_process_peaks_call_count": 0,
                "total_process_peaks_call_count_for_trial_prediction": 1,
                "representative_and_centroid_shared_process_peaks_call": True,
                "centroid_hit_table_count": int(len(centroid_hit_table_list)),
                "centroid_finite_hit_row_total": int(centroid_finite_hit_row_total),
                "centroid_grid_carrier_strategy": ("central_ray_single_candidate_moment"),
                "centroid_grid_unique_carrier_count": 1,
                "centroid_grid_phase_count": 1,
                "centroid_expected_event_count": 0,
                "centroid_quantile_event_budget_ignored": True,
                "centroid_candidate_weight_mode": ("continuous_projected_candidate_mass"),
                "centroid_grid_beam_divergence_enabled": False,
                "centroid_grid_collapsed_identical_central_carrier": True,
                "centroid_mosaic_sigma_deg": float(mosaic.get("sigma_mosaic_deg", 0.0)),
                "centroid_mosaic_gamma_deg": float(mosaic.get("gamma_mosaic_deg", 0.0)),
                "centroid_mosaic_eta": float(mosaic.get("eta", 0.0)),
                "centroid_mosaic_event_density_retained": True,
                "centroid_sampling_signature": (
                    "geometry_fit_shared_frozen_central_candidate_moment_v1",
                    1,
                    1,
                    float(np.asarray(theta_array, dtype=float).reshape(-1)[0]),
                    float(np.asarray(phi_array, dtype=float).reshape(-1)[0]),
                    float(np.asarray(wavelength_arg, dtype=float).reshape(-1)[0]),
                ),
            }
        )
    if use_miller_prefilter and required_targets and not shared_representative_centroid_call:
        try:
            density_centroid_hit_tables, density_centroid_diag = (
                _simulate_geometry_fit_density_centroid_hit_tables(
                    miller_array=filtered_miller_array,
                    intensity_array=filtered_intensity_array,
                    image_size=int(image_size),
                    params_local={**dict(params_local), "mosaic_params": dict(mosaic)},
                    process_peaks_parallel=process_peaks_parallel,
                    default_solve_q_steps=int(default_solve_q_steps),
                    default_solve_q_rel_tol=float(default_solve_q_rel_tol),
                    default_solve_q_mode=int(default_solve_q_mode),
                    source_indices=filtered_source_indices,
                )
            )
            if density_centroid_hit_tables:
                centroid_hit_table_list = list(density_centroid_hit_tables)
                diagnostics.update(density_centroid_diag)
                diagnostics["centroid_source_authority"] = (
                    "fresh_process_peaks_branch_peak_centroid"
                )
                diagnostics["total_process_peaks_call_count_for_trial_prediction"] = int(
                    diagnostics.get("process_peaks_call_count_for_trial_prediction", 1) or 1
                ) + int(
                    density_centroid_diag.get(
                        "centroid_process_peaks_call_count_for_trial_prediction",
                        0,
                    )
                    or 0
                )
            else:
                diagnostics["centroid_source_strategy"] = "density_grid_3x3_process_peaks_empty"
        except Exception as exc:
            # Generic/quantile hit tables are not a substitute for the fresh
            # all-candidate density cloud.  Leaving them attached here would
            # silently relabel a different objective surface as authoritative.
            centroid_hit_table_list = None
            diagnostics["centroid_source_strategy"] = "density_grid_3x3_process_peaks_failed"
            for key, value in _geometry_fit_exception_diagnostics(exc).items():
                diagnostics[f"centroid_{key}"] = value
    if use_miller_prefilter:
        hit_table_list = _geometry_fit_attach_targeted_hit_table_provenance(
            hit_table_list,
            filtered_source_indices,
        )
        if centroid_hit_table_list is not None:
            centroid_hit_table_list = _geometry_fit_attach_targeted_hit_table_provenance(
                centroid_hit_table_list,
                filtered_source_indices,
            )
        diagnostics["targeted_source_index_preview_after_filter"] = [
            int(value) for value in list(filtered_source_indices[:12])
        ]
        diagnostics["targeted_source_index_count_after_filter"] = int(len(filtered_source_indices))
    diagnostics["fresh_hit_table_hkl_inventory_before_filter"] = (
        _geometry_fit_hit_table_hkl_inventory(hit_table_list)
    )
    diagnostics["fresh_hit_table_hkl_branch_inventory_before_filter"] = (
        _geometry_fit_hit_table_hkl_branch_inventory(hit_table_list)
    )
    diagnostics["fresh_hit_table_source_index_inventory_before_filter"] = (
        _geometry_fit_hit_table_source_index_inventory(hit_table_list)
    )

    def _filter_required_hit_tables(tables: Sequence[object]) -> list[object]:
        required_hkls = {tuple(key[0]) for key in required_branch_keys}
        filtered_hit_tables: list[object] = []
        for table in tables:
            arr = np.asarray(table, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 7:
                continue
            try:
                table_hkl = (
                    int(np.rint(float(arr[0, 4]))),
                    int(np.rint(float(arr[0, 5]))),
                    int(np.rint(float(arr[0, 6]))),
                )
            except Exception:
                continue
            if table_hkl not in required_hkls:
                continue
            if required_hkls:
                row_mask = np.asarray(
                    [_row_matches_required_hkl(row[4:7]) for row in arr],
                    dtype=bool,
                )
                if np.any(row_mask):
                    filtered_hit_tables.append(np.asarray(arr[row_mask], dtype=np.float64).copy())
            else:
                filtered_hit_tables.append(np.asarray(arr, dtype=np.float64).copy())
        return filtered_hit_tables

    if required_branch_keys:
        hit_table_list = _filter_required_hit_tables(hit_table_list)
        if centroid_hit_table_list is not None:
            centroid_hit_table_list = _filter_required_hit_tables(centroid_hit_table_list)
        diagnostics["targeted_hit_table_branch_filter_deferred"] = True
        diagnostics["targeted_hit_table_filter_policy"] = "exact_hkl"
    if shared_representative_centroid_call:
        covered_targets, missing_targets = _shared_centroid_required_target_coverage(
            centroid_hit_table_list or ()
        )
        diagnostics.update(
            {
                "shared_centroid_required_target_count": int(len(required_targets)),
                "shared_centroid_covered_required_target_count": int(len(covered_targets)),
                "shared_centroid_missing_required_targets": missing_targets,
            }
        )
        if missing_targets:
            _raise_shared_centroid_validation(
                "geometry_fit_shared_centroid_call_incomplete_required_target_coverage",
                shared_centroid_required_target_count=int(len(required_targets)),
                shared_centroid_covered_required_target_count=int(len(covered_targets)),
                shared_centroid_missing_required_targets=missing_targets,
            )
        diagnostics["centroid_source_authority"] = "fresh_process_peaks_branch_peak_centroid"
    diagnostics["fresh_hit_table_hkl_inventory"] = _geometry_fit_hit_table_hkl_inventory(
        hit_table_list
    )
    diagnostics["fresh_hit_table_hkl_branch_inventory"] = (
        _geometry_fit_hit_table_hkl_branch_inventory(hit_table_list)
    )
    diagnostics["fresh_hit_table_source_index_inventory"] = (
        _geometry_fit_hit_table_source_index_inventory(hit_table_list)
    )
    hit_row_counts = [int(len(geometry_reference_hit_rows(table))) for table in hit_table_list]
    centroid_hit_row_counts = [
        int(len(geometry_reference_hit_rows(table))) for table in (centroid_hit_table_list or [])
    ]
    row_count_preview = [int(count) for count in hit_row_counts[:16]]
    diagnostics.update(
        {
            "status": ("success" if int(sum(hit_row_counts)) > 0 else "empty_hit_tables"),
            "hit_table_count": int(len(hit_table_list)),
            "hit_row_counts": hit_row_counts,
            "nonempty_hit_table_count": int(sum(1 for count in hit_row_counts if count > 0)),
            "finite_hit_row_total": int(sum(hit_row_counts)),
            "row_count_preview_per_table": row_count_preview,
            "row_count_preview_truncated": bool(len(hit_row_counts) > len(row_count_preview)),
            "projected_peak_count": int(sum(hit_row_counts)),
            "simulation_image_shape": _array_shape_list(sim_buffer),
            "simulation_image_nonzero_count": int(np.count_nonzero(sim_buffer)),
            "centroid_source_hit_tables_available": bool(centroid_hit_table_list),
            "centroid_source_hit_table_count": int(len(centroid_hit_table_list or [])),
            "centroid_source_finite_hit_row_total": int(sum(centroid_hit_row_counts)),
            "centroid_source_authority": (
                "fresh_process_peaks_branch_peak_centroid" if centroid_hit_table_list else None
            ),
        }
    )
    if centroid_hit_table_list:
        return (
            GeometryFitSourceRowsHitTables(
                hit_table_list,
                centroid_hit_tables=centroid_hit_table_list,
            ),
            diagnostics,
        )
    return hit_table_list, diagnostics


def make_runtime_geometry_fit_simulation_callbacks(
    *,
    build_geometry_fit_central_mosaic_params: (
        Callable[[Mapping[str, object]], Mapping[str, object]] | None
    ) = None,
    process_peaks_parallel: Callable[..., object],
    default_solve_q_steps: int,
    default_solve_q_rel_tol: float,
    default_solve_q_mode: int,
) -> GeometryFitSimulationRuntimeCallbacks:
    """Return live geometry-fit simulation callbacks from runtime value sources."""

    def _simulate_hit_tables(
        miller_array: np.ndarray,
        intensity_array: np.ndarray,
        image_size: int,
        param_set: Mapping[str, object] | dict[str, object],
        *,
        required_branch_group_keys: Sequence[tuple[tuple[int, int, int], int | None, object | None]]
        | None = None,
        required_manual_fit_targets: Sequence[Mapping[str, object]] | None = None,
        **_kwargs: object,
    ) -> tuple[list[object], dict[str, object]]:
        simulate_kwargs = {
            "build_geometry_fit_central_mosaic_params": (
                build_geometry_fit_central_mosaic_params
            ),
            "process_peaks_parallel": process_peaks_parallel,
            "default_solve_q_steps": default_solve_q_steps,
            "default_solve_q_rel_tol": default_solve_q_rel_tol,
            "default_solve_q_mode": default_solve_q_mode,
        }
        if required_branch_group_keys is not None:
            simulate_kwargs["required_branch_group_keys"] = required_branch_group_keys
        if required_manual_fit_targets is not None:
            simulate_kwargs["required_manual_fit_targets"] = required_manual_fit_targets
        if bool(_kwargs.get("hit_tables_only", False)):
            simulate_kwargs["hit_tables_only"] = True
        return simulate_geometry_fit_hit_tables(
            miller_array,
            intensity_array,
            image_size,
            param_set,
            **simulate_kwargs,
        )

    return GeometryFitSimulationRuntimeCallbacks(
        simulate_hit_tables=_simulate_hit_tables,
    )


def make_runtime_geometry_q_group_value_callbacks(
    *,
    simulation_runtime_state,
    preview_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names_factory: object,
    primary_a_factory: object,
    primary_c_factory: object,
    image_size_factory: object,
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ],
    caked_view_enabled_factory: object = False,
    project_peaks_to_current_view: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None = None,
) -> GeometryQGroupRuntimeValueCallbacks:
    """Return live Qr/Qz selector value callbacks from runtime sources."""

    _last_live_preview_cache_metadata_state: dict[str, object] = {}

    def _set_live_preview_cache_metadata(**fields: object) -> None:
        _last_live_preview_cache_metadata_state.clear()
        _last_live_preview_cache_metadata_state.update(fields)

    def _last_live_preview_cache_metadata() -> dict[str, object]:
        return dict(_last_live_preview_cache_metadata_state)

    def _current_geometry_fit_var_names() -> list[object]:
        raw_value = _resolve_runtime_value(current_geometry_fit_var_names_factory)
        if raw_value is None:
            return []
        if isinstance(raw_value, Sequence) and not isinstance(
            raw_value,
            (str, bytes),
        ):
            return list(raw_value)
        try:
            return list(raw_value)
        except Exception:
            return []

    def _primary_a() -> float:
        return _coerce_float(_resolve_runtime_value(primary_a_factory), float("nan"))

    def _primary_c() -> float:
        return _coerce_float(_resolve_runtime_value(primary_c_factory), float("nan"))

    def _detector_preview_image_shape() -> tuple[int, int]:
        stored_sim_image = getattr(simulation_runtime_state, "stored_sim_image", None)
        if stored_sim_image is not None:
            try:
                return tuple(int(v) for v in np.asarray(stored_sim_image).shape[:2])
            except Exception:
                pass
        image_size = max(
            0,
            _coerce_int(_resolve_runtime_value(image_size_factory), 0),
        )
        return (image_size, image_size)

    def _stored_peak_table_lattice_signature() -> object:
        return _geometry_q_group_signature_value(
            getattr(simulation_runtime_state, "stored_peak_table_lattice", None)
        )

    def _stored_q_group_content_signature() -> object:
        content_signature = getattr(
            simulation_runtime_state,
            "stored_q_group_content_signature",
            None,
        )
        if content_signature is not None:
            return _geometry_q_group_signature_value(content_signature)
        return _geometry_q_group_signature_value(
            getattr(simulation_runtime_state, "stored_max_positions_local", None)
        )

    def _stored_hit_table_q_group_signature() -> tuple[object, ...]:
        return (
            "geometry_q_group_entries",
            2,
            _geometry_q_group_signature_value(
                getattr(simulation_runtime_state, "stored_hit_table_signature", None)
            ),
            _stored_q_group_content_signature(),
            _stored_peak_table_lattice_signature(),
            _geometry_q_group_cache_scalar(_primary_a()),
            _geometry_q_group_cache_scalar(_primary_c()),
        )

    def _stored_hit_tables_available() -> bool:
        hit_tables = getattr(simulation_runtime_state, "stored_max_positions_local", None)
        if not isinstance(hit_tables, Sequence) or isinstance(hit_tables, (str, bytes)):
            return False
        return any(geometry_reference_hit_rows(table) for table in hit_tables)

    def _build_simulated_peaks_from_stored_hit_tables() -> list[dict[str, object]]:
        hit_tables = getattr(simulation_runtime_state, "stored_max_positions_local", None)
        if not isinstance(hit_tables, Sequence) or isinstance(hit_tables, (str, bytes)):
            return []
        return build_projected_geometry_fit_simulated_peaks(
            hit_tables,
            image_shape=_detector_preview_image_shape(),
            native_sim_to_display_coords=native_sim_to_display_coords,
            peak_table_lattice=getattr(
                simulation_runtime_state,
                "stored_peak_table_lattice",
                None,
            ),
            primary_a=_primary_a(),
            primary_c=_primary_c(),
            default_source_label="primary",
            allow_nominal_hkl_indices=True,
            project_peaks_to_current_view=project_peaks_to_current_view,
            caked_view_enabled=_caked_view_enabled(),
            profile_cache=getattr(simulation_runtime_state, "profile_cache", None),
        )

    def _caked_view_enabled() -> bool:
        try:
            return bool(_resolve_runtime_value(caked_view_enabled_factory))
        except Exception:
            return False

    def _build_live_preview_simulated_peaks() -> list[dict[str, object]]:
        rows = _build_simulated_peaks_from_stored_hit_tables()
        _set_live_preview_cache_metadata(
            cache_source="stored_hit_tables",
            stored_max_positions_table_count=int(
                len(getattr(simulation_runtime_state, "stored_max_positions_local", ()) or ())
            ),
            simulated_peak_count=int(len(rows)),
            reason="ready" if rows else "stored_hit_tables_missing_or_empty",
        )
        return rows

    def _filter_simulated_peaks(
        simulated_peaks: Sequence[dict[str, object]] | None,
    ) -> tuple[list[dict[str, object]], int, int]:
        return filter_geometry_fit_simulated_peaks(
            simulated_peaks,
            listed_keys=gui_controllers.listed_geometry_q_group_keys(q_group_state),
            q_group_state=q_group_state,
        )

    def _collapse_simulated_peaks(
        simulated_peaks: Sequence[dict[str, object]] | None,
    ) -> tuple[list[dict[str, object]], int]:
        return collapse_qr_qz_selection_peaks(
            simulated_peaks,
            profile_cache=getattr(simulation_runtime_state, "profile_cache", None),
        )

    def _build_entries_snapshot() -> list[dict[str, object]]:
        if _stored_hit_tables_available():
            cache_signature = _stored_hit_table_q_group_signature()
            if (
                getattr(
                    simulation_runtime_state,
                    "geometry_q_group_entries_cache_signature",
                    None,
                )
                == cache_signature
            ):
                cached_entries = gui_controllers.clone_geometry_q_group_entries(
                    getattr(
                        simulation_runtime_state,
                        "geometry_q_group_entries_cache",
                        [],
                    )
                )
                if cached_entries:
                    return cached_entries
            else:
                entries = build_geometry_q_group_entries(
                    getattr(simulation_runtime_state, "stored_max_positions_local", None),
                    peak_table_lattice=simulation_runtime_state.stored_peak_table_lattice,
                    peak_records=None,
                    primary_a=_primary_a(),
                    primary_c=_primary_c(),
                    allow_nominal_hkl_indices=True,
                )
                try:
                    simulation_runtime_state.geometry_q_group_entries_cache_signature = (
                        cache_signature
                    )
                    simulation_runtime_state.geometry_q_group_entries_cache = (
                        gui_controllers.clone_geometry_q_group_entries(entries)
                    )
                except Exception:
                    pass
                if entries:
                    return gui_controllers.clone_geometry_q_group_entries(entries)

        return []

    def _listed_entries() -> list[dict[str, object]]:
        return gui_controllers.listed_geometry_q_group_entries(q_group_state)

    def _listed_keys(
        entries: Sequence[dict[str, object]] | None = None,
    ) -> set[tuple[object, ...]]:
        return gui_controllers.listed_geometry_q_group_keys(q_group_state, entries)

    def _export_rows(
        entries: Sequence[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        return build_geometry_q_group_export_rows(
            q_group_state=q_group_state,
            entries=entries,
        )

    def _current_min_matches() -> int:
        return current_geometry_auto_match_min_matches(
            fit_config,
            _current_geometry_fit_var_names(),
        )

    def _excluded_count(
        entries: Sequence[dict[str, object]] | None = None,
    ) -> int:
        return geometry_q_group_excluded_count(
            q_group_state,
            entries,
        )

    def _build_window_status(
        entries: Sequence[dict[str, object]] | None = None,
    ) -> str:
        return build_geometry_q_group_window_status_text(
            q_group_state=q_group_state,
            fit_config=fit_config,
            current_geometry_fit_var_names=_current_geometry_fit_var_names(),
            entries=entries,
        )

    def _build_preview_exclude_button_label(
        entries: Sequence[dict[str, object]] | None = None,
    ) -> str:
        return build_geometry_preview_exclude_button_label(
            q_group_state=q_group_state,
            entries=entries,
        )

    def _live_preview_match_key(
        entry: dict[str, object] | None,
    ) -> tuple[object, ...] | None:
        return live_geometry_preview_match_key(entry)

    def _live_preview_match_hkl(
        entry: dict[str, object] | None,
    ) -> tuple[int, int, int] | None:
        return live_geometry_preview_match_hkl(entry)

    def _live_preview_match_is_excluded(entry: dict[str, object] | None) -> bool:
        return live_geometry_preview_match_is_excluded(
            preview_state,
            entry,
        )

    def _filter_live_preview_matches(
        matched_pairs: Sequence[dict[str, object]] | None,
    ) -> tuple[list[dict[str, object]], int]:
        return filter_live_geometry_preview_matches(
            preview_state,
            matched_pairs,
        )

    def _apply_live_preview_match_exclusions(
        matched_pairs: Sequence[dict[str, object]] | None,
        match_stats: dict[str, object] | None,
    ) -> tuple[list[dict[str, object]], dict[str, object], int]:
        return apply_live_geometry_preview_match_exclusions(
            preview_state,
            matched_pairs,
            match_stats,
        )

    return GeometryQGroupRuntimeValueCallbacks(
        build_live_preview_simulated_peaks_from_cache=(_build_live_preview_simulated_peaks),
        last_live_preview_cache_metadata=_last_live_preview_cache_metadata,
        filter_simulated_peaks=_filter_simulated_peaks,
        collapse_simulated_peaks=_collapse_simulated_peaks,
        build_entries_snapshot=_build_entries_snapshot,
        clone_entries=gui_controllers.clone_geometry_q_group_entries,
        listed_entries=_listed_entries,
        listed_keys=_listed_keys,
        key_from_jsonable=geometry_q_group_key_from_jsonable,
        export_rows=_export_rows,
        format_line=format_geometry_q_group_line,
        current_min_matches=_current_min_matches,
        excluded_count=_excluded_count,
        build_window_status=_build_window_status,
        build_preview_exclude_button_label=_build_preview_exclude_button_label,
        live_preview_match_key=_live_preview_match_key,
        live_preview_match_hkl=_live_preview_match_hkl,
        live_preview_match_is_excluded=_live_preview_match_is_excluded,
        filter_live_preview_matches=_filter_live_preview_matches,
        apply_live_preview_match_exclusions=_apply_live_preview_match_exclusions,
    )


def filter_geometry_fit_simulated_peaks(
    simulated_peaks: Sequence[dict[str, object]] | None,
    *,
    listed_keys: Sequence[tuple[object, ...]] | None = None,
    q_group_state,
) -> tuple[list[dict[str, object]], int, int]:
    """Apply the current Qr/Qz selector state to geometry-fit seeds."""

    filtered: list[dict[str, object]] = []
    excluded_count = 0
    available_keys: set[tuple[object, ...]] = set()
    listed_keys_local = set(listed_keys or ())
    restrict_to_listed = bool(listed_keys_local)
    if restrict_to_listed:
        available_keys = set(listed_keys_local)

    for raw_entry in simulated_peaks or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        group_key = geometry_q_group_key_from_entry(entry)
        if group_key is None:
            excluded_count += 1
            continue
        entry["q_group_key"] = group_key
        if restrict_to_listed and group_key not in listed_keys_local:
            excluded_count += 1
            continue
        if not restrict_to_listed:
            available_keys.add(group_key)
        if not gui_controllers.effective_q_group_enabled_state(entry, q_group_state):
            excluded_count += 1
            continue
        filtered.append(entry)

    return filtered, int(excluded_count), int(len(available_keys))


def _geometry_fit_branch_key(entry: Mapping[str, object]) -> int | str:
    hkl = _centroid_hkl_token(entry)
    if hkl is None:
        raise ValueError("geometry-fit simulated peak requires hkl")
    if hkl[0] == 0 and hkl[1] == 0 and hkl[2] > 0:
        return "00l"
    try:
        branch = int(entry["source_branch_index"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "non-00l geometry-fit simulated peak requires source_branch_index"
        ) from exc
    if branch not in {0, 1}:
        raise ValueError("source_branch_index must be 0 or 1")
    return branch


def collapse_geometry_fit_simulated_peaks(
    simulated_peaks: Sequence[dict[str, object]] | None,
    *,
    profile_cache: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, object]], int]:
    """Collapse each Qr/Qz group to one representative per physical branch."""

    grouped_entries: dict[
        tuple[object, ...],
        dict[int | str, list[dict[str, object]]],
    ] = {}
    ordered_groups: list[tuple[object, ...]] = []
    branch_order: dict[tuple[object, ...], list[int | str]] = {}

    for raw_entry in simulated_peaks or ():
        if not isinstance(raw_entry, Mapping):
            raise TypeError("geometry-fit simulated peaks must be mappings")
        entry = dict(raw_entry)
        group_key = gui_mosaic_top.normalize_q_group_key(entry.get("q_group_key"))
        if group_key is None:
            raise ValueError("geometry-fit simulated peak requires q_group_key")
        entry["q_group_key"] = group_key
        branch_key = _geometry_fit_branch_key(entry)
        entry = gui_mosaic_top.annotate_selection_metadata(
            entry,
            target_key=group_key,
            profile_cache=profile_cache,
        )
        entry.pop("mosaic_top_rank_key", None)

        if group_key not in grouped_entries:
            grouped_entries[group_key] = {}
            ordered_groups.append(group_key)
            branch_order[group_key] = []
        if branch_key not in grouped_entries[group_key]:
            grouped_entries[group_key][branch_key] = []
            branch_order[group_key].append(branch_key)
        grouped_entries[group_key][branch_key].append(entry)

    collapsed: list[dict[str, object]] = []
    collapsed_degenerate_count = 0
    for group_key in ordered_groups:
        for branch_key in branch_order[group_key]:
            cluster_entries = grouped_entries[group_key][branch_key]
            representative = gui_mosaic_top.select_mosaic_top_representative(
                cluster_entries,
                target_key=group_key,
                profile_cache=profile_cache,
            )
            if representative is None:
                raise RuntimeError("mosaic-top representative selection returned no row")
            merged = dict(representative)
            degenerate_hkls: list[tuple[int, int, int]] = []
            seen_hkls: set[tuple[int, int, int]] = set()
            total_weight = 0.0

            for entry in cluster_entries:
                hkl = _centroid_hkl_token(entry)
                if hkl is None:
                    raise ValueError("geometry-fit simulated peak requires hkl")
                if hkl not in seen_hkls:
                    seen_hkls.add(hkl)
                    degenerate_hkls.append(hkl)
                try:
                    weight = float(entry["weight"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        "geometry-fit simulated peak requires finite nonnegative weight"
                    ) from exc
                if not np.isfinite(weight) or weight < 0.0:
                    raise ValueError(
                        "geometry-fit simulated peak requires finite nonnegative weight"
                    )
                total_weight += weight

            merged["weight"] = total_weight
            merged["degenerate_count"] = len(cluster_entries)
            merged["degenerate_hkls"] = degenerate_hkls
            collapsed_degenerate_count += max(0, len(cluster_entries) - 1)
            collapsed.append(merged)

    return collapsed, collapsed_degenerate_count


def collapse_qr_qz_selection_peaks(
    simulated_peaks: Sequence[dict[str, object]] | None,
    *,
    profile_cache: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, object]], int]:
    """Collapse Qr/Qz UI selection rows to one ray per branch."""

    return collapse_geometry_fit_simulated_peaks(
        simulated_peaks,
        profile_cache=profile_cache,
    )


def format_geometry_q_group_line(entry: Mapping[str, object]) -> str:
    """Return a compact label for one Qr/Qz selector row."""

    qr_val = _coerce_float(entry.get("qr", np.nan), float("nan"))
    qz_val = _coerce_float(entry.get("qz", np.nan), float("nan"))
    total_intensity = _coerce_float(entry.get("total_intensity", 0.0), 0.0)
    peak_count = _coerce_int(entry.get("peak_count", 0), 0)
    source_label = str(entry.get("source_label", ""))
    ml_components = geometry_q_group_ml_from_key(entry)
    if ml_components is None:
        m_index = entry.get("m_index")
        l_index = entry.get("l_index", entry.get("gz_index"))
    else:
        m_index, l_index = ml_components
    hkl_items = entry.get("hkl_preview", [])
    if isinstance(hkl_items, np.ndarray):
        hkl_items = list(hkl_items)
    elif not isinstance(hkl_items, Sequence) or isinstance(hkl_items, str | bytes):
        hkl_items = []
    else:
        hkl_items = list(hkl_items)
    hkl_preview = ", ".join(str(hkl) for hkl in hkl_items[:3])
    if len(hkl_items) > 3:
        hkl_preview += ", ..."

    def _component_text(value: object) -> str:
        try:
            numeric = float(value)
        except Exception:
            return " n/a"
        if not np.isfinite(numeric):
            return " n/a"
        if abs(numeric - round(numeric)) <= 1.0e-6:
            return f"{int(round(numeric)):4d}"
        return f"{numeric:7.3f}"

    m_text = _component_text(m_index)
    l_text = _component_text(l_index)
    return (
        f"{source_label:<9}  "
        f"Qr={qr_val:8.2f}  "
        f"m={m_text}  "
        f"L={l_text}  "
        f"Qz={qz_val:8.2f}  "
        f"I={total_intensity:10.3f}  "
        f"hits={peak_count:4d}" + (f"  HKL={hkl_preview}" if hkl_preview else "")
    )


def _geometry_auto_match_config(
    fit_config: Mapping[str, object] | None,
) -> Mapping[str, object]:
    geometry_refine_cfg = (
        fit_config.get("geometry", {})
        if isinstance(
            fit_config,
            Mapping,
        )
        else {}
    )
    if not isinstance(geometry_refine_cfg, Mapping):
        geometry_refine_cfg = {}
    auto_match_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(auto_match_cfg, Mapping):
        return {}
    return auto_match_cfg


def current_geometry_auto_match_min_matches(
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
) -> int:
    """Return the current geometry auto-match minimum peak count."""

    auto_match_cfg = _geometry_auto_match_config(fit_config)
    default_min_matches = max(6, len(list(current_geometry_fit_var_names or ())) + 2)
    try:
        min_matches = int(auto_match_cfg.get("min_matches", default_min_matches))
    except Exception:
        min_matches = int(default_min_matches)
    return max(1, int(min_matches))


def _effective_disabled_geometry_q_group_keys(
    q_group_state,
    entries: Sequence[dict[str, object]] | None = None,
) -> set[tuple[object, ...]]:
    """Return listed structural child keys that are effectively disabled by masks."""

    rows = (
        list(entries)
        if entries is not None
        else gui_controllers.listed_geometry_q_group_entries(q_group_state)
    )
    return {
        key
        for entry in rows
        if isinstance(entry, Mapping)
        and (key := entry.get("key")) is not None
        and not gui_controllers.effective_q_group_enabled_state(entry, q_group_state)
    }


def build_live_geometry_preview_auto_match_config(
    fit_config: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return the normalized auto-match config used for live preview refreshes."""

    auto_match_cfg = _geometry_auto_match_config(fit_config)

    preview_auto_match_cfg = dict(auto_match_cfg)
    preview_auto_match_cfg["relax_on_low_matches"] = False
    search_radius = _coerce_float(auto_match_cfg.get("search_radius_px", 24.0), 24.0)
    preview_auto_match_cfg.setdefault(
        "context_margin_px",
        max(192.0, 8.0 * float(search_radius)),
    )
    return preview_auto_match_cfg


def geometry_q_group_excluded_count(
    q_group_state,
    entries: Sequence[dict[str, object]] | None = None,
) -> int:
    """Count excluded Qr/Qz rows, optionally scoped to one entry list."""

    return int(len(_effective_disabled_geometry_q_group_keys(q_group_state, entries)))


def build_geometry_preview_exclude_button_label(
    *,
    q_group_state,
    entries: Sequence[dict[str, object]] | None = None,
) -> str:
    """Return the toolbar label for the Qr/Qz preview-selector action."""

    label = "Choose Active Qr/Qz Groups"
    excluded_count = geometry_q_group_excluded_count(
        q_group_state,
        entries,
    )
    if excluded_count > 0:
        label += f" ({excluded_count} off)"
    return label


def build_geometry_q_group_window_status_text(
    *,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
    entries: Sequence[dict[str, object]] | None = None,
) -> str:
    """Build the summary text shown above the Qr/Qz selector rows."""

    rows = (
        list(entries)
        if entries is not None
        else gui_controllers.listed_geometry_q_group_entries(q_group_state)
    )
    total_count = len(rows)
    included_rows = gui_controllers.filter_enabled_q_group_rows(rows, q_group_state)
    selected_peak_count = int(
        sum(_coerce_int(entry.get("peak_count", 0), 0) for entry in included_rows)
    )
    total_peak_count = int(sum(_coerce_int(entry.get("peak_count", 0), 0) for entry in rows))
    min_matches = current_geometry_auto_match_min_matches(
        fit_config,
        current_geometry_fit_var_names,
    )
    shortfall = max(0, int(min_matches - selected_peak_count))
    selected_intensity = float(
        sum(_coerce_float(entry.get("total_intensity", 0.0), 0.0) for entry in included_rows)
    )
    total_intensity = float(
        sum(_coerce_float(entry.get("total_intensity", 0.0), 0.0) for entry in rows)
    )
    return (
        f"Included Qr/Qz groups: {len(included_rows)}/{total_count}  "
        f"Selected peaks: {selected_peak_count}/{total_peak_count}  "
        f"Need >= {min_matches}"
        + (f"  short {shortfall}" if shortfall > 0 else "  ready")
        + "\n"
        + f"Intensity={selected_intensity:.3f}/{total_intensity:.3f}  "
        + 'Listed peaks stay fixed until you press "Update Listed Peaks".'
    )


def update_geometry_q_group_window_status(
    *,
    view_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
    entries: Sequence[dict[str, object]] | None = None,
) -> None:
    """Refresh the summary label shown at the top of the selector window."""

    gui_views.set_geometry_q_group_status_text(
        view_state,
        build_geometry_q_group_window_status_text(
            q_group_state=q_group_state,
            fit_config=fit_config,
            current_geometry_fit_var_names=current_geometry_fit_var_names,
            entries=entries,
        ),
    )


def refresh_geometry_q_group_window(
    *,
    view_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
    on_toggle: Callable[[tuple[object, ...] | None, object], None],
) -> bool:
    """Redraw the Qr/Qz selector window from the stored manual snapshot."""

    entries = gui_controllers.listed_geometry_q_group_entries(q_group_state)
    return gui_views.refresh_geometry_q_group_window(
        view_state=view_state,
        entries=entries,
        excluded_q_groups=_effective_disabled_geometry_q_group_keys(
            q_group_state,
            entries,
        ),
        status_text=build_geometry_q_group_window_status_text(
            q_group_state=q_group_state,
            fit_config=fit_config,
            current_geometry_fit_var_names=current_geometry_fit_var_names,
            entries=entries,
        ),
        format_line=format_geometry_q_group_line,
        on_toggle=on_toggle,
        clear_row_vars=lambda: gui_controllers.clear_geometry_q_group_row_vars(
            q_group_state,
        ),
        register_row_var=lambda group_key, row_var: gui_controllers.set_geometry_q_group_row_var(
            q_group_state,
            group_key,
            row_var,
        ),
    )


def apply_geometry_q_group_checkbox_change(
    q_group_state,
    group_key: tuple[object, ...] | None,
    row_var: object,
    *,
    entries: Sequence[dict[str, object]] | None = None,
) -> str | None:
    """Apply one Qr/Qz include/exclude toggle from the selector window."""

    if group_key is None:
        return None
    if row_var is None:
        enabled = False
    else:
        try:
            enabled = bool(row_var.get())
        except Exception:
            enabled = bool(row_var)
    gui_controllers.set_geometry_q_group_row_enabled(
        q_group_state,
        group_key,
        enabled=enabled,
        entries=entries,
    )
    return "Enabled" if enabled else "Disabled"


def set_all_geometry_q_groups_enabled(
    q_group_state,
    *,
    enabled: bool,
) -> tuple[str, int]:
    """Enable or disable every currently listed Qr/Qz group."""

    entries = gui_controllers.listed_geometry_q_group_entries(q_group_state)
    if enabled:
        gui_controllers.clear_geometry_q_group_masks(q_group_state)
        action = "Enabled"
    else:
        gui_controllers.replace_geometry_q_group_masks(
            q_group_state,
            disabled_qr_sets=sorted(
                {
                    parent_key
                    for entry in entries
                    if (
                        parent_key := gui_controllers.qr_set_mask_key(
                            entry.get("key") if isinstance(entry, Mapping) else entry
                        )
                    )
                    is not None
                }
            ),
            disabled_qz_sections=[],
        )
        action = "Disabled"
    return action, len(entries)


def request_geometry_q_group_window_update(q_group_state) -> None:
    """Mark the Qr/Qz selector listing for refresh on the next update."""

    gui_controllers.request_geometry_q_group_refresh(q_group_state)


def replace_geometry_q_group_entries_snapshot_with_side_effects(
    *,
    q_group_state,
    entries: Sequence[dict[str, object]] | None,
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    update_geometry_preview_exclude_button_label: Callable[[], None],
) -> list[dict[str, object]]:
    """Replace the stored Qr/Qz snapshot and keep dependent state in sync."""

    gui_controllers.replace_geometry_q_group_cached_entries(
        q_group_state,
        entries,
    )
    current_entries = gui_controllers.listed_geometry_q_group_entries(q_group_state)
    gui_controllers.prune_geometry_q_group_masks(q_group_state, current_entries)
    invalidate_geometry_manual_pick_cache()
    update_geometry_preview_exclude_button_label()
    return gui_controllers.listed_geometry_q_group_entries(q_group_state)


def close_geometry_q_group_window(view_state, q_group_state) -> None:
    """Destroy the Qr/Qz selector window and clear its row-var map."""

    gui_views.close_geometry_q_group_window(view_state)
    gui_controllers.clear_geometry_q_group_row_vars(q_group_state)


def open_geometry_q_group_window(
    *,
    root,
    view_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
    on_toggle: Callable[[tuple[object, ...] | None, object], None],
    on_include_all: Callable[[], None],
    on_exclude_all: Callable[[], None],
    on_update_listed_peaks: Callable[[], None],
    on_save: Callable[[], None],
    on_load: Callable[[], None],
    on_close: Callable[[], None],
) -> bool:
    """Open and refresh the geometry Q-group selector window."""

    opened = gui_views.open_geometry_q_group_window(
        root=root,
        view_state=view_state,
        on_include_all=on_include_all,
        on_exclude_all=on_exclude_all,
        on_update_listed_peaks=on_update_listed_peaks,
        on_save=on_save,
        on_load=on_load,
        on_close=on_close,
    )
    refresh_geometry_q_group_window(
        view_state=view_state,
        q_group_state=q_group_state,
        fit_config=fit_config,
        current_geometry_fit_var_names=current_geometry_fit_var_names,
        on_toggle=on_toggle,
    )
    return opened


def geometry_q_group_key_to_jsonable(group_key: object) -> list[object] | None:
    """Convert one stable Qr/Qz group key into a JSON-safe list."""

    return gui_manual_geometry.geometry_q_group_key_to_jsonable(group_key)


def geometry_q_group_key_from_jsonable(value: object) -> tuple[object, ...] | None:
    """Rebuild one stable Qr/Qz group key from JSON-loaded data."""

    return gui_manual_geometry.geometry_q_group_key_from_jsonable(value)


def geometry_q_group_float_for_json(value: object) -> float | None:
    """Return a finite float for JSON export, or ``None`` when unavailable."""

    numeric = _coerce_float(value, float("nan"))
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def build_geometry_q_group_export_rows(
    *,
    q_group_state,
    entries: Sequence[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Build JSON export rows for the current Qr/Qz selector listing."""

    rows: list[dict[str, object]] = []
    source_entries = (
        list(entries)
        if entries is not None
        else gui_controllers.listed_geometry_q_group_entries(q_group_state)
    )
    for entry in source_entries:
        if not isinstance(entry, Mapping):
            continue
        group_key = entry.get("key")
        serialized_key = geometry_q_group_key_to_jsonable(group_key)
        if serialized_key is None:
            continue
        ml_components = geometry_q_group_ml_from_key(entry)
        if ml_components is None:
            m_index = serialized_key[2]
            l_index = serialized_key[3]
        else:
            m_index, l_index = ml_components
        hkl_preview = []
        for hkl_value in entry.get("hkl_preview", [])[:8]:
            if not isinstance(hkl_value, (list, tuple, np.ndarray)) or len(hkl_value) < 3:
                continue
            try:
                hkl_preview.append([int(hkl_value[0]), int(hkl_value[1]), int(hkl_value[2])])
            except Exception:
                continue
        row = {
            "key": serialized_key,
            "included": bool(gui_controllers.effective_q_group_enabled_state(entry, q_group_state)),
            "source_label": str(entry.get("source_label", "")),
            "qr": geometry_q_group_float_for_json(entry.get("qr", np.nan)),
            "qz": geometry_q_group_float_for_json(entry.get("qz", np.nan)),
            "m_index": m_index,
            "l_index": int(l_index),
            "gz_index": int(entry.get("gz_index", serialized_key[3])),
            "total_intensity": geometry_q_group_float_for_json(
                entry.get("total_intensity", np.nan)
            ),
            "peak_count": int(entry.get("peak_count", 0)),
            "hkl_preview": hkl_preview,
            "display_label": format_geometry_q_group_line(entry),
        }
        for key in ("phase_label", "structure_role", "overlap_identity"):
            if entry.get(key) is not None:
                value = entry.get(key)
                if isinstance(value, tuple):
                    value = list(value)
                row[key] = value
        rows.append(row)
    return rows


def build_geometry_q_group_save_payload(
    export_rows: Sequence[Mapping[str, object]],
    *,
    q_group_state,
    saved_at: str,
) -> dict[str, object]:
    """Build the JSON payload written by the Qr/Qz selector save action."""

    disabled_qr_sets, disabled_qz_sections = gui_controllers.normalized_geometry_q_group_masks(
        q_group_state
    )
    return {
        "type": "ra_sim.geometry_q_group_selection",
        "version": 2,
        "saved_at": str(saved_at),
        "row_count": int(len(export_rows)),
        "included_count": int(sum(1 for row in export_rows if bool(row.get("included", False)))),
        "disabled_qr_sets": [
            [str(source_label), int(m_index)]
            for source_label, m_index in sorted(
                disabled_qr_sets,
                key=lambda item: (str(item[0]), int(item[1])),
            )
        ],
        "disabled_qz_sections": [
            [str(source_label), int(m_index), int(gz_index)]
            for source_label, m_index, gz_index in sorted(
                disabled_qz_sections,
                key=lambda item: (str(item[0]), int(item[1]), int(item[2])),
            )
        ],
        "rows": [dict(row) for row in export_rows],
    }


def load_geometry_q_group_saved_state(
    payload: object,
) -> tuple[dict[str, object] | None, str | None]:
    """Validate one saved selector payload and rebuild its inclusion map."""

    if not isinstance(payload, Mapping):
        return None, "Invalid Qr/Qz peak list file: expected a JSON object."
    expected_keys = {
        "type",
        "version",
        "saved_at",
        "row_count",
        "included_count",
        "disabled_qr_sets",
        "disabled_qz_sections",
        "rows",
    }
    if set(payload) != expected_keys:
        return None, "Qr/Qz peak list must use the exact version 2 schema."
    if payload["type"] != "ra_sim.geometry_q_group_selection":
        return None, "Invalid Qr/Qz peak list file type."
    if type(payload["version"]) is not int or payload["version"] != 2:
        return None, "Only Qr/Qz peak list version 2 is supported."

    saved_rows = payload["rows"]
    if not isinstance(saved_rows, list):
        return None, "Qr/Qz peak list rows must be a list."

    saved_state: dict[tuple[object, ...], bool] = {}
    for row in saved_rows:
        if not isinstance(row, Mapping) or type(row.get("included")) is not bool:
            return None, "Every Qr/Qz peak list row must contain a boolean included field."
        group_key = geometry_q_group_key_from_jsonable(row.get("key"))
        if group_key is None:
            return None, "Every Qr/Qz peak list row must contain a current group key."
        saved_state[group_key] = row["included"]

    raw_disabled_qr_sets = payload["disabled_qr_sets"]
    raw_disabled_qz_sections = payload["disabled_qz_sections"]
    if not isinstance(raw_disabled_qr_sets, list) or not isinstance(
        raw_disabled_qz_sections, list
    ):
        return None, "Qr/Qz peak list masks must be lists."
    disabled_qr_sets = set()
    for raw_key in raw_disabled_qr_sets:
        parent_key = gui_controllers.qr_set_mask_key(raw_key)
        if parent_key is None:
            return None, "Qr/Qz peak list contains an invalid Qr mask key."
        disabled_qr_sets.add(parent_key)
    disabled_qz_sections = set()
    for raw_key in raw_disabled_qz_sections:
        child_key = gui_controllers.qz_section_mask_key(raw_key)
        if child_key is None:
            return None, "Qr/Qz peak list contains an invalid Qz mask key."
        disabled_qz_sections.add(child_key)
    if type(payload["row_count"]) is not int or payload["row_count"] != len(saved_rows):
        return None, "Qr/Qz peak list row_count does not match rows."
    included_count = sum(1 for included in saved_state.values() if included)
    if type(payload["included_count"]) is not int or payload["included_count"] != included_count:
        return None, "Qr/Qz peak list included_count does not match rows."
    if not saved_state:
        return None, "Loaded Qr/Qz peak list does not contain any valid rows."
    return {
        "saved_rows": saved_state,
        "disabled_qr_sets": sorted(
            disabled_qr_sets,
            key=lambda item: (str(item[0]), int(item[1])),
        ),
        "disabled_qz_sections": sorted(
            disabled_qz_sections,
            key=lambda item: (str(item[0]), int(item[1]), int(item[2])),
        ),
    }, None


def apply_loaded_geometry_q_group_saved_state(
    *,
    q_group_state,
    saved_state: Mapping[str, object] | None,
) -> tuple[dict[str, int] | None, str | None]:
    """Apply one loaded selector inclusion map to the current listed rows."""

    if not isinstance(saved_state, Mapping) or not saved_state:
        return None, "Loaded Qr/Qz peak list does not contain any valid rows."

    current_entries = gui_controllers.listed_geometry_q_group_entries(q_group_state)
    current_keys = [
        entry.get("key")
        for entry in current_entries
        if isinstance(entry, Mapping) and entry.get("key") is not None
    ]
    if not current_keys:
        return None, (
            "No listed Qr/Qz groups are available to match against the saved list. "
            'Press "Update Listed Peaks" first.'
        )

    saved_rows = saved_state.get("saved_rows", {})
    if not isinstance(saved_rows, Mapping):
        saved_rows = {}
    current_key_set = set(current_keys)
    matched_keys = current_key_set.intersection(saved_rows.keys())
    if saved_rows and not matched_keys:
        return None, "Loaded Qr/Qz peak list does not match any currently listed groups."

    gui_controllers.replace_geometry_q_group_masks(
        q_group_state,
        disabled_qr_sets=saved_state.get("disabled_qr_sets", ()),
        disabled_qz_sections=saved_state.get("disabled_qz_sections", ()),
    )

    gui_controllers.prune_geometry_q_group_masks(q_group_state, current_entries)
    enabled_rows = gui_controllers.filter_enabled_q_group_rows(current_entries, q_group_state)
    return {
        "matched_total": int(len(matched_keys) if saved_rows else len(current_keys)),
        "included_total": int(
            sum(1 for key in matched_keys if bool(saved_rows.get(key, False)))
            if saved_rows
            else len(enabled_rows)
        ),
        "current_only": int(sum(1 for key in current_keys if key not in saved_rows))
        if saved_rows
        else 0,
        "saved_only": int(sum(1 for key in saved_rows if key not in current_key_set))
        if saved_rows
        else 0,
    }, None


def live_geometry_preview_match_hkl(
    entry: dict[str, object] | None,
) -> tuple[int, int, int] | None:
    """Return the exact HKL tuple for one live preview match."""

    if not isinstance(entry, dict):
        return None
    hkl = entry.get("hkl")
    if not isinstance(hkl, (list, tuple)) or len(hkl) != 3:
        return None
    try:
        return tuple(int(value) for value in hkl)
    except (TypeError, ValueError):
        return None


def live_geometry_preview_match_key(
    entry: dict[str, object] | None,
) -> tuple[object, ...] | None:
    """Build the current exact exclusion key for one live preview match."""

    if not isinstance(entry, dict):
        return None
    hkl = live_geometry_preview_match_hkl(entry)
    if hkl is None:
        return None
    source_label = entry.get("source_label")
    source_reflection_index = _nonnegative_identity_index(
        entry.get("source_reflection_index")
    )
    source_row_index = _nonnegative_identity_index(entry.get("source_row_index"))
    try:
        source_branch = _geometry_fit_branch_key(entry)
    except ValueError:
        return None
    if (
        not isinstance(source_label, str)
        or not source_label
        or source_reflection_index is None
        or source_row_index is None
    ):
        return None
    return (
        "peak",
        source_label,
        source_reflection_index,
        source_row_index,
        source_branch,
        *hkl,
    )


def live_geometry_preview_match_is_excluded(
    preview_state,
    entry: dict[str, object] | None,
) -> bool:
    """Return whether the exact current match key is excluded."""

    key = live_geometry_preview_match_key(entry)
    return key is not None and key in preview_state.excluded_keys


def filter_live_geometry_preview_matches(
    preview_state,
    matched_pairs: Sequence[dict[str, object]] | None,
) -> tuple[list[dict[str, object]], int]:
    """Return live preview matches after applying exact user exclusions."""

    filtered: list[dict[str, object]] = []
    excluded_count = 0
    for raw_entry in matched_pairs or ():
        if not isinstance(raw_entry, dict):
            raise TypeError("Live preview matches must contain objects.")
        if live_geometry_preview_match_is_excluded(preview_state, raw_entry):
            excluded_count += 1
        else:
            filtered.append(dict(raw_entry))
    return filtered, excluded_count


def apply_live_geometry_preview_match_exclusions(
    preview_state,
    matched_pairs: Sequence[dict[str, object]] | None,
    match_stats: dict[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object], int]:
    """Apply live preview exclusions and refresh fit-facing summary metrics."""

    filtered_pairs, excluded_count = filter_live_geometry_preview_matches(
        preview_state,
        matched_pairs,
    )
    stats = dict(match_stats) if isinstance(match_stats, dict) else {}
    stats["excluded_count"] = int(excluded_count)
    stats["matched_count"] = int(len(filtered_pairs))
    stats["matched_after_exclusions"] = int(len(filtered_pairs))

    match_dists = np.asarray(
        [float(entry.get("distance_px", np.nan)) for entry in filtered_pairs],
        dtype=float,
    )
    match_dists = match_dists[np.isfinite(match_dists)]
    match_conf = np.asarray(
        [float(entry.get("confidence", np.nan)) for entry in filtered_pairs],
        dtype=float,
    )
    match_conf = match_conf[np.isfinite(match_conf)]

    stats["mean_match_distance_px"] = (
        float(np.mean(match_dists)) if match_dists.size else float("nan")
    )
    stats["p90_match_distance_px"] = (
        float(np.percentile(match_dists, 90.0)) if match_dists.size else float("nan")
    )
    stats["median_match_confidence"] = (
        float(np.median(match_conf)) if match_conf.size else float("nan")
    )
    return filtered_pairs, stats, int(excluded_count)


def build_empty_live_geometry_preview_overlay_state(
    *,
    signature: object,
    min_matches: int,
    max_display_markers: int,
    q_group_total: int,
    q_group_excluded: int,
    excluded_q_peaks: int,
    collapsed_degenerate_peaks: int = 0,
) -> dict[str, object]:
    """Return one empty cached live-preview overlay-state payload."""

    return {
        "signature": signature,
        "pairs": [],
        "simulated_count": 0,
        "min_matches": int(min_matches),
        "best_radius": float("nan"),
        "mean_dist": float("nan"),
        "p90_dist": float("nan"),
        "quality_fail": False,
        "max_display_markers": int(max_display_markers),
        "auto_match_attempts": [],
        "q_group_total": int(q_group_total),
        "q_group_excluded": int(q_group_excluded),
        "excluded_q_peaks": int(excluded_q_peaks),
        "collapsed_degenerate_peaks": int(collapsed_degenerate_peaks),
    }


def build_live_geometry_preview_overlay_state(
    *,
    signature: object,
    matched_pairs: Sequence[Mapping[str, object]] | None,
    match_stats: Mapping[str, object] | None,
    preview_auto_match_cfg: Mapping[str, object] | None,
    auto_match_attempts: Sequence[Mapping[str, object]] | None,
    min_matches: int,
    q_group_total: int,
    q_group_excluded: int,
    excluded_q_peaks: int,
    collapsed_degenerate_peaks: int = 0,
) -> dict[str, object]:
    """Return one cached live-preview overlay-state payload from match results."""

    match_stats_local = match_stats if isinstance(match_stats, Mapping) else {}
    preview_cfg = preview_auto_match_cfg if isinstance(preview_auto_match_cfg, Mapping) else {}
    matched_pairs_local = [dict(entry) for entry in matched_pairs or ()]
    attempts_local = [dict(entry) for entry in auto_match_attempts or ()]

    simulated_count = _coerce_int(
        match_stats_local.get("simulated_count", len(matched_pairs_local)),
        len(matched_pairs_local),
    )
    best_radius = _coerce_float(
        match_stats_local.get("search_radius_px", np.nan),
        float("nan"),
    )
    p90_dist = _coerce_float(
        match_stats_local.get("p90_match_distance_px", np.nan),
        float("nan"),
    )
    mean_dist = _coerce_float(
        match_stats_local.get("mean_match_distance_px", np.nan),
        float("nan"),
    )
    max_auto_p90 = _coerce_float(
        preview_cfg.get("max_p90_distance_px", 35.0),
        35.0,
    )
    max_auto_mean = _coerce_float(
        preview_cfg.get("max_mean_distance_px", 22.0),
        22.0,
    )
    quality_fail = bool(
        (np.isfinite(max_auto_p90) and np.isfinite(p90_dist) and p90_dist > max_auto_p90)
        or (np.isfinite(max_auto_mean) and np.isfinite(mean_dist) and mean_dist > max_auto_mean)
    )

    return {
        "signature": signature,
        "pairs": matched_pairs_local,
        "simulated_count": int(simulated_count),
        "min_matches": int(min_matches),
        "best_radius": float(best_radius),
        "mean_dist": float(mean_dist),
        "p90_dist": float(p90_dist),
        "quality_fail": bool(quality_fail),
        "max_display_markers": _coerce_int(
            preview_cfg.get("max_display_markers", 120),
            120,
        ),
        "auto_match_attempts": attempts_local,
        "q_group_total": int(q_group_total),
        "q_group_excluded": int(q_group_excluded),
        "excluded_q_peaks": int(excluded_q_peaks),
        "collapsed_degenerate_peaks": int(collapsed_degenerate_peaks),
    }


def build_live_geometry_preview_status_text(
    preview_overlay_state: object,
    *,
    active_pair_count: int,
    excluded_count: int,
) -> str:
    """Build the status line shown after one live-preview redraw."""

    pairs = list(getattr(preview_overlay_state, "pairs", []) or [])
    simulated_count = _coerce_int(
        getattr(preview_overlay_state, "simulated_count", 0),
        0,
    )
    min_matches = _coerce_int(
        getattr(preview_overlay_state, "min_matches", 0),
        0,
    )
    best_radius = _coerce_float(
        getattr(preview_overlay_state, "best_radius", np.nan),
        float("nan"),
    )
    mean_dist = _coerce_float(
        getattr(preview_overlay_state, "mean_dist", np.nan),
        float("nan"),
    )
    p90_dist = _coerce_float(
        getattr(preview_overlay_state, "p90_dist", np.nan),
        float("nan"),
    )
    quality_fail = bool(getattr(preview_overlay_state, "quality_fail", False))
    q_group_total = _coerce_int(
        getattr(preview_overlay_state, "q_group_total", 0),
        0,
    )
    q_group_excluded = _coerce_int(
        getattr(preview_overlay_state, "q_group_excluded", 0),
        0,
    )
    collapsed_deg = _coerce_int(
        getattr(preview_overlay_state, "collapsed_degenerate_peaks", 0),
        0,
    )
    max_display_markers = max(
        1,
        _coerce_int(getattr(preview_overlay_state, "max_display_markers", 120), 120),
    )
    shown_count = min(len(pairs), max_display_markers)

    summary = (
        "Live auto-match preview: "
        f"{int(active_pair_count)}/{simulated_count} active peaks "
        f"(need {min_matches}, local-peak match"
    )
    if np.isfinite(best_radius):
        summary += f", limit={best_radius:.1f}px"
    if np.isfinite(mean_dist):
        summary += f", mean={mean_dist:.1f}px"
    if np.isfinite(p90_dist):
        summary += f", p90={p90_dist:.1f}px"
    summary += ")."
    if int(excluded_count) > 0:
        summary += f" Excluded={int(excluded_count)}."
    if q_group_total > 0:
        summary += f" Qr/Qz groups on={max(0, q_group_total - q_group_excluded)}/{q_group_total}."
    if collapsed_deg > 0:
        summary += f" Degenerate collapsed={collapsed_deg}."
    if int(active_pair_count) < min_matches:
        summary += " Geometry fit would stop on the minimum-match gate."
    elif quality_fail:
        summary += " Geometry fit would stop on the quality gate."
    else:
        summary += " Geometry fit gates pass."
    if shown_count < len(pairs):
        summary += f" Showing {shown_count}/{len(pairs)} overlays."
    return summary


def render_live_geometry_preview_overlay_state(
    *,
    preview_state,
    draw_live_geometry_preview_overlay: Callable[..., None],
    filter_live_preview_matches: Callable[
        [Sequence[dict[str, object]]], tuple[Sequence[dict[str, object]], int]
    ],
    set_status_text: Callable[[str], None] | None = None,
    update_status: bool = True,
) -> bool:
    """Redraw the cached live-preview overlay and optionally refresh its status."""

    preview_overlay_state = getattr(preview_state, "overlay", None)
    pairs = list(getattr(preview_overlay_state, "pairs", []) or [])
    max_display_markers = _coerce_int(
        getattr(preview_overlay_state, "max_display_markers", 120),
        120,
    )
    draw_live_geometry_preview_overlay(
        pairs,
        max_display_markers=max_display_markers,
    )
    if not update_status:
        return bool(pairs)

    active_pairs, excluded_count = filter_live_preview_matches(pairs)
    _set_status_text(
        set_status_text,
        build_live_geometry_preview_status_text(
            preview_overlay_state,
            active_pair_count=len(list(active_pairs)),
            excluded_count=int(excluded_count),
        ),
    )
    return bool(list(active_pairs))


def runtime_live_geometry_preview_enabled(
    bindings: GeometryQGroupRuntimeBindings,
) -> bool:
    """Return whether the runtime live-preview checkbox is currently enabled."""

    try:
        return bool(bindings.live_geometry_preview_enabled())
    except Exception:
        return False


def draw_runtime_live_geometry_preview_overlay(
    bindings: GeometryQGroupRuntimeBindings,
    matched_pairs: Sequence[dict[str, object]] | None,
    *,
    max_display_markers: int = 120,
) -> None:
    """Draw the runtime live-preview overlay using the bound axis/artists."""

    if bindings.axis is None:
        return
    clear_geometry_preview_artists = bindings.clear_geometry_preview_artists or (lambda: None)
    draw_idle = bindings.draw_idle or (lambda: None)
    normalize_hkl_key = bindings.normalize_hkl_key or (lambda _value: None)
    live_preview_match_is_excluded = bindings.live_preview_match_is_excluded or (
        lambda _entry: False
    )
    gui_overlays.draw_live_geometry_preview_overlay(
        bindings.axis,
        matched_pairs,
        geometry_preview_artists=(
            bindings.geometry_preview_artists
            if bindings.geometry_preview_artists is not None
            else []
        ),
        clear_geometry_preview_artists=clear_geometry_preview_artists,
        draw_idle=draw_idle,
        normalize_hkl_key=normalize_hkl_key,
        live_preview_match_is_excluded=live_preview_match_is_excluded,
        max_display_markers=max_display_markers,
        show_pair_connectors=False,
    )


def render_runtime_live_geometry_preview_state(
    bindings: GeometryQGroupRuntimeBindings,
    *,
    update_status: bool = True,
) -> bool:
    """Redraw the cached runtime live-preview overlay from bound state."""

    filter_live_preview_matches = bindings.filter_live_preview_matches or (
        lambda pairs: (list(pairs or []), 0)
    )
    return render_live_geometry_preview_overlay_state(
        preview_state=bindings.preview_state,
        draw_live_geometry_preview_overlay=lambda pairs, *, max_display_markers: (
            draw_runtime_live_geometry_preview_overlay(
                bindings,
                pairs,
                max_display_markers=max_display_markers,
            )
        ),
        filter_live_preview_matches=filter_live_preview_matches,
        set_status_text=bindings.set_status_text,
        update_status=update_status,
    )


def resolve_runtime_live_geometry_preview_simulated_peaks(
    bindings: GeometryQGroupRuntimeBindings,
    *,
    update_status: bool = True,
) -> list[dict[str, object]] | None:
    """Return runtime live-preview peaks from the current simulation cache."""

    build_cached_peaks = bindings.build_live_preview_simulated_peaks_from_cache
    simulated_peaks = list(build_cached_peaks() or []) if callable(build_cached_peaks) else []
    if simulated_peaks:
        return simulated_peaks

    if callable(bindings.clear_geometry_preview_artists):
        bindings.clear_geometry_preview_artists()
    if update_status:
        _set_status_text(
            bindings.set_status_text,
            "Live auto-match preview unavailable: no simulated peaks are available.",
        )
    return None


def resolve_runtime_live_geometry_preview_background(
    bindings: GeometryQGroupRuntimeBindings,
    *,
    update_status: bool = True,
) -> object | None:
    """Return the display background when live preview is available."""

    if not runtime_live_geometry_preview_enabled(bindings):
        if callable(bindings.clear_geometry_preview_artists):
            bindings.clear_geometry_preview_artists()
        return None

    try:
        caked_view_enabled = bool(
            bindings.caked_view_enabled() if callable(bindings.caked_view_enabled) else False
        )
    except Exception:
        caked_view_enabled = False
    if caked_view_enabled:
        if callable(bindings.clear_geometry_preview_artists):
            bindings.clear_geometry_preview_artists()
        if update_status:
            _set_status_text(
                bindings.set_status_text,
                "Live auto-match preview unavailable in 2D caked view.",
            )
        return None

    display_background = (
        bindings.current_background_display_factory()
        if callable(bindings.current_background_display_factory)
        else None
    )
    if not bool(_resolve_runtime_value(bindings.background_visible)) or display_background is None:
        if callable(bindings.clear_geometry_preview_artists):
            bindings.clear_geometry_preview_artists()
        if update_status:
            _set_status_text(
                bindings.set_status_text,
                "Live auto-match preview unavailable: background image is hidden.",
            )
        return None
    return display_background


def resolve_runtime_live_geometry_preview_seed_state(
    bindings: GeometryQGroupRuntimeBindings,
    simulated_peaks: Sequence[dict[str, object]] | None,
    *,
    preview_auto_match_cfg: Mapping[str, object] | None,
    min_matches: int,
    signature: object,
    update_status: bool = True,
) -> tuple[list[dict[str, object]], int, int, int] | None:
    """Filter/collapse runtime live-preview seeds and handle empty-state exits."""

    filter_simulated_peaks = bindings.filter_simulated_peaks
    if callable(filter_simulated_peaks):
        filtered_peaks, excluded_q_peaks, q_group_total = filter_simulated_peaks(simulated_peaks)
    else:
        filtered_peaks = list(simulated_peaks or [])
        excluded_q_peaks = 0
        q_group_total = 0

    if not filtered_peaks:
        if callable(bindings.clear_geometry_preview_artists):
            bindings.clear_geometry_preview_artists()
        if update_status:
            _set_status_text(
                bindings.set_status_text,
                "Live auto-match preview unavailable: no Qr/Qz groups are selected.",
            )
        excluded_q_group_count = (
            _coerce_int(bindings.excluded_q_group_count(), 0)
            if callable(bindings.excluded_q_group_count)
            else 0
        )
        gui_controllers.replace_geometry_preview_overlay_state(
            bindings.preview_state,
            build_empty_live_geometry_preview_overlay_state(
                signature=signature,
                min_matches=int(min_matches),
                max_display_markers=_coerce_int(
                    (
                        preview_auto_match_cfg.get("max_display_markers", 120)
                        if isinstance(preview_auto_match_cfg, Mapping)
                        else 120
                    ),
                    120,
                ),
                q_group_total=int(q_group_total),
                q_group_excluded=int(excluded_q_group_count),
                excluded_q_peaks=int(excluded_q_peaks),
            ),
        )
        return None

    collapse_simulated_peaks = bindings.collapse_simulated_peaks
    if callable(collapse_simulated_peaks):
        collapsed_peaks, collapsed_deg_preview = collapse_simulated_peaks(filtered_peaks)
    else:
        collapsed_peaks = list(filtered_peaks)
        collapsed_deg_preview = 0

    if not collapsed_peaks:
        if callable(bindings.clear_geometry_preview_artists):
            bindings.clear_geometry_preview_artists()
        if update_status:
            _set_status_text(
                bindings.set_status_text,
                (
                    "Live auto-match preview unavailable: no geometry-fit seeds "
                    "remain after Qr/Qz collapse."
                ),
            )
        return None

    return (
        list(collapsed_peaks),
        int(excluded_q_peaks),
        int(q_group_total),
        int(collapsed_deg_preview),
    )


def apply_runtime_live_geometry_preview_match_results(
    bindings: GeometryQGroupRuntimeBindings,
    *,
    signature: object,
    matched_pairs: Sequence[Mapping[str, object]] | None,
    match_stats: Mapping[str, object] | None,
    preview_auto_match_cfg: Mapping[str, object] | None,
    auto_match_attempts: Sequence[Mapping[str, object]] | None,
    min_matches: int,
    q_group_total: int,
    excluded_q_peaks: int,
    collapsed_deg_preview: int = 0,
    update_status: bool = True,
) -> bool:
    """Store runtime live-preview match results and redraw the cached overlay."""

    q_group_excluded = (
        _coerce_int(bindings.excluded_q_group_count(), 0)
        if callable(bindings.excluded_q_group_count)
        else 0
    )
    gui_controllers.replace_geometry_preview_overlay_state(
        bindings.preview_state,
        build_live_geometry_preview_overlay_state(
            signature=signature,
            matched_pairs=matched_pairs,
            match_stats=match_stats,
            preview_auto_match_cfg=preview_auto_match_cfg,
            auto_match_attempts=auto_match_attempts,
            min_matches=int(min_matches),
            q_group_total=int(q_group_total),
            q_group_excluded=int(q_group_excluded),
            excluded_q_peaks=int(excluded_q_peaks),
            collapsed_degenerate_peaks=int(collapsed_deg_preview),
        ),
    )
    return render_runtime_live_geometry_preview_state(
        bindings,
        update_status=update_status,
    )


def distance_point_to_segment_sq(
    px: float,
    py: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    """Return squared distance from one display-space point to one segment."""

    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return (float(px) - float(x0)) ** 2 + (float(py) - float(y0)) ** 2

    t = ((float(px) - float(x0)) * dx + (float(py) - float(y0)) * dy) / (dx * dx + dy * dy)
    t = min(1.0, max(0.0, float(t)))
    cx = float(x0) + t * dx
    cy = float(y0) + t * dy
    return (float(px) - cx) ** 2 + (float(py) - cy) ** 2


def clear_live_geometry_preview_exclusions_with_side_effects(
    *,
    preview_state,
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    update_geometry_preview_exclude_button_label: Callable[[], None],
    refresh_geometry_q_group_window: Callable[[], None],
    live_geometry_preview_enabled: Callable[[], bool],
    refresh_live_geometry_preview: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
) -> None:
    """Clear preview exclusions and apply the dependent runtime side effects."""

    gui_controllers.clear_geometry_preview_excluded_keys(preview_state)
    invalidate_geometry_manual_pick_cache()
    update_geometry_preview_exclude_button_label()
    refresh_geometry_q_group_window()
    if live_geometry_preview_enabled():
        refresh_live_geometry_preview()
    else:
        _set_status_text(
            set_status_text,
            "Reset live preview pair exclusions.",
        )


def toggle_live_geometry_preview_exclusion_at(
    *,
    preview_state,
    col: float,
    row: float,
    live_preview_match_key: Callable[[dict[str, object] | None], tuple[object, ...] | None],
    live_preview_match_hkl: Callable[[dict[str, object] | None], tuple[int, int, int] | None],
    render_live_geometry_preview_state: Callable[[], None],
    max_distance_px: float,
    set_status_text: Callable[[str], None] | None = None,
) -> bool:
    """Toggle the nearest live-preview pair in or out of geometry fitting."""

    preview_overlay_state = getattr(preview_state, "overlay", None)
    pairs = list(getattr(preview_overlay_state, "pairs", []) or [])
    if not pairs:
        _set_status_text(
            set_status_text,
            "No live preview pairs are available to exclude.",
        )
        return False

    best_entry: dict[str, object] | None = None
    best_d2 = float("inf")
    for raw_entry in pairs:
        if not isinstance(raw_entry, dict):
            continue
        try:
            sim_col = float(raw_entry["sim_x"])
            sim_row = float(raw_entry["sim_y"])
            bg_col = float(raw_entry["x"])
            bg_row = float(raw_entry["y"])
        except Exception:
            continue
        d2 = min(
            (float(col) - sim_col) ** 2 + (float(row) - sim_row) ** 2,
            (float(col) - bg_col) ** 2 + (float(row) - bg_row) ** 2,
            distance_point_to_segment_sq(
                float(col),
                float(row),
                sim_col,
                sim_row,
                bg_col,
                bg_row,
            ),
        )
        if d2 < best_d2:
            best_d2 = d2
            best_entry = raw_entry

    if best_entry is None or best_d2 > float(max_distance_px) ** 2:
        _set_status_text(
            set_status_text,
            f"No preview pair within {float(max_distance_px):.0f}px to toggle.",
        )
        return False

    callback_key = live_preview_match_key(best_entry)
    key = callback_key
    hkl_key = live_preview_match_hkl(best_entry)
    if key is None or hkl_key is None:
        _set_status_text(
            set_status_text,
            "The selected preview pair cannot be excluded.",
        )
        return False

    if key in preview_state.excluded_keys:
        gui_controllers.set_geometry_preview_match_included(
            preview_state,
            key,
            included=True,
        )
        action = "Included"
    else:
        gui_controllers.set_geometry_preview_match_included(
            preview_state,
            key,
            included=False,
        )
        action = "Excluded"

    render_live_geometry_preview_state()
    _set_status_text(
        set_status_text,
        f"{action} live preview peak HKL={hkl_key} from geometry fit.",
    )
    return True


def make_runtime_geometry_q_group_bindings_factory(
    *,
    view_state,
    preview_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names_factory: object,
    build_entries_snapshot: Callable[[], Sequence[dict[str, object]] | None] | None = None,
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    update_geometry_preview_exclude_button_label: Callable[[], None],
    live_geometry_preview_enabled: Callable[[], bool],
    refresh_live_geometry_preview: Callable[[], None],
    set_hkl_pick_mode: Callable[[bool], None] | None = None,
    live_preview_match_key: (
        Callable[[dict[str, object] | None], tuple[object, ...] | None] | None
    ) = None,
    live_preview_match_hkl: (
        Callable[[dict[str, object] | None], tuple[int, int, int] | None] | None
    ) = None,
    render_live_geometry_preview_state: Callable[[], object] | None = None,
    clear_geometry_preview_artists: Callable[[], None] | None = None,
    preview_toggle_max_distance_px: float = 20.0,
    update_running_factory: object | None = None,
    has_cached_hit_tables_factory: object | None = None,
    build_live_preview_simulated_peaks_from_cache: Callable[[], list[dict[str, object]]]
    | None = None,
    filter_simulated_peaks: (
        Callable[
            [Sequence[dict[str, object]] | None],
            tuple[list[dict[str, object]], int, int],
        ]
        | None
    ) = None,
    collapse_simulated_peaks: Callable[..., tuple[list[dict[str, object]], int]] | None = None,
    excluded_q_group_count: Callable[[], int] | None = None,
    caked_view_enabled: Callable[[], bool] | None = None,
    background_visible_factory: object | None = None,
    current_background_display_factory: Callable[[], object] | None = None,
    axis: object | None = None,
    geometry_preview_artists: list[object] | None = None,
    draw_idle_factory: object | None = None,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] | None = None,
    live_preview_match_is_excluded: (Callable[[dict[str, object] | None], bool] | None) = None,
    filter_live_preview_matches: (
        Callable[[Sequence[dict[str, object]] | None], tuple[list[dict[str, object]], int]] | None
    ) = None,
    refresh_live_geometry_preview_quiet: Callable[[], None] | None = None,
    clear_last_simulation_signature: Callable[[], None] | None = None,
    schedule_update_factory: object | None = None,
    set_status_text_factory: object | None = None,
    file_dialog_dir_factory: object | None = None,
    asksaveasfilename: Callable[..., object] | None = None,
    askopenfilename: Callable[..., object] | None = None,
    warm_detector_mode_qr_caked_cache: Callable[[], object] | None = None,
) -> Callable[[], GeometryQGroupRuntimeBindings]:
    """Return a zero-arg factory for live geometry Q-group runtime bindings."""

    def _build_bindings() -> GeometryQGroupRuntimeBindings:
        return GeometryQGroupRuntimeBindings(
            view_state=view_state,
            preview_state=preview_state,
            q_group_state=q_group_state,
            fit_config=fit_config,
            current_geometry_fit_var_names_factory=current_geometry_fit_var_names_factory,
            build_entries_snapshot=build_entries_snapshot,
            invalidate_geometry_manual_pick_cache=invalidate_geometry_manual_pick_cache,
            update_geometry_preview_exclude_button_label=update_geometry_preview_exclude_button_label,
            live_geometry_preview_enabled=live_geometry_preview_enabled,
            refresh_live_geometry_preview=refresh_live_geometry_preview,
            set_hkl_pick_mode=set_hkl_pick_mode,
            live_preview_match_key=live_preview_match_key,
            live_preview_match_hkl=live_preview_match_hkl,
            render_live_geometry_preview_state=render_live_geometry_preview_state,
            clear_geometry_preview_artists=clear_geometry_preview_artists,
            preview_toggle_max_distance_px=_coerce_float(
                preview_toggle_max_distance_px,
                20.0,
            ),
            update_running=_resolve_runtime_value(update_running_factory),
            has_cached_hit_tables=_resolve_runtime_value(has_cached_hit_tables_factory),
            build_live_preview_simulated_peaks_from_cache=(
                build_live_preview_simulated_peaks_from_cache
            ),
            filter_simulated_peaks=filter_simulated_peaks,
            collapse_simulated_peaks=collapse_simulated_peaks or collapse_qr_qz_selection_peaks,
            excluded_q_group_count=excluded_q_group_count,
            caked_view_enabled=caked_view_enabled,
            background_visible=_resolve_runtime_value(background_visible_factory),
            current_background_display_factory=current_background_display_factory,
            axis=axis,
            geometry_preview_artists=geometry_preview_artists,
            draw_idle=_resolve_runtime_value(draw_idle_factory),
            normalize_hkl_key=normalize_hkl_key,
            live_preview_match_is_excluded=live_preview_match_is_excluded,
            filter_live_preview_matches=filter_live_preview_matches,
            refresh_live_geometry_preview_quiet=refresh_live_geometry_preview_quiet,
            clear_last_simulation_signature=clear_last_simulation_signature,
            schedule_update=_resolve_runtime_value(schedule_update_factory),
            set_status_text=_resolve_runtime_value(set_status_text_factory),
            file_dialog_dir=_resolve_runtime_value(file_dialog_dir_factory),
            asksaveasfilename=asksaveasfilename,
            askopenfilename=askopenfilename,
            warm_detector_mode_qr_caked_cache=warm_detector_mode_qr_caked_cache,
        )

    return _build_bindings


def update_runtime_geometry_q_group_window_status(
    bindings: GeometryQGroupRuntimeBindings,
    entries: Sequence[dict[str, object]] | None = None,
) -> None:
    """Refresh the runtime selector status text from live bindings."""

    update_geometry_q_group_window_status(
        view_state=bindings.view_state,
        q_group_state=bindings.q_group_state,
        fit_config=bindings.fit_config,
        current_geometry_fit_var_names=_runtime_geometry_fit_var_names(bindings),
        entries=entries,
    )


def on_runtime_geometry_q_group_checkbox_changed(
    bindings: GeometryQGroupRuntimeBindings,
    group_key: tuple[object, ...] | None,
    row_var: object,
) -> bool:
    """Apply one runtime Qr/Qz selector checkbox toggle."""

    return apply_geometry_q_group_checkbox_change_with_side_effects(
        q_group_state=bindings.q_group_state,
        group_key=group_key,
        row_var=row_var,
        invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
        update_geometry_preview_exclude_button_label=bindings.update_geometry_preview_exclude_button_label,
        update_geometry_q_group_window_status=lambda: update_runtime_geometry_q_group_window_status(
            bindings
        ),
        live_geometry_preview_enabled=bindings.live_geometry_preview_enabled,
        refresh_live_geometry_preview=bindings.refresh_live_geometry_preview,
        set_status_text=bindings.set_status_text,
        warm_detector_mode_qr_caked_cache=bindings.warm_detector_mode_qr_caked_cache,
    )


def refresh_runtime_geometry_q_group_window(
    bindings: GeometryQGroupRuntimeBindings,
) -> bool:
    """Redraw the runtime Qr/Qz selector window from live bindings."""

    return refresh_geometry_q_group_window(
        view_state=bindings.view_state,
        q_group_state=bindings.q_group_state,
        fit_config=bindings.fit_config,
        current_geometry_fit_var_names=_runtime_geometry_fit_var_names(bindings),
        on_toggle=lambda group_key, row_var: on_runtime_geometry_q_group_checkbox_changed(
            bindings,
            group_key,
            row_var,
        ),
    )


def set_all_geometry_q_groups_enabled_runtime(
    bindings: GeometryQGroupRuntimeBindings,
    *,
    enabled: bool,
) -> bool:
    """Apply one runtime bulk include/exclude selector action."""

    return set_all_geometry_q_groups_enabled_with_side_effects(
        q_group_state=bindings.q_group_state,
        enabled=enabled,
        invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
        update_geometry_preview_exclude_button_label=bindings.update_geometry_preview_exclude_button_label,
        refresh_geometry_q_group_window=lambda: refresh_runtime_geometry_q_group_window(bindings),
        live_geometry_preview_enabled=bindings.live_geometry_preview_enabled,
        refresh_live_geometry_preview=bindings.refresh_live_geometry_preview,
        set_status_text=bindings.set_status_text,
        warm_detector_mode_qr_caked_cache=bindings.warm_detector_mode_qr_caked_cache,
    )


def request_runtime_geometry_q_group_window_update(
    bindings: GeometryQGroupRuntimeBindings,
) -> None:
    """Request one runtime refresh of the listed Qr/Qz peaks."""

    request_geometry_q_group_window_update_with_side_effects(
        q_group_state=bindings.q_group_state,
        clear_last_simulation_signature=(
            bindings.clear_last_simulation_signature or (lambda: None)
        ),
        invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
        set_status_text=bindings.set_status_text,
        schedule_update=bindings.schedule_update or (lambda: None),
    )


def capture_runtime_geometry_q_group_entries_snapshot(
    bindings: GeometryQGroupRuntimeBindings,
) -> list[dict[str, object]]:
    """Rebuild and store the runtime Qr/Qz selector snapshot from live data."""

    build_entries_snapshot = bindings.build_entries_snapshot
    if not callable(build_entries_snapshot):
        return gui_controllers.listed_geometry_q_group_entries(bindings.q_group_state)

    built_entries = build_entries_snapshot()
    existing_entries = gui_controllers.listed_geometry_q_group_entries(bindings.q_group_state)
    preserve_imported_rows = (
        not built_entries
        and existing_entries
        and bool(
            getattr(
                bindings.q_group_state,
                "restored_q_group_rows_pending_live_refresh",
                False,
            )
        )
        and not bool(_resolve_runtime_value(bindings.has_cached_hit_tables))
    )
    if preserve_imported_rows:
        gui_controllers.request_geometry_q_group_refresh(bindings.q_group_state)
        if gui_views.geometry_q_group_window_open(bindings.view_state):
            refresh_runtime_geometry_q_group_window(bindings)
        return existing_entries

    entries = replace_geometry_q_group_entries_snapshot_with_side_effects(
        q_group_state=bindings.q_group_state,
        entries=built_entries,
        invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
        update_geometry_preview_exclude_button_label=(
            bindings.update_geometry_preview_exclude_button_label
        ),
    )
    bindings.q_group_state.restored_q_group_rows_pending_live_refresh = False
    if gui_views.geometry_q_group_window_open(bindings.view_state):
        refresh_runtime_geometry_q_group_window(bindings)
    return entries


def save_geometry_q_group_selection_runtime(
    bindings: GeometryQGroupRuntimeBindings,
) -> bool:
    """Export the runtime Qr/Qz selector state through the configured dialog."""

    if not callable(bindings.asksaveasfilename):
        _set_status_text(bindings.set_status_text, "Save Qr/Qz peak list unavailable.")
        return False
    return save_geometry_q_group_selection_with_dialog(
        q_group_state=bindings.q_group_state,
        file_dialog_dir=bindings.file_dialog_dir,
        asksaveasfilename=bindings.asksaveasfilename,
        set_status_text=bindings.set_status_text,
    )


def load_geometry_q_group_selection_runtime(
    bindings: GeometryQGroupRuntimeBindings,
) -> bool:
    """Import the runtime Qr/Qz selector state through the configured dialog."""

    if not callable(bindings.askopenfilename):
        _set_status_text(bindings.set_status_text, "Load Qr/Qz peak list unavailable.")
        return False
    return load_geometry_q_group_selection_with_dialog(
        q_group_state=bindings.q_group_state,
        file_dialog_dir=bindings.file_dialog_dir,
        askopenfilename=bindings.askopenfilename,
        invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
        update_geometry_preview_exclude_button_label=bindings.update_geometry_preview_exclude_button_label,
        refresh_geometry_q_group_window=lambda: refresh_runtime_geometry_q_group_window(bindings),
        live_geometry_preview_enabled=bindings.live_geometry_preview_enabled,
        refresh_live_geometry_preview=(
            bindings.refresh_live_geometry_preview_quiet or bindings.refresh_live_geometry_preview
        ),
        set_status_text=bindings.set_status_text,
        warm_detector_mode_qr_caked_cache=bindings.warm_detector_mode_qr_caked_cache,
    )


def close_runtime_geometry_q_group_window(
    bindings: GeometryQGroupRuntimeBindings,
) -> None:
    """Close the runtime Qr/Qz selector window using the live bindings."""

    close_geometry_q_group_window(bindings.view_state, bindings.q_group_state)


def open_runtime_geometry_q_group_window(
    *,
    root,
    bindings_factory: Callable[[], GeometryQGroupRuntimeBindings],
) -> bool:
    """Open the runtime Qr/Qz selector window and wire live callbacks."""

    bindings = bindings_factory()
    return open_geometry_q_group_window(
        root=root,
        view_state=bindings.view_state,
        q_group_state=bindings.q_group_state,
        fit_config=bindings.fit_config,
        current_geometry_fit_var_names=_runtime_geometry_fit_var_names(bindings),
        on_toggle=lambda group_key, row_var: on_runtime_geometry_q_group_checkbox_changed(
            bindings_factory(),
            group_key,
            row_var,
        ),
        on_include_all=lambda: set_all_geometry_q_groups_enabled_runtime(
            bindings_factory(),
            enabled=True,
        ),
        on_exclude_all=lambda: set_all_geometry_q_groups_enabled_runtime(
            bindings_factory(),
            enabled=False,
        ),
        on_update_listed_peaks=lambda: request_runtime_geometry_q_group_window_update(
            bindings_factory()
        ),
        on_save=lambda: save_geometry_q_group_selection_runtime(bindings_factory()),
        on_load=lambda: load_geometry_q_group_selection_runtime(bindings_factory()),
        on_close=lambda: close_runtime_geometry_q_group_window(bindings_factory()),
    )


def open_runtime_geometry_q_group_preview_exclusion_window(
    *,
    root,
    bindings_factory: Callable[[], GeometryQGroupRuntimeBindings],
) -> bool:
    """Open the runtime selector in preview-exclusion mode and report status."""

    opened = open_runtime_geometry_q_group_window(
        root=root,
        bindings_factory=bindings_factory,
    )
    _set_status_text(
        bindings_factory().set_status_text,
        (
            "Opened the Qr/Qz group selector. "
            "Unchecked rows are skipped during manual picking and geometry fitting."
        ),
    )
    return opened


def set_runtime_geometry_preview_exclude_mode(
    bindings: GeometryQGroupRuntimeBindings,
    enabled: bool,
    *,
    message: str | None = None,
) -> bool:
    """Apply one runtime preview-exclude mode toggle from live bindings."""

    changed = gui_controllers.set_geometry_preview_exclude_mode(
        bindings.preview_state,
        enabled,
    )
    if changed and callable(bindings.set_hkl_pick_mode):
        bindings.set_hkl_pick_mode(False)
    bindings.update_geometry_preview_exclude_button_label()
    if message:
        _set_status_text(bindings.set_status_text, message)
    return bool(changed)


def clear_runtime_live_geometry_preview_exclusions(
    bindings: GeometryQGroupRuntimeBindings,
) -> None:
    """Clear runtime live-preview exclusions through the bound workflow surface."""

    clear_live_geometry_preview_exclusions_with_side_effects(
        preview_state=bindings.preview_state,
        invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
        update_geometry_preview_exclude_button_label=bindings.update_geometry_preview_exclude_button_label,
        refresh_geometry_q_group_window=lambda: refresh_runtime_geometry_q_group_window(bindings),
        live_geometry_preview_enabled=bindings.live_geometry_preview_enabled,
        refresh_live_geometry_preview=bindings.refresh_live_geometry_preview,
        set_status_text=bindings.set_status_text,
    )


def toggle_runtime_live_geometry_preview_exclusion_at(
    bindings: GeometryQGroupRuntimeBindings,
    col: float,
    row: float,
) -> bool:
    """Toggle one runtime preview exclusion using the live binding surface."""

    if not (
        callable(bindings.live_preview_match_key)
        and callable(bindings.live_preview_match_hkl)
        and callable(bindings.render_live_geometry_preview_state)
    ):
        _set_status_text(
            bindings.set_status_text,
            "Live preview exclusion toggle unavailable.",
        )
        return False

    return toggle_live_geometry_preview_exclusion_at(
        preview_state=bindings.preview_state,
        col=col,
        row=row,
        live_preview_match_key=bindings.live_preview_match_key,
        live_preview_match_hkl=bindings.live_preview_match_hkl,
        render_live_geometry_preview_state=bindings.render_live_geometry_preview_state,
        max_distance_px=_coerce_float(bindings.preview_toggle_max_distance_px, 20.0),
        set_status_text=bindings.set_status_text,
    )


def toggle_live_geometry_preview_with_side_effects(
    *,
    enabled: bool,
    disable_preview_exclude_mode: Callable[[], None],
    clear_geometry_preview_artists: Callable[[], None],
    open_geometry_q_group_window: Callable[[], object],
    update_running: bool,
    has_cached_hit_tables: bool,
    schedule_update: Callable[[], None],
    refresh_live_geometry_preview: Callable[[], bool],
    set_status_text: Callable[[str], None] | None = None,
) -> bool:
    """Apply the live-preview checkbox action and its follow-on workflow."""

    if not enabled:
        disable_preview_exclude_mode()
        clear_geometry_preview_artists()
        _set_status_text(
            set_status_text,
            "Live auto-match preview disabled.",
        )
        return False

    open_geometry_q_group_window()
    if bool(update_running) or not bool(has_cached_hit_tables):
        schedule_update()
        return True

    refreshed = bool(refresh_live_geometry_preview())
    if not refreshed:
        schedule_update()
    return refreshed


def toggle_runtime_live_geometry_preview(
    bindings: GeometryQGroupRuntimeBindings,
    *,
    root,
    bindings_factory: Callable[[], GeometryQGroupRuntimeBindings],
) -> bool:
    """Apply one runtime live-preview checkbox action through live bindings."""

    return toggle_live_geometry_preview_with_side_effects(
        enabled=bool(_resolve_runtime_value(bindings.live_geometry_preview_enabled)),
        disable_preview_exclude_mode=lambda: set_runtime_geometry_preview_exclude_mode(
            bindings,
            False,
        ),
        clear_geometry_preview_artists=(bindings.clear_geometry_preview_artists or (lambda: None)),
        open_geometry_q_group_window=lambda: open_runtime_geometry_q_group_window(
            root=root,
            bindings_factory=bindings_factory,
        ),
        update_running=bool(_resolve_runtime_value(bindings.update_running)),
        has_cached_hit_tables=bool(_resolve_runtime_value(bindings.has_cached_hit_tables)),
        schedule_update=bindings.schedule_update or (lambda: None),
        refresh_live_geometry_preview=bindings.refresh_live_geometry_preview,
        set_status_text=bindings.set_status_text,
    )


def make_runtime_geometry_q_group_callbacks(
    *,
    root,
    bindings_factory: Callable[[], GeometryQGroupRuntimeBindings],
) -> GeometryQGroupRuntimeCallbacks:
    """Return bound zero-arg callbacks for the runtime Qr/Qz selector workflow."""

    def _set_preview_exclude_mode(
        enabled: bool,
        message: str | None = None,
    ) -> bool:
        return set_runtime_geometry_preview_exclude_mode(
            bindings_factory(),
            enabled,
            message=message,
        )

    return GeometryQGroupRuntimeCallbacks(
        update_window_status=lambda entries=None: update_runtime_geometry_q_group_window_status(
            bindings_factory(),
            entries=entries,
        ),
        refresh_window=lambda: refresh_runtime_geometry_q_group_window(bindings_factory()),
        on_toggle=lambda group_key, row_var: on_runtime_geometry_q_group_checkbox_changed(
            bindings_factory(),
            group_key,
            row_var,
        ),
        include_all=lambda: set_all_geometry_q_groups_enabled_runtime(
            bindings_factory(),
            enabled=True,
        ),
        exclude_all=lambda: set_all_geometry_q_groups_enabled_runtime(
            bindings_factory(),
            enabled=False,
        ),
        update_listed_peaks=lambda: request_runtime_geometry_q_group_window_update(
            bindings_factory()
        ),
        save_selection=lambda: save_geometry_q_group_selection_runtime(bindings_factory()),
        load_selection=lambda: load_geometry_q_group_selection_runtime(bindings_factory()),
        close_window=lambda: close_runtime_geometry_q_group_window(bindings_factory()),
        open_window=lambda: open_runtime_geometry_q_group_window(
            root=root,
            bindings_factory=bindings_factory,
        ),
        open_preview_exclusion_window=lambda: (
            open_runtime_geometry_q_group_preview_exclusion_window(
                root=root,
                bindings_factory=bindings_factory,
            )
        ),
        set_preview_exclude_mode=_set_preview_exclude_mode,
        clear_preview_exclusions=lambda: clear_runtime_live_geometry_preview_exclusions(
            bindings_factory()
        ),
        toggle_preview_exclusion_at=lambda col, row: (
            toggle_runtime_live_geometry_preview_exclusion_at(
                bindings_factory(),
                col,
                row,
            )
        ),
        toggle_live_preview=lambda: toggle_runtime_live_geometry_preview(
            bindings_factory(),
            root=root,
            bindings_factory=bindings_factory,
        ),
        live_preview_enabled=lambda: runtime_live_geometry_preview_enabled(bindings_factory()),
        render_live_preview_state=lambda update_status=True: (
            render_runtime_live_geometry_preview_state(
                bindings_factory(),
                update_status=update_status,
            )
        ),
    )


def apply_geometry_q_group_checkbox_change_with_side_effects(
    *,
    q_group_state,
    group_key: tuple[object, ...] | None,
    row_var: object,
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    update_geometry_preview_exclude_button_label: Callable[[], None],
    update_geometry_q_group_window_status: Callable[[], None],
    live_geometry_preview_enabled: Callable[[], bool],
    refresh_live_geometry_preview: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
    warm_detector_mode_qr_caked_cache: Callable[[], object] | None = None,
) -> bool:
    """Apply one checkbox toggle and the dependent live-preview/status effects."""

    action = apply_geometry_q_group_checkbox_change(
        q_group_state,
        group_key,
        row_var,
        entries=gui_controllers.listed_geometry_q_group_entries(q_group_state),
    )
    if action is None:
        return False

    invalidate_geometry_manual_pick_cache()
    update_geometry_preview_exclude_button_label()
    update_geometry_q_group_window_status()

    if live_geometry_preview_enabled():
        refresh_live_geometry_preview()
    else:
        _set_status_text(
            set_status_text,
            f"{action} one Qr/Qz group.",
        )
    _warm_detector_mode_qr_caked_cache(
        warm_detector_mode_qr_caked_cache,
        set_status_text=set_status_text,
    )
    return True


def _warm_detector_mode_qr_caked_cache(
    callback: Callable[[], object] | None,
    *,
    set_status_text: Callable[[str], None] | None = None,
) -> bool:
    if not callable(callback):
        return False
    try:
        return bool(callback())
    except Exception as exc:
        _set_status_text(
            set_status_text,
            f"Qr/Qz caked cache warm failed: {exc}",
        )
        return False


def set_all_geometry_q_groups_enabled_with_side_effects(
    *,
    q_group_state,
    enabled: bool,
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    update_geometry_preview_exclude_button_label: Callable[[], None],
    refresh_geometry_q_group_window: Callable[[], None],
    live_geometry_preview_enabled: Callable[[], bool],
    refresh_live_geometry_preview: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
    warm_detector_mode_qr_caked_cache: Callable[[], object] | None = None,
) -> bool:
    """Apply a bulk include/exclude action and the dependent side effects."""

    action, count = set_all_geometry_q_groups_enabled(
        q_group_state,
        enabled=enabled,
    )
    if count <= 0:
        _set_status_text(
            set_status_text,
            'No listed Qr/Qz groups are available. Press "Update Listed Peaks" first.',
        )
        return False

    invalidate_geometry_manual_pick_cache()
    update_geometry_preview_exclude_button_label()
    refresh_geometry_q_group_window()

    if live_geometry_preview_enabled():
        refresh_live_geometry_preview()
    else:
        _set_status_text(
            set_status_text,
            f"{action} {count} Qr/Qz groups.",
        )
    _warm_detector_mode_qr_caked_cache(
        warm_detector_mode_qr_caked_cache,
        set_status_text=set_status_text,
    )
    return True


def request_geometry_q_group_window_update_with_side_effects(
    *,
    q_group_state,
    clear_last_simulation_signature: Callable[[], None],
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
    schedule_update: Callable[[], None],
) -> None:
    """Request a listed-peak refresh and trigger the dependent runtime effects."""

    request_geometry_q_group_window_update(q_group_state)
    clear_last_simulation_signature()
    invalidate_geometry_manual_pick_cache()
    _set_status_text(
        set_status_text,
        "Updating listed Qr/Qz peaks from the current simulation...",
    )
    schedule_update()


def save_geometry_q_group_selection_with_dialog(
    *,
    q_group_state,
    file_dialog_dir: object,
    asksaveasfilename: Callable[..., object],
    set_status_text: Callable[[str], None] | None = None,
    save_payload: Callable[[str, Mapping[str, object]], None] | None = None,
    now: Callable[[], datetime] | None = None,
) -> bool:
    """Export the current selector rows through a save-file dialog."""

    export_rows = build_geometry_q_group_export_rows(
        q_group_state=q_group_state,
    )
    if not export_rows:
        _set_status_text(
            set_status_text,
            "No listed Qr/Qz groups are available to save. Press Update Listed Peaks first.",
        )
        return False

    now_value = now() if callable(now) else datetime.now()
    file_path = asksaveasfilename(
        title="Save Geometry Fit Qr/Qz Peak List",
        initialdir=str(file_dialog_dir),
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialfile=(f"geometry_q_groups_{now_value.strftime('%Y%m%d_%H%M%S')}.json"),
    )
    if not file_path:
        _set_status_text(set_status_text, "Save Qr/Qz peak list canceled.")
        return False

    payload = build_geometry_q_group_save_payload(
        export_rows,
        q_group_state=q_group_state,
        saved_at=now_value.isoformat(timespec="seconds"),
    )
    try:
        if save_payload is None:
            with open(str(file_path), "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        else:
            save_payload(str(file_path), payload)
    except Exception as exc:
        _set_status_text(
            set_status_text,
            f"Failed to save Qr/Qz peak list: {exc}",
        )
        return False

    _set_status_text(
        set_status_text,
        f"Saved {len(export_rows)} Qr/Qz groups to {file_path}",
    )
    return True


def load_geometry_q_group_selection_with_dialog(
    *,
    q_group_state,
    file_dialog_dir: object,
    askopenfilename: Callable[..., object],
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    update_geometry_preview_exclude_button_label: Callable[[], None],
    refresh_geometry_q_group_window: Callable[[], None],
    live_geometry_preview_enabled: Callable[[], bool],
    refresh_live_geometry_preview: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
    load_payload: Callable[[str], object] | None = None,
    warm_detector_mode_qr_caked_cache: Callable[[], object] | None = None,
) -> bool:
    """Import selector rows through an open-file dialog and apply them."""

    file_path = askopenfilename(
        title="Load Geometry Fit Qr/Qz Peak List",
        initialdir=str(file_dialog_dir),
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    if not file_path:
        _set_status_text(set_status_text, "Load Qr/Qz peak list canceled.")
        return False

    try:
        if load_payload is None:
            with open(str(file_path), encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = load_payload(str(file_path))
    except Exception as exc:
        _set_status_text(
            set_status_text,
            f"Failed to load Qr/Qz peak list: {exc}",
        )
        return False

    saved_state, error = load_geometry_q_group_saved_state(payload)
    if error is not None:
        _set_status_text(set_status_text, error)
        return False

    summary, error = apply_loaded_geometry_q_group_saved_state(
        q_group_state=q_group_state,
        saved_state=saved_state,
    )
    if error is not None:
        _set_status_text(set_status_text, error)
        return False

    invalidate_geometry_manual_pick_cache()
    update_geometry_preview_exclude_button_label()
    refresh_geometry_q_group_window()
    if live_geometry_preview_enabled():
        refresh_live_geometry_preview()
    _warm_detector_mode_qr_caked_cache(
        warm_detector_mode_qr_caked_cache,
        set_status_text=set_status_text,
    )

    _set_status_text(
        set_status_text,
        (
            f"Loaded Qr/Qz peak list from {Path(str(file_path)).name}: "
            f"matched {summary['matched_total']}, enabled {summary['included_total']}, "
            f"current-only unmatched {summary['current_only']}, "
            f"saved-only missing {summary['saved_only']}."
        ),
    )
    return True
