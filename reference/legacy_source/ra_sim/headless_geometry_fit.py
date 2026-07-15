"""Run geometry fitting from a saved GUI state without launching Tk."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
import copy
import hashlib
import importlib
import json
import math
import os
import time
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from ra_sim.fitting._numeric import coerce_nonnegative_int as _coerce_nonnegative_int
from ra_sim.fitting._numeric import positive_shape_2d as _headless_geometry_fit_positive_shape
from ra_sim.fitting._numeric import safe_finite_float_or_none as _headless_progress_float
from ra_sim.fitting._numeric import safe_int as _headless_progress_int
from ra_sim.fitting.caked_geometry_objective import ACTIVE_PARAMETER_RUNGS
from ra_sim.fitting.optimization_mosaic_profiles import focus_mosaic_profile_dataset_specs

from ra_sim.config.loader import get_instrument_config, get_path
from ra_sim.fitting.geometry_fit_parameters import (
    geometry_fit_configured_bounds,
    geometry_fit_pixel_size_m,
    safe_float as _coerce_float,
)
from ra_sim.gui._runtime.live_cache_helpers import (
    empty_peak_overlay_cache as _empty_peak_overlay_cache,
)
from ra_sim.gui.runtime_values import RuntimeValueBinding
from ra_sim.io.file_parsing import parse_poni_file
from ra_sim.io.osc_reader import read_osc
from ra_sim.utils import _LazyModuleProxy
from ra_sim.utils import stable_type_label as _headless_progress_type_label

if TYPE_CHECKING:
    from ra_sim.gui.state import (
        AtomSiteOverrideState,
        SimulationRuntimeState,
    )


DISPLAY_ROTATE_K = -1
SIM_DISPLAY_ROTATE_K = 0
HEADLESS_GEOMETRY_CAKED_RADIAL_BINS = 1000
HEADLESS_GEOMETRY_CAKED_AZIMUTH_BINS = 720


def _geometry_fit_rebuild_stage_callback_for_consumer(
    consumer: object,
    callback: Callable[..., object] | None,
) -> Callable[..., object] | None:
    if str(consumer or "") == "geometry_fit_trial_source_rows":
        return None
    return callback


_HEADLESS_GEOMETRY_FIT_DYNAMIC_OBJECTIVE_TRIAL_COVERAGE_KEYS = (
    "dynamic_objective_trial_locked_qr_row_count",
    "dynamic_objective_trial_locked_qr_row_keys",
    "dynamic_objective_trial_saved_source_count",
    "dynamic_objective_trial_saved_source_row_keys",
    "dynamic_objective_trial_coverage_complete",
)
_HEADLESS_GEOMETRY_FIT_SAVED_MANUAL_CAKED_DEFAULT_ACTIVE_VAR_NAMES = (
    "gamma",
    "Gamma",
)


@dataclass(frozen=True)
class HeadlessGeometryFitResult:
    """Result metadata for one saved-state geometry fit."""

    state: dict[str, object]
    log_path: Path
    accepted: bool
    rejection_reason: str | None = None
    rms_px: float | None = None
    mosaic_shape_fit: dict[str, object] | None = None


class HeadlessGeometryFitConfiguredInputError(RuntimeError):
    """Configured file input required before row selection was unavailable."""

    classification = "configured_input_missing"

    def __init__(
        self,
        *,
        key: str | None = None,
        path: object = None,
        operation: str | None = None,
        exc: BaseException | None = None,
        issues: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        issue_rows = [dict(issue) for issue in (issues or ()) if isinstance(issue, Mapping)]
        if not issue_rows:
            issue_rows = [
                _headless_configured_input_issue(
                    key=str(key or "configured_input"),
                    path=path,
                    operation=str(operation or "read_configured_input"),
                    exc=exc or FileNotFoundError(str(path)),
                )
            ]
        self.issues = issue_rows
        first_issue = issue_rows[0]
        self.key = str(first_issue.get("key") or "configured_input")
        self.path = str(first_issue.get("path") or "")
        self.operation = str(first_issue.get("operation") or "read_configured_input")
        self.exception_type = str(first_issue.get("exception_type") or "FileNotFoundError")
        self.original_message = str(first_issue.get("error") or "")
        if len(issue_rows) == 1:
            message = (
                "configured_input_missing: "
                f"{self.key} is required for {self.operation} but is not available at "
                f"{self.path} ({self.exception_type}: {self.original_message})"
            )
        else:
            missing_summary = ", ".join(
                f"{issue.get('key')}={issue.get('path')}" for issue in issue_rows
            )
            message = (
                "configured_input_missing: "
                f"{len(issue_rows)} configured inputs are required before geometry fitting "
                f"but are not available: {missing_summary}"
            )
        super().__init__(message)

    def progress_fields(self) -> dict[str, object]:
        return {
            "preflight_status": self.classification,
            "preflight_classification": self.classification,
            "missing_configured_input_key": self.key,
            "missing_configured_input_path": self.path,
            "missing_configured_input_count": len(self.issues),
            "missing_configured_inputs": [dict(issue) for issue in self.issues],
            "configured_input_operation": self.operation,
            "configured_input_exception_type": self.exception_type,
            "configured_input_error": self.original_message,
            "configured_input_missing_before_row_selection": True,
        }


def _headless_configured_input_path_text(path: object) -> str:
    try:
        return str(Path(str(path)).expanduser())
    except Exception:
        return str(path)


def _headless_configured_input_issue(
    *,
    key: str,
    path: object,
    operation: str,
    exc: BaseException,
) -> dict[str, object]:
    return {
        "key": str(key),
        "path": _headless_configured_input_path_text(path),
        "operation": str(operation),
        "exception_type": type(exc).__name__,
        "error": str(exc),
    }


def _headless_missing_configured_file_issue(
    *,
    key: str,
    path: object,
    operation: str,
) -> dict[str, object] | None:
    path_text = _headless_configured_input_path_text(path)
    try:
        if Path(path_text).expanduser().exists():
            return None
    except OSError as exc:
        return _headless_configured_input_issue(
            key=key,
            path=path_text,
            operation=operation,
            exc=exc,
        )
    return _headless_configured_input_issue(
        key=key,
        path=path_text,
        operation=operation,
        exc=FileNotFoundError(f"No such file or directory: {path_text}"),
    )


def _headless_config_path_issue(
    *,
    key: str,
    operation: str,
    exc: BaseException,
) -> dict[str, object]:
    return _headless_configured_input_issue(
        key=key,
        path=f"<unresolved:{key}>",
        operation=operation,
        exc=exc,
    )


def _headless_geometry_fit_configured_input_issues(
    saved_state: Mapping[str, object],
) -> list[dict[str, object]]:
    files_state = saved_state.get("files", {}) if isinstance(saved_state.get("files"), dict) else {}
    issues: list[dict[str, object]] = []

    try:
        geometry_poni_path = get_path("geometry_poni")
    except Exception as exc:
        issues.append(
            _headless_config_path_issue(
                key="geometry_poni",
                operation="resolve_config_path",
                exc=exc,
            )
        )
    else:
        issue = _headless_missing_configured_file_issue(
            key="geometry_poni",
            path=geometry_poni_path,
            operation="parse_poni_file",
        )
        if issue is not None:
            issues.append(issue)

    primary_cif_raw = files_state.get("primary_cif_path")
    if primary_cif_raw:
        primary_cif_path = primary_cif_raw
    else:
        try:
            primary_cif_path = get_path("cif_file")
        except Exception as exc:
            primary_cif_path = None
            issues.append(
                _headless_config_path_issue(
                    key="primary_cif_path",
                    operation="resolve_config_path",
                    exc=exc,
                )
            )
    if primary_cif_path:
        issue = _headless_missing_configured_file_issue(
            key="primary_cif_path",
            path=primary_cif_path,
            operation="read_primary_cif",
        )
        if issue is not None:
            issues.append(issue)

    secondary_cif_raw = files_state.get("secondary_cif_path")
    if secondary_cif_raw:
        secondary_cif_path = secondary_cif_raw
    else:
        try:
            secondary_cif_path = get_path("cif_file2")
        except KeyError:
            secondary_cif_path = None
        except Exception as exc:
            secondary_cif_path = None
            issues.append(
                _headless_config_path_issue(
                    key="secondary_cif_path",
                    operation="resolve_config_path",
                    exc=exc,
                )
            )
    if secondary_cif_path:
        issue = _headless_missing_configured_file_issue(
            key="secondary_cif_path",
            path=secondary_cif_path,
            operation="read_secondary_cif",
        )
        if issue is not None:
            issues.append(issue)

    osc_files_raw = files_state.get("background_files", [])
    if isinstance(osc_files_raw, Sequence) and not isinstance(osc_files_raw, (str, bytes)):
        for index, raw_path in enumerate(osc_files_raw):
            path_text = str(raw_path).strip()
            if not path_text:
                continue
            issue = _headless_missing_configured_file_issue(
                key=f"background_files[{index}]",
                path=path_text,
                operation="read_background_file",
            )
            if issue is not None:
                issues.append(issue)
    return issues


def headless_geometry_fit_result_report_fields(result: object) -> dict[str, object]:
    log_path = getattr(result, "log_path", None)
    report: dict[str, object] = {
        "accepted": bool(getattr(result, "accepted", False)),
        "log_path": str(log_path) if log_path is not None else None,
        "matched_peaks_path": None,
    }
    rejection_reason = getattr(result, "rejection_reason", None)
    if rejection_reason:
        report["rejection_reason"] = str(rejection_reason)
    rms_px = getattr(result, "rms_px", None)
    if rms_px is not None:
        report["rms_px"] = float(rms_px)
    mosaic_shape_fit = getattr(result, "mosaic_shape_fit", None)
    if isinstance(mosaic_shape_fit, Mapping):
        report["mosaic_shape_fit"] = copy.deepcopy(dict(mosaic_shape_fit))
    return report


@dataclass(frozen=True)
class _RuntimeDefaults:
    primary_cif_path: str
    secondary_cif_path: str | None
    osc_files: list[str]
    current_background_index: int
    image_size: int
    pixel_size_m: float
    lambda_angstrom: float
    psi_deg: float
    defaults: dict[str, object]
    fit_config: dict[str, object]
    intensity_threshold: float
    include_rods_flag: bool
    two_theta_range: tuple[float, float]
    mx: int


def normalize_headless_geometry_fit_active_var_names(
    active_var_names: Sequence[object] | str | None,
) -> list[str] | None:
    """Normalize one optional ordered active-variable override for headless fits."""

    if active_var_names is None:
        return None
    if isinstance(active_var_names, str):
        if not active_var_names.strip():
            raise ValueError("Geometry fit active-vars override cannot be empty.")
        raw_names = active_var_names.split(",")
    else:
        raw_names = list(active_var_names)
        if not raw_names:
            raise ValueError("Geometry fit active-vars override cannot be empty.")

    normalized_names: list[str] = []
    seen_names: set[str] = set()
    for raw_name in raw_names:
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Geometry fit active-vars override contains an empty name.")
        if name in seen_names:
            raise ValueError(f"Duplicate geometry fit active var '{name}'.")
        seen_names.add(name)
        normalized_names.append(name)

    if tuple(normalized_names) not in ACTIVE_PARAMETER_RUNGS:
        supported = "; ".join(",".join(rung) for rung in ACTIVE_PARAMETER_RUNGS)
        raise ValueError(
            f"Geometry fit active vars must be one exact-caked parameter rung: {supported}."
        )
    return normalized_names


def _headless_geometry_fit_runtime_active_var_names(
    active_var_names: Sequence[str],
) -> list[str]:
    """Return the exact-caked parameter rung requested for this fit."""

    names = normalize_headless_geometry_fit_active_var_names(active_var_names)
    if names is None:
        raise ValueError("Geometry fit active variables are required.")
    return names


def _headless_geometry_fit_bounds_section(
    fit_config: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the required headless geometry-fit bounds mapping."""

    bounds_cfg = fit_config.get("bounds")
    if not isinstance(bounds_cfg, Mapping):
        raise TypeError("geometry-fit configuration requires a bounds mapping")
    return bounds_cfg


def _headless_runtime_geometry_fit_parameter_domains(
    *,
    fit_config: Mapping[str, object],
    current_params: Mapping[str, object],
    names: Sequence[str],
) -> dict[str, tuple[float, float]]:
    """Build strict headless geometry-fit domains from configured bounds."""

    bounds_cfg = _headless_geometry_fit_bounds_section(fit_config)
    domains: dict[str, tuple[float, float]] = {}
    for raw_name in names:
        name = str(raw_name)
        current_value = float(current_params[name])
        if not np.isfinite(current_value):
            raise ValueError(f"geometry-fit parameter {name!r} must be finite")
        lo, hi = geometry_fit_configured_bounds(
            name,
            current_value,
            bounds_cfg=bounds_cfg,
        )
        if not lo < hi:
            raise ValueError(f"geometry-fit bounds for {name!r} must satisfy min < max")
        domains[name] = (float(lo), float(hi))
    return domains


def _read_first_cif_block(path: str) -> tuple[object, object]:
    """Load one CIF and return the container plus its first block."""

    import CifFile

    cf = CifFile.ReadCif(path)
    keys = list(cf.keys())
    if not keys:
        raise ValueError(f"No CIF data blocks found in {path}")
    return cf, cf[keys[0]]


def _lazy_import_module(module_name: str):
    @lru_cache(maxsize=1)
    def _load():
        return importlib.import_module(module_name)

    return _load


def _lazy_import_namespace(module_name: str, *names: str):
    @lru_cache(maxsize=1)
    def _load() -> SimpleNamespace:
        module = importlib.import_module(module_name)
        return SimpleNamespace(**{name: getattr(module, name) for name in names})

    return _load


_load_gui_background_module = _lazy_import_module("ra_sim.gui.background")
_load_gui_background_theta_module = _lazy_import_module("ra_sim.gui.background_theta")
_load_gui_controllers_module = _lazy_import_module("ra_sim.gui.controllers")
_load_gui_geometry_fit_module = _lazy_import_module("ra_sim.gui.geometry_fit")
_load_gui_geometry_overlay_module = _lazy_import_module("ra_sim.gui.geometry_overlay")
_load_gui_geometry_q_group_manager_module = _lazy_import_module(
    "ra_sim.gui.geometry_q_group_manager"
)
_load_gui_manual_geometry_module = _lazy_import_module("ra_sim.gui.manual_geometry")
_load_gui_structure_model_module = _lazy_import_module("ra_sim.gui.structure_model")
_load_gui_state_types = _lazy_import_namespace(
    "ra_sim.gui.state",
    "AtomSiteOverrideState",
    "BackgroundRuntimeState",
    "SimulationRuntimeState",
)
_load_exact_cake_portable_module = _lazy_import_module("ra_sim.simulation.exact_cake_portable")
_load_simulation_diffraction = _lazy_import_module("ra_sim.simulation.diffraction")
_load_simulation_mosaic_profiles = _lazy_import_module("ra_sim.simulation.mosaic_profiles")
_load_fitting_optimization = _lazy_import_module("ra_sim.fitting.optimization")
_load_intersection_cache_schema = _lazy_import_module("ra_sim.simulation.intersection_cache_schema")
_load_stacking_fault_runtime = _lazy_import_module("ra_sim.utils.stacking_fault")
_load_diffraction_tools = _lazy_import_module("ra_sim.utils.diffraction_tools")
_load_calculation_runtime = _lazy_import_module("ra_sim.utils.calculations")


gui_background = _LazyModuleProxy(_load_gui_background_module)
gui_background_theta = _LazyModuleProxy(_load_gui_background_theta_module)
gui_controllers = _LazyModuleProxy(_load_gui_controllers_module)
gui_geometry_fit = _LazyModuleProxy(_load_gui_geometry_fit_module)
gui_geometry_overlay = _LazyModuleProxy(_load_gui_geometry_overlay_module)
gui_geometry_q_group_manager = _LazyModuleProxy(_load_gui_geometry_q_group_manager_module)
gui_manual_geometry = _LazyModuleProxy(_load_gui_manual_geometry_module)
gui_structure_model = _LazyModuleProxy(_load_gui_structure_model_module)


def _headless_native_detector_coords_to_detector_display_coords_for_background(
    load_background_by_index,
    background_index: int,
    *,
    display_rotate_k: int = DISPLAY_ROTATE_K,
):
    try:
        bg_idx = int(background_index)
    except Exception:
        return None
    try:
        native_background, _display_background = load_background_by_index(bg_idx)
        native_shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
    except Exception:
        return None
    if len(native_shape) < 2 or min(native_shape) <= 0:
        return None

    def _to_display(col: float, row: float):
        return gui_geometry_overlay.rotate_point_for_display(
            float(col),
            float(row),
            native_shape,
            int(display_rotate_k),
        )

    return _to_display


def _headless_background_display_to_native_detector_coords_for_background(
    load_background_by_index,
    background_index: int,
    *,
    display_rotate_k: int = DISPLAY_ROTATE_K,
):
    try:
        bg_idx = int(background_index)
    except Exception:
        return None
    try:
        native_background, _display_background = load_background_by_index(bg_idx)
        native_shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
    except Exception:
        return None
    if len(native_shape) < 2 or min(native_shape) <= 0:
        return None

    def _to_native(col: float, row: float):
        return gui_geometry_overlay.display_point_to_native_for_rotation(
            float(col),
            float(row),
            native_shape,
            int(display_rotate_k),
        )

    return _to_native


def _headless_native_detector_coords_to_caked_display_coords_for_payload(
    payload: Mapping[str, object] | None,
    gui_manual_geometry,
    *,
    native_detector_coords_to_bundle_detector_coords: Callable[[float, float], object]
    | None = None,
):
    if not isinstance(payload, Mapping):
        return None
    return partial(
        gui_manual_geometry.native_detector_coords_to_caked_display_coords,
        transform_bundle=payload.get("transform_bundle"),
        native_detector_coords_to_bundle_detector_coords=(
            native_detector_coords_to_bundle_detector_coords
        ),
    )


_HEADLESS_FROZEN_CENTROID_BINDING_SCHEMA = "geometry_fit_frozen_centroid_projection_binding_v1"


def _headless_geometry_fit_frozen_centroid_binding(
    *,
    params: Mapping[str, object],
    projection_payload: Mapping[str, object],
    expected_native_detector_shape: object,
    hit_table_detector_shape: object,
    hit_table_detector_coords_to_bundle_detector_coords: Callable[[float, float], object],
) -> tuple[object, dict[str, object]] | None:
    """Validate and describe the exact frozen projection used by centroid hits."""

    exact_cake = _load_exact_cake_portable_module()
    transform_bundle = projection_payload.get("transform_bundle")
    native_shape = _headless_geometry_fit_positive_shape(expected_native_detector_shape)
    hit_shape = _headless_geometry_fit_positive_shape(hit_table_detector_shape)
    payload_shape = _headless_geometry_fit_positive_shape(projection_payload.get("detector_shape"))
    bundle_shape = _headless_geometry_fit_positive_shape(
        getattr(transform_bundle, "detector_shape", None)
    )
    canonical_native_shape = _headless_geometry_fit_positive_shape(
        projection_payload.get("native_detector_shape")
    )
    canonical_bundle_shape = _headless_geometry_fit_positive_shape(
        projection_payload.get("bundle_detector_shape")
    )
    if (
        not isinstance(transform_bundle, exact_cake.CakeTransformBundle)
        or native_shape is None
        or hit_shape is None
        or payload_shape is None
        or bundle_shape is None
        or canonical_native_shape != native_shape
        or canonical_bundle_shape != bundle_shape
        or payload_shape != bundle_shape
    ):
        return None

    center = gui_geometry_fit._geometry_fit_center_from_params(params)
    pixel_size = geometry_fit_pixel_size_m(params)
    try:
        distance = float(params.get("corto_detector", np.nan))
        pixel_size_value = float(pixel_size)
        canonical_center = np.asarray(
            projection_payload.get("bundle_center_row_col"),
            dtype=np.float64,
        ).reshape(-1)
    except (TypeError, ValueError):
        return None
    if (
        center is None
        or not np.isfinite(distance)
        or distance <= 0.0
        or not np.isfinite(pixel_size_value)
        or pixel_size_value <= 0.0
        or canonical_center.size != 2
        or not np.all(np.isfinite(canonical_center))
        or not np.allclose(
            canonical_center,
            np.asarray(center, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-9,
        )
        or str(projection_payload.get("center_input_frame") or "") != "simulation_cake_raster"
    ):
        return None

    try:
        radial_axis = np.asarray(projection_payload.get("radial_axis"), dtype=np.float64).reshape(
            -1
        )
        azimuth_axis = np.asarray(projection_payload.get("azimuth_axis"), dtype=np.float64).reshape(
            -1
        )
        raw_azimuth_axis = np.asarray(
            projection_payload.get("raw_azimuth_axis"), dtype=np.float64
        ).reshape(-1)
        permutation = np.asarray(
            projection_payload.get("raw_to_gui_row_permutation"), dtype=np.int64
        ).reshape(-1)
        bundle_radial = np.asarray(transform_bundle.radial_deg, dtype=np.float64).reshape(-1)
        bundle_gui = np.asarray(transform_bundle.gui_azimuth_deg, dtype=np.float64).reshape(-1)
        bundle_raw = np.asarray(transform_bundle.raw_azimuth_deg, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if (
        radial_axis.size < 2
        or azimuth_axis.size < 2
        or raw_azimuth_axis.size < 2
        or radial_axis.shape != bundle_radial.shape
        or azimuth_axis.shape != bundle_gui.shape
        or raw_azimuth_axis.shape != bundle_raw.shape
        or permutation.shape != azimuth_axis.shape
        or not all(
            np.all(np.isfinite(axis))
            for axis in (
                radial_axis,
                azimuth_axis,
                raw_azimuth_axis,
                bundle_radial,
                bundle_gui,
                bundle_raw,
            )
        )
        or not np.allclose(radial_axis, bundle_radial, rtol=0.0, atol=1.0e-9)
        or not np.allclose(raw_azimuth_axis, bundle_raw, rtol=0.0, atol=1.0e-9)
        or not np.array_equal(
            np.sort(permutation),
            np.arange(int(azimuth_axis.size), dtype=np.int64),
        )
        or not np.allclose(
            azimuth_axis,
            bundle_gui[permutation],
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        return None

    lut = getattr(transform_bundle, "lut", None)
    lut_shape = _headless_geometry_fit_positive_shape(getattr(lut, "image_shape", None))
    try:
        lut_n_rad = int(getattr(lut, "n_rad"))
        lut_n_az = int(getattr(lut, "n_az"))
    except Exception:
        return None
    if (
        lut_shape != bundle_shape
        or lut_n_rad != int(radial_axis.size)
        or lut_n_az != int(raw_azimuth_axis.size)
    ):
        return None

    hit_height, hit_width = hit_shape
    expected_corners = {
        (0.0, 0.0),
        (float(bundle_shape[1] - 1), 0.0),
        (0.0, float(bundle_shape[0] - 1)),
        (float(bundle_shape[1] - 1), float(bundle_shape[0] - 1)),
    }
    mapped_corners: set[tuple[float, float]] = set()
    try:
        for col, row in (
            (0.0, 0.0),
            (float(hit_width - 1), 0.0),
            (0.0, float(hit_height - 1)),
            (float(hit_width - 1), float(hit_height - 1)),
        ):
            point = hit_table_detector_coords_to_bundle_detector_coords(col, row)
            mapped_corners.add((round(float(point[0]), 9), round(float(point[1]), 9)))
    except Exception:
        return None
    if mapped_corners != expected_corners:
        return None

    projection_content_token = gui_geometry_fit._geometry_fit_digest_payload(
        {
            "kind": "headless_frozen_centroid_projection_binding_v1",
            "detector_shape": list(bundle_shape),
            "native_detector_shape": list(native_shape),
            "radial_axis": gui_manual_geometry.geometry_manual_stable_axis_value_token(radial_axis),
            "azimuth_axis": gui_manual_geometry.geometry_manual_stable_axis_value_token(
                azimuth_axis
            ),
            "raw_azimuth_axis": gui_manual_geometry.geometry_manual_stable_axis_value_token(
                raw_azimuth_axis
            ),
            "row_permutation": (
                gui_manual_geometry.geometry_manual_stable_permutation_value_token(permutation)
            ),
            "center_row_col": [float(center[0]), float(center[1])],
            "distance_m": float(distance),
            "pixel_size_m": float(pixel_size_value),
            "lut_shape": list(lut_shape),
            "lut_n_rad": int(lut_n_rad),
            "lut_n_az": int(lut_n_az),
        }
    )
    projection_digest = hashlib.sha256(str(projection_content_token).encode("utf-8")).hexdigest()
    geometry = exact_cake.PortableGeometry(
        pixel_size_m=float(pixel_size_value),
        distance_m=float(distance),
        center_row_px=float(center[0]),
        center_col_px=float(center[1]),
    )
    return geometry, {
        "schema": _HEADLESS_FROZEN_CENTROID_BINDING_SCHEMA,
        "projection_payload_digest": str(projection_digest),
        "native_detector_shape": [int(value) for value in native_shape],
        "bundle_detector_shape": [int(value) for value in bundle_shape],
        "hit_table_detector_shape": [int(value) for value in hit_shape],
        "center_row_col": [float(center[0]), float(center[1])],
        "distance_m": float(distance),
        "pixel_size_m": float(pixel_size_value),
        "center_input_frame": "simulation_cake_raster",
        "hit_table_point_frame": "simulation_detector_native",
        "projector_point_frame": "frozen_exact_cake_bundle_detector",
    }


def _headless_geometry_fit_build_frozen_centroid_hit_projector(
    *,
    params: Mapping[str, object],
    projection_payload: Mapping[str, object] | None,
    expected_native_detector_shape: object,
    hit_table_detector_shape: object,
    hit_table_detector_coords_to_bundle_detector_coords: (Callable[[float, float], object] | None),
) -> tuple[Callable[[float, float], tuple[float, float]], dict[str, object]] | None:
    """Build the strict raw-simulation-hit to continuous-cake projector.

    Process hit-table coordinates are emitted in the simulation detector frame.
    They therefore enter the frozen cake bundle through the simulation display
    transform, never through a background-native transform.
    """

    if not isinstance(projection_payload, Mapping) or not callable(
        hit_table_detector_coords_to_bundle_detector_coords
    ):
        return None
    bound = _headless_geometry_fit_frozen_centroid_binding(
        params=params,
        projection_payload=projection_payload,
        expected_native_detector_shape=expected_native_detector_shape,
        hit_table_detector_shape=hit_table_detector_shape,
        hit_table_detector_coords_to_bundle_detector_coords=(
            hit_table_detector_coords_to_bundle_detector_coords
        ),
    )
    if bound is None:
        return None
    geometry, provenance = bound
    exact_cake = _load_exact_cake_portable_module()

    def _project(col: float, row: float) -> tuple[float, float]:
        raw_bundle_point = hit_table_detector_coords_to_bundle_detector_coords(
            float(col),
            float(row),
        )
        try:
            bundle_col = float(raw_bundle_point[0])  # type: ignore[index]
            bundle_row = float(raw_bundle_point[1])  # type: ignore[index]
        except Exception as exc:
            raise ValueError("strict_centroid_hit_to_bundle_point_invalid") from exc
        if not (np.isfinite(bundle_col) and np.isfinite(bundle_row)):
            raise ValueError("strict_centroid_hit_to_bundle_point_nonfinite")
        return exact_cake.detector_pixel_to_continuous_caked_angles(
            geometry,
            float(bundle_col),
            float(bundle_row),
        )

    return _project, provenance


def _headless_geometry_fit_fresh_simulation_hit_projector(
    *,
    params: Mapping[str, object],
    projection_payload: Mapping[str, object] | None,
    expected_native_detector_shape: object,
    simulation_detector_shape: object,
    native_sim_to_display_coords: Callable[[float, float, object], object] | None,
) -> tuple[Callable[[float, float], tuple[float, float]], dict[str, object]] | None:
    """Map fresh simulation-native hits into one frozen exact-cake bundle."""

    if not callable(native_sim_to_display_coords):
        return None
    try:
        detector_shape = tuple(int(value) for value in simulation_detector_shape[:2])
    except (TypeError, ValueError, IndexError):
        return None
    if len(detector_shape) != 2 or min(detector_shape) <= 0:
        return None

    def _raw_sim_hit_to_bundle(col: float, row: float) -> tuple[float, float]:
        point = native_sim_to_display_coords(
            float(col),
            float(row),
            detector_shape,
        )
        return float(point[0]), float(point[1])  # type: ignore[index]

    return _headless_geometry_fit_build_frozen_centroid_hit_projector(
        params=params,
        projection_payload=projection_payload,
        expected_native_detector_shape=expected_native_detector_shape,
        hit_table_detector_shape=detector_shape,
        hit_table_detector_coords_to_bundle_detector_coords=_raw_sim_hit_to_bundle,
    )


def _headless_geometry_fit_outcome_status_fields(
    *,
    apply_accepted: bool,
    solver_result: object,
    final_summary: Mapping[str, object] | None,
) -> dict[str, object]:
    summary = final_summary if isinstance(final_summary, Mapping) else {}
    if summary.get("schema") != "caked_geometry_first_rung_result_v2":
        raise ValueError("Geometry fit did not return the exact caked result schema.")
    if summary.get("acceptance_metric_space") != "caked_deg":
        raise ValueError("Geometry fit did not return exact caked angular metrics.")
    exact_fit_result = summary.get("fit_result")
    if not isinstance(exact_fit_result, Mapping):
        raise ValueError("Geometry fit did not return its caked solver result.")
    fit_quality_pass = bool(exact_fit_result.get("success", False))
    fit_quality_reason = "caked_solver_converged" if fit_quality_pass else "caked_solver_failed"
    solver_success = bool(getattr(solver_result, "success", False))
    state_write_accepted = bool(apply_accepted and solver_success and fit_quality_pass)
    accepted = state_write_accepted
    if not state_write_accepted:
        if not bool(apply_accepted):
            outcome_rejection_reason = "state_write_rejected"
        elif not solver_success:
            outcome_rejection_reason = "solver_failed"
        else:
            outcome_rejection_reason = fit_quality_reason
    else:
        outcome_rejection_reason = None
    outcome = {
        "accepted": bool(accepted),
        "solver_success": bool(solver_success),
        "fit_quality_pass": bool(fit_quality_pass),
        "fit_quality_reason": fit_quality_reason,
        "state_write_accepted": bool(state_write_accepted),
        "outcome_rejection_reason": outcome_rejection_reason,
    }
    outcome["acceptance_policy_id"] = getattr(
        gui_geometry_fit,
        "CAKED_GEOMETRY_FIRST_RUNG_ACCEPTANCE_POLICY_ID",
        None,
    )
    return outcome


def _coerce_int(value: object, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = int(default)
    if minimum is not None:
        parsed = max(int(minimum), parsed)
    return int(parsed)


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _build_headless_geometry_fit_caked_projection_payload(
    detector_shape: object,
    *,
    params: Mapping[str, object],
    pixel_size_m: float,
    bundle_detector_shape: object | None = None,
    native_detector_coords_to_bundle_detector_coords: Callable[[float, float], object]
    | None = None,
) -> dict[str, object] | None:
    """Build exact caked projector payload without integrating a display image."""

    optional_kwargs: dict[str, object] = {}
    if bundle_detector_shape is not None:
        optional_kwargs["bundle_detector_shape"] = bundle_detector_shape
    if callable(native_detector_coords_to_bundle_detector_coords):
        optional_kwargs["native_detector_coords_to_bundle_detector_coords"] = (
            native_detector_coords_to_bundle_detector_coords
        )
    return gui_geometry_fit.build_geometry_fit_frozen_caked_projection_payload(
        detector_shape,
        params=params,
        pixel_size_m=float(pixel_size_m),
        npt_rad=HEADLESS_GEOMETRY_CAKED_RADIAL_BINS,
        npt_azim=HEADLESS_GEOMETRY_CAKED_AZIMUTH_BINS,
        **optional_kwargs,
    )


def _require_positive_float(raw_value: object, field_name: str) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return value


def _require_phase_delta_expression(raw_value: object, validator: Callable[[str], str]) -> str:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("phase_delta_expression must be a nonempty string")
    return validator(raw_value)


def _resolve_solve_q_mode(mode_raw: object) -> int:
    if mode_raw == "uniform":
        return 0
    if mode_raw == "adaptive":
        return 1
    raise ValueError("solve_q_mode must be 'uniform' or 'adaptive'")


def _build_runtime_defaults(saved_state: dict[str, object]) -> _RuntimeDefaults:
    diffraction = _load_simulation_diffraction()
    pixel_tools = _load_diffraction_tools()
    gui_geometry_overlay = _load_gui_geometry_overlay_module()
    stack = _load_stacking_fault_runtime()
    instrument = get_instrument_config().get("instrument", {})
    detector_cfg = instrument.get("detector", {})
    geometry_cfg = instrument.get("geometry_defaults", {})
    beam_cfg = instrument.get("beam", {})
    sample_cfg = instrument.get("sample_orientation", {})
    debye_cfg = instrument.get("debye_waller", {})
    ht_cfg = instrument.get("hendricks_teller", {})
    fit_config = instrument.get("fit", {})
    files_state = saved_state.get("files")
    if not isinstance(files_state, Mapping):
        raise ValueError("GUI state v1 is missing files.")
    primary_cif_raw = files_state.get("primary_cif_path")
    if not isinstance(primary_cif_raw, str) or not primary_cif_raw.strip():
        raise ValueError("GUI state v1 files.primary_cif_path must be a path.")
    primary_cif_path = str(Path(primary_cif_raw).expanduser())
    secondary_cif_raw = files_state.get("secondary_cif_path")
    if secondary_cif_raw:
        if not isinstance(secondary_cif_raw, str):
            raise ValueError("GUI state v1 files.secondary_cif_path must be a path or null.")
        secondary_cif_path = str(Path(secondary_cif_raw).expanduser())
    else:
        secondary_cif_path = None

    osc_files_raw = files_state.get("background_files")
    if not isinstance(osc_files_raw, list) or not all(
        isinstance(path, str) and path.strip() for path in osc_files_raw
    ):
        raise ValueError("GUI state v1 files.background_files must be a list of paths.")
    osc_files = [str(Path(path).expanduser()) for path in osc_files_raw]
    if not osc_files:
        raise ValueError("Saved GUI state does not include any background files.")

    try:
        current_background_index = int(files_state["current_background_index"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("GUI state v1 files.current_background_index must be an integer.") from exc
    if not 0 <= current_background_index < len(osc_files):
        raise ValueError("GUI state v1 files.current_background_index is out of range.")

    image_size = int(detector_cfg.get("image_size", 3000))
    pixel_size_m = float(detector_cfg.get("pixel_size_m", pixel_tools.DEFAULT_PIXEL_SIZE_M))

    geometry_poni_path = get_path("geometry_poni")
    try:
        poni = parse_poni_file(geometry_poni_path)
    except OSError as exc:
        raise HeadlessGeometryFitConfiguredInputError(
            key="geometry_poni",
            path=geometry_poni_path,
            operation="parse_poni_file",
            exc=exc,
        ) from exc
    distance_m = float(poni.get("Dist", geometry_cfg.get("distance_m", 0.075)))
    gamma_initial = float(poni.get("Rot2", geometry_cfg.get("rot2", 0.0)))
    Gamma_initial = float(poni.get("Rot1", geometry_cfg.get("rot1", 0.0)))
    poni1 = float(poni.get("Poni1", geometry_cfg.get("poni1_m", 0.0)))
    poni2 = float(poni.get("Poni2", geometry_cfg.get("poni2_m", 0.0)))
    wave_m = float(poni.get("Wavelength", geometry_cfg.get("wavelength_m", 1.0e-10)))
    lambda_from_poni = wave_m * 1.0e10
    lambda_override = beam_cfg.get("wavelength_angstrom")
    lambda_angstrom = float(lambda_override if lambda_override is not None else lambda_from_poni)

    center_default = list(
        gui_geometry_overlay.beam_center_row_col_from_poni(
            float(poni1),
            float(poni2),
            float(pixel_size_m),
        )
    )
    two_theta_max = pixel_tools.detector_two_theta_max(
        image_size,
        center_default,
        distance_m,
        pixel_size=pixel_size_m,
    )

    try:
        cf, blk = _read_first_cif_block(primary_cif_path)
    except OSError as exc:
        raise HeadlessGeometryFitConfiguredInputError(
            key="primary_cif_path",
            path=primary_cif_path,
            operation="read_primary_cif",
            exc=exc,
        ) from exc
    av = gui_structure_model.parse_cif_num(blk.get("_cell_length_a"))
    bv = gui_structure_model.parse_cif_num(blk.get("_cell_length_b"))
    cv = gui_structure_model.parse_cif_num(blk.get("_cell_length_c"))
    if secondary_cif_path:
        try:
            cf2, blk2 = _read_first_cif_block(secondary_cif_path)
        except OSError as exc:
            raise HeadlessGeometryFitConfiguredInputError(
                key="secondary_cif_path",
                path=secondary_cif_path,
                operation="read_secondary_cif",
                exc=exc,
            ) from exc
        av2 = gui_structure_model.parse_cif_num(blk2.get("_cell_length_a") or av)
        cv2 = gui_structure_model.parse_cif_num(blk2.get("_cell_length_c") or cv)
    else:
        av2 = None
        cv2 = None

    rich_phase_components = stack.canonicalize_rich_phase_components(
        ht_cfg.get("rich_phase_components")
    )
    p_defaults = [rich_phase_components[key]["epsilon"] for key in ("2H", "4H", "6H")]
    w_defaults = [rich_phase_components[key]["w"] for key in ("2H", "4H", "6H")]
    phase_delta_raw = ht_cfg.get("phase_delta_expression")
    if not isinstance(phase_delta_raw, str) or not phase_delta_raw.strip():
        raise ValueError("ht.phase_delta_expression must be a nonempty string")
    phase_delta_default = stack.validate_phase_delta_expression(phase_delta_raw)

    film_thickness_nm_default = gui_controllers.normalize_finite_stack_thickness_nm(
        ht_cfg.get(
            "film_thickness_nm",
            gui_controllers.DEFAULT_FINITE_STACK_FILM_THICKNESS_NM,
        ),
        gui_controllers.DEFAULT_FINITE_STACK_FILM_THICKNESS_NM,
    )
    stack_layers_default = gui_controllers.finite_stack_layers_from_thickness_nm(
        film_thickness_nm=film_thickness_nm_default,
        c_axis_angstrom=cv,
    )

    defaults = {
        "theta_initial": float(sample_cfg.get("theta_initial_deg", 6.0)),
        "cor_angle": float(sample_cfg.get("cor_deg", 0.0)),
        "gamma": float(Gamma_initial),
        "Gamma": float(gamma_initial),
        "chi": float(sample_cfg.get("chi_deg", 0.0)),
        "psi_z": float(sample_cfg.get("psi_z_deg", 0.0)),
        "zs": float(sample_cfg.get("zs", 0.0)),
        "zb": float(sample_cfg.get("zb", 0.0)),
        "sample_width_m": float(sample_cfg.get("width_m", 0.0)),
        "sample_length_m": float(sample_cfg.get("length_m", 0.0)),
        "sample_depth_m": float(sample_cfg.get("depth_m", 0.0)),
        "debye_x": float(debye_cfg.get("x", 0.0)),
        "debye_y": float(debye_cfg.get("y", 0.0)),
        "corto_detector": float(distance_m),
        "sigma_mosaic_deg": float(beam_cfg.get("sigma_mosaic_fwhm_deg", 0.8)),
        "gamma_mosaic_deg": float(beam_cfg.get("gamma_mosaic_fwhm_deg", 0.7)),
        "eta": float(beam_cfg.get("eta", 0.0)),
        "a": float(av),
        "b": float(bv),
        "c": float(cv),
        "a2": float(av2) if av2 is not None else None,
        "c2": float(cv2) if cv2 is not None else None,
        "p0": float(p_defaults[0]),
        "p1": float(p_defaults[1]),
        "p2": float(p_defaults[2]),
        "w0": float(w_defaults[0]),
        "w1": float(w_defaults[1]),
        "w2": float(w_defaults[2]),
        "rich_phase_components": rich_phase_components,
        "iodine_z": 0.0,
        "phase_delta_expression": str(phase_delta_default),
        "phi_l_divisor": float(
            _require_positive_float(ht_cfg.get("phi_l_divisor"), "ht.phi_l_divisor")
        ),
        "center_x": float(center_default[0]),
        "center_y": float(center_default[1]),
        "bandwidth_percent": float(
            np.clip(float(beam_cfg.get("bandwidth_percent", 0.7)), 0.0, 10.0)
        ),
        "divergence_fwhm_deg": float(beam_cfg["divergence_fwhm_deg"]),
        "bandwidth_sigma_fraction": float(beam_cfg["bandwidth_sigma_fraction"]),
        "solve_q_steps": int(beam_cfg.get("solve_q_steps", diffraction.DEFAULT_SOLVE_Q_STEPS)),
        "solve_q_rel_tol": float(
            beam_cfg.get("solve_q_rel_tol", diffraction.DEFAULT_SOLVE_Q_REL_TOL)
        ),
        "solve_q_mode": _resolve_solve_q_mode(
            beam_cfg.get("solve_q_mode", "uniform")
        ),
        "finite_stack": bool(ht_cfg.get("finite_stack", True)),
        "film_thickness_nm": float(film_thickness_nm_default),
        "stack_layers": int(stack_layers_default),
        "weight1": 0.5 if secondary_cif_path else 1.0,
        "weight2": 0.5 if secondary_cif_path else 0.0,
    }

    return _RuntimeDefaults(
        primary_cif_path=primary_cif_path,
        secondary_cif_path=secondary_cif_path,
        osc_files=osc_files,
        current_background_index=current_background_index,
        image_size=image_size,
        pixel_size_m=pixel_size_m,
        lambda_angstrom=lambda_angstrom,
        psi_deg=float(sample_cfg.get("psi_deg", 0.0)),
        defaults=defaults,
        fit_config=dict(fit_config) if isinstance(fit_config, dict) else {},
        intensity_threshold=float(detector_cfg.get("intensity_threshold", 1.0)),
        include_rods_flag=bool(ht_cfg.get("include_rods", False)),
        two_theta_range=(0.0, float(two_theta_max)),
        mx=int(ht_cfg.get("max_miller_index", 19)),
    )


def _default_finite_stack_film_thickness_nm(default_values: Mapping[str, object]) -> float:
    return gui_controllers.normalize_finite_stack_thickness_nm(
        default_values.get(
            "film_thickness_nm",
            gui_controllers.DEFAULT_FINITE_STACK_FILM_THICKNESS_NM,
        ),
        gui_controllers.DEFAULT_FINITE_STACK_FILM_THICKNESS_NM,
    )


def _build_var_store(
    saved_state: dict[str, object],
) -> dict[str, object]:
    saved_variables = saved_state.get("variables")
    if not isinstance(saved_variables, dict):
        raise ValueError("GUI state v1 variables must be an object.")
    required_names = (
        "fit_rung_var",
        "zb_var",
        "zs_var",
        "theta_initial_var",
        "psi_z_var",
        "chi_var",
        "cor_angle_var",
        "sample_width_var",
        "sample_length_var",
        "sample_depth_var",
        "gamma_var",
        "Gamma_var",
        "corto_detector_var",
        "a_var",
        "c_var",
        "center_x_var",
        "center_y_var",
        "debye_x_var",
        "debye_y_var",
        "geometry_theta_offset_var",
        "background_theta_list_var",
        "geometry_fit_background_selection_var",
        "sigma_mosaic_var",
        "gamma_mosaic_var",
        "eta_var",
        "bandwidth_percent_var",
        "solve_q_steps_var",
        "solve_q_rel_tol_var",
        "solve_q_mode_var",
        "p0_var",
        "p1_var",
        "p2_var",
        "w0_var",
        "w1_var",
        "w2_var",
        "finite_stack_var",
        "film_thickness_nm_var",
        "phase_delta_expr_var",
        "phi_l_divisor_var",
        "weight1_var",
        "weight2_var",
        "sample_count_var",
    )
    missing = sorted(set(required_names) - set(saved_variables))
    if missing:
        raise ValueError(f"GUI state v1 is missing required fit variables: {missing}")
    return {name: saved_variables[name] for name in required_names}


def _restore_manual_pairs(
    osc_files: list[str],
    saved_rows: list[object] | None,
) -> dict[int, list[dict[str, object]]]:
    pairs_by_background: dict[int, list[dict[str, object]]] = {}

    def _pairs_for_index(index: int) -> list[dict[str, object]]:
        return gui_manual_geometry.geometry_manual_pairs_for_index(
            int(index), pairs_by_background=pairs_by_background
        )

    def _replace(payload: dict[int, list[dict[str, object]]]) -> None:
        pairs_by_background.clear()
        pairs_by_background.update(
            {
                int(idx): [dict(entry) for entry in entries]
                for idx, entries in payload.items()
                if entries
            }
        )

    gui_manual_geometry.apply_geometry_manual_pairs_rows(
        saved_rows,
        osc_files=osc_files,
        pairs_for_index=_pairs_for_index,
        replace_pairs_by_background=_replace,
        clear_preview_artists=lambda **_kwargs: None,
        cancel_pick_session=lambda **_kwargs: None,
        invalidate_pick_cache=lambda: None,
        clear_manual_undo_stack=lambda: None,
        clear_geometry_fit_undo_stack=lambda: None,
        render_current_pairs=lambda **_kwargs: None,
        update_button_label=lambda: None,
        refresh_status=lambda: None,
    )
    return pairs_by_background


def headless_geometry_fit_required_background_indices(
    selected_indices: Sequence[object],
    *,
    current_background_index: int,
    uses_shared_theta_offset: bool,
) -> list[int]:
    current_idx = int(current_background_index)
    selected = [int(idx) for idx in selected_indices]
    return (
        selected
        if bool(uses_shared_theta_offset)
        else [current_idx if current_idx in selected or not selected else int(selected[0])]
    )


def _headless_geometry_fit_locked_qr_runtime_readiness(
    summary: Mapping[str, object],
) -> dict[str, object]:
    expected = _coerce_nonnegative_int(summary.get("expected_locked_qr_rows"))
    if expected is None:
        expected = _coerce_nonnegative_int(summary.get("qr_fit_expected_count"))
    projected = _coerce_nonnegative_int(summary.get("projected_locked_qr_rows"))
    if projected is None:
        projected = _coerce_nonnegative_int(summary.get("qr_fit_resolved_count"))
    finite = _coerce_nonnegative_int(summary.get("finite_locked_qr_rows"))
    if finite is None:
        finite = _coerce_nonnegative_int(summary.get("manual_caked_residual_row_count"))
    if finite is None:
        finite = _coerce_nonnegative_int(summary.get("raw_angular_row_count"))
    if expected is None or projected is None or finite is None or int(expected) <= 0:
        return {}

    missing_pairs = summary.get("qr_fit_missing_pairs")
    missing_pair_count = (
        len(missing_pairs)
        if isinstance(missing_pairs, Sequence) and not isinstance(missing_pairs, (str, bytes))
        else 0
    )
    projection_ready = bool(
        int(expected) == int(projected) == int(finite)
        and int(missing_pair_count) == 0
        and not bool(summary.get("qr_fit_objective_incomplete", False))
    )
    storage_required = bool(summary.get("caked_view_storage_required_for_fit", False))
    storage_timeout_fatal = bool(summary.get("storage_timeout_fatal", False))
    return {
        "expected_locked_qr_rows": int(expected),
        "projected_locked_qr_rows": int(projected),
        "finite_locked_qr_rows": int(finite),
        "projection_ready": bool(projection_ready),
        "storage_required_for_fit": bool(storage_required),
        "storage_timeout_fatal": bool(storage_timeout_fatal),
    }


def _headless_geometry_fit_load_progress(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _load_structure_model(
    defaults: _RuntimeDefaults,
    saved_state: dict[str, object],
    var_store: dict[str, object],
    simulation_runtime_state: SimulationRuntimeState,
) -> tuple[object, AtomSiteOverrideState, str, complex]:
    calc_runtime = _load_calculation_runtime()
    stack = _load_stacking_fault_runtime()
    state_types = _load_gui_state_types()
    diffraction_tools = _load_diffraction_tools()
    dynamic_lists = saved_state["dynamic_lists"]
    cf, blk = _read_first_cif_block(defaults.primary_cif_path)
    occupancy_site_labels, occupancy_site_expanded_map = (
        gui_structure_model.extract_occupancy_site_metadata(
            blk,
            defaults.primary_cif_path,
        )
    )
    occ_source = dynamic_lists["occupancy_values"]
    if len(occ_source) != len(occupancy_site_labels):
        raise ValueError(
            "GUI state v1 occupancy count must match the primary CIF occupancy sites."
        )
    occ_values = [float(value) for value in occ_source]

    atom_site_fractional_metadata = gui_structure_model.extract_atom_site_fractional_metadata(blk)
    saved_atom_sites = dynamic_lists["atom_site_fractional_values"]
    if len(saved_atom_sites) != len(atom_site_fractional_metadata):
        raise ValueError(
            "GUI state v1 atom-site count must match the primary CIF atom sites."
        )
    atom_site_values = [
        (float(row["x"]), float(row["y"]), float(row["z"]))
        for row in saved_atom_sites
    ]

    defaults_map = copy.deepcopy(defaults.defaults)
    for target_name, var_name in (
        ("a", "a_var"),
        ("c", "c_var"),
        ("p0", "p0_var"),
        ("p1", "p1_var"),
        ("p2", "p2_var"),
        ("w0", "w0_var"),
        ("w1", "w1_var"),
        ("w2", "w2_var"),
    ):
        defaults_map[target_name] = _coerce_float(
            var_store[var_name],
            float(defaults_map[target_name]),
        )
    defaults_map["finite_stack"] = _coerce_bool(
        var_store["finite_stack_var"],
        defaults.defaults["finite_stack"],
    )
    default_film_thickness_nm = _default_finite_stack_film_thickness_nm(defaults.defaults)
    defaults_map["film_thickness_nm"] = gui_controllers.normalize_finite_stack_thickness_nm(
        var_store["film_thickness_nm_var"],
        default_film_thickness_nm,
    )
    defaults_map["stack_layers"] = gui_controllers.finite_stack_layers_from_thickness_nm(
        film_thickness_nm=defaults_map["film_thickness_nm"],
        c_axis_angstrom=defaults_map["c"],
    )
    defaults_map["phase_delta_expression"] = _require_phase_delta_expression(
        var_store["phase_delta_expr_var"],
        stack.validate_phase_delta_expression,
    )
    defaults_map["phi_l_divisor"] = _require_positive_float(
        var_store["phi_l_divisor_var"],
        "phi_l_divisor",
    )

    structure_state = gui_structure_model.build_initial_structure_model_state(
        cif_file=defaults.primary_cif_path,
        cf=cf,
        blk=blk,
        cif_file2=defaults.secondary_cif_path,
        occupancy_site_labels=occupancy_site_labels,
        occupancy_site_expanded_map=occupancy_site_expanded_map,
        occ=occ_values,
        atom_site_fractional_metadata=atom_site_fractional_metadata,
        av=float(defaults.defaults["a"]),
        bv=float(defaults.defaults["b"]),
        cv=float(defaults.defaults["c"]),
        av2=defaults.defaults.get("a2"),
        cv2=defaults.defaults.get("c2"),
        defaults=defaults_map,
        mx=defaults.mx,
        lambda_angstrom=defaults.lambda_angstrom,
        intensity_threshold=defaults.intensity_threshold,
        two_theta_range=defaults.two_theta_range,
        include_rods_flag=defaults.include_rods_flag,
        combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
        miller_generator=diffraction_tools.miller_generator,
        inject_fractional_reflections=diffraction_tools.inject_fractional_reflections,
    )
    atom_site_override_state = state_types.AtomSiteOverrideState()
    active_cif_path = gui_structure_model.active_primary_cif_path(
        structure_state,
        atom_site_override_state,
        atom_site_values=atom_site_values,
    )
    rich_phase_layer_form_factor_provider = (
        gui_structure_model._cached_rich_phase_layer_form_factor_provider(
            structure_state,
            active_cif_path,
            source_cif_path=structure_state.cif_file,
        )
    )
    gui_structure_model.rebuild_diffraction_inputs(
        structure_state,
        new_occ=occ_values,
        p_vals=[
            _coerce_float(var_store["p0_var"], defaults.defaults["p0"]),
            _coerce_float(var_store["p1_var"], defaults.defaults["p1"]),
            _coerce_float(var_store["p2_var"], defaults.defaults["p2"]),
        ],
        weights=gui_controllers.normalize_stacking_weight_values(
            [
                var_store["w0_var"],
                var_store["w1_var"],
                var_store["w2_var"],
            ]
        ),
        a_axis=_coerce_float(var_store["a_var"], defaults.defaults["a"]),
        c_axis=_coerce_float(var_store["c_var"], defaults.defaults["c"]),
        finite_stack_flag=_coerce_bool(
            var_store["finite_stack_var"],
            defaults.defaults["finite_stack"],
        ),
        layers=int(defaults_map["stack_layers"]),
        phase_delta_expression_current=_require_phase_delta_expression(
            var_store["phase_delta_expr_var"],
            stack.validate_phase_delta_expression,
        ),
        phi_l_divisor_current=_require_positive_float(
            var_store["phi_l_divisor_var"],
            "phi_l_divisor",
        ),
        atom_site_values=atom_site_values,
        iodine_z_current=gui_structure_model.current_iodine_z(
            structure_state,
            atom_site_override_state,
            active_cif_path=active_cif_path,
            atom_site_values=atom_site_values,
            layer_form_factor_provider=rich_phase_layer_form_factor_provider,
        ),
        atom_site_override_state=atom_site_override_state,
        simulation_runtime_state=simulation_runtime_state,
        combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
        apply_bragg_qr_filters=lambda **_kwargs: None,
        schedule_update=lambda: None,
        weight1=_coerce_float(var_store["weight1_var"], defaults.defaults["weight1"]),
        weight2=_coerce_float(var_store["weight2_var"], defaults.defaults["weight2"]),
        primary_source_mode="rich_phase",
        force=True,
        trigger_update=False,
    )
    nominal_n2 = calc_runtime.resolve_index_of_refraction(
        defaults.lambda_angstrom * 1.0e-10,
        cif_path=active_cif_path,
    )
    return structure_state, atom_site_override_state, str(active_cif_path), nominal_n2


def _sync_background_theta_state(
    defaults: _RuntimeDefaults,
    var_store: dict[str, object],
) -> None:
    selection = RuntimeValueBinding(
        partial(var_store.__getitem__, "geometry_fit_background_selection_var"),
        partial(var_store.__setitem__, "geometry_fit_background_selection_var"),
    )
    theta = RuntimeValueBinding(
        partial(var_store.__getitem__, "theta_initial_var"),
        partial(var_store.__setitem__, "theta_initial_var"),
    )
    theta_list = RuntimeValueBinding(
        partial(var_store.__getitem__, "background_theta_list_var"),
        partial(var_store.__setitem__, "background_theta_list_var"),
    )
    if gui_background_theta.geometry_fit_uses_shared_theta_offset(
        osc_files=defaults.osc_files,
        current_background_index=defaults.current_background_index,
        geometry_fit_background_selection=selection,
    ):
        return
    theta_values = gui_background_theta.current_background_theta_values(
        osc_files=defaults.osc_files,
        theta_initial=theta,
        defaults={"theta_initial": defaults.defaults["theta_initial"]},
        theta_initial_fallback=defaults.defaults["theta_initial"],
        background_theta_list=theta_list,
        strict_count=True,
    )
    idx = min(max(defaults.current_background_index, 0), len(theta_values) - 1)
    theta_value = float(var_store["theta_initial_var"])
    if not np.isfinite(theta_value):
        raise ValueError("GUI state v1 theta_initial_var must be finite.")
    theta_values[idx] = theta_value
    var_store["background_theta_list_var"] = (
        gui_background_theta.format_background_theta_values(theta_values)
    )


def _updated_state_snapshot(
    saved_state: dict[str, object],
    defaults: _RuntimeDefaults,
    var_store: dict[str, object],
) -> dict[str, object]:
    _sync_background_theta_state(defaults, var_store)
    updated_state = copy.deepcopy(saved_state)
    variables = updated_state["variables"]
    variables.update(var_store)
    return updated_state




def _copy_intersection_cache_tables(
    cache: Sequence[object] | None,
) -> list[np.ndarray]:
    schema = _load_intersection_cache_schema()
    copied: list[np.ndarray] = []
    if not isinstance(cache, Sequence) or isinstance(cache, (str, bytes)):
        return copied
    for table in cache:
        copied.append(schema.coerce_intersection_cache_table(table))
    return copied


def _set_runtime_peak_cache_from_source_rows(
    simulation_runtime_state: SimulationRuntimeState,
    source_rows: Sequence[object] | None,
) -> None:
    restored_records: list[dict[str, object]] = []
    restored_positions: list[tuple[float, float]] = []
    restored_millers: list[tuple[int, int, int]] = []
    restored_intensities: list[float] = []

    for raw_entry in source_rows or ():
        if not isinstance(raw_entry, Mapping):
            continue
        peak_record = gui_manual_geometry.geometry_manual_canonicalize_live_source_entry(
            raw_entry,
            normalize_hkl_key=gui_geometry_overlay.normalize_hkl_key,
        )
        if peak_record is None:
            continue
        try:
            display_col = float(peak_record.get("sim_col", np.nan))
            display_row = float(peak_record.get("sim_row", np.nan))
        except Exception:
            continue
        if not (np.isfinite(display_col) and np.isfinite(display_row)):
            continue

        hkl_value = peak_record.get("hkl")
        if not isinstance(hkl_value, tuple) or len(hkl_value) < 3:
            continue
        try:
            hkl_triplet = (
                int(hkl_value[0]),
                int(hkl_value[1]),
                int(hkl_value[2]),
            )
        except Exception:
            continue

        try:
            intensity = float(peak_record.get("weight", peak_record.get("intensity", 0.0)))
        except Exception:
            intensity = 0.0
        if not np.isfinite(intensity):
            intensity = 0.0

        peak_record["display_col"] = float(display_col)
        peak_record["display_row"] = float(display_row)
        peak_record["intensity"] = float(intensity)
        peak_record["weight"] = float(intensity)

        restored_records.append(peak_record)
        restored_positions.append((float(display_col), float(display_row)))
        restored_millers.append(hkl_triplet)
        restored_intensities.append(float(intensity))

    simulation_runtime_state.peak_records = restored_records
    simulation_runtime_state.peak_positions = restored_positions
    simulation_runtime_state.peak_millers = restored_millers
    simulation_runtime_state.peak_intensities = restored_intensities
    simulation_runtime_state.selected_peak_record = None
    simulation_runtime_state.peak_overlay_cache = (
        {
            "sig": None,
            "positions": list(restored_positions),
            "millers": list(restored_millers),
            "intensities": list(restored_intensities),
            "records": [dict(record) for record in restored_records],
        }
        if restored_records
        else _empty_peak_overlay_cache()
    )


def _build_source_rows_from_hit_tables(
    hit_tables: Sequence[object] | None,
    *,
    image_size_value: int,
    params_local: Mapping[str, object],
    native_sim_to_display_coords,
    allow_nominal_hkl_indices: bool,
    project_source_rows_to_caked=None,
    native_detector_coords_to_caked_display_coords=None,
    centroid_hit_table_coords_to_caked_display_coords=None,
    centroid_projection_provenance: Mapping[str, object] | None = None,
    centroid_frozen_roi_native_detector_coords_to_caked_display_coords=None,
    required_manual_fit_targets: Sequence[Mapping[str, object]] | None = None,
) -> tuple[
    list[dict[str, object]],
    list[tuple[float, float, str]],
    list[np.ndarray],
]:
    copied_hit_tables = gui_geometry_q_group_manager.copy_geometry_fit_hit_tables(hit_tables)
    if not copied_hit_tables:
        return [], [], []

    try:
        primary_a = float(params_local.get("a", np.nan))
    except Exception:
        primary_a = float("nan")
    try:
        primary_c = float(params_local.get("c", np.nan))
    except Exception:
        primary_c = float("nan")
    try:
        theta_initial_value = float(params_local.get("theta_initial", np.nan))
    except Exception:
        theta_initial_value = float("nan")

    raw_rows, peak_table_lattice = (
        gui_geometry_q_group_manager.build_geometry_fit_full_order_source_rows(
            copied_hit_tables,
            image_shape=(int(image_size_value), int(image_size_value)),
            native_sim_to_display_coords=native_sim_to_display_coords,
            primary_a=primary_a,
            primary_c=primary_c,
            default_source_label="primary",
            round_pixel_centers=False,
            allow_nominal_hkl_indices=bool(allow_nominal_hkl_indices),
        )
    )
    raw_rows = [dict(entry) for entry in (raw_rows or ()) if isinstance(entry, Mapping)]
    if np.isfinite(theta_initial_value):
        for entry in raw_rows:
            entry["theta_initial"] = float(theta_initial_value)
            entry["theta_initial_deg"] = float(theta_initial_value)
    centroid_hit_tables = getattr(copied_hit_tables, "centroid_hit_tables", None)
    if (
        raw_rows
        and centroid_hit_tables is not None
        and callable(project_source_rows_to_caked)
        and callable(native_detector_coords_to_caked_display_coords)
    ):
        try:
            projected_target_rows = [
                dict(entry)
                for entry in (project_source_rows_to_caked(raw_rows) or ())
                if isinstance(entry, Mapping)
            ]
            projected_target_rows = gui_geometry_q_group_manager.stamp_selected_branch_frozen_caked_centroid_roi_centers(
                projected_target_rows,
                required_manual_fit_targets=required_manual_fit_targets,
                frozen_roi_native_detector_coords_to_caked_display_coords=(
                    centroid_frozen_roi_native_detector_coords_to_caked_display_coords
                ),
            )
            projected_target_rows = (
                gui_geometry_q_group_manager.attach_selected_branch_caked_centroids_from_hit_tables(
                    projected_target_rows,
                    centroid_hit_tables,
                    native_detector_coords_to_caked_display_coords=(
                        centroid_hit_table_coords_to_caked_display_coords
                        if callable(centroid_hit_table_coords_to_caked_display_coords)
                        else native_detector_coords_to_caked_display_coords
                    ),
                )
            )
            projected_target_rows = gui_geometry_q_group_manager.stamp_selected_branch_caked_centroid_projector_provenance(
                projected_target_rows,
                centroid_projection_provenance,
            )
        except Exception:
            projected_target_rows = []
        if projected_target_rows:
            centroid_fields_by_key: dict[tuple[object, ...], dict[str, object]] = {}
            for projected_entry in projected_target_rows:
                key = gui_geometry_q_group_manager._centroid_source_row_group_key(projected_entry)
                if key is None or "selected_branch_caked_centroid_deg" not in projected_entry:
                    continue
                centroid_fields = {
                    field: value
                    for field, value in projected_entry.items()
                    if str(field).startswith("selected_branch_caked_centroid")
                    or str(field).startswith("process_peaks_centroid")
                    or str(field).startswith("predicted_centroid_")
                }
                if centroid_fields:
                    centroid_fields_by_key[key] = centroid_fields
            if centroid_fields_by_key:
                attached_rows: list[dict[str, object]] = []
                for raw_entry in raw_rows:
                    copied = dict(raw_entry)
                    key = gui_geometry_q_group_manager._centroid_source_row_group_key(copied)
                    if key in centroid_fields_by_key:
                        copied.update(centroid_fields_by_key[key])
                    attached_rows.append(copied)
                raw_rows = attached_rows
    return raw_rows, peak_table_lattice, copied_hit_tables


_HEADLESS_GEOMETRY_FIT_PROGRESS_PHASES = frozenset(
    {
        "preflight",
        "runtime_config_ready",
        "solve_start",
        "final_validation",
        "output_state_write",
    }
)


def _headless_geometry_fit_state_provenance(state_path: str | Path) -> dict[str, object]:
    state_file = Path(state_path).expanduser().resolve()
    try:
        with state_file.open("rb") as stream:
            state_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        state_hash = None
    return {
        "input_state_path": state_file,
        "input_state_sha256": state_hash,
    }


def _headless_progress_jsonable(
    value: object,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> object:
    if _seen is None:
        _seen = set()
    if int(_depth) > 20:
        return {"status": "max_depth"}
    if isinstance(value, np.ndarray):
        shape = [int(dim) for dim in value.shape]
        dtype = str(value.dtype)
        if value.dtype.kind != "O" and int(value.size) <= 256:
            try:
                return _headless_progress_jsonable(value.tolist(), _seen, int(_depth) + 1)
            except Exception:
                pass
        return {"type": "ndarray", "shape": shape, "dtype": dtype}
    if isinstance(value, np.generic):
        try:
            return _headless_progress_jsonable(value.item(), _seen, int(_depth) + 1)
        except Exception:
            return {"type": "numpy_scalar", "dtype": str(getattr(value, "dtype", ""))}
    if isinstance(value, (Path, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except Exception:
            return {"type": _headless_progress_type_label(value)}
        if math.isfinite(number):
            return value
        return str(value)
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in _seen:
            return {"type": _headless_progress_type_label(value), "status": "cycle"}
        _seen.add(marker)
        try:
            result: dict[str, object] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= 500:
                    result["<truncated>"] = int(len(value))
                    break
                if type(key) in (str, int, float, bool):
                    key_text = str(key)
                else:
                    key_text = f"<{_headless_progress_type_label(key)}>"
                result[key_text] = _headless_progress_jsonable(item, _seen, int(_depth) + 1)
            return result
        finally:
            _seen.discard(marker)
    if isinstance(value, (list, tuple, set)):
        marker = id(value)
        if marker in _seen:
            return {"type": _headless_progress_type_label(value), "status": "cycle"}
        _seen.add(marker)
        try:
            result = []
            for index, item in enumerate(value):
                if index >= 1000:
                    result.append({"status": "truncated", "length": int(len(value))})
                    break
                result.append(_headless_progress_jsonable(item, _seen, int(_depth) + 1))
            return result
        finally:
            _seen.discard(marker)
    return {"type": _headless_progress_type_label(value)}


_HEADLESS_GEOMETRY_FIT_PREFLIGHT_EVENT_DETAIL_KEYS = frozenset(
    (
        "status",
        "valid",
        "reason",
        "validator_failure_reason",
        "consumer",
        "required_pair_count",
        "validated_pair_count",
        "missing_required_pair_count",
        "branch_mismatch_count",
        "hkl_missing_candidate_count",
        "row_count",
        "raw_row_count",
        "validator_finite_detector_rows",
        "validator_finite_caked_rows",
        "validator_caked_only_rows_allowed",
        "signature_match",
        "trial_source_rows_signature_mismatch_reused",
        "pair_failure_reasons",
    )
)


def _headless_geometry_fit_compact_preflight_event(
    *,
    stage: str,
    payload: Mapping[str, object] | object,
    event_index: int,
    elapsed_s: float,
) -> dict[str, object]:
    event_payload = dict(payload) if isinstance(payload, Mapping) else {"payload": str(payload)}
    compact_event: dict[str, object] = {
        "event_index": int(event_index),
        "stage": str(stage),
        "message": event_payload.get("message"),
        "background_index": event_payload.get("background_index"),
        "dataset_index": event_payload.get("dataset_index"),
        "elapsed_s": float(max(0.0, elapsed_s)),
    }
    for key in sorted(_HEADLESS_GEOMETRY_FIT_PREFLIGHT_EVENT_DETAIL_KEYS):
        if key in event_payload:
            compact_event[key] = _headless_progress_jsonable(event_payload.get(key))
    return compact_event


_HEADLESS_PROGRESS_LIVE_CACHE_RECORD_KEYS = frozenset(
    (
        "dataset_index",
        "dataset_label",
        "background_index",
        "background_path",
        "cif_path",
        "pair_id",
        "q_group_key",
        "hkl",
        "source_branch_index",
        "observed_caked_deg",
        "predicted_caked_deg",
        "fit_observed_caked_deg",
        "fit_prediction_caked_deg",
        "fit_prediction_caked_deg_source",
        "background_two_theta_deg",
        "background_phi_deg",
        "caked_x",
        "caked_y",
        "sim_nominal_caked_deg",
        "sim_refined_caked_deg",
        "simulated_two_theta_deg",
        "simulated_phi_deg",
        "fit_prediction_detector_display_px",
        "fit_prediction_detector_display_px_source",
        "fit_prediction_detector_display_px_unavailable_reason",
        "fit_prediction_detector_native_px",
        "fit_prediction_detector_native_px_source",
        "final_prediction_detector_native_px",
        "final_prediction_detector_native_px_source",
        "final_prediction_caked_deg",
        "final_prediction_source",
        "objective_source_authority",
        "fit_prediction_caked_authority",
        "locked_qr_detector_point_source",
        "resolver_path",
        "actual_source",
        "source_kind",
        "coordinate_provenance",
        "projection_frame",
        "is_dynamic_trial_row",
        "residual_caked_deg",
        "fit_residual_caked_deg",
        "angular_residual_norm_deg",
        "delta_two_theta_deg",
        "wrapped_delta_phi_deg",
        "source_authority_match",
        "visual_objective_surface_match",
        "branch_match",
    )
)


_HEADLESS_PROGRESS_LIVE_POINT_SUMMARY_KEYS = frozenset(
    (
        "acceptance_metric_space",
        "metric_unit",
        "fixed_source_resolved_count",
        "matched_pair_count",
        "missing_pair_count",
        "qr_fit_expected_count",
        "qr_fit_resolved_count",
        "qr_fit_missing_count",
        "source_authority_mismatch_count",
        "visual_objective_surface_mismatch_count",
        "branch_mismatch_count",
        "dynamic_source_coordinate_recompute_count",
        "dynamic_objective_trial_locked_qr_row_count",
        "dynamic_objective_trial_saved_source_count",
        "dynamic_objective_trial_coverage_complete",
        "line_group_count",
        "resolved_line_group_count",
        "missing_line_group_count",
        "line_angle_rms_deg",
        "line_offset_rms_deg",
        "line_fit_space_span_deg_mean",
        "manual_pick_cache_rebuild_count",
        "caked_projection_rebuild_count",
        "failure_classification",
    )
)


def _headless_compact_live_cache_record(record: object) -> object:
    if not isinstance(record, Mapping):
        return record
    return {
        str(key): record.get(key)
        for key in _HEADLESS_PROGRESS_LIVE_CACHE_RECORD_KEYS
        if key in record
    }


def _headless_compact_live_point_match_summary(summary: object) -> object:
    if not isinstance(summary, Mapping):
        return {}
    compact: dict[str, object] = {}
    for key in _HEADLESS_PROGRESS_LIVE_POINT_SUMMARY_KEYS:
        if key not in summary:
            continue
        value = summary.get(key)
        if isinstance(value, np.generic):
            try:
                value = value.item()
            except Exception:
                value = str(value)
        if (
            value is None
            or isinstance(value, (str, bool, int, float))
            or type(value).__module__ == "builtins"
            and type(value).__name__ in {"str", "bool", "int", "float"}
        ):
            compact[str(key)] = value
    return compact


def _headless_progress_live_payload(value: object) -> object:
    if isinstance(value, Mapping):
        compact_value: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text == "live_cache_records":
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                    compact_value[key_text] = [
                        _headless_compact_live_cache_record(record)
                        for record in item
                        if isinstance(record, Mapping)
                    ]
                else:
                    compact_value[key_text] = []
                continue
            if key_text == "point_match_summary":
                compact_value[key_text] = _headless_compact_live_point_match_summary(item)
                continue
            compact_value[key_text] = item
        payload = _headless_progress_jsonable(compact_value)
    else:
        payload = _headless_progress_jsonable(value)
    if not isinstance(payload, Mapping):
        return payload
    records = payload.get("live_cache_records")
    if not isinstance(records, list):
        return payload
    updated = dict(payload)
    updated["live_cache_records"] = records
    return updated


class _HeadlessGeometryFitProgressWriter:
    """Private JSON sidecar for long headless geometry fits."""

    def __init__(
        self,
        path: str | Path | None,
        *,
        active_vars: Sequence[object] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve() if path is not None else None
        self.started_at = time.monotonic()
        self.data: dict[str, object] = {
            "phase": "preflight",
            "elapsed_s": 0.0,
            "pid": int(os.getpid()),
            "active_vars": [str(name) for name in (active_vars or ())],
            "request_build_s": None,
            "initial_objective_s": None,
            "least_squares_s": None,
            "residual_eval_count": 0,
            "mean_residual_eval_s": None,
            "max_residual_eval_s": None,
            "optimizer_nfev": None,
            "optimizer_njev": None,
            "manual_pick_cache_rebuild_count": 0,
            "caked_projection_rebuild_count": 0,
            "dynamic_source_coordinate_recompute_count": 0,
            "fixed_source_resolved_count": 0,
            "matched_pair_count": 0,
            "missing_pair_count": 0,
        }

    def update_static(
        self,
        *,
        active_vars: Sequence[object] | None = None,
        runtime_cfg: Mapping[str, object] | None = None,
        max_nfev_override: int | None = None,
    ) -> None:
        updates: dict[str, object] = {}
        if active_vars is not None:
            updates["active_vars"] = [str(name) for name in active_vars]
        if max_nfev_override is not None:
            updates["max_nfev_override"] = int(max_nfev_override)
        if isinstance(runtime_cfg, Mapping):
            optimizer = runtime_cfg.get("optimizer")
            optimizer_map = optimizer if isinstance(optimizer, Mapping) else {}
            updates.update(
                {
                    "optimizer_max_nfev": optimizer_map.get("max_nfev"),
                }
            )
        if updates:
            self.write(str(self.data.get("phase", "preflight")), **updates)

    def _merge_point_match_summary(self, summary: Mapping[str, object]) -> dict[str, object]:
        updates: dict[str, object] = {}
        updates.update(_headless_geometry_fit_locked_qr_runtime_readiness(summary))
        worst_failure_row = _headless_geometry_fit_summary_worst_failure_row(
            {"point_match_summary": summary}
        )
        if worst_failure_row is not None:
            failing_pair = worst_failure_row.get("pair_id")
            failing_branch = worst_failure_row.get("source_branch_index")
            failure_classification = _headless_geometry_fit_summary_first_available(
                summary.get("failure_classification"),
                summary.get("dynamic_angular_failure_classification"),
                worst_failure_row.get("failure_classification"),
            )
            if failing_pair is not None:
                updates["first_failing_pair_id"] = failing_pair
            if failing_branch is not None:
                updates["first_failing_branch_index"] = failing_branch
            if failure_classification is not None:
                updates["failure_classification"] = failure_classification
        coverage_payload = dict(summary)
        coverage_payload.update(updates)
        coverage = gui_geometry_fit._geometry_fit_dynamic_objective_trial_locked_qr_coverage(
            coverage_payload
        )
        for key in _HEADLESS_GEOMETRY_FIT_DYNAMIC_OBJECTIVE_TRIAL_COVERAGE_KEYS:
            if key in coverage:
                updates[key] = coverage.get(key)
        for key in (
            "fixed_source_resolved_count",
            "matched_pair_count",
            "missing_pair_count",
            "dynamic_source_coordinate_recompute_count",
            "line_group_count",
            "resolved_line_group_count",
            "missing_line_group_count",
        ):
            if key in summary:
                updates[key] = _headless_progress_int(summary.get(key), 0)
        for key in (
            "line_angle_rms_deg",
            "line_offset_rms_deg",
            "line_fit_space_span_deg_mean",
        ):
            if key in summary:
                updates[key] = _headless_progress_float(summary.get(key))
        for key in (
            "manual_pick_cache_rebuild_count",
            "caked_projection_rebuild_count",
        ):
            if key in summary:
                updates[key] = _headless_progress_int(summary.get(key), 0)
        return updates

    def status(self, text: object) -> None:
        try:
            message = str(text).strip()
        except Exception:
            message = ""
        if not message:
            return
        phase = str(self.data.get("phase", "preflight"))
        if " eval=" in message:
            phase = "solve_start"
        self.write(phase, status_text=message)

    def live_update(self, payload: Mapping[str, object]) -> None:
        if not isinstance(payload, Mapping):
            return
        updates: dict[str, object] = {}
        eval_count: int | None = None
        if "evaluation_count" in payload:
            eval_count = _headless_progress_int(payload.get("evaluation_count"), 0)
            updates["residual_eval_count"] = int(eval_count)
            updates["optimizer_nfev"] = updates["residual_eval_count"]
        for source_key, dest_key in (
            ("mean_residual_eval_s", "mean_residual_eval_s"),
            ("max_residual_eval_s", "max_residual_eval_s"),
            ("last_residual_eval_s", "last_residual_eval_s"),
        ):
            if source_key in payload:
                updates[dest_key] = _headless_progress_float(payload.get(source_key))
        summary = payload.get("point_match_summary")
        if isinstance(summary, Mapping):
            updates.update(self._merge_point_match_summary(summary))
        if eval_count is not None:
            history = [
                dict(row)
                for row in (self.data.get("cost_history", []) or [])
                if isinstance(row, Mapping)
            ]
            history.append(
                {
                    "eval": int(eval_count),
                    "current_cost": _headless_progress_float(payload.get("current_cost")),
                    "best_cost": _headless_progress_float(payload.get("best_cost")),
                    "weighted_rms_px": _headless_progress_float(payload.get("weighted_rms_px")),
                    "improved": bool(payload.get("improved", False)),
                    "qr_fit_resolved_count": updates.get("qr_fit_resolved_count"),
                    "qr_fit_expected_count": updates.get("qr_fit_expected_count"),
                    "dynamic_objective_trial_coverage_complete": updates.get(
                        "dynamic_objective_trial_coverage_complete"
                    ),
                    "fixed_source_resolved_count": updates.get("fixed_source_resolved_count"),
                    "matched_pair_count": updates.get("matched_pair_count"),
                    "missing_pair_count": updates.get("missing_pair_count"),
                }
            )
            updates["cost_history"] = history
        updates["last_live_update"] = _headless_progress_live_payload(payload)
        self.write("solve_start", **updates)

    def write(self, phase: str, **updates: object) -> None:
        if self.path is None:
            return
        phase_name = str(phase or self.data.get("phase") or "preflight")
        if phase_name not in _HEADLESS_GEOMETRY_FIT_PROGRESS_PHASES:
            phase_name = str(self.data.get("phase") or "preflight")
        self.data["phase"] = phase_name
        self.data["elapsed_s"] = float(max(0.0, time.monotonic() - self.started_at))
        self.data["pid"] = int(os.getpid())
        self.data.update(
            {str(key): _headless_progress_jsonable(value) for key, value in updates.items()}
        )
        with suppress(Exception):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_name(f"{self.path.name}.tmp")
            tmp_path.write_text(
                json.dumps(_headless_progress_jsonable(self.data), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp_path.replace(self.path)


def headless_geometry_fit_progress_report_fields(
    progress_path: str | Path | None,
) -> dict[str, object]:
    if progress_path is None:
        return {}
    path = Path(progress_path)
    return {"progress_json": str(path), **_headless_geometry_fit_load_progress(path)}


def _headless_geometry_fit_summary_first_available(*values: object) -> object:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _headless_geometry_fit_summary_worst_failure_row(
    progress_payload: Mapping[str, object],
) -> Mapping[str, object] | None:
    summary = progress_payload.get("point_match_summary")
    if not isinstance(summary, Mapping):
        return None
    rows = summary.get("worst_angular_residual_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return None

    best_row: Mapping[str, object] | None = None
    best_norm = float("-inf")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        norm = _headless_progress_float(row.get("angular_residual_norm_deg"))
        if norm is None:
            norm = _headless_progress_float(row.get("final_residual_norm"))
        if norm is None:
            norm = _headless_progress_float(row.get("residual_norm_deg"))
        norm_value = float(norm) if norm is not None else float("nan")
        if best_row is None or (math.isfinite(norm_value) and norm_value > best_norm):
            best_row = row
            if math.isfinite(norm_value):
                best_norm = float(norm_value)
    return best_row


def _headless_geometry_fit_param_delta_fields(
    *,
    active_var_names: Sequence[object],
    initial_params: Mapping[str, object],
    final_params: Mapping[str, object],
) -> dict[str, object]:
    fields: dict[str, object] = {}
    seen: set[str] = set()
    for raw_name in active_var_names or ():
        name = str(raw_name)
        if not name or name in seen:
            continue
        seen.add(name)
        fields[f"{name}_before"] = initial_params.get(name)
        fields[f"{name}_after"] = final_params.get(name)
    return fields


def _headless_geometry_fit_report_final_params(
    *,
    current_params: Mapping[str, object],
    active_var_names: Sequence[object],
    solver_result: object | None,
    progress_data: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return final params for diagnostics, even when the state write is rejected."""

    final_params = dict(current_params or {})
    names = [str(name) for name in (active_var_names or ())]
    if not names:
        return final_params

    candidate_values: list[object] = []
    if solver_result is not None:
        candidate_values.append(getattr(solver_result, "x", None))
        progress = getattr(solver_result, "geometry_fit_progress", None)
        if isinstance(progress, Mapping):
            candidate_values.append(progress.get("end_x"))
            candidate_values.append(progress.get("last_x_trial"))
    if isinstance(progress_data, Mapping):
        live_update = progress_data.get("last_live_update")
        if isinstance(live_update, Mapping):
            candidate_values.append(live_update.get("x_trial"))

    for raw_values in candidate_values:
        if raw_values is None:
            continue
        try:
            values = np.asarray(raw_values, dtype=float).reshape(-1)
        except Exception:
            continue
        if values.size < len(names) or not np.all(np.isfinite(values[: len(names)])):
            continue
        for idx, name in enumerate(names):
            final_params[name] = float(values[idx])
        final_params["_diagnostic_final_params_source"] = "solver_result"
        return final_params

    final_params["_diagnostic_final_params_source"] = "current_params"
    return final_params


def _headless_geometry_fit_final_reporting_fields(
    *,
    solver_result: object | None,
    active_var_names: Sequence[object],
    initial_params: Mapping[str, object],
    current_params: Mapping[str, object],
    progress_data: Mapping[str, object] | None,
) -> dict[str, object]:
    fields: dict[str, object] = {}
    final_summary = getattr(solver_result, "point_match_summary", None)
    summary = final_summary if isinstance(final_summary, Mapping) else {}
    for name in (
        "initial_residual_rms",
        "final_residual_rms",
        "initial_residual_norm",
        "final_residual_norm",
    ):
        value = _headless_progress_float(getattr(solver_result, name, None))
        if value is not None:
            fields[name] = float(value)
    if "initial_residual_rms" not in fields:
        initial_rms = _headless_progress_float(
            summary.get(
                "identity_baseline_point_rms_deg",
                summary.get("initial_residual_rms"),
            )
        )
        if initial_rms is not None:
            fields["initial_residual_rms"] = float(initial_rms)
    if "final_residual_rms" not in fields:
        final_rms = _headless_progress_float(
            summary.get("raw_angular_rms_deg", summary.get("final_rms_deg"))
        )
        if final_rms is not None:
            fields["final_residual_rms"] = float(final_rms)
    for name in ("initial_residual_count", "final_residual_count"):
        raw_value = getattr(solver_result, name, None)
        if raw_value is not None:
            fields[name] = _headless_progress_int(raw_value, 0)

    for attr_name, field_name in (
        ("success", "least_squares_success"),
        ("status", "least_squares_status"),
        ("message", "least_squares_message"),
        ("nfev", "least_squares_nfev"),
        ("njev", "least_squares_njev"),
        ("cost", "least_squares_cost"),
        ("initial_cost", "least_squares_initial_cost"),
        ("final_cost", "least_squares_final_cost"),
        ("cost_reduction", "least_squares_cost_reduction"),
        ("optimality", "least_squares_optimality"),
    ):
        if solver_result is None:
            continue
        raw_value = getattr(solver_result, attr_name, None)
        if raw_value is None:
            continue
        if attr_name == "message":
            fields[field_name] = str(raw_value)
        elif attr_name == "success":
            fields[field_name] = bool(raw_value)
        elif attr_name in {"status", "nfev", "njev"}:
            fields[field_name] = _headless_progress_int(raw_value, 0)
        else:
            value = _headless_progress_float(raw_value)
            if value is not None:
                fields[field_name] = float(value)
    if solver_result is not None:
        grad = getattr(solver_result, "grad", None)
        if grad is not None:
            try:
                grad_arr = np.asarray(grad, dtype=float).reshape(-1)
            except Exception:
                grad_arr = np.asarray([], dtype=float)
            if grad_arr.size and np.all(np.isfinite(grad_arr)):
                fields["least_squares_gradient_norm"] = float(np.linalg.norm(grad_arr))
        active_mask = getattr(solver_result, "active_mask", None)
        if active_mask is not None:
            try:
                active_arr = np.asarray(active_mask, dtype=int).reshape(-1)
            except Exception:
                active_arr = np.asarray([], dtype=int)
            if active_arr.size:
                fields["least_squares_active_mask"] = active_arr.tolist()
                fields["least_squares_active_bounds"] = [
                    {
                        "parameter": str(active_var_names[index]),
                        "active_mask": int(active_arr[index]),
                    }
                    for index in range(min(len(active_var_names), active_arr.size))
                    if int(active_arr[index]) != 0
                ]
        if bool(getattr(solver_result, "point_only_residual_eval_cap_reached", False)):
            fields["point_only_residual_eval_cap_reached"] = True
        x_final = getattr(solver_result, "x", None)
        x_initial = getattr(solver_result, "x0", None)
        if x_initial is None:
            x_initial = getattr(solver_result, "initial_x", None)
        if x_initial is None:
            progress = getattr(solver_result, "geometry_fit_progress", None)
            if isinstance(progress, Mapping):
                x_initial = progress.get("start_x")
        if x_initial is not None and x_final is not None:
            try:
                start_arr = np.asarray(x_initial, dtype=float).reshape(-1)
                final_arr = np.asarray(x_final, dtype=float).reshape(-1)
            except Exception:
                start_arr = np.asarray([], dtype=float)
                final_arr = np.asarray([], dtype=float)
            count = min(start_arr.size, final_arr.size)
            if (
                count
                and np.all(np.isfinite(start_arr[:count]))
                and np.all(np.isfinite(final_arr[:count]))
            ):
                fields["least_squares_step_norm"] = float(
                    np.linalg.norm(final_arr[:count] - start_arr[:count])
                )
        if "least_squares_final_cost" not in fields and "least_squares_cost" in fields:
            fields["least_squares_final_cost"] = float(fields["least_squares_cost"])

    final_params = _headless_geometry_fit_report_final_params(
        current_params=current_params,
        active_var_names=active_var_names,
        solver_result=solver_result,
        progress_data=progress_data,
    )
    fields.update(
        _headless_geometry_fit_param_delta_fields(
            active_var_names=active_var_names,
            initial_params=initial_params,
            final_params=final_params,
        )
    )
    final_params_source = final_params.get("_diagnostic_final_params_source")
    if final_params_source:
        fields["diagnostic_final_params_source"] = final_params_source
    return fields


def _headless_source_row_native_detector_point(
    entry: Mapping[str, object],
) -> tuple[float, float] | None:
    for x_key, y_key in (
        ("refined_sim_native_x", "refined_sim_native_y"),
        ("sim_native_x", "sim_native_y"),
        ("native_col", "native_row"),
        ("sim_detector_anchor_x", "sim_detector_anchor_y"),
    ):
        try:
            point = (float(entry.get(x_key)), float(entry.get(y_key)))
        except Exception:
            continue
        if np.isfinite(point[0]) and np.isfinite(point[1]):
            return float(point[0]), float(point[1])
    return None


def _headless_project_source_rows_to_exact_caked_view(
    rows: Sequence[Mapping[str, object]],
    *,
    background_index: int,
    native_detector_coords_to_caked_display_coords: Callable[[float, float], object] | None,
    simulation_native_coords_to_caked_display_coords: (
        Callable[[float, float], object] | None
    ) = None,
    radial_axis: object = None,
    azimuth_axis: object = None,
) -> list[dict[str, object]]:
    if not (
        callable(native_detector_coords_to_caked_display_coords)
        or callable(simulation_native_coords_to_caked_display_coords)
    ):
        return []

    projected: list[dict[str, object]] = []
    for raw_entry in rows:
        if not isinstance(raw_entry, Mapping):
            continue
        native_point = _headless_source_row_native_detector_point(raw_entry)
        if native_point is None:
            continue
        simulation_native_frame = bool(
            gui_manual_geometry.geometry_manual_entry_is_simulation_native_frame(raw_entry)
            and not gui_manual_geometry.geometry_manual_entry_is_background_detector_frame(
                raw_entry
            )
        )
        projector = (
            simulation_native_coords_to_caked_display_coords
            if simulation_native_frame
            and callable(simulation_native_coords_to_caked_display_coords)
            else native_detector_coords_to_caked_display_coords
        )
        if not callable(projector):
            continue
        try:
            raw_caked = projector(
                float(native_point[0]),
                float(native_point[1]),
            )
        except Exception:
            continue
        if not isinstance(raw_caked, (tuple, list, np.ndarray)) or len(raw_caked) < 2:
            continue
        try:
            caked_point = (float(raw_caked[0]), float(raw_caked[1]))
        except Exception:
            continue
        if not (np.isfinite(caked_point[0]) and np.isfinite(caked_point[1])):
            continue

        entry = dict(raw_entry)
        entry["background_index"] = int(background_index)
        entry["native_col"] = float(native_point[0])
        entry["native_row"] = float(native_point[1])
        entry["sim_native_x"] = float(native_point[0])
        entry["sim_native_y"] = float(native_point[1])
        entry["sim_detector_anchor_x"] = float(native_point[0])
        entry["sim_detector_anchor_y"] = float(native_point[1])
        entry.setdefault(
            "sim_detector_frame_provenance",
            "simulation_native" if simulation_native_frame else "native_detector",
        )
        entry["caked_x"] = float(caked_point[0])
        entry["caked_y"] = float(caked_point[1])
        entry["raw_caked_x"] = float(caked_point[0])
        entry["raw_caked_y"] = float(caked_point[1])
        entry["two_theta_deg"] = float(caked_point[0])
        entry["phi_deg"] = float(caked_point[1])
        entry["simulated_two_theta_deg"] = float(caked_point[0])
        entry["simulated_phi_deg"] = float(caked_point[1])
        entry["sim_nominal_caked_deg"] = (float(caked_point[0]), float(caked_point[1]))
        entry["sim_refined_caked_deg"] = (float(caked_point[0]), float(caked_point[1]))
        entry["sim_refined_caked_authority"] = "dynamic_trial_projection_from_prediction_native"
        entry["display_col"] = float(caked_point[0])
        entry["display_row"] = float(caked_point[1])
        entry["display_frame"] = "caked_display"
        entry["current_view_frame"] = "caked_display"
        entry["sim_col_global"] = float(caked_point[0])
        entry["sim_row_global"] = float(caked_point[1])
        entry["fit_space_projector_kind"] = "exact_caked_bundle"
        entry["point_only_detector_projection_used"] = True
        entry["caked_projection_input_frame"] = (
            "simulation_native" if simulation_native_frame else "native_detector"
        )
        entry["caked_projection_frame_route"] = (
            "simulation_native_to_bundle_to_caked"
            if simulation_native_frame
            else "background_native_to_bundle_to_caked"
        )
        with suppress(Exception):
            entry["sim_col_local"] = float(
                gui_manual_geometry.caked_axis_to_image_index(
                    float(caked_point[0]),
                    radial_axis,
                )
            )
        with suppress(Exception):
            entry["sim_row_local"] = float(
                gui_manual_geometry.caked_axis_to_image_index(
                    float(caked_point[1]),
                    azimuth_axis,
                )
            )
        projected.append(entry)
    return projected


def _apply_headless_geometry_fit_max_nfev_override(
    runtime_cfg: dict[str, object],
    max_nfev: int | None,
) -> int | None:
    if max_nfev is None:
        return None
    try:
        budget = int(max_nfev)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_nfev must be a positive integer.") from exc
    if budget < 1:
        raise ValueError("max_nfev must be a positive integer.")
    section = runtime_cfg.get("optimizer")
    optimizer = section if isinstance(section, dict) else {}
    if not isinstance(section, dict):
        optimizer = dict(section) if isinstance(section, Mapping) else {}
        runtime_cfg["optimizer"] = optimizer
    optimizer["max_nfev"] = int(budget)
    return int(budget)


def _headless_geometry_entry_has_fixed_manual_caked_qr(entry: Mapping[str, object]) -> bool:
    return gui_geometry_fit.geometry_fit_entry_has_fixed_manual_caked_qr(entry)


def _headless_geometry_fit_attach_selected_pair_provenance(
    row: Mapping[str, object],
    *,
    native_to_display: Callable[[float, float], tuple[float, float]] | None = None,
    display_to_native: Callable[[float, float], tuple[float, float]] | None = None,
    native_to_caked: Callable[[float, float], tuple[float, float]] | None = None,
    measured_reprojection_projector_kind: object = "",
) -> dict[str, object]:
    from ra_sim.gui.geometry_fit_coordinates import (
        selected_pair_measurement_authority_audit,
        selected_pair_coordinate_frame_audit,
        selected_pair_measured_detector_reprojection,
    )

    output = dict(row)
    authority_audit = selected_pair_measurement_authority_audit(output)
    output["selected_pair_measurement_authority_audit"] = authority_audit
    output.update(authority_audit)
    audit = selected_pair_coordinate_frame_audit(
        output,
        native_to_display=native_to_display,
        display_to_native=display_to_native,
    )
    output["selected_pair_coordinate_frame_audit"] = audit
    output["chosen_measured_point_field"] = audit.get("chosen_measured_point_field")
    output["chosen_measured_frame"] = audit.get("chosen_measured_frame")
    output["chosen_measured_point_frame"] = audit.get("chosen_measured_point_frame")
    output["chosen_measured_point"] = audit.get("chosen_measured_point")
    output["chosen_prediction_point_field"] = audit.get("chosen_prediction_point_field")
    output["chosen_prediction_point_frame"] = audit.get("chosen_prediction_point_frame")
    output["chosen_prediction_point"] = audit.get("chosen_prediction_point")
    output["chosen_sim_identity_fields"] = audit.get("chosen_sim_identity_fields")
    output["measured_point_source"] = (
        "selected_manual_pair" if audit.get("chosen_measured_point") is not None else "unknown"
    )
    output["measured_point_fields"] = audit.get("chosen_measured_point_field")
    measured_reprojection = selected_pair_measured_detector_reprojection(
        output,
        display_to_native=display_to_native,
        native_to_caked=native_to_caked,
        projector_kind=measured_reprojection_projector_kind,
    )
    output["selected_pair_measured_reprojection"] = measured_reprojection
    output.update(measured_reprojection)
    return output


def _headless_geometry_fit_theta_contract_payload(
    prepared_run: object,
    *,
    theta_offset_override: object | None = None,
) -> dict[str, object]:
    """Return the fixed per-background theta_i contract used by headless fits."""

    selected_indices = [
        int(idx) for idx in (getattr(prepared_run, "selected_background_indices", ()) or ())
    ]
    raw_background_theta_values = list(getattr(prepared_run, "background_theta_values", ()) or ())
    fit_params = getattr(prepared_run, "fit_params", {}) or {}
    theta_offset = _coerce_float(
        (
            theta_offset_override
            if theta_offset_override is not None
            else fit_params.get("theta_offset", 0.0)
            if isinstance(fit_params, Mapping)
            else 0.0
        ),
        0.0,
    )
    joint_background_mode = bool(getattr(prepared_run, "joint_background_mode", False))
    theta_base_by_background: dict[str, float] = {}
    theta_effective_by_background: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    complete = bool(selected_indices)
    for idx in selected_indices:
        base_value: float | None = None
        if 0 <= int(idx) < len(raw_background_theta_values):
            try:
                base_value = float(raw_background_theta_values[int(idx)])
            except Exception:
                base_value = None
        if base_value is None or not math.isfinite(float(base_value)):
            complete = False
            continue
        applied_offset = float(theta_offset) if joint_background_mode else 0.0
        effective_value = float(base_value) + applied_offset
        theta_base_by_background[str(int(idx))] = float(base_value)
        theta_effective_by_background[str(int(idx))] = float(effective_value)
        rows.append(
            {
                "background_index": int(idx),
                "background_number": int(idx) + 1,
                "theta_initial_base_deg": float(base_value),
                "shared_theta_offset_deg": float(applied_offset),
                "theta_initial_effective_deg": float(effective_value),
            }
        )
    return {
        "theta_contract": "fixed_background_theta_plus_shared_offset",
        "theta_contract_complete": bool(complete),
        "theta_offset_applies_equally_to_selected_backgrounds": bool(joint_background_mode),
        "selected_background_indices_zero_based": selected_indices,
        "selected_background_numbers_one_based": [int(idx) + 1 for idx in selected_indices],
        "background_theta_base_values_deg": [float(value) for value in raw_background_theta_values],
        "background_theta_base_deg_by_index": theta_base_by_background,
        "shared_theta_offset_deg": float(theta_offset) if joint_background_mode else 0.0,
        "effective_theta_initial_deg_by_background": theta_effective_by_background,
        "effective_theta_initial_rows": rows,
    }


def run_headless_geometry_fit(
    saved_state: dict[str, object],
    *,
    state_path: str | Path,
    downloads_dir: str | Path | None = None,
    stamp: str | None = None,
    active_var_names: Sequence[object] | str | None = None,
    max_nfev: int | None = None,
    progress_path: str | Path | None = None,
    weighted_event_workers: int | None = None,
    fit_mosaic_shape: bool = False,
) -> HeadlessGeometryFitResult:
    """Run the geometry fit described by ``saved_state`` and return the updated state."""

    if not isinstance(saved_state, dict):
        raise ValueError("Saved GUI state must be a dictionary.")
    resolved_active_var_names = (
        normalize_headless_geometry_fit_active_var_names(active_var_names)
        if active_var_names is not None
        else None
    )
    progress_writer = _HeadlessGeometryFitProgressWriter(
        progress_path,
        active_vars=resolved_active_var_names,
    )
    progress_writer.write(
        "preflight",
        state_path=Path(state_path),
        **_headless_geometry_fit_state_provenance(state_path),
        downloads_dir=downloads_dir,
        max_nfev=max_nfev,
    )
    weighted_event_worker_count = None
    if weighted_event_workers is not None:
        try:
            weighted_event_worker_count = int(weighted_event_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError("Weighted-event workers must be a positive integer.") from exc
        if weighted_event_worker_count < 1:
            raise ValueError("Weighted-event workers must be a positive integer.")

    downloads_path = (
        Path(downloads_dir)
        if downloads_dir is not None
        else Path(state_path).expanduser().resolve().parent
    )
    downloads_path.mkdir(parents=True, exist_ok=True)
    fit_stamp = str(stamp or Path(state_path).stem)

    diffraction = _load_simulation_diffraction()
    state_types = _load_gui_state_types()
    try:
        missing_configured_inputs = _headless_geometry_fit_configured_input_issues(saved_state)
        if missing_configured_inputs:
            raise HeadlessGeometryFitConfiguredInputError(issues=missing_configured_inputs)
        defaults = _build_runtime_defaults(saved_state)
    except HeadlessGeometryFitConfiguredInputError as exc:
        rejection_reason = str(exc)
        progress_writer.write(
            "final_validation",
            accepted=False,
            rejection_reason=rejection_reason,
            solver_success=False,
            fit_quality_pass=False,
            state_write_accepted=False,
            **exc.progress_fields(),
        )
        raise
    var_store = _build_var_store(saved_state)
    geometry_state = saved_state["geometry"]
    pairs_by_background = _restore_manual_pairs(
        defaults.osc_files,
        geometry_state["manual_pairs"],
    )

    background_state = state_types.BackgroundRuntimeState(
        osc_files=list(defaults.osc_files),
        background_images=[None] * len(defaults.osc_files),
        background_images_native=[None] * len(defaults.osc_files),
        background_images_display=[None] * len(defaults.osc_files),
        current_background_index=int(defaults.current_background_index),
        visible=True,
    )
    simulation_runtime_state = state_types.SimulationRuntimeState()

    structure_state, _atom_site_override_state, active_cif_path, nominal_n2 = _load_structure_model(
        defaults,
        saved_state,
        var_store,
        simulation_runtime_state,
    )

    def _load_background_by_index(index: int) -> tuple[np.ndarray, np.ndarray]:
        loaded = gui_background.load_background_image_by_index(
            int(index),
            osc_files=background_state.osc_files,
            background_images=background_state.background_images,
            background_images_native=background_state.background_images_native,
            background_images_display=background_state.background_images_display,
            display_rotate_k=DISPLAY_ROTATE_K,
            read_osc=read_osc,
        )
        background_state.background_images = list(loaded["background_images"])
        background_state.background_images_native = list(loaded["background_images_native"])
        background_state.background_images_display = list(loaded["background_images_display"])
        background_state.current_background_index = int(index)
        background_state.current_background_image = np.asarray(loaded["background_image"])
        background_state.current_background_display = np.asarray(loaded["background_display"])
        return (
            background_state.current_background_image,
            background_state.current_background_display,
        )

    def _restore_background_index(index: int) -> None:
        if int(background_state.current_background_index) == int(index):
            return
        with suppress(Exception):
            _load_background_by_index(index)

    _load_background_by_index(background_state.current_background_index)

    def _pairs_for_index(index: int) -> list[dict[str, object]]:
        return gui_manual_geometry.geometry_manual_pairs_for_index(
            int(index), pairs_by_background=pairs_by_background
        )

    theta_defaults = {"theta_initial": defaults.defaults["theta_initial"]}
    theta_controls: dict[str, object] = {}
    theta_initial_value = RuntimeValueBinding(
        partial(var_store.__getitem__, "theta_initial_var"),
        partial(var_store.__setitem__, "theta_initial_var"),
    )
    background_theta_list_value = RuntimeValueBinding(
        partial(var_store.__getitem__, "background_theta_list_var"),
        partial(var_store.__setitem__, "background_theta_list_var"),
    )
    geometry_theta_offset_value = RuntimeValueBinding(
        partial(var_store.__getitem__, "geometry_theta_offset_var"),
        partial(var_store.__setitem__, "geometry_theta_offset_var"),
    )
    geometry_fit_selection_value = RuntimeValueBinding(
        partial(var_store.__getitem__, "geometry_fit_background_selection_var"),
        partial(var_store.__setitem__, "geometry_fit_background_selection_var"),
    )

    def _current_geometry_fit_background_indices(*, strict: bool = False) -> list[int]:
        return gui_background_theta.current_geometry_fit_background_indices(
            osc_files=defaults.osc_files,
            current_background_index=background_state.current_background_index,
            geometry_fit_background_selection=geometry_fit_selection_value,
            strict=strict,
        )

    def _geometry_fit_uses_shared_theta_offset(
        selected_indices: list[int] | None = None,
    ) -> bool:
        return gui_background_theta.geometry_fit_uses_shared_theta_offset(
            selected_indices,
            osc_files=defaults.osc_files,
            current_background_index=background_state.current_background_index,
            geometry_fit_background_selection=geometry_fit_selection_value,
        )

    _current_geometry_theta_offset = partial(
        gui_background_theta.current_geometry_theta_offset,
        geometry_theta_offset=geometry_theta_offset_value,
    )
    _current_background_theta_values = partial(
        gui_background_theta.current_background_theta_values,
        osc_files=defaults.osc_files,
        theta_initial=theta_initial_value,
        defaults=theta_defaults,
        theta_initial_fallback=defaults.defaults["theta_initial"],
        background_theta_list=background_theta_list_value,
    )

    def _background_theta_for_index(index: int, *, strict_count: bool = False) -> float:
        return gui_background_theta.background_theta_for_index(
            int(index),
            osc_files=defaults.osc_files,
            theta_initial=theta_initial_value,
            defaults=theta_defaults,
            theta_initial_fallback=defaults.defaults["theta_initial"],
            background_theta_list=background_theta_list_value,
            geometry_theta_offset=geometry_theta_offset_value,
            geometry_fit_background_selection=geometry_fit_selection_value,
            current_background_index=background_state.current_background_index,
            strict_count=strict_count,
        )

    def _apply_background_theta_metadata(
        *,
        trigger_update: bool = False,
        sync_live_theta: bool = True,
    ) -> bool:
        return gui_background_theta.apply_background_theta_metadata(
            osc_files=defaults.osc_files,
            current_background_index=background_state.current_background_index,
            theta_initial=theta_initial_value,
            defaults=theta_defaults,
            theta_initial_fallback=defaults.defaults["theta_initial"],
            background_theta_list=background_theta_list_value,
            geometry_theta_offset=geometry_theta_offset_value,
            geometry_fit_background_selection=geometry_fit_selection_value,
            theta_controls=theta_controls,
            set_background_file_status_text=None,
            schedule_update=None,
            progress_label=None,
            trigger_update=trigger_update,
            sync_live_theta=sync_live_theta,
        )

    def _apply_geometry_fit_background_selection(
        *,
        trigger_update: bool = False,
        sync_live_theta: bool = True,
    ) -> bool:
        return gui_background_theta.apply_geometry_fit_background_selection(
            osc_files=defaults.osc_files,
            current_background_index=background_state.current_background_index,
            theta_initial=theta_initial_value,
            defaults=theta_defaults,
            theta_initial_fallback=defaults.defaults["theta_initial"],
            background_theta_list=background_theta_list_value,
            geometry_theta_offset=geometry_theta_offset_value,
            geometry_fit_background_selection=geometry_fit_selection_value,
            theta_controls=theta_controls,
            set_background_file_status_text=None,
            schedule_update=None,
            progress_label_geometry=None,
            trigger_update=trigger_update,
            sync_live_theta=sync_live_theta,
        )

    solve_q_steps = _coerce_int(
        var_store["solve_q_steps_var"],
        defaults.defaults["solve_q_steps"],
        minimum=32,
    )
    solve_q_rel_tol = float(
        np.clip(
            _coerce_float(
                var_store["solve_q_rel_tol_var"],
                defaults.defaults["solve_q_rel_tol"],
            ),
            1.0e-6,
            5.0e-2,
        )
    )
    nominal_lambda = float(defaults.lambda_angstrom)
    calc_runtime = _load_calculation_runtime()
    wavelength_array = np.array([nominal_lambda], dtype=np.float64)
    n2_source_meta = calc_runtime._normalize_n2_source_meta(("cif_path", active_cif_path))
    n2_wavelength_snapshot = calc_runtime._n2_wavelength_snapshot_from_angstrom(wavelength_array)
    mosaic_params = {
        "beam_x_array": np.zeros(1, dtype=np.float64),
        "beam_y_array": np.zeros(1, dtype=np.float64),
        "theta_array": np.zeros(1, dtype=np.float64),
        "phi_array": np.zeros(1, dtype=np.float64),
        "wavelength_array": wavelength_array,
        "wavelength_i_array": wavelength_array,
        "n2_sample_array": np.array([nominal_n2], dtype=np.complex128),
        "_n2_sample_array_source": n2_source_meta,
        "_n2_sample_array_wavelength_snapshot": n2_wavelength_snapshot,
        "sigma_mosaic_deg": _coerce_float(
            var_store["sigma_mosaic_var"],
            defaults.defaults["sigma_mosaic_deg"],
        ),
        "gamma_mosaic_deg": _coerce_float(
            var_store["gamma_mosaic_var"],
            defaults.defaults["gamma_mosaic_deg"],
        ),
        "eta": _coerce_float(var_store["eta_var"], defaults.defaults["eta"]),
        "solve_q_steps": solve_q_steps,
        "solve_q_rel_tol": solve_q_rel_tol,
        "solve_q_mode": _resolve_solve_q_mode(var_store["solve_q_mode_var"]),
    }

    geometry_values = {
        name: RuntimeValueBinding(
            partial(var_store.__getitem__, state_name),
            partial(var_store.__setitem__, state_name),
        )
        for name, state_name in (
            ("fit_rung", "fit_rung_var"),
            ("zb", "zb_var"),
            ("zs", "zs_var"),
            ("theta_initial", "theta_initial_var"),
            ("theta_offset", "geometry_theta_offset_var"),
            ("psi_z", "psi_z_var"),
            ("chi", "chi_var"),
            ("cor_angle", "cor_angle_var"),
            ("sample_width_m", "sample_width_var"),
            ("sample_length_m", "sample_length_var"),
            ("sample_depth_m", "sample_depth_var"),
            ("gamma", "gamma_var"),
            ("Gamma", "Gamma_var"),
            ("corto_detector", "corto_detector_var"),
            ("a", "a_var"),
            ("c", "c_var"),
            ("center_x", "center_x_var"),
            ("center_y", "center_y_var"),
            ("debye_x", "debye_x_var"),
            ("debye_y", "debye_y_var"),
        )
    }
    value_callbacks = gui_geometry_fit.build_runtime_geometry_fit_value_callbacks(
        gui_geometry_fit.GeometryFitRuntimeValueBindings(
            values=geometry_values,
            current_background_index=lambda: background_state.current_background_index,
            geometry_fit_uses_shared_theta_offset=_geometry_fit_uses_shared_theta_offset,
            current_geometry_theta_offset=_current_geometry_theta_offset,
            background_theta_for_index=_background_theta_for_index,
            build_mosaic_params=lambda: dict(mosaic_params),
            lambda_value=lambda: nominal_lambda,
            psi=lambda: float(defaults.psi_deg),
            n2=lambda: nominal_n2,
            pixel_size_value=lambda: float(defaults.pixel_size_m),
        )
    )

    def _process_peaks_parallel_for_headless(*args, **kwargs):
        if weighted_event_worker_count is not None and kwargs.get("numba_thread_count") is None:
            kwargs["numba_thread_count"] = int(weighted_event_worker_count)
        return diffraction.process_peaks_parallel(*args, **kwargs)

    process_peaks_parallel_for_fit = _process_peaks_parallel_for_headless

    def _native_sim_to_display_coords(col: float, row: float, image_shape: object):
        return gui_geometry_overlay.native_sim_to_display_coords(
            col,
            row,
            image_shape,
            sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
        )

    def _display_to_native_sim_coords(col: float, row: float, image_shape: object):
        return gui_geometry_overlay.display_to_native_sim_coords(
            col,
            row,
            image_shape,
            sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
        )

    simulation_callbacks = (
        gui_geometry_q_group_manager.make_runtime_geometry_fit_simulation_callbacks(
            build_geometry_fit_central_mosaic_params=(
                lambda fit_params: (
                    gui_geometry_q_group_manager.build_locked_detector_native_central_mosaic_params(
                        (
                            fit_params.get("mosaic_params")
                            if isinstance(fit_params, Mapping)
                            else None
                        ),
                        fit_params if isinstance(fit_params, Mapping) else {},
                    )
                )
            ),
            process_peaks_parallel=process_peaks_parallel_for_fit,
            default_solve_q_steps=solve_q_steps,
            default_solve_q_rel_tol=solve_q_rel_tol,
            default_solve_q_mode=_resolve_solve_q_mode(var_store["solve_q_mode_var"]),
        )
    )

    def _signature_numeric(value: object) -> object:
        try:
            parsed = float(value)
        except Exception:
            return None
        if not np.isfinite(parsed):
            return None
        return round(float(parsed), 9)

    _signature_summary = gui_geometry_q_group_manager._signature_summary

    frozen_caked_projection_params = dict(value_callbacks.current_params())
    frozen_caked_projection_payload_by_background: dict[int, dict[str, object]] = {}

    def _geometry_fit_caked_projection_for_index(
        index: int,
    ) -> dict[str, object] | None:
        background_idx = int(index)
        cached_payload = frozen_caked_projection_payload_by_background.get(background_idx)
        if isinstance(cached_payload, dict):
            return dict(cached_payload)
        native_background, display_background = _load_background_by_index(background_idx)
        backend_background = gui_background.apply_background_backend_orientation(
            np.asarray(native_background, dtype=np.float64),
        )
        if backend_background is None:
            backend_background = native_background
        raw_backend_image = np.asarray(backend_background, dtype=np.float64)
        if raw_backend_image.ndim != 2:
            return None
        detector_shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
        params_local = dict(frozen_caked_projection_params)
        payload = _build_headless_geometry_fit_caked_projection_payload(
            detector_shape,
            params=params_local,
            pixel_size_m=float(defaults.pixel_size_m),
            bundle_detector_shape=np.asarray(display_background).shape[:2],
            native_detector_coords_to_bundle_detector_coords=(
                _headless_native_detector_coords_to_detector_display_coords_for_background(
                    _load_background_by_index,
                    int(background_idx),
                    display_rotate_k=DISPLAY_ROTATE_K,
                )
            ),
        )
        if not isinstance(payload, Mapping):
            return None
        normalized_payload = gui_geometry_fit.normalize_geometry_fit_caked_view_payload(
            {
                **dict(payload),
                "background_index": int(background_idx),
                "payload_kind": "projection",
                "projection_view_mode": "caked",
            },
            detector_shape=detector_shape,
        )
        if not isinstance(normalized_payload, dict):
            return None
        hydrated_payload = gui_geometry_fit._geometry_fit_hydrate_exact_caked_payload(
            normalized_payload,
            detector_shape=detector_shape,
            require_background=False,
        )
        projection = gui_geometry_fit.geometry_fit_caked_projection_payload(hydrated_payload)
        if not isinstance(projection, Mapping):
            return None
        transform_bundle = projection.get("transform_bundle")
        exact_cake = _load_exact_cake_portable_module()
        if not isinstance(transform_bundle, exact_cake.CakeTransformBundle):
            return None
        stored = dict(projection)
        for metadata_key in (
            "native_detector_shape",
            "bundle_detector_shape",
            "native_center_row_col",
            "bundle_center_row_col",
            "center_input_frame",
        ):
            if metadata_key in payload:
                stored[metadata_key] = copy.deepcopy(payload.get(metadata_key))
        stored.update(
            {
                "background_index": int(background_idx),
                "payload_kind": "projection",
                "projection_view_mode": "caked",
            }
        )
        frozen_caked_projection_payload_by_background[background_idx] = stored
        return dict(stored)

    def _native_detector_coords_to_caked_coords_for_background(
        background_index: int,
    ):
        background_idx = int(background_index)
        payload = _geometry_fit_caked_projection_for_index(background_idx)
        return _headless_native_detector_coords_to_caked_display_coords_for_payload(
            payload,
            gui_manual_geometry,
            native_detector_coords_to_bundle_detector_coords=(
                _headless_native_detector_coords_to_detector_display_coords_for_background(
                    _load_background_by_index,
                    background_idx,
                    display_rotate_k=DISPLAY_ROTATE_K,
                )
            ),
        )

    def _simulation_native_coords_to_caked_coords_for_background(
        background_index: int,
        *,
        projection_payload: Mapping[str, object] | None = None,
        projection_params: Mapping[str, object] | None = None,
    ) -> tuple[Callable[[float, float], tuple[float, float]], dict[str, object]] | None:
        background_idx = int(background_index)
        payload = (
            projection_payload
            if isinstance(projection_payload, Mapping)
            else _geometry_fit_caked_projection_for_index(background_idx)
        )
        if not isinstance(payload, Mapping):
            return None
        params_for_projection = (
            dict(projection_params)
            if isinstance(projection_params, Mapping)
            else dict(value_callbacks.current_params())
        )
        return _headless_geometry_fit_fresh_simulation_hit_projector(
            params=params_for_projection,
            projection_payload=payload,
            expected_native_detector_shape=np.asarray(
                _load_background_by_index(background_idx)[0]
            ).shape[:2],
            simulation_detector_shape=(
                int(defaults.image_size),
                int(defaults.image_size),
            ),
            native_sim_to_display_coords=_native_sim_to_display_coords,
        )

    def _geometry_fit_required_background_indices() -> list[int]:
        selected = _current_geometry_fit_background_indices(strict=True)
        return headless_geometry_fit_required_background_indices(
            selected,
            current_background_index=int(background_state.current_background_index),
            uses_shared_theta_offset=_geometry_fit_uses_shared_theta_offset(selected),
        )

    def _native_to_display_for_background(background_index: int):
        return _headless_native_detector_coords_to_detector_display_coords_for_background(
            _load_background_by_index, int(background_index), display_rotate_k=DISPLAY_ROTATE_K
        )

    def _display_to_native_for_background(background_index: int):
        return _headless_background_display_to_native_detector_coords_for_background(
            _load_background_by_index, int(background_index), display_rotate_k=DISPLAY_ROTATE_K
        )

    def _attach_headless_manual_pair_selected_provenance(
        row: Mapping[str, object],
        *,
        background_index: int,
    ) -> dict[str, object]:
        output = dict(row)
        if not _headless_geometry_entry_has_fixed_manual_caked_qr(output):
            return output
        simulation_projection = _simulation_native_coords_to_caked_coords_for_background(
            int(background_index)
        )
        native_to_caked = simulation_projection[0] if simulation_projection is not None else None
        return _headless_geometry_fit_attach_selected_pair_provenance(
            output,
            native_to_display=_native_to_display_for_background(int(background_index)),
            display_to_native=_display_to_native_for_background(int(background_index)),
            native_to_caked=native_to_caked,
            measured_reprojection_projector_kind=(
                "exact_caked_bundle" if native_to_caked is not None else ""
            ),
        )

    def _headless_manual_pair_rows_for_background(
        background_index: int,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        background_idx = int(background_index)
        for entry in _pairs_for_index(background_idx):
            if not isinstance(entry, Mapping):
                continue
            row = dict(entry)
            row["background_index"] = int(background_idx)
            rows.append(
                _attach_headless_manual_pair_selected_provenance(
                    row,
                    background_index=int(background_idx),
                )
            )
        return rows

    def _headless_geometry_fit_pairs_for_index(index: int) -> list[dict[str, object]]:
        return _headless_manual_pair_rows_for_background(int(index))

    def _ensure_geometry_fit_caked_view() -> None:
        previous_background_idx = int(background_state.current_background_index)
        try:
            for background_idx in _geometry_fit_required_background_indices():
                if not gui_geometry_fit.geometry_manual_pairs_use_caked_fit_space(
                    _pairs_for_index(int(background_idx))
                ):
                    continue
                if _geometry_fit_caked_projection_for_index(int(background_idx)) is None:
                    raise RuntimeError(
                        "exact caked projector unavailable for "
                        f"background {int(background_idx) + 1}"
                    )
        finally:
            _restore_background_index(previous_background_idx)

    def _headless_current_caked_projection_for_callbacks() -> dict[str, object] | None:
        try:
            background_idx = int(background_state.current_background_index)
        except Exception:
            return None
        if not gui_geometry_fit.geometry_manual_pairs_use_caked_fit_space(
            _pairs_for_index(background_idx)
        ):
            return None
        payload = _geometry_fit_caked_projection_for_index(background_idx)
        return payload if isinstance(payload, dict) else None

    def _headless_projection_callbacks(
        *,
        caked_view_enabled: Callable[[], bool],
        caked_projection_payload: Callable[[], Mapping[str, object] | None],
        current_background_display: Callable[[], object],
        current_background_native: Callable[[], object],
        current_background_index: Callable[[], int],
        native_detector_coords_to_bundle_detector_coords: Callable[..., object],
        bundle_detector_coords_to_background_display_coords: Callable[..., object],
    ) -> object:
        def _payload_value(key: str) -> object:
            payload = caked_projection_payload()
            return payload.get(key) if isinstance(payload, Mapping) else None

        return gui_manual_geometry.make_runtime_geometry_manual_projection_callbacks(
            caked_view_enabled=caked_view_enabled,
            last_caked_background_image_unscaled=lambda: None,
            last_caked_radial_values=lambda: _payload_value("radial_axis"),
            last_caked_azimuth_values=lambda: _payload_value("azimuth_axis"),
            current_background_display=current_background_display,
            current_background_native=current_background_native,
            caked_transform_bundle=lambda: _payload_value("transform_bundle"),
            current_background_index=current_background_index,
            caked_projection_payload=caked_projection_payload,
            last_caked_raw_azimuth_values=lambda: _payload_value("raw_azimuth_axis"),
            last_caked_raw_to_gui_row_permutation=lambda: _payload_value(
                "raw_to_gui_row_permutation"
            ),
            rotate_point_for_display=gui_geometry_overlay.rotate_point_for_display,
            display_rotate_k=DISPLAY_ROTATE_K,
            image_size=int(defaults.image_size),
            display_to_native_sim_coords=_display_to_native_sim_coords,
            native_detector_coords_to_bundle_detector_coords=(
                native_detector_coords_to_bundle_detector_coords
            ),
            bundle_detector_coords_to_background_display_coords=(
                bundle_detector_coords_to_background_display_coords
            ),
        )

    projection_callbacks = _headless_projection_callbacks(
        caked_view_enabled=lambda: isinstance(
            _headless_current_caked_projection_for_callbacks(), Mapping
        ),
        caked_projection_payload=_headless_current_caked_projection_for_callbacks,
        current_background_display=lambda: _load_background_by_index(
            background_state.current_background_index
        )[1],
        current_background_native=lambda: _load_background_by_index(
            background_state.current_background_index
        )[0],
        current_background_index=lambda: int(background_state.current_background_index),
        native_detector_coords_to_bundle_detector_coords=(
            lambda col, row: (
                _native_to_display_for_background(int(background_state.current_background_index))
                or (lambda _col, _row: (None, None))
            )(float(col), float(row))
        ),
        bundle_detector_coords_to_background_display_coords=lambda col, row: (
            float(col),
            float(row),
        ),
    )

    def _project_peaks_for_background_view(
        background_index: int,
        rows: Sequence[dict[str, object]] | None,
        *,
        mode_override: str | None = None,
        strict_caked_projection: bool = True,
    ) -> list[dict[str, object]]:
        normalized_rows = [dict(entry) for entry in (rows or ()) if isinstance(entry, Mapping)]
        if not normalized_rows:
            return []
        background_idx = int(background_index)
        if mode_override is None:
            use_caked_projection = gui_geometry_fit.geometry_manual_pairs_use_caked_fit_space(
                _pairs_for_index(background_idx)
            )
        else:
            use_caked_projection = str(mode_override).strip().lower() == "caked"
        if not use_caked_projection:
            return [
                dict(entry)
                for entry in (
                    projection_callbacks.project_peaks_to_current_view(normalized_rows) or ()
                )
                if isinstance(entry, Mapping)
            ]
        previous_background_idx = int(background_state.current_background_index)
        try:
            payload = _geometry_fit_caked_projection_for_index(background_idx)
            if not isinstance(payload, Mapping):
                if not bool(strict_caked_projection):
                    return []
                raise RuntimeError(
                    f"exact caked projector unavailable for background {int(background_idx) + 1}"
                )
            simulation_projection = _simulation_native_coords_to_caked_coords_for_background(
                background_idx,
                projection_payload=payload,
            )
            simulation_native_projector = (
                simulation_projection[0] if simulation_projection is not None else None
            )
            simulation_native_rows_present = any(
                gui_manual_geometry.geometry_manual_entry_is_simulation_native_frame(entry)
                and not gui_manual_geometry.geometry_manual_entry_is_background_detector_frame(
                    entry
                )
                for entry in normalized_rows
            )
            if simulation_native_rows_present and not callable(simulation_native_projector):
                if not bool(strict_caked_projection):
                    return []
                raise RuntimeError(
                    "frozen simulation-native caked projector unavailable for "
                    f"background {int(background_idx) + 1}"
                )
            direct_projected_rows = _headless_project_source_rows_to_exact_caked_view(
                normalized_rows,
                background_index=int(background_idx),
                native_detector_coords_to_caked_display_coords=(
                    _native_detector_coords_to_caked_coords_for_background(background_idx)
                ),
                simulation_native_coords_to_caked_display_coords=(simulation_native_projector),
                radial_axis=payload.get("radial_axis"),
                azimuth_axis=payload.get("azimuth_axis"),
            )
            if direct_projected_rows:
                return direct_projected_rows
            native_background, display_background = _load_background_by_index(background_idx)
            background_projection_callbacks = _headless_projection_callbacks(
                caked_view_enabled=lambda: True,
                caked_projection_payload=lambda: payload,
                current_background_display=lambda: display_background,
                current_background_native=lambda: native_background,
                current_background_index=lambda: int(background_idx),
                native_detector_coords_to_bundle_detector_coords=(
                    _native_to_display_for_background(int(background_idx))
                    or (lambda _col, _row: (None, None))
                ),
                bundle_detector_coords_to_background_display_coords=(
                    lambda col, row: (float(col), float(row))
                ),
            )
            projected_rows = background_projection_callbacks.project_peaks_to_current_view(
                normalized_rows
            )
            return [
                {**dict(entry), "background_index": int(background_idx)}
                for entry in (projected_rows or ())
                if isinstance(entry, Mapping)
            ]
        finally:
            _restore_background_index(previous_background_idx)

    def _project_peaks_to_current_view_for_dataset(
        rows: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        return _project_peaks_for_background_view(
            int(background_state.current_background_index),
            rows,
        )

    def _source_rows_signature_for_background(
        background_index: int,
        param_set: dict[str, object] | None = None,
    ) -> tuple[object, ...]:
        params_local = dict(value_callbacks.current_params())
        if isinstance(param_set, Mapping):
            params_local.update(dict(param_set))
        center_value = params_local.get("center", [np.nan, np.nan])
        if isinstance(center_value, (list, tuple, np.ndarray)) and len(center_value) >= 2:
            center_signature = (
                _signature_numeric(center_value[0]),
                _signature_numeric(center_value[1]),
            )
        else:
            center_signature = (None, None)
        return (
            int(background_index),
            int(defaults.image_size),
            tuple(np.asarray(structure_state.miller).shape),
            tuple(np.asarray(structure_state.intensities).shape),
            _signature_numeric(params_local.get("a")),
            _signature_numeric(params_local.get("c")),
            _signature_numeric(params_local.get("lambda")),
            _signature_numeric(params_local.get("theta_initial")),
            _signature_numeric(params_local.get("theta_offset")),
            _signature_numeric(params_local.get("corto_detector")),
            _signature_numeric(params_local.get("gamma")),
            _signature_numeric(params_local.get("Gamma")),
            _signature_numeric(params_local.get("chi")),
            _signature_numeric(params_local.get("psi_z")),
            _signature_numeric(params_local.get("cor_angle")),
            _signature_numeric(params_local.get("zb")),
            _signature_numeric(params_local.get("zs")),
            center_signature,
        )

    def _simulate_hit_tables_for_fit(
        miller_array: np.ndarray,
        intensity_array: np.ndarray,
        image_size_value: int,
        params_local: Mapping[str, object],
        *,
        required_branch_group_keys: (
            Sequence[tuple[tuple[int, int, int], int | None, object | None]] | None
        ) = None,
        required_manual_fit_targets: Sequence[Mapping[str, object]] | None = None,
        preflight_mode: str = "full",
        hit_tables_only: bool = False,
    ) -> tuple[list[object], dict[str, object]]:
        return simulation_callbacks.simulate_hit_tables(
            np.asarray(miller_array, dtype=np.float64),
            np.asarray(intensity_array, dtype=np.float64),
            int(image_size_value),
            dict(params_local),
            required_branch_group_keys=required_branch_group_keys,
            required_manual_fit_targets=required_manual_fit_targets,
            hit_tables_only=bool(hit_tables_only),
        )

    def _background_label_for_index(background_index: int) -> str:
        try:
            osc_path = background_state.osc_files[int(background_index)]
        except Exception:
            osc_path = None
        if osc_path is not None:
            try:
                label = Path(str(osc_path)).name
            except Exception:
                label = ""
            if str(label).strip():
                return str(label)
        return f"background {int(background_index) + 1}"

    def _commit_source_row_rebuild_result(
        rebuild_result: gui_geometry_fit.GeometryFitSourceRowRebuildResult,
    ) -> list[dict[str, object]]:
        if not isinstance(rebuild_result, gui_geometry_fit.GeometryFitSourceRowRebuildResult):
            return []

        background_idx = int(rebuild_result.background_index)
        _preflight_stage_callback(
            "source_rows_rebuild_commit_start",
            {
                "background_index": int(background_idx),
                "message": (
                    "preflight: committing rebuilt source rows for "
                    f"background {int(background_idx) + 1}"
                ),
                "rebuild_source": str(rebuild_result.rebuild_source or ""),
                "stored_row_count": int(len(rebuild_result.stored_rows or ())),
                "projected_row_count": int(len(rebuild_result.projected_rows or ())),
                "has_hit_tables": bool(rebuild_result.hit_tables is not None),
                "has_intersection_cache": bool(rebuild_result.intersection_cache is not None),
            },
        )
        stored_rows = [
            dict(entry)
            for entry in (rebuild_result.stored_rows or ())
            if isinstance(entry, Mapping)
        ]
        projected_rows = [
            dict(entry)
            for entry in (rebuild_result.projected_rows or ())
            if isinstance(entry, Mapping)
        ]
        diagnostics = (
            dict(rebuild_result.diagnostics)
            if isinstance(rebuild_result.diagnostics, Mapping)
            else {}
        )
        _preflight_stage_callback(
            "source_rows_rebuild_commit_rows_ready",
            {
                "background_index": int(background_idx),
                "stored_row_count": int(len(stored_rows)),
                "projected_row_count": int(len(projected_rows)),
                "diagnostic_key_count": int(len(diagnostics)),
            },
        )
        if stored_rows:
            if rebuild_result.hit_tables is not None:
                _preflight_stage_callback(
                    "source_rows_rebuild_commit_hit_tables_start",
                    {
                        "background_index": int(background_idx),
                        "hit_table_count": int(len(rebuild_result.hit_tables or ())),
                    },
                )
                try:
                    max_positions_local = diffraction.hit_tables_to_max_positions(
                        rebuild_result.hit_tables
                    )
                except Exception:
                    max_positions_local = np.empty((0, 6), dtype=np.float64)
                simulation_runtime_state.stored_max_positions_local = list(max_positions_local)
                _preflight_stage_callback(
                    "source_rows_rebuild_commit_hit_tables_ready",
                    {
                        "background_index": int(background_idx),
                        "max_position_count": int(len(max_positions_local)),
                    },
                )
            if rebuild_result.peak_table_lattice is not None:
                simulation_runtime_state.stored_peak_table_lattice = list(
                    rebuild_result.peak_table_lattice
                )
            simulation_runtime_state.stored_sim_image = np.zeros(
                (int(defaults.image_size), int(defaults.image_size)),
                dtype=np.float64,
            )
            if rebuild_result.intersection_cache is not None:
                _preflight_stage_callback(
                    "source_rows_rebuild_commit_intersection_cache_start",
                    {
                        "background_index": int(background_idx),
                        "intersection_cache_count": int(
                            len(rebuild_result.intersection_cache or ())
                        ),
                    },
                )
                simulation_runtime_state.stored_intersection_cache = (
                    _copy_intersection_cache_tables(rebuild_result.intersection_cache)
                )
                simulation_runtime_state.stored_hit_table_signature = (
                    rebuild_result.requested_signature
                )
                _preflight_stage_callback(
                    "source_rows_rebuild_commit_intersection_cache_ready",
                    {
                        "background_index": int(background_idx),
                        "intersection_cache_count": int(
                            len(rebuild_result.intersection_cache or ())
                        ),
                    },
                )
            simulation_runtime_state.last_simulation_signature = rebuild_result.requested_signature
            _preflight_stage_callback(
                "source_rows_rebuild_commit_peak_cache_start",
                {
                    "background_index": int(background_idx),
                    "stored_row_count": int(len(stored_rows)),
                },
            )
            _set_runtime_peak_cache_from_source_rows(
                simulation_runtime_state,
                stored_rows,
            )
            _preflight_stage_callback(
                "source_rows_rebuild_commit_peak_cache_ready",
                {
                    "background_index": int(background_idx),
                    "peak_record_count": int(len(simulation_runtime_state.peak_records or ())),
                },
            )
        if diagnostics:
            _preflight_stage_callback(
                "source_rows_rebuild_commit_diagnostics_start",
                {
                    "background_index": int(background_idx),
                    "diagnostic_key_count": int(len(diagnostics)),
                },
            )
            _preflight_stage_callback(
                "source_rows_rebuild_commit_diagnostics_ready",
                {
                    "background_index": int(background_idx),
                    "diagnostic_key_count": int(len(diagnostics)),
                },
            )
        result_rows = projected_rows if projected_rows else stored_rows
        _preflight_stage_callback(
            "source_rows_rebuild_commit_ready",
            {
                "background_index": int(background_idx),
                "result_row_count": int(len(result_rows)),
            },
        )
        return [dict(entry) for entry in result_rows if isinstance(entry, Mapping)]

    def _geometry_manual_rebuild_source_rows_for_background(
        background_index: int,
        param_set: dict[str, object] | None = None,
        *,
        consumer: str | None = None,
        required_pairs: Sequence[Mapping[str, object]] | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        background_idx = int(background_index)
        consumer_name = str(consumer or "unspecified")
        params_local = dict(value_callbacks.current_params())
        if isinstance(param_set, Mapping):
            params_local.update(dict(param_set))
        requested_signature = _source_rows_signature_for_background(
            background_idx,
            params_local,
        )
        requested_signature_summary = _signature_summary(requested_signature)

        def _build_source_rows_for_rebuild(
            source_tables: Sequence[object] | None,
            *,
            required_branch_group_keys: (
                Sequence[tuple[tuple[int, int, int], int | None, object | None]] | None
            ) = None,
            required_manual_fit_targets: Sequence[Mapping[str, object]] | None = None,
            preflight_mode: str = "full",
            consumer: str | None = None,
        ) -> tuple[list[dict[str, object]], list[tuple[float, float, str]], list[object]]:
            schema = _load_intersection_cache_schema()
            table_list = list(source_tables or ())
            if not table_list:
                return [], [], []
            trial_source_rows_consumer = (
                str(consumer or consumer_name) == "geometry_fit_trial_source_rows"
            )
            if schema.is_intersection_cache_table(table_list[0]):
                hit_tables_local = diffraction.intersection_cache_to_hit_tables(table_list)
            elif trial_source_rows_consumer:
                hit_tables_local = source_tables
            else:
                hit_tables_local = gui_geometry_q_group_manager.copy_geometry_fit_hit_tables(
                    source_tables
                )
            if (
                str(preflight_mode or "full") == "manual_geometry_targeted"
                and not trial_source_rows_consumer
            ):
                hit_tables_local = (
                    gui_geometry_q_group_manager.filter_geometry_fit_hit_tables_for_required_branch_groups(
                        hit_tables_local,
                        required_branch_group_keys=required_branch_group_keys,
                    )
                )
            native_to_display = _native_sim_to_display_coords
            fresh_projection_payload = _geometry_fit_caked_projection_for_index(background_idx)
            fresh_projection = _simulation_native_coords_to_caked_coords_for_background(
                background_idx,
                projection_payload=(
                    fresh_projection_payload
                    if isinstance(fresh_projection_payload, Mapping)
                    else None
                ),
                projection_params=frozen_caked_projection_params,
            )
            fresh_simulation_hit_projector = (
                fresh_projection[0] if fresh_projection is not None else None
            )
            fresh_projection_provenance = (
                fresh_projection[1] if fresh_projection is not None else None
            )
            frozen_roi_native_projector = None
            if callable(fresh_simulation_hit_projector):
                native_detector_shape = np.asarray(
                    _load_background_by_index(background_idx)[0]
                ).shape[:2]
                simulation_detector_shape = (
                    int(defaults.image_size),
                    int(defaults.image_size),
                )
                bundle_detector_shape = (
                    fresh_projection_provenance.get("bundle_detector_shape")
                    if isinstance(fresh_projection_provenance, Mapping)
                    else None
                )
                native_to_bundle = (
                    _headless_native_detector_coords_to_detector_display_coords_for_background(
                        _load_background_by_index,
                        background_idx,
                        display_rotate_k=DISPLAY_ROTATE_K,
                    )
                )

                def _bundle_detector_coords_to_hit_table_detector_coords(
                    col: float,
                    row: float,
                ) -> tuple[float, float]:
                    point = _display_to_native_sim_coords(
                        float(col),
                        float(row),
                        simulation_detector_shape,
                    )
                    return float(point[0]), float(point[1])

                def _hit_table_detector_coords_to_bundle_detector_coords(
                    col: float,
                    row: float,
                ) -> tuple[float, float]:
                    point = _native_sim_to_display_coords(
                        float(col),
                        float(row),
                        simulation_detector_shape,
                    )
                    return float(point[0]), float(point[1])

                frozen_roi_native_projector = (
                    gui_geometry_q_group_manager.build_frozen_caked_centroid_native_roi_projector(
                        centroid_hit_projector=fresh_simulation_hit_projector,
                        native_detector_shape=native_detector_shape,
                        bundle_detector_shape=bundle_detector_shape,
                        hit_table_detector_shape=simulation_detector_shape,
                        native_detector_coords_to_bundle_detector_coords=(native_to_bundle),
                        bundle_detector_coords_to_hit_table_detector_coords=(
                            _bundle_detector_coords_to_hit_table_detector_coords
                        ),
                        hit_table_detector_coords_to_bundle_detector_coords=(
                            _hit_table_detector_coords_to_bundle_detector_coords
                        ),
                    )
                )
            if (
                required_manual_fit_targets
                and not False
                and not callable(frozen_roi_native_projector)
            ):
                raise RuntimeError("required frozen centroid native ROI projector is unavailable")

            def _project_rebuild_source_rows_to_caked(
                rows: Sequence[Mapping[str, object]] | None,
            ) -> list[dict[str, object]]:
                if callable(fresh_simulation_hit_projector):
                    return _headless_project_source_rows_to_exact_caked_view(
                        [dict(row) for row in (rows or ()) if isinstance(row, Mapping)],
                        background_index=int(background_idx),
                        native_detector_coords_to_caked_display_coords=(
                            _native_detector_coords_to_caked_coords_for_background(background_idx)
                        ),
                        simulation_native_coords_to_caked_display_coords=(
                            fresh_simulation_hit_projector
                        ),
                        radial_axis=(
                            fresh_projection_payload.get("radial_axis")
                            if isinstance(fresh_projection_payload, Mapping)
                            else None
                        ),
                        azimuth_axis=(
                            fresh_projection_payload.get("azimuth_axis")
                            if isinstance(fresh_projection_payload, Mapping)
                            else None
                        ),
                    )
                return _project_peaks_for_background_view(
                    background_idx,
                    [dict(row) for row in (rows or ()) if isinstance(row, Mapping)],
                )

            return _build_source_rows_from_hit_tables(
                hit_tables_local,
                image_size_value=int(defaults.image_size),
                params_local=params_local,
                native_sim_to_display_coords=native_to_display,
                allow_nominal_hkl_indices=True,
                project_source_rows_to_caked=_project_rebuild_source_rows_to_caked,
                native_detector_coords_to_caked_display_coords=(
                    _native_detector_coords_to_caked_coords_for_background(background_idx)
                ),
                centroid_hit_table_coords_to_caked_display_coords=(fresh_simulation_hit_projector),
                centroid_projection_provenance=fresh_projection_provenance,
                centroid_frozen_roi_native_detector_coords_to_caked_display_coords=(
                    frozen_roi_native_projector
                ),
                required_manual_fit_targets=required_manual_fit_targets,
            )

        rebuild_result = gui_geometry_fit.rebuild_geometry_fit_source_rows(
            background_index=int(background_idx),
            background_label=_background_label_for_index(background_idx),
            params_local=params_local,
            consumer=consumer_name,
            requested_signature=requested_signature,
            requested_signature_summary=requested_signature_summary,
            build_source_rows_from_hit_tables=_build_source_rows_for_rebuild,
            simulate_hit_tables=(
                lambda normalized_params, **kwargs: _simulate_hit_tables_for_fit(
                    structure_state.miller,
                    structure_state.intensities,
                    int(defaults.image_size),
                    normalized_params,
                    **kwargs,
                )
            ),
            required_pairs=required_pairs,
            stage_callback=_geometry_fit_rebuild_stage_callback_for_consumer(
                consumer_name,
                _preflight_stage_callback,
            ),
            retain_rebuild_source_tables=False,
        )
        strict_fresh_rows = gui_geometry_fit.strict_caked_geometry_fresh_native_rows(rebuild_result)
        if strict_fresh_rows:
            rows = strict_fresh_rows
        else:
            rows = _commit_source_row_rebuild_result(rebuild_result)
        return rows, dict(rebuild_result.diagnostics)

    manual_dataset_bindings = gui_geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=tuple(defaults.osc_files),
        current_background_index=int(background_state.current_background_index),
        image_size=int(defaults.image_size),
        display_rotate_k=int(DISPLAY_ROTATE_K),
        geometry_manual_pairs_for_index=_headless_geometry_fit_pairs_for_index,
        load_background_by_index=_load_background_by_index,
        apply_background_backend_orientation=gui_background.apply_background_backend_orientation,
        backend_detector_coords_to_native_detector_coords=lambda col, row, native_shape=None: (
            gui_background.background_backend_point_to_native_coords(
                float(col),
                float(row),
                native_shape=(
                    tuple(int(v) for v in tuple(native_shape)[:2])
                    if native_shape is not None
                    else np.asarray(
                        _load_background_by_index(int(background_state.current_background_index))[0]
                    ).shape[:2]
                ),
            )
        ),
        native_detector_coords_to_detector_display_coords_for_background=(
            _native_to_display_for_background
        ),
        detector_display_to_native_detector_coords_for_background=(
            _display_to_native_for_background
        ),
        geometry_manual_simulated_lookup=projection_callbacks.simulated_lookup,
        geometry_manual_rebuild_source_rows_for_background=(
            _geometry_manual_rebuild_source_rows_for_background
        ),
        pick_uses_caked_space=lambda: gui_geometry_fit.geometry_manual_pairs_use_caked_fit_space(
            _pairs_for_index(int(background_state.current_background_index))
        ),
        geometry_manual_caked_view_for_index=None,
        geometry_manual_caked_projection_for_index=_geometry_fit_caked_projection_for_index,
        geometry_manual_entry_display_coords=projection_callbacks.entry_display_coords,
        geometry_manual_project_peaks_to_current_view=(_project_peaks_to_current_view_for_dataset),
        geometry_manual_project_peaks_for_background_view=(_project_peaks_for_background_view),
        unrotate_display_peaks=lambda measured, rotated_shape, *, k=None: (
            gui_geometry_overlay.unrotate_display_peaks(
                measured,
                rotated_shape,
                k=k,
                default_display_rotate_k=DISPLAY_ROTATE_K,
            )
        ),
        display_to_native_sim_coords=_display_to_native_sim_coords,
    )

    if resolved_active_var_names is None:
        resolved_active_var_names = list(
            _HEADLESS_GEOMETRY_FIT_SAVED_MANUAL_CAKED_DEFAULT_ACTIVE_VAR_NAMES
        )
    progress_writer.update_static(active_vars=resolved_active_var_names)

    params = value_callbacks.current_params()
    var_names = _headless_geometry_fit_runtime_active_var_names(
        resolved_active_var_names,
    )
    progress_writer.update_static(active_vars=var_names)

    def _headless_failure_status_fields(reason: object) -> dict[str, object]:
        return {
            "accepted": False,
            "rejection_reason": str(reason),
            "solver_success": False,
            "fit_quality_pass": False,
            "state_write_accepted": False,
        }

    preserve_live_theta = "theta_initial" not in var_names and "theta_offset" not in var_names
    headless_fit_config = (
        copy.deepcopy(defaults.fit_config) if isinstance(defaults.fit_config, dict) else {}
    )

    def _build_headless_runtime_config(_fit_params: Mapping[str, object]) -> dict[str, object]:
        base_runtime_cfg = copy.deepcopy(
            headless_fit_config.get("geometry", {}) if isinstance(headless_fit_config, dict) else {}
        )
        if not isinstance(base_runtime_cfg, dict):
            base_runtime_cfg = {}
        candidate_params = {str(name): _fit_params.get(str(name)) for name in var_names}
        parameter_domains = _headless_runtime_geometry_fit_parameter_domains(
            fit_config=base_runtime_cfg,
            current_params=_fit_params,
            names=var_names,
        )
        runtime_cfg = gui_geometry_fit.build_geometry_fit_runtime_config(
            base_runtime_cfg,
            candidate_params,
            parameter_domains,
            candidate_param_names=var_names,
        )
        return runtime_cfg

    preflight_stage_event_index = 0

    def _preflight_stage_callback(stage: str, payload: Mapping[str, object]) -> None:
        nonlocal preflight_stage_event_index
        preflight_stage_event_index += 1
        event_payload = dict(payload) if isinstance(payload, Mapping) else {"payload": str(payload)}
        compact_event = _headless_geometry_fit_compact_preflight_event(
            stage=str(stage),
            payload=event_payload,
            event_index=int(preflight_stage_event_index),
            elapsed_s=float(max(0.0, time.monotonic() - progress_writer.started_at)),
        )
        recent_events = [
            dict(event)
            for event in (progress_writer.data.get("preflight_stage_events", ()) or ())
            if isinstance(event, Mapping)
        ]
        recent_events.append(compact_event)
        progress_writer.write(
            "preflight",
            preflight_stage=str(stage),
            preflight_stage_message=event_payload.get("message"),
            preflight_stage_event_index=int(preflight_stage_event_index),
            preflight_stage_elapsed_s=compact_event["elapsed_s"],
            preflight_stage_payload=_headless_progress_jsonable(event_payload),
            preflight_stage_events=recent_events[-100:],
            preflight_wait_condition=str(stage),
        )

    preflight_started_at = time.monotonic()
    preparation = gui_geometry_fit.prepare_runtime_geometry_fit_run(
        params=params,
        var_names=var_names,
        preserve_live_theta=preserve_live_theta,
        bindings=gui_geometry_fit.GeometryFitRuntimePreparationBindings(
            fit_config=headless_fit_config,
            theta_initial=var_store["theta_initial_var"],
            apply_geometry_fit_background_selection=_apply_geometry_fit_background_selection,
            current_geometry_fit_background_indices=_current_geometry_fit_background_indices,
            geometry_fit_uses_shared_theta_offset=_geometry_fit_uses_shared_theta_offset,
            apply_background_theta_metadata=_apply_background_theta_metadata,
            current_background_theta_values=_current_background_theta_values,
            current_geometry_theta_offset=_current_geometry_theta_offset,
            ensure_geometry_fit_caked_view=_ensure_geometry_fit_caked_view,
            manual_dataset_bindings=manual_dataset_bindings,
            build_runtime_config=_build_headless_runtime_config,
            include_all_selected_backgrounds=None,
        ),
        stage_callback=_preflight_stage_callback,
    )
    if preparation.prepared_run is None:
        failure_status = _headless_failure_status_fields(
            preparation.error_text or "Geometry fit preparation failed."
        )
        raise RuntimeError(str(failure_status["rejection_reason"]))
    prepared_run = preparation.prepared_run
    progress_writer.write(
        str(progress_writer.data.get("phase", "preflight")),
        **_headless_geometry_fit_theta_contract_payload(prepared_run),
    )
    preflight_elapsed_s = float(max(0.0, time.monotonic() - preflight_started_at))
    headless_geometry_cfg = (
        copy.deepcopy(prepared_run.geometry_runtime_cfg)
        if isinstance(prepared_run.geometry_runtime_cfg, Mapping)
        else {}
    )
    applied_max_nfev_override = _apply_headless_geometry_fit_max_nfev_override(
        headless_geometry_cfg,
        max_nfev,
    )
    progress_writer.update_static(
        runtime_cfg=headless_geometry_cfg,
        max_nfev_override=applied_max_nfev_override,
    )
    prepared_run = replace(
        prepared_run,
        start_log_sections=gui_geometry_fit.build_geometry_fit_start_log_sections(
            params=prepared_run.fit_params,
            var_names=var_names,
            dataset_infos=prepared_run.dataset_infos,
        ),
        geometry_runtime_cfg=headless_geometry_cfg,
        stage_timing_s={
            **(
                dict(prepared_run.stage_timing_s)
                if isinstance(prepared_run.stage_timing_s, Mapping)
                else {}
            ),
            "preflight_rebind": float(preflight_elapsed_s),
            "dataset construction": float(preflight_elapsed_s),
        },
    )
    progress_writer.write(
        "runtime_config_ready",
        request_build_s=preflight_elapsed_s,
        active_vars=var_names,
    )

    initial_fit_params = dict(value_callbacks.current_params())
    progress_writer.write("solve_start")
    execution = gui_geometry_fit.execute_runtime_geometry_fit(
        prepared_run=prepared_run,
        var_names=var_names,
        preserve_live_theta=preserve_live_theta,
        setup=gui_geometry_fit.build_runtime_geometry_fit_execution_setup_from_bindings(
            prepared_run=prepared_run,
            mosaic_params=mosaic_params,
            stamp=fit_stamp,
            bindings=gui_geometry_fit.GeometryFitRuntimeActionExecutionBindings(
                downloads_dir=downloads_path,
                simulation_runtime_state=simulation_runtime_state,
                background_runtime_state=background_state,
                current_ui_params=value_callbacks.current_ui_params,
                values=value_callbacks.values,
                background_theta_for_index=_background_theta_for_index,
                refresh_status=lambda: None,
                update_manual_pick_button_label=lambda: None,
                capture_undo_state=lambda: {},
                push_undo_state=lambda _state: None,
                replace_dataset_cache=lambda _payload: None,
                request_preview_skip_once=lambda: None,
                schedule_update=lambda: None,
                draw_overlay_records=lambda _records, _marker_limit: None,
                draw_initial_pairs_overlay=lambda _pairs, _marker_limit: None,
                set_last_overlay_state=lambda _state: None,
                set_progress_text=progress_writer.status,
                cmd_line=progress_writer.status,
                background_display_rotate_k=DISPLAY_ROTATE_K,
                build_overlay_records=gui_geometry_overlay.build_geometry_fit_overlay_records,
                compute_frame_diagnostics=partial(
                    gui_geometry_overlay.compute_geometry_overlay_frame_diagnostics,
                    show_caked_2d=False,
                    native_detector_coords_to_caked_display_coords=None,
                ),
                live_update_callback=progress_writer.live_update,
            ),
        ),
    )

    def _fail_headless_geometry_fit_execution(reason: object) -> None:
        failure_status = _headless_failure_status_fields(reason)
        progress_writer.write("final_validation", **failure_status)
        raise RuntimeError(str(failure_status["rejection_reason"]))

    if execution.error_text:
        _fail_headless_geometry_fit_execution(execution.error_text)
    if execution.apply_result is None:
        _fail_headless_geometry_fit_execution("Geometry fit finished without an apply result.")
    solver_result = getattr(execution, "solver_result", None)
    final_summary = getattr(solver_result, "point_match_summary", None)
    final_progress: dict[str, object] = {
        "accepted": bool(execution.apply_result.accepted),
        "rejection_reason": execution.apply_result.rejection_reason,
        "rms_px": execution.apply_result.rms,
    }
    final_params_for_theta_contract = dict(value_callbacks.current_params())
    if isinstance(final_summary, Mapping):
        final_progress.update(progress_writer._merge_point_match_summary(final_summary))
        final_progress["point_match_summary"] = copy.deepcopy(dict(final_summary))
        for metric_name in (
            "raw_angular_rms_deg",
            "raw_angular_max_deg",
            "final_rms_deg",
            "final_max_deg",
        ):
            if metric_name in final_summary:
                final_progress[metric_name] = final_summary.get(metric_name)
    if solver_result is not None:
        final_progress["optimizer_nfev"] = getattr(solver_result, "nfev", None)
        final_progress["optimizer_njev"] = getattr(solver_result, "njev", None)
        with suppress(Exception):
            final_solver_residual = np.asarray(
                getattr(solver_result, "fun", ()),
                dtype=float,
            ).reshape(-1)
            if final_solver_residual.size:
                final_progress["least_squares_final_residual_vector"] = (
                    final_solver_residual.tolist()
                )
                final_progress["least_squares_final_residual_count"] = int(
                    final_solver_residual.size
                )
        objective_eval_count = getattr(solver_result, "objective_eval_count", None)
        if objective_eval_count is not None:
            final_progress["objective_eval_count"] = _headless_progress_int(
                objective_eval_count,
                0,
            )
        final_progress.update(
            _headless_geometry_fit_outcome_status_fields(
                apply_accepted=bool(execution.apply_result.accepted),
                solver_result=solver_result,
                final_summary=final_summary if isinstance(final_summary, Mapping) else None,
            )
        )
        final_progress.update(
            _headless_geometry_fit_final_reporting_fields(
                solver_result=solver_result,
                active_var_names=var_names,
                initial_params=initial_fit_params,
                current_params=final_params_for_theta_contract,
                progress_data=progress_writer.data,
            )
        )
        final_params_for_theta_contract = _headless_geometry_fit_report_final_params(
            current_params=final_params_for_theta_contract,
            active_var_names=var_names,
            solver_result=solver_result,
            progress_data=progress_writer.data,
        )
        final_progress.update(
            _headless_geometry_fit_theta_contract_payload(
                prepared_run,
                theta_offset_override=final_params_for_theta_contract.get("theta_offset"),
            )
        )
        if not bool(final_progress.get("accepted")) and not final_progress.get("rejection_reason"):
            final_progress["rejection_reason"] = final_progress.get("outcome_rejection_reason")
    effective_accepted = bool(final_progress.get("accepted"))
    effective_rejection_reason = final_progress.get("rejection_reason")
    progress_writer.write("final_validation", **final_progress)

    mosaic_shape_report: dict[str, object] | None = None
    if fit_mosaic_shape:
        if not effective_accepted:
            raise RuntimeError(
                str(effective_rejection_reason or "Geometry fit solution was rejected.")
            )

        raw_mosaic_shape_cfg = defaults.fit_config.get("mosaic_shape")
        if not isinstance(raw_mosaic_shape_cfg, Mapping):
            raise TypeError("fit.mosaic_shape must be a mapping")
        raw_solver_cfg = raw_mosaic_shape_cfg.get("solver")
        raw_roi_cfg = raw_mosaic_shape_cfg.get("roi")
        raw_sampling_cfg = raw_mosaic_shape_cfg.get("sampling")
        if not isinstance(raw_solver_cfg, Mapping):
            raise TypeError("fit.mosaic_shape.solver must be a mapping")
        if not isinstance(raw_roi_cfg, Mapping):
            raise TypeError("fit.mosaic_shape.roi must be a mapping")
        if not isinstance(raw_sampling_cfg, Mapping):
            raise TypeError("fit.mosaic_shape.sampling must be a mapping")
        mosaic_solver_cfg = dict(raw_solver_cfg)
        mosaic_roi_cfg = dict(raw_roi_cfg)

        try:
            saved_sample_count = int(var_store["sample_count_var"])
            minimum_sample_count = int(raw_sampling_cfg["min_num_samples"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Mosaic shape fitting requires positive integer sample counts."
            ) from exc
        if saved_sample_count <= 0 or minimum_sample_count <= 0:
            raise ValueError("Mosaic shape fitting requires positive integer sample counts.")

        divergence_fwhm_deg = float(defaults.defaults["divergence_fwhm_deg"])
        bandwidth_sigma_fraction = float(defaults.defaults["bandwidth_sigma_fraction"])
        bandwidth_percent = float(var_store["bandwidth_percent_var"])
        if not (
            np.isfinite(divergence_fwhm_deg)
            and divergence_fwhm_deg >= 0.0
            and np.isfinite(bandwidth_sigma_fraction)
            and bandwidth_sigma_fraction >= 0.0
            and np.isfinite(bandwidth_percent)
            and 0.0 <= bandwidth_percent <= 10.0
        ):
            raise ValueError("Mosaic shape sampling settings must be finite and nonnegative.")
        fwhm_to_sigma = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        (
            beam_x_array,
            beam_y_array,
            theta_array,
            phi_array,
            fit_wavelength_array,
        ) = _load_simulation_mosaic_profiles().generate_random_profiles(
            max(saved_sample_count, minimum_sample_count),
            math.radians(divergence_fwhm_deg) * fwhm_to_sigma,
            bandwidth_sigma_fraction * fwhm_to_sigma,
            nominal_lambda,
            bandwidth_percent / 100.0,
            rng=0,
        )
        mosaic_params.update(
            {
                "beam_x_array": beam_x_array,
                "beam_y_array": beam_y_array,
                "theta_array": theta_array,
                "phi_array": phi_array,
                "wavelength_array": fit_wavelength_array,
                "wavelength_i_array": fit_wavelength_array,
                "n2_sample_array": calc_runtime.resolve_index_of_refraction_array(
                    np.asarray(fit_wavelength_array, dtype=np.float64) * 1.0e-10,
                    cif_path=str(active_cif_path),
                ),
                "_n2_sample_array_wavelength_snapshot": (
                    calc_runtime._n2_wavelength_snapshot_from_angstrom(fit_wavelength_array)
                ),
            }
        )

        optimization = _load_fitting_optimization()
        dataset_specs = [
            gui_geometry_fit.copy_geometry_fit_state_value(dict(spec))
            for spec in prepared_run.dataset_specs
            if isinstance(spec, Mapping)
        ]
        dataset_specs, _focused_peak_selection = focus_mosaic_profile_dataset_specs(
            dataset_specs,
            source_miller=np.asarray(structure_state.miller, dtype=np.float64),
            source_intensities=np.asarray(structure_state.intensities, dtype=np.float64),
            reference_dataset_index=int(defaults.current_background_index),
        )
        focused_dataset_peak_counts = [
            len(spec.get("measured_peaks", ()))
            for spec in dataset_specs
            if isinstance(spec, Mapping)
        ]
        if not dataset_specs or not focused_dataset_peak_counts:
            raise RuntimeError("Mosaic shape fit could not prepare any datasets.")

        configured_min_total_rois = int(mosaic_roi_cfg["min_total_rois"])
        configured_min_per_dataset_rois = int(mosaic_roi_cfg["min_per_dataset_rois"])
        min_total_rois = max(
            1,
            min(configured_min_total_rois, sum(focused_dataset_peak_counts)),
        )
        min_per_dataset_rois = max(
            1,
            min(configured_min_per_dataset_rois, min(focused_dataset_peak_counts)),
        )

        fit_sigma_mosaic = mosaic_solver_cfg["fit_sigma_mosaic"]
        fit_gamma_mosaic = mosaic_solver_cfg["fit_gamma_mosaic"]
        fit_eta = mosaic_solver_cfg["fit_eta"]
        fit_theta_i = mosaic_solver_cfg["fit_theta_i"]
        if not all(
            type(value) is bool
            for value in (fit_sigma_mosaic, fit_gamma_mosaic, fit_eta, fit_theta_i)
        ):
            raise TypeError("Mosaic shape fit parameter toggles must be booleans.")
        if not any((fit_sigma_mosaic, fit_gamma_mosaic, fit_eta, fit_theta_i)):
            raise RuntimeError("Mosaic shape fit requires at least one enabled parameter.")

        theta_i_mode = mosaic_solver_cfg["theta_i_mode"]
        if theta_i_mode == "auto":
            if len(dataset_specs) == 1:
                theta_i_mode = "single"
            elif prepared_run.joint_background_mode:
                theta_i_mode = "shared_offset"
            else:
                theta_i_mode = "per_dataset"
        if theta_i_mode not in {"single", "shared_offset", "per_dataset"}:
            raise ValueError(
                "fit.mosaic_shape.solver.theta_i_mode must be auto, single, "
                "shared_offset, or per_dataset."
            )

        raw_theta_i_bounds = mosaic_solver_cfg["theta_i_bounds_deg"]
        if not isinstance(raw_theta_i_bounds, (list, tuple)) or len(raw_theta_i_bounds) != 2:
            raise ValueError("Mosaic theta_i bounds must contain two values.")
        theta_i_bounds_deg = (
            float(raw_theta_i_bounds[0]),
            float(raw_theta_i_bounds[1]),
        )
        if not (
            np.isfinite(theta_i_bounds_deg[0])
            and np.isfinite(theta_i_bounds_deg[1])
            and theta_i_bounds_deg[0] < theta_i_bounds_deg[1]
        ):
            raise ValueError("Mosaic theta_i bounds must be finite and increasing.")

        mosaic_fit_params = dict(value_callbacks.current_params())
        mosaic_fit_params["mosaic_params"] = dict(mosaic_params)
        result = optimization.fit_mosaic_shape_parameters(
            np.asarray(structure_state.miller, dtype=np.float64),
            np.asarray(structure_state.intensities, dtype=np.float64),
            int(defaults.image_size),
            mosaic_fit_params,
            dataset_specs=dataset_specs,
            loss=str(mosaic_solver_cfg["loss"]),
            f_scale=float(mosaic_solver_cfg["f_scale_px"]),
            max_nfev=int(mosaic_solver_cfg["max_nfev"]),
            max_restarts=int(mosaic_solver_cfg["restarts"]),
            min_total_rois=int(min_total_rois),
            min_per_dataset_rois=int(min_per_dataset_rois),
            equal_dataset_weights=bool(mosaic_roi_cfg["equal_dataset_weights"]),
            workers=mosaic_solver_cfg["workers"],
            parallel_mode=str(mosaic_solver_cfg["parallel_mode"]),
            worker_numba_threads=mosaic_solver_cfg["worker_numba_threads"],
            restart_jitter=float(mosaic_solver_cfg["restart_jitter"]),
            ridge_weight=float(mosaic_solver_cfg["ridge_weight"]),
            specular_relative_intensity_weight=float(
                mosaic_solver_cfg["specular_relative_intensity_weight"]
            ),
            fit_theta_i=fit_theta_i,
            theta_i_mode=str(theta_i_mode),
            theta_i_bounds_deg=theta_i_bounds_deg,
            fit_sigma_mosaic=fit_sigma_mosaic,
            fit_gamma_mosaic=fit_gamma_mosaic,
            fit_eta=fit_eta,
            progress_callback=progress_writer.status,
        )
        result_values = np.asarray(result.x, dtype=np.float64).reshape(-1)
        if result_values.size < 3 or not np.all(np.isfinite(result_values)):
            raise RuntimeError("Mosaic shape fit returned invalid parameters.")
        if not bool(result.acceptance_passed):
            message = str(result.message or "").strip()
            raise RuntimeError(
                f"Mosaic shape fit rejected: {message or 'acceptance criteria failed.'}"
            )

        sigma_mosaic_deg, gamma_mosaic_deg, eta = map(float, result_values[:3])
        var_store["sigma_mosaic_var"] = sigma_mosaic_deg
        var_store["gamma_mosaic_var"] = gamma_mosaic_deg
        var_store["eta_var"] = eta
        mosaic_shape_report = {
            "accepted": True,
            "success": bool(result.success),
            "message": str(result.message or "").strip(),
            "sigma_mosaic_deg": sigma_mosaic_deg,
            "gamma_mosaic_deg": gamma_mosaic_deg,
            "eta": eta,
            "dataset_count": len(dataset_specs),
            "roi_count": int(result.total_roi_count),
            "cost_reduction": float(result.cost_reduction),
            "boundary_warning": str(result.boundary_warning or "").strip(),
        }

    updated_state = _updated_state_snapshot(saved_state, defaults, var_store)
    return HeadlessGeometryFitResult(
        state=updated_state,
        log_path=Path(execution.log_path),
        accepted=effective_accepted,
        rejection_reason=(str(effective_rejection_reason) if effective_rejection_reason else None),
        rms_px=(
            float(execution.apply_result.rms) if execution.apply_result.rms is not None else None
        ),
        mosaic_shape_fit=mosaic_shape_report,
    )
