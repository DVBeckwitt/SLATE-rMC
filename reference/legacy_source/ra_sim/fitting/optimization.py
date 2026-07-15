"""Optimization routines for fitting simulated data to experiments."""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import math
from threading import Lock
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
from scipy.optimize import (
    OptimizeResult,
    least_squares,
)

from ra_sim.fitting._numeric import (
    coerce_nonnegative_int as _nonnegative_index,
    safe_finite_float as _safe_float,
)
from ra_sim.fitting.caked_geometry_solver import solve_caked_geometry_first_rung
from ra_sim.fitting.geometry_fit_parameters import (
    geometry_fit_pixel_size_m,
    set_mosaic_wavelength_array as _set_parameter_mosaic_wavelength_array,
)
from ra_sim.fitting import optimization_mosaic_profiles as _mosaic_profiles
from ra_sim.fitting.optimization_mosaic_profiles import (
    MosaicProfileDatasetContext,
    MosaicProfileROI,
)
from ra_sim.fitting.optimization_runtime import (
    available_parallel_thread_budget as _available_parallel_thread_budget,
    coerce_sequence_items as _coerce_sequence_items,
    threaded_map as _threaded_map,
)
from ra_sim.simulation.diffraction import process_peaks_parallel
from ra_sim.utils.calculations import (
    _n2_wavelength_snapshot_from_angstrom,
    _normalize_n2_source_meta,
    entry_is_nonzero_00l_reflection,
    resolve_canonical_branch,
    resolve_index_of_refraction_array,
)
from ra_sim.utils.parallel import (
    numba_threads_per_worker as _shared_numba_threads_per_worker,
    reserved_worker_count,
)


def _resolve_parallel_worker_count(
    raw_value: object,
    *,
    max_tasks: int,
) -> int:
    """Resolve the bounded worker count for one optimization batch."""

    if max_tasks <= 1:
        return 1

    requested = 0
    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if text in {"", "auto", "default"}:
            requested = 0
        else:
            try:
                requested = int(float(text))
            except Exception:
                requested = 1
    elif raw_value is None:
        requested = 0
    elif isinstance(raw_value, (int, float)):
        requested = int(raw_value)
    else:
        try:
            requested = int(str(raw_value).strip())
        except Exception:
            requested = 1

    if requested <= 0:
        requested = reserved_worker_count(
            thread_budget=_available_parallel_thread_budget(),
        )
    return max(1, min(int(requested), int(max_tasks)))


def _resolve_numba_threads_per_worker(
    worker_count: int,
    raw_value: object,
) -> Optional[int]:
    """Resolve the numba thread count assigned to each worker."""

    if worker_count <= 1:
        return None

    requested = 0
    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if text not in {"", "auto", "default"}:
            try:
                requested = int(float(text))
            except Exception:
                requested = 0
    elif isinstance(raw_value, (int, float)):
        requested = int(raw_value)
    elif raw_value is not None:
        try:
            requested = int(str(raw_value).strip())
        except Exception:
            requested = 0

    if requested > 0:
        return max(int(requested), 1)

    return _shared_numba_threads_per_worker(
        worker_count,
        thread_budget=_available_parallel_thread_budget(),
    )


def _normalize_measured_peaks(
    measured_peaks: Optional[Sequence[object]],
) -> List[Dict[str, object]]:
    """Validate current measured-peak mappings and normalize numeric values."""

    normalized: List[Dict[str, object]] = []
    measured_entries = _coerce_sequence_items(measured_peaks)
    if not measured_entries:
        return normalized

    for entry_index, entry in enumerate(measured_entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"Measured geometry row {entry_index} must be a mapping.")
        missing = sorted({"hkl", "x", "y"} - set(entry))
        if missing:
            raise ValueError(
                f"Measured geometry row {entry_index} is missing required fields: {missing}."
            )
        normalized_entry = dict(entry)
        raw_hkl = entry["hkl"]
        if not isinstance(raw_hkl, (list, tuple, np.ndarray)) or len(raw_hkl) != 3:
            raise ValueError(f"Measured geometry row {entry_index} hkl must have three values.")
        try:
            hkl_values = tuple(float(value) for value in raw_hkl)
            x = float(entry["x"])
            y = float(entry["y"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Measured geometry row {entry_index} hkl, x, and y must be numeric."
            ) from exc
        if not all(np.isfinite(value) and value.is_integer() for value in hkl_values):
            raise ValueError(
                f"Measured geometry row {entry_index} hkl must contain three finite integers."
            )
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError(f"Measured geometry row {entry_index} x and y must be finite.")

        normalized_entry["hkl"] = tuple(int(value) for value in hkl_values)
        normalized_entry["x"] = x
        normalized_entry["y"] = y

        for key in ("sigma_px", "sigma_radial_px", "sigma_tangential_px"):
            if key not in normalized_entry:
                continue
            try:
                sigma_value = float(normalized_entry[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Measured geometry row {entry_index} {key} must be numeric."
                ) from exc
            if not np.isfinite(sigma_value) or sigma_value <= 0.0:
                raise ValueError(
                    f"Measured geometry row {entry_index} {key} must be finite and positive."
                )
            normalized_entry[key] = sigma_value
        normalized.append(normalized_entry)
    return normalized


@dataclass
class ReflectionSimulationSubset:
    """Reduced reflection list and remapped measured entries for fitting."""

    miller: np.ndarray
    intensities: np.ndarray
    measured_entries: List[Dict[str, object]]


@dataclass
class GeometryFitDatasetContext:
    """One measured-peak dataset used in a geometry fit."""

    dataset_index: int
    label: str
    theta_initial: float
    subset: ReflectionSimulationSubset
    experimental_image: Optional[np.ndarray] = None


def _prepare_reflection_subset(
    miller: np.ndarray,
    intensities: np.ndarray,
    measured_peaks: Optional[Sequence[object]],
) -> ReflectionSimulationSubset:
    """Restrict simulation to measured reflections with canonical full provenance."""

    miller_arr = np.asarray(miller, dtype=np.float64)
    intensities_arr = np.asarray(intensities, dtype=np.float64)
    if miller_arr.ndim != 2 or miller_arr.shape[1] != 3:
        raise ValueError("Geometry fitting requires an N x 3 Miller array.")
    if intensities_arr.ndim != 1 or intensities_arr.shape[0] != miller_arr.shape[0]:
        raise ValueError("Geometry fitting requires one intensity per Miller row.")

    measured_entries = _normalize_measured_peaks(measured_peaks)
    if not measured_entries:
        return ReflectionSimulationSubset(
            miller=miller_arr,
            intensities=intensities_arr,
            measured_entries=[],
        )

    reflection_indices: list[int] = []
    seen_indices: set[int] = set()
    validated_entries: list[dict[str, object]] = []
    for entry_index, raw_entry in enumerate(measured_entries):
        entry = dict(raw_entry)
        hkl = _mosaic_profiles._normalized_hkl_key(entry.get("hkl"))
        if hkl is None:
            raise ValueError(f"Measured geometry row {entry_index} requires canonical hkl.")
        reflection_index = _nonnegative_index(entry.get("source_reflection_index"))
        if reflection_index is None or reflection_index >= miller_arr.shape[0]:
            raise ValueError(
                f"Measured geometry row {entry_index} has an invalid source_reflection_index."
            )
        source_hkl = _mosaic_profiles._miller_key_from_row(miller_arr[reflection_index])
        if source_hkl != hkl:
            raise ValueError(
                f"Measured geometry row {entry_index} HKL does not match its source reflection."
            )
        source_row_index = _nonnegative_index(entry.get("source_row_index"))
        if source_row_index is None:
            raise ValueError(
                f"Measured geometry row {entry_index} requires source_row_index provenance."
            )
        branch_index, _branch_source, branch_reason = resolve_canonical_branch(entry)
        if not entry_is_nonzero_00l_reflection(entry) and branch_index not in {0, 1}:
            raise ValueError(
                f"Measured geometry row {entry_index} requires source_branch_index: "
                f"{branch_reason or 'missing_source_branch_index'}"
            )

        if reflection_index not in seen_indices:
            seen_indices.add(reflection_index)
            reflection_indices.append(reflection_index)
        entry["hkl"] = hkl
        entry["source_reflection_index"] = int(reflection_index)
        entry["source_row_index"] = int(source_row_index)
        if branch_index in {0, 1}:
            entry["source_branch_index"] = int(branch_index)
        else:
            entry.pop("source_branch_index", None)
        validated_entries.append(entry)

    original_indices = np.asarray(reflection_indices, dtype=np.int64)
    local_index_map = {
        int(original_index): int(local_index)
        for local_index, original_index in enumerate(original_indices)
    }
    remapped_entries: list[dict[str, object]] = []
    for entry in validated_entries:
        remapped = dict(entry)
        local_index = local_index_map[int(entry["source_reflection_index"])]
        remapped["source_table_index"] = int(local_index)
        remapped["resolved_table_index"] = int(local_index)
        remapped_entries.append(remapped)

    return ReflectionSimulationSubset(
        miller=miller_arr[original_indices],
        intensities=intensities_arr[original_indices],
        measured_entries=remapped_entries,
    )


def _build_geometry_fit_dataset_contexts(
    miller: np.ndarray,
    intensities: np.ndarray,
    dataset_specs: Sequence[Dict[str, object]],
) -> List[GeometryFitDatasetContext]:
    """Validate geometry-fit dataset specs and build internal contexts."""

    raw_specs: List[Dict[str, object]] = []
    dataset_spec_entries = _coerce_sequence_items(dataset_specs)
    for entry_index, raw_entry in enumerate(dataset_spec_entries):
        if not isinstance(raw_entry, dict):
            raise TypeError("geometry fit dataset_specs entries must be dictionaries")
        missing = sorted(
            {"dataset_index", "label", "theta_initial", "measured_peaks", "experimental_image"}
            - set(raw_entry)
        )
        if missing:
            raise ValueError(f"Geometry fit dataset {entry_index} is missing fields: {missing}.")
        raw_specs.append(dict(raw_entry))

    contexts: List[GeometryFitDatasetContext] = []
    for entry_index, entry in enumerate(raw_specs):
        try:
            dataset_index = int(entry["dataset_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Geometry fit dataset {entry_index} dataset_index must be an integer."
            ) from exc
        if dataset_index < 0:
            raise ValueError(
                f"Geometry fit dataset {entry_index} dataset_index must be non-negative."
            )

        try:
            theta_initial = float(entry["theta_initial"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Geometry fit dataset {entry_index} theta_initial must be numeric."
            ) from exc
        if not np.isfinite(theta_initial):
            raise ValueError(f"Geometry fit dataset {entry_index} theta_initial must be finite.")

        label = str(entry["label"])
        measured_local = entry["measured_peaks"]
        experimental_local = entry["experimental_image"]
        subset = _prepare_reflection_subset(miller, intensities, measured_local)
        contexts.append(
            GeometryFitDatasetContext(
                dataset_index=int(dataset_index),
                label=label,
                theta_initial=float(theta_initial),
                subset=subset,
                experimental_image=(
                    None
                    if experimental_local is None
                    else np.asarray(experimental_local, dtype=np.float64)
                ),
            )
        )
    return contexts


def _robust_cost(residual: np.ndarray, loss: str, f_scale: float) -> float:
    """Compute the least_squares robust objective for a residual vector."""

    residual = np.asarray(residual, dtype=np.float64)
    if residual.size == 0:
        return 0.0
    f_scale = max(float(f_scale), 1e-12)
    z = (residual / f_scale) ** 2
    loss_key = str(loss).strip().lower()

    if loss_key == "linear":
        rho = z
    elif loss_key == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss_key == "huber":
        rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif loss_key == "cauchy":
        rho = np.log1p(z)
    elif loss_key == "arctan":
        rho = np.arctan(z)
    else:
        raise ValueError(f"Unsupported loss '{loss}'.")
    return 0.5 * (f_scale * f_scale) * float(np.sum(rho))


def _estimate_mosaic_shape_roi_half_width(
    params: Dict[str, object],
    upper_bounds: Sequence[float],
    image_size: int,
) -> int:
    """Estimate one fixed ROI radius from detector geometry and width bounds."""

    detector_distance = max(float(params.get("corto_detector", 0.0)), 1.0e-6)
    pixel_size = geometry_fit_pixel_size_m(params)
    width_upper = max(float(upper_bounds[0]), float(upper_bounds[1]), 0.03)
    projected_half_width_px = (
        detector_distance * math.radians(width_upper) / max(pixel_size, 1.0e-9)
    )
    half_width = int(
        np.clip(
            math.ceil(max(8.0, 2.0 * projected_half_width_px)),
            8,
            max(8, min(64, int(image_size) // 4)),
        )
    )
    return max(int(half_width), 1)


def _fit_mosaic_shape_parameters_profiles(
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, object],
    *,
    dataset_specs: Sequence[Dict[str, object]],
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    max_nfev: int = 80,
    max_restarts: int = 2,
    roi_half_width: Optional[int] = None,
    min_total_rois: int = 8,
    min_per_dataset_rois: int = 3,
    equal_dataset_weights: bool = True,
    workers: object = "auto",
    parallel_mode: str = "auto",
    worker_numba_threads: object = 0,
    restart_jitter: float = 0.15,
    ridge_weight: float = 1.0,
    specular_relative_intensity_weight: float = 0.0,
    fit_theta_i: bool = True,
    theta_i_mode: str = "auto",
    theta_i_bounds_deg: Optional[Tuple[float, float]] = None,
    fit_sigma_mosaic: bool = True,
    fit_gamma_mosaic: bool = True,
    fit_eta: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> OptimizeResult:
    """Fit sigma/gamma/eta using in-plane phi profiles plus specular 00L profiles."""

    dataset_spec_entries = _coerce_sequence_items(dataset_specs)
    miller = np.asarray(miller, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    if miller.ndim != 2 or miller.shape[1] != 3:
        raise ValueError("miller must be an array of shape (N, 3)")
    if intensities.ndim != 1 or intensities.shape[0] != miller.shape[0]:
        raise ValueError("intensities and miller must have matching lengths")
    if int(image_size) <= 0:
        raise ValueError("image_size must be positive")
    if not dataset_spec_entries:
        raise RuntimeError("Mosaic shape fit requires at least one prepared dataset_spec.")

    mosaic_params = dict(params.get("mosaic_params", {}))
    if not mosaic_params:
        raise ValueError("params['mosaic_params'] is required")
    required_keys = ("beam_x_array", "beam_y_array", "theta_array", "phi_array")
    missing_keys = [key for key in required_keys if not np.asarray(mosaic_params.get(key)).size]
    if missing_keys:
        raise ValueError("mosaic_params must include beam and divergence samples for shape fitting")

    wavelength_array = mosaic_params.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic_params.get("wavelength_i_array")
    if wavelength_array is None:
        base_lambda = float(params.get("lambda", 1.0))
        wavelength_array = np.full(
            np.asarray(mosaic_params["beam_x_array"]).shape,
            base_lambda,
            dtype=np.float64,
        )
    else:
        wavelength_array = np.asarray(wavelength_array, dtype=np.float64)

    sigma0 = float(mosaic_params.get("sigma_mosaic_deg", 0.5))
    gamma0 = float(mosaic_params.get("gamma_mosaic_deg", 0.5))
    eta0 = float(mosaic_params.get("eta", 0.05))
    if bounds is None:
        bounds = (
            np.array([0.03, 0.03, 0.0], dtype=np.float64),
            np.array([3.0, 3.0, 1.0], dtype=np.float64),
        )
    lower = np.asarray(bounds[0], dtype=np.float64).reshape(-1)
    upper = np.asarray(bounds[1], dtype=np.float64).reshape(-1)
    if lower.size != 3 or upper.size != 3:
        raise ValueError("bounds must contain exactly 3 lower/upper values")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be strictly less than upper bound")

    profile_weight = max(float(ridge_weight), 0.0)
    specular_ratio_weight = max(float(specular_relative_intensity_weight), 0.0)
    if profile_weight <= 0.0 and specular_ratio_weight <= 0.0:
        raise ValueError(
            "At least one mosaic objective term must be enabled "
            "(ridge_weight > 0 or specular_relative_intensity_weight > 0)."
        )

    x0 = np.clip(np.array([sigma0, gamma0, eta0], dtype=np.float64), lower, upper)
    fit_sigma_mosaic = bool(fit_sigma_mosaic)
    fit_gamma_mosaic = bool(fit_gamma_mosaic)
    fit_eta = bool(fit_eta)
    min_total_rois = max(int(min_total_rois), 1)
    min_per_dataset_rois = max(int(min_per_dataset_rois), 1)
    max_restarts = max(int(max_restarts), 0)
    restart_jitter = max(float(restart_jitter), 0.0)
    progress_lock = Lock()

    def _emit_progress(message: str) -> None:
        if not callable(progress_callback):
            return
        text = str(message).strip()
        if not text:
            return
        with progress_lock:
            try:
                progress_callback(text)
            except Exception:
                pass

    if roi_half_width is None:
        roi_half_width = _estimate_mosaic_shape_roi_half_width(
            params,
            upper,
            int(image_size),
        )
    roi_half_width = int(roi_half_width)
    if roi_half_width <= 0:
        raise ValueError("roi_half_width must be a positive integer")

    _emit_progress(
        "Preparing dataset ROIs: "
        f"datasets={len(dataset_spec_entries)}, roi_half_width={int(roi_half_width)}"
    )
    prepared_datasets, rejected_rois = _mosaic_profiles._build_mosaic_profile_dataset_contexts(
        miller,
        intensities,
        int(image_size),
        dict(params),
        list(dataset_spec_entries),
        roi_half_width=int(roi_half_width),
        build_geometry_fit_dataset_contexts=_build_geometry_fit_dataset_contexts,
        miller_key_from_row=_mosaic_profiles._miller_key_from_row,
        normalized_hkl_key=_mosaic_profiles._normalized_hkl_key,
        measured_detector_anchor=_measured_detector_anchor,
        detector_pixels_to_fit_space=_detector_pixels_to_fit_space,
    )
    if not prepared_datasets:
        raise RuntimeError("Mosaic shape fit found no prepared datasets.")

    rejected_reason_counts_by_dataset: Dict[str, Dict[str, int]] = {}
    for rejected in rejected_rois:
        dataset_label = str(rejected.get("dataset_label") or "")
        reason = str(rejected.get("reason") or "unknown")
        if not dataset_label:
            continue
        bucket = rejected_reason_counts_by_dataset.setdefault(dataset_label, {})
        bucket[reason] = int(bucket.get(reason, 0)) + 1

    dataset_failures = [
        dataset_ctx
        for dataset_ctx in prepared_datasets
        if len(dataset_ctx.rois) < int(min_per_dataset_rois)
    ]
    if dataset_failures:
        dataset_text = ", ".join(f"{ctx.label}={len(ctx.rois)}" for ctx in dataset_failures)
        rejection_text = ", ".join(
            f"{ctx.label}["
            + ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(
                    rejected_reason_counts_by_dataset.get(str(ctx.label), {}).items(),
                    key=lambda item: (-int(item[1]), str(item[0])),
                )[:3]
            )
            + "]"
            for ctx in dataset_failures
            if rejected_reason_counts_by_dataset.get(str(ctx.label))
        )
        detail_suffix = f" Rejections: {rejection_text}." if rejection_text else ""
        raise RuntimeError(
            "Mosaic shape fit needs at least "
            f"{int(min_per_dataset_rois)} usable ROIs per dataset; got {dataset_text}."
            f"{detail_suffix}"
        )

    total_rois = int(sum(len(dataset_ctx.rois) for dataset_ctx in prepared_datasets))
    if total_rois < int(min_total_rois):
        raise RuntimeError(
            f"Mosaic shape fit needs at least {int(min_total_rois)} usable ROIs; got {total_rois}."
        )

    total_in_plane = int(sum(ctx.in_plane_roi_count for ctx in prepared_datasets))
    total_specular = int(sum(ctx.specular_roi_count for ctx in prepared_datasets))
    specular_required = bool(specular_ratio_weight > 0.0)
    if total_in_plane <= 0:
        raise RuntimeError("Mosaic shape fit needs at least one usable off-specular profile.")
    if specular_required and total_specular <= 0:
        raise RuntimeError("Mosaic shape fit needs at least one usable specular (00l) profile.")
    if specular_required and total_specular < 2:
        raise RuntimeError(
            "Mosaic shape fit needs at least two specular (00l) profiles when relative-intensity fitting is enabled."
        )

    def _json_safe(value: object) -> object:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, complex):
            return {
                "real": _json_safe(value.real),
                "imag": _json_safe(value.imag),
            }
        if isinstance(value, np.generic):
            return _json_safe(value.item())
        if isinstance(value, np.ndarray):
            return [_json_safe(item) for item in value.tolist()]
        if isinstance(value, Mapping):
            return {str(key): _json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)

    def _array_summary(values: object) -> Dict[str, object]:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = arr[np.isfinite(arr)]
        summary: Dict[str, object] = {
            "count": int(arr.size),
            "finite_count": int(finite.size),
        }
        if finite.size:
            summary.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                }
            )
        else:
            summary.update(
                {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                }
            )
        return summary

    def _image_summary(image: object) -> Dict[str, object]:
        arr = np.asarray(image, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        summary: Dict[str, object] = {
            "shape": [int(dim) for dim in arr.shape],
            "finite_count": int(finite.size),
        }
        if finite.size:
            summary.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "sum": float(np.sum(finite)),
                }
            )
        else:
            summary.update(
                {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "sum": None,
                }
            )
        return summary

    def _measured_peak_summaries(entries: object) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for entry in entries if isinstance(entries, (list, tuple)) else []:
            if not isinstance(entry, Mapping):
                continue
            peak_summary: Dict[str, object] = {
                "hkl": _json_safe(entry.get("hkl")),
                "x": _json_safe(entry.get("x")),
                "y": _json_safe(entry.get("y")),
            }
            if "source_table_index" in entry:
                peak_summary["source_table_index"] = _json_safe(entry.get("source_table_index"))
            if "source_row_index" in entry:
                peak_summary["source_row_index"] = _json_safe(entry.get("source_row_index"))
            out.append(peak_summary)
        return out

    input_dataset_summaries: List[Dict[str, object]] = []
    for spec in dataset_spec_entries:
        if not isinstance(spec, Mapping):
            continue
        measured_peaks = spec.get("measured_peaks", [])
        experimental_image = spec.get("experimental_image")
        input_dataset_summaries.append(
            {
                "dataset_index": _json_safe(spec.get("dataset_index")),
                "dataset_label": _json_safe(spec.get("label", spec.get("dataset_index", "?"))),
                "theta_initial_deg": _json_safe(spec.get("theta_initial")),
                "measured_peak_count": int(
                    len(measured_peaks) if isinstance(measured_peaks, (list, tuple)) else 0
                ),
                "experimental_image": (
                    _image_summary(experimental_image) if experimental_image is not None else None
                ),
                "measured_peaks": _measured_peak_summaries(measured_peaks),
            }
        )

    prepared_dataset_summaries: List[Dict[str, object]] = []
    for dataset_ctx in prepared_datasets:
        roi_entries = [
            {
                "reflection_index": int(roi.reflection_index),
                "hkl": [int(v) for v in roi.hkl],
                "family": str(roi.family),
                "axis_name": str(roi.axis_name),
                "center_xy_px": [float(roi.center_col), float(roi.center_row)],
                "row_bounds": [int(v) for v in roi.row_bounds],
                "col_bounds": [int(v) for v in roi.col_bounds],
                "measured_two_theta_deg": float(roi.measured_two_theta_deg),
                "measured_phi_deg": float(roi.measured_phi_deg),
                "measured_area": float(roi.measured_area),
            }
            for roi in dataset_ctx.rois
        ]
        prepared_dataset_summaries.append(
            {
                "dataset_index": int(dataset_ctx.dataset_index),
                "dataset_label": str(dataset_ctx.label),
                "theta_initial_deg": float(dataset_ctx.theta_initial),
                "experimental_image": _image_summary(dataset_ctx.experimental_image),
                "measured_peak_count": int(dataset_ctx.measured_peak_count),
                "reflection_count": int(dataset_ctx.miller.shape[0]),
                "intensity_count": int(dataset_ctx.intensities.shape[0]),
                "roi_count": int(len(dataset_ctx.rois)),
                "in_plane_roi_count": int(dataset_ctx.in_plane_roi_count),
                "specular_roi_count": int(dataset_ctx.specular_roi_count),
                "roi_hkls": [[int(v) for v in roi.hkl] for roi in dataset_ctx.rois],
                "rois": roi_entries,
            }
        )

    rejected_roi_reason_counts: Dict[str, int] = {}
    rejected_roi_counts_by_dataset: Dict[str, int] = {}
    for rejected in rejected_rois:
        if not isinstance(rejected, Mapping):
            continue
        dataset_key = str(rejected.get("dataset_label", rejected.get("dataset_index", "?")))
        reason_key = str(rejected.get("reason", "unknown"))
        stage_key = str(rejected.get("stage", "unknown"))
        rejected_roi_reason_counts[f"{dataset_key}:{stage_key}:{reason_key}"] = (
            rejected_roi_reason_counts.get(f"{dataset_key}:{stage_key}:{reason_key}", 0) + 1
        )
        rejected_roi_counts_by_dataset[dataset_key] = (
            rejected_roi_counts_by_dataset.get(dataset_key, 0) + 1
        )

    base_params = dict(params)
    beam_x = np.asarray(mosaic_params.get("beam_x_array"), dtype=np.float64)
    beam_y = np.asarray(mosaic_params.get("beam_y_array"), dtype=np.float64)
    theta_array = np.asarray(mosaic_params.get("theta_array"), dtype=np.float64)
    phi_array = np.asarray(mosaic_params.get("phi_array"), dtype=np.float64)
    uv1 = np.asarray(
        base_params.get("uv1", np.array([1.0, 0.0, 0.0])),
        dtype=np.float64,
    )
    uv2 = np.asarray(
        base_params.get("uv2", np.array([0.0, 1.0, 0.0])),
        dtype=np.float64,
    )

    dataset_count = int(len(prepared_datasets))
    optimize_theta = bool(fit_theta_i)

    theta_mode_key = str(theta_i_mode).strip().lower()
    if theta_mode_key not in {"auto", "single", "shared_offset", "per_dataset"}:
        raise ValueError(
            "theta_i_mode must be one of {'auto', 'single', 'shared_offset', 'per_dataset'}"
        )
    if theta_mode_key == "auto":
        resolved_theta_mode = "single" if dataset_count == 1 else "per_dataset"
    else:
        resolved_theta_mode = str(theta_mode_key)
    if resolved_theta_mode == "single" and dataset_count != 1:
        raise ValueError("theta_i_mode='single' requires exactly one prepared dataset")

    param_names = ["sigma_mosaic_deg", "gamma_mosaic_deg", "eta"]
    theta_param_names: List[str] = []
    theta_param_dataset_indices: List[Optional[int]] = []
    if optimize_theta:
        theta_bound_lower: float
        theta_bound_upper: float
        if theta_i_bounds_deg is None:
            theta_bound_lower = -0.5
            theta_bound_upper = 0.5
        else:
            theta_bound_lower = float(theta_i_bounds_deg[0])
            theta_bound_upper = float(theta_i_bounds_deg[1])
        if not (np.isfinite(theta_bound_lower) and np.isfinite(theta_bound_upper)):
            raise ValueError("theta_i_bounds_deg must be finite when fit_theta_i is enabled")
        if theta_bound_lower >= theta_bound_upper:
            raise ValueError(
                "theta_i_bounds_deg lower bound must be strictly less than the upper bound"
            )

        theta_seeds: List[float] = []
        theta_lowers: List[float] = []
        theta_uppers: List[float] = []
        if resolved_theta_mode == "shared_offset":
            theta_seed = _safe_float(params.get("theta_offset", 0.0), 0.0)
            theta_param_names.append("theta_offset")
            theta_param_dataset_indices.append(None)
            theta_seeds.append(float(np.clip(theta_seed, theta_bound_lower, theta_bound_upper)))
            theta_lowers.append(float(theta_bound_lower))
            theta_uppers.append(float(theta_bound_upper))
        elif resolved_theta_mode == "single":
            theta_seed = _safe_float(
                params.get("theta_initial", prepared_datasets[0].theta_initial),
                float(prepared_datasets[0].theta_initial),
            )
            theta_param_names.append("theta_initial")
            theta_param_dataset_indices.append(int(prepared_datasets[0].dataset_index))
            theta_seeds.append(float(np.clip(theta_seed, theta_bound_lower, theta_bound_upper)))
            theta_lowers.append(float(theta_bound_lower))
            theta_uppers.append(float(theta_bound_upper))
        else:
            theta_seed_map = dict(base_params.get("_mosaic_theta_initials_by_dataset", {}))
            for dataset_ctx in prepared_datasets:
                dataset_idx = int(dataset_ctx.dataset_index)
                dataset_theta = float(dataset_ctx.theta_initial)
                theta_seed = _safe_float(
                    theta_seed_map.get(dataset_idx, dataset_theta),
                    dataset_theta,
                )
                lower_i = float(dataset_theta + theta_bound_lower)
                upper_i = float(dataset_theta + theta_bound_upper)
                theta_param_names.append(f"theta_initial[{dataset_idx}]")
                theta_param_dataset_indices.append(int(dataset_idx))
                theta_seeds.append(float(np.clip(theta_seed, lower_i, upper_i)))
                theta_lowers.append(float(lower_i))
                theta_uppers.append(float(upper_i))

        x0 = np.concatenate([x0, np.asarray(theta_seeds, dtype=np.float64)])
        lower = np.concatenate([lower, np.asarray(theta_lowers, dtype=np.float64)])
        upper = np.concatenate([upper, np.asarray(theta_uppers, dtype=np.float64)])

    if optimize_theta and theta_param_names:
        param_names.extend(str(name) for name in theta_param_names)
    active_mask = np.array(
        [
            fit_sigma_mosaic,
            fit_gamma_mosaic,
            fit_eta,
            *([True] * len(theta_param_names) if optimize_theta else []),
        ],
        dtype=bool,
    )
    active_indices = np.flatnonzero(active_mask)
    fixed_indices = np.flatnonzero(~active_mask)
    if active_indices.size == 0:
        raise ValueError(
            "At least one mosaic fit parameter must be enabled "
            "(fit_sigma_mosaic, fit_gamma_mosaic, fit_eta, or fit_theta_i)."
        )

    full_x0 = np.asarray(x0, dtype=np.float64)
    full_lower = np.asarray(lower, dtype=np.float64)
    full_upper = np.asarray(upper, dtype=np.float64)
    active_x0 = np.asarray(full_x0[active_indices], dtype=np.float64)
    active_lower = np.asarray(full_lower[active_indices], dtype=np.float64)
    active_upper = np.asarray(full_upper[active_indices], dtype=np.float64)
    active_parameter_names = [str(param_names[idx]) for idx in active_indices.tolist()]
    fixed_parameter_names = [str(param_names[idx]) for idx in fixed_indices.tolist()]

    configured_parallel_workers = _resolve_parallel_worker_count(
        workers,
        max_tasks=max(len(prepared_datasets), max_restarts + 1, 1),
    )
    parallel_mode_key = str(parallel_mode).strip().lower()
    if parallel_mode_key not in {"auto", "datasets", "restarts", "off"}:
        raise ValueError("parallel_mode must be one of {'auto', 'datasets', 'restarts', 'off'}")
    dataset_parallel_workers = 1
    restart_parallel_workers = 1
    if configured_parallel_workers > 1 and parallel_mode_key != "off":
        if parallel_mode_key in {"auto", "datasets"} and len(prepared_datasets) > 1:
            dataset_parallel_workers = min(
                int(configured_parallel_workers),
                len(prepared_datasets),
            )
        elif parallel_mode_key in {"auto", "restarts"} and max_restarts > 1:
            restart_parallel_workers = min(
                int(configured_parallel_workers),
                int(max_restarts),
            )
    active_outer_workers = max(dataset_parallel_workers, restart_parallel_workers)
    numba_threads = _resolve_numba_threads_per_worker(
        active_outer_workers,
        worker_numba_threads,
    )
    parallelization_summary = {
        "mode": str(parallel_mode_key),
        "configured_workers": int(configured_parallel_workers),
        "dataset_workers": int(dataset_parallel_workers),
        "restart_workers": int(restart_parallel_workers),
        "worker_numba_threads": (None if numba_threads is None else int(numba_threads)),
        "numba_thread_budget": int(_available_parallel_thread_budget()),
    }

    def _format_cost(value: object) -> str:
        try:
            out = float(value)
        except Exception:
            return "n/a"
        if not np.isfinite(out):
            return "n/a"
        return f"{out:.6g}"

    def _format_rms(values: Sequence[float]) -> str:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return "0"
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return "n/a"
        return f"{float(np.sqrt(np.mean(finite * finite))):.6g}"

    def _format_params(x_values: Sequence[float]) -> str:
        arr = np.asarray(x_values, dtype=np.float64).reshape(-1)
        if arr.size < 3:
            return "params=n/a"
        parts = [
            f"sigma={float(arr[0]):.4f}",
            f"gamma={float(arr[1]):.4f}",
            f"eta={float(arr[2]):.4f}",
        ]
        if optimize_theta and arr.size > 3:
            theta_slice = np.asarray(arr[3:], dtype=np.float64).reshape(-1)
            if resolved_theta_mode == "shared_offset" and theta_slice.size >= 1:
                parts.append(f"theta_offset={float(theta_slice[0]):+.4f}")
            elif resolved_theta_mode == "single" and theta_slice.size >= 1:
                parts.append(f"theta={float(theta_slice[0]):.4f}")
            elif theta_slice.size:
                theta_finite = theta_slice[np.isfinite(theta_slice)]
                if theta_finite.size:
                    parts.append(
                        "theta_range="
                        f"[{float(np.min(theta_finite)):.4f},{float(np.max(theta_finite)):.4f}]"
                    )
        return ", ".join(parts)

    rejection_preview = sorted(
        rejected_roi_reason_counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )
    _emit_progress(
        "Prepared "
        f"{len(prepared_datasets)} dataset(s): total_rois={int(total_rois)}, "
        f"in_plane={int(total_in_plane)}, specular={int(total_specular)}, "
        f"active_params={','.join(active_parameter_names)}, "
        f"fixed_params={','.join(fixed_parameter_names) if fixed_parameter_names else '-'}"
    )
    for dataset_ctx in prepared_datasets:
        _emit_progress(
            "Dataset "
            f"{dataset_ctx.label}: theta={float(dataset_ctx.theta_initial):.4f} deg, "
            f"rois={len(dataset_ctx.rois)}, in_plane={int(dataset_ctx.in_plane_roi_count)}, "
            f"specular={int(dataset_ctx.specular_roi_count)}, "
            f"reflections={int(dataset_ctx.miller.shape[0])}"
        )
    if rejected_rois:
        preview_text = ", ".join(f"{key}={count}" for key, count in rejection_preview[:6])
        if len(rejection_preview) > 6:
            preview_text += ", ..."
        _emit_progress(f"ROI rejections during prep: count={len(rejected_rois)} [{preview_text}]")
    _emit_progress(
        "Parallel plan: "
        f"mode={parallel_mode_key}, dataset_workers={int(dataset_parallel_workers)}, "
        f"restart_workers={int(restart_parallel_workers)}, "
        "worker_numba_threads="
        f"{'auto' if numba_threads is None else int(numba_threads)}"
    )

    def _expand_active_vector(x_trial: Sequence[float]) -> np.ndarray:
        x_arr = np.asarray(x_trial, dtype=np.float64).reshape(-1)
        if active_indices.size == full_x0.size:
            if x_arr.size != full_x0.size:
                raise ValueError("mosaic fit parameter vector length mismatch")
            return x_arr
        if x_arr.size != active_indices.size:
            raise ValueError("mosaic fit active parameter vector length mismatch")
        full = np.array(full_x0, copy=True)
        full[active_indices] = x_arr
        return full

    def _apply_trial_params(x_values: Sequence[float]) -> Dict[str, object]:
        local = dict(base_params)
        local_mosaic = dict(mosaic_params)
        local_mosaic["sigma_mosaic_deg"] = float(x_values[0])
        local_mosaic["gamma_mosaic_deg"] = float(x_values[1])
        local_mosaic["eta"] = float(x_values[2])
        local_mosaic["beam_x_array"] = beam_x
        local_mosaic["beam_y_array"] = beam_y
        local_mosaic["theta_array"] = theta_array
        local_mosaic["phi_array"] = phi_array
        _set_parameter_mosaic_wavelength_array(local_mosaic, wavelength_array)
        local["mosaic_params"] = local_mosaic
        local.setdefault("theta_offset", 0.0)
        if optimize_theta:
            theta_slice = np.asarray(x_values[3:], dtype=np.float64)
            if resolved_theta_mode == "shared_offset" and theta_slice.size >= 1:
                local["theta_offset"] = float(theta_slice[0])
            elif resolved_theta_mode == "single" and theta_slice.size >= 1:
                local["theta_initial"] = float(theta_slice[0])
            elif resolved_theta_mode == "per_dataset" and theta_slice.size:
                local["_mosaic_theta_initials_by_dataset"] = {
                    int(dataset_idx): float(theta_slice[idx])
                    for idx, dataset_idx in enumerate(theta_param_dataset_indices)
                    if dataset_idx is not None and idx < theta_slice.size
                }
        return local

    def _theta_initial_for_dataset(
        local: Dict[str, object],
        dataset_ctx: MosaicProfileDatasetContext,
    ) -> float:
        theta_base = _safe_float(dataset_ctx.theta_initial, 0.0)
        if resolved_theta_mode == "shared_offset":
            return float(theta_base + _safe_float(local.get("theta_offset", 0.0), 0.0))
        if resolved_theta_mode == "single":
            return _safe_float(local.get("theta_initial", theta_base), theta_base)
        theta_map = local.get("_mosaic_theta_initials_by_dataset", {})
        if isinstance(theta_map, Mapping):
            return _safe_float(
                theta_map.get(int(dataset_ctx.dataset_index), theta_base),
                theta_base,
            )
        return float(theta_base)

    image_cache: Dict[Tuple[int, float, float, float, float], np.ndarray] = {}
    image_cache_lock = Lock()

    def _simulate_dataset_image(
        local: Dict[str, object],
        dataset_ctx: MosaicProfileDatasetContext,
        *,
        theta_value: float,
    ) -> np.ndarray:
        key = (
            int(dataset_ctx.dataset_index),
            float(local["mosaic_params"]["sigma_mosaic_deg"]),
            float(local["mosaic_params"]["gamma_mosaic_deg"]),
            float(local["mosaic_params"]["eta"]),
            float(theta_value),
        )
        with image_cache_lock:
            cached = image_cache.get(key)
        if cached is not None:
            return cached

        local_mosaic = dict(local["mosaic_params"])
        wave_local = local_mosaic.get("wavelength_array")
        if wave_local is None:
            wave_local = local_mosaic.get("wavelength_i_array")
        if wave_local is None:
            wave_local = wavelength_array

        buffer = np.zeros((image_size, image_size), dtype=np.float64)
        image, *_ = process_peaks_parallel(
            dataset_ctx.miller,
            dataset_ctx.intensities,
            image_size,
            local["a"],
            local["c"],
            wave_local,
            buffer,
            local["corto_detector"],
            local["gamma"],
            local["Gamma"],
            local["chi"],
            local.get("psi", 0.0),
            local.get("psi_z", 0.0),
            local["zs"],
            local["zb"],
            local["n2"],
            local_mosaic["beam_x_array"],
            local_mosaic["beam_y_array"],
            local_mosaic["theta_array"],
            local_mosaic["phi_array"],
            local_mosaic["sigma_mosaic_deg"],
            local_mosaic["gamma_mosaic_deg"],
            local_mosaic["eta"],
            wave_local,
            local["debye_x"],
            local["debye_y"],
            local["center"],
            float(theta_value),
            local.get("cor_angle", 0.0),
            uv1,
            uv2,
            save_flag=0,
            collect_hit_tables=False,
            **_simulation_kernel_kwargs(local, local_mosaic),
        )
        cached = np.asarray(image, dtype=np.float64)
        with image_cache_lock:
            existing = image_cache.get(key)
            if existing is not None:
                return existing
            image_cache[key] = cached
        return cached

    def _evaluate_one_dataset(
        item: Tuple[Dict[str, object], MosaicProfileDatasetContext, bool],
    ) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
        local, dataset_ctx, collect_diagnostics = item
        theta_value = _theta_initial_for_dataset(local, dataset_ctx)
        sim_image = _simulate_dataset_image(
            local,
            dataset_ctx,
            theta_value=float(theta_value),
        )
        flat_sim = np.asarray(sim_image, dtype=np.float64).ravel()
        residual_blocks: List[np.ndarray] = []
        roi_diags: List[Dict[str, object]] = []
        specular_bundle: List[Tuple[MosaicProfileROI, np.ndarray, float]] = []

        for roi in dataset_ctx.rois:
            sim_profile = _mosaic_profiles._extract_profile_from_flat_image(flat_sim, roi)
            sim_area = float(np.sum(sim_profile))
            if roi.family == "in_plane":
                sim_shape = (
                    np.asarray(sim_profile, dtype=np.float64) / sim_area
                    if np.isfinite(sim_area) and sim_area > 0.0
                    else np.zeros_like(roi.measured_shape_profile, dtype=np.float64)
                )
                residual_raw = sim_shape - roi.measured_shape_profile
                residual = np.asarray(residual_raw, dtype=np.float64)
                if residual.size:
                    residual = residual / math.sqrt(max(residual.size, 1))
                if profile_weight > 0.0:
                    residual_blocks.append(residual * float(profile_weight))
                if collect_diagnostics:
                    roi_diags.append(
                        {
                            "dataset_index": int(dataset_ctx.dataset_index),
                            "dataset_label": str(dataset_ctx.label),
                            "hkl": tuple(int(v) for v in roi.hkl),
                            "family": str(roi.family),
                            "axis_name": str(roi.axis_name),
                            "center": (float(roi.center_col), float(roi.center_row)),
                            "measured_area": float(roi.measured_area),
                            "simulated_area": float(sim_area),
                            "rms": (
                                float(np.sqrt(np.mean(residual_raw * residual_raw)))
                                if residual_raw.size
                                else 0.0
                            ),
                        }
                    )
            else:
                specular_bundle.append(
                    (roi, np.asarray(sim_profile, dtype=np.float64), float(sim_area))
                )

        shared_specular_scale = float("nan")
        ratio_term_count = 0
        if specular_bundle:
            sim_concat = np.concatenate([entry[1] for entry in specular_bundle]).astype(
                np.float64,
                copy=False,
            )
            meas_concat = np.concatenate(
                [entry[0].measured_profile for entry in specular_bundle]
            ).astype(np.float64, copy=False)
            denom = float(np.dot(sim_concat, sim_concat))
            if np.isfinite(denom) and denom > 1.0e-12:
                shared_specular_scale = max(
                    0.0,
                    float(np.dot(meas_concat, sim_concat) / denom),
                )
            else:
                shared_specular_scale = 0.0

            for roi, sim_profile, sim_area in specular_bundle:
                residual_raw = (
                    float(shared_specular_scale) * np.asarray(sim_profile, dtype=np.float64)
                    - roi.measured_profile
                )
                residual = np.asarray(residual_raw, dtype=np.float64)
                if residual.size:
                    residual = residual / math.sqrt(max(roi.measured_area, 1.0))
                if profile_weight > 0.0:
                    residual_blocks.append(residual * float(profile_weight))
                if collect_diagnostics:
                    roi_diags.append(
                        {
                            "dataset_index": int(dataset_ctx.dataset_index),
                            "dataset_label": str(dataset_ctx.label),
                            "hkl": tuple(int(v) for v in roi.hkl),
                            "family": str(roi.family),
                            "axis_name": str(roi.axis_name),
                            "center": (float(roi.center_col), float(roi.center_row)),
                            "measured_area": float(roi.measured_area),
                            "simulated_area": float(sim_area),
                            "shared_specular_scale": float(shared_specular_scale),
                            "rms": (
                                float(np.sqrt(np.mean(residual_raw * residual_raw)))
                                if residual_raw.size
                                else 0.0
                            ),
                        }
                    )

            if specular_ratio_weight > 0.0 and len(specular_bundle) > 1:
                ref_index = int(
                    np.argmax([float(entry[0].measured_area) for entry in specular_bundle])
                )
                ref_measured_area = max(
                    float(specular_bundle[ref_index][0].measured_area),
                    1.0e-12,
                )
                ref_simulated_area = max(
                    float(specular_bundle[ref_index][2]),
                    1.0e-12,
                )
                ratio_terms: List[float] = []
                for idx, (roi, _sim_profile, sim_area) in enumerate(specular_bundle):
                    if idx == ref_index:
                        continue
                    log_sim = math.log(max(float(sim_area), 1.0e-12) / ref_simulated_area)
                    log_meas = math.log(max(float(roi.measured_area), 1.0e-12) / ref_measured_area)
                    ratio_terms.append(float(log_sim - log_meas))
                ratio_term_count = int(len(ratio_terms))
                if ratio_terms:
                    residual_blocks.append(
                        np.asarray(ratio_terms, dtype=np.float64) * float(specular_ratio_weight)
                    )

        dataset_residual = (
            np.concatenate(residual_blocks) if residual_blocks else np.zeros(0, dtype=np.float64)
        )
        dataset_weight = (
            1.0 / math.sqrt(max(len(dataset_ctx.rois), 1)) if bool(equal_dataset_weights) else 1.0
        )
        dataset_residual = np.asarray(dataset_residual, dtype=np.float64) * dataset_weight

        ordered_roi_diags = sorted(
            roi_diags,
            key=lambda item: float(item["rms"]),
            reverse=True,
        )
        dataset_summary = {
            "dataset_index": int(dataset_ctx.dataset_index),
            "dataset_label": str(dataset_ctx.label),
            "theta_initial_deg": float(theta_value),
            "roi_count": int(len(dataset_ctx.rois)),
            "measured_peak_count": int(dataset_ctx.measured_peak_count),
            "simulated_reflection_count": int(dataset_ctx.miller.shape[0]),
            "dataset_weight": float(dataset_weight),
            "in_plane_roi_count": int(dataset_ctx.in_plane_roi_count),
            "specular_roi_count": int(dataset_ctx.specular_roi_count),
            "profile_weight": float(profile_weight),
            "specular_ratio_weight": float(specular_ratio_weight),
            "shared_specular_scale": (
                float(shared_specular_scale) if np.isfinite(shared_specular_scale) else None
            ),
            "relative_intensity_term_count": int(ratio_term_count),
            "worst_hkls": [tuple(int(v) for v in diag["hkl"]) for diag in ordered_roi_diags[:3]],
        }
        if collect_diagnostics:
            dataset_summary.update(
                {
                    "residual_norm": float(np.linalg.norm(dataset_residual)),
                    "cost": float(_robust_cost(dataset_residual, loss=loss, f_scale=f_scale)),
                    "max_roi_rms": (
                        float(ordered_roi_diags[0]["rms"]) if ordered_roi_diags else 0.0
                    ),
                }
            )
        return dataset_residual, roi_diags, dataset_summary

    def _evaluate_residual(
        theta: np.ndarray,
        *,
        collect_diagnostics: bool = False,
    ) -> Tuple[
        np.ndarray,
        Optional[List[Dict[str, object]]],
        Optional[List[Dict[str, object]]],
        Optional[Dict[int, int]],
    ]:
        local = _apply_trial_params(theta)
        dataset_items = [
            (local, dataset_ctx, bool(collect_diagnostics)) for dataset_ctx in prepared_datasets
        ]
        if dataset_parallel_workers > 1 and len(dataset_items) > 1:
            dataset_results = _threaded_map(
                _evaluate_one_dataset,
                dataset_items,
                max_workers=dataset_parallel_workers,
                numba_threads=numba_threads,
            )
        else:
            dataset_results = [_evaluate_one_dataset(item) for item in dataset_items]
        residual_blocks = [item[0] for item in dataset_results]
        all_roi_diags: List[Dict[str, object]] = []
        dataset_diagnostics: List[Dict[str, object]] = []
        roi_count_by_dataset: Dict[int, int] = {}
        for (_, roi_diags, dataset_summary), dataset_ctx in zip(
            dataset_results,
            prepared_datasets,
        ):
            if collect_diagnostics:
                all_roi_diags.extend(list(roi_diags))
                dataset_diagnostics.append(dict(dataset_summary))
            roi_count_by_dataset[int(dataset_ctx.dataset_index)] = int(len(dataset_ctx.rois))
        residual = (
            np.concatenate(residual_blocks) if residual_blocks else np.zeros(0, dtype=np.float64)
        )
        return (
            np.asarray(residual, dtype=np.float64),
            all_roi_diags if collect_diagnostics else None,
            dataset_diagnostics if collect_diagnostics else None,
            roi_count_by_dataset if collect_diagnostics else None,
        )

    initial_residual, _, _, _ = _evaluate_residual(full_x0)
    initial_cost = _robust_cost(initial_residual, loss=loss, f_scale=f_scale)
    _emit_progress(
        "Initial objective: "
        f"cost={_format_cost(initial_cost)}, residuals={int(np.asarray(initial_residual).size)}, "
        f"rms={_format_rms(initial_residual)}, {_format_params(full_x0)}"
    )

    def _run_solver(
        seed: np.ndarray,
        *,
        attempt_label: str,
        emit_evaluations: bool,
    ) -> Tuple[OptimizeResult, Dict[str, object]]:
        objective_state: Dict[str, object] = {
            "eval_count": 0,
            "best_cost": float("inf"),
        }

        def _objective_active(active_x: Sequence[float]) -> np.ndarray:
            full = _expand_active_vector(active_x)
            residual, _, _, _ = _evaluate_residual(full)
            if emit_evaluations:
                residual_arr = np.asarray(residual, dtype=np.float64)
                objective_state["eval_count"] = int(objective_state["eval_count"]) + 1
                eval_count = int(objective_state["eval_count"])
                cost = float(_robust_cost(residual_arr, loss=loss, f_scale=f_scale))
                best_cost = float(objective_state["best_cost"])
                improvement_margin = max(1.0e-6, 5.0e-3 * max(abs(best_cost), 1.0))
                improved = not np.isfinite(best_cost) or cost < (best_cost - improvement_margin)
                if improved:
                    objective_state["best_cost"] = float(cost)
                if eval_count == 1 or eval_count % 10 == 0 or improved:
                    _emit_progress(
                        f"{attempt_label} eval {eval_count}: "
                        f"cost={_format_cost(cost)}, rms={_format_rms(residual_arr)}, "
                        f"{_format_params(full)}"
                    )
            return np.asarray(residual, dtype=np.float64)

        result = least_squares(
            _objective_active,
            np.asarray(seed, dtype=np.float64),
            bounds=(active_lower, active_upper),
            loss=str(loss),
            f_scale=float(f_scale),
            max_nfev=int(max_nfev),
        )
        meta = {
            "eval_count": int(objective_state["eval_count"]),
            "best_cost": (
                None
                if not np.isfinite(float(objective_state["best_cost"]))
                else float(objective_state["best_cost"])
            ),
        }
        return result, meta

    _emit_progress(f"Primary solve start: max_nfev={int(max_nfev)}, {_format_params(full_x0)}")
    best_result, primary_meta = _run_solver(
        active_x0,
        attempt_label="Primary",
        emit_evaluations=True,
    )
    best_cost = _robust_cost(
        np.asarray(best_result.fun, dtype=np.float64),
        loss=loss,
        f_scale=f_scale,
    )
    _emit_progress(
        "Primary solve done: "
        f"success={bool(best_result.success)}, cost={_format_cost(best_cost)}, "
        f"nfev={_json_safe(getattr(best_result, 'nfev', None))}, "
        f"evals={int(primary_meta.get('eval_count', 0))}, "
        f"message={str(getattr(best_result, 'message', '') or '').strip() or 'n/a'}, "
        f"{_format_params(_expand_active_vector(np.asarray(best_result.x, dtype=np.float64)))}"
    )
    restart_history: List[Dict[str, object]] = []

    if max_restarts > 0 and active_x0.size:
        restart_rng = np.random.default_rng(42)
        span = (active_upper - active_lower) * float(restart_jitter)
        restart_starts = [
            np.clip(
                active_x0
                + restart_rng.uniform(-1.0, 1.0, size=active_x0.size).astype(np.float64) * span,
                active_lower,
                active_upper,
            )
            for _ in range(int(max_restarts))
        ]
        if restart_starts:
            _emit_progress(
                "Restarts scheduled: "
                f"count={len(restart_starts)}, mode="
                f"{'parallel' if restart_parallel_workers > 1 and len(restart_starts) > 1 else 'sequential'}"
            )

        def _solve_restart(
            seed: np.ndarray,
        ) -> Tuple[np.ndarray, OptimizeResult, float, Dict[str, object]]:
            trial, trial_meta = _run_solver(
                seed,
                attempt_label="Restart",
                emit_evaluations=False,
            )
            trial_cost = _robust_cost(
                np.asarray(trial.fun, dtype=np.float64),
                loss=loss,
                f_scale=f_scale,
            )
            return (
                np.asarray(seed, dtype=np.float64),
                trial,
                float(trial_cost),
                dict(trial_meta),
            )

        if restart_parallel_workers > 1 and len(restart_starts) > 1:
            restart_results = _threaded_map(
                _solve_restart,
                restart_starts,
                max_workers=restart_parallel_workers,
                numba_threads=numba_threads,
            )
        else:
            restart_results = []
            for restart_idx, seed in enumerate(restart_starts, start=1):
                _emit_progress(
                    f"Restart {restart_idx}/{len(restart_starts)} start: "
                    f"{_format_params(_expand_active_vector(seed))}"
                )
                trial, trial_meta = _run_solver(
                    seed,
                    attempt_label=f"Restart {restart_idx}/{len(restart_starts)}",
                    emit_evaluations=True,
                )
                trial_cost = _robust_cost(
                    np.asarray(trial.fun, dtype=np.float64),
                    loss=loss,
                    f_scale=f_scale,
                )
                restart_results.append(
                    (
                        np.asarray(seed, dtype=np.float64),
                        trial,
                        float(trial_cost),
                        dict(trial_meta),
                    )
                )

        for restart_idx, (seed, trial, trial_cost, trial_meta) in enumerate(
            restart_results,
            start=1,
        ):
            restart_history.append(
                {
                    "restart": int(restart_idx),
                    "start_x": np.asarray(_expand_active_vector(seed), dtype=np.float64).tolist(),
                    "end_x": np.asarray(_expand_active_vector(trial.x), dtype=np.float64).tolist(),
                    "cost": float(trial_cost),
                    "success": bool(trial.success),
                    "message": str(trial.message),
                }
            )
            _emit_progress(
                f"Restart {restart_idx}/{len(restart_starts)} done: "
                f"success={bool(trial.success)}, cost={_format_cost(trial_cost)}, "
                f"nfev={_json_safe(getattr(trial, 'nfev', None))}, "
                f"evals={int(trial_meta.get('eval_count', 0))}, "
                f"{_format_params(_expand_active_vector(np.asarray(trial.x, dtype=np.float64)))}"
            )
            if float(trial_cost) < float(best_cost):
                _emit_progress(
                    f"Restart {restart_idx}/{len(restart_starts)} is the new best solution."
                )
                best_result = trial
                best_cost = float(trial_cost)

    best_full_x = _expand_active_vector(np.asarray(best_result.x, dtype=np.float64))
    final_residual, roi_diagnostics, dataset_diagnostics, roi_count_by_dataset = _evaluate_residual(
        best_full_x, collect_diagnostics=True
    )
    best_result.x = np.asarray(best_full_x, dtype=np.float64)
    best_result.fun = np.asarray(final_residual, dtype=np.float64)
    final_cost = _robust_cost(best_result.fun, loss=loss, f_scale=f_scale)

    initial_residual_count = int(np.asarray(initial_residual, dtype=np.float64).size)
    final_residual_count = int(np.asarray(best_result.fun, dtype=np.float64).size)
    initial_residual_norm = float(np.linalg.norm(initial_residual))
    final_residual_norm = float(np.linalg.norm(best_result.fun))
    initial_residual_rms = (
        float(np.sqrt(np.mean(np.asarray(initial_residual, dtype=np.float64) ** 2)))
        if initial_residual_count
        else 0.0
    )
    final_residual_rms = (
        float(np.sqrt(np.mean(np.asarray(best_result.fun, dtype=np.float64) ** 2)))
        if final_residual_count
        else 0.0
    )

    best_params = dict(base_params)
    best_params["mosaic_params"] = dict(mosaic_params)
    best_params["mosaic_params"].update(
        {
            "beam_x_array": beam_x,
            "beam_y_array": beam_y,
            "theta_array": theta_array,
            "phi_array": phi_array,
            "wavelength_array": wavelength_array,
            "sigma_mosaic_deg": float(best_result.x[0]),
            "gamma_mosaic_deg": float(best_result.x[1]),
            "eta": float(best_result.x[2]),
        }
    )
    refined_theta_values_by_dataset: Dict[int, float] = {}
    refined_theta_offset: Optional[float] = None
    if optimize_theta:
        theta_slice = np.asarray(best_result.x[3:], dtype=np.float64)
        if resolved_theta_mode == "shared_offset" and theta_slice.size >= 1:
            refined_theta_offset = float(theta_slice[0])
            best_params["theta_offset"] = float(refined_theta_offset)
            for dataset_ctx in prepared_datasets:
                refined_theta_values_by_dataset[int(dataset_ctx.dataset_index)] = float(
                    float(dataset_ctx.theta_initial) + refined_theta_offset
                )
        elif resolved_theta_mode == "single" and theta_slice.size >= 1:
            refined_theta = float(theta_slice[0])
            best_params["theta_initial"] = float(refined_theta)
            refined_theta_values_by_dataset[int(prepared_datasets[0].dataset_index)] = float(
                refined_theta
            )
        elif resolved_theta_mode == "per_dataset" and theta_slice.size:
            theta_map = {
                int(dataset_idx): float(theta_slice[idx])
                for idx, dataset_idx in enumerate(theta_param_dataset_indices)
                if dataset_idx is not None and idx < theta_slice.size
            }
            best_params["_mosaic_theta_initials_by_dataset"] = dict(theta_map)
            refined_theta_values_by_dataset.update(theta_map)

    cost_reduction = 0.0
    if initial_cost > 1.0e-12 and np.isfinite(initial_cost):
        cost_reduction = float((float(initial_cost) - float(final_cost)) / float(initial_cost))
    bound_hits = [
        str(param_names[idx])
        for idx in active_indices.tolist()
        if np.isclose(best_result.x[idx], full_lower[idx], rtol=0.0, atol=1.0e-6)
        or np.isclose(best_result.x[idx], full_upper[idx], rtol=0.0, atol=1.0e-6)
    ]
    boundary_warning = None
    if bound_hits:
        boundary_warning = "Parameters finished on bounds: " + ", ".join(
            str(name) for name in bound_hits
        )

    ordered_roi_diagnostics = sorted(
        list(roi_diagnostics or []),
        key=lambda item: float(item["rms"]),
        reverse=True,
    )
    best_result.best_params = best_params
    best_result.initial_cost = float(initial_cost)
    best_result.final_cost = float(final_cost)
    best_result.cost_reduction = float(cost_reduction)
    best_result.restart_history = restart_history
    best_result.boundary_warning = boundary_warning
    best_result.bound_hits = list(bound_hits)
    best_result.roi_diagnostics = ordered_roi_diagnostics
    best_result.rejected_rois = list(rejected_rois)
    best_result.dataset_diagnostics = list(dataset_diagnostics or [])
    best_result.roi_count_by_dataset = dict(roi_count_by_dataset or {})
    best_result.top_worst_rois = ordered_roi_diagnostics[:10]
    best_result.parallelization_summary = dict(parallelization_summary)
    best_result.roi_half_width = int(roi_half_width)
    best_result.total_roi_count = int(total_rois)
    best_result.solver_loss = str(loss)
    best_result.solver_f_scale = float(f_scale)
    best_result.active_parameters = list(active_parameter_names)
    best_result.fixed_parameters = list(fixed_parameter_names)
    best_result.active_parameter_indices = active_indices.tolist()
    best_result.fixed_parameter_indices = fixed_indices.tolist()
    best_result.fit_sigma_mosaic = bool(fit_sigma_mosaic)
    best_result.fit_gamma_mosaic = bool(fit_gamma_mosaic)
    best_result.fit_eta = bool(fit_eta)
    best_result.fit_theta_i = bool(optimize_theta)
    best_result.theta_refinement_mode = str(resolved_theta_mode) if optimize_theta else None
    best_result.theta_param_name = (
        str(theta_param_names[0]) if optimize_theta and len(theta_param_names) == 1 else None
    )
    best_result.theta_param_names = list(theta_param_names)
    best_result.refined_theta_value = (
        float(best_result.x[3])
        if optimize_theta and len(theta_param_names) == 1 and best_result.x.size >= 4
        else None
    )
    best_result.refined_theta_offset = (
        float(refined_theta_offset) if refined_theta_offset is not None else None
    )
    best_result.refined_theta_values_by_dataset = dict(refined_theta_values_by_dataset)
    best_result.ridge_weight = float(profile_weight)
    best_result.specular_relative_intensity_weight = float(specular_ratio_weight)
    best_result.acceptance_passed = bool(
        float(cost_reduction) >= 0.20
        and not bound_hits
        and total_in_plane > 0
        and (total_specular > 0 or not specular_required)
        and all(
            int(count) >= int(min_per_dataset_rois)
            for count in (roi_count_by_dataset or {}).values()
        )
    )

    parameter_debug: Dict[str, Dict[str, object]] = {}
    for idx, name in enumerate(param_names):
        parameter_debug[str(name)] = {
            "index": int(idx),
            "active": bool(active_mask[idx]),
            "initial": float(full_x0[idx]),
            "final": float(best_result.x[idx]),
            "delta": float(best_result.x[idx] - full_x0[idx]),
            "lower": float(full_lower[idx]),
            "upper": float(full_upper[idx]),
        }

    geometry_parameter_summary = {
        "a": _json_safe(base_params.get("a")),
        "c": _json_safe(base_params.get("c")),
        "lambda": _json_safe(base_params.get("lambda")),
        "psi": _json_safe(base_params.get("psi")),
        "psi_z": _json_safe(base_params.get("psi_z")),
        "zs": _json_safe(base_params.get("zs")),
        "zb": _json_safe(base_params.get("zb")),
        "sample_width_m": _json_safe(base_params.get("sample_width_m")),
        "sample_length_m": _json_safe(base_params.get("sample_length_m")),
        "sample_depth_m": _json_safe(base_params.get("sample_depth_m")),
        "chi": _json_safe(base_params.get("chi")),
        "n2": _json_safe(base_params.get("n2")),
        "center_xy_px": _json_safe(base_params.get("center")),
        "theta_initial_deg": _json_safe(base_params.get("theta_initial")),
        "theta_offset_deg": _json_safe(base_params.get("theta_offset")),
        "corto_detector_m": _json_safe(base_params.get("corto_detector")),
        "gamma_deg": _json_safe(base_params.get("gamma")),
        "Gamma_deg": _json_safe(base_params.get("Gamma")),
        "pixel_size_m": _json_safe(base_params.get("pixel_size_m")),
        "pixel_size": _json_safe(base_params.get("pixel_size")),
        "uv1": _json_safe(uv1),
        "uv2": _json_safe(uv2),
    }
    mosaic_sample_summary = {
        "beam_x": _array_summary(beam_x),
        "beam_y": _array_summary(beam_y),
        "theta": _array_summary(theta_array),
        "phi": _array_summary(phi_array),
        "wavelength": _array_summary(wavelength_array),
        "n2_sample_array": _array_summary(mosaic_params.get("n2_sample_array", [])),
        "solve_q_steps": int(mosaic_params.get("solve_q_steps", 1000)),
        "solve_q_rel_tol": float(mosaic_params.get("solve_q_rel_tol", 5.0e-4)),
        "solve_q_mode": int(mosaic_params.get("solve_q_mode", 1)),
    }
    acceptance_summary = {
        "passed": bool(best_result.acceptance_passed),
        "cost_reduction_threshold": 0.20,
        "cost_reduction": float(cost_reduction),
        "bound_hits": list(bound_hits),
        "boundary_warning": boundary_warning,
        "min_total_rois": int(min_total_rois),
        "min_per_dataset_rois": int(min_per_dataset_rois),
        "total_roi_count": int(total_rois),
        "roi_count_by_dataset": {
            str(key): int(val) for key, val in dict(roi_count_by_dataset or {}).items()
        },
        "total_in_plane_roi_count": int(total_in_plane),
        "total_specular_roi_count": int(total_specular),
        "specular_required": bool(specular_required),
    }
    mosaic_fit_debug_summary = {
        "inputs": {
            "image_size": int(image_size),
            "dataset_count": int(len(prepared_datasets)),
            "miller_count": int(miller.shape[0]),
            "intensity_count": int(intensities.shape[0]),
            "roi_half_width": int(roi_half_width),
            "input_datasets": input_dataset_summaries,
            "prepared_datasets": prepared_dataset_summaries,
            "rejected_rois": _json_safe(rejected_rois),
            "rejected_roi_reason_counts": _json_safe(rejected_roi_reason_counts),
            "rejected_roi_counts_by_dataset": _json_safe(rejected_roi_counts_by_dataset),
            "geometry_parameters": geometry_parameter_summary,
            "mosaic_samples": mosaic_sample_summary,
        },
        "solver": {
            "loss": str(loss),
            "f_scale": float(f_scale),
            "max_nfev": int(max_nfev),
            "max_restarts": int(max_restarts),
            "restart_jitter": float(restart_jitter),
            "progress_callback_enabled": bool(callable(progress_callback)),
            "success": bool(best_result.success),
            "status": _json_safe(getattr(best_result, "status", None)),
            "message": str(best_result.message),
            "nfev": _json_safe(getattr(best_result, "nfev", None)),
            "njev": _json_safe(getattr(best_result, "njev", None)),
            "optimality": _json_safe(getattr(best_result, "optimality", None)),
            "active_parameters": list(active_parameter_names),
            "fixed_parameters": list(fixed_parameter_names),
            "parameter_bounds": parameter_debug,
            "initial_cost": float(initial_cost),
            "final_cost": float(final_cost),
            "cost_reduction": float(cost_reduction),
            "initial_residual_count": int(initial_residual_count),
            "final_residual_count": int(final_residual_count),
            "initial_residual_norm": float(initial_residual_norm),
            "final_residual_norm": float(final_residual_norm),
            "initial_residual_rms": float(initial_residual_rms),
            "final_residual_rms": float(final_residual_rms),
            "restart_history": _json_safe(restart_history),
        },
        "objective_terms": {
            "profile_shape_enabled": bool(profile_weight > 0.0),
            "specular_relative_intensity_enabled": bool(specular_ratio_weight > 0.0),
            "profile_weight": float(profile_weight),
            "specular_ratio_weight": float(specular_ratio_weight),
            "equal_dataset_weights": bool(equal_dataset_weights),
        },
        "theta": {
            "optimize_theta": bool(optimize_theta),
            "requested_mode": str(theta_i_mode),
            "resolved_mode": (str(resolved_theta_mode) if optimize_theta else None),
            "theta_i_bounds_deg": _json_safe(theta_i_bounds_deg),
            "theta_parameter_names": list(theta_param_names),
            "refined_theta_value": _json_safe(best_result.refined_theta_value),
            "refined_theta_offset": _json_safe(best_result.refined_theta_offset),
            "refined_theta_values_by_dataset": _json_safe(
                best_result.refined_theta_values_by_dataset
            ),
        },
        "diagnostics": {
            "parallelization": _json_safe(parallelization_summary),
            "dataset_diagnostics": _json_safe(dataset_diagnostics or []),
            "roi_diagnostics": _json_safe(ordered_roi_diagnostics),
            "top_worst_rois": _json_safe(ordered_roi_diagnostics[:10]),
        },
        "acceptance": acceptance_summary,
    }

    _emit_progress(
        "Final result: "
        f"success={bool(best_result.success)}, accepted={bool(best_result.acceptance_passed)}, "
        f"cost={_format_cost(final_cost)}, reduction={100.0 * float(cost_reduction):.1f}%, "
        f"rms={_format_rms(best_result.fun)}, "
        f"{_format_params(best_result.x)}"
    )
    if bound_hits:
        _emit_progress("Final bound hits: " + ", ".join(str(name) for name in bound_hits))
    for dataset_summary in list(dataset_diagnostics or []):
        label = str(dataset_summary.get("dataset_label", dataset_summary.get("dataset_index", "?")))
        worst_hkls = list(dataset_summary.get("worst_hkls", []) or [])
        worst_text = ", ".join(str(tuple(hkl)) for hkl in worst_hkls[:3]) if worst_hkls else "-"
        _emit_progress(
            f"Final dataset {label}: "
            f"theta={_format_cost(dataset_summary.get('theta_initial_deg'))} deg, "
            f"rois={int(dataset_summary.get('roi_count', 0))}, "
            f"cost={_format_cost(dataset_summary.get('cost'))}, "
            f"max_roi_rms={_format_cost(dataset_summary.get('max_roi_rms'))}, "
            f"worst_hkls={worst_text}"
        )

    best_result.parameter_bounds = parameter_debug
    best_result.initial_parameter_values = {
        str(name): float(full_x0[idx]) for idx, name in enumerate(param_names)
    }
    best_result.final_parameter_values = {
        str(name): float(best_result.x[idx]) for idx, name in enumerate(param_names)
    }
    best_result.initial_residual_count = int(initial_residual_count)
    best_result.final_residual_count = int(final_residual_count)
    best_result.initial_residual_norm = float(initial_residual_norm)
    best_result.final_residual_norm = float(final_residual_norm)
    best_result.initial_residual_rms = float(initial_residual_rms)
    best_result.final_residual_rms = float(final_residual_rms)
    best_result.input_dataset_summaries = list(input_dataset_summaries)
    best_result.prepared_dataset_summaries = list(prepared_dataset_summaries)
    best_result.rejected_roi_reason_counts = dict(rejected_roi_reason_counts)
    best_result.rejected_roi_counts_by_dataset = dict(rejected_roi_counts_by_dataset)
    best_result.geometry_parameter_summary = dict(geometry_parameter_summary)
    best_result.mosaic_sample_summary = dict(mosaic_sample_summary)
    best_result.acceptance_summary = dict(acceptance_summary)
    best_result.mosaic_fit_debug_summary = dict(mosaic_fit_debug_summary)
    return best_result


def fit_mosaic_shape_parameters(
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, object],
    *,
    dataset_specs: Sequence[Dict[str, object]],
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    max_nfev: int = 80,
    max_restarts: int = 2,
    roi_half_width: Optional[int] = None,
    min_total_rois: int = 8,
    min_per_dataset_rois: int = 3,
    equal_dataset_weights: bool = True,
    workers: object = "auto",
    parallel_mode: str = "auto",
    worker_numba_threads: object = 0,
    restart_jitter: float = 0.15,
    ridge_weight: float = 1.0,
    specular_relative_intensity_weight: float = 0.0,
    fit_theta_i: bool = True,
    theta_i_mode: str = "auto",
    theta_i_bounds_deg: Optional[Tuple[float, float]] = None,
    fit_sigma_mosaic: bool = True,
    fit_gamma_mosaic: bool = True,
    fit_eta: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> OptimizeResult:
    """Public entry point for the profile-based mosaic fit."""

    return _fit_mosaic_shape_parameters_profiles(
        miller,
        intensities,
        image_size,
        params,
        dataset_specs=dataset_specs,
        bounds=bounds,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        max_restarts=max_restarts,
        roi_half_width=roi_half_width,
        min_total_rois=min_total_rois,
        min_per_dataset_rois=min_per_dataset_rois,
        equal_dataset_weights=equal_dataset_weights,
        workers=workers,
        parallel_mode=parallel_mode,
        worker_numba_threads=worker_numba_threads,
        restart_jitter=restart_jitter,
        ridge_weight=ridge_weight,
        specular_relative_intensity_weight=specular_relative_intensity_weight,
        fit_theta_i=fit_theta_i,
        theta_i_mode=theta_i_mode,
        theta_i_bounds_deg=theta_i_bounds_deg,
        fit_sigma_mosaic=fit_sigma_mosaic,
        fit_gamma_mosaic=fit_gamma_mosaic,
        fit_eta=fit_eta,
        progress_callback=progress_callback,
    )


def _simulation_kernel_kwargs(
    params: Dict[str, object],
    mosaic: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    mosaic_params = params.get("mosaic_params", {}) if mosaic is None else mosaic
    if not isinstance(mosaic_params, dict):
        mosaic_params = {}

    kwargs: Dict[str, object] = {
        "solve_q_steps": int(mosaic_params.get("solve_q_steps", 1000)),
        "solve_q_rel_tol": float(mosaic_params.get("solve_q_rel_tol", 5.0e-4)),
        "solve_q_mode": int(mosaic_params.get("solve_q_mode", 1)),
        "thickness": float(params.get("sample_depth_m", params.get("thickness", 0.0))),
        "pixel_size_m": float(params.get("pixel_size_m", params.get("pixel_size", 100e-6))),
        "sample_width_m": float(params.get("sample_width_m", 0.0)),
        "sample_length_m": float(params.get("sample_length_m", 0.0)),
    }

    beam_sample_count = int(
        np.asarray(mosaic_params.get("beam_x_array", []), dtype=np.float64).reshape(-1).size
    )
    wavelength_array = mosaic_params.get("wavelength_array")
    if wavelength_array is None:
        raise ValueError("mosaic_params.wavelength_array is required.")
    wavelength_snapshot = _n2_wavelength_snapshot_from_angstrom(wavelength_array)
    if wavelength_snapshot.size != beam_sample_count:
        raise ValueError(
            "mosaic_params.wavelength_array length does not match beam_x_array length."
        )
    source_meta = _normalize_n2_source_meta(mosaic_params.get("_n2_sample_array_source"))
    if source_meta is None or source_meta[0] != "cif_path":
        raise ValueError("CIF provenance is required for n2_sample_array.")

    source_wavelength_snapshot = mosaic_params.get("_n2_sample_array_wavelength_snapshot")
    n2_sample_array = mosaic_params.get("n2_sample_array")
    cached_snapshot = (
        _n2_wavelength_snapshot_from_angstrom(source_wavelength_snapshot)
        if source_wavelength_snapshot is not None
        else None
    )
    if (
        n2_sample_array is not None
        and cached_snapshot is not None
        and cached_snapshot.size == wavelength_snapshot.size
        and np.array_equal(cached_snapshot, wavelength_snapshot, equal_nan=True)
    ):
        n2_override = np.ascontiguousarray(
            np.asarray(n2_sample_array, dtype=np.complex128).reshape(-1),
            dtype=np.complex128,
        )
    else:
        n2_override = np.ascontiguousarray(
            resolve_index_of_refraction_array(
                wavelength_snapshot * 1.0e-10,
                cif_path=str(source_meta[1]),
            ),
            dtype=np.complex128,
        )
        mosaic_params["n2_sample_array"] = n2_override
        mosaic_params["_n2_sample_array_source"] = source_meta
        mosaic_params["_n2_sample_array_wavelength_snapshot"] = wavelength_snapshot.copy()
    if n2_override.size != beam_sample_count:
        raise ValueError("n2_sample_array length does not match beam_x_array length.")
    kwargs["n2_sample_array_override"] = n2_override
    return kwargs








def _detector_pixels_to_fit_space(
    cols: Sequence[float] | np.ndarray,
    rows: Sequence[float] | np.ndarray,
    *,
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert detector pixels into flat-detector ``(2theta_deg, phi_deg)``."""

    cols_arr = np.asarray(cols, dtype=float).reshape(-1)
    rows_arr = np.asarray(rows, dtype=float).reshape(-1)
    two_theta = np.full(cols_arr.shape, np.nan, dtype=np.float64)
    phi = np.full(cols_arr.shape, np.nan, dtype=np.float64)

    if cols_arr.shape != rows_arr.shape:
        return two_theta, phi
    if center is None or len(center) < 2:
        return two_theta, phi
    if not np.isfinite(detector_distance) or detector_distance <= 0.0:
        return two_theta, phi
    if not np.isfinite(pixel_size) or pixel_size <= 0.0:
        return two_theta, phi

    try:
        centre_row = float(center[0])
        centre_col = float(center[1])
    except (TypeError, ValueError, IndexError):
        return two_theta, phi

    x = (cols_arr - centre_col) * float(pixel_size)
    z = (centre_row - rows_arr) * float(pixel_size)
    two_theta[:] = np.degrees(np.arctan2(np.hypot(x, z), float(detector_distance)))
    phi[:] = np.degrees(np.arctan2(x, z))
    phi[:] = (phi + 180.0) % 360.0 - 180.0
    return two_theta, phi


def _detector_anchor_from_entry(
    entry: Mapping[str, object],
    *anchor_keys: tuple[str, str, str],
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Return one finite detector anchor from the requested entry key pairs."""

    for x_key, y_key, reason in anchor_keys:
        try:
            col = float(entry.get(x_key, np.nan))
            row = float(entry.get(y_key, np.nan))
        except Exception:
            continue
        if np.isfinite(col) and np.isfinite(row):
            return (float(col), float(row)), reason
    return None, "missing_detector_anchor"


def _measured_detector_anchor(
    entry: Mapping[str, object],
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Return one measured detector anchor in native/oriented detector pixels."""

    return _detector_anchor_from_entry(
        entry,
        ("native_col", "native_row", "resolved_native_anchor"),
        ("background_detector_x", "background_detector_y", "resolved_background_detector_anchor"),
        ("detector_x", "detector_y", "resolved_detector_anchor"),
        ("x", "y", "resolved_display_anchor"),
    )


def fit_geometry_parameters(
    *,
    params: Mapping[str, object],
    var_names: Sequence[str],
    dataset_specs: Optional[Sequence[Dict[str, object]]] = None,
    caked_geometry_problem: object,
    refinement_config: Optional[Dict[str, Dict[str, float]]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
):
    """Fit the typed exact-caked geometry problem."""

    return solve_caked_geometry_first_rung(
        problem=caked_geometry_problem,
        params=params,
        var_names=var_names,
        dataset_specs=_coerce_sequence_items(dataset_specs),
        refinement_config=refinement_config,
        least_squares_fn=least_squares,
        status_callback=status_callback,
    )
