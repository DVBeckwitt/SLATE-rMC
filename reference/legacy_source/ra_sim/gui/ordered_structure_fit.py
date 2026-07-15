"""Ordered-structure fit helpers for detector-space intensity refinement."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares

from .state import OrderedStructureFitSnapshot


@dataclass(frozen=True)
class OrderedStructureParameterSpec:
    """One active nonlinear parameter for ordered-structure fitting."""

    name: str
    value: float
    lower: float
    upper: float


@dataclass
class OrderedStructureMask:
    """Fixed detector-space mask and equalized pixel weights for the fit."""

    pixel_mask: np.ndarray
    weight_map: np.ndarray
    roi_count: int
    bragg_roi_count: int
    specular_roi_count: int
    selected_hkls: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class OrderedStructureFitResult:
    """Summary of one ordered-structure refinement attempt."""

    success: bool
    acceptance_passed: bool
    message: str
    parameter_values: dict[str, float] = field(default_factory=dict)
    scale: float = 0.0
    initial_objective: float = float("nan")
    final_objective: float = float("nan")
    objective_reduction: float = 0.0
    changed_parameter_names: list[str] = field(default_factory=list)


def solve_positive_weighted_scale(
    measured: object,
    primary_model: object,
    *,
    fixed_component: object | None = None,
    weights: object | None = None,
) -> float:
    """Return the analytic non-negative scale minimizing weighted SSE."""

    measured_arr = np.asarray(measured, dtype=np.float64)
    primary_arr = np.asarray(primary_model, dtype=np.float64)
    fixed_arr = (
        np.zeros_like(measured_arr, dtype=np.float64)
        if fixed_component is None
        else np.asarray(fixed_component, dtype=np.float64)
    )
    weight_arr = (
        np.ones_like(measured_arr, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )

    valid = (
        np.isfinite(measured_arr)
        & np.isfinite(primary_arr)
        & np.isfinite(fixed_arr)
        & np.isfinite(weight_arr)
        & (weight_arr > 0.0)
    )
    if not np.any(valid):
        return 0.0

    primary_valid = primary_arr[valid]
    weight_valid = weight_arr[valid]
    target_valid = measured_arr[valid] - fixed_arr[valid]
    denom = float(np.sum(weight_valid * primary_valid * primary_valid))
    if not math.isfinite(denom) or denom <= 0.0:
        return 0.0

    numer = float(np.sum(weight_valid * primary_valid * target_valid))
    if not math.isfinite(numer):
        return 0.0
    return float(max(0.0, numer / denom))


def build_hybrid_ordered_structure_mask(
    *,
    image_shape: Sequence[int],
    primary_hit_tables: Sequence[object],
    max_reflections: int = 24,
    tube_width_scale: float = 1.0,
    specular_width_scale: float = 2.5,
    equal_peak_weights: bool = True,
    base_tube_radius_px: float = 3.0,
) -> OrderedStructureMask:
    """Build the fixed Bragg/specular detector mask for ordered refinement."""

    image_shape_tuple = tuple(int(value) for value in image_shape[:2])
    if len(image_shape_tuple) != 2 or min(image_shape_tuple) <= 0:
        raise ValueError("image_shape must describe a 2D detector image")

    max_count = max(0, int(max_reflections))
    tube_radius = max(0.5, float(base_tube_radius_px) * float(tube_width_scale))
    specular_radius = max(
        tube_radius,
        float(base_tube_radius_px) * float(tube_width_scale) * float(specular_width_scale),
    )

    non_specular: list[tuple[float, tuple[int, int, int], object]] = []
    specular: list[tuple[float, tuple[int, int, int], object]] = []
    for hit_table in primary_hit_tables or ():
        arr = np.asarray(hit_table, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 7:
            continue
        finite_hkl_rows = arr[np.all(np.isfinite(arr[:, 4:7]), axis=1)]
        if finite_hkl_rows.size == 0:
            continue
        hkl = tuple(int(np.rint(value)) for value in finite_hkl_rows[0, 4:7])
        try:
            intensity = float(np.nanmax(arr[:, 0]))
        except ValueError:
            intensity = 0.0
        if not math.isfinite(intensity):
            intensity = 0.0
        intensity = float(max(0.0, intensity))
        if hkl[0] == 0 and hkl[1] == 0:
            specular.append((intensity, hkl, hit_table))
        else:
            non_specular.append((intensity, hkl, hit_table))

    non_specular.sort(key=lambda item: item[0], reverse=True)
    specular.sort(key=lambda item: item[0], reverse=True)
    selected_non_specular = non_specular[:max_count]

    weight_map = np.zeros(image_shape_tuple, dtype=np.float64)
    selected_hkls: list[tuple[int, int, int]] = []
    roi_count = 0

    for entries, radius_px in ((selected_non_specular, tube_radius), (specular, specular_radius)):
        radius = max(0.0, float(radius_px))
        max_offset = int(math.ceil(radius))
        offsets = [
            (delta_row, delta_col)
            for delta_row in range(-max_offset, max_offset + 1)
            for delta_col in range(-max_offset, max_offset + 1)
            if delta_row * delta_row + delta_col * delta_col <= radius * radius
        ] or [(0, 0)]
        for _intensity, hkl, hit_table in entries:
            arr = np.asarray(hit_table, dtype=np.float64)
            roi_mask = np.zeros(image_shape_tuple, dtype=bool)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 3:
                finite_rows = arr[np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])]
                if finite_rows.size:
                    row_max, col_max = image_shape_tuple
                    for row_val, col_val in zip(
                        np.rint(finite_rows[:, 2]).astype(np.int32, copy=False),
                        np.rint(finite_rows[:, 1]).astype(np.int32, copy=False),
                    ):
                        for delta_row, delta_col in offsets:
                            rr = int(row_val + delta_row)
                            cc = int(col_val + delta_col)
                            if 0 <= rr < row_max and 0 <= cc < col_max:
                                roi_mask[rr, cc] = True
            pixel_count = int(np.count_nonzero(roi_mask))
            if pixel_count <= 0:
                continue
            roi_count += 1
            if equal_peak_weights:
                weight_map[roi_mask] += 1.0 / float(pixel_count)
            else:
                weight_map[roi_mask] += 1.0
            selected_hkls.append(tuple(hkl))

    pixel_mask = weight_map > 0.0
    if np.any(pixel_mask):
        mean_weight = float(np.mean(weight_map[pixel_mask]))
        if math.isfinite(mean_weight) and mean_weight > 0.0:
            weight_map[pixel_mask] /= mean_weight

    return OrderedStructureMask(
        pixel_mask=pixel_mask,
        weight_map=weight_map,
        roi_count=int(roi_count),
        bragg_roi_count=int(len(selected_non_specular)),
        specular_roi_count=int(len(specular)),
        selected_hkls=list(selected_hkls),
    )


def capture_ordered_structure_snapshot(
    *,
    occupancy_values: Sequence[object],
    atom_site_values: Sequence[Sequence[object]],
    debye_x: object,
    debye_y: object,
    ordered_scale: object,
) -> OrderedStructureFitSnapshot:
    """Capture one ordered-structure revert snapshot."""

    return OrderedStructureFitSnapshot(
        occupancy_values=[float(value) for value in occupancy_values],
        atom_site_values=[
            (
                float(row[0]),
                float(row[1]),
                float(row[2]),
            )
            for row in atom_site_values
        ],
        debye_x=float(debye_x),
        debye_y=float(debye_y),
        ordered_scale=float(ordered_scale),
    )


def restore_ordered_structure_snapshot(
    snapshot: OrderedStructureFitSnapshot | None,
    *,
    occupancy_vars: Sequence[Any],
    atom_site_vars: Sequence[dict[str, Any]],
    debye_x_var: Any,
    debye_y_var: Any,
    ordered_scale_var: Any,
) -> bool:
    """Restore one snapshot into the provided Tk-like variables."""

    if snapshot is None:
        return False

    for var, value in zip(occupancy_vars, snapshot.occupancy_values):
        setter = getattr(var, "set", None)
        if callable(setter):
            setter(float(value))

    for axis_vars, values in zip(atom_site_vars, snapshot.atom_site_values):
        for axis, value in zip(("x", "y", "z"), values):
            setter = getattr(axis_vars.get(axis), "set", None)
            if callable(setter):
                setter(float(value))

    for var, value in (
        (debye_x_var, snapshot.debye_x),
        (debye_y_var, snapshot.debye_y),
        (ordered_scale_var, snapshot.ordered_scale),
    ):
        setter = getattr(var, "set", None)
        if callable(setter):
            setter(float(value))
    return True


def apply_ordered_structure_values(
    parameter_values: Mapping[str, float],
    *,
    occupancy_vars: Sequence[Any],
    occupancy_param_names: Sequence[str],
    atom_site_vars: Sequence[dict[str, Any]],
    atom_param_names: Sequence[dict[str, str]],
    debye_x_var: Any,
    debye_y_var: Any,
    ordered_scale_var: Any,
    scale_value: float,
) -> None:
    """Apply fitted ordered-structure values to Tk-like runtime variables."""

    for var, name in zip(occupancy_vars, occupancy_param_names):
        if name not in parameter_values:
            continue
        setter = getattr(var, "set", None)
        if callable(setter):
            setter(float(parameter_values[name]))

    for axis_vars, name_map in zip(atom_site_vars, atom_param_names):
        for axis, name in name_map.items():
            if name not in parameter_values:
                continue
            setter = getattr(axis_vars.get(axis), "set", None)
            if callable(setter):
                setter(float(parameter_values[name]))

    if "debye_x" in parameter_values:
        setter = getattr(debye_x_var, "set", None)
        if callable(setter):
            setter(float(parameter_values["debye_x"]))
    if "debye_y" in parameter_values:
        setter = getattr(debye_y_var, "set", None)
        if callable(setter):
            setter(float(parameter_values["debye_y"]))

    scale_setter = getattr(ordered_scale_var, "set", None)
    if callable(scale_setter):
        scale_setter(float(scale_value))


def _mean_pool_2d(image: np.ndarray, factor: int) -> np.ndarray:
    factor_int = max(1, int(factor))
    if factor_int <= 1:
        return np.asarray(image, dtype=np.float64)

    arr = np.asarray(image, dtype=np.float64)
    rows = (arr.shape[0] // factor_int) * factor_int
    cols = (arr.shape[1] // factor_int) * factor_int
    if rows <= 0 or cols <= 0:
        return arr
    trimmed = arr[:rows, :cols]
    reshaped = trimmed.reshape(rows // factor_int, factor_int, cols // factor_int, factor_int)
    return np.mean(reshaped, axis=(1, 3))


def fit_ordered_structure_parameters(
    *,
    measured_image: object,
    mask: OrderedStructureMask,
    parameter_specs: Sequence[OrderedStructureParameterSpec],
    simulate_components: Callable[[dict[str, float]], tuple[np.ndarray, np.ndarray]],
    coarse_downsample_factor: int = 2,
    loss: str = "soft_l1",
    f_scale: float = 2.0,
    coarse_max_nfev: int = 15,
    polish_max_nfev: int = 10,
    restarts: int = 2,
) -> OrderedStructureFitResult:
    """Fit detector-space ordered-structure parameters against one masked image."""

    measured_arr = np.asarray(measured_image, dtype=np.float64)
    if measured_arr.ndim != 2:
        raise ValueError("measured_image must be a 2D detector image")

    mask_pixels = np.asarray(mask.pixel_mask, dtype=bool)
    weight_map = np.asarray(mask.weight_map, dtype=np.float64)
    if mask_pixels.shape != measured_arr.shape or weight_map.shape != measured_arr.shape:
        raise ValueError("mask arrays must match measured_image shape")

    valid_pixels = (
        mask_pixels
        & np.isfinite(measured_arr)
        & np.isfinite(weight_map)
        & (weight_map > 0.0)
    )
    if not np.any(valid_pixels):
        return OrderedStructureFitResult(
            success=False,
            acceptance_passed=False,
            message="Ordered-structure fit unavailable: mask has no valid pixels.",
        )

    specs = list(parameter_specs)
    if not specs:
        return OrderedStructureFitResult(
            success=False,
            acceptance_passed=False,
            message="Ordered-structure fit unavailable: no parameters selected.",
        )

    x0 = np.asarray([float(spec.value) for spec in specs], dtype=np.float64)
    lower = np.asarray([float(spec.lower) for spec in specs], dtype=np.float64)
    upper = np.asarray([float(spec.upper) for spec in specs], dtype=np.float64)
    if (
        x0.shape != lower.shape
        or x0.shape != upper.shape
        or np.any(~np.isfinite(x0))
        or np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or np.any(upper < lower)
    ):
        return OrderedStructureFitResult(
            success=False,
            acceptance_passed=False,
            message="Ordered-structure fit failed: invalid parameter bounds.",
        )

    coarse_factor = max(1, int(coarse_downsample_factor))
    measured_levels = {1: measured_arr, coarse_factor: _mean_pool_2d(measured_arr, coarse_factor)}
    weight_levels = {1: weight_map, coarse_factor: _mean_pool_2d(weight_map, coarse_factor)}

    def _evaluate_trial(x_vec: np.ndarray, *, downsample_factor: int) -> dict[str, object]:
        param_values = {
            spec.name: float(value)
            for spec, value in zip(specs, np.asarray(x_vec, dtype=np.float64))
        }
        primary_image, fixed_image = simulate_components(param_values)
        primary_arr = np.asarray(primary_image, dtype=np.float64)
        fixed_arr = np.asarray(fixed_image, dtype=np.float64)
        if primary_arr.shape != measured_arr.shape or fixed_arr.shape != measured_arr.shape:
            raise ValueError("simulate_components returned images with unexpected shapes")

        factor = max(1, int(downsample_factor))
        primary_level = (
            primary_arr if factor <= 1 else _mean_pool_2d(primary_arr, factor)
        )
        fixed_level = fixed_arr if factor <= 1 else _mean_pool_2d(fixed_arr, factor)
        measured_level = measured_levels[factor]
        weight_level = weight_levels[factor]
        valid_level = (
            np.isfinite(primary_level)
            & np.isfinite(fixed_level)
            & np.isfinite(measured_level)
            & np.isfinite(weight_level)
            & (weight_level > 0.0)
        )
        if not np.any(valid_level):
            raise ValueError("ordered-structure fit produced no valid masked pixels")

        scale_val = solve_positive_weighted_scale(
            measured_level,
            primary_level,
            fixed_component=fixed_level,
            weights=weight_level,
        )
        predicted = scale_val * primary_level + fixed_level
        residual_image = (predicted - measured_level) * np.sqrt(weight_level)
        residual_vector = residual_image[valid_level].reshape(-1)
        objective = 0.5 * float(np.dot(residual_vector, residual_vector))
        return {
            "scale": float(scale_val),
            "objective": float(objective),
            "residual_vector": residual_vector,
        }

    try:
        initial_eval = _evaluate_trial(x0, downsample_factor=1)
    except Exception as exc:
        return OrderedStructureFitResult(
            success=False,
            acceptance_passed=False,
            message=f"Ordered-structure fit failed: {exc}",
        )

    best_x = x0.copy()
    best_scale = float(initial_eval["scale"])
    best_objective = float(initial_eval["objective"])
    best_success = False
    best_message = ""

    rng = np.random.default_rng(0)
    span = np.maximum(upper - lower, 1.0e-12)
    guesses = [x0]
    for _ in range(max(0, int(restarts))):
        jittered = x0 + rng.normal(loc=0.0, scale=0.10, size=x0.shape) * span
        guesses.append(np.clip(jittered, lower, upper))

    for guess in guesses:
        coarse_x = np.asarray(guess, dtype=np.float64)
        attempt_success = False
        attempt_message = ""

        if coarse_factor > 1:
            coarse_result = least_squares(
                lambda x_vec: np.asarray(
                    _evaluate_trial(x_vec, downsample_factor=coarse_factor)["residual_vector"],
                    dtype=np.float64,
                ),
                coarse_x,
                bounds=(lower, upper),
                method="trf",
                loss=str(loss),
                f_scale=float(f_scale),
                max_nfev=max(1, int(coarse_max_nfev)),
            )
            coarse_x = np.asarray(coarse_result.x, dtype=np.float64)
            attempt_success = attempt_success or bool(getattr(coarse_result, "success", False))
            attempt_message = str(getattr(coarse_result, "message", "") or "").strip()

        polish_result = least_squares(
            lambda x_vec: np.asarray(
                _evaluate_trial(x_vec, downsample_factor=1)["residual_vector"],
                dtype=np.float64,
            ),
            coarse_x,
            bounds=(lower, upper),
            method="trf",
            loss=str(loss),
            f_scale=float(f_scale),
            max_nfev=max(1, int(polish_max_nfev)),
        )
        attempt_x = np.asarray(polish_result.x, dtype=np.float64)
        attempt_success = attempt_success or bool(getattr(polish_result, "success", False))
        polish_message = str(getattr(polish_result, "message", "") or "").strip()
        if polish_message:
            attempt_message = polish_message

        try:
            attempt_eval = _evaluate_trial(attempt_x, downsample_factor=1)
        except Exception:
            continue

        attempt_objective = float(attempt_eval["objective"])
        if attempt_objective < best_objective and np.all(np.isfinite(attempt_x)):
            best_x = attempt_x.copy()
            best_scale = float(attempt_eval["scale"])
            best_objective = attempt_objective
            best_success = bool(attempt_success)
            best_message = attempt_message

    objective_reduction = 0.0
    if initial_eval["objective"] > 0.0:
        objective_reduction = max(
            0.0,
            float(initial_eval["objective"] - best_objective) / float(initial_eval["objective"]),
        )

    parameter_values = {
        spec.name: float(value)
        for spec, value in zip(specs, best_x)
    }
    changed_names = [
        spec.name
        for spec, start_value, end_value in zip(specs, x0, best_x)
        if not math.isclose(float(start_value), float(end_value), rel_tol=1.0e-9, abs_tol=1.0e-9)
    ]
    acceptance_passed = bool(
        np.all(np.isfinite(best_x))
        and math.isfinite(best_scale)
        and best_scale >= 0.0
        and best_objective < float(initial_eval["objective"])
    )
    if not acceptance_passed and not best_message:
        best_message = "objective did not improve"

    return OrderedStructureFitResult(
        success=bool(best_success),
        acceptance_passed=acceptance_passed,
        message=str(best_message),
        parameter_values=parameter_values,
        scale=float(best_scale),
        initial_objective=float(initial_eval["objective"]),
        final_objective=float(best_objective),
        objective_reduction=float(objective_reduction),
        changed_parameter_names=changed_names,
    )
