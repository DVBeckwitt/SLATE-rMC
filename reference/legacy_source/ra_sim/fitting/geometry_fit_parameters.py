"""Internal parameter-vector helpers for geometry fitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ra_sim.fitting._numeric import finite_float_or_none


def safe_float(value: object, fallback: float) -> float:
    parsed = finite_float_or_none(value)
    return float(fallback) if parsed is None else parsed


def set_mosaic_wavelength_array(
    mosaic: dict[str, object],
    wavelength_array: object,
    *,
    key: str = "wavelength_array",
) -> None:
    mosaic[key] = np.ascontiguousarray(
        np.asarray(wavelength_array, dtype=np.float64).reshape(-1),
        dtype=np.float64,
    )
    mosaic.pop("n2_sample_array", None)
    mosaic.pop("_n2_sample_array_wavelength_snapshot", None)


def initial_geometry_fit_value(
    name: str,
    params: Mapping[str, object],
) -> float:
    value = finite_float_or_none(params.get(name))
    if value is None:
        raise ValueError(f"geometry-fit parameter {name!r} must be finite")
    return value


def geometry_fit_pixel_size_m(params: Mapping[str, object]) -> float:
    """Return the required finite positive detector pixel size in metres."""

    value = finite_float_or_none(params.get("pixel_size_m"))
    if value is None or value <= 0.0:
        raise ValueError("geometry parameters require finite positive pixel_size_m")
    return value


@dataclass(frozen=True)
class GeometryFitParameterBounds:
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    bounds_table: list[dict[str, object]]


_LOCAL_BOUND_POLICY: dict[str, dict[str, object]] = {
    "zb": {
        "state_field": "zb",
        "units": "m",
        "mode": "relative",
        "min": -1.0e-4,
        "max": 1.0e-4,
        "source_units": "0.1_mm",
    },
    "zs": {
        "state_field": "zs",
        "units": "m",
        "mode": "relative",
        "min": -1.0e-4,
        "max": 1.0e-4,
        "source_units": "0.1_mm",
    },
    "gamma": {
        "state_field": "gamma",
        "units": "deg",
        "mode": "relative",
        "min": -10.0,
        "max": 10.0,
    },
    "Gamma": {
        "state_field": "Gamma",
        "units": "deg",
        "mode": "relative",
        "min": -10.0,
        "max": 10.0,
    },
    "chi": {
        "state_field": "chi",
        "units": "deg",
        "mode": "relative",
        "min": -5.0,
        "max": 5.0,
    },
    "psi_z": {
        "state_field": "psi_z",
        "units": "deg",
        "mode": "relative",
        "min": -5.0,
        "max": 5.0,
    },
    "cor_angle": {
        "state_field": "cor_angle",
        "units": "deg",
        "mode": "relative",
        "min": -5.0,
        "max": 5.0,
    },
}


def geometry_fit_configured_bounds(
    name: str,
    current_val: float,
    *,
    bounds_cfg: Mapping[str, object],
) -> tuple[float, float]:
    entry = bounds_cfg.get(name)
    if entry is None:
        raise ValueError(f"missing bounds for active geometry-fit variable {name!r}")
    if not isinstance(entry, dict):
        raise TypeError(f"bounds for active geometry-fit variable {name!r} must be a mapping")
    expected_keys = {"mode", "min", "max"}
    if set(entry) != expected_keys:
        raise ValueError(
            f"bounds for active geometry-fit variable {name!r} must contain exactly "
            "mode, min, and max"
        )

    mode = str(entry["mode"]).lower()
    min_raw = float(entry["min"])
    max_raw = float(entry["max"])
    if not (np.isfinite(min_raw) and np.isfinite(max_raw)):
        raise ValueError(f"bounds for active geometry-fit variable {name!r} must be finite")

    if mode == "relative":
        if not np.isfinite(current_val):
            raise ValueError(f"initial geometry-fit variable {name!r} must be finite")
        return current_val + min_raw, current_val + max_raw
    if mode == "absolute":
        return min_raw, max_raw
    raise ValueError(f"unsupported bounds mode {mode!r} for geometry-fit variable {name!r}")


def _geometry_fit_resolved_bounds_inventory_entry(
    name: str,
    current_val: float,
    *,
    bounds_cfg: Mapping[str, object],
) -> dict[str, object]:
    name_s = str(name)
    policy = _LOCAL_BOUND_POLICY.get(name_s)
    if policy is None:
        raise ValueError(f"unknown units for active geometry fit variable '{name_s}'")
    local_lo, local_hi = geometry_fit_configured_bounds(
        name_s,
        float(current_val),
        bounds_cfg={
            name_s: {
                "mode": policy["mode"],
                "min": policy["min"],
                "max": policy["max"],
            }
        },
    )
    if not np.isfinite(local_lo) or not np.isfinite(local_hi):
        raise ValueError(f"nonfinite local bounds for active geometry fit variable '{name_s}'")

    configured_present = name_s in bounds_cfg
    configured_lo = float("nan")
    configured_hi = float("nan")
    configured_finite = False
    if configured_present:
        configured_lo, configured_hi = geometry_fit_configured_bounds(
            name_s,
            float(current_val),
            bounds_cfg=bounds_cfg,
        )
        configured_finite = bool(np.isfinite(configured_lo) and np.isfinite(configured_hi))

    if configured_finite:
        lower = max(float(local_lo), float(configured_lo))
        upper = min(float(local_hi), float(configured_hi))
        local_span = float(local_hi - local_lo)
        configured_span = float(configured_hi - configured_lo)
        if np.isclose(lower, configured_lo) and np.isclose(upper, configured_hi):
            source = "configured" if configured_span <= local_span else "local_policy"
        elif np.isclose(lower, local_lo) and np.isclose(upper, local_hi):
            source = "local_policy"
        else:
            source = "local_policy+configured"
    else:
        lower = float(local_lo)
        upper = float(local_hi)
        source = "local_policy"

    if not np.isfinite(current_val):
        raise ValueError(f"nonfinite initial value for active geometry fit variable '{name_s}'")
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"nonfinite bounds for active geometry fit variable '{name_s}'")
    if not lower < upper:
        raise ValueError(
            f"invalid bounds for active geometry fit variable '{name_s}': "
            f"lower={lower} upper={upper}"
        )

    return {
        "active_name": name_s,
        "state_field": str(policy["state_field"]),
        "units": str(policy["units"]),
        "initial_value": float(current_val),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "scale": float("nan"),
        "bound_source": str(source),
        "local_lower_bound": float(local_lo),
        "local_upper_bound": float(local_hi),
        "configured_lower_bound": float(configured_lo),
        "configured_upper_bound": float(configured_hi),
        "configured_bounds_present": bool(configured_present),
        "configured_bounds_finite": bool(configured_finite),
    }


def build_geometry_fit_parameter_bounds(
    *,
    var_names: Sequence[str],
    x0: Sequence[float],
    bounds_cfg: Mapping[str, object],
) -> GeometryFitParameterBounds:
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    bounds_table: list[dict[str, object]] = []
    for name, value in zip(var_names, x0, strict=True):
        inventory_entry = _geometry_fit_resolved_bounds_inventory_entry(
            str(name),
            float(value),
            bounds_cfg=bounds_cfg,
        )
        lower_bounds.append(float(inventory_entry["lower_bound"]))
        upper_bounds.append(float(inventory_entry["upper_bound"]))
        bounds_table.append(inventory_entry)

    return GeometryFitParameterBounds(
        lower_bounds=np.asarray(lower_bounds, dtype=float),
        upper_bounds=np.asarray(upper_bounds, dtype=float),
        bounds_table=bounds_table,
    )
