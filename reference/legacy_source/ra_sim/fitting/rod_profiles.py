"""Support-normalized rod/profile integration helpers."""

from __future__ import annotations

import numpy as np


def _same_shape_optional(
    name: str,
    value: object | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} shape must match image.shape.")
    return array


def _fill_nan(size: int) -> np.ndarray:
    return np.full(size, np.nan, dtype=np.float64)


def _curve_arrays(
    *,
    x_name: str,
    x_values: object,
    y_name: str,
    y_values: object,
) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y_array = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if x_array.size != y_array.size:
        raise ValueError(f"{x_name} and {y_name} must have the same length.")
    return x_array, y_array


def _unit_scaled_curve(model_x_values: np.ndarray, model_y_values: np.ndarray) -> dict[str, object]:
    return {
        "x": model_x_values.copy(),
        "y": model_y_values.copy(),
        "scale": 1.0,
        "overlap_count": 0,
    }


def _nan_to_zero(values: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(values), 0.0, values)


def _bincount_float(
    bins: np.ndarray,
    values: np.ndarray,
    bin_count: int,
) -> np.ndarray:
    if bins.size <= 0:
        return np.zeros(bin_count, dtype=np.float64)
    return np.bincount(
        bins.astype(np.intp, copy=False),
        weights=np.asarray(values, dtype=np.float64),
        minlength=bin_count,
    )[:bin_count].astype(np.float64, copy=False)


def _store_weighted_density(
    *,
    acc_all: np.ndarray,
    background_all: np.ndarray,
    model_all: np.ndarray | None,
    has_support: np.ndarray,
    acceptance_sum: np.ndarray,
    background_weighted_sum: np.ndarray,
    background_density: np.ndarray,
    fit_weighted_sum: np.ndarray,
    fit_density: np.ndarray,
) -> None:
    acceptance_sum[has_support] = acc_all[has_support]
    background_weighted_sum[has_support] = background_all[has_support]
    positive_acceptance = has_support & (acc_all > 0.0)
    background_density[positive_acceptance] = (
        background_all[positive_acceptance] / acc_all[positive_acceptance]
    )
    if model_all is not None:
        fit_weighted_sum[has_support] = model_all[has_support]
        fit_density[positive_acceptance] = (
            model_all[positive_acceptance] / acc_all[positive_acceptance]
        )


def binned_caked_mask_profile(
    *,
    image: object,
    coord_map: object,
    coord_edges: object,
    mask: object,
    model: object | None = None,
    signal_sum: object | None = None,
    normalization_sum: object | None = None,
    acceptance: object | None = None,
    theta_map: object | None = None,
    coord_name: str = "qz",
) -> dict[str, np.ndarray]:
    """Bin a caked mask profile and return support-normalized intensity densities."""

    image_array = np.asarray(image, dtype=np.float64)
    if image_array.ndim != 2:
        raise ValueError("image must be a 2D array.")
    shape = image_array.shape
    coord_array = _same_shape_optional("coord_map", coord_map, shape)
    if coord_array is None:
        raise ValueError("coord_map is required.")
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != shape:
        raise ValueError("mask shape must match image.shape.")
    model_array = _same_shape_optional("model", model, shape)
    signal_array = _same_shape_optional("signal_sum", signal_sum, shape)
    normalization_array = _same_shape_optional("normalization_sum", normalization_sum, shape)
    acceptance_array = _same_shape_optional("acceptance", acceptance, shape)
    theta_array = _same_shape_optional("theta_map", theta_map, shape)
    edges = np.asarray(coord_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError(
            "coord_edges must be a strictly increasing 1D array with at least 2 values."
        )

    if (signal_array is None) != (normalization_array is None):
        raise ValueError("signal_sum and normalization_sum must be supplied together.")
    source = "sum_normalization" if signal_array is not None else "pixel_count"
    if source == "pixel_count" and acceptance_array is not None:
        source = "acceptance"
    bin_count = int(edges.size - 1)
    pixel_count = np.zeros(bin_count, dtype=np.int64)
    acceptance_sum = _fill_nan(bin_count)
    background_sum = _fill_nan(bin_count)
    fit_sum = _fill_nan(bin_count)
    background_mean = _fill_nan(bin_count)
    fit_mean = _fill_nan(bin_count)
    background_weighted_sum = _fill_nan(bin_count)
    fit_weighted_sum = _fill_nan(bin_count)
    background_density = _fill_nan(bin_count)
    fit_density = _fill_nan(bin_count)
    two_theta_min = _fill_nan(bin_count)
    two_theta_max = _fill_nan(bin_count)
    two_theta_mean = _fill_nan(bin_count)

    coord_flat = coord_array.reshape(-1)
    mask_flat = mask_array.reshape(-1)
    bin_index = np.searchsorted(edges, coord_flat, side="right") - 1
    finite_coord = np.isfinite(coord_flat)
    last_edge = finite_coord & (coord_flat == edges[-1])
    if np.any(last_edge):
        bin_index[last_edge] = bin_count - 1
    support = (
        mask_flat
        & finite_coord
        & (bin_index >= 0)
        & (bin_index < bin_count)
    )
    support_bins = bin_index[support].astype(np.intp, copy=False)
    if support_bins.size > 0:
        pixel_count = np.bincount(support_bins, minlength=bin_count)[:bin_count].astype(
            np.int64,
            copy=False,
        )
    has_support = pixel_count > 0

    image_flat = image_array.reshape(-1)
    background_sum_all = _bincount_float(
        support_bins,
        _nan_to_zero(image_flat[support]),
        bin_count,
    )
    background_sum[has_support] = background_sum_all[has_support]
    background_mean[has_support] = (
        background_sum_all[has_support] / pixel_count[has_support].astype(np.float64)
    )

    if model_array is not None:
        model_flat = model_array.reshape(-1)
        fit_sum_all = _bincount_float(
            support_bins,
            _nan_to_zero(model_flat[support]),
            bin_count,
        )
        fit_sum[has_support] = fit_sum_all[has_support]
        fit_mean[has_support] = (
            fit_sum_all[has_support] / pixel_count[has_support].astype(np.float64)
        )
    else:
        model_flat = None

    if theta_array is not None:
        theta_flat = theta_array.reshape(-1)
        theta_support = support & np.isfinite(theta_flat)
        theta_bins = bin_index[theta_support].astype(np.intp, copy=False)
        theta_values = theta_flat[theta_support]
        if theta_bins.size > 0:
            theta_min = np.full(bin_count, np.inf, dtype=np.float64)
            theta_max = np.full(bin_count, -np.inf, dtype=np.float64)
            np.minimum.at(theta_min, theta_bins, theta_values)
            np.maximum.at(theta_max, theta_bins, theta_values)
            theta_count = np.bincount(theta_bins, minlength=bin_count)[:bin_count]
            theta_sum = _bincount_float(theta_bins, theta_values, bin_count)
            valid_theta = theta_count > 0
            two_theta_min[valid_theta] = theta_min[valid_theta]
            two_theta_max[valid_theta] = theta_max[valid_theta]
            two_theta_mean[valid_theta] = theta_sum[valid_theta] / theta_count[
                valid_theta
            ].astype(np.float64)

    if source == "sum_normalization":
        signal_flat = signal_array.reshape(-1)
        normalization_flat = normalization_array.reshape(-1)
        weighted = (
            support
            & np.isfinite(signal_flat)
            & np.isfinite(normalization_flat)
            & (normalization_flat > 0.0)
        )
        weighted_bins = bin_index[weighted].astype(np.intp, copy=False)
        acc_all = _bincount_float(weighted_bins, normalization_flat[weighted], bin_count)
        sig_all = _bincount_float(weighted_bins, signal_flat[weighted], bin_count)
        fit_sig_all = None
        if model_flat is not None:
            fit_sig_all = _bincount_float(
                weighted_bins,
                _nan_to_zero(model_flat[weighted] * normalization_flat[weighted]),
                bin_count,
            )
        _store_weighted_density(
            acc_all=acc_all,
            background_all=sig_all,
            model_all=fit_sig_all,
            has_support=has_support,
            acceptance_sum=acceptance_sum,
            background_weighted_sum=background_weighted_sum,
            background_density=background_density,
            fit_weighted_sum=fit_weighted_sum,
            fit_density=fit_density,
        )
    elif source == "acceptance":
        acceptance_flat = acceptance_array.reshape(-1)
        weighted = support & np.isfinite(acceptance_flat) & (acceptance_flat > 0.0)
        weighted_bins = bin_index[weighted].astype(np.intp, copy=False)
        acc_all = _bincount_float(weighted_bins, acceptance_flat[weighted], bin_count)
        background_sig_all = _bincount_float(
            weighted_bins,
            _nan_to_zero(image_flat[weighted] * acceptance_flat[weighted]),
            bin_count,
        )
        fit_sig_all = None
        if model_flat is not None:
            fit_sig_all = _bincount_float(
                weighted_bins,
                _nan_to_zero(model_flat[weighted] * acceptance_flat[weighted]),
                bin_count,
            )
        _store_weighted_density(
            acc_all=acc_all,
            background_all=background_sig_all,
            model_all=fit_sig_all,
            has_support=has_support,
            acceptance_sum=acceptance_sum,
            background_weighted_sum=background_weighted_sum,
            background_density=background_density,
            fit_weighted_sum=fit_weighted_sum,
            fit_density=fit_density,
        )
    else:
        acceptance_sum[~has_support] = 0.0
        acceptance_sum[has_support] = pixel_count[has_support].astype(np.float64)
        background_weighted_sum[has_support] = background_sum_all[has_support]
        background_density[has_support] = (
            background_sum_all[has_support] / pixel_count[has_support].astype(np.float64)
        )
        if model_flat is not None:
            fit_weighted_sum[has_support] = fit_sum_all[has_support]
            fit_density[has_support] = (
                fit_sum_all[has_support] / pixel_count[has_support].astype(np.float64)
            )

    result: dict[str, np.ndarray] = {
        f"{coord_name}_bin": np.arange(1, bin_count + 1, dtype=np.int64),
        f"{coord_name}_min": edges[:-1].copy(),
        f"{coord_name}_max": edges[1:].copy(),
        f"{coord_name}_center": 0.5 * (edges[:-1] + edges[1:]),
        "pixel_count": pixel_count,
        "acceptance_sum": acceptance_sum,
        "acceptance_source": np.full(bin_count, source, dtype=object),
        "background_sum": background_sum,
        "fit_sum": fit_sum,
        "background_mean": background_mean,
        "fit_mean": fit_mean,
        "background_weighted_sum": background_weighted_sum,
        "fit_weighted_sum": fit_weighted_sum,
        "background_density": background_density,
        "fit_density": fit_density,
    }
    if theta_array is not None:
        result.update(
            {
                "two_theta_min": two_theta_min,
                "two_theta_max": two_theta_max,
                "two_theta_mean": two_theta_mean,
            }
        )
    return result


def finite_profile_curve(
    profile: object,
    *,
    coord_name: str = "qz",
    density_key: str = "background_density",
) -> dict[str, np.ndarray | str]:
    """Return finite coordinate/density samples from a profile payload."""

    coord_key = f"{coord_name}_center"
    x_values, y_values = _curve_arrays(
        x_name=coord_key,
        x_values=profile[coord_key],  # type: ignore[index]
        y_name=density_key,
        y_values=profile[density_key],  # type: ignore[index]
    )
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    return {
        "x": x_values[finite].copy(),
        "y": y_values[finite].copy(),
        "coord_key": coord_key,
        "density_key": density_key,
    }


def normalize_model_curve_to_measured(
    *,
    model_x: object,
    model_y: object,
    measured_x: object,
    measured_y: object,
) -> dict[str, np.ndarray | float | int]:
    """Scale a model curve to measured samples over their overlapping coordinate range."""

    model_x_values, model_y_values = _curve_arrays(
        x_name="model_x",
        x_values=model_x,
        y_name="model_y",
        y_values=model_y,
    )
    measured_x_values, measured_y_values = _curve_arrays(
        x_name="measured_x",
        x_values=measured_x,
        y_name="measured_y",
        y_values=measured_y,
    )

    finite_model = np.isfinite(model_x_values) & np.isfinite(model_y_values)
    if not np.any(finite_model):
        return _unit_scaled_curve(model_x_values, model_y_values)
    order = np.argsort(model_x_values[finite_model])
    interp_x = model_x_values[finite_model][order]
    interp_y = model_y_values[finite_model][order]
    unique_x, inverse = np.unique(interp_x, return_inverse=True)
    if unique_x.size != interp_x.size:
        interp_y = np.bincount(inverse, weights=interp_y, minlength=unique_x.size) / np.bincount(
            inverse, minlength=unique_x.size
        ).astype(np.float64)
        interp_x = unique_x

    measured_finite = np.isfinite(measured_x_values) & np.isfinite(measured_y_values)
    overlap = (
        measured_finite
        & (measured_x_values >= float(interp_x[0]))
        & (measured_x_values <= float(interp_x[-1]))
    )
    if not np.any(overlap):
        return _unit_scaled_curve(model_x_values, model_y_values)

    sampled_model = np.interp(measured_x_values[overlap], interp_x, interp_y)
    measured_overlap = measured_y_values[overlap]
    valid = np.isfinite(sampled_model) & np.isfinite(measured_overlap)
    if not np.any(valid):
        return _unit_scaled_curve(model_x_values, model_y_values)

    sampled_model = sampled_model[valid]
    measured_overlap = measured_overlap[valid]
    denominator = float(np.sum(sampled_model * sampled_model))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return _unit_scaled_curve(model_x_values, model_y_values)

    scale = float(np.sum(measured_overlap * sampled_model) / denominator)
    if not np.isfinite(scale):
        scale = 1.0
        overlap_count = 0
    else:
        overlap_count = int(sampled_model.size)
    return {
        "x": model_x_values.copy(),
        "y": model_y_values * scale,
        "scale": scale,
        "overlap_count": overlap_count,
    }
