"""Local count-space elliptical Gaussian proposals; never a geometric calibration."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


@dataclass(frozen=True, slots=True)
class GaussianSpotProposal:
    # Background, positive amplitude, native column/row, widths in px, orientation in rad.
    values: tuple[float, ...]
    center_standard_error_px: tuple[float, float] | None
    residual_rms_count: float
    valid_fraction: float
    jacobian_rank: int
    condition: float
    reliable_initial_estimate: bool
    limitations: tuple[str, ...]
    model: NDArray[np.float64]
    residual: NDArray[np.float64]


def gaussian_spot_model(values, column_px, row_px):
    background, amplitude, column, row, sigma1, sigma2, angle = values
    dc, dr = column_px - column, row_px - row
    x = np.cos(angle) * dc + np.sin(angle) * dr
    y = -np.sin(angle) * dc + np.cos(angle) * dr
    return background + amplitude * np.exp(-0.5 * ((x / sigma1) ** 2 + (y / sigma2) ** 2))


def estimate_gaussian_spot(
    counts,
    *,
    roi_column_row_bounds,
    inclusion_mask=None,
    canceled: Callable[[], bool] | None = None,
):
    """Unweighted least squares of original valid ROI values, with a constant background."""
    image = np.asarray(counts)
    c0, c1, r0, r1 = roi_column_row_bounds
    if (
        image.ndim != 2
        or any(type(v) is not int for v in (c0, c1, r0, r1))
        or not 0 <= c0 < c1 <= image.shape[1]
        or not 0 <= r0 < r1 <= image.shape[0]
        or (c1 - c0) * (r1 - r0) > 65536
        or min(c1 - c0, r1 - r0) < 5
    ):
        raise ValueError(
            "spot ROI must be native column/row bounds, at least 5x5 and at most 65536 pixels"
        )
    data = np.array(image[r0:r1, c0:c1], dtype=np.float64, copy=True)
    valid = np.isfinite(data)
    if inclusion_mask is not None:
        mask = np.asarray(inclusion_mask)
        if mask.dtype != np.bool_ or mask.shape != image.shape:
            raise ValueError("spot mask must be aligned Boolean native support")
        valid &= mask[r0:r1, c0:c1]
    if np.count_nonzero(valid) < 30:
        raise ValueError("spot proposal unavailable: fewer than 30 valid ROI pixels")
    row, column = np.mgrid[r0:r1, c0:c1]
    x, y, z = column[valid], row[valid], data[valid]
    background = float(np.quantile(z, 0.2))
    signal = np.maximum(z - background, 0)
    total = float(signal.sum())
    amplitude = float(z.max() - background)
    if total <= 0 or amplitude <= 1e-12 * max(1, float(np.max(np.abs(z)))):
        raise ValueError("spot proposal unavailable: no positive contrast")
    cc = float(x @ signal / total)
    rr = float(y @ signal / total)
    sx = max(0.5, float(np.sqrt((x - cc) ** 2 @ signal / total)))
    sy = max(0.5, float(np.sqrt((y - rr) ** 2 @ signal / total)))
    lower = (-np.inf, np.finfo(float).tiny, c0, r0, 0.25, 0.25, -np.pi / 2)
    upper = (np.inf, np.inf, c1 - 1, r1 - 1, 2 * (c1 - c0), 2 * (r1 - r0), np.pi / 2)
    initial = [background, amplitude, np.clip(cc, c0, c1 - 1), np.clip(rr, r0, r1 - 1), sx, sy, 0]

    def residual(values):
        if canceled is not None and canceled():
            raise RuntimeError("spot proposal canceled")
        return gaussian_spot_model(values, x, y) - z

    fit = least_squares(residual, initial, bounds=(lower, upper), x_scale="jac", max_nfev=300)
    raw = residual(fit.x)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * max(fit.jac.shape) * np.finfo(float).eps))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    uncertainty = None
    limitations = []
    if rank == 7 and condition < 1e8:
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * float(raw @ raw) / max(len(raw) - 7, 1)
        errors = np.sqrt(np.maximum(np.diag(covariance), 0))
        uncertainty = (float(errors[2]), float(errors[3]))
    else:
        limitations.append("center uncertainty unavailable: rank/conditioning")
    fraction = float(np.mean(valid))
    c, r, w1, w2 = fit.x[2:6]
    edge = min(c - c0, c1 - 1 - c, r - r0, r1 - 1 - r)
    if edge < 2 * max(w1, w2):
        limitations.append("ROI may clip the spot; tails/overlap are not qualified")
    if fraction < 0.8:
        limitations.append("more than 20 percent invalid or excluded ROI support")
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    bound = np.any(
        np.isclose(fit.x[finite_lower], np.asarray(lower)[finite_lower], atol=1e-6, rtol=0)
    ) or np.any(np.isclose(fit.x[finite_upper], np.asarray(upper)[finite_upper], atol=1e-6, rtol=0))
    if bound:
        limitations.append("Gaussian parameter touches a bound")
    if not fit.success:
        limitations.append("Gaussian optimizer did not terminate successfully")
    model = gaussian_spot_model(fit.x, column, row)
    difference = data - model
    difference[~valid] = np.nan
    model.setflags(write=False)
    difference.setflags(write=False)
    return GaussianSpotProposal(
        tuple(float(v) for v in fit.x),
        uncertainty,
        float(np.sqrt(np.mean(raw**2))),
        fraction,
        rank,
        condition,
        not limitations,
        tuple(limitations),
        model,
        difference,
    )
