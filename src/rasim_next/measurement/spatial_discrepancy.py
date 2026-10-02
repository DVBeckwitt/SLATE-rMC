"""A declared smooth native-pixel count discrepancy and its exact footprint modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix


@dataclass(frozen=True, slots=True)
class SpatialCountDiscrepancy:
    """Zero-mean Gaussian discrepancy, with fixed pointwise count variance.

    Independent unit-normal mode amplitudes give pixel covariance Phi @ Phi.T.
    Normalized radial Gaussian modes have pointwise standard deviation
    sigma_count_per_pixel. This is a declared model, not calibrated coverage,
    detector noise, a fitted background mean or a change to physical transport.
    basis_width_px is the Gaussian basis width. Finite normalized centers induce
    the nonstationary cosine covariance of their mode vectors, not a stationary
    squared-exponential kernel with that correlation length.
    """

    centers_column_row_px: np.ndarray
    basis_width_px: float
    sigma_count_per_pixel: float

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers_column_row_px)
        if np.iscomplexobj(centers):
            raise ValueError("spatial centers must be finite real (column,row) coordinates")
        centers = np.array(centers, dtype=float, copy=True, order="C")
        if (
            centers.ndim != 2
            or centers.shape[1] != 2
            or not len(centers)
            or np.any(~np.isfinite(centers))
        ):
            raise ValueError("spatial centers must be a nonempty finite (mode,2) array")
        width, sigma = np.asarray(self.basis_width_px), np.asarray(self.sigma_count_per_pixel)
        if (
            width.shape != ()
            or sigma.shape != ()
            or width.dtype.kind not in "fiu"
            or sigma.dtype.kind not in "fiu"
            or not np.isfinite(width)
            or not np.isfinite(sigma)
            or width <= 0
            or sigma < 0
        ):
            raise ValueError(
                "spatial basis width must be positive; count sigma must be nonnegative"
            )
        width, sigma = float(width), float(sigma)
        if not np.isfinite(width * width) or width * width == 0:
            raise ValueError("squared spatial basis width must be finite and positive")
        centers.setflags(write=False)
        object.__setattr__(self, "centers_column_row_px", centers)
        object.__setattr__(self, "basis_width_px", width)
        object.__setattr__(self, "sigma_count_per_pixel", float(sigma))

    def project_modes(
        self,
        ownership: csc_matrix,
        native_flat_pixel_index: np.ndarray,
        detector_shape_rc: tuple[int, int],
        *,
        chunk_size: int = 16384,
    ) -> np.ndarray:
        """Return U = W Phi in counts, using literal native pixel memberships.

        W may contain fractional overlaps or signed contrast weights. Repeated
        pixel indices share the same field. No support normalization or extra
        likelihood rows are introduced. Observation discrepancy covariance is
        U @ U.T; cross-footprint covariance is U1 @ U2.T.
        """
        if (
            len(detector_shape_rc) != 2
            or any(type(n) is not int or n <= 0 for n in detector_shape_rc)
            or type(chunk_size) is not int
            or chunk_size <= 0
        ):
            raise ValueError("detector shape and projection chunk size must be positive integers")
        indices = np.asarray(native_flat_pixel_index)
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= detector_shape_rc[0] * detector_shape_rc[1])
        ):
            raise ValueError("pixel indices must be aligned native flat indices within the panel")
        operator = csc_matrix(ownership, copy=True)
        if (
            operator.shape[1] != len(indices)
            or not operator.shape[0]
            or np.iscomplexobj(operator.data)
            or np.any(~np.isfinite(operator.data))
        ):
            raise ValueError("footprint ownership must be finite real and align with native pixels")
        modes = np.zeros((operator.shape[0], len(self.centers_column_row_px)))
        width = detector_shape_rc[1]
        for first in range(0, len(indices), chunk_size):
            last = first + chunk_size
            pixel = indices[first:last]
            coordinates = np.column_stack((pixel % width, pixel // width))
            offset = coordinates[:, None, :] - self.centers_column_row_px
            distance2 = np.einsum("ijk,ijk->ij", offset, offset)
            if np.any(~np.isfinite(distance2)):
                raise ValueError("spatial distances overflowed the declared pixel coordinate model")
            # Subtracting the closest squared distance cancels in normalization
            # and prevents underflow of every mode at a point far from the grid.
            distance2 -= distance2.min(axis=1, keepdims=True)
            weights = np.exp(-0.5 * distance2 / self.basis_width_px**2)
            weights *= self.sigma_count_per_pixel / np.linalg.norm(weights, axis=1, keepdims=True)
            modes += operator[:, first:last] @ weights
        if np.any(~np.isfinite(modes)):
            raise ValueError("projected discrepancy modes are nonfinite")
        modes.setflags(write=False)
        return modes
