"""Active column-vector rotations used by the mosaic orientation measure."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def mosaic_axes(reciprocal_basis_Ainv: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the mean reciprocal axis and deterministic perpendicular tilt axis."""

    mean_axis = np.asarray(reciprocal_basis_Ainv[:, 2], dtype=np.float64)
    mean_axis = mean_axis / np.linalg.norm(mean_axis)
    reference = (
        reciprocal_basis_Ainv[:, 0] - np.dot(reciprocal_basis_Ainv[:, 0], mean_axis) * mean_axis
    )
    reference_norm = np.linalg.norm(reference)
    scale = float(np.linalg.norm(reciprocal_basis_Ainv[:, 0]))
    if reference_norm <= 128.0 * np.finfo(np.float64).eps * scale:
        raise ValueError("first and third reciprocal basis vectors must not be parallel")
    reference = reference / reference_norm
    tilt_axis = np.cross(mean_axis, reference)
    return mean_axis, tilt_axis


def axis_angle_rotation_batch(axis: FloatArray, angle_rad: FloatArray) -> FloatArray:
    """Evaluate Rodrigues' active rotation for every angle around one unit axis."""

    x, y, z = axis
    cross_matrix = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = np.asarray(angle_rad, dtype=np.float64)
    cosine = np.cos(angle)[:, None, None]
    sine = np.sin(angle)[:, None, None]
    outer = np.outer(axis, axis)
    return (
        cosine * np.eye(3)[None, :, :]
        + (1.0 - cosine) * outer[None, :, :]
        + sine * cross_matrix[None, :, :]
    )
