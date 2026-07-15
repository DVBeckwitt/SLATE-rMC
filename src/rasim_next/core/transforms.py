"""Rigid transforms for column-vector, active-rotation geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.frames import FrameId

FloatArray = NDArray[np.float64]


def _readonly_float_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """A named ``target_from_source`` rigid transform in metres."""

    rotation: FloatArray
    translation_m: FloatArray
    source_frame: FrameId
    target_frame: FrameId

    def __post_init__(self) -> None:
        rotation = _readonly_float_array(self.rotation, (3, 3), "rotation")
        translation = _readonly_float_array(self.translation_m, (3,), "translation_m")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12):
            raise ValueError("rotation must be orthogonal")
        if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("rotation determinant must be +1")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "source_frame", FrameId(self.source_frame))
        object.__setattr__(self, "target_frame", FrameId(self.target_frame))

    @classmethod
    def identity(cls, frame: FrameId) -> RigidTransform:
        return cls(np.eye(3), np.zeros(3), frame, frame)

    @classmethod
    def around_pivot(
        cls,
        *,
        rotation: ArrayLike,
        pivot_m: ArrayLike,
        frame: FrameId,
    ) -> RigidTransform:
        rotation_array = _readonly_float_array(rotation, (3, 3), "rotation")
        pivot = _readonly_float_array(pivot_m, (3,), "pivot_m")
        translation = pivot - rotation_array @ pivot
        return cls(rotation_array, translation, frame, frame)

    def apply_point(self, point_source_m: ArrayLike) -> FloatArray:
        points = np.asarray(point_source_m, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError("point_source_m must end with a length-3 coordinate axis")
        return np.asarray(points @ self.rotation.T + self.translation_m, dtype=np.float64)

    def apply_vector(self, vector_source: ArrayLike) -> FloatArray:
        vectors = np.asarray(vector_source, dtype=np.float64)
        if vectors.shape[-1:] != (3,):
            raise ValueError("vector_source must end with a length-3 coordinate axis")
        return np.asarray(vectors @ self.rotation.T, dtype=np.float64)

    def inverse(self) -> RigidTransform:
        inverse_rotation = self.rotation.T
        return RigidTransform(
            inverse_rotation,
            -(inverse_rotation @ self.translation_m),
            self.target_frame,
            self.source_frame,
        )

    def compose(self, source_transform: RigidTransform) -> RigidTransform:
        """Return ``self @ source_transform`` with frame continuity checked."""

        if self.source_frame != source_transform.target_frame:
            raise ValueError(
                "frame mismatch: "
                f"{source_transform.target_frame} cannot feed {self.source_frame}"
            )
        return RigidTransform(
            self.rotation @ source_transform.rotation,
            self.rotation @ source_transform.translation_m + self.translation_m,
            source_transform.source_frame,
            self.target_frame,
        )
