"""The single OSC-raw to detector-native orientation boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class OscRawIndex:
    row: int
    column: int

    def __post_init__(self) -> None:
        if type(self.row) is not int or type(self.column) is not int or min(self.row, self.column) < 0:
            raise ValueError("OSC raw indices must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class DetectorIndex:
    row: int
    column: int

    def __post_init__(self) -> None:
        if type(self.row) is not int or type(self.column) is not int or min(self.row, self.column) < 0:
            raise ValueError("detector indices must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class DetectorCoordinate:
    column_px: float
    row_px: float

    def __post_init__(self) -> None:
        column = float(self.column_px)
        row = float(self.row_px)
        if not math.isfinite(column) or not math.isfinite(row):
            raise ValueError("detector coordinates must be finite")
        object.__setattr__(self, "column_px", column)
        object.__setattr__(self, "row_px", row)


def _raw_shape(raw_shape: tuple[int, int]) -> tuple[int, int]:
    if len(raw_shape) != 2 or any(type(size) is not int or size <= 0 for size in raw_shape):
        raise ValueError("raw_shape must contain two positive integers")
    return raw_shape


def raw_to_detector_index(
    raw_index: OscRawIndex, raw_shape: tuple[int, int]
) -> DetectorIndex:
    if not isinstance(raw_index, OscRawIndex):
        raise TypeError("raw_index must be an OscRawIndex")
    height, width = _raw_shape(raw_shape)
    if not (0 <= raw_index.row < height and 0 <= raw_index.column < width):
        raise ValueError(f"raw index {raw_index} is outside raw shape {raw_shape}")
    return DetectorIndex(row=raw_index.column, column=height - 1 - raw_index.row)


def detector_to_raw_index(
    detector_index: DetectorIndex, raw_shape: tuple[int, int]
) -> OscRawIndex:
    if not isinstance(detector_index, DetectorIndex):
        raise TypeError("detector_index must be a DetectorIndex")
    height, width = _raw_shape(raw_shape)
    if not (0 <= detector_index.row < width and 0 <= detector_index.column < height):
        raise ValueError(f"detector index {detector_index} is outside native shape {(width, height)}")
    return OscRawIndex(row=height - 1 - detector_index.column, column=detector_index.row)


def index_to_coordinate(detector_index: DetectorIndex) -> DetectorCoordinate:
    if not isinstance(detector_index, DetectorIndex):
        raise TypeError("detector_index must be a DetectorIndex")
    return DetectorCoordinate(
        column_px=float(detector_index.column), row_px=float(detector_index.row)
    )


def raw_to_detector_native(raw_array: NDArray[np.generic]) -> NDArray[np.generic]:
    array = np.asarray(raw_array)
    if array.ndim != 2:
        raise ValueError(f"raw_array must be two-dimensional, got {array.ndim} dimensions")
    return np.ascontiguousarray(np.rot90(array, -1))


def detector_native_to_raw(detector_native_array: NDArray[np.generic]) -> NDArray[np.generic]:
    array = np.asarray(detector_native_array)
    if array.ndim != 2:
        raise ValueError(
            f"detector_native_array must be two-dimensional, got {array.ndim} dimensions"
        )
    return np.ascontiguousarray(np.rot90(array, 1))
