"""Conservative bilinear deposition in native detector coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import DetectorHitBatch


@dataclass(frozen=True, slots=True)
class DepositionResult:
    """Native image mass and the deposited/clipped conservation ledger."""

    image_A2: NDArray[np.float64]
    deposited_mass_A2: float
    clipped_mass_A2: float

    def __post_init__(self) -> None:
        supplied = np.asarray(self.image_A2)
        if np.iscomplexobj(supplied):
            raise ValueError("image_A2 must be real")
        image = np.asarray(supplied, dtype=np.float64, order="C")
        deposited = float(self.deposited_mass_A2)
        clipped = float(self.clipped_mass_A2)
        if (
            image.ndim != 2
            or not np.all(np.isfinite(image))
            or np.any(image < 0.0)
            or not np.isfinite(deposited)
            or deposited < 0.0
            or not np.isfinite(clipped)
            or clipped < 0.0
        ):
            raise ValueError("deposition outputs must be finite and nonnegative")
        image.setflags(write=False)
        object.__setattr__(self, "image_A2", image)
        object.__setattr__(self, "deposited_mass_A2", deposited)
        object.__setattr__(self, "clipped_mass_A2", clipped)


def deposit_bilinear(
    detector_hits: DetectorHitBatch,
    *,
    event_row: ArrayLike,
    event_id: ArrayLike,
    assigned_mass_A2: ArrayLike,
    detector_shape_rc: tuple[int, int],
) -> DepositionResult:
    """Deposit selected mass once at continuous ``(column, row)`` hit coordinates."""

    if not isinstance(detector_hits, DetectorHitBatch):
        raise TypeError("detector_hits must be a DetectorHitBatch")
    try:
        shape = tuple(detector_shape_rc)
    except TypeError as error:
        raise ValueError("detector_shape_rc must contain two positive integers") from error
    if (
        len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in shape)
        or any(value <= 0 for value in shape)
    ):
        raise ValueError("detector_shape_rc must contain two positive integers")
    row_count, column_count = map(int, shape)

    supplied_rows = np.asarray(event_row)
    supplied_ids = np.asarray(event_id)
    if supplied_rows.dtype.kind not in "iu" or supplied_rows.ndim != 1:
        raise ValueError("event_row must be a one-dimensional integer array")
    if supplied_ids.dtype.kind not in "iu" or supplied_ids.shape != supplied_rows.shape:
        raise ValueError("event_id must be an event-row-aligned integer array")
    if np.any(supplied_rows < 0) or np.any(supplied_rows >= detector_hits.event_id.size):
        raise ValueError("event_row contains an out-of-range detector-hit row")
    selected_row = np.array(supplied_rows, dtype=np.int64, copy=True)
    selected_id = np.array(supplied_ids, dtype=np.int64, copy=True)
    if not np.array_equal(detector_hits.event_id[selected_row], selected_id):
        raise ValueError("event_id must match detector_hits.event_id at every event_row")
    if not np.all(detector_hits.valid[selected_row]):
        raise ValueError("every selected event_row must reference a valid detector hit")

    supplied_mass = np.asarray(assigned_mass_A2)
    if np.iscomplexobj(supplied_mass):
        raise ValueError("assigned_mass_A2 must be real")
    assigned_mass = np.array(supplied_mass, dtype=np.float64, copy=True)
    if (
        assigned_mass.shape != selected_row.shape
        or not np.all(np.isfinite(assigned_mass))
        or np.any(assigned_mass < 0.0)
    ):
        raise ValueError("assigned_mass_A2 must be finite, nonnegative, and event-row aligned")

    column = detector_hits.column_px[selected_row]
    row = detector_hits.row_px[selected_row]
    if (
        not np.all(np.isfinite(column))
        or not np.all(np.isfinite(row))
        or np.any((column < -0.5) | (column > column_count - 0.5))
        or np.any((row < -0.5) | (row > row_count - 0.5))
    ):
        raise ValueError("selected coordinates lie outside valid half-pixel detector support")

    base_column = np.floor(column).astype(np.int64)
    base_row = np.floor(row).astype(np.int64)
    column_fraction = column - base_column
    row_fraction = row - base_row
    image = np.zeros((row_count, column_count), dtype=np.float64)
    clipped_mass = 0.0
    for row_offset in (0, 1):
        target_row = base_row + row_offset
        row_weight = 1.0 - row_fraction if row_offset == 0 else row_fraction
        for column_offset in (0, 1):
            target_column = base_column + column_offset
            column_weight = 1.0 - column_fraction if column_offset == 0 else column_fraction
            contribution = assigned_mass * row_weight * column_weight
            inside = (
                (target_row >= 0)
                & (target_row < row_count)
                & (target_column >= 0)
                & (target_column < column_count)
            )
            np.add.at(image, (target_row[inside], target_column[inside]), contribution[inside])
            clipped_mass += float(np.sum(contribution[~inside], dtype=np.float64))

    return DepositionResult(
        image_A2=image,
        deposited_mass_A2=float(np.sum(image, dtype=np.float64)),
        clipped_mass_A2=clipped_mass,
    )
