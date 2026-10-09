"""Simulator-only summed display levels and bounded contrast statistics.

These are presentation arrays. The original detector data and numerical measure
remain owned by the immutable simulation result.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DisplayLevel:
    bin_size: int
    values: np.ndarray
    positive_quantiles: np.ndarray
    positive_count: int
    invalid_count: int
    minimum: float
    maximum: float

    def above_fraction(self, upper):
        """Approximate fraction of positive display cells above a level (CDF step 0.1%)."""
        q = self.positive_quantiles
        if not self.positive_count or upper >= q[-1]:
            return 0.0
        if upper < q[0]:
            return 1.0
        index = int(np.searchsorted(q, upper, side="right"))
        fraction = (index - 1 + (upper - q[index - 1]) / (q[index] - q[index - 1])) / 1000
        return float(1 - fraction)


def _level(values, bin_size):
    finite = np.isfinite(values)
    positive = values[finite & (values > 0)]
    quantiles = np.quantile(positive, np.linspace(0, 1, 1001)) if positive.size else np.zeros(1001)
    quantiles.setflags(write=False)
    minimum = float(np.min(values, where=finite, initial=np.inf))
    maximum = float(np.max(values, where=finite, initial=-np.inf))
    if not np.isfinite(minimum):
        minimum, maximum = 0.0, 0.0
    values.setflags(write=False)
    return DisplayLevel(
        bin_size,
        values,
        quantiles,
        int(positive.size),
        int(values.size - np.count_nonzero(finite)),
        minimum,
        maximum,
    )


def prepare_display_levels(image, display=None, inclusion=None):
    """Sum disjoint native cells on the worker; final edge bins may be partial.

    Nonfinite/excluded input cells contribute no signal. A bin with no included
    finite cells remains NaN. No intensity, profile or cursor array is modified.
    """
    source = np.asarray(image)
    if source.ndim != 2 or not source.size:
        raise ValueError("display preparation needs a nonempty detector plane")
    if inclusion is not None and (inclusion.shape != source.shape or inclusion.dtype != np.bool_):
        raise ValueError("display inclusion must be a Boolean native-shape mask")
    if display is None:
        display = np.ascontiguousarray(source, dtype=np.float32)
    if inclusion is not None:
        display = np.where(inclusion, display, np.nan).astype(np.float32)
    levels = [_level(display, 1)]
    # float64 finite sums and support counts; row blocks avoid a full float64 copy.
    sums = None
    support = None
    factor = 2
    while max(source.shape) > 1:
        rows, columns = source.shape
        result = np.empty(((rows + 1) // 2, (columns + 1) // 2), dtype=np.float64)
        counts = np.empty(result.shape, dtype=np.int32)
        for start in range(0, rows, 64):
            block = source[start : start + 64]
            if sums is None:
                valid = np.isfinite(block)
                if inclusion is not None:
                    valid &= inclusion[start : start + 64]
                block = np.where(valid, block, 0.0)
                count_block = valid.astype(np.int32)
            else:
                count_block = support[start : start + 64]
            row_sums = np.add.reduceat(block, np.arange(0, len(block), 2), axis=0, dtype=np.float64)
            row_counts = np.add.reduceat(count_block, np.arange(0, len(block), 2), axis=0)
            result[start // 2 : (start + len(block) + 1) // 2] = np.add.reduceat(
                row_sums, np.arange(0, columns, 2), axis=1
            )
            counts[start // 2 : (start + len(block) + 1) // 2] = np.add.reduceat(
                row_counts, np.arange(0, columns, 2), axis=1
            )
        values = result.astype(np.float32)
        values[counts == 0] = np.nan
        levels.append(_level(values, factor))
        sums, support, source = result, counts, result
        factor *= 2
        if max(result.shape) == 1:
            break
    return tuple(levels)


def select_display_level(levels, native_shape, viewport_width, viewport_height):
    """Choose the finest sum grid that is not minified at the device pixel size."""
    rows, columns = native_shape
    needed = max(columns / max(viewport_width, 1), rows / max(viewport_height, 1))
    return next((level for level in levels if level.bin_size + 1e-12 >= needed), levels[-1])
