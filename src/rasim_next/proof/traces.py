"""Trace records and first-divergence comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rasim_next.core import traces as _core_traces


@dataclass(frozen=True, slots=True)
class Tolerance:
    atol: float = 0.0
    rtol: float = 0.0
    scale: float = 0.0

    @property
    def limit(self) -> float:
        return self.atol + self.rtol * self.scale


@dataclass(frozen=True, slots=True)
class TraceComparison:
    passed: bool
    first_failing_stage: str | None
    failure_metric: str | None
    maximum_error: float | None
    percentile_95_error: float | None


def compare_traces(
    reference: Sequence[_core_traces.TraceRecord],
    candidate: Sequence[_core_traces.TraceRecord],
    tolerances: Mapping[str, Tolerance] | None = None,
) -> TraceComparison:
    """Compare in reference order and return the first divergent stage."""

    actual = {(record.case_id, record.stage_id): record for record in candidate}
    if len(actual) != len(candidate):
        raise ValueError("candidate trace keys must be unique")
    expected_keys = {(record.case_id, record.stage_id) for record in reference}
    for expected in reference:
        observed = actual.get((expected.case_id, expected.stage_id))
        metric: str | None = None
        error: NDArray[np.float64] | None = None
        if observed is None:
            metric = "missing_stage"
        elif expected.value.shape != observed.value.shape:
            metric = "shape"
        elif expected.value.dtype != observed.value.dtype:
            metric = "dtype"
        elif any(
            getattr(expected, field) != getattr(observed, field)
            for field in ("unit", "frame", "measure", "quantity_kind", "model_version", "provenance")
        ):
            metric = "metadata"
        elif expected.value.dtype.kind in "biuSU":
            metric = None if np.array_equal(expected.value, observed.value) else "exact_value"
        else:
            error = np.abs(observed.value - expected.value).astype(np.float64, copy=False)
            limit = (tolerances or {}).get(expected.stage_id, Tolerance()).limit
            metric = "numeric_value" if np.any(~np.isfinite(error)) or np.any(error > limit) else None
        if metric:
            return TraceComparison(
                False,
                expected.stage_id,
                metric,
                None if error is None else float(np.max(error)),
                None if error is None else float(np.percentile(error, 95.0)),
            )
    for record in candidate:
        if (record.case_id, record.stage_id) not in expected_keys:
            return TraceComparison(False, record.stage_id, "unexpected_stage", None, None)
    return TraceComparison(True, None, None, 0.0, 0.0)
