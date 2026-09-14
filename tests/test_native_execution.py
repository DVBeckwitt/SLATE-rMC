"""Interrupted, out-of-order raw work remains reusable without stale objectives."""

import numpy as np
import pytest

from rasim_next.fitting.native_execution import NativePredictionStore


def test_partial_unordered_batch_resume_preserves_exact_rows_and_rejects_other_physics():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    store = NativePredictionStore("explicit-physics-and-numerics", 2, 3)
    checkpoints = []

    def model(v):
        return np.r_[v, v.sum()]

    def interrupted(rows, repeats):
        assert repeats == 13
        yield 2, model(rows[2])
        yield 0, model(rows[0])
        raise RuntimeError("worker stopped")

    with pytest.raises(RuntimeError, match="worker stopped"):
        store.evaluate(
            values, 13, interrupted, checkpoint=lambda: checkpoints.append(store.state())
        )
    assert checkpoints[-1]["completed_count"] == 2
    np.testing.assert_array_equal(store.pending_values, values[1:2])
    restored = NativePredictionStore(store.revision, 2, 3)
    restored.restore(store.arrays(), store.state())
    with pytest.raises(ValueError, match="revision"):
        NativePredictionStore("changed-source-rule", 2, 3).restore(store.arrays(), store.state())
    dispatched = []

    def execute(rows, repeats):
        dispatched.extend(rows.tolist())
        for i in range(len(rows) - 1, -1, -1):
            yield i, model(rows[i])

    result = restored.evaluate(values[[1, 2, 0, 1]], 13, execute)
    np.testing.assert_array_equal(result, [model(v) for v in values[[1, 2, 0, 1]]])
    assert dispatched == values[1:2].tolist()
    restored.evaluate(values[:1], 14, execute)
    assert len(dispatched) == 2  # A different discrete physical model cannot reuse the row.
    with pytest.raises(ValueError, match="unique indices"):
        NativePredictionStore("proof", 2, 3).evaluate(
            values, 13, lambda rows, n: [(0, model(rows[0])), (0, model(rows[0]))]
        )
