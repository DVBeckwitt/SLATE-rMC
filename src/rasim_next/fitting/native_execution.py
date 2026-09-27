"""Parent-owned exact raw predictions, including partially completed batches."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NativePredictionStore:
    """Recover raw forward work; optimizers always restart through their public API.

    The caller binds revision to input hashes, implementation, physical/numerical
    model and parameter ordering. Residuals and Jacobians are deliberately absent:
    scale, covariance, calibration and bounds are evaluated by the current search.
    Checkpoint callbacks own external I/O. Workers never mutate this owner.
    """

    revision: str
    parameter_count: int
    observation_count: int
    _completed: dict = field(default_factory=dict, init=False, repr=False)
    pending_values: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), init=False)
    pending_repeats: int | None = field(default=None, init=False)
    dispatched_count: int = field(default=0, init=False)
    replayed_count: int = field(default=0, init=False)

    def __post_init__(self):
        if not self.revision or any(
            type(n) is not int or n < 1 for n in (self.parameter_count, self.observation_count)
        ):
            raise ValueError("prediction store requires explicit revision and positive dimensions")
        self.pending_values = np.empty((0, self.parameter_count))

    @property
    def values(self):
        return [entry[0] for entry in self._completed.values()]

    @property
    def repeats(self):
        return [entry[1] for entry in self._completed.values()]

    @property
    def raw(self):
        return [entry[2] for entry in self._completed.values()]

    def arrays(self):
        return dict(
            prediction_values=np.asarray(self.values).reshape(-1, self.parameter_count),
            prediction_repeats=np.asarray(self.repeats, dtype=np.int64),
            prediction_raw=np.asarray(self.raw).reshape(-1, self.observation_count),
            pending_prediction_values=self.pending_values,
        )

    def state(self):
        return dict(
            revision=self.revision,
            pending_repeats=self.pending_repeats,
            dispatched_count=self.dispatched_count,
            completed_count=len(self._completed),
            replayed_count=self.replayed_count,
        )

    def restore(self, arrays, state):
        if self._completed or state["revision"] != self.revision:
            raise ValueError("prediction checkpoint revision differs or store is not empty")
        values, repeats, raw = (
            arrays[k] for k in ("prediction_values", "prediction_repeats", "prediction_raw")
        )
        if (
            values.shape != (len(repeats), self.parameter_count)
            or raw.shape != (len(repeats), self.observation_count)
            or repeats.dtype.kind not in "iu"
            or np.any(repeats < 1)
            or any(np.iscomplexobj(a) or np.any(~np.isfinite(a)) for a in (values, raw))
        ):
            raise ValueError("invalid completed prediction checkpoint")
        for v, n, row in zip(values, repeats, raw, strict=True):
            self._insert(v, int(n), row)
        self.dispatched_count = int(state["dispatched_count"])
        self.replayed_count = int(state["replayed_count"])
        # Pending work is regenerated from the declared public search. Completed
        # exact rows are reused regardless of the order in which it asks for them.

    def _insert(self, values, repeats, raw):
        key = repeats, np.asarray(values, dtype=float).tobytes()
        if key in self._completed:
            raise ValueError("duplicate completed prediction in checkpoint or worker output")
        values = np.array(values, dtype=float, copy=True)
        values.setflags(write=False)
        row = np.array(raw, dtype=float, copy=True)
        row.setflags(write=False)
        # Publish one complete record: an interrupt cannot leave the serialized
        # parameter, N and raw arrays at different lengths.
        self._completed[key] = values, repeats, row

    def evaluate(self, values, repeats, execute, *, checkpoint=None):
        """Execute missing rows; execute yields (input index, raw row), in any order."""
        values = np.asarray(values)
        if (
            values.ndim != 2
            or values.shape[1] != self.parameter_count
            or np.iscomplexobj(values)
            or np.any(~np.isfinite(values))
            or type(repeats) is not int
            or repeats < 1
        ):
            raise ValueError("prediction batch requires finite full vectors and positive N")
        values = np.asarray(values, dtype=float)
        keys = [(repeats, row.tobytes()) for row in values]
        missing = {}
        for key, row in zip(keys, values, strict=True):
            if key in self._completed:
                self.replayed_count += 1
            else:
                missing.setdefault(key, row)
        if missing:
            trials = np.array(list(missing.values()))
            complete = np.zeros(len(trials), dtype=bool)
            self.pending_values, self.pending_repeats = trials, repeats
            self.dispatched_count += len(trials)
            if checkpoint is not None:
                checkpoint()
            try:
                for index, raw in execute(trials, repeats):
                    raw = np.asarray(raw)
                    if (
                        type(index) is not int
                        or not 0 <= index < len(trials)
                        or complete[index]
                        or raw.shape != (self.observation_count,)
                        or np.iscomplexobj(raw)
                        or np.any(~np.isfinite(raw))
                    ):
                        raise ValueError(
                            "worker must return unique indices and finite aligned raw rows"
                        )
                    self._insert(trials[index], repeats, raw)
                    complete[index] = True
                    self.pending_values = trials[~complete]
                    if checkpoint is not None:
                        checkpoint()
                if not np.all(complete):
                    raise ValueError("prediction worker omitted requested rows")
            finally:
                if not len(self.pending_values):
                    self.pending_repeats = None
                if checkpoint is not None:
                    checkpoint()
        return np.array([self._completed[key][2] for key in keys])
