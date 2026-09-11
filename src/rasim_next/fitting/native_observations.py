"""Frozen detector-native observations with one count/background covariance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from rasim_next.measurement.continuous_regions import NativePixelRegionProjection


@dataclass(frozen=True, slots=True)
class NativeFitObservations:
    """Counts and immutable GLS whitening; historical guards remain separate."""

    projection: NativePixelRegionProjection
    net_count: np.ndarray
    valid: np.ndarray
    covariance_count2: np.ndarray
    fit_operator: np.ndarray
    fit_target: np.ndarray
    guard_operator: np.ndarray
    guard_pointer: np.ndarray
    guard_limit: np.ndarray
    input_revision: str
    _cholesky: np.ndarray = field(init=False, repr=False)
    _whitened_net: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = self.projection.observation_count
        for name in (
            "net_count",
            "valid",
            "covariance_count2",
            "fit_operator",
            "fit_target",
            "guard_operator",
            "guard_pointer",
            "guard_limit",
        ):
            raw = np.asarray(getattr(self, name))
            if np.iscomplexobj(raw) or np.any(~np.isfinite(raw)):
                raise ValueError("observation arrays must be finite and real")
            if name == "valid" and raw.dtype.kind != "b":
                raise TypeError("valid must be a boolean vector")
            if name == "guard_pointer" and raw.dtype.kind not in "iu":
                raise TypeError("guard pointers must be integers")
            value = np.array(raw, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if (
            self.net_count.shape != (n,)
            or self.valid.shape != (n,)
            or not np.any(self.valid)
            or self.covariance_count2.shape != (n, n)
            or self.fit_target.ndim != 1
            or self.guard_limit.ndim != 1
            or self.fit_operator.shape != (len(self.fit_target), n)
            or self.guard_operator.ndim != 2
            or self.guard_operator.shape[1] != n
            or self.guard_pointer.shape != (len(self.guard_limit) + 1,)
            or self.guard_pointer[0] != 0
            or self.guard_pointer[-1] != self.guard_operator.shape[0]
            or np.any(np.diff(self.guard_pointer) <= 0)
            or np.any(self.guard_limit <= 0)
        ):
            raise ValueError("native observations, covariance and historical guards must align")
        if not np.allclose(
            self.covariance_count2, self.covariance_count2.T, rtol=1e-13, atol=1e-10
        ):
            raise ValueError("count covariance must be symmetric")
        covariance = self.covariance_count2[np.ix_(self.valid, self.valid)]
        factor = cholesky((covariance + covariance.T) * 0.5, lower=True)
        factor.setflags(write=False)
        object.__setattr__(self, "_cholesky", factor)
        net = self.whiten(self.net_count)
        net.setflags(write=False)
        object.__setattr__(self, "_whitened_net", net)

    def whiten(self, values: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(values):
            raise ValueError("count predictions must be real")
        values = np.asarray(values, dtype=float)
        if values.shape != self.net_count.shape or np.any(~np.isfinite(values)):
            raise ValueError("predictions must be finite and match the frozen observations")
        return solve_triangular(self._cholesky, values[self.valid], lower=True)

    def profile_scale(
        self, raw_prediction: np.ndarray, *, enforce_guards: bool = False
    ) -> tuple[float, np.ndarray]:
        """Exact nonnegative GLS scale, including correlated observation errors."""
        shape = self.whiten(raw_prediction)
        denominator = float(shape @ shape)
        if denominator == 0:
            raise ValueError("a zero prediction cannot determine an intensity scale")
        scale = max(0.0, float(shape @ self._whitened_net) / denominator)
        if enforce_guards:
            interval = self.guard_scale_interval(raw_prediction)
            if interval is not None:
                scale = float(np.clip(scale, *interval))
        return scale, scale * shape - self._whitened_net

    def guard_scale_interval(self, raw_prediction: np.ndarray) -> tuple[float, float] | None:
        """Intersect the exact nonnegative scale intervals allowed by every old guard.

        None means this shape cannot pass at any scale. A constrained optimizer
        must then change the shape; the GLS scale is only a search diagnostic.
        """
        raw_prediction = np.asarray(raw_prediction)
        if (
            np.iscomplexobj(raw_prediction)
            or raw_prediction.shape != self.net_count.shape
            or np.any(~np.isfinite(raw_prediction))
        ):
            raise ValueError("guard predictions must be finite real aligned counts")
        shape = self.guard_operator @ raw_prediction
        target = self.guard_operator @ self.net_count
        pointer = self.guard_pointer[:-1]
        a = np.add.reduceat(shape * shape, pointer)
        b = np.add.reduceat(shape * target, pointer)
        c = np.add.reduceat(target * target, pointer) - self.guard_limit**2
        if np.any((a == 0) & (c > 0)):
            return None
        nonzero = a > 0
        a, b, c = a[nonzero], b[nonzero], c[nonzero]
        discriminant = b * b - a * c
        tolerance = 64 * np.finfo(float).eps * (b * b + abs(a * c))
        if np.any(discriminant < -tolerance):
            return None
        root = np.sqrt(np.maximum(discriminant, 0))
        low = max(0.0, float(np.max((b - root) / a, initial=0)))
        high = float(np.min((b + root) / a, initial=np.inf))
        return (low, high) if low <= high else None

    def scores(self, prediction_count: np.ndarray) -> dict:
        residual = self.whiten(prediction_count - self.net_count)
        old = self.fit_operator @ prediction_count - self.fit_target
        guard = self.guard_operator @ (prediction_count - self.net_count)
        rms = np.sqrt(np.add.reduceat(guard * guard, self.guard_pointer[:-1]))
        return dict(
            gls_chi_square=float(residual @ residual),
            historical_loss=float(old @ old),
            guard_scores=rms,
            guards_pass=bool(np.all(rms <= self.guard_limit + 1e-6)),
        )


def load_native_fit_observations(path: Path) -> NativeFitObservations:
    """Read numeric NPZ only, verify its hash and retain the declared native support."""
    path = Path(path)
    payload = path.read_bytes()
    record = json.loads(payload)
    if record.get("schema") != "rasim-native-fit-observations-v1":
        raise ValueError("unsupported native observation schema")
    arrays_path = path.parent / record["arrays"]["path"]
    with arrays_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != record["arrays"]["sha256"]:
        raise ValueError("native observation array SHA256 mismatch")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        data = record["signal_projection"]
        projection = NativePixelRegionProjection(
            detector_shape_rc=tuple(data["detector_shape_rc"]),
            observation_count=data["observation_count"],
            quadrature_revision=data["quadrature_revision"],
            **{key: arrays[value] for key, value in data["arrays"].items()},
        )
        if projection.projection_revision != data["projection_revision"]:
            raise ValueError("native observation projection revision mismatch")
        net = arrays["measured"] - arrays["background"]
        if not np.allclose(net, arrays["frozen_net"], rtol=0, atol=1e-8):
            raise ValueError("historical operators require the unchanged net observations")
        modes = arrays["background_modes"]
        return NativeFitObservations(
            projection,
            net,
            arrays["valid"],
            arrays["count_covariance"] + modes.T @ modes,
            arrays["frozen_fit_operator"],
            arrays["frozen_target"],
            arrays["frozen_guard_operator"],
            arrays["frozen_guard_pointer"],
            arrays["frozen_guard_limit"],
            hashlib.sha256(payload).hexdigest(),
        )
