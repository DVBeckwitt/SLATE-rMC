"""Frozen detector-native observations with one count/background covariance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls

from rasim_next.measurement.continuous_regions import NativePixelRegionProjection


@dataclass(frozen=True, slots=True)
class NativeFitObservations:
    """Frozen counts, covariance and an explicitly selected fitting objective."""

    projection: NativePixelRegionProjection | None
    net_count: np.ndarray
    valid: np.ndarray
    covariance_count2: np.ndarray
    fit_operator: np.ndarray
    fit_target: np.ndarray
    guard_operator: np.ndarray
    guard_pointer: np.ndarray
    guard_limit: np.ndarray
    input_revision: str
    allow_guard_constraints: bool = True
    objective_kind: str = "gls"
    exposure_index: np.ndarray | None = None
    _cholesky: np.ndarray = field(init=False, repr=False)
    _whitened_net: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.objective_kind not in ("gls", "historical"):
            raise ValueError("objective_kind must be gls or historical")
        if self.objective_kind == "historical" and not self.allow_guard_constraints:
            raise ValueError("historical objective requires the complete frozen observation roster")
        n = len(self.net_count) if self.projection is None else self.projection.observation_count
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
        if self.exposure_index is not None:
            indices = np.asarray(self.exposure_index)
            if (
                indices.shape != (n,)
                or indices.dtype.kind not in "iu"
                or not np.array_equal(np.unique(indices), np.arange(indices.max() + 1))
                or not np.array_equal(np.unique(indices[self.valid]), np.unique(indices))
                or self.objective_kind != "gls"
                or self.allow_guard_constraints
            ):
                raise ValueError(
                    "exposure groups require aligned contiguous GLS indices and no historical guards"
                )
            indices = np.array(indices, dtype=np.int64, copy=True)
            indices.setflags(write=False)
            object.__setattr__(self, "exposure_index", indices)
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

    def objective_residual(self, prediction_count: np.ndarray) -> np.ndarray:
        """Residual in the declared fitting measure at literal predicted counts."""
        if self.objective_kind == "historical":
            return self.fit_operator @ prediction_count - self.fit_target
        return self.whiten(prediction_count - self.net_count)

    def scalar_scale_design(self, raw_prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scalar scale design and target in the declared fitting measure."""
        if self.exposure_index is not None:
            raise ValueError("scalar scale design is unavailable for exposure groups")
        whitened = self.whiten(raw_prediction)
        if self.objective_kind == "historical":
            return self.fit_operator @ raw_prediction, self.fit_target
        return whitened, self._whitened_net

    def profile_scale(
        self, raw_prediction: np.ndarray, *, enforce_guards: bool = False
    ) -> tuple[float | np.ndarray, np.ndarray]:
        """Exact nonnegative scale for the declared quadratic objective."""
        if enforce_guards and not self.allow_guard_constraints:
            raise ValueError("historical guards may only be diagnostic on a training split")
        if self.exposure_index is not None:
            design = np.zeros((len(raw_prediction), int(self.exposure_index.max()) + 1))
            design[np.arange(len(raw_prediction)), self.exposure_index] = raw_prediction
            design = solve_triangular(self._cholesky, design[self.valid], lower=True)
            if np.any(np.linalg.norm(design, axis=0) <= np.finfo(float).tiny):
                raise ValueError("each acquisition requires scale-identifying signal")
            scales, _ = nnls(design, self._whitened_net)
            return scales, self.objective_residual(self.apply_scale(raw_prediction, scales))
        shape, target = self.scalar_scale_design(raw_prediction)
        denominator = float(shape @ shape)
        if denominator == 0:
            raise ValueError("a zero prediction cannot determine an intensity scale")
        scale = max(0.0, float(shape @ target) / denominator)
        if enforce_guards:
            interval = self.guard_scale_interval(raw_prediction)
            if interval is not None:
                scale = float(np.clip(scale, *interval))
        return scale, self.objective_residual(self.apply_scale(raw_prediction, scale))

    def apply_scale(self, raw_prediction, scale):
        """Apply declared exposure scales using the explicit acquisition row mapping."""
        scale = np.asarray(scale)
        expected = () if self.exposure_index is None else (int(self.exposure_index.max()) + 1,)
        if (
            scale.shape != expected
            or np.iscomplexobj(scale)
            or np.any(~np.isfinite(scale))
            or np.any(scale < 0)
        ):
            raise ValueError("scale must be finite nonnegative and match the declared acquisitions")
        raw_prediction = np.asarray(raw_prediction)
        if (
            raw_prediction.shape != self.net_count.shape
            or np.iscomplexobj(raw_prediction)
            or np.any(~np.isfinite(raw_prediction))
        ):
            raise ValueError("predictions must be finite real aligned counts")
        if self.exposure_index is not None:
            scale = scale[self.exposure_index]
        return scale * raw_prediction

    def guard_scale_interval(self, raw_prediction: np.ndarray) -> tuple[float, float] | None:
        """Intersect the exact nonnegative scale intervals allowed by every old guard.

        None means this shape cannot pass at any scale. A constrained optimizer
        must then change the shape; the unconstrained scale is only a search diagnostic.
        """
        if not self.allow_guard_constraints:
            raise ValueError("historical scale constraints are unavailable for this objective")
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
        declared = self.objective_residual(prediction_count)
        residual = (
            declared
            if self.objective_kind == "gls"
            else self.whiten(prediction_count - self.net_count)
        )
        old = (
            declared
            if self.objective_kind == "historical"
            else self.fit_operator @ prediction_count - self.fit_target
        )
        guard = self.guard_operator @ (prediction_count - self.net_count)
        rms = np.sqrt(np.add.reduceat(guard * guard, self.guard_pointer[:-1]))
        return dict(
            objective_kind=self.objective_kind,
            data_objective=float(old @ old)
            if self.objective_kind == "historical"
            else float(residual @ residual),
            gls_chi_square=float(residual @ residual),
            historical_loss=float(old @ old) if len(old) else None,
            guard_scores=rms,
            guards_pass=bool(np.all(rms <= self.guard_limit + 1e-6)),
        )


def load_native_fit_observations(
    path: Path, *, arrays_path: Path | None = None
) -> NativeFitObservations:
    """Read numeric NPZ only, verify its hash and retain the declared native support."""
    path = Path(path)
    payload = path.read_bytes()
    record = json.loads(payload)
    if record.get("schema") != "rasim-native-fit-observations-v1":
        raise ValueError("unsupported native observation schema")
    arrays_path = (
        path.parent / record["arrays"]["path"] if arrays_path is None else Path(arrays_path)
    )
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


@dataclass(frozen=True, slots=True)
class NativeBackgroundControl:
    """Frozen control support; measurement noise is distinct from calibration weighting."""

    projection: NativePixelRegionProjection
    measurement_variance_count2: np.ndarray
    split: np.ndarray
    revision: str

    def __post_init__(self):
        n = self.projection.observation_count
        for name in ("measurement_variance_count2", "split"):
            value = np.array(getattr(self, name), copy=True)
            if value.shape != (n,) or np.iscomplexobj(value) or np.any(~np.isfinite(value)):
                raise ValueError("control arrays must be aligned finite real vectors")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if (
            np.any(self.measurement_variance_count2 <= 0)
            or self.split.dtype.kind not in "iu"
            or not np.all(np.isin(self.split, (0, 1, 2)))
            or not self.revision
        ):
            raise ValueError(
                "controls require positive measurement variance and declared split labels"
            )

    def signal_diagnostic(self, predicted_signal_count):
        """Predicted diffraction contamination, without fitting a scale or changing background."""
        signal = np.asarray(predicted_signal_count)
        if (
            signal.shape != self.split.shape
            or np.iscomplexobj(signal)
            or np.any(~np.isfinite(signal))
            or np.any(signal < 0)
        ):
            raise ValueError("control signal must be a finite nonnegative count vector")
        sigma = signal / np.sqrt(self.measurement_variance_count2)
        density = signal / self.projection.observation_measure_px2
        return dict(
            signal_measurement_sigma=sigma,
            signal_density_count_per_px2=density,
            split_summary=[
                dict(
                    split=i,
                    row_count=int(np.sum(self.split == i)),
                    predicted_signal_count=float(signal[self.split == i].sum()),
                    maximum_signal_measurement_sigma=float(
                        np.max(sigma[self.split == i], initial=0)
                    ),
                    squared_signal_norm=float(np.sum(sigma[self.split == i] ** 2)),
                )
                for i in (0, 1, 2)
            ],
            interpretation="predicted signal norm, not a goodness-of-fit chi-square",
        )


def load_native_background_controls(path: Path) -> tuple[NativeBackgroundControl, ...]:
    """Load independent control layouts; do not combine their overlapping cells as evidence."""
    path = Path(path)
    record = json.loads(path.read_bytes())
    arrays_path = path.parent / record["arrays"]["path"]
    with arrays_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != record["arrays"]["sha256"]:
            raise ValueError("native observation array SHA256 mismatch")
    controls = (record["background"]["controls"], *record["background"]["guard_controls"])
    result = []
    with np.load(arrays_path, allow_pickle=False) as arrays:
        signal_pixels = arrays[record["signal_projection"]["arrays"]["flat_pixel_index"]]
        for control in controls:
            data = control["projection"]
            projection = NativePixelRegionProjection(
                detector_shape_rc=tuple(data["detector_shape_rc"]),
                observation_count=data["observation_count"],
                quadrature_revision=data["quadrature_revision"],
                **{key: arrays[value] for key, value in data["arrays"].items()},
            )
            if projection.projection_revision != data["projection_revision"] or len(
                np.intersect1d(signal_pixels, projection.flat_pixel_index)
            ):
                raise ValueError("control projection is changed or overlaps frozen signal pixels")
            result.append(
                NativeBackgroundControl(
                    projection,
                    arrays[control["arrays"]["measurement_variance_count2"]],
                    arrays[control["arrays"]["split"]],
                    control["revision"],
                )
            )
    return tuple(result)
