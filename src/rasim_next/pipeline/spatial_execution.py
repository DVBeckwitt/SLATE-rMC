"""Caller-owned timing calibration and automatic native-pixel device selection.

This execution resource changes only where the shared float64 integral runs.
It retains timings and decisions, never physical arrays or detector predictions.
"""

import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from os import getpid
from pathlib import Path
from threading import get_ident
from time import perf_counter

import numpy as np
from numba.core.errors import NumbaPerformanceWarning


@dataclass(frozen=True, slots=True)
class SpatialExecutionDecision:
    backend: str
    reason: str
    pixel_visits: int = 0
    cpu_seconds: float | None = None
    cuda_seconds: float | None = None


@dataclass(slots=True)
class NativeSpatialExecutor:
    """Explicit, thread-confined runtime resource shared across native batches.

    Auto calibration occurs once, lazily, on the first substantial batch. Small
    batches never initialize CUDA. A changed quadrature rule or process/thread
    requires a fresh resource. Explicit CPU/CUDA never calibrate or substitute.
    ``summary()`` exposes admission reasons, calibration cost and batch counts.
    """

    calibration: dict | None = field(default=None, init=False)
    last_decision: SpatialExecutionDecision | None = field(default=None, init=False)
    counts: dict[str, int] = field(default_factory=dict, init=False)
    _owner: tuple[int, int] | None = field(default=None, init=False, repr=False)
    region_workers: int = 0
    region_workspace_bytes: int = 64 * 1024**2

    def __post_init__(self):
        if type(self.region_workers) is not int or self.region_workers < 0:
            raise ValueError(
                "region_workers must be a nonnegative integer (zero selects automatically)"
            )
        if type(self.region_workspace_bytes) is not int or self.region_workspace_bytes < 1:
            raise ValueError("region workspace budget must be a positive integer")

    def region_worker_count(self, event_count):
        """Bound CPU concurrency by the caller's Numba thread allowance."""
        import numba

        owner = (getpid(), get_ident())
        if self._owner is not None and self._owner != owner:
            raise RuntimeError("native spatial executor must stay in its owning process/thread")
        self._owner = owner
        available = numba.get_num_threads()
        workers = min(self.region_workers or 4, available, max(1, event_count // 1024))
        return workers

    def summary(self):
        return {
            "policy": "native_spatial_calibrated.v1",
            "calibration": deepcopy(self.calibration),
            "decisions": dict(self.counts),
            "last_decision": None if self.last_decision is None else asdict(self.last_decision),
        }

    def _record(self, backend, reason, visits=0, cpu=None, gpu=None):
        self.last_decision = SpatialExecutionDecision(backend, reason, visits, cpu, gpu)
        key = backend + ":" + reason
        self.counts[key] = self.counts.get(key, 0) + 1
        return backend

    def select(self, mode, mean, factor, mass, shape, nodes, weights, radius, row, column):
        from rasim_next.pipeline.source_spatial import _pixel_work

        if mode not in {"auto", "cpu", "cuda"}:
            raise ValueError("spatial execution must be auto, cpu or cuda")
        owner = (getpid(), get_ident())
        if self._owner is not None and self._owner != owner:
            raise RuntimeError("native spatial executor must stay in its owning process/thread")
        self._owner = owner
        if mode != "auto":
            return self._record(mode, "explicit")
        # Eight-event adaptive patches measured slower on CUDA. This is a
        # conservative admission floor, not a claimed universal crossover.
        if np.count_nonzero(mass) < 64:
            return self._record("cpu", "small_batch")
        work, active, degenerate = _pixel_work(mean, factor, mass, shape, radius, row, column)
        visits = int(work.sum())
        if active < 64 or visits < 262144:
            return self._record("cpu", "small_batch", visits)
        if degenerate:
            return self._record("cpu", "uncalibrated_degenerate_branch", visits)
        from numba import cuda

        if not cuda.is_available():
            return self._record("cpu", "cuda_unavailable", visits)
        device = cuda.get_current_device()
        if device.compute_capability < (6, 0):
            return self._record("cpu", "float64_atomics_unavailable", visits)
        from numba.cuda.cudadrv import nvvm
        from numba.cuda.cudadrv.libs import get_libdevice

        libdevice = get_libdevice()
        if not nvvm.is_available() or not libdevice or not Path(libdevice).is_file():
            return self._record("cpu", "cuda_toolkit_unavailable", visits)
        device_bytes = (
            8 * shape[0] * shape[1] + len(mean) * (56 + 48 * len(nodes)) + 16 * len(nodes)
        )
        free, _ = cuda.current_context().get_memory_info()
        calibration_bytes = (
            max(
                8 * 1024**2 + 56 + 64 * len(nodes),
                8 * 256**2 + 256 * (56 + 48 * len(nodes)) + 16 * len(nodes),
            )
            if self.calibration is None
            else 0
        )
        if max(device_bytes, calibration_bytes) > 0.8 * free:
            return self._record("cpu", "device_memory", visits)
        identity = (int(device.id), str(device.uuid), len(nodes), float(radius))
        if self.calibration is None:
            self.calibration = _calibrate(nodes, weights, radius, identity)
        profile = self.calibration
        if profile["identity"] != identity:
            return self._record("cpu", "calibration_identity_changed", visits)
        if not profile["usable"]:
            return self._record("cpu", "calibration_unresolved", visits)
        free, _ = cuda.current_context().get_memory_info()
        if device_bytes > 0.8 * free:
            return self._record("cpu", "device_memory", visits)
        cpu = float(work @ np.asarray(profile["cpu_seconds_per_visit"]))
        # Include host parameter preparation, all transfers and readback. Limit
        # extrapolation below the calibrated number of independent event blocks.
        gpu = float(work @ np.asarray(profile["cuda_seconds_per_visit"]))
        gpu *= max(1.0, 256.0 / active)
        gpu += profile["launch_seconds"]
        gpu += device_bytes * profile["transfer_seconds_per_device_byte"]
        gpu += len(mean) * profile["event_seconds"]
        if gpu * 1.35 < cpu:
            return self._record("cuda", "predicted_faster", visits, cpu, gpu)
        return self._record("cpu", "insufficient_predicted_gain", visits, cpu, gpu)


def _calibrate(nodes, weights, radius, identity):
    """Bounded local timing of the production integral, never a physical fit.

    Four correlation branches use 256 events on 256x256 windows. Zero-mass
    calls separate launch, output traffic and per-event preparation overhead.
    Every timed GPU call includes transfers, synchronization and readback.
    """
    from rasim_next.pipeline._source_spatial_cuda import deposit_gaussian_pixels_cuda
    from rasim_next.pipeline.source_spatial import _deposit_gaussian_pixels, _pixel_work

    started = perf_counter()
    shape = (256, 256)
    indices = np.arange(256)
    mean = np.column_stack((16.125 + 14 * (indices % 16), 16.375 + 14 * (indices // 16)))
    mass = np.ones(256)
    cpu_rates, gpu_rates, errors = [], [], []

    def run(backend, positions, factors, masses, window):
        start = perf_counter()
        if backend == "cpu":
            result = _deposit_gaussian_pixels(
                positions, factors, masses, window, nodes, weights, nodes, weights, radius, 0, 0
            )
        else:
            result = deposit_gaussian_pixels_cuda(
                positions, factors, masses, window, nodes, weights, radius, 0, 0, None
            )
        return result, perf_counter() - start

    factor = np.tile(np.eye(2) * 2.0, (256, 1, 1))
    # Compile both paths before timing. Compilation is retained in total setup time.
    for backend in ("cpu", "cuda"):
        run(backend, mean, factor, mass, shape)
    zero = np.zeros(256)
    # These launches intentionally measure overhead at low occupancy.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumbaPerformanceWarning)
        base = min(run("cuda", mean[:1], factor[:1], zero[:1], (1, 1))[1] for _ in range(3))
        large = min(run("cuda", mean[:1], factor[:1], zero[:1], (1024, 1024))[1] for _ in range(3))
    traffic_rate = max(0.0, large - base) / (8 * (1024 * 1024 - 1))
    events = min(run("cuda", mean, factor, zero, (1, 1))[1] for _ in range(3))
    event_rate = max(0.0, events - base) / 255
    for rho in (0.0, 0.6, 0.9, 0.99):
        factor[:, 1, 0] = 2.0 * rho
        factor[:, 1, 1] = 2.0 * np.sqrt(1.0 - rho * rho)
        work, _, _ = _pixel_work(mean, factor, mass, shape, radius, 0, 0)
        cpu, cpu_time = run("cpu", mean, factor, mass, shape)
        gpu, gpu_time = run("cuda", mean, factor, mass, shape)
        cpu_time = min(cpu_time, run("cpu", mean, factor, mass, shape)[1])
        gpu_time = max(gpu_time, run("cuda", mean, factor, mass, shape)[1])
        error = float(np.max(np.abs(cpu - gpu)))
        if (
            not np.all(np.isfinite(cpu))
            or not np.all(np.isfinite(gpu))
            or error > 1e-11 * float(np.max(cpu))
        ):
            raise FloatingPointError("CUDA spatial calibration disagrees with the CPU integral")
        errors.append(error)
        cpu_rates.append(cpu_time / float(work.sum()))
        overhead = base + 8 * np.prod(shape) * traffic_rate + 256 * event_rate
        gpu_rates.append((gpu_time - overhead) / float(work.sum()))
    return {
        "identity": identity,
        "usable": bool(
            np.all(np.isfinite(cpu_rates + gpu_rates)) and min(cpu_rates + gpu_rates) > 0
        ),
        "cpu_seconds_per_visit": cpu_rates,
        "cuda_seconds_per_visit": gpu_rates,
        "launch_seconds": base,
        "transfer_seconds_per_device_byte": float(traffic_rate),
        "event_seconds": event_rate,
        "parity_max_absolute": errors,
        "setup_seconds": perf_counter() - started,
    }
