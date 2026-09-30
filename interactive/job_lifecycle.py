"""Bounded, GUI-owned lifecycle for one background operation at a time.

Workers receive immutable request metadata and a cooperative control object. They
never touch Qt widgets. The GUI polls one result slot; no worker-to-GUI event queue
can accumulate large payloads or progress messages.
"""

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from types import FunctionType
from typing import Any
from uuid import UUID

from comparison_state import LineWork
from mask_state import MaskWork
from PySide6.QtCore import QObject, QTimer, Signal

MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESULT_BYTES = 96 * 1024 * 1024
MAX_SUMMARIES = 8


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel-requested"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Revisions:
    data: int = 0
    mask: int = 0
    calibration: int = 0
    model: int = 0

    def __post_init__(self) -> None:
        if min(self.data, self.mask, self.calibration, self.model) < 0:
            raise ValueError("Revisions must be nonnegative")


@dataclass(frozen=True, slots=True)
class JobIdentity:
    project_id: UUID
    acquisition_id: UUID | None
    revisions: Revisions
    generation: int


@dataclass(frozen=True, slots=True)
class JobResult:
    value: Any
    resident_bytes: int


class JobControl:
    """Thread-safe cancellation and a single replaceable progress message."""

    def __init__(self) -> None:
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._progress: str | None = None

    @property
    def canceled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            self._progress = None

    def report(self, message: str) -> None:
        if self.canceled:
            return
        with self._lock:
            if not self.canceled:
                self._progress = message[:160]

    def take_progress(self) -> str | None:
        with self._lock:
            message, self._progress = self._progress, None
        return message


@dataclass(frozen=True, slots=True)
class JobRequest:
    project_id: UUID
    acquisition_id: UUID | None
    revisions: Revisions
    argument: bytes | str | Path | memoryview | MaskWork | LineWork
    argument_bytes: int
    expected_result_bytes: int
    run: Callable[[Any, JobControl], JobResult]


@dataclass(frozen=True, slots=True)
class JobSummary:
    identity: JobIdentity
    state: JobState
    detail: str = ""
    safe_stop_ms: float | None = None


@dataclass(slots=True)
class _Scheduled:
    identity: JobIdentity
    request: JobRequest


@dataclass(slots=True)
class _Active:
    scheduled: _Scheduled
    control: JobControl
    thread: threading.Thread | None = None
    outcome: tuple[JobResult | None, str | None] | None = None
    cancel_at: float | None = None


def _known_bytes(value: Any) -> int:
    if isinstance(value, (MaskWork, LineWork)):
        return value.argument_bytes
    if isinstance(value, memoryview):
        return value.nbytes
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, (str, Path)):
        return len(str(value).encode("utf-8"))
    size = getattr(value, "nbytes", 0)
    return int(size) if isinstance(size, int) else 0


class JobOwner(QObject):
    """One global active worker, one newest pending request, eight tiny summaries."""

    state_changed = Signal(object)
    progress_changed = Signal(object, str)
    result_ready = Signal(object, object)
    drained = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._generation = 0
        self._active: _Active | None = None
        self._pending: _Scheduled | None = None
        self._closing = False
        self._awaiting_drain = False
        self._publishing: JobIdentity | None = None
        self._publication_canceled = False
        self.summaries: deque[JobSummary] = deque(maxlen=MAX_SUMMARIES)
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._poll)

    @property
    def busy(self) -> bool:
        return self._active is not None or self._pending is not None

    @property
    def latest_generation(self) -> int:
        return self._generation

    def submit(self, request: JobRequest) -> JobIdentity:
        if self._closing:
            raise RuntimeError("Application close has been requested")
        if isinstance(request.argument, (MaskWork, LineWork)):
            request.argument.validate()
        if not isinstance(request.argument, (bytes, str, Path, memoryview, MaskWork, LineWork)):
            raise TypeError(
                "Request argument must be immutable bytes, text, path, byte view, mask or line work"
            )
        if not isinstance(request.run, FunctionType) or request.run.__closure__ is not None:
            raise TypeError("Worker must be a plain function without captured state")
        if not 0 <= request.argument_bytes <= MAX_REQUEST_BYTES:
            raise ValueError("Request exceeds the 4 MiB application limit")
        if _known_bytes(request.argument) > request.argument_bytes:
            raise ValueError("Request byte count understates its payload")
        if not 0 <= request.expected_result_bytes <= MAX_RESULT_BYTES:
            raise ValueError("Expected result exceeds the 96 MiB application limit")
        if isinstance(request.argument, memoryview):
            owned_bytes = request.argument.tobytes()
            request = replace(request, argument=owned_bytes, argument_bytes=len(owned_bytes))
        self._generation += 1
        identity = JobIdentity(
            request.project_id, request.acquisition_id, request.revisions, self._generation
        )
        old_pending = self._pending
        self._pending = _Scheduled(identity, request)
        if old_pending is not None:
            self._terminal(old_pending.identity, JobState.CANCELED, "Replaced before start")
        if self._pending is None or self._pending.identity != identity:
            return identity
        if self._active is not None:
            self._request_active_cancel()
        if self._pending is None or self._pending.identity != identity:
            return identity
        self.state_changed.emit(JobSummary(identity, JobState.QUEUED))
        if (
            self._active is None
            and self._pending is not None
            and self._pending.identity == identity
        ):
            self._start_pending()
        return identity

    def cancel(self) -> None:
        if self._publishing is not None:
            self._publication_canceled = True
        pending, self._pending = self._pending, None
        active = self._active
        canceled_active = None
        if active is not None and not active.control.canceled:
            active.cancel_at = perf_counter()
            active.control.cancel()
            canceled_active = JobSummary(active.scheduled.identity, JobState.CANCEL_REQUESTED)
        if pending is not None:
            self._terminal(pending.identity, JobState.CANCELED, "Canceled before start")
        if canceled_active is not None:
            self.state_changed.emit(canceled_active)

    def invalidate(self) -> None:
        """Reject prior publications when selection or input revisions change."""
        self._generation += 1
        self.cancel()

    def request_close(self) -> bool:
        self._closing = True
        self.cancel()
        self._awaiting_drain = self.busy
        return not self._awaiting_drain

    def _request_active_cancel(self) -> None:
        active = self._active
        if active is not None and not active.control.canceled:
            active.cancel_at = perf_counter()
            active.control.cancel()
            self.state_changed.emit(
                JobSummary(active.scheduled.identity, JobState.CANCEL_REQUESTED)
            )

    def _start_pending(self) -> None:
        scheduled, self._pending = self._pending, None
        if scheduled is None:
            return
        active = _Active(scheduled, JobControl())
        self._active = active

        def work() -> None:
            try:
                result = scheduled.request.run(scheduled.request.argument, active.control)
                if not isinstance(result, JobResult):
                    raise TypeError("Worker must return a JobResult")
                if not 0 <= result.resident_bytes <= MAX_RESULT_BYTES:
                    raise ValueError("Result exceeds the 96 MiB application limit")
                if result.resident_bytes > scheduled.request.expected_result_bytes:
                    raise ValueError("Result exceeds its declared request budget")
                if _known_bytes(result.value) > result.resident_bytes:
                    raise ValueError("Result byte count understates its payload")
                active.outcome = (result, None)
            except Exception as error:
                active.outcome = (None, f"{type(error).__name__}: {error}"[:160])

        active.thread = threading.Thread(target=work, name="slate-background-job")
        active.thread.start()
        self.state_changed.emit(JobSummary(scheduled.identity, JobState.RUNNING))
        self._timer.start()

    def _terminal(
        self,
        identity: JobIdentity,
        state: JobState,
        detail: str = "",
        safe_stop_ms: float | None = None,
    ) -> None:
        summary = JobSummary(identity, state, detail[:160], safe_stop_ms)
        self.summaries.append(summary)
        self.state_changed.emit(summary)

    def _poll(self) -> None:
        active = self._active
        if active is None or active.thread is None:
            self._timer.stop()
            return
        if not active.control.canceled:
            progress = active.control.take_progress()
            if progress is not None and active.scheduled.identity.generation == self._generation:
                self.progress_changed.emit(active.scheduled.identity, progress)
        if active.thread.is_alive():
            return
        active.thread.join(timeout=0)
        self._active = None
        identity = active.scheduled.identity
        result, error = active.outcome or (None, "Worker ended without an outcome")
        safe_stop_ms = (
            (perf_counter() - active.cancel_at) * 1000 if active.cancel_at is not None else None
        )
        if active.control.canceled or identity.generation != self._generation or self._closing:
            self._terminal(identity, JobState.CANCELED, safe_stop_ms=safe_stop_ms)
        elif error is not None:
            self._terminal(identity, JobState.FAILED, error)
        else:
            self._publishing = identity
            self._publication_canceled = False
            try:
                self._terminal(identity, JobState.COMPLETED)
                if (
                    not self._closing
                    and not self._publication_canceled
                    and identity.generation == self._generation
                ):
                    self.result_ready.emit(identity, result.value)
            finally:
                self._publishing = None
        if self._closing:
            self._timer.stop()
            if self._awaiting_drain:
                self._awaiting_drain = False
                self.drained.emit()
        elif self._active is not None:
            return
        elif self._pending is not None:
            self._start_pending()
        else:
            self._timer.stop()
