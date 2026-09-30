"""Bounded native-pixel display sampling and immutable comparison references."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from detector_panel import BandProfiles
    from job_lifecycle import JobControl, JobResult

    from rasim_next.geometry.instrument import CompiledInstrument

MAX_LINE_SAMPLES = 8192


@dataclass(frozen=True, slots=True)
class LineDefinition:
    start_column_row: tuple[float, float]
    end_column_row: tuple[float, float]
    spacing_px: float = 1.0
    revision: int = 0

    def __post_init__(self) -> None:
        for point in (self.start_column_row, self.end_column_row):
            if (
                type(point) is not tuple
                or len(point) != 2
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 32768
                    for v in point
                )
            ):
                raise ValueError("Line endpoints need finite native (column,row) coordinates")
        if self.start_column_row == self.end_column_row:
            raise ValueError("Line endpoints must differ")
        if (
            type(self.spacing_px) not in (int, float)
            or not math.isfinite(self.spacing_px)
            or not 0.25 <= self.spacing_px <= 64
        ):
            raise ValueError("Line spacing must be between 0.25 and 64 native pixels")
        if type(self.revision) is not int or not 0 <= self.revision < 2**63:
            raise ValueError("Line revision must be a nonnegative signed 64-bit integer")
        if self.sample_count > MAX_LINE_SAMPLES:
            raise ValueError("Line exceeds 8192 samples; increase spacing or shorten the line")

    @property
    def length_px(self) -> float:
        return math.dist(self.start_column_row, self.end_column_row)

    @property
    def sample_count(self) -> int:
        return math.ceil(self.length_px / self.spacing_px) + 1


@dataclass(frozen=True, slots=True)
class LineSamples:
    definition: LineDefinition
    distance_px: np.ndarray
    column_px: np.ndarray
    row_px: np.ndarray
    nearest_column: np.ndarray
    nearest_row: np.ndarray
    values: np.ndarray
    support: np.ndarray

    @property
    def storage_bytes(self) -> int:
        return sum(
            a.nbytes
            for a in (
                self.distance_px,
                self.column_px,
                self.row_px,
                self.nearest_column,
                self.nearest_row,
                self.values,
                self.support,
            )
        )


@dataclass(frozen=True, slots=True)
class LineWork:
    native: np.ndarray
    inclusion: np.ndarray | None
    definition: LineDefinition

    @property
    def argument_bytes(self) -> int:
        return 256

    def validate(self) -> None:
        if (
            self.native.ndim != 2
            or not self.native.size
            or self.native.dtype.kind not in "iuf"
            or self.native.flags.writeable
            or not self.native.flags.c_contiguous
            or self.native.nbytes > 96 * 1024 * 1024
        ):
            raise ValueError("Line work needs a bounded immutable native plane")
        if self.inclusion is not None and (
            self.inclusion.shape != self.native.shape
            or self.inclusion.dtype != np.bool_
            or self.inclusion.flags.writeable
        ):
            raise ValueError("Line work inclusion must be immutable and aligned")
        if not isinstance(self.definition, LineDefinition):
            raise ValueError("Line work definition is invalid")


def prepare_line(work: LineWork, control: JobControl) -> JobResult:
    from job_lifecycle import JobResult

    work.validate()
    control.report("Sampling line")
    if control.canceled:
        raise ValueError("Line sampling canceled")
    sampled = sample_line(work.native, work.inclusion, work.definition)
    if control.canceled:
        raise ValueError("Line sampling canceled")
    return JobResult(sampled, sampled.storage_bytes)


def sample_line(
    native: np.ndarray, inclusion: np.ndarray | None, line: LineDefinition
) -> LineSamples:
    """Nearest pixel centers, ties toward the larger index; no strip integration."""
    if not isinstance(line, LineDefinition):
        raise ValueError("Line sampling needs a bounded native line definition")
    if native.ndim != 2 or not native.size or native.dtype.kind not in "iuf":
        raise ValueError("Line sampling needs a real native [row,column] plane")
    if inclusion is not None and (inclusion.dtype != np.bool_ or inclusion.shape != native.shape):
        raise ValueError("Line inclusion must match native shape")
    t = np.linspace(0, 1, line.sample_count, dtype=np.float64)
    a, b = line.start_column_row, line.end_column_row
    column = a[0] + t * (b[0] - a[0])
    row = a[1] + t * (b[1] - a[1])
    ci = np.floor(column + 0.5).astype(np.int64)
    ri = np.floor(row + 0.5).astype(np.int64)
    inside = (ci >= 0) & (ci < native.shape[1]) & (ri >= 0) & (ri < native.shape[0])
    values = (
        np.zeros(t.shape, dtype=native.dtype)
        if native.dtype.kind in "iu"
        else np.full(t.shape, np.nan, dtype=native.dtype)
    )
    selected = np.flatnonzero(inside)
    sampled = native[ri[selected], ci[selected]]
    valid = np.isfinite(sampled)
    if inclusion is not None:
        valid &= inclusion[ri[selected], ci[selected]]
    values[selected[valid]] = sampled[valid]
    support = np.zeros(t.shape, dtype=np.int64)
    support[selected[valid]] = 1
    arrays = (t * line.length_px, column, row, ci, ri, values, support)
    for array in arrays:
        array.setflags(write=False)
    return LineSamples(line, *arrays)


@dataclass(frozen=True, slots=True)
class PinIdentity:
    acquisition_id: UUID
    source_sha256: str
    native_shape_rc: tuple[int, int]
    mask_revision: int
    query: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.acquisition_id, UUID):
            raise ValueError("Pinned profile needs an acquisition UUID")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("Pinned profile needs a decoded source SHA-256")
        if (
            type(self.native_shape_rc) is not tuple
            or len(self.native_shape_rc) != 2
            or any(type(v) is not int or not 0 < v <= 16384 for v in self.native_shape_rc)
            or math.prod(self.native_shape_rc) > 12_000_000
        ):
            raise ValueError("Pinned profile shape is invalid")
        if type(self.mask_revision) is not int or not 0 <= self.mask_revision < 2**63:
            raise ValueError("Pinned profile mask revision is invalid")
        q = self.query
        if (
            type(q) is not tuple
            or len(q) != 7
            or any(type(v) is not int for v in q[:4])
            or not 0 <= q[0] < self.native_shape_rc[1]
            or not 0 <= q[1] < self.native_shape_rc[0]
            or not 1 <= min(q[2:4]) <= max(q[2:4]) <= 16384
            or q[4] not in ("band", "full", "roi")
            or q[6] not in ("sum", "mean")
        ):
            raise ValueError("Pinned profile settings are invalid")
        roi = q[5]
        if roi is not None and (
            type(roi) is not tuple
            or len(roi) != 4
            or any(type(v) is not int for v in roi)
            or not 0 <= roi[0] < roi[1] <= self.native_shape_rc[1]
            or not 0 <= roi[2] < roi[3] <= self.native_shape_rc[0]
        ):
            raise ValueError("Pinned ROI is outside native shape")
        if q[4] == "roi" and roi is None:
            raise ValueError("Pinned ROI scope needs explicit bounds")


@dataclass(frozen=True, slots=True)
class PinnedProfiles:
    identity: PinIdentity
    horizontal: np.ndarray
    vertical: np.ndarray
    horizontal_support: np.ndarray
    vertical_support: np.ndarray

    @classmethod
    def capture(cls, identity: PinIdentity, profiles: BandProfiles) -> PinnedProfiles:
        arrays = tuple(
            np.array(a, copy=True)
            for a in (
                profiles.horizontal,
                profiles.vertical,
                profiles.horizontal_support,
                profiles.vertical_support,
            )
        )
        rows, columns = identity.native_shape_rc
        if (
            any(a.ndim != 1 for a in arrays)
            or tuple(a.size for a in arrays) != (columns, rows, columns, rows)
            or any(a.dtype.kind not in "iuf" for a in arrays[:2])
            or any(a.dtype != np.int64 or np.any(a < 0) for a in arrays[2:])
        ):
            raise ValueError("Pinned values and support must match the native vectors")
        for a in arrays:
            a.setflags(write=False)
        return cls(identity, *arrays)

    @property
    def storage_bytes(self) -> int:
        return sum(
            a.nbytes
            for a in (
                self.horizontal,
                self.vertical,
                self.horizontal_support,
                self.vertical_support,
            )
        )


def detector_frame_key(instrument: CompiledInstrument) -> str:
    """Identity of the canonical calibrated detector lattice in LAB metres."""
    digest = hashlib.sha256(b"native-calibrated-detector-lab.v1")
    values = (
        instrument.detector_shape_rc,
        instrument.detector_row_pitch_m,
        instrument.detector_column_pitch_m,
        instrument.detector_reference_coordinate_px,
        instrument.lab_from_detector.rotation,
        instrument.lab_from_detector.translation_m,
    )
    for value in values:
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()
