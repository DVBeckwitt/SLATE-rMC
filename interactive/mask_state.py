"""Bounded detector-native exclusions; no data correction or automatic censoring."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np

REASONS = ("included", "user exclusion", "beamstop", "detector gap", "saturation")
MAX_SPANS = 8192
MAX_POINTS = 2048
MAX_ACTIONS = 64
HISTORY_BYTES = 512 * 1024
HISTORY_COUNT = 32
IMPORT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class NativeMask:
    source_sha256: str
    shape: tuple[int, int]
    revision: int = 0
    spans: tuple[tuple[int, int, int, int], ...] = ()
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("Mask needs the decoded OSC SHA-256")
        if (
            type(self.shape) is not tuple
            or len(self.shape) != 2
            or any(type(v) is not int or not 0 < v <= 16384 for v in self.shape)
            or math.prod(self.shape) > 12_000_000
        ):
            raise ValueError("Mask shape must be a bounded native (row, column) shape")
        if type(self.revision) is not int or not 0 <= self.revision < 2**63:
            raise ValueError("Mask revision must be a nonnegative signed 64-bit integer")
        if type(self.spans) is not tuple or len(self.spans) > MAX_SPANS:
            raise ValueError(f"Mask exceeds {MAX_SPANS} native row spans")
        previous = None
        for span in self.spans:
            if type(span) is not tuple or len(span) != 4 or any(type(v) is not int for v in span):
                raise ValueError("Mask spans must be integer (row, start, stop, reason)")
            r, start, stop, reason = span
            if not (
                0 <= r < self.shape[0]
                and 0 <= start < stop <= self.shape[1]
                and 0 < reason < len(REASONS)
            ):
                raise ValueError("Mask span exceeds native shape or reason domain")
            if previous is not None:
                pr, _ps, pe, pv = previous
                if r < pr or (r == pr and (start < pe or (start == pe and reason == pv))):
                    raise ValueError("Mask spans must be canonical, sorted and disjoint")
            previous = span
        if (
            type(self.provenance) is not tuple
            or len(self.provenance) > 32
            or any(type(v) is not str or not 0 < len(v) <= 512 for v in self.provenance)
        ):
            raise ValueError("Mask provenance needs at most 32 bounded descriptions")

    @property
    def storage_bytes(self) -> int:
        return (
            200 * len(self.spans) + sum(len(v.encode("utf-8")) + 64 for v in self.provenance) + 512
        )


def mask_document(mask: NativeMask | None) -> dict | None:
    if mask is None:
        return None
    return {
        "source_sha256": mask.source_sha256,
        "shape": list(mask.shape),
        "revision": mask.revision,
        "spans": [list(s) for s in mask.spans],
        "provenance": list(mask.provenance),
    }


def mask_from_document(value: object) -> NativeMask | None:
    if value is None:
        return None
    if type(value) is not dict or set(value) != {
        "source_sha256",
        "shape",
        "revision",
        "spans",
        "provenance",
    }:
        raise ValueError("Unsupported mask document")
    if (
        type(value["shape"]) is not list
        or type(value["spans"]) is not list
        or type(value["provenance"]) is not list
        or len(value["spans"]) > MAX_SPANS
        or any(type(s) is not list for s in value["spans"])
    ):
        raise ValueError("Invalid native mask arrays")
    return NativeMask(
        value["source_sha256"],
        tuple(value["shape"]),
        value["revision"],
        tuple(tuple(s) for s in value["spans"]),
        tuple(value["provenance"]),
    )


@dataclass(frozen=True, slots=True)
class MaskGesture:
    kind: str
    reason: int
    points: tuple[tuple[float, float], ...] = ()
    radius_px: float = 8.0
    import_path: Path | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("rectangle", "polygon", "brush", "import"):
            raise ValueError("Unsupported mask tool")
        if type(self.reason) is not int or not 0 <= self.reason < len(REASONS):
            raise ValueError("Unsupported mask reason")
        if (
            type(self.points) is not tuple
            or len(self.points) > MAX_POINTS
            or any(
                type(p) is not tuple
                or len(p) != 2
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 32768 for v in p
                )
                for p in self.points
            )
        ):
            raise ValueError("Mask points must be finite native (column, row) coordinates")
        if (
            type(self.radius_px) not in (int, float)
            or not math.isfinite(self.radius_px)
            or not 0.5 <= self.radius_px <= 256
        ):
            raise ValueError("Brush radius must be between 0.5 and 256 native pixels")
        if self.kind == "import":
            if not isinstance(self.import_path, Path) or self.points or self.reason == 0:
                raise ValueError("Import needs one Boolean native .npy file")
        elif (
            self.import_path is not None
            or len(self.points) < {"rectangle": 2, "polygon": 3, "brush": 1}[self.kind]
        ):
            raise ValueError("Mask gesture is incomplete")
        if self.kind == "rectangle" and (
            len(self.points) != 2
            or self.points[0][0] == self.points[1][0]
            or self.points[0][1] == self.points[1][1]
        ):
            raise ValueError("Rectangle must have nonzero width and height")
        if self.kind == "polygon":
            area = sum(
                a[0] * b[1] - b[0] * a[1]
                for a, b in zip(self.points, self.points[1:] + self.points[:1], strict=True)
            )
            if abs(area) < 1e-9:
                raise ValueError("Polygon must have nonzero signed area")


def _spans(plane: np.ndarray, canceled) -> tuple[tuple[int, int, int, int], ...]:
    spans = []
    rows, columns = plane.shape
    for first in range(0, rows, 64):
        if canceled():
            raise ValueError("Mask preparation canceled")
        block = plane[first : first + 64]
        edges = np.empty((block.shape[0], columns + 1), dtype=np.bool_)
        edges[:, 0] = block[:, 0] != 0
        edges[:, 1:columns] = block[:, 1:] != block[:, :-1]
        edges[:, columns] = block[:, -1] != 0
        edge_rows, edge_columns = np.nonzero(edges)
        for index in range(len(edge_rows) - 1):
            row, start = int(edge_rows[index]), int(edge_columns[index])
            if start == columns:
                continue
            reason = int(block[row, start])
            if reason:
                spans.append((row + first, start, int(edge_columns[index + 1]), reason))
                if len(spans) > MAX_SPANS:
                    raise ValueError(f"Mask exceeds {MAX_SPANS} row spans; simplify the exclusion")
    return tuple(spans)


def rasterize_gesture(plane: np.ndarray, gesture: MaskGesture, canceled=lambda: False) -> str:
    """Assign reasons at native pixel centers. Reinclude (0) clears the prior reason."""
    rows, columns = plane.shape
    if gesture.kind == "import":
        path = gesture.import_path
        if path.suffix.lower() != ".npy" or path.stat().st_size > IMPORT_BYTES:
            raise ValueError("Mask import supports only native Boolean .npy files up to 16 MiB")
        inclusion = np.load(path, mmap_mode="r", allow_pickle=False, max_header_size=4096)
        if (
            not isinstance(inclusion, np.ndarray)
            or inclusion.dtype != np.bool_
            or inclusion.shape != plane.shape
            or not inclusion.flags.c_contiguous
        ):
            raise ValueError(
                "Imported mask must be C-order Boolean [row,column], matching native shape; True includes"
            )
        for r0 in range(0, rows, 32):
            if canceled():
                raise ValueError("Mask import canceled")
            block = plane[r0 : r0 + 32]
            block[~inclusion[r0 : r0 + 32]] = gesture.reason
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return f"Boolean native .npy True=include; {path.name[:120]}; sha256={digest}; additive {REASONS[gesture.reason]}"
    points = gesture.points
    pad = gesture.radius_px if gesture.kind == "brush" else 0
    c0 = max(0, math.ceil(min(p[0] for p in points) - pad))
    c1 = min(columns, math.floor(max(p[0] for p in points) + pad) + 1)
    r0 = max(0, math.ceil(min(p[1] for p in points) - pad))
    r1 = min(rows, math.floor(max(p[1] for p in points) + pad) + 1)
    if c0 >= c1 or r0 >= r1:
        return f"{gesture.kind}: {REASONS[gesture.reason]}"
    x = np.arange(c0, c1, dtype=np.float64)[None, :]
    for start in range(r0, r1, 16):
        if canceled():
            raise ValueError("Mask rasterization canceled")
        stop = min(start + 16, r1)
        y = np.arange(start, stop, dtype=np.float64)[:, None]
        selected = np.zeros((stop - start, c1 - c0), dtype=np.bool_)
        if gesture.kind == "rectangle":
            selected[:] = True
        elif gesture.kind == "polygon":
            for a, b in zip(points, points[1:] + points[:1], strict=True):
                if a[1] != b[1]:
                    selected ^= ((a[1] > y) != (b[1] > y)) & (
                        x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
                    )
        else:
            segments = pairwise(points) if len(points) > 1 else [(points[0], points[0])]
            for a, b in segments:
                dx, dy = b[0] - a[0], b[1] - a[1]
                length2 = dx * dx + dy * dy
                t = np.clip(((x - a[0]) * dx + (y - a[1]) * dy) / length2, 0, 1) if length2 else 0
                selected |= (x - a[0] - t * dx) ** 2 + (
                    y - a[1] - t * dy
                ) ** 2 <= gesture.radius_px**2
        plane[start:stop, c0:c1][selected] = gesture.reason
    return f"{gesture.kind}: {REASONS[gesture.reason]}"


@dataclass(frozen=True, slots=True)
class MaskWork:
    native: np.ndarray
    mask: NativeMask
    gestures: tuple[MaskGesture, ...]
    query: tuple
    prepared: PreparedMask | None = None

    @property
    def argument_bytes(self) -> int:
        return self.mask.storage_bytes + sum(
            256 + 128 * len(g.points) + (len(str(g.import_path)) if g.import_path else 0)
            for g in self.gestures
        )

    def validate(self) -> None:
        if (
            self.native.shape != self.mask.shape
            or self.native.flags.writeable
            or not self.native.flags.c_contiguous
            or self.native.dtype.kind not in "iuf"
            or self.native.nbytes > 96 * 1024 * 1024
            or type(self.gestures) is not tuple
            or len(self.gestures) > MAX_ACTIONS
            or any(not isinstance(g, MaskGesture) for g in self.gestures)
        ):
            raise ValueError("Mask preparation needs a bounded immutable native plane and gestures")
        if (
            type(self.query) is not tuple
            or len(self.query) != 7
            or any(type(v) is not int for v in self.query[:4])
            or not 0 <= self.query[0] < self.mask.shape[1]
            or not 0 <= self.query[1] < self.mask.shape[0]
            or min(self.query[2:4]) < 1
            or max(self.query[2:4]) > 16384
            or self.query[4] not in ("band", "full", "roi")
            or self.query[6] not in ("sum", "mean")
        ):
            raise ValueError("Mask profile query is outside the native domain")


@dataclass(frozen=True, slots=True)
class PreparedMask:
    mask: NativeMask
    reasons: np.ndarray
    inclusion: np.ndarray
    full_profiles: object
    profiles: object
    query: tuple
    edits: tuple[tuple[np.ndarray, np.ndarray], ...]
    phases_ms: tuple[float, float]
    base_reasons_id: int | None = None
    changed_bounds: tuple[int, int, int, int] | None = None


def prepare_mask(work: MaskWork, control) -> object:
    from detector_panel import exact_band_profiles
    from job_lifecycle import JobResult

    work.validate()
    control.report("Preparing profiles")
    began = perf_counter()
    reuse = work.prepared is not None and work.prepared.mask == work.mask and not work.gestures
    if reuse:
        reasons = work.prepared.reasons
    else:
        reasons = np.zeros(work.mask.shape, dtype=np.uint8)
        for r, start, stop, reason in work.mask.spans:
            reasons[r, start:stop] = reason
    state, edits = work.mask, []
    for gesture in work.gestures:
        provenance = rasterize_gesture(reasons, gesture, lambda: control.canceled)
        spans = _spans(reasons, lambda: control.canceled)
        if spans != state.spans:
            updated = replace(
                state,
                revision=state.revision + 1,
                spans=spans,
                provenance=(*state.provenance, provenance)[-32:],
            )
            before = np.asarray(state.spans, dtype=np.int32).reshape(-1, 4)
            after = np.asarray(updated.spans, dtype=np.int32).reshape(-1, 4)
            before.setflags(write=False)
            after.setflags(write=False)
            edits.append((before, after))
            state = updated
    if work.prepared is not None and state == work.prepared.mask:
        reuse = True
        reasons = work.prepared.reasons
    inclusion = work.prepared.inclusion if reuse else reasons == 0
    base_id, changed_bounds = None, None
    if work.prepared is not None and reasons is not work.prepared.reasons:
        base_id = id(work.prepared.reasons)
        changed = reasons != work.prepared.reasons
        changed_rows = np.flatnonzero(np.any(changed, axis=1))
        changed_columns = np.flatnonzero(np.any(changed, axis=0))
        if changed_rows.size:
            changed_bounds = (
                int(changed_columns[0]),
                int(changed_columns[-1]) + 1,
                int(changed_rows[0]),
                int(changed_rows[-1]) + 1,
            )
        del changed
    raster_end = perf_counter()
    if control.canceled:
        raise ValueError("Profile preparation canceled")
    column, row, row_width, column_width, scope, roi, measure = work.query
    full = (
        work.prepared.full_profiles
        if reuse
        else exact_band_profiles(
            work.native, column_px=column, row_px=row, scope="full", mask=inclusion
        )
    )
    if control.canceled:
        raise ValueError("Profile preparation canceled")
    profiles = exact_band_profiles(
        work.native,
        column_px=column,
        row_px=row,
        row_width=row_width,
        column_width=column_width,
        scope=scope,
        roi_column_row_bounds=roi,
        measure=measure,
        mask=inclusion,
    )
    for a in (
        reasons,
        inclusion,
        full.horizontal,
        full.vertical,
        full.horizontal_support,
        full.vertical_support,
        profiles.horizontal,
        profiles.vertical,
        profiles.horizontal_support,
        profiles.vertical_support,
    ):
        a.setflags(write=False)
    result = PreparedMask(
        state,
        reasons,
        inclusion,
        full,
        profiles,
        work.query,
        tuple(edits),
        ((raster_end - began) * 1000, (perf_counter() - raster_end) * 1000),
        base_id,
        changed_bounds,
    )
    arrays = (
        reasons,
        inclusion,
        full.horizontal,
        full.vertical,
        full.horizontal_support,
        full.vertical_support,
        profiles.horizontal,
        profiles.vertical,
        profiles.horizontal_support,
        profiles.vertical_support,
    )
    return JobResult(
        result,
        sum(a.nbytes for a in arrays)
        + sum(a.nbytes + b.nbytes for a, b in edits)
        + state.storage_bytes,
    )


class MaskHistory:
    """Sparse immutable checkpoints: 32 actions / 512 KiB per acquisition."""

    def __init__(self) -> None:
        self.undo: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self.redo: deque[tuple[np.ndarray, np.ndarray]] = deque()

    @property
    def storage_bytes(self) -> int:
        return sum(a.nbytes + b.nbytes + 256 for a, b in (*self.undo, *self.redo))

    def push(self, before: np.ndarray, after: np.ndarray) -> None:
        before, after = (
            np.array(a, dtype=np.int32, copy=True) if a.flags.writeable else a
            for a in (before, after)
        )
        before.setflags(write=False)
        after.setflags(write=False)
        if np.array_equal(before, after):
            return
        self.redo.clear()
        self.undo.append((before, after))
        while len(self.undo) > HISTORY_COUNT or self.storage_bytes > HISTORY_BYTES:
            self.undo.popleft()

    def move(self, current: NativeMask, *, undo: bool) -> NativeMask:
        source, destination = (self.undo, self.redo) if undo else (self.redo, self.undo)
        if not source:
            return current
        before, after = source[-1]
        target = before if undo else after
        updated = replace(
            current,
            spans=tuple(tuple(int(v) for v in span) for span in target),
            revision=current.revision + 1,
            provenance=(*current.provenance, "Undo" if undo else "Redo")[-32:],
        )
        source.pop()
        destination.append((before, after))
        return updated
