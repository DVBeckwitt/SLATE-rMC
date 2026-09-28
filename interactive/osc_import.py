"""Bounded OSC import preparation for the desktop worker."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from detector_panel import BandProfiles, exact_band_profiles
from job_lifecycle import JobControl, JobResult
from numpy.typing import NDArray
from project_state import linear_display_limits

SOURCE_LIMIT_BYTES = 64 * 1024 * 1024
DECODED_LIMIT_BYTES = 32 * 1024 * 1024
PIXEL_LIMIT = 12_000_000
AXIS_LIMIT = 16_384


def encode_bounded_path(path: Path, axis_limit: int) -> bytes:
    if not 0 < axis_limit <= AXIS_LIMIT:
        raise ValueError("OSC texture axis limit is outside the supported range")
    return str(axis_limit).encode("ascii") + b"\0" + os.fsencode(path)


def decode_bounded_path(argument: bytes) -> tuple[Path, int]:
    axis_bytes, separator, path_bytes = argument.partition(b"\0")
    if not separator or not path_bytes:
        raise ValueError("OSC request must include a texture limit and path")
    axis_limit = int(axis_bytes)
    if not 0 < axis_limit <= AXIS_LIMIT:
        raise ValueError("OSC texture axis limit is outside the supported range")
    return Path(os.fsdecode(path_bytes)), axis_limit


@dataclass(frozen=True, slots=True)
class PreparedOsc:
    source_path: Path
    decoded_sha256: str
    native_counts: NDArray[np.int32]
    display: NDArray[np.float32]
    profiles: BandProfiles
    low_value: float
    high_value: float
    max_value: float
    min_positive: float | None
    version: int
    byte_order: str
    raw_shape: tuple[int, int]


def prepare_osc(argument: bytes, control: JobControl) -> JobResult:
    """Decode and prepare one image; Qt widgets consume only the bounded result."""

    from rasim_next.io.osc import OscReadCancelled, OscReadLimits, read_osc

    path, axis_limit = decode_bounded_path(argument)
    control.report(f"Reading {path.name}")
    image = read_osc(
        path,
        limits=OscReadLimits(SOURCE_LIMIT_BYTES, DECODED_LIMIT_BYTES, PIXEL_LIMIT, axis_limit),
        canceled=lambda: control.canceled,
    )
    if control.canceled:
        raise OscReadCancelled("OSC import canceled")
    native = image.detector_native_counts
    display = native.astype(np.float32)
    display.setflags(write=False)
    minimum, maximum = float(np.min(native)), float(np.max(native))
    positive = native > 0
    min_positive = (
        float(np.min(native, where=positive, initial=np.iinfo(np.int32).max))
        if np.any(positive)
        else None
    )
    del positive
    profiles = exact_band_profiles(
        native, column_px=native.shape[1] // 2, row_px=native.shape[0] // 2
    )
    if control.canceled:
        raise OscReadCancelled("OSC import canceled")
    assert image.decoded_sha256 is not None
    low_value, high_value = linear_display_limits(minimum, maximum)
    prepared = PreparedOsc(
        path,
        image.decoded_sha256,
        native,
        display,
        profiles,
        low_value,
        high_value,
        maximum,
        min_positive,
        image.metadata.version,
        image.metadata.byte_order,
        image.metadata.raw_shape,
    )
    resident_bytes = (
        native.nbytes
        + display.nbytes
        + sum(
            array.nbytes
            for array in (
                profiles.horizontal,
                profiles.vertical,
                profiles.horizontal_support,
                profiles.vertical_support,
            )
        )
        + len(image.metadata.header)
    )
    control.report("Detector image ready")
    return JobResult(prepared, resident_bytes)
