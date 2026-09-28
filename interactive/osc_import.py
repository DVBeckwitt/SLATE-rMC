"""Bounded OSC import preparation for the desktop worker."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from detector_panel import BandProfiles, exact_band_profiles
from job_lifecycle import JobControl, JobResult
from numpy.typing import NDArray

SOURCE_LIMIT_BYTES = 64 * 1024 * 1024
DECODED_LIMIT_BYTES = 32 * 1024 * 1024
PIXEL_LIMIT = 12_000_000


@dataclass(frozen=True, slots=True)
class PreparedOsc:
    source_path: Path
    decoded_sha256: str
    native_counts: NDArray[np.int32]
    display: NDArray[np.float32]
    profiles: BandProfiles
    low_value: float
    high_value: float
    version: int
    byte_order: str
    raw_shape: tuple[int, int]


def prepare_osc(argument: Path, control: JobControl) -> JobResult:
    """Decode and prepare one image; Qt widgets consume only the bounded result."""

    from rasim_next.io.osc import OscReadCancelled, OscReadLimits, read_osc

    path = Path(argument)
    control.report(f"Reading {path.name}")
    image = read_osc(
        path,
        limits=OscReadLimits(SOURCE_LIMIT_BYTES, DECODED_LIMIT_BYTES, PIXEL_LIMIT),
        canceled=lambda: control.canceled,
    )
    if control.canceled:
        raise OscReadCancelled("OSC import canceled")
    native = image.detector_native_counts
    display = native.astype(np.float32)
    display.setflags(write=False)
    minimum, maximum = float(np.min(native)), float(np.max(native))
    profiles = exact_band_profiles(
        native, column_px=native.shape[1] // 2, row_px=native.shape[0] // 2
    )
    if control.canceled:
        raise OscReadCancelled("OSC import canceled")
    assert image.decoded_sha256 is not None
    prepared = PreparedOsc(
        path,
        image.decoded_sha256,
        native,
        display,
        profiles,
        minimum,
        max(maximum, minimum + 1.0),
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
