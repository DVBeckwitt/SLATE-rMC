"""Minimal Rigaku RAXIS ``.osc`` reader used by SLATE-rMC runtime workflows."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np

RAXIS_SIGNATURE = b"RAXIS"
RAXIS_HEADER_BYTES = 6000


class ShapeError(Exception):
    """Raised when an OSC payload cannot be reshaped to its declared size."""


def read_osc(
    filename: str | PathLike[str],
) -> np.ndarray:
    """Read one OSC file and return its interpreted detector image."""

    path = Path(filename)
    file_size = path.stat().st_size
    if file_size < RAXIS_HEADER_BYTES:
        raise ShapeError(
            f"OSC file is shorter than its {RAXIS_HEADER_BYTES}-byte header."
        )
    with path.open("rb") as handle:
        header = handle.read(RAXIS_HEADER_BYTES)
    if header[: len(RAXIS_SIGNATURE)] != RAXIS_SIGNATURE:
        raise IOError(
            "This file does not start with the expected 'RAXIS' signature."
        )

    version = int.from_bytes(header[796:800], byteorder="big", signed=False)
    endian = ">" if version < 20 else "<"
    byteorder = "big" if endian == ">" else "little"
    width = int.from_bytes(header[768:772], byteorder=byteorder, signed=False)
    height = int.from_bytes(header[772:776], byteorder=byteorder, signed=False)
    if width <= 0 or height <= 0:
        raise ShapeError(f"OSC header declares invalid dimensions {height}x{width}.")
    pixel_count = width * height
    expected_size = RAXIS_HEADER_BYTES + pixel_count * np.dtype("u2").itemsize
    if file_size != expected_size:
        raise ShapeError(
            f"OSC file size {file_size} does not match declared {height}x{width} "
            f"payload size {expected_size}."
        )
    pixel_data = np.fromfile(
        path,
        dtype=endian + "u2",
        count=pixel_count,
        offset=RAXIS_HEADER_BYTES,
    )
    if pixel_data.size != pixel_count:
        raise ShapeError(
            f"Could not read the declared {height}x{width} OSC pixel payload."
        )
    pixel_data = pixel_data.reshape((height, width))
    image = pixel_data.astype(np.int32, copy=False)
    signed_mask = image >= 0x8000
    image[signed_mask] -= 0x10000
    image[signed_mask] += 0x8000
    image[signed_mask] *= 32
    return image

