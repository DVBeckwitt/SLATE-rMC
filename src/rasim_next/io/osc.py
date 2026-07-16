"""Rigaku RAXIS OSC decoding at the detector-native orientation boundary."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rasim_next.io.orientation import raw_to_detector_native

_HEADER_BYTES = 6000
_SIGNATURE = b"RAXIS"


class OscFormatError(ValueError):
    """Raised when an OSC byte stream violates the tracked RAXIS layout."""


@dataclass(frozen=True, slots=True)
class OscMetadata:
    """Raw OSC header facts retained without interpreting detector coordinates."""

    version: int
    byte_order: Literal["big", "little"]
    raw_shape: tuple[int, int]
    header: bytes


@dataclass(frozen=True, slots=True)
class OscImage:
    """One decoded OSC image in distinct raw and detector-native arrays."""

    metadata: OscMetadata
    raw_counts: NDArray[np.int32]
    detector_native_counts: NDArray[np.int32]


def _read_bytes(path: Path) -> bytes:
    name = path.name.lower()
    if name.endswith(".osc.gz"):
        try:
            with gzip.open(path, "rb") as handle:
                return handle.read()
        except (OSError, EOFError) as exc:
            raise OscFormatError(f"cannot decode gzip OSC file {path}") from exc
    if path.suffix.lower() == ".osc":
        return path.read_bytes()
    raise ValueError("OSC input path must end in .osc or .osc.gz")


def read_osc(path: str | PathLike[str]) -> OscImage:
    """Decode one plain or gzip OSC file and apply the canonical orientation once."""

    source = Path(path)
    content = _read_bytes(source)
    if len(content) < _HEADER_BYTES:
        raise OscFormatError(f"OSC file is shorter than its {_HEADER_BYTES}-byte header")

    header = content[:_HEADER_BYTES]
    if header[: len(_SIGNATURE)] != _SIGNATURE:
        raise OscFormatError("OSC file does not start with the RAXIS signature")

    version = int.from_bytes(header[796:800], byteorder="big", signed=False)
    byte_order: Literal["big", "little"] = "big" if version < 20 else "little"
    dtype = np.dtype(">u2" if byte_order == "big" else "<u2")
    width = int.from_bytes(header[768:772], byteorder=byte_order, signed=False)
    height = int.from_bytes(header[772:776], byteorder=byte_order, signed=False)
    if width <= 0 or height <= 0:
        raise OscFormatError(f"OSC header declares invalid dimensions {height}x{width}")

    pixel_count = height * width
    expected_bytes = _HEADER_BYTES + pixel_count * dtype.itemsize
    if len(content) != expected_bytes:
        raise OscFormatError(
            f"OSC byte length {len(content)} does not match declared "
            f"{height}x{width} length {expected_bytes}"
        )

    encoded = np.frombuffer(
        content,
        dtype=dtype,
        count=pixel_count,
        offset=_HEADER_BYTES,
    )
    raw_counts = encoded.astype(np.int32).reshape(height, width)
    high_range = raw_counts >= 0x8000
    raw_counts[high_range] = (raw_counts[high_range] - 0x8000) * 32
    detector_native_counts = raw_to_detector_native(raw_counts)
    raw_counts.setflags(write=False)
    detector_native_counts.setflags(write=False)

    return OscImage(
        metadata=OscMetadata(version, byte_order, (height, width), header),
        raw_counts=raw_counts,
        detector_native_counts=detector_native_counts,
    )
