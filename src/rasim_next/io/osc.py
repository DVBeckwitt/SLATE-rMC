"""Rigaku RAXIS OSC decoding at the detector-native orientation boundary."""

from __future__ import annotations

import gzip
import hashlib
import os
from collections.abc import Callable
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


class OscReadCancelled(Exception):
    """Raised at a bounded read boundary after cancellation was requested."""


@dataclass(frozen=True, slots=True)
class OscReadLimits:
    """Admission limits for an interactive OSC read, in bytes and pixels."""

    source_bytes: int
    decoded_bytes: int
    pixels: int


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
    decoded_sha256: str | None = None


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


def _layout(header: bytes) -> tuple[OscMetadata, np.dtype, int]:
    if header[: len(_SIGNATURE)] != _SIGNATURE:
        raise OscFormatError("OSC file does not start with the RAXIS signature")
    version = int.from_bytes(header[796:800], byteorder="big", signed=False)
    byte_order: Literal["big", "little"] = "big" if version < 20 else "little"
    dtype = np.dtype(">u2" if byte_order == "big" else "<u2")
    width = int.from_bytes(header[768:772], byteorder=byte_order, signed=False)
    height = int.from_bytes(header[772:776], byteorder=byte_order, signed=False)
    if width <= 0 or height <= 0:
        raise OscFormatError(f"OSC header declares invalid dimensions {height}x{width}")
    return OscMetadata(version, byte_order, (height, width), header), dtype, height * width


def _bounded_bytes(
    path: Path, limits: OscReadLimits, canceled: Callable[[], bool] | None
) -> tuple[bytearray, str]:
    if min(limits.source_bytes, limits.decoded_bytes, limits.pixels) <= 0:
        raise ValueError("OSC read limits must be positive")
    before = path.stat()
    if before.st_size > limits.source_bytes:
        raise OscFormatError(f"OSC source exceeds the {limits.source_bytes} byte import limit")
    compressed = path.name.lower().endswith(".osc.gz")
    if not compressed and path.suffix.lower() != ".osc":
        raise ValueError("OSC input path must end in .osc or .osc.gz")

    def source_changed() -> bool:
        try:
            current = path.stat()
        except FileNotFoundError:
            return True
        return (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        )

    def check_cancel() -> None:
        if canceled is not None and canceled():
            raise OscReadCancelled("OSC import canceled")

    check_cancel()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as raw:
            opened = os.fstat(raw.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ):
                raise OscFormatError("OSC source changed during load")
            with gzip.GzipFile(fileobj=raw) if compressed else raw as stream:
                header = stream.read(_HEADER_BYTES)
                if len(header) < _HEADER_BYTES:
                    raise OscFormatError(
                        f"OSC file is shorter than its {_HEADER_BYTES}-byte header"
                    )
                metadata, dtype, pixels = _layout(header)
                expected = _HEADER_BYTES + pixels * dtype.itemsize
                if pixels > limits.pixels or expected > limits.decoded_bytes:
                    raise OscFormatError(
                        f"OSC {metadata.raw_shape[0]}x{metadata.raw_shape[1]} image "
                        f"exceeds the {limits.pixels} pixel / {limits.decoded_bytes} byte import limit"
                    )
                if not compressed and before.st_size != expected:
                    raise OscFormatError(
                        f"OSC byte length {before.st_size} does not match declared "
                        f"{metadata.raw_shape[0]}x{metadata.raw_shape[1]} length {expected}"
                    )
                content = bytearray(expected)
                content[:_HEADER_BYTES] = header
                digest.update(header)
                position = _HEADER_BYTES
                while position < expected:
                    check_cancel()
                    chunk = stream.read(min(1024 * 1024, expected - position))
                    if not chunk:
                        raise OscFormatError(
                            f"OSC byte length {position} does not match declared length {expected}"
                        )
                    content[position : position + len(chunk)] = chunk
                    digest.update(chunk)
                    position += len(chunk)
                check_cancel()
                if stream.read(1):
                    raise OscFormatError(f"OSC byte length exceeds declared length {expected}")
    except (OSError, EOFError, OscFormatError) as exc:
        if source_changed():
            raise OscFormatError("OSC source changed during load") from exc
        if isinstance(exc, OscFormatError):
            raise
        raise OscFormatError(f"cannot read OSC source {path.name}: {exc}") from exc
    if source_changed():
        raise OscFormatError("OSC source changed during load")
    check_cancel()
    return content, digest.hexdigest()


def read_osc(
    path: str | PathLike[str],
    *,
    limits: OscReadLimits | None = None,
    canceled: Callable[[], bool] | None = None,
) -> OscImage:
    """Decode one plain or gzip OSC file and apply the canonical orientation once."""

    source = Path(path)
    if limits is None:
        content = _read_bytes(source)
        digest = None
    else:
        content, digest = _bounded_bytes(source, limits, canceled)
    if len(content) < _HEADER_BYTES:
        raise OscFormatError(f"OSC file is shorter than its {_HEADER_BYTES}-byte header")

    header = bytes(content[:_HEADER_BYTES])
    metadata, dtype, pixel_count = _layout(header)
    height, width = metadata.raw_shape
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
    np.subtract(raw_counts, 0x8000, out=raw_counts, where=high_range)
    np.multiply(raw_counts, 32, out=raw_counts, where=high_range)
    detector_native_counts = raw_to_detector_native(raw_counts)
    raw_counts.setflags(write=False)
    detector_native_counts.setflags(write=False)

    return OscImage(
        metadata=metadata,
        raw_counts=raw_counts,
        detector_native_counts=detector_native_counts,
        decoded_sha256=digest,
    )
