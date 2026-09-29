"""Exact inspection CSV and bounded paired figure publication."""

import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from job_lifecycle import MAX_REQUEST_BYTES, JobControl, JobResult
from project_state import ProjectFormatError

MAX_PROFILE_CSV_BYTES = 3 * 1024 * 1024
MAX_EXPORT_HEADER_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class InspectionExportReceipt:
    figure: Path
    profiles: Path
    figure_sha256: str
    profiles_sha256: str


def external_export_destination(
    destination: Path,
    project_path: Path | None,
    input_paths: tuple[Path, ...],
    *,
    suffix: str,
) -> Path:
    """Resolve an external target without aliasing Git, project, or input files."""

    path = Path(destination).absolute()
    if path.suffix.lower() != suffix:
        path = path.with_suffix(suffix)
    if len(str(path)) > 4096:
        raise ProjectFormatError("export path exceeds 4096 characters")
    resolved = path.resolve(strict=False)
    for candidate in (path, resolved):
        if any((parent / ".git").exists() for parent in candidate.parents):
            raise ProjectFormatError("choose an export destination outside Git checkouts")
    for protected in (project_path, *input_paths):
        if protected is None:
            continue
        source = Path(protected).absolute()
        if resolved == source.resolve(strict=False) or (
            path.exists() and source.exists() and os.path.samefile(path, source)
        ):
            raise ProjectFormatError("export would overwrite a project or input reference")
    return path


def profile_csv(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    horizontal_support: np.ndarray,
    vertical_support: np.ndarray,
    metadata: dict[str, str],
) -> bytes:
    """Write each canonical native bin once, with its measure and support."""

    if (
        horizontal.ndim != 1
        or vertical.ndim != 1
        or horizontal.shape != horizontal_support.shape
        or vertical.shape != vertical_support.shape
        or horizontal.dtype not in (np.int64, np.float64)
        or vertical.dtype not in (np.int64, np.float64)
        or horizontal_support.dtype != np.int64
        or vertical_support.dtype != np.int64
        or np.any(horizontal_support < 0)
        or np.any(vertical_support < 0)
    ):
        raise ValueError("inspection profiles must be aligned exact native vectors")
    measure = metadata.get("profile_measure")
    if measure not in ("sum", "mean"):
        raise ValueError("inspection profile measure is invalid")
    if measure == "sum" and (horizontal.dtype != vertical.dtype):
        raise ValueError("sum vectors must share one numeric kind")
    if measure == "mean" and (horizontal.dtype != np.float64 or vertical.dtype != np.float64):
        raise ValueError("mean vectors must be float64")

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "record_type",
            "axis",
            "native_index_px",
            "value",
            "support",
            "missing",
            "unit",
            "field",
            "metadata_value",
        )
    )
    for key, value in metadata.items():
        writer.writerow(("metadata", "", "", "", "", "", "", key, value))
    unit = "counts" if measure == "sum" else "counts_per_valid_pixel"
    for axis, values, support in (
        ("horizontal_column", horizontal, horizontal_support),
        ("vertical_row", vertical, vertical_support),
    ):
        for index, (value, count) in enumerate(zip(values, support, strict=True)):
            numeric = str(int(value)) if values.dtype == np.int64 else format(float(value), ".17g")
            writer.writerow(
                ("profile", axis, index, numeric, int(count), int(count == 0), unit, "", "")
            )
    encoded = output.getvalue().encode("utf-8")
    if len(encoded) > MAX_PROFILE_CSV_BYTES:
        raise ProjectFormatError("inspection profile CSV exceeds 3 MiB")
    return encoded


def inspection_export_request(
    figure: Path,
    profiles: Path,
    figure_png: bytes,
    profiles_csv: bytes,
    project_path: Path | None,
    input_paths: tuple[Path, ...],
) -> bytes:
    header = json.dumps(
        {
            "figure": str(figure),
            "profiles": str(profiles),
            "project": str(project_path) if project_path is not None else None,
            "inputs": [str(path) for path in input_paths],
            "figure_bytes": len(figure_png),
            "profiles_bytes": len(profiles_csv),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    if not figure_png.startswith(b"\x89PNG\r\n\x1a\n") or not profiles_csv.startswith(
        b"record_type,"
    ):
        raise ValueError("inspection export payload is invalid")
    if len(header) > MAX_EXPORT_HEADER_BYTES:
        raise ProjectFormatError("inspection export paths exceed the request budget")
    request = len(header).to_bytes(4, "little") + header + figure_png + profiles_csv
    if len(request) > MAX_REQUEST_BYTES:
        raise ProjectFormatError("inspection export exceeds the 4 MiB background request limit")
    return request


def publish_inspection_export(argument: bytes, control: JobControl) -> JobResult:
    """Create a new PNG/CSV pair; failure leaves every prior file untouched."""

    if len(argument) < 4:
        raise ValueError("inspection export request is incomplete")
    header_length = int.from_bytes(argument[:4], "little")
    if not 0 < header_length <= MAX_EXPORT_HEADER_BYTES:
        raise ValueError("inspection export header is invalid")
    header = json.loads(argument[4 : 4 + header_length].decode("utf-8"))
    if set(header) != {
        "figure",
        "profiles",
        "project",
        "inputs",
        "figure_bytes",
        "profiles_bytes",
    }:
        raise ValueError("inspection export header has unexpected fields")
    figure_size, profiles_size = header["figure_bytes"], header["profiles_bytes"]
    if (
        type(figure_size) is not int
        or type(profiles_size) is not int
        or figure_size < 8
        or profiles_size < 1
        or 4 + header_length + figure_size + profiles_size != len(argument)
    ):
        raise ValueError("inspection export payload length is invalid")
    input_paths = tuple(Path(path) for path in header["inputs"])
    project_path = Path(header["project"]) if header["project"] is not None else None
    figure = external_export_destination(
        Path(header["figure"]), project_path, input_paths, suffix=".png"
    )
    profiles = external_export_destination(
        Path(header["profiles"]), project_path, input_paths, suffix=".csv"
    )
    if figure != Path(header["figure"]) or profiles != Path(header["profiles"]):
        raise ValueError("inspection export destination changed")
    if not figure.parent.is_dir() or not profiles.parent.is_dir():
        raise ProjectFormatError("inspection export destination directory does not exist")
    if any(path.exists() or path.is_symlink() for path in (figure, profiles)):
        raise FileExistsError("inspection export uses new filenames; choose another name")
    if control.canceled:
        raise RuntimeError("inspection export canceled before writing")
    png_start = 4 + header_length
    png = argument[png_start : png_start + figure_size]
    csv_data = argument[png_start + figure_size :]
    created: list[Path] = []
    try:
        for path, data in ((figure, png), (profiles, csv_data)):
            with path.open("xb") as handle:
                created.append(path)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        figure_hash = hashlib.sha256(figure.read_bytes()).hexdigest()
        profiles_hash = hashlib.sha256(profiles.read_bytes()).hexdigest()
        if (
            figure_hash != hashlib.sha256(png).hexdigest()
            or profiles_hash != hashlib.sha256(csv_data).hexdigest()
        ):
            raise OSError("inspection export readback differs from captured bytes")
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    receipt = InspectionExportReceipt(figure, profiles, figure_hash, profiles_hash)
    return JobResult(receipt, len(str(figure)) + len(str(profiles)) + 160)
