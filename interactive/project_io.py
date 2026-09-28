"""Bounded project-document workers using the canonical atomic JSON publisher."""

import json
import os
import zlib
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal
from uuid import UUID

from job_lifecycle import JobControl, JobResult
from osc_import import (
    DECODED_LIMIT_BYTES,
    PIXEL_LIMIT,
    SOURCE_LIMIT_BYTES,
    decode_bounded_path,
)
from project_state import (
    MAX_PROJECT_BYTES,
    ProjectDocument,
    ProjectFormatError,
    project_from_document,
    read_project_document,
)

MAX_RECOVERY_DRAFTS = 32


@dataclass(frozen=True, slots=True)
class SourceCheck:
    acquisition_id: UUID
    state: Literal["verified", "missing", "changed", "unreadable"]
    detail: str = ""


@dataclass(frozen=True, slots=True)
class LoadedProject:
    document: ProjectDocument
    path: Path
    sources: tuple[SourceCheck, ...]


@dataclass(frozen=True, slots=True)
class PublishedProject:
    path: Path
    cleanup_warning: str = ""


def load_project(argument: bytes, control: JobControl) -> JobResult:
    """Validate JSON and source hashes one at a time without retaining image planes."""
    from rasim_next.io.osc import OscReadLimits, read_osc

    path, axis_limit = decode_bounded_path(argument)
    control.report(f"Opening {path.name}")
    document = read_project_document(path)
    seen: dict[Path, tuple[str, str, tuple[int, int] | None]] = {}
    checks: list[SourceCheck] = []
    limits = OscReadLimits(SOURCE_LIMIT_BYTES, DECODED_LIMIT_BYTES, PIXEL_LIMIT, axis_limit)
    for acquisition in document.project.acquisitions:
        if control.canceled:
            raise RuntimeError("Project opening canceled")
        source = acquisition.source_path
        if source not in seen:
            try:
                image = read_osc(source, limits=limits, canceled=lambda: control.canceled)
                seen[source] = (
                    "verified",
                    image.decoded_sha256 or "",
                    image.detector_native_counts.shape,
                )
                del image
            except FileNotFoundError:
                seen[source] = ("missing", str(source), None)
            except (OSError, ValueError, zlib.error) as exc:
                seen[source] = ("unreadable", str(exc)[:120], None)
        state, detail, shape = seen[source]
        if state == "missing":
            checks.append(SourceCheck(acquisition.acquisition_id, "missing", detail))
        elif state == "unreadable":
            checks.append(SourceCheck(acquisition.acquisition_id, "unreadable", detail))
        elif detail != acquisition.source_sha256:
            checks.append(
                SourceCheck(acquisition.acquisition_id, "changed", "Decoded OSC SHA-256 differs")
            )
        else:
            checks.append(SourceCheck(acquisition.acquisition_id, "verified"))
            if acquisition.acquisition_id == document.view.selected_acquisition_id:
                detector = document.view.detector
                if (
                    detector is not None
                    and shape is not None
                    and (detector.column_px >= shape[1] or detector.row_px >= shape[0])
                ):
                    raise ProjectFormatError("saved crosshair is outside its detector source")
    loaded = LoadedProject(document, path, tuple(checks))
    resident = path.stat().st_size + sum(len(item.detail.encode("utf-8")) + 128 for item in checks)
    control.report("Project and source references checked")
    return JobResult(loaded, resident)


def write_project(argument: bytes, control: JobControl) -> JobResult:
    """Publish a validated snapshot; never cancel after starting the atomic replacement."""
    from rasim_next.io.json_publication import publish_json_document

    if control.canceled:
        raise RuntimeError("Project save canceled before publication")
    try:
        request = json.loads(argument.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProjectFormatError(f"invalid project save request: {exc}") from exc
    if type(request) is not dict or set(request) != {
        "destination",
        "document",
        "recovery",
        "retire_draft",
    }:
        raise ProjectFormatError("project save request has unexpected fields")
    destination_text = request["destination"]
    recovery = request["recovery"]
    retire_text = request["retire_draft"]
    if type(destination_text) is not str or not destination_text:
        raise ProjectFormatError("project destination is invalid")
    if type(recovery) is not bool:
        raise ProjectFormatError("recovery flag is invalid")
    if retire_text is not None and (type(retire_text) is not str or not retire_text):
        raise ProjectFormatError("draft cleanup path is invalid")
    destination = Path(destination_text).absolute()
    document = project_from_document(request["document"], destination)
    encoded = (
        json.dumps(request["document"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_PROJECT_BYTES:
        raise ProjectFormatError(f"project document exceeds {MAX_PROJECT_BYTES} bytes")
    if not destination.name.lower().endswith(".slate.json"):
        raise ProjectFormatError("project destination must end in .slate.json")
    for acquisition in document.project.acquisitions:
        source = acquisition.source_path
        if destination.resolve(strict=False) == source.resolve(strict=False) or (
            destination.exists() and source.exists() and os.path.samefile(destination, source)
        ):
            raise ProjectFormatError("project destination would overwrite an OSC source")
    if recovery:
        if destination.name != f"{document.project.project_id}.slate.json":
            raise ProjectFormatError("recovery destination must use the project UUID")
        destination.parent.mkdir(parents=True, exist_ok=True)
        drafts = tuple(islice(destination.parent.glob("*.slate.json"), MAX_RECOVERY_DRAFTS + 1))
        if len(drafts) > MAX_RECOVERY_DRAFTS or (
            len(drafts) == MAX_RECOVERY_DRAFTS and not destination.exists()
        ):
            raise ProjectFormatError(
                f"recovery location is limited to {MAX_RECOVERY_DRAFTS} drafts"
            )
        other_bytes = sum(
            item.stat().st_size for item in drafts if item.is_file() and item != destination
        )
        if other_bytes + len(encoded) > MAX_RECOVERY_DRAFTS * MAX_PROJECT_BYTES:
            raise ProjectFormatError("recovery location exceeds its byte limit")
    elif not destination.parent.is_dir():
        raise ProjectFormatError("project destination directory does not exist")
    retire = None
    if retire_text is not None:
        retire = Path(retire_text).absolute()
        aliases_destination = retire.resolve(strict=False) == destination.resolve(strict=False) or (
            retire.exists() and destination.exists() and os.path.samefile(retire, destination)
        )
        if retire.name != f"{document.project.project_id}.slate.json" or aliases_destination:
            raise ProjectFormatError("draft cleanup target is invalid")
    control.report(f"Saving {destination.name}")
    publish_json_document(destination, request["document"], allow_nan=False)
    warning = ""
    if retire is not None:
        try:
            retire.unlink(missing_ok=True)
        except OSError as exc:
            warning = f"Saved, but could not remove old recovery draft: {exc}"
    receipt = PublishedProject(destination, warning)
    return JobResult(receipt, len(str(destination).encode("utf-8")) + 128)


def discard_recovery(argument: Path, control: JobControl) -> JobResult:
    """Remove only the current project's explicitly selected recovery file."""
    path = Path(argument)
    if not path.name.endswith(".slate.json"):
        raise ProjectFormatError("invalid recovery draft path")
    UUID(path.name.removesuffix(".slate.json"))
    control.report("Discarding recoverable draft")
    path.unlink(missing_ok=True)
    return JobResult(PublishedProject(path), len(str(path).encode("utf-8")) + 128)
