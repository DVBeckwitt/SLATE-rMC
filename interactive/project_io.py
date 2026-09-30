"""Bounded project-document workers using the canonical atomic JSON publisher."""

import hashlib
import json
import os
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal
from uuid import UUID

from job_lifecycle import JobControl, JobResult
from metadata_review import bounded_reference_snapshot, reference_path_key
from osc_import import (
    DECODED_LIMIT_BYTES,
    PIXEL_LIMIT,
    SOURCE_LIMIT_BYTES,
    decode_bounded_path,
)
from project_state import (
    MAX_PROJECT_BYTES,
    Project,
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
class ReferenceCheck:
    acquisition_id: UUID
    kind: Literal["cif", "configuration", "configuration_cif"]
    state: Literal["verified", "missing", "changed", "unreadable", "unverified"]
    detail: str = ""
    file_identity: tuple[int, int] | None = None


def reference_bindings(project: Project) -> Iterator[tuple[UUID, str, Path, str]]:
    """Yield each saved identity without accessing its file."""
    for acquisition in project.acquisitions:
        metadata = acquisition.metadata
        for kind, path, digest in (
            ("cif", metadata.cif_path, metadata.cif_sha256),
            ("configuration", metadata.configuration_path, metadata.configuration_sha256),
            (
                "configuration_cif",
                metadata.configuration_cif_path,
                metadata.configuration_cif_sha256,
            ),
        ):
            if path is not None and digest is not None:
                yield acquisition.acquisition_id, kind, path, digest


def invalidate_reference_checks(
    project: Project,
    checks: dict[tuple[UUID, str], ReferenceCheck],
    kind: str,
    path: Path,
    detail: str,
) -> dict[tuple[UUID, str], ReferenceCheck]:
    """Clear one reference and its known aliases in one bounded in-memory pass."""
    bindings = tuple(reference_bindings(project))
    requested_key = reference_path_key(path)
    file_identities = {
        check.file_identity
        for acquisition_id, binding_kind, bound_path, _ in bindings
        if reference_path_key(bound_path) == requested_key
        if (check := checks.get((acquisition_id, binding_kind))) is not None
        and check.file_identity is not None
    }
    path_keys = {requested_key}
    if kind == "configuration":
        for acquisition in project.acquisitions:
            metadata = acquisition.metadata
            if metadata.configuration_path is None or metadata.configuration_cif_path is None:
                continue
            check = checks.get((acquisition.acquisition_id, "configuration"))
            if reference_path_key(metadata.configuration_path) == requested_key or (
                check is not None and check.file_identity in file_identities
            ):
                path_keys.add(reference_path_key(metadata.configuration_cif_path))
                dependent = checks.get((acquisition.acquisition_id, "configuration_cif"))
                if dependent is not None and dependent.file_identity is not None:
                    file_identities.add(dependent.file_identity)
    updated = checks.copy()
    for acquisition_id, binding_kind, bound_path, _ in bindings:
        key = (acquisition_id, binding_kind)
        old = checks.get(key)
        if reference_path_key(bound_path) in path_keys or (
            old is not None
            and old.file_identity is not None
            and old.file_identity in file_identities
        ):
            updated[key] = ReferenceCheck(
                acquisition_id,
                binding_kind,
                "unverified",
                detail,
                None if old is None else old.file_identity,
            )
    return updated


def publish_reference_checks(
    project: Project,
    checks: dict[tuple[UUID, str], ReferenceCheck],
    path: Path,
    observed_sha256: str,
    file_identity: tuple[int, int] | None,
    matching_bindings: tuple[tuple[UUID, str], ...],
) -> dict[tuple[UUID, str], ReferenceCheck]:
    """Compare a worker observation with each affected saved hash."""
    observed_key = reference_path_key(path)
    matched = set(matching_bindings)
    updated = checks.copy()
    for acquisition_id, kind, bound_path, expected in reference_bindings(project):
        key = (acquisition_id, kind)
        if key not in matched and reference_path_key(bound_path) != observed_key:
            continue
        state = "verified" if expected == observed_sha256 else "changed"
        updated[key] = ReferenceCheck(
            acquisition_id,
            kind,
            state,
            "SHA-256 matches bound bytes"
            if state == "verified"
            else "SHA-256 differs from bound bytes",
            file_identity,
        )
    return updated


@dataclass(frozen=True, slots=True)
class LoadedProject:
    document: ProjectDocument
    path: Path
    sources: tuple[SourceCheck, ...]
    references: tuple[ReferenceCheck, ...]
    numeric_validated: bool
    simulation_detail: str = ""


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
    reference_checks: list[ReferenceCheck] = []
    seen_references: dict[Path, tuple[str, str, tuple[int, int] | None]] = {}
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
        metadata = acquisition.metadata
        for kind, reference_path, expected in (
            ("cif", metadata.cif_path, metadata.cif_sha256),
            ("configuration", metadata.configuration_path, metadata.configuration_sha256),
            (
                "configuration_cif",
                metadata.configuration_cif_path,
                metadata.configuration_cif_sha256,
            ),
        ):
            if control.canceled:
                raise RuntimeError("Project opening canceled")
            if reference_path is None:
                if kind == "configuration_cif" and metadata.configuration_path is not None:
                    reference_checks.append(
                        ReferenceCheck(
                            acquisition.acquisition_id,
                            kind,
                            "unverified",
                            "No recorded dependent CIF identity; choose configuration again",
                        )
                    )
                continue
            if reference_path not in seen_references:
                try:
                    source_bytes, file_identity = bounded_reference_snapshot(reference_path)
                    actual = hashlib.sha256(source_bytes).hexdigest()
                    seen_references[reference_path] = ("readable", actual, file_identity)
                except FileNotFoundError:
                    seen_references[reference_path] = ("missing", str(reference_path), None)
                except (OSError, ProjectFormatError) as exc:
                    seen_references[reference_path] = ("unreadable", str(exc)[:120], None)
            read_state, detail, file_identity = seen_references[reference_path]
            state = (
                ("verified" if detail == expected else "changed")
                if read_state == "readable"
                else read_state
            )
            reference_checks.append(
                ReferenceCheck(
                    acquisition.acquisition_id,
                    kind,
                    state,
                    "SHA-256 matches recorded bytes"
                    if state == "verified"
                    else "SHA-256 differs from recorded bytes"
                    if state == "changed"
                    else detail,
                    file_identity,
                )
            )
    draft = document.numeric_draft
    numeric_validated = False
    if draft is not None:
        acquisition = next(
            item
            for item in document.project.acquisitions
            if item.acquisition_id == draft.acquisition_id
        )
        metadata = acquisition.metadata
        matching_identity = (
            acquisition.source_sha256 == draft.source_sha256
            and metadata.configuration_path == draft.configuration_path
            and metadata.configuration_sha256 == draft.configuration_sha256
            and metadata.configuration_cif_path == draft.cif_path
            and metadata.configuration_cif_sha256 == draft.cif_sha256
        )
        source_verified = next(
            check.state == "verified"
            for check in checks
            if check.acquisition_id == draft.acquisition_id
        )
        reference_states = {
            check.kind: check.state
            for check in reference_checks
            if check.acquisition_id == draft.acquisition_id
        }
        verified = source_verified and all(
            reference_states.get(kind) == "verified"
            for kind in ("configuration", "configuration_cif")
        )
        if matching_identity and verified:
            if control.canceled:
                raise RuntimeError("Project opening canceled")
            from parameter_state import configured_draft

            try:
                configured_draft(draft)
            except (OSError, ValueError) as exc:
                raise ProjectFormatError(f"saved numeric draft is invalid: {exc}") from exc
            numeric_validated = True
    simulation_detail = ""
    if document.view.simulation_draft is not None:
        from simulation_io import canonical_configuration

        try:
            canonical_configuration(document.view.simulation_draft)
        except (OSError, ValueError) as exc:
            simulation_detail = f"Saved independent draft requires input review: {exc}"
    loaded = LoadedProject(
        document, path, tuple(checks), tuple(reference_checks), numeric_validated, simulation_detail
    )
    resident = path.stat().st_size + sum(
        len(item.detail.encode("utf-8")) + 128 for item in (*checks, *reference_checks)
    )
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
    independent = document.view.simulation_draft
    if independent is not None:
        for source in (independent.configuration_path, independent.cif_path):
            if destination.resolve(strict=False) == source.resolve(strict=False) or (
                destination.exists() and source.exists() and os.path.samefile(destination, source)
            ):
                raise ProjectFormatError(
                    "project destination would overwrite an independent simulation input"
                )
    if document.view.simulation_result is not None and destination.resolve(
        strict=False
    ) == document.view.simulation_result.path.resolve(strict=False):
        raise ProjectFormatError("project destination would overwrite a simulation result")
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
