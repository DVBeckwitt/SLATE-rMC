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
    sample_input_checks: tuple[tuple[str, str], ...] = ()
    document_sha256: str = ""


@dataclass(frozen=True, slots=True)
class PublishedProject:
    path: Path
    cleanup_warning: str = ""
    sha256: str = ""

    @property
    def nbytes(self):
        return (
            len(str(self.path).encode())
            + len(self.cleanup_warning.encode())
            + len(self.sha256)
            + 128
        )


def load_project(argument: bytes, control: JobControl) -> JobResult:
    """Validate JSON and source hashes one at a time without retaining image planes."""
    from rasim_next.io.osc import OscReadLimits, read_osc

    expected = None
    if argument.startswith(b'{"reviewed_path":'):
        request = json.loads(argument)
        if set(request) != {"reviewed_path", "axis_limit", "sha256"}:
            raise ProjectFormatError("Malformed reviewed project request")
        from osc_import import encode_bounded_path

        path, axis_limit = decode_bounded_path(
            encode_bounded_path(Path(request["reviewed_path"]), request["axis_limit"])
        )
        expected = request["sha256"]
        if type(expected) is not str or len(expected) != 64:
            raise ProjectFormatError("Reviewed project requires its exact SHA256")
    else:
        path, axis_limit = decode_bounded_path(argument)
    control.report(f"Opening {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(MAX_PROJECT_BYTES + 1)
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ProjectFormatError("Reviewed project changed; review again before opening")
    document = read_project_document(path, source_bytes=raw)
    from archive_storage import storage_files

    storage = storage_files(document.view.archive_storage_json)
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
                configured_draft(draft, storage)
            except (OSError, ValueError) as exc:
                raise ProjectFormatError(f"saved numeric draft is invalid: {exc}") from exc
            numeric_validated = True
    simulation_detail = ""
    if document.view.simulation_draft is not None:
        from simulation_io import canonical_configuration

        try:
            canonical_configuration(document.view.simulation_draft, storage)
        except (OSError, ValueError) as exc:
            simulation_detail = f"Saved independent draft requires input review: {exc}"
    if document.view.native_simulation_draft is not None:
        from native_simulation_io import canonical_native

        try:
            canonical_native(document.view.native_simulation_draft)
        except (OSError, ValueError) as exc:
            simulation_detail += f" Native draft requires input review: {exc}"
    if document.view.hbn_sessions:
        from hbn_io import validate_hbn_result

        for session in document.view.hbn_sessions:
            for text in session.results_json:
                if control.canceled:
                    raise RuntimeError("hBN project opening canceled")
                validate_hbn_result(json.loads(text), session)
    sample_checks = ()
    from attempt_state import read_history

    inherited = read_history(document.view.attempts_json).inherited_from is not None
    if document.view.sample_session is not None and not storage and not inherited:
        from sample_io import sample_input_checks
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            sample_checks = sample_input_checks(document.view.sample_session, control)
    if document.view.sample_session is not None and (storage or inherited):
        from sample_io import _validate_input_bindings, validate_sample_record
        from sample_state import payload_hash

        session = document.view.sample_session
        _validate_input_bindings(json.loads(session.inputs_json))
        for text in session.results_json:
            validate_sample_record(json.loads(text), session)
        inputs = [json.loads(session.inputs_json)] + [
            json.loads(v)["launch"]["inputs"] for v in session.results_json
        ]
        sample_checks = tuple(
            (
                payload_hash(v),
                "Archived exact bytes/history; live canonical geometry revalidation is unavailable until explicitly requested through existing owners",
            )
            for v in inputs
        )
    loaded = LoadedProject(
        document,
        path,
        tuple(checks),
        tuple(reference_checks),
        numeric_validated,
        simulation_detail,
        sample_checks,
        digest,
    )
    resident = (
        len(raw)
        + sum(len(item.detail.encode("utf-8")) + 128 for item in (*checks, *reference_checks))
        + sum(len(identity) + len(detail.encode()) + 128 for identity, detail in sample_checks)
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
    required = {"destination", "document", "recovery", "retire_draft"}
    if (
        type(request) is not dict
        or not required <= set(request)
        or set(request) - required - {"exclusive", "retire_sha256", "recovery_sha256"}
    ):
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
    exclusive = request.get("exclusive", False)
    if type(exclusive) is not bool:
        raise ProjectFormatError("Exclusive publication flag is invalid")
    destination = Path(destination_text).absolute()
    if exclusive:
        from simulation_io import _external

        _external(destination)
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
    from attempt_state import read_history

    history = read_history(document.view.attempts_json)
    protected = [ref.path for ref in (*history.configured, *history.native)]
    protected.extend(path for _, _, path, _ in reference_bindings(document.project))
    from preparation_state import read_preparation

    protected.extend(
        Path(file["path"])
        for review in read_preparation(document.view.preparation_json)["reviews"]
        for file in review["files"]
    )
    if document.view.native_simulation_draft is not None:
        protected.append(document.view.native_simulation_draft.physics_path)
    if document.view.native_simulation_result is not None:
        protected.append(document.view.native_simulation_result.path)
    for session in document.view.hbn_sessions:
        inputs = json.loads(session.inputs_json)
        protected.extend(
            Path(inputs[k]) for k in ("source_path", "dark_path", "configuration_path", "cif_path")
        )
        protected.extend(Path(p) for p, _sha in session.exports)
    if document.view.sample_session is not None:
        from sample_io import sample_protected_paths

        protected.extend(sample_protected_paths(document.view.sample_session))
    if document.view.joint_session is not None:
        from joint_io import joint_protected_paths

        protected.extend(joint_protected_paths(document.view.joint_session))
    if document.view.native_fit_session is not None:
        from native_fit_io import prepared_paths

        protected.extend(prepared_paths(document.view.native_fit_session))
    from archive_storage import storage_files

    protected.extend(
        Path(row["stored"]) for row in storage_files(document.view.archive_storage_json)
    )
    for source in protected:
        if destination.resolve(strict=False) == source.resolve(strict=False) or (
            destination.exists() and source.exists() and os.path.samefile(destination, source)
        ):
            raise ProjectFormatError(
                "project destination would overwrite a native simulation, hBN or sample input/result"
            )
    if recovery:
        if destination.name != f"{document.project.project_id}.slate.json":
            raise ProjectFormatError("recovery destination must use the project UUID")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            with destination.open("rb") as stream:
                previous = stream.read(MAX_PROJECT_BYTES + 1)
            if (
                destination.is_symlink()
                or len(previous) > MAX_PROJECT_BYTES
                or hashlib.sha256(previous).hexdigest() != request.get("recovery_sha256")
            ):
                raise ProjectFormatError(
                    "Existing recovery draft is not the exact owned snapshot; review it explicitly"
                )
        else:
            exclusive = True
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
    publish_json_document(
        destination,
        request["document"],
        allow_nan=False,
        overwrite=not exclusive,
        canceled=lambda: control.canceled,
    )
    warning = ""
    if retire is not None:
        try:
            warning = retire_recovery(retire, request.get("retire_sha256"))
        except OSError as exc:
            warning = "Saved, but could not remove old recovery draft: " + str(exc)[:512]
    receipt = PublishedProject(destination, warning, hashlib.sha256(encoded).hexdigest())
    return JobResult(receipt, receipt.nbytes)


def retire_recovery(path: Path, expected: str | None) -> str:
    """Retire only an exact owned/reviewed draft; preserve unknown or changed bytes."""
    if not path.exists():
        return ""
    if path.is_symlink() or expected is None:
        return "Recovery draft retained: no exact owned identity available"
    with path.open("rb") as stream:
        raw = stream.read(MAX_PROJECT_BYTES + 1)
    if len(raw) > MAX_PROJECT_BYTES or hashlib.sha256(raw).hexdigest() != expected:
        return "Recovery draft retained: bytes changed after ownership/review"
    document = read_project_document(path, source_bytes=raw)
    if path.name != f"{document.project.project_id}.slate.json":
        return "Recovery draft retained: UUID differs"
    path.unlink()
    return ""


def discard_recovery(argument: bytes, control: JobControl) -> JobResult:
    """Remove only an exact owned recovery snapshot; no filesystem transaction is claimed."""
    request = json.loads(argument)
    if type(request) is not dict or set(request) != {"path", "sha256"}:
        raise ProjectFormatError("Invalid recovery discard request")
    path = Path(request["path"])
    UUID(path.name.removesuffix(".slate.json"))
    if control.canceled:
        raise RuntimeError("Draft discard canceled")
    control.report("Discarding owned recovery draft")
    warning = retire_recovery(path, request["sha256"])
    receipt = PublishedProject(path, warning)
    return JobResult(receipt, receipt.nbytes)
