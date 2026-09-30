"""Single-worker template, identity relocation and reviewed bounded source copying."""

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID, uuid4

import yaml
from job_lifecycle import JobControl, JobResult
from metadata_review import MAX_REFERENCE_BYTES, bounded_reference_snapshot, validate_reference
from osc_import import AXIS_LIMIT, DECODED_LIMIT_BYTES, PIXEL_LIMIT, SOURCE_LIMIT_BYTES
from project_state import (
    ProjectDocument,
    ProjectFormatError,
    StorageOrigin,
    project_from_document,
    project_to_document,
)
from setup_state import MAX_TEMPLATE_BYTES, apply_template, template_bytes, template_from_bytes

COPY_BUFFER_BYTES = 1024 * 1024
MAX_COPY_BYTES = 512 * 1024 * 1024
MAX_COPY_FILES = 512


@dataclass(frozen=True, slots=True)
class SetupApplication:
    document: ProjectDocument
    review: tuple[str, ...]
    verified_kinds: tuple[tuple[UUID, str], ...] = ()


@dataclass(frozen=True, slots=True)
class TemplateFile:
    path: Path
    encoded: bytes
    sha256: str


def _stop(control: JobControl) -> None:
    if control.canceled:
        raise RuntimeError("setup/storage canceled; no new project binding")


def _hash_file(path: Path, limit: int, control: JobControl) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(COPY_BUFFER_BYTES):
            _stop(control)
            size += len(block)
            if size > limit:
                raise ProjectFormatError(f"{path.name} exceeds its source size limit")
            digest.update(block)
    return size, digest.hexdigest()


def _external(path: Path, protected: tuple[Path, ...], *, directory: bool = False) -> Path:
    resolved = path.resolve(strict=False)
    if len(str(resolved)) > 4096:
        raise ProjectFormatError("storage path exceeds 4096 characters")
    if any((parent / ".git").exists() for parent in (resolved, *resolved.parents)):
        raise ProjectFormatError("choose storage outside Git checkouts")
    for source in protected:
        if resolved == source.resolve(strict=False) or (
            resolved.exists() and source.exists() and os.path.samefile(resolved, source)
        ):
            raise ProjectFormatError("destination aliases a project or source")
    if directory and not resolved.is_dir():
        raise ProjectFormatError("choose an existing external project data folder")
    return resolved


def _bindings(document: ProjectDocument, ids: tuple[UUID, ...]):
    for item in document.project.acquisitions:
        if item.acquisition_id not in ids:
            continue
        yield item.acquisition_id, "osc", item.source_path, item.source_sha256
        for kind in ("cif", "configuration", "configuration_cif"):
            path = getattr(item.metadata, f"{kind}_path")
            if path is not None:
                yield item.acquisition_id, kind, path, getattr(item.metadata, f"{kind}_sha256")


def _protected(document: ProjectDocument, project_path: Path) -> tuple[Path, ...]:
    return (
        project_path,
        *(
            path
            for _, _, path, _ in _bindings(
                document, tuple(v.acquisition_id for v in document.project.acquisitions)
            )
        ),
    )


def _verify(path: Path, kind: str, expected: str, control: JobControl):
    _stop(control)
    if kind == "osc":
        from rasim_next.io.osc import OscReadLimits, read_osc

        image = read_osc(
            path,
            limits=OscReadLimits(SOURCE_LIMIT_BYTES, DECODED_LIMIT_BYTES, PIXEL_LIMIT, AXIS_LIMIT),
            canceled=lambda: control.canceled,
        )
        actual = image.decoded_sha256
        del image
        value = None
    else:
        value = validate_reference(path, "cif" if kind == "configuration_cif" else kind)
        actual = value.sha256
    if actual != expected:
        raise ProjectFormatError(
            f"{kind} identity mismatch for {path.name}; different content is not a relocation"
        )
    _stop(control)
    return value


def review_copy(
    document: ProjectDocument,
    ids: tuple[UUID, ...],
    folder: Path,
    project_path: Path,
    control: JobControl,
) -> dict:
    protected = _protected(document, project_path)
    root = _external(folder, protected, directory=True)
    by_id = {item.acquisition_id: item for item in document.project.acquisitions}
    rows = []
    seen = {}
    for acquisition_id, kind, source, expected in _bindings(document, ids):
        source = source.resolve(strict=True)
        key = (kind, source, expected)
        control.report(f"Reviewing {source.name}")
        if key not in seen:
            value = _verify(source, kind, expected, control)
            size, raw = _hash_file(
                source, SOURCE_LIMIT_BYTES if kind == "osc" else MAX_REFERENCE_BYTES, control
            )
            seen[key] = (value, size, raw)
        value, size, raw = seen[key]
        derived = None
        if kind in ("configuration", "configuration_cif"):
            metadata = by_id[acquisition_id].metadata
            config_path = metadata.configuration_path.resolve(strict=True)
            dependent = metadata.configuration_cif_path
            if dependent is None:
                raise ProjectFormatError(
                    "configuration needs its recorded dependent CIF before copying"
                )
            dependent = dependent.resolve(strict=True)
            reference = _verify(
                config_path, "configuration", metadata.configuration_sha256, control
            )
            if (
                reference.dependent_cif_path.resolve(strict=True) != dependent
                or reference.dependent_cif_sha256 != metadata.configuration_cif_sha256
            ):
                raise ProjectFormatError(
                    "configuration-dependent CIF differs from the recorded binding"
                )
            source_mapping_bytes, _ = bounded_reference_snapshot(config_path)
            from rasim_next.pipeline.configured_simulation import load_strict_yaml_mapping

            mapping = load_strict_yaml_mapping(config_path, source_bytes=source_mapping_bytes)
            declared = Path(mapping["material"]["cif_path"])
            bundle = root / ("configuration_" + metadata.configuration_sha256[:16])
            if declared.is_absolute() or config_path.drive != dependent.drive:
                destination = bundle / (
                    config_path.name
                    if kind == "configuration"
                    else "dependency" / Path(dependent.name)
                )
                if kind == "configuration":
                    mapping["material"]["cif_path"] = "dependency/" + dependent.name
                    derived = yaml.safe_dump(mapping, sort_keys=False)
            else:
                common = Path(os.path.commonpath((config_path.parent, dependent.parent)))
                destination = bundle / source.relative_to(common)
        else:
            destination = root / (kind + "_" + raw[:16]) / source.name
        destination = _external(destination, protected)
        if not destination.is_relative_to(root) or destination.exists():
            raise ProjectFormatError(f"destination collision or escape: {destination}")
        output = derived.encode() if derived is not None else None
        rows.append(
            {
                "acquisition_id": str(acquisition_id),
                "kind": kind,
                "source": str(source),
                "destination": str(destination),
                "expected_identity": expected,
                "source_bytes": size,
                "source_raw_sha256": raw,
                "destination_bytes": len(output) if output is not None else size,
                "destination_raw_sha256": hashlib.sha256(output).hexdigest()
                if output is not None
                else raw,
                "derived_yaml": derived,
            }
        )
    unique = {}
    for row in rows:
        prior = unique.setdefault(row["destination"], row)
        if (prior["source"], prior["destination_raw_sha256"]) != (
            row["source"],
            row["destination_raw_sha256"],
        ):
            raise ProjectFormatError("two different inputs would share a destination")
    total = sum(row["destination_bytes"] for row in unique.values())
    if not rows or len(unique) > MAX_COPY_FILES or total > MAX_COPY_BYTES:
        raise ProjectFormatError(
            "copy review is limited to 512 files / 512 MiB; select fewer acquisitions"
        )
    return {"root": str(root), "total_bytes": total, "rows": rows}


def _bind(document: ProjectDocument, rows: list[dict], mode: str) -> SetupApplication:
    changes = {}
    review = []
    verified = []
    for row in rows:
        acquisition_id, kind = UUID(row["acquisition_id"]), row["kind"]
        changes.setdefault(acquisition_id, {})[kind] = row
        verified.append((acquisition_id, kind))
        review.append(
            f"{acquisition_id} {kind}: {row['source']} -> {row['destination']} ({mode}); identity {row['expected_identity']}; observed raw SHA-256 {row['source_raw_sha256']}"
        )
    items = []
    draft = document.numeric_draft
    for item in document.project.acquisitions:
        rows_by_kind = changes.get(item.acquisition_id, {})
        if not rows_by_kind:
            items.append(item)
            continue
        metadata_changes = {}
        provenance = dict(item.metadata.provenance)
        storage = {v.kind: v for v in item.storage}
        source = item.source_path
        changed = False
        for kind, row in rows_by_kind.items():
            destination = Path(row["destination"])
            digest = row["destination_raw_sha256"] if kind != "osc" else item.source_sha256
            old_path = item.source_path if kind == "osc" else getattr(item.metadata, f"{kind}_path")
            if destination == old_path:
                continue
            changed = True
            previous = storage.get(kind)
            storage[kind] = StorageOrigin(
                kind,
                mode,
                previous.original_path if previous else old_path,
                Path(row["source"]) if mode == "copy" else destination,
                row["source_raw_sha256"],
            )
            if kind == "osc":
                source = destination
            else:
                metadata_changes[f"{kind}_path"] = destination
                metadata_changes[f"{kind}_sha256"] = digest
                provenance[f"{kind}_path"] = (
                    "derived configuration path rewrite; verified new hash"
                    if row.get("derived_yaml")
                    else f"verified {mode}; saved identity matched"
                )
        if not changed:
            items.append(item)
            continue
        metadata = replace(
            item.metadata,
            **metadata_changes,
            provenance=tuple(sorted(provenance.items())),
            revision=item.metadata.revision + 1,
        )
        items.append(
            replace(
                item,
                source_path=source,
                metadata=metadata,
                storage=tuple(storage[key] for key in sorted(storage)),
            )
        )
        if (
            draft is not None
            and draft.acquisition_id == item.acquisition_id
            and any(kind in rows_by_kind for kind in ("configuration", "configuration_cif"))
        ):
            # Rebuild from the destination reader; rewritten YAML has its own exact baseline/hash.
            from parameter_state import prepare_numeric_draft

            request = {
                "acquisition_id": str(item.acquisition_id),
                "source_sha256": item.source_sha256,
                "configuration_path": str(metadata.configuration_path),
                "configuration_sha256": metadata.configuration_sha256,
                "cif_path": str(metadata.configuration_cif_path),
                "cif_sha256": metadata.configuration_cif_sha256,
                "proposed": draft.proposed,
                "revision": draft.revision + 1,
            }
            draft = prepare_numeric_draft(json.dumps(request).encode(), JobControl()).value
    return SetupApplication(
        replace(
            document,
            project=replace(document.project, acquisitions=tuple(items)),
            numeric_draft=draft,
        ),
        tuple(review),
        tuple(verified),
    )


def copy_sources(
    document: ProjectDocument, plan: dict, project_path: Path, control: JobControl
) -> SetupApplication:
    """No overwrite; verify every byte before publishing any project reference."""
    if (
        type(plan) is not dict
        or set(plan) != {"root", "total_bytes", "rows"}
        or type(plan["rows"]) is not list
        or len(plan["rows"]) > MAX_COPY_FILES
    ):
        raise ProjectFormatError("invalid copy plan")
    protected = _protected(document, project_path)
    root = _external(Path(plan["root"]), protected, directory=True)
    bindings = {
        (str(a), k, str(p.resolve(strict=True)), h)
        for a, k, p, h in _bindings(
            document, tuple(v.acquisition_id for v in document.project.acquisitions)
        )
    }
    unique = {}
    for row in plan["rows"]:
        if type(row) is not dict or set(row) != {
            "acquisition_id",
            "kind",
            "source",
            "destination",
            "expected_identity",
            "source_bytes",
            "source_raw_sha256",
            "destination_bytes",
            "destination_raw_sha256",
            "derived_yaml",
        }:
            raise ProjectFormatError("invalid copy row")
        if (
            row["acquisition_id"],
            row["kind"],
            row["source"],
            row["expected_identity"],
        ) not in bindings:
            raise ProjectFormatError("copy plan no longer matches project sources")
        destination = _external(Path(row["destination"]), protected)
        if not destination.is_relative_to(root) or destination.exists():
            raise ProjectFormatError(f"copy destination collision: {destination}")
        for name in ("source_bytes", "destination_bytes"):
            if type(row[name]) is not int or not 0 < row[name] <= SOURCE_LIMIT_BYTES:
                raise ProjectFormatError("invalid copy size")
        for name in ("source_raw_sha256", "destination_raw_sha256"):
            if (
                type(row[name]) is not str
                or len(row[name]) != 64
                or any(c not in "0123456789abcdef" for c in row[name])
            ):
                raise ProjectFormatError("invalid copy hash")
        if row["derived_yaml"] is not None and (
            row["kind"] != "configuration"
            or type(row["derived_yaml"]) is not str
            or len(row["derived_yaml"].encode()) > MAX_REFERENCE_BYTES
        ):
            raise ProjectFormatError("invalid derived configuration")
        prior = unique.setdefault(destination, row)
        if prior != row and (prior["source"], prior["destination_raw_sha256"]) != (
            row["source"],
            row["destination_raw_sha256"],
        ):
            raise ProjectFormatError("conflicting copy destinations")
    total = sum(row["destination_bytes"] for row in unique.values())
    if not unique or total != plan["total_bytes"] or total > MAX_COPY_BYTES:
        raise ProjectFormatError("copy total is invalid")
    completed, copied = [], 0
    checked_sources = set()
    try:
        for destination, row in unique.items():
            _stop(control)
            source = Path(row["source"])
            source_key = (source, row["kind"], row["expected_identity"])
            if source_key not in checked_sources:
                _verify(source, row["kind"], row["expected_identity"], control)
                checked_sources.add(source_key)
            size, digest = _hash_file(source, SOURCE_LIMIT_BYTES, control)
            if (size, digest) != (row["source_bytes"], row["source_raw_sha256"]):
                raise ProjectFormatError("source bytes changed after copy review")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Recheck after parent creation to reject symlink races before opening.
            if destination.resolve(strict=False) != destination:
                raise ProjectFormatError("copy destination changed during preparation")
            temporary = destination.with_name(destination.name + "." + uuid4().hex + ".part")
            try:
                output_hash, output_size = hashlib.sha256(), 0
                with temporary.open("xb") as target:
                    if row["derived_yaml"] is not None:
                        block = row["derived_yaml"].encode()
                        target.write(block)
                        output_hash.update(block)
                        output_size = len(block)
                    else:
                        with source.open("rb") as original:
                            while block := original.read(COPY_BUFFER_BYTES):
                                _stop(control)
                                output_size += len(block)
                                if output_size > row["destination_bytes"]:
                                    raise ProjectFormatError("copy source grew after review")
                                target.write(block)
                                output_hash.update(block)
                                control.report(
                                    f"Copying {source.name}: {copied + output_size}/{total} bytes"
                                )
                    target.flush()
                    os.fsync(target.fileno())
                if (output_size, output_hash.hexdigest()) != (
                    row["destination_bytes"],
                    row["destination_raw_sha256"],
                ):
                    raise ProjectFormatError("copied bytes differ from review")
                _stop(control)
                if _hash_file(temporary, SOURCE_LIMIT_BYTES, control) != (
                    output_size,
                    output_hash.hexdigest(),
                ):
                    raise ProjectFormatError("destination readback failed")
                os.link(temporary, destination)
                completed.append(str(destination))
                copied += output_size
            finally:
                temporary.unlink(missing_ok=True)
        for row in plan["rows"]:
            expected = (
                row["expected_identity"] if row["kind"] == "osc" else row["destination_raw_sha256"]
            )
            value = _verify(Path(row["destination"]), row["kind"], expected, control)
            if row["kind"] == "configuration":
                dependent = next(
                    (
                        r
                        for r in plan["rows"]
                        if r["acquisition_id"] == row["acquisition_id"]
                        and r["kind"] == "configuration_cif"
                    ),
                    None,
                )
                if (
                    dependent is None
                    or value.dependent_cif_path.resolve(strict=True)
                    != Path(dependent["destination"])
                    or value.dependent_cif_sha256 != dependent["destination_raw_sha256"]
                ):
                    raise ProjectFormatError("copied configuration resolved a different CIF")
        _stop(control)
        return _bind(document, plan["rows"], "copy")
    except (OSError, ValueError, RuntimeError) as exc:
        raise ProjectFormatError(
            f"Copy not bound: {exc}. {len(completed)} completed byte-verified file(s) retained in {root}; originals unchanged. No partial project binding."
        ) from exc


def relink_references(
    document: ProjectDocument, ids: tuple[UUID, ...], kind: str, path: Path, control: JobControl
) -> SetupApplication:
    rows = []
    for acquisition_id, bound_kind, source, expected in _bindings(document, ids):
        if bound_kind != kind:
            continue
        value = _verify(path, kind, expected, control)
        _size, raw = _hash_file(
            path, SOURCE_LIMIT_BYTES if kind == "osc" else MAX_REFERENCE_BYTES, control
        )
        rows.append(
            {
                "acquisition_id": str(acquisition_id),
                "kind": kind,
                "source": str(source),
                "destination": str(path.absolute()),
                "expected_identity": expected,
                "source_raw_sha256": raw,
                "destination_raw_sha256": raw,
            }
        )
        if kind == "configuration":
            item = next(
                v for v in document.project.acquisitions if v.acquisition_id == acquisition_id
            )
            if value.dependent_cif_sha256 != item.metadata.configuration_cif_sha256:
                raise ProjectFormatError(
                    "dependent CIF identity differs; relocate the matching configuration/CIF bundle"
                )
            rows.append(
                {
                    "acquisition_id": str(acquisition_id),
                    "kind": "configuration_cif",
                    "source": str(item.metadata.configuration_cif_path),
                    "destination": str(value.dependent_cif_path),
                    "expected_identity": value.dependent_cif_sha256,
                    "source_raw_sha256": value.dependent_cif_sha256,
                    "destination_raw_sha256": value.dependent_cif_sha256,
                }
            )
    if not rows:
        raise ProjectFormatError("no saved identity exists for selected reference kind")
    return _bind(document, rows, "relink")


def prepare_setup(argument: bytes, control: JobControl) -> JobResult:
    request = json.loads(argument)
    if type(request) is not dict or set(request) != {
        "operation",
        "project_path",
        "document",
        "ids",
        "payload",
    }:
        raise ProjectFormatError("invalid setup request")
    operation = request["operation"]
    project_path = Path(request["project_path"])
    document = project_from_document(request["document"], project_path)
    ids = tuple(UUID(value) for value in request["ids"])
    if not set(ids) <= {v.acquisition_id for v in document.project.acquisitions} or len(
        set(ids)
    ) != len(ids):
        raise ProjectFormatError("invalid setup target IDs")
    payload = request["payload"]
    _stop(control)
    if operation in ("load_template", "save_template"):
        path = Path(payload["path"]).absolute()
        if not path.name.endswith(".slate-template.json"):
            raise ProjectFormatError("template file must end in .slate-template.json")
        if operation == "load_template":
            with path.open("rb") as stream:
                encoded = stream.read(MAX_TEMPLATE_BYTES + 1)
            template_from_bytes(encoded)
        else:
            path = _external(path, _protected(document, project_path))
            encoded = template_bytes(template_from_bytes(payload["encoded"].encode()))
            expected = payload["expected_sha256"]
            if path.exists():
                if expected is None or _hash_file(path, MAX_TEMPLATE_BYTES, control)[1] != expected:
                    raise ProjectFormatError(
                        "template destination exists or changed; choose a new name"
                    )
                from rasim_next.io.json_publication import publish_json_document

                _stop(control)
                publish_json_document(path, json.loads(encoded), allow_nan=False)
            else:
                temporary = path.with_name(path.name + "." + uuid4().hex + ".part")
                try:
                    with temporary.open("xb") as stream:
                        stream.write(encoded)
                        stream.flush()
                        os.fsync(stream.fileno())
                    _stop(control)
                    os.link(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            with path.open("rb") as stream:
                observed = stream.read(MAX_TEMPLATE_BYTES + 1)
            if observed != encoded:
                raise ProjectFormatError("template publication readback mismatch")
        value = TemplateFile(path, encoded, hashlib.sha256(encoded).hexdigest())
    elif operation == "apply_template":
        encoded = payload["encoded"].encode()
        template = template_from_bytes(encoded)
        project, draft, review = apply_template(
            document.project,
            document.numeric_draft,
            template,
            ids,
            hashlib.sha256(encoded).hexdigest(),
            control,
        )
        value = SetupApplication(replace(document, project=project, numeric_draft=draft), review)
    elif operation == "review_copy":
        value = review_copy(document, ids, Path(payload["folder"]), project_path, control)
    elif operation == "copy":
        value = copy_sources(document, payload["plan"], project_path, control)
    elif operation == "relink":
        if payload["kind"] not in ("osc", "cif", "configuration"):
            raise ProjectFormatError(
                "dependent CIF relocation uses the matching configuration bundle"
            )
        value = relink_references(
            document, ids, payload["kind"], Path(payload["path"]).absolute(), control
        )
    else:
        raise ProjectFormatError("unsupported setup operation")
    _stop(control)
    if isinstance(value, SetupApplication):
        project_to_document(value.document, project_path)
    return JobResult((operation, value), 2 * len(argument) + 128 * 1024)
