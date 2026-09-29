"""Bounded acquisition metadata import/export and validated reference pickers."""

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID

from job_lifecycle import JobControl, JobResult
from project_state import SOURCE_HASH_KIND, AcquisitionMetadata, Project, ProjectFormatError

MAX_METADATA_TEXT_BYTES = 64 * 1024
MAX_METADATA_ROWS = 128
MAX_METADATA_COLUMNS = 24
MAX_METADATA_FIELD_LENGTH = 256
MAX_REFERENCE_BYTES = 1024 * 1024
MAX_METADATA_EXPORT_BYTES = 1024 * 1024
MAPPED_FIELDS = (
    "acquisition_id",
    "role",
    "specimen",
    "mount",
    "incidence_deg",
    "exposure_s",
    "detector_setup",
    "material_id",
    "calibrant_id",
    "dark_acquisition_id",
    "mask_acquisition_id",
)


@dataclass(frozen=True, slots=True)
class ValidatedReference:
    kind: str
    path: Path
    sha256: str
    material_id: str | None


def prepare_reference(argument: bytes, control: JobControl) -> JobResult:
    """Validate one explicit reference in the existing global background owner."""
    request = json.loads(argument.decode("utf-8"))
    if type(request) is not dict or set(request) != {"kind", "path"}:
        raise ProjectFormatError("invalid reference request")
    kind, path = request["kind"], request["path"]
    if type(path) is not str or not path or len(path) > 4096:
        raise ProjectFormatError("invalid reference path")
    control.report(f"Checking {Path(path).name}")
    digest, material_id = validate_reference(Path(path), kind)
    if control.canceled:
        raise RuntimeError("reference validation canceled")
    value = ValidatedReference(kind, Path(path).absolute(), digest, material_id)
    return JobResult(value, len(path.encode("utf-8")) + 256)


def parse_table_text(text: str, *, delimiter: str) -> tuple[tuple[str, ...], ...]:
    """Bound text and row/field counts before any project mutation."""
    if len(text.encode("utf-8")) > MAX_METADATA_TEXT_BYTES:
        raise ProjectFormatError("metadata text exceeds 64 KiB")
    if delimiter not in (",", "\t"):
        raise ValueError("metadata delimiter must be CSV or tab")
    rows = []
    for row in csv.reader(io.StringIO(text), delimiter=delimiter, strict=True):
        if len(row) > MAX_METADATA_COLUMNS:
            raise ProjectFormatError("metadata row has too many columns")
        rows.append(tuple(row))
        if len(rows) > MAX_METADATA_ROWS + 1:
            raise ProjectFormatError("metadata text exceeds 128 data rows")
    if not rows:
        raise ProjectFormatError("metadata text is empty")
    return tuple(rows)


def filename_angle_proposal(path: Path) -> tuple[tuple[str, str, str], ...]:
    """Recognize one explicit `5d` token as an unconfirmed degree suggestion."""
    matches = tuple(re.finditer(r"(?i)(?:^|[_-])(-?\d+(?:\.\d+)?)d(?=[_.-]|$)", path.name))
    if len(matches) != 1:
        return ()
    degrees = float(matches[0].group(1))
    if not math.isfinite(degrees):
        return ()
    return (("incidence_rad", f"{math.radians(degrees):.17g}", "filename:degree token"),)


def apply_mapped_rows(
    project: Project,
    rows: tuple[tuple[str, ...], ...],
    mapping: tuple[str | None, ...],
    *,
    selected_ids: tuple[UUID, ...] = (),
    origin: str,
) -> Project:
    """Apply explicit columns by UUID or selected UUID order, atomically."""
    if not rows or len(rows) > MAX_METADATA_ROWS or len(mapping) > MAX_METADATA_COLUMNS:
        raise ProjectFormatError("metadata mapping is out of bounds")
    if any(field is not None and field not in MAPPED_FIELDS for field in mapping):
        raise ProjectFormatError("unsupported metadata column")
    active = [field for field in mapping if field is not None]
    if len(active) != len(set(active)):
        raise ProjectFormatError("metadata fields must map to unique columns")
    if "acquisition_id" not in active and len(rows) != len(selected_ids):
        raise ProjectFormatError("paste row count must match selected acquisitions")
    by_id = {item.acquisition_id: item for item in project.acquisitions}
    seen: set[UUID] = set()
    for index, row in enumerate(rows):
        if len(row) > len(mapping):
            raise ProjectFormatError("metadata row exceeds its declared columns")
        fields = {
            field: row[column].strip()
            for column, field in enumerate(mapping)
            if field is not None and column < len(row)
        }
        if any(len(value) > MAX_METADATA_FIELD_LENGTH for value in fields.values()):
            raise ProjectFormatError(f"row {index + 1} has an overlong mapped field")
        try:
            acquisition_id = (
                UUID(fields.pop("acquisition_id"))
                if "acquisition_id" in fields
                else selected_ids[index]
            )
        except ValueError as exc:
            raise ProjectFormatError(f"row {index + 1} has invalid acquisition UUID") from exc
        if acquisition_id not in by_id:
            raise ProjectFormatError(f"row {index + 1} names an unknown acquisition")
        if acquisition_id in seen:
            raise ProjectFormatError(f"row {index + 1} repeats an acquisition UUID")
        seen.add(acquisition_id)
        old = by_id[acquisition_id].metadata
        updates = {}
        sources = dict(old.provenance)
        for name, supplied in fields.items():
            if not supplied:
                continue
            if supplied == "?":
                updates["incidence_rad" if name == "incidence_deg" else name] = None
                sources.pop("incidence_rad" if name == "incidence_deg" else name, None)
                continue
            key = "incidence_rad" if name == "incidence_deg" else name
            if name == "incidence_deg":
                try:
                    numeric = float(supplied)
                except ValueError as exc:
                    raise ProjectFormatError(f"row {index + 1} has invalid angle degrees") from exc
                updates[key] = math.radians(numeric)
            elif name == "exposure_s":
                try:
                    updates[key] = float(supplied)
                except ValueError as exc:
                    raise ProjectFormatError(
                        f"row {index + 1} has invalid exposure seconds"
                    ) from exc
            elif name in ("dark_acquisition_id", "mask_acquisition_id"):
                try:
                    updates[key] = UUID(supplied)
                except ValueError as exc:
                    raise ProjectFormatError(
                        f"row {index + 1} has invalid association UUID"
                    ) from exc
            else:
                updates[key] = supplied
            sources[key] = origin
        if updates:
            updates["provenance"] = tuple(sorted(sources.items()))
            try:
                metadata = replace(old, **updates, revision=old.revision + 1)
            except (ProjectFormatError, ValueError) as exc:
                raise ProjectFormatError(f"row {index + 1}: {exc}") from exc
            by_id[acquisition_id] = replace(by_id[acquisition_id], metadata=metadata)
    try:
        return replace(
            project,
            acquisitions=tuple(by_id[item.acquisition_id] for item in project.acquisitions),
        )
    except ValueError as exc:
        raise ProjectFormatError(f"metadata associations after all rows: {exc}") from exc


def validate_reference(path: Path, kind: str) -> tuple[str, str | None]:
    """Use current scientific readers on a bounded, explicit local reference."""
    source = Path(path).absolute()
    if kind not in ("cif", "configuration"):
        raise ValueError("unsupported reference kind")
    with source.open("rb") as stream:
        source_bytes = stream.read(MAX_REFERENCE_BYTES + 1)
    if len(source_bytes) > MAX_REFERENCE_BYTES:
        raise ProjectFormatError("reference exceeds 1 MiB picker limit")
    digest = hashlib.sha256(source_bytes).hexdigest()
    if kind == "cif":
        from rasim_next.materials.crystal import read_crystal

        crystal = read_crystal(source, expected_sha256=digest, source_bytes=source_bytes)
        return digest, crystal.phase_id
    from rasim_next.pipeline.configured_simulation import load_simulation_config

    config = load_simulation_config(
        source, source_bytes=source_bytes, max_referenced_cif_bytes=MAX_REFERENCE_BYTES
    )
    return digest, config.material.phase_id


def metadata_csv(project: Project) -> str:
    """Export stable UUIDs, declared units and provenance without pixel data."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        (
            *MAPPED_FIELDS,
            "native_rows",
            "native_columns",
            "source_path",
            "source_sha256",
            "source_hash_kind",
            "cif_path",
            "cif_sha256",
            "configuration_path",
            "configuration_sha256",
            "provenance_json",
            "proposals_json",
            "metadata_revision",
        )
    )
    for item in project.acquisitions:
        metadata: AcquisitionMetadata = item.metadata
        writer.writerow(
            (
                str(item.acquisition_id),
                metadata.role or "",
                metadata.specimen or "",
                metadata.mount or "",
                ""
                if metadata.incidence_rad is None
                else f"{math.degrees(metadata.incidence_rad):.17g}",
                "" if metadata.exposure_s is None else f"{metadata.exposure_s:.17g}",
                metadata.detector_setup or "",
                metadata.material_id or "",
                metadata.calibrant_id or "",
                "" if metadata.dark_acquisition_id is None else str(metadata.dark_acquisition_id),
                "" if metadata.mask_acquisition_id is None else str(metadata.mask_acquisition_id),
                "" if metadata.native_shape is None else metadata.native_shape[0],
                "" if metadata.native_shape is None else metadata.native_shape[1],
                str(item.source_path),
                item.source_sha256,
                SOURCE_HASH_KIND,
                "" if metadata.cif_path is None else str(metadata.cif_path),
                metadata.cif_sha256 or "",
                "" if metadata.configuration_path is None else str(metadata.configuration_path),
                metadata.configuration_sha256 or "",
                json.dumps(metadata.provenance, separators=(",", ":")),
                json.dumps(metadata.proposals, separators=(",", ":")),
                metadata.revision,
            )
        )
    result = stream.getvalue()
    if len(result.encode("utf-8")) > MAX_METADATA_EXPORT_BYTES:
        raise ProjectFormatError("metadata export exceeds 1 MiB")
    return result
