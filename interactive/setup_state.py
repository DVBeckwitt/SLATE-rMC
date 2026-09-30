"""Bounded reusable setup values using canonical metadata and configured inputs."""

import json
from dataclasses import dataclass, replace
from uuid import UUID, uuid4

from numeric_fields import PARAMETERS
from parameter_state import _source_mapping, _value_at, freeze_draft, prepare_numeric_draft
from project_state import (
    AcquisitionMetadata,
    NumericDraft,
    Project,
    ProjectFormatError,
    SetupReceipt,
    validate_initial_values,
)

MAX_TEMPLATE_BYTES = 64 * 1024
TEMPLATE_FIELDS = (
    "role",
    "specimen",
    "mount",
    "incidence_rad",
    "exposure_s",
    "detector_setup",
    "material_id",
    "calibrant_id",
)
METADATA_UNITS = ("label", "label", "label", "rad", "s", "label", "label", "preset")


@dataclass(frozen=True, slots=True)
class SetupTemplate:
    template_id: UUID
    name: str
    revision: int
    metadata: tuple[tuple[str, object, str, str], ...] = ()
    initial_values: tuple[tuple[str, float, str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.template_id, UUID)
            or type(self.name) is not str
            or not 0 < len(self.name.strip()) <= 256
            or self.name != self.name.strip()
        ):
            raise ProjectFormatError("template needs a UUID and a name of at most 256 characters")
        if type(self.revision) is not int or not 0 <= self.revision < 2**63:
            raise ProjectFormatError("invalid template revision")
        if type(self.metadata) is not tuple or len(self.metadata) > len(TEMPLATE_FIELDS):
            raise ProjectFormatError("invalid reusable metadata")
        values = {}
        for row in self.metadata:
            if type(row) is not tuple or len(row) != 4:
                raise ProjectFormatError("invalid reusable metadata entry")
            name, value, unit, origin = row
            if name not in TEMPLATE_FIELDS or name in values or value is None:
                raise ProjectFormatError("unsupported or duplicate reusable metadata")
            if unit != METADATA_UNITS[TEMPLATE_FIELDS.index(name)]:
                raise ProjectFormatError("incorrect template metadata unit")
            if type(origin) is not str or not 0 < len(origin) <= 256 or origin != origin.strip():
                raise ProjectFormatError("invalid template field origin")
            values[name] = value
        AcquisitionMetadata(**values)
        validate_initial_values(self.initial_values)


def template_document(template: SetupTemplate) -> dict:
    return {
        "schema": "slate.setup.v1",
        "id": str(template.template_id),
        "name": template.name,
        "revision": template.revision,
        "metadata": [list(row) for row in template.metadata],
        "initial_values": [list(row) for row in template.initial_values],
    }


def template_bytes(template: SetupTemplate) -> bytes:
    encoded = (
        json.dumps(template_document(template), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if len(encoded) > MAX_TEMPLATE_BYTES:
        raise ProjectFormatError("template exceeds 64 KiB")
    return encoded


def template_from_bytes(encoded: bytes) -> SetupTemplate:
    if not 0 < len(encoded) <= MAX_TEMPLATE_BYTES:
        raise ProjectFormatError("template must be at most 64 KiB")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProjectFormatError("duplicate template JSON key")
            result[key] = value
        return result

    try:
        row = json.loads(encoded, object_pairs_hook=unique)
        if (
            type(row) is not dict
            or set(row) != {"schema", "id", "name", "revision", "metadata", "initial_values"}
            or row["schema"] != "slate.setup.v1"
        ):
            raise ProjectFormatError("unsupported template document")
        if any(
            type(row[name]) is not list or any(type(v) is not list for v in row[name])
            for name in ("metadata", "initial_values")
        ):
            raise ProjectFormatError("template entries must be lists")
        return SetupTemplate(
            UUID(row["id"]),
            row["name"],
            row["revision"],
            tuple(tuple(v) for v in row["metadata"]),
            tuple(tuple(v) for v in row["initial_values"]),
        )
    except (UnicodeError, TypeError, ValueError, RecursionError) as exc:
        raise ProjectFormatError(f"invalid template: {exc}") from exc


def capture_template(
    name: str, metadata: AcquisitionMetadata, draft: NumericDraft | None
) -> SetupTemplate:
    origins = dict(metadata.provenance)
    reusable = tuple(
        (field, getattr(metadata, field), unit, origins.get(field, "declared project value"))
        for field, unit in zip(TEMPLATE_FIELDS, METADATA_UNITS, strict=True)
        if getattr(metadata, field) is not None
    )
    values = ()
    if draft is not None:
        mapping = _source_mapping(draft)
        proposed = {row[0]: row[1:] for row in draft.proposed}
        rows = []
        for item in PARAMETERS:
            if not item.editable:
                continue
            try:
                value = _value_at(mapping, item.path)
            except ProjectFormatError:
                continue
            stored, unit, origin = proposed.get(
                item.field, (value, item.stored_unit, "configuration file declared value")
            )
            rows.append((item.field, stored, unit, origin))
        values = tuple(rows)
    return SetupTemplate(uuid4(), name, 0, reusable, values)


def apply_template(
    project: Project,
    draft: NumericDraft | None,
    template: SetupTemplate,
    acquisition_ids: tuple[UUID, ...],
    digest: str,
    control,
) -> tuple[Project, NumericDraft | None, tuple[str, ...]]:
    """Copy only reviewed fields, validating complete target configurations in a worker."""
    if (
        not acquisition_ids
        or len(set(acquisition_ids)) != len(acquisition_ids)
        or not set(acquisition_ids) <= {item.acquisition_id for item in project.acquisitions}
    ):
        raise ProjectFormatError("select unique current acquisitions for template review")
    receipt = SetupReceipt(template.template_id, template.name, digest)
    origin = f"template:{template.template_id}:{digest[:16]}:{template.name[:100]}"
    review, result = [], []
    next_draft = draft
    for acquisition in project.acquisitions:
        if control.canceled:
            raise RuntimeError("template validation canceled")
        if acquisition.acquisition_id not in acquisition_ids:
            result.append(acquisition)
            continue
        metadata, changes = acquisition.metadata, {}
        provenance = dict(metadata.provenance)
        for field, value, unit, prior_origin in template.metadata:
            if getattr(metadata, field) != value:
                review.append(
                    f"{acquisition.name} [{acquisition.acquisition_id}] {field}: {getattr(metadata, field)!r} -> {value!r} {unit}; {prior_origin} -> {origin}"
                )
                changes[field], provenance[field] = value, origin
        values = {row[0]: row[1:] for row in acquisition.initial_values}
        matching = draft is not None and draft.acquisition_id == acquisition.acquisition_id
        if matching:
            values.update({row[0]: row[1:] for row in draft.proposed})
        for field, value, unit, prior_origin in template.initial_values:
            previous = values.get(field)
            if previous is None or previous[0] != value:
                review.append(
                    f"{acquisition.name} [{acquisition.acquisition_id}] {field}: {previous} -> {value} {unit}; {prior_origin} -> {origin}"
                )
                values[field] = (value, unit, origin)
        initial = tuple((key, *value) for key, value in sorted(values.items()))
        if template.initial_values:
            if metadata.configuration_path is None or metadata.configuration_cif_path is None:
                raise ProjectFormatError(
                    f"{acquisition.name}: numeric defaults need a compatible configuration"
                )
            if matching:
                target = replace(draft, proposed=initial, revision=draft.revision + 1)
            else:
                request = {
                    "acquisition_id": str(acquisition.acquisition_id),
                    "source_sha256": acquisition.source_sha256,
                    "configuration_path": str(metadata.configuration_path),
                    "configuration_sha256": metadata.configuration_sha256,
                    "cif_path": str(metadata.configuration_cif_path),
                    "cif_sha256": metadata.configuration_cif_sha256,
                    "proposed": initial,
                    "revision": 0,
                }
                target = prepare_numeric_draft(json.dumps(request).encode(), control).value
            configured = freeze_draft(project, acquisition, target).configured
            if (
                metadata.native_shape is not None
                and tuple(configured.instrument.detector_shape_rc) != metadata.native_shape
            ):
                raise ProjectFormatError(
                    f"{acquisition.name}: configuration detector shape is incompatible"
                )
            if matching and target.proposed != draft.proposed:
                next_draft = target
        if changes:
            metadata = replace(
                metadata,
                **changes,
                provenance=tuple(sorted(provenance.items())),
                revision=metadata.revision + 1,
            )
        changed = bool(changes) or initial != acquisition.initial_values
        result.append(
            replace(acquisition, metadata=metadata, initial_values=initial, setup_receipt=receipt)
            if changed
            else acquisition
        )
    return replace(project, acquisitions=tuple(result)), next_draft, tuple(review)
