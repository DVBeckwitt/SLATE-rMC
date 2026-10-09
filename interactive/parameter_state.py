"""Bounded numeric draft and per-field session history; no fitting or rendering owner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import yaml
from numeric_fields import description, validate_proposal
from project_state import (
    MAX_NUMERIC_BASELINE_BYTES,
    Acquisition,
    AcquisitionMetadata,
    NumericDraft,
    Project,
    ProjectFormatError,
)

if TYPE_CHECKING:
    from rasim_next.pipeline.configured_simulation import SimulationConfiguration

MAX_HISTORY_ACTIONS = 32
MAX_HISTORY_BYTES = 256 * 1024


def _value_at(mapping: dict[str, Any], path: tuple[str | int, ...]) -> Any:
    value: Any = mapping
    try:
        for part in path:
            value = value[part]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProjectFormatError("numeric field is absent from this configuration") from exc
    return value


def _set_at(mapping: dict[str, Any], path: tuple[str | int, ...], value: float) -> None:
    target: Any = mapping
    try:
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
    except (KeyError, IndexError, TypeError) as exc:
        raise ProjectFormatError("numeric field is absent from this configuration") from exc


def _source_mapping(draft: NumericDraft) -> dict[str, Any]:
    from rasim_next.pipeline.configured_simulation import load_strict_yaml_mapping

    mapping = load_strict_yaml_mapping(
        draft.configuration_path, source_bytes=draft.baseline_yaml.encode("utf-8")
    )
    mapping["instrument"].setdefault(
        "detector_tilt",
        {
            "about_column_axis_deg": 0.0,
            "about_row_axis_deg": 0.0,
        },
    )
    return mapping


def configured_draft(draft: NumericDraft, storage=()) -> SimulationConfiguration:
    """Validate the complete edited input with the authoritative configured reader."""
    from rasim_next.pipeline.configured_simulation import load_simulation_config

    mapping = _source_mapping(draft)
    for field, value, unit, _origin in draft.proposed:
        try:
            validate_proposal(field, value, unit)
        except ValueError as exc:
            raise ProjectFormatError(str(exc)) from exc
        item = description(field)
        _set_at(mapping, item.path, value)
    encoded = yaml.safe_dump(mapping, sort_keys=False).encode("utf-8")
    if len(encoded) > MAX_NUMERIC_BASELINE_BYTES:
        raise ProjectFormatError("edited configuration exceeds the 64 KiB draft limit")
    config = load_simulation_config(
        draft.configuration_path,
        source_bytes=encoded,
        max_referenced_cif_bytes=1024 * 1024,
        stored_paths={Path(r["original"]).resolve(): Path(r["stored"]) for r in storage},
    )
    from archive_storage import stored_path

    if (
        config.material.cif_path != stored_path(draft.cif_path, storage, draft.cif_sha256)
        or config.cif_sha256 != draft.cif_sha256
    ):
        raise ProjectFormatError("configuration-dependent CIF identity changed")
    return config


def prepare_numeric_draft(argument: bytes, control: Any) -> Any:
    """Load and validate bounded configuration bytes on the existing job worker."""
    from job_lifecycle import JobResult

    request = json.loads(argument)
    if type(request) is not dict or set(request) != {
        "acquisition_id",
        "source_sha256",
        "configuration_path",
        "configuration_sha256",
        "cif_path",
        "cif_sha256",
        "proposed",
        "revision",
    }:
        raise ProjectFormatError("invalid numeric draft request")
    path = Path(request["configuration_path"])
    with path.open("rb") as handle:
        encoded = handle.read(MAX_NUMERIC_BASELINE_BYTES + 1)
    if control.canceled:
        return JobResult(None, 0)
    if not encoded or len(encoded) > MAX_NUMERIC_BASELINE_BYTES:
        raise ProjectFormatError("numeric draft configuration must be at most 64 KiB")
    draft = NumericDraft(
        UUID(request["acquisition_id"]),
        request["source_sha256"],
        path,
        request["configuration_sha256"],
        Path(request["cif_path"]),
        request["cif_sha256"],
        encoded.decode("utf-8"),
        tuple(tuple(entry) for entry in request["proposed"]),
        request["revision"],
    )
    configured_draft(draft)
    return JobResult(draft, 4 * len(encoded) + 1024)


def displayed_value(draft: NumericDraft, field: str) -> str:
    item = description(field)
    if not item.editable:
        return "Read only"
    values = {name: value for name, value, _, _ in draft.proposed}
    baseline = _source_mapping(draft)
    stored = values.get(field, _value_at(baseline, item.path))
    return format(float(stored) / item.display_to_stored, ".17g")


def edit_draft(draft: NumericDraft, field: str, display_text: str) -> NumericDraft:
    item = description(field)
    if not item.editable:
        raise ProjectFormatError(item.reason)
    try:
        display_value = float(display_text.strip())
    except (ValueError, TypeError) as exc:
        raise ProjectFormatError(
            f"{item.label} must be a finite {item.display_unit} number"
        ) from exc
    stored = display_value * item.display_to_stored
    try:
        validate_proposal(field, stored, item.stored_unit)
    except ValueError as exc:
        raise ProjectFormatError(str(exc)) from exc
    baseline = _value_at(_source_mapping(draft), item.path)
    proposals = {name: (value, unit, origin) for name, value, unit, origin in draft.proposed}
    if stored == baseline:
        proposals.pop(field, None)
    else:
        proposals[field] = (stored, item.stored_unit, "manual proposed initial geometry")
    updated = replace(
        draft,
        proposed=tuple((name, *values) for name, values in sorted(proposals.items())),
        revision=draft.revision + 1,
    )
    if updated.proposed == draft.proposed:
        return draft
    configured_draft(updated)
    return updated


@dataclass(frozen=True, slots=True)
class LaunchSnapshot:
    project_id: UUID
    acquisition_id: UUID
    source_sha256: str
    configuration_sha256: str
    cif_sha256: str
    draft_revision: int
    proposed: tuple[tuple[str, float, str, str], ...]
    configured: SimulationConfiguration
    kind: str = "configured_geometry_draft.v1"


def freeze_draft(project: Project, acquisition: Acquisition, draft: NumericDraft) -> LaunchSnapshot:
    metadata = acquisition.metadata
    if (
        draft.acquisition_id != acquisition.acquisition_id
        or draft.source_sha256 != acquisition.source_sha256
        or metadata.configuration_path != draft.configuration_path
        or metadata.configuration_sha256 != draft.configuration_sha256
        or metadata.configuration_cif_path != draft.cif_path
        or metadata.configuration_cif_sha256 != draft.cif_sha256
    ):
        raise ProjectFormatError("numeric draft is stale for this acquisition or reference")
    for path, digest in (
        (draft.configuration_path, draft.configuration_sha256),
        (draft.cif_path, draft.cif_sha256),
    ):
        with path.open("rb") as handle:
            encoded = handle.read(1024 * 1024 + 1)
        if len(encoded) > 1024 * 1024 or hashlib.sha256(encoded).hexdigest() != digest:
            raise ProjectFormatError(f"{path.name} changed since reference validation")
    return LaunchSnapshot(
        project.project_id,
        acquisition.acquisition_id,
        acquisition.source_sha256,
        draft.configuration_sha256,
        draft.cif_sha256,
        draft.revision,
        draft.proposed,
        configured_draft(draft),
    )


@dataclass(frozen=True, slots=True)
class FieldChange:
    acquisition_id: UUID
    field: str
    before: Any
    after: Any


@dataclass(frozen=True, slots=True)
class HistoryAction:
    label: str
    changes: tuple[FieldChange, ...]
    bytes_used: int


def metadata_action(before: Project, after: Project, label: str) -> HistoryAction | None:
    if before.project_id != after.project_id or {
        item.acquisition_id for item in before.acquisitions
    } != {item.acquisition_id for item in after.acquisitions}:
        raise ProjectFormatError("metadata transaction changed project or acquisition membership")
    old_items = {item.acquisition_id: item for item in before.acquisitions}
    old = {key: item.metadata for key, item in old_items.items()}
    changes = []
    for item in after.acquisitions:
        prior = old[item.acquisition_id]
        current = item.metadata
        for field_info in fields(AcquisitionMetadata):
            name = field_info.name
            if name == "revision":
                continue
            before_value, after_value = getattr(prior, name), getattr(current, name)
            if name in ("provenance", "proposals"):
                old_entries = {entry[0]: entry[1:] for entry in before_value}
                new_entries = {entry[0]: entry[1:] for entry in after_value}
                for key in sorted(old_entries.keys() | new_entries.keys()):
                    if old_entries.get(key) != new_entries.get(key):
                        changes.append(
                            FieldChange(
                                item.acquisition_id,
                                f"{name}.{key}",
                                old_entries.get(key),
                                new_entries.get(key),
                            )
                        )
            elif before_value != after_value:
                changes.append(FieldChange(item.acquisition_id, name, before_value, after_value))
        prior_item = old_items[item.acquisition_id]
        for name in ("source_path", "initial_values", "setup_receipt", "storage"):
            if getattr(prior_item, name) != getattr(item, name):
                changes.append(
                    FieldChange(
                        item.acquisition_id,
                        f"acquisition.{name}",
                        getattr(prior_item, name),
                        getattr(item, name),
                    )
                )
    return _action(label, changes)


def combined_action(
    before: Project,
    after: Project,
    old_draft: NumericDraft | None,
    new_draft: NumericDraft | None,
    label: str,
) -> HistoryAction | None:
    action = metadata_action(before, after, label)
    changes = list(action.changes) if action is not None else []
    if old_draft is not None and new_draft is not None and old_draft != new_draft:
        if (old_draft.configuration_sha256, old_draft.configuration_path, old_draft.cif_path) != (
            new_draft.configuration_sha256,
            new_draft.configuration_path,
            new_draft.cif_path,
        ):
            changes.append(
                FieldChange(old_draft.acquisition_id, "draft_snapshot", old_draft, new_draft)
            )
        else:
            numeric = draft_action(old_draft, new_draft, label)
            if numeric is not None:
                changes.extend(numeric.changes)
    return _action(label, changes)


def draft_action(before: NumericDraft, after: NumericDraft, label: str) -> HistoryAction | None:
    if (before.acquisition_id, before.configuration_sha256) != (
        after.acquisition_id,
        after.configuration_sha256,
    ):
        raise ProjectFormatError("numeric draft identity changed within one edit")
    old = {entry[0]: entry[1:] for entry in before.proposed}
    new = {entry[0]: entry[1:] for entry in after.proposed}
    changes = [
        FieldChange(before.acquisition_id, f"draft.{key}", old.get(key), new.get(key))
        for key in sorted(old.keys() | new.keys())
        if old.get(key) != new.get(key)
    ]
    return _action(label, changes)


def _action(label: str, changes: list[FieldChange]) -> HistoryAction | None:
    if not changes:
        return None
    size = sum(len(repr(change).encode("utf-8")) for change in changes)
    if size > MAX_HISTORY_BYTES:
        raise ProjectFormatError("one edit exceeds the 256 KiB undo limit")
    return HistoryAction(label, tuple(changes), size)


def _apply_action(
    project: Project, draft: NumericDraft | None, action: HistoryAction, *, undo: bool
) -> tuple[Project, NumericDraft | None, tuple[tuple[UUID, str], ...]]:
    by_id = {item.acquisition_id: item for item in project.acquisitions}
    grouped: dict[UUID, list[FieldChange]] = {}
    for change in action.changes:
        grouped.setdefault(change.acquisition_id, []).append(change)
    replacement = {}
    reference_changes = []
    next_draft = draft
    for acquisition_id, changes in grouped.items():
        if acquisition_id not in by_id:
            raise ProjectFormatError("undo target acquisition was removed")
        item = by_id[acquisition_id]
        direct = {}
        acquisition_changes = {}
        provenance = {key: (value,) for key, value in item.metadata.provenance}
        proposals = {key: (value, origin) for key, value, origin in item.metadata.proposals}
        draft_values = (
            {entry[0]: entry[1:] for entry in next_draft.proposed}
            if next_draft is not None and next_draft.acquisition_id == acquisition_id
            else None
        )
        for change in changes:
            expected, restored = (
                (change.after, change.before) if undo else (change.before, change.after)
            )
            if change.field == "draft_snapshot":
                if (
                    next_draft is None
                    or replace(next_draft, revision=expected.revision) != expected
                ):
                    raise ProjectFormatError("reference draft undo conflicts with current draft")
                next_draft = replace(restored, revision=next_draft.revision + 1)
                reference_changes.append((acquisition_id, "configuration"))
            elif change.field.startswith("acquisition."):
                name = change.field.split(".", 1)[1]
                if (
                    name not in ("source_path", "initial_values", "setup_receipt", "storage")
                    or getattr(item, name) != expected
                ):
                    raise ProjectFormatError("acquisition undo conflicts with current field")
                acquisition_changes[name] = restored
                if name == "source_path":
                    reference_changes.append((acquisition_id, "osc"))
                if name == "initial_values":
                    reference_changes.append((acquisition_id, "initial_values"))
            elif change.field.startswith("draft."):
                if draft_values is None or draft_values.get(change.field[6:]) != expected:
                    raise ProjectFormatError("numeric undo conflicts with current draft")
                if restored is None:
                    draft_values.pop(change.field[6:], None)
                else:
                    draft_values[change.field[6:]] = restored
            elif change.field.startswith("provenance.") or change.field.startswith("proposals."):
                kind, key = change.field.split(".", 1)
                values = provenance if kind == "provenance" else proposals
                if values.get(key) != expected:
                    raise ProjectFormatError("metadata undo conflicts with current provenance")
                if restored is None:
                    values.pop(key, None)
                else:
                    values[key] = restored
            else:
                if getattr(item.metadata, change.field) != expected:
                    raise ProjectFormatError("metadata undo conflicts with current field")
                direct[change.field] = restored
                if change.field.endswith(("_path", "_sha256")):
                    kind = (
                        "configuration_cif"
                        if change.field.startswith("configuration_cif_")
                        else change.field.split("_")[0]
                    )
                    reference_changes.append((acquisition_id, kind))
        if draft_values is not None and any(
            change.field.startswith("draft.") for change in changes
        ):
            next_draft = replace(
                next_draft,
                proposed=tuple((name, *values) for name, values in sorted(draft_values.items())),
                revision=next_draft.revision + 1,
            )
            configured_draft(next_draft)
        if any(not change.field.startswith("draft.") for change in changes):
            replacement[acquisition_id] = replace(
                item,
                **acquisition_changes,
                metadata=replace(
                    item.metadata,
                    **direct,
                    provenance=tuple((key, value[0]) for key, value in sorted(provenance.items())),
                    proposals=tuple((key, *value) for key, value in sorted(proposals.items())),
                    revision=item.metadata.revision + 1,
                ),
            )
    next_project = replace(
        project,
        acquisitions=tuple(
            replacement.get(item.acquisition_id, item) for item in project.acquisitions
        ),
    )
    return next_project, next_draft, tuple(sorted(set(reference_changes)))


class SessionHistory:
    def __init__(self) -> None:
        self.undo_actions: list[HistoryAction] = []
        self.redo_actions: list[HistoryAction] = []
        self.bytes_used = 0

    def push(self, action: HistoryAction | None) -> bool:
        if action is None:
            return False
        self.bytes_used -= sum(item.bytes_used for item in self.redo_actions)
        self.redo_actions.clear()
        self.undo_actions.append(action)
        self.bytes_used += action.bytes_used
        while len(self.undo_actions) > MAX_HISTORY_ACTIONS or self.bytes_used > MAX_HISTORY_BYTES:
            self.bytes_used -= self.undo_actions.pop(0).bytes_used
        return True

    def discard_draft_actions(self) -> None:
        for stack in (self.undo_actions, self.redo_actions):
            retained = []
            for action in stack:
                changes = [
                    change
                    for change in action.changes
                    if not change.field.startswith("draft.") and change.field != "draft_snapshot"
                ]
                if len(changes) == len(action.changes):
                    retained.append(action)
                elif changes:
                    retained_action = _action(action.label, changes)
                    assert retained_action is not None
                    retained.append(retained_action)
            stack[:] = retained
        self.bytes_used = sum(
            action.bytes_used for action in (*self.undo_actions, *self.redo_actions)
        )

    def apply(
        self,
        project: Project,
        draft: NumericDraft | None,
        *,
        undo: bool,
        validate: Callable[[Project, NumericDraft | None], None] | None = None,
    ) -> tuple[Project, NumericDraft | None, tuple[tuple[UUID, str], ...], str]:
        source = self.undo_actions if undo else self.redo_actions
        if not source:
            raise ProjectFormatError("nothing to undo" if undo else "nothing to redo")
        action = source[-1]
        updated, next_draft, references = _apply_action(project, draft, action, undo=undo)
        if validate is not None:
            validate(updated, next_draft)
        source.pop()
        (self.redo_actions if undo else self.undo_actions).append(action)
        return updated, next_draft, references, action.label
