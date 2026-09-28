"""Qt-free project identity, strict JSON state and decoded-source references."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import UUID, uuid4

PROJECT_SCHEMA_VERSION = 1
SOURCE_HASH_KIND = "sha256:decoded-osc-header-and-payload"
MAX_PROJECT_BYTES = 1024 * 1024
MAX_ACQUISITIONS = 128
MAX_NAME_LENGTH = 256
MAX_SOURCE_PATH_LENGTH = 4096


class ProjectFormatError(ValueError):
    """An unsupported or malformed desktop project document."""


@dataclass(frozen=True, slots=True)
class Acquisition:
    acquisition_id: UUID
    name: str
    source_path: Path
    source_sha256: str

    @classmethod
    def create(
        cls,
        name: str,
        source_path: Path,
        source_sha256: str,
        *,
        acquisition_id: UUID | None = None,
    ) -> "Acquisition":
        name = _name(name.strip(), "acquisition name")
        if len(source_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in source_sha256
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        return cls(acquisition_id or uuid4(), name, Path(source_path), source_sha256)


@dataclass(frozen=True, slots=True)
class Project:
    project_id: UUID
    name: str
    acquisitions: tuple[Acquisition, ...] = ()
    schema_version: int = PROJECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _name(self.name, "project name")
        if self.schema_version != PROJECT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported project schema version: {self.schema_version}")
        ids = [item.acquisition_id for item in self.acquisitions]
        if len(ids) != len(set(ids)):
            raise ValueError("Acquisition IDs must be unique within a project")
        if len(ids) > MAX_ACQUISITIONS:
            raise ValueError(f"Project exceeds {MAX_ACQUISITIONS} acquisitions")

    @classmethod
    def create(cls, name: str = "Untitled project") -> "Project":
        name = _name(name.strip(), "project name")
        return cls(uuid4(), name)

    def add_acquisition(self, name: str, source_path: Path, source_sha256: str) -> "Project":
        acquisition = Acquisition.create(name, source_path, source_sha256)
        return replace(self, acquisitions=(*self.acquisitions, acquisition))

    def rename_acquisition(self, acquisition_id: UUID, name: str) -> "Project":
        name = _name(name.strip(), "acquisition name")
        if not any(item.acquisition_id == acquisition_id for item in self.acquisitions):
            raise KeyError(acquisition_id)
        return replace(
            self,
            acquisitions=tuple(
                replace(item, name=name) if item.acquisition_id == acquisition_id else item
                for item in self.acquisitions
            ),
        )

    def reorder_acquisitions(self, order: Sequence[UUID]) -> "Project":
        by_id = {item.acquisition_id: item for item in self.acquisitions}
        if len(order) != len(by_id) or set(order) != set(by_id):
            raise ValueError("Order must contain each acquisition ID exactly once")
        return replace(self, acquisitions=tuple(by_id[acquisition_id] for acquisition_id in order))


@dataclass(frozen=True, slots=True)
class DetectorViewState:
    column_px: int
    row_px: int
    zoom: float
    pan_x_px: float
    pan_y_px: float
    low_value: float
    high_value: float
    positive_log: bool

    def __post_init__(self) -> None:
        if (
            type(self.column_px) is not int
            or type(self.row_px) is not int
            or min(self.column_px, self.row_px) < 0
        ):
            raise ProjectFormatError("detector crosshair coordinates must be nonnegative integers")
        zoom = _float(self.zoom, "zoom")
        _float(self.pan_x_px, "pan_x_px")
        _float(self.pan_y_px, "pan_y_px")
        low = _float(self.low_value, "low_value")
        high = _float(self.high_value, "high_value")
        if not 0.25 <= zoom <= 30.0 or high <= low or type(self.positive_log) is not bool:
            raise ProjectFormatError("detector view has invalid zoom, levels or contrast mode")
        if self.positive_log and low <= 0.0:
            raise ProjectFormatError("positive-log contrast needs a positive lower level")


@dataclass(frozen=True, slots=True)
class ProjectViewState:
    selected_acquisition_id: UUID | None = None
    workspace: str = "fit_experiments"
    detector: DetectorViewState | None = None


@dataclass(frozen=True, slots=True)
class ProjectDocument:
    project: Project
    view: ProjectViewState

    def __post_init__(self) -> None:
        selected = self.view.selected_acquisition_id
        if selected is not None and not any(
            item.acquisition_id == selected for item in self.project.acquisitions
        ):
            raise ProjectFormatError("selected acquisition is absent from the project")
        if self.view.workspace not in ("fit_experiments", "simulator"):
            raise ProjectFormatError("unsupported workspace view")
        if self.view.detector is not None and selected is None:
            raise ProjectFormatError("detector view needs a selected acquisition")


def _object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ProjectFormatError(f"{label} must contain exactly {', '.join(sorted(keys))}")
    return value


def _name(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > MAX_NAME_LENGTH
    ):
        raise ProjectFormatError(
            f"{label} must be a nonempty name of at most {MAX_NAME_LENGTH} characters"
        )
    return value


def _uuid(value: Any, label: str) -> UUID:
    if type(value) is not str:
        raise ProjectFormatError(f"{label} must be a UUID string")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ProjectFormatError(f"{label} must be a UUID string") from exc
    if str(parsed) != value:
        raise ProjectFormatError(f"{label} must use canonical lowercase UUID form")
    return parsed


def _digest(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ProjectFormatError("source_sha256 must be a lowercase decoded-stream SHA-256")
    return value


def _float(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ProjectFormatError(f"{label} must be a finite number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ProjectFormatError(f"{label} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ProjectFormatError(f"{label} must be a finite number")
    return converted


def _detector_view(value: Any) -> DetectorViewState | None:
    if value is None:
        return None
    fields = {
        "column_px",
        "row_px",
        "zoom",
        "pan_x_px",
        "pan_y_px",
        "low_value",
        "high_value",
        "positive_log",
    }
    data = _object(value, fields, "detector view")
    return DetectorViewState(
        data["column_px"],
        data["row_px"],
        data["zoom"],
        data["pan_x_px"],
        data["pan_y_px"],
        data["low_value"],
        data["high_value"],
        data["positive_log"],
    )


def _source_reference(source: Path, document_path: Path) -> str:
    absolute = source.absolute()
    try:
        reference = absolute.relative_to(document_path.parent.absolute()).as_posix()
    except ValueError:
        reference = str(absolute)
    if not reference or len(reference) > MAX_SOURCE_PATH_LENGTH:
        raise ProjectFormatError(f"source path exceeds {MAX_SOURCE_PATH_LENGTH} characters")
    return reference


def project_to_document(state: ProjectDocument, document_path: Path) -> dict[str, Any]:
    """Encode only the currently supported numeric project and view state."""
    if len(state.project.acquisitions) > MAX_ACQUISITIONS:
        raise ProjectFormatError(f"project exceeds {MAX_ACQUISITIONS} acquisitions")
    acquisitions = [
        {
            "id": str(item.acquisition_id),
            "name": _name(item.name, "acquisition name"),
            "source_path": _source_reference(item.source_path, document_path),
            "source_sha256": _digest(item.source_sha256),
        }
        for item in state.project.acquisitions
    ]
    detector = state.view.detector
    view = {
        "selected_acquisition_id": (
            str(state.view.selected_acquisition_id)
            if state.view.selected_acquisition_id is not None
            else None
        ),
        "workspace": state.view.workspace,
        "detector": (
            None
            if detector is None
            else {
                "column_px": detector.column_px,
                "row_px": detector.row_px,
                "zoom": detector.zoom,
                "pan_x_px": detector.pan_x_px,
                "pan_y_px": detector.pan_y_px,
                "low_value": detector.low_value,
                "high_value": detector.high_value,
                "positive_log": detector.positive_log,
            }
        ),
    }
    document = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "source_hash_kind": SOURCE_HASH_KIND,
        "project": {
            "id": str(state.project.project_id),
            "name": _name(state.project.name, "project name"),
            "acquisitions": acquisitions,
        },
        "view": view,
    }
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > MAX_PROJECT_BYTES:
        raise ProjectFormatError(f"project document exceeds {MAX_PROJECT_BYTES} bytes")
    return document


def project_from_document(value: Any, document_path: Path) -> ProjectDocument:
    """Validate a complete project before replacing the current shell state."""
    top = _object(value, {"schema_version", "source_hash_kind", "project", "view"}, "document")
    if type(top["schema_version"]) is not int or top["schema_version"] != PROJECT_SCHEMA_VERSION:
        raise ProjectFormatError(f"unsupported project schema version {top['schema_version']!r}")
    if top["source_hash_kind"] != SOURCE_HASH_KIND:
        raise ProjectFormatError("unsupported source hash kind; expected decoded OSC bytes")
    data = _object(top["project"], {"id", "name", "acquisitions"}, "project")
    items = data["acquisitions"]
    if type(items) is not list or len(items) > MAX_ACQUISITIONS:
        raise ProjectFormatError(f"acquisitions must be a list of at most {MAX_ACQUISITIONS}")
    acquisitions = []
    for index, item in enumerate(items):
        row = _object(item, {"id", "name", "source_path", "source_sha256"}, f"acquisition {index}")
        reference = row["source_path"]
        if (
            type(reference) is not str
            or not reference
            or len(reference) > MAX_SOURCE_PATH_LENGTH
            or "\0" in reference
        ):
            raise ProjectFormatError(f"acquisition {index} has an invalid source path")
        source = Path(reference)
        if not source.is_absolute():
            windows = PureWindowsPath(reference)
            if windows.drive or windows.root:
                raise ProjectFormatError(
                    f"acquisition {index} source path is drive-relative or rooted"
                )
            source = document_path.parent / source
        acquisitions.append(
            Acquisition(
                _uuid(row["id"], f"acquisition {index} ID"),
                _name(row["name"], f"acquisition {index} name"),
                source.absolute(),
                _digest(row["source_sha256"]),
            )
        )
    try:
        project = Project(
            _uuid(data["id"], "project ID"),
            _name(data["name"], "project name"),
            tuple(acquisitions),
        )
    except ValueError as exc:
        raise ProjectFormatError(str(exc)) from exc
    view_data = _object(top["view"], {"selected_acquisition_id", "workspace", "detector"}, "view")
    selected = view_data["selected_acquisition_id"]
    selected_id = None if selected is None else _uuid(selected, "selected acquisition ID")
    return ProjectDocument(
        project,
        ProjectViewState(
            selected_id, view_data["workspace"], _detector_view(view_data["detector"])
        ),
    )


def read_project_document(path: Path) -> ProjectDocument:
    """Read at most one bounded JSON document; reject nonstandard constants."""
    with path.open("rb") as stream:
        encoded = stream.read(MAX_PROJECT_BYTES + 1)
    if len(encoded) > MAX_PROJECT_BYTES:
        raise ProjectFormatError(f"project document exceeds {MAX_PROJECT_BYTES} bytes")

    def reject_constant(token: str) -> None:
        raise ValueError(f"unsupported numeric constant {token}")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            encoded.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_keys
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProjectFormatError(f"invalid project JSON: {exc}") from exc
    return project_from_document(value, path)
