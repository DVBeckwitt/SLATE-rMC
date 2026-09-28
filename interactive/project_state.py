"""Qt-free project identity, strict JSON state and decoded-source references."""

import json
import math
import struct
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
FLOAT32_DISPLAY_MAX = 3.4028234663852886e38
FLOAT32_NORMAL_MIN = 1.1754943508222875e-38


class ProjectFormatError(ValueError):
    """An unsupported or malformed desktop project document."""


def _display_float32(value: float) -> float:
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError as exc:
        raise ProjectFormatError("display value exceeds the float32 shader range") from exc


def validate_display_limits(low: float, high: float, mode: str) -> None:
    if not all(math.isfinite(value) and abs(value) <= FLOAT32_DISPLAY_MAX for value in (low, high)):
        raise ProjectFormatError("display limits must be finite within the float32 shader range")
    low32, high32 = _display_float32(low), _display_float32(high)
    if high32 <= low32 or max(abs(low32), abs(high32)) < FLOAT32_NORMAL_MIN:
        raise ProjectFormatError(
            "display limits must be distinguishable at float32 shader precision"
        )
    if mode not in ("linear", "signed", "positive_log"):
        raise ProjectFormatError("unsupported detector contrast mode")
    if mode == "positive_log":
        if low32 < FLOAT32_NORMAL_MIN:
            raise ProjectFormatError(
                "positive-log display requires a normal positive float32 lower level"
            )
        if math.log(high32) - math.log(low32) <= 1.0e-5:
            raise ProjectFormatError("positive-log limits are too close for shader precision")
    if mode == "signed" and not (low32 <= -FLOAT32_NORMAL_MIN and high32 >= FLOAT32_NORMAL_MIN):
        raise ProjectFormatError("signed display needs normal float32 limits around zero")


def linear_display_limits(minimum: float, maximum: float) -> tuple[float, float]:
    """Choose finite float32-distinguishable display bounds without changing source values."""
    if not all(
        math.isfinite(value) and abs(value) <= FLOAT32_DISPLAY_MAX for value in (minimum, maximum)
    ):
        raise ProjectFormatError("finite image values exceed the float32 display range")
    if maximum < minimum:
        raise ProjectFormatError("image extrema are reversed")
    if (
        _display_float32(maximum) > _display_float32(minimum)
        and max(abs(minimum), abs(maximum)) >= FLOAT32_NORMAL_MIN
    ):
        return minimum, maximum
    span = 1.0 if minimum == 0 else max(abs(minimum) * 0.01, 1.0e-30)
    if minimum + span <= FLOAT32_DISPLAY_MAX:
        return minimum, minimum + span
    return minimum - span, minimum


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
    contrast_mode: str
    device_pixel_ratio: float | None = None
    scale_mode: str = "custom"
    show_image: bool = True
    show_crosshair: bool = True
    show_markers: bool = True
    profile_follow: bool = True
    profile_row_width: int = 1
    profile_column_width: int = 1
    profile_measure: str = "sum"
    profile_scope: str = "band"
    profile_roi: tuple[int, int, int, int] | None = None
    horizontal_intensity_limits: tuple[float, float] | None = None
    vertical_intensity_limits: tuple[float, float] | None = None

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
        dpr = (
            None
            if self.device_pixel_ratio is None
            else _float(self.device_pixel_ratio, "device_pixel_ratio")
        )
        if not 0.0001 <= zoom <= 10000.0:
            raise ProjectFormatError("detector view has invalid zoom, levels or contrast mode")
        if dpr is not None and not 0.25 <= dpr <= 16.0:
            raise ProjectFormatError("detector device pixel ratio is outside the supported range")
        validate_display_limits(low, high, self.contrast_mode)
        if self.scale_mode not in ("fit", "native", "custom"):
            raise ProjectFormatError("unsupported detector scale mode")
        if any(
            type(value) is not bool
            for value in (self.show_image, self.show_crosshair, self.show_markers)
        ):
            raise ProjectFormatError("detector layer visibility must be Boolean")
        if type(self.profile_follow) is not bool:
            raise ProjectFormatError("profile follow state must be Boolean")
        if any(
            type(width) is not int or not 1 <= width <= 16_384
            for width in (self.profile_row_width, self.profile_column_width)
        ):
            raise ProjectFormatError("profile band widths must be integers from 1 to 16384")
        if self.profile_measure not in ("sum", "mean") or self.profile_scope not in (
            "band",
            "full",
            "roi",
        ):
            raise ProjectFormatError("unsupported profile measure or scope")
        if self.profile_roi is not None:
            roi = self.profile_roi
            if (
                type(roi) is not tuple
                or len(roi) != 4
                or any(type(bound) is not int for bound in roi)
            ):
                raise ProjectFormatError("profile ROI needs four integer native bounds")
            c0, c1, r0, r1 = roi
            if not (0 <= c0 < c1 <= 16_384 and 0 <= r0 < r1 <= 16_384):
                raise ProjectFormatError("profile ROI bounds are invalid")
        if self.profile_scope == "roi" and self.profile_roi is None:
            raise ProjectFormatError("profile ROI scope needs explicit bounds")
        for label, limits in (
            ("horizontal", self.horizontal_intensity_limits),
            ("vertical", self.vertical_intensity_limits),
        ):
            if limits is None:
                continue
            if type(limits) is not tuple or len(limits) != 2:
                raise ProjectFormatError(f"{label} intensity limits need two numbers")
            low_limit = _float(limits[0], f"{label} intensity minimum")
            high_limit = _float(limits[1], f"{label} intensity maximum")
            if not low_limit < high_limit or not math.isfinite(high_limit - low_limit):
                raise ProjectFormatError(f"{label} intensity limits must have finite span")


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
    legacy_fields = {
        "column_px",
        "row_px",
        "zoom",
        "pan_x_px",
        "pan_y_px",
        "low_value",
        "high_value",
        "positive_log",
    }
    current_fields = (legacy_fields - {"positive_log"}) | {
        "contrast_mode",
        "device_pixel_ratio",
        "scale_mode",
        "show_image",
        "show_crosshair",
        "show_markers",
    }
    profile_fields = current_fields | {
        "profile_follow",
        "profile_row_width",
        "profile_column_width",
        "profile_measure",
        "profile_scope",
        "profile_roi",
        "horizontal_intensity_limits",
        "vertical_intensity_limits",
    }
    if type(value) is not dict:
        raise ProjectFormatError("detector view must be an object")
    if set(value) == legacy_fields:
        data = value
        if type(data["positive_log"]) is not bool:
            raise ProjectFormatError("legacy detector contrast flag must be Boolean")
        mode = "positive_log" if data["positive_log"] else "linear"
        return DetectorViewState(
            data["column_px"],
            data["row_px"],
            data["zoom"],
            data["pan_x_px"],
            data["pan_y_px"],
            data["low_value"],
            data["high_value"],
            mode,
        )
    data = _object(
        value, profile_fields if set(value) == profile_fields else current_fields, "detector view"
    )
    earlier = set(data) == current_fields

    def optional_tuple(field: str, length: int) -> tuple[Any, ...] | None:
        supplied = data[field]
        if supplied is None:
            return None
        if type(supplied) is not list or len(supplied) != length:
            raise ProjectFormatError(f"{field} needs a {length}-element array")
        return tuple(supplied)

    return DetectorViewState(
        data["column_px"],
        data["row_px"],
        data["zoom"],
        data["pan_x_px"],
        data["pan_y_px"],
        data["low_value"],
        data["high_value"],
        data["contrast_mode"],
        data["device_pixel_ratio"],
        data["scale_mode"],
        data["show_image"],
        data["show_crosshair"],
        data["show_markers"],
        True if earlier else data["profile_follow"],
        1 if earlier else data["profile_row_width"],
        1 if earlier else data["profile_column_width"],
        "sum" if earlier else data["profile_measure"],
        "band" if earlier else data["profile_scope"],
        None if earlier else optional_tuple("profile_roi", 4),
        None if earlier else optional_tuple("horizontal_intensity_limits", 2),
        None if earlier else optional_tuple("vertical_intensity_limits", 2),
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
                "contrast_mode": detector.contrast_mode,
                "device_pixel_ratio": detector.device_pixel_ratio,
                "scale_mode": detector.scale_mode,
                "show_image": detector.show_image,
                "show_crosshair": detector.show_crosshair,
                "show_markers": detector.show_markers,
                "profile_follow": detector.profile_follow,
                "profile_row_width": detector.profile_row_width,
                "profile_column_width": detector.profile_column_width,
                "profile_measure": detector.profile_measure,
                "profile_scope": detector.profile_scope,
                "profile_roi": (
                    None if detector.profile_roi is None else list(detector.profile_roi)
                ),
                "horizontal_intensity_limits": (
                    None
                    if detector.horizontal_intensity_limits is None
                    else list(detector.horizontal_intensity_limits)
                ),
                "vertical_intensity_limits": (
                    None
                    if detector.vertical_intensity_limits is None
                    else list(detector.vertical_intensity_limits)
                ),
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
