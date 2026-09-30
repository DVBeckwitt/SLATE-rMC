"""Qt-free project identity, strict JSON state and decoded-source references."""

import hashlib
import json
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import UUID, uuid4

from comparison_state import LineDefinition, PinIdentity
from mask_state import NativeMask, mask_document, mask_from_document
from numeric_fields import validate_proposal

PROJECT_SCHEMA_VERSION = 7
SOURCE_HASH_KIND = "sha256:decoded-osc-header-and-payload"
MAX_PROJECT_BYTES = 1024 * 1024
MAX_ACQUISITIONS = 128
MAX_NAME_LENGTH = 256
MAX_SOURCE_PATH_LENGTH = 4096
MAX_NUMERIC_BASELINE_BYTES = 64 * 1024
MAX_NUMERIC_PROPOSALS = 32
METADATA_FIELDS = (
    "role",
    "specimen",
    "mount",
    "incidence_rad",
    "exposure_s",
    "detector_setup",
    "material_id",
    "cif_path",
    "configuration_path",
    "configuration_cif_path",
    "calibrant_id",
    "dark_acquisition_id",
    "mask_acquisition_id",
)
HBN_CU_K_ALPHA_PRESET = "hbn_cu_ka_5rings"
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
class AcquisitionMetadata:
    role: str | None = None
    specimen: str | None = None
    mount: str | None = None
    incidence_rad: float | None = None
    exposure_s: float | None = None
    native_shape: tuple[int, int] | None = None
    detector_setup: str | None = None
    material_id: str | None = None
    cif_path: Path | None = None
    cif_sha256: str | None = None
    configuration_path: Path | None = None
    configuration_sha256: str | None = None
    configuration_cif_path: Path | None = None
    configuration_cif_sha256: str | None = None
    calibrant_id: str | None = None
    dark_acquisition_id: UUID | None = None
    mask_acquisition_id: UUID | None = None
    provenance: tuple[tuple[str, str], ...] = ()
    proposals: tuple[tuple[str, str, str], ...] = ()
    revision: int = 0

    def __post_init__(self) -> None:
        if self.role is not None and self.role not in ("sample", "calibrant", "dark", "mask"):
            raise ProjectFormatError("unsupported acquisition role")
        for name in ("specimen", "mount", "detector_setup", "material_id", "calibrant_id"):
            value = getattr(self, name)
            if value is not None:
                _name(value, name)
        if self.calibrant_id is not None and self.calibrant_id != HBN_CU_K_ALPHA_PRESET:
            raise ProjectFormatError("unsupported calibrant preset")
        if self.incidence_rad is not None:
            _float(self.incidence_rad, "incidence_rad")
        if self.exposure_s is not None and _float(self.exposure_s, "exposure_s") <= 0:
            raise ProjectFormatError("exposure must be positive seconds")
        if self.native_shape is not None and (
            type(self.native_shape) is not tuple
            or len(self.native_shape) != 2
            or any(type(axis) is not int or not 1 <= axis <= 16_384 for axis in self.native_shape)
            or self.native_shape[0] * self.native_shape[1] > 12_000_000
        ):
            raise ProjectFormatError("invalid detector-native shape")
        for label in ("cif", "configuration", "configuration_cif"):
            path = getattr(self, f"{label}_path")
            digest = getattr(self, f"{label}_sha256")
            if (path is None) != (digest is None):
                raise ProjectFormatError(f"{label} reference needs path and SHA-256")
            if path is not None:
                if (
                    not isinstance(path, Path)
                    or not str(path)
                    or len(str(path)) > MAX_SOURCE_PATH_LENGTH
                ):
                    raise ProjectFormatError(f"invalid {label} path")
                _digest(digest)
        if self.configuration_cif_path is not None and self.configuration_path is None:
            raise ProjectFormatError("dependent CIF identity needs a configuration reference")
        if type(self.revision) is not int or self.revision < 0:
            raise ProjectFormatError("metadata revision must be nonnegative")
        if type(self.provenance) is not tuple or any(
            type(entry) is not tuple
            or len(entry) != 2
            or any(type(value) is not str for value in entry)
            for entry in self.provenance
        ):
            raise ProjectFormatError("metadata provenance must be immutable string pairs")
        if type(self.proposals) is not tuple or any(
            type(entry) is not tuple
            or len(entry) != 3
            or any(type(value) is not str for value in entry)
            for entry in self.proposals
        ):
            raise ProjectFormatError("metadata proposals must be immutable string triples")
        sources = [field for field, _ in self.provenance]
        if len(sources) != len(set(sources)):
            raise ProjectFormatError("duplicate metadata provenance")
        for field_name, origin in self.provenance:
            if field_name not in METADATA_FIELDS or getattr(self, field_name) is None:
                raise ProjectFormatError("metadata provenance references an unset field")
            _name(origin, "metadata provenance")
        for field_name, value, origin in self.proposals:
            if field_name != "incidence_rad" or type(value) is not str or not value:
                raise ProjectFormatError("invalid metadata proposal")
            if len(value) > MAX_NAME_LENGTH:
                raise ProjectFormatError("metadata proposal is too long")
            try:
                proposed_angle = float(value)
            except ValueError as exc:
                raise ProjectFormatError("invalid incidence proposal") from exc
            if not math.isfinite(proposed_angle):
                raise ProjectFormatError("incidence proposal must be finite")
            _name(origin, "proposal provenance")
        if len(self.proposals) > len(METADATA_FIELDS):
            raise ProjectFormatError("too many metadata proposals")
        if len({field for field, _, _ in self.proposals}) != len(self.proposals):
            raise ProjectFormatError("duplicate metadata proposals")


@dataclass(frozen=True, slots=True)
class Acquisition:
    acquisition_id: UUID
    name: str
    source_path: Path
    source_sha256: str
    metadata: AcquisitionMetadata = field(default_factory=AcquisitionMetadata)
    mask: NativeMask | None = None

    def __post_init__(self) -> None:
        if self.mask is not None and (
            not isinstance(self.mask, NativeMask) or self.mask.source_sha256 != self.source_sha256
        ):
            raise ProjectFormatError("Mask must bind the acquisition's decoded source identity")

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
        by_id = {item.acquisition_id: item for item in self.acquisitions}
        for item in self.acquisitions:
            for field_name, role in (
                ("dark_acquisition_id", "dark"),
                ("mask_acquisition_id", "mask"),
            ):
                linked = getattr(item.metadata, field_name)
                if linked is not None and (
                    linked == item.acquisition_id
                    or linked not in by_id
                    or by_id[linked].metadata.role != role
                ):
                    raise ValueError(f"{field_name} must name another {role} acquisition")

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

    def update_metadata(self, acquisition_id: UUID, **changes: Any) -> "Project":
        if not changes or not set(changes) <= set(METADATA_FIELDS) | {
            "native_shape",
            "provenance",
            "proposals",
            "cif_sha256",
            "configuration_sha256",
            "configuration_cif_sha256",
        }:
            raise ValueError("unsupported metadata change")
        if acquisition_id not in {item.acquisition_id for item in self.acquisitions}:
            raise KeyError(acquisition_id)
        if "configuration_path" in changes and not any(
            field_name in changes
            for field_name in ("configuration_cif_path", "configuration_cif_sha256")
        ):
            changes["configuration_cif_path"] = None
            changes["configuration_cif_sha256"] = None
            existing = next(
                item for item in self.acquisitions if item.acquisition_id == acquisition_id
            )
            excluded = (
                {"configuration_cif_path"}
                if "provenance" in changes
                else {"configuration_path", "configuration_cif_path"}
            )
            changes["provenance"] = tuple(
                (field_name, origin)
                for field_name, origin in changes.get("provenance", existing.metadata.provenance)
                if field_name not in excluded
            )
        for label in ("cif", "configuration", "configuration_cif"):
            path_field, hash_field = f"{label}_path", f"{label}_sha256"
            if (path_field in changes) != (hash_field in changes):
                raise ValueError(f"{label} path and SHA-256 must change together")
        return replace(
            self,
            acquisitions=tuple(
                replace(
                    item,
                    metadata=replace(item.metadata, **changes, revision=item.metadata.revision + 1),
                )
                if item.acquisition_id == acquisition_id
                else item
                for item in self.acquisitions
            ),
        )

    def remove_acquisition(self, acquisition_id: UUID) -> "Project":
        if acquisition_id not in {item.acquisition_id for item in self.acquisitions}:
            raise KeyError(acquisition_id)
        remaining = []
        for item in self.acquisitions:
            if item.acquisition_id == acquisition_id:
                continue
            changes = {
                field_name: None
                for field_name in ("dark_acquisition_id", "mask_acquisition_id")
                if getattr(item.metadata, field_name) == acquisition_id
            }
            metadata = item.metadata
            if changes:
                sources = tuple(
                    (name, source) for name, source in metadata.provenance if name not in changes
                )
                metadata = replace(
                    metadata, **changes, provenance=sources, revision=metadata.revision + 1
                )
            remaining.append(replace(item, metadata=metadata))
        return replace(self, acquisitions=tuple(remaining))


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
    show_mask: bool = True

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
            for value in (self.show_image, self.show_crosshair, self.show_markers, self.show_mask)
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
class SceneViewState:
    yaw_rad: float
    pitch_rad: float
    zoom: float
    target_lab_m: tuple[float, float, float]
    visible: bool

    def __post_init__(self) -> None:
        for name in ("yaw_rad", "pitch_rad", "zoom"):
            _float(getattr(self, name), name)
        if not -math.pi / 2 <= self.pitch_rad <= math.pi / 2 or not 0.2 <= self.zoom <= 12.0:
            raise ProjectFormatError("scene camera is outside its supported range")
        if type(self.target_lab_m) is not tuple or len(self.target_lab_m) != 3:
            raise ProjectFormatError("scene target must be one lab-frame metre point")
        for value in self.target_lab_m:
            _float(value, "scene target")
        if type(self.visible) is not bool:
            raise ProjectFormatError("scene visibility must be Boolean")


@dataclass(frozen=True, slots=True)
class ComparisonState:
    acquisition_ids: tuple[UUID | None, UUID | None] = (None, None)
    views: tuple[DetectorViewState | None, DetectorViewState | None] = (None, None)
    lines: tuple[LineDefinition | None, LineDefinition | None] = (None, None)
    pin: PinIdentity | None = None
    pin_axis: str = "horizontal"
    active_slot: int = 0
    linked_navigation: bool = False
    shared_limits: tuple[float, float, str] | None = None
    visible: bool = False

    def __post_init__(self) -> None:
        for name, kind in (
            ("acquisition_ids", UUID),
            ("views", DetectorViewState),
            ("lines", LineDefinition),
        ):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or len(values) != 2
                or any(v is not None and not isinstance(v, kind) for v in values)
            ):
                raise ProjectFormatError(f"Comparison {name} needs two bounded entries")
        if self.pin is not None and not isinstance(self.pin, PinIdentity):
            raise ProjectFormatError("Comparison reference identity is invalid")
        if (
            self.pin_axis not in ("horizontal", "vertical")
            or type(self.active_slot) is not int
            or self.active_slot not in (0, 1)
        ):
            raise ProjectFormatError("Comparison axis or active image is invalid")
        if type(self.linked_navigation) is not bool or type(self.visible) is not bool:
            raise ProjectFormatError("Comparison visibility and navigation flags must be Boolean")
        if self.shared_limits is not None:
            limits = self.shared_limits
            if (
                type(limits) is not tuple
                or len(limits) != 3
                or limits[2] not in ("linear", "signed", "positive_log")
            ):
                raise ProjectFormatError("Shared comparison limits need low, high and display mode")
            validate_display_limits(limits[0], limits[1], limits[2])


@dataclass(frozen=True, slots=True)
class ProjectViewState:
    selected_acquisition_id: UUID | None = None
    workspace: str = "fit_experiments"
    detector: DetectorViewState | None = None
    scene: SceneViewState | None = None
    comparison: ComparisonState | None = None


@dataclass(frozen=True, slots=True)
class NumericDraft:
    acquisition_id: UUID
    source_sha256: str
    configuration_path: Path
    configuration_sha256: str
    cif_path: Path
    cif_sha256: str
    baseline_yaml: str
    proposed: tuple[tuple[str, float, str, str], ...] = ()
    revision: int = 0

    def __post_init__(self) -> None:
        for digest in (self.source_sha256, self.configuration_sha256, self.cif_sha256):
            _digest(digest)
        for path in (self.configuration_path, self.cif_path):
            if (
                not isinstance(path, Path)
                or not str(path)
                or len(str(path)) > MAX_SOURCE_PATH_LENGTH
            ):
                raise ProjectFormatError("numeric draft reference path is invalid")
        if type(self.baseline_yaml) is not str:
            raise ProjectFormatError("numeric baseline must be UTF-8 text")
        try:
            encoded = self.baseline_yaml.encode("utf-8")
        except UnicodeError as exc:
            raise ProjectFormatError("numeric baseline must be UTF-8 text") from exc
        if not encoded or len(encoded) > MAX_NUMERIC_BASELINE_BYTES:
            raise ProjectFormatError("numeric baseline exceeds its 64 KiB limit")
        if hashlib.sha256(encoded).hexdigest() != self.configuration_sha256:
            raise ProjectFormatError("numeric baseline hash differs from configuration identity")
        if type(self.revision) is not int or self.revision < 0:
            raise ProjectFormatError("numeric draft revision must be nonnegative")
        if type(self.proposed) is not tuple or len(self.proposed) > MAX_NUMERIC_PROPOSALS:
            raise ProjectFormatError("too many numeric draft proposals")
        names = []
        for entry in self.proposed:
            if type(entry) is not tuple or len(entry) != 4:
                raise ProjectFormatError("numeric proposal needs field, value, unit and provenance")
            name, value, unit, provenance = entry
            if type(name) is not str or not name or len(name) > 128:
                raise ProjectFormatError("numeric proposal field is invalid")
            _float(value, name)
            _name(unit, "numeric proposal unit")
            _name(provenance, "numeric proposal provenance")
            try:
                validate_proposal(name, value, unit)
            except ValueError as exc:
                raise ProjectFormatError(str(exc)) from exc
            names.append(name)
        if len(names) != len(set(names)):
            raise ProjectFormatError("duplicate numeric proposal field")


@dataclass(frozen=True, slots=True)
class ProjectDocument:
    project: Project
    view: ProjectViewState
    numeric_draft: NumericDraft | None = None

    def __post_init__(self) -> None:
        selected = self.view.selected_acquisition_id
        if self.view.comparison is not None and not isinstance(
            self.view.comparison, ComparisonState
        ):
            raise ProjectFormatError("Comparison view state is invalid")
        if selected is not None and not any(
            item.acquisition_id == selected for item in self.project.acquisitions
        ):
            raise ProjectFormatError("selected acquisition is absent from the project")
        if self.view.workspace not in ("fit_experiments", "simulator"):
            raise ProjectFormatError("unsupported workspace view")
        if self.view.detector is not None and selected is None:
            raise ProjectFormatError("detector view needs a selected acquisition")
        if self.view.scene is not None and selected is None:
            raise ProjectFormatError("scene view needs a selected acquisition")
        if self.numeric_draft is not None and not any(
            item.acquisition_id == self.numeric_draft.acquisition_id
            for item in self.project.acquisitions
        ):
            raise ProjectFormatError("numeric draft acquisition is absent from the project")


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
        value,
        profile_fields | {"show_mask"}
        if set(value) == profile_fields | {"show_mask"}
        else profile_fields
        if set(value) == profile_fields
        else current_fields,
        "detector view",
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
        data.get("show_mask", True),
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


def _resolved_reference(value: Any, document_path: Path, label: str) -> Path:
    if type(value) is not str or not value or len(value) > MAX_SOURCE_PATH_LENGTH or "\0" in value:
        raise ProjectFormatError(f"{label} has an invalid path")
    source = Path(value)
    if not source.is_absolute():
        windows = PureWindowsPath(value)
        if windows.drive or windows.root:
            raise ProjectFormatError(f"{label} path is drive-relative or rooted")
        source = document_path.parent / source
    return source.absolute()


def _metadata_document(metadata: AcquisitionMetadata, document_path: Path) -> dict[str, Any]:
    return {
        **{
            field_name: (
                str(getattr(metadata, field_name))
                if field_name.endswith("_id") and isinstance(getattr(metadata, field_name), UUID)
                else getattr(metadata, field_name)
            )
            for field_name in METADATA_FIELDS
            if field_name not in ("cif_path", "configuration_path", "configuration_cif_path")
        },
        "native_shape": None if metadata.native_shape is None else list(metadata.native_shape),
        "cif_path": None
        if metadata.cif_path is None
        else _source_reference(metadata.cif_path, document_path),
        "cif_sha256": metadata.cif_sha256,
        "configuration_path": None
        if metadata.configuration_path is None
        else _source_reference(metadata.configuration_path, document_path),
        "configuration_sha256": metadata.configuration_sha256,
        "configuration_cif_path": None
        if metadata.configuration_cif_path is None
        else _source_reference(metadata.configuration_cif_path, document_path),
        "configuration_cif_sha256": metadata.configuration_cif_sha256,
        "provenance": [list(entry) for entry in metadata.provenance],
        "proposals": [list(entry) for entry in metadata.proposals],
        "revision": metadata.revision,
    }


def _metadata_from_document(
    value: Any, document_path: Path, schema_version: int
) -> AcquisitionMetadata:
    keys = set(METADATA_FIELDS) | {
        "native_shape",
        "cif_sha256",
        "configuration_sha256",
        "provenance",
        "proposals",
        "revision",
    }
    if schema_version >= 3:
        keys |= {"configuration_cif_sha256"}
    else:
        keys -= {"configuration_cif_path"}
    data = _object(value, keys, "acquisition metadata")
    shape = data["native_shape"]
    if shape is not None and (type(shape) is not list or len(shape) != 2):
        raise ProjectFormatError("native shape needs two axes")
    provenance = data["provenance"]
    proposals = data["proposals"]
    if type(provenance) is not list or type(proposals) is not list:
        raise ProjectFormatError("metadata provenance and proposals must be arrays")
    if any(type(entry) is not list or len(entry) != 2 for entry in provenance):
        raise ProjectFormatError("invalid metadata provenance entry")
    if any(type(entry) is not list or len(entry) != 3 for entry in proposals):
        raise ProjectFormatError("invalid metadata proposal entry")
    return AcquisitionMetadata(
        role=data["role"],
        specimen=data["specimen"],
        mount=data["mount"],
        incidence_rad=data["incidence_rad"],
        exposure_s=data["exposure_s"],
        native_shape=None if shape is None else tuple(shape),
        detector_setup=data["detector_setup"],
        material_id=data["material_id"],
        cif_path=None
        if data["cif_path"] is None
        else _resolved_reference(data["cif_path"], document_path, "CIF"),
        cif_sha256=None if data["cif_sha256"] is None else _digest(data["cif_sha256"]),
        configuration_path=None
        if data["configuration_path"] is None
        else _resolved_reference(data["configuration_path"], document_path, "configuration"),
        configuration_sha256=None
        if data["configuration_sha256"] is None
        else _digest(data["configuration_sha256"]),
        configuration_cif_path=None
        if schema_version < 3 or data["configuration_cif_path"] is None
        else _resolved_reference(
            data["configuration_cif_path"], document_path, "configuration CIF"
        ),
        configuration_cif_sha256=None
        if schema_version < 3 or data["configuration_cif_sha256"] is None
        else _digest(data["configuration_cif_sha256"]),
        calibrant_id=data["calibrant_id"],
        dark_acquisition_id=None
        if data["dark_acquisition_id"] is None
        else _uuid(data["dark_acquisition_id"], "dark acquisition ID"),
        mask_acquisition_id=None
        if data["mask_acquisition_id"] is None
        else _uuid(data["mask_acquisition_id"], "mask acquisition ID"),
        provenance=tuple(tuple(entry) for entry in provenance),
        proposals=tuple(tuple(entry) for entry in proposals),
        revision=data["revision"],
    )


def _detector_document(detector: DetectorViewState | None) -> dict | None:
    if detector is None:
        return None
    return {
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
        "show_mask": detector.show_mask,
        "profile_follow": detector.profile_follow,
        "profile_row_width": detector.profile_row_width,
        "profile_column_width": detector.profile_column_width,
        "profile_measure": detector.profile_measure,
        "profile_scope": detector.profile_scope,
        "profile_roi": (None if detector.profile_roi is None else list(detector.profile_roi)),
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


def _comparison_document(state: ComparisonState | None) -> dict | None:
    if state is None:
        return None
    pin = state.pin
    return {
        "acquisition_ids": [str(v) if v is not None else None for v in state.acquisition_ids],
        "views": [_detector_document(v) for v in state.views],
        "lines": [
            None
            if v is None
            else {
                "start_column_row": list(v.start_column_row),
                "end_column_row": list(v.end_column_row),
                "spacing_px": v.spacing_px,
                "revision": v.revision,
            }
            for v in state.lines
        ],
        "pin": None
        if pin is None
        else {
            "acquisition_id": str(pin.acquisition_id),
            "source_sha256": pin.source_sha256,
            "native_shape_rc": list(pin.native_shape_rc),
            "mask_revision": pin.mask_revision,
            "query": [
                *pin.query[:5],
                None if pin.query[5] is None else list(pin.query[5]),
                pin.query[6],
            ],
        },
        "pin_axis": state.pin_axis,
        "active_slot": state.active_slot,
        "linked_navigation": state.linked_navigation,
        "shared_limits": None if state.shared_limits is None else list(state.shared_limits),
        "visible": state.visible,
    }


def _comparison_view(value: Any) -> ComparisonState | None:
    if value is None:
        return None
    data = _object(
        value,
        {
            "acquisition_ids",
            "views",
            "lines",
            "pin",
            "pin_axis",
            "active_slot",
            "linked_navigation",
            "shared_limits",
            "visible",
        },
        "comparison",
    )
    for name in ("acquisition_ids", "views", "lines"):
        if type(data[name]) is not list or len(data[name]) != 2:
            raise ProjectFormatError(f"Comparison {name} needs two entries")
    lines = []
    for value in data["lines"]:
        if value is None:
            lines.append(None)
            continue
        row = _object(
            value, {"start_column_row", "end_column_row", "spacing_px", "revision"}, "line"
        )
        if any(type(row[n]) is not list for n in ("start_column_row", "end_column_row")):
            raise ProjectFormatError("Line endpoints must be arrays")
        lines.append(
            LineDefinition(
                tuple(row["start_column_row"]),
                tuple(row["end_column_row"]),
                row["spacing_px"],
                row["revision"],
            )
        )
    pin = None
    if data["pin"] is not None:
        row = _object(
            data["pin"],
            {"acquisition_id", "source_sha256", "native_shape_rc", "mask_revision", "query"},
            "pinned reference",
        )
        query = row["query"]
        if (
            type(query) is not list
            or len(query) != 7
            or type(row["native_shape_rc"]) is not list
            or (query[5] is not None and type(query[5]) is not list)
        ):
            raise ProjectFormatError("Pinned profile arrays are invalid")
        pin = PinIdentity(
            _uuid(row["acquisition_id"], "pinned acquisition"),
            _digest(row["source_sha256"]),
            tuple(row["native_shape_rc"]),
            row["mask_revision"],
            (*query[:5], None if query[5] is None else tuple(query[5]), query[6]),
        )
    limits = data["shared_limits"]
    if limits is not None and type(limits) is not list:
        raise ProjectFormatError("Shared comparison limits must be an array")
    return ComparisonState(
        tuple(
            None if v is None else _uuid(v, "comparison acquisition")
            for v in data["acquisition_ids"]
        ),
        tuple(_detector_view(v) for v in data["views"]),
        tuple(lines),
        pin,
        data["pin_axis"],
        data["active_slot"],
        data["linked_navigation"],
        None if limits is None else tuple(limits),
        data["visible"],
    )


def project_to_document(
    state: ProjectDocument, document_path: Path, *, reserved_bytes: int = 0
) -> dict[str, Any]:
    """Encode only the currently supported numeric project and view state."""
    if type(reserved_bytes) is not int or reserved_bytes < 0:
        raise ValueError("reserved project bytes must be nonnegative")
    if len(state.project.acquisitions) > MAX_ACQUISITIONS:
        raise ProjectFormatError(f"project exceeds {MAX_ACQUISITIONS} acquisitions")
    acquisitions = [
        {
            "id": str(item.acquisition_id),
            "name": _name(item.name, "acquisition name"),
            "source_path": _source_reference(item.source_path, document_path),
            "source_sha256": _digest(item.source_sha256),
            "metadata": _metadata_document(item.metadata, document_path),
            "mask": mask_document(item.mask),
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
        "comparison": _comparison_document(state.view.comparison),
        "detector": _detector_document(detector),
        "scene": (
            None
            if state.view.scene is None
            else {
                "yaw_rad": state.view.scene.yaw_rad,
                "pitch_rad": state.view.scene.pitch_rad,
                "zoom": state.view.scene.zoom,
                "target_lab_m": list(state.view.scene.target_lab_m),
                "visible": state.view.scene.visible,
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
        "numeric_draft": (
            None
            if state.numeric_draft is None
            else {
                "acquisition_id": str(state.numeric_draft.acquisition_id),
                "source_sha256": state.numeric_draft.source_sha256,
                "configuration_path": _source_reference(
                    state.numeric_draft.configuration_path, document_path
                ),
                "configuration_sha256": state.numeric_draft.configuration_sha256,
                "cif_path": _source_reference(state.numeric_draft.cif_path, document_path),
                "cif_sha256": state.numeric_draft.cif_sha256,
                "baseline_yaml": state.numeric_draft.baseline_yaml,
                "proposed": [list(entry) for entry in state.numeric_draft.proposed],
                "revision": state.numeric_draft.revision,
            }
        ),
    }
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if len(encoded) + reserved_bytes > MAX_PROJECT_BYTES:
        raise ProjectFormatError(f"project document exceeds {MAX_PROJECT_BYTES} bytes")
    return document


def project_from_document(value: Any, document_path: Path) -> ProjectDocument:
    """Validate a complete project before replacing the current shell state."""
    if type(value) is not dict:
        raise ProjectFormatError("document must be an object")
    draft_fields = {"numeric_draft"} if value.get("schema_version", 0) >= 4 else set()
    top = _object(
        value, {"schema_version", "source_hash_kind", "project", "view"} | draft_fields, "document"
    )
    if type(top["schema_version"]) is not int or top["schema_version"] not in (
        1,
        2,
        3,
        4,
        5,
        6,
        PROJECT_SCHEMA_VERSION,
    ):
        raise ProjectFormatError(f"unsupported project schema version {top['schema_version']!r}")
    if top["source_hash_kind"] != SOURCE_HASH_KIND:
        raise ProjectFormatError("unsupported source hash kind; expected decoded OSC bytes")
    data = _object(top["project"], {"id", "name", "acquisitions"}, "project")
    items = data["acquisitions"]
    if type(items) is not list or len(items) > MAX_ACQUISITIONS:
        raise ProjectFormatError(f"acquisitions must be a list of at most {MAX_ACQUISITIONS}")
    acquisitions = []
    for index, item in enumerate(items):
        fields = {"id", "name", "source_path", "source_sha256"}
        row = _object(
            item,
            (fields if top["schema_version"] == 1 else fields | {"metadata"})
            | ({"mask"} if top["schema_version"] >= 6 else set()),
            f"acquisition {index}",
        )
        source = _resolved_reference(row["source_path"], document_path, f"acquisition {index}")
        acquisitions.append(
            Acquisition(
                _uuid(row["id"], f"acquisition {index} ID"),
                _name(row["name"], f"acquisition {index} name"),
                source,
                _digest(row["source_sha256"]),
                AcquisitionMetadata()
                if top["schema_version"] == 1
                else _metadata_from_document(row["metadata"], document_path, top["schema_version"]),
                mask_from_document(row["mask"]) if top["schema_version"] >= 6 else None,
            )
        )
    try:
        project = Project(
            _uuid(data["id"], "project ID"),
            _name(data["name"], "project name"),
            tuple(acquisitions),
            PROJECT_SCHEMA_VERSION,
        )
    except ValueError as exc:
        raise ProjectFormatError(str(exc)) from exc
    view_fields = {"selected_acquisition_id", "workspace", "detector"}
    view_data = _object(
        top["view"],
        view_fields
        | ({"scene"} if top["schema_version"] >= 5 else set())
        | ({"comparison"} if top["schema_version"] >= 7 else set()),
        "view",
    )
    selected = view_data["selected_acquisition_id"]
    selected_id = None if selected is None else _uuid(selected, "selected acquisition ID")
    scene = None
    if view_data.get("scene") is not None:
        scene_data = _object(
            view_data["scene"],
            {"yaw_rad", "pitch_rad", "zoom", "target_lab_m", "visible"},
            "scene view",
        )
        target = scene_data["target_lab_m"]
        if type(target) is not list or len(target) != 3:
            raise ProjectFormatError("scene target must have three metre coordinates")
        scene = SceneViewState(
            scene_data["yaw_rad"],
            scene_data["pitch_rad"],
            scene_data["zoom"],
            tuple(target),
            scene_data["visible"],
        )
    numeric = top.get("numeric_draft")
    draft = None
    if numeric is not None:
        supplied = _object(
            numeric,
            {
                "acquisition_id",
                "source_sha256",
                "configuration_path",
                "configuration_sha256",
                "cif_path",
                "cif_sha256",
                "baseline_yaml",
                "proposed",
                "revision",
            },
            "numeric draft",
        )
        proposals = supplied["proposed"]
        if type(proposals) is not list or any(type(entry) is not list for entry in proposals):
            raise ProjectFormatError("numeric proposals must be arrays")
        draft = NumericDraft(
            _uuid(supplied["acquisition_id"], "numeric acquisition ID"),
            _digest(supplied["source_sha256"]),
            _resolved_reference(supplied["configuration_path"], document_path, "configuration"),
            _digest(supplied["configuration_sha256"]),
            _resolved_reference(supplied["cif_path"], document_path, "configuration CIF"),
            _digest(supplied["cif_sha256"]),
            supplied["baseline_yaml"],
            tuple(tuple(entry) for entry in proposals),
            supplied["revision"],
        )
    return ProjectDocument(
        project,
        ProjectViewState(
            selected_id,
            view_data["workspace"],
            _detector_view(view_data["detector"]),
            scene,
            _comparison_view(view_data.get("comparison")),
        ),
        draft,
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
