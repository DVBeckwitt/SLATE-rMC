"""Native detector presentation primitives for the desktop application.

The texture is display only. Profiles always reduce the owned native data plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from comparison_state import LineDefinition
from mask_state import MAX_POINTS, REASONS, MaskGesture
from numpy.typing import NDArray
from project_state import (
    FLOAT32_DISPLAY_MAX,
    DetectorViewState,
    linear_display_limits,
    validate_display_limits,
)
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPolygonF,
    QShortcut,
    QTransform,
)
from PySide6.QtOpenGL import (
    QOpenGLPixelTransferOptions,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QWidget,
)
from shiboken6 import VoidPtr


@dataclass(frozen=True, slots=True)
class BandProfiles:
    horizontal: NDArray[np.int64] | NDArray[np.float64]
    vertical: NDArray[np.int64] | NDArray[np.float64]
    horizontal_support: NDArray[np.int64]
    vertical_support: NDArray[np.int64]
    row_bounds: tuple[int, int]
    column_bounds: tuple[int, int]


def _mean_per_valid(
    total: NDArray[np.int64] | NDArray[np.float64], support: NDArray[np.int64]
) -> NDArray[np.float64]:
    mean = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, support, out=mean, where=support > 0)
    mean.setflags(write=False)
    return mean


def _effective_band_bounds(
    rows: int, columns: int, column_px: int, row_px: int, row_width: int, column_width: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    row_start = row_px - (row_width - 1) // 2
    column_start = column_px - (column_width - 1) // 2
    return (
        (max(0, row_start), min(rows, row_start + row_width)),
        (max(0, column_start), min(columns, column_start + column_width)),
    )


def _segment_crossing(a: float, b: float, bound: float) -> float:
    """Position where a finite segment crosses a bound without overflow."""

    difference = b - a
    distance = bound - a
    if math.isfinite(difference) and math.isfinite(distance):
        return distance / difference
    scale = max(abs(a), abs(b), abs(bound))
    return ((bound / scale) - (a / scale)) / ((b / scale) - (a / scale))


def exact_band_profiles(
    native_image: NDArray[np.generic],
    *,
    column_px: int,
    row_px: int,
    row_width: int = 1,
    column_width: int = 1,
    mask: NDArray[np.bool] | None = None,
    scope: str = "band",
    roi_column_row_bounds: tuple[int, int, int, int] | None = None,
    measure: str = "sum",
) -> BandProfiles:
    """Reduce native values over bands, the full panel, or a half-open native ROI.

    ROI bounds are ``(column_start, column_stop, row_start, row_stop)``.
    Both returned vectors retain full native column/row alignment.
    """

    image = np.asarray(native_image)
    if image.ndim != 2 or image.dtype.kind not in "iuf":
        raise ValueError("native_image must be a two-dimensional real numeric plane")
    if image.dtype.kind == "f" and image.dtype.itemsize < 8:
        raise ValueError("exact profiles require the original float64 quantitative plane")
    rows, columns = image.shape
    if not rows or not columns:
        raise ValueError("native_image must be nonempty")
    if not (0 <= column_px < columns and 0 <= row_px < rows):
        raise ValueError("crosshair is outside the native detector")
    if row_width < 1 or column_width < 1:
        raise ValueError("band widths must be positive")
    if measure not in ("sum", "mean"):
        raise ValueError("profile measure must be sum or mean per valid pixel")
    if scope == "band":
        (r0, r1), (c0, c1) = _effective_band_bounds(
            rows, columns, column_px, row_px, row_width, column_width
        )
    elif scope == "full":
        r0, r1, c0, c1 = 0, rows, 0, columns
    elif scope == "roi":
        if (
            roi_column_row_bounds is None
            or len(roi_column_row_bounds) != 4
            or any(type(value) is not int for value in roi_column_row_bounds)
        ):
            raise ValueError("ROI needs four integer native column/row bounds")
        c0, c1, r0, r1 = roi_column_row_bounds
        if not (0 <= c0 < c1 <= columns and 0 <= r0 < r1 <= rows):
            raise ValueError("ROI bounds are outside the native detector")
    else:
        raise ValueError("profile scope must be band, full, or roi")
    if mask is not None and (mask.dtype != np.bool_ or mask.shape != image.shape):
        raise ValueError("mask must be a Boolean native-detector plane")
    integer = image.dtype.kind in "iu"
    if integer:
        limits = np.iinfo(image.dtype)
        maximum_band = max(r1 - r0, c1 - c0)
        if max(abs(int(limits.min)), int(limits.max)) * maximum_band > np.iinfo(np.int64).max:
            raise ValueError("integer band may overflow int64 exact accumulation")
    accumulator = np.int64 if integer else np.float64

    def reduce_band(values: NDArray[np.generic], included: NDArray[np.bool] | None, axis: int):
        if integer and included is None:
            total = np.sum(values, axis=axis, dtype=np.int64)
            support = np.full(total.shape, values.shape[axis], dtype=np.int64)
            if measure == "mean":
                return _mean_per_valid(total, support), support
            return total, support
        valid = np.isfinite(values)
        if included is not None:
            valid &= included
        safe = np.where(valid, values, 0)
        total = np.sum(safe, axis=axis, dtype=accumulator)
        support = np.sum(valid, axis=axis, dtype=np.int64)
        if measure == "mean":
            return _mean_per_valid(total, support), support
        return total, support

    if scope == "roi":
        values = image[r0:r1, c0:c1]
        included = None if mask is None else mask[r0:r1, c0:c1]
        h_values, h_support = reduce_band(values, included, 0)
        v_values, v_support = reduce_band(values, included, 1)
        empty = np.nan if measure == "mean" else 0
        horizontal = np.full(columns, empty, dtype=np.float64 if measure == "mean" else accumulator)
        vertical = np.full(rows, empty, dtype=np.float64 if measure == "mean" else accumulator)
        horizontal_support = np.zeros(columns, dtype=np.int64)
        vertical_support = np.zeros(rows, dtype=np.int64)
        horizontal[c0:c1], horizontal_support[c0:c1] = h_values, h_support
        vertical[r0:r1], vertical_support[r0:r1] = v_values, v_support
    elif (r0, r1, c0, c1) == (0, rows, 0, columns) and (mask is not None or not integer):
        valid = np.isfinite(image)
        if mask is not None:
            valid &= mask
        safe = np.where(valid, image, 0)
        horizontal = np.sum(safe, axis=0, dtype=accumulator)
        vertical = np.sum(safe, axis=1, dtype=accumulator)
        horizontal_support = np.sum(valid, axis=0, dtype=np.int64)
        vertical_support = np.sum(valid, axis=1, dtype=np.int64)
        if measure == "mean":
            horizontal = _mean_per_valid(horizontal, horizontal_support)
            vertical = _mean_per_valid(vertical, vertical_support)
    else:
        horizontal_values = image[r0:r1]
        vertical_values = image[:, c0:c1]
        horizontal_mask = None if mask is None else mask[r0:r1]
        vertical_mask = None if mask is None else mask[:, c0:c1]
        horizontal, horizontal_support = reduce_band(horizontal_values, horizontal_mask, 0)
        vertical, vertical_support = reduce_band(vertical_values, vertical_mask, 1)
    for array in (horizontal, vertical, horizontal_support, vertical_support):
        array.setflags(write=False)
    return BandProfiles(
        horizontal, vertical, horizontal_support, vertical_support, (r0, r1), (c0, c1)
    )


_VERTEX = """#version 330 core
out vec2 uv;
uniform vec4 rect;
uniform int plane;
uniform float yaw;
uniform float pitch;
const vec2 corners[6] = vec2[6](vec2(0,0),vec2(1,0),vec2(0,1),vec2(0,1),vec2(1,0),vec2(1,1));
void main() {
    uv = corners[gl_VertexID];
    vec2 p = rect.xy + uv * rect.zw;
    if (plane == 1) {
        vec2 q = (uv - 0.5) * vec2(1.55, 1.55);
        float y = q.y * cos(pitch);
        float z = q.y * sin(pitch);
        p = vec2(0.5, 0.5) + 0.45 * vec2(q.x * cos(yaw) + z * sin(yaw), y);
    }
    gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
}
"""
DETECTOR_FRAGMENT_SHADER = """#version 330 core
in vec2 uv;
out vec4 color;
uniform sampler2D detector;
uniform float low_value;
uniform float high_value;
uniform int contrast_mode;
uniform sampler2D exclusion;
uniform int show_exclusion;
void detector_color() {
    float value = texture(detector, uv).r;
    if (isnan(value) || isinf(value) || (contrast_mode == 2 && value <= 0.0)) {
        float check = mod(floor(gl_FragCoord.x / 6.0) + floor(gl_FragCoord.y / 6.0), 2.0);
        color = vec4(vec3(0.10 + 0.06 * check), 1.0);
        return;
    }
    if (contrast_mode == 1) {
        float negative = clamp(value / low_value, 0.0, 1.0);
        float positive = clamp(value / high_value, 0.0, 1.0);
        vec3 neutral = vec3(0.17, 0.19, 0.22);
        color = vec4(neutral + negative * (vec3(0.20, 0.64, 0.92) - neutral)
            + positive * (vec3(1.0, 0.65, 0.19) - neutral), 1.0);
        return;
    }
    float magnitude = max(abs(low_value), abs(high_value));
    float multiplier = magnitude > 1e18 ? 5.421010862427522e-20
        : (magnitude < 1e-18 ? 1.8446744073709552e19 : 1.0);
    float level = contrast_mode == 2
        ? (log(value) - log(low_value)) / max(log(high_value) - log(low_value), 1e-12)
        : (value * multiplier - low_value * multiplier)
            / (high_value * multiplier - low_value * multiplier);
    level = clamp(level, 0.0, 1.0);
    color = vec4(level, level * 0.82 + 0.08, 0.12 + level * 0.68, 1.0);
}
void main() {
    detector_color();
    if (show_exclusion == 1) {
        int reason = int(round(texture(exclusion, uv).r * 255.0));
        if (reason > 0) {
            vec3 tint = reason == 1 ? vec3(1.0,0.25,0.35)
                : reason == 2 ? vec3(0.85,0.35,1.0)
                : reason == 3 ? vec3(0.2,0.75,1.0) : vec3(1.0,0.8,0.1);
            color.rgb = mix(color.rgb, tint, 0.55);
        }
    }
}
"""
_GL_MAX_TEXTURE_SIZE = 0x0D33


class DetectorTextureView(QOpenGLWidget):
    """One persistent detector texture with native coordinate overlays."""

    painted = Signal(int, float)
    crosshair_changed = Signal()
    view_state_changed = Signal()
    pin_requested = Signal()
    cursor_changed = Signal(object)
    overlays_changed = Signal(int)
    band_edge_dragged = Signal(str, int)
    roi_selected = Signal(object)
    mask_gesture_ready = Signal(object)
    mask_mode_changed = Signal(str)
    mask_input_error = Signal(str)
    line_endpoints_ready = Signal(object)
    line_mode_changed = Signal(bool)

    def __init__(self, *, plane: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plane = plane
        self.image: NDArray[np.generic] | None = None
        self._display: NDArray[np.float32] | None = None
        self.column_axis_label = "column_px"
        self.row_axis_label = "row_px"
        self._texture: QOpenGLTexture | None = None
        self._mask_texture: QOpenGLTexture | None = None
        self.mask_reasons = None
        self.mask_revision = 0
        self.mask_upload_count = 0
        self.mask_uploaded_bytes = 0
        self._mask_epoch = 0
        self._mask_uploaded_epoch = -1
        self._mask_update_bounds = None
        self.show_mask = True
        self.mask_mode = "inspect"
        self.mask_reason = 1
        self.brush_radius_px = 8.0
        self._mask_points: list[tuple[float, float]] = []
        self._mask_hover: tuple[float, float] | None = None
        self.line_mode = False
        self.inspection_line: LineDefinition | None = None
        self._line_start: tuple[float, float] | None = None
        self._line_end: tuple[float, float] | None = None
        self._program: QOpenGLShaderProgram | None = None
        self._vao: QOpenGLVertexArrayObject | None = None
        self._uploaded_revision = -1
        self.max_texture_axis: int | None = None
        self.data_revision = 0
        self.request_generation = 0
        self.upload_count = 0
        self.low_value = 1.0
        self.high_value = 100.0
        self.contrast_mode = "linear"
        self.data_min = 0.0
        self.data_max = 1.0
        self.min_positive: float | None = None
        self.zoom = 1.0
        self.scale_mode = "fit"
        self.pan = QPointF(0.0, 0.0)
        self._pan_dpr = self.devicePixelRatioF()
        self.show_image = True
        self.show_crosshair = True
        self.show_markers = True
        self.yaw = 0.35
        self.pitch = 0.45
        self._last_pointer: QPointF | None = None
        self._box_start: QPointF | None = None
        self._box_end: QPointF | None = None
        self.box_zoom_enabled = False
        self.roi_select_enabled = False
        self._roi_start: QPointF | None = None
        self._roi_end: QPointF | None = None
        self._press_position: QPointF | None = None
        self._dragging = False
        self._band_drag: str | None = None
        self.follow_cursor = True
        self.profile_scope = "band"
        self.profile_row_bounds = (0, 1)
        self.profile_column_bounds = (0, 1)
        self.profile_roi: tuple[int, int, int, int] | None = None
        self.crosshair = (0, 0)
        self.overlays = np.empty((0, 2), dtype=np.float64)
        self._overlay_points = QPolygonF()
        self.setMinimumSize(180, 180)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _request_paint(self) -> None:
        self.request_generation += 1
        self.update()

    def set_mask_mode(self, mode: str) -> None:
        if mode not in ("inspect", "rectangle", "polygon", "brush"):
            raise ValueError("Unsupported detector mode")
        self._mask_points.clear()
        if self.line_mode:
            self.line_mode = False
            self._line_start = self._line_end = None
            self.line_mode_changed.emit(False)
        self._mask_hover = None
        self.mask_mode = mode
        if mode != "inspect":
            self._box_start = self._box_end = self._roi_start = self._roi_end = None
            self._band_drag = None
            self.box_zoom_enabled = self.roi_select_enabled = False
        self.mask_mode_changed.emit(mode)
        self._request_paint()

    def set_line_mode(self, enabled: bool) -> None:
        self.set_mask_mode("inspect")
        self.line_mode = enabled
        self._line_start = self._line_end = None
        self._box_start = self._box_end = self._roi_start = self._roi_end = None
        self.box_zoom_enabled = self.roi_select_enabled = False
        self.line_mode_changed.emit(enabled)
        self._request_paint()

    def set_mask_plane(
        self, reasons, revision: int, *, base_reasons_id=None, changed_bounds=None
    ) -> None:
        if reasons is not None and (
            self.image is None
            or reasons.shape != self.image.shape
            or reasons.dtype != np.uint8
            or reasons.flags.writeable
            or not reasons.flags.c_contiguous
        ):
            raise ValueError("Mask overlay needs an immutable aligned native reason plane")
        self._mask_update_bounds = (
            changed_bounds
            if self.mask_reasons is not None
            and base_reasons_id == id(self.mask_reasons)
            and self._mask_uploaded_epoch == self._mask_epoch
            else None
        )
        self.mask_reasons, self.mask_revision = reasons, revision
        self._mask_epoch += 1
        self._request_paint()

    def _mask_position(self, position: QPointF) -> tuple[float, float] | None:
        if self.image is None:
            return None
        rect = self._rect()
        rows, columns = self.image.shape
        return (
            max(
                -0.5,
                min(columns - 0.5, (position.x() - rect.left()) * columns / rect.width() - 0.5),
            ),
            max(-0.5, min(rows - 0.5, (position.y() - rect.top()) * rows / rect.height() - 0.5)),
        )

    def _finish_mask_gesture(self) -> None:
        try:
            gesture = MaskGesture(
                self.mask_mode, self.mask_reason, tuple(self._mask_points), self.brush_radius_px
            )
        except ValueError:
            gesture = None
        self._mask_points.clear()
        self._request_paint()
        if gesture is not None:
            self.mask_gesture_ready.emit(gesture)

    def set_image(self, image: NDArray[np.generic]) -> None:
        supplied = np.asarray(image)
        if supplied.ndim != 2 or supplied.dtype.kind not in "iuf" or not supplied.size:
            raise ValueError("image must be a nonempty real native-detector plane")
        finite = supplied[np.isfinite(supplied)]
        if finite.size:
            data_min = float(np.min(finite))
            data_max = float(np.max(finite))
            positive = finite[finite > 0]
            min_positive = float(np.min(positive)) if positive.size else None
        else:
            data_min, data_max, min_positive = 0.0, 1.0, None
        low, high = linear_display_limits(data_min, data_max)
        native = np.array(supplied, copy=True, order="C")
        native.setflags(write=False)
        display = np.ascontiguousarray(native, dtype=np.float32)
        self.image, self._display = native, display
        self.set_mask_plane(None, 0)
        self.set_mask_mode("inspect")
        self.data_min, self.data_max, self.min_positive = data_min, data_max, min_positive
        self.low_value, self.high_value = low, high
        self.contrast_mode = "linear"
        self.zoom, self.pan, self.scale_mode = 1.0, QPointF(), "fit"
        self._pan_dpr = self.devicePixelRatioF()
        self.crosshair = (supplied.shape[1] // 2, supplied.shape[0] // 2)
        self.follow_cursor = True
        self.profile_scope = "band"
        self.profile_row_bounds = (self.crosshair[1], self.crosshair[1] + 1)
        self.profile_column_bounds = (self.crosshair[0], self.crosshair[0] + 1)
        self.profile_roi = None
        self._clear_overlays()
        self.cursor_changed.emit(None)
        self.data_revision += 1
        self._request_paint()

    def set_prepared_image(
        self,
        native_counts: NDArray[np.int32],
        display: NDArray[np.float32],
        low_value: float,
        high_value: float,
        max_value: float,
        min_positive: float | None = None,
    ) -> None:
        """Adopt an immutable worker-prepared OSC plane without a GUI copy or scan."""

        if (
            native_counts.ndim != 2
            or not native_counts.size
            or native_counts.dtype
            not in (np.dtype(np.int32), np.dtype(np.float32), np.dtype(np.float64))
            or not native_counts.flags.c_contiguous
            or native_counts.flags.writeable
            or display.shape != native_counts.shape
            or display.dtype != np.float32
            or not display.flags.c_contiguous
            or display.flags.writeable
        ):
            raise ValueError("prepared image needs aligned, read-only native and display planes")
        validate_display_limits(low_value, high_value, "linear")
        if self.max_texture_axis is None:
            raise ValueError("Detector OpenGL context is unavailable; reopen the display")
        if max(native_counts.shape) > self.max_texture_axis:
            raise ValueError(
                f"Detector axis exceeds this OpenGL display's {self.max_texture_axis} pixel limit"
            )
        if min_positive is not None and (not math.isfinite(min_positive) or min_positive <= 0):
            raise ValueError("prepared positive minimum must be positive and finite")
        if not math.isfinite(max_value) or not low_value <= max_value <= high_value:
            raise ValueError("prepared maximum is inconsistent with display levels")
        self.image, self._display = native_counts, display
        self.set_mask_plane(None, 0)
        self.set_mask_mode("inspect")
        self.data_min, self.data_max, self.min_positive = low_value, max_value, min_positive
        self.inspection_line = None
        self.low_value, self.high_value, self.contrast_mode = low_value, high_value, "linear"
        self.zoom, self.pan, self.scale_mode = 1.0, QPointF(), "fit"
        self._pan_dpr = self.devicePixelRatioF()
        self.crosshair = (native_counts.shape[1] // 2, native_counts.shape[0] // 2)
        self.follow_cursor = True
        self.profile_scope = "band"
        self.profile_row_bounds = (self.crosshair[1], self.crosshair[1] + 1)
        self.profile_column_bounds = (self.crosshair[0], self.crosshair[0] + 1)
        self.profile_roi = None
        self._clear_overlays()
        self.cursor_changed.emit(None)
        self.data_revision += 1
        self._request_paint()

    def set_overlays(self, column_row_px: NDArray[np.float64]) -> None:
        points = np.asarray(column_row_px, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or points.shape[0] > 4000
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("overlays must be finite (column_px, row_px) pairs")
        self.overlays = np.array(points, copy=True, order="C")
        self._overlay_points = QPolygonF(
            [QPointF(float(c) + 0.5, float(r) + 0.5) for c, r in self.overlays]
        )
        self._request_paint()
        self.overlays_changed.emit(len(points))

    def _clear_overlays(self) -> None:
        self.overlays = np.empty((0, 2), dtype=np.float64)
        self._overlay_points = QPolygonF()
        self.overlays_changed.emit(0)

    def set_levels(self, low: float, high: float, *, mode: str = "linear") -> None:
        validate_display_limits(low, high, mode)
        self.low_value, self.high_value, self.contrast_mode = low, high, mode
        self._request_paint()
        self.view_state_changed.emit()

    def auto_levels(self, mode: str) -> None:
        if mode == "linear":
            low, high = linear_display_limits(self.data_min, self.data_max)
        elif mode == "signed":
            extent = max(abs(self.data_min), abs(self.data_max), 1.0)
            low, high = -extent, extent
        elif mode == "positive_log":
            if self.min_positive is None:
                raise ValueError("positive-log display requires at least one positive pixel")
            low = self.min_positive
            high = max(self.data_max, low * 1.001)
            if high > FLOAT32_DISPLAY_MAX:
                high = self.data_max
                low = high / 1.001
        else:
            raise ValueError("unsupported contrast mode")
        self.set_levels(low, high, mode=mode)

    def _rect(self) -> QRectF:
        assert self.image is not None
        rows, columns = self.image.shape
        scale = min(self.width() / columns, self.height() / rows) * self.effective_zoom()
        width, height = columns * scale, rows * scale
        return QRectF(
            (self.width() - width) / 2 + self.pan.x(),
            (self.height() - height) / 2 + self.pan.y(),
            width,
            height,
        )

    def _native_zoom(self) -> float:
        assert self.image is not None
        rows, columns = self.image.shape
        return 1.0 / (self.devicePixelRatioF() * min(self.width() / columns, self.height() / rows))

    def effective_zoom(self) -> float:
        return (
            self._native_zoom()
            if self.image is not None and self.scale_mode == "native"
            else self.zoom
        )

    def native_to_widget(self, column_px: float, row_px: float) -> QPointF:
        return QPointF(
            float(self.axis_position(column_px, vertical=False)),
            float(self.axis_position(row_px, vertical=True)),
        )

    def axis_position(
        self, index: float | NDArray[np.float64], *, vertical: bool
    ) -> float | NDArray[np.float64]:
        """Center of one native row/column on the shared viewport transform."""
        rect = self._rect()
        assert self.image is not None
        rows, columns = self.image.shape
        origin = rect.top() if vertical else rect.left()
        extent = rect.height() if vertical else rect.width()
        count = rows if vertical else columns
        return origin + (index + 0.5) * extent / count

    def set_profile_overlay(
        self,
        *,
        scope: str,
        row_bounds: tuple[int, int],
        column_bounds: tuple[int, int],
        roi: tuple[int, int, int, int] | None,
    ) -> None:
        state = (scope, row_bounds, column_bounds, roi)
        if state == (
            self.profile_scope,
            self.profile_row_bounds,
            self.profile_column_bounds,
            self.profile_roi,
        ):
            return
        (
            self.profile_scope,
            self.profile_row_bounds,
            self.profile_column_bounds,
            self.profile_roi,
        ) = state
        self._request_paint()

    def _band_hit(self, position: QPointF) -> str | None:
        if self.image is None or self.profile_scope != "band" or self.follow_cursor:
            return None
        rect = self._rect()
        if not rect.contains(position):
            return None
        rows, columns = self.image.shape
        edges = (
            ("row_start", rect.top() + self.profile_row_bounds[0] * rect.height() / rows),
            ("row_stop", rect.top() + self.profile_row_bounds[1] * rect.height() / rows),
            ("column_start", rect.left() + self.profile_column_bounds[0] * rect.width() / columns),
            ("column_stop", rect.left() + self.profile_column_bounds[1] * rect.width() / columns),
        )
        candidates = [
            (abs((position.y() if name.startswith("row") else position.x()) - edge), name)
            for name, edge in edges
        ]
        nearest, name = min(candidates)
        return name if nearest <= 6 else None

    def widget_to_native(self, position: QPointF) -> tuple[int, int] | None:
        rect = self._rect()
        assert self.image is not None
        if not (
            rect.left() <= position.x() < rect.right()
            and rect.top() <= position.y() < rect.bottom()
        ):
            return None
        rows, columns = self.image.shape
        column = math.floor((position.x() - rect.left()) * columns / rect.width())
        row = math.floor((position.y() - rect.top()) * rows / rect.height())
        if not (0 <= column < columns and 0 <= row < rows):
            return None
        return column, row

    def _zoom_bounds(self) -> tuple[float, float]:
        assert self.image is not None
        native_zoom = self._native_zoom()
        return max(0.0001, min(0.25, native_zoom)), min(10000.0, max(30.0, native_zoom))

    def set_zoom_at(self, zoom: float, anchor: QPointF) -> None:
        if self.image is None:
            return
        minimum, maximum = self._zoom_bounds()
        changed = min(max(zoom, minimum), maximum)
        if changed == self.effective_zoom():
            return
        before = self._rect()
        fraction_x = (anchor.x() - before.left()) / before.width()
        fraction_y = (anchor.y() - before.top()) / before.height()
        self.zoom = changed
        self.scale_mode = "custom"
        after = self._rect()
        self.pan += QPointF(
            anchor.x() - (after.left() + fraction_x * after.width()),
            anchor.y() - (after.top() + fraction_y * after.height()),
        )
        self._request_paint()
        self.view_state_changed.emit()

    def fit_image(self) -> None:
        if self.image is None:
            return
        self.zoom, self.pan, self.scale_mode = 1.0, QPointF(), "fit"
        self._request_paint()
        self.view_state_changed.emit()

    def native_pixel_scale(self) -> None:
        if self.image is None:
            return
        self.zoom = self._native_zoom()
        self.scale_mode = "native"
        scale = 1.0 / self.devicePixelRatioF()
        self.pan = QPointF(
            self.image.shape[1] * scale / 2 - (self.crosshair[0] + 0.5) * scale,
            self.image.shape[0] * scale / 2 - (self.crosshair[1] + 0.5) * scale,
        )
        self._request_paint()
        self.view_state_changed.emit()

    def set_layer_visibility(self, *, image: bool, crosshair: bool, markers: bool) -> None:
        if any(type(value) is not bool for value in (image, crosshair, markers)):
            raise ValueError("layer visibility must be Boolean")
        self.show_image, self.show_crosshair, self.show_markers = image, crosshair, markers
        self._request_paint()
        self.view_state_changed.emit()

    def initializeGL(self) -> None:
        self.max_texture_axis = int(self.context().functions().glGetIntegerv(_GL_MAX_TEXTURE_SIZE))
        if self.max_texture_axis <= 0:
            raise RuntimeError("OpenGL did not report a usable detector texture size")
        program = QOpenGLShaderProgram()
        for kind, source in (
            (QOpenGLShader.ShaderTypeBit.Vertex, _VERTEX),
            (QOpenGLShader.ShaderTypeBit.Fragment, DETECTOR_FRAGMENT_SHADER),
        ):
            if not program.addCacheableShaderFromSourceCode(kind, source):
                raise RuntimeError(program.log())
        if not program.link():
            raise RuntimeError(program.log())
        vao = QOpenGLVertexArrayObject()
        if not vao.create():
            raise RuntimeError("OpenGL vertex array creation failed")
        self._program, self._vao = program, vao
        self._uploaded_revision = -1
        self._mask_uploaded_epoch = -1
        self.context().aboutToBeDestroyed.connect(self.release_resources)

    def release_resources(self) -> None:
        if self._texture is None and self._vao is None:
            return
        self.makeCurrent()
        try:
            if self._texture is not None:
                self._texture.destroy()
            if self._mask_texture is not None:
                self._mask_texture.destroy()
            if self._vao is not None:
                self._vao.destroy()
        finally:
            self._texture = None
            self._mask_texture = None
            self._mask_uploaded_epoch = -1
            self._vao = None
            self._program = None
            self._uploaded_revision = -1
            self.max_texture_axis = None
            self.doneCurrent()

    def _upload_if_needed(self) -> None:
        if self._display is None or self._uploaded_revision == self.data_revision:
            return
        if self._texture is not None:
            self._texture.destroy()
        rows, columns = self._display.shape
        texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        texture.setFormat(QOpenGLTexture.TextureFormat.R32F)
        texture.setSize(columns, rows)
        texture.allocateStorage(QOpenGLTexture.PixelFormat.Red, QOpenGLTexture.PixelType.Float32)
        texture.setMinMagFilters(QOpenGLTexture.Filter.Nearest, QOpenGLTexture.Filter.Nearest)
        texture.setData(
            QOpenGLTexture.PixelFormat.Red,
            QOpenGLTexture.PixelType.Float32,
            VoidPtr(self._display.ctypes.data, self._display.nbytes, False),
        )
        self._texture = texture
        self._uploaded_revision = self.data_revision
        self.upload_count += 1

    def _upload_mask_if_needed(self) -> None:
        if self._mask_uploaded_epoch == self._mask_epoch:
            return
        if self._mask_texture is not None and self.mask_reasons is None:
            self._mask_texture.destroy()
            self._mask_texture = None
        if self.mask_reasons is not None:
            rows, columns = self.mask_reasons.shape
            if self._mask_texture is not None and (
                self._mask_texture.width() != columns or self._mask_texture.height() != rows
            ):
                self._mask_texture.destroy()
                self._mask_texture = None
            if self._mask_texture is None:
                texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
                texture.setFormat(QOpenGLTexture.TextureFormat.R8_UNorm)
                texture.setSize(columns, rows)
                texture.allocateStorage(
                    QOpenGLTexture.PixelFormat.Red, QOpenGLTexture.PixelType.UInt8
                )
                texture.setMinMagFilters(
                    QOpenGLTexture.Filter.Nearest, QOpenGLTexture.Filter.Nearest
                )
                self._mask_texture = texture
                self._mask_update_bounds = None
            if self._mask_update_bounds is None:
                options = QOpenGLPixelTransferOptions()
                options.setAlignment(1)
                self._mask_texture.setData(
                    QOpenGLTexture.PixelFormat.Red,
                    QOpenGLTexture.PixelType.UInt8,
                    VoidPtr(self.mask_reasons.ctypes.data, self.mask_reasons.nbytes, False),
                    options,
                )
                uploaded_bytes = self.mask_reasons.nbytes
            else:
                c0, c1, r0, r1 = self._mask_update_bounds
                block = np.ascontiguousarray(self.mask_reasons[r0:r1, c0:c1])
                functions = self.context().functions()
                alignment = int(functions.glGetIntegerv(0x0CF5))
                self._mask_texture.bind(1)
                try:
                    functions.glPixelStorei(0x0CF5, 1)
                    functions.glTexSubImage2D(
                        0x0DE1,
                        0,
                        c0,
                        r0,
                        c1 - c0,
                        r1 - r0,
                        0x1903,
                        0x1401,
                        VoidPtr(block.ctypes.data, block.nbytes, False),
                    )
                finally:
                    functions.glPixelStorei(0x0CF5, alignment)
                    self._mask_texture.release(1)
                uploaded_bytes = block.nbytes
            self.mask_upload_count += 1
            self.mask_uploaded_bytes += uploaded_bytes
        self._mask_uploaded_epoch = self._mask_epoch

    def paintGL(self) -> None:
        functions = self.context().functions()
        functions.glClearColor(0.08, 0.09, 0.11, 1.0)
        functions.glClear(0x00004000)
        if self.image is None or self._program is None or self._vao is None:
            self.painted.emit(self.request_generation, perf_counter())
            return
        self._upload_if_needed()
        self._upload_mask_if_needed()
        rect = self._rect() if not self.plane else QRectF(0.1, 0.1, 0.8, 0.8)
        if self.show_image:
            self._program.bind()
            self._vao.bind()
            assert self._texture is not None
            self._texture.bind(0)
            if self._mask_texture is not None:
                self._mask_texture.bind(1)
            functions.glUniform1i(self._program.uniformLocation("exclusion"), 1)
            functions.glUniform1i(
                self._program.uniformLocation("show_exclusion"),
                int(self.show_mask and self._mask_texture is not None),
            )
            functions.glUniform1i(self._program.uniformLocation("detector"), 0)
            functions.glUniform4f(
                self._program.uniformLocation("rect"),
                float(rect.x() / self.width()),
                float(rect.y() / self.height()),
                float(rect.width() / self.width()),
                float(rect.height() / self.height()),
            )
            functions.glUniform1i(self._program.uniformLocation("plane"), int(self.plane))
            functions.glUniform1f(self._program.uniformLocation("yaw"), float(self.yaw))
            functions.glUniform1f(self._program.uniformLocation("pitch"), float(self.pitch))
            functions.glUniform1f(self._program.uniformLocation("low_value"), float(self.low_value))
            functions.glUniform1f(
                self._program.uniformLocation("high_value"), float(self.high_value)
            )
            functions.glUniform1i(
                self._program.uniformLocation("contrast_mode"),
                {"linear": 0, "signed": 1, "positive_log": 2}[self.contrast_mode],
            )
            functions.glDrawArrays(0x0004, 0, 6)
            self._texture.release()
            if self._mask_texture is not None:
                self._mask_texture.release(1)
            self._vao.release()
            self._program.release()
        if not self.plane:
            painter = QPainter(self)
            rows, columns = self.image.shape
            row_scale = rect.height() / rows
            column_scale = rect.width() / columns
            painter.setClipRect(rect)
            if self.profile_scope == "band":
                r0, r1 = self.profile_row_bounds
                c0, c1 = self.profile_column_bounds
                painter.fillRect(
                    QRectF(
                        rect.left(),
                        rect.top() + r0 * row_scale,
                        rect.width(),
                        (r1 - r0) * row_scale,
                    ),
                    QColor(110, 215, 170, 55),
                )
                painter.fillRect(
                    QRectF(
                        rect.left() + c0 * column_scale,
                        rect.top(),
                        (c1 - c0) * column_scale,
                        rect.height(),
                    ),
                    QColor(120, 170, 255, 55),
                )
                painter.setPen(QPen(QColor(130, 240, 185, 220), 1))
                for edge in (r0, r1):
                    y = rect.top() + edge * row_scale
                    painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
                painter.setPen(QPen(QColor(145, 190, 255, 220), 1))
                for edge in (c0, c1):
                    x = rect.left() + edge * column_scale
                    painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            elif self.profile_scope == "roi" and self.profile_roi is not None:
                c0, c1, r0, r1 = self.profile_roi
                bounds = QRectF(
                    rect.left() + c0 * column_scale,
                    rect.top() + r0 * row_scale,
                    (c1 - c0) * column_scale,
                    (r1 - r0) * row_scale,
                )
                painter.fillRect(bounds, QColor(160, 220, 160, 58))
                painter.setPen(QPen(QColor(175, 235, 175, 230), 1))
                painter.drawRect(bounds)
            painter.setClipping(False)
            if self.show_markers and self.overlays.size:
                pen = QPen(QColor(100, 245, 235), 1)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setClipRect(rect)
                painter.setTransform(
                    QTransform(
                        rect.width() / self.image.shape[1],
                        0,
                        0,
                        rect.height() / self.image.shape[0],
                        rect.left(),
                        rect.top(),
                    )
                )
                painter.drawPoints(self._overlay_points)
                painter.resetTransform()
                painter.setClipping(False)
            if self.show_crosshair:
                painter.setPen(QPen(QColor(255, 221, 96, 220), 1))
                point = self.native_to_widget(*self.crosshair)
                painter.drawLine(QPointF(point.x(), rect.top()), QPointF(point.x(), rect.bottom()))
                painter.drawLine(QPointF(rect.left(), point.y()), QPointF(rect.right(), point.y()))
            if self._box_start is not None and self._box_end is not None:
                painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine))
                painter.drawRect(QRectF(self._box_start, self._box_end).normalized())
            if self._roi_start is not None and self._roi_end is not None:
                painter.setPen(QPen(QColor(175, 235, 175), 1, Qt.PenStyle.DashLine))
                painter.drawRect(QRectF(self._roi_start, self._roi_end).normalized())
            if self._mask_points:
                painter.setPen(QPen(QColor(255, 100, 145), 2, Qt.PenStyle.DashLine))
                points = QPolygonF([self.native_to_widget(*p) for p in self._mask_points])
                if self.mask_mode == "rectangle" and len(points) == 2:
                    painter.drawRect(QRectF(points[0], points[1]).normalized())
                else:
                    painter.drawPolyline(points)
                    if self.mask_mode == "brush":
                        radius = self.brush_radius_px * column_scale
                        painter.drawEllipse(points[-1], radius, radius)
            if self.mask_mode != "inspect" and self._mask_hover is not None:
                point = self.native_to_widget(*self._mask_hover)
                painter.setPen(QPen(QColor(255, 100, 145), 1))
                radius = self.brush_radius_px * column_scale if self.mask_mode == "brush" else 3.0
                painter.drawEllipse(point, radius, radius)
                if self.mask_mode == "polygon" and self._mask_points:
                    painter.drawLine(self.native_to_widget(*self._mask_points[-1]), point)
            line_points = (
                (self._line_start, self._line_end)
                if self._line_start is not None and self._line_end is not None
                else None
                if self.inspection_line is None
                else (self.inspection_line.start_column_row, self.inspection_line.end_column_row)
            )
            if line_points is not None:
                painter.setClipRect(rect)
                painter.setPen(
                    QPen(
                        QColor(255, 135, 235),
                        2,
                        Qt.PenStyle.DashLine
                        if self._line_start is not None
                        else Qt.PenStyle.SolidLine,
                    )
                )
                first, last = (self.native_to_widget(*p) for p in line_points)
                painter.drawLine(first, last)
                painter.drawEllipse(first, 3, 3)
                painter.drawEllipse(last, 3, 3)
                painter.setClipping(False)
            painter.end()
        self.painted.emit(self.request_generation, perf_counter())

    def wheelEvent(self, event) -> None:
        if self.image is not None and event.angleDelta().y():
            self.set_zoom_at(self.zoom * 1.2 ** (event.angleDelta().y() / 120), event.position())
            event.accept()

    def mousePressEvent(self, event) -> None:
        if self.line_mode and event.button() == Qt.MouseButton.LeftButton:
            self._line_start = self._line_end = self._mask_position(event.position())
            self._request_paint()
            event.accept()
            return
        if self.mask_mode != "inspect" and event.button() == Qt.MouseButton.LeftButton:
            point = self._mask_position(event.position())
            if point is not None:
                if len(self._mask_points) >= MAX_POINTS:
                    self.set_mask_mode("inspect")
                    self.mask_input_error.emit(
                        f"Gesture canceled: maximum {MAX_POINTS} native vertices"
                    )
                    event.accept()
                    return
                if self.mask_mode != "polygon":
                    self._mask_points = [point]
                else:
                    self._mask_points.append(point)
                self._request_paint()
            event.accept()
            return
        self._last_pointer = event.position()
        self._press_position = (
            event.position() if event.button() == Qt.MouseButton.LeftButton else None
        )
        self._dragging = False
        if self.image is None or event.button() != Qt.MouseButton.LeftButton or self.plane:
            return
        if self.roi_select_enabled:
            self._roi_start = self._roi_end = event.position()
            self._request_paint()
        elif self.box_zoom_enabled:
            self._box_start = event.position()
            self._box_end = event.position()
            self._request_paint()
        else:
            self._band_drag = self._band_hit(event.position())

    def mouseMoveEvent(self, event) -> None:
        if (
            self.line_mode
            and self._line_start is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._line_end = self._mask_position(event.position())
            self._request_paint()
            event.accept()
            return
        if self.mask_mode != "inspect":
            self._mask_hover = self._mask_position(event.position())
            self._request_paint()
            if event.buttons() & Qt.MouseButton.LeftButton and self._mask_points:
                point = self._mask_position(event.position())
                if self.mask_mode == "rectangle":
                    self._mask_points = [self._mask_points[0], point]
                elif self.mask_mode == "brush" and point != self._mask_points[-1]:
                    if len(self._mask_points) >= MAX_POINTS:
                        self.set_mask_mode("inspect")
                        self.mask_input_error.emit(
                            f"Gesture canceled: maximum {MAX_POINTS} native vertices"
                        )
                        event.accept()
                        return
                    self._mask_points.append(point)
                self._request_paint()
            elif event.buttons() & Qt.MouseButton.MiddleButton and self._last_pointer is not None:
                self.pan += event.position() - self._last_pointer
                self._request_paint()
                self.view_state_changed.emit()
            self._last_pointer = event.position()
            native = self.widget_to_native(event.position())
            if native is not None:
                c, r = native
                self.cursor_changed.emit((c, r, self.image[r, c].item()))
            event.accept()
            return
        if self._roi_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._roi_end = event.position()
            self._request_paint()
        elif self._box_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._box_end = event.position()
            self._request_paint()
        elif self._band_drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if self._press_position is not None and (
                self._dragging or (event.position() - self._press_position).manhattanLength() >= 5
            ):
                self._dragging = True
                rect = self._rect()
                rows, columns = self.image.shape
                vertical = self._band_drag.startswith("row")
                origin = rect.top() if vertical else rect.left()
                extent = rect.height() if vertical else rect.width()
                count = rows if vertical else columns
                position = event.position().y() if vertical else event.position().x()
                edge = round((position - origin) * count / extent)
                self.band_edge_dragged.emit(self._band_drag, max(0, min(count, edge)))
        elif self._last_pointer is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._last_pointer
            if self._press_position is not None and (
                self._dragging or (event.position() - self._press_position).manhattanLength() >= 5
            ):
                self._dragging = True
                if self.plane:
                    self.yaw += delta.x() * 0.008
                    self.pitch = max(-1.3, min(1.3, self.pitch + delta.y() * 0.008))
                else:
                    self.pan += delta
                self._request_paint()
                self.view_state_changed.emit()
        if self.image is not None and not self.plane:
            native = self.widget_to_native(event.position())
            if native is None:
                self.cursor_changed.emit(None)
            else:
                row, column = native[1], native[0]
                self.cursor_changed.emit((column, row, self.image[row, column].item()))
                if not event.buttons() and self.follow_cursor and native != self.crosshair:
                    self.crosshair = native
                    self._request_paint()
                    self.crosshair_changed.emit()
        self._last_pointer = event.position()

    def mouseReleaseEvent(self, event) -> None:
        if self.line_mode and event.button() == Qt.MouseButton.LeftButton:
            start, end = self._line_start, self._mask_position(event.position())
            self._line_start = self._line_end = None
            if start is not None and end is not None and start != end:
                self.line_endpoints_ready.emit((start, end))
            self._request_paint()
            event.accept()
            return
        if self.mask_mode != "inspect":
            if (
                event.button() == Qt.MouseButton.LeftButton
                and self._mask_points
                and self.mask_mode != "polygon"
            ):
                point = self._mask_position(event.position())
                if self.mask_mode == "rectangle":
                    self._mask_points = [self._mask_points[0], point]
                elif len(self._mask_points) < MAX_POINTS:
                    self._mask_points.append(point)
                elif point != self._mask_points[-1]:
                    self.set_mask_mode("inspect")
                    self.mask_input_error.emit(
                        f"Gesture canceled: maximum {MAX_POINTS} native vertices"
                    )
                    event.accept()
                    return
                self._finish_mask_gesture()
            self._last_pointer = None
            event.accept()
            return
        if self._roi_start is not None and self.image is not None:
            box = QRectF(self._roi_start, event.position()).normalized().intersected(self._rect())
            self._roi_start = self._roi_end = None
            if box.width() >= 4 and box.height() >= 4:
                rect = self._rect()
                rows, columns = self.image.shape
                c0 = max(0, math.floor((box.left() - rect.left()) * columns / rect.width()))
                c1 = min(columns, math.ceil((box.right() - rect.left()) * columns / rect.width()))
                r0 = max(0, math.floor((box.top() - rect.top()) * rows / rect.height()))
                r1 = min(rows, math.ceil((box.bottom() - rect.top()) * rows / rect.height()))
                if c0 < c1 and r0 < r1:
                    self.roi_selected.emit((c0, c1, r0, r1))
            self._request_paint()
        elif self._band_drag is not None:
            self._band_drag = None
            if not self._dragging and self.image is not None:
                native = self.widget_to_native(event.position())
                if native is not None:
                    self.pin_requested.emit()
                    if native != self.crosshair:
                        self.crosshair = native
                        self._request_paint()
                        self.crosshair_changed.emit()
        elif self._box_start is not None and self.image is not None:
            box = QRectF(self._box_start, event.position()).normalized().intersected(self._rect())
            self._box_start = self._box_end = None
            if box.width() >= 8 and box.height() >= 8:
                fraction = min(self.width() / box.width(), self.height() / box.height())
                self.set_zoom_at(self.zoom * fraction, box.center())
                self.pan += QPointF(
                    self.width() / 2 - box.center().x(),
                    self.height() / 2 - box.center().y(),
                )
                self._request_paint()
                self.view_state_changed.emit()
            else:
                self._request_paint()
        elif (
            self.image is not None
            and not self.plane
            and event.button() == Qt.MouseButton.LeftButton
            and not self._dragging
        ):
            native = self.widget_to_native(event.position())
            if native is not None:
                self.pin_requested.emit()
                if native != self.crosshair:
                    self.crosshair = native
                    self._request_paint()
                    self.crosshair_changed.emit()
        self._last_pointer = None
        self._press_position = None
        self._dragging = False

    def leaveEvent(self, event) -> None:
        self.cursor_changed.emit(None)
        self._last_pointer = None
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.image is not None and self.scale_mode == "native":
            self.zoom = self._native_zoom()
            self.view_state_changed.emit()

    def event(self, event) -> bool:
        result = super().event(event)
        if event.type() in (
            QEvent.Type.DevicePixelRatioChange,
            QEvent.Type.ScreenChangeInternal,
        ):
            current_dpr = self.devicePixelRatioF()
            if getattr(self, "image", None) is not None and current_dpr != self._pan_dpr:
                self.pan *= self._pan_dpr / current_dpr
                if self.scale_mode == "native":
                    self.zoom = self._native_zoom()
                self._request_paint()
                self.view_state_changed.emit()
            self._pan_dpr = current_dpr
        return result

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.set_mask_mode("inspect")
            self._box_start = self._box_end = self._roi_start = self._roi_end = None
            self.box_zoom_enabled = self.roi_select_enabled = False
            event.accept()
            return
        if self.mask_mode == "polygon" and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish_mask_gesture()
            event.accept()
            return
        if self.image is not None and event.key() in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        ):
            dx = 32 * (int(event.key() == Qt.Key.Key_Right) - int(event.key() == Qt.Key.Key_Left))
            dy = 32 * (int(event.key() == Qt.Key.Key_Down) - int(event.key() == Qt.Key.Key_Up))
            self.pan += QPointF(dx, dy)
            self._request_paint()
            self.view_state_changed.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class ProfilePlot(QWidget):
    """Persistent lightweight profile plot; the native coordinate axis is shared with the image."""

    painted = Signal(int, float)

    def __init__(
        self, *, vertical: bool, source_view: DetectorTextureView, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.vertical = vertical
        self.source_view = source_view
        self.values: NDArray[np.generic] | None = None
        self.support: NDArray[np.int64] | None = None
        self.generation = 0
        self.intensity_limits: tuple[float, float] | None = None
        self.setMinimumSize(50 if vertical else 180, 180 if vertical else 50)
        if vertical:
            self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def set_intensity_limits(self, limits: tuple[float, float] | None) -> None:
        if self.intensity_limits != limits:
            self.intensity_limits = limits
            self.generation = self.source_view.request_generation
            self.update()

    def set_values(self, values: NDArray[np.generic], support: NDArray[np.int64]) -> None:
        self.values = values
        self.support = support
        self.generation = self.source_view.request_generation
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 28, 34))
        if (
            self.values is not None
            and self.support is not None
            and self.source_view.image is not None
        ):
            valid = np.isfinite(self.values) & (self.support > 0)
            selected = self.values[valid]
            if self.intensity_limits is None:
                minimum = min(float(np.min(selected, initial=0)), 0.0)
                maximum = max(float(np.max(selected, initial=0)), 1.0)
            else:
                minimum, maximum = self.intensity_limits
            span = maximum - minimum
            rect = self.source_view._rect()
            painter.setPen(QPen(QColor(104, 218, 243), 1))
            origin = rect.top() if self.vertical else rect.left()
            extent = rect.height() if self.vertical else rect.width()
            limit = self.height() if self.vertical else self.width()
            starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
            stops = np.flatnonzero(valid & ~np.r_[valid[1:], False]) + 1
            for start, stop in zip(starts, stops, strict=True):
                positions = self.source_view.axis_position(
                    np.arange(start, stop, dtype=np.float64), vertical=self.vertical
                )
                if positions[0] > limit or positions[-1] < -1:
                    continue
                first_inside = int(np.searchsorted(positions, -1, side="left"))
                last_inside = int(np.searchsorted(positions, limit, side="right"))
                first = max(0, first_inside - 1)
                last = min(positions.size, last_inside + 1)
                positions = positions[first:last]
                amplitudes = self.values[start + first : start + last]
                if amplitudes.size > limit * 2:
                    pixel = np.floor(positions).astype(np.int64)
                    group_start = np.r_[0, np.flatnonzero(np.diff(pixel)) + 1]
                    low = np.minimum.reduceat(amplitudes, group_start)
                    high = np.maximum.reduceat(amplitudes, group_start)
                    first = amplitudes[group_start]
                    last = amplitudes[np.r_[group_start[1:] - 1, amplitudes.size - 1]]
                    positions = pixel[group_start].astype(np.float64) + 0.5
                    amplitudes = np.column_stack(
                        (np.where(first <= last, low, high), np.where(first <= last, high, low))
                    ).ravel()
                    positions = np.repeat(positions, 2)

                def plot_point(position: float, amplitude: float) -> QPointF:
                    normalized = (amplitude - minimum) / span
                    if self.vertical:
                        return QPointF(normalized * (self.width() - 1), position)
                    return QPointF(position, (1 - normalized) * (self.height() - 1))

                if np.all((amplitudes >= minimum) & (amplitudes <= maximum)):
                    points = QPolygonF(
                        [
                            plot_point(float(position), float(amplitude))
                            for position, amplitude in zip(positions, amplitudes, strict=True)
                        ]
                    )
                    if len(points) == 1:
                        painter.drawPoint(points[0])
                    else:
                        painter.drawPolyline(points)
                    continue

                segment: list[QPointF] = []
                for index in range(len(amplitudes) - 1):
                    a, b = float(amplitudes[index]), float(amplitudes[index + 1])
                    if max(a, b) < minimum or min(a, b) > maximum:
                        if len(segment) > 1:
                            painter.drawPolyline(QPolygonF(segment))
                        segment = []
                        continue

                    first_position, last_position = (
                        float(positions[index]),
                        float(positions[index + 1]),
                    )
                    first_value, last_value = a, b
                    if a < minimum or a > maximum:
                        first_value = minimum if a < minimum else maximum
                        first_position += (last_position - first_position) * _segment_crossing(
                            a, b, first_value
                        )
                    if b < minimum or b > maximum:
                        last_value = minimum if b < minimum else maximum
                        last_position = float(positions[index]) + (
                            float(positions[index + 1]) - float(positions[index])
                        ) * _segment_crossing(a, b, last_value)
                    first_point = plot_point(first_position, first_value)
                    last_point = plot_point(last_position, last_value)
                    if segment and segment[-1] != first_point:
                        if len(segment) > 1:
                            painter.drawPolyline(QPolygonF(segment))
                        segment = []
                    if not segment:
                        segment.append(first_point)
                    segment.append(last_point)
                if len(segment) > 1:
                    painter.drawPolyline(QPolygonF(segment))
                if len(amplitudes) == 1 and minimum <= float(amplitudes[0]) <= maximum:
                    painter.drawPoint(plot_point(float(positions[0]), float(amplitudes[0])))
            count = self.source_view.image.shape[0 if self.vertical else 1]
            origin = rect.top() if self.vertical else rect.left()
            extent = rect.height() if self.vertical else rect.width()
            limit = self.height() if self.vertical else self.width()
            first_visible = max(0, math.floor(-origin * count / extent))
            last_visible = min(count - 1, math.ceil((limit - origin) * count / extent))
            span = max(1, last_visible - first_visible)
            rough_step = max(1, math.ceil(span / 4))
            magnitude = 10 ** math.floor(math.log10(rough_step))
            step = next(
                (part * magnitude for part in (1, 2, 5, 10) if part * magnitude >= rough_step),
                10 * magnitude,
            )
            painter.setPen(QColor(215, 225, 231))
            painter.drawText(
                4,
                12,
                self.source_view.row_axis_label
                if self.vertical
                else self.source_view.column_axis_label,
            )
            for index in range((first_visible + step - 1) // step * step, last_visible + 1, step):
                position = self.source_view.axis_position(index, vertical=self.vertical)
                if not 0 <= position < limit:
                    continue
                if self.vertical:
                    if position - 9 < 18:
                        continue
                    painter.drawText(
                        QRectF(0, position - 9, self.width() - 3, 18),
                        Qt.AlignmentFlag.AlignRight,
                        str(index),
                    )
                else:
                    painter.drawText(
                        QRectF(position - 28, self.height() - 17, 56, 16),
                        Qt.AlignmentFlag.AlignHCenter,
                        str(index),
                    )
        painter.end()
        self.generation = self.source_view.request_generation
        self.painted.emit(self.generation, perf_counter())


class DetectorPanel(QWidget):
    profile_requested = Signal()
    """A reusable native image with exact linked horizontal and vertical bands."""

    def __init__(self, parent: QWidget | None = None, *, compact: bool = False) -> None:
        super().__init__(parent)
        self.view = DetectorTextureView(parent=self)
        self.horizontal = ProfilePlot(vertical=False, source_view=self.view, parent=self)
        self.vertical = ProfilePlot(vertical=True, source_view=self.view, parent=self)
        navigation = QWidget(self)
        nav_layout = QHBoxLayout(navigation)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip(
            "Fit the whole detector (Ctrl+0). Wheel zooms at the pointer; drag pans."
        )
        self.native_button = QPushButton("1:1 px")
        self.native_button.setToolTip("One detector pixel per physical display pixel (Ctrl+1).")
        self.box_button = QPushButton("Box zoom")
        self.box_button.setCheckable(True)
        self.box_button.setToolTip("Drag a detector rectangle to magnify it (Ctrl+B).")
        self.export_button = QPushButton("Export figure + profiles")
        self.export_button.setToolTip("Save the visible detector figure and exact native profiles.")
        self.export_button.setEnabled(False)
        for button in (self.fit_button, self.native_button, self.box_button, self.export_button):
            nav_layout.addWidget(button)
        navigation_hint = QLabel("Drag: pan · Wheel: zoom · Arrows: pan")
        navigation_hint.setWordWrap(True)
        nav_layout.addWidget(navigation_hint, 1)

        contrast = QWidget(self)
        contrast_layout = QHBoxLayout(contrast)
        contrast_layout.setContentsMargins(0, 0, 0, 0)
        self.mode_control = QComboBox()
        for label, mode in (
            ("Linear", "linear"),
            ("Signed ±", "signed"),
            ("Positive log", "positive_log"),
        ):
            self.mode_control.addItem(label, mode)
        self.low_control = QLineEdit()
        self.high_control = QLineEdit()
        for control in (self.low_control, self.high_control):
            control.setPlaceholderText("scientific value")
            control.setMinimumWidth(100)
        self.apply_button = QPushButton("Apply")
        self.auto_button = QPushButton("Auto")
        for widget in (
            self.mode_control,
            QLabel("Low"),
            self.low_control,
            QLabel("High"),
            self.high_control,
            self.apply_button,
            self.auto_button,
        ):
            contrast_layout.addWidget(widget)

        profile_controls = QWidget(self)
        profile_layout = QGridLayout(profile_controls)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        self.pin_center_button = QPushButton("Follow pointer")
        self.pin_center_button.setCheckable(True)
        self.pin_center_button.setToolTip(
            "Pin the integration center; pointer readout still follows the mouse."
        )
        self.column_control = QSpinBox()
        self.row_control = QSpinBox()
        self.row_width_control = QSpinBox()
        self.column_width_control = QSpinBox()
        for control in (
            self.column_control,
            self.row_control,
            self.row_width_control,
            self.column_width_control,
        ):
            control.setKeyboardTracking(False)
            control.setMaximum(16_384)
        for control in (self.row_width_control, self.column_width_control):
            control.setMinimum(1)
        for label, control, row, column in (
            ("column_px", self.column_control, 0, 1),
            ("row_px", self.row_control, 0, 3),
            ("row width", self.row_width_control, 1, 1),
            ("column width", self.column_width_control, 1, 3),
        ):
            coordinate_label = QLabel(label)
            if label == "column_px":
                self.column_coordinate_label = coordinate_label
            elif label == "row_px":
                self.row_coordinate_label = coordinate_label
            profile_layout.addWidget(coordinate_label, row, column - 1)
            profile_layout.addWidget(control, row, column)
        profile_layout.addWidget(self.pin_center_button, 2, 0)
        self.profile_measure_control = QComboBox()
        self.profile_measure_control.addItem("Sum", "sum")
        self.profile_measure_control.addItem("Mean / valid px", "mean")
        self.profile_scope_control = QComboBox()
        for label, scope in (
            ("Bands", "band"),
            ("Full detector", "full"),
            ("Inspection ROI", "roi"),
        ):
            self.profile_scope_control.addItem(label, scope)
        self.profile_scope_control.model().item(2).setEnabled(False)
        self.draw_roi_button = QPushButton("Draw ROI")
        self.draw_roi_button.setCheckable(True)
        profile_layout.addWidget(self.profile_measure_control, 2, 1)
        profile_layout.addWidget(self.profile_scope_control, 2, 2)
        profile_layout.addWidget(self.draw_roi_button, 2, 3)

        scale_controls = QWidget(self)
        scale_layout = QHBoxLayout(scale_controls)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        self.horizontal_pin_scale = QCheckBox("H pinned scale")
        self.vertical_pin_scale = QCheckBox("V pinned scale")
        self.horizontal_scale_low = QLineEdit("0")
        self.horizontal_scale_high = QLineEdit("1")
        self.vertical_scale_low = QLineEdit("0")
        self.vertical_scale_high = QLineEdit("1")
        for entry in (
            self.horizontal_scale_low,
            self.horizontal_scale_high,
            self.vertical_scale_low,
            self.vertical_scale_high,
        ):
            entry.setMaximumWidth(90)
            entry.setPlaceholderText("scientific value")
        self.apply_profile_scale = QPushButton("Apply scales")
        for widget in (
            self.horizontal_pin_scale,
            self.horizontal_scale_low,
            self.horizontal_scale_high,
            self.vertical_pin_scale,
            self.vertical_scale_low,
            self.vertical_scale_high,
            self.apply_profile_scale,
        ):
            scale_layout.addWidget(widget)
        self.profile_status = QLabel("Bands: 1 row x 1 column · sum")
        self.profile_status.setWordWrap(True)

        layers = QWidget(self)
        layer_layout = QHBoxLayout(layers)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        self.image_layer = QCheckBox("Image")
        self.crosshair_layer = QCheckBox("Crosshair")
        self.marker_layer = QCheckBox("Markers (0)")
        self.fitted_layer = QCheckBox("Fitted results unavailable")
        self.fitted_layer.setEnabled(False)
        self.fitted_layer.setToolTip("No fitted result is loaded for this acquisition.")
        for control in (
            self.image_layer,
            self.crosshair_layer,
            self.marker_layer,
            self.fitted_layer,
        ):
            layer_layout.addWidget(control)
        layer_layout.addStretch()
        self.cursor_label = QLabel(
            "Pointer outside detector · Q/angles unavailable · saturation unknown"
        )
        self.cursor_label.setWordWrap(True)
        self.contrast_note = QLabel(
            "Contrast clips colors only; checkerboard marks invalid or hidden log pixels."
        )
        self.contrast_note.setWordWrap(True)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(navigation, 0, 0, 1, 2)
        layout.addWidget(contrast, 1, 0, 1, 2)
        layout.addWidget(profile_controls, 3, 0, 1, 2)
        layout.addWidget(scale_controls, 4, 0, 1, 2)
        layout.addWidget(self.horizontal, 5, 0)
        layout.addWidget(self.view, 6, 0)
        layout.addWidget(self.vertical, 6, 1)
        layout.addWidget(self.profile_status, 7, 0, 1, 2)
        layout.addWidget(self.cursor_label, 8, 0, 1, 2)
        layout.addWidget(self.contrast_note, 9, 0, 1, 2)
        layout.addWidget(layers, 10, 0, 1, 2)
        mask_controls = QWidget(self)
        mask_layout = QGridLayout(mask_controls)
        mask_layout.setContentsMargins(0, 0, 0, 0)
        self.mask_tool = QComboBox()
        for label, mode in (
            ("Inspect", "inspect"),
            ("Rectangle", "rectangle"),
            ("Polygon", "polygon"),
            ("Brush", "brush"),
        ):
            self.mask_tool.addItem(label, mode)
        self.mask_reason = QComboBox()
        for reason in (1, 2, 3, 4, 0):
            self.mask_reason.addItem(REASONS[reason] if reason else "Reinclude", reason)
        self.brush_radius = QSpinBox()
        self.brush_radius.setRange(1, 256)
        self.brush_radius.setValue(8)
        self.brush_radius.setSuffix(" px radius")
        self.mask_import_button = QPushButton("Import .npy")
        self.mask_undo_button = QPushButton("Undo mask")
        self.mask_redo_button = QPushButton("Redo mask")
        self.mask_cancel_button = QPushButton("Cancel prep")
        self.mask_cancel_button.setEnabled(False)
        self.mask_layer = QCheckBox("Exclusions visible")
        self.mask_layer.setChecked(True)
        self.mask_history_label = QLabel("Undo limit: 32 actions / 512 KiB; session only")
        self.mask_history_label.setWordWrap(True)
        self.mask_hint = QLabel(
            "Inspect: pointer profiles; masking preserves the integration center"
        )
        self.mask_hint.setWordWrap(False)
        self.mask_preparation_status = QLabel(
            "No exclusions; native masks bind this acquisition/source"
        )
        self.mask_preparation_status.setWordWrap(False)
        line_height = self.fontMetrics().lineSpacing()
        self.profile_status.setFixedHeight(4 * line_height)
        self.mask_hint.setFixedHeight(line_height)
        self.mask_history_label.setFixedHeight(line_height)
        self.cursor_label.setFixedHeight(2 * line_height)
        self.mask_preparation_status.setFixedHeight(line_height)
        for label in (
            self.mask_hint,
            self.mask_history_label,
            self.mask_preparation_status,
            self.cursor_label,
        ):
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        for column, widget in enumerate(
            (self.mask_tool, self.mask_reason, self.brush_radius, self.mask_import_button)
        ):
            mask_layout.addWidget(widget, 0, column)
        for column, widget in enumerate(
            (self.mask_undo_button, self.mask_redo_button, self.mask_layer)
        ):
            mask_layout.addWidget(widget, 1, column)
        mask_layout.addWidget(self.mask_cancel_button, 1, 3)
        mask_layout.addWidget(self.mask_history_label, 2, 2, 1, 2)
        mask_layout.addWidget(self.mask_hint, 2, 0, 1, 2)
        mask_layout.addWidget(self.mask_preparation_status, 3, 0, 1, 4)
        layout.addWidget(mask_controls, 2, 0, 1, 2)
        if compact:
            for widget in (
                navigation,
                contrast,
                mask_controls,
                profile_controls,
                scale_controls,
                self.contrast_note,
                layers,
            ):
                widget.hide()
            self.profile_status.setFixedHeight(2 * line_height)
            self.view.setMinimumSize(140, 140)
            self.vertical.setMinimumHeight(140)
            self.horizontal.setMinimumHeight(40)
        self.mask_tool.currentIndexChanged.connect(
            lambda: self.view.set_mask_mode(self.mask_tool.currentData())
        )
        self.view.mask_mode_changed.connect(self._mask_mode_changed)
        self.mask_reason.currentIndexChanged.connect(
            lambda: setattr(self.view, "mask_reason", self.mask_reason.currentData())
        )
        self.brush_radius.valueChanged.connect(
            lambda radius: setattr(self.view, "brush_radius_px", float(radius))
        )
        self.mask_layer.toggled.connect(self._mask_visibility_changed)
        layout.setRowStretch(6, 1)
        layout.setColumnStretch(0, 1)
        self.view.crosshair_changed.connect(self._refresh_profile_if_needed)
        self.view.crosshair_changed.connect(self._sync_profile_position)
        self.view.pin_requested.connect(lambda: self.pin_center_button.setChecked(True))
        self.view.band_edge_dragged.connect(self._drag_band_edge)
        self.view.roi_selected.connect(self._apply_roi)
        self.view.painted.connect(self._refresh_axes)
        self.view.view_state_changed.connect(self._sync_controls)
        self.view.cursor_changed.connect(self._show_cursor)
        self.view.overlays_changed.connect(self._marker_count_changed)
        self.fit_button.clicked.connect(self.view.fit_image)
        self.native_button.clicked.connect(self.view.native_pixel_scale)
        self.box_button.toggled.connect(self._set_box_zoom)
        self.draw_roi_button.toggled.connect(self._set_roi_selection)
        self.pin_center_button.toggled.connect(self._set_pin_center)
        for control in (self.column_control, self.row_control):
            control.valueChanged.connect(self._center_controls_changed)
        for control in (self.row_width_control, self.column_width_control):
            control.valueChanged.connect(self._query_changed)
        self.profile_measure_control.currentIndexChanged.connect(self._query_changed)
        self.profile_scope_control.currentIndexChanged.connect(self._query_changed)
        self.apply_profile_scale.clicked.connect(self._apply_profile_scales)
        self.horizontal_pin_scale.toggled.connect(self._apply_profile_scales)
        self.vertical_pin_scale.toggled.connect(self._apply_profile_scales)
        self._shortcuts = []
        for sequence, action in (
            ("Ctrl+0", self.view.fit_image),
            ("Ctrl+1", self.view.native_pixel_scale),
            ("Ctrl+B", self.box_button.toggle),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(action)
            self._shortcuts.append(shortcut)
        self.mode_control.currentIndexChanged.connect(self._mode_changed)
        self.apply_button.clicked.connect(self._apply_contrast)
        self.auto_button.clicked.connect(self._auto_contrast)
        for control in (self.image_layer, self.crosshair_layer, self.marker_layer):
            control.toggled.connect(self._apply_layers)
        self._profile_key: tuple[object, ...] | None = None
        self._full_profiles: BandProfiles | None = None
        self._full_mean_profiles: BandProfiles | None = None
        self._roi_cache_key: tuple[object, ...] | None = None
        self._roi_sum_profiles: BandProfiles | None = None
        self._roi_mean_profiles: BandProfiles | None = None
        self._current_profiles: BandProfiles | None = None
        self._mask_inclusion = None
        self._mask_pending = False
        self._profile_pending = False
        self._mask_description = "No exclusions; native masks bind this acquisition/source"
        self.mask_undo_button.setEnabled(False)
        self.mask_redo_button.setEnabled(False)
        self._roi_bounds: tuple[int, int, int, int] | None = None
        self.acquisition_identity: object = None
        self.observable_unit = "counts"
        self.quantitative_ready = True
        self.profile_work_required = False
        self.coalesce_profile_updates = False
        self._profile_refresh_queued = False
        self._profile_axis_rect: QRectF | None = None
        self._sync_controls()
        self._marker_count_changed(0)

    def _mask_mode_changed(self, mode: str) -> None:
        self.mask_tool.blockSignals(True)
        self.mask_tool.setCurrentIndex(self.mask_tool.findData(mode))
        self.mask_tool.blockSignals(False)
        self.mask_hint.setText(
            {
                "inspect": "Inspect: pointer profiles",
                "rectangle": "Drag rectangle; Esc cancels",
                "polygon": "Click vertices; Enter commits; Esc cancels",
                "brush": "Drag brush; Esc cancels",
            }[mode]
        )

    def _mask_visibility_changed(self, visible: bool) -> None:
        self.view.show_mask = visible
        self.view._request_paint()
        self.view.view_state_changed.emit()

    def profile_query(self) -> tuple:
        return (
            *self.view.crosshair,
            self.row_width_control.value(),
            self.column_width_control.value(),
            self.profile_scope_control.currentData(),
            self._roi_bounds,
            self.profile_measure_control.currentData(),
        )

    def set_mask_pending(self, pending: bool) -> None:
        self._mask_pending = pending
        self.mask_preparation_status.setText(
            f"Preparing profiles · showing prior mask revision {self.view.mask_revision}"
            if pending
            else self._mask_description
        )
        self.export_button.setEnabled(
            self.view.image is not None and not pending and not self._profile_pending
        )
        if self._current_profiles is not None:
            self._update_profile_status(self._current_profiles)

    def publish_mask(self, prepared) -> None:
        self._mask_inclusion = prepared.inclusion
        if (
            self.view.mask_reasons is not prepared.reasons
            or self.view.mask_revision != prepared.mask.revision
        ):
            self.view.set_mask_plane(
                prepared.reasons,
                prepared.mask.revision,
                base_reasons_id=prepared.base_reasons_id,
                changed_bounds=prepared.changed_bounds,
            )
        self._mask_pending = self._profile_pending = False
        self._full_profiles = prepared.full_profiles
        provenance = prepared.mask.provenance[-1] if prepared.mask.provenance else "No exclusions"
        self._mask_description = (
            f"Mask revision {prepared.mask.revision} · {provenance.split(';')[0][:70]}"
        )
        self.mask_preparation_status.setText(self._mask_description)
        self.mask_preparation_status.setToolTip("\n".join(prepared.mask.provenance))
        self._full_mean_profiles = self._roi_sum_profiles = self._roi_mean_profiles = None
        self._roi_cache_key = self._profile_key = None
        if prepared.query == self.profile_query():
            self._present_profiles(prepared.profiles, prepared.query[4], self._query_key())
        else:
            self._refresh_profile_if_needed()

    def _set_box_zoom(self, enabled: bool) -> None:
        if enabled:
            self.view.set_mask_mode("inspect")
        if enabled and self.draw_roi_button.isChecked():
            self.draw_roi_button.setChecked(False)
        self.view.box_zoom_enabled = enabled
        self.view.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def _set_roi_selection(self, enabled: bool) -> None:
        if enabled:
            self.view.set_mask_mode("inspect")
        if enabled and self.box_button.isChecked():
            self.box_button.setChecked(False)
        self.view.roi_select_enabled = enabled
        self.view.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def _set_pin_center(self, pinned: bool) -> None:
        self.view.follow_cursor = not pinned
        self.pin_center_button.setText("Pinned center" if pinned else "Follow pointer")
        self.view.view_state_changed.emit()

    def _sync_profile_position(self) -> None:
        if self.view.image is None:
            return
        for control, value in zip(
            (self.column_control, self.row_control), self.view.crosshair, strict=True
        ):
            if control.value() != value:
                control.blockSignals(True)
                control.setValue(value)
                control.blockSignals(False)

    def _center_controls_changed(self) -> None:
        if self.view.image is None:
            return
        center = (self.column_control.value(), self.row_control.value())
        if center == self.view.crosshair:
            return
        if not self.pin_center_button.isChecked():
            self.pin_center_button.setChecked(True)
        self.view.crosshair = center
        self.view._request_paint()
        self.view.crosshair_changed.emit()

    def _query_changed(self) -> None:
        if self.view.image is None:
            return
        self._refresh_profile_if_needed()
        self.view.view_state_changed.emit()

    def _drag_band_edge(self, edge: str, index: int) -> None:
        if self.view.image is None or self.profile_scope_control.currentData() != "band":
            return
        current = (
            self.view.profile_row_bounds
            if edge.startswith("row")
            else self.view.profile_column_bounds
        )
        start, stop = current
        if edge.endswith("start"):
            start = min(index, stop - 1)
        else:
            stop = max(index, start + 1)
        width = stop - start
        center = start + (width - 1) // 2
        vertical = edge.startswith("row")
        width_control = self.row_width_control if vertical else self.column_width_control
        width_control.blockSignals(True)
        width_control.setValue(width)
        width_control.blockSignals(False)
        column, row = self.view.crosshair
        self.view.crosshair = (column, center) if vertical else (center, row)
        self._sync_profile_position()
        if not self.pin_center_button.isChecked():
            self.pin_center_button.setChecked(True)
        self.view._request_paint()
        self._query_changed()

    def _apply_roi(self, bounds: object) -> None:
        if (
            self.view.image is None
            or type(bounds) is not tuple
            or len(bounds) != 4
            or any(type(bound) is not int for bound in bounds)
        ):
            raise ValueError("inspection ROI needs four integer native bounds")
        c0, c1, r0, r1 = bounds
        rows, columns = self.view.image.shape
        if not (0 <= c0 < c1 <= columns and 0 <= r0 < r1 <= rows):
            raise ValueError("inspection ROI is outside this detector")
        self._roi_bounds = bounds
        self.profile_scope_control.model().item(2).setEnabled(True)
        already_selected = self.profile_scope_control.currentIndex() == 2
        self.profile_scope_control.setCurrentIndex(2)
        self.draw_roi_button.setChecked(False)
        if already_selected:
            self._query_changed()

    def _apply_profile_scales(self) -> None:
        def limits(checked: bool, low_text: str, high_text: str) -> tuple[float, float] | None:
            if not checked:
                return None
            low, high = float(low_text), float(high_text)
            if (
                not math.isfinite(low)
                or not math.isfinite(high)
                or not low < high
                or not math.isfinite(high - low)
            ):
                raise ValueError("pinned profile limits need finite low < high")
            return low, high

        try:
            horizontal = limits(
                self.horizontal_pin_scale.isChecked(),
                self.horizontal_scale_low.text(),
                self.horizontal_scale_high.text(),
            )
            vertical = limits(
                self.vertical_pin_scale.isChecked(),
                self.vertical_scale_low.text(),
                self.vertical_scale_high.text(),
            )
        except (ValueError, OverflowError) as exc:
            for checkbox, low_entry, high_entry, plot in (
                (
                    self.horizontal_pin_scale,
                    self.horizontal_scale_low,
                    self.horizontal_scale_high,
                    self.horizontal,
                ),
                (
                    self.vertical_pin_scale,
                    self.vertical_scale_low,
                    self.vertical_scale_high,
                    self.vertical,
                ),
            ):
                applied = plot.intensity_limits
                checkbox.blockSignals(True)
                checkbox.setChecked(applied is not None)
                checkbox.blockSignals(False)
                if applied is not None:
                    low_entry.setText(format(applied[0], ".17g"))
                    high_entry.setText(format(applied[1], ".17g"))
            self.profile_status.setText(str(exc))
            return
        if (
            self.horizontal.intensity_limits != horizontal
            or self.vertical.intensity_limits != vertical
        ):
            self.view._request_paint()
        self.horizontal.set_intensity_limits(horizontal)
        self.vertical.set_intensity_limits(vertical)
        if self._current_profiles is not None:
            self._update_profile_status(self._current_profiles)
        self.view.view_state_changed.emit()

    def _sync_controls(self) -> None:
        view = self.view
        mode_index = self.mode_control.findData(view.contrast_mode)
        if self.mode_control.currentIndex() != mode_index:
            self.mode_control.blockSignals(True)
            self.mode_control.setCurrentIndex(mode_index)
            self.mode_control.blockSignals(False)
        low_text, high_text = format(view.low_value, ".17g"), format(view.high_value, ".17g")
        if self.low_control.text() != low_text:
            self.low_control.setText(low_text)
        if self.high_control.text() != high_text:
            self.high_control.setText(high_text)
        for control, state in (
            (self.image_layer, view.show_image),
            (self.crosshair_layer, view.show_crosshair),
            (self.marker_layer, view.show_markers),
        ):
            if control.isChecked() != state:
                control.blockSignals(True)
                control.setChecked(state)
                control.blockSignals(False)

    def _mode_changed(self) -> None:
        try:
            self.view.auto_levels(self.mode_control.currentData())
        except ValueError as exc:
            self.contrast_note.setText(str(exc))
            self._sync_controls()
        else:
            self.contrast_note.setText(
                "Contrast clips colors only; checkerboard marks invalid or hidden log pixels."
            )

    def _apply_contrast(self) -> None:
        try:
            self.view.set_levels(
                float(self.low_control.text()),
                float(self.high_control.text()),
                mode=self.mode_control.currentData(),
            )
        except ValueError as exc:
            self.contrast_note.setText(str(exc))
        else:
            self.contrast_note.setText(
                "Contrast clips colors only; checkerboard marks invalid or hidden log pixels."
            )

    def _auto_contrast(self) -> None:
        self._mode_changed()

    def _apply_layers(self) -> None:
        self.view.set_layer_visibility(
            image=self.image_layer.isChecked(),
            crosshair=self.crosshair_layer.isChecked(),
            markers=self.marker_layer.isChecked(),
        )

    def _marker_count_changed(self, count: int) -> None:
        self.marker_layer.setText(f"Markers ({count})")
        self.marker_layer.setEnabled(count > 0)

    def _show_cursor(self, pixel: object) -> None:
        if pixel is None:
            self.cursor_label.setText(
                "Pointer outside detector · Q/angles unavailable · saturation unknown"
            )
        elif not self.quantitative_ready:
            column, row, _ = pixel
            self.cursor_label.setText(
                f"{self.view.column_axis_label}={column}, {self.view.row_axis_label}={row}: Awaiting quantitative snapshot"
            )
        else:
            column, row, intensity = pixel
            reason = (
                0 if self.view.mask_reasons is None else int(self.view.mask_reasons[row, column])
            )
            self.cursor_label.setText(
                f"{self.view.column_axis_label}={column}, {self.view.row_axis_label}={row}, intensity={intensity!r} {self.observable_unit}"
                f" · {REASONS[reason]} · Q/angles unavailable"
            )

    def _reset_profile_state(self, acquisition_identity: object = None) -> None:
        self._mask_description = "No exclusions; native masks bind this acquisition/source"
        self.mask_preparation_status.setText(self._mask_description)
        self.mask_preparation_status.setToolTip("")
        self._mask_inclusion = None
        self._mask_pending = self._profile_pending = False
        self.acquisition_identity = acquisition_identity
        self._profile_key = None
        self._full_profiles = None
        self._full_mean_profiles = None
        self._roi_cache_key = None
        self._roi_sum_profiles = None
        self._roi_mean_profiles = None
        self._current_profiles = None
        self._roi_bounds = None
        self.view.follow_cursor = True
        self.view.roi_select_enabled = False
        self.view.profile_scope = "band"
        self.profile_scope_control.model().item(2).setEnabled(False)
        for control, value in (
            (self.pin_center_button, False),
            (self.draw_roi_button, False),
            (self.horizontal_pin_scale, False),
            (self.vertical_pin_scale, False),
        ):
            control.blockSignals(True)
            control.setChecked(value)
            control.blockSignals(False)
        self.pin_center_button.setText("Follow pointer")
        for control, index in (
            (self.profile_measure_control, 0),
            (self.profile_scope_control, 0),
        ):
            control.blockSignals(True)
            control.setCurrentIndex(index)
            control.blockSignals(False)
        rows, columns = self.view.image.shape
        for control, maximum, value in (
            (self.column_control, columns - 1, self.view.crosshair[0]),
            (self.row_control, rows - 1, self.view.crosshair[1]),
            (self.row_width_control, 16_384, 1),
            (self.column_width_control, 16_384, 1),
        ):
            control.blockSignals(True)
            control.setMaximum(maximum)
            control.setValue(value)
            control.blockSignals(False)
        self.horizontal.set_intensity_limits(None)
        self.vertical.set_intensity_limits(None)
        self.profile_status.setText("Bands: 1 row x 1 column · sum")

    def restore_profile_state(self, state: DetectorViewState) -> None:
        """Apply a validated saved inspection query before publishing its profiles."""

        self.mask_layer.setChecked(state.show_mask)
        if self.view.image is None:
            raise ValueError("cannot restore profiles before the native image")
        rows, columns = self.view.image.shape
        if state.profile_roi is not None:
            _, c1, _, r1 = state.profile_roi
            if c1 > columns or r1 > rows:
                raise ValueError("saved inspection ROI is outside this detector")
        self._roi_bounds = state.profile_roi
        self.profile_scope_control.model().item(2).setEnabled(self._roi_bounds is not None)
        for control, value in (
            (self.row_width_control, state.profile_row_width),
            (self.column_width_control, state.profile_column_width),
        ):
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        for control, value in (
            (
                self.profile_measure_control,
                self.profile_measure_control.findData(state.profile_measure),
            ),
            (self.profile_scope_control, self.profile_scope_control.findData(state.profile_scope)),
        ):
            control.blockSignals(True)
            control.setCurrentIndex(value)
            control.blockSignals(False)
        self.pin_center_button.blockSignals(True)
        self.pin_center_button.setChecked(not state.profile_follow)
        self.pin_center_button.blockSignals(False)
        self.pin_center_button.setText(
            "Follow pointer" if state.profile_follow else "Pinned center"
        )
        self.view.follow_cursor = state.profile_follow
        for checkbox, low_entry, high_entry, limits, plot in (
            (
                self.horizontal_pin_scale,
                self.horizontal_scale_low,
                self.horizontal_scale_high,
                state.horizontal_intensity_limits,
                self.horizontal,
            ),
            (
                self.vertical_pin_scale,
                self.vertical_scale_low,
                self.vertical_scale_high,
                state.vertical_intensity_limits,
                self.vertical,
            ),
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(limits is not None)
            checkbox.blockSignals(False)
            if limits is not None:
                low_entry.setText(format(limits[0], ".17g"))
                high_entry.setText(format(limits[1], ".17g"))
            plot.set_intensity_limits(limits)
        self._sync_profile_position()
        self._profile_key = None
        self._refresh_profile_if_needed()

    def _query_key(self) -> tuple[object, ...]:
        scope = self.profile_scope_control.currentData()
        band_bounds = None
        if scope == "band":
            rows, columns = self.view.image.shape
            band_bounds = _effective_band_bounds(
                rows,
                columns,
                *self.view.crosshair,
                self.row_width_control.value(),
                self.column_width_control.value(),
            )
        return (
            self.acquisition_identity,
            id(self.view.image),
            self.view.data_revision,
            scope,
            self.profile_measure_control.currentData(),
            band_bounds,
            self._roi_bounds if scope == "roi" else None,
            self.view.mask_revision,
        )

    def _update_profile_status(self, profiles: BandProfiles) -> None:
        column, row = self.view.crosshair
        h_support = int(profiles.horizontal_support[column])
        v_support = int(profiles.vertical_support[row])
        r0, r1 = profiles.row_bounds
        c0, c1 = profiles.column_bounds
        scope = self.profile_scope_control.currentData()
        measure = self.profile_measure_control.currentData()
        self.profile_status.setText(
            f"{scope} · {measure} · rows [{r0},{r1}) · columns [{c0},{c1})"
            f" · valid support H@column={h_support}, V@row={v_support}"
            " · zero support is missing"
            + (
                " · Preparing profiles; displayed revision is old"
                if self._mask_pending or self._profile_pending
                else f" · mask revision {self.view.mask_revision}"
            )
        )

    def _present_profiles(
        self, profiles: BandProfiles, scope: str, key: tuple[object, ...]
    ) -> None:
        prior_generation = self.view.request_generation
        self.view.set_profile_overlay(
            scope=scope,
            row_bounds=profiles.row_bounds,
            column_bounds=profiles.column_bounds,
            roi=self._roi_bounds,
        )
        if self.view.request_generation == prior_generation:
            self.view._request_paint()
        self.horizontal.set_values(profiles.horizontal, profiles.horizontal_support)
        self.vertical.set_values(profiles.vertical, profiles.vertical_support)
        self._current_profiles = profiles
        self._profile_key = key
        self._update_profile_status(profiles)

    def set_image(self, image: NDArray[np.generic]) -> None:
        supplied = np.asarray(image)
        if supplied.ndim != 2 or not all(supplied.shape):
            raise ValueError("detector panels require a nonempty native plane")
        exact_band_profiles(
            supplied, column_px=supplied.shape[1] // 2, row_px=supplied.shape[0] // 2
        )
        self.view.set_image(image)
        self.export_button.setEnabled(True)
        self._reset_profile_state()
        self._profile_axis_rect = None
        self._sync_controls()
        self._refresh_profile_if_needed()

    def set_prepared_image(
        self,
        native_counts: NDArray[np.int32],
        display: NDArray[np.float32],
        profiles: BandProfiles,
        full_profiles: BandProfiles,
        low_value: float,
        high_value: float,
        max_value: float,
        min_positive: float | None = None,
        acquisition_identity: object = None,
    ) -> None:
        """Publish a worker-prepared OSC result with its exact center bands."""

        rows, columns = native_counts.shape
        if (
            full_profiles.row_bounds != (0, rows)
            or full_profiles.column_bounds != (0, columns)
            or full_profiles.horizontal.shape != (columns,)
            or full_profiles.vertical.shape != (rows,)
        ):
            raise ValueError("prepared full marginals do not cover the native detector")
        self.view.set_prepared_image(
            native_counts, display, low_value, high_value, max_value, min_positive
        )
        self.export_button.setEnabled(True)
        self._reset_profile_state(acquisition_identity)
        self._profile_axis_rect = None
        self._sync_controls()
        self._full_profiles = full_profiles
        self._present_profiles(profiles, "band", self._query_key())

    def capture_inspection_figure(self, title: str) -> QImage:
        """Capture the current detector viewport and its two visible profile axes."""

        if self.view.image is None or not self.view.isVisible():
            raise ValueError("no visible detector image is ready for export")
        dpr = self.view.devicePixelRatioF()
        width = math.ceil((self.view.width() + self.vertical.width()) * dpr)
        height = math.ceil((self.horizontal.height() + self.view.height() + 32) * dpr)
        if width <= 0 or height <= 0 or width * height * 4 > 16 * 1024 * 1024:
            raise ValueError("figure capture exceeds the 16 MiB display buffer limit")
        detector = self.view.grabFramebuffer()
        horizontal = self.horizontal.grab().toImage()
        vertical = self.vertical.grab().toImage()
        if any(image.isNull() for image in (detector, horizontal, vertical)):
            raise ValueError("a detector or profile layer could not be captured")
        figure_width = max(detector.width(), horizontal.width()) + vertical.width()
        figure_height = horizontal.height() + max(detector.height(), vertical.height())
        header = math.ceil(32 * dpr)
        if figure_width * (figure_height + header) * 4 > 16 * 1024 * 1024:
            raise ValueError("captured figure exceeds the 16 MiB display buffer limit")
        figure = QImage(
            figure_width, figure_height + header, QImage.Format.Format_ARGB32_Premultiplied
        )
        figure.fill(QColor(24, 28, 34))
        painter = QPainter(figure)
        painter.setPen(QColor(225, 237, 241))
        font = painter.font()
        font.setPixelSize(max(12, round(14 * dpr)))
        painter.setFont(font)
        painter.drawText(
            QRectF(8 * dpr, 0, figure_width - 16 * dpr, header),
            Qt.AlignmentFlag.AlignVCenter,
            title,
        )
        for image, x, y in (
            (horizontal, 0, header),
            (detector, 0, header + horizontal.height()),
            (vertical, max(detector.width(), horizontal.width()), header + horizontal.height()),
        ):
            painter.drawImage(
                QRectF(x, y, image.width(), image.height()),
                image,
                QRectF(0, 0, image.width(), image.height()),
            )
        painter.end()
        figure.setDevicePixelRatio(dpr)
        return figure

    def _refresh_axes(self, _generation: int, _timestamp: float) -> None:
        if self.view.image is None:
            return
        current = self.view._rect()
        if self._profile_axis_rect == current:
            return
        self._profile_axis_rect = QRectF(current)
        self.horizontal.update()
        self.vertical.update()

    def _flush_profile_update(self) -> None:
        self._profile_refresh_queued = False
        self._profile_pending = False
        self._refresh_profile_if_needed(immediate=True)

    def _refresh_profile_if_needed(self, *, immediate: bool = False) -> None:
        if self.coalesce_profile_updates and not immediate:
            if not self._profile_refresh_queued:
                self._profile_refresh_queued = True
                self._profile_pending = True
                if self._current_profiles is not None:
                    self._update_profile_status(self._current_profiles)
                QTimer.singleShot(0, self._flush_profile_update)
            return
        if self.view.image is None:
            return
        if not self.quantitative_ready:
            self.profile_status.setText(
                "Awaiting quantitative snapshot; retained profiles are historical"
            )
            return
        key = self._query_key()
        if self._profile_key == key and self._current_profiles is not None:
            self._update_profile_status(self._current_profiles)
            return
        scope = self.profile_scope_control.currentData()
        measure = self.profile_measure_control.currentData()
        rows, columns = self.view.image.shape
        bounds = key[5]
        direct_pixels = (
            rows * columns
            if scope != "band"
            else (bounds[0][1] - bounds[0][0]) * columns + (bounds[1][1] - bounds[1][0]) * rows
        )
        if self._mask_pending and direct_pixels > 131072:
            self._update_profile_status(self._current_profiles)
            return
        if (
            (self._mask_inclusion is not None or self.profile_work_required)
            and direct_pixels > 131072
            and scope != "full"
            and not (scope == "band" and bounds == ((0, rows), (0, columns)))
            and not (
                scope == "roi"
                and self._roi_cache_key
                == (
                    self.acquisition_identity,
                    id(self.view.image),
                    self.view.data_revision,
                    self._roi_bounds,
                )
            )
        ):
            self._profile_pending = True
            self.mask_preparation_status.setText(
                f"Preparing profiles · mask revision {self.view.mask_revision}; prior bands shown"
            )
            self._update_profile_status(self._current_profiles)
            self.profile_requested.emit()
            return
        covers_full_detector = scope == "full" or (
            scope == "band" and key[5] == ((0, rows), (0, columns))
        )
        if covers_full_detector:
            if self._full_profiles is None:
                self._full_profiles = exact_band_profiles(
                    self.view.image,
                    column_px=self.view.crosshair[0],
                    row_px=self.view.crosshair[1],
                    scope="full",
                    mask=self._mask_inclusion,
                )
            if measure == "mean":
                if self._full_mean_profiles is None:
                    sums = self._full_profiles
                    self._full_mean_profiles = BandProfiles(
                        _mean_per_valid(sums.horizontal, sums.horizontal_support),
                        _mean_per_valid(sums.vertical, sums.vertical_support),
                        sums.horizontal_support,
                        sums.vertical_support,
                        sums.row_bounds,
                        sums.column_bounds,
                    )
                profiles = self._full_mean_profiles
            else:
                profiles = self._full_profiles
        elif scope == "roi":
            roi_key = (
                self.acquisition_identity,
                id(self.view.image),
                self.view.data_revision,
                self._roi_bounds,
            )
            if roi_key != self._roi_cache_key:
                sums = exact_band_profiles(
                    self.view.image,
                    column_px=self.view.crosshair[0],
                    row_px=self.view.crosshair[1],
                    scope="roi",
                    roi_column_row_bounds=self._roi_bounds,
                    mask=self._mask_inclusion,
                )
                self._roi_cache_key = roi_key
                self._roi_sum_profiles = sums
                self._roi_mean_profiles = None
            if measure == "mean":
                if self._roi_mean_profiles is None:
                    sums = self._roi_sum_profiles
                    self._roi_mean_profiles = BandProfiles(
                        _mean_per_valid(sums.horizontal, sums.horizontal_support),
                        _mean_per_valid(sums.vertical, sums.vertical_support),
                        sums.horizontal_support,
                        sums.vertical_support,
                        sums.row_bounds,
                        sums.column_bounds,
                    )
                profiles = self._roi_mean_profiles
            else:
                profiles = self._roi_sum_profiles
        else:
            profiles = exact_band_profiles(
                self.view.image,
                column_px=self.view.crosshair[0],
                row_px=self.view.crosshair[1],
                row_width=self.row_width_control.value(),
                column_width=self.column_width_control.value(),
                scope=scope,
                roi_column_row_bounds=self._roi_bounds if scope == "roi" else None,
                measure=measure,
                mask=self._mask_inclusion,
            )
        self._present_profiles(profiles, scope, key)
