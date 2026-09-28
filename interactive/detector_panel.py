"""Native detector presentation primitives for the desktop application.

The texture is display only. Profiles always reduce the owned native data plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from project_state import FLOAT32_DISPLAY_MAX, linear_display_limits, validate_display_limits
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QPolygonF, QShortcut, QTransform
from PySide6.QtOpenGL import (
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


def exact_band_profiles(
    native_image: NDArray[np.generic],
    *,
    column_px: int,
    row_px: int,
    row_width: int = 1,
    column_width: int = 1,
    mask: NDArray[np.bool] | None = None,
) -> BandProfiles:
    """Sum native pixel values and valid support over two clipped bands."""

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
    row_start = row_px - (row_width - 1) // 2
    column_start = column_px - (column_width - 1) // 2
    r0, r1 = max(0, row_start), min(rows, row_start + row_width)
    c0, c1 = max(0, column_start), min(columns, column_start + column_width)
    if mask is not None and (mask.dtype != np.bool_ or mask.shape != image.shape):
        raise ValueError("mask must be a Boolean native-detector plane")
    integer = image.dtype.kind in "iu"
    if integer:
        limits = np.iinfo(image.dtype)
        maximum_band = max(r1 - r0, c1 - c0)
        if max(abs(int(limits.min)), int(limits.max)) * maximum_band > np.iinfo(np.int64).max:
            raise ValueError("integer band may overflow int64 exact accumulation")
    accumulator = np.int64 if integer else np.float64

    def reduce_band(values: NDArray[np.generic], included: NDArray[np.bool], axis: int):
        valid = included & np.isfinite(values)
        safe = np.where(valid, values, 0)
        return (
            np.sum(safe, axis=axis, dtype=accumulator),
            np.sum(valid, axis=axis, dtype=np.int64),
        )

    horizontal_values = image[r0:r1]
    vertical_values = image[:, c0:c1]
    horizontal_mask = (
        np.ones(horizontal_values.shape, dtype=np.bool_) if mask is None else mask[r0:r1]
    )
    vertical_mask = (
        np.ones(vertical_values.shape, dtype=np.bool_) if mask is None else mask[:, c0:c1]
    )
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
_FRAGMENT = """#version 330 core
in vec2 uv;
out vec4 color;
uniform sampler2D detector;
uniform float low_value;
uniform float high_value;
uniform int contrast_mode;
void main() {
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
"""
_GL_MAX_TEXTURE_SIZE = 0x0D33


class DetectorTextureView(QOpenGLWidget):
    """One persistent detector texture with native coordinate overlays."""

    painted = Signal(int, float)
    crosshair_changed = Signal()
    view_state_changed = Signal()
    cursor_changed = Signal(object)
    overlays_changed = Signal(int)

    def __init__(self, *, plane: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plane = plane
        self.image: NDArray[np.generic] | None = None
        self._display: NDArray[np.float32] | None = None
        self._texture: QOpenGLTexture | None = None
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
        self.crosshair = (0, 0)
        self.overlays = np.empty((0, 2), dtype=np.float64)
        self._overlay_points = QPolygonF()
        self.setMinimumSize(180, 180)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _request_paint(self) -> None:
        self.request_generation += 1
        self.update()

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
        self.data_min, self.data_max, self.min_positive = data_min, data_max, min_positive
        self.low_value, self.high_value = low, high
        self.contrast_mode = "linear"
        self.zoom, self.pan, self.scale_mode = 1.0, QPointF(), "fit"
        self._pan_dpr = self.devicePixelRatioF()
        self.crosshair = (supplied.shape[1] // 2, supplied.shape[0] // 2)
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
            or native_counts.dtype != np.int32
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
        self.data_min, self.data_max, self.min_positive = low_value, max_value, min_positive
        self.low_value, self.high_value, self.contrast_mode = low_value, high_value, "linear"
        self.zoom, self.pan, self.scale_mode = 1.0, QPointF(), "fit"
        self._pan_dpr = self.devicePixelRatioF()
        self.crosshair = (native_counts.shape[1] // 2, native_counts.shape[0] // 2)
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
        program = QOpenGLShaderProgram(self)
        for kind, source in (
            (QOpenGLShader.ShaderTypeBit.Vertex, _VERTEX),
            (QOpenGLShader.ShaderTypeBit.Fragment, _FRAGMENT),
        ):
            if not program.addShaderFromSourceCode(kind, source):
                raise RuntimeError(program.log())
        if not program.link():
            raise RuntimeError(program.log())
        vao = QOpenGLVertexArrayObject(self)
        if not vao.create():
            raise RuntimeError("OpenGL vertex array creation failed")
        self._program, self._vao = program, vao
        self._uploaded_revision = -1
        self.context().aboutToBeDestroyed.connect(self.release_resources)

    def release_resources(self) -> None:
        if self._texture is None and self._vao is None:
            return
        self.makeCurrent()
        try:
            if self._texture is not None:
                self._texture.destroy()
            if self._vao is not None:
                self._vao.destroy()
        finally:
            self._texture = None
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

    def paintGL(self) -> None:
        functions = self.context().functions()
        functions.glClearColor(0.08, 0.09, 0.11, 1.0)
        functions.glClear(0x00004000)
        if self.image is None or self._program is None or self._vao is None:
            self.painted.emit(self.request_generation, perf_counter())
            return
        self._upload_if_needed()
        rect = self._rect() if not self.plane else QRectF(0.1, 0.1, 0.8, 0.8)
        if self.show_image:
            self._program.bind()
            self._vao.bind()
            assert self._texture is not None
            self._texture.bind(0)
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
            self._vao.release()
            self._program.release()
        if not self.plane:
            painter = QPainter(self)
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
            painter.end()
        self.painted.emit(self.request_generation, perf_counter())

    def wheelEvent(self, event) -> None:
        if self.image is not None and event.angleDelta().y():
            self.set_zoom_at(self.zoom * 1.2 ** (event.angleDelta().y() / 120), event.position())
            event.accept()

    def mousePressEvent(self, event) -> None:
        self._last_pointer = event.position()
        if (
            self.image is not None
            and self.box_zoom_enabled
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._box_start = event.position()
            self._box_end = event.position()
            self._request_paint()

    def mouseMoveEvent(self, event) -> None:
        if self._box_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._box_end = event.position()
            self._request_paint()
        elif self._last_pointer is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._last_pointer
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
                if not event.buttons() and native != self.crosshair:
                    self.crosshair = native
                    self._request_paint()
                    self.crosshair_changed.emit()
        self._last_pointer = event.position()

    def mouseReleaseEvent(self, event) -> None:
        if self._box_start is not None and self.image is not None:
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
        self._last_pointer = None

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
        self.setMinimumSize(50 if vertical else 180, 50)

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
            minimum = min(float(np.min(selected, initial=0)), 0.0)
            maximum = max(float(np.max(selected, initial=0)), 1.0)
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
                normalized = (amplitudes - minimum) / span
                if self.vertical:
                    points = QPolygonF(
                        [
                            QPointF(float(value) * (self.width() - 1), float(position))
                            for position, value in zip(positions, normalized, strict=True)
                        ]
                    )
                else:
                    points = QPolygonF(
                        [
                            QPointF(float(position), (1 - float(value)) * (self.height() - 1))
                            for position, value in zip(positions, normalized, strict=True)
                        ]
                    )
                if len(points) == 1:
                    painter.drawPoint(points[0])
                else:
                    painter.drawPolyline(points)
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
            painter.drawText(4, 12, "row_px" if self.vertical else "column_px")
            for index in range((first_visible + step - 1) // step * step, last_visible + 1, step):
                position = self.source_view.axis_position(index, vertical=self.vertical)
                if not 0 <= position < limit:
                    continue
                if self.vertical:
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
        self.painted.emit(self.generation, perf_counter())


class DetectorPanel(QWidget):
    """A reusable native image with exact linked horizontal and vertical bands."""

    def __init__(self, parent: QWidget | None = None) -> None:
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
        for button in (self.fit_button, self.native_button, self.box_button):
            nav_layout.addWidget(button)
        nav_layout.addWidget(QLabel("Drag: pan · Wheel: zoom · Arrows: pan"), 1)

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
        layout.addWidget(self.horizontal, 2, 0)
        layout.addWidget(self.view, 3, 0)
        layout.addWidget(self.vertical, 3, 1)
        layout.addWidget(self.cursor_label, 4, 0, 1, 2)
        layout.addWidget(self.contrast_note, 5, 0, 1, 2)
        layout.addWidget(layers, 6, 0, 1, 2)
        layout.setRowStretch(3, 1)
        layout.setColumnStretch(0, 1)
        self.view.crosshair_changed.connect(self._refresh_profile_if_needed)
        self.view.painted.connect(self._refresh_axes)
        self.view.view_state_changed.connect(self._sync_controls)
        self.view.cursor_changed.connect(self._show_cursor)
        self.view.overlays_changed.connect(self._marker_count_changed)
        self.fit_button.clicked.connect(self.view.fit_image)
        self.native_button.clicked.connect(self.view.native_pixel_scale)
        self.box_button.toggled.connect(self._set_box_zoom)
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
        self._profile_crosshair: tuple[int, int] | None = None
        self._profile_axis_rect: QRectF | None = None
        self._sync_controls()
        self._marker_count_changed(0)

    def _set_box_zoom(self, enabled: bool) -> None:
        self.view.box_zoom_enabled = enabled
        self.view.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

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
        else:
            column, row, intensity = pixel
            self.cursor_label.setText(
                f"column_px={column}, row_px={row}, intensity={intensity!r} counts"
                " · Q/angles unavailable · saturation unknown"
            )

    def set_image(self, image: NDArray[np.generic]) -> None:
        supplied = np.asarray(image)
        if supplied.ndim != 2 or not all(supplied.shape):
            raise ValueError("detector panels require a nonempty native plane")
        exact_band_profiles(
            supplied, column_px=supplied.shape[1] // 2, row_px=supplied.shape[0] // 2
        )
        self.view.set_image(image)
        self._profile_axis_rect = None
        self._sync_controls()
        self._profile_crosshair = None
        self._refresh_profile_if_needed()

    def set_prepared_image(
        self,
        native_counts: NDArray[np.int32],
        display: NDArray[np.float32],
        profiles: BandProfiles,
        low_value: float,
        high_value: float,
        max_value: float,
        min_positive: float | None = None,
    ) -> None:
        """Publish a worker-prepared OSC result with its exact center bands."""

        self.view.set_prepared_image(
            native_counts, display, low_value, high_value, max_value, min_positive
        )
        self._profile_axis_rect = None
        self._sync_controls()
        self.horizontal.set_values(profiles.horizontal, profiles.horizontal_support)
        self.vertical.set_values(profiles.vertical, profiles.vertical_support)
        self._profile_crosshair = self.view.crosshair

    def _refresh_axes(self, _generation: int, _timestamp: float) -> None:
        if self.view.image is None:
            return
        current = self.view._rect()
        if self._profile_axis_rect == current:
            return
        self._profile_axis_rect = QRectF(current)
        self.horizontal.update()
        self.vertical.update()

    def _refresh_profile_if_needed(self) -> None:
        if self.view.image is None or self._profile_crosshair == self.view.crosshair:
            return
        profiles = exact_band_profiles(
            self.view.image, column_px=self.view.crosshair[0], row_px=self.view.crosshair[1]
        )
        self.horizontal.set_values(profiles.horizontal, profiles.horizontal_support)
        self.vertical.set_values(profiles.vertical, profiles.vertical_support)
        self._profile_crosshair = self.view.crosshair
