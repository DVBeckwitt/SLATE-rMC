"""Native detector presentation primitives for the desktop application.

The texture is display only. Profiles always reduce the owned native data plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QTransform
from PySide6.QtOpenGL import (
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QGridLayout, QWidget
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
uniform int positive_log;
void main() {
    float value = texture(detector, uv).r;
    if (isnan(value) || isinf(value) || (positive_log == 1 && value <= 0.0)) {
        color = vec4(0.08, 0.09, 0.11, 1.0);
        return;
    }
    float level = positive_log == 1
        ? (log(value) - log(low_value)) / max(log(high_value / low_value), 1e-12)
        : (value - low_value) / max(high_value - low_value, 1e-12);
    level = clamp(level, 0.0, 1.0);
    color = vec4(level, level * 0.82 + 0.08, 0.12 + level * 0.68, 1.0);
}
"""


class DetectorTextureView(QOpenGLWidget):
    """One persistent detector texture with native coordinate overlays."""

    painted = Signal(int, float)
    crosshair_changed = Signal()

    def __init__(self, *, plane: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plane = plane
        self.image: NDArray[np.generic] | None = None
        self._display: NDArray[np.float32] | None = None
        self._texture: QOpenGLTexture | None = None
        self._program: QOpenGLShaderProgram | None = None
        self._vao: QOpenGLVertexArrayObject | None = None
        self._uploaded_revision = -1
        self.data_revision = 0
        self.request_generation = 0
        self.upload_count = 0
        self.low_value = 1.0
        self.high_value = 100.0
        self.positive_log = False
        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)
        self.yaw = 0.35
        self.pitch = 0.45
        self._last_pointer: QPointF | None = None
        self.crosshair = (0, 0)
        self.overlays = np.empty((0, 2), dtype=np.float64)
        self._overlay_points = QPolygonF()
        self.setMinimumSize(180, 180)
        self.setMouseTracking(True)

    def _request_paint(self) -> None:
        self.request_generation += 1
        self.update()

    def set_image(self, image: NDArray[np.generic]) -> None:
        supplied = np.asarray(image)
        if supplied.ndim != 2 or supplied.dtype.kind not in "iuf" or not supplied.size:
            raise ValueError("image must be a nonempty real native-detector plane")
        self.image = np.array(supplied, copy=True, order="C")
        self.image.setflags(write=False)
        self._display = np.ascontiguousarray(self.image, dtype=np.float32)
        finite = self._display[np.isfinite(self._display)]
        if self.positive_log:
            positive = finite[finite > 0]
            if positive.size:
                self.low_value = float(np.min(positive))
                self.high_value = max(float(np.max(positive)), self.low_value * 1.001)
            else:
                self.low_value, self.high_value = 1.0e-8, 1.0
        elif finite.size:
            self.low_value = float(np.min(finite))
            self.high_value = max(float(np.max(finite)), self.low_value + 1.0)
        else:
            self.low_value, self.high_value = 0.0, 1.0
        self.crosshair = (supplied.shape[1] // 2, supplied.shape[0] // 2)
        self.data_revision += 1
        self._request_paint()

    def set_overlays(self, column_row_px: NDArray[np.float64]) -> None:
        points = np.asarray(column_row_px, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("overlays must be finite (column_px, row_px) pairs")
        self.overlays = np.array(points, copy=True, order="C")
        self._overlay_points = QPolygonF(
            [QPointF(float(c) + 0.5, float(r) + 0.5) for c, r in self.overlays]
        )
        self._request_paint()

    def set_levels(self, low: float, high: float, *, positive_log: bool = False) -> None:
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("display levels must be finite and increasing")
        if positive_log and low <= 0.0:
            raise ValueError("positive-log display requires a positive lower level")
        self.low_value, self.high_value, self.positive_log = low, high, positive_log
        self._request_paint()

    def _rect(self) -> QRectF:
        assert self.image is not None
        rows, columns = self.image.shape
        scale = min(self.width() / columns, self.height() / rows) * self.zoom
        width, height = columns * scale, rows * scale
        return QRectF(
            (self.width() - width) / 2 + self.pan.x(),
            (self.height() - height) / 2 + self.pan.y(),
            width,
            height,
        )

    def native_to_widget(self, column_px: float, row_px: float) -> QPointF:
        rect = self._rect()
        assert self.image is not None
        rows, columns = self.image.shape
        return QPointF(
            rect.left() + (column_px + 0.5) * rect.width() / columns,
            rect.top() + (row_px + 0.5) * rect.height() / rows,
        )

    def widget_to_native(self, position: QPointF) -> tuple[int, int]:
        rect = self._rect()
        assert self.image is not None
        rows, columns = self.image.shape
        column = math.floor((position.x() - rect.left()) * columns / rect.width())
        row = math.floor((position.y() - rect.top()) * rows / rect.height())
        return min(max(column, 0), columns - 1), min(max(row, 0), rows - 1)

    def initializeGL(self) -> None:
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
        functions.glUniform1f(self._program.uniformLocation("high_value"), float(self.high_value))
        functions.glUniform1i(self._program.uniformLocation("positive_log"), int(self.positive_log))
        functions.glDrawArrays(0x0004, 0, 6)
        self._texture.release()
        self._vao.release()
        self._program.release()
        if not self.plane:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 221, 96, 220), 1))
            point = self.native_to_widget(*self.crosshair)
            painter.drawLine(QPointF(point.x(), rect.top()), QPointF(point.x(), rect.bottom()))
            painter.drawLine(QPointF(rect.left(), point.y()), QPointF(rect.right(), point.y()))
            if self.overlays.size:
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
            painter.end()
        self.painted.emit(self.request_generation, perf_counter())

    def wheelEvent(self, event) -> None:
        self.zoom = min(
            30.0, max(0.25, self.zoom * (1.2 if event.angleDelta().y() > 0 else 1 / 1.2))
        )
        self._request_paint()

    def mousePressEvent(self, event) -> None:
        self._last_pointer = event.position()

    def mouseMoveEvent(self, event) -> None:
        if self._last_pointer is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._last_pointer
            if self.plane:
                self.yaw += delta.x() * 0.008
                self.pitch = max(-1.3, min(1.3, self.pitch + delta.y() * 0.008))
            else:
                self.pan += delta
            self._request_paint()
        elif self.image is not None and not self.plane:
            self.crosshair = self.widget_to_native(event.position())
            self._request_paint()
            self.crosshair_changed.emit()
        self._last_pointer = event.position()

    def mouseReleaseEvent(self, event) -> None:
        self._last_pointer = None


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
            points: list[QPointF] = []
            for index, value in enumerate(self.values):
                if not valid[index]:
                    if points:
                        painter.drawPolyline(QPolygonF(points))
                        points = []
                    continue
                position = (index + 0.5) / self.values.size
                magnitude = (float(value) - minimum) / span
                points.append(
                    QPointF(magnitude * (self.width() - 1), rect.top() + position * rect.height())
                    if self.vertical
                    else QPointF(
                        rect.left() + position * rect.width(), (1 - magnitude) * (self.height() - 1)
                    )
                )
            if points:
                painter.drawPolyline(QPolygonF(points))
        painter.end()
        self.painted.emit(self.generation, perf_counter())


class DetectorPanel(QWidget):
    """A reusable native image with exact linked horizontal and vertical bands."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.view = DetectorTextureView(parent=self)
        self.horizontal = ProfilePlot(vertical=False, source_view=self.view, parent=self)
        self.vertical = ProfilePlot(vertical=True, source_view=self.view, parent=self)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.horizontal, 0, 0)
        layout.addWidget(self.view, 1, 0)
        layout.addWidget(self.vertical, 1, 1)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 1)
        self.view.crosshair_changed.connect(self._refresh_profile_if_needed)
        self.view.painted.connect(self._refresh_axes)
        self._profile_crosshair: tuple[int, int] | None = None

    def set_image(self, image: NDArray[np.generic]) -> None:
        supplied = np.asarray(image)
        if supplied.ndim != 2 or not all(supplied.shape):
            raise ValueError("detector panels require a nonempty native plane")
        exact_band_profiles(
            supplied, column_px=supplied.shape[1] // 2, row_px=supplied.shape[0] // 2
        )
        self.view.set_image(image)
        self._profile_crosshair = None
        self._refresh_profile_if_needed()

    def _refresh_axes(self, _generation: int, _timestamp: float) -> None:
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
