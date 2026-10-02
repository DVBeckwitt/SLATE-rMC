"""Retained detector-textured view of the configured experiment geometry."""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
from detector_panel import DETECTOR_FRAGMENT_SHADER
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtOpenGL import (
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from shiboken6 import VoidPtr

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from osc_import import PreparedOsc
    from reciprocal_preview import ReciprocalMapping


_VERTEX = """#version 330 core
out vec2 uv;
uniform vec3 corner;
uniform vec3 column_axis;
uniform vec3 row_axis;
uniform vec3 target;
uniform vec3 camera_right;
uniform vec3 camera_up;
uniform vec3 camera_forward;
uniform float scale_m;
uniform float aspect;
const vec2 corners[6] = vec2[6](vec2(0,0),vec2(1,0),vec2(0,1),vec2(0,1),vec2(1,0),vec2(1,1));
void main() {
    uv = corners[gl_VertexID];
    vec3 delta = corner + uv.x * column_axis + uv.y * row_axis - target;
    float depth = dot(delta, camera_forward) / scale_m;
    float w = max(0.15, 1.0 - depth / 4.0);
    gl_Position = vec4(dot(delta,camera_right) / (scale_m * aspect),
                       dot(delta,camera_up) / scale_m, 0.0, w);
}
"""


def _point(values: object) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("scene landmark must be a finite lab-frame 3-vector")
    return tuple(float(v) for v in array)


@dataclass(frozen=True, slots=True)
class SceneGeometry:
    """Canonical lab-frame landmarks; lengths are metres and corners are native edges."""

    detector_corners_lab_m: tuple[tuple[float, float, float], ...]
    sample_lab_m: tuple[float, float, float]
    beam_source_lab_m: tuple[float, float, float]
    beam_direction_lab: tuple[float, float, float]
    detector_normal_lab: tuple[float, float, float]
    sample_axes_lab: tuple[tuple[float, float, float], ...]
    goniometer_axes_lab: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float], float], ...
    ]
    sample_width_m: float | None
    sample_length_m: float | None

    @classmethod
    def from_mapping(cls, mapping: ReciprocalMapping) -> SceneGeometry:
        instrument = mapping.instrument
        beam = instrument.lab_from_sample.rotation @ mapping.incident.states.k_air_sample_Ainv[0]
        rotations = tuple(
            (r.pivot_lab_m, r.axis_lab, math.radians(r.angle_deg)) for r in mapping.axis_rotations
        )
        sample = instrument.lab_from_sample.translation_m
        origin = mapping.source_origin_lab_m
        geometry = cls.from_instrument(
            instrument, sample if origin is None else origin, beam, rotations
        )
        if origin is None:
            extent = max(
                np.linalg.norm(np.asarray(corner) - sample)
                for corner in geometry.detector_corners_lab_m
            )
            geometry = replace(
                geometry,
                beam_source_lab_m=_point(sample - np.asarray(geometry.beam_direction_lab) * extent),
            )
        return geometry

    @classmethod
    def from_instrument(cls, instrument, origin_lab_m, direction_lab, rotations=()):
        """Use the compiled rigid instrument; no optics or intensity evaluation."""
        rows, columns = instrument.detector_shape_rc
        reference_column, reference_row = instrument.detector_reference_coordinate_px
        corners = tuple(
            _point(
                instrument.lab_from_detector.apply_point(
                    np.array(
                        (
                            (c - reference_column) * instrument.detector_column_pitch_m,
                            (r - reference_row) * instrument.detector_row_pitch_m,
                            0.0,
                        )
                    )
                )
            )
            for c, r in (
                (-0.5, -0.5),
                (columns - 0.5, -0.5),
                (-0.5, rows - 0.5),
                (columns - 0.5, rows - 0.5),
            )
        )
        beam = np.asarray(direction_lab, dtype=np.float64)
        beam = beam / np.linalg.norm(beam)
        return cls(
            corners,
            _point(instrument.lab_from_sample.translation_m),
            _point(origin_lab_m),
            _point(beam),
            _point(instrument.lab_from_detector.rotation[:, 2]),
            tuple(_point(instrument.lab_from_sample.rotation[:, i]) for i in range(3)),
            tuple((_point(p), _point(a), float(v)) for p, a, v in rotations),
            instrument.sample_width_m,
            instrument.sample_length_m,
        )


@dataclass(frozen=True, slots=True)
class PhysicalHandle:
    name: str
    pivot_lab_m: tuple[float, float, float]
    axis_lab: tuple[float, float, float]
    kind: str
    value: float
    unit: str
    metres_per_unit: float = 1.0
    label: str = ""


class ExperimentSceneView(QOpenGLWidget):
    """Camera-only redraws reuse geometry and the current context's display texture."""

    painted = Signal(int, float, object)
    camera_changed = Signal()
    object_picked = Signal(str)
    physical_started = Signal()
    physical_preview = Signal(float)
    physical_committed = Signal(float)
    physical_canceled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.geometry: SceneGeometry | None = None
        self.selection_mode = False
        self.selected_object = ""
        self.hover_object = ""
        self.callouts = ()
        self.keyboard_step = 0.0001
        self.physical_handle: PhysicalHandle | None = None
        self._handle_path: list[QPointF] = []
        self._physical_drag = False
        self._gesture_handle: PhysicalHandle | None = None
        self._gesture_path: list[QPointF] = []
        self._gesture_value = 0.0
        self._gesture_radius = 1.0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.overlay_points_lab_m: tuple[tuple[float, float, float], ...] = ()
        self._image: NDArray[np.float32] | None = None
        self._display_bin = 1
        self._native_image_shape = (1, 1)
        self._image_identity: tuple[UUID, str, int] | None = None
        self._geometry_identity: tuple[object, ...] | None = None
        self._scene_radius_m = 1.0
        self._projection_cache: (
            tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray, float] | None
        ) = None
        self._texture: QOpenGLTexture | None = None
        self._textures: OrderedDict[tuple[UUID, str, int], QOpenGLTexture] = OrderedDict()
        self._low = 0.0
        self._high = 1.0
        self._contrast_mode = "linear"
        self._program: QOpenGLShaderProgram | None = None
        self._vao: QOpenGLVertexArrayObject | None = None
        self._uploaded_identity: tuple[UUID, str, int] | None = None
        self.context_serial = 0
        self.upload_count = 0
        self.texture_evictions = 0
        self.upload_records: list[tuple[int, tuple[UUID, str, int]]] = []
        self.request_generation = 0
        self.yaw = 0.55
        self.pitch = 0.30
        self.zoom = 1.0
        self.target_lab_m = (0.0, 0.0, 0.0)
        self._press: QPointF | None = None
        self._last: QPointF | None = None
        self._dragged = False
        self._animate = QTimer(self)
        self._animate.setInterval(16)
        self._animate.timeout.connect(self._animation_step)
        self._animation: tuple[int, tuple[float, ...], tuple[float, ...]] | None = None
        self._picks: list[tuple[str, tuple[float, float, float], QPointF, QRectF]] = []
        self._pick_lines: list[tuple[str, tuple[float, float, float], QPointF, QPointF]] = []
        self._pick_polygons: list[tuple[str, tuple[float, float, float], QPolygonF]] = []
        self.focus_name: str | None = None
        self.setMinimumSize(240, 200)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_scene(
        self,
        acquisition_id: UUID | None,
        prepared: PreparedOsc | None,
        mapping: ReciprocalMapping | None,
        geometry_identity: tuple[object, ...] | None,
    ) -> None:
        image_identity = (
            (acquisition_id, prepared.decoded_sha256, 1)
            if (acquisition_id is not None and prepared is not None and mapping is not None)
            else None
        )
        if image_identity == self._image_identity and geometry_identity == self._geometry_identity:
            return
        image_changed = image_identity != self._image_identity
        first_geometry = self.geometry is None and mapping is not None
        self._image_identity = image_identity
        self._image = prepared.display if image_identity is not None else None
        if geometry_identity != self._geometry_identity or (mapping is None) != (
            self.geometry is None
        ):
            self.geometry = SceneGeometry.from_mapping(mapping) if mapping is not None else None
            self._geometry_identity = geometry_identity
            if self.geometry is not None:
                center = np.asarray(self.geometry.sample_lab_m)
                self._scene_radius_m = max(
                    0.001,
                    max(
                        np.linalg.norm(np.asarray(point) - center)
                        for point in (
                            *self.geometry.detector_corners_lab_m,
                            self.geometry.beam_source_lab_m,
                        )
                    ),
                )
            else:
                self._scene_radius_m = 1.0
            if self.geometry is not None and (image_changed or first_geometry):
                self.target_lab_m = self.geometry.sample_lab_m
        self.request_generation += 1
        self.update()

    def set_geometry(self, geometry, identity):
        if identity == self._geometry_identity and geometry is self.geometry:
            return
        first = self.geometry is None
        self.geometry, self._geometry_identity = geometry, identity
        self._scene_radius_m = (
            1.0
            if geometry is None
            else max(
                0.001,
                max(
                    np.linalg.norm(np.asarray(p) - geometry.sample_lab_m)
                    for p in (*geometry.detector_corners_lab_m, geometry.beam_source_lab_m)
                ),
            )
        )
        if first and geometry is not None:
            self.target_lab_m = geometry.sample_lab_m
        self.request_generation += 1
        self.update()

    def set_detector_image(self, display, identity, *, display_bin=1, native_shape=(1, 1)):
        if identity == self._image_identity:
            return
        self._image, self._image_identity = display, identity
        self._display_bin, self._native_image_shape = display_bin, native_shape
        self.request_generation += 1
        self.update()

    def camera_state(self) -> tuple[float, float, float, tuple[float, float, float]]:
        return self.yaw, self.pitch, self.zoom, self.target_lab_m

    def set_overlays(
        self, native_column_row_px: NDArray[np.float64], mapping: ReciprocalMapping | None
    ) -> None:
        points: tuple[tuple[float, float, float], ...] = ()
        if mapping is not None:
            supplied = np.asarray(native_column_row_px, dtype=np.float64)
            if supplied.ndim != 2 or supplied.shape[1] != 2 or len(supplied) > 5000:
                raise ValueError("scene overlays need at most 5000 native detector coordinates")
            instrument = mapping.instrument
            rows, columns = instrument.detector_shape_rc
            valid = (
                np.isfinite(supplied).all(axis=1)
                & (supplied[:, 0] >= -0.5)
                & (supplied[:, 0] <= columns - 0.5)
                & (supplied[:, 1] >= -0.5)
                & (supplied[:, 1] <= rows - 0.5)
            )
            reference_column, reference_row = instrument.detector_reference_coordinate_px
            points = tuple(
                _point(
                    instrument.lab_from_detector.apply_point(
                        np.array(
                            (
                                (column - reference_column) * instrument.detector_column_pitch_m,
                                (row - reference_row) * instrument.detector_row_pitch_m,
                                0.0,
                            )
                        )
                    )
                )
                for column, row in supplied[valid]
            )
        if points != self.overlay_points_lab_m:
            self.overlay_points_lab_m = points
            self.request_generation += 1
            self.update()

    def restore_camera(self, state: tuple[float, float, float, tuple[float, float, float]]) -> None:
        yaw, pitch, zoom, target = state
        if (
            not all(math.isfinite(v) for v in (yaw, pitch, zoom, *target))
            or not -math.pi / 2 <= pitch <= math.pi / 2
            or not 0.2 <= zoom <= 12
        ):
            raise ValueError("invalid scene camera state")
        self.yaw, self.pitch, self.zoom, self.target_lab_m = yaw, pitch, zoom, target
        self.request_generation += 1
        self.update()

    def _radius(self) -> float:
        return self._scene_radius_m

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        right = np.array((-math.sin(self.yaw), math.cos(self.yaw), 0.0))
        forward = np.array(
            (
                -math.cos(self.yaw) * math.cos(self.pitch),
                -math.sin(self.yaw) * math.cos(self.pitch),
                -math.sin(self.pitch),
            )
        )
        return right, np.cross(right, forward), forward

    def _project(self, point: tuple[float, float, float]) -> QPointF:
        key = self.request_generation, self.width(), self.height()
        if self._projection_cache is None or self._projection_cache[0] != key:
            right, up, forward = self._basis()
            scale = self._radius() * 1.35 / self.zoom
            self._projection_cache = key, right, up, forward, scale
        _, right, up, forward, scale = self._projection_cache
        delta = np.asarray(point) - np.asarray(self.target_lab_m)
        depth = max(0.15, 1.0 - float(delta @ forward) / (scale * 4.0))
        height = max(self.height(), 1)
        return QPointF(
            self.width() / 2 + float(delta @ right) * height / (2 * scale * depth),
            self.height() / 2 - float(delta @ up) * height / (2 * scale * depth),
        )

    def _request_camera(self) -> None:
        self.request_generation += 1
        self.update()
        self.camera_changed.emit()

    def _view_along(
        self,
        direction_lab: tuple[float, float, float],
        target_lab_m: tuple[float, float, float],
        zoom: float,
    ) -> tuple[float, float, float, tuple[float, float, float]]:
        direction = np.asarray(direction_lab, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        horizontal = math.hypot(direction[0], direction[1])
        yaw = math.atan2(-direction[1], -direction[0]) if horizontal > 1e-12 else self.yaw
        pitch = math.atan2(-direction[2], horizontal)
        return yaw, pitch, zoom, target_lab_m

    def _reduced_motion(self) -> bool:
        if self.window().property("reduced_motion"):
            return True
        if sys.platform == "win32":
            import ctypes

            enabled = ctypes.c_int(1)
            if ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0):
                return not bool(enabled.value)
        return False

    def preset(self, name: str) -> None:
        geometry = self.geometry
        if geometry is None:
            return
        detector_center = np.mean(np.asarray(geometry.detector_corners_lab_m), axis=0)
        choices = {
            "context": (0.55, 0.30, 1.0, geometry.sample_lab_m),
            "sample": (self.yaw, self.pitch, 3.5, geometry.sample_lab_m),
            "front": self._view_along(
                _point(-np.asarray(geometry.sample_axes_lab[2])), geometry.sample_lab_m, 1.4
            ),
            "side": self._view_along(
                _point(-np.asarray(geometry.sample_axes_lab[0])), geometry.sample_lab_m, 1.4
            ),
            "beam": self._view_along(geometry.beam_direction_lab, geometry.sample_lab_m, 1.4),
            "detector": self._view_along(
                _point(-np.asarray(geometry.detector_normal_lab)), _point(detector_center), 2.1
            ),
        }
        if name not in choices:
            raise ValueError("unsupported scene camera preset")
        if name == "context":
            self.focus_name = None
        else:
            self.focus_name = {
                "sample": "sample",
                "front": "sample front",
                "side": "sample side",
                "beam": "incident beam",
                "detector": "detector",
            }[name]
        self._move_camera(choices[name])

    def _move_camera(self, end: tuple[float, float, float, tuple[float, float, float]]) -> None:
        self._animate.stop()
        start = (self.yaw, self.pitch, self.zoom, *self.target_lab_m)
        finish = (end[0], end[1], end[2], *end[3])
        if self._reduced_motion():
            self.restore_camera(end)
            self.camera_changed.emit()
            return
        self._animation = (0, start, finish)
        self._animate.start()

    def focus(self, name: str, target_lab_m: tuple[float, float, float]) -> None:
        self.focus_name = name
        zoom = 2.1 if name.startswith("detector") else 3.0 if name.startswith("axis ") else 3.5
        self._move_camera((self.yaw, self.pitch, zoom, target_lab_m))

    def _draw_pick_label(
        self,
        painter: QPainter,
        name: str,
        anchor_lab_m: tuple[float, float, float],
        target_lab_m: tuple[float, float, float],
        offset: QPointF,
    ) -> None:
        if self.selection_mode:
            return
        anchor = self._project(anchor_lab_m)
        baseline = anchor + offset
        painter.drawText(baseline, name)
        metrics = painter.fontMetrics()
        region = QRectF(
            baseline.x() - 3,
            baseline.y() - metrics.ascent() - 3,
            metrics.horizontalAdvance(name) + 6,
            metrics.height() + 6,
        )
        self._picks.append((name, target_lab_m, anchor, region))

    def _animation_step(self) -> None:
        if self._animation is None:
            self._animate.stop()
            return
        step, start, end = self._animation
        step += 1
        t = min(step / 8, 1.0)
        t = t * t * (3 - 2 * t)
        values = tuple(a + (b - a) * t for a, b in zip(start, end, strict=True))
        self.yaw, self.pitch, self.zoom = values[:3]
        self.target_lab_m = values[3:6]
        self._request_camera()
        if step >= 8:
            self._animate.stop()
            self._animation = None
        else:
            self._animation = (step, start, end)

    def initializeGL(self) -> None:
        program = QOpenGLShaderProgram(self)
        for kind, source in (
            (QOpenGLShader.ShaderTypeBit.Vertex, _VERTEX),
            (QOpenGLShader.ShaderTypeBit.Fragment, DETECTOR_FRAGMENT_SHADER),
        ):
            if not program.addShaderFromSourceCode(kind, source):
                raise RuntimeError(program.log())
        if not program.link():
            raise RuntimeError(program.log())
        vao = QOpenGLVertexArrayObject(self)
        if not vao.create():
            raise RuntimeError("scene vertex array creation failed")
        self._program, self._vao = program, vao
        self.context_serial += 1
        self._uploaded_identity = None
        self.context().aboutToBeDestroyed.connect(self.release_resources)

    def release_resources(self) -> None:
        if not self._textures and self._vao is None:
            return
        self.makeCurrent()
        try:
            for texture in self._textures.values():
                texture.destroy()
            if self._vao is not None:
                self._vao.destroy()
        finally:
            self._textures.clear()
            self._texture = None
            self._vao = None
            self._program = None
            self._uploaded_identity = None
            self.doneCurrent()

    def _upload_if_needed(self) -> None:
        if self._image is None or self._image_identity is None:
            return
        cached = self._textures.get(self._image_identity)
        if cached is not None:
            self._textures.move_to_end(self._image_identity)
            self._texture = cached
            self._uploaded_identity = self._image_identity
            return
        rows, columns = self._image.shape
        texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        texture.setFormat(QOpenGLTexture.TextureFormat.R32F)
        texture.setSize(columns, rows)
        texture.allocateStorage(QOpenGLTexture.PixelFormat.Red, QOpenGLTexture.PixelType.Float32)
        texture.setMinMagFilters(QOpenGLTexture.Filter.Nearest, QOpenGLTexture.Filter.Nearest)
        texture.setData(
            QOpenGLTexture.PixelFormat.Red,
            QOpenGLTexture.PixelType.Float32,
            VoidPtr(self._image.ctypes.data, self._image.nbytes, False),
        )
        self._texture = texture
        self._uploaded_identity = self._image_identity
        self._textures[self._image_identity] = texture
        if len(self._textures) > 2:
            _victim_identity, victim = self._textures.popitem(last=False)
            victim.destroy()
            self.texture_evictions += 1
        self.upload_count += 1
        assert self._image_identity is not None
        self.upload_records.append((self.context_serial, self._image_identity))
        if len(self.upload_records) > 128:
            self.upload_records.pop(0)

    def paintGL(self) -> None:
        functions = self.context().functions()
        functions.glClearColor(0.07, 0.10, 0.14, 1.0)
        functions.glClear(0x00004000)
        geometry = self.geometry
        if geometry is not None and self._image is not None and self._program and self._vao:
            self._upload_if_needed()
            program = self._program
            program.bind()
            self._vao.bind()
            assert self._texture is not None
            self._texture.bind(0)
            right, up, forward = self._basis()
            corners = np.asarray(geometry.detector_corners_lab_m)
            vectors = {
                "corner": corners[0],
                "column_axis": corners[1] - corners[0],
                "row_axis": corners[2] - corners[0],
                "target": self.target_lab_m,
                "camera_right": right,
                "camera_up": up,
                "camera_forward": forward,
            }
            for name, vector in vectors.items():
                functions.glUniform3f(program.uniformLocation(name), *(float(v) for v in vector))
            functions.glUniform1f(
                program.uniformLocation("scale_m"), self._radius() * 1.35 / self.zoom
            )
            functions.glUniform1f(
                program.uniformLocation("aspect"), self.width() / max(self.height(), 1)
            )
            functions.glUniform1i(program.uniformLocation("detector"), 0)
            functions.glUniform1f(program.uniformLocation("display_bin"), float(self._display_bin))
            functions.glUniform2f(
                program.uniformLocation("native_shape"),
                float(self._native_image_shape[1]),
                float(self._native_image_shape[0]),
            )
            functions.glUniform1f(program.uniformLocation("low_value"), self._low)
            functions.glUniform1f(program.uniformLocation("high_value"), self._high)
            functions.glUniform1i(
                program.uniformLocation("contrast_mode"),
                {"linear": 0, "signed": 1, "positive_log": 2}[self._contrast_mode],
            )
            functions.glDrawArrays(0x0004, 0, 6)
            self._texture.release()
            self._vao.release()
            program.release()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._picks = []
        self._pick_lines = []
        self._pick_polygons = []
        if geometry is None:
            painter.setPen(QColor("#b7c5cf"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Select an image and map verified geometry to view the experiment",
            )
        else:
            if self._image is None and not self.selection_mode:
                painter.setPen(QColor("#b7c5cf"))
                painter.drawText(
                    10, 20, "Configured geometry; matching detector texture unavailable"
                )
            corners = [self._project(point) for point in geometry.detector_corners_lab_m]
            detector_center = _point(np.mean(np.asarray(geometry.detector_corners_lab_m), axis=0))
            self._pick_polygons.append(
                (
                    "detector",
                    detector_center,
                    QPolygonF([corners[0], corners[1], corners[3], corners[2]]),
                )
            )
            painter.setPen(QPen(QColor("#e0c887"), 2))
            for a, b in ((0, 1), (1, 3), (3, 2), (2, 0)):
                painter.drawLine(corners[a], corners[b])
            sample = self._project(geometry.sample_lab_m)
            source = self._project(geometry.beam_source_lab_m)
            painter.setPen(QPen(QColor("#e1a34d"), 2))
            normal = np.asarray(geometry.sample_axes_lab[2])
            direction = np.asarray(geometry.beam_direction_lab)
            denominator = float(normal @ direction)
            distance = (
                float(normal @ (np.asarray(geometry.sample_lab_m) - geometry.beam_source_lab_m))
                / denominator
                if abs(denominator) > 1e-14
                else self._radius()
            )
            beam_end = self._project(
                _point(
                    np.asarray(geometry.beam_source_lab_m)
                    + direction * (distance if distance > 0 else self._radius())
                )
            )
            painter.drawLine(source, beam_end)
            if self.selection_mode:
                delta = beam_end - source
                length = math.hypot(delta.x(), delta.y())
                if length > 1e-8:
                    direction_2d = delta / length
                    perpendicular = QPointF(-direction_2d.y(), direction_2d.x())
                    painter.drawLine(beam_end, beam_end - direction_2d * 10 + perpendicular * 5)
                    painter.drawLine(beam_end, beam_end - direction_2d * 10 - perpendicular * 5)
            self._pick_lines.append(("incident beam", geometry.beam_source_lab_m, source, beam_end))
            self._draw_pick_label(
                painter,
                "incident beam",
                geometry.beam_source_lab_m,
                geometry.sample_lab_m,
                QPointF(5, -5),
            )
            radius = self._radius() * 0.14
            width = geometry.sample_width_m or radius * 0.35
            length = geometry.sample_length_m or radius * 0.35
            center_lab = np.asarray(geometry.sample_lab_m)
            axis_x, axis_y = (np.asarray(axis) for axis in geometry.sample_axes_lab[:2])
            sample_corners = QPolygonF(
                [
                    self._project(
                        _point(center_lab + sx * width * axis_x / 2 + sy * length * axis_y / 2)
                    )
                    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                ]
            )
            self._pick_polygons.append(("sample", geometry.sample_lab_m, sample_corners))
            painter.setPen(QPen(QColor("#d9d0b5"), 1))
            painter.setBrush(QColor(217, 208, 181, 70))
            painter.drawPolygon(sample_corners)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#af92d5"), 1, Qt.PenStyle.DashLine))
            painter.drawEllipse(sample, 14, 8)
            self._draw_pick_label(
                painter,
                "holder · schematic",
                geometry.sample_lab_m,
                geometry.sample_lab_m,
                QPointF(16, 18),
            )
            for axis, color, label in zip(
                geometry.sample_axes_lab,
                ("#da6c63", "#70b782", "#72aee8"),
                ("x", "y", "z"),
                strict=True,
            ):
                if self.selection_mode and (
                    label != "z" or self.selected_object not in ("Sample", "Mount")
                ):
                    continue
                tip = _point(np.asarray(geometry.sample_lab_m) + np.asarray(axis) * radius)
                painter.setPen(QPen(QColor(color), 2))
                painter.drawLine(sample, self._project(tip))
                self._draw_pick_label(painter, label, tip, geometry.sample_lab_m, QPointF(3, -3))
            painter.setBrush(QColor("#f3e7c8"))
            painter.drawEllipse(sample, 5, 5)
            self._draw_pick_label(
                painter, "sample", geometry.sample_lab_m, geometry.sample_lab_m, QPointF(7, -7)
            )
            for index, (pivot, axis, angle_rad) in enumerate(geometry.goniometer_axes_lab):
                if self.selection_mode and f"Axis {index + 1}" not in (
                    self.selected_object,
                    self.hover_object,
                ):
                    continue
                start = self._project(_point(np.asarray(pivot) - np.asarray(axis) * radius))
                tip = _point(np.asarray(pivot) + np.asarray(axis) * radius)
                end = self._project(tip)
                painter.setPen(QPen(QColor("#af92d5"), 2, Qt.PenStyle.DashLine))
                painter.drawLine(start, end)
                axis_name = f"axis {index + 1} · {math.degrees(angle_rad):g}° / pivot"
                self._pick_lines.append((axis_name, pivot, start, end))
                self._draw_pick_label(painter, axis_name, tip, pivot, QPointF(3, -3))
            painter.setPen(QColor("#d9e4eb"))
            self._draw_pick_label(
                painter,
                "detector · native image",
                geometry.detector_corners_lab_m[1],
                detector_center,
                QPointF(4, -4),
            )
            painter.setPen(QPen(QColor("#75d9dd"), 2))
            for point in self.overlay_points_lab_m:
                center = self._project(point)
                painter.drawEllipse(center, 3, 3)
            if self.selection_mode:
                if self._image is None:
                    painter.drawText(
                        10, self.height() - 10, "Schematic detector: no matching intensity"
                    )
            else:
                painter.drawText(
                    10,
                    self.height() - 10,
                    painter.fontMetrics().elidedText(
                        "Schematic holder/axes; detector and configured sample positions are in metres",
                        Qt.TextElideMode.ElideRight,
                        self.width() - 20,
                    ),
                )
            if self.focus_name is not None and not self.selection_mode:
                painter.drawText(10, 20, f"Focused: {self.focus_name} · Back to experiment")
        if self.selection_mode and geometry is not None:
            self._draw_device_callouts(painter)
        self._draw_physical_handle(painter)
        painter.end()
        self.painted.emit(self.request_generation, perf_counter(), self._image_identity)

    def _draw_device_callouts(self, painter):
        # Leaders are selectors, separate from the selected physical arrow/arc.
        width = max(self.width() - 20, 1)
        axis_limit = (
            0
            if self.height() < 260 and self.selected_object.startswith("Axis ")
            else (1 if self.height() < 260 else 3)
        )
        visible = tuple(
            item
            for item in self.callouts
            if (
                item[0] not in ("Crystal / material", "Mosaic", "External path")
                or item[0] == self.selected_object
            )
            and (
                not item[0].startswith("Axis ")
                or int(item[0].split()[1]) <= axis_limit
                or item[0] == self.selected_object
            )
        )
        for name, point in visible:
            anchor = self._project(point)
            height = self.height()
            metrics = painter.fontMetrics()
            label_width = metrics.horizontalAdvance(name)
            middle = (55 + height - 35) / 2
            positions = {
                "Beam": (10, 55),
                "Crystal / material": (10, middle),
                "Goniometer base": ((width - label_width) / 2, height - 35),
                "Mount": (width - label_width - 4, height - 35),
                "Sample": ((width - label_width) / 2, 55),
                "Detector": (width - label_width - 4, 55),
                "Mosaic": (width - label_width - 4, middle),
                "External path": (10, middle),
            }
            axes = [item[0] for item in visible if item[0].startswith("Axis ")]
            axis_index = axes.index(name) if name in axes else 0
            spacing = metrics.height() + 8
            x, y = positions.get(name, (10, height - 35 - (len(axes) - 1 - axis_index) * spacing))
            position = QPointF(x, y)
            metrics = painter.fontMetrics()
            region = QRectF(
                position.x() - 3,
                position.y() - metrics.ascent() - 3,
                metrics.horizontalAdvance(name) + 8,
                metrics.height() + 6,
            )
            color = QColor(
                "#ffcd70" if name in (self.selected_object, self.hover_object) else "#d9e4eb"
            )
            painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
            painter.drawLine(anchor, region.center())
            self._pick_lines.append((name, point, anchor, region.center()))
            painter.fillRect(region, QColor("#15232c"))
            painter.setPen(color)
            painter.drawText(position, name)
            self._picks.append((name, point, anchor, region))
            painter.drawEllipse(anchor, 4, 4)
        # Separate schematic base/mount glyphs share their actual declared anchors.
        landmarks = dict(self.callouts)
        for name, radius in (("Goniometer base", 12), ("Mount", 7)):
            if name not in landmarks:
                continue
            point = self._project(landmarks[name])
            painter.setPen(QPen(QColor("#b3afc5"), 2))
            painter.drawEllipse(point, radius, radius / 2)
        # Selected spatial details use colored arrows; their meanings are in the
        # inline caption, rather than stacked text at nearly coincident LAB points.
        if self.selected_object == "Detector" or self.hover_object == "Detector":
            reference = self._project(landmarks["Detector"])
            painter.setPen(QColor("#d9e4eb"))
            painter.drawLine(reference - QPointF(5, 0), reference + QPointF(5, 0))
            painter.drawLine(reference - QPointF(0, 5), reference + QPointF(0, 5))
            corners = self.geometry.detector_corners_lab_m
            for index, color in ((1, "#75d9dd"), (2, "#df9aca")):
                direction = np.asarray(corners[index]) - corners[0]
                direction /= np.linalg.norm(direction)
                endpoint = self._project(
                    _point(np.asarray(landmarks["Detector"]) + direction * self._radius() * 0.14)
                )
                painter.setPen(QPen(QColor(color), 2))
                painter.drawLine(reference, endpoint)
        # This camera-oriented LAB triad has fixed screen size. Its colored legend
        # keeps separate text positions even when axes project onto the same ray.
        origin = QPointF(self.width() - 35, 24)
        right, up, _forward = self._basis()
        painter.setPen(QColor("#d9e4eb"))
        painter.drawText(10, 25, "LAB")
        x = 45
        for axis, color, name in zip(
            np.eye(3), ("#ef8880", "#9cd6ae", "#92c6fa"), ("+X", "+Y", "+Z"), strict=True
        ):
            end = origin + QPointF(float(axis @ right) * 14, -float(axis @ up) * 14)
            painter.setPen(QPen(QColor(color), 2))
            painter.drawLine(origin, end)
            painter.drawText(x, 25, name)
            x += painter.fontMetrics().horizontalAdvance(name) + 18

    def set_physical_handle(self, handle: PhysicalHandle | None) -> None:
        if handle is not None:
            axis = np.asarray(handle.axis_lab)
            if handle.kind not in ("translation", "rotation") or not np.isclose(
                np.linalg.norm(axis), 1.0
            ):
                raise ValueError("Physical handle needs one constrained unit axis")
        self.physical_handle = handle
        self._handle_path = []
        self.update()

    def _draw_physical_handle(self, painter) -> None:
        handle = self.physical_handle
        self._handle_path = []
        if handle is None or self.geometry is None:
            return
        pivot, axis = np.asarray(handle.pivot_lab_m), np.asarray(handle.axis_lab)
        size = self._radius() * 0.28
        if handle.kind == "translation":
            self._handle_path = [
                self._project(_point(pivot)),
                self._project(_point(pivot + size * axis)),
            ]
        else:
            seed = np.eye(3)[np.argmin(np.abs(axis))]
            first = np.cross(axis, seed)
            first /= np.linalg.norm(first)
            second = np.cross(axis, first)
            self._handle_path = [
                self._project(_point(pivot + size * (first * math.cos(a) + second * math.sin(a))))
                for a in np.linspace(-0.5, 2.4, 40)
            ]
        painter.setPen(QPen(QColor("#ffcd70"), 3))
        for start, end in zip(self._handle_path[:-1], self._handle_path[1:], strict=True):
            painter.drawLine(start, end)
        end, prior = self._handle_path[-1], self._handle_path[-2]
        d = end - prior
        norm = math.hypot(d.x(), d.y())
        if norm > 1e-8:
            tangent = QPointF(d.x() / norm, d.y() / norm)
            normal = QPointF(-tangent.y(), tangent.x())
            painter.drawLine(end, end - tangent * 12 + normal * 6)
            painter.drawLine(end, end - tangent * 12 - normal * 6)
        painter.setPen(QColor("#ffcd70"))
        if self.selection_mode:
            return  # The exact selected parameter and units are already visible inline.
        painter.drawText(
            QRectF(10, 30, max(1, self.width() - 20), 80),
            Qt.TextFlag.TextWordWrap,
            handle.label
            or f"{handle.name}: {handle.value:.17g} {handle.unit}; positive arrow; Escape cancels",
        )

    def _handle_hit(self, position) -> bool:
        for start, end in zip(self._handle_path[:-1], self._handle_path[1:], strict=True):
            delta = end - start
            norm2 = delta.x() ** 2 + delta.y() ** 2
            t = min(
                1,
                max(
                    0,
                    ((position - start).x() * delta.x() + (position - start).y() * delta.y())
                    / max(norm2, 1e-12),
                ),
            )
            if (
                math.hypot((position - start - delta * t).x(), (position - start - delta * t).y())
                <= 12
            ):
                return True
        return False

    def cancel_physical_gesture(self) -> None:
        if self._physical_drag:
            self._physical_drag = False
            self._press = self._last = None
            self._gesture_handle = None
            self.physical_canceled.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self._physical_drag:
            self.cancel_physical_gesture()
            event.accept()
        elif (
            self.selection_mode
            and self.physical_handle is not None
            and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down)
        ):
            value = self.physical_handle.value + self.keyboard_step * (
                -1 if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Down) else 1
            )
            self.physical_started.emit()
            self.physical_preview.emit(value)
            self.physical_committed.emit(value)
            event.accept()
        else:
            super().keyPressEvent(event)

    def set_levels(self, low: float, high: float, mode: str = "linear") -> None:
        if mode not in ("linear", "signed", "positive_log"):
            raise ValueError("unsupported scene contrast mode")
        if (low, high, mode) == (self._low, self._high, self._contrast_mode):
            return
        self._low, self._high, self._contrast_mode = float(low), float(high), mode
        self.request_generation += 1
        self.update()

    def mousePressEvent(self, event) -> None:
        if self._physical_drag:
            self.cancel_physical_gesture()
        self._animate.stop()
        self._animation = None
        self._press = event.position()
        self._last = event.position()
        self._dragged = False
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.physical_handle is not None
            and self._handle_hit(event.position())
        ):
            self.setFocus()
            self._physical_drag = True
            self._gesture_handle = self.physical_handle
            self._gesture_path = list(self._handle_path)
            self._gesture_radius = self._radius()
            self._gesture_value = self.physical_handle.value
            self.physical_started.emit()

    def mouseMoveEvent(self, event) -> None:
        if not event.buttons():
            if self.selection_mode:
                hit = self._pick_at(event.position())
                name = "" if hit is None else hit[0]
                if name != self.hover_object:
                    self.hover_object = name
                    self.setToolTip(
                        "Select " + name if name else "Empty drag orbits; right drag pans"
                    )
                    self.update()
            return
        if self._last is None:
            return
        if self._physical_drag:
            handle = self._gesture_handle
            path = self._gesture_path
            delta = event.position() - self._press
            if handle.kind == "translation":
                vector = path[-1] - path[0]
                norm2 = vector.x() ** 2 + vector.y() ** 2
                if norm2 <= 4:
                    return
                amount = (delta.x() * vector.x() + delta.y() * vector.y()) / norm2
                value = handle.value + amount * self._gesture_radius * 0.28 / handle.metres_per_unit
            else:
                nearest = min(
                    range(len(path)), key=lambda i: (path[i] - self._press).manhattanLength()
                )
                lo, hi = max(0, nearest - 1), min(len(path) - 1, nearest + 1)
                tangent = path[hi] - path[lo]
                norm2 = tangent.x() ** 2 + tangent.y() ** 2
                if norm2 <= 4:
                    return
                angle = (
                    (delta.x() * tangent.x() + delta.y() * tangent.y())
                    / norm2
                    * (hi - lo)
                    * 2.9
                    / 39
                )
                value = handle.value + (math.degrees(angle) if handle.unit == "deg" else angle)
            self._gesture_value = value
            self.physical_preview.emit(value)
            return
        delta = event.position() - self._last
        if (event.position() - self._press).manhattanLength() > 3:
            self._dragged = True
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.yaw += delta.x() * 0.008
            self.pitch = min(max(self.pitch + delta.y() * 0.008, -math.pi / 2), math.pi / 2)
        elif event.buttons() & Qt.MouseButton.RightButton:
            right, up, _ = self._basis()
            target = np.asarray(self.target_lab_m) - (
                right * delta.x() - up * delta.y()
            ) * self._radius() / max(self.height(), 1)
            self.target_lab_m = _point(target)
        self._last = event.position()
        self._request_camera()

    def _pick_at(self, position):
        hit = next(
            (
                (name, target)
                for name, target, _anchor, region in reversed(self._picks)
                if region.contains(position)
            ),
            None,
        )
        if hit is None and self.selection_mode:
            hit = next(
                (
                    (name, target)
                    for name, target, polygon in reversed(self._pick_polygons)
                    if polygon.containsPoint(position, Qt.FillRule.OddEvenFill)
                ),
                None,
            )
        if hit is None:
            near = sorted(
                (math.hypot(anchor.x() - position.x(), anchor.y() - position.y()), name, target)
                for name, target, anchor, _region in self._picks
            )
            if near and near[0][0] <= 16:
                hit = near[0][1:]
        if hit is None:
            for name, target, start, end in self._pick_lines:
                dx, dy = end.x() - start.x(), end.y() - start.y()
                fraction = min(
                    max(
                        ((position.x() - start.x()) * dx + (position.y() - start.y()) * dy)
                        / max(dx * dx + dy * dy, 1e-12),
                        0,
                    ),
                    1,
                )
                distance = math.hypot(
                    position.x() - start.x() - fraction * dx,
                    position.y() - start.y() - fraction * dy,
                )
                if distance <= 8:
                    hit = name, target
                    break
        if hit is None:
            hit = next(
                (
                    (name, target)
                    for name, target, polygon in reversed(self._pick_polygons)
                    if polygon.containsPoint(position, Qt.FillRule.OddEvenFill)
                ),
                None,
            )
        return hit

    def mouseReleaseEvent(self, event) -> None:
        if self._physical_drag:
            if event.button() != Qt.MouseButton.LeftButton:
                return
            self._physical_drag = False
            self._press = self._last = None
            self._gesture_handle = None
            self.physical_committed.emit(self._gesture_value)
            return
        if (
            not self._dragged
            and self.geometry is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            hit = self._pick_at(event.position())
            if hit is not None:
                if self.selection_mode:
                    self.object_picked.emit(hit[0])
                else:
                    self.focus(*hit)
        self._press = self._last = None

    def wheelEvent(self, event) -> None:
        if self._physical_drag:
            return
        self.zoom = min(max(self.zoom * (1.12 ** (event.angleDelta().y() / 120)), 0.2), 12.0)
        self._request_camera()


class ExperimentScenePanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QGridLayout()
        self.preset_buttons: dict[str, QPushButton] = {}
        for title, preset in (
            ("Back to experiment", "context"),
            ("Front", "front"),
            ("Side", "side"),
            ("Along beam", "beam"),
            ("Detector normal", "detector"),
            ("Sample", "sample"),
        ):
            button = QPushButton(title)
            button.clicked.connect(lambda _checked=False, name=preset: self.view.preset(name))
            index = len(self.preset_buttons)
            controls.addWidget(button, index // 3, index % 3)
            self.preset_buttons[preset] = button
        layout.addLayout(controls)
        self.view = ExperimentSceneView()
        layout.addWidget(self.view, 1)
        hint = self.hint = QLabel(
            "Drag to orbit · right drag to pan · wheel to zoom · click a landmark to inspect"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
