"""Two native detector presenters with bounded display cuts and reference profiles."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from comparison_state import LineDefinition, LineSamples, PinIdentity, PinnedProfiles, sample_line
from detector_panel import BandProfiles, DetectorPanel, DetectorTextureView
from project_state import Acquisition, ComparisonState, DetectorViewState
from PySide6.QtCore import QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from mask_state import PreparedMask
    from osc_import import PreparedOsc


class SamplePlot(QWidget):
    painted = Signal(int, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.x = self.values = self.support = None
        self.title = "Unavailable"
        self.generation = 0
        self.unit = "counts"
        self.setMinimumSize(150, 60)
        self.setMaximumHeight(80)

    def set_samples(
        self,
        x: np.ndarray | None,
        values: np.ndarray | None,
        support: np.ndarray | None,
        title: str,
        *,
        unit: str = "counts",
    ) -> None:
        self.x, self.values, self.support, self.title = x, values, support, title
        self.unit = unit
        self.generation += 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 29, 36))
        painter.setPen(QColor(220, 230, 235))
        painter.drawText(4, 14, self.title)
        if self.values is not None:
            valid = np.isfinite(self.values) & (self.support > 0)
            if np.any(valid):
                lo, hi = float(np.min(self.values[valid])), float(np.max(self.values[valid]))
                if lo == hi:
                    lo, hi = lo - 0.5, hi + 0.5
                first, last = float(self.x[0]), float(self.x[-1])
                span = max(last - first, 1.0)
                path = QPainterPath()
                connected = False
                for x, y, accepted in zip(self.x, self.values, valid, strict=True):
                    if not accepted:
                        connected = False
                        continue
                    point = QPointF(
                        4 + (float(x) - first) / span * (self.width() - 8),
                        20 + (hi - float(y)) / (hi - lo) * (self.height() - 38),
                    )
                    if connected:
                        path.lineTo(point)
                    else:
                        path.moveTo(point)
                        painter.drawPoint(point)
                    connected = True
                painter.setPen(QPen(QColor(130, 225, 190), 1))
                painter.drawPath(path)
                painter.setPen(QColor(220, 230, 235))
                painter.drawText(
                    4, self.height() - 3, f"{first:g}..{last:g} px; {lo:g}..{hi:g} {self.unit}"
                )
            else:
                painter.drawText(4, 35, "No valid support; samples missing")
        painter.end()
        self.painted.emit(self.generation, perf_counter())


def panel_view_state(panel: DetectorPanel) -> DetectorViewState | None:
    v = panel.view
    if v.image is None:
        return None
    return DetectorViewState(
        *v.crosshair,
        v.effective_zoom(),
        v.pan.x(),
        v.pan.y(),
        v.low_value,
        v.high_value,
        v.contrast_mode,
        v.devicePixelRatioF(),
        v.scale_mode,
        v.show_image,
        v.show_crosshair,
        v.show_markers,
        v.follow_cursor,
        panel.row_width_control.value(),
        panel.column_width_control.value(),
        panel.profile_measure_control.currentData(),
        panel.profile_scope_control.currentData(),
        panel._roi_bounds,
        panel.horizontal.intensity_limits,
        panel.vertical.intensity_limits,
        v.show_mask,
    )


class ComparisonPanel(QWidget):
    selection_requested = Signal(int, object)
    profiles_requested = Signal()
    cut_requested = Signal(int)
    state_changed = Signal()
    frames_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.desired = [None, None]
        self.acquisitions = [None, None]
        self.planes = [None, None]
        self.ready = [False, False]
        self.frames = [None, None]
        self.lines = [None, None]
        self.samples = [None, None]
        self.cut_keys = [None, None]
        self.cut_pending = [False, False]
        self.pin_identity = None
        self.pin = None
        self.pin_pending = False
        self.pin_unavailable_reason = None
        self._restore = [None, None]
        self._independent_levels = [None, None]
        self._link_guard = False
        self._magnifier_key = None
        self._pin_plot_key = None
        self._restored_link = False
        self._restored_limits = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        choices = QHBoxLayout()
        self.selectors = [QComboBox(), QComboBox()]
        for slot, selector in enumerate(self.selectors):
            selector.setMinimumContentsLength(10)
            selector.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            choices.addWidget(QLabel("A" if slot == 0 else "B"))
            choices.addWidget(selector, 1)
            selector.currentIndexChanged.connect(lambda _index, s=slot: self._choose(s))
        layout.addLayout(choices)
        tools = QHBoxLayout()
        self.active = QComboBox()
        self.active.addItems(("Inspect A", "Inspect B"))
        self.link = QCheckBox("Link calibrated navigation")
        self.lock_limits = QCheckBox("Lock raw-count limits")
        self.frames_button = QPushButton("Check detector frames")
        self.fit_button = QPushButton("Fit active")
        self.auto_button = QPushButton("Auto active limits")
        for widget in (
            self.active,
            self.link,
            self.lock_limits,
            self.frames_button,
            self.fit_button,
            self.auto_button,
        ):
            tools.addWidget(widget)
        layout.addLayout(tools)
        self.frame_status = QLabel("Native pixel frames independent; calibration unavailable")
        self.frame_status.setFixedHeight(self.fontMetrics().lineSpacing())
        self.frame_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.frame_status)
        profiles = QHBoxLayout()
        self.row_width = QSpinBox()
        self.row_width.setRange(1, 16384)
        self.row_width.setValue(1)
        self.column_width = QSpinBox()
        self.column_width.setRange(1, 16384)
        self.column_width.setValue(1)
        self.measure = QComboBox()
        self.measure.addItems(("sum", "mean"))
        self.scope = QComboBox()
        self.scope.addItems(("band", "full"))
        self.pin_center = QCheckBox("Pin crosshair")
        self.mode = QComboBox()
        self.mode.addItems(("linear", "signed", "positive_log"))
        for title, widget in (
            ("Rows", self.row_width),
            ("Columns", self.column_width),
            ("Measure", self.measure),
            ("Scope", self.scope),
            ("Colors", self.mode),
        ):
            profiles.addWidget(QLabel(title))
            profiles.addWidget(widget)
        profiles.addWidget(self.pin_center)
        layout.addLayout(profiles)
        editor = QHBoxLayout()
        self.draw_line = QCheckBox("Draw line")
        editor.addWidget(self.draw_line)
        self.endpoints = []
        for name in ("c0", "r0", "c1", "r1"):
            spin = QDoubleSpinBox()
            spin.setRange(-32768, 32768)
            spin.setDecimals(3)
            self.endpoints.append(spin)
            editor.addWidget(QLabel(name))
            editor.addWidget(spin)
        self.spacing = QDoubleSpinBox()
        self.spacing.setRange(0.25, 64)
        self.spacing.setValue(1)
        self.spacing.setSuffix(" px")
        self.apply_line_button = QPushButton("Apply line")
        self.clear_line_button = QPushButton("Clear line")
        for widget in (self.spacing, self.apply_line_button, self.clear_line_button):
            editor.addWidget(widget)
        layout.addLayout(editor)
        self.policy = QLabel(
            "Line: nearest native pixel (ties upward); spacing ≤ requested; no strip average. Esc cancels drawing. Raw counts; no exposure normalization."
        )
        self.policy.setWordWrap(True)
        self.policy.setFixedHeight(2 * self.fontMetrics().lineSpacing())
        self.policy.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.policy)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.panels = [DetectorPanel(compact=True), DetectorPanel(compact=True)]
        self.labels = [QLabel("A unavailable"), QLabel("B unavailable")]
        self.cut_plots = [SamplePlot(), SamplePlot()]
        for slot, panel in enumerate(self.panels):
            self.labels[slot].setWordWrap(True)
            self.labels[slot].setFixedHeight(3 * self.fontMetrics().lineSpacing())
            self.labels[slot].setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            grid.addWidget(self.labels[slot], 0, slot)
            grid.addWidget(panel, 1, slot)
            grid.addWidget(self.cut_plots[slot], 2, slot)
            panel.profile_requested.connect(self.profiles_requested)
            panel.view.view_state_changed.connect(lambda s=slot: self._navigate(s))
            panel.view.crosshair_changed.connect(lambda s=slot: self._cursor(s))
            panel.view.line_endpoints_ready.connect(
                lambda points, s=slot: self.set_line(s, *points)
            )
            panel.view.line_mode_changed.connect(
                lambda enabled, s=slot: self._line_mode(s, enabled)
            )
        sidebar = QVBoxLayout()
        self.pin_button = QPushButton("Pin / replace active profiles")
        self.unpin_button = QPushButton("Unpin reference")
        self.pin_axis = QComboBox()
        self.pin_axis.addItems(("horizontal", "vertical"))
        self.pin_status = QLabel("No reference pinned")
        self.pin_status.setWordWrap(True)
        self.pin_status.setFixedHeight(3 * self.fontMetrics().lineSpacing())
        self.pin_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.pin_plot = SamplePlot()
        self.magnifier = DetectorTextureView()
        self.magnifier.setMinimumSize(140, 100)
        self.magnifier.setMaximumHeight(140)
        self.magnifier.show_crosshair = False
        self.magnifier_label = QLabel("Magnifier: select an image")
        self.magnifier_label.setWordWrap(True)
        self.magnifier_label.setFixedHeight(2 * self.fontMetrics().lineSpacing())
        self.magnifier_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        for widget in (
            self.pin_button,
            self.unpin_button,
            self.pin_axis,
            self.pin_status,
            self.pin_plot,
            self.magnifier_label,
            self.magnifier,
        ):
            sidebar.addWidget(widget)
        grid.addLayout(sidebar, 0, 2, 3, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid, 1)
        footer = QHBoxLayout()
        self.export_button = QPushButton("Export comparison + cuts")
        self.cancel_button = QPushButton("Cancel preparation")
        self.cancel_button.setEnabled(False)
        self.message = QLabel(
            "Measured/model/residual intensity comparison unavailable: no admitted intensity result"
        )
        self.message.setWordWrap(True)
        self.message.setFixedHeight(2 * self.fontMetrics().lineSpacing())
        self.message.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        footer.addWidget(self.export_button)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.message, 1)
        layout.addLayout(footer)
        self.active.currentIndexChanged.connect(self._sync_active)
        self.active.currentIndexChanged.connect(self.state_changed)
        for widget in (self.row_width, self.column_width):
            widget.valueChanged.connect(self._query)
        self.measure.currentIndexChanged.connect(self._query)
        self.scope.currentIndexChanged.connect(self._query)
        self.pin_center.toggled.connect(self._query)
        self.mode.currentIndexChanged.connect(self._color_mode)
        self.link.toggled.connect(
            lambda enabled: (
                self._navigate(self.active.currentIndex()) if enabled else self.state_changed.emit()
            )
        )
        self.lock_limits.toggled.connect(self._limits)
        self.frames_button.clicked.connect(self.frames_requested)
        self.fit_button.clicked.connect(
            lambda: self.panels[self.active.currentIndex()].fit_button.click()
        )
        self.auto_button.clicked.connect(
            lambda: self.panels[self.active.currentIndex()].auto_button.click()
        )
        self.draw_line.toggled.connect(
            lambda enabled: self.panels[self.active.currentIndex()].view.set_line_mode(enabled)
        )
        self.apply_line_button.clicked.connect(self._apply_line)
        self.clear_line_button.clicked.connect(self._clear_line)
        self.pin_button.clicked.connect(self.pin_current)
        self.unpin_button.clicked.connect(self.unpin)
        self.pin_axis.currentIndexChanged.connect(self._show_pin)
        self.pin_axis.currentIndexChanged.connect(self.state_changed)
        self.export_button.clicked.connect(self.export_requested)
        self._sync_active()

    def _choose(self, slot: int) -> None:
        chosen = self.selectors[slot].currentData()
        if chosen == self.desired[slot]:
            return
        self.desired[slot] = chosen
        self.ready[slot] = False
        old = self.acquisitions[slot]
        self.labels[slot].setText(
            f"Preparing {chosen}; prior {old.name if old else 'none'} retained"
        )
        self.panels[slot].set_mask_pending(True)
        self.samples[slot] = None
        self.cut_keys[slot] = None
        self.cut_pending[slot] = False
        self.cut_plots[slot].set_samples(None, None, None, "Preparing current image")
        self.selection_requested.emit(slot, chosen)
        self.state_changed.emit()

    def set_choices(self, acquisitions: tuple[Acquisition, ...]) -> None:
        for slot, selector in enumerate(self.selectors):
            selector.blockSignals(True)
            selector.clear()
            selector.addItem("Choose acquisition", None)
            for acquisition in acquisitions:
                selector.addItem(acquisition.name, acquisition.acquisition_id)
            selector.setCurrentIndex(max(0, selector.findData(self.desired[slot])))
            selector.blockSignals(False)

    def unavailable(self, slot: int, reason: str) -> None:
        self.ready[slot] = False
        self.frames[slot] = None
        self.labels[slot].setText(
            f"{self.desired[slot]} unavailable: {reason}; prior snapshot retained"
        )
        self.update_frames()

    def admit(
        self,
        slot: int,
        acquisition: Acquisition,
        plane: PreparedOsc,
        mask: PreparedMask | None = None,
        frame: str | None = None,
    ) -> None:
        if (
            acquisition.acquisition_id != self.desired[slot]
            or acquisition.source_sha256 != plane.decoded_sha256
        ):
            return
        panel = self.panels[slot]
        changed = (
            self.planes[slot] is not plane
            or panel.acquisition_identity != acquisition.acquisition_id
        )
        self.acquisitions[slot] = acquisition
        self.planes[slot] = plane
        self.frames[slot] = frame
        if changed:
            panel.set_prepared_image(
                plane.native_counts,
                plane.display,
                plane.profiles,
                plane.full_profiles,
                plane.low_value,
                plane.high_value,
                plane.max_value,
                plane.min_positive,
                acquisition.acquisition_id,
            )
            restore = self._restore[slot]
            self._restore[slot] = None
            if restore is not None:
                v = panel.view
                v.zoom = restore.zoom
                v.pan = QPointF(restore.pan_x_px, restore.pan_y_px)
                v.scale_mode = restore.scale_mode
                v.crosshair = (restore.column_px, restore.row_px)
                panel.restore_profile_state(restore)
                v.set_levels(restore.low_value, restore.high_value, mode=restore.contrast_mode)
            panel.view.inspection_line = self.lines[slot]
        expected = acquisition.mask.revision if acquisition.mask else 0
        matched = (
            mask is not None
            and mask.mask.revision == expected
            and mask.mask.source_sha256 == acquisition.source_sha256
        )
        if matched and (panel.view.mask_reasons is not mask.reasons or panel._mask_pending):
            panel.publish_mask(mask)
        self.ready[slot] = not acquisition.mask or matched
        panel.set_mask_pending(not self.ready[slot])
        if not self.ready[slot]:
            panel._profile_pending = True
            self.profiles_requested.emit()
        exposure = (
            f"{acquisition.metadata.exposure_s:g} s"
            if acquisition.metadata.exposure_s is not None
            else "unknown"
        )
        self.labels[slot].setText(
            f"{'A' if slot == 0 else 'B'}: {acquisition.name}; {acquisition.acquisition_id}\nsource {acquisition.source_sha256[:12]}; exposure {exposure}; raw counts\nmask {expected}; {'current' if self.ready[slot] else 'old / preparing'}"
        )
        self.labels[slot].setToolTip(
            f"Source {acquisition.source_path}\nDecoded SHA-256 {acquisition.source_sha256}\nVerified at admission; no correction or normalization"
        )
        self.update_frames()
        if self.lines[slot] is None and self.cut_plots[slot].title != "No line defined":
            self.cut_plots[slot].set_samples(None, None, None, "No line defined")
        self.refresh_cut(slot)
        self._restore_pin()
        self._sync_active()

    def update_frames(self) -> None:
        compatible = (
            all(self.ready) and self.frames[0] is not None and self.frames[0] == self.frames[1]
        )
        self.link.setEnabled(compatible)
        if not compatible:
            self.link.setChecked(False)
        self.frame_status.setText(
            "Linked frame available: matching verified LAB detector calibration; raw counts"
            if compatible
            else "Independent native pixel frames; link unavailable: missing or incompatible verified detector calibration"
        )
        self.lock_limits.setEnabled(all(self.ready))
        settled = (
            all(self.ready)
            and not any(self.cut_pending)
            and not any(p._profile_pending for p in self.panels)
            and all(p._profile_key == p._query_key() for p in self.panels)
            and not self.pin_pending
            and all(
                line is None or self.cut_keys[slot] == self.line_key(slot)
                for slot, line in enumerate(self.lines)
            )
        )
        self.export_button.setEnabled(settled)
        if all(self.ready) and self._restored_limits is not None:
            limits, self._restored_limits = self._restored_limits, None
            self.lock_limits.setChecked(True)
            self.panels[self.active.currentIndex()].view.set_levels(
                limits[0], limits[1], mode=limits[2]
            )
        if compatible and self._restored_link:
            self._restored_link = False
            self.link.setChecked(True)

    def _navigate(self, slot: int) -> None:
        if self._link_guard:
            return
        self._link_guard = True
        try:
            source = self.panels[slot].view
            target = self.panels[1 - slot].view
            if self.link.isChecked() and all(self.ready):
                rect = source._rect()
                rows, cols = source.image.shape
                scale = rect.width() / cols
                center = (
                    (source.width() / 2 - rect.x()) / scale - 0.5,
                    (source.height() / 2 - rect.y()) / scale - 0.5,
                )
                target.scale_mode = "fit"
                target.zoom = scale / min(target.width() / cols, target.height() / rows)
                target.pan = QPointF(
                    (cols / 2 - center[0] - 0.5) * scale,
                    (rows / 2 - center[1] - 0.5) * scale,
                )
                target._request_paint()
                target.view_state_changed.emit()
            if self.lock_limits.isChecked() and all(self.ready):
                other = self.panels[1 - slot].view
                other.set_levels(source.low_value, source.high_value, mode=source.contrast_mode)
            self.refresh_magnifier()
            self.state_changed.emit()
        finally:
            self._link_guard = False

    def _limits(self, enabled: bool) -> None:
        if self._link_guard:
            return
        if enabled:
            self._independent_levels = [
                (p.view.low_value, p.view.high_value, p.view.contrast_mode) for p in self.panels
            ]
            self._navigate(self.active.currentIndex())
        else:
            self._link_guard = True
            try:
                for panel, levels in zip(self.panels, self._independent_levels, strict=True):
                    if panel.view.image is not None and levels is not None:
                        panel.view.set_levels(levels[0], levels[1], mode=levels[2])
            finally:
                self._link_guard = False
        self._sync_active()
        self.state_changed.emit()

    def _color_mode(self) -> None:
        slot = self.active.currentIndex()
        panel = self.panels[slot]
        if not self.ready[slot]:
            return
        try:
            panel.view.auto_levels(self.mode.currentText())
        except ValueError as exc:
            self.message.setText(str(exc))
            self._sync_active()

    def _query(self) -> None:
        p = self.panels[self.active.currentIndex()]
        p.row_width_control.setValue(self.row_width.value())
        p.column_width_control.setValue(self.column_width.value())
        p.profile_measure_control.setCurrentIndex(
            p.profile_measure_control.findData(self.measure.currentText())
        )
        p.profile_scope_control.setCurrentIndex(
            p.profile_scope_control.findData(self.scope.currentText())
        )
        p.pin_center_button.setChecked(self.pin_center.isChecked())
        self.state_changed.emit()

    def _sync_active(self) -> None:
        slot = self.active.currentIndex()
        p = self.panels[slot]
        for widget, value in (
            (self.row_width, p.row_width_control.value()),
            (self.column_width, p.column_width_control.value()),
        ):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        for widget, text in (
            (self.measure, p.profile_measure_control.currentData()),
            (self.scope, p.profile_scope_control.currentData()),
            (self.mode, p.view.contrast_mode),
        ):
            widget.blockSignals(True)
            widget.setCurrentText(text)
            widget.blockSignals(False)
        self.pin_center.blockSignals(True)
        self.pin_center.setChecked(not p.view.follow_cursor)
        self.pin_center.blockSignals(False)
        self.draw_line.blockSignals(True)
        self.draw_line.setChecked(p.view.line_mode)
        self.draw_line.blockSignals(False)
        line = self.lines[slot]
        if line is not None:
            for spin, value in zip(
                self.endpoints, (*line.start_column_row, *line.end_column_row), strict=True
            ):
                spin.setValue(value)
                spin.setToolTip(f"Exact native coordinate: {value:.17g} px")
            self.spacing.setValue(line.spacing_px)
        self.refresh_magnifier()
        self._show_pin()

    def _line_mode(self, slot: int, enabled: bool) -> None:
        if slot == self.active.currentIndex():
            self.draw_line.blockSignals(True)
            self.draw_line.setChecked(enabled)
            self.draw_line.blockSignals(False)

    def _cursor(self, slot: int) -> None:
        if slot == self.active.currentIndex():
            self.refresh_magnifier()

    def _apply_line(self) -> None:
        self.set_line(
            self.active.currentIndex(),
            tuple(s.value() for s in self.endpoints[:2]),
            tuple(s.value() for s in self.endpoints[2:]),
        )

    def set_line(self, slot: int, start: tuple[float, float], end: tuple[float, float]) -> None:
        try:
            line = LineDefinition(
                tuple(start),
                tuple(end),
                self.spacing.value(),
                (self.lines[slot].revision + 1) if self.lines[slot] else 1,
            )
        except ValueError as exc:
            self.message.setText(str(exc))
            return
        self.lines[slot] = line
        self.panels[slot].view.inspection_line = line
        self.panels[slot].view._request_paint()
        self.refresh_cut(slot)
        self._sync_active()
        self.state_changed.emit()

    def _clear_line(self) -> None:
        slot = self.active.currentIndex()
        self.lines[slot] = None
        self.samples[slot] = None
        self.cut_keys[slot] = None
        self.cut_pending[slot] = False
        self.panels[slot].view.inspection_line = None
        self.panels[slot].view._request_paint()
        self.cut_plots[slot].set_samples(None, None, None, "No line defined")
        self.update_frames()
        self.state_changed.emit()

    def line_key(self, slot: int) -> tuple[object, ...] | None:
        p = self.panels[slot]
        a = self.acquisitions[slot]
        return (
            None
            if a is None or self.lines[slot] is None
            else (
                a.acquisition_id,
                a.source_sha256,
                id(p.view.image),
                p.view.data_revision,
                p.view.mask_revision,
                id(p._mask_inclusion),
                self.lines[slot],
            )
        )

    def refresh_cut(self, slot: int) -> None:
        key = self.line_key(slot)
        if key is None or not self.ready[slot] or self.panels[slot]._mask_pending:
            return
        if key == self.cut_keys[slot]:
            return
        line = self.lines[slot]
        p = self.panels[slot]
        self.cut_pending[slot] = True
        self.update_frames()
        if line.sample_count > 2048:
            self.cut_plots[slot].title = "Preparing line; prior samples shown"
            self.cut_plots[slot].update()
            self.cut_requested.emit(slot)
        else:
            self.publish_cut(slot, key, sample_line(p.view.image, p._mask_inclusion, line))

    def publish_cut(self, slot: int, key: tuple[object, ...], samples: LineSamples) -> None:
        if key != self.line_key(slot) or not self.ready[slot]:
            return
        self.samples[slot] = samples
        self.cut_keys[slot] = key
        self.cut_pending[slot] = False
        self.cut_plots[slot].set_samples(
            samples.distance_px,
            samples.values,
            samples.support,
            f"{'A' if slot == 0 else 'B'} line r{samples.definition.revision}: nearest counts; support 0/1",
        )
        self.update_frames()

    def pin_current(self) -> None:
        slot = self.active.currentIndex()
        p = self.panels[slot]
        a = self.acquisitions[slot]
        if (
            not self.ready[slot]
            or p._mask_pending
            or p._profile_pending
            or p._current_profiles is None
        ):
            return
        self.pin_identity = PinIdentity(
            a.acquisition_id,
            a.source_sha256,
            p.view.image.shape,
            p.view.mask_revision,
            p.profile_query(),
        )
        self.pin = PinnedProfiles.capture(self.pin_identity, p._current_profiles)
        self.pin_unavailable_reason = None
        self.pin_pending = False
        self._show_pin()
        self.state_changed.emit()

    def unpin(self) -> None:
        self.pin_identity = self.pin = None
        self.pin_unavailable_reason = None
        self.pin_pending = False
        self._show_pin()
        self.state_changed.emit()

    def _restore_pin(self) -> None:
        if (
            self.pin is not None
            or self.pin_identity is None
            or self.pin_unavailable_reason is not None
        ):
            return
        for slot, a in enumerate(self.acquisitions):
            p = self.panels[slot]
            identity = self.pin_identity
            if (
                a is not None
                and self.ready[slot]
                and (
                    a.acquisition_id,
                    a.source_sha256,
                    p.view.image.shape,
                    p.view.mask_revision,
                    p.profile_query(),
                )
                == (
                    identity.acquisition_id,
                    identity.source_sha256,
                    identity.native_shape_rc,
                    identity.mask_revision,
                    identity.query,
                )
                and not p._profile_pending
            ):
                self.pin = PinnedProfiles.capture(identity, p._current_profiles)
                self.pin_pending = False
                self._show_pin()
                return
        self.pin_pending = True
        self.profiles_requested.emit()
        self._show_pin()

    def publish_pin(self, identity: PinIdentity, profiles: BandProfiles) -> None:
        if identity != self.pin_identity:
            return
        self.pin = PinnedProfiles.capture(identity, profiles)
        self.pin_pending = False
        self._show_pin()

    def _show_pin(self) -> None:
        identity = self.pin_identity
        if identity is None:
            self.pin_status.setText("No reference pinned")
            if self._pin_plot_key is not None:
                self.pin_plot.set_samples(None, None, None, "Reference unavailable")
                self._pin_plot_key = None
            return
        label = f"Reference {str(identity.acquisition_id)[:12]}\nsource {identity.source_sha256[:10]}; mask {identity.mask_revision}"
        current = next(
            (
                a
                for a in self.acquisitions
                if a is not None and a.acquisition_id == identity.acquisition_id
            ),
            None,
        )
        stale = current is not None and (
            current.source_sha256 != identity.source_sha256
            or (current.mask.revision if current.mask else 0) != identity.mask_revision
        )
        self.pin_status.setText(
            label
            + "\n"
            + (
                "Unavailable / stale; prior samples retained"
                if self.pin_unavailable_reason is not None
                else "Stale; immutable prior samples"
                if stale
                else "Frozen samples retained"
                if self.pin is not None
                else "Pending rebind / unavailable"
            )
        )
        self.pin_status.setToolTip(
            str(identity)
            + (f"\n{self.pin_unavailable_reason}" if self.pin_unavailable_reason else "")
        )
        if self.pin is not None:
            horizontal = self.pin_axis.currentText() == "horizontal"
            key = (id(self.pin), horizontal)
            if key == self._pin_plot_key:
                return
            self._pin_plot_key = key
            values = self.pin.horizontal if horizontal else self.pin.vertical
            support = self.pin.horizontal_support if horizontal else self.pin.vertical_support
            self.pin_plot.set_samples(
                np.arange(values.size, dtype=np.float64),
                values,
                support,
                f"Pinned {self.pin_axis.currentText()}: {identity.query[6]}",
                unit="counts" if identity.query[6] == "sum" else "counts / valid pixel",
            )

    def refresh_magnifier(self) -> None:
        slot = self.active.currentIndex()
        p = self.panels[slot]
        v = p.view
        if not self.ready[slot] or v.image is None:
            return
        key = (
            slot,
            id(v.image),
            id(v.mask_reasons),
            v.crosshair,
            v.low_value,
            v.high_value,
            v.contrast_mode,
        )
        if key == self._magnifier_key:
            return
        self._magnifier_key = key
        c, r = v.crosshair
        rows, cols = v.image.shape
        c0, c1 = max(0, c - 20), min(cols, c + 21)
        r0, r1 = max(0, r - 20), min(rows, r + 21)
        native = np.ascontiguousarray(v.image[r0:r1, c0:c1])
        display = np.ascontiguousarray(native, dtype=np.float32)
        native.setflags(write=False)
        display.setflags(write=False)
        self.magnifier.set_prepared_image(
            native,
            display,
            self.planes[slot].low_value,
            self.planes[slot].high_value,
            self.planes[slot].max_value,
            self.planes[slot].min_positive,
        )
        self.magnifier.set_levels(v.low_value, v.high_value, mode=v.contrast_mode)
        self.magnifier.show_crosshair = False
        if v.mask_reasons is not None:
            reasons = np.ascontiguousarray(v.mask_reasons[r0:r1, c0:c1])
            reasons.setflags(write=False)
            self.magnifier.set_mask_plane(reasons, v.mask_revision)
        self.magnifier_label.setText(
            f"{'A' if slot == 0 else 'B'} magnifier c[{c0},{c1}) r[{r0},{r1}); native counts\n41 px maximum; masked/nonfinite samples remain explicit"
        )

    def capture_state(self, visible: bool = False) -> ComparisonState:
        views = [
            panel_view_state(p) if self.ready[slot] else self._restore[slot]
            for slot, p in enumerate(self.panels)
        ]
        shared = None
        if self.lock_limits.isChecked() and all(self.ready):
            v = self.panels[self.active.currentIndex()].view
            shared = (v.low_value, v.high_value, v.contrast_mode)
            for slot, levels in enumerate(self._independent_levels):
                if views[slot] is not None and levels is not None:
                    views[slot] = replace(
                        views[slot],
                        low_value=levels[0],
                        high_value=levels[1],
                        contrast_mode=levels[2],
                    )
        return ComparisonState(
            tuple(self.desired),
            tuple(views),
            tuple(self.lines),
            self.pin_identity,
            self.pin_axis.currentText(),
            self.active.currentIndex(),
            self.link.isChecked(),
            shared,
            visible,
        )

    def restore_state(self, state: ComparisonState) -> None:
        self.acquisitions = [None, None]
        self.planes = [None, None]
        self.ready = [False, False]
        self.frames = [None, None]
        self.samples = [None, None]
        self.cut_keys = [None, None]
        self.cut_pending = [False, False]
        self._magnifier_key = self._pin_plot_key = None
        self.link.setChecked(False)
        self.lock_limits.setChecked(False)
        self.desired = list(state.acquisition_ids)
        self._restore = list(state.views)
        self.lines = list(state.lines)
        self.pin_identity = state.pin
        self.pin = None
        self.pin_unavailable_reason = None
        self.pin_pending = state.pin is not None
        self.active.setCurrentIndex(state.active_slot)
        self.pin_axis.setCurrentText(state.pin_axis)
        self._restored_link = state.linked_navigation
        self._restored_limits = state.shared_limits
        for panel, line in zip(self.panels, self.lines, strict=True):
            panel.view.inspection_line = line
        self.update_frames()
        self._show_pin()

    def release_resources(self) -> None:
        for view in (*[p.view for p in self.panels], self.magnifier):
            view.release_resources()

    def capture_figure(self) -> QImage:
        """Capture both admitted native views and the displayed cuts and reference."""
        if not self.isVisible() or not self.export_button.isEnabled():
            raise ValueError("Comparison must be visible and settled for figure export")
        columns = []
        for slot, panel in enumerate(self.panels):
            detector = panel.capture_inspection_figure(
                self.labels[slot].text().replace("\n", " | ")
            )
            cut = self.cut_plots[slot].grab().toImage()
            columns.append((detector, cut))
        pin = self.pin_plot.grab().toImage()
        magnifier = self.magnifier.grabFramebuffer()
        if magnifier.isNull():
            raise ValueError("Magnifier capture is unavailable")
        columns.append((pin, magnifier))
        widths = [max(image.width() for image in column) for column in columns]
        height = max(sum(image.height() for image in column) for column in columns)
        width = sum(widths)
        if width * height * 4 > 16 * 1024 * 1024:
            raise ValueError("Comparison figure exceeds the 16 MiB display buffer limit")
        figure = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        figure.fill(QColor(24, 28, 34))
        painter = QPainter(figure)
        x = 0
        for column, column_width in zip(columns, widths, strict=True):
            y = 0
            for image in column:
                painter.drawImage(
                    QRectF(x, y, image.width(), image.height()),
                    image,
                    QRectF(0, 0, image.width(), image.height()),
                )
                y += image.height()
            x += column_width
        painter.end()
        figure.setDevicePixelRatio(self.devicePixelRatioF())
        return figure
