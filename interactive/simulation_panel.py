"""Independent configured forms and shared detector inspection, without experiment bindings."""

import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml
from comparison_panel import panel_view_state
from detector_panel import DetectorPanel
from mask_state import MaskWork, NativeMask
from native_simulation_panel import NativeDraftPanel
from native_simulation_state import (
    NativeSimulationDraft,
    NativeSimulationReference,
    native_draft_document,
    native_reference_document,
)
from parameter_state import FieldChange, SessionHistory, _action
from project_state import ProjectFormatError
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from simulation_contrast import SimulationContrast
from simulation_dashboard import SimulationDashboard
from simulation_fields import SIMULATION_FIELDS
from simulation_io import SimulationFrame
from simulation_state import (
    SimulationDraft,
    SimulationExportWork,
    SimulationReference,
    simulation_draft_document,
)


class ArrayEditor(QWidget):
    changed = Signal()

    def __init__(self, kind: str, value, parent=None) -> None:
        super().__init__(parent)
        self.kind = kind
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget()
        if kind == "rotations":
            headers = (
                "axis X",
                "axis Y",
                "axis Z",
                "angle deg",
                "pivot X m",
                "pivot Y m",
                "pivot Z m",
            )
            rows = [[*v["axis_lab"], v["angle_deg"], *v["pivot_lab_m"]] for v in value]
        elif kind == "sequence":
            headers, rows = ("Value",), [[v] for v in value]
        elif kind in ("matrix", "axes"):
            headers, rows = ("X", "Y", "Z"), value
        else:
            headers = (
                ("row", "column")
                if kind == "integer_vector2"
                else ("X / column", "Y / row", "Z")[: 2 if kind == "vector2" else 3]
            )
            rows = [value]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, item in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(str(item)))
        self.table.setMaximumHeight(140 if kind not in ("rotations", "sequence") else 210)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        if kind in ("rotations", "sequence"):
            actions = QHBoxLayout()
            add, remove = QPushButton("Add row"), QPushButton("Remove selected row")
            actions.addWidget(add)
            actions.addWidget(remove)
            layout.addLayout(actions)
            add.clicked.connect(self._add)
            remove.clicked.connect(self._remove)
        self.table.cellChanged.connect(lambda *_: self.changed.emit())

    def _add(self) -> None:
        if self.table.rowCount() >= 512:
            return
        values = [1, 0, 0, 0, 0, 0, 0] if self.kind == "rotations" else [0]
        row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)
        for column, value in enumerate(values):
            self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.table.blockSignals(False)
        self.changed.emit()

    def _remove(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self.changed.emit()

    def value(self):
        rows = []
        for r in range(self.table.rowCount()):
            values = []
            for c in range(self.table.columnCount()):
                cell = self.table.item(r, c)
                text = cell.text().strip() if cell is not None else ""
                try:
                    value = int(text) if self.kind == "integer_vector2" else float(text)
                except ValueError:
                    value = text
                values.append(value)
            rows.append(values)
        if self.kind == "rotations":
            return [{"axis_lab": v[:3], "angle_deg": v[3], "pivot_lab_m": v[4:]} for v in rows]
        if self.kind == "sequence":
            return [v[0] for v in rows]
        return rows if self.kind in ("matrix", "axes") else rows[0]


class FieldEditor(QWidget):
    changed = Signal()

    def __init__(self, field, value, present: bool, parent=None) -> None:
        super().__init__(parent)
        self.field = field
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.included = QCheckBox("Declare this optional value") if field.optional else None
        if self.included is not None:
            self.included.setChecked(present)
            layout.addWidget(self.included)
        if field.kind in (
            "vector2",
            "vector3",
            "integer_vector2",
            "matrix",
            "axes",
            "rotations",
            "sequence",
        ):
            self.control = ArrayEditor(field.kind, value, self)
            self.control.changed.connect(self.changed)
        elif field.kind == "bool":
            self.control = QCheckBox("Enabled")
            self.control.setChecked(bool(value))
            self.control.toggled.connect(self.changed)
        elif field.kind == "choice":
            self.control = QComboBox()
            self.control.addItems(field.choices)
            if value not in field.choices:
                self.control.addItem(str(value))
            self.control.setCurrentText(str(value))
            self.control.currentTextChanged.connect(self.changed)
        else:
            self.control = QLineEdit("" if value is None else str(value))
            self.control.editingFinished.connect(self.changed)
        self.control.setObjectName("simulation_" + field.path)
        self.control.setEnabled(present or not field.optional)
        layout.addWidget(self.control)
        default_note = (
            f"default: {field.default!r}" if field.optional else "required explicit value"
        )
        note = QLabel(
            f"{field.unit} | {field.domain} | {default_note}\n{field.description}\n{field.applicability}"
        )
        note.setWordWrap(True)
        note.setObjectName("mutedText")
        layout.addWidget(note)
        if self.included is not None:
            self.included.toggled.connect(self.control.setEnabled)
            self.included.toggled.connect(self.changed)

    def value(self):
        kind = self.field.kind
        if isinstance(self.control, ArrayEditor):
            return self.control.value()
        if kind == "choice":
            return self.control.currentText()
        if kind == "bool":
            return self.control.isChecked()
        text = self.control.text().strip()
        try:
            if kind == "int":
                return int(text)
            if kind in ("float", "nullable_float"):
                return None if kind == "nullable_float" and not text else float(text)
        except ValueError:
            self.control.setStyleSheet("border:1px solid #d88370")
            return text
        self.control.setStyleSheet("")
        return text


class SimulationScatterView(QWidget):
    """Retained sampled Qx/Qz display; complete canonical arrays stay in the result."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.points = QPolygonF()
        self.colors = ()
        self.title = "No configured reciprocal output"
        self.setMinimumSize(250, 200)

    def set_data(self, points: np.ndarray, density: np.ndarray, title: str) -> None:
        self.points = QPolygonF()
        self.colors = ()
        if density.shape != (points.shape[0],):
            raise ValueError("scatter coordinates and canonical density must align")
        if points.size:
            valid = np.all(np.isfinite(points), axis=1)
            indices = np.flatnonzero(valid)
            if len(indices) > 512:
                indices = indices[np.linspace(0, len(indices) - 1, 512, dtype=np.intp)]
            sample = points[indices][:, (0, 2)]
            if len(sample):
                low, high = sample.min(axis=0), sample.max(axis=0)
                span = np.maximum(high - low, 1e-12)
                normalized = (sample - low) / span
                self.points = QPolygonF([QPointF(float(x), float(y)) for x, y in normalized])
                displayed = density[indices]
                finite = np.isfinite(displayed)
                lower = float(np.min(displayed, where=finite, initial=np.inf))
                upper = float(np.max(displayed, where=finite, initial=-np.inf))
                span_density = max(upper - lower, 1e-300) if np.any(finite) else 1.0
                levels = np.clip((displayed - lower) / span_density, 0, 1)
                self.colors = tuple(
                    QColor.fromRgbF(float(level), 0.4, 1.0 - float(level))
                    if valid
                    else QColor(130, 130, 130)
                    for level, valid in zip(levels, finite, strict=True)
                )
                title += f"; linear density colors [{lower:.4g},{upper:.4g}], nonfinite values grey"
        else:
            self.points = QPolygonF()
            self.colors = ()
        self.title = (
            title
            + f"; drawing {len(self.points)} points, full values retained; sample Qx/Qz (angstrom^-1)"
        )
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 28, 34))
        painter.setPen(QColor(225, 237, 241))
        painter.drawText(
            self.rect().adjusted(8, 5, -8, -5),
            Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            self.title,
        )
        painter.translate(20, self.height() - 20)
        painter.scale(max(1, self.width() - 40), -max(1, self.height() - 85))
        painter.setPen(QPen(QColor(99, 185, 201), 0))
        for point, color in zip(self.points, self.colors, strict=True):
            painter.setPen(QPen(color, 0))
            painter.drawPoint(point)
        painter.end()


class SimulatorPanel(QWidget):
    def __init__(self, shell) -> None:
        super().__init__(shell)
        self.shell = shell
        self.draft: SimulationDraft | None = None
        self.result_reference: SimulationReference | None = None
        self.validated: SimulationDraft | None = None
        self.frame: SimulationFrame | None = None
        self.latest_frame: SimulationFrame | None = None
        self.history = SessionHistory()
        self.epoch = 0
        self.hold = False
        self._inspection_pending = False
        self._mapping = None
        self.editors = {}
        self._restoring = False
        self._pending_detector_state = None
        self._transfer_context = None
        self._transfer_request = None
        self._transfer_review = None
        self._transfer_dialog = None
        self._update_request = None
        self._last_requested = None
        self._pending_quick = None
        self._fresh_live_pending = False
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(300)
        self._live_timer.timeout.connect(self.request_update)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        controls = QHBoxLayout()
        self.live = QCheckBox("&Live")
        self.live.setToolTip(
            "Validate and run the latest complete draft after a 300 ms pause; nominal results."
        )
        self.advanced_button = QPushButton("&Advanced/actions")
        controls.addWidget(self.live)
        self.advanced = QDialog(self)
        self.advanced.setWindowTitle("Simulator — complete editor and actions")
        self.advanced.resize(760, 640)
        advanced_layout = QVBoxLayout(self.advanced)
        action_scroll = QScrollArea()
        action_scroll.setWidgetResizable(True)
        action_content = QWidget()
        action_layout = QVBoxLayout(action_content)
        action_scroll.setWidget(action_content)
        advanced_layout.addWidget(action_scroll, 1)
        secondary = QFormLayout()
        action_layout.addLayout(secondary)
        self.load_button = QPushButton("Load configuration")
        self.validate_button = QPushButton("Validate complete draft")
        self.run_button = QPushButton("&Run/update")
        self.inspect_button = QPushButton("Inspect this snapshot")
        self.resume_button = QPushButton("Follow progression")
        self.cancel_button = QPushButton("&Stop")
        controls.addWidget(self.run_button)
        controls.addWidget(self.cancel_button)
        controls.addStretch()
        controls.addWidget(self.advanced_button)
        secondary.addRow(self.load_button, self.validate_button)
        secondary.addRow(self.inspect_button, self.resume_button)
        layout.addLayout(controls)
        actions = QFormLayout()
        self.undo_button = QPushButton("Undo draft")
        self.redo_button = QPushButton("Redo draft")
        self.save_configuration_button = QPushButton("Export configuration YAML")
        self.export_button = QPushButton("Export exact snapshot")
        self.reopen_button = QPushButton("Reopen saved snapshot")
        for button in (
            self.undo_button,
            self.redo_button,
            self.save_configuration_button,
            self.export_button,
            self.reopen_button,
        ):
            actions.addRow(button)
        action_layout.addLayout(actions)
        transfer_actions = QHBoxLayout()
        self.transfer_target = QComboBox()
        self.transfer_target.addItem("Experiment -> configured draft", "configured")
        for recipe in ("bi2se3", "bi2te3", "gd1", "sid1", "clean1", "b4"):
            self.transfer_target.addItem("Experiment -> native " + recipe, recipe)
        self.transfer_button = QPushButton("Review experiment transfer")
        self.transfer_button.clicked.connect(self.review_transfer)
        transfer_actions.addWidget(self.transfer_target)
        transfer_actions.addWidget(self.transfer_button)
        action_layout.addLayout(transfer_actions)
        self.status = QLabel(
            "Load a supported rasim-simulation-v2 configuration. No acquisition or fit is required."
        )
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(42)
        layout.addWidget(self.status)
        self.draft_kind = QComboBox()
        self.draft_kind.addItem("Independent configured YAML", "configured")
        self.draft_kind.addItem("Independent native recipe", "native")
        action_layout.addWidget(self.draft_kind)
        splitter = QSplitter()
        forms = QWidget()
        forms_layout = QVBoxLayout(forms)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search parameters, units, domains and descriptions")
        forms_layout.addWidget(self.search)
        run_form = QFormLayout()
        self.route = QComboBox()
        for title, key in (
            ("Native pixel MC mass", "monte_carlo"),
            ("Native pixel-center density (display)", "pixel_centers"),
            ("Macrobin display preview", "macrobins"),
            ("Reciprocal display only", "reciprocal_space"),
            ("Nominal Ewald display only", "ewald_surface"),
        ):
            self.route.addItem(title, key)
        self.position = QComboBox()
        self.position.addItem("Sample source position", "sampled")
        self.position.addItem("Smooth conditional-position integral", "conditional_position")
        self.draw_count = QLineEdit("8")
        self.detector_seed = QLineEdit("1729")
        for label, widget in (
            ("Detector route", self.route),
            ("Beam position terminal", self.position),
            ("MC draws per source state", self.draw_count),
            ("Detector seed (separate from source seed)", self.detector_seed),
        ):
            run_form.addRow(label, widget)
        forms_layout.addLayout(run_form)
        self.groups = QTabWidget()
        self._group_layouts = {}
        self._group_order = []
        self.groups.currentChanged.connect(self._show_group)
        forms_layout.addWidget(self.groups, 1)
        forms.setMinimumWidth(300)
        self.form_stack = QStackedWidget()
        self.form_stack.addWidget(forms)
        self.native = NativeDraftPanel(self)
        self.form_stack.addWidget(self.native)
        action_layout.addWidget(self.form_stack, 1)
        self.form_stack.setMinimumHeight(380)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.advanced.hide)
        advanced_layout.addWidget(close)
        self.dashboard = SimulationDashboard(self)
        splitter.addWidget(self.dashboard)
        self.outputs = QTabWidget()
        self.detector = DetectorPanel(compact=True)
        for control in (
            self.detector.fit_button,
            self.detector.mode_control,
            self.detector.pin_center_button,
        ):
            control.parentWidget().show()
        self.detector.observable_unit = "angstrom^2 (simulation mass, not experimental counts)"
        self.detector.quantitative_ready = False
        self.display_contrast = SimulationContrast(self.detector)
        self.detector.profile_work_required = True
        self.detector.coalesce_profile_updates = True
        for widget in (
            self.detector.mask_tool,
            self.detector.mask_reason,
            self.detector.brush_radius,
            self.detector.mask_import_button,
            self.detector.mask_undo_button,
            self.detector.mask_redo_button,
            self.detector.mask_cancel_button,
        ):
            widget.setEnabled(False)
        self.detector.mask_hint.setText("Independent simulation; no acquisition exclusion editor")
        self.detector.export_button.hide()
        self.reciprocal = SimulationScatterView()
        self.ewald = SimulationScatterView()
        self.outputs.addTab(self.detector, "Detector / exact profiles")
        self.outputs.addTab(self.reciprocal, "Reciprocal output")
        self.outputs.addTab(self.ewald, "Ewald output")
        splitter.addWidget(self.outputs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([290, 850])
        self.dashboard_splitter = splitter
        layout.addWidget(splitter, 1)
        self.figures = QCheckBox(
            "Export configured figures using selected output names and directory"
        )
        self.figures.setChecked(True)
        action_layout.addWidget(self.figures)
        self.identity = QLabel("Awaiting quantitative snapshot")
        self.identity.setWordWrap(True)
        self.identity.setMinimumHeight(46)
        layout.addWidget(self.identity)
        self.draft_kind.currentIndexChanged.connect(self._kind_changed)
        self.load_button.clicked.connect(self.load)
        self.validate_button.clicked.connect(self.validate)
        self.run_button.clicked.connect(lambda: self.request_update(force=True))
        self.live.toggled.connect(self._live_changed)
        self.advanced_button.clicked.connect(self.advanced.show)
        self.inspect_button.clicked.connect(self.inspect)
        self.resume_button.clicked.connect(self.follow)
        self.cancel_button.clicked.connect(self.stop_live)
        self.save_configuration_button.clicked.connect(self.save_configuration)
        self.export_button.clicked.connect(self.export)
        self.reopen_button.clicked.connect(self.reopen)
        self.undo_button.clicked.connect(lambda: self.history_step(True))
        self.redo_button.clicked.connect(lambda: self.history_step(False))
        self.search.textChanged.connect(self.filter_fields)
        self.route.currentIndexChanged.connect(self._route_changed)
        self.position.currentIndexChanged.connect(self.propose)
        self.draw_count.editingFinished.connect(self.propose)
        self.detector_seed.editingFinished.connect(self.propose)
        self.draw_count.textEdited.connect(self._pending_control_edited)
        self.detector_seed.textEdited.connect(self._pending_control_edited)
        self.figures.toggled.connect(self.propose)
        self.detector.profile_requested.connect(self.request_profiles)
        self.detector.view.view_state_changed.connect(self.shell._mark_dirty)
        self.refresh()

    def update_context(self):
        return (
            self.shell.project.project_id,
            self.epoch,
            self.draft_kind.currentData(),
            self.active_draft,
        )

    def _live_changed(self, enabled):
        if enabled:
            self.request_update(force=True)
        else:
            self.stop_live()

    def stop_live(self):
        self._live_timer.stop()
        self._fresh_live_pending = False
        self.live.blockSignals(True)
        self.live.setChecked(False)
        self.live.blockSignals(False)
        self.flush_quick_edit()
        self._supersede()
        self.refresh()

    def update_failed(self, context):
        if self._update_request is not None and self._update_request[:2] == context:
            self._update_request = None

    def _route_changed(self):
        if not self._restoring:
            self.stop_live()
            self.propose()

    def input_pending(self, schedule=True):
        if self._restoring or self.native._restoring:
            return
        self._live_timer.stop()
        self._supersede()
        if schedule and self.live.isChecked():
            self._live_timer.start()

    def quick_pending(self, key, text, schedule):
        self._pending_quick = (self.draft_kind.currentData(), key, text) if schedule else None
        self.input_pending(schedule)

    def draft_changed(self):
        if self.live.isChecked():
            self._live_timer.start()

    def request_update(self, *, force=False):
        self._live_timer.stop()
        if (
            self.shell._close_intent
            or self.shell._pending_open is not None
            or self.shell.workspaces.currentIndex() != 1
        ):
            return
        if not self.flush_quick_edit():
            return
        native = self.draft_kind.currentData() == "native"
        if not (self.native.propose() if native else self.propose()):
            self._live_timer.stop()
            self._update_request = None
            return
        self._live_timer.stop()
        draft = self.active_draft
        if draft is None:
            self.status.setText("Load a complete draft in Advanced/actions before enabling Live.")
            return
        if not force and self._last_requested == (self.draft_kind.currentData(), draft):
            return
        self._last_requested = (self.draft_kind.currentData(), draft)
        validated = self.native.validated if native else self.validated
        if validated == draft:
            self.run()
        else:
            self._update_request = self.update_context()
            self.validate()

    def flush_quick_edit(self):
        pending, self._pending_quick = self._pending_quick, None
        if pending is None or pending[0] != self.draft_kind.currentData():
            return True
        return self.quick_edit(pending[1], pending[2])

    def quick_edit(self, key, text):
        self._pending_quick = None
        path, indices = key
        native = self.draft_kind.currentData() == "native"
        if path[0] == "run":
            control = getattr(self.native if native else self, path[1])
            control.setText(text)
        else:
            if native:
                self.native._show_group(self.native._group_order.index(str(path[0])))
                editor = self.native.editors[path][3]
            else:
                field_path = ".".join(path)
                self._show_group(self._group_order.index(str(path[0])))
                editor = self.editors[field_path][2]
            control = editor.control
            if indices:
                table = control if native else control.table
                row, column = (
                    indices
                    if len(indices) == 2
                    else ((indices[0], 0) if native else (0, indices[0]))
                )
                table.blockSignals(True)
                table.item(row, column).setText(text)
                table.blockSignals(False)
            else:
                control.setText(text)
        accepted = self.native.propose() if native else self.propose()
        if not accepted:
            self.input_pending(schedule=False)
        return accepted

    def quick_fields(self):
        native = self.draft_kind.currentData() == "native"
        if self.active_draft is None:
            return []
        entries = []

        def add(path, group, label, unit, value, indices=(), integer=False, probability=False):
            if value is None or type(value) is bool or not isinstance(value, (float, int, str)):
                return
            entries.append(
                ((tuple(path), tuple(indices)), group, label, unit, value, integer, probability)
            )

        def components(path, group, label, unit, value, names):
            if isinstance(value, (list, tuple)) and len(value) == len(names):
                for index, (name, item) in enumerate(zip(names, value, strict=True)):
                    add(path, group, f"{label} {name}", unit, item, (index,))

        geometry, mosaic, detector, beam, sample, sampling = (
            "Incident angle / Geometry",
            "Mosaic Broadening",
            "Detector",
            "Beam Controls",
            "Sample / Structure",
            "Sampling / Optics",
        )
        if native:
            for path, (value, unit, _label, _editor) in self.native.editors.items():
                name = str(path[-1])
                if path[0] == "specimen":
                    group = (
                        mosaic
                        if name
                        in (
                            "gaussian_sigma_rad",
                            "lorentzian_half_width_rad",
                            "lorentzian_probability",
                        )
                        else sample
                    )
                    add(
                        path,
                        group,
                        name.replace("_", " "),
                        unit,
                        value,
                        probability=name == "lorentzian_probability",
                    )
                elif path[0] == "source":
                    if name in ("mean_origin_lab_m", "divergence_sigma_rad", "spatial_sigma_m"):
                        components(
                            path,
                            beam,
                            name.replace("_", " "),
                            unit,
                            value,
                            ("X", "Y", "Z")
                            if name == "mean_origin_lab_m"
                            else ("axis 1", "axis 2"),
                        )
                    elif name in ("mean_wavelength_A", "common_wavelength_sigma_A"):
                        add(path, beam, name.replace("_", " "), unit, value)
                elif path[0] == "instrument":
                    if name == "translation_m" and path[1] in (
                        "lab_from_sample",
                        "lab_from_detector",
                    ):
                        components(
                            path,
                            geometry if path[1] == "lab_from_sample" else detector,
                            path[1].replace("_", " "),
                            "m LAB",
                            value,
                            ("X", "Y", "Z"),
                        )
                    elif name == "detector_reference_coordinate_px":
                        components(
                            path, detector, "Reference coordinate", "px", value, ("column", "row")
                        )
                    elif name in ("detector_row_pitch_m", "detector_column_pitch_m"):
                        add(path, detector, name.replace("_", " "), unit, value)
                elif path[0] in ("source_rule", "integration_rule") and type(value) in (float, int):
                    add(
                        path,
                        sampling,
                        ".".join(map(str, path[1:])),
                        unit,
                        value,
                        integer=type(value) is int,
                    )
            add(
                ("run", "repeats"),
                sample,
                "Coherent repeats",
                "cells",
                self.native.draft.coherent_repeats,
                integer=True,
            )
            add(
                ("run", "bin_size"),
                sampling,
                "Integrated rectangle width",
                "native px",
                self.native.draft.bin_size_px,
                integer=True,
            )
            return entries
        mapping = self._mapping
        if mapping is None:
            return []
        instrument = mapping["instrument"]
        rotations = instrument.get("axis_rotations", [])
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for index, rotation in enumerate(rotations):
            if not isinstance(rotation, dict):
                continue
            angle = rotation.get("angle_deg")
            incidence = (
                len(rotations) == 1
                and rotation.get("axis_lab") == [1, 0, 0]
                and mapping["source"].get("mean_direction_lab") == [0, 1, 0]
                and instrument["lab_from_goniometer_zero"]["rotation"] == identity
                and instrument["goniometer_from_sample"]["rotation"] == identity
                and isinstance(angle, (int, float))
                and -90 <= angle <= 90
            )
            label = (
                "Incident angle (+X rotation, +Y beam)"
                if incidence
                else f"Axis {index + 1} rotation, LAB {rotation.get('axis_lab')}"
            )
            add(("instrument", "axis_rotations"), geometry, label, "deg", angle, (index, 3))
        for field in SIMULATION_FIELDS:
            path = tuple(field.path.split("."))
            value = mapping
            for name in path:
                if not isinstance(value, dict) or name not in value:
                    value = None
                    break
                value = value[name]
            if path[0] == "mosaic" and field.kind == "float":
                add(
                    path,
                    mosaic,
                    field.label,
                    field.unit,
                    value,
                    probability=path[-1] == "lorentzian_probability",
                )
            elif path[-1] == "translation_m" and path[1] in (
                "goniometer_from_sample",
                "lab_from_detector",
            ):
                components(
                    path,
                    geometry if path[1] == "goniometer_from_sample" else detector,
                    "Sample offset" if path[1] == "goniometer_from_sample" else "Detector position",
                    "m GONIOMETER" if path[1] == "goniometer_from_sample" else "m LAB",
                    value,
                    ("X", "Y", "Z"),
                )
            elif path[-1] == "detector_reference_coordinate_px":
                components(path, detector, "Reference coordinate", "px", value, ("column", "row"))
            elif path[0] == "instrument" and (
                "detector_tilt" in path
                or path[-1] in ("detector_row_pitch_m", "detector_column_pitch_m")
            ):
                add(path, detector, field.label, field.unit, value)
            elif path[0] == "source" and path[-1] in (
                "spatial_sigma_m",
                "divergence_sigma_rad",
                "mean_origin_lab_m",
            ):
                components(
                    path,
                    beam,
                    field.label,
                    field.unit,
                    value,
                    ("X", "Y", "Z") if path[-1] == "mean_origin_lab_m" else ("axis 1", "axis 2"),
                )
            elif path[0] == "source" and field.kind == "float":
                add(path, beam, field.label, field.unit, value)
            elif path[0] == "structure_factor" and field.kind in ("int", "float"):
                add(path, sample, field.label, field.unit, value, integer=field.kind == "int")
            elif (path[0] == "numerics" and field.kind == "int") or (
                field.path
                in (
                    "source.sample_count",
                    "source.seed",
                    "mosaic.alpha_panel_count",
                    "mosaic.alpha_gauss_order",
                    "mosaic.azimuth_count",
                    "instrument.film_thickness_A",
                )
            ):
                add(path, sampling, field.label, field.unit, value, integer=field.kind == "int")
        add(
            ("run", "draw_count"),
            sampling,
            "MC draws per source",
            "draws",
            self.draft.draw_count,
            integer=True,
        )
        add(
            ("run", "detector_seed"),
            sampling,
            "Detector seed",
            "integer",
            self.draft.detector_seed,
            integer=True,
        )
        return entries

    @property
    def active_draft(self):
        return self.native.draft if self.draft_kind.currentData() == "native" else self.draft

    def _kind_changed(self):
        self.stop_live()
        self.form_stack.setCurrentIndex(self.draft_kind.currentIndex())
        self._supersede()
        self.figures.setVisible(self.draft_kind.currentData() == "configured")
        self.save_configuration_button.setText(
            "Export native draft JSON"
            if self.draft_kind.currentData() == "native"
            else "Export configuration YAML"
        )
        self.shell._mark_dirty()
        self.refresh()

    def refresh(self) -> None:
        native = self.draft_kind.currentData() == "native"
        draft = self.active_draft
        validated = self.native.validated if native else self.validated
        history = self.native.history if native else self.history
        available = draft is not None
        self.load_button.setText("Load native physics" if native else "Load configuration")
        busy = self.shell._active_kind == "simulation" or self.shell._pending_simulation is not None
        self.validate_button.setEnabled(available)
        self.run_button.setEnabled(available)
        self.cancel_button.setEnabled(busy or self.live.isChecked() or self._live_timer.isActive())
        self.inspect_button.setEnabled(busy or self.frame is not None)
        self.resume_button.setEnabled(self.hold)
        self.save_configuration_button.setEnabled(available and validated == draft)
        self.export_button.setEnabled(
            self.frame is not None
            and self.frame.quantitative
            and not busy
            and not self.detector._profile_pending
        )
        self.reopen_button.setEnabled(
            (self.native.result_reference if native else self.result_reference) is not None
        )
        self.undo_button.setEnabled(bool(history.undo_actions))
        self.redo_button.setEnabled(bool(history.redo_actions))
        self.dashboard.sync()

    def _transfer_source_context(self):
        return (
            self.shell.project.project_id,
            self.shell.selected_acquisition_id,
            self.shell._numeric_draft,
            self.shell._numeric_acquisition(),
        )

    def transfer_context_current(self, token):
        if self._transfer_context is None or self._transfer_request is None:
            return False
        source, target, destination, epoch = self._transfer_context
        actual = self.draft if target == "configured" else self.native.draft
        return (
            token == self._transfer_request["token"]
            and source == self._transfer_source_context()
            and actual == destination
            and epoch == self.epoch
        )

    def review_transfer(self):
        from simulation_transfer import numeric_snapshot_document

        try:
            acquisition = self.shell._numeric_validated_acquisition()
            source = self.shell._numeric_draft
            if source is None or not self.shell._numeric_identity_matches(acquisition):
                raise ProjectFormatError("load a compatible experiment numeric snapshot first")
            selected = self.transfer_target.currentData()
            target = "configured" if selected == "configured" else "native"
            destination = self.draft if target == "configured" else self.native.draft
            if target == "native" and (destination is None or destination.recipe != selected):
                raise ProjectFormatError(
                    "load the explicitly selected native recipe before reviewing this transfer"
                )
            self._supersede()
            token = str(uuid4())
            self._transfer_context = (
                self._transfer_source_context(),
                target,
                destination,
                self.epoch,
            )
            request = {
                "token": token,
                "source": numeric_snapshot_document(source),
                "target_kind": target,
                "recipe": selected,
                "destination": simulation_draft_document(destination)
                if target == "configured"
                else native_draft_document(destination),
            }
            self._transfer_request = request
            self._transfer_review = None
            self.shell._request_simulation("transfer_review", json.dumps(request).encode())
        except (ValueError, ProjectFormatError) as exc:
            self.status.setText(f"Transfer unavailable: {exc}")

    def present_transfer(self, review):
        if not self.transfer_context_current(review.token):
            self.status.setText("Stale transfer review rejected; source or destination changed")
            return
        self._transfer_review = review
        if self._transfer_dialog is not None:
            self._transfer_dialog.close()
        dialog = QDialog(self)
        self._transfer_dialog = dialog
        dialog.setWindowTitle("Review immutable experiment transfer")
        dialog.resize(1050, 650)
        layout = QVBoxLayout(dialog)
        note = QLabel(
            f"Destination: {review.target_kind}; source snapshot SHA256 {review.source_sha256}. Apply copies the included values into an independent draft. Target-retained and incompatible fields are explicit below."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        table = QTableWidget(len(review.rows), 5)
        table.setHorizontalHeaderLabels(
            ["Mapping", "Field", "Value", "Units / frame", "Provenance / reason"]
        )
        for row, values in enumerate(review.rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(value)
                table.setItem(row, column, item)
        table.setColumnWidth(0, 160)
        table.setColumnWidth(1, 240)
        table.setColumnWidth(2, 260)
        table.setColumnWidth(3, 220)
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            lambda: self.confirm_transfer(review, dialog)
        )
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.show()

    def confirm_transfer(self, review, dialog=None):
        if review is not self._transfer_review or not self.transfer_context_current(review.token):
            self.status.setText(
                "Stale transfer confirmation rejected before mutation; review again"
            )
            return False
        request = {
            **self._transfer_request,
            "confirmed_candidate": (
                simulation_draft_document(review.candidate)
                if review.target_kind == "configured"
                else native_draft_document(review.candidate)
            ),
        }
        self.shell._request_simulation("transfer_apply", json.dumps(request).encode())
        if dialog is not None:
            dialog.accept()
        return True

    def apply_transfer(self, review):
        if not self.transfer_context_current(review.token):
            self.status.setText("Deferred transfer confirmation became stale; prior state retained")
            return False
        if review.target_kind == "native":
            accepted = self.native.adopt(
                review.candidate, "Apply reviewed experiment transfer", validated=True
            )
        else:
            if not self._can_persist(review.candidate, self.result_reference):
                return False
            if self.draft is not None:
                self.history.push(
                    _action(
                        "Apply reviewed experiment transfer",
                        [
                            FieldChange(
                                review.candidate.draft_id,
                                "simulation_draft",
                                self.draft,
                                review.candidate,
                            )
                        ],
                    )
                )
            self.draft = self.validated = review.candidate
            self._populate(review.candidate)
            self._supersede()
            self.shell._mark_dirty()
            accepted = True
        if accepted:
            self.draft_kind.setCurrentIndex(self.draft_kind.findData(review.target_kind))
            self.status.setText(
                "Reviewed immutable experiment transfer applied; source unchanged. Run is explicit."
            )
            self._transfer_review = self._transfer_request = self._transfer_context = None
            self.refresh()
        return accepted

    def _settings(self) -> dict:
        return {
            "route": self.route.currentData(),
            "position_mode": self.position.currentData(),
            "draw_count": int(self.draw_count.text()),
            "detector_seed": int(self.detector_seed.text()),
            "export_figures": self.figures.isChecked(),
        }

    def _populate(self, draft: SimulationDraft) -> None:
        from rasim_next.pipeline.configured_simulation import load_strict_yaml_mapping

        self._restoring = True
        self.groups.blockSignals(True)
        try:
            self._mapping = load_strict_yaml_mapping(
                draft.configuration_path, source_bytes=draft.yaml_text.encode()
            )
            while self.groups.count():
                widget = self.groups.widget(0)
                self.groups.removeTab(0)
                widget.deleteLater()
            self.editors = {field.path: (field, None, None) for field in SIMULATION_FIELDS}
            self._group_layouts = {}
            self._group_order = []
            for field in SIMULATION_FIELDS:
                group = field.path.split(".", 1)[0]
                if group in self._group_layouts:
                    continue
                content = QWidget()
                form = QFormLayout(content)
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setWidget(content)
                self.groups.addTab(scroll, group.replace("_", " ").title())
                self._group_layouts[group] = form
                self._group_order.append(group)
            self.route.setCurrentIndex(self.route.findData(draft.route))
            self.position.setCurrentIndex(self.position.findData(draft.position_mode))
            self.draw_count.setText(str(draft.draw_count))
            self.detector_seed.setText(str(draft.detector_seed))
            self.figures.setChecked(draft.export_figures)
        finally:
            self.groups.blockSignals(False)
            self._restoring = False
        self._show_group(self.groups.currentIndex())
        self.filter_fields(self.search.text())

    def _show_group(self, index: int) -> None:
        if not 0 <= index < len(self._group_order):
            return
        group = self._group_order[index]
        self._restoring = True
        try:
            for path, (field, label, editor) in tuple(self.editors.items()):
                if path.split(".", 1)[0] != group or editor is not None:
                    continue
                value = self._mapping
                present = True
                for key in path.split("."):
                    if not isinstance(value, dict) or key not in value:
                        present, value = False, field.default
                        break
                    value = value[key]
                editor = FieldEditor(field, value, present, self)
                editor.changed.connect(self.propose)
                for line in editor.findChildren(QLineEdit):
                    line.textEdited.connect(self._pending_control_edited)
                label = QLabel(field.label)
                label.setWordWrap(True)
                self._group_layouts[group].addRow(label, editor)
                self.editors[path] = (field, label, editor)
        finally:
            self._restoring = False
        self.filter_fields(self.search.text())

    def filter_fields(self, text: str) -> None:
        text = text.casefold()
        visible_groups = set()
        for field, label, editor in self.editors.values():
            shown = (
                text
                in " ".join(
                    (
                        field.path,
                        field.label,
                        field.unit,
                        field.domain,
                        field.description,
                        field.applicability,
                    )
                ).casefold()
            )
            if shown:
                visible_groups.add(field.path.split(".", 1)[0])
            if editor is not None:
                label.setVisible(shown)
                editor.setVisible(shown)
        for index, group in enumerate(self._group_order):
            self.groups.setTabVisible(index, group in visible_groups)

    def _can_persist(
        self, draft: SimulationDraft | None, reference: SimulationReference | None
    ) -> bool:
        try:
            view = replace(
                self.shell._capture_view(), simulation_draft=draft, simulation_result=reference
            )
            self.shell._validate_project_admission(self.shell.project, view=view)
        except ValueError as exc:
            self.status.setText(f"Simulation change rejected; prior savable state retained: {exc}")
            return False
        return True

    def _pending_control_edited(self, _text: str) -> None:
        self.input_pending()

    def propose(self, *_) -> bool:
        if self._restoring:
            return False
        if self.draft is None:
            self._supersede()
            return False
        mapping = json.loads(json.dumps(self._mapping))
        for path, (_field, _label, editor) in self.editors.items():
            if editor is None:
                continue
            keys = path.split(".")
            parent = mapping
            included = editor.included is None or editor.included.isChecked()
            if not included:
                for key in keys[:-1]:
                    if key not in parent:
                        break
                    parent = parent[key]
                else:
                    parent.pop(keys[-1], None)
                continue
            for key in keys[:-1]:
                parent = parent.setdefault(key, {})
            parent[keys[-1]] = editor.value()
        # The optional tilt mapping must either declare both components or be absent.
        tilt = mapping["instrument"].get("detector_tilt")
        if tilt == {}:
            del mapping["instrument"]["detector_tilt"]
        try:
            settings = self._settings()
            if mapping == self._mapping and all(
                getattr(self.draft, key) == value for key, value in settings.items()
            ):
                return True
            updated = replace(
                self.draft,
                yaml_text=yaml.safe_dump(mapping, sort_keys=False),
                revision=self.draft.revision + 1,
                **self._settings(),
            )
            action = _action(
                "Edit independent simulation draft",
                [FieldChange(self.draft.draft_id, "simulation_draft", self.draft, updated)],
            )
        except (ValueError, ProjectFormatError) as exc:
            self.validated = None
            self.status.setText(
                f"Invalid run control: {exc}; correct it before validating or saving"
            )
            self._supersede()
            return False
        if replace(updated, revision=self.draft.revision) == self.draft:
            return True
        if not self._can_persist(updated, self.result_reference):
            return False
        self.history.push(action)
        self.draft = updated
        self._mapping = mapping
        self.validated = None
        self.status.setText(
            f"Draft revision {updated.revision} saved in editable state; complete validation required. No run started."
        )
        self._supersede()
        self.shell._mark_dirty()
        self.refresh()
        self.draft_changed()
        return True

    def _supersede(self) -> None:
        self._update_request = None
        self.dashboard.incidence.setText("Mean-ray incidence: validate the current geometry")
        self.epoch += 1
        self.shell._supersede_simulation()
        if self.frame is not None:
            progress = (
                "completed batches"
                if isinstance(self.frame.draft, NativeSimulationDraft)
                else "draw prefix"
            )
            self.identity.setText(
                f"Historical snapshot: draft {self.frame.draft.draft_id} revision {self.frame.draft.revision}, {progress} {self.frame.draw_prefix}. Current draft changed; no current quantitative values."
            )

    def history_step(self, undo: bool) -> None:
        if self.draft_kind.currentData() == "native":
            self.native.history_step(undo)
            return
        source = self.history.undo_actions if undo else self.history.redo_actions
        if not source or self.draft is None:
            return
        action = source[-1]
        change = action.changes[0]
        expected, restored = (
            (change.after, change.before) if undo else (change.before, change.after)
        )
        if replace(self.draft, revision=expected.revision) != expected:
            self.status.setText("Draft history conflicts with current values")
            return
        updated = replace(restored, revision=self.draft.revision + 1)
        if not self._can_persist(updated, self.result_reference):
            return
        self.draft = updated
        source.pop()
        (self.history.redo_actions if undo else self.history.undo_actions).append(action)
        self.validated = None
        self._populate(self.draft)
        self._supersede()
        self.shell._mark_dirty()
        self.status.setText("Independent draft restored; validate again before execution")
        self.refresh()
        self.draft_changed()

    def load(self) -> None:
        if self.draft_kind.currentData() == "native":
            self.native.load()
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load independent configured simulation", "", "Simulation YAML (*.yaml *.yml)"
        )
        if path:
            self.stop_live()
            self.shell._request_simulation(
                "load", json.dumps({"operation": "load", "path": path}).encode()
            )

    def validate(self) -> None:
        if self.draft_kind.currentData() == "native":
            self.native.validate()
            return
        if self.draft is None:
            return
        if not self.propose():
            return
        try:
            settings = self._settings()
        except ValueError as exc:
            self.status.setText(f"Invalid run control: {exc}")
            return
        self.shell._request_simulation(
            "validate",
            json.dumps(
                {
                    "operation": "validate",
                    "draft": simulation_draft_document(self.draft),
                    "yaml_text": self.draft.yaml_text,
                    "settings": settings,
                }
            ).encode(),
        )

    def run(self) -> None:
        native = self.draft_kind.currentData() == "native"
        draft = self.active_draft
        validated = self.native.validated if native else self.validated
        if draft is None or validated != draft:
            self.status.setText("Validate the complete current draft before Run")
            return
        self.hold = False
        self.latest_frame = None
        self._inspection_pending = False
        self.detector.quantitative_ready = False
        self.detector.profile_status.setText(
            "Awaiting quantitative snapshot; any retained values are historical"
        )
        self.identity.setText(
            f"Updating current draft revision {draft.revision}; displayed historical snapshot "
            f"{self.frame.run_id}, revision {self.frame.draft.revision}, prefix {self.frame.draw_prefix}"
            if self.frame is not None
            else f"Starting current draft revision {draft.revision}; Awaiting quantitative snapshot"
        )
        self.shell._request_simulation(
            "native_run" if native else "run",
            json.dumps(
                {
                    "draft": native_draft_document(draft)
                    if native
                    else simulation_draft_document(draft),
                    **self.shell._simulation_resource_charge(),
                }
            ).encode(),
        )

    def inspect(self) -> None:
        if self.latest_frame is not None and self.latest_frame.quantitative:
            self.hold = True
            self.admit(self.latest_frame, force=True)
            self.status.setText(
                "Holding this exact snapshot image/profiles; later publications remain historical until selected"
            )
        elif self.shell._active_kind == "simulation" and self.shell._simulation_operation in (
            "run",
            "native_run",
        ):
            self.shell.jobs.inspect_active(self.shell._active_generation)
            self._inspection_pending = True
            self.hold = False
            self.status.setText(
                "Inspection requested at the next canonical batch boundary; Awaiting quantitative snapshot"
            )
        self.refresh()

    def follow(self) -> None:
        self.hold = False
        if self.latest_frame is not None:
            self.admit(self.latest_frame, force=True)
        self.refresh()

    def save_configuration(self) -> None:
        if self.draft_kind.currentData() == "native":
            if self.native.draft is None or self.native.validated != self.native.draft:
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export independent native draft",
                "native-draft.json",
                "Native draft (*.json)",
            )
            if path:
                self.shell._request_simulation(
                    "native_save",
                    json.dumps(
                        {
                            "operation": "native_save",
                            "draft": native_draft_document(self.native.draft),
                            "path": path,
                        }
                    ).encode(),
                )
            return
        if self.draft is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export a new canonical YAML", "simulation.yaml", "Simulation YAML (*.yaml *.yml)"
        )
        if path:
            self.shell._request_simulation(
                "save_configuration",
                json.dumps(
                    {
                        "operation": "save_configuration",
                        "draft": simulation_draft_document(self.draft),
                        "path": path,
                    }
                ).encode(),
            )

    def export(self) -> None:
        frame = self.frame
        if frame is None or not frame.quantitative:
            self.status.setText("Awaiting quantitative snapshot")
            return
        if frame.image is not None and (
            self.detector._profile_pending
            or self.detector._profile_key != self.detector._query_key()
        ):
            self.status.setText("Wait for exact profiles matching this snapshot and query")
            return
        query = self.detector.profile_query()
        profiles = self.detector._current_profiles
        manifest = json.loads(frame.manifest)
        manifest["write_configured_figures"] = (
            isinstance(frame.draft, SimulationDraft) and self.figures.isChecked()
        )
        config_outputs = self._mapping["outputs"] if self._mapping is not None else {}
        directory = str(config_outputs.get("output_directory", ""))
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export new exact numeric snapshot",
            str(Path(directory).expanduser() / "simulation-snapshot.npz"),
            "Numeric snapshot (*.npz)",
        )
        if not path:
            return
        if self.frame is not frame or self.detector.profile_query() != query:
            self.status.setText("Snapshot or inspection query changed; choose the export again")
            return
        arrays = list(frame.arrays)
        if frame.image is not None:
            arrays.append(("image", frame.image))
            manifest["inspection_query"] = query
            manifest["inspection_identity"] = [
                frame.run_id,
                frame.draft.revision,
                frame.draw_prefix,
            ]
            arrays.extend(
                (
                    ("horizontal_profile", profiles.horizontal),
                    ("vertical_profile", profiles.vertical),
                    ("horizontal_support", profiles.horizontal_support),
                    ("vertical_support", profiles.vertical_support),
                )
            )
        work = SimulationExportWork(
            Path(path).absolute(),
            json.dumps(manifest, sort_keys=True, allow_nan=False).encode(),
            tuple(arrays),
        )
        self.shell._request_simulation("export", work)

    def reopen(self) -> None:
        if self.draft_kind.currentData() == "native":
            if self.native.result_reference is not None:
                self.shell._request_simulation(
                    "reopen",
                    json.dumps(native_reference_document(self.native.result_reference)).encode(),
                )
            return
        if self.result_reference is not None:
            from simulation_state import simulation_reference_document

            self.shell._request_simulation(
                "reopen", json.dumps(simulation_reference_document(self.result_reference)).encode()
            )

    def request_profiles(self) -> None:
        if (
            self.frame is None
            or not self.frame.quantitative
            or self.frame.image is None
            or self._update_request is not None
            or self._live_timer.isActive()
            or (
                self.shell._pending_simulation is not None
                and self.shell._pending_simulation[0] != "profiles"
            )
            or (
                self.shell._simulation_operation
                in ("run", "native_run", "validate", "native_validate")
                and self.shell._active_kind == "simulation"
            )
        ):
            return
        shape = self.frame.image.shape
        mask = NativeMask(
            self.frame.draft.configuration_sha256
            if isinstance(self.frame.draft, SimulationDraft)
            else self.frame.draft.physics_sha256,
            shape,
        )
        work = MaskWork(self.frame.image, mask, (), self.detector.profile_query())
        self.shell._request_simulation("profiles", work)

    def admit(self, frame: SimulationFrame, *, force: bool = False) -> bool:
        if not self._can_persist(self.draft, self.result_reference):
            return False
        previous_run = self.frame.run_id if self.frame is not None else None
        self.latest_frame = frame
        if self._inspection_pending and frame.quantitative:
            self._inspection_pending = False
            self.hold = True
            force = True
        if self.hold and not force and self.frame is not None and self.frame.quantitative:
            progress = (
                "completed batches"
                if isinstance(frame.draft, NativeSimulationDraft)
                else "draw prefix"
            )
            self.identity.setText(
                f"Held snapshot draft {self.frame.draft.revision} {progress} {self.frame.draw_prefix}; latest {progress} {frame.draw_prefix}. Image and profiles remain held together."
            )
            return True
        self.frame = frame
        if frame.image is None:
            self._clear_detector()
        else:
            self.detector.setEnabled(True)
            previous = panel_view_state(self.detector)
            self.detector.view.column_axis_label = (
                "macrobin column index"
                if (
                    (isinstance(frame.draft, NativeSimulationDraft) and frame.draft.bin_size_px > 1)
                    or (
                        isinstance(frame.draft, SimulationDraft)
                        and frame.draft.route == "macrobins"
                    )
                )
                else "column_px"
            )
            self.detector.view.row_axis_label = (
                "macrobin row index"
                if (
                    (isinstance(frame.draft, NativeSimulationDraft) and frame.draft.bin_size_px > 1)
                    or (
                        isinstance(frame.draft, SimulationDraft)
                        and frame.draft.route == "macrobins"
                    )
                )
                else "row_px"
            )
            self.detector.column_coordinate_label.setText(self.detector.view.column_axis_label)
            self.detector.row_coordinate_label.setText(self.detector.view.row_axis_label)
            self.detector.profile_measure_control.setItemText(
                1,
                "Mean / valid macrobin"
                if (
                    (isinstance(frame.draft, NativeSimulationDraft) and frame.draft.bin_size_px > 1)
                    or (
                        isinstance(frame.draft, SimulationDraft)
                        and frame.draft.route == "macrobins"
                    )
                )
                else "Mean / valid px",
            )
            self.detector.quantitative_ready = frame.quantitative
            self.detector.observable_unit = (
                "angstrom^2/pixel^2 (display density)"
                if isinstance(frame.draft, SimulationDraft) and frame.draft.route == "pixel_centers"
                else "angstrom^2 (simulation mass)"
            )
            if (isinstance(frame.draft, NativeSimulationDraft) and frame.draft.bin_size_px > 1) or (
                isinstance(frame.draft, SimulationDraft) and frame.draft.route == "macrobins"
            ):
                self.detector.observable_unit = (
                    "angstrom^2/macrobin (display quadrature; cursor uses macrobin indices)"
                )
            if frame.quantitative:
                self.detector.set_prepared_image(
                    frame.image,
                    frame.display,
                    frame.profiles,
                    frame.full_profiles,
                    frame.low,
                    frame.high,
                    frame.maximum,
                    frame.min_positive,
                    (frame.run_id, frame.draft.revision, frame.draw_prefix),
                )
            else:
                self.detector.view.set_prepared_image(
                    frame.image,
                    frame.display,
                    frame.low,
                    frame.high,
                    frame.maximum,
                    frame.min_positive,
                )
                self.detector._reset_profile_state(
                    (frame.run_id, frame.draft.revision, frame.draw_prefix)
                )
                self.detector.quantitative_ready = False
                self.detector.profile_status.setText(
                    "Awaiting quantitative snapshot; retained profiles are historical"
                )
            state = self._pending_detector_state or previous
            self._pending_detector_state = None
            if (
                state is not None
                and state.column_px < frame.image.shape[1]
                and state.row_px < frame.image.shape[0]
            ):
                view = self.detector.view
                view.crosshair = (state.column_px, state.row_px)
                view.zoom = state.zoom
                view.pan = QPointF(state.pan_x_px, state.pan_y_px)
                view.scale_mode = state.scale_mode
                view.set_levels(state.low_value, state.high_value, mode=state.contrast_mode)
                self.detector.restore_profile_state(state)
                self.detector._sync_controls()
        if frame.image is not None:
            self.display_contrast.admit(frame)
        arrays = dict(frame.arrays)
        if previous_run != frame.run_id:
            for display, key in (
                (self.reciprocal, "reciprocal_q_sample_Ainv"),
                (self.ewald, "ewald_q_sample_Ainv"),
            ):
                if key not in arrays:
                    display.set_data(
                        np.empty((0, 3)), np.empty(0), "No selected output in this snapshot"
                    )
        if previous_run != frame.run_id and "reciprocal_q_sample_Ainv" in arrays:
            self.reciprocal.set_data(
                arrays["reciprocal_q_sample_Ainv"],
                arrays["reciprocal_density_A2_rad2_inv"],
                "Canonical latent reciprocal display; angstrom^2 rad^-2",
            )
        if previous_run != frame.run_id and "ewald_q_sample_Ainv" in arrays:
            self.ewald.set_data(
                arrays["ewald_q_sample_Ainv"],
                arrays["ewald_density_A2_rad2_inv"],
                "Canonical detector-visible nominal Ewald coating; angstrom^2 rad^-2",
            )
        current_draft = self.active_draft
        validated = (
            self.native.validated
            if isinstance(current_draft, NativeSimulationDraft)
            else self.validated
        )
        current = (
            current_draft is not None
            and frame.draft == current_draft
            and validated == current_draft
        )
        label = "Current" if current else "Historical"
        progress = (
            f"completed event batches {frame.draw_prefix}; integral complete={json.loads(frame.manifest).get('integration_complete')}"
            if isinstance(frame.draft, NativeSimulationDraft)
            else f"draw prefix {frame.draw_prefix}; detector seed {frame.draft.detector_seed}"
        )
        self.identity.setText(
            f"{label} draft {frame.draft.draft_id} revision {frame.draft.revision}; run {frame.run_id}; {frame.measure}; backend {frame.backend}; {progress}. "
            + (
                "Immutable float64; nominal, not converged or fit qualified."
                if frame.quantitative
                else "Float32 presentation only; Awaiting quantitative snapshot."
            )
        )
        if (isinstance(frame.draft, NativeSimulationDraft) and frame.draft.bin_size_px > 1) or (
            isinstance(frame.draft, SimulationDraft) and frame.draft.route == "macrobins"
        ):
            self.identity.setText(
                self.identity.text()
                + " Display coordinates are macrobin column/row indices; exported center arrays provide native pixel coordinates."
            )
        self.refresh()
        return True

    def ready(self, operation: str, value, incidence_deg=None) -> None:
        continue_update = (
            operation in ("validate", "native_validate")
            and self._update_request == self.update_context()
        )
        self._update_request = None
        if operation == "transfer_review":
            self.present_transfer(value)
        elif operation == "transfer_apply":
            self.apply_transfer(value)
        elif isinstance(value, NativeSimulationDraft):
            if operation == "native_load":
                if not self.native.can_persist(value, None):
                    return
                self.native.result_reference = None
                self.native.history = SessionHistory()
            if not self.native.adopt(
                value,
                "Load native draft" if operation == "native_load" else "Validate native draft",
                validated=True,
                record_history=operation != "native_load",
            ):
                return
        elif isinstance(value, NativeSimulationReference):
            if not self.native.can_persist(self.native.draft, value):
                self.status.setText(
                    self.status.text()
                    + f" Exported snapshot remains at {value.path}; it was not added to this project."
                )
                return
            self.native.result_reference = value
            self.status.setText(
                f"Exact native snapshot exported/read back: {value.path}; SHA256 {value.sha256}"
            )
            self.shell._mark_dirty()
        elif isinstance(value, SimulationDraft):
            reference = (
                None
                if operation in ("load", "load_default", "load_limited")
                else self.result_reference
            )
            if not self._can_persist(value, reference):
                return
            previous_yaml = self.draft.yaml_text if self.draft is not None else None
            if operation in ("load", "load_default", "load_limited"):
                self.history = SessionHistory()
                self.result_reference = None
            self.draft = value
            self.validated = None if operation == "load_limited" else value
            if (
                operation in ("load", "load_default", "load_limited")
                or self._mapping is None
                or previous_yaml != value.yaml_text
            ):
                self._populate(value)
            self.status.setText(
                f"Complete canonical configuration admitted: revision {value.revision}, config SHA-256 {value.configuration_sha256}; CIF SHA-256 {value.cif_sha256}. Run is explicit; detector routes require supported finite-stack strength."
            )
            if operation == "load_default":
                self.status.setText(
                    "Bi2Se3 starting preview: editable, nominal, not converged or qualified. "
                    "Canonical inputs validated; Run selected outputs is explicit. "
                    "64 source samples, 4 CPU workers, 8 draws/source, detector seed 1729; "
                    "detector only, configured figure export off."
                )
            if operation == "load_limited":
                self.status.setText(
                    "Imported declaration retained completely. Its source count exceeds the desktop 256-row cap; edit explicitly and Validate before execution. No source support was reduced."
                )
            self.shell._mark_dirty()
        elif isinstance(value, SimulationFrame):
            if not self.admit(value):
                return
            self.status.setText(
                "Requested outputs ready; quantitative snapshots remain nominal and unqualified"
            )
        elif isinstance(value, SimulationReference):
            if not self._can_persist(self.draft, value):
                self.status.setText(
                    self.status.text()
                    + f" Exported snapshot remains at {value.path}; it was not added to this project."
                )
                return
            self.result_reference = value
            self.status.setText(
                f"Exact snapshot written and reopened successfully: {value.path}; SHA-256 {value.sha256}"
            )
            self.shell._mark_dirty()
        elif operation == "profiles":
            image_identity, query, profiles = value
            if (
                self.frame is not None
                and self.frame.quantitative
                and self.frame.image is not None
                and id(self.frame.image) == image_identity
                and self.detector.profile_query() == query
            ):
                self.detector._profile_pending = False
                self.detector._present_profiles(profiles, query[4], self.detector._query_key())
                self.status.setText(
                    "Exact float64 profiles match the inspected snapshot and current query"
                )
            else:
                self.detector._profile_pending = (
                    self.frame is not None and self.frame.image is not None
                )
        elif operation in ("save_configuration", "native_save"):
            self.status.setText(f"Canonical YAML exported and read back: {value}")
        self.refresh()
        if incidence_deg is not None:
            self.dashboard.incidence.setText(
                f"Mean-ray glancing incidence: {incidence_deg:.12g} deg\n"
                "Signed toward the sample; derived from the canonical LAB → SAMPLE transform."
            )
        if operation == "load_default" and self._fresh_live_pending:
            self._fresh_live_pending = False
            self.live.setChecked(True)
        elif continue_update:
            self._last_requested = (self.draft_kind.currentData(), self.active_draft)
            self.run()

    def _clear_detector(self) -> None:
        self.detector.view.set_display_levels(())
        self.display_contrast.frame = None
        if self.detector.view.image is not None:
            self.detector._reset_profile_state(None)
        self.detector.view.image = self.detector.view._display = None
        self.detector.view._request_paint()
        self.detector.quantitative_ready = False
        for plot in (self.detector.horizontal, self.detector.vertical):
            plot.values = plot.support = None
            plot.update()
        self.detector._current_profiles = self.detector._full_profiles = None
        self.detector._profile_key = None
        self.detector._profile_pending = False
        self.detector.setEnabled(False)
        self.detector.export_button.setEnabled(False)
        self.detector.cursor_label.setText("No detector output in this snapshot")
        self.detector.profile_status.setText(
            "No detector output; exact detector profiles unavailable"
        )
        self._pending_detector_state = None
        pending = self.shell._pending_simulation
        if pending is not None and pending[0] == "profiles":
            self.shell._pending_simulation = None

    def restore(self, view, detail: str = "") -> None:
        self.stop_live()
        self._last_requested = None
        self.epoch += 1
        self.native.draft = view.native_simulation_draft
        self.native.result_reference = view.native_simulation_result
        self.native.validated = None
        self.native.history = SessionHistory()
        if self.native.draft is not None:
            self.native.populate(self.native.draft)
        self.draft_kind.blockSignals(True)
        self.draft_kind.setCurrentIndex(self.draft_kind.findData(view.simulator_kind))
        self.form_stack.setCurrentIndex(self.draft_kind.currentIndex())
        self.figures.setVisible(view.simulator_kind == "configured")
        self.draft_kind.blockSignals(False)
        self.draft = view.simulation_draft
        self.result_reference = view.simulation_result
        self.validated = None
        self.frame = self.latest_frame = None
        self._clear_detector()
        self.identity.setText("Awaiting quantitative snapshot")
        for display in (self.reciprocal, self.ewald):
            display.set_data(np.empty((0, 3)), np.empty(0), "Awaiting selected output snapshot")
        self.hold = False
        self._inspection_pending = False
        self.history = SessionHistory()
        self._pending_detector_state = view.simulation_detector
        self.display_contrast.restore(view.simulation_detector)
        if self.draft is not None:
            self._populate(self.draft)
            self.status.setText(
                detail
                or "Independent draft restored; validate current canonical inputs before execution. Saved results reopen by exact hash and may be historical."
            )
        elif self.native.draft is not None:
            self.status.setText(
                "Native draft restored; Live is off. Run/update validates the current complete inputs."
            )
        else:
            self.status.setText("No independent simulation draft in this project")
        self.refresh()
