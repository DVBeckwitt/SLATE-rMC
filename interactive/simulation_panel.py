"""Independent configured forms and shared detector inspection, without experiment bindings."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from comparison_panel import panel_view_state
from detector_panel import DetectorPanel
from mask_state import MaskWork, NativeMask
from parameter_state import FieldChange, SessionHistory, _action
from project_state import ProjectFormatError
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        controls = QHBoxLayout()
        self.load_button = QPushButton("Load configuration")
        self.validate_button = QPushButton("Validate complete draft")
        self.run_button = QPushButton("Run selected outputs")
        self.inspect_button = QPushButton("Inspect this snapshot")
        self.resume_button = QPushButton("Follow progression")
        self.cancel_button = QPushButton("Cancel simulation")
        for button in (
            self.load_button,
            self.validate_button,
            self.run_button,
            self.inspect_button,
            self.resume_button,
            self.cancel_button,
        ):
            controls.addWidget(button)
        layout.addLayout(controls)
        actions = QHBoxLayout()
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
            actions.addWidget(button)
        layout.addLayout(actions)
        self.status = QLabel(
            "Load a supported rasim-simulation-v2 configuration. No acquisition or fit is required."
        )
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(42)
        layout.addWidget(self.status)
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
        splitter.addWidget(forms)
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
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self.figures = QCheckBox(
            "Export configured figures using selected output names and directory"
        )
        self.figures.setChecked(True)
        layout.addWidget(self.figures)
        self.identity = QLabel("Awaiting quantitative snapshot")
        self.identity.setWordWrap(True)
        self.identity.setMinimumHeight(46)
        layout.addWidget(self.identity)
        self.load_button.clicked.connect(self.load)
        self.validate_button.clicked.connect(self.validate)
        self.run_button.clicked.connect(self.run)
        self.inspect_button.clicked.connect(self.inspect)
        self.resume_button.clicked.connect(self.follow)
        self.cancel_button.clicked.connect(shell._cancel_current)
        self.save_configuration_button.clicked.connect(self.save_configuration)
        self.export_button.clicked.connect(self.export)
        self.reopen_button.clicked.connect(self.reopen)
        self.undo_button.clicked.connect(lambda: self.history_step(True))
        self.redo_button.clicked.connect(lambda: self.history_step(False))
        self.search.textChanged.connect(self.filter_fields)
        self.route.currentIndexChanged.connect(self.propose)
        self.position.currentIndexChanged.connect(self.propose)
        self.draw_count.editingFinished.connect(self.propose)
        self.detector_seed.editingFinished.connect(self.propose)
        self.detector.profile_requested.connect(self.request_profiles)
        self.detector.view.view_state_changed.connect(self.shell._mark_dirty)
        self.refresh()

    def refresh(self) -> None:
        available = self.draft is not None
        busy = self.shell._active_kind == "simulation" or self.shell._pending_simulation is not None
        self.validate_button.setEnabled(available)
        self.run_button.setEnabled(available and self.validated == self.draft)
        self.cancel_button.setEnabled(busy)
        self.inspect_button.setEnabled(busy or self.frame is not None)
        self.resume_button.setEnabled(self.hold)
        self.save_configuration_button.setEnabled(available and self.validated == self.draft)
        self.export_button.setEnabled(
            self.frame is not None
            and self.frame.quantitative
            and not busy
            and not self.detector._profile_pending
        )
        self.reopen_button.setEnabled(self.result_reference is not None)
        self.undo_button.setEnabled(bool(self.history.undo_actions))
        self.redo_button.setEnabled(bool(self.history.redo_actions))

    def _settings(self) -> dict:
        return {
            "route": self.route.currentData(),
            "position_mode": self.position.currentData(),
            "draw_count": int(self.draw_count.text()),
            "detector_seed": int(self.detector_seed.text()),
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
        view = replace(
            self.shell._capture_view(), simulation_draft=draft, simulation_result=reference
        )
        try:
            self.shell._validate_project_admission(self.shell.project, view=view)
        except ProjectFormatError as exc:
            self.status.setText(f"Simulation change rejected; prior savable state retained: {exc}")
            return False
        return True

    def propose(self, *_) -> bool:
        if self._restoring or self.draft is None:
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
        return True

    def _supersede(self) -> None:
        self.epoch += 1
        self.shell._supersede_simulation()
        if self.frame is not None:
            self.identity.setText(
                f"Historical snapshot: draft {self.frame.draft.draft_id} revision {self.frame.draft.revision}, prefix {self.frame.draw_prefix}. Current draft changed; no current quantitative values."
            )

    def history_step(self, undo: bool) -> None:
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

    def load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load independent configured simulation", "", "Simulation YAML (*.yaml *.yml)"
        )
        if path:
            self._supersede()
            self.shell._request_simulation(
                "load", json.dumps({"operation": "load", "path": path}).encode()
            )

    def validate(self) -> None:
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
        if self.draft is None or self.validated != self.draft:
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
            f"Starting current draft revision {self.draft.revision}; Awaiting quantitative snapshot"
        )
        self.shell._request_simulation(
            "run",
            json.dumps(
                {
                    "draft": simulation_draft_document(self.draft),
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
        elif self.shell._active_kind == "simulation" and self.shell._simulation_operation == "run":
            self.shell.jobs.inspect_active(self.shell._active_generation)
            self._inspection_pending = True
            self.hold = False
            self.status.setText(
                "Inspection requested at the next canonical prefix boundary; Awaiting quantitative snapshot"
            )
        self.refresh()

    def follow(self) -> None:
        self.hold = False
        if self.latest_frame is not None:
            self.admit(self.latest_frame, force=True)
        self.refresh()

    def save_configuration(self) -> None:
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
        manifest["write_configured_figures"] = self.figures.isChecked()
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
            or (
                self.shell._simulation_operation == "run"
                and self.shell._active_kind == "simulation"
            )
        ):
            return
        shape = self.frame.image.shape
        mask = NativeMask(self.frame.draft.configuration_sha256, shape)
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
            self.identity.setText(
                f"Held snapshot draft {self.frame.draft.revision} prefix {self.frame.draw_prefix}; latest prefix {frame.draw_prefix}. Image and profiles remain held together."
            )
            return True
        self.frame = frame
        if frame.image is None:
            self._clear_detector()
        else:
            self.detector.setEnabled(True)
            previous = panel_view_state(self.detector)
            self.detector.view.column_axis_label = (
                "macrobin column index" if frame.draft.route == "macrobins" else "column_px"
            )
            self.detector.view.row_axis_label = (
                "macrobin row index" if frame.draft.route == "macrobins" else "row_px"
            )
            self.detector.column_coordinate_label.setText(self.detector.view.column_axis_label)
            self.detector.row_coordinate_label.setText(self.detector.view.row_axis_label)
            self.detector.profile_measure_control.setItemText(
                1,
                "Mean / valid macrobin" if frame.draft.route == "macrobins" else "Mean / valid px",
            )
            self.detector.quantitative_ready = frame.quantitative
            self.detector.observable_unit = (
                "angstrom^2/pixel^2 (display density)"
                if frame.draft.route == "pixel_centers"
                else "angstrom^2 (simulation mass)"
            )
            if frame.draft.route == "macrobins":
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
        current = (
            self.draft is not None and frame.draft == self.draft and self.validated == self.draft
        )
        label = "Current" if current else "Historical"
        self.identity.setText(
            f"{label} draft {frame.draft.draft_id} revision {frame.draft.revision}; run {frame.run_id}; {frame.measure}; backend {frame.backend}; draw prefix {frame.draw_prefix}; detector seed {frame.draft.detector_seed}. "
            + (
                "Immutable float64; nominal, not converged or fit qualified."
                if frame.quantitative
                else "Float32 presentation only; Awaiting quantitative snapshot."
            )
        )
        if frame.draft.route == "macrobins":
            self.identity.setText(
                self.identity.text()
                + " Display coordinates are macrobin column/row indices; exported center arrays provide native pixel coordinates."
            )
        self.refresh()
        return True

    def ready(self, operation: str, value) -> None:
        if isinstance(value, SimulationDraft):
            reference = None if operation in ("load", "load_limited") else self.result_reference
            if not self._can_persist(value, reference):
                return
            previous_yaml = self.draft.yaml_text if self.draft is not None else None
            if operation in ("load", "load_limited"):
                self.history = SessionHistory()
                self.result_reference = None
            self.draft = value
            self.validated = None if operation == "load_limited" else value
            if (
                operation in ("load", "load_limited")
                or self._mapping is None
                or previous_yaml != value.yaml_text
            ):
                self._populate(value)
            self.status.setText(
                f"Complete canonical configuration admitted: revision {value.revision}, config SHA-256 {value.configuration_sha256}; CIF SHA-256 {value.cif_sha256}. Run is explicit; detector routes require supported finite-stack strength."
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
        elif operation == "save_configuration":
            self.status.setText(f"Canonical YAML exported and read back: {value}")
        self.refresh()

    def _clear_detector(self) -> None:
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
        self.epoch += 1
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
        if self.draft is not None:
            self._populate(self.draft)
            self.status.setText(
                detail
                or "Independent draft restored; validate current canonical inputs before execution. Saved results reopen by exact hash and may be historical."
            )
        else:
            self.status.setText("No independent simulation draft in this project")
        self.refresh()
