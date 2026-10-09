"""Compact scalar navigation over the simulator's existing schema editors."""

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class ScalarControl(QWidget):
    edited = Signal(str)
    pending = Signal(bool)
    drag_started = Signal()
    dragged = Signal(str)
    drag_finished = Signal(str)

    def __init__(self, label, unit, value, *, integer=False, probability=False):
        super().__init__()
        self.integer = integer
        self.probability = probability
        self.center = (
            float(value) if isinstance(value, (float, int)) and math.isfinite(value) else 0.0
        )
        self.span = 1.0 if integer else (abs(self.center) * 0.2 if self.center else 0.01)
        self.step = 1.0 if integer else self.span / 100
        self.drag_origin = self.center
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        title = self.title = QLabel(f"{label} ({unit})")
        title.setWordWrap(True)
        layout.addWidget(title)
        row = QHBoxLayout()
        row.setSpacing(2)
        self.text = QLineEdit()
        self.text.setAccessibleName(f"{label}, {unit}, exact value")
        self.text.setToolTip("Exact value; canonical validation determines the physical domain.")
        title.setBuddy(self.text)
        row.addWidget(self.text, 1)
        for sign, direction in (("-", -1), ("+", 1)):
            button = QToolButton()
            button.setText(sign)
            button.setAccessibleName(f"{label}: step {'down' if direction < 0 else 'up'}")
            button.clicked.connect(lambda _checked=False, d=direction: self.nudge(d))
            row.addWidget(button)
        navigation = QToolButton()
        navigation.setText("⋯")
        navigation.setAccessibleName(f"{label}: slider span and keyboard step")
        navigation.clicked.connect(self.navigation)
        row.addWidget(navigation)
        layout.addLayout(row)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000) if probability else self.slider.setRange(-1000, 1000)
        self.slider.setSingleStep(
            max(1, round(1000 * self.step / (1 if probability else self.span)))
        )
        self.slider.setAccessibleName(f"{label}, slider navigation in {unit}")
        self.slider.setToolTip(
            "Held drag dispatches latest values every 200 ms when Live is on. One undo gesture; exact final value on release."
        )
        self.slider.sliderPressed.connect(self.begin_drag)
        self.slider.sliderReleased.connect(self.finish_drag)
        self.slider.valueChanged.connect(self.slide)
        layout.addWidget(self.slider)
        self.text.textEdited.connect(lambda _text: self.pending.emit(True))
        self.text.editingFinished.connect(lambda: self.edited.emit(self.text.text()))
        self.set_value(value)

    def set_value(self, value):
        if self.slider.isSliderDown():
            return
        self.center = (
            float(value) if isinstance(value, (float, int)) and math.isfinite(value) else 0.0
        )
        self.drag_origin = self.center
        self.text.setText(
            repr(float(value))
            if isinstance(value, (float, int)) and not self.integer
            else str(value)
        )
        self.slider.blockSignals(True)
        self.slider.setValue(round(self.center * 1000) if self.probability else 0)
        self.slider.blockSignals(False)

    def begin_drag(self):
        self.drag_origin = self.center
        self.drag_started.emit()

    def slide(self, position):
        value = (
            position / 1000 if self.probability else self.drag_origin + position * self.span / 1000
        )
        if self.integer:
            value = round(value)
        self.text.setText(repr(value))
        if self.slider.isSliderDown():
            self.dragged.emit(self.text.text())
        else:
            self.edited.emit(self.text.text())

    def finish_drag(self):
        self.drag_finished.emit(self.text.text())

    def nudge(self, direction):
        try:
            value = float(self.text.text()) + direction * self.step
        except ValueError:
            self.edited.emit(self.text.text())
            return
        if self.integer:
            value = round(value)
        if self.probability:
            value = min(1.0, max(0.0, value))
        self.text.setText(repr(value))
        self.edited.emit(self.text.text())

    def navigation(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Navigation span and step")
        form = QFormLayout(dialog)
        span, step = QLineEdit(repr(self.span)), QLineEdit(repr(self.step))
        span.setEnabled(not self.probability)
        form.addRow("Slider half span (navigation only)", span)
        form.addRow("Step in the displayed units", step)
        error = QLabel("The exact value is unrestricted by this span.")
        error.setWordWrap(True)
        form.addRow(error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(buttons)

        def accept():
            try:
                values = float(span.text()), float(step.text())
                if any(not math.isfinite(v) or v <= 0 for v in values):
                    raise ValueError("Use finite positive span and step")
                if self.integer and any(v < 1 or not v.is_integer() for v in values):
                    raise ValueError("Integer controls require whole steps and span >= 1")
            except ValueError as exc:
                error.setText(str(exc))
                return
            self.span, self.step = values
            self.slider.setSingleStep(
                max(1, round(1000 * self.step / (1 if self.probability else self.span)))
            )
            dialog.accept()

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        dialog.exec()


class SimulationDashboard(QScrollArea):
    """Quick controls delegate all edits to existing immutable draft/history owners."""

    def __init__(self, simulator):
        super().__init__()
        self.simulator = simulator
        self.setWidgetResizable(True)
        self.setMinimumWidth(240)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        self.body = QVBoxLayout(content)
        self.body.setContentsMargins(6, 6, 6, 6)
        self.incidence = QLabel("Mean-ray incidence: validate the current geometry")
        self.incidence.setWordWrap(True)
        self.body.addWidget(self.incidence)
        self.sampling = QLabel()
        self.sampling.setWordWrap(True)
        self.body.addWidget(self.sampling)
        self.sections = {}
        for index, name in enumerate(
            (
                "Incident angle / Geometry",
                "Mosaic Broadening",
                "Detector",
                "Beam Controls",
                "Sample / Structure",
                "Sampling / Optics",
            )
        ):
            header = QToolButton()
            header.setText(name)
            header.setCheckable(True)
            header.setChecked(index < 2)
            header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            header.setArrowType(Qt.ArrowType.DownArrow if index < 2 else Qt.ArrowType.RightArrow)
            self.body.addWidget(header)
            section = QWidget()
            form = QVBoxLayout(section)
            form.setContentsMargins(0, 2, 0, 6)
            section.setVisible(index < 2)
            header.toggled.connect(section.setVisible)
            header.toggled.connect(
                lambda expanded, h=header: h.setArrowType(
                    Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
                )
            )
            self.body.addWidget(section)
            self.sections[name] = form
        self.body.addStretch()
        self.setWidget(content)
        self.controls = {}
        self.structure = None
        self.draft = None

    def sync(self):
        draft = self.simulator.active_draft
        if draft == self.draft:
            return
        self.draft = draft
        entries = self.simulator.quick_fields()
        structure = tuple(
            (key, group, unit, integer, probability)
            for key, group, label, unit, _value, integer, probability in entries
        )
        if structure != self.structure:
            self.structure = structure
            self.controls.clear()
            for form in self.sections.values():
                while form.count():
                    item = form.takeAt(0)
                    item.widget().deleteLater()
            for key, group, label, unit, value, integer, probability in entries:
                control = ScalarControl(
                    label, unit, value, integer=integer, probability=probability
                )
                control.setObjectName("quick_" + str(key))
                control.edited.connect(lambda text, k=key: self.simulator.quick_edit(k, text))
                control.pending.connect(
                    lambda schedule, k=key, c=control: self.simulator.quick_pending(
                        k, c.text.text(), schedule
                    )
                )
                control.drag_started.connect(lambda k=key: self.simulator.begin_slider_drag(k))
                control.dragged.connect(lambda text, k=key: self.simulator.drag_slider(k, text))
                control.drag_finished.connect(
                    lambda text, k=key: self.simulator.end_slider_drag(k, text)
                )
                self.sections[group].addWidget(control)
                self.controls[key] = control
            note = QLabel(
                "Full orientation matrices, axes/pivots and optional declarations: Advanced/actions."
            )
            note.setWordWrap(True)
            self.sections["Incident angle / Geometry"].addWidget(note)
            if self.simulator.draft_kind.currentData() == "configured":
                note = QLabel(
                    "Lattice and site ADPs come from the CIF. This route has no independent lattice/Debye sliders."
                )
                note.setWordWrap(True)
                self.sections["Sample / Structure"].addWidget(note)
            if not entries:
                note = QLabel("Load a configuration or native recipe in Advanced/actions.")
                note.setWordWrap(True)
                self.sections["Incident angle / Geometry"].addWidget(note)
        for key, _group, label, unit, value, _integer, _probability in entries:
            self.controls[key].title.setText(f"{label} ({unit})")
            self.controls[key].set_value(value)
        if draft is not None and hasattr(draft, "draw_count"):
            source = self.simulator._mapping.get("source", {})
            self.sampling.setText(
                f"Requested: {source.get('sample_count')} source states; "
                f"{draft.draw_count} detector draws/source; source seed {source.get('seed')}; "
                f"detector seed {draft.detector_seed}. Counts are nominal, not convergence."
            )
        else:
            self.sampling.setText(
                "Native deterministic integration: Advanced/actions declares its complete source rule."
            )
