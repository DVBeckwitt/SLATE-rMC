"""Device picking and exact shared-draft controls in the Simulator workspace."""

import json
import math

import numpy as np
from experiment_scene import ExperimentScenePanel, PhysicalHandle
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from simulation_dashboard import ScalarControl
from simulation_state import simulation_draft_document
from simulation_widgets import NumberEdit


class SimulatorScene(ExperimentScenePanel):
    def __init__(self, simulator):
        super().__init__(simulator, camera_controls=False)
        self.simulator = simulator
        self.current_draft = None
        self.state = None
        self._seen = None
        self._requested = None
        self._continue = False
        self._key = None
        self._entries = []
        self._syncing = False
        self._steps = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self.request)
        self.escape = QShortcut("Escape", self)
        self.escape.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.escape.activated.connect(self.cancel)
        self.view.selection_mode = True
        self.view.setMinimumHeight(140)
        self.view.object_picked.connect(self.pick)
        self.view.physical_started.connect(self.begin)
        self.view.physical_preview.connect(self.preview)
        self.view.physical_committed.connect(self.finish)
        self.view.physical_canceled.connect(self.cancel)
        self.view.camera_changed.connect(self.update_handle)
        # Keep the shared physical editor's camera controls unchanged.
        camera = QHBoxLayout()
        self.camera = QComboBox()
        self.camera.setAccessibleName("Scene camera preset; view only")
        for label, name in (
            ("Experiment", "context"),
            ("Front", "front"),
            ("Side", "side"),
            ("Along beam", "beam"),
            ("Detector normal", "detector"),
            ("Sample", "sample"),
        ):
            self.camera.addItem(label, name)
        self.camera.activated.connect(lambda: self.view.preset(self.camera.currentData()))
        camera.addWidget(self.camera, 1)
        reset = QPushButton("Reset")
        reset.setToolTip("Restore the experiment camera; view only")
        reset.clicked.connect(lambda: self.view.preset("context"))
        camera.addWidget(reset)
        self.focus_button = QPushButton("Focus")
        self.focus_button.setToolTip("Frame the selected device; view only")
        self.focus_button.clicked.connect(self.focus_device)
        camera.addWidget(self.focus_button)
        self.maximize_button = QPushButton("Maximize")
        camera.addWidget(self.maximize_button)
        self.layout().insertLayout(0, camera)
        self.hint.hide()
        self.hint.setToolTip(
            "Click a device or arrow to edit. Empty left drag: orbit; right drag: pan; wheel: zoom. Esc cancels a handle; Undo reverses one gesture."
        )
        self.devices = QComboBox()
        self.devices.setAccessibleName("Scene device selection; same targets as figure labels")
        self.devices.addItems(
            [
                "Beam",
                "Goniometer base",
                "Mount",
                "Sample",
                "Crystal / material",
                "Mosaic",
                "Detector",
                "External path",
                "Numerics / optics",
                "Display",
                "Identity",
            ]
        )
        self.devices.setToolTip("All devices and secondary declarations")
        self.layout().addWidget(self.devices)
        self.incidence = QLabel("Mean-ray incidence: current geometry unavailable")
        self.incidence.setWordWrap(True)
        self.layout().addWidget(self.incidence)
        self.motor_order = QLabel()
        self.motor_order.setWordWrap(True)
        self.layout().addWidget(self.motor_order)
        self.pinned_mosaic = QWidget()
        self.pinned_layout = QGridLayout(self.pinned_mosaic)
        self.pinned_layout.setContentsMargins(0, 0, 0, 0)
        self.pinned_controls = {}
        self._pinned_keys = ()
        self.layout().addWidget(self.pinned_mosaic)
        self.parameter = QComboBox()
        self.parameter.setAccessibleName("Selected device parameter")
        self.parameter.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.parameter.setMinimumContentsLength(15)
        self.fields = QWidget()
        self.fields_layout = QGridLayout(self.fields)
        self.fields_layout.setContentsMargins(0, 0, 0, 0)
        self.common_controls = {}
        self._common_keys = ()
        self.fields_scroll = QScrollArea()
        self.fields_scroll.setWidgetResizable(True)
        self.fields_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.fields_scroll.setWidget(self.fields)
        self.layout().addWidget(self.fields_scroll)
        fine_row = QHBoxLayout()
        self.fine_button = QPushButton("Fine adjust / more")
        self.fine_button.setCheckable(True)
        fine_row.addWidget(self.fine_button)
        undo, redo = QPushButton("Undo"), QPushButton("Redo")
        undo.clicked.connect(lambda: simulator.history_step(True))
        redo.clicked.connect(lambda: simulator.history_step(False))
        fine_row.addWidget(undo)
        fine_row.addWidget(redo)
        self.layout().addLayout(fine_row)
        self.fine = QWidget()
        fine_layout = QVBoxLayout(self.fine)
        fine_layout.setContentsMargins(0, 0, 0, 0)
        fine_layout.addWidget(self.parameter)
        self.control = ScalarControl("Select a parameter", "declared units", 0.0)
        fine_layout.addWidget(self.control)
        row = QHBoxLayout()
        self.step = QLineEdit("0.0001")
        self.step.setMaximumWidth(100)
        self.step.setAccessibleName("Exact keyboard step in selected parameter units")
        label = QLabel("Step")
        label.setBuddy(self.step)
        row.addWidget(label)
        row.addWidget(self.step)
        fine_layout.addLayout(row)
        self.layout().addWidget(self.fine)
        self.fine.hide()
        self.fine_button.toggled.connect(self.fine.setVisible)
        self.note = QLabel("Load a draft through Menu > Advanced parameters.")
        self.note.setWordWrap(True)
        self.layout().addWidget(self.note)
        self.step.editingFinished.connect(self.set_step)
        self.devices.currentTextChanged.connect(self.select)
        self.parameter.currentIndexChanged.connect(self.select_parameter)
        self.control.edited.connect(self.edit)
        self.control.drag_started.connect(self.begin)
        self.control.dragged.connect(lambda text: self.preview(float(text)))
        self.control.drag_finished.connect(lambda text: self.finish(float(text)))

    def set_step(self):
        try:
            value = float(self.step.text())
            if (
                not math.isfinite(value)
                or value <= 0
                or (self.control.integer and not value.is_integer())
            ):
                raise ValueError("Step must be positive, finite and whole for integer parameters")
            self.control.step = value
            self._steps[self._key] = value
            self.view.keyboard_step = value
        except ValueError as exc:
            self.note.setText(str(exc))

    def sync(self):
        draft = self.simulator.active_draft
        owner = (
            self.simulator.shell.project.project_id,
            self.simulator.draft_kind.currentData(),
            draft,
        )
        if owner != self._seen:
            if self._seen is not None and owner[:2] != self._seen[:2]:
                self.current_draft = None
            self._seen = owner
            self._requested = None
            self.select()
            self.note.setText(
                "Updating current draft geometry; previous geometry is historical. No intensity calculated here."
            )
            self.view.set_physical_handle(None)
            if draft is None:
                self.current_draft = self.state = None
                self.view.set_geometry(None, None)
            else:
                self._timer.start()
        self.texture()

    def workspace_changed(self, index):
        if index == 1:
            self.sync()
            if self.current_draft != self.simulator.active_draft:
                self.request()

    def request(self, *, continue_update=False):
        self._timer.stop()
        s = self.simulator
        if (
            s.active_draft is None
            or s.shell._close_intent
            or s.shell._pending_open is not None
            or s.shell.workspaces.currentIndex() != 1
        ):
            return
        self._continue |= continue_update
        identity = s.shell.project.project_id, s.epoch, s.active_draft
        if identity == self._requested:
            return
        self._requested = identity
        if s.draft_kind.currentData() == "native":
            from native_simulation_state import native_draft_document

            document = native_draft_document(s.active_draft)
        else:
            document = simulation_draft_document(s.active_draft)
        s.shell._request_simulation(
            "scene_geometry",
            json.dumps({"kind": s.draft_kind.currentData(), "draft": document}).encode(),
        )

    def ready(self, state):
        if state.draft != self.simulator.active_draft:
            return
        self.state = state
        self.current_draft = state.draft
        identity = (
            self.simulator.shell.project.project_id,
            state.draft.draft_id,
            state.draft.revision,
        )
        self.view.set_geometry(state.geometry, identity)
        self.view.callouts = (
            ("Beam", state.geometry.beam_source_lab_m),
            ("Goniometer base", state.base_lab_m),
            ("Mount", state.mount_lab_m),
            ("Sample", state.geometry.sample_lab_m),
            ("Detector", tuple(state.instrument.lab_from_detector.translation_m)),
            ("Mosaic", state.geometry.sample_lab_m),
            ("Crystal / material", state.geometry.sample_lab_m),
            (
                "External path",
                tuple(
                    (
                        np.asarray(state.geometry.sample_lab_m)
                        + state.instrument.lab_from_detector.translation_m
                    )
                    / 2
                ),
            ),
            *tuple(
                (f"Axis {i + 1}", pivot)
                for i, (pivot, _axis, _angle) in enumerate(state.geometry.goniometer_axes_lab)
            ),
        )
        for index in range(self.devices.count() - 1, -1, -1):
            name = self.devices.itemText(index)
            if name.startswith("Axis ") and int(name.split()[1]) > len(
                state.geometry.goniometer_axes_lab
            ):
                self.devices.removeItem(index)
        if not hasattr(state.draft, "yaml_text"):
            self.view.callouts = tuple(
                item for item in self.view.callouts if item[0] not in ("Goniometer base", "Mount")
            )
        for i in range(len(state.geometry.goniometer_axes_lab)):
            name = f"Axis {i + 1}"
            if self.devices.findText(name) < 0:
                self.devices.addItem(name)
        self.select()
        self.texture()
        if self._continue:
            self._continue = False
            self.simulator.request_update(force=True)

    def texture(self):
        frame = self.simulator.frame
        matching = (
            self.state is not None
            and self.current_draft == self.simulator.active_draft
            and frame is not None
            and frame.draft == self.current_draft
            and frame.display is not None
        )
        level = self.simulator.detector.view.active_display_level if matching else None
        bin_size = 1 if level is None else level.bin_size
        identity = (
            (frame.draft.draft_id, f"{frame.run_id}/display-sum:{bin_size}", frame.draw_prefix)
            if matching
            else None
        )
        display = None if not matching else frame.display if level is None else level.values
        self.view.set_detector_image(
            display,
            identity,
            display_bin=bin_size,
            native_shape=(1, 1) if not matching else frame.image.shape,
        )
        detector = self.simulator.detector.view
        self.view.set_levels(detector.low_value, detector.high_value, detector.contrast_mode)

    def pick(self, name):
        if name.startswith("axis "):
            name = "Axis " + name.split()[1]
        elif name.startswith("detector"):
            name = "Detector"
        elif name == "incident beam":
            name = "Beam"
        elif name.startswith("holder"):
            name = "Mount"
        elif name in ("sample", "x", "y", "z"):
            name = "Sample"
        index = self.devices.findText(name)
        if index >= 0:
            self.devices.setCurrentIndex(index)
            self.parameter.setFocus()

    def entries(self):
        s = self.simulator
        device = self.devices.currentText()
        result = []
        if s.draft_kind.currentData() == "native" and device in ("Mount", "Goniometer base"):
            return result
        for entry in s.quick_fields():
            (path, indices), group, _label, _unit, value, _integer, _probability = entry
            target = "Numerics / optics"
            if group == "Beam Controls":
                target = "Beam"
            elif group == "Mosaic Broadening":
                target = "Mosaic"
            elif group == "Detector":
                target = "Detector"
            elif group == "Sample / Structure":
                target = "Crystal / material"
            elif group == "Incident angle / Geometry":
                target = f"Axis {indices[0] + 1}" if path[-1] == "axis_rotations" else "Sample"
            if target == device or (device == "Mount" and target == "Sample"):
                result.append(entry)
        if s.draft_kind.currentData() == "configured" and s._mapping is not None:
            instrument = s._mapping["instrument"]
            if device == "Goniometer base":
                for i, value in enumerate(instrument["lab_from_goniometer_zero"]["translation_m"]):
                    result.append(
                        (
                            (("instrument", "lab_from_goniometer_zero", "translation_m"), (i,)),
                            "",
                            f"Base {'XYZ'[i]}",
                            "m LAB",
                            value,
                            False,
                            False,
                        )
                    )
            if device == "External path":
                value = instrument.get("detector_path_linear_attenuation_m_inv")
                if value is not None:
                    result.append(
                        (
                            (("instrument", "detector_path_linear_attenuation_m_inv"), ()),
                            "",
                            "External attenuation",
                            "1/m",
                            value,
                            False,
                            False,
                        )
                    )
        return result

    def select(self, *_):
        old = self._key
        self._entries = self.entries()
        self._syncing = True
        self.parameter.clear()
        for key, _group, label, unit, _value, _integer, _probability in self._entries:
            self.parameter.addItem(f"{label} [{unit}]", key)
        index = next((i for i, entry in enumerate(self._entries) if entry[0] == old), 0)
        self.parameter.setCurrentIndex(max(0, index))
        self._syncing = False
        self.select_parameter()
        self.sync_common()

    def select_parameter(self, *_):
        if self._syncing:
            return
        index = self.parameter.currentIndex()
        self._key = self._entries[index][0] if 0 <= index < len(self._entries) else None
        available = 0 <= index < len(self._entries)
        self.control.setVisible(available)
        self.step.setEnabled(available)
        device = self.devices.currentText()
        self.view.selected_object = device
        self.focus_button.setEnabled(self.state is not None)
        legend = "Left drag: orbit; right: pan; wheel: zoom."
        if device in ("Sample", "Mount"):
            legend = "Sample normal +n: blue."
        elif device == "Detector":
            legend = "+column: cyan; +row: pink; crosshair: pixel reference."
        elif device.startswith("Axis "):
            legend = f"{device}: declared LAB axis/pivot; angle in degrees."
        self.hint.setText("Holder/axes schematic; positions in metres.\n" + legend)
        note = "Declared rigid orientations, axes/pivots and coupled beam direction/basis: Menu > Advanced parameters."
        if (
            device in ("Mount", "Goniometer base", "Sample")
            and self.simulator.draft_kind.currentData() == "native"
        ):
            note = "Native recipe supplies a LAB sample pose; no separately declared mount or motor chain. Exact pose values are shared with Advanced."
        elif device == "Identity":
            draft = self.simulator.active_draft
            note = (
                "No draft"
                if draft is None
                else f"Draft {draft.draft_id}, revision {draft.revision}; geometry and intensity identities are separate. Full provenance: Advanced parameters."
            )
        elif device == "Display":
            note = "Detector: Auto 99%, Display exposure, Upper and Full range. Scene camera controls change the view only."
        elif device == "Beam":
            note += " Source reference plane; glyph is not an asserted tube/slit. Width and divergence are nonspatial parameters."
        elif device == "Mosaic":
            note = "Physical distribution parameters; no unqualified probability cone. Angular integration controls belong to Numerics / optics."
        if self.state is not None:
            note += f" Mean-ray glancing incidence (derived): {self.state.incidence_deg:.6g} deg. Geometry revision {self.state.draft.revision}; sample patch/holder schematic where support is unbounded."
        self.note.setToolTip(note + "\n" + self.hint.text())
        if self.state is not None and available:
            note = f"Incidence {self.state.incidence_deg:.6g} deg (derived) · " + legend
        elif (
            self.state is not None
            and device in ("Mount", "Goniometer base")
            and self.simulator.draft_kind.currentData() == "native"
        ):
            note = "Native LAB sample pose; no separately declared mount or motor chain. See Advanced parameters."
        self.note.setText(note)

        if available:
            _key, _group, label, unit, value, integer, probability = self._entries[index]
            self.control.integer, self.control.probability = integer, probability
            self.control.slider.blockSignals(True)
            self.control.slider.setRange(0, 1000) if probability else self.control.slider.setRange(
                -1000, 1000
            )
            self.control.slider.blockSignals(False)
            self.control.title.setText(f"{device}: {label} ({unit})")
            self.control.text.setAccessibleName(f"{device}: {label}, exact {unit}")
            self.control.set_value(value)
            self.control.span = (
                1.0 if integer else (0.005 if unit.startswith("m ") or unit == "m" else 1.0)
            )
            self.control.step = self._steps.get(
                self._key, 1.0 if integer else self.control.span / 100
            )
            self.step.setText(repr(self.control.step))
            self.view.keyboard_step = self.control.step
        self.update_handle()

    def focus_device(self):
        target = next(
            (point for name, point in self.view.callouts if name == self.devices.currentText()),
            None,
        )
        if target is not None:
            self.view.focus(self.devices.currentText().lower(), target)

    def activate_field(self, key):
        index = next((i for i, entry in enumerate(self._entries) if entry[0] == key), -1)
        if index >= 0:
            self.parameter.setCurrentIndex(index)

    def edit_field(self, key, text):
        self.activate_field(key)
        self.edit(text)

    def sync_common(self):
        self.sync_pinned()
        device = self.devices.currentText()
        entries = self._entries
        if device == "Beam":
            entries = [
                e
                for e in entries
                if e[0][0][-1] in ("spatial_sigma_m", "divergence_sigma_rad", "mean_wavelength_A")
            ]
        elif device == "Detector":
            entries = [
                e
                for e in entries
                if e[0][0][-1] in ("translation_m", "about_row_axis_deg", "about_column_axis_deg")
            ]
        elif device not in (
            "Sample",
            "Mount",
            "Goniometer base",
            "Mosaic",
            "Crystal / material",
        ) and not device.startswith("Axis "):
            entries = []
        keys = tuple(e[0] for e in entries)
        if keys != self._common_keys:
            while self.fields_layout.count():
                item = self.fields_layout.takeAt(0)
                item.widget().hide()
                item.widget().deleteLater()
            self.common_controls.clear()
            self._common_keys = keys
            for i, (key, _group, label, unit, _value, _integer, _probability) in enumerate(entries):
                field = NumberEdit()
                field.setAccessibleName(f"{device}: {label}, exact {unit}")
                short = (
                    label.removeprefix("Sample offset ")
                    .removeprefix("Detector ")
                    .removeprefix("Beam origin ")
                )
                if device.startswith("Axis "):
                    short = f"{device} motor angle"
                caption = QLabel(f"{short} ({unit})")
                caption.setWordWrap(False)
                caption.setBuddy(field)
                caption.setToolTip(field.accessibleName())
                row, col = i, 0
                self.fields_layout.addWidget(caption, row, col)
                self.fields_layout.addWidget(field, row, col + 1)
                field.activated.connect(lambda k=key: self.activate_field(k))
                field.committed.connect(lambda value, k=key: self.edit_field(k, value))
                self.common_controls[key] = field
        for key, _group, _label, _unit, value, integer, _probability in entries:
            self.common_controls[key].setText(str(value) if integer else repr(float(value)))
        self.fields_scroll.setVisible(bool(entries))
        if entries:
            rows = len(entries)
            row_height = max(field.sizeHint().height() for field in self.common_controls.values())
            height = min(
                180, rows * row_height + (rows - 1) * self.fields_layout.verticalSpacing() + 8
            )
            self.fields_scroll.setFixedHeight(height)

    def sync_pinned(self):
        s = self.simulator
        matching = self.state is not None and self.current_draft == s.active_draft
        self.incidence.setText(
            f"Mean-ray incidence {self.state.incidence_deg:.7g}° (derived from current geometry)"
            if matching
            else "Mean-ray incidence: waiting for current geometry"
        )
        rotations = [] if s._mapping is None else s._mapping["instrument"].get("axis_rotations", [])
        self.motor_order.setText(
            "Motor order: "
            + " → ".join(f"Axis {i + 1} LAB {r['axis_lab']}" for i, r in enumerate(rotations))
            if rotations
            else "Native LAB sample pose; no separately declared motor chain"
            if s.draft_kind.currentData() == "native"
            else "No declared motors"
        )
        entries = [e for e in s.quick_fields() if e[1] == "Mosaic Broadening"]
        keys = tuple(e[0] for e in entries)
        if keys != self._pinned_keys:
            while self.pinned_layout.count():
                self.pinned_layout.takeAt(0).widget().deleteLater()
            self.pinned_controls.clear()
            self._pinned_keys = keys
            for column, (key, _group, label, unit, _value, _integer, _probability) in enumerate(
                entries
            ):
                field = NumberEdit()
                field.setAccessibleName(f"Pinned mosaic: {label}, exact {unit}")
                caption = QLabel(f"{label} ({unit})")
                caption.setWordWrap(True)
                caption.setBuddy(field)
                self.pinned_layout.addWidget(caption, 0, column)
                self.pinned_layout.addWidget(field, 1, column)
                field.committed.connect(lambda text, k=key: s.quick_edit(k, text))
                self.pinned_controls[key] = field
        for key, _group, _label, _unit, value, integer, _probability in entries:
            self.pinned_controls[key].setText(str(value) if integer else repr(float(value)))
            self.pinned_controls[key].setEnabled(s._drag is None)
        self.pinned_mosaic.setVisible(bool(entries))

    def update_handle(self):
        if (
            self._key is None
            or self.state is None
            or self.current_draft != self.simulator.active_draft
        ):
            self.view.set_physical_handle(None)
            return
        path, indices = self._key
        state = self.state
        instrument = state.instrument
        pivot, axis, kind, scale = None, None, "translation", 1.0
        value = self._entries[self.parameter.currentIndex()][4]
        unit = self._entries[self.parameter.currentIndex()][3]
        if path == ("source", "mean_origin_lab_m"):
            pivot, axis = state.geometry.beam_source_lab_m, np.eye(3)[indices[0]]
        elif path[-1] == "translation_m" and indices:
            target = path[1]
            if target == "lab_from_detector":
                pivot, axes = instrument.lab_from_detector.translation_m, np.eye(3)
            elif target == "lab_from_goniometer_zero":
                pivot, axes = state.base_lab_m, np.eye(3)
            elif target == "goniometer_from_sample":
                pivot, axes = state.geometry.sample_lab_m, np.asarray(state.mount_axes_lab).T
            elif target == "lab_from_sample":
                pivot, axes = state.geometry.sample_lab_m, np.eye(3)
            else:
                axes = None
            if axes is not None:
                axis = axes[:, indices[0]]
        elif path[-1] == "axis_rotations" and len(indices) == 2 and indices[1] == 3:
            pivot, axis, _angle = state.geometry.goniometer_axes_lab[indices[0]]
            kind = "rotation"
        elif path[-1] == "detector_reference_coordinate_px":
            pivot, axis = (
                instrument.lab_from_detector.translation_m,
                -instrument.lab_from_detector.rotation[:, indices[0]],
            )
            scale = (
                instrument.detector_column_pitch_m
                if indices[0] == 0
                else instrument.detector_row_pitch_m
            )
        elif "detector_tilt" in path:
            pivot = instrument.lab_from_detector.translation_m
            kind = "rotation"
            if path[-1] == "about_row_axis_deg":
                axis = instrument.lab_from_detector.rotation[:, 1]
            else:
                tilt = self.simulator._mapping["instrument"]["detector_tilt"]["about_row_axis_deg"]
                angle = math.radians(tilt)
                axis = instrument.lab_from_detector.rotation @ np.asarray(
                    (math.cos(angle), 0, math.sin(angle))
                )
        self.view.selected_object = self.devices.currentText()
        handle = (
            None
            if axis is None
            else PhysicalHandle(
                str(self._key),
                tuple(pivot),
                tuple(axis),
                kind,
                float(value),
                "deg" if unit == "deg" else unit,
                scale,
                f"{self.devices.currentText()}: {self.parameter.currentText()}; positive handle; Esc cancels",
            )
        )
        self.view.set_physical_handle(handle)

    def edit(self, text):
        if self._key is None:
            return
        try:
            value = float(text)
            path, _indices = self._key
            if not math.isfinite(value):
                raise ValueError("Use a finite exact value")
            if self.control.integer and not value.is_integer():
                raise ValueError("This declaration requires an integer")
            if self.control.probability and not 0 <= value <= 1:
                raise ValueError("Probability must be between zero and one")
            if path[-1] in ("detector_row_pitch_m", "detector_column_pitch_m") and value <= 0:
                raise ValueError("Pixel pitch must be positive")
        except ValueError as exc:
            self.note.setText(str(exc))
            return
        self.simulator.quick_edit(self._key, text)

    def begin(self):
        if self._key is not None:
            self.devices.setEnabled(False)
            self.parameter.setEnabled(False)
            self.fields.setEnabled(False)
            for button in self.device_buttons.values():
                button.setEnabled(False)
            self.simulator.begin_slider_drag(self._key)

    def preview(self, value):
        if self._key is not None and self.simulator._drag is not None:
            self.simulator.drag_slider(self._key, format(value, ".17g"))
            # Current immutable draft advances during the gesture; canonical landmarks
            # follow on the same latest-only worker, never in the pointer handler.
            self.simulator.flush_quick_edit()

    def finish(self, value):
        if self._key is not None and self.simulator._drag is not None:
            self.simulator.end_slider_drag(self._key, format(value, ".17g"))
            self.devices.setEnabled(True)
            self.parameter.setEnabled(True)
            self.fields.setEnabled(True)
            self.select_parameter()

    def cancel(self):
        if self.view._physical_drag:
            self.view.cancel_physical_gesture()
            return
        s = self.simulator
        self.devices.setEnabled(True)
        self.parameter.setEnabled(True)
        self.fields.setEnabled(True)
        s._pending_quick = None
        action = s._drag_action
        s._drag_timer.stop()
        s._drag = s._drag_action = None
        if action is not None:
            s.history_step(True)
            history = s.native.history if s.draft_kind.currentData() == "native" else s.history
            if history.redo_actions and history.redo_actions[-1] is action:
                history.redo_actions.pop()
                history.bytes_used -= action.bytes_used
        self.select_parameter()

    def stop(self):
        self.devices.setEnabled(True)
        self.parameter.setEnabled(True)
        self.fields.setEnabled(True)
        self.select_parameter()
        # Stop retains the latest draft while ending pointer ownership.
        self.view._physical_drag = False
        self.view._press = self.view._last = None
        self.view._gesture_handle = None
        self._timer.stop()
        self._continue = False
        self._requested = None
