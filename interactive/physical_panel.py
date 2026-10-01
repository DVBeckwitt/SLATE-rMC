"""Physical initial-value editor, explicit center adoption and local sensitivity."""

import json
import math
from dataclasses import replace
from uuid import UUID

import numpy as np
from experiment_scene import ExperimentScenePanel, PhysicalHandle
from hbn_state import hbn_session_document, hbn_session_from_document
from joint_state import joint_session_from_document
from parameter_state import FieldChange, SessionHistory, _action
from physical_io import (
    changed_snapshot,
    fields,
    joint_image_identity,
    numeric_from_document,
    snapshot,
    split_joint_image_identity,
)
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from reciprocal_preview import ReciprocalCoverageView, ReciprocalPreview
from sample_state import encoded, payload_hash, sample_session_from_document
from simulation_state import simulation_draft_from_document


class NativeFeatureView(QWidget):
    """Small native-coordinate overlay; no image transform or scientific calculation."""

    def __init__(self):
        super().__init__()
        self.views = ()
        self.identity = None
        self.background = None
        self.setMinimumHeight(180)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#18232b"))
        good = [v for v in self.views if v.mapping is not None and not v.error]
        if not good:
            painter.setPen(QColor("#bbcbd3"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Native feature overlay unavailable")
            return
        rows, columns = good[0].mapping.instrument.detector_shape_rc
        box = QRectF(30, 10, max(1, self.width() - 45), max(1, self.height() - 35))
        painter.setPen(QPen(QColor("#68838e"), 1))
        painter.drawRect(box)
        if self.background is not None:
            painter.drawPixmap(box, self.background, QRectF(self.background.rect()))
        for view, color in zip(good, ("#61d5bd", "#ffcd70", "#aab7ff")[: len(good)], strict=True):
            painter.setPen(QPen(QColor(color), 1))
            for column, row in view.coordinates_px:
                if -0.5 <= column <= columns - 0.5 and -0.5 <= row <= rows - 0.5:
                    p = QPointF(
                        box.left() + (column + 0.5) / columns * box.width(),
                        box.top() + (row + 0.5) / rows * box.height(),
                    )
                    painter.drawEllipse(p, 1.5, 1.5)
        painter.setPen(QColor("#bbcbd3"))
        painter.drawText(
            30,
            self.height() - 5,
            "column_px →; row_px ↓; thumbnail context; native-coordinate features",
        )


class PhysicalPanel(QDialog):
    def __init__(self, shell):
        super().__init__(shell)
        self.shell = shell
        self.epoch = 0
        self.baseline = None
        self.candidate = None
        self.result = None
        self.proposal = None
        self.history = SessionHistory()
        self.settings_json = "{}"
        self._rendering = False
        self._commit_requested = False
        self._gesture = False
        self._drag_value = None
        self._proposal_source = None
        self.setWindowTitle("Physical geometry / initial estimates / sensitivity")
        self.resize(1190, 850)
        layout = QVBoxLayout(self)
        self.status = QLabel(
            "Select a route and load its admitted canonical initial values. No action starts a fit."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.route = QComboBox()
        for title, key in (
            ("Acquisition configuration", "configuration"),
            ("Independent simulator", "simulator"),
            ("hBN private calibration", "hbn"),
            ("Sample geometry series", "sample"),
            ("Joint geometry vector", "joint"),
        ):
            self.route.addItem(title, key)
        row.addWidget(self.route)
        self.button(row, "Load current route", self.load)
        self.images = QComboBox()
        self.images.setMinimumWidth(200)
        row.addWidget(self.images)
        self.button(row, "Cancel calculation / gesture", self.cancel)
        split = QSplitter()
        layout.addWidget(split, 1)
        left, right = QWidget(), QWidget()
        split.addWidget(left)
        split.addWidget(right)
        a, b = QVBoxLayout(left), QVBoxLayout(right)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "Owner coordinate",
                "Initial",
                "Unit",
                "Bounds / domain",
                "Frame",
                "Role",
                "Affected images",
            ]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.selected)
        a.addWidget(self.table, 1)
        self.callout = QLabel()
        self.callout.setWordWrap(True)
        a.addWidget(self.callout)
        values = QHBoxLayout()
        a.addLayout(values)
        self.value = QLineEdit()
        values.addWidget(self.value)
        self.apply_button = self.button(values, "Preview / apply exact initial", self.apply_typed)
        self.button(values, "- fine", lambda: self.fine(-1))
        self.button(values, "+ fine", lambda: self.fine(1))
        history = QHBoxLayout()
        a.addLayout(history)
        self.button(history, "Undo", lambda: self.history_step(True))
        self.button(history, "Redo", lambda: self.history_step(False))
        self.button(history, "Cancel preview", self.cancel)
        self.tabs = QTabWidget()
        a.addWidget(self.tabs)
        centers, sensitivity = QWidget(), QWidget()
        self.tabs.addTab(centers, "Geometry-derived center")
        self.tabs.addTab(sensitivity, "Bounded sensitivity")
        c, d = QVBoxLayout(centers), QVBoxLayout(sensitivity)
        self.source = QComboBox()
        self.source.addItem("Genuine hBN calibration result", "hbn")
        self.source.addItem("Supported sample geometry result", "sample")
        c.addWidget(self.source)
        self.source.currentIndexChanged.connect(self.fill_results)
        self.results = QComboBox()
        c.addWidget(self.results)
        self.button(c, "Preview / compare center", self.request_proposal)
        self.target = QComboBox()
        self.target.addItem("hBN initial beam center", "hbn")
        self.target.addItem("Sample detector calibration starts (coupled pose)", "sample")
        self.target.currentIndexChanged.connect(self.compare_proposal)
        c.addWidget(self.target)
        self.shared = QCheckBox(
            "Adopt in compatible retained hBN starts (lists affected images below)"
        )
        self.shared.toggled.connect(self.compare_proposal)
        c.addWidget(self.shared)
        self.proposal_text = QTextBrowser()
        c.addWidget(self.proposal_text)
        self.adopt_button = self.button(c, "Use as initial estimate", self.adopt)
        self.step = QLineEdit("0.0001")
        d.addWidget(
            QLabel("Positive perturbation in the selected stored unit; baseline and ± step only")
        )
        d.addWidget(self.step)
        self.sensitivity_button = self.button(d, "Request local sensitivity", self.sensitivity)
        self.sensitivity_text = QTextBrowser()
        d.addWidget(self.sensitivity_text)
        self.sensitivity_text.setPlainText(
            "Sensitivity is feature motion with other coordinates held fixed. Measurement error and parameter propagation are separate. Prediction bands unavailable: these owners do not retain a qualified full covariance with a supported observable mapping."
        )
        self.scene = ExperimentScenePanel()
        b.addWidget(self.scene, 2)
        self.features = NativeFeatureView()
        b.addWidget(self.features, 1)
        self.reciprocal = ReciprocalCoverageView()
        b.addWidget(self.reciprocal, 1)
        self.identity = QLabel()
        self.identity.setWordWrap(True)
        b.addWidget(self.identity)
        self.route.currentIndexChanged.connect(self.route_changed)
        self.images.currentIndexChanged.connect(self.image_changed)
        view = self.scene.view
        view.physical_started.connect(self.begin_gesture)
        view.physical_preview.connect(self.drag_preview)
        view.physical_committed.connect(self.end_gesture)
        view.physical_canceled.connect(self.cancel)
        self._drag_timer = QTimer(self)
        self._drag_timer.setSingleShot(True)
        self._drag_timer.setInterval(100)
        self._drag_timer.timeout.connect(self.flush_drag)

    def button(self, layout, title, action):
        button = QPushButton(title)
        button.clicked.connect(lambda: self.guard(action))
        layout.addWidget(button)
        return button

    def guard(self, action):
        try:
            return action()
        except (ValueError, TypeError, RuntimeError, KeyError, OSError, StopIteration) as exc:
            self.status.setText(str(exc) or "Requested source is unavailable")
            return None

    def resident_bytes(self):
        result_bytes = 0 if self.result is None else self.result.nbytes
        thumbnail_bytes = (
            0
            if self.features.background is None
            else self.features.background.width() * self.features.background.height() * 4
        )
        return (
            result_bytes
            + thumbnail_bytes
            + sum(
                len(encoded(v).encode()) for v in (self.baseline, self.candidate) if v is not None
            )
        )

    def owner_changed(self):
        if self.baseline is not None:
            try:
                current = snapshot(self.shell, self.route.currentData())
            except (ValueError, TypeError, KeyError, OSError):
                current = None
            if current != self.baseline:
                self.cancel(clear=True)
                self.scene.view.set_scene(None, None, None, None)
                self.status.setText(
                    "Canonical owner changed; old physical preview is stale. Load current route."
                )

    def context(self):
        return (
            self.shell.project.project_id,
            self.shell.selected_acquisition_id,
            self.shell._revision,
            self.epoch,
            self.route.currentData(),
            self.images.currentData(),
        )

    def route_changed(self, *_args):
        if not self._rendering:
            self.cancel(clear=True)
            self.baseline = None
            self.table.setRowCount(0)
            self.guard(self.load)

    def image_changed(self, *_args):
        if not self._rendering and self.baseline is not None:
            self.cancel(clear=True)
            self.guard(lambda: self.request("preview"))

    def load(self):
        self.cancel(clear=True)
        self.baseline = snapshot(self.shell, self.route.currentData())
        self.candidate = None
        self._rendering = True
        self.images.clear()
        scopes = tuple(
            dict.fromkeys(
                im for f in fields(self.route.currentData(), self.baseline) for im in f.scope
            )
        )
        for identity in scopes:
            self.images.addItem(identity, identity)
        selected_identity = str(self.shell.selected_acquisition_id)
        if self.route.currentData() == "joint":
            selected_identity = joint_image_identity("hbn", selected_identity)
        index = self.images.findData(selected_identity)
        if index >= 0:
            self.images.setCurrentIndex(index)
        self._rendering = False
        self.render_fields()
        self.fill_results()
        self.request("preview")

    def render_fields(self, name=None):
        self._rendering = True
        rows = fields(self.route.currentData(), self.candidate or self.baseline)
        self.table.setRowCount(len(rows))
        for i, f in enumerate(rows):
            for j, value in enumerate(
                (
                    f.name,
                    "—" if f.value is None else format(f.value, ".17g"),
                    f.unit,
                    f"[{f.lower}, {f.upper}]" if f.lower is not None else "canonical input domain",
                    f.frame,
                    f.role,
                    ", ".join(f.scope),
                )
            ):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))
            if f.name == name:
                self.table.selectRow(i)
        if self.table.currentRow() < 0 and rows:
            self.table.selectRow(0)
        self._rendering = False
        self.selected()

    def field(self):
        if self.baseline is None or self.table.currentRow() < 0:
            raise ValueError("Select an admitted coordinate")
        return fields(self.route.currentData(), self.candidate or self.baseline)[
            self.table.currentRow()
        ]

    def selected(self):
        if self._rendering or self.baseline is None:
            return
        f = self.field()
        self.value.setText("" if f.value is None else format(f.value, ".17g"))
        self.value.setEnabled(f.editable)
        self.apply_button.setEnabled(f.editable)
        self.sensitivity_button.setEnabled(
            f.editable
            and f.name
            not in (
                "source.spatial_sigma_x",
                "source.spatial_sigma_y",
                "source.divergence_x",
                "source.divergence_y",
            )
        )
        self.callout.setText(
            f"{f.name}: {f.value} {f.unit}; {f.frame}; {f.role}; bounds [{f.lower}, {f.upper}]. Affected: {', '.join(f.scope)}. {f.reason}\nProvenance: canonical {self.route.currentData()} draft "
            + payload_hash(self.candidate or self.baseline)
            + (
                "\nJoint reference: Bi2Se3 sample x-tilt is fixed at zero and has no fitted coordinate."
                if self.route.currentData() == "joint"
                else ""
            )
        )
        self.bind_handle()

    def bind_handle(self):
        f = self.field()
        view = self.scene.view
        m = (
            None
            if self.result is None or not self.result.views
            else self.result.views[0 if self.result.step is not None else -1].mapping
        )
        if m is None or not f.editable:
            view.set_physical_handle(None)
            return
        instrument = m.instrument
        detector, sample = instrument.lab_from_detector, instrument.lab_from_sample
        name = f.name
        calibrant_view = self.result.views[0].observable.startswith("hBN")
        pivot, axis, kind, scale = detector.translation_m, None, "translation", 1.0
        if name.startswith("detector.") and name[-1] in "xyz" and name.count(".") == 1:
            axis = np.eye(3)["xyz".index(name[-1])]
        elif name.startswith("source.origin_"):
            pivot = m.source_origin_lab_m
            axis = np.eye(3)["xyz".index(name[-1])]
        elif name in (
            "detector.reference_column",
            "detector_reference_column_offset_px",
            "beam_center_column_px",
        ):
            axis = detector.rotation[:, 0] * (
                -1 if name.startswith("detector") or calibrant_view else 1
            )
            scale = instrument.detector_column_pitch_m
        elif name in (
            "detector.reference_row",
            "detector_reference_row_offset_px",
            "beam_center_row_px",
        ):
            axis = detector.rotation[:, 1] * (
                -1 if name.startswith("detector") or calibrant_view else 1
            )
            scale = instrument.detector_row_pitch_m
        elif name == "detector_plane_normal_offset_m":
            axis = detector.rotation[:, 2]
        elif name in ("calibrant_distance_m", "hbn_calibrant_distance_m") and calibrant_view:
            axis = np.asarray(view.geometry.beam_direction_lab)
        elif name in ("sample_plane_normal_offset_m", "bi2se3_zs_m", "bi2te3_zs_m"):
            pivot, axis = sample.translation_m, sample.rotation[:, 2]
        elif name in (
            "detector.column_tilt",
            "detector.row_tilt",
            "detector_column_tilt_rad",
            "detector_row_tilt_rad",
        ):
            y_name = (
                "detector.row_tilt" if name.startswith("detector.") else "detector_row_tilt_rad"
            )
            y = next(
                r.value
                for r in fields(self.route.currentData(), self.candidate or self.baseline)
                if r.name == y_name
            )
            y = math.radians(y) if f.unit == "deg" else y
            axis = (
                detector.rotation @ np.asarray((math.cos(y), 0, math.sin(y)))
                if "column" in name
                else detector.rotation[:, 1]
            )
            kind = "rotation"
            if calibrant_view:
                distance = next(
                    r.value
                    for r in fields(self.route.currentData(), self.candidate or self.baseline)
                    if r.name in ("calibrant_distance_m", "hbn_calibrant_distance_m")
                )
                pivot = sample.translation_m + distance * np.asarray(
                    view.geometry.beam_direction_lab
                )
        elif name in ("goniometer.angle_0", "incidence_angle_delta_rad") and m.axis_rotations:
            rotation = m.axis_rotations[0]
            pivot, axis, kind = rotation.pivot_lab_m, rotation.axis_lab, "rotation"
        elif name.startswith("goniometer_") and m.axis_rotations:
            rotation = m.axis_rotations[0]
            pivot = rotation.pivot_lab_m
            direction = np.asarray(rotation.axis_lab)
            horizontal = float(np.hypot(*direction[:2]))
            if horizontal > 1e-12:
                yaw = math.atan2(-direction[1], direction[0])
                pitch = math.atan2(direction[2], horizontal)
                if name == "goniometer_axis_yaw_rad":
                    axis, kind = np.asarray((0.0, 0.0, -1.0)), "rotation"
                elif name == "goniometer_axis_pitch_rad":
                    axis, kind = np.asarray((-math.sin(yaw), -math.cos(yaw), 0.0)), "rotation"
                elif name == "goniometer_pivot_yaw_offset_m":
                    axis = np.asarray((-math.sin(yaw), -math.cos(yaw), 0.0))
                elif name == "goniometer_pivot_pitch_offset_m":
                    axis = np.asarray(
                        (
                            -math.cos(yaw) * math.sin(pitch),
                            math.sin(yaw) * math.sin(pitch),
                            math.cos(pitch),
                        )
                    )
        elif (
            name
            in (
                "sample_normal_x_tilt_rad",
                "sample_normal_y_tilt_rad",
                "bi2te3_sample_x_tilt_rad",
                "bi2te3_sample_y_tilt_rad",
                "bi2se3_sample_y_tilt_rad",
            )
            and m.axis_rotations
        ):
            pivot, kind = m.axis_rotations[0].pivot_lab_m, "rotation"
            y_name = (
                "sample_normal_y_tilt_rad"
                if name.startswith("sample_")
                else name.split("_sample_")[0] + "_sample_y_tilt_rad"
            )
            y = next(
                r.value
                for r in fields(self.route.currentData(), self.candidate or self.baseline)
                if r.name == y_name
            )
            axis = (
                sample.rotation @ np.asarray((math.cos(y), 0, math.sin(y)))
                if "x_tilt" in name
                else sample.rotation[:, 1]
            )
        if (
            name in ("beam_center_column_px", "beam_center_row_px")
            and self.route.currentData() == "joint"
            and not calibrant_view
            and axis is not None
        ):
            beam = np.asarray(view.geometry.beam_direction_lab)
            axis = axis - beam * float(axis @ beam)
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                axis = None
            else:
                scale *= norm
                pivot = m.source_origin_lab_m
        if axis is None:
            view.set_physical_handle(None)
            self.callout.setText(
                self.callout.text()
                + "\nNo independent rigid handle for this source/model or coupled correction coordinate; exact numeric editing retains the canonical owner."
            )
            return
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        label = f"{name}: {f.value:.17g} {f.unit}; {f.role}; bounds [{f.lower}, {f.upper}]; affected {', '.join(f.scope)}; positive arrow; Escape cancels"
        view.set_physical_handle(
            PhysicalHandle(name, tuple(pivot), tuple(axis), kind, f.value, f.unit, scale, label)
        )

    def ensure_current(self):
        if self.baseline is None or snapshot(self.shell, self.route.currentData()) != self.baseline:
            raise ValueError("Canonical owner changed; load the current route before editing")

    def request(self, operation, **extra):
        self.ensure_current()
        name = extra.get("parameter")
        if name:
            f = next(f for f in fields(self.route.currentData(), self.baseline) if f.name == name)
            if self.images.currentData() not in f.scope and f.scope:
                self._rendering = True
                self.images.setCurrentIndex(self.images.findData(f.scope[0]))
                self._rendering = False
        self.epoch += 1
        if operation != "proposal":
            self.result = None
            self.identity.setText(
                "Canonical preview pending; retained visual is historical. Requested initial draft "
                + payload_hash(self.candidate or self.baseline)
                + (
                    "\nJoint reference: Bi2Se3 sample x-tilt is fixed at zero and has no fitted coordinate."
                    if self.route.currentData() == "joint"
                    else ""
                )
            )
        request = dict(
            operation=operation,
            route=self.route.currentData(),
            snapshot=self.baseline,
            image_id=self.images.currentData(),
            **extra,
        )
        self.shell._request_physical(encoded(request).encode(), self.context())

    def propose(self, value, commit=False):
        self.ensure_current()
        name = self.field().name
        candidate = changed_snapshot(self.route.currentData(), self.baseline, name, value)
        self.candidate = candidate
        self._commit_requested = commit
        self.render_fields(name)
        self.request("preview", parameter=name, candidate=candidate)

    def apply_typed(self):
        self.propose(float(self.value.text()), commit=True)

    def fine(self, direction):
        f = self.field()
        step = float(self.step.text())
        if not math.isfinite(step) or step <= 0:
            raise ValueError("Fine step must be positive and finite in the stored unit")
        self.propose(f.value + direction * step, commit=True)

    def begin_gesture(self):
        self.ensure_current()
        self._gesture = True
        self._commit_requested = False

    def drag_preview(self, value):
        self._drag_value = value
        self._drag_timer.start()

    def flush_drag(self):
        if self._gesture and self._drag_value is not None:
            self.guard(lambda: self.propose(self._drag_value))

    def end_gesture(self, value):
        self._drag_timer.stop()
        self._gesture = False
        self.guard(lambda: self.propose(value, commit=True))

    def cancel(self, *, clear=False):
        self._drag_timer.stop()
        self._gesture = False
        self._drag_value = None
        self.scene.view._physical_drag = False
        self.scene.view._press = self.scene.view._last = None
        self.candidate = None
        self._commit_requested = False
        self.epoch += 1
        self.shell._supersede_physical()
        self.result = None
        self.features.views = ()
        self.features.background = None
        self.features.update()
        self.reciprocal.set_preview(None)
        self.scene.view.set_physical_handle(None)
        self.identity.setText(
            "Preview canceled / obsolete; canonical starts and immutable results unchanged"
        )
        if not clear and self.baseline is not None:
            self.render_fields()
            self.guard(lambda: self.request("preview"))

    def _editable_state(self, route, data):
        if route == "configuration":
            owner_keys = (
                "acquisition_id",
                "source_sha256",
                "configuration_path",
                "configuration_sha256",
                "cif_path",
                "cif_sha256",
                "baseline_yaml",
            )
            keys = ("proposed",)
        elif route == "simulator":
            owner_keys = (
                "draft_id",
                "configuration_path",
                "imported_sha256",
                "cif_path",
                "cif_sha256",
            )
            keys = ("yaml_text",)
        elif route == "hbn":
            owner_keys = (
                "session_id",
                "acquisition_id",
                "inputs_json",
                "candidates",
                "exclusions",
                "lower",
                "upper",
            )
            keys = (
                "initial",
                "revision",
                "frozen_json",
                "center_proposal_json",
                "selected_result_id",
                "initial_provenance_json",
            )
        elif route == "sample":
            owner_keys = ("session_id", "inputs_json", "prepared_json", "frozen_json", "exclusions")
            keys = ("controls_json", "selected_result_id", "initial_provenance_json")
        else:
            owner_keys = ("session_id", "captures_json")
            keys = ("controls_json", "selected_result_id")
        return {
            "owner": payload_hash({k: data[k] for k in owner_keys}),
            "values": {k: data[k] for k in keys},
        }

    def install(self, route, data):
        shell = self.shell
        if route == "configuration":
            from parameter_state import configured_draft

            draft = numeric_from_document(data)
            configured_draft(draft)
            shell._validate_project_admission(shell.project, numeric_draft=draft)
            shell._numeric_draft = draft
            shell._validated_numeric = (shell.project.project_id, draft)
            shell._launch_snapshot = None
            shell._refresh_numeric_editor()
            shell._mark_dirty()
        elif route == "simulator":
            from simulation_io import canonical_configuration

            draft = simulation_draft_from_document(data)
            canonical_configuration(draft)
            if not shell.simulator._can_persist(draft, shell.simulator.result_reference):
                raise ValueError("Simulation edit exceeds project admission")
            shell.simulator.draft = draft
            shell.simulator.validated = None
            shell.simulator._populate(draft)
            shell.simulator._supersede()
            shell._mark_dirty()
            shell.simulator.refresh()
        elif route == "hbn":
            shell.hbn.store(hbn_session_from_document(data))
        elif route == "sample":
            shell._supersede_sample()
            shell.sample.store(sample_session_from_document(data))
        else:
            shell.joint.store(joint_session_from_document(data))

    def commit(self):
        self.ensure_current()
        if self.candidate is None:
            return
        if self.result is None or any(v.error for v in self.result.views):
            raise ValueError("The canonical preview is invalid; initial values were not committed")
        route = self.route.currentData()
        action = _action(
            "Physical initial-value edit",
            [
                FieldChange(
                    self.shell.project.project_id,
                    route,
                    self._editable_state(route, self.baseline),
                    self._editable_state(route, self.candidate),
                )
            ],
        )
        name = self.field().name
        self.install(route, self.candidate)
        self.history.push(action)
        self.baseline = snapshot(self.shell, route)
        self.candidate = None
        self._commit_requested = False
        self.render_fields(name)
        self.status.setText(
            "One canonical initial-value transaction committed; downstream state invalidated; no fit ran"
        )
        self.request("preview")

    def history_step(self, undo):
        source = self.history.undo_actions if undo else self.history.redo_actions
        if not source:
            raise ValueError("Nothing to undo/redo")
        action = source[-1]
        updates = []
        for change in action.changes:
            if change.acquisition_id != self.shell.project.project_id:
                raise ValueError("History belongs to another project")
            route = change.field
            if route.startswith("hbn:"):
                acq = UUID(route[4:])
                session = next(
                    (s for s in self.shell.hbn.sessions if s.acquisition_id == acq), None
                )
                if session is None:
                    raise ValueError("Physical history owner is no longer available")
                current = hbn_session_document(session)
                key = "hbn"
            else:
                key = route
                current = snapshot(self.shell, route)
            expected, restored = (
                (change.after, change.before) if undo else (change.before, change.after)
            )
            current_state = self._editable_state(key, current)
            if current_state["owner"] != expected["owner"]:
                raise ValueError(
                    "Physical history belongs to another or replaced owner/input scope"
                )
            if current_state["values"] != expected["values"]:
                raise ValueError("Physical undo conflicts with newer route edits")
            current.update(restored["values"])
            if key != "hbn":
                current["revision"] += 1
            updates.append((key, current))
        if all(key == "hbn" for key, _data in updates):
            replacements = {
                hbn_session_from_document(data).acquisition_id: hbn_session_from_document(data)
                for _key, data in updates
            }
            sessions = tuple(replacements.get(s.acquisition_id, s) for s in self.shell.hbn.sessions)
            self.shell._validate_project_admission(
                self.shell.project, view=replace(self.shell._capture_view(), hbn_sessions=sessions)
            )
            self.shell.hbn.sessions = sessions
            self.shell.hbn.epoch += 1
            self.shell.hbn.render()
            self.shell._mark_dirty()
        else:
            for key, data in updates:
                self.install(key, data)
        source.pop()
        (self.history.redo_actions if undo else self.history.undo_actions).append(action)
        self.load()
        self.status.setText(action.label + (" undone" if undo else " redone"))

    def fill_results(self, *_args):
        self.results.clear()
        source = self.source.currentData()
        session = self.shell.hbn.session if source == "hbn" else self.shell.sample.session
        for text in () if session is None else session.results_json:
            record = json.loads(text)
            self.results.addItem(record["name"], record["result_id"])
        self.proposal = None
        self.adopt_button.setEnabled(False)
        self.proposal_text.setPlainText(
            "Select a genuine result. Missing identity, inactive calibration and deficient qualification remain unavailable. hBN does not require a sample fit. Center, ellipse center and reference pixel are distinct."
        )

    def request_proposal(self):
        route = self.source.currentData()
        result_id = self.results.currentData()
        if self.route.currentData() != route:
            self.route.setCurrentIndex(self.route.findData(route))
        self.ensure_current()
        if result_id is None:
            raise ValueError("No retained genuine result available for this source")
        self._proposal_source = self.baseline
        self.request("proposal", result_id=result_id)

    def proposal_targets(self):
        if self.proposal is None:
            raise ValueError("Preview a matching genuine center first")
        source = self._proposal_source
        route = self.source.currentData()
        if route == "hbn":
            inp = json.loads(hbn_session_from_document(source).inputs_json)
        else:
            inp = json.loads(sample_session_from_document(source).inputs_json)
            inp["configuration_sha256"] = next(
                v["sha256"] for v in inp["files"] if v["path"] == inp["configuration_path"]
            )
            inp["cif_sha256"] = next(
                v["sha256"] for v in inp["files"] if v["path"] == inp["cif_path"]
            )
        selected = self.shell.hbn.session
        if selected is None:
            raise ValueError(
                "Load a matching hBN initial-value draft to adopt this intercept; detector reference coordinates are not beam centers"
            )
        candidates = self.shell.hbn.sessions if self.shared.isChecked() else (selected,)
        targets = []
        for session in candidates:
            target_inputs = json.loads(session.inputs_json)
            if (
                target_inputs["configuration_sha256"] != inp["configuration_sha256"]
                or target_inputs["configuration_path"] != inp["configuration_path"]
                or target_inputs["cif_sha256"] != inp["cif_sha256"]
            ):
                if not self.shared.isChecked():
                    raise ValueError(
                        "Target setup/beam/detector identity differs; retain a per-acquisition proposal"
                    )
                continue
            acquisition = next(
                (
                    a
                    for a in self.shell.project.acquisitions
                    if a.acquisition_id == session.acquisition_id
                ),
                None,
            )
            if (
                acquisition is None
                or acquisition.source_sha256 != target_inputs["source_sha256"]
                or acquisition.metadata.revision != target_inputs["metadata_revision"]
            ):
                raise ValueError(
                    "Target acquisition identity/revision changed; admit matching hBN inputs first"
                )
            if route == "hbn" and any(
                target_inputs[k] != inp[k]
                for k in (
                    "base_detector_rotation",
                    "beam_direction_lab",
                    "detector_column_pitch_m",
                    "detector_row_pitch_m",
                )
            ):
                raise ValueError("Shared target detector/beam settings disagree")
            targets.append(session)
        if not targets:
            raise ValueError(
                "No compatible hBN starts share the source setup/beam/detector identity"
            )
        return tuple(targets)

    def compare_proposal(self, *_args):
        if self.proposal is None:
            return
        self.adopt_button.setEnabled(False)
        self.shared.setEnabled(self.target.currentData() == "hbn")
        try:
            if self.target.currentData() == "sample":
                candidate = self.sample_center_candidate()
                affected = fields("sample", candidate)[0].scope
                lines = [
                    "Affected sample images: " + ", ".join(affected),
                    "Coupled canonical initial updates: "
                    + encoded(self.proposal["sample_initial_updates"]),
                ]
                self.adopt_button.setEnabled(True)
                self.proposal_text.setPlainText(
                    "\n".join(lines)
                    + "\n"
                    + json.dumps(self.proposal, indent=2)
                    + "\nNo observations added. This remains an explicitly unqualified initial estimate; full covariance/prediction bands unavailable."
                )
                return
            targets = self.proposal_targets()
            lines = [
                f"{s.acquisition_id}: current {s.initial[2:4]} → proposed {tuple(self.proposal['center_px'])}; private distance stays {s.initial[4]} m"
                for s in targets
            ]
            self.adopt_button.setEnabled(True)
        except ValueError as exc:
            lines = [str(exc)]
        self.proposal_text.setPlainText(
            "\n".join(lines)
            + "\n"
            + json.dumps(self.proposal, indent=2)
            + "\nDependence: hBN private distance/tilt and declared wavelength/lattice; sample incidence/alignment and active calibration. Marginal errors are not prediction bands. This estimate is never appended to observations."
        )

    def sample_center_candidate(self):
        if (
            self.source.currentData() != "sample"
            or self.route.currentData() != "sample"
            or self.proposal is None
        ):
            raise ValueError("Sample starts require a matching supported sample-derived proposal")
        self.ensure_current()
        if self.baseline != self._proposal_source:
            raise ValueError("Sample proposal baseline changed; preview again")
        session = sample_session_from_document(self.baseline)
        controls = json.loads(session.controls_json)
        editable = {f.name for f in fields("sample", self.baseline) if f.editable}
        for name, value in self.proposal["sample_initial_updates"].items():
            index = controls["names"].index(name)
            if name not in editable and value != controls["initial"][index]:
                raise ValueError("Coupled estimate conflicts with the current fixed scope: " + name)
            controls["initial"][index] = value
        from sample_io import _fit_arguments
        from sample_state import sample_session_document

        _fit_arguments(controls, len(json.loads(session.inputs_json)["images"]))
        return sample_session_document(
            replace(
                session,
                controls_json=encoded(controls),
                revision=session.revision + 1,
                initial_provenance_json=encoded(self.proposal),
                selected_result_id=None,
            )
        )

    def adopt(self):
        self.ensure_current()
        if snapshot(self.shell, self.source.currentData()) != self._proposal_source:
            raise ValueError("Center proposal source changed; preview again")
        proposal = self.proposal
        if self.target.currentData() == "sample":
            self.candidate = self.sample_center_candidate()
            self._commit_requested = True
            self.render_fields("detector_reference_column_offset_px")
            self.request(
                "preview", parameter="detector_reference_column_offset_px", candidate=self.candidate
            )
            return
        targets = self.proposal_targets()
        changes, replacements = [], {}
        for s in targets:
            updated = replace(
                s,
                initial=(*s.initial[:2], *proposal["center_px"], s.initial[4]),
                revision=s.revision + 1,
                initial_provenance_json=encoded(proposal),
                center_proposal_json="",
                frozen_json="",
                selected_result_id=None,
            )
            before, after = hbn_session_document(s), hbn_session_document(updated)
            changes.append(
                FieldChange(
                    self.shell.project.project_id,
                    "hbn:" + str(s.acquisition_id),
                    self._editable_state("hbn", before),
                    self._editable_state("hbn", after),
                )
            )
            replacements[s.acquisition_id] = updated
        action = _action("Adopt geometry-derived initial beam center", changes)
        sessions = tuple(replacements.get(s.acquisition_id, s) for s in self.shell.hbn.sessions)
        self.shell._validate_project_admission(
            self.shell.project, view=replace(self.shell._capture_view(), hbn_sessions=sessions)
        )
        self.shell.hbn.sessions = sessions
        self.shell.hbn.epoch += 1
        self.shell.hbn.render()
        self.shell.detector_panel.view.set_ring_curves()
        self.history.push(action)
        self.shell._mark_dirty()
        self.proposal = None
        self.load()
        self.status.setText(
            "Initial centers adopted in one undo action: "
            + ", ".join(str(s.acquisition_id) for s in targets)
            + "; source results and observations unchanged"
        )

    def sensitivity(self):
        self.ensure_current()
        f = self.field()
        step = float(self.step.text())
        if (
            not f.editable
            or f.name
            in (
                "source.spatial_sigma_x",
                "source.spatial_sigma_y",
                "source.divergence_x",
                "source.divergence_y",
            )
            or not math.isfinite(step)
            or step <= 0
        ):
            raise ValueError(
                "Choose an observed geometry coordinate and positive finite step; nominal maps do not observe source spread"
            )
        settings = dict(
            route=self.route.currentData(),
            parameter=f.name,
            unit=f.unit,
            step=step,
            baseline_sha256=payload_hash(self.baseline),
            affected_images=list(f.scope),
            observable="canonical geometry forward preview",
        )
        self.settings_json = encoded(settings)
        self.shell._mark_dirty()
        self.candidate = None
        self._commit_requested = False
        self.request("sensitivity", parameter=f.name, step=step)

    def matched_acquisition(self):
        route, data = self.route.currentData(), self.baseline
        if route == "simulator":
            return None
        if route == "configuration":
            identity = numeric_from_document(data).acquisition_id
            return next(
                (a for a in self.shell.project.acquisitions if a.acquisition_id == identity), None
            )
        if route == "hbn":
            identity = hbn_session_from_document(data).acquisition_id
            return next(
                (a for a in self.shell.project.acquisitions if a.acquisition_id == identity), None
            )
        if route == "sample":
            sessions = (sample_session_from_document(data),)
        else:
            session = joint_session_from_document(data)
            captures = json.loads(session.captures_json)
            hbn = hbn_session_from_document(captures["hbn"]["session"])
            group, selected_image_id = split_joint_image_identity(self.images.currentData())
            if group == "hbn":
                if selected_image_id != str(hbn.acquisition_id):
                    return None
                return next(
                    (
                        a
                        for a in self.shell.project.acquisitions
                        if a.acquisition_id == hbn.acquisition_id
                    ),
                    None,
                )
            if group not in captures:
                return None
            sessions = (sample_session_from_document(captures[group]["session"]),)
        for session in sessions:
            for row in json.loads(session.inputs_json)["images"]:
                selected = self.images.currentData() if route == "sample" else selected_image_id
                if row["image_id"] == selected:
                    return next(
                        (
                            a
                            for a in self.shell.project.acquisitions
                            if str(a.source_path.resolve()) == row["path"]
                            and a.source_sha256 == row["decoded_sha256"]
                        ),
                        None,
                    )
        return None

    def ready(self, result):
        self.ensure_current()
        if result.snapshot_sha256 != payload_hash(self.baseline):
            raise ValueError("Obsolete physical baseline rejected")
        if result.proposal is not None:
            self.proposal = result.proposal
            self.compare_proposal()
            self.status.setText(
                "Explicit geometry-derived estimate ready; no data/result/qualification changed"
            )
            return
        self.result = result
        views = result.views
        self.features.views = views
        self.features.identity = (result.snapshot_sha256, self.epoch)
        self.features.update()
        current = views[1] if result.step is not None and len(views) > 1 else views[-1]
        self.identity.setText(
            f"Transient preview: baseline {result.snapshot_sha256}; request {result.request_sha256}; route {self.route.currentData()}; {current.observable}; {current.error or 'no fit'}"
        )
        acquisition = self.matched_acquisition()
        acq = None if acquisition is None else acquisition.acquisition_id
        prepared = None if acq is None else self.shell._resident_planes.get(acq)
        if prepared is not None and (
            prepared.decoded_sha256 != acquisition.source_sha256
            or current.mapping is None
            or prepared.display.shape != current.mapping.instrument.detector_shape_rc
        ):
            prepared = None
        thumbnail = None if prepared is None else self.shell._thumbnails.get(acq)
        if thumbnail is None:
            self.features.background = None
        else:
            pixels, rows, columns = thumbnail
            image = QImage(pixels, columns, rows, columns, QImage.Format.Format_Grayscale8)
            self.features.background = QPixmap.fromImage(image.copy())
        if prepared is not None:
            self.scene.view.set_levels(prepared.low_value, prepared.high_value)
        self.scene.view.set_scene(
            acq,
            prepared,
            current.mapping,
            (result.snapshot_sha256, result.request_sha256, self.epoch),
        )
        if current.mapping is not None:
            self.scene.view.set_overlays(current.coordinates_px, current.mapping)
        if views[0].mapping is not None and current.mapping is not None:
            preview = ReciprocalPreview(
                result.request_sha256,
                self.shell.project.project_id,
                acq or self.shell.project.project_id,
                views[0].mapping,
                current.mapping if len(views) > 1 else None,
                self.epoch,
            )
            self.reciprocal.set_preview(preview)
        else:
            self.reciprocal.set_preview(None)
        self.bind_handle()
        if result.step is not None:
            lines = [
                f"Local sensitivity: {result.parameter}, ±{result.step} {self.field().unit}",
                "Held fixed (exact stored values): " + encoded(result.held_fixed),
            ]
            baseline = views[0]
            for sign, v in zip(("+", "-"), views[1:], strict=True):
                if v.error or baseline.error:
                    lines.append(sign + ": invalid: " + (v.error or baseline.error))
                elif v.observable.startswith("13x13"):
                    valid = v.mapping.valid & baseline.mapping.valid
                    if not np.array_equal(
                        v.mapping.valid, baseline.mapping.valid
                    ) or not np.array_equal(v.mapping.status, baseline.mapping.status):
                        lines.append(
                            sign
                            + ": invalid branch/domain: map validity/status changed; no derivative"
                        )
                    elif not valid.any():
                        lines.append(sign + ": no valid Q samples; derivative unavailable")
                    else:
                        delta = (
                            v.mapping.q_sample_Ainv[valid] - baseline.mapping.q_sample_Ainv[valid]
                        )
                        lines.append(
                            sign
                            + f": maximum ΔQ norm {np.linalg.norm(delta, axis=1).max():.9g} Å⁻¹; {len(delta)} fixed native grid points"
                        )
                else:
                    delta = v.coordinates_px - baseline.coordinates_px
                    lines.append(
                        sign
                        + f": maximum native feature motion {np.linalg.norm(delta, axis=1).max():.9g} px; {len(delta)} held-identity features"
                    )
            lines += [
                "Measurement covariance is attached to frozen observations only. Parameter propagation / prediction bands unavailable: no saved qualified full covariance and supported mapping. Hard bounds are not uncertainty.",
                self.identity.text(),
            ]
            self.sensitivity_text.setPlainText("\n".join(lines))
        if self._commit_requested:
            self.commit()
        else:
            self.status.setText(
                "Canonical preview ready"
                if not current.error
                else "Invalid canonical preview: " + current.error
            )

    def restore(self, settings_json="{}"):
        self.cancel(clear=True)
        self.settings_json = settings_json
        self.history = SessionHistory()
        self.baseline = self.candidate = self.result = self.proposal = None
        self.table.setRowCount(0)
        settings = json.loads(settings_json)
        if settings:
            self._rendering = True
            self.route.setCurrentIndex(self.route.findData(settings["route"]))
            self.step.setText(format(settings["step"], ".17g"))
            self._rendering = False
            self.sensitivity_text.setPlainText(
                "Saved request settings / provenance (transient results not persisted):\n"
                + json.dumps(settings, indent=2)
                + "\nPrediction bands remain unavailable without saved qualified full covariance."
            )

    def closeEvent(self, event):
        self.cancel(clear=True)
        self.scene.view.release_resources()
        super().closeEvent(event)
