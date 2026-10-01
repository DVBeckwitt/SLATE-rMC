"""Explicit hBN review and fitting controls over the shared application worker."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from hbn_state import encoded, freeze_hbn_observations, hbn_session_document
from mask_state import mask_document
from parameter_state import FieldChange, SessionHistory, _action
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

RINGS = ("002", "100", "101", "102", "004")
EDIT_FIELDS = (
    "revision",
    "initial",
    "lower",
    "upper",
    "f_scale",
    "max_nfev",
    "exclusions",
    "frozen_json",
    "center_proposal_json",
    "initial_provenance_json",
    "selected_result_id",
)


class HbnPanel(QDialog):
    def __init__(self, shell):
        super().__init__(shell)
        self.shell = shell
        self.sessions = ()
        self.history = SessionHistory()
        self.epoch = 0
        self._displayed_acquisition = None
        self._rendering = False
        self._proposal_epoch = None
        self._spot_arrays = ()
        self._canvas = None
        self._showing_result_points = False
        self.setWindowTitle("hBN calibration / Choose beam center")
        self.resize(850, 740)
        layout = QVBoxLayout(self)
        self.identity = QLabel("Select an acquisition first.")
        self.identity.setWordWrap(True)
        layout.addWidget(self.identity)
        self.status = QLabel("No hBN draft. Admit the selected image, dark and base configuration.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_inputs()
        self._build_spot()
        self._build_review()
        self._build_results()
        controls = QHBoxLayout()
        self.undo_button = QPushButton("Undo draft edit")
        self.redo_button = QPushButton("Redo draft edit")
        self.cancel_button = QPushButton("Cancel")
        for button in (self.undo_button, self.redo_button, self.cancel_button):
            controls.addWidget(button)
        layout.addLayout(controls)
        self.undo_button.clicked.connect(lambda: self.guard(lambda: self.undo_redo(True)))
        self.redo_button.clicked.connect(lambda: self.guard(lambda: self.undo_redo(False)))
        self.cancel_button.clicked.connect(shell._cancel_current)
        shell.detector_panel.view.center_picked.connect(self.manual_pick)
        shell.detector_panel.view.roi_selected.connect(self.roi_picked)

    def _button(self, layout, text, action):
        button = QPushButton(text)
        button.clicked.connect(lambda: self.guard(action))
        layout.addWidget(button)
        return button

    def _page(self, name, *, scroll=False):
        page = QWidget()
        layout = QVBoxLayout(page)
        if scroll:
            container = QScrollArea()
            container.setWidgetResizable(True)
            container.setWidget(page)
            self.tabs.addTab(container, name)
        else:
            self.tabs.addTab(page, name)
        return layout

    def _build_inputs(self):
        layout = self._page("1 Inputs / initial values")
        self.powder = QCheckBox(
            "I identify the selected image as hBN powder for ring preparation/fitting"
        )
        layout.addWidget(self.powder)
        self.powder.toggled.connect(self.changed)
        form = QFormLayout()
        layout.addLayout(form)
        self.dark = QLineEdit()
        self.configuration = QLineEdit()
        for label, control, pattern in (
            ("Raw dark OSC", self.dark, "OSC (*.osc *.osc.gz)"),
            ("Base detector configuration", self.configuration, "YAML (*.yaml *.yml)"),
        ):
            row = QHBoxLayout()
            row.addWidget(control)
            choose = QPushButton("Choose…")
            choose.clicked.connect(
                lambda _checked=False, c=control, f=pattern: self.choose_input(c, f)
            )
            row.addWidget(choose)
            form.addRow(label, row)
            control.textChanged.connect(self.changed)
        self.load_button = self._button(
            layout, "Admit selected image / dark / configuration", self.load
        )
        self.parameters = QTableWidget(5, 4)
        self.parameters.setHorizontalHeaderLabels(
            ["Actual fitted coordinate", "Initial", "Lower", "Upper"]
        )
        names = (
            "Column tilt (degree)",
            "Row tilt (degree)",
            "Center column (px)",
            "Center row (px)",
            "Private calibrant distance (m)",
        )
        for i, name in enumerate(names):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.parameters.setItem(i, 0, item)
            for j in range(1, 4):
                self.parameters.setItem(i, j, QTableWidgetItem(""))
        self.parameters.cellChanged.connect(self.changed)
        self.parameters.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.parameters)
        controls = QFormLayout()
        self.f_scale = QLineEdit("1")
        self.max_nfev = QLineEdit("1000")
        controls.addRow("soft_l1 f_scale (px)", self.f_scale)
        controls.addRow("Maximum function evaluations (1-1000)", self.max_nfev)
        self.f_scale.textChanged.connect(self.changed)
        self.max_nfev.textChanged.connect(self.changed)
        layout.addLayout(controls)
        self._button(layout, "Apply seeds and bounds", self.apply_parameters)
        self.fixed = QTextBrowser()
        layout.addWidget(self.fixed)

    def _build_spot(self):
        layout = self._page("2 Center proposal", scroll=True)
        explanation = QLabel(
            "Observed spot center is an approximate initial estimate. It does not determine the geometric beam intercept. Inspection crosshairs never change geometry."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        form = QFormLayout()
        layout.addLayout(form)
        self.manual_column = QLineEdit()
        self.manual_row = QLineEdit()
        form.addRow("Proposed native column_px", self.manual_column)
        form.addRow("Proposed native row_px", self.manual_row)
        for control in (self.manual_column, self.manual_row):
            control.textChanged.connect(self.changed)
        row = QHBoxLayout()
        self._button(row, "Click a proposed center on detector", self.pick_center)
        self._button(row, "Use manual center as initial estimate", self.adopt_manual)
        layout.addLayout(row)
        self.roi_fields = tuple(QLineEdit("0") for _ in range(4))
        for label, control in zip(
            (
                "ROI column start",
                "ROI column stop (exclusive)",
                "ROI row start",
                "ROI row stop (exclusive)",
            ),
            self.roi_fields,
            strict=True,
        ):
            form.addRow(label, control)
            control.textChanged.connect(self.changed)
        self.saturation = QLineEdit()
        self.saturation.setPlaceholderText("Blank: no declared saturation threshold")
        form.addRow("Exclude raw counts ≥ threshold", self.saturation)
        self.saturation.textChanged.connect(self.changed)
        row = QHBoxLayout()
        self._button(row, "Drag ROI on detector", self.pick_roi)
        self.spot_button = self._button(
            row, "Fit ROI Gaussian proposal", lambda: self.request("spot")
        )
        self.adopt_button = self._button(row, "Adopt reviewed Gaussian center", self.adopt_spot)
        layout.addLayout(row)
        self.spot_summary = QTextBrowser()
        self.spot_summary.setMinimumHeight(120)
        self.spot_summary.setMaximumHeight(190)
        layout.addWidget(self.spot_summary)
        self.spot_plot_layout = QVBoxLayout()
        layout.addLayout(self.spot_plot_layout, 1)

    def _build_review(self):
        layout = self._page("3 Prepare / review / freeze")
        self.prepare_button = self._button(
            layout, "Prepare ring observations for review", lambda: self.request("prepare")
        )
        self.review_summary = QLabel("No discovered observations.")
        self.review_summary.setWordWrap(True)
        layout.addWidget(self.review_summary)
        self.review = QTableWidget(0, 7)
        self.review.setHorizontalHeaderLabels(
            ["Use", "Column_px", "Row_px", "Ring hkl", "Sector", "Exclusion reason", "Residual px"]
        )
        self.review.cellChanged.connect(self.changed)
        self.review.cellClicked.connect(self.inspect_point)
        self.review.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.review, 1)
        self._button(layout, "Show draft review", self.render)
        self._button(layout, "Commit review decisions", self.commit_review)
        self.freeze_button = self._button(layout, "Freeze reviewed observations", self.freeze)
        self._button(
            layout, "Export frozen observations…", lambda: self.export("export_observations")
        )

    def _build_results(self):
        layout = self._page("4 Fit / results")
        self.name = QLineEdit("hBN calibration")
        layout.addWidget(QLabel("Name for the next immutable result"))
        layout.addWidget(self.name)
        self.name.textChanged.connect(self.changed)
        self.fit_button = self._button(
            layout, "Fit the frozen observation pack", lambda: self.request("fit")
        )
        self.results = QComboBox()
        self.results.currentIndexChanged.connect(self.render_result)
        layout.addWidget(self.results)
        row = QHBoxLayout()
        self._button(row, "Inspect result / show ring curves", lambda: self.request("present"))
        self._button(row, "Select this result", self.select_result)
        self._button(row, "Remove this result", self.remove_result)
        layout.addLayout(row)
        self.result_summary = QTextBrowser()
        layout.addWidget(self.result_summary, 1)
        row = QHBoxLayout()
        self._button(row, "Export result…", lambda: self.export("export"))
        self._button(row, "Export figure + exact values…", lambda: self.export("export_figure"))
        self._button(row, "Import bound result…", self.import_result)
        layout.addLayout(row)

    @property
    def session(self):
        return next(
            (s for s in self.sessions if s.acquisition_id == self.shell.selected_acquisition_id),
            None,
        )

    def acquisition(self):
        return next(
            (
                a
                for a in self.shell.project.acquisitions
                if a.acquisition_id == self.shell.selected_acquisition_id
            ),
            None,
        )

    def context(self):
        a = self.acquisition()
        return (
            self.shell.project.project_id,
            self.epoch,
            self.shell.selected_acquisition_id,
            None
            if a is None
            else (
                a.source_sha256,
                str(a.source_path),
                a.metadata.revision,
                encoded(mask_document(a.mask)),
            ),
            self.shell._visible_acquisition_id,
        )

    def changed(self, *_args):
        if self._rendering:
            return
        self.epoch += 1
        self._proposal_epoch = None
        self.shell._supersede_hbn()
        self.adopt_button.setEnabled(False)

    def guard(self, action):
        try:
            action()
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            self.status.setText(f"Unavailable: {exc}")

    def input_identity_current(self):
        session = self.session
        a = self.acquisition()
        if session is None or a is None:
            return False
        inputs = json.loads(session.inputs_json)
        return (
            inputs["source_sha256"] == a.source_sha256
            and inputs["source_path"] == str(a.source_path)
            and inputs["metadata_revision"] == a.metadata.revision
            and inputs["mask"] == mask_document(a.mask)
            and Path(self.dark.text()).resolve() == Path(inputs["dark_path"])
            and Path(self.configuration.text()).resolve() == Path(inputs["configuration_path"])
        )

    def current_inputs(self):
        session = self.session
        a = self.acquisition()
        if session is None or a is None:
            raise ValueError("admit image/dark/configuration first")
        inputs = json.loads(session.inputs_json)
        if (
            inputs["source_sha256"] != a.source_sha256
            or inputs["source_path"] != str(a.source_path)
            or inputs["metadata_revision"] != a.metadata.revision
            or inputs["mask"] != mask_document(a.mask)
            or Path(self.dark.text()).resolve() != Path(inputs["dark_path"])
            or Path(self.configuration.text()).resolve() != Path(inputs["configuration_path"])
        ):
            raise ValueError(
                "input or mask changed; admit a fresh draft (historical results are retained)"
            )
        if self.shell._visible_acquisition_id != a.acquisition_id:
            raise ValueError("wait for the selected native detector image")
        return session

    def store(self, session, label=None):
        old = self.session
        sessions = (
            *(s for s in self.sessions if s.acquisition_id != session.acquisition_id),
            session,
        )
        if len(sessions) > 8:
            raise ValueError("at most eight retained hBN acquisition drafts")
        self.shell._validate_project_admission(
            self.shell.project, view=replace(self.shell._capture_view(), hbn_sessions=sessions)
        )
        action = None
        if label and old is not None and old.session_id == session.session_id:
            before = {k: getattr(old, k) for k in EDIT_FIELDS}
            after = {k: getattr(session, k) for k in EDIT_FIELDS}
            if before != after:
                action = _action(
                    label, [FieldChange(session.acquisition_id, "hbn_edit", before, after)]
                )
        if old is None or old.launch_sha256 != session.launch_sha256:
            self.shell.detector_panel.view.set_ring_curves()
        self.sessions = sessions
        self.history.push(action)
        self.epoch += 1
        self._proposal_epoch = None
        self.shell._mark_dirty()
        self.render()

    def load(self):
        a = self.acquisition()
        if a is None or self.shell._visible_acquisition_id != a.acquisition_id:
            raise ValueError("select and open the native image first")
        self.shell._request_hbn(
            "load",
            encoded(
                {
                    "operation": "load",
                    "source_path": str(a.source_path),
                    "source_sha256": a.source_sha256,
                    "acquisition_id": str(a.acquisition_id),
                    "metadata_revision": a.metadata.revision,
                    "mask": mask_document(a.mask),
                    "dark_path": self.dark.text(),
                    "configuration_path": self.configuration.text(),
                }
            ).encode(),
            self.context(),
        )

    def choose_input(self, control, pattern):
        path, _ = QFileDialog.getOpenFileName(self, "Select input", "", pattern)
        if path:
            control.setText(path)

    def apply_parameters(self):
        session = self.current_inputs()
        columns = []
        for j in range(1, 4):
            prior = (session.initial, session.lower, session.upper)[j - 1]
            values = []
            for i, stored in enumerate(prior):
                text = self.parameters.item(i, j).text()
                display = format(stored * 180 / math.pi if i < 2 else stored, ".17g")
                value = stored if text == display else float(text) * (math.pi / 180 if i < 2 else 1)
                values.append(value)
            columns.append(tuple(values))
        if (
            tuple(columns) == (session.initial, session.lower, session.upper)
            and float(self.f_scale.text()) == session.f_scale
            and int(self.max_nfev.text()) == session.max_nfev
        ):
            self.status.setText("Seeds/bounds are unchanged")
            return
        updated = replace(
            session,
            initial=columns[0],
            lower=columns[1],
            upper=columns[2],
            f_scale=float(self.f_scale.text()),
            max_nfev=int(self.max_nfev.text()),
            revision=session.revision + 1,
            frozen_json="",
            center_proposal_json="",
            selected_result_id=None,
        )
        self.store(updated, "Edit hBN seeds/bounds")
        self.status.setText("Seeds/bounds applied; review and freeze a new pack before fitting.")

    def request(self, operation, **extra):
        session = self.current_inputs()
        for j, values in enumerate((session.initial, session.lower, session.upper), 1):
            for i, v in enumerate(values):
                if self.parameters.item(i, j).text() != format(
                    v * 180 / math.pi if i < 2 else v, ".17g"
                ):
                    raise ValueError("apply or undo edited seeds/bounds before launching work")
        if self.f_scale.text() != str(session.f_scale) or self.max_nfev.text() != str(
            session.max_nfev
        ):
            raise ValueError("apply edited solver controls before launching work")
        if operation in ("prepare", "fit") and not self.powder.isChecked():
            raise ValueError("identify the selected image as hBN powder first")
        if operation == "fit" and not session.frozen_json:
            raise ValueError("freeze reviewed observations before Fit")
        if operation == "fit" and len(session.results_json) >= 4:
            raise ValueError("four retained results; remove one before another Fit")
        if operation == "spot":
            extra.update(
                roi=[int(c.text()) for c in self.roi_fields],
                saturation_count=float(self.saturation.text())
                if self.saturation.text().strip()
                else None,
            )
        if operation in ("present", "export", "export_figure"):
            record = self.record()
            if record is None:
                raise ValueError("choose a retained result")
            extra["record"] = encoded(record)
        if operation == "fit":
            extra["name"] = self.name.text().strip()
        argument = encoded(
            {"operation": operation, "session": hbn_session_document(session), **extra}
        ).encode()
        self.shell._request_hbn(operation, argument, self.context())

    def pick_center(self):
        self.current_inputs()
        view = self.shell.detector_panel.view
        view.set_mask_mode("inspect")
        view.roi_select_enabled = False
        view.center_pick_enabled = True
        self.status.setText(
            "Click a native detector point. Then use the manual center as an initial estimate."
        )

    def manual_pick(self, column, row):
        if not self.isVisible():
            return
        self.manual_column.setText(format(column, ".17g"))
        self.manual_row.setText(format(row, ".17g"))
        self.status.setText(
            "Manual center proposal received; geometry is unchanged until explicit adoption."
        )

    def pick_roi(self):
        self.current_inputs()
        view = self.shell.detector_panel.view
        view.center_pick_enabled = False
        view.set_mask_mode("inspect")
        view.roi_select_enabled = True
        self.status.setText(
            "Drag a native ROI on the detector; at most 65,536 pixels for the Gaussian proposal."
        )

    def roi_picked(self, roi):
        if not self.isVisible():
            return
        for control, value in zip(self.roi_fields, roi, strict=True):
            control.setText(str(value))

    def adopt_manual(self):
        session = self.current_inputs()
        center = (float(self.manual_column.text()), float(self.manual_row.text()))
        proposal = encoded(
            {
                "method": "manual native click/numeric",
                "center_px": center,
                "inputs_sha256": session.inputs_sha256,
                "draft_revision": session.revision,
                "interpretation": "approximate initial estimate; not a geometric beam intercept",
            }
        )
        self.store(
            replace(
                session,
                initial=(*session.initial[:2], *center, session.initial[4]),
                revision=session.revision + 1,
                center_proposal_json=proposal,
                initial_provenance_json=proposal,
                frozen_json="",
                selected_result_id=None,
            ),
            "Adopt manual initial center",
        )
        self.status.setText(
            "Manual center adopted as initial estimate; private distance remains local."
        )

    def adopt_spot(self):
        session = self.current_inputs()
        proposal = json.loads(session.center_proposal_json)
        if (
            self._proposal_epoch != self.epoch
            or proposal["inputs_sha256"] != session.inputs_sha256
            or proposal["draft_revision"] != session.revision
        ):
            raise ValueError("Gaussian proposal is stale; fit the current ROI again")
        if not proposal["reliable_initial_estimate"]:
            raise ValueError("Gaussian center is unreliable: " + "; ".join(proposal["limitations"]))
        center = proposal["values"][2:4]
        self.store(
            replace(
                session,
                initial=(*session.initial[:2], *center, session.initial[4]),
                revision=session.revision + 1,
                initial_provenance_json=session.center_proposal_json,
                frozen_json="",
                selected_result_id=None,
            ),
            "Adopt Gaussian initial center",
        )
        self.status.setText("Reviewed Gaussian center adopted as an initial estimate.")

    def commit_review(self):
        session = self.current_inputs()
        if self._showing_result_points:
            raise ValueError(
                "result points are immutable; show the draft review before editing exclusions"
            )
        exclusions = []
        for i in range(self.review.rowCount()):
            if self.review.item(i, 0).checkState() != Qt.CheckState.Checked:
                reason = self.review.item(i, 5).text().strip()
                if not reason:
                    raise ValueError(f"excluded candidate {i} needs a reason")
                exclusions.append((i, reason))
        if tuple(exclusions) != session.exclusions:
            self.store(
                replace(
                    session,
                    exclusions=tuple(exclusions),
                    revision=session.revision + 1,
                    frozen_json="",
                    selected_result_id=None,
                ),
                "Review hBN exclusions",
            )

    def freeze(self):
        self.commit_review()
        session = self.current_inputs()
        pack = freeze_hbn_observations(session)
        self.store(replace(session, frozen_json=pack), "Freeze reviewed hBN observations")
        self.status.setText(
            "Immutable observations frozen. Fit consumes these exact coordinates and identities."
        )

    def record(self):
        session = self.session
        result_id = self.results.currentData()
        return (
            None
            if session is None
            else next(
                (
                    json.loads(t)
                    for t in session.results_json
                    if json.loads(t)["result_id"] == result_id
                ),
                None,
            )
        )

    def select_result(self):
        session = self.current_inputs()
        record = self.record()
        if record is None:
            raise ValueError("choose a retained result")
        if record["launch"]["launch_sha256"] != session.launch_sha256:
            raise ValueError("historical/stale result cannot be selected for the current draft")
        self.store(replace(session, selected_result_id=record["result_id"]), "Select hBN result")
        self.status.setText(
            "Result selected; qualification is shown separately. Calibrant-private distance stays local."
        )

    def remove_result(self):
        session = self.session
        record = self.record()
        if record is None:
            raise ValueError("choose a retained result")
        result_id = record["result_id"]
        self.store(
            replace(
                session,
                results_json=tuple(
                    t for t in session.results_json if json.loads(t)["result_id"] != result_id
                ),
                selected_result_id=None
                if session.selected_result_id == result_id
                else session.selected_result_id,
            )
        )
        self.history = SessionHistory()
        self.render()
        self.shell.detector_panel.view.set_ring_curves()

    def export(self, operation):
        pattern = "PNG (*.png)" if operation == "export_figure" else "JSON (*.json)"
        path, _ = QFileDialog.getSaveFileName(self, "New external hBN export", "", pattern)
        if path:
            self.request(operation, path=path)

    def import_result(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import exact bound hBN result", "", "JSON (*.json)"
        )
        if path:
            self.request("import", path=path)

    def inspect_point(self, row, _column):
        session = self.session
        record = self.record()
        points = (
            record["launch"]["pack"]["coordinates_px"]
            if record is not None and self._showing_result_points
            else ([] if session is None else session.candidates)
        )
        if (
            0 <= row < len(points)
            and self.shell._visible_acquisition_id == self.shell.selected_acquisition_id
        ):
            c, r = points[row][:2]
            view = self.shell.detector_panel.view
            view.crosshair = (round(c), round(r))
            view._request_paint()
            view.crosshair_changed.emit()

    def undo_redo(self, undo):
        source = self.history.undo_actions if undo else self.history.redo_actions
        if not source:
            raise ValueError("nothing to undo/redo")
        action = source[-1]
        if self.session is None or action.changes[0].acquisition_id != self.session.acquisition_id:
            raise ValueError("select the acquisition owning this draft edit")
        updated = replace(
            self.session, **(action.changes[0].before if undo else action.changes[0].after)
        )
        self.store(updated)
        source.pop()
        (self.history.redo_actions if undo else self.history.undo_actions).append(action)
        self.status.setText(action.label + (" undone" if undo else " redone"))
        self.render()

    def restore(self, sessions):
        self.sessions = sessions
        self.history = SessionHistory()
        self.epoch += 1
        self._proposal_epoch = None
        self._displayed_acquisition = None
        self._spot_arrays = ()
        if self._canvas is not None:
            self._canvas.setVisible(False)
        self.refresh()

    def restore_proposal_controls(self, session):
        inputs = json.loads(session.inputs_json)
        rows, columns = inputs["shape_rc"]
        c, r = session.initial[2:4]
        proposal = json.loads(session.center_proposal_json) if session.center_proposal_json else {}
        roi = proposal.get(
            "roi",
            (
                max(0, int(c) - 32),
                min(columns, int(c) + 32),
                max(0, int(r) - 32),
                min(rows, int(r) + 32),
            ),
        )
        for control, number in zip(self.roi_fields, roi, strict=True):
            control.setText(str(number))
        saturation = proposal.get("saturation_raw_count")
        self.saturation.setText("" if saturation is None else str(saturation))
        center = proposal.get("center_px", session.initial[2:4])
        self.manual_column.setText(format(center[0], ".17g"))
        self.manual_row.setText(format(center[1], ".17g"))

    def refresh(self):
        a = self.acquisition()
        self.identity.setText(
            "Select an acquisition first."
            if a is None
            else a.name + " — native column,row; array row,column"
        )
        if self._displayed_acquisition != self.shell.selected_acquisition_id:
            self._displayed_acquisition = self.shell.selected_acquisition_id
            self.epoch += 1
            self._proposal_epoch = None
            self._spot_arrays = ()
            if self._canvas is not None:
                self._canvas.setVisible(False)
            self.shell.detector_panel.view.set_ring_curves()
            self.shell.detector_panel.view.center_pick_enabled = False
            self._rendering = True
            self.powder.setChecked(False)
            session = self.session
            if session is not None:
                inputs = json.loads(session.inputs_json)
                self.dark.setText(inputs["dark_path"])
                self.configuration.setText(inputs["configuration_path"])
                self.restore_proposal_controls(session)
            self._rendering = False
            self.render()
        self.cancel_button.setEnabled(
            self.shell._active_kind == "hbn" or self.shell._pending_hbn is not None
        )
        self.undo_button.setEnabled(bool(self.history.undo_actions))
        self.redo_button.setEnabled(bool(self.history.redo_actions))

    def render(self):
        self._rendering = True
        try:
            session = self.session
            for button in (
                self.spot_button,
                self.prepare_button,
                self.freeze_button,
                self.fit_button,
            ):
                button.setEnabled(session is not None)
            self.adopt_button.setEnabled(False)
            self._showing_result_points = False
            self.review.setRowCount(0)
            self.results.clear()
            if session is None:
                self.fixed.setPlainText("No admitted fixed inputs.")
                self.result_summary.setPlainText("No fit result.")
                self.spot_summary.setPlainText("No proposal.")
                return
            for j, values in enumerate((session.initial, session.lower, session.upper), 1):
                for i, v in enumerate(values):
                    self.parameters.item(i, j).setText(
                        format(v * 180 / math.pi if i < 2 else v, ".17g")
                    )
            self.f_scale.setText(str(session.f_scale))
            self.max_nfev.setText(str(session.max_nfev))
            self.fixed.setPlainText(json.dumps(json.loads(session.inputs_json), indent=2))
            self.review.setRowCount(len(session.candidates))
            exclusions = dict(session.exclusions)
            for i, (c, r, ring, sector) in enumerate(session.candidates):
                use = QTableWidgetItem()
                use.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsSelectable
                )
                use.setCheckState(
                    Qt.CheckState.Unchecked if i in exclusions else Qt.CheckState.Checked
                )
                self.review.setItem(i, 0, use)
                for j, text in enumerate(
                    (
                        format(c, ".17g"),
                        format(r, ".17g"),
                        RINGS[ring],
                        str(sector),
                        exclusions.get(i, ""),
                        "",
                    ),
                    1,
                ):
                    item = QTableWidgetItem(text)
                    if j != 5:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.review.setItem(i, j, item)
            counts = [
                sum(p[2] == i for n, p in enumerate(session.candidates) if n not in exclusions)
                for i in range(5)
            ]
            self.review_summary.setText(
                f"{len(session.candidates)} discovered; included counts {dict(zip(RINGS, counts, strict=True))}; each sector 10°. "
                + (
                    "Frozen pack: " + json.loads(session.frozen_json)["pack_id"]
                    if session.frozen_json
                    else "Draft: no frozen pack."
                )
            )
            for text in session.results_json:
                record = json.loads(text)
                kind = (
                    "selected" if record["result_id"] == session.selected_result_id else "candidate"
                )
                if (
                    record["launch"]["launch_sha256"] != session.launch_sha256
                    or not self.input_identity_current()
                ):
                    kind = "historical / stale"
                self.results.addItem(
                    record["name"] + " — " + kind + " — " + record["qualification"],
                    record["result_id"],
                )
            if session.selected_result_id:
                self.results.setCurrentIndex(self.results.findData(session.selected_result_id))
            self.spot_summary.setPlainText(
                session.center_proposal_json
                or "No proposal. Reopened proposals require a fresh explicit ROI fit before adoption."
            )
        finally:
            self._rendering = False
        self.render_result()
        self.refresh()

    def render_result(self, *_args):
        if self._rendering:
            return
        record = self.record()
        if record is None:
            self.result_summary.setPlainText(
                "No fit. Preparation/preliminary refinement does not qualify a result."
            )
            return
        session = self.session
        calibration = record["calibration"]
        lines = [
            record["name"],
            record["qualification"],
            "Solver termination: "
            + str(calibration["solver_success"])
            + "; "
            + calibration["solver_message"],
            f"Evaluations: {calibration['function_evaluations']}; rank {calibration['jacobian_rank']}/5; scaled condition {calibration['scaled_jacobian_condition']}",
            f"RMS {calibration['residual_rms_px']} px; maximum {calibration['residual_max_px']} px; active bounds {calibration['active_bounds']}",
        ]
        for i, name in enumerate(
            (
                "Column tilt degree",
                "Row tilt degree",
                "Center column px",
                "Center row px",
                "Private distance m",
            )
        ):
            factor = 180 / math.pi if i < 2 else 1
            stderr = calibration["standard_error"][i]
            available = (
                calibration["jacobian_rank"] == 5
                and calibration["scaled_jacobian_condition"] is not None
                and calibration["scaled_jacobian_condition"] < 1e8
                and stderr is not None
            )
            lines.append(
                f"{name}: launch {record['launch']['initial'][i] * factor:.12g}; current {session.initial[i] * factor:.12g}; fitted {record['fitted_values'][i] * factor:.12g}; standard error {stderr * factor if available else 'unavailable (rank/conditioning)'}"
            )
        for i, ring in enumerate(RINGS):
            lines.append(
                f"{ring}: {calibration['ring_point_count'][i]} points; sector coverage {calibration['ring_angular_coverage_fraction'][i]}; RMS {calibration['ring_rms_px'][i]} px"
            )
        lines.extend(
            (
                "Pack: " + record["launch"]["pack"]["pack_id"],
                "Exact inputs / seeds / bounds / residuals:",
                json.dumps(record["launch"], indent=2),
            )
        )
        if (
            record["launch"]["launch_sha256"] != session.launch_sha256
            or not self.input_identity_current()
        ):
            lines.insert(0, "HISTORICAL / STALE for the current draft/input identity")
        self.result_summary.setPlainText("\n".join(lines))

    def show_result_points(self, record):
        pack = record["launch"]["pack"]
        self._rendering = True
        try:
            self.review.setRowCount(len(pack["ring_index"]))
            for i, (point, ring, sector, residual) in enumerate(
                zip(
                    pack["coordinates_px"],
                    pack["ring_index"],
                    pack["angular_sector"],
                    record["residual_px"],
                    strict=True,
                )
            ):
                for j, text in enumerate(
                    (
                        "frozen",
                        str(point[0]),
                        str(point[1]),
                        RINGS[ring],
                        str(sector),
                        "",
                        str(residual),
                    )
                ):
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.review.setItem(i, j, item)
        finally:
            self._rendering = False
        self._showing_result_points = True
        self.review_summary.setText(
            "Inspecting immutable result observations and native residuals. Click a row to inspect its detector profiles; show draft review before editing exclusions."
        )

    def show_spot(self, value):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        if self._canvas is None:
            self._canvas = FigureCanvasQTAgg(Figure(figsize=(7, 4), layout="constrained"))
            self._canvas.setMinimumHeight(400)
            self.spot_plot_layout.addWidget(self._canvas)
        self._canvas.setVisible(True)
        self._spot_arrays = (value.spot_data, value.spot_model, value.spot_residual)
        figure = self._canvas.figure
        figure.clear()
        axes = figure.subplots(2, 3)
        roi = json.loads(value.session.center_proposal_json)["roi"]
        for axis, data, label in zip(
            axes[0],
            self._spot_arrays,
            ("Original valid raw-dark counts", "Gaussian model counts", "Residual counts"),
            strict=True,
        ):
            axis.imshow(
                np.ma.masked_where(~np.isfinite(value.spot_residual), data),
                origin="upper",
                extent=(roi[0] - 0.5, roi[1] - 0.5, roi[3] - 0.5, roi[2] - 0.5),
                aspect="auto",
            )
            axis.set(title=label, xlabel="native column_px", ylabel="native row_px")
        valid = np.isfinite(value.spot_residual)
        for axis, along, coord, label in (
            (axes[1, 0], 0, np.arange(roi[0], roi[1]), "column_px"),
            (axes[1, 1], 1, np.arange(roi[2], roi[3]), "row_px"),
        ):
            axis.plot(coord, np.sum(np.where(valid, value.spot_data, 0), axis=along), label="data")
            axis.plot(
                coord, np.sum(np.where(valid, value.spot_model, 0), axis=along), label="model"
            )
            axis.set(xlabel="native " + label, ylabel="valid-support count sum")
            axis.legend()
        axes[1, 2].plot(value.spot_residual[valid], ".", markersize=1)
        axes[1, 2].set(xlabel="valid ROI pixel index", ylabel="residual count")
        self._canvas.draw_idle()

    def ready(self, value):
        old = self.session
        session = value.session
        if value.operation == "load":
            if old is not None:
                session = replace(session, results_json=old.results_json, exports=old.exports)
            self.history = SessionHistory()
        self.store(session)
        if value.operation == "load":
            self._rendering = True
            self.restore_proposal_controls(session)
            self._rendering = False
        if value.operation == "spot":
            self.show_spot(value)
            self._proposal_epoch = self.epoch
            proposal = json.loads(session.center_proposal_json)
            self.adopt_button.setEnabled(proposal["reliable_initial_estimate"])
        if value.presented_result_id is not None:
            self.results.setCurrentIndex(self.results.findData(value.presented_result_id))
            record = self.record()
            if self.shell._visible_acquisition_id == session.acquisition_id:
                self.shell.detector_panel.view.set_overlays(
                    np.asarray(record["launch"]["pack"]["coordinates_px"])
                )
                self.shell.detector_panel.view.set_ring_curves(value.curves)
            self.show_result_points(record)
            self.render_result()
        elif session.candidates and self.shell._visible_acquisition_id == session.acquisition_id:
            self.shell.detector_panel.view.set_overlays(
                np.asarray([p[:2] for p in session.candidates])
            )
            self.shell.detector_panel.view.set_ring_curves()
        self.status.setText(
            value.operation.replace("_", " ").title()
            + " completed. "
            + (
                "Result is a candidate; qualification is shown separately."
                if value.operation == "fit"
                else ""
            )
        )

    def closeEvent(self, event):
        self.shell.detector_panel.view.center_pick_enabled = False
        self.shell.detector_panel.view.roi_select_enabled = False
        self.shell._supersede_hbn()
        event.accept()
