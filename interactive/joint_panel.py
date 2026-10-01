"""Explicit three-pack joint geometry controls and immutable result inspection."""

import json
import math
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from hbn_state import hbn_session_document
from joint_state import JointSession, joint_controls, joint_session_document
from mask_state import mask_document
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
from sample_state import encoded, payload_hash, sample_session_document


class JointPanel(QDialog):
    def __init__(self, shell):
        super().__init__(shell)
        self.shell = shell
        self.session = None
        self.epoch = 0
        self._rendering = False
        self._draft_dirty = False
        self._presentation_record = None
        self.setWindowTitle("Joint geometry — hBN + Bi2Se3 + Bi2Te3")
        self.resize(1080, 850)
        layout = QVBoxLayout(self)
        self.status = QLabel(
            "Capture each reviewed frozen group explicitly. No new standalone fit is required."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        controls = self.page("1 Captures / controls")
        buttons = QHBoxLayout()
        controls.addLayout(buttons)
        for group, label in (
            ("hbn", "Capture current hBN"),
            ("bi2se3", "Capture current Bi2Se3"),
            ("bi2te3", "Capture current Bi2Te3"),
        ):
            self.button(buttons, label, lambda g=group: self.capture(g))
        self.captures = QTextBrowser()
        self.captures.setMinimumHeight(150)
        controls.addWidget(self.captures)
        hint = QLabel(
            "Shared: detector column/row tilt and beam center. Bi2Se3 sample-x, axis pitch and pivot pitch offset are fixed zero gauge references. PbI2 capture is unsupported here; its unobserved coordinates stay zero. hBN distance is private. Derived zB is conditional on fixed mechanical references."
        )
        hint.setWordWrap(True)
        controls.addWidget(hint)
        self.parameters = QTableWidget(0, 6)
        self.parameters.setHorizontalHeaderLabels(
            ["Canonical parameter", "Unit", "Initial", "Lower", "Upper", "Role / scope"]
        )
        self.parameters.setMinimumHeight(470)
        self.parameters.itemChanged.connect(self.changed)
        controls.addWidget(self.parameters)
        self.button(controls, "Reset controls from captured hBN seed", self.reset_controls)
        self.button(controls, "Commit displayed joint controls", self.commit)
        self.name = QLineEdit("Joint geometry candidate")
        form = QFormLayout()
        form.addRow("Result name", self.name)
        controls.addLayout(form)
        self.fit_button = self.button(
            controls,
            "Fit captured frozen packs",
            lambda: self.request("fit", name=self.name.text()),
        )
        results = self.page("2 Results / inspection")
        self.results = QComboBox()
        self.results.currentIndexChanged.connect(self.present)
        results.addWidget(self.results)
        row = QHBoxLayout()
        results.addLayout(row)
        self.button(row, "Select current-launch candidate", self.select)
        self.button(row, "Remove unselected result", self.remove)
        self.button(row, "Import exact result / historical report", self.import_result)
        self.button(
            row, "Export exact result", lambda: self.file_request("export_result", save=True)
        )
        self.summary = QTextBrowser()
        self.summary.setMinimumHeight(200)
        results.addWidget(self.summary)
        self.comparison = QTableWidget(0, 8)
        self.comparison.setHorizontalHeaderLabels(
            [
                "Name / scope",
                "Unit",
                "Initial",
                "Fitted",
                "Std error",
                "Active bound",
                "Confidence",
                "Weak direction",
            ]
        )
        self.comparison.setMinimumHeight(420)
        results.addWidget(self.comparison)
        self.images = QComboBox()
        self.images.currentIndexChanged.connect(self.present_points)
        results.addWidget(self.images)
        self.points = QTableWidget(0, 7)
        self.points.setHorizontalHeaderLabels(
            [
                "Group / image / observation",
                "Observed column",
                "Observed row",
                "Predicted column",
                "Predicted row",
                "Residual column",
                "Residual row",
            ]
        )
        self.points.setMinimumHeight(260)
        self.points.itemSelectionChanged.connect(self.inspect)
        results.addWidget(self.points)
        self.inspection = QTextBrowser()
        results.addWidget(self.inspection)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(7, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(360)
        results.addWidget(self.canvas)
        handoff = self.page("3 Qualified handoff")
        hint = QLabel(
            "Save or reload the owner's hash-bound GEOMETRY_ONLY handoff. All predecessors must still match. Historical report inspection works without the source files. Joint handoffs are separate from indexed results and are not adopted into an experiment here."
        )
        hint.setWordWrap(True)
        handoff.addWidget(hint)
        self.specimen = QComboBox()
        self.specimen.addItems(["bi2se3", "bi2te3"])
        self.manifest = QLineEdit()
        self.detector_base = QLineEdit()
        form = QFormLayout()
        form.addRow("Handoff specimen", self.specimen)
        form.addRow("Matching specimen manifest", self.manifest)
        form.addRow("Detector base configuration", self.detector_base)
        handoff.addLayout(form)
        self.button(
            handoff, "Use displayed result's captured predecessor paths", self.handoff_paths
        )
        self.button(
            handoff, "Save qualified handoff", lambda: self.file_request("save_handoff", save=True)
        )
        self.button(handoff, "Reload / verify handoff", lambda: self.file_request("reload_handoff"))
        handoff.addStretch()
        self.cancel_button = self.button(layout, "Cancel", shell._cancel_current)
        self.render()

    def page(self, label):
        page = QWidget()
        layout = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, label)
        return layout

    def button(self, layout, label, action):
        button = QPushButton(label)
        button.clicked.connect(lambda: self.guard(action))
        layout.addWidget(button)
        return button

    def guard(self, action):
        try:
            action()
        except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def receipt(self, paths):
        return [
            {
                "id": str(a.acquisition_id),
                "path": str(a.source_path),
                "source_sha256": a.source_sha256,
                "metadata_revision": a.metadata.revision,
                "mask": mask_document(a.mask),
            }
            for a in self.shell.project.acquisitions
            if str(a.source_path) in paths
        ]

    def context(self):
        return (self.shell.project.project_id, self.epoch, tuple(self.shell.project.acquisitions))

    def store(self, session, *, render=True, invalidate=True):
        self.shell._validate_project_admission(
            self.shell.project, view=replace(self.shell._capture_view(), joint_session=session)
        )
        self.shell._supersede_joint()
        self.session = session
        self.epoch += 1
        self.shell._mark_dirty()
        if render:
            self.render()
        else:
            self.refresh()

    def capture(self, group):
        if self.session is not None and self.session.draft_json is not None:
            raise ValueError("Commit or restore pending joint edits before changing captures")
        if group == "hbn":
            panel = self.shell.hbn
            session = panel.current_inputs()
            if panel.review_pending():
                raise ValueError("Commit and refreeze pending hBN review first")
            for j, values in enumerate((session.initial, session.lower, session.upper), 1):
                for i, v in enumerate(values):
                    if panel.parameters.item(i, j).text() != format(
                        v * 180 / math.pi if i < 2 else v, ".17g"
                    ):
                        raise ValueError("Commit pending hBN controls before capture")
            seed = panel.record()
            if not session.frozen_json or seed is None:
                raise ValueError(
                    "Freeze hBN and choose a genuine retained calibration candidate (a prior candidate may be imported); a new successful fit is not required"
                )
            inputs = json.loads(session.inputs_json)
            paths = [inputs["source_path"]]
            captured = replace(session, results_json=(), selected_result_id=None, exports=())
            document = hbn_session_document(captured)
            extra = {"calibration_record": seed}
        else:
            panel = self.shell.sample
            session = panel.session
            if (
                session is None
                or not session.frozen_json
                or panel._draft_dirty
                or not panel.inputs_match()
            ):
                raise ValueError("Load, review, commit and freeze the requested sample group first")
            if json.loads(session.inputs_json)["phase_id"].lower() != group:
                raise ValueError(
                    "Current sample material is not " + group + "; no material aliases are admitted"
                )
            inputs = json.loads(session.inputs_json)
            paths = [v["path"] for v in inputs["images"]]
            captured = replace(session, results_json=(), selected_result_id=None, exports=())
            document = sample_session_document(captured)
            extra = {}
        capture = {"session": document, "project_receipt": self.receipt(paths), **extra}
        capture["sha256"] = payload_hash(capture)
        current = self.session or JointSession(uuid4())
        captures = json.loads(current.captures_json)
        captures[group] = capture
        updated = replace(
            current,
            captures_json=encoded(captures),
            revision=current.revision + 1,
            selected_result_id=None,
        )
        if group == "hbn" and current.controls_json is None:
            updated = replace(updated, controls_json=encoded(self.seed_controls(capture)))
        self.store(updated)
        self.status.setText(
            group
            + " frozen pack captured explicitly; other captures and existing controls retained"
        )

    def seed_controls(self, capture):
        from rasim_next.fitting.hbn import HbnDetectorCalibration
        from rasim_next.fitting.joint_geometry import JointGeometryBounds, JointGeometryState

        seed = HbnDetectorCalibration(**capture["calibration_record"]["calibration"])
        state = JointGeometryState.from_hbn(seed)
        bounds = JointGeometryBounds.around_hbn(seed)
        return joint_controls(
            state.as_array().tolist(),
            bounds.lower.as_array().tolist(),
            bounds.upper.as_array().tolist(),
        )

    def reset_controls(self):
        self._draft_dirty = False
        if self.session is None or "hbn" not in json.loads(self.session.captures_json):
            raise ValueError("Capture hBN first")
        controls = self.seed_controls(json.loads(self.session.captures_json)["hbn"])
        self.store(
            replace(
                self.session,
                controls_json=encoded(controls),
                draft_json=None,
                selected_result_id=None,
                revision=self.session.revision + 1,
            )
        )

    def changed(self, *_):
        if self._rendering or self.session is None or self.parameters.rowCount() != 21:
            return
        self._draft_dirty = True
        self.shell._supersede_joint()
        rows = [[self.parameters.item(i, j).text() for j in (2, 3, 4)] for i in range(21)]
        try:
            self.store(
                replace(
                    self.session,
                    draft_json=encoded(rows),
                    selected_result_id=None,
                    revision=self.session.revision + 1,
                ),
                render=False,
            )
            self.status.setText("Visible joint edits saved as pending; commit before Fit")
        except (ValueError, TypeError) as exc:
            self.shell._supersede_joint()
            self.status.setText(str(exc))

    def commit(self):
        if self.session is None or self.session.controls_json is None:
            raise ValueError("Capture hBN to initialize controls first")
        controls = json.loads(self.session.controls_json)
        arrays = []
        for j, key in enumerate(("initial", "lower", "upper"), 2):
            values = []
            for i, stored in enumerate(controls[key]):
                text = self.parameters.item(i, j).text()
                values.append(stored if text == format(stored, ".17g") else float(text))
            arrays.append(values)
        validated = joint_controls(*arrays)
        self._draft_dirty = False
        self.store(
            replace(
                self.session,
                controls_json=encoded(validated),
                draft_json=None,
                selected_result_id=None,
                revision=self.session.revision + 1,
            )
        )
        self.status.setText(
            "Displayed initial values and bounds committed exactly in canonical units"
        )

    def readiness(self, *, history_capacity=True):
        if self.session is None or self.session.controls_json is None:
            return "Capture reviewed hBN and initialize the canonical controls"
        if self._draft_dirty or self.session.draft_json is not None:
            return "Pending visible joint edits must be committed"
        captures = json.loads(self.session.captures_json)
        missing = {"hbn", "bi2se3", "bi2te3"} - set(captures)
        if missing:
            return "Capture frozen " + ", ".join(sorted(missing))
        for capture in captures.values():
            expected = capture["project_receipt"]
            if self.receipt([v["path"] for v in expected]) != expected:
                return "Captured source, metadata or mask changed; recapture the affected group"
        if history_capacity and len(self.session.results_json) >= 4:
            return "History is full; remove an unselected record"
        return None

    def request(self, operation, **extra):
        session = self.session or JointSession(uuid4())
        if operation == "fit":
            reason = self.readiness()
            if reason:
                raise ValueError(reason)
            if not extra["name"].strip() or len(extra["name"]) > 256:
                raise ValueError("Enter a result name with 1-256 characters")
        if self.session is None:
            self.store(session)
        self.shell._request_joint(
            operation,
            encoded(
                {"operation": operation, "session": joint_session_document(session), **extra}
            ).encode(),
            self.context(),
        )

    def file_request(self, operation, save=False):
        record = self._presentation_record
        if operation in ("export_result", "save_handoff") and record is None:
            raise ValueError("Display a retained joint result first")
        chooser = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
        path, _ = chooser(self, operation.replace("_", " "), "", "JSON (*.json)")
        if not path:
            return
        extra = {"path": path}
        if record is not None and save:
            extra["record"] = encoded(record)
        if operation == "save_handoff":
            extra.update(
                specimen_id=self.specimen.currentText(),
                manifest_path=self.manifest.text(),
                detector_base_config_path=self.detector_base.text(),
            )
        self.request(operation, **extra)

    def import_result(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import exact joint result / historical report", "", "JSON (*.json)"
        )
        if path:
            self.request("import_result", path=path, name=self.name.text())

    def handoff_paths(self):
        record = self._presentation_record
        if record is None or record["launch"] is None:
            raise ValueError(
                "Historical report has no desktop captures; supply its actual matching predecessor paths"
            )
        captures = record["launch"]["captures"]
        self.manifest.setText(
            json.loads(captures[self.specimen.currentText()]["session"]["inputs_json"])[
                "manifest_path"
            ]
        )
        self.detector_base.setText(
            json.loads(captures["bi2se3"]["session"]["inputs_json"])["configuration_path"]
        )

    def select(self):
        record = self._presentation_record
        if (
            record is None
            or self.readiness(history_capacity=False)
            or record["launch_sha256"] != self.session.launch_sha256
        ):
            raise ValueError("Select only a result from the current committed frozen launch")
        self.store(replace(self.session, selected_result_id=record["result_id"]))
        self.status.setText(
            "Candidate selected explicitly; qualification remains the recorded owner verdict. Dependent experiment adoption is unavailable here."
        )

    def remove(self):
        record = self._presentation_record
        if record is None:
            return
        if self.session.selected_result_id == record["result_id"]:
            raise ValueError(
                "Replace captures or select another candidate before removing the selected result"
            )
        self.store(
            replace(
                self.session,
                results_json=tuple(
                    v
                    for v in self.session.results_json
                    if json.loads(v)["result_id"] != record["result_id"]
                ),
            )
        )

    def ready(self, value):
        self.store(value.session, invalidate=False)
        if value.presented_result_id:
            self.results.setCurrentIndex(self.results.findData(value.presented_result_id))
        self.status.setText(
            value.detail or (value.operation + " complete; prior records remain immutable")
        )

    def restore(self, session):
        self.session = session
        self.epoch += 1
        self.render()

    def invalidate_inputs(self):
        if self.session is None:
            return
        captures = json.loads(self.session.captures_json)
        if any(
            self.receipt([v["path"] for v in c["project_receipt"]]) != c["project_receipt"]
            for c in captures.values()
        ):
            self.shell._supersede_joint()
            if self.session.selected_result_id is not None:
                self.session = replace(
                    self.session, selected_result_id=None, revision=self.session.revision + 1
                )
                self.epoch += 1
            self.refresh()

    def refresh(self):
        reason = self.readiness()
        self.fit_button.setEnabled(
            reason is None
            and self.shell._pending_joint is None
            and self.shell._active_kind != "joint"
        )
        self.fit_button.setToolTip(
            reason
            or "Call fit_joint_geometry with the exact captured packs and displayed committed controls"
        )
        self.cancel_button.setEnabled(
            self.shell._active_kind == "joint" or self.shell._pending_joint is not None
        )

    def item(self, table, row, column, value, identity=None, editable=False):
        item = QTableWidgetItem(str(value))
        if not editable:
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setData(Qt.UserRole, identity)
        table.setItem(row, column, item)

    def render(self):
        self._rendering = True
        try:
            previous = self.results.currentData()
            self.results.clear()
            self.parameters.setRowCount(0)
            self.captures.clear()
            if self.session is not None:
                captures = json.loads(self.session.captures_json)
                lines = []
                for group, capture in captures.items():
                    doc = capture["session"]
                    inputs = json.loads(doc["inputs_json"])
                    roster = (
                        ", ".join(v["image_id"] for v in inputs.get("images", []))
                        or Path(inputs["source_path"]).name
                    )
                    lines.append(
                        f"{group}: {roster}\nFrozen capture {capture['sha256'][:16]}… · review revision {doc['revision']}"
                    )
                self.captures.setPlainText(
                    "\n".join(lines)
                    + "\nReadiness: "
                    + (self.readiness() or "ready; worker checks live file hashes before fitting")
                )
                if self.session.controls_json is not None:
                    c = json.loads(self.session.controls_json)
                    drafts = (
                        None
                        if self.session.draft_json is None
                        else json.loads(self.session.draft_json)
                    )
                    self.parameters.setRowCount(len(c["names"]))
                    for i, n in enumerate(c["names"]):
                        fixed = n in (
                            "goniometer_axis_pitch_rad",
                            "goniometer_pivot_pitch_offset_m",
                        ) or n.startswith("pbi2_")
                        scope = (
                            "fixed zero reference"
                            if fixed
                            else "private hBN nuisance"
                            if n.startswith("hbn_")
                            else "specimen local"
                            if n.startswith("bi2")
                            else "shared geometry"
                        )
                        unit = "rad" if n.endswith("_rad") else "px" if n.endswith("_px") else "m"
                        self.item(self.parameters, i, 0, n)
                        self.item(self.parameters, i, 1, unit)
                        for j, key in enumerate(("initial", "lower", "upper"), 2):
                            self.item(
                                self.parameters,
                                i,
                                j,
                                drafts[i][j - 2] if drafts else format(c[key][i], ".17g"),
                                editable=not fixed,
                            )
                        self.item(self.parameters, i, 5, scope)
                for text in self.session.results_json:
                    record = json.loads(text)
                    label = (
                        "selected"
                        if record["result_id"] == self.session.selected_result_id
                        else "candidate"
                        if record["launch_sha256"] == self.session.launch_sha256
                        else "historical"
                    )
                    verdict = (
                        "qualified" if record["report"]["confidence_qualified"] else "unqualified"
                    )
                    self.results.addItem(
                        record["name"] + " / " + label + " / " + verdict, record["result_id"]
                    )
                index = self.results.findData(previous or self.session.selected_result_id)
                if index >= 0:
                    self.results.setCurrentIndex(index)
            self.parameters.resizeColumnsToContents()
            self.present()
        finally:
            self._rendering = False
            self.refresh()

    def present(self, *_):
        self._presentation_record = None
        self.summary.clear()
        self.comparison.setRowCount(0)
        self.inspection.clear()
        self.images.blockSignals(True)
        self.images.clear()
        identity = self.results.currentData()
        if self.session is not None:
            self._presentation_record = next(
                (
                    json.loads(v)
                    for v in self.session.results_json
                    if json.loads(v)["result_id"] == identity
                ),
                None,
            )
        record = self._presentation_record
        if record is not None:
            report = record["report"]
            ident = report["identifiability"]
            hbn = report["hbn_automatic_trace"]
            metrics = report["crystalline_metrics"]
            qualification = (
                "Qualified by the existing joint geometry owner"
                if report["confidence_qualified"]
                else "Unqualified candidate: " + "; ".join(report["qualification_failures"])
            )
            condition = ident["scaled_jacobian_condition"]
            lines = [
                record["name"] + " · " + qualification,
                "Optimizer: " + report["message"],
                f"Jacobian rank {ident['jacobian_rank']}/{ident['parameter_count']} · scaled condition {condition if condition is not None else 'unavailable'}",
                f"hBN RMS {hbn['joint_rms_px']:.6g} px · max {hbn['joint_max_px']:.6g} px",
                f"Crystalline pooled RMS {metrics['pooled_site_rms_px']:.6g} px · max {metrics['pooled_site_max_px']:.6g} px",
                "Per-image diagnostics (native pixels):",
            ]
            lines.extend(
                f"{v['specimen_id']} / {v['image_id']}: {v['site_count']} sites · RMS {v['site_rms_px']:.6g} · max {v['site_max_px']:.6g}"
                for v in metrics["per_image"]
            )
            z = report["global"]["z_b_m"]
            lines.append(
                f"Derived zB {z['value']:.6g} m · standard error {z['standard_error']} m; conditional on fixed references, not an independent mechanical measurement."
            )
            if record["launch"] is None:
                lines.append(
                    "Historical report: original desktop frozen coordinates and starting values are unavailable. Source files are required only for a new fit or handoff verification."
                )
            self.summary.setPlainText("\n".join(lines))
            initial = (
                {}
                if record["launch"] is None
                else dict(
                    zip(
                        record["launch"]["controls"]["names"],
                        record["launch"]["controls"]["initial"],
                        strict=True,
                    )
                )
            )
            from rasim_next.fitting.joint_geometry import (
                GLOBAL_PARAMETER_NAMES,
                LOCAL_PARAMETER_NAMES,
                NUISANCE_PARAMETER_NAMES,
            )

            for section, names in (
                ("global", GLOBAL_PARAMETER_NAMES),
                ("local", LOCAL_PARAMETER_NAMES),
                ("nuisance", NUISANCE_PARAMETER_NAMES),
            ):
                for n in (*names, *(("z_b_m",) if section == "global" else ())):
                    param = report[section][n]
                    row = self.comparison.rowCount()
                    self.comparison.insertRow(row)
                    unit = "rad" if n.endswith("_rad") else "px" if n.endswith("_px") else "m"
                    values = [
                        n + " / " + section + " / " + param["role"],
                        unit,
                        initial.get(n, "unavailable"),
                        param["value"],
                        param["standard_error"],
                        param["active_bound"],
                        param["confidence_qualified"],
                        report["identifiability"]["weakest_direction"].get(n, "derived"),
                    ]
                    for j, v in enumerate(values):
                        self.item(
                            self.comparison,
                            row,
                            j,
                            "unavailable" if v is None else v,
                            (record["result_id"], n),
                        )
            for metric in report["crystalline_metrics"]["per_image"]:
                self.images.addItem(
                    metric["specimen_id"] + " / " + metric["image_id"],
                    (metric["specimen_id"], metric["image_id"]),
                )
            self.images.addItem("hBN frozen observations", ("hbn", "hbn"))
        self.images.blockSignals(False)
        self.comparison.resizeColumnsToContents()
        self.present_points()

    def present_points(self, *_):
        record = self._presentation_record
        self.points.setRowCount(0)
        self.inspection.clear()
        self.figure.clear()
        axes = self.figure.subplots()
        axes.set(xlabel="native column_px", ylabel="native row_px")
        axes.invert_yaxis()
        axes.set_aspect("equal")
        if record is not None:
            scope = self.images.currentData()
            rows = [p for p in record["points"] if (p["specimen_id"], p["image_id"]) == scope]
            if scope == ("hbn", "hbn") and record["launch"] is not None:
                from hbn_state import hbn_session_from_document

                hs = hbn_session_from_document(record["launch"]["captures"]["hbn"]["session"])
                pack = json.loads(hs.frozen_json)
                for xy in pack["coordinates_px"]:
                    axes.scatter(*xy, s=10, color="tab:blue")
                import numpy as np

                for curve in record.get("hbn_curves_px", []):
                    xy = np.asarray(curve, dtype=np.float64)
                    axes.plot(xy[:, 0], xy[:, 1], color="tab:orange", linewidth=0.8)
                self.inspection.setPlainText(
                    encoded(
                        {
                            "result_id": record["result_id"],
                            "frozen_pack": pack,
                            "radial_residual_px": record["hbn_residual_px"],
                        }
                    )
                )
            self.points.setRowCount(len(rows))
            for i, p in enumerate(rows):
                identity = (
                    record["result_id"],
                    p["specimen_id"],
                    p["image_id"],
                    p["observation_id"],
                )
                for j, v in enumerate(
                    [
                        p["specimen_id"] + " / " + p["image_id"] + " / " + p["observation_id"],
                        *p["observed_px"],
                        *p["predicted_px"],
                        *p["residual_px"],
                    ]
                ):
                    self.item(self.points, i, j, v, identity)
                axes.scatter(*p["observed_px"], s=16, color="tab:blue")
                axes.scatter(*p["predicted_px"], s=24, marker="+", color="tab:orange")
            axes.set_title(
                record["name"]
                + " / "
                + str(scope)
                + (
                    " / coordinates unavailable in historical report"
                    if not rows
                    else " / observed blue, predicted orange"
                )
            )
        self.points.resizeColumnsToContents()
        self.canvas.draw_idle()

    def inspect(self):
        row = self.points.currentRow()
        record = self._presentation_record
        if row < 0 or record is None or self.points.item(row, 0) is None:
            return
        bound = self.points.item(row, 0).data(Qt.UserRole)
        point = next(
            (
                p
                for p in record["points"]
                if (record["result_id"], p["specimen_id"], p["image_id"], p["observation_id"])
                == bound
            ),
            None,
        )
        self.inspection.setPlainText(
            encoded({"result_id": record["result_id"], "point": point})
            if point is not None
            else "Stale point binding; display the result again"
        )
