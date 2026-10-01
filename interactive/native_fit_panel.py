"""Prepared native data inspection and plan-derived editing; Run is unavailable."""

import json
from dataclasses import replace

from native_fit_profiles import NativeProfilesView
from native_fit_state import native_fit_session_document, reuse_plan
from parameter_state import MAX_HISTORY_BYTES, FieldChange, HistoryAction, SessionHistory
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
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from sample_state import encoded


class NativeFitPanel(QDialog):
    def __init__(self, shell):
        super().__init__(shell)
        self.shell, self.session, self.profiles = shell, None, None
        self.epoch, self._rendering = 0, False
        self.profile_description_sha256 = None
        self.history = SessionHistory()
        self.setWindowTitle("Prepared native inputs / draft plan")
        self.resize(1150, 800)
        layout = QVBoxLayout(self)
        self.status = QLabel(
            "Load existing physics, observation and plan JSON. No observation preparation or fitting."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        page = QWidget()
        form = QFormLayout(page)
        self.paths = {}
        for key, label in (
            ("physics_path", "Physics JSON"),
            ("observation_path", "Observations JSON"),
            ("plan_path", "Plan JSON"),
        ):
            row = QHBoxLayout()
            field = QLineEdit()
            self.paths[key] = field
            row.addWidget(field)
            self.button(row, "Browse", lambda k=key: self.browse(k))
            form.addRow(label, row)
        self.button(form, "Load prepared set", self.load)
        self.button(form, "Reload original sources / definitions", lambda: self.request("reload"))
        self.button(
            form,
            "Inspect displayed frozen output",
            lambda: self.request("inspect", description_sha256=self.displayed()["sha256"]),
        )
        self.summary = QTextBrowser()
        form.addRow(self.summary)
        self.tabs.addTab(page, "Inputs / provenance")
        page = QWidget()
        body = QVBoxLayout(page)
        self.profile_view = NativeProfilesView()
        body.addWidget(self.profile_view)
        self.row = QSpinBox()
        self.row.setPrefix("Frozen row ")
        self.row.valueChanged.connect(self.inspect_row)
        body.addWidget(self.row)
        self.row_detail = QTextBrowser()
        body.addWidget(self.row_detail)
        self.covariance = QTableWidget(0, 2)
        self.covariance.setHorizontalHeaderLabels(
            ["Other frozen row ID", "Full covariance, counts²"]
        )
        body.addWidget(self.covariance)
        self.observation_metadata = QTextBrowser()
        body.addWidget(self.observation_metadata)
        self.tabs.addTab(page, "Measured profiles / covariance")
        page = QWidget()
        body = QVBoxLayout(page)
        self.descriptions = QComboBox()
        self.descriptions.currentIndexChanged.connect(self.present)
        body.addWidget(self.descriptions)
        self.start = QSpinBox()
        self.start.setPrefix("Start index ")
        self.start.valueChanged.connect(self.present)
        body.addWidget(self.start)
        self.parameters = QTableWidget(0, 11)
        self.parameters.setHorizontalHeaderLabels(
            [
                "Canonical name",
                "Owner / scope",
                "Canonical = displayed unit",
                "Initial",
                "Lower",
                "Lower kind",
                "Upper",
                "Upper kind",
                "Sensitivity scale",
                "Fixed / active stages",
                "Support / constraint",
            ]
        )
        self.parameters.itemChanged.connect(self.changed)
        body.addWidget(self.parameters)
        self.stages = QTableWidget(0, 6)
        self.stages.setHorizontalHeaderLabels(
            [
                "Stage name / order",
                "Active canonical names",
                "Method",
                "Iterations",
                "Function budget",
                "Historical guards",
            ]
        )
        self.stages.itemChanged.connect(self.changed)
        body.addWidget(self.stages)
        self.step = QLineEdit()
        self.step.editingFinished.connect(self.changed)
        body.addWidget(self.step)
        row = QHBoxLayout()
        body.addLayout(row)
        self.button(row, "Commit displayed draft", self.commit)
        self.button(row, "Discard pending edits", self.discard)
        self.button(row, "Undo", lambda: self.undo(True))
        self.button(row, "Redo", lambda: self.undo(False))
        self.button(row, "Reuse compatible historical starts", self.reuse)
        self.button(row, "Remove displayed historical description", self.remove_history)
        self.button(row, "Export committed plan", self.export)
        self.button(row, "Export exact measured data", self.export_profiles)
        run = QPushButton("Run unavailable - pending R4 engine integration")
        run.setEnabled(False)
        run.setToolTip(
            "Draft parsing is not full engine launch admission; indexed adoption is unavailable"
        )
        body.addWidget(run)
        self.tabs.addTab(page, "Parameters / staged declarations")
        self.button(layout, "Cancel file operation", shell._cancel_current)
        self.render()

    def button(self, layout, text, action):
        button = QPushButton(text)
        button.clicked.connect(lambda: self.guard(action))
        layout.addWidget(button)
        return button

    def guard(self, action):
        try:
            action()
        except (ValueError, TypeError, RuntimeError, OSError, KeyError) as exc:
            self.status.setText(str(exc))

    def context(self):
        value = self.displayed()
        return self.shell.project.project_id, self.epoch, None if value is None else value["sha256"]

    def browse(self, key):
        path, _ = QFileDialog.getOpenFileName(self, key.replace("_", " "), "", "JSON (*.json)")
        if path:
            self.paths[key].setText(path)

    def request(self, operation, **extra):
        if operation != "load" and self.session is None:
            raise ValueError("Load a prepared set first")
        self.shell._request_prepared(
            operation,
            encoded(
                {
                    "operation": operation,
                    "session": native_fit_session_document(self.session),
                    "storage_json": self.shell.archive_storage_json,
                    **extra,
                }
            ).encode(),
            self.context(),
        )

    def load(self):
        if self.session is not None and self.session.draft_json is not None:
            raise ValueError("Commit or discard pending edits before replacing inputs")
        self.request("load", **{k: v.text() for k, v in self.paths.items()})

    def store(self, session, *, record_history=False, render=True, supersede=True):
        self.shell._validate_project_admission(
            self.shell.project, view=replace(self.shell._capture_view(), native_fit_session=session)
        )
        action = None
        if record_history:
            before = (
                None
                if self.session is None
                else replace(self.session, draft_json=None)
                if session is not None and session.current_json != self.session.current_json
                else self.session
            )
            old_strings = set() if before is None else {before.current_json, *before.history_json}
            new_strings = (
                set() if session is None else {session.current_json, *session.history_json}
            )
            retained_changes = old_strings ^ new_strings
            if before is not None:
                retained_changes.add(before.current_json)
            if session is not None:
                retained_changes.add(session.current_json)
            size = sum(len(v.encode()) for v in retained_changes) + 4096
            if size > MAX_HISTORY_BYTES:
                raise ValueError("Prepared edit exceeds the existing undo memory limit")
            action = HistoryAction(
                "Prepared input/plan edit",
                (FieldChange(self.shell.project.project_id, "prepared_snapshot", before, session),),
                size,
            )
        if supersede:
            self.shell._supersede_prepared()
        self.session = session
        self.epoch += 1
        self.history.push(action)
        self.shell._mark_dirty()
        if render:
            self.render()

    def ready(self, value):
        if value.inspection_json is None:
            self.store(value.session, record_history=value.session != self.session, supersede=False)
        elif value.session != self.session:
            raise ValueError("Inspection session changed; stale completion rejected")
        if value.profiles is not None:
            self.profiles = value.profiles
            self.profile_description_sha256 = self.displayed()["sha256"]
        self.observation_metadata.setPlainText(
            value.inspection_json
            or "Original observation JSON remains in the exact source inventory; inspect displayed output to review all declared metadata."
        )
        self.profile_view.set_profiles(self.profiles)
        self.row.setRange(0, 0 if self.profiles is None else len(self.profiles.row_ids) - 1)
        self.inspect_row()
        self.status.setText(value.detail)

    def restore(self, session):
        self.session, self.profiles = session, None
        self.profile_description_sha256 = None
        self.observation_metadata.clear()
        self.epoch += 1
        self.history = SessionHistory()
        self.profile_view.set_profiles(None)
        self.render()
        self.inspect_row()
        self.status.setText(
            "Saved immutable descriptions restored. Original sources/definitions are unverified; reload to inspect arrays. Run/adoption unavailable."
        )

    def render(self):
        self._rendering = True
        try:
            self.descriptions.clear()
            self.descriptions.addItem("Current committed draft", 0)
            if self.session:
                for i, text in enumerate(self.session.history_json, 1):
                    value = json.loads(text)
                    self.descriptions.addItem(
                        f"Historical {i} / {value['sha256'][:12]} / {value['inputs']['sample_id']}",
                        i,
                    )
                value = json.loads(self.session.current_json)
                for row in value["inputs"]["files"]:
                    key = {
                        "physics": "physics_path",
                        "observations": "observation_path",
                        "plan": "plan_path",
                    }.get(row["kind"])
                    if key:
                        self.paths[key].setText(row["path"])
            self.present()
        finally:
            self._rendering = False

    def displayed(self):
        if self.session is None:
            return None
        index = self.descriptions.currentData() or 0
        return json.loads(
            self.session.current_json if index == 0 else self.session.history_json[index - 1]
        )

    def item(self, table, row, column, value, editable=False, identity=None):
        item = QTableWidgetItem(str(value))
        item.setData(Qt.UserRole, identity)
        if not editable:
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        table.setItem(row, column, item)

    def present(self, *_):
        rendering = self._rendering
        self._rendering = True
        try:
            self.parameters.setRowCount(0)
            self.stages.setRowCount(0)
            value = self.displayed()
            if value is None or value["sha256"] != self.profile_description_sha256:
                self.profiles = None
                self.profile_description_sha256 = None
                self.profile_view.set_profiles(None)
                self.observation_metadata.clear()
                self.inspect_row()
            if value is None:
                self.summary.clear()
                return
            self.summary.setPlainText(
                encoded(
                    {
                        "description_sha256": value["sha256"],
                        "definition": value["definition"],
                        "inputs": value["inputs"],
                        "other_plan_settings": {
                            k: v
                            for k, v in value["plan"].items()
                            if k not in ("parameters", "starts", "stages")
                        },
                        "readiness": "Immutable parsed draft; full launch and adoption unavailable. Missing sources retain history. Marginal sigma is not full covariance. Loaded settings without controls are inert declarations, not engine admission.",
                    }
                )
            )
            plan = value["plan"]
            if self.session.draft_json is not None and self.descriptions.currentData() == 0:
                pending_index = json.loads(self.session.draft_json)["start_index"]
                if self.start.value() != pending_index:
                    self.start.blockSignals(True)
                    self.start.setValue(pending_index)
                    self.start.blockSignals(False)
                    self.status.setText(
                        "Commit or discard pending edits before changing the start index"
                    )
            self.start.setRange(0, max(0, len(plan["starts"]) - 1))
            current = self.descriptions.currentData() == 0
            edits = (
                {}
                if not current or self.session.draft_json is None
                else json.loads(self.session.draft_json)
            )
            self.parameters.setRowCount(len(plan["parameters"]))
            for i, (p, capability) in enumerate(
                zip(plan["parameters"], value["definition"]["capabilities"], strict=True)
            ):
                identity = (p["owner"], p["name"])
                key = encoded(identity)
                fixed = p["name"] in plan.get("fixed_parameters", {})
                stages = [
                    s["name"] for s in plan.get("stages", ()) if p["name"] in s["active_parameters"]
                ]
                reason = (
                    capability["reason"]
                    or "Loaded search declaration; full coupled domain/gauge admission remains R4"
                )
                values = [
                    p["name"],
                    p["owner"],
                    p["unit"],
                    plan["starts"][self.start.value()][i],
                    p["lower"],
                    p.get("lower_kind", "search"),
                    p["upper"],
                    p.get("upper_kind", "search"),
                    p["sensitivity_scale"],
                    "fixed" if fixed else ", ".join(stages) or "free (no staged declaration)",
                    reason,
                ]
                for col, text in enumerate(values):
                    editable = (
                        current
                        and not capability["reason"]
                        and (
                            col == 3
                            or (col == 4 and p.get("lower_kind", "search") == "search")
                            or (col == 6 and p.get("upper_kind", "search") == "search")
                        )
                    )
                    self.item(
                        self.parameters,
                        i,
                        col,
                        edits.get("parameters", {}).get(key, {}).get(str(col), text)
                        if editable
                        else text,
                        editable,
                        identity,
                    )
            stages = plan.get("stages", ())
            self.stages.setRowCount(len(stages))
            for i, s in enumerate(stages):
                for col, k in enumerate(
                    (
                        "name",
                        "active_parameters",
                        "method",
                        "maximum_iterations",
                        "maximum_function_evaluations",
                        "enforce_historical_guards",
                    )
                ):
                    editable = (
                        current
                        and "settings_reasons" in value["definition"]
                        and not value["definition"]["settings_reasons"]
                        and col in (2, 3, 4)
                        and k in s
                    )
                    self.item(
                        self.stages,
                        i,
                        col,
                        edits.get("stages", {})
                        .get(s["name"], {})
                        .get(k, s.get(k, "owner launch default; unavailable here")),
                        editable,
                        s["name"],
                    )
            self.step.setReadOnly(not current or "finite_difference_step" not in plan)
            self.step.setText(
                edits.get("step", str(plan.get("finite_difference_step", "Not declared")))
            )
            self.parameters.resizeColumnsToContents()
        finally:
            self._rendering = rendering

    def changed(self, *_):
        if self._rendering or self.session is None or self.descriptions.currentData() != 0:
            return
        edits = {
            "parameters": {},
            "stages": {},
            "step": self.step.text(),
            "start_index": self.start.value(),
        }
        for i in range(self.parameters.rowCount()):
            key = encoded(self.parameters.item(i, 0).data(Qt.UserRole))
            edits["parameters"][key] = {
                str(c): self.parameters.item(i, c).text()
                for c in (3, 4, 6)
                if self.parameters.item(i, c).flags() & Qt.ItemIsEditable
            }
        for i in range(self.stages.rowCount()):
            name = self.stages.item(i, 0).data(Qt.UserRole)
            edits["stages"][name] = {
                k: self.stages.item(i, c).text()
                for c, k in (
                    (2, "method"),
                    (3, "maximum_iterations"),
                    (4, "maximum_function_evaluations"),
                )
                if self.stages.item(i, c).flags() & Qt.ItemIsEditable
            }
        self.guard(
            lambda: self.store(replace(self.session, draft_json=encoded(edits)), render=False)
        )

    def plan_from_edits(self):
        plan = json.loads(self.session.current_json)["plan"]
        if self.session.draft_json is None:
            return plan
        edits = json.loads(self.session.draft_json)
        index = edits["start_index"]
        for i, p in enumerate(plan["parameters"]):
            row = edits["parameters"].get(encoded((p["owner"], p["name"])), {})
            if "3" in row:
                value = float(row["3"])
                if p["name"] in plan.get("fixed_parameters", {}):
                    plan["fixed_parameters"][p["name"]] = value
                    for start in plan["starts"]:
                        start[i] = value
                else:
                    plan["starts"][index][i] = value
            for col, name in (("4", "lower"), ("6", "upper")):
                if col in row:
                    p[name] = float(row[col])
        for s in plan.get("stages", ()):
            for k, text in edits["stages"].get(s["name"], {}).items():
                s[k] = text if k == "method" else int(text)
        if "finite_difference_step" in plan:
            plan["finite_difference_step"] = float(edits["step"])
        return plan

    def commit(self):
        if self.session is None or self.descriptions.currentData() != 0:
            raise ValueError("Review the current draft before committing")
        self.request("commit", plan_json=encoded(self.plan_from_edits()))

    def discard(self):
        if self.session:
            self.store(replace(self.session, draft_json=None))

    def reuse(self):
        old = self.displayed()
        if old is None or self.descriptions.currentData() == 0:
            raise ValueError("Display a historical description to review compatible starts")
        plan = reuse_plan(old, json.loads(self.session.current_json))
        self.request("commit", plan_json=encoded(plan))

    def remove_history(self):
        index = self.descriptions.currentData()
        if self.session is None or not index:
            raise ValueError("Display an unselected historical description first")
        self.store(
            replace(
                self.session,
                history_json=tuple(
                    v for i, v in enumerate(self.session.history_json, 1) if i != index
                ),
            ),
            record_history=True,
        )

    def undo(self, undo):
        stack = self.history.undo_actions if undo else self.history.redo_actions
        if not stack:
            raise ValueError("Nothing to undo" if undo else "Nothing to redo")
        change = stack[-1].changes[0]
        expected, restored = (
            (change.after, change.before) if undo else (change.before, change.after)
        )
        if self.session != expected:
            raise ValueError(
                "Undo conflicts with pending/current state; discard pending edits first"
            )
        self.store(restored)
        (self.history.redo_actions if undo else self.history.undo_actions).append(stack.pop())
        self.profiles = None
        self.profile_view.set_profiles(None)
        self.inspect_row()
        self.status.setText(
            "Description restored exactly; reload its original inputs to inspect profiles"
        )

    def export(self):
        if self.session is None or self.session.draft_json is not None:
            raise ValueError("Commit or discard pending edits before exporting")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export committed draft plan", "", "JSON (*.json)"
        )
        if path:
            self.request("export", path=path)

    def export_profiles(self):
        if self.session is None:
            raise ValueError("Load a prepared set first")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export exact measured counts and full covariance", "", "Numeric NPZ (*.npz)"
        )
        if path:
            self.request("export_profiles", path=path)

    def inspect_row(self, *_):
        self.covariance.setRowCount(0)
        if self.profiles is None:
            self.row_detail.setPlainText(
                "Frozen arrays unavailable; reload original files. Saved descriptions remain inspectable."
            )
            return
        p = self.profiles
        i = self.row.value()
        self.row_detail.setPlainText(
            encoded(
                {
                    "row_id": p.row_ids[i],
                    "observation_sha256": p.input_sha256,
                    "raw_count": float(p.raw_count[i]),
                    "background_count": float(p.background_count[i]),
                    "signed_corrected_count": float(p.corrected_count[i]),
                    "valid": bool(p.valid[i]),
                    "marginal_standard_error_count": float(p.standard_error_count[i]),
                    "covariance_policy": "Full covariance retained; marginal sigma is not a diagonal substitute",
                }
            )
        )
        self.covariance.setRowCount(len(p.row_ids))
        for j, row_id in enumerate(p.row_ids):
            self.item(self.covariance, j, 0, row_id)
            self.item(self.covariance, j, 1, float(p.covariance_count2[i, j]))
