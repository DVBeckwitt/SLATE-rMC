"""Prepared native data inspection and plan-derived editing; Run is unavailable."""

import json
from dataclasses import replace

from native_fit_profiles import NativeProfilesView
from native_fit_state import native_fit_session_document, reuse_plan, stage_review
from parameter_state import MAX_HISTORY_BYTES, FieldChange, HistoryAction, SessionHistory
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
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
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
        self.setWindowTitle("Prepare a fit — inputs, geometry and staged plan")
        self.resize(1150, 800)
        layout = QVBoxLayout(self)
        self.status = QLabel(
            "Load existing physics, observation and plan JSON. No observation preparation or fitting."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.readiness = QLabel("Inputs → Exclusions → Geometry review → Model / stages → Results")
        self.readiness.setWordWrap(True)
        layout.addWidget(self.readiness)
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, "1 · Inputs")
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, "2 · Exclusions / counts")
        page = QWidget()
        geometry = QVBoxLayout(page)
        self.geometry_review = QLabel()
        self.geometry_review.setWordWrap(True)
        geometry.addWidget(self.geometry_review)
        self.button(geometry, "Review acquisition exclusions", self.review_exclusions)
        self.button(geometry, "Review physical geometry", lambda: shell.physical.show())
        self.button(geometry, "Review joint geometry", lambda: shell.joint.show())
        geometry.addStretch()
        self.tabs.addTab(page, "3 · Geometry review")
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
        self.parameters.setMaximumHeight(180)
        body.addWidget(self.parameters)
        self.stages = QTableWidget(0, 6)
        self.stages.setHorizontalHeaderLabels(
            [
                "Stage name / order",
                "Active canonical names (JSON list; draft)",
                "Method",
                "Iterations",
                "Function budget",
                "Historical guards",
            ]
        )
        self.stages.itemChanged.connect(self.changed)
        self.stages.setMaximumHeight(80)
        body.addWidget(self.stages)
        declaration_notice = QLabel(
            "Active lists and historical guards are draft proposals; global fixed/gauge definitions and final stage order are read-only. No execution or native stage-result import."
        )
        declaration_notice.setWordWrap(True)
        body.addWidget(declaration_notice)
        self.selected_stage = QComboBox()
        self.selected_stage.currentIndexChanged.connect(self.select_stage)
        body.addWidget(self.selected_stage)
        self.active_parameters = QListWidget()
        self.active_parameters.setAccessibleName(
            "Declared active parameters for selected fitting stage"
        )
        self.active_parameters.setMinimumHeight(80)
        self.active_parameters.setMaximumHeight(110)
        self.active_parameters.itemChanged.connect(self.active_changed)
        body.addWidget(QLabel("Select free parameters for this stage (declared order retained):"))
        body.addWidget(self.active_parameters)
        self.advanced_declarations = QCheckBox("Advanced definitions / raw stage declarations")
        self.advanced_declarations.toggled.connect(self.show_advanced)
        body.addWidget(self.advanced_declarations)
        self.stage_details = QTextBrowser()
        body.addWidget(self.stage_details)
        self.step = QLineEdit()
        self.step.editingFinished.connect(self.changed)
        body.addWidget(self.step)
        actions = (
            ("Undo", lambda: self.undo(True)),
            ("Redo", lambda: self.undo(False)),
            ("Reuse compatible historical starts", self.reuse),
            ("Remove displayed historical description", self.remove_history),
            ("Export exact measured data", self.export_profiles),
        )
        for index, (text, action) in enumerate(actions):
            if index % 2 == 0:
                row = QHBoxLayout()
                body.addLayout(row)
            self.button(row, text, action)
        run = QPushButton("Native fit execution unavailable in this build")
        run.setEnabled(False)
        run.setToolTip(
            "Draft parsing is not full engine launch admission; indexed adoption is unavailable"
        )
        body.addWidget(run)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, "4 · Model / stages")
        results = QWidget()
        result_layout = QVBoxLayout(results)
        message = QLabel(
            "No native fit result is admitted in this build. Solver termination, convergence, rank, parameter covariance and qualification are unavailable. Preparation and saved descriptions do not constitute a qualified fit."
        )
        message.setWordWrap(True)
        result_layout.addWidget(message)
        self.button(
            result_layout,
            "Review frozen measured counts / full covariance",
            lambda: self.tabs.setCurrentIndex(1),
        )
        self.button(result_layout, "Export committed plan", self.export)
        self.button(result_layout, "Export exact measured data", self.export_profiles)
        result_layout.addStretch()
        self.tabs.addTab(results, "5 · Results")
        self.show_advanced(False)
        primary = QHBoxLayout()
        self.button(primary, "Commit displayed draft", self.commit)
        self.button(primary, "Discard pending edits", self.discard)
        self.button(primary, "Export committed plan", self.export)
        layout.addLayout(primary)
        self.button(layout, "Cancel file operation", shell._cancel_current)
        self.render()

    def showEvent(self, event):
        screen = self.screen().availableGeometry()
        self.resize(
            min(self.width(), screen.width() - 32), min(self.height(), screen.height() - 48)
        )
        super().showEvent(event)

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
        self.sync_preparation()

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
        item.setToolTip(str(value))
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
                self.selected_stage.clear()
                self.stage_details.clear()
                self.summary.clear()
                self.sync_preparation()
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
                    or "Loaded search declaration; native launch admission is unavailable in this build"
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
                    "fixed"
                    if fixed
                    else "free · " + (", ".join(stages) or "no staged declaration"),
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
            selected = self.session.selected_stage
            self.selected_stage.blockSignals(True)
            self.selected_stage.clear()
            for i, stage in enumerate(stages):
                self.selected_stage.addItem(f"{i + 1}: {stage['name']}", stage["name"])
            self.selected_stage.setCurrentIndex(max(0, self.selected_stage.findData(selected)))
            self.selected_stage.blockSignals(False)
            self.stage_details.setPlainText(
                encoded(stage_review(value, self.selected_stage.currentData(), edits))
            )
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
                        and not any(p["reason"] for p in value["definition"]["capabilities"])
                        and (col in (2, 3, 4, 5) or (col == 1 and i < len(stages) - 1))
                        and k in s
                    )
                    self.item(
                        self.stages,
                        i,
                        col,
                        edits.get("stages", {})
                        .get(s["name"], {})
                        .get(
                            k,
                            encoded(s[k])
                            if k == "active_parameters"
                            else s.get(k, "owner launch default; unavailable here"),
                        ),
                        editable,
                        s["name"],
                    )
            self.step.setReadOnly(not current or "finite_difference_step" not in plan)
            self.step.setText(
                edits.get("step", str(plan.get("finite_difference_step", "Not declared")))
            )
            self.parameters.resizeColumnsToContents()
            self.show_advanced(self.advanced_declarations.isChecked())
            self.sync_preparation()
        finally:
            self._rendering = rendering

    def select_stage(self, *_):
        if self._rendering or self.session is None:
            return
        name = self.selected_stage.currentData()
        if self.descriptions.currentData() == 0 and name != self.session.selected_stage:
            self.guard(lambda: self.store(replace(self.session, selected_stage=name)))
        else:
            edits = {} if self.session.draft_json is None else json.loads(self.session.draft_json)
            self.stage_details.setPlainText(
                encoded(
                    stage_review(
                        self.displayed(),
                        name,
                        edits if self.descriptions.currentData() == 0 else None,
                    )
                )
            )
            self.sync_preparation()

    def review_exclusions(self):
        self.shell.workspaces.setCurrentIndex(0)
        self.shell.scene_tabs.setCurrentIndex(0)
        self.status.setText(
            "Review exclusions on the acquisition detector. Loaded prepared rows/covariance remain frozen; changed exclusions require a separately admitted preparation recipe."
        )

    def show_advanced(self, shown):
        for column in (5, 7, 8, 10):
            self.parameters.setColumnHidden(column, not shown)
        self.stages.setColumnHidden(1, not shown)
        self.stage_details.setVisible(shown)
        self.step.setVisible(shown)
        if not shown:
            for column, width in (
                (0, 230),
                (1, 145),
                (2, 75),
                (3, 110),
                (4, 110),
                (6, 110),
                (9, 155),
            ):
                self.parameters.setColumnWidth(column, width)
        self.parameters.setHorizontalHeaderLabels(
            [
                "Scientific parameter",
                "Owner / scope",
                "Unit",
                "Initial",
                "Lower",
                "Lower kind",
                "Upper",
                "Upper kind",
                "Sensitivity scale",
                "Fixed / stage active",
                "Support / constraint",
            ]
        )

    def sync_preparation(self):
        value = self.displayed()
        if value is None:
            self.readiness.setText(
                "Inputs required: load physics, observations and a plan. Native execution unavailable."
            )
            self.geometry_review.setText(
                "No prepared geometry declaration. Review acquisition geometry through its existing owner; no default calibration is assumed."
            )
            self.active_parameters.clear()
            return
        self.readiness.setText(
            "Inputs: recorded · Exclusions/counts: frozen"
            + (
                " arrays inspected"
                if self.profiles is not None
                else " arrays unverified — inspect/reload"
            )
            + " · Geometry: review bound declaration · Plan: "
            + ("pending edits" if self.session.draft_json is not None else "committed draft")
            + " · Execution/results: unavailable"
        )
        self.geometry_review.setText(
            f"Prepared projection {value['inputs']['projection_revision']}\nPhysics {value['inputs']['physics_sha256']}\nThe loaded input geometry stays bound to this description. Geometry review does not adopt a new calibration into frozen observations. Review the source declarations in Inputs; covariance and native memberships are unchanged."
        )
        name = self.selected_stage.currentData()
        row = next(
            (
                r
                for r in range(self.stages.rowCount())
                if self.stages.item(r, 0).data(Qt.UserRole) == name
            ),
            None,
        )
        self.active_parameters.blockSignals(True)
        try:
            self.active_parameters.clear()
            if row is None:
                return
            try:
                active = json.loads(self.stages.item(row, 1).text())
                if not isinstance(active, list) or any(type(v) is not str for v in active):
                    raise ValueError("Invalid active parameter declaration; correct it in Advanced")
            except (ValueError, TypeError) as exc:
                self.status.setText(str(exc))
                return
            editable = bool(self.stages.item(row, 1).flags() & Qt.ItemIsEditable)
            fixed = value["plan"].get("fixed_parameters", {})
            for p, capability in zip(
                value["plan"]["parameters"], value["definition"]["capabilities"], strict=True
            ):
                item = QListWidgetItem(
                    f"{p['name']} [{p['unit']}] · {p['owner']}"
                    + (" · fixed" if p["name"] in fixed else "")
                )
                item.setData(Qt.UserRole, p["name"])
                item.setCheckState(
                    Qt.CheckState.Checked if p["name"] in active else Qt.CheckState.Unchecked
                )
                if not editable or p["name"] in fixed or capability["reason"]:
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                else:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setToolTip(
                    capability["reason"]
                    or (
                        "Final stage order and global fixed/gauge definitions are read-only"
                        if not editable
                        else "Draft selection; full launch admission remains unavailable"
                    )
                )
                self.active_parameters.addItem(item)
        finally:
            self.active_parameters.blockSignals(False)

    def active_changed(self, item):
        if self._rendering or self.session is None:
            return
        name = self.selected_stage.currentData()
        row = next(
            r
            for r in range(self.stages.rowCount())
            if self.stages.item(r, 0).data(Qt.UserRole) == name
        )
        cell = self.stages.item(row, 1)
        if not cell.flags() & Qt.ItemIsEditable:
            return
        previous = json.loads(cell.text())
        selected = [
            self.active_parameters.item(i).data(Qt.UserRole)
            for i in range(self.active_parameters.count())
            if self.active_parameters.item(i).checkState() == Qt.CheckState.Checked
        ]
        ordered = [v for v in previous if v in selected] + [
            v for v in selected if v not in previous
        ]
        cell.setText(encoded(ordered))

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
                    (1, "active_parameters"),
                    (2, "method"),
                    (3, "maximum_iterations"),
                    (4, "maximum_function_evaluations"),
                    (5, "enforce_historical_guards"),
                )
                if self.stages.item(i, c).flags() & Qt.ItemIsEditable
            }
        self.guard(
            lambda: self.store(replace(self.session, draft_json=encoded(edits)), render=False)
        )
        if self.session.draft_json is not None:
            self.stage_details.setPlainText(
                encoded(
                    stage_review(
                        self.displayed(),
                        self.selected_stage.currentData(),
                        json.loads(self.session.draft_json),
                    )
                )
            )

        self.sync_preparation()

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
                if k == "method":
                    s[k] = text
                elif k == "active_parameters":
                    active = json.loads(text)
                    if type(active) is not list or any(type(v) is not str for v in active):
                        raise ValueError(
                            "Active coordinates require a JSON list of canonical names"
                        )
                    s[k] = active
                elif k == "enforce_historical_guards":
                    if text.lower() not in ("true", "false"):
                        raise ValueError("Historical guard proposal must be true or false")
                    s[k] = text.lower() == "true"
                else:
                    s[k] = int(text)
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
