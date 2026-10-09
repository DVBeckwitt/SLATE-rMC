"""Named immutable attempts, reviewed independent copies and bounded recovery choices."""

import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from attempt_state import attempt_rows, history_text, read_history
from native_simulation_state import native_reference_document
from parameter_state import FieldChange, HistoryAction, SessionHistory
from project_state import ProjectDocument, project_to_document
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
)
from sample_state import encoded, payload_hash
from simulation_state import simulation_reference_document


class ProjectToolsPanel(QDialog):
    def __init__(self, shell):
        super().__init__(shell)
        self.shell = shell
        self.history = SessionHistory()
        self.epoch = 0
        self.copy_review = None
        self.copy_path = None
        self.copy_sha = None
        self.recovery_rows = ()
        self.setWindowTitle("Attempts / independent copy / recovery")
        self.resize(1050, 760)
        layout = QVBoxLayout(self)
        self.status = QLabel(
            "Names and inspection selections preserve immutable scientific records. Owner selection/reuse uses each route's existing checks."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.rows = QTableWidget(0, 4)
        self.rows.setHorizontalHeaderLabels(("Route", "Name", "State", "Stable owner / attempt"))
        self.rows.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.rows.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.rows.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.rows.itemSelectionChanged.connect(self.inspect)
        layout.addWidget(self.rows)
        self.name = QLineEdit()
        self.name.setMaxLength(256)
        layout.addWidget(self.name)
        self.buttons(
            layout,
            [
                ("Refresh", self.refresh),
                ("Rename display", self.rename),
                ("Select for inspection", self.select_inspection),
                ("Open existing owner", self.open_owner),
                ("Remove old simulation reference", self.remove_reference),
                ("Undo name/inspection", lambda: self.undo(True)),
                ("Redo", lambda: self.undo(False)),
            ],
        )
        self.copy_name = QLineEdit("Independent experiment")
        self.copy_name.setMaxLength(256)
        self.copy_name.textChanged.connect(self.invalidate)
        layout.addWidget(self.copy_name)
        self.buttons(
            layout,
            [
                ("Review independent copy", self.review_copy),
                ("Publish reviewed copy", self.publish_copy),
                ("Open published copy", self.open_copy),
                ("Review recovery folder", self.review_recovery),
                ("Open reviewed recovery", self.open_recovery),
                ("Cancel", shell._cancel_current),
            ],
        )
        self.recovery = QComboBox()
        self.recovery.currentIndexChanged.connect(self.inspect_recovery)
        layout.addWidget(self.recovery)
        self.details = QTextBrowser()
        layout.addWidget(self.details)

    def buttons(self, layout, actions):
        row = QHBoxLayout()
        layout.addLayout(row)
        for label, action in actions:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, a=action: self.guard(a))
            row.addWidget(button)

    def guard(self, action):
        try:
            action()
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def resident_bytes(self):
        duplicate = (
            0
            if self.copy_review is None
            else len(
                json.dumps(
                    project_to_document(
                        self.copy_review[1], self.shell._project_path or self.shell._recovery_path()
                    )
                ).encode()
            )
        )
        return (
            self.history.bytes_used
            + duplicate
            + len(encoded(self.recovery_rows).encode())
            + len(self.details.toPlainText().encode())
            + 8192
        )

    def snapshot(self):
        anchor = self.shell._project_path or self.shell._recovery_path()
        state = ProjectDocument(
            self.shell.project, self.shell._capture_view(), self.shell._numeric_draft
        )
        return anchor, state, project_to_document(state, anchor)

    def context(self):
        return (self.shell.project.project_id, self.epoch, payload_hash(self.snapshot()[2]))

    def invalidate(self, *_):
        self.epoch += 1
        self.copy_review = None
        self.shell._supersede_project_tools()

    def refresh(self):
        self.rows.blockSignals(True)
        try:
            inventory = attempt_rows(self.shell._capture_view())
            history = read_history(self.shell.attempts_json)
            self.rows.setRowCount(len(inventory))
            for i, row in enumerate(inventory):
                for j, text in enumerate(
                    (row.key[0], row.name, row.state, " / ".join(row.key[1:]))
                ):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, row.key)
                    self.rows.setItem(i, j, item)
                if row.key == history.inspection:
                    self.rows.selectRow(i)
            self.rows.resizeColumnsToContents()
        finally:
            self.rows.blockSignals(False)
        self.inspect()

    def selected(self):
        index = self.rows.currentRow()
        if index < 0 or self.rows.item(index, 0) is None:
            raise ValueError("Choose an attempt")
        key = tuple(self.rows.item(index, 0).data(Qt.ItemDataRole.UserRole))
        row = next((v for v in attempt_rows(self.shell._capture_view()) if v.key == key), None)
        if row is None:
            raise ValueError("Attempt changed; refresh history")
        return row

    def inspect(self):
        if self.rows.currentRow() < 0:
            return
        self.guard(lambda: self.details.setPlainText(self.selected().detail))

    def store(self, text, label):
        read_history(text)
        view = replace(self.shell._capture_view(), attempts_json=text)
        self.shell._validate_project_admission(self.shell.project, view=view)
        old = self.shell.attempts_json
        if old == text:
            return
        action = HistoryAction(
            label,
            (FieldChange(self.shell.project.project_id, "attempts_json", old, text),),
            len(old.encode()) + len(text.encode()) + 256,
        )
        self.history.push(action)
        self.shell.attempts_json = text
        self.shell._mark_dirty()
        self.refresh()

    def rename(self):
        row = self.selected()
        name = self.name.text().strip()
        if not name:
            raise ValueError("Enter a display name")
        history = read_history(self.shell.attempts_json)
        names = (*(v for v in history.names if v[:3] != row.key), (*row.key, name))
        text = history_text(replace(history, names=names))
        read_history(text)
        self.store(text, "Rename attempt display")

    def select_inspection(self):
        self.store(
            history_text(
                replace(read_history(self.shell.attempts_json), inspection=self.selected().key)
            ),
            "Select immutable attempt for inspection",
        )

    def remove_reference(self):
        row = self.selected()
        route, owner, identity = row.key
        if route not in ("configured", "native"):
            raise ValueError("Remove other history through its existing owner")
        current = (
            self.shell.simulator.result_reference
            if route == "configured"
            else self.shell.simulator.native.result_reference
        )
        if current is not None and current.sha256 == identity:
            raise ValueError("Current selected output cannot be removed from retained history")
        history = read_history(self.shell.attempts_json)
        refs = getattr(history, route)
        if not any(v.sha256 == identity and str(v.draft_id) == owner for v in refs):
            raise ValueError("Choose a retained simulation output")
        self.store(
            history_text(
                replace(
                    history,
                    **{route: tuple(v for v in refs if v.sha256 != identity)},
                    inspection=None if history.inspection == row.key else history.inspection,
                    names=tuple(v for v in history.names if v[:3] != row.key),
                )
            ),
            "Remove retained output reference",
        )

    def undo(self, undo):
        source = self.history.undo_actions if undo else self.history.redo_actions
        if not source:
            raise ValueError("Nothing to undo" if undo else "Nothing to redo")
        action = source[-1]
        change = action.changes[0]
        if change.acquisition_id != self.shell.project.project_id:
            raise ValueError("Undo belongs to another project")
        expected = change.after if undo else change.before
        if self.shell.attempts_json != expected:
            raise ValueError("Attempt metadata changed independently; refresh before editing")
        text = change.before if undo else change.after
        self.shell._validate_project_admission(
            self.shell.project, view=replace(self.shell._capture_view(), attempts_json=text)
        )
        source.pop()
        (self.history.redo_actions if undo else self.history.undo_actions).append(action)
        self.shell.attempts_json = text
        self.shell._mark_dirty()
        self.refresh()

    def open_owner(self):
        route, owner, identity = self.selected().key
        if route == "prepared":
            panel = self.shell.prepared
            panel.show()
            session = panel.session
            texts = (session.current_json, *session.history_json)
            for i, text in enumerate(texts):
                d = json.loads(text)
                if d["sha256"] == identity and d["definition"]["sha256"] == owner:
                    panel.descriptions.setCurrentIndex(panel.descriptions.findData(i))
                    panel.present()
                    return
        elif route in ("sample", "joint"):
            panel = getattr(self.shell, route)
            if panel.session is None or str(panel.session.session_id) != owner:
                raise ValueError("Historical owner no longer current")
            panel.show()
            index = panel.results.findData(identity)
            if index >= 0:
                panel.results.setCurrentIndex(index)
            return
        elif route == "hbn":
            panel = self.shell.hbn
            session = next((v for v in panel.sessions if str(v.session_id) == owner), None)
            if session is None or self.shell.selected_acquisition_id != session.acquisition_id:
                raise ValueError(
                    "Choose this owner's acquisition first; current pending state remains"
                )
            panel.show()
            panel.render()
            index = panel.results.findData(identity)
            if index >= 0:
                panel.results.setCurrentIndex(index)
            return
        elif route in ("configured", "native"):
            history = read_history(self.shell.attempts_json)
            ref = next(
                (
                    v
                    for v in getattr(history, route)
                    if str(v.draft_id) == owner and v.sha256 == identity
                ),
                None,
            )
            self.shell.workspaces.setCurrentIndex(1)
            if ref is not None:
                document = (
                    simulation_reference_document(ref)
                    if route == "configured"
                    else native_reference_document(ref)
                )
                self.shell._request_simulation("reopen", encoded(document).encode())
            return
        raise ValueError("Attempt is available for immutable inspection only")

    def review_copy(self):
        anchor, state, source = self.snapshot()
        name = self.copy_name.text().strip()
        if not name:
            raise ValueError("Name the independent experiment")
        history = replace(
            read_history(state.view.attempts_json), inherited_from=state.project.project_id
        )
        duplicate = replace(
            state,
            project=replace(state.project, project_id=uuid4(), name=name),
            view=replace(state.view, attempts_json=history_text(history)),
        )
        document = project_to_document(duplicate, anchor)
        self.copy_review = (self.context(), duplicate)
        self.details.setPlainText(
            json.dumps(
                {
                    "source_id": str(state.project.project_id),
                    "new_id": str(duplicate.project.project_id),
                    "name": name,
                    "source_snapshot_sha256": payload_hash(source),
                    "reviewed_copy": document,
                    "sharing": "Original scientific owners/results and exact files retained as inherited inspection. Edits, selections, pending text, undo and UUID recovery are independent. Portable archive copies the files.",
                },
                indent=2,
            )
        )

    def publish_copy(self):
        if self.copy_review is None or self.copy_review[0] != self.context():
            raise ValueError("Review the current independent copy first")
        path, _ = QFileDialog.getSaveFileName(
            self, "Publish new independent project", "", "SLATE (*.slate.json)"
        )
        if not path:
            return
        context, duplicate = self.copy_review
        # File dialog runs an event loop: recheck state before committing the reviewed request.
        if context != self.context():
            raise ValueError("Project changed during destination choice; review again")
        destination = Path(path).absolute()
        if destination.exists():
            raise ValueError("Choose a new destination; existing files are protected")
        self.request(
            "duplicate",
            document=project_to_document(duplicate, destination),
            destination=str(destination),
            source_id=str(self.shell.project.project_id),
            new_id=str(duplicate.project.project_id),
        )

    def request(self, operation, **fields):
        self.shell._request_project_tools(
            encoded({"operation": operation, **fields}).encode(), self.context()
        )

    def ready(self, value):
        self.details.setPlainText(json.dumps(json.loads(value.review_json), indent=2))
        if value.operation == "duplicate":
            self.copy_path, self.copy_sha = value.path, value.sha256
            self.status.setText(
                "Independent copy published. Open separately through ordinary dirty-state handling."
            )
        else:
            self.recovery_rows = tuple(json.loads(value.review_json)["drafts"])
            self.recovery.blockSignals(True)
            self.recovery.clear()
            for row in self.recovery_rows:
                self.recovery.addItem(
                    (
                        row.get("name", "Invalid draft")
                        + " · "
                        + row.get("project_id", Path(row["path"]).name)
                        + " · "
                        + (f"{row['age_seconds']:.0f}s old" if row["valid"] else row["reason"])
                    ),
                    row,
                )
            self.recovery.blockSignals(False)
            self.status.setText(
                "Recovery candidates reviewed. Invalid/changed files cannot replace current state."
            )

    def open_copy(self):
        if self.copy_path is None:
            raise ValueError("Publish an independent copy first")
        self.shell.open_project(self.copy_path, expected_sha=self.copy_sha)

    def review_recovery(self):
        self.request("recovery", root=str(self.shell.recovery_root))

    def inspect_recovery(self, *_):
        row = self.recovery.currentData()
        if row is not None:
            self.details.setPlainText(json.dumps(row, indent=2))

    def open_recovery(self):
        row = self.recovery.currentData()
        if row is None or not row["valid"]:
            raise ValueError("Choose a valid reviewed recovery draft")
        self.shell.open_project(Path(row["path"]), recovery=True, expected_sha=row["sha256"])
