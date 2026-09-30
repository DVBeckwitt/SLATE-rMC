"""Compact setup/source review controls; ShellWindow retains jobs, state and history."""

from dataclasses import replace
from pathlib import Path

from numeric_fields import PARAMETERS
from project_state import ProjectFormatError
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)
from setup_io import SetupApplication, TemplateFile
from setup_state import SetupTemplate, capture_template, template_bytes, template_from_bytes


class TemplateEditor(QDialog):
    def __init__(self, template: SetupTemplate, parent=None) -> None:
        super().__init__(parent)
        self.template = template
        self.setWindowTitle("Edit reusable setup")
        self.resize(800, 560)
        layout = QVBoxLayout(self)
        self.name = QLineEdit(template.name)
        self.name.setObjectName("template_name")
        layout.addWidget(QLabel("Template name (edits affect this template only)"))
        layout.addWidget(self.name)
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("template_fields")
        self.table.setHorizontalHeaderLabels(("Field", "Value (stored units)", "Unit", "Origin"))
        self.rows = (*template.metadata, *template.initial_values)
        self.metadata_count = len(template.metadata)
        for row, values in enumerate(self.rows):
            self.table.insertRow(row)
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if column != 1:
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, cell)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)
        label = QLabel(
            "Blank value omits that reusable default. Unexposed canonical values stay in the target configuration. Fixed/derived values: "
            + "; ".join(item.label + ": " + item.reason for item in PARAMETERS if not item.editable)
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        metadata, numeric = [], []
        try:
            for index, (field, old, unit, origin) in enumerate(self.rows):
                text = self.table.item(index, 1).text().strip()
                if not text:
                    continue
                value = (
                    float(text)
                    if index >= self.metadata_count or field in ("incidence_rad", "exposure_s")
                    else text
                )
                changed = value != old
                row = (field, value, unit, "manual template edit" if changed else origin)
                (metadata if index < self.metadata_count else numeric).append(row)
            self.template = replace(
                self.template,
                name=self.name.text().strip(),
                revision=self.template.revision + 1,
                metadata=tuple(metadata),
                initial_values=tuple(numeric),
            )
            template_bytes(self.template)
        except (ValueError, ProjectFormatError) as exc:
            QMessageBox.warning(self, "Invalid template", str(exc))
            return
        self.accept()


class SetupDialog(QDialog):
    def __init__(self, shell) -> None:
        super().__init__(shell)
        self.shell = shell
        self.setWindowTitle("Reusable setup and sources")
        self.resize(950, 650)
        layout = QVBoxLayout(self)
        self.template_status = QLabel(
            "No template loaded. Capture declared values from the active acquisition or load a named template."
        )
        self.template_status.setWordWrap(True)
        layout.addWidget(self.template_status)
        actions = QGridLayout()
        self.capture_button = QPushButton("Capture setup")
        self.load_button = QPushButton("Load template")
        self.edit_button = QPushButton("Edit template")
        self.save_button = QPushButton("Save template")
        self.apply_button = QPushButton("Review / apply to selected")
        self.copy_button = QPushButton("Review copy to data folder")
        self.relink_button = QPushButton("Locate / review relocation")
        self.kind = QComboBox()
        self.kind.addItem("OSC decoded identity", "osc")
        self.kind.addItem("CIF byte identity", "cif")
        self.kind.addItem("Configuration + dependent CIF byte identities", "configuration")
        for index, button in enumerate(
            (
                self.capture_button,
                self.load_button,
                self.edit_button,
                self.save_button,
                self.apply_button,
                self.copy_button,
                self.relink_button,
            )
        ):
            actions.addWidget(button, index // 3, index % 3)
        actions.addWidget(self.kind, 2, 1, 1, 2)
        layout.addLayout(actions)
        notice = QLabel(
            "Default: reference inputs in place. Copy requires reviewed destinations and byte totals; originals stay unchanged. Remove from project never deletes files. For a missing dependent CIF, locate the matching configuration/CIF bundle. Different content is rejected by Relink; Replace CIF / configuration is a separate action in the main review."
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.references = QTableWidget(0, 5)
        self.references.setHorizontalHeaderLabels(
            ("Acquisition", "Reference", "Status", "Current path", "Expected identity")
        )
        self.references.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.references.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.references, 1)
        self.message = QLabel("Ready")
        self.message.setWordWrap(True)
        self.message.setObjectName("setup_message")
        layout.addWidget(self.message)
        self.cancel_button = QPushButton("Cancel operation")
        self.cancel_button.clicked.connect(shell._cancel_current)
        layout.addWidget(self.cancel_button)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        self.capture_button.clicked.connect(self.capture)
        self.edit_button.clicked.connect(self.edit)
        self.load_button.clicked.connect(self.load)
        self.save_button.clicked.connect(self.save)
        self.apply_button.clicked.connect(self.apply)
        self.copy_button.clicked.connect(self.copy)
        self.relink_button.clicked.connect(self.relink)
        self.refresh()

    def refresh(self) -> None:
        shell = self.shell
        template = shell._setup_template
        if template is not None:
            self.template_status.setText(
                f"{template.name} | {template.template_id} | revision {template.revision} | {len(template.metadata)} metadata and {len(template.initial_values)} numeric defaults | file: {shell._setup_template_path or 'unsaved'}"
            )
        busy = shell._active_kind == "setup"
        for button in (self.capture_button, self.load_button, self.copy_button, self.relink_button):
            button.setEnabled(not busy)
        for button in (self.edit_button, self.save_button, self.apply_button):
            button.setEnabled(template is not None and not busy)
        self.cancel_button.setEnabled(busy and shell.cancel_button.isEnabled())
        self.references.setRowCount(0)
        ids = shell._selected_project_ids()
        for item in shell.project.acquisitions:
            if item.acquisition_id not in ids:
                continue
            check = shell._source_checks.get(item.acquisition_id)
            values = [
                (
                    "osc",
                    item.source_path,
                    item.source_sha256,
                    check.state if check else "unverified",
                )
            ]
            values.extend(
                (
                    v.kind,
                    getattr(item.metadata, f"{v.kind}_path"),
                    getattr(item.metadata, f"{v.kind}_sha256"),
                    v.state,
                )
                for v in shell._reference_states(item)
            )
            for kind, path, digest, state in values:
                row = self.references.rowCount()
                self.references.insertRow(row)
                for col, value in enumerate(
                    (item.name, kind, state, str(path), digest or "unknown")
                ):
                    cell = QTableWidgetItem(str(value))
                    cell.setToolTip(f"{item.acquisition_id}\n{value}")
                    self.references.setItem(row, col, cell)

    def capture(self) -> None:
        item = self.shell._numeric_acquisition()
        if item is None:
            self.message.setText("Select an active acquisition first")
            return
        draft = (
            self.shell._numeric_draft
            if self.shell._numeric_identity_matches(item)
            and self.shell._validated_numeric
            == (self.shell.project.project_id, self.shell._numeric_draft)
            else None
        )
        editor = TemplateEditor(capture_template(item.name + " setup", item.metadata, draft), self)
        if editor.exec() == QDialog.DialogCode.Accepted:
            self.shell._setup_template = editor.template
            self.shell._setup_template_path = None
            self.shell._setup_template_hash = None
            self.refresh()

    def edit(self) -> None:
        editor = TemplateEditor(self.shell._setup_template, self)
        if editor.exec() == QDialog.DialogCode.Accepted:
            self.shell._setup_template = editor.template
            self.refresh()

    def load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load setup template", "", "SLATE setup (*.slate-template.json)"
        )
        if path:
            self.shell._start_setup("load_template", {"path": path})

    def save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save setup template",
            str(
                self.shell._setup_template_path
                or (self.shell._setup_template.name + ".slate-template.json")
            ),
            "SLATE setup (*.slate-template.json)",
        )
        if path:
            if not path.endswith(".slate-template.json"):
                path += ".slate-template.json"
            expected = (
                self.shell._setup_template_hash
                if Path(path).absolute() == self.shell._setup_template_path
                else None
            )
            self.shell._start_setup(
                "save_template",
                {
                    "path": path,
                    "encoded": template_bytes(self.shell._setup_template).decode(),
                    "expected_sha256": expected,
                },
            )

    def apply(self) -> None:
        self.shell._start_setup(
            "apply_template", {"encoded": template_bytes(self.shell._setup_template).decode()}
        )

    def copy(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose external project data folder")
        if folder:
            self.shell._start_setup("review_copy", {"folder": folder})

    def relink(self) -> None:
        kind = self.kind.currentData()
        extension = (
            "OSC (*.osc *.osc.gz)"
            if kind == "osc"
            else "CIF (*.cif)"
            if kind == "cif"
            else "Configuration (*.yaml *.yml)"
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate matching " + kind + " reference", "", extension
        )
        if path:
            self.shell._start_setup("relink", {"kind": kind, "path": path})

    def confirm(self, heading: str, lines: tuple[str, ...]) -> bool:
        review = QDialog(self)
        review.setWindowTitle(heading)
        review.resize(850, 500)
        layout = QVBoxLayout(review)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText("\n\n".join(lines))
        layout.addWidget(text)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(review.accept)
        buttons.rejected.connect(review.reject)
        layout.addWidget(buttons)
        return review.exec() == QDialog.DialogCode.Accepted

    def ready(self, operation: str, value: object) -> None:
        shell = self.shell
        if isinstance(value, TemplateFile):
            shell._setup_template = template_from_bytes(value.encoded)
            shell._setup_template_path, shell._setup_template_hash = value.path, value.sha256
            self.message.setText(
                f"Template {operation.replace('_', ' ')} completed and read back: {value.sha256}"
            )
        elif operation == "review_copy":
            rows = value["rows"]
            lines = (
                f"Copy {len({r['destination'] for r in rows})} files; {value['total_bytes']} bytes total. No binding until every destination passes verification. Cancellation/failure may leave completed byte-verified copies in {value['root']}.",
            )
            lines += tuple(
                f"{r['acquisition_id']} {r['kind']}: {r['source']} ({r['source_bytes']} bytes) -> {r['destination']} ({r['destination_bytes']} bytes)\nSource raw hash {r['source_raw_sha256']}\nDestination hash {r['destination_raw_sha256']}\n{'Derived YAML path rewrite; new configuration identity' if r['derived_yaml'] else 'Exact original bytes'}"
                for r in rows
            )
            if self.confirm("Review source storage", lines) and shell._setup_context_current():
                shell._pending_setup_copy = value
            else:
                self.message.setText("Copy review canceled or stale; no files copied")
        elif isinstance(value, SetupApplication):
            if operation == "copy" or (
                value.review
                and self.confirm(
                    "Review compatible setup changes"
                    if operation == "apply_template"
                    else "Review identity-preserving relocation",
                    value.review,
                )
            ):
                shell._commit_setup(value)
            else:
                self.message.setText("No change committed (canceled or no-op)")
        self.refresh()
