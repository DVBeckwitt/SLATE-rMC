"""Review selected delivered products before portable archive publication/reopen."""

import json
from pathlib import Path

from archive_io import product_choices
from archive_storage import storage_files
from project_state import ProjectDocument, project_to_document
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)
from sample_state import encoded, payload_hash


class ArchivePanel(QDialog):
    def __init__(self, shell):
        super().__init__(shell)
        self.shell = shell
        self.review = None
        self.imported_path = None
        self.epoch = 0
        self.boxes = {}
        self.setWindowTitle("Portable archive / exact exports")
        self.resize(1050, 750)
        layout = QVBoxLayout(self)
        self.status = QLabel(
            "Review current delivered products. Unavailable selected predecessors block self-contained export. Run/adoption remains unavailable."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.selection_layout = QVBoxLayout()
        layout.addLayout(self.selection_layout)
        row = QHBoxLayout()
        layout.addLayout(row)
        for label, action in [
            ("Refresh products", self.refresh),
            ("Review selection", self.request_review),
            ("Export reviewed archive", self.export),
            ("Import archive elsewhere", self.import_archive),
            ("Open imported project", self.reopen),
            ("Cancel", shell._cancel_current),
        ]:
            b = QPushButton(label)
            b.clicked.connect(lambda checked=False, a=action: self.guard(a))
            row.addWidget(b)
        self.details = QTextBrowser()
        layout.addWidget(self.details)
        exports = QHBoxLayout()
        layout.addLayout(exports)
        for label, action in [
            ("Configured/native result exports", lambda: shell.workspaces.setCurrentIndex(1)),
            ("Prepared profiles / plan export", lambda: shell.prepared.show()),
            ("hBN result exports", shell._show_hbn),
            ("Sample result exports", shell._show_sample),
            ("Joint report exports", shell._show_joint),
        ]:
            b = QPushButton(label)
            b.clicked.connect(lambda checked=False, a=action: self.guard(a))
            exports.addWidget(b)

    def guard(self, action):
        try:
            action()
        except (ValueError, TypeError, RuntimeError, OSError, KeyError) as exc:
            self.status.setText(str(exc))

    def snapshot(self):
        state = ProjectDocument(
            self.shell.project, self.shell._capture_view(), self.shell._numeric_draft
        )
        anchor = self.shell._project_path or Path.home() / ".codex" / "unsaved-archive.slate.json"
        return anchor, project_to_document(state, anchor)

    def context(self):
        _, document = self.snapshot()
        return (
            self.shell.project.project_id,
            self.epoch,
            payload_hash(document),
            tuple(k for k, b in self.boxes.items() if b.isChecked()),
        )

    def invalidate(self):
        self.review = None
        self.epoch += 1
        self.shell._supersede_archive()
        self.status.setText("Selection/state changed; review again before export")

    def refresh(self):
        self.invalidate()
        while self.selection_layout.count():
            item = self.selection_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.boxes = {}
        state = ProjectDocument(
            self.shell.project, self.shell._capture_view(), self.shell._numeric_draft
        )
        for key, label in product_choices(state):
            box = QCheckBox(label)
            box.setChecked(True)
            box.toggled.connect(self.invalidate)
            self.boxes[key] = box
            self.selection_layout.addWidget(box)
        storage = storage_files(state.view.archive_storage_json)
        self.details.setPlainText(
            "Current archive storage: "
            + encoded(storage)
            + "\nExisting inspection/comparison/simulation figure exports retain exact paired values. This archive includes immutable saved state and already retained files; it computes no output. Future Run/stages/preparation remain unavailable."
        )

    def request_review(self):
        anchor, document = self.snapshot()
        self.request(
            "review",
            anchor=str(anchor),
            document=document,
            selection=[k for k, b in self.boxes.items() if b.isChecked()],
        )

    def request(self, operation, **fields):
        self.shell._request_archive(
            operation, encoded({"operation": operation, **fields}).encode(), self.context()
        )

    def ready(self, value):
        if value.operation == "review":
            self.review = value.review_json
            review = json.loads(value.review_json)
            self.details.setPlainText(
                json.dumps({k: v for k, v in review.items() if k != "document"}, indent=2)
            )
        elif value.operation == "import":
            self.imported_path = value.path
            self.details.setPlainText(
                str(value.path)
                + "\n"
                + json.dumps(json.loads(value.review_json), indent=2)
                + "\nImport is complete. Open adopts this reviewed project separately; current dirty/pending state is retained until ordinary Open."
            )
        self.status.setText(value.detail)

    def export(self):
        if self.review is None:
            raise ValueError("Review the current selection first")
        review = json.loads(self.review)
        if (
            review["snapshot_sha256"] != self.context()[2]
            or tuple(review["selection"]) != self.context()[3]
        ):
            raise ValueError("Project or selection changed; review again")
        if review["unavailable"]:
            raise ValueError(
                "Unavailable selected predecessors block a self-contained archive; review or deselect their products explicitly"
            )
        path, _ = QFileDialog.getSaveFileName(
            self, "Export reviewed self-contained archive", "", "SLATE archive (*.slatezip)"
        )
        if path:
            self.request("export", review_json=self.review, path=path)

    def import_archive(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose portable archive", "", "SLATE archive (*.slatezip)"
        )
        if not path:
            return
        parent = QFileDialog.getExistingDirectory(
            self, "Choose external parent for a new archive directory"
        )
        if not parent:
            return
        destination = Path(parent) / (Path(path).stem + "-reopened")
        self.request("import", path=path, destination=str(destination))

    def reopen(self):
        if self.imported_path is None:
            raise ValueError("Import a validated archive first")
        self.shell.open_project(self.imported_path)
