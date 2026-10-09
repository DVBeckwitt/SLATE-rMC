"""Acquisition recipe review and explicit routing to existing frozen-input owners."""

import json
from dataclasses import replace
from pathlib import Path

from attempt_state import attempt_rows
from preparation_state import RECIPES, read_preparation, review_inputs
from project_state import ProjectDocument, project_to_document
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
)
from sample_state import encoded, payload_hash


class PreparationPanel(QDialog):
    def __init__(self, shell):
        super().__init__(shell)
        self.shell = shell
        self.epoch = 0
        self.setWindowTitle("Acquisition preparation / frozen output review")
        self.resize(1050, 800)
        layout = QVBoxLayout(self)
        self.status = QLabel(
            "Select an acquisition and review its exact inputs. Existing frozen outputs remain independently inspectable."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.acquisitions = QComboBox()
        self.recipes = QComboBox()
        for key, label, _ in RECIPES:
            self.recipes.addItem(label, key)
        layout.addWidget(self.acquisitions)
        layout.addWidget(self.recipes)
        self.reason = QLabel()
        self.reason.setWordWrap(True)
        layout.addWidget(self.reason)
        layout.addWidget(
            QLabel(
                "Pending review notes (proposals only; no scientific defaults or background controls)"
            )
        )
        self.notes = QTextEdit()
        self.notes.setMaximumHeight(85)
        layout.addWidget(self.notes)
        self.buttons(
            layout, [("Review current inputs", self.review), ("Cancel", shell._cancel_current)]
        )
        self.prepare = QPushButton("Numerical Prepare unavailable")
        self.prepare.setEnabled(False)
        layout.addWidget(self.prepare)
        self.reviews = QComboBox()
        layout.addWidget(self.reviews)
        self.buttons(
            layout,
            [
                ("Inspect retained review", self.inspect_review),
                ("Remove selected review", self.remove_review),
            ],
        )
        self.outputs = QComboBox()
        layout.addWidget(self.outputs)
        self.buttons(
            layout,
            [
                ("Inspect frozen counts / full covariance", self.inspect_output),
                ("Open prepared draft / stage editor", self.handoff),
                ("Browse existing prepared inputs", self.browse_inputs),
            ],
        )
        self.details = QTextBrowser()
        layout.addWidget(self.details)
        for control in (self.acquisitions, self.recipes, self.outputs):
            control.currentIndexChanged.connect(lambda *_: self.guard(self.changed))
        self.notes.textChanged.connect(lambda: self.guard(self.changed))

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
        return (
            len(self.details.toPlainText().encode()) + len(self.notes.toPlainText().encode()) + 8192
        )

    def choices(self):
        return read_preparation(self.shell.preparation_json)

    def snapshot(self):
        anchor = self.shell._project_path or self.shell._recovery_path()
        return project_to_document(
            ProjectDocument(
                self.shell.project, self.shell._capture_view(), self.shell._numeric_draft
            ),
            anchor,
        )

    def inputs(self):
        value = review_inputs(
            self.snapshot(), self.choices(), self.shell._project_path or self.shell._recovery_path()
        )
        value["delivered_geometry"] = [
            dict(owner=list(row.key), state=row.state)
            for row in attempt_rows(self.shell._capture_view())
            if row.key[0] in ("hbn", "sample", "joint")
        ]
        return value

    def context(self):
        try:
            digest = payload_hash(self.inputs())
        except ValueError:
            digest = None
        return self.shell.project.project_id, self.epoch, digest

    def invalidate(self):
        self.epoch += 1
        self.shell._supersede_preparation()
        self.status.setText(
            "Inputs or choices changed. Review again; retained reviews and frozen outputs remain historical inspection."
        )

    def store(self, value):
        text = encoded(value)
        read_preparation(text)
        self.shell._validate_project_admission(
            self.shell.project, view=replace(self.shell._capture_view(), preparation_json=text)
        )
        self.shell.preparation_json = text
        self.shell._mark_dirty()

    def refresh(self):
        choices = self.choices()
        for control in (self.acquisitions, self.recipes, self.notes, self.outputs):
            control.blockSignals(True)
        try:
            self.acquisitions.clear()
            self.acquisitions.addItem("Select an imported acquisition", None)
            for acquisition in self.shell.project.acquisitions:
                self.acquisitions.addItem(acquisition.name, str(acquisition.acquisition_id))
            self.acquisitions.setCurrentIndex(
                max(0, self.acquisitions.findData(choices["acquisition_id"]))
            )
            self.recipes.setCurrentIndex(self.recipes.findData(choices["recipe"]))
            self.notes.setPlainText(choices["notes"])
            self.outputs.clear()
            session = self.shell.prepared.session
            if session:
                for i, text in enumerate((session.current_json, *session.history_json)):
                    value = json.loads(text)
                    self.outputs.addItem(
                        f"{'Current draft' if i == 0 else 'Historical'} / {value['inputs']['sample_id']} / {value['sha256'][:12]}",
                        value["sha256"],
                    )
                self.outputs.setCurrentIndex(
                    max(0, self.outputs.findData(choices["selected_description"]))
                )
            self.reason.setText(next(v[2] for v in RECIPES if v[0] == choices["recipe"]))
            self.reviews.clear()
            for row in choices["reviews"]:
                self.reviews.addItem(
                    row["inputs"]["acquisition"]["name"] + " / " + row["sha256"][:12], row["sha256"]
                )
        finally:
            for control in (self.acquisitions, self.recipes, self.notes, self.outputs):
                control.blockSignals(False)

    def changed(self):
        choices = self.choices()
        choices.update(
            acquisition_id=self.acquisitions.currentData(),
            recipe=self.recipes.currentData(),
            notes=self.notes.toPlainText(),
            selected_description=self.outputs.currentData(),
        )
        self.store(choices)
        self.reason.setText(next(v[2] for v in RECIPES if v[0] == choices["recipe"]))

    def review(self):
        self.changed()
        if len(self.choices()["reviews"]) >= 8:
            raise ValueError(
                "Eight reviews retained; explicitly remove an old review before another"
            )
        self.shell._request_preparation(encoded(self.inputs()).encode(), self.context())

    def ready(self, value):
        row = json.loads(value.review_json)
        if row["inputs"] != self.inputs():
            raise ValueError("Stale review inputs rejected")
        for file in row["files"]:
            if file["state"] == "verified":
                stat = Path(file["path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (file["size"], file["mtime_ns"]):
                    raise ValueError("Source changed after worker review; review again")
        choices = self.choices()
        if not any(v["sha256"] == row["sha256"] for v in choices["reviews"]):
            choices["reviews"].append(row)
            self.store(choices)
        self.refresh()
        self.reviews.setCurrentIndex(self.reviews.findData(row["sha256"]))
        self.inspect_review()
        self.status.setText(
            "Exact input identity review retained. Numerical Prepare unavailable; no scientific output generated."
        )

    def inspect_review(self):
        row = next(
            (v for v in self.choices()["reviews"] if v["sha256"] == self.reviews.currentData()),
            None,
        )
        if row is None:
            raise ValueError("Choose a retained review")
        try:
            current = row["inputs"] == self.inputs()
        except ValueError:
            current = False
        self.details.setPlainText(
            json.dumps(
                dict(
                    review=row,
                    current_declared_inputs=current,
                    readiness="Historical identity receipt; source bytes must be verified again before any later scientific use. Numerical Prepare remains unavailable.",
                ),
                indent=2,
            )
        )

    def remove_review(self):
        choices = self.choices()
        choices["reviews"] = [
            v for v in choices["reviews"] if v["sha256"] != self.reviews.currentData()
        ]
        self.store(choices)
        self.refresh()

    def open_output(self):
        session = self.shell.prepared.session
        if session is None:
            raise ValueError(
                "Load existing physics/observation/plan inputs through the prepared owner first"
            )
        digest = self.outputs.currentData()
        index = next(
            (
                i
                for i, text in enumerate((session.current_json, *session.history_json))
                if json.loads(text)["sha256"] == digest
            ),
            None,
        )
        if index is None:
            raise ValueError("Selected immutable description is no longer retained")
        self.changed()
        self.details.setPlainText(
            json.dumps(
                dict(
                    frozen_description=json.loads(
                        (session.current_json, *session.history_json)[index]
                    ),
                    association="Independently retained frozen output. Selection does not assert this output belongs to the selected OSC or current geometry; original raw kind/hash and geometry remain authoritative.",
                ),
                indent=2,
            )
        )
        self.shell.prepared.descriptions.setCurrentIndex(index)
        self.shell.prepared.show()

    def inspect_output(self):
        self.open_output()
        self.shell.prepared.request("inspect", description_sha256=self.outputs.currentData())
        self.shell.prepared.tabs.setCurrentIndex(1)

    def handoff(self):
        self.open_output()
        self.shell.prepared.tabs.setCurrentIndex(2)
        self.status.setText(
            "Frozen observations retained; review original parameter definitions and compatibility in the prepared editor. Historical reuse is explicit. Fitting Run/indexed adoption unavailable."
        )

    def browse_inputs(self):
        self.shell.prepared.show()
        self.shell.prepared.tabs.setCurrentIndex(0)
