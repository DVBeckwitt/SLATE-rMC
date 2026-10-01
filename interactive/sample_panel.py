"""Explicit sample-series review, frozen fitting and immutable result inspection."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from mask_state import mask_document
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
from sample_io import observation_id
from sample_state import encoded, sample_session_document


class SamplePanel(QDialog):
    def __init__(self, shell):
        super().__init__(shell)
        self.shell = shell
        self.session = None
        self.epoch = 0
        self._rendering = False
        self._draft_dirty = False
        self._presentation_record = None
        self.setWindowTitle("Sample-only geometry · reviewed OSC series")
        self.resize(980, 800)
        layout = QVBoxLayout(self)
        self.status = QLabel(
            "Load a supported multi-image OSC manifest; sample-only geometry needs no calibrant."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._inputs()
        self._review()
        self._results()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(shell._cancel_current)
        layout.addWidget(self.cancel_button)

    def _page(self, title):
        page = QWidget()
        layout = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, title)
        return layout

    def _button(self, layout, label, action):
        button = QPushButton(label)
        button.clicked.connect(lambda: self.guard(action))
        layout.addWidget(button)
        return button

    def _inputs(self):
        layout = self._page("1 Inputs / controls")
        form = QFormLayout()
        layout.addLayout(form)
        self.manifest = QLineEdit()
        self.manifest.setPlaceholderText("Supported OSC series manifest (.yaml)")
        self.manifest.textEdited.connect(self.changed)
        form.addRow("Series manifest", self.manifest)
        row = QHBoxLayout()
        layout.addLayout(row)
        self._button(row, "Browse manifest", self.browse)
        self.load_button = self._button(row, "Admit complete series / masks", self.load)
        self.fixed = QTextBrowser()
        self.fixed.setMinimumHeight(160)
        layout.addWidget(self.fixed)
        self.parameters = QTableWidget(0, 5)
        self.parameters.setHorizontalHeaderLabels(
            ["Correction / display unit", "Initial / current", "Lower", "Upper", "Scope"]
        )
        self.parameters.setMinimumHeight(365)
        self.parameters.itemChanged.connect(self.changed)
        layout.addWidget(self.parameters)
        form = QFormLayout()
        layout.addLayout(form)
        self.prior = QLineEdit()
        self.prior.textEdited.connect(self.changed)
        form.addRow("Helmert trim prior sigma (degrees)", self.prior)
        hint = QLabel(
            "All corrections are shared. Fixed fields retain their exact stored values. Delta and sample-normal x cannot both fit. Trims fit as one complete zero-sum Helmert span with equal symmetric bounds. Geometry, source, material, detector pixel pitch and commanded angles remain fixed references."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.commit_button = self._button(layout, "Commit displayed controls / review", self.commit)
        self.prepare_button = self._button(
            layout, "Prepare discovery / indexing", lambda: self.request("prepare")
        )

    def _review(self):
        layout = self._page("2 Candidates / review / freeze")
        self.readiness = QLabel("No prepared candidates.")
        self.readiness.setWordWrap(True)
        layout.addWidget(self.readiness)
        self.review = QTableWidget(0, 7)
        self.review.setHorizontalHeaderLabels(
            [
                "Image / observation ID",
                "Keep candidate",
                "Observed column px",
                "Observed row px",
                "Rod / branch / L",
                "Owner admission / reason",
                "Exclusion reason",
            ]
        )
        self.review.setMinimumHeight(420)
        self.review.itemChanged.connect(self.changed)
        layout.addWidget(self.review)
        self.candidates = QTableWidget(0, 5)
        self.candidates.setHorizontalHeaderLabels(
            [
                "Image / candidate",
                "Original column px",
                "Original row px",
                "Detection z",
                "Assignment",
            ]
        )
        self.candidates.setMinimumHeight(220)
        self.candidates.itemSelectionChanged.connect(self.inspect_candidate)
        layout.addWidget(self.candidates)
        self.discovery = QTextBrowser()
        self.discovery.setMinimumHeight(150)
        layout.addWidget(self.discovery)
        self.freeze_button = self._button(
            layout, "Commit review and freeze exact fit-ready pack", self.freeze
        )
        self.export_observations_button = self._button(
            layout, "Export exact frozen observations", lambda: self.export("export_observations")
        )

    def _results(self):
        layout = self._page("3 Fit / results / inspection")
        form = QFormLayout()
        layout.addLayout(form)
        self.name = QLineEdit("Sample geometry candidate")
        self.name.setMaxLength(256)
        form.addRow("New result name", self.name)
        self.fit_button = self._button(layout, "Fit exact frozen observations", self.fit)
        self.results = QComboBox()
        self.results.currentIndexChanged.connect(self.present)
        form.addRow("Inspect retained result", self.results)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.select_button = self._button(row, "Select named result", self.select)
        self._button(row, "Remove inspected history record", self.remove)
        self._button(row, "Import result", self.import_result)
        self._button(row, "Export result", lambda: self.export("export"))
        self._button(row, "Export figure + exact values", lambda: self.export("export_figure"))
        self.summary = QTextBrowser()
        self.summary.setMinimumHeight(200)
        layout.addWidget(self.summary)
        self.comparison = QTableWidget(0, 5)
        self.comparison.setMinimumHeight(300)
        self.comparison.setHorizontalHeaderLabels(
            ["Correction / display unit", "Launch initial", "Current draft", "Fitted", "Scope"]
        )
        layout.addWidget(self.comparison)
        self.points = QTableWidget(0, 7)
        self.points.setHorizontalHeaderLabels(
            [
                "Image / observation ID",
                "Observed column",
                "Observed row",
                "Predicted column",
                "Predicted row",
                "Residual column px",
                "Residual row px",
            ]
        )
        self.points.setMinimumHeight(230)
        self.points.itemSelectionChanged.connect(self.inspect)
        layout.addWidget(self.points)
        self.inspection = QTextBrowser()
        self.inspection.setMinimumHeight(100)
        layout.addWidget(self.inspection)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(7, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(400)
        layout.addWidget(self.canvas)

    def guard(self, action):
        try:
            return action()
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            self.status.setText(str(exc))
            self.refresh()

    def project_masks(self):
        masks = {}
        paths = (
            None
            if self.session is None
            else {v["path"] for v in json.loads(self.session.inputs_json)["images"]}
        )
        for a in self.shell.project.acquisitions:
            path = str(a.source_path.resolve())
            if paths is not None and path not in paths:
                continue
            value = mask_document(a.mask)
            if path in masks and masks[path] != value:
                raise ValueError(
                    "same sample path has conflicting acquisition fitting masks; resolve the mask assignment"
                )
            masks[path] = value
        return masks

    def context(self):
        paths = (
            None
            if self.session is None
            else {v["path"] for v in json.loads(self.session.inputs_json)["images"]}
        )
        signatures = tuple(
            (a.acquisition_id, str(a.source_path.resolve()), a.source_sha256, a.mask)
            for a in self.shell.project.acquisitions
            if paths is None or str(a.source_path.resolve()) in paths
        )
        return (self.shell.project.project_id, self.epoch, signatures, self.manifest.text())

    def inputs_match(self):
        if (
            self.session is None
            or str(Path(self.manifest.text()).resolve())
            != json.loads(self.session.inputs_json)["manifest_path"]
        ):
            return False
        try:
            current = self.project_masks()
        except ValueError:
            return False
        for row in json.loads(self.session.inputs_json)["images"]:
            if any(
                str(a.source_path.resolve()) == row["path"]
                and a.source_sha256 != row["decoded_sha256"]
                for a in self.shell.project.acquisitions
            ):
                return False
            if (
                row["path"] in current
                and current[row["path"]] is not None
                and current[row["path"]] != row["mask"]
            ):
                return False
            if row["path"] in current and current[row["path"]] is None and row["mask"]["spans"]:
                return False
        return True

    def changed(self, *_args):
        if self._rendering:
            return
        self._draft_dirty = True
        self.epoch += 1
        self.shell._supersede_sample()
        self.status.setText("Displayed edits are pending. Commit/refreeze before fitting.")
        self.refresh()

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "OSC series manifest", "", "YAML (*.yaml *.yml)"
        )
        if path:
            self.manifest.setText(path)
            self.changed()

    def load(self):
        # Re-admission intentionally captures the current complete project mask assignments.
        previous = self.session
        self.session = None
        try:
            masks = self.project_masks()
        finally:
            self.session = previous
        self.shell._request_sample(
            "load",
            encoded(
                {
                    "operation": "load",
                    "manifest_path": self.manifest.text(),
                    "project_masks": masks,
                    "previous_session": sample_session_document(previous),
                }
            ).encode(),
            self.context(),
        )

    def request(self, operation, **extra):
        if self.session is None or not self.inputs_match():
            raise ValueError("Admit the current series and fitting masks first")
        if self._draft_dirty:
            raise ValueError(
                "Pending visible edits block work; commit the displayed controls/review first"
            )
        self.shell._request_sample(
            operation,
            encoded(
                {"operation": operation, "session": sample_session_document(self.session), **extra}
            ).encode(),
            self.context(),
        )

    def store(self, session):
        view = replace(self.shell._capture_view(), sample_session=session)
        self.shell._validate_project_admission(self.shell.project, view=view)
        self.session = session
        self.epoch += 1
        self._draft_dirty = False
        self.shell._mark_dirty()
        self.render()

    def _value(self, text, stored, angular):
        displayed = math.degrees(stored) if angular else stored
        if text == format(displayed, ".17g"):
            return stored
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("all displayed sample values must be finite")
        return math.radians(value) if angular else value

    def commit(self):
        if self.session is None or not self.inputs_match():
            raise ValueError("Admit changed inputs/masks before committing a draft")
        controls = json.loads(self.session.controls_json)
        for i, name in enumerate(controls["names"]):
            for j, key in enumerate(("initial", "lower", "upper"), 1):
                controls[key][i] = self._value(
                    self.parameters.item(i, j).text(), controls[key][i], name.endswith("_rad")
                )
        controls["fitted"] = [
            name
            for i, name in enumerate(controls["names"])
            if self.parameters.cellWidget(i, 4).isChecked()
        ]
        controls["trim_prior_sigma_rad"] = self._value(
            self.prior.text(), controls["trim_prior_sigma_rad"], True
        )
        # Lightweight control validation runs on the worker before launching the solver.
        exclusions = []
        for i in range(self.review.rowCount()):
            item = self.review.item(i, 0)
            if self.review.item(i, 1).checkState() != Qt.Checked:
                reason = self.review.item(i, 6).text().strip()
                if not reason:
                    raise ValueError("Each excluded candidate needs an explicit reason")
                exclusions.append((item.data(Qt.UserRole), reason))
        exclusions = tuple(sorted(exclusions))
        changed = exclusions != self.session.exclusions
        self.store(
            replace(
                self.session,
                controls_json=encoded(controls),
                initial_provenance_json=self.session.initial_provenance_json
                if controls["initial"] == json.loads(self.session.controls_json)["initial"]
                else "",
                exclusions=exclusions,
                frozen_json=None if changed else self.session.frozen_json,
                revision=self.session.revision + 1,
            )
        )
        self.status.setText("Draft committed. Refreeze review if observations changed.")

    def freeze(self):
        if self._draft_dirty:
            self.commit()
        self.request("freeze")

    def fit(self):
        if self._draft_dirty or self.session is None or not self.session.frozen_json:
            raise ValueError("Commit pending edits and freeze the reviewed pack before Fit")
        name = self.name.text().strip()
        if not name:
            raise ValueError("Name the candidate result")
        self.request("fit", name=name)

    def ready(self, value):
        previous_id = self.results.currentData()
        self.store(value.session)
        if value.presented_result_id:
            self.results.setCurrentIndex(self.results.findData(value.presented_result_id))
        elif previous_id:
            self.results.setCurrentIndex(self.results.findData(previous_id))
        self.status.setText(
            value.operation.title() + " completed; qualification and selection remain explicit"
        )

    def restore(self, session):
        self.session = session
        self.epoch += 1
        self._draft_dirty = False
        self.render()
        self.status.setText(
            "Sample series restored; review readiness and inspect retained results."
            if session is not None
            else "Load a supported multi-image OSC manifest; sample-only geometry needs no calibrant."
        )

    def refresh(self):
        valid = self.session is not None and self.inputs_match()
        pending = self.shell._active_kind == "sample" or self.shell._pending_sample is not None
        self.cancel_button.setEnabled(pending)
        self.prepare_button.setEnabled(valid and not self._draft_dirty)
        self.freeze_button.setEnabled(valid and self.session.prepared_json is not None)
        self.fit_button.setEnabled(
            valid
            and not self._draft_dirty
            and self.session.frozen_json is not None
            and len(self.session.results_json) < 4
        )
        self.export_observations_button.setEnabled(
            valid and not self._draft_dirty and self.session.frozen_json is not None
        )
        self.commit_button.setEnabled(valid)
        self.select_button.setEnabled(
            self._presentation_record is not None
            and not self._draft_dirty
            and self._presentation_record["launch_sha256"] == self.session.launch_sha256
        )
        if self.session is not None:
            self.readiness.setText(
                "Pending visible edits; Fit blocked"
                if self._draft_dirty
                else "Inputs/mask assignment changed; re-admit and prepare"
                if not valid
                else "Frozen exact reviewed pack: " + json.loads(self.session.frozen_json)["sha256"]
                if self.session.frozen_json
                else "Review not frozen; Fit unavailable"
            )

    def render(self):
        self._rendering = True
        try:
            previous = self.results.currentData()
            self.parameters.setRowCount(0)
            self.review.setRowCount(0)
            self.candidates.setRowCount(0)
            self.results.clear()
            self._presentation_record = None
            if self.session is None:
                self.fixed.setPlainText("No admitted sample series.")
                self.discovery.clear()
                self.present()
                return
            inputs = json.loads(self.session.inputs_json)
            controls = json.loads(self.session.controls_json)
            self.manifest.setText(inputs["manifest_path"])
            self.fixed.setPlainText(
                json.dumps(
                    {"fixed_inputs": inputs, "solver_defaults": controls["solver"]}, indent=2
                )
            )
            self.parameters.setRowCount(len(controls["names"]))
            for i, name in enumerate(controls["names"]):
                angular = name.endswith("_rad")
                label = name.removesuffix("_rad") + " (degrees; stored rad)" if angular else name
                self._item(self.parameters, i, 0, label)
                for j, key in enumerate(("initial", "lower", "upper"), 1):
                    value = controls[key][i]
                    displayed = math.degrees(value) if angular else value
                    self.parameters.setItem(i, j, QTableWidgetItem(format(displayed, ".17g")))
                check = QCheckBox(
                    "Fit shared"
                    if i < 9 or i == 12
                    else "Fit detector calibration"
                    if i < 12
                    else "Fit Helmert contrast"
                )
                check.setChecked(name in controls["fitted"])
                check.toggled.connect(self.changed)
                self.parameters.setCellWidget(i, 4, check)
            self.prior.setText(format(math.degrees(controls["trim_prior_sigma_rad"]), ".17g"))
            if self.session.prepared_json:
                prepared = json.loads(self.session.prepared_json)
                self.discovery.setPlainText(
                    json.dumps(
                        {
                            "availability": prepared["availability"],
                            "manifest_hash": prepared["selection"]["manifest_hash"],
                            "tracks": prepared["selection"]["tracks"],
                        },
                        indent=2,
                    )
                )
                for image in prepared["discoveries"]:
                    decisions = next(
                        v["marker_decisions"]
                        for v in prepared["selection"]["images"]
                        if v["image_id"] == image["image_id"]
                    )
                    for i, peak in enumerate(image["peaks"]):
                        row = self.candidates.rowCount()
                        self.candidates.insertRow(row)
                        assignments = [
                            v
                            for v in decisions
                            if v["observed_column_px"] == peak["column_px"]
                            and v["observed_row_px"] == peak["row_px"]
                        ]
                        self._item(
                            self.candidates,
                            row,
                            0,
                            image["image_id"] + " / " + str(i),
                            encoded(
                                {
                                    "image_id": image["image_id"],
                                    "discovery_hash": image["discovery_hash"],
                                    "candidate_index": i,
                                    "original": peak,
                                    "assignments": assignments,
                                }
                            ),
                        )
                        for j, key in enumerate(("column_px", "row_px", "z_score"), 1):
                            self._item(self.candidates, row, j, format(peak[key], ".17g"))
                        self._item(
                            self.candidates,
                            row,
                            4,
                            " / ".join(v["status"] + ": " + v["reason"] for v in assignments)
                            or "No admitted discrete assignment",
                        )
                self.candidates.resizeColumnsToContents()
                admitted = set(prepared["admitted_observation_ids"])
                if self.session.frozen_json:
                    admitted = {
                        v
                        for image in json.loads(self.session.frozen_json)["observations"]
                        for v in image["observation_ids"]
                    }
                excluded = dict(self.session.exclusions)
                for image in prepared["selection"]["images"]:
                    for decision in image["marker_decisions"]:
                        if decision["observed_column_px"] is None:
                            continue
                        row = self.review.rowCount()
                        self.review.insertRow(row)
                        identity = observation_id(image["image_id"], decision["key"])
                        self._item(
                            self.review, row, 0, image["image_id"] + " / " + identity[:12], identity
                        )
                        keep = QTableWidgetItem()
                        keep.setFlags(
                            Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
                        )
                        keep.setCheckState(Qt.Unchecked if identity in excluded else Qt.Checked)
                        self.review.setItem(row, 1, keep)
                        self._item(
                            self.review, row, 2, format(decision["observed_column_px"], ".17g")
                        )
                        self._item(self.review, row, 3, format(decision["observed_row_px"], ".17g"))
                        self._item(self.review, row, 4, encoded(decision["key"]))
                        self._item(
                            self.review,
                            row,
                            5,
                            ("ADMITTED" if identity in admitted else "NOT ADMITTED")
                            + " · "
                            + decision["status"]
                            + " · "
                            + decision["reason"],
                        )
                        self.review.setItem(row, 6, QTableWidgetItem(excluded.get(identity, "")))
            else:
                self.discovery.setPlainText(
                    "Prepare uses native valid masks plus reviewed exclusions, canonical discovery/indexing and replicated-track admission. Fit never rediscovers or reindexes."
                )
            for text in self.session.results_json:
                record = json.loads(text)
                state = (
                    "selected"
                    if record["result_id"] == self.session.selected_result_id
                    else "candidate"
                )
                if record["launch_sha256"] != self.session.launch_sha256:
                    state = "stale / historical"
                self.results.addItem(
                    record["name"] + " · " + state + " · unqualified", record["result_id"]
                )
            preferred = previous or self.session.selected_result_id
            index = self.results.findData(preferred)
            if index >= 0:
                self.results.setCurrentIndex(index)
            self.parameters.resizeColumnsToContents()
            self.review.resizeColumnsToContents()
            self.present()
        finally:
            self._rendering = False
            self.refresh()

    def _item(self, table, row, column, text, identity=None):
        item = QTableWidgetItem(str(text))
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        if identity is not None:
            item.setData(Qt.UserRole, identity)
        table.setItem(row, column, item)

    def record(self):
        if self.session is None:
            return None
        identity = self.results.currentData()
        return next(
            (
                json.loads(v)
                for v in self.session.results_json
                if json.loads(v)["result_id"] == identity
            ),
            None,
        )

    def present(self, *_args):
        # A whole presentation changes atomically to one immutable record; inspection never
        # consults the combo box again and cannot combine another result's points/table.
        record = self.record()
        self._presentation_record = record
        self.points.setRowCount(0)
        self.comparison.setRowCount(0)
        self.inspection.clear()
        self.figure.clear()
        axes = self.figure.subplots()
        axes.set(xlabel="native column_px", ylabel="native row_px")
        axes.invert_yaxis()
        axes.set_aspect("equal")
        if record is None:
            self.summary.setPlainText(
                "No result. Prepare, review, freeze and explicitly Fit the complete series."
            )
            self.canvas.draw_idle()
            self.refresh()
            return
        summary = {
            k: v for k, v in record.items() if k not in ("launch", "points", "objective_residual")
        }
        fit = record["fit"]
        status_lines = [
            record["name"],
            "Result ID: " + record["result_id"],
            "Dataset / parameter precision: UNQUALIFIED; downstream adoption unavailable.",
            "Solver: unavailable — " + record["unavailable_reason"]
            if fit is None
            else "Solver: "
            + ("terminated successfully" if fit["success"] else "did not converge")
            + " — "
            + fit["message"],
        ]
        if fit is not None:
            status_lines.append(
                f"Data Jacobian: rank {fit['jacobian_rank']}/{len(fit['active_bounds'])}; condition {fit['jacobian_condition']:.8g}; active bounds {sum(fit['active_bounds'])}."
            )
            status_lines.extend(
                f"{v['image_id']}: {v['site_count']} sites; RMS {v['site_rms_px']:.8g} px; maximum {v['site_max_px']:.8g} px; chord RMS {math.degrees(v['chord_angle_rms_rad']):.8g} degrees."
                for v in fit["per_image"]
            )
        status_lines += [
            record["qualification_reason"],
            record["uncertainty_reason"],
            "Recorded checks:\n" + json.dumps(summary, indent=2),
        ]
        self.summary.setPlainText("\n".join(status_lines))
        launch = record["launch"]["controls"]
        current = json.loads(self.session.controls_json)
        fit = record["fit"]
        historical_inputs = record["launch"]["inputs"]
        current_inputs = json.loads(self.session.inputs_json)

        def setup_identity(inputs):
            keys = (
                "phase_id",
                "fixed_instrument",
                "nominal_mean_direction_lab",
                "nominal_mean_wavelength_A",
                "source_policy",
                "incidence_axis_index",
                "configuration_path",
                "cif_path",
            )
            paths = {inputs["configuration_path"], inputs["cif_path"]}
            return [
                {k: inputs[k] for k in keys},
                sorted((v["path"], v["sha256"]) for v in inputs["files"] if v["path"] in paths),
            ]

        def ordered_images(inputs):
            # The canonical fit constructs its Helmert basis in sorted image-ID order.
            return sorted(
                (v["image_id"], v["decoded_sha256"], v["axis_rotation_angles_deg"])
                for v in inputs["images"]
            )

        compatible_setup = setup_identity(historical_inputs) == setup_identity(current_inputs)
        compatible_trims = compatible_setup and ordered_images(historical_inputs) == ordered_images(
            current_inputs
        )
        current_values = dict(zip(current["names"], current["initial"], strict=True))
        fitted = (
            {}
            if fit is None
            else {
                **fit["corrections"],
                **fit["detector_calibration_corrections"],
                "incidence_angle_delta_rad": fit["incidence_angle_delta_rad"],
            }
        )
        trim_names = [n for n in launch["names"] if n.startswith("incidence_angle_trim_helmert_")]
        if fit is not None:
            contrasts = fit["incidence_angle_trim_contrast_rad"]
            if contrasts:
                fitted.update(zip(trim_names, contrasts, strict=True))
            else:
                fitted.update(
                    (n, v)
                    for n, v in zip(launch["names"], launch["initial"], strict=True)
                    if n in trim_names and n not in launch["fitted"]
                )
        self.comparison.setRowCount(len(launch["names"]))
        for i, name in enumerate(launch["names"]):
            angular = name.endswith("_rad")
            trim = name in trim_names
            compatible = compatible_trims if trim else compatible_setup
            self._item(
                self.comparison,
                i,
                0,
                name.removesuffix("_rad") + " (degrees; stored rad)" if angular else name,
            )
            values = (
                launch["initial"][i],
                current_values.get(name) if compatible else None,
                fitted.get(name),
            )
            for j, value in enumerate(values, 1):
                text = (
                    "unavailable / not comparable"
                    if j == 2 and value is None
                    else "unavailable"
                    if value is None
                    else format(math.degrees(value) if angular else value, ".17g")
                )
                self._item(self.comparison, i, j, text)
            scope = "shared fitted" if name in launch["fitted"] else "fixed reference"
            if trim:
                scope += "; ordered images: " + ", ".join(
                    v[0] for v in ordered_images(historical_inputs)
                )
            self._item(self.comparison, i, 4, scope)
        self.points.setRowCount(len(record["points"]))
        for i, point in enumerate(record["points"]):
            self._item(
                self.points,
                i,
                0,
                point["image_id"] + " / " + point["observation_id"][:12],
                (record["result_id"], i),
            )
            for j, value in enumerate(
                point["observed_px"] + point["predicted_px"] + point["residual_px"], 1
            ):
                self._item(self.points, i, j, format(value, ".17g"))
        for image in record["launch"]["inputs"]["images"]:
            points = [v for v in record["points"] if v["image_id"] == image["image_id"]]
            if points:
                observed = np.asarray([v["observed_px"] for v in points])
                predicted = np.asarray([v["predicted_px"] for v in points])
                axes.scatter(
                    observed[:, 0], observed[:, 1], s=14, label=image["image_id"] + " observed"
                )
                axes.scatter(
                    predicted[:, 0],
                    predicted[:, 1],
                    s=22,
                    marker="+",
                    label=image["image_id"] + " predicted",
                )
        if record["points"]:
            axes.legend(fontsize="small")
        axes.set_title(record["name"] + " · " + record["result_id"] + " · unqualified")
        self.points.resizeColumnsToContents()
        self.comparison.resizeColumnsToContents()
        self.canvas.draw_idle()
        self.refresh()

    def inspect_candidate(self):
        row = self.candidates.currentRow()
        if row >= 0 and self.candidates.item(row, 0) is not None:
            self.discovery.setPlainText(
                json.dumps(json.loads(self.candidates.item(row, 0).data(Qt.UserRole)), indent=2)
            )

    def inspect(self):
        record = self._presentation_record
        row = self.points.currentRow()
        if record is None or not 0 <= row < len(record["points"]):
            return
        if self.points.item(row, 0).data(Qt.UserRole) != (record["result_id"], row):
            self.inspection.setPlainText("Stale row binding; choose the result again")
            return
        self.inspection.setPlainText(
            json.dumps({"result_id": record["result_id"], "point": record["points"][row]}, indent=2)
        )

    def select(self):
        record = self._presentation_record
        if (
            record is None
            or self._draft_dirty
            or not self.inputs_match()
            or record["launch_sha256"] != self.session.launch_sha256
        ):
            raise ValueError("Only a result of the current committed frozen launch can be selected")
        self.store(replace(self.session, selected_result_id=record["result_id"]))
        self.status.setText(
            "Selected for inspection only. Dataset/precision qualification and downstream geometry adoption remain unavailable."
        )

    def remove(self):
        record = self._presentation_record
        if record is None:
            return
        if self._draft_dirty:
            raise ValueError("Commit pending draft edits before changing history")
        identity = record["result_id"]
        self.store(
            replace(
                self.session,
                results_json=tuple(
                    v for v in self.session.results_json if json.loads(v)["result_id"] != identity
                ),
                selected_result_id=None
                if self.session.selected_result_id == identity
                else self.session.selected_result_id,
            )
        )

    def import_result(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import hash-bound sample record", "", "JSON (*.json)"
        )
        if path:
            self.request("import", path=path)

    def export(self, operation):
        record = self._presentation_record
        if operation != "export_observations" and record is None:
            raise ValueError("Inspect a retained result first")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to a new external destination",
            "",
            "PNG (*.png)" if operation == "export_figure" else "JSON (*.json)",
        )
        if path:
            self.request(operation, path=path, record=None if record is None else encoded(record))
