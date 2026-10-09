"""Snapshot state, recorded provenance and strictly compatible inspection."""

import json

import yaml
from detector_panel import DetectorTextureView
from project_state import validate_display_limits
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from simulation_contrast import ColorLegend, display_limits
from simulation_display import select_display_level
from simulation_state import SimulationDraft
from simulation_widgets import NumberEdit


def snapshot_state(frame, current_draft, *, held=False, compact=False):
    if frame is None:
        return (
            "No detector snapshot · Run or Project > Show saved image"
            if compact
            else "No detector snapshot; Run/update or Show saved image"
        )
    manifest = json.loads(frame.manifest)
    relation = "Inputs match draft" if frame.draft == current_draft else "Historical inputs"
    precision = "Quantitative snapshot" if frame.quantitative else "Presentation preview"
    if frame.image is None:
        precision += " · no detector output"
    complete = manifest.get("execution_complete", manifest.get("integration_complete"))
    progress = (
        "Complete"
        if complete is True
        else "Partial"
        if complete is False
        else "Completion unrecorded"
    )
    if isinstance(frame.draft, SimulationDraft) and frame.draft.route == "monte_carlo":
        progress += f" · draws {frame.draw_prefix}/{frame.draft.draw_count} per source"
    else:
        progress += f" · recorded batches/prefix {frame.draw_prefix}"
    qualification = manifest.get("qualification", "Qualification unrecorded")
    if compact:
        declared = str(qualification).split(";", 1)[0].strip().lower()
        qualification = (
            "Unqualified"
            if declared in ("nominal", "unqualified")
            else "Qualified"
            if declared == "qualified"
            else "Qualification unknown"
        )
        progress = (
            "Complete"
            if complete is True
            else "Partial"
            if complete is False
            else "Completion unknown"
        )
        if isinstance(frame.draft, SimulationDraft) and frame.draft.route == "monte_carlo":
            progress += f" · draws {frame.draw_prefix}/{frame.draft.draw_count}"
        elif complete is False:
            progress += f" · batches {frame.draw_prefix}"
        return (
            ("Current" if frame.draft == current_draft else "Historical")
            + (" · Quantitative" if frame.quantitative else " · Preview")
            + (" · no detector output" if frame.image is None else "")
            + f" · {progress} · {qualification}"
            + (" · Held" if held else "")
        )
    return (
        f"{relation} · {precision} · {progress}"
        + (" · Held" if held else "")
        + f"\n{frame.measure}\n{qualification}"
    )


def comparison_identity(frame):
    """No inferred alignment: require literal declared geometry and support."""
    if not frame.quantitative or frame.image is None:
        raise ValueError("Compare requires a quantitative detector snapshot; hold one first")
    if not isinstance(frame.draft, SimulationDraft):
        raise ValueError(
            "Native saved-result comparison has no admitted detector geometry identity in this build"
        )
    if frame.draft.route not in ("monte_carlo", "pixel_centers"):
        raise ValueError(
            "Compare supports native detector cells; macrobin/auxiliary alignment is unavailable"
        )
    instrument = yaml.safe_load(frame.draft.yaml_text)["instrument"]
    # The entire declaration binds detector pose/frame, sample support and ordered motors.
    return (
        frame.measure,
        tuple(frame.image.shape),
        json.dumps(instrument, sort_keys=True, allow_nan=False),
    )


class SnapshotInspection(QWidget):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.simulator = simulator
        self.reference = None
        self.comparison_dialog = None
        self.comparison_views = ()
        self.compared_frame = None
        self.comparison_limits = None
        self.comparison_controls = None
        self.comparison_error = None
        self.comparison_bin = None
        self.comparison_mode = None
        self.comparison_note = None
        self.comparison_legend = None
        self.comparison_low = self.comparison_high = None
        self._context_attempt = 0
        self.comparison_timer = QTimer(self)
        self.comparison_timer.setSingleShot(True)
        self.comparison_timer.timeout.connect(self.populate_comparison)
        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        actions = QHBoxLayout()
        simulator.inspect_button.setText("&Hold")
        simulator.resume_button.setText("&Follow")
        for button in (
            simulator.inspect_button,
            simulator.resume_button,
            simulator.profile_controls_button,
            simulator.export_button,
        ):
            actions.addWidget(button)
        simulator.profile_controls_button.setText("Profiles / ROI")
        simulator.export_button.setText("Export snapshot")
        self.compare_button = QPushButton("Compare saved result")
        self.compare_button.clicked.connect(self.choose_reference)
        actions.addWidget(self.compare_button)
        body.addLayout(actions)
        self.availability = QLabel()
        self.availability.setWordWrap(True)
        body.addWidget(self.availability)
        self.provenance = QPlainTextEdit()
        self.provenance.setReadOnly(True)
        self.provenance.setAccessibleName("Displayed snapshot recorded parameters and provenance")
        body.addWidget(self.provenance, 1)
        row = QHBoxLayout()
        self.launch_button = QPushButton("Export recorded launch")
        self.launch_button.clicked.connect(self.export_launch)
        row.addWidget(self.launch_button)
        experiments = QPushButton("Compare acquisitions")
        experiments.setToolTip(
            "Open the existing compatible-count acquisition comparison; no model residual admission"
        )
        experiments.clicked.connect(self.open_acquisitions)
        row.addWidget(experiments)
        body.addLayout(row)
        self.sync()

    def open_acquisitions(self):
        shell = self.simulator.shell
        shell.workspaces.setCurrentIndex(0)
        shell.scene_tabs.setCurrentIndex(2)

    def sync(self):
        s = self.simulator
        frame = s.frame
        busy = s.shell._active_kind == "simulation" or s.shell._pending_simulation is not None
        quantitative = frame is not None and frame.quantitative and frame.image is not None
        reason = "Exact profiles/ROI and export use this quantitative snapshot."
        if not quantitative:
            reason = "Exact profiles/ROI/export unavailable until a quantitative snapshot is received; preview readout is presentation only."
        self.availability.setText(reason)
        self.availability.setToolTip(
            reason
            + " Comparison requires the same declared measure, detector geometry/frame and sample support; no alignment or normalization."
        )
        self.launch_button.setEnabled(frame is not None and not busy)
        self.compare_button.setEnabled(quantitative and not busy)
        if quantitative:
            try:
                comparison_identity(frame)
            except ValueError as exc:
                self.compare_button.setEnabled(False)
                self.availability.setText(reason + " " + str(exc))
                self.compare_button.setToolTip(str(exc))
        if self.comparison_error is not None:
            self.availability.setText(self.comparison_error)
        if frame is None:
            self.provenance.setPlainText(
                "No displayed result. Load a draft or reopen its saved image."
            )
            return
        manifest = json.loads(frame.manifest)
        draft = frame.draft
        if isinstance(draft, SimulationDraft):
            config = yaml.safe_load(draft.yaml_text)
            parameters = {
                k: config[k]
                for k in ("source", "instrument", "mosaic", "structure_factor")
                if k in config
            }
            budget = f"{manifest.get('source_state_count', 'unrecorded')} source states; {draft.draw_count} draws/source; seed {draft.detector_seed}; {draft.route}/{draft.position_mode}"
        else:
            parameters = json.loads(draft.physics_json)
            budget = f"{draft.coherent_repeats} repeats; {draft.bin_size_px} native px integration rectangle"
        self.provenance.setPlainText(
            snapshot_state(frame, s.active_draft, held=s.hold)
            + f"\nRun {frame.run_id}; draft {draft.draft_id} revision {draft.revision}\nBackend {frame.backend}\nBudget {budget}\nRecorded launch parameters (declared units):\n"
            + yaml.safe_dump(parameters, sort_keys=False)
            + "\nFull recorded identity/manifest:\n"
            + json.dumps(manifest, indent=2)
        )

    def choose_reference(self):
        try:
            comparison_identity(self.simulator.frame)
        except (ValueError, AttributeError) as exc:
            self.availability.setText(str(exc))
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Compare exact saved simulation from project", "", "SLATE project (*.slate.json)"
        )
        if path:
            self.simulator.shell._request_simulation(
                "compare_reopen", json.dumps({"path": path}).encode()
            )

    def admit_reference(self, frame):
        try:
            if comparison_identity(self.simulator.frame) != comparison_identity(frame):
                raise ValueError(
                    "Compare rejected: measure, detector geometry/frame or sample support differs"
                )
        except (ValueError, AttributeError) as exc:
            self.comparison_error = str(exc)
            self.simulator.status.setText(str(exc))
            self.availability.setText(str(exc))
            return
        common_bins = {level.bin_size for level in self.simulator.frame.display_levels} & {
            level.bin_size for level in frame.display_levels
        }
        if not common_bins:
            self.availability.setText("Comparison unavailable: no common prepared display bins")
            return
        self.clear_comparison()
        self.reference = frame
        self.compared_frame = self.simulator.frame
        dialog = QDialog(self)
        self.comparison_dialog = dialog
        dialog.setWindowTitle("Compatible saved snapshots — shared display limits, raw values")
        dialog.resize(1120, 680)
        body = QVBoxLayout(dialog)
        self.comparison_note = QLabel()
        self.comparison_note.setWordWrap(True)
        body.addWidget(self.comparison_note)
        source = self.simulator.detector.view
        self.comparison_source_bin = (
            None if source.active_display_level is None else source.active_display_level.bin_size
        )
        self.comparison_limits = source.low_value, source.high_value, source.contrast_mode
        self.comparison_controls = QWidget()
        controls = QHBoxLayout(self.comparison_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        self.comparison_bin = QComboBox()
        self.comparison_bin.setAccessibleName("Shared comparison display bin")
        self.comparison_bin.setToolTip(
            "Both panes use this fixed sum bin across resize/zoom. Changing bins uses shared Auto 99% of the displayed snapshot."
        )
        for size in sorted(common_bins):
            self.comparison_bin.addItem(f"{size} x {size} sum", size)
        self.comparison_mode = QComboBox()
        self.comparison_mode.setAccessibleName("Shared comparison contrast mode")
        for label, mode in (
            ("Linear", "linear"),
            ("Positive log", "positive_log"),
            ("Signed", "signed"),
        ):
            self.comparison_mode.addItem(label, mode)
        controls.addWidget(self.comparison_bin)
        controls.addWidget(self.comparison_mode)
        self.comparison_low, self.comparison_high = NumberEdit(), NumberEdit()
        for label, field in (("Low", self.comparison_low), ("Upper", self.comparison_high)):
            caption = QLabel(label)
            caption.setBuddy(field)
            field.setAccessibleName(f"Shared comparison {label.lower()} display limit")
            controls.addWidget(caption)
            controls.addWidget(field, 1)
        for label, action in (
            ("Apply", lambda: self.apply_comparison_limits(manual=True)),
            ("Auto 99%", self.apply_comparison_limits),
            ("Full range", lambda: self.apply_comparison_limits(full=True)),
            ("Fit both", self.fit_comparison),
            ("Details", self.show_comparison_details),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, a=action: a())
            controls.addWidget(button)
        body.addWidget(self.comparison_controls)
        self.comparison_controls.setEnabled(False)
        split = QSplitter()
        views = []
        for label, current in (("Displayed", self.compared_frame), ("Reference", frame)):
            pane = QWidget()
            column = QVBoxLayout(pane)
            caption = QLabel(f"{label} snapshot · recorded prefix {current.draw_prefix}")
            caption.setToolTip(
                f"Run {current.run_id}; draft {current.draft.draft_id}, revision {current.draft.revision}"
            )
            caption.setWordWrap(True)
            column.addWidget(caption)
            detector = DetectorTextureView()
            column.addWidget(detector, 1)
            split.addWidget(pane)
            views.append(detector)
        self.comparison_views = tuple(views)
        body.addWidget(split, 1)
        self.comparison_legend = ColorLegend(dialog)
        body.addWidget(self.comparison_legend)
        self.comparison_bin.currentIndexChanged.connect(self.change_comparison_bin)
        self.comparison_mode.currentIndexChanged.connect(self.apply_comparison_limits)
        dialog.finished.connect(self.clear_comparison)
        dialog.show()
        self._context_attempt = 0
        self.comparison_timer.start(0)
        self.availability.setText(
            "Compatible frozen snapshots opened with shared display-sum bins and limits. Counts-to-model residuals remain unavailable."
        )

    def clear_comparison(self, *_):
        self.comparison_timer.stop()
        dialog = self.comparison_dialog
        self.reference = self.compared_frame = self.comparison_dialog = None
        self.comparison_views = ()
        self.comparison_controls = self.comparison_bin = self.comparison_mode = None
        self.comparison_note = self.comparison_legend = None
        self.comparison_low = self.comparison_high = self.comparison_limits = None
        self.comparison_source_bin = None
        self.comparison_error = None
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()

    def populate_comparison(self):
        if self.comparison_dialog is None:
            return
        if any(view.max_texture_axis is None for view in self.comparison_views):
            self._context_attempt += 1
            if self._context_attempt < 8:
                self.comparison_timer.start(25)
            else:
                self.availability.setText(
                    "Comparison display context unavailable; reopen the inspection view"
                )
                self.clear_comparison()
            return
        try:
            for view, frame in zip(
                self.comparison_views, (self.compared_frame, self.reference), strict=True
            ):
                view.set_prepared_image(
                    frame.image,
                    frame.display,
                    frame.low,
                    frame.high,
                    frame.maximum,
                    frame.min_positive,
                )
            levels = tuple(
                level
                for level in self.compared_frame.display_levels
                if self.comparison_bin.findData(level.bin_size) >= 0
            )
            size = max(
                select_display_level(
                    levels,
                    self.compared_frame.image.shape,
                    view._rect().width() * view.devicePixelRatioF(),
                    view._rect().height() * view.devicePixelRatioF(),
                ).bin_size
                for view in self.comparison_views
            )
            self.comparison_bin.blockSignals(True)
            self.comparison_bin.setCurrentIndex(self.comparison_bin.findData(size))
            self.comparison_bin.blockSignals(False)
            recorded_limits = self.comparison_limits
            self.comparison_mode.blockSignals(True)
            self.comparison_mode.setCurrentIndex(self.comparison_mode.findData(recorded_limits[2]))
            self.comparison_mode.blockSignals(False)
            self.change_comparison_bin()
            if self.comparison_source_bin == size:
                self.set_comparison_limits(
                    *recorded_limits,
                    "copied from displayed snapshot's same-bin contrast",
                )
            self.comparison_controls.setEnabled(True)
        except ValueError as exc:
            self.availability.setText(f"Comparison display unavailable: {exc}")
            self.clear_comparison()

    def change_comparison_bin(self, *_):
        if not self.comparison_views or any(view.image is None for view in self.comparison_views):
            return
        size = self.comparison_bin.currentData()
        for view, frame in zip(
            self.comparison_views, (self.compared_frame, self.reference), strict=True
        ):
            level = next(level for level in frame.display_levels if level.bin_size == size)
            view.set_display_levels((level,))
        self.apply_comparison_limits()

    def apply_comparison_limits(self, *_, manual=False, full=False):
        if not self.comparison_views or self.comparison_views[0].active_display_level is None:
            return
        mode = self.comparison_mode.currentData()
        try:
            low, high = (
                (float(self.comparison_low.exact_text()), float(self.comparison_high.exact_text()))
                if manual
                else display_limits(self.comparison_views[0].active_display_level, mode, full=full)
            )
            self.set_comparison_limits(
                low,
                high,
                mode,
                "manual shared limits"
                if manual
                else "displayed snapshot full range"
                if full
                else "displayed snapshot Auto 99%",
            )
        except ValueError as exc:
            self.comparison_note.setText(f"Display error: {exc}; choose valid limits or Auto 99%")

    def set_comparison_limits(self, low, high, mode, anchor):
        validate_display_limits(low, high, mode)
        for view in self.comparison_views:
            view.set_levels(low, high, mode=mode)
        self.comparison_limits = low, high, mode
        self.comparison_low.setText(repr(low))
        self.comparison_high.setText(repr(high))
        legend = self.comparison_legend
        legend.low, legend.high, legend.mode = low, high, mode
        legend.update()
        size = self.comparison_bin.currentData()
        unit = (
            "angstrom^2/pixel^2 display density"
            if self.compared_frame.draft.route == "pixel_centers"
            else "angstrom^2 simulation mass"
        )
        self.comparison_note.setText(
            f"{self.compared_frame.measure}\nShared {mode} [{low:.7g}, {high:.7g}] in sums of {size} x {size} native display cells ({unit}; edge bins may be smaller).\n"
            f"Binning fixed across resize/zoom; scale anchor: {anchor}. No per-image normalization. Raw native arrays/identities unchanged."
        )

    def show_comparison_details(self):
        if self.comparison_dialog is None:
            return
        dialog = QDialog(self.comparison_dialog)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setWindowTitle("Frozen comparison identities and display details")
        dialog.resize(720, 480)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(
            json.dumps(
                {
                    "displayed_identity": {
                        "run_id": str(self.compared_frame.run_id),
                        "draft_id": str(self.compared_frame.draft.draft_id),
                        "revision": self.compared_frame.draft.revision,
                    },
                    "reference_identity": {
                        "run_id": str(self.reference.run_id),
                        "draft_id": str(self.reference.draft.draft_id),
                        "revision": self.reference.draft.revision,
                    },
                    "displayed_snapshot": json.loads(self.compared_frame.manifest),
                    "reference_snapshot": json.loads(self.reference.manifest),
                    "shared_display_limits": self.comparison_limits,
                    "shared_sum_bin": self.comparison_bin.currentData(),
                    "display_policy": "Fixed common sums; edge bins may be partial; no per-image normalization. Exact native data is unchanged.",
                },
                indent=2,
            )
        )
        layout = QVBoxLayout(dialog)
        layout.addWidget(text)
        dialog.show()

    def fit_comparison(self):
        for view in self.comparison_views:
            view.fit_image()

    def export_launch(self):
        frame = self.simulator.frame
        if frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export displayed result's recorded launch",
            "recorded-launch.json",
            "JSON (*.json)",
        )
        if not path or self.simulator.frame is not frame:
            return
        draft = frame.draft
        protected = (
            [str(draft.configuration_path), str(draft.cif_path)]
            if isinstance(draft, SimulationDraft)
            else []
        )
        if self.simulator.shell._project_path is not None:
            protected.append(str(self.simulator.shell._project_path))
        self.simulator.shell._request_simulation(
            "export_launch",
            json.dumps(
                {"path": path, "manifest": json.loads(frame.manifest), "protected": protected}
            ).encode(),
        )
