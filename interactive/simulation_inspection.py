"""Snapshot state, recorded provenance and strictly compatible inspection."""

import json

import yaml
from detector_panel import DetectorTextureView
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
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
from simulation_state import SimulationDraft


def snapshot_state(frame, current_draft, *, held=False):
    if frame is None:
        return "No detector snapshot; Run/update or Show saved image"
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
        self._context_attempt = 0
        self.comparison_timer = QTimer(self)
        self.comparison_timer.setSingleShot(True)
        self.comparison_timer.timeout.connect(self.populate_comparison)
        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        actions = QHBoxLayout()
        simulator.inspect_button.setText("&Hold snapshot")
        simulator.resume_button.setText("&Follow")
        for button in (
            simulator.inspect_button,
            simulator.resume_button,
            simulator.profile_controls_button,
            simulator.export_button,
        ):
            actions.addWidget(button)
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
        self.availability.setText(
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
            self.availability.setText(str(exc))
            self.simulator.status.setText(str(exc))
            return
        self.clear_comparison()
        self.reference = frame
        self.compared_frame = self.simulator.frame
        dialog = QDialog(self)
        self.comparison_dialog = dialog
        dialog.setWindowTitle("Compatible saved snapshots — shared display limits, raw values")
        dialog.resize(1120, 680)
        body = QVBoxLayout(dialog)
        view = self.simulator.detector.view
        native = view.active_display_level is None or view.active_display_level.bin_size == 1
        self.comparison_limits = (
            (view.low_value, view.high_value, view.contrast_mode)
            if native
            else (self.compared_frame.low, self.compared_frame.high, "linear")
        )
        low, high, mode = self.comparison_limits
        body.addWidget(
            QLabel(
                f"{frame.measure}\nShared {mode} native-cell limits [{low:.7g}, {high:.7g}]; "
                + (
                    "copied from current native-cell display"
                    if native
                    else "native full range of displayed snapshot; aggregated display limits not reused"
                )
                + "; no per-image normalization. Frozen comparison of displayed identities."
            )
        )
        split = QSplitter()
        views = []
        for label, current in (("Displayed", self.compared_frame), ("Reference", frame)):
            pane = QWidget()
            column = QVBoxLayout(pane)
            caption = QLabel(f"{label}: {current.run_id}; prefix {current.draw_prefix}")
            caption.setWordWrap(True)
            column.addWidget(caption)
            detector = DetectorTextureView()
            column.addWidget(detector, 1)
            split.addWidget(pane)
            views.append(detector)
        self.comparison_views = tuple(views)
        body.addWidget(split, 1)
        dialog.finished.connect(self.clear_comparison)
        dialog.show()
        self._context_attempt = 0
        self.comparison_timer.start(0)
        self.availability.setText(
            "Compatible frozen snapshots opened with shared native-cell display limits. Counts-to-model residuals remain unavailable."
        )

    def clear_comparison(self, *_):
        self.comparison_timer.stop()
        dialog = self.comparison_dialog
        self.reference = self.compared_frame = self.comparison_dialog = None
        self.comparison_views = ()
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
                low, high, mode = self.comparison_limits
                view.set_levels(low, high, mode=mode)
        except ValueError as exc:
            self.availability.setText(f"Comparison display unavailable: {exc}")
            self.clear_comparison()

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
