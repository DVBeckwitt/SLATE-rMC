"""Native desktop shell for SLATE-rMC; launch with ``python interactive/slate_app.py``."""

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from detector_panel import DetectorPanel
from job_lifecycle import (
    MAX_RESULT_BYTES,
    JobIdentity,
    JobOwner,
    JobRequest,
    JobState,
    JobSummary,
    Revisions,
)
from osc_import import AXIS_LIMIT, PreparedOsc, prepare_osc
from project_state import Acquisition, Project
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


class StatusView(QFrame):
    """A visible empty, loading or error state for an unpopulated workspace."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusView")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(12)
        layout.addStretch()
        self.eyebrow = QLabel()
        self.eyebrow.setObjectName("statusEyebrow")
        self.title = QLabel()
        self.title.setObjectName("statusTitle")
        self.detail = QLabel()
        self.detail.setObjectName("statusDetail")
        self.detail.setWordWrap(True)
        for widget in (self.eyebrow, self.title, self.detail):
            layout.addWidget(widget)
        layout.addStretch()

    def set_state(
        self, kind: Literal["empty", "loading", "error"], title: str, detail: str
    ) -> None:
        if kind not in ("empty", "loading", "error"):
            raise ValueError(f"Unknown status: {kind}")
        self.setProperty("state", kind)
        self.eyebrow.setText(
            {"empty": "GET STARTED", "loading": "LOADING", "error": "NEEDS ATTENTION"}[kind]
        )
        self.title.setText(title)
        self.detail.setText(detail)
        self.style().unpolish(self)
        self.style().polish(self)


class ShellWindow(QMainWindow):
    def __init__(self, project: Project | None = None) -> None:
        super().__init__()
        self.project = project if project is not None else Project.create()
        self.selected_acquisition_id: UUID | None = None
        self._visible_acquisition_id: UUID | None = None
        self._visible_details = ""
        self._obsolete_pending = False
        self.setAcceptDrops(True)
        self.setWindowTitle(f"SLATE · {self.project.name}")
        self.resize(1280, 800)
        self.setMinimumSize(820, 540)

        tabs = QTabWidget()
        tabs.setObjectName("workspaces")
        tabs.addTab(self._build_experiments(), "Fit experiments")
        tabs.addTab(self._build_simulator(), "Simulator")
        self.setCentralWidget(tabs)
        self.workspaces = tabs
        self.jobs = JobOwner(self)
        self.jobs.state_changed.connect(self._job_state_changed)
        self.jobs.progress_changed.connect(self._job_progress)
        self.jobs.result_ready.connect(self._import_ready)
        self.jobs.drained.connect(self.close)
        self.cancel_button.clicked.connect(self.jobs.cancel)
        self.import_button.clicked.connect(self._choose_import)
        self.statusBar().showMessage("Local project · Not saved")
        self.refresh_project()
        self._max_import_axis = 0
        try:
            frame = self.detector_panel.view.grabFramebuffer()
            texture_axis = self.detector_panel.view.max_texture_axis
            if frame.isNull() or texture_axis is None:
                raise RuntimeError("Detector OpenGL context is unavailable")
            self._max_import_axis = min(AXIS_LIMIT, texture_axis)
        except (RuntimeError, ValueError) as exc:
            self.import_button.setEnabled(False)
            self.import_button.setToolTip(str(exc))
            self._show_state("error", "Detector display unavailable", str(exc))

    def _build_experiments(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(16)
        heading = QLabel("Fit experiments")
        heading.setObjectName("workspaceTitle")
        outer.addWidget(heading)
        subtitle = QLabel("Inspect acquisitions and prepare supported fitting stages.")
        subtitle.setObjectName("mutedText")
        outer.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        browser = QFrame()
        browser.setObjectName("sidePanel")
        browser_layout = QVBoxLayout(browser)
        browser_layout.setContentsMargins(18, 18, 18, 18)
        browser_layout.setSpacing(12)
        browser_layout.addWidget(QLabel("PROJECT / ACQUISITIONS"))
        self.project_tree = QTreeWidget()
        self.project_tree.setObjectName("projectTree")
        self.project_tree.setHeaderHidden(True)
        self.project_tree.itemSelectionChanged.connect(self._selection_changed)
        browser_layout.addWidget(self.project_tree, 1)
        self.acquisition_count = QLabel("0 acquisitions")
        self.acquisition_count.setObjectName("mutedText")
        browser_layout.addWidget(self.acquisition_count)
        splitter.addWidget(browser)

        center = QFrame()
        center.setObjectName("centerPanel")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(20, 20, 20, 20)
        center_layout.setSpacing(14)
        center_layout.addWidget(QLabel("Detector view"))
        self.experiment_status = StatusView()
        self.experiment_status.set_state(
            "empty",
            "No acquisitions yet",
            "Choose one OSC or OSC.GZ file, or drop it onto this window.",
        )
        self.detector_panel = DetectorPanel()
        self.detector_notice = QLabel()
        self.detector_notice.setWordWrap(True)
        detector_page = QWidget()
        detector_layout = QVBoxLayout(detector_page)
        detector_layout.setContentsMargins(0, 0, 0, 0)
        detector_layout.addWidget(self.detector_notice)
        detector_layout.addWidget(self.detector_panel, 1)
        self.detector_stack = QStackedWidget()
        self.detector_stack.addWidget(self.experiment_status)
        self.detector_stack.addWidget(detector_page)
        center_layout.addWidget(self.detector_stack, 1)
        self.import_button = QPushButton("Import OSC")
        controls = QHBoxLayout()
        controls.addWidget(self.import_button)
        self.cancel_button = QPushButton("Cancel operation")
        self.cancel_button.setEnabled(False)
        controls.addWidget(self.cancel_button)
        controls.addStretch()
        center_layout.addLayout(controls)
        splitter.addWidget(center)

        inspector = QFrame()
        inspector.setObjectName("sidePanel")
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(18, 18, 18, 18)
        inspector_layout.setSpacing(12)
        inspector_layout.addWidget(QLabel("INSPECTOR"))
        self.selection_label = QLabel("No acquisition selected")
        self.selection_label.setObjectName("mutedText")
        self.selection_label.setWordWrap(True)
        inspector_layout.addWidget(self.selection_label)
        inspector_layout.addStretch()
        project_limit = QLabel(
            "Angles and calibration are unknown until entered explicitly. Project saving is unavailable in this version."
        )
        project_limit.setWordWrap(True)
        project_limit.setObjectName("mutedText")
        inspector_layout.addWidget(project_limit)
        splitter.addWidget(inspector)
        splitter.setSizes([250, 680, 250])
        return page

    def _build_simulator(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        heading = QLabel("Simulator")
        heading.setObjectName("workspaceTitle")
        layout.addWidget(heading)
        subtitle = QLabel("An independent workspace for supported model configurations.")
        subtitle.setObjectName("mutedText")
        layout.addWidget(subtitle)
        status = StatusView()
        status.set_state(
            "empty",
            "No simulation draft",
            "Configuration editing and simulation runs are unavailable in this version.",
        )
        layout.addWidget(status, 1)
        return page

    def refresh_project(self) -> None:
        selected_id = self.selected_acquisition_id
        self.project_tree.blockSignals(True)
        self.project_tree.clear()
        root = QTreeWidgetItem([self.project.name])
        root.setData(0, Qt.ItemDataRole.UserRole, None)
        self.project_tree.addTopLevelItem(root)
        selected_item = None
        for acquisition in self.project.acquisitions:
            item = QTreeWidgetItem([acquisition.name])
            item.setData(0, Qt.ItemDataRole.UserRole, str(acquisition.acquisition_id))
            root.addChild(item)
            if acquisition.acquisition_id == selected_id:
                selected_item = item
        root.setExpanded(True)
        if selected_item is not None:
            self.project_tree.setCurrentItem(selected_item)
            self._update_selection_label()
        else:
            self.selected_acquisition_id = None
            self.selection_label.setText("No acquisition selected")
        self.acquisition_count.setText(f"{len(self.project.acquisitions)} acquisitions")
        if self.project.acquisitions:
            self._show_state("empty", "No image open", "Choose an OSC acquisition to view it.")
        else:
            self._show_state(
                "empty", "No acquisitions yet", "Choose one OSC or OSC.GZ file, or drop it here."
            )
        self.project_tree.blockSignals(False)

    def _update_selection_label(self) -> None:
        acquisition = next(
            (
                item
                for item in self.project.acquisitions
                if item.acquisition_id == self.selected_acquisition_id
            ),
            None,
        )
        if acquisition is None:
            self.selection_label.setText("No acquisition selected")
            return
        details = (
            self._visible_details
            if acquisition.acquisition_id == self._visible_acquisition_id
            else "Image not resident"
        )
        self.selection_label.setText(
            f"{acquisition.name}\n{details}\nAngles: unknown\nCalibration: unknown"
        )

    def _show_state(
        self, kind: Literal["empty", "loading", "error"], title: str, detail: str
    ) -> None:
        self.experiment_status.set_state(kind, title, detail)
        show_image = (
            self._visible_acquisition_id == self.selected_acquisition_id
            and self._visible_acquisition_id is not None
        )
        self.detector_stack.setCurrentIndex(1 if show_image else 0)
        self.detector_notice.setText(f"{title}: {detail}" if kind != "empty" else "")
        self.detector_notice.setVisible(show_image and kind != "empty")

    def _choose_import(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import one OSC image",
            str(Path.home()),
            "OSC images (*.osc *.OSC *.osc.gz *.OSC.GZ)",
        )
        if filename:
            self.start_import(Path(filename))

    def start_import(self, path: Path) -> None:
        source = Path(path).absolute()
        if not source.name.lower().endswith((".osc", ".osc.gz")):
            self._show_state("error", "Unsupported file", "Choose one .osc or .osc.gz file.")
            return
        self._submit_import(source, uuid4())

    def _submit_import(self, source: Path, acquisition_id: UUID) -> None:
        texture_axis = self.detector_panel.view.max_texture_axis
        if texture_axis is None or texture_axis <= 0:
            self._show_state(
                "error", "Detector display unavailable", "An OpenGL detector context is required."
            )
            return
        self._max_import_axis = min(AXIS_LIMIT, texture_axis)
        argument = str(self._max_import_axis).encode("ascii") + b"\0" + os.fsencode(source)
        try:
            self.jobs.submit(
                JobRequest(
                    self.project.project_id,
                    acquisition_id,
                    Revisions(data=1),
                    argument,
                    len(argument),
                    MAX_RESULT_BYTES,
                    prepare_osc,
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self._show_state("error", "Import could not start", str(exc))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if len(paths) != 1 or len(event.mimeData().urls()) != 1:
            self._show_state(
                "error",
                "One file at a time",
                "Drop exactly one local OSC file. Multiple-file import will arrive in a later version.",
            )
        else:
            self.start_import(paths[0])
        event.acceptProposedAction()

    def _import_ready(self, identity: JobIdentity, value: object) -> None:
        if (
            identity.project_id != self.project.project_id
            or identity.generation != self.jobs.latest_generation
            or identity.acquisition_id is None
            or identity.revisions != Revisions(data=1)
            or not isinstance(value, PreparedOsc)
        ):
            return
        existing = next(
            (
                item
                for item in self.project.acquisitions
                if item.acquisition_id == identity.acquisition_id
            ),
            None,
        )
        if existing is not None:
            if (
                existing.source_path != value.source_path
                or existing.source_sha256 != value.decoded_sha256
            ):
                self._show_state(
                    "error",
                    "Source changed",
                    "The OSC bytes no longer match this acquisition. Import the changed file as a new acquisition.",
                )
                self.statusBar().showMessage("Source changed · No image applied")
                return
            acquisition = existing
        else:
            acquisition = Acquisition.create(
                value.source_path.name,
                value.source_path,
                value.decoded_sha256,
                acquisition_id=identity.acquisition_id,
            )
        try:
            self.detector_panel.set_prepared_image(
                value.native_counts,
                value.display,
                value.profiles,
                value.low_value,
                value.high_value,
            )
        except ValueError as exc:
            self._show_state("error", "Unsupported detector size", str(exc))
            self.statusBar().showMessage("Unsupported detector size · No image applied")
            return
        if existing is None:
            self.project = replace(
                self.project, acquisitions=(*self.project.acquisitions, acquisition)
            )
        self.selected_acquisition_id = acquisition.acquisition_id
        self._visible_acquisition_id = acquisition.acquisition_id
        self._visible_details = (
            f"Native detector: {value.native_counts.shape[0]} rows x "
            f"{value.native_counts.shape[1]} columns\n"
            f"OSC header: version {value.version}, {value.byte_order} endian"
        )
        self.refresh_project()
        self.statusBar().showMessage(f"Imported {value.source_path.name} · detector-native counts")

    def _selection_changed(self) -> None:
        selected = self.project_tree.selectedItems()
        item = selected[0] if selected else None
        value = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        selected_id = UUID(value) if value else None
        if selected_id == self.selected_acquisition_id:
            return
        self.selected_acquisition_id = selected_id
        self._update_selection_label()
        had_work = self.jobs.busy
        self.jobs.invalidate()
        self.cancel_button.setEnabled(False)
        self._obsolete_pending = had_work and self.jobs.busy
        if self._obsolete_pending:
            self._show_state(
                "loading", "Stopping obsolete work", "Waiting for the prior operation to stop."
            )
            self.statusBar().showMessage("Selection changed · Waiting for prior work to stop")
        else:
            self._show_state("empty", "No image open", "No image is resident for this selection.")
            self.statusBar().showMessage("Selection changed · Ready")
        if selected_id is not None and selected_id != self._visible_acquisition_id:
            acquisition = next(
                item for item in self.project.acquisitions if item.acquisition_id == selected_id
            )
            self._submit_import(acquisition.source_path, acquisition.acquisition_id)

    def _job_state_changed(self, summary: JobSummary) -> None:
        if summary.identity.generation != self.jobs.latest_generation:
            if self._obsolete_pending and not self.jobs.busy:
                self._obsolete_pending = False
                self.cancel_button.setEnabled(False)
                self._show_state("empty", "No image open", "The previous operation was discarded.")
                self.statusBar().showMessage("Prior operation discarded · Ready")
            return
        self._obsolete_pending = False
        state = summary.state
        self.cancel_button.setEnabled(state in (JobState.QUEUED, JobState.RUNNING))
        if state in (JobState.QUEUED, JobState.RUNNING):
            self._show_state("loading", "Importing OSC", "Reading and preparing detector counts.")
        elif state == JobState.CANCEL_REQUESTED:
            self._show_state(
                "loading",
                "Stopping safely",
                "Waiting for the current operation to release its resources.",
            )
        elif state == JobState.FAILED:
            self._show_state("error", "Import failed", f"{summary.detail} Choose another OSC file.")
        elif state == JobState.CANCELED:
            self._show_state("empty", "Operation canceled", "No result was applied.")
        else:
            self._show_state("empty", "Import complete", "Detector counts are ready.")
        self.statusBar().showMessage(f"Operation: {state.value}")

    def _job_progress(self, identity: JobIdentity, message: str) -> None:
        if identity.generation == self.jobs.latest_generation:
            self.experiment_status.detail.setText(message)
            if self.detector_notice.isVisible():
                self.detector_notice.setText(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.jobs.request_close():
            event.ignore()
            self.cancel_button.setEnabled(False)
            self._show_state(
                "loading", "Closing safely", "Waiting for the current operation to stop."
            )
            self.statusBar().showMessage("Close requested · Waiting for safe stop")
            return
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(
        """
        QWidget { background: #151d24; color: #e9eff2; }
        QLabel { background: transparent; }
        QTabWidget::pane { border: 0; }
        QTabBar::tab { padding: 12px 24px; color: #b7c5cf; border-bottom: 2px solid transparent; }
        QTabBar::tab:selected { color: #f2f7f8; border-bottom-color: #61d5bd; }
        QFrame#sidePanel, QFrame#centerPanel, QFrame#statusView {
            background: #202a33; border: 1px solid #364550; border-radius: 6px;
        }
        QFrame#statusView[state="error"] { border-color: #d8776f; }
        QLabel#workspaceTitle { font-size: 22px; font-weight: 600; }
        QLabel#statusTitle { font-size: 21px; font-weight: 600; }
        QLabel#statusEyebrow { color: #61d5bd; font-size: 11px; font-weight: 700; }
        QFrame#statusView[state="error"] QLabel#statusEyebrow { color: #eb958d; }
        QLabel#mutedText, QLabel#statusDetail { color: #b7c5cf; }
        QTreeWidget { border: 0; background: transparent; outline: none; }
        QTreeWidget::item { padding: 6px 2px; }
        QTreeWidget::item:selected { background: #315669; color: #ffffff; }
        QPushButton { padding: 8px 14px; border: 1px solid #547066; border-radius: 4px; }
        QPushButton:disabled { color: #b5bfc4; border-color: #4b5960; }
        QStatusBar { color: #b7c5cf; }
        """
    )
    window = ShellWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
