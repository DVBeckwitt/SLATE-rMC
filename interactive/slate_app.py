"""Native desktop shell for SLATE-rMC; launch with ``python interactive/slate_app.py``."""

import json
import os
import sys
from collections import deque
from dataclasses import dataclass, replace
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
from osc_import import AXIS_LIMIT, PreparedOsc, encode_bounded_path, prepare_osc
from project_io import (
    LoadedProject,
    PublishedProject,
    SourceCheck,
    discard_recovery,
    load_project,
    write_project,
)
from project_state import (
    MAX_ACQUISITIONS,
    Acquisition,
    DetectorViewState,
    Project,
    ProjectDocument,
    ProjectFormatError,
    ProjectViewState,
    project_to_document,
)
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
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

MAX_PENDING_WRITES = 8
MAX_PENDING_WRITE_BYTES = 3 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class WriteTask:
    project_id: UUID
    revision: int
    destination: Path
    argument: bytes | Path
    recovery: bool
    explicit: bool
    adopt_destination: bool = False
    discard: bool = False


def _request_size(argument: bytes | Path) -> int:
    return len(argument) if isinstance(argument, bytes) else len(str(argument).encode("utf-8"))


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
    def __init__(
        self, project: Project | None = None, *, recovery_root: Path | None = None
    ) -> None:
        super().__init__()
        self.project = project if project is not None else Project.create()
        local_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
        self.recovery_root = (
            Path(recovery_root).absolute()
            if recovery_root is not None
            else local_data / "SLATE-rMC" / "recovery"
        )
        self._project_path: Path | None = None
        self._revision = 0
        self._saved_revision = -1
        self._draft_revision = -1
        self._save_failure = ""
        self._restoring_view = False
        self._pending_view_restore: ProjectViewState | None = None
        self._source_checks: dict[UUID, SourceCheck] = {}
        self._write_queue: deque[WriteTask] = deque()
        self._active_write: WriteTask | None = None
        self._active_kind: Literal["import", "relink", "open", "save", "discard"] | None = None
        self._active_generation: int | None = None
        self._deferred_import: tuple[Path, UUID, Literal["import", "relink"], UUID] | None = None
        self._pending_open: tuple[Path, bool] | None = None
        self._opening_recovery = False
        self._opening_path: Path | None = None
        self._close_intent = False
        self._close_after_write = False
        self._discard_confirmed = False
        self._allow_close = False
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
        self.jobs.result_ready.connect(self._job_result_ready)
        self.jobs.drained.connect(self.close)
        self.cancel_button.clicked.connect(self.jobs.cancel)
        self.import_button.clicked.connect(self._choose_import)
        self.relink_button.clicked.connect(self._choose_relink)
        self.open_button.clicked.connect(self._choose_open)
        self.save_button.clicked.connect(self.save_project)
        self.save_as_button.clicked.connect(self.save_project_as)
        self.recover_button.clicked.connect(self._choose_recovery)
        self.rename_button.clicked.connect(self._choose_project_name)
        self.rename_acquisition_button.clicked.connect(self._choose_acquisition_name)
        self.move_up_button.clicked.connect(lambda: self.move_selected_acquisition(-1))
        self.move_down_button.clicked.connect(lambda: self.move_selected_acquisition(1))
        self.workspaces.currentChanged.connect(self._mark_dirty)
        self.detector_panel.view.crosshair_changed.connect(self._mark_dirty)
        self.detector_panel.view.view_state_changed.connect(self._mark_dirty)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(750)
        self._autosave_timer.timeout.connect(self._autosave)
        self.statusBar().showMessage("Local project · Not saved")
        self.refresh_project()
        self._update_save_status()
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
        project_controls = QHBoxLayout()
        self.open_button = QPushButton("Open")
        self.save_button = QPushButton("Save")
        self.save_as_button = QPushButton("Save As")
        self.recover_button = QPushButton("Recover Draft")
        self.rename_button = QPushButton("Rename Project")
        for button in (
            self.open_button,
            self.save_button,
            self.save_as_button,
            self.recover_button,
            self.rename_button,
        ):
            project_controls.addWidget(button)
        self.save_status = QLabel()
        self.save_status.setObjectName("mutedText")
        project_controls.addWidget(self.save_status, 1)
        outer.addLayout(project_controls)

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
        row_controls = QHBoxLayout()
        self.rename_acquisition_button = QPushButton("Rename")
        self.move_up_button = QPushButton("Up")
        self.move_down_button = QPushButton("Down")
        for button in (self.rename_acquisition_button, self.move_up_button, self.move_down_button):
            row_controls.addWidget(button)
        browser_layout.addLayout(row_controls)
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
        self.relink_button = QPushButton("Relink OSC")
        self.relink_button.setEnabled(False)
        controls.addWidget(self.relink_button)
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
            "Angles and calibration remain unknown. Saving a draft never resumes a solver."
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

    def _recovery_path(self) -> Path:
        return self.recovery_root / f"{self.project.project_id}.slate.json"

    def _capture_view(self) -> ProjectViewState:
        detector = None
        view = self.detector_panel.view
        if (
            self.selected_acquisition_id is not None
            and self.selected_acquisition_id == self._visible_acquisition_id
            and view.image is not None
        ):
            detector = DetectorViewState(
                view.crosshair[0],
                view.crosshair[1],
                view.zoom,
                view.pan.x(),
                view.pan.y(),
                view.low_value,
                view.high_value,
                view.positive_log,
            )
        elif (
            self._pending_view_restore is not None
            and self._pending_view_restore.selected_acquisition_id == self.selected_acquisition_id
        ):
            detector = self._pending_view_restore.detector
        return ProjectViewState(
            self.selected_acquisition_id,
            "fit_experiments" if self.workspaces.currentIndex() == 0 else "simulator",
            detector,
        )

    def _update_save_status(self) -> None:
        if self._save_failure:
            label = f"Save failed · {self._save_failure}"
        elif self._active_write is not None:
            label = "Saving…" if self._active_write.explicit else "Autosaving…"
        elif self._project_path is not None:
            label = (
                f"Saved · {self._project_path.name}"
                if self._saved_revision >= self._revision
                else "Unsaved changes"
            )
        elif self._draft_revision >= self._revision:
            label = "Recoverable draft · Save As to name it"
        elif self._revision == 0:
            label = "Unsaved project"
        else:
            label = "Unsaved changes · Recovery pending"
        self.save_status.setText(label)
        dirty = (
            self._revision
            > (self._saved_revision if self._project_path is not None else self._draft_revision)
            and self._revision > 0
        )
        self.setWindowTitle(f"SLATE · {self.project.name}{' *' if dirty else ''}")

    def _mark_dirty(self, _value: object = None) -> None:
        if self._restoring_view:
            return
        self._revision += 1
        self._save_failure = ""
        self._autosave_timer.start()
        self._update_save_status()

    def _autosave(self) -> None:
        if (self._project_path is not None and self._saved_revision >= self._revision) or (
            self._project_path is None and self._draft_revision >= self._revision
        ):
            return
        if (
            self._close_intent
            or self._pending_open is not None
            or any(task.adopt_destination for task in self._write_queue)
            or (self._active_write is not None and self._active_write.adopt_destination)
        ):
            return
        destination = self._project_path or self._recovery_path()
        if not self._queue_write(destination, recovery=self._project_path is None, explicit=False):
            self._autosave_timer.start()

    def _queue_write(
        self, destination: Path, *, recovery: bool, explicit: bool, adopt_destination: bool = False
    ) -> bool:
        path = Path(destination).absolute()
        try:
            document = project_to_document(
                ProjectDocument(self.project, self._capture_view()), path
            )
            retire = (
                str(self._recovery_path())
                if not recovery and (adopt_destination or self._project_path is None)
                else None
            )
            argument = json.dumps(
                {
                    "destination": str(path),
                    "document": document,
                    "recovery": recovery,
                    "retire_draft": retire,
                },
                allow_nan=False,
            ).encode("utf-8")
            if len(argument) > 4 * 1024 * 1024:
                raise ProjectFormatError("project save request exceeds 4 MiB")
        except (ProjectFormatError, ValueError) as exc:
            self._save_failure = str(exc)
            self._update_save_status()
            return False
        task = WriteTask(
            self.project.project_id,
            self._revision,
            path,
            argument,
            recovery,
            explicit,
            adopt_destination,
        )
        if not explicit:
            self._write_queue = deque(
                queued
                for queued in self._write_queue
                if queued.explicit
                or queued.destination != path
                or queued.project_id != task.project_id
            )
        queued_bytes = sum(_request_size(queued.argument) for queued in self._write_queue)
        if (
            len(self._write_queue) >= MAX_PENDING_WRITES
            or queued_bytes + len(argument) > MAX_PENDING_WRITE_BYTES
        ):
            self._save_failure = "Save queue is full; retry after the current write"
            self._update_save_status()
            return False
        self._write_queue.append(task)
        self._save_failure = ""
        self._update_save_status()
        QTimer.singleShot(0, self._dispatch_pending)
        return True

    def save_project(self) -> bool:
        if self._project_path is None:
            return self.save_project_as()
        return self._queue_write(self._project_path, recovery=False, explicit=True)

    def save_project_as(self) -> bool:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save project",
            str(Path.home() / f"{self.project.name}.slate.json"),
            "SLATE projects (*.slate.json)",
        )
        return self.save_as(Path(filename)) if filename else False

    def save_as(self, path: Path) -> bool:
        destination = Path(path).absolute()
        if not destination.name.lower().endswith(".slate.json"):
            destination = destination.with_name(destination.name + ".slate.json")
        return self._queue_write(destination, recovery=False, explicit=True, adopt_destination=True)

    def rename_project(self, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError("A project needs a name")
        self.project = replace(self.project, name=name)
        self.refresh_project()
        self._mark_dirty()

    def _choose_project_name(self) -> None:
        name, accepted = QInputDialog.getText(
            self, "Rename project", "Project name", text=self.project.name
        )
        if accepted:
            try:
                self.rename_project(name)
            except ValueError as exc:
                self._show_state("error", "Invalid project name", str(exc))

    def rename_selected_acquisition(self, name: str) -> None:
        if self.selected_acquisition_id is None:
            raise ValueError("Select an acquisition to rename")
        self.project = self.project.rename_acquisition(self.selected_acquisition_id, name)
        self.refresh_project()
        self._mark_dirty()

    def _choose_acquisition_name(self) -> None:
        selected = next(
            (
                item
                for item in self.project.acquisitions
                if item.acquisition_id == self.selected_acquisition_id
            ),
            None,
        )
        if selected is None:
            return
        name, accepted = QInputDialog.getText(
            self, "Rename acquisition", "Acquisition name", text=selected.name
        )
        if accepted:
            try:
                self.rename_selected_acquisition(name)
            except ValueError as exc:
                self._show_state("error", "Invalid acquisition name", str(exc))

    def move_selected_acquisition(self, direction: int) -> None:
        selected = self.selected_acquisition_id
        if selected is None or direction not in (-1, 1):
            return
        order = [item.acquisition_id for item in self.project.acquisitions]
        index = order.index(selected)
        target = index + direction
        if not 0 <= target < len(order):
            return
        order[index], order[target] = order[target], order[index]
        self.project = self.project.reorder_acquisitions(order)
        self.refresh_project()
        self._mark_dirty()

    def _has_unsaved_edits(self) -> bool:
        if self._project_path is not None:
            return self._revision > self._saved_revision
        return self._revision > 0 and self._revision > self._draft_revision

    def _ask_unsaved(self, action: str) -> QMessageBox.StandardButton:
        destination = (
            "the project file" if self._project_path is not None else "a recoverable draft"
        )
        return QMessageBox.question(
            self,
            f"Save before {action}?",
            f"Save the current editable state to {destination} before {action}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )

    def _dispatch_pending(self) -> None:
        if self.jobs.busy or self._active_kind is not None:
            return
        if self._write_queue:
            task = self._write_queue.popleft()
            self._active_write = task
            self._active_kind = "discard" if task.discard else "save"
            run = discard_recovery if task.discard else write_project
            request = JobRequest(
                task.project_id,
                None,
                Revisions(data=task.revision),
                task.argument,
                _request_size(task.argument),
                16 * 1024,
                run,
            )
            try:
                identity = self.jobs.submit(request)
            except (ValueError, RuntimeError, TypeError) as exc:
                self._active_kind = None
                self._active_write = None
                self._save_failure = str(exc)
                self._update_save_status()
                return
            self._active_generation = identity.generation
            self._update_save_status()
            return
        if self._pending_open is not None:
            self._advance_open()
            return
        if self._close_intent:
            self._advance_close()
            return
        if self._deferred_import is not None:
            source, acquisition_id, mode, project_id = self._deferred_import
            self._deferred_import = None
            if project_id == self.project.project_id:
                self._submit_import(source, acquisition_id, mode=mode)

    def _choose_open(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open project", str(Path.home()), "SLATE projects (*.slate.json)"
        )
        if filename:
            self.open_project(Path(filename))

    def _choose_recovery(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            f"Recover a draft from {self.recovery_root}",
            str(self.recovery_root),
            "SLATE drafts (*.slate.json)",
        )
        if filename:
            self.open_project(Path(filename), recovery=True)

    def open_project(self, path: Path, *, recovery: bool = False) -> None:
        candidate = Path(path).absolute()
        if recovery and candidate.parent != self.recovery_root:
            self._show_state(
                "error",
                "Recovery path unavailable",
                "Choose a draft in the configured recovery location.",
            )
            return
        self._pending_open = (candidate, recovery)
        self._discard_confirmed = False
        if self._active_kind in ("import", "relink"):
            self.jobs.cancel()
        QTimer.singleShot(0, self._dispatch_pending)

    def _advance_open(self) -> None:
        assert self._pending_open is not None
        if self._has_unsaved_edits() and not self._discard_confirmed:
            choice = self._ask_unsaved("opening another project")
            if choice == QMessageBox.StandardButton.Cancel:
                self._pending_open = None
                self._autosave_timer.start()
                return
            if choice == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    self._pending_open = None
                return
            self._discard_confirmed = True
            self._autosave_timer.stop()
        path, recovery = self._pending_open
        self._pending_open = None
        self._discard_confirmed = False
        self._active_kind = "open"
        argument = encode_bounded_path(path, self._max_import_axis or AXIS_LIMIT)
        try:
            identity = self.jobs.submit(
                JobRequest(
                    self.project.project_id,
                    None,
                    Revisions(data=self._revision),
                    argument,
                    len(argument),
                    4 * 1024 * 1024,
                    load_project,
                )
            )
        except (ValueError, RuntimeError, TypeError) as exc:
            self._active_kind = None
            self._show_state("error", "Project open failed", str(exc))
            return
        self._active_generation = identity.generation
        self._opening_recovery = recovery
        self._opening_path = path

    def _choose_relink(self) -> None:
        if self.selected_acquisition_id is None:
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Relink the selected OSC source",
            str(Path.home()),
            "OSC images (*.osc *.OSC *.osc.gz *.OSC.GZ)",
        )
        if filename:
            self.relink_selected(Path(filename))

    def relink_selected(self, path: Path) -> None:
        if self.selected_acquisition_id is None:
            self._show_state(
                "error", "No acquisition selected", "Select an acquisition before relinking."
            )
            return
        source = Path(path).absolute()
        if not source.name.lower().endswith((".osc", ".osc.gz")):
            self._show_state("error", "Unsupported file", "Choose one .osc or .osc.gz file.")
            return
        self._submit_import(source, self.selected_acquisition_id, mode="relink")

    def _advance_close(self) -> None:
        if self._discard_confirmed:
            self._allow_close = True
            self.close()
            return
        if self._close_after_write:
            if self._has_unsaved_edits():
                destination = self._project_path or self._recovery_path()
                if not self._queue_write(
                    destination, recovery=self._project_path is None, explicit=True
                ):
                    self._close_intent = False
                    self._close_after_write = False
                return
            self._allow_close = True
            self.close()
            return
        needs_choice = self._has_unsaved_edits() or (
            self._project_path is None and (self._revision > 0 or self._draft_revision >= 0)
        )
        if not needs_choice:
            self._allow_close = True
            self.close()
            return
        choice = self._ask_unsaved("closing")
        if choice == QMessageBox.StandardButton.Cancel:
            self._close_intent = False
            self._autosave_timer.start()
            self.statusBar().showMessage("Close canceled · Work retained")
        elif choice == QMessageBox.StandardButton.Save:
            destination = self._project_path or self._recovery_path()
            if self._queue_write(destination, recovery=self._project_path is None, explicit=True):
                self._close_after_write = True
            else:
                self._close_intent = False
        else:
            self._discard_confirmed = True
            if self._project_path is None and self._draft_revision >= 0:
                path = self._recovery_path()
                task = WriteTask(
                    self.project.project_id,
                    self._revision,
                    path,
                    path,
                    True,
                    True,
                    discard=True,
                )
                self._write_queue.append(task)
                QTimer.singleShot(0, self._dispatch_pending)
            else:
                self._allow_close = True
                self.close()

    def refresh_project(self) -> None:
        selected_id = self.selected_acquisition_id
        self.project_tree.blockSignals(True)
        self.project_tree.clear()
        root = QTreeWidgetItem([self.project.name])
        root.setData(0, Qt.ItemDataRole.UserRole, None)
        self.project_tree.addTopLevelItem(root)
        selected_item = None
        for acquisition in self.project.acquisitions:
            check = self._source_checks.get(acquisition.acquisition_id)
            suffix = f" · {check.state}" if check is not None and check.state != "verified" else ""
            item = QTreeWidgetItem([acquisition.name + suffix])
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
            self.relink_button.setEnabled(False)
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
            self.relink_button.setEnabled(False)
            return
        details = (
            self._visible_details
            if acquisition.acquisition_id == self._visible_acquisition_id
            else "Image not resident"
        )
        check = self._source_checks.get(acquisition.acquisition_id)
        source_state = (
            f"Source: {check.state} · {check.detail}\nRelink with the original OSC bytes."
            if check is not None and check.state != "verified"
            else "Source: verified"
        )
        self.selection_label.setText(
            f"{acquisition.name}\n{details}\n{source_state}\nAngles: unknown\nCalibration: unknown"
        )
        self.relink_button.setEnabled(True)

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

    def _submit_import(
        self, source: Path, acquisition_id: UUID, *, mode: Literal["import", "relink"] = "import"
    ) -> None:
        if (
            self._active_kind in ("save", "discard", "open")
            or self._write_queue
            or self._pending_open
        ):
            self._deferred_import = (source, acquisition_id, mode, self.project.project_id)
            self.statusBar().showMessage("Import queued behind project I/O")
            return
        texture_axis = self.detector_panel.view.max_texture_axis
        if texture_axis is None or texture_axis <= 0:
            self._show_state(
                "error", "Detector display unavailable", "An OpenGL detector context is required."
            )
            return
        self._max_import_axis = min(AXIS_LIMIT, texture_axis)
        argument = encode_bounded_path(source, self._max_import_axis)
        previous_kind = self._active_kind
        previous_generation = self._active_generation
        self._active_kind = mode
        self._active_generation = None
        try:
            identity = self.jobs.submit(
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
            self._active_kind = previous_kind
            self._active_generation = previous_generation
            self._show_state("error", "Import could not start", str(exc))
            return
        self._active_generation = identity.generation

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
        relinking = self._active_kind == "relink"
        if existing is not None:
            if existing.source_sha256 != value.decoded_sha256 or (
                not relinking and existing.source_path != value.source_path
            ):
                self._show_state(
                    "error",
                    "Source mismatch",
                    "Decoded OSC bytes do not match this acquisition. Choose the original source or import a new acquisition.",
                )
                self.statusBar().showMessage("Source mismatch · No image applied")
                return
            acquisition = (
                replace(existing, source_path=value.source_path) if relinking else existing
            )
        else:
            if relinking:
                self._show_state("error", "Relink unavailable", "The acquisition no longer exists.")
                return
            if len(self.project.acquisitions) >= MAX_ACQUISITIONS:
                self._show_state(
                    "error",
                    "Project is full",
                    f"This project supports at most {MAX_ACQUISITIONS} acquisitions.",
                )
                return
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
        elif relinking and acquisition.source_path != existing.source_path:
            self.project = replace(
                self.project,
                acquisitions=tuple(
                    acquisition if item.acquisition_id == acquisition.acquisition_id else item
                    for item in self.project.acquisitions
                ),
            )
        self.selected_acquisition_id = acquisition.acquisition_id
        self._visible_acquisition_id = acquisition.acquisition_id
        self._visible_details = (
            f"Native detector: {value.native_counts.shape[0]} rows x "
            f"{value.native_counts.shape[1]} columns\n"
            f"OSC header: version {value.version}, {value.byte_order} endian"
        )
        self._source_checks[acquisition.acquisition_id] = SourceCheck(
            acquisition.acquisition_id, "verified"
        )
        self.refresh_project()
        self._apply_restored_view()
        if existing is None or (relinking and acquisition.source_path != existing.source_path):
            self._mark_dirty()
        self.statusBar().showMessage(f"Imported {value.source_path.name} · detector-native counts")

    def _apply_restored_view(self) -> None:
        saved = self._pending_view_restore
        if saved is None or saved.selected_acquisition_id != self.selected_acquisition_id:
            return
        detector = saved.detector
        self._pending_view_restore = None
        if detector is None:
            return
        view = self.detector_panel.view
        if view.image is None or not (
            0 <= detector.column_px < view.image.shape[1]
            and 0 <= detector.row_px < view.image.shape[0]
        ):
            self._show_state(
                "error", "Saved view unavailable", "Crosshair is outside this detector."
            )
            return
        self._restoring_view = True
        try:
            view.zoom = detector.zoom
            view.pan = QPointF(detector.pan_x_px, detector.pan_y_px)
            view.crosshair = (detector.column_px, detector.row_px)
            view.set_levels(
                detector.low_value, detector.high_value, positive_log=detector.positive_log
            )
            self.detector_panel._refresh_profile_if_needed()
        finally:
            self._restoring_view = False

    def _selection_changed(self) -> None:
        selected = self.project_tree.selectedItems()
        item = selected[0] if selected else None
        value = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        selected_id = UUID(value) if value else None
        if selected_id == self.selected_acquisition_id:
            return
        self.selected_acquisition_id = selected_id
        self._deferred_import = None
        self._update_selection_label()
        had_work = self.jobs.busy and self._active_kind not in ("save", "discard")
        if self._active_kind not in ("save", "discard"):
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
            check = self._source_checks.get(selected_id)
            if check is None or check.state == "verified":
                self._submit_import(acquisition.source_path, acquisition.acquisition_id)
            else:
                self._show_state(
                    "error",
                    "Source unavailable",
                    f"{check.state}: {check.detail}. Use Relink OSC with the original bytes.",
                )
        self._mark_dirty()

    def _job_state_changed(self, summary: JobSummary) -> None:
        state = summary.state
        terminal = state in (JobState.COMPLETED, JobState.CANCELED, JobState.FAILED)
        if summary.identity.generation != self.jobs.latest_generation:
            if terminal and summary.identity.generation == self._active_generation:
                self._active_kind = None
                self._active_generation = None
                QTimer.singleShot(0, self._dispatch_pending)
            if self._obsolete_pending and not self.jobs.busy:
                self._obsolete_pending = False
                self.cancel_button.setEnabled(False)
                self._show_state("empty", "No image open", "The previous operation was discarded.")
                self.statusBar().showMessage("Prior operation discarded · Ready")
            return
        kind = self._active_kind
        self._obsolete_pending = False
        self.cancel_button.setEnabled(
            kind in ("import", "relink", "open") and state in (JobState.QUEUED, JobState.RUNNING)
        )
        if kind in ("save", "discard"):
            if state == JobState.FAILED or state == JobState.CANCELED:
                self._save_failure = (
                    summary.detail
                    if state == JobState.FAILED
                    else "Save outcome uncertain after cancellation; inspect the destination"
                )
                self._active_write = None
                self._active_kind = None
                self._active_generation = None
                self._close_intent = False
                self._close_after_write = False
                self._discard_confirmed = False
                self._pending_open = None
                QTimer.singleShot(0, self._dispatch_pending)
            self._update_save_status()
            self.statusBar().showMessage(f"Project write: {state.value}")
            return
        if kind == "open":
            if state in (JobState.QUEUED, JobState.RUNNING):
                self._show_state(
                    "loading", "Opening project", "Checking project and OSC references."
                )
            elif state == JobState.FAILED:
                self._show_state("error", "Project open failed", summary.detail)
            elif state == JobState.CANCELED:
                self._show_state(
                    "empty", "Open canceled", "The previous project remains available."
                )
            if state in (JobState.FAILED, JobState.CANCELED):
                self._active_kind = None
                self._active_generation = None
                QTimer.singleShot(0, self._dispatch_pending)
            self.statusBar().showMessage(f"Project open: {state.value}")
            return
        if state in (JobState.QUEUED, JobState.RUNNING):
            title = "Relinking OSC" if kind == "relink" else "Importing OSC"
            self._show_state("loading", title, "Reading and preparing detector counts.")
        elif state == JobState.CANCEL_REQUESTED:
            self._show_state(
                "loading",
                "Stopping safely",
                "Waiting for the current operation to release its resources.",
            )
        elif state == JobState.FAILED:
            self._show_state(
                "error", "OSC read failed", f"{summary.detail} Choose another OSC file."
            )
        elif state == JobState.CANCELED:
            self._show_state("empty", "Operation canceled", "No result was applied.")
        else:
            self._show_state("empty", "OSC ready", "Detector counts are ready.")
        if state in (JobState.FAILED, JobState.CANCELED):
            self._active_kind = None
            self._active_generation = None
            QTimer.singleShot(0, self._dispatch_pending)
        self.statusBar().showMessage(f"OSC operation: {state.value}")

    def _job_result_ready(self, identity: JobIdentity, value: object) -> None:
        if identity.generation != self._active_generation:
            return
        kind = self._active_kind
        try:
            if kind in ("save", "discard"):
                self._write_ready(identity, value)
            elif kind == "open":
                self._open_ready(identity, value)
            elif kind in ("import", "relink"):
                self._import_ready(identity, value)
        finally:
            self._active_kind = None
            self._active_generation = None
            QTimer.singleShot(0, self._dispatch_pending)

    def _write_ready(self, identity: JobIdentity, value: object) -> None:
        task = self._active_write
        self._active_write = None
        if (
            task is None
            or not isinstance(value, PublishedProject)
            or identity.project_id != task.project_id
            or identity.revisions.data != task.revision
            or value.path != task.destination
        ):
            self._save_failure = "Project write receipt did not match its snapshot"
            self._close_intent = False
            self._pending_open = None
            self._update_save_status()
            return
        if task.discard:
            self._draft_revision = -1
            self.statusBar().showMessage("Recovery draft discarded")
        elif task.project_id == self.project.project_id:
            if task.adopt_destination:
                self._project_path = task.destination
                self._saved_revision = task.revision
                self._draft_revision = -1
            elif task.recovery:
                self._draft_revision = max(self._draft_revision, task.revision)
            elif self._project_path == task.destination:
                self._saved_revision = max(self._saved_revision, task.revision)
            self.statusBar().showMessage(value.cleanup_warning or f"Saved {task.destination.name}")
        self._save_failure = ""
        self._update_save_status()
        if self._has_unsaved_edits() and not self._close_intent and self._pending_open is None:
            self._autosave_timer.start()

    def _open_ready(self, identity: JobIdentity, value: object) -> None:
        if (
            not isinstance(value, LoadedProject)
            or self._opening_path != value.path
            or identity.project_id != self.project.project_id
        ):
            self._show_state(
                "error", "Project open failed", "Project read receipt did not match its request."
            )
            return
        if (
            self._opening_recovery
            and value.path.name != f"{value.document.project.project_id}.slate.json"
        ):
            self._show_state(
                "error",
                "Recovery draft name mismatch",
                "Choose the UUID-named draft for this project in the recovery location.",
            )
            return
        if identity.revisions.data != self._revision:
            self._show_state(
                "error",
                "Project changed while opening",
                "Current edits were kept. Open the project again to replace them.",
            )
            if self._has_unsaved_edits():
                self._autosave_timer.start()
            return
        self._restoring_view = True
        try:
            self.project = value.document.project
            self.selected_acquisition_id = value.document.view.selected_acquisition_id
            self._visible_acquisition_id = None
            self._visible_details = ""
            self._source_checks = {item.acquisition_id: item for item in value.sources}
            self._pending_view_restore = value.document.view
            self._deferred_import = None
            self._project_path = None if self._opening_recovery else value.path
            self._revision = 0
            self._saved_revision = -1 if self._opening_recovery else 0
            self._draft_revision = 0 if self._opening_recovery else -1
            self._save_failure = ""
            self.workspaces.setCurrentIndex(
                0 if value.document.view.workspace == "fit_experiments" else 1
            )
            self.refresh_project()
            self._update_save_status()
        finally:
            self._restoring_view = False
        selected = self.selected_acquisition_id
        if selected is not None:
            check = self._source_checks[selected]
            if check.state == "verified":
                source = next(
                    item.source_path
                    for item in self.project.acquisitions
                    if item.acquisition_id == selected
                )
                self._deferred_import = (source, selected, "import", self.project.project_id)
            else:
                self._show_state(
                    "error",
                    "Source unavailable",
                    f"{check.state}: {check.detail}. Use Relink OSC with the original bytes.",
                )
        self.statusBar().showMessage(
            "Recovered editable draft · No solver resumed"
            if self._opening_recovery
            else f"Opened {value.path.name} · References checked"
        )

    def _job_progress(self, identity: JobIdentity, message: str) -> None:
        if identity.generation == self.jobs.latest_generation:
            if self._active_kind not in ("save", "discard"):
                self.experiment_status.detail.setText(message)
                if self.detector_notice.isVisible():
                    self.detector_notice.setText(message)
            self.statusBar().showMessage(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_close:
            if not self.jobs.request_close():
                event.ignore()
                self.cancel_button.setEnabled(False)
                self._show_state(
                    "loading", "Closing safely", "Waiting for the current operation to stop."
                )
                self.statusBar().showMessage("Close requested · Waiting for safe stop")
                return
            super().closeEvent(event)
            return
        event.ignore()
        if not self._close_intent:
            self._close_intent = True
            self._pending_open = None
            self._deferred_import = None
            self._autosave_timer.stop()
            self._write_queue = deque(task for task in self._write_queue if task.explicit)
            if self._active_kind in ("import", "relink", "open"):
                self.jobs.cancel()
            self.statusBar().showMessage("Close requested · Preserving accepted state")
        QTimer.singleShot(0, self._dispatch_pending)


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
