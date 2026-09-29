"""Native desktop shell for SLATE-rMC; launch with ``python interactive/slate_app.py``."""

import csv
import json
import math
import os
import sys
from collections import OrderedDict, deque
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
from metadata_review import (
    MAPPED_FIELDS,
    ValidatedReference,
    apply_mapped_rows,
    filename_angle_proposal,
    metadata_csv,
    parse_table_text,
    prepare_reference,
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
    AcquisitionMetadata,
    DetectorViewState,
    Project,
    ProjectDocument,
    ProjectFormatError,
    ProjectViewState,
    project_to_document,
)
from PySide6.QtCore import QItemSelectionModel, QPointF, Qt, QTimer
from PySide6.QtGui import (
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
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
MAX_IMPORT_CANDIDATES = 128
MAX_IMPORT_PATH_BYTES = 512 * 1024
MAX_CACHED_PLANES = 2
MAX_THUMBNAIL_BYTES = 128 * 96 * 96


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


@dataclass(slots=True)
class ImportCandidate:
    acquisition_id: UUID
    path: Path
    status: Literal["queued", "reading", "imported", "unsupported", "failed", "canceled"]
    detail: str = ""


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
        self._active_kind: (
            Literal["import", "relink", "batch", "reference", "open", "save", "discard"] | None
        ) = None
        self._active_generation: int | None = None
        self._active_load_id: UUID | None = None
        self._active_load_revision: int | None = None
        self._active_load_hash: str | None = None
        self._deferred_import: tuple[Path, UUID, Literal["import", "relink"], UUID] | None = None
        self._candidates: dict[UUID, ImportCandidate] = {}
        self._candidate_queue: deque[UUID] = deque()
        self._active_candidate_id: UUID | None = None
        self._batch_auto_select = False
        self._pending_reference: tuple[str, Path, tuple[UUID, ...], UUID, int] | None = None
        self._active_reference: tuple[str, Path, tuple[UUID, ...], UUID, int] | None = None
        self._resident_planes: OrderedDict[UUID, PreparedOsc] = OrderedDict()
        self._thumbnails: dict[UUID, tuple[bytes, int, int]] = {}
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
        self._selection_from_review = False
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
        self.cancel_button.clicked.connect(self._cancel_current)
        self.import_button.clicked.connect(self._choose_import)
        self.folder_button.clicked.connect(self._choose_folder)
        self.review_table.itemSelectionChanged.connect(self._review_selection_changed)
        self.filmstrip.itemClicked.connect(self._filmstrip_clicked)
        self.retry_button.clicked.connect(self._retry_review_selection)
        self.cancel_row_button.clicked.connect(self._cancel_review_selection)
        self.remove_button.clicked.connect(self._remove_review_selection)
        self.metadata_button.clicked.connect(self._edit_selected_metadata)
        self.confirm_button.clicked.connect(self._confirm_selected_proposals)
        self.cif_button.clicked.connect(lambda: self._choose_reference("cif"))
        self.configuration_button.clicked.connect(lambda: self._choose_reference("configuration"))
        self.paste_button.clicked.connect(lambda: self._map_metadata_text("\t"))
        self.csv_button.clicked.connect(lambda: self._map_metadata_text(","))
        self.export_metadata_button.clicked.connect(self._choose_metadata_export)
        self.relink_button.clicked.connect(self._choose_relink)
        self.open_button.clicked.connect(self._choose_open)
        self.save_button.clicked.connect(self.save_project)
        self.save_as_button.clicked.connect(self.save_project_as)
        self.recover_button.clicked.connect(self._choose_recovery)
        self.rename_button.clicked.connect(self._choose_project_name)
        self.rename_acquisition_button.clicked.connect(self._choose_acquisition_name)
        self.move_up_button.clicked.connect(lambda: self.move_selected_acquisition(-1))
        self.move_down_button.clicked.connect(lambda: self.move_selected_acquisition(1))
        self.previous_shortcut = QShortcut(QKeySequence("Alt+Left"), self)
        self.next_shortcut = QShortcut(QKeySequence("Alt+Right"), self)
        self.previous_shortcut.activated.connect(lambda: self._step_acquisition(-1))
        self.next_shortcut.activated.connect(lambda: self._step_acquisition(1))
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
        self.filmstrip = QListWidget()
        self.filmstrip.setObjectName("acquisitionFilmstrip")
        self.filmstrip.setFlow(QListWidget.Flow.LeftToRight)
        self.filmstrip.setViewMode(QListWidget.ViewMode.IconMode)
        self.filmstrip.setIconSize(QPixmap(96, 96).size())
        self.filmstrip.setFixedHeight(120)
        center_layout.addWidget(self.filmstrip)
        self.review_table = QTableWidget(0, 11)
        self.review_table.setObjectName("acquisitionReview")
        self.review_table.setHorizontalHeaderLabels(
            (
                "Preview",
                "Filename",
                "Role",
                "Specimen / mount",
                "Angle (deg)",
                "Exposure (s)",
                "Native shape",
                "Detector setup",
                "Material / references",
                "Calibrant",
                "Status",
            )
        )
        self.review_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.review_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.review_table.setSortingEnabled(True)
        self.review_table.setMinimumHeight(165)
        center_layout.addWidget(self.review_table)
        self.import_button = QPushButton("Import files")
        controls = QHBoxLayout()
        controls.addWidget(self.import_button)
        self.folder_button = QPushButton("Review folder")
        controls.addWidget(self.folder_button)
        self.retry_button = QPushButton("Retry row")
        controls.addWidget(self.retry_button)
        self.cancel_row_button = QPushButton("Cancel row")
        controls.addWidget(self.cancel_row_button)
        self.remove_button = QPushButton("Remove from project")
        controls.addWidget(self.remove_button)
        self.metadata_button = QPushButton("Apply to selected")
        controls.addWidget(self.metadata_button)
        self.confirm_button = QPushButton("Confirm suggestions")
        controls.addWidget(self.confirm_button)
        self.relink_button = QPushButton("Relink OSC")
        self.relink_button.setEnabled(False)
        controls.addWidget(self.relink_button)
        self.cancel_button = QPushButton("Cancel operation")
        self.cancel_button.setEnabled(False)
        controls.addWidget(self.cancel_button)
        controls.addStretch()
        center_layout.addLayout(controls)
        metadata_controls = QHBoxLayout()
        self.cif_button = QPushButton("Choose CIF")
        self.configuration_button = QPushButton("Choose configuration")
        self.paste_button = QPushButton("Paste table")
        self.csv_button = QPushButton("Map CSV")
        self.export_metadata_button = QPushButton("Export metadata CSV")
        for button in (
            self.cif_button,
            self.configuration_button,
            self.paste_button,
            self.csv_button,
            self.export_metadata_button,
        ):
            metadata_controls.addWidget(button)
        center_layout.addLayout(metadata_controls)
        storage_notice = QLabel(
            "Storage: reference OSC files in place. Originals are never copied or modified; copy-storage review is a later task."
        )
        storage_notice.setWordWrap(True)
        storage_notice.setObjectName("mutedText")
        center_layout.addWidget(storage_notice)
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
                view.effective_zoom(),
                view.pan.x(),
                view.pan.y(),
                view.low_value,
                view.high_value,
                view.contrast_mode,
                view.devicePixelRatioF(),
                view.scale_mode,
                view.show_image,
                view.show_crosshair,
                view.show_markers,
                view.follow_cursor,
                self.detector_panel.row_width_control.value(),
                self.detector_panel.column_width_control.value(),
                self.detector_panel.profile_measure_control.currentData(),
                self.detector_panel.profile_scope_control.currentData(),
                self.detector_panel._roi_bounds,
                self.detector_panel.horizontal.intensity_limits,
                self.detector_panel.vertical.intensity_limits,
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

    def _validate_project_admission(
        self, candidate: Project, *, selected_id: UUID | None = None
    ) -> None:
        view = self._capture_view()
        if selected_id is not None:
            view = replace(view, selected_acquisition_id=selected_id, detector=None)
        project_to_document(
            ProjectDocument(candidate, view), self._project_path or self._recovery_path()
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
        candidate = replace(self.project, name=name)
        self._validate_project_admission(candidate)
        self.project = candidate
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
        candidate = self.project.rename_acquisition(self.selected_acquisition_id, name)
        self._validate_project_admission(candidate)
        self.project = candidate
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

    def _step_acquisition(self, direction: int) -> None:
        order = [item.acquisition_id for item in self.project.acquisitions]
        if not order or direction not in (-1, 1):
            return
        if self.selected_acquisition_id not in order:
            target = 0 if direction > 0 else len(order) - 1
        else:
            target = min(
                max(order.index(self.selected_acquisition_id) + direction, 0), len(order) - 1
            )
        self._activate_acquisition(order[target])

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
        if self._pending_reference is not None:
            kind, path, ids, project_id, revision = self._pending_reference
            self._pending_reference = None
            if project_id == self.project.project_id and revision == self._revision:
                argument = json.dumps({"kind": kind, "path": str(path)}).encode("utf-8")
                self._active_kind = "reference"
                self._active_reference = (kind, path, ids, project_id, revision)
                try:
                    identity = self.jobs.submit(
                        JobRequest(
                            project_id,
                            None,
                            Revisions(data=revision),
                            argument,
                            len(argument),
                            8192,
                            prepare_reference,
                        )
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    self._active_kind = None
                    self._active_reference = None
                    self.statusBar().showMessage(f"Reference check could not start: {exc}")
                else:
                    self._active_generation = identity.generation
                    return
        if self._deferred_import is not None:
            source, acquisition_id, mode, project_id = self._deferred_import
            self._deferred_import = None
            if project_id == self.project.project_id:
                self._submit_import(source, acquisition_id, mode=mode)
            return
        self._start_next_candidate()

    def _start_next_candidate(self) -> None:
        if self.jobs.busy or self._active_kind is not None or self._close_intent:
            return
        while self._candidate_queue:
            candidate_id = self._candidate_queue.popleft()
            candidate = self._candidates.get(candidate_id)
            if candidate is None or candidate.status != "queued":
                continue
            if len(self.project.acquisitions) >= MAX_ACQUISITIONS:
                candidate.status = "failed"
                candidate.detail = f"Project limit: {MAX_ACQUISITIONS} acquisitions"
                continue
            candidate.status = "reading"
            self._active_kind = "batch"
            self._active_candidate_id = candidate_id
            try:
                argument = encode_bounded_path(candidate.path, self._max_import_axis)
                identity = self.jobs.submit(
                    JobRequest(
                        self.project.project_id,
                        candidate_id,
                        Revisions(data=1),
                        argument,
                        len(argument),
                        MAX_RESULT_BYTES,
                        prepare_osc,
                    )
                )
            except (OSError, RuntimeError, ValueError) as exc:
                candidate.status = "failed"
                candidate.detail = str(exc)[:160]
                self._active_kind = None
                self._active_candidate_id = None
                continue
            self._active_generation = identity.generation
            self._refresh_review_table()
            return
        self._refresh_review_table()

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
        for candidate_id in self._candidate_queue:
            queued = self._candidates.get(candidate_id)
            if queued is not None and queued.status == "queued":
                queued.status = "canceled"
                queued.detail = "Interrupted by project open; retry if this project remains"
        self._pending_reference = None
        if self._active_kind in ("import", "relink", "batch", "reference"):
            self.jobs.cancel()
        self._candidate_queue.clear()
        self._refresh_review_table()
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

    def _choose_reference(self, kind: str) -> None:
        ids = self._selected_project_ids()
        if not ids:
            self.statusBar().showMessage("Select acquisitions before choosing a reference")
            return
        extension = "CIF (*.cif)" if kind == "cif" else "Simulation configurations (*.yaml *.yml)"
        filename, _ = QFileDialog.getOpenFileName(
            self, f"Choose {kind} reference", str(Path.home()), extension
        )
        if not filename:
            return
        path = Path(filename).absolute()
        if len(str(path)) > 4096:
            self.statusBar().showMessage("Reference path exceeds 4096 characters")
            return
        if self._pending_reference is not None:
            self.statusBar().showMessage("Finish the pending reference check first")
            return
        self._pending_reference = (kind, path, ids, self.project.project_id, self._revision)
        QTimer.singleShot(0, self._dispatch_pending)

    def _reference_ready(self, identity: JobIdentity, value: object) -> None:
        task = self._active_reference
        self._active_reference = None
        if (
            task is None
            or not isinstance(value, ValidatedReference)
            or identity.project_id != self.project.project_id
            or identity.revisions.data != self._revision
            or (value.kind, value.path) != task[:2]
        ):
            self.statusBar().showMessage("Reference result became stale; choose it again")
            return
        kind, path, ids, _, _ = task
        updated = self.project
        try:
            for acquisition_id in ids:
                if acquisition_id not in {item.acquisition_id for item in updated.acquisitions}:
                    continue
                metadata = next(
                    item.metadata
                    for item in updated.acquisitions
                    if item.acquisition_id == acquisition_id
                )
                provenance = dict(metadata.provenance)
                provenance[f"{kind}_path"] = f"validated {kind} reader"
                changes = {
                    f"{kind}_path": path,
                    f"{kind}_sha256": value.sha256,
                    "provenance": tuple(sorted(provenance.items())),
                }
                if value.material_id:
                    changes["material_id"] = value.material_id
                    provenance["material_id"] = f"validated {kind} reader"
                    changes["provenance"] = tuple(sorted(provenance.items()))
                updated = updated.update_metadata(acquisition_id, **changes)
            self._validate_project_admission(updated)
        except (ProjectFormatError, ValueError) as exc:
            self.statusBar().showMessage(f"Reference binding rejected: {exc}")
            return
        if updated != self.project:
            self.project = updated
            self.refresh_project()
            self._mark_dirty()
        self.statusBar().showMessage(
            f"Validated {kind} reference and SHA-256 for {len(ids)} acquisition(s)"
        )

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
        self._refresh_review_table()

    def _thumbnail_icon(self, acquisition_id: UUID) -> QIcon:
        item = self._thumbnails.get(acquisition_id)
        if item is None:
            return QIcon()
        pixels, rows, columns = item
        image = QImage(pixels, columns, rows, columns, QImage.Format.Format_Grayscale8)
        return QIcon(QPixmap.fromImage(image.copy()))

    def _refresh_review_table(self) -> None:
        table = self.review_table
        selected_ids = set(self._selected_review_ids())
        current_item = table.item(table.currentRow(), 0) if table.currentRow() >= 0 else None
        current_id = (
            current_item.data(Qt.ItemDataRole.UserRole) if current_item is not None else None
        )
        if not selected_ids and self.selected_acquisition_id is not None:
            selected_ids.add(self.selected_acquisition_id)
        selected_values = {str(item) for item in selected_ids}
        table.blockSignals(True)
        table.clearSelection()
        table.setSortingEnabled(False)
        candidate_rows = [item for item in self._candidates.values() if item.status != "imported"]
        table.setRowCount(len(self.project.acquisitions) + len(candidate_rows))
        for row, acquisition in enumerate(self.project.acquisitions):
            metadata = acquisition.metadata
            check = self._source_checks.get(acquisition.acquisition_id)
            duplicate_count = (
                sum(
                    other.source_sha256 == acquisition.source_sha256
                    for other in self.project.acquisitions
                )
                - 1
            )
            status = check.state if check is not None else "verified"
            if duplicate_count:
                status += f" · exact-content duplicate ({duplicate_count})"
            if (
                metadata.detector_setup
                and metadata.native_shape
                and any(
                    other.acquisition_id != acquisition.acquisition_id
                    and other.metadata.detector_setup == metadata.detector_setup
                    and other.metadata.native_shape is not None
                    and other.metadata.native_shape != metadata.native_shape
                    for other in self.project.acquisitions
                )
            ):
                status += " · detector-setup shape conflict"
            if metadata.proposals:
                status += f" · {len(metadata.proposals)} unconfirmed suggestion(s)"
            source = self._candidates.get(acquisition.acquisition_id)
            if source is not None and source.detail:
                status += f" · {source.detail}"
            references = ", ".join(
                value
                for value in (
                    metadata.material_id,
                    metadata.cif_path.name if metadata.cif_path else None,
                    metadata.configuration_path.name if metadata.configuration_path else None,
                )
                if value
            )
            cells = (
                "",
                acquisition.source_path.name,
                metadata.role or "unknown",
                " / ".join(value or "unknown" for value in (metadata.specimen, metadata.mount)),
                "unknown"
                if metadata.incidence_rad is None
                else f"{math.degrees(metadata.incidence_rad):.6g}",
                "unknown" if metadata.exposure_s is None else f"{metadata.exposure_s:.6g}",
                "unknown"
                if metadata.native_shape is None
                else f"{metadata.native_shape[0]} x {metadata.native_shape[1]}",
                metadata.detector_setup or "unknown",
                references or "unknown",
                metadata.calibrant_id or "unknown",
                status,
            )
            for column, value in enumerate(cells):
                cell = QTableWidgetItem(value)
                cell.setData(Qt.ItemDataRole.UserRole, str(acquisition.acquisition_id))
                if column == 0:
                    cell.setIcon(self._thumbnail_icon(acquisition.acquisition_id))
                table.setItem(row, column, cell)
        for offset, candidate in enumerate(candidate_rows):
            row = len(self.project.acquisitions) + offset
            cells = (
                "",
                candidate.path.name,
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                f"{candidate.status}: {candidate.detail}",
            )
            for column, value in enumerate(cells):
                cell = QTableWidgetItem(value)
                cell.setData(Qt.ItemDataRole.UserRole, str(candidate.acquisition_id))
                table.setItem(row, column, cell)
        table.setSortingEnabled(True)
        for row in range(table.rowCount()):
            cell = table.item(row, 0)
            if cell is None:
                continue
            row_id = cell.data(Qt.ItemDataRole.UserRole)
            if row_id in selected_values:
                table.selectionModel().select(
                    table.model().index(row, 0),
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
            if row_id == current_id:
                table.setCurrentCell(row, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
        self.filmstrip.blockSignals(True)
        self.filmstrip.clear()
        for acquisition in self.project.acquisitions:
            item = QListWidgetItem(
                self._thumbnail_icon(acquisition.acquisition_id), acquisition.name
            )
            item.setData(Qt.ItemDataRole.UserRole, str(acquisition.acquisition_id))
            self.filmstrip.addItem(item)
        self.filmstrip.blockSignals(False)
        table.blockSignals(False)

    def _review_selection_changed(self) -> None:
        row = self.review_table.currentRow()
        item = self.review_table.item(row, 0) if row >= 0 else None
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if value is not None:
            self._selection_from_review = True
            try:
                self._activate_acquisition(UUID(value))
            finally:
                self._selection_from_review = False

    def _filmstrip_clicked(self, item: QListWidgetItem) -> None:
        self._activate_acquisition(UUID(item.data(Qt.ItemDataRole.UserRole)))

    def _activate_acquisition(self, acquisition_id: UUID) -> None:
        root = self.project_tree.topLevelItem(0)
        if root is None:
            return
        for index in range(root.childCount()):
            item = root.child(index)
            if item.data(0, Qt.ItemDataRole.UserRole) == str(acquisition_id):
                self.project_tree.setCurrentItem(item)
                return

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
        metadata = acquisition.metadata
        angle = (
            "unknown"
            if metadata.incidence_rad is None
            else f"{math.degrees(metadata.incidence_rad):.6g} deg (commanded)"
        )
        exposure = "unknown" if metadata.exposure_s is None else f"{metadata.exposure_s:.6g} s"
        references = (
            ", ".join(
                value
                for value in (
                    metadata.material_id,
                    str(metadata.cif_path) if metadata.cif_path else None,
                    str(metadata.configuration_path) if metadata.configuration_path else None,
                    metadata.calibrant_id,
                )
                if value
            )
            or "unknown"
        )
        missing = []
        if metadata.role is None:
            missing.append("role")
        if metadata.detector_setup is None:
            missing.append("detector setup")
        if metadata.role == "sample":
            if metadata.incidence_rad is None:
                missing.append("commanded incidence")
            if metadata.exposure_s is None:
                missing.append("exposure")
            if metadata.material_id is None and metadata.cif_path is None:
                missing.append("material/CIF")
        if metadata.role == "calibrant" and metadata.calibrant_id is None:
            missing.append("calibrant reference")
        source_detail = (
            "; ".join(f"{field}: {origin}" for field, origin in metadata.provenance) or "none"
        )
        proposals = (
            "; ".join(
                f"{field}={value} from {origin} (unconfirmed)"
                for field, value, origin in metadata.proposals
            )
            or "none"
        )
        self.selection_label.setText(
            f"{acquisition.name}\n{details}\n{source_state}\n"
            f"Source: {acquisition.source_path}\nProject: {self._project_path or 'unsaved'}\n"
            f"Role: {metadata.role or 'unknown'} · Specimen: {metadata.specimen or 'unknown'} · Mount: {metadata.mount or 'unknown'}\n"
            f"Angle: {angle} · Exposure: {exposure}\n"
            f"Detector setup: {metadata.detector_setup or 'unknown'} · References: {references}\n"
            f"Metadata origin: {source_detail}\nSuggestions: {proposals}\n"
            f"Missing for review: {', '.join(missing) if missing else 'none declared'}\n"
            "Raw browsing remains available; no fitting, subtraction or mask application is implied."
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
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Import files",
            str(Path.home()),
            "OSC images (*.osc *.OSC *.osc.gz *.OSC.GZ)",
        )
        if filenames:
            self.start_import_files(Path(filename) for filename in filenames)

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Review one folder", str(Path.home()))
        if not folder:
            return
        paths = []
        try:
            with os.scandir(folder) as entries:
                for index, entry in enumerate(entries):
                    if index >= 256:
                        raise ValueError("Folder has more than 256 direct entries")
                    if entry.is_file():
                        paths.append(Path(entry.path))
                    if len(paths) > MAX_IMPORT_CANDIDATES:
                        raise ValueError(
                            f"Folder has more than {MAX_IMPORT_CANDIDATES} direct files"
                        )
        except (OSError, ValueError) as exc:
            self._show_state(
                "error",
                "Folder scope too large",
                f"{exc}. Subfolders were not scanned.",
            )
            return
        paths.sort()
        names = "\n".join(path.name for path in paths[:20])
        detail = f"{len(paths)} direct files in {folder}; no subfolders.\n{names}"
        if len(paths) > 20:
            detail += f"\n… and {len(paths) - 20} more"
        if (
            QMessageBox.question(
                self, "Review folder candidates", detail + "\nImport this explicit list?"
            )
            == QMessageBox.StandardButton.Yes
        ):
            try:
                self.start_import_files(paths)
            except ValueError as exc:
                self._show_state("error", "Folder scope rejected", str(exc))

    def start_import(self, path: Path) -> None:
        self.start_import_files((path,))

    def start_import_files(self, paths: object) -> tuple[UUID, ...]:
        sources = tuple(Path(path).absolute() for path in paths)
        if not sources or len(sources) > MAX_IMPORT_CANDIDATES:
            raise ValueError(f"Choose 1 to {MAX_IMPORT_CANDIDATES} files")
        encoded = sum(len(os.fsencode(source)) for source in sources)
        if encoded > MAX_IMPORT_PATH_BYTES or any(len(str(source)) > 4096 for source in sources):
            raise ValueError("Import candidate paths exceed the bounded request limit")
        if len(self._candidates) + len(sources) > MAX_IMPORT_CANDIDATES:
            raise ValueError("Import review is full; clear completed candidates first")
        if not self.project.acquisitions and self.selected_acquisition_id is None:
            self._batch_auto_select = True
        added = []
        for source in sources:
            candidate = ImportCandidate(
                uuid4(),
                source,
                "queued" if source.name.lower().endswith((".osc", ".osc.gz")) else "unsupported",
                ""
                if source.name.lower().endswith((".osc", ".osc.gz"))
                else "Only OSC/OSC.GZ is supported",
            )
            self._candidates[candidate.acquisition_id] = candidate
            added.append(candidate.acquisition_id)
            if candidate.status == "queued":
                self._candidate_queue.append(candidate.acquisition_id)
        self._refresh_review_table()
        QTimer.singleShot(0, self._dispatch_pending)
        return tuple(added)

    def retry_candidate(self, candidate_id: UUID) -> None:
        candidate = self._candidates[candidate_id]
        if candidate.status not in ("failed", "canceled"):
            raise ValueError("Only failed or canceled candidates can be retried")
        candidate.status = "queued"
        candidate.detail = ""
        self._candidate_queue.append(candidate_id)
        self._refresh_review_table()
        QTimer.singleShot(0, self._dispatch_pending)

    def _selected_review_ids(self) -> tuple[UUID, ...]:
        rows = sorted({index.row() for index in self.review_table.selectedIndexes()})
        return tuple(
            UUID(self.review_table.item(row, 0).data(Qt.ItemDataRole.UserRole))
            for row in rows
            if self.review_table.item(row, 0) is not None
        )

    def _selected_project_ids(self) -> tuple[UUID, ...]:
        valid = {item.acquisition_id for item in self.project.acquisitions}
        review_ids = self._selected_review_ids()
        selected = tuple(item for item in review_ids if item in valid)
        if not review_ids and self.selected_acquisition_id in valid:
            return (self.selected_acquisition_id,)
        return selected

    def _edit_selected_metadata(self) -> None:
        ids = self._selected_project_ids()
        if not ids:
            self.statusBar().showMessage("Select one or more admitted acquisitions")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Apply metadata to {len(ids)} selected acquisition(s)")
        layout = QFormLayout(dialog)
        guidance = QLabel(
            "Blank keeps a field. Enter ? to clear it. Angle is degrees; stored value is radians. Nothing is inferred from filenames."
        )
        guidance.setWordWrap(True)
        layout.addRow(guidance)
        role = QComboBox()
        for label, value in (
            ("Keep role", None),
            ("Unknown", "?"),
            ("Sample", "sample"),
            ("Calibrant", "calibrant"),
            ("Dark", "dark"),
            ("Mask", "mask"),
        ):
            role.addItem(label, value)
        layout.addRow("Role", role)
        fields = {}
        for field_name, label in (
            ("specimen", "Specimen"),
            ("mount", "Mount"),
            ("incidence_deg", "Commanded incidence (deg)"),
            ("exposure_s", "Exposure (s)"),
            ("detector_setup", "Detector setup"),
            ("material_id", "Material phase ID (label only)"),
        ):
            editor = QLineEdit()
            editor.setMaxLength(256)
            layout.addRow(label, editor)
            fields[field_name] = editor
        calibrant = QComboBox()
        calibrant.addItem("Keep calibrant", None)
        calibrant.addItem("Unknown", "?")
        calibrant.addItem("hBN · Cu K-alpha · five declared rings", "hbn_cu_ka_5rings")
        layout.addRow("Calibrant preset", calibrant)
        associations = {}
        for name, linked_role in (("dark_acquisition_id", "dark"), ("mask_acquisition_id", "mask")):
            picker = QComboBox()
            picker.addItem("Keep association", None)
            picker.addItem("None", "?")
            for item in self.project.acquisitions:
                if item.metadata.role == linked_role and item.acquisition_id not in ids:
                    picker.addItem(f"{item.name} · {item.acquisition_id}", str(item.acquisition_id))
            layout.addRow(f"{linked_role.title()} association", picker)
            associations[name] = picker
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        mapping = ["role", *fields, "calibrant_id", *associations]
        values = [
            role.currentData(),
            *(editor.text().strip() for editor in fields.values()),
            calibrant.currentData(),
            *(picker.currentData() for picker in associations.values()),
        ]
        row = tuple("" if value is None else value for value in values)
        try:
            updated = apply_mapped_rows(
                self.project,
                tuple(row for _ in ids),
                tuple(mapping),
                selected_ids=ids,
                origin="manual review",
            )
            self._validate_project_admission(updated)
        except (ProjectFormatError, ValueError) as exc:
            QMessageBox.warning(self, "Metadata rejected", str(exc))
            return
        if updated != self.project:
            self.project = updated
            self.refresh_project()
            self._mark_dirty()

    def _confirm_selected_proposals(self) -> None:
        ids = self._selected_project_ids()
        updated = self.project
        try:
            for acquisition_id in ids:
                metadata = next(
                    item.metadata
                    for item in updated.acquisitions
                    if item.acquisition_id == acquisition_id
                )
                if not metadata.proposals:
                    continue
                changes = {}
                provenance = dict(metadata.provenance)
                for field_name, supplied, origin in metadata.proposals:
                    changes[field_name] = float(supplied)
                    provenance[field_name] = f"confirmed {origin}"
                updated = updated.update_metadata(
                    acquisition_id,
                    **changes,
                    provenance=tuple(sorted(provenance.items())),
                    proposals=(),
                )
            self._validate_project_admission(updated)
        except (ProjectFormatError, ValueError) as exc:
            QMessageBox.warning(self, "Suggestion rejected", str(exc))
            return
        if updated != self.project:
            self.project = updated
            self.refresh_project()
            self._mark_dirty()
            self.statusBar().showMessage(
                "Confirmed selected filename suggestions as commanded values"
            )

    def _map_metadata_text(self, delimiter: str) -> None:
        origin = "spreadsheet paste"
        target_ids = self._selected_project_ids()
        if delimiter == ",":
            filename, _ = QFileDialog.getOpenFileName(
                self, "Choose metadata CSV", str(Path.home()), "CSV (*.csv)"
            )
            if not filename:
                return
            path = Path(filename)
            try:
                with path.open("rb") as stream:
                    encoded = stream.read(64 * 1024 + 1)
                if len(encoded) > 64 * 1024:
                    raise ProjectFormatError("metadata CSV exceeds 64 KiB")
                text = encoded.decode("utf-8-sig")
            except (OSError, UnicodeError, ProjectFormatError) as exc:
                QMessageBox.warning(self, "CSV unavailable", str(exc))
                return
            origin = f"CSV: {path.name}"
        else:
            clipboard_text = QApplication.clipboard().text()
            if len(clipboard_text.encode("utf-8")) > 64 * 1024:
                QMessageBox.warning(self, "Paste rejected", "Spreadsheet paste exceeds 64 KiB")
                return
            dialog = QDialog(self)
            dialog.setWindowTitle("Paste spreadsheet metadata with a header row")
            layout = QVBoxLayout(dialog)
            editor = QPlainTextEdit()
            editor.setPlainText(clipboard_text)
            layout.addWidget(editor)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            text = editor.toPlainText()
        try:
            parsed = parse_table_text(text, delimiter=delimiter)
            if len(parsed) < 2:
                raise ProjectFormatError("metadata table needs a header and at least one data row")
            headers, rows = parsed[0], parsed[1:]
            if any(len(row) != len(headers) for row in rows):
                raise ProjectFormatError("metadata rows must match header column count")
        except (ProjectFormatError, csv.Error) as exc:
            QMessageBox.warning(self, "Metadata table rejected", str(exc))
            return
        mapping_dialog = QDialog(self)
        mapping_dialog.setWindowTitle("Map columns and preview metadata")
        layout = QFormLayout(mapping_dialog)
        preview = QLabel("\n".join(" | ".join(row) for row in parsed[:6]))
        preview.setWordWrap(True)
        layout.addRow("First rows", preview)
        pickers = []
        for index, header in enumerate(headers):
            picker = QComboBox()
            picker.addItem("Ignore", None)
            for field_name in MAPPED_FIELDS:
                picker.addItem(field_name, field_name)
            if header.strip() in MAPPED_FIELDS:
                picker.setCurrentIndex(MAPPED_FIELDS.index(header.strip()) + 1)
            layout.addRow(f"Column {index + 1}: {header}", picker)
            pickers.append(picker)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(mapping_dialog.accept)
        buttons.rejected.connect(mapping_dialog.reject)
        layout.addRow(buttons)
        if mapping_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            updated = apply_mapped_rows(
                self.project,
                rows,
                tuple(picker.currentData() for picker in pickers),
                selected_ids=target_ids,
                origin=origin,
            )
            self._validate_project_admission(updated)
        except (ProjectFormatError, ValueError) as exc:
            QMessageBox.warning(self, "Metadata mapping rejected", str(exc))
            return
        if updated != self.project:
            self.project = updated
            self.refresh_project()
            self._mark_dirty()

    def _choose_metadata_export(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export acquisition metadata",
            str(Path.home() / "acquisitions.csv"),
            "CSV (*.csv)",
        )
        if not filename:
            return
        path = Path(filename).absolute()
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        if path == self._project_path or any(
            path == item.source_path for item in self.project.acquisitions
        ):
            QMessageBox.warning(
                self,
                "Export rejected",
                "Choose a destination separate from project and source files",
            )
            return
        try:
            path.write_text(metadata_csv(self.project), encoding="utf-8", newline="")
        except (OSError, ProjectFormatError) as exc:
            QMessageBox.warning(self, "Metadata export failed", str(exc))
            return
        self.statusBar().showMessage(
            f"Exported metadata with explicit degrees and provenance: {path}"
        )

    def _retry_review_selection(self) -> None:
        for candidate_id in self._selected_review_ids():
            candidate = self._candidates.get(candidate_id)
            if candidate is not None and candidate.status in ("failed", "canceled"):
                if not candidate.path.exists():
                    filename, _ = QFileDialog.getOpenFileName(
                        self,
                        "Locate failed OSC candidate",
                        str(candidate.path.parent),
                        "OSC images (*.osc *.OSC *.osc.gz *.OSC.GZ)",
                    )
                    if not filename:
                        continue
                    candidate.path = Path(filename).absolute()
                self.retry_candidate(candidate_id)

    def _cancel_review_selection(self) -> None:
        for candidate_id in self._selected_review_ids():
            if candidate_id in self._candidates:
                self.cancel_candidate(candidate_id)

    def _remove_review_selection(self) -> None:
        ids = self._selected_review_ids()
        if not ids:
            return
        if (
            QMessageBox.question(
                self,
                "Remove references",
                f"Remove {len(ids)} selected rows from this project? Source files remain untouched.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        changed = False
        for acquisition_id in ids:
            candidate = self._candidates.pop(acquisition_id, None)
            if candidate is not None and candidate.status == "reading":
                self.jobs.cancel()
            if candidate is not None and candidate.status == "queued":
                self._candidate_queue = deque(
                    item for item in self._candidate_queue if item != acquisition_id
                )
            if any(item.acquisition_id == acquisition_id for item in self.project.acquisitions):
                if self._deferred_import is not None and self._deferred_import[1] == acquisition_id:
                    self._deferred_import = None
                if (
                    self._active_kind in ("import", "relink")
                    and self._active_load_id == acquisition_id
                ):
                    self.jobs.invalidate()
                self.project = self.project.remove_acquisition(acquisition_id)
                self._resident_planes.pop(acquisition_id, None)
                self._thumbnails.pop(acquisition_id, None)
                self._source_checks.pop(acquisition_id, None)
                if self.selected_acquisition_id == acquisition_id:
                    self.selected_acquisition_id = None
                if self._visible_acquisition_id == acquisition_id:
                    self._visible_acquisition_id = None
                changed = True
        if changed:
            self._mark_dirty()
        self.refresh_project()

    def cancel_candidate(self, candidate_id: UUID) -> None:
        candidate = self._candidates[candidate_id]
        if candidate.status == "queued":
            candidate.status = "canceled"
            candidate.detail = "Canceled before decode"
            self._candidate_queue = deque(
                item for item in self._candidate_queue if item != candidate_id
            )
        elif candidate_id == self._active_candidate_id and candidate.status == "reading":
            self.jobs.cancel()
        self._refresh_review_table()

    def _cancel_current(self) -> None:
        if self._active_kind == "batch":
            for candidate_id in self._candidate_queue:
                candidate = self._candidates.get(candidate_id)
                if candidate is not None:
                    candidate.status = "canceled"
                    candidate.detail = "Batch canceled"
            self._candidate_queue.clear()
            self._refresh_review_table()
        self.jobs.cancel()

    def _submit_import(
        self, source: Path, acquisition_id: UUID, *, mode: Literal["import", "relink"] = "import"
    ) -> None:
        if (
            self._active_kind in ("save", "discard", "open")
            or self.jobs.busy
            or self._active_kind is not None
            or self._write_queue
            or self._pending_open
        ):
            self._deferred_import = (source, acquisition_id, mode, self.project.project_id)
            self.statusBar().showMessage("Import queued behind project I/O")
            return
        acquisition = next(
            (item for item in self.project.acquisitions if item.acquisition_id == acquisition_id),
            None,
        )
        if acquisition is None:
            self.statusBar().showMessage("Removed acquisition load ignored")
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
        previous_load = self._active_load_id
        previous_revision = self._active_load_revision
        previous_hash = self._active_load_hash
        self._active_kind = mode
        self._active_generation = None
        self._active_load_id = acquisition_id
        self._active_load_revision = self._revision
        self._active_load_hash = acquisition.source_sha256
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
            self._active_load_id = previous_load
            self._active_load_revision = previous_revision
            self._active_load_hash = previous_hash
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
        try:
            self.start_import_files(paths)
        except ValueError as exc:
            self._show_state("error", "Import scope rejected", str(exc))
        event.acceptProposedAction()

    def _batch_ready(self, identity: JobIdentity, value: object) -> None:
        candidate = self._candidates.get(identity.acquisition_id)
        if (
            candidate is None
            or candidate.status != "reading"
            or identity.project_id != self.project.project_id
            or identity.generation != self.jobs.latest_generation
            or identity.revisions != Revisions(data=1)
            or not isinstance(value, PreparedOsc)
            or candidate.path != value.source_path
        ):
            return
        if len(self.project.acquisitions) >= MAX_ACQUISITIONS:
            candidate.status = "failed"
            candidate.detail = "Project acquisition limit reached"
            return
        duplicate_ids = [
            item.acquisition_id
            for item in self.project.acquisitions
            if item.source_sha256 == value.decoded_sha256
        ]
        metadata = AcquisitionMetadata(
            native_shape=value.native_counts.shape,
            proposals=filename_angle_proposal(value.source_path),
        )
        acquisition = Acquisition.create(
            value.source_path.name,
            value.source_path,
            value.decoded_sha256,
            acquisition_id=candidate.acquisition_id,
        )
        acquisition = replace(acquisition, metadata=metadata)
        next_project = replace(self.project, acquisitions=(*self.project.acquisitions, acquisition))
        select_on_admission = self._batch_auto_select and self.selected_acquisition_id is None
        try:
            self._validate_project_admission(
                next_project,
                selected_id=acquisition.acquisition_id if select_on_admission else None,
            )
        except (ProjectFormatError, ValueError) as exc:
            candidate.status = "failed"
            candidate.detail = f"Project admission rejected: {exc}"
            return
        if select_on_admission:
            try:
                self._publish_plane(acquisition.acquisition_id, value)
            except ValueError as exc:
                candidate.status = "failed"
                candidate.detail = f"Detector publication rejected: {exc}"
                return
        self.project = next_project
        candidate.status = "imported"
        candidate.detail = (
            f"Exact decoded-content duplicate of {len(duplicate_ids)} acquisition(s)"
            if duplicate_ids
            else "Decoded source verified"
        )
        self._source_checks[acquisition.acquisition_id] = SourceCheck(
            acquisition.acquisition_id, "verified"
        )
        self._thumbnails[acquisition.acquisition_id] = (
            value.thumbnail.tobytes(),
            *value.thumbnail.shape,
        )
        if sum(len(item[0]) for item in self._thumbnails.values()) > MAX_THUMBNAIL_BYTES:
            raise RuntimeError("thumbnail residency exceeded its declared cap")
        self._remember_plane(acquisition.acquisition_id, value)
        if select_on_admission:
            self.selected_acquisition_id = acquisition.acquisition_id
            self._batch_auto_select = False
        self.refresh_project()
        self._refresh_review_table()
        self._mark_dirty()

    def _remember_plane(self, acquisition_id: UUID, value: PreparedOsc) -> None:
        self._resident_planes[acquisition_id] = value
        self._resident_planes.move_to_end(acquisition_id)
        while len(self._resident_planes) > MAX_CACHED_PLANES:
            victim = next(
                item for item in self._resident_planes if item != self._visible_acquisition_id
            )
            del self._resident_planes[victim]

    def _publish_plane(self, acquisition_id: UUID, value: PreparedOsc) -> None:
        self.detector_panel.set_prepared_image(
            value.native_counts,
            value.display,
            value.profiles,
            value.full_profiles,
            value.low_value,
            value.high_value,
            value.max_value,
            value.min_positive,
            acquisition_id,
        )
        self._visible_acquisition_id = acquisition_id
        self._visible_details = (
            f"Native detector: {value.native_counts.shape[0]} rows x "
            f"{value.native_counts.shape[1]} columns\n"
            f"OSC header: version {value.version}, {value.byte_order} endian"
        )

    def _import_ready(self, identity: JobIdentity, value: object) -> None:
        if (
            identity.project_id != self.project.project_id
            or identity.generation != self.jobs.latest_generation
            or identity.acquisition_id is None
            or identity.acquisition_id != self.selected_acquisition_id
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
        if existing is None:
            self.statusBar().showMessage("Removed acquisition ignored · No image applied")
            return
        if (
            self._active_load_revision != self._revision
            or self._active_load_hash != existing.source_sha256
        ):
            self._deferred_import = (
                existing.source_path,
                existing.acquisition_id,
                "import",
                self.project.project_id,
            )
            self.statusBar().showMessage(
                "Project changed during image load · Reloading current source"
            )
            return
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
        acquisition = replace(existing, source_path=value.source_path) if relinking else existing
        if relinking and acquisition.source_path != existing.source_path:
            candidate_project = replace(
                self.project,
                acquisitions=tuple(
                    acquisition if item.acquisition_id == acquisition.acquisition_id else item
                    for item in self.project.acquisitions
                ),
            )
            try:
                self._validate_project_admission(candidate_project)
            except (ProjectFormatError, ValueError) as exc:
                self._show_state("error", "Relink rejected", str(exc))
                self.statusBar().showMessage("Relink exceeds project document limit")
                return
        try:
            self._publish_plane(identity.acquisition_id, value)
        except ValueError as exc:
            self._show_state("error", "Unsupported detector size", str(exc))
            self.statusBar().showMessage("Unsupported detector size · No image applied")
            return
        if relinking and acquisition.source_path != existing.source_path:
            self.project = replace(
                self.project,
                acquisitions=tuple(
                    acquisition if item.acquisition_id == acquisition.acquisition_id else item
                    for item in self.project.acquisitions
                ),
            )
        self.selected_acquisition_id = acquisition.acquisition_id
        self._remember_plane(acquisition.acquisition_id, value)
        self._thumbnails[acquisition.acquisition_id] = (
            value.thumbnail.tobytes(),
            *value.thumbnail.shape,
        )
        self._source_checks[acquisition.acquisition_id] = SourceCheck(
            acquisition.acquisition_id, "verified"
        )
        self.refresh_project()
        self._apply_restored_view()
        if relinking and acquisition.source_path != existing.source_path:
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
        if detector.profile_roi is not None:
            _, c1, _, r1 = detector.profile_roi
            if c1 > view.image.shape[1] or r1 > view.image.shape[0]:
                self._show_state(
                    "error", "Saved view unavailable", "Inspection ROI is outside this detector."
                )
                return
        self._restoring_view = True
        try:
            current_dpr = view.devicePixelRatioF()
            view.zoom = detector.zoom
            if detector.scale_mode == "native":
                view.zoom = 1.0 / (
                    current_dpr
                    * min(view.width() / view.image.shape[1], view.height() / view.image.shape[0])
                )
            elif detector.scale_mode == "fit":
                view.zoom = 1.0
            else:
                minimum, maximum = view._zoom_bounds()
                view.zoom = min(max(view.zoom, minimum), maximum)
            view.scale_mode = detector.scale_mode
            view.pan = QPointF(
                detector.pan_x_px
                * (
                    detector.device_pixel_ratio / current_dpr
                    if detector.device_pixel_ratio is not None
                    else 1.0
                ),
                detector.pan_y_px
                * (
                    detector.device_pixel_ratio / current_dpr
                    if detector.device_pixel_ratio is not None
                    else 1.0
                ),
            )
            view.crosshair = (detector.column_px, detector.row_px)
            view.show_image = detector.show_image
            view.show_crosshair = detector.show_crosshair
            view.show_markers = detector.show_markers
            view.set_levels(detector.low_value, detector.high_value, mode=detector.contrast_mode)
            self.detector_panel._sync_controls()
            self.detector_panel.restore_profile_state(detector)
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
        self._batch_auto_select = False
        self._deferred_import = None
        self._mark_dirty()
        if not self._selection_from_review:
            self.review_table.blockSignals(True)
            self.review_table.clearSelection()
            if selected_id is not None:
                for row in range(self.review_table.rowCount()):
                    cell = self.review_table.item(row, 0)
                    if cell is not None and cell.data(Qt.ItemDataRole.UserRole) == str(selected_id):
                        self.review_table.selectRow(row)
                        break
            self.review_table.blockSignals(False)
        self.filmstrip.blockSignals(True)
        for index in range(self.filmstrip.count()):
            item = self.filmstrip.item(index)
            item.setSelected(item.data(Qt.ItemDataRole.UserRole) == str(selected_id))
        self.filmstrip.blockSignals(False)
        self._update_selection_label()
        had_work = self.jobs.busy and self._active_kind not in ("save", "discard", "batch")
        if self._active_kind not in ("save", "discard", "batch"):
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
                resident = self._resident_planes.get(selected_id)
                if resident is not None:
                    self._resident_planes.move_to_end(selected_id)
                    self._publish_plane(selected_id, resident)
                    self._apply_restored_view()
                    self._show_state(
                        "empty", "Image ready", "Resident source verified at admission"
                    )
                else:
                    self._submit_import(acquisition.source_path, acquisition.acquisition_id)
            else:
                self._show_state(
                    "error",
                    "Source unavailable",
                    f"{check.state}: {check.detail}. Use Relink OSC with the original bytes.",
                )

    def _job_state_changed(self, summary: JobSummary) -> None:
        state = summary.state
        terminal = state in (JobState.COMPLETED, JobState.CANCELED, JobState.FAILED)
        if summary.identity.generation != self.jobs.latest_generation:
            if terminal and summary.identity.generation == self._active_generation:
                self._active_kind = None
                self._active_generation = None
                self._active_load_id = None
                self._active_load_revision = None
                self._active_load_hash = None
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
            kind in ("import", "relink", "batch", "reference", "open")
            and state in (JobState.QUEUED, JobState.RUNNING)
        )
        if kind == "batch":
            candidate = self._candidates.get(summary.identity.acquisition_id)
            if state in (JobState.FAILED, JobState.CANCELED):
                if candidate is not None:
                    candidate.status = "failed" if state == JobState.FAILED else "canceled"
                    candidate.detail = summary.detail or state.value
                self._active_kind = None
                self._active_generation = None
                self._active_candidate_id = None
                QTimer.singleShot(0, self._dispatch_pending)
            self._refresh_review_table()
            self.statusBar().showMessage(
                f"Import {candidate.path.name if candidate else ''}: {state.value}"
            )
            return
        if kind == "reference":
            if state in (JobState.FAILED, JobState.CANCELED):
                self.statusBar().showMessage(f"Reference check {state.value}: {summary.detail}")
                self._active_kind = None
                self._active_generation = None
                self._active_reference = None
                QTimer.singleShot(0, self._dispatch_pending)
            return
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
            self._active_load_id = None
            self._active_load_revision = None
            self._active_load_hash = None
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
            elif kind == "batch":
                self._batch_ready(identity, value)
            elif kind == "reference":
                self._reference_ready(identity, value)
            elif kind in ("import", "relink"):
                self._import_ready(identity, value)
        finally:
            self._active_kind = None
            self._active_generation = None
            self._active_candidate_id = None
            self._active_load_id = None
            self._active_load_revision = None
            self._active_load_hash = None
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
            self._candidates.clear()
            self._candidate_queue.clear()
            self._batch_auto_select = False
            self._resident_planes.clear()
            self._thumbnails.clear()
            self._pending_view_restore = value.document.view
            self._deferred_import = None
            self._pending_reference = None
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
            self._pending_reference = None
            for candidate_id in self._candidate_queue:
                candidate = self._candidates.get(candidate_id)
                if candidate is not None and candidate.status == "queued":
                    candidate.status = "canceled"
                    candidate.detail = "Window closing"
            self._candidate_queue.clear()
            self._autosave_timer.stop()
            self._write_queue = deque(task for task in self._write_queue if task.explicit)
            if self._active_kind in ("import", "relink", "batch", "reference", "open"):
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
