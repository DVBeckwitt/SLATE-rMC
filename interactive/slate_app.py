"""Native desktop shell for SLATE-rMC; launch with ``python interactive/slate_app.py``."""

import csv
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
from comparison_panel import ComparisonPanel, panel_view_state
from comparison_state import LineSamples, LineWork, detector_frame_key, prepare_line
from detector_panel import DetectorPanel
from experiment_scene import ExperimentScenePanel
from hbn_io import HbnWorkResult, hbn_work, hbn_work_budget
from hbn_panel import HbnPanel
from inspection_export import (
    InspectionExportReceipt,
    comparison_csv,
    external_export_destination,
    inspection_export_request,
    profile_csv,
    publish_inspection_export,
)
from job_lifecycle import (
    MAX_RESULT_BYTES,
    JobIdentity,
    JobOwner,
    JobRequest,
    JobState,
    JobSummary,
    Revisions,
)
from joint_io import JointWorkResult, joint_work, joint_work_budget
from joint_panel import JointPanel
from mask_state import (
    MAX_ACTIONS,
    MaskGesture,
    MaskHistory,
    MaskWork,
    NativeMask,
    PreparedMask,
    prepare_mask,
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
from native_fit_io import PreparedWorkResult, prepared_budget, prepared_work
from native_fit_panel import NativeFitPanel
from native_simulation_io import prepare_native_draft, run_native_simulation
from numeric_fields import PARAMETERS
from osc_import import AXIS_LIMIT, PreparedOsc, encode_bounded_path, prepare_osc
from parameter_state import (
    SessionHistory,
    combined_action,
    description,
    displayed_value,
    draft_action,
    edit_draft,
    freeze_draft,
    metadata_action,
    prepare_numeric_draft,
)
from physical_io import PhysicalResult, physical_work
from physical_panel import PhysicalPanel
from project_io import (
    LoadedProject,
    PublishedProject,
    ReferenceCheck,
    SourceCheck,
    discard_recovery,
    invalidate_reference_checks,
    load_project,
    publish_reference_checks,
    reference_bindings,
    write_project,
)
from project_state import (
    MAX_ACQUISITIONS,
    Acquisition,
    AcquisitionMetadata,
    ComparisonState,
    DetectorViewState,
    NumericDraft,
    Project,
    ProjectDocument,
    ProjectFormatError,
    ProjectViewState,
    SceneViewState,
    project_to_document,
)
from PySide6.QtCore import QBuffer, QIODevice, QItemSelectionModel, QPointF, QSize, Qt, QTimer
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
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
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
    QScrollArea,
    QSizePolicy,
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
from reciprocal_preview import (
    MAX_PREVIEW_BYTES,
    ReciprocalCoverageView,
    ReciprocalPreview,
    prepare_reciprocal_preview,
)
from sample_io import SampleWorkResult, sample_work, sample_work_budget
from sample_panel import SamplePanel
from setup_io import SetupApplication, prepare_setup
from setup_panel import SetupDialog
from simulation_io import (
    SimulationFrame,
    export_simulation,
    prepare_simulation_draft,
    prepare_simulation_profiles,
    reopen_simulation_result,
    run_simulation,
)
from simulation_panel import SimulatorPanel
from simulation_transfer import prepare_simulation_transfer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

MAX_PENDING_WRITES = 8
MAX_PENDING_WRITE_BYTES = 3 * 1024 * 1024
MAX_IMPORT_CANDIDATES = 128
MAX_IMPORT_PATH_BYTES = 512 * 1024
MAX_CACHED_PLANES = 2
MAX_THUMBNAIL_BYTES = 128 * 96 * 96
# A fully populated detector view with native ROI bounds, both intensity ranges,
# all flags and finite double-precision values is below this serialization margin.
# Reserve it at project admission so later browsing cannot exhaust the document cap.
MAX_FUTURE_VIEW_BYTES = 16 * 1024


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


@dataclass(frozen=True, slots=True)
class InspectionExportTask:
    project_id: UUID
    acquisition_id: UUID
    data_revision: int
    acquisition_name: str
    figure: Path
    profiles: Path
    figure_sha256: str
    profiles_sha256: str
    capture_ms: float


@dataclass(slots=True)
class ImportCandidate:
    acquisition_id: UUID
    path: Path
    status: Literal["queued", "reading", "imported", "unsupported", "failed", "canceled"]
    detail: str = ""


class NumericReviewItem(QTableWidgetItem):
    """Sort displayed angle/exposure values numerically, with unknowns last."""

    def __init__(self, text: str, value: float | None) -> None:
        super().__init__(text)
        self.value = value

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, NumericReviewItem):
            return (self.value is None, self.value if self.value is not None else 0.0) < (
                other.value is None,
                other.value if other.value is not None else 0.0,
            )
        return super().__lt__(other)


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


class InspectionTabs(QTabWidget):
    """Use the visible inspection page's minimum, including its tab navigation."""

    def minimumSizeHint(self) -> QSize:
        page = self.currentWidget()
        if page is None:
            return super().minimumSizeHint()
        content, tabs = page.minimumSizeHint(), self.tabBar().minimumSizeHint()
        return QSize(max(content.width(), tabs.width()), content.height() + tabs.height() + 2)

    def sizeHint(self) -> QSize:
        return self.minimumSizeHint()


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
        self._setup_template = None
        self._setup_template_path: Path | None = None
        self._setup_template_hash: str | None = None
        self._setup_dialog: SetupDialog | None = None
        self._setup_context = None
        self._pending_setup_request = None
        self._pending_setup_copy = None
        self._project_path: Path | None = None
        self._revision = 0
        self._saved_revision = -1
        self._draft_revision = -1
        self._save_failure = ""
        self._restoring_view = False
        self._pending_view_restore: ProjectViewState | None = None
        self._pending_scene_restore: SceneViewState | None = None
        self._scene_cameras: dict[UUID, tuple[float, float, float, tuple[float, float, float]]] = {}
        self._source_checks: dict[UUID, SourceCheck] = {}
        self._reference_checks: dict[tuple[UUID, str], ReferenceCheck] = {}
        self._numeric_draft: NumericDraft | None = None
        self._validated_numeric: tuple[UUID, NumericDraft] | None = None
        self._numeric_history = SessionHistory()
        self._mask_history: OrderedDict[UUID, MaskHistory] = OrderedDict()
        self._mask_cache: OrderedDict[UUID, PreparedMask] = OrderedDict()
        self._pending_masks: dict[UUID, tuple[MaskGesture, ...]] = {}
        self._active_mask: tuple[UUID, NativeMask, tuple[MaskGesture, ...]] | None = None
        self._active_profile_target: int | str | None = None
        self._active_profile_query: tuple | None = None
        self._active_line: tuple[int, tuple[object, ...]] | None = None
        self._comparison_frames_pending: deque[UUID] = deque()
        self._deferred_mask_write: tuple[Path, bool, bool, bool] | None = None
        self._launch_snapshot = None
        self._active_numeric: tuple[UUID, int, str, NumericDraft | None] | None = None
        self._reciprocal_cache: OrderedDict[UUID, tuple[tuple[object, ...], ReciprocalPreview]] = (
            OrderedDict()
        )
        self._active_reciprocal: tuple[tuple[object, ...], int, str] | None = None
        self._reciprocal_epoch = 0
        self._reciprocal_detector_q_text: str | None = None
        self._reciprocal_pointer_status: str | None = None
        self._write_queue: deque[WriteTask] = deque()
        self._active_write: WriteTask | None = None
        self._active_export: InspectionExportTask | None = None
        self._active_kind: (
            Literal[
                "import",
                "relink",
                "batch",
                "reference",
                "open",
                "save",
                "discard",
                "export",
                "numeric",
                "reciprocal",
                "mask",
                "line",
                "setup",
                "simulation",
                "hbn",
                "sample",
                "joint",
                "prepared",
                "physical",
            ]
            | None
        ) = None
        self._pending_hbn = None
        self._hbn_context = None
        self._pending_prepared = None
        self._pending_joint = None
        self._prepared_context = None
        self._joint_context = None
        self._pending_physical = None
        self._physical_context = None
        self._pending_sample = None
        self._sample_context = None
        self._pending_simulation = None
        self._simulation_operation = None
        self._simulation_context = None
        self._active_generation: int | None = None
        self._active_load_id: UUID | None = None
        self._active_load_path: Path | None = None
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
        self.physical = PhysicalPanel(self)
        self.jobs = JobOwner(self)
        self.jobs.state_changed.connect(self._job_state_changed)
        self.jobs.progress_changed.connect(self._job_progress)
        self.jobs.result_ready.connect(self._job_result_ready)
        self.jobs.publication_ready.connect(self._simulation_publication)
        tabs.currentChanged.connect(self._simulation_workspace_changed)
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
        self.detector_panel.export_button.clicked.connect(self._choose_inspection_export)
        self.relink_button.clicked.connect(self._choose_relink)
        self.setup_button.clicked.connect(self._show_setup)
        self.open_button.clicked.connect(self._choose_open)
        self.save_button.clicked.connect(self.save_project)
        self.save_as_button.clicked.connect(self.save_project_as)
        self.recover_button.clicked.connect(self._choose_recovery)
        self.rename_button.clicked.connect(self._choose_project_name)
        self.rename_acquisition_button.clicked.connect(self._choose_acquisition_name)
        self.move_up_button.clicked.connect(lambda: self.move_selected_acquisition(-1))
        self.move_down_button.clicked.connect(lambda: self.move_selected_acquisition(1))
        self.numeric_load_button.clicked.connect(self._load_numeric_draft)
        self.numeric_field.currentIndexChanged.connect(self._refresh_numeric_editor)
        self.numeric_apply_button.clicked.connect(self._apply_numeric_value)
        self.numeric_undo_button.clicked.connect(lambda: self._numeric_undo_redo(undo=True))
        self.numeric_redo_button.clicked.connect(lambda: self._numeric_undo_redo(undo=False))
        self.numeric_revert_button.clicked.connect(self._revert_numeric_draft)
        self.numeric_freeze_button.clicked.connect(self._freeze_numeric_draft)
        self.reciprocal_button.clicked.connect(self._request_reciprocal_preview)
        self.previous_shortcut = QShortcut(QKeySequence("Alt+Left"), self)
        self.next_shortcut = QShortcut(QKeySequence("Alt+Right"), self)
        self.previous_shortcut.activated.connect(lambda: self._step_acquisition(-1))
        self.next_shortcut.activated.connect(lambda: self._step_acquisition(1))
        self.workspaces.currentChanged.connect(self._mark_dirty)
        self.detector_panel.view.crosshair_changed.connect(self._mark_dirty)
        self.detector_panel.view.crosshair_changed.connect(self._reciprocal_selection_changed)
        self.detector_panel.view.cursor_changed.connect(self._reciprocal_cursor_changed)
        self.detector_panel.view.view_state_changed.connect(self._mark_dirty)
        self.detector_panel.view.view_state_changed.connect(self._sync_scene)
        self.detector_panel.view.overlays_changed.connect(self._sync_scene)
        self.detector_panel.view.mask_gesture_ready.connect(self._mask_gesture)
        self.detector_panel.view.mask_input_error.connect(self.statusBar().showMessage)
        self.detector_panel.mask_import_button.clicked.connect(self._choose_mask)
        self.detector_panel.mask_undo_button.clicked.connect(
            lambda: self._mask_undo_redo(undo=True)
        )
        self.detector_panel.mask_redo_button.clicked.connect(
            lambda: self._mask_undo_redo(undo=False)
        )
        self.detector_panel.profile_requested.connect(self._request_mask_profiles)
        self.comparison_panel.selection_requested.connect(self._comparison_selection)
        self.comparison_panel.profiles_requested.connect(self._request_mask_profiles)
        self.comparison_panel.cut_requested.connect(self._request_line)
        self.comparison_panel.state_changed.connect(self._mark_dirty)
        self.comparison_panel.frames_requested.connect(self._request_comparison_frames)
        self.comparison_panel.export_requested.connect(self._choose_comparison_export)
        self.comparison_panel.cancel_button.clicked.connect(self._cancel_current)
        self.detector_panel.mask_cancel_button.clicked.connect(self._cancel_current)
        self.scene_panel.view.camera_changed.connect(self._remember_scene_camera)
        self.scene_tabs.currentChanged.connect(self._mark_dirty)
        self.scene_tabs.currentChanged.connect(self._comparison_layout_changed)
        self.scene_tabs.currentChanged.connect(
            lambda _index: QTimer.singleShot(0, self._refresh_comparison)
        )
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
        self._center_layout = center_layout
        center_layout.setContentsMargins(20, 20, 20, 20)
        center_layout.setSpacing(14)
        self.detector_heading = QLabel("Detector view")
        center_layout.addWidget(self.detector_heading)
        self.choose_center_button = QPushButton("Choose beam center")
        self.choose_center_button.clicked.connect(lambda: self._show_hbn(1))
        center_layout.addWidget(self.choose_center_button)
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
        self.scene_tabs = InspectionTabs()
        self.scene_tabs.addTab(self.detector_panel, "Detector / profiles")
        self.scene_panel = ExperimentScenePanel()
        self.scene_tabs.addTab(self.scene_panel, "Experiment 3D")
        self.comparison_panel = ComparisonPanel()
        self.scene_tabs.addTab(self.comparison_panel, "Compare images")
        detector_layout.addWidget(self.scene_tabs, 1)
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
        self.review_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.review_table.setSortingEnabled(True)
        self.review_table.setMinimumHeight(165)
        center_layout.addWidget(self.review_table)
        self.import_button = QPushButton("Import files")
        controls = QGridLayout()
        self.folder_button = QPushButton("Review folder")
        self.retry_button = QPushButton("Retry row")
        self.cancel_row_button = QPushButton("Cancel row")
        self.remove_button = QPushButton("Remove from project")
        self.metadata_button = QPushButton("Apply to selected")
        self.confirm_button = QPushButton("Confirm suggestions")
        self.relink_button = QPushButton("Relink OSC")
        self.relink_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel operation")
        self.cancel_button.setEnabled(False)
        for index, button in enumerate(
            (
                self.import_button,
                self.folder_button,
                self.retry_button,
                self.cancel_row_button,
                self.remove_button,
                self.metadata_button,
                self.confirm_button,
                self.relink_button,
                self.cancel_button,
            )
        ):
            controls.addWidget(button, index // 3, index % 3)
        center_layout.addLayout(controls)
        metadata_controls = QGridLayout()
        self.cif_button = QPushButton("Replace CIF")
        self.configuration_button = QPushButton("Replace configuration")
        self.paste_button = QPushButton("Paste table")
        self.csv_button = QPushButton("Map CSV")
        self.export_metadata_button = QPushButton("Export metadata CSV")
        for index, button in enumerate(
            (
                self.cif_button,
                self.configuration_button,
                self.paste_button,
                self.csv_button,
                self.export_metadata_button,
            )
        ):
            metadata_controls.addWidget(button, index // 3, index % 3)
        self.setup_button = QPushButton("Setup / sources")
        metadata_controls.addWidget(self.setup_button, 1, 2)
        center_layout.addLayout(metadata_controls)
        storage_notice = QLabel(
            "Storage: reference inputs in place by default. Setup / sources reviews reusable defaults, verified copies and identity-preserving relocation."
        )
        storage_notice.setWordWrap(True)
        storage_notice.setObjectName("mutedText")
        center_layout.addWidget(storage_notice)
        center_scroll = QScrollArea()
        center_scroll.setWidgetResizable(True)
        center_scroll.setWidget(center)
        splitter.addWidget(center_scroll)

        inspector = QFrame()
        inspector.setObjectName("sidePanel")
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(18, 18, 18, 18)
        inspector_layout.setSpacing(12)
        inspector_layout.addWidget(QLabel("INSPECTOR"))
        self.hbn = HbnPanel(self)
        self.hbn_button = QPushButton("hBN calibration")
        self.hbn_button.clicked.connect(lambda: self._show_hbn(0))
        inspector_layout.addWidget(self.hbn_button)
        self.sample = SamplePanel(self)
        self.sample_button = QPushButton("Sample geometry series")
        self.sample_button.clicked.connect(self._show_sample)
        inspector_layout.addWidget(self.sample_button)
        self.prepared = NativeFitPanel(self)
        self.prepared_button = QPushButton("Prepared inputs / draft plan")
        self.prepared_button.clicked.connect(lambda: self.prepared.show())
        inspector_layout.addWidget(self.prepared_button)
        self.joint = JointPanel(self)
        self.joint_button = QPushButton("Joint geometry")
        self.joint_button.clicked.connect(self._show_joint)
        inspector_layout.addWidget(self.joint_button)
        self.physical_button = QPushButton("Physical geometry / centers / sensitivity")
        self.physical_button.clicked.connect(self._show_physical)
        inspector_layout.addWidget(self.physical_button)
        self.selection_label = QLabel("No acquisition selected")
        self.selection_label.setObjectName("mutedText")
        self.selection_label.setWordWrap(True)
        self.selection_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.selection_scroll = QScrollArea()
        self.selection_scroll.setWidgetResizable(True)
        self.selection_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.selection_scroll.setMinimumHeight(140)
        self.selection_scroll.setWidget(self.selection_label)
        inspector_layout.addWidget(self.selection_scroll, 1)
        numeric_heading = QLabel("NUMERIC INITIAL VALUES")
        numeric_heading.setWordWrap(True)
        inspector_layout.addWidget(numeric_heading)
        self.numeric_status = QLabel("Load a validated configuration for one acquisition.")
        self.numeric_status.setObjectName("mutedText")
        self.numeric_status.setWordWrap(True)
        inspector_layout.addWidget(self.numeric_status)
        self.numeric_load_button = QPushButton("Load draft")
        self.numeric_load_button.setToolTip("Load numeric draft")
        inspector_layout.addWidget(self.numeric_load_button)
        self.numeric_field = QComboBox()
        for item in PARAMETERS:
            self.numeric_field.addItem(item.label, item.field)
        self.numeric_field.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.numeric_field.setMinimumContentsLength(12)
        inspector_layout.addWidget(self.numeric_field)
        self.numeric_description = QLabel()
        self.numeric_description.setObjectName("mutedText")
        self.numeric_description.setWordWrap(True)
        inspector_layout.addWidget(self.numeric_description)
        self.numeric_value = QLineEdit()
        self.numeric_value.setPlaceholderText("Initial value")
        inspector_layout.addWidget(self.numeric_value)
        numeric_buttons = QGridLayout()
        self.numeric_apply_button = QPushButton("Apply initial")
        self.numeric_undo_button = QPushButton("Undo")
        self.numeric_redo_button = QPushButton("Redo")
        self.numeric_revert_button = QPushButton("Revert draft")
        self.numeric_freeze_button = QPushButton("Freeze")
        self.numeric_freeze_button.setToolTip("Freeze an immutable initial snapshot")
        for index, button in enumerate(
            (
                self.numeric_apply_button,
                self.numeric_undo_button,
                self.numeric_redo_button,
                self.numeric_revert_button,
                self.numeric_freeze_button,
            )
        ):
            numeric_buttons.addWidget(button, index, 0)
        inspector_layout.addLayout(numeric_buttons)
        reciprocal_heading = QLabel("RECIPROCAL COVERAGE")
        reciprocal_heading.setWordWrap(True)
        inspector_layout.addWidget(reciprocal_heading)
        self.reciprocal_button = QPushButton("Map geometry")
        self.reciprocal_button.setToolTip("Map selected geometry")
        inspector_layout.addWidget(self.reciprocal_button)
        self.reciprocal_status = QLabel("Select an acquisition with verified geometry.")
        self.reciprocal_status.setObjectName("mutedText")
        self.reciprocal_status.setWordWrap(True)
        inspector_layout.addWidget(self.reciprocal_status)
        self.reciprocal_view = ReciprocalCoverageView()
        inspector_layout.addWidget(self.reciprocal_view)
        self.reciprocal_cursor = QLabel("Pointer Q unavailable")
        self.reciprocal_cursor.setObjectName("mutedText")
        self.reciprocal_cursor.setWordWrap(True)
        inspector_layout.addWidget(self.reciprocal_cursor)
        self.reciprocal_selection = QLabel("Selected Q unavailable")
        self.reciprocal_selection.setObjectName("mutedText")
        self.reciprocal_selection.setWordWrap(True)
        inspector_layout.addWidget(self.reciprocal_selection)
        reciprocal_limit = QLabel("No reviewed reciprocal features are available.")
        reciprocal_limit.setWordWrap(True)
        inspector_layout.addWidget(reciprocal_limit)
        project_limit = QLabel(
            "Angles and calibration remain unknown. Saving a draft never resumes a solver."
        )
        project_limit.setWordWrap(True)
        project_limit.setObjectName("mutedText")
        inspector_layout.addWidget(project_limit)
        self.inspector_scroll = QScrollArea()
        self.inspector_scroll.setWidgetResizable(True)
        self.inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inspector_scroll.setWidget(inspector)
        splitter.addWidget(self.inspector_scroll)
        self._center_scroll = center_scroll
        self._comparison_hidden_widgets = (
            browser,
            self.detector_heading,
            self.cancel_button,
            self.detector_notice,
            self.filmstrip,
            self.review_table,
            storage_notice,
            self.import_button,
            self.folder_button,
            self.retry_button,
            self.cancel_row_button,
            self.remove_button,
            self.metadata_button,
            self.confirm_button,
            self.relink_button,
            self.cif_button,
            self.configuration_button,
            self.paste_button,
            self.csv_button,
            self.export_metadata_button,
        )
        splitter.setSizes([250, 680, 250])
        return page

    def _comparison_layout_changed(self, index: int) -> None:
        comparing = index == 2
        vertical = QSizePolicy.Policy.Ignored if comparing else QSizePolicy.Policy.Preferred
        self.detector_stack.setSizePolicy(QSizePolicy.Policy.Expanding, vertical)
        self.scene_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, vertical)
        for widget in self._comparison_hidden_widgets:
            widget.setVisible(not comparing)
        self.inspector_scroll.setVisible(not comparing)
        margin, spacing = (8, 4) if comparing else (20, 14)
        self._center_layout.setContentsMargins(margin, margin, margin, margin)
        self._center_layout.setSpacing(spacing)
        self._center_scroll.verticalScrollBar().setValue(0)
        self._center_scroll.horizontalScrollBar().setValue(0)
        QTimer.singleShot(0, self._sync_comparison_height)

    def _sync_comparison_height(self) -> None:
        available = self._center_scroll.viewport().height()
        minimum = (
            self.comparison_panel.minimumSizeHint().height()
            + self.scene_tabs.tabBar().height()
            + 20
        )
        maximum = (
            available if self.scene_tabs.currentIndex() == 2 and available >= minimum else 16777215
        )
        self._center_scroll.widget().setMaximumHeight(maximum)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_center_scroll"):
            QTimer.singleShot(0, self._sync_comparison_height)

    def _build_simulator(self) -> QWidget:
        self.simulator = SimulatorPanel(self)
        return self.simulator

    def _simulation_workspace_changed(self, index: int) -> None:
        if index != 1:
            self._supersede_simulation()

    def _simulation_resource_charge(self) -> dict:
        arrays = {}
        for plane in self._resident_planes.values():
            for name in ("native_counts", "display", "thumbnail"):
                value = getattr(plane, name, None)
                if isinstance(value, np.ndarray):
                    arrays[id(value)] = value
        views = (
            self.detector_panel.view,
            *[p.view for p in self.comparison_panel.panels],
            self.comparison_panel.magnifier,
        )
        for view in views:
            for value in (view.image, view._display, view.mask_reasons):
                if isinstance(value, np.ndarray):
                    arrays[id(value)] = value
        for mask in self._mask_cache.values():
            for name in ("reasons", "inclusion"):
                value = getattr(mask, name, None)
                if isinstance(value, np.ndarray):
                    arrays[id(value)] = value
        for frame in (self.simulator.frame, self.simulator.latest_frame):
            if frame is None:
                continue
            for value in (frame.image, frame.display, *[a for _, a in frame.arrays]):
                if isinstance(value, np.ndarray):
                    arrays[id(value)] = value
            for profile in (frame.profiles, frame.full_profiles):
                if profile is not None:
                    for value in (
                        profile.horizontal,
                        profile.vertical,
                        profile.horizontal_support,
                        profile.vertical_support,
                    ):
                        arrays[id(value)] = value
        scene = self.scene_panel.view
        if scene._image is not None:
            arrays[id(scene._image)] = scene._image
        gpu = sum(view._display.nbytes for view in views if view._display is not None)
        gpu += sum(view.mask_reasons.nbytes for view in views if view.mask_reasons is not None)
        if self.simulator.detector.view._display is not None:
            gpu += self.simulator.detector.view._display.nbytes
        # The scene's retained owner admits at most two native R32F detector textures.
        gpu += len(scene._textures) * 12_000_000 * 4
        if hasattr(self, "physical"):
            physical_scene = self.physical.scene.view
            gpu += (
                max(2 if self.physical.isVisible() else 0, len(physical_scene._textures))
                * 12_000_000
                * 4
            )
            if physical_scene._image is not None:
                arrays[id(physical_scene._image)] = physical_scene._image
        return {
            "other_cpu_bytes": sum(a.nbytes for a in arrays.values())
            + self._numeric_history.bytes_used
            + self.physical.history.bytes_used
            + self.physical.resident_bytes()
            + self.hbn.history.bytes_used
            + sum(a.nbytes for a in self.hbn._spot_arrays)
            + sum(v.nbytes for v in self.hbn.sessions)
            + (0 if self.sample.session is None else 3 * self.sample.session.nbytes)
            + sum(
                len(identity) + len(detail.encode()) + 128
                for identity, detail in self.sample.input_checks.items()
            )
            + (0 if self.joint.session is None else 3 * self.joint.session.nbytes)
            + (0 if self.prepared.session is None else 3 * self.prepared.session.nbytes)
            + (0 if self.prepared.profiles is None else self.prepared.profiles.nbytes)
            + self.prepared.history.bytes_used
            + self.simulator.history.bytes_used
            + self.simulator.native.history.bytes_used
            + sum(history.storage_bytes for history in self._mask_history.values()),
            "other_gpu_bytes": gpu,
        }

    def _show_physical(self):
        self.physical.show()
        self.physical.raise_()
        self.physical.guard(self.physical.load)

    def _supersede_physical(self):
        self._pending_physical = None
        if self._active_kind == "physical":
            self.jobs.invalidate()

    def _request_physical(self, argument, context):
        if self._close_intent or self._pending_open is not None:
            return
        if type(argument) is not bytes or len(argument) > 4 * 1024**2:
            raise ValueError("Physical request exceeds 4 MiB")
        self._pending_physical = (argument, context)
        if self._active_kind == "physical":
            self.jobs.invalidate()
        self.physical.status.setText("Canonical preview requested; waiting for the shared worker")
        QTimer.singleShot(0, self._dispatch_pending)

    def _submit_physical(self, argument, context):
        if context != self.physical.context():
            self.physical.status.setText("Queued physical request is stale; no work launched")
            return
        self._active_kind = "physical"
        self._physical_context = context
        try:
            request = json.loads(argument)
            request["resources"] = sample_work_budget(**self._simulation_resource_charge())
            argument = json.dumps(request, allow_nan=False).encode()
            identity = self.jobs.submit(
                JobRequest(
                    self.project.project_id,
                    None,
                    Revisions(calibration=self.physical.epoch),
                    argument,
                    len(argument),
                    2 * 1024**2,
                    physical_work,
                )
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            self._active_kind = self._physical_context = None
            self.physical.status.setText(f"Physical calculation could not start: {exc}")
        else:
            self._active_generation = identity.generation

    def _physical_current(self, identity):
        return (
            identity.generation == self._active_generation == self.jobs.latest_generation
            and self._physical_context == self.physical.context()
            and not self._close_intent
            and self._pending_open is None
        )

    def _supersede_prepared(self):
        self._pending_prepared = None
        if self._active_kind == "prepared":
            self.jobs.invalidate()
            self.prepared.status.setText("Prepared request superseded; prior description retained")

    def _request_prepared(self, operation, argument, context):
        if self._close_intent or self._pending_open is not None:
            return
        if type(argument) is not bytes or len(argument) > 1024 * 1024:
            raise ValueError("Prepared request exceeds 1 MiB")
        self._pending_prepared = (operation, argument, context)
        if self._active_kind == "prepared":
            self.jobs.invalidate()
        self.prepared.status.setText(operation.title() + " queued on the shared worker")
        QTimer.singleShot(0, self._dispatch_pending)

    def _submit_prepared(self, operation, argument, context):
        if context != self.prepared.context():
            self.prepared.status.setText("Stale prepared request rejected before file work")
            return
        self._active_kind = "prepared"
        self._prepared_context = context
        try:
            request = json.loads(argument)
            request["resources"] = prepared_budget(**self._simulation_resource_charge())
            payload = json.dumps(request, allow_nan=False).encode()
            identity = self.jobs.submit(
                JobRequest(
                    self.project.project_id,
                    None,
                    Revisions(calibration=self.prepared.epoch),
                    payload,
                    len(payload),
                    MAX_RESULT_BYTES,
                    prepared_work,
                )
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            self._active_kind = self._prepared_context = None
            self.prepared.status.setText("Prepared operation could not start: " + str(exc))
        else:
            self._active_generation = identity.generation

    def _prepared_current(self, identity):
        return (
            identity.generation == self._active_generation == self.jobs.latest_generation
            and self._prepared_context == self.prepared.context()
            and not self._close_intent
            and self._pending_open is None
        )

    def _show_joint(self):
        self.joint.refresh()
        self.joint.show()
        self.joint.raise_()

    def _supersede_joint(self):
        self._pending_joint = None
        if self._active_kind == "joint":
            self.jobs.invalidate()
            self.joint.status.setText("Joint request superseded; draining safely")

    def _request_joint(self, operation, argument, context):
        if self._close_intent or self._pending_open is not None:
            return
        if type(argument) is not bytes or len(argument) > 32 * 1024**2:
            raise ValueError("joint request exceeds 32 MiB")
        self._pending_joint = (operation, argument, context)
        if self._active_kind == "joint":
            self.jobs.invalidate()
        self.joint.status.setText(operation.title() + " requested; waiting for the shared worker")
        self.joint.refresh()
        QTimer.singleShot(0, self._dispatch_pending)

    def _submit_joint(self, operation, argument, context):
        if context != self.joint.context():
            self.joint.status.setText("Queued joint request is stale; no work launched")
            return
        self._active_kind = "joint"
        self._joint_context = context
        try:
            request = json.loads(argument)
            request["resources"] = joint_work_budget(**self._simulation_resource_charge())
            argument = json.dumps(request, allow_nan=False).encode()
            identity = self.jobs.submit(
                JobRequest(
                    self.project.project_id,
                    None,
                    Revisions(calibration=self.joint.epoch),
                    argument,
                    len(argument),
                    64 * 1024**2,
                    joint_work,
                )
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            self._active_kind = self._joint_context = None
            self.joint.status.setText(f"Joint geometry could not start: {exc}")
        else:
            self._active_generation = identity.generation
        self.joint.refresh()

    def _joint_current(self, identity):
        return (
            identity.generation == self._active_generation == self.jobs.latest_generation
            and self._joint_context == self.joint.context()
            and not self._close_intent
            and self._pending_open is None
        )

    def _show_sample(self):
        self.sample.refresh()
        self.sample.show()
        self.sample.raise_()

    def _supersede_sample(self):
        self._pending_sample = None
        if self._active_kind == "sample":
            self.jobs.invalidate()
            self.sample.status.setText("Sample request superseded; draining safely")

    def _request_sample(self, operation, argument, context):
        if self._close_intent or self._pending_open is not None:
            return
        if type(argument) is not bytes or len(argument) > 4 * 1024**2:
            raise ValueError("sample request exceeds 4 MiB")
        self._pending_sample = (operation, argument, context)
        if self._active_kind == "sample":
            self.jobs.invalidate()
        self.sample.status.setText(operation.title() + " requested; waiting for the shared worker")
        self.sample.refresh()
        QTimer.singleShot(0, self._dispatch_pending)

    def _submit_sample(self, operation, argument, context):
        if context != self.sample.context():
            self.sample.status.setText("Queued sample request is stale; no work launched")
            return
        self._active_kind = "sample"
        self._sample_context = context
        try:
            request = json.loads(argument)
            request["resources"] = sample_work_budget(**self._simulation_resource_charge())
            argument = json.dumps(request, allow_nan=False).encode()
            identity = self.jobs.submit(
                JobRequest(
                    self.project.project_id,
                    None,
                    Revisions(calibration=self.sample.epoch),
                    argument,
                    len(argument),
                    16 * 1024**2,
                    sample_work,
                )
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            self._active_kind = self._sample_context = None
            self.sample.status.setText(f"Sample geometry could not start: {exc}")
        else:
            self._active_generation = identity.generation
        self.sample.refresh()

    def _sample_current(self, identity):
        return (
            identity.generation == self._active_generation == self.jobs.latest_generation
            and self._sample_context == self.sample.context()
            and not self._close_intent
            and self._pending_open is None
        )

    def _show_hbn(self, tab=0):
        self.hbn.refresh()
        self.hbn.tabs.setCurrentIndex(tab)
        self.hbn.show()
        self.hbn.raise_()

    def _supersede_hbn(self):
        self._pending_hbn = None
        if self._active_kind == "hbn":
            self.jobs.invalidate()
            self.hbn.status.setText(
                "hBN request superseded; obsolete result rejected. Waiting for cooperative safe stop."
            )
        self.hbn.refresh()

    def _request_hbn(self, operation, argument, context):
        if self._close_intent or self._pending_open is not None:
            return
        if type(argument) is not bytes or len(argument) > 4 * 1024**2:
            raise ValueError("hBN request exceeds 4 MiB")
        self._pending_hbn = (operation, argument, context)
        if self._active_kind == "hbn":
            self.jobs.invalidate()
        self.hbn.status.setText(operation.title() + " requested; waiting for the shared worker")
        self.hbn.refresh()
        QTimer.singleShot(0, self._dispatch_pending)

    def _submit_hbn(self, operation, argument, context):
        if context != self.hbn.context():
            self.hbn.status.setText("Queued hBN request is stale; no work launched")
            return
        self._active_kind = "hbn"
        self._hbn_context = context
        try:
            request = json.loads(argument)
            shape = (
                tuple(self.detector_panel.view.image.shape)
                if operation == "load"
                else tuple(json.loads(request["session"]["inputs_json"])["shape_rc"])
            )
            request["resources"] = hbn_work_budget(
                operation, shape, **self._simulation_resource_charge()
            )
            argument = json.dumps(request, allow_nan=False).encode()
            identity = self.jobs.submit(
                JobRequest(
                    self.project.project_id,
                    self.selected_acquisition_id,
                    Revisions(calibration=self.hbn.epoch),
                    argument,
                    len(argument),
                    4 * 1024**2,
                    hbn_work,
                )
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            self._active_kind = self._hbn_context = None
            self.hbn.status.setText(f"hBN could not start: {exc}")
        else:
            self._active_generation = identity.generation
        self.hbn.refresh()

    def _hbn_current(self, identity):
        return (
            identity.generation == self._active_generation == self.jobs.latest_generation
            and self._hbn_context == self.hbn.context()
            and not self._close_intent
            and self._pending_open is None
        )

    def _supersede_simulation(self) -> None:
        self._pending_simulation = None
        if self._active_kind == "simulation":
            self.jobs.invalidate()
            self.simulator.status.setText(
                "Simulation superseded; obsolete publications rejected immediately. Waiting for canonical safe stop."
            )
        self.simulator.refresh()

    def _request_simulation(self, operation: str, argument) -> None:
        if self._close_intent or self._pending_open is not None:
            return
        context = (self.project.project_id, self.simulator.epoch)
        self._pending_simulation = (operation, argument, context)
        if self._active_kind == "simulation" and operation != "profiles":
            self.jobs.invalidate()
        self.simulator.status.setText(
            f"{operation.replace('_', ' ').title()} requested; waiting for the global worker"
        )
        self.simulator.refresh()
        QTimer.singleShot(0, self._dispatch_pending)

    def _submit_simulation(self, operation: str, argument, context) -> None:
        if context != (self.project.project_id, self.simulator.epoch):
            return
        if operation in ("transfer_review", "transfer_apply"):
            if not self.simulator.transfer_context_current(json.loads(argument)["token"]):
                self.simulator.status.setText(
                    "Queued transfer review/confirmation is stale; prior draft retained"
                )
                return
            run, budget = prepare_simulation_transfer, 256 * 1024
        elif operation in ("native_load", "native_validate", "native_save"):
            run, budget = prepare_native_draft, 192 * 1024
        elif operation == "native_run":
            run, budget = run_native_simulation, 160 * 1024**2
        elif operation in ("load", "validate", "save_configuration"):
            run, budget = prepare_simulation_draft, 96 * 1024
        elif operation == "run":
            run, budget = run_simulation, 160 * 1024**2
        elif operation == "profiles":
            run, budget = prepare_simulation_profiles, 512 * 1024
        elif operation == "export":
            run, budget = export_simulation, 8192
        elif operation == "reopen":
            run, budget = reopen_simulation_result, 160 * 1024**2
        else:
            raise ValueError("unknown simulation operation")
        self._simulation_operation, self._simulation_context = operation, context
        self._active_kind = "simulation"
        try:
            identity = self.jobs.submit(
                JobRequest(
                    self.project.project_id,
                    None,
                    Revisions(model=self.simulator.epoch),
                    argument,
                    getattr(argument, "argument_bytes", None) or _request_size(argument),
                    budget,
                    run,
                )
            )
        except (ValueError, RuntimeError, TypeError) as exc:
            self._active_kind = self._simulation_operation = self._simulation_context = None
            self.simulator.status.setText(f"Simulation could not start: {exc}")
        else:
            self._active_generation = identity.generation
        self.simulator.refresh()

    def _simulation_current(self, identity: JobIdentity) -> bool:
        return (
            identity.generation == self.jobs.latest_generation == self._active_generation
            and self._simulation_context == (self.project.project_id, self.simulator.epoch)
            and not self._close_intent
            and self._pending_open is None
        )

    def _simulation_publication(self, identity: JobIdentity, value: object) -> None:
        if (
            self._active_kind == "simulation"
            and self._simulation_current(identity)
            and isinstance(value, SimulationFrame)
        ):
            self.simulator.admit(value)

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
                view.show_mask,
            )
        elif (
            self._pending_view_restore is not None
            and self._pending_view_restore.selected_acquisition_id == self.selected_acquisition_id
        ):
            detector = self._pending_view_restore.detector
        scene = None
        if self.selected_acquisition_id is not None:
            yaw, pitch, zoom, target = self.scene_panel.view.camera_state()
            scene = SceneViewState(yaw, pitch, zoom, target, self.scene_tabs.currentIndex() == 1)
        return ProjectViewState(
            self.selected_acquisition_id,
            "fit_experiments" if self.workspaces.currentIndex() == 0 else "simulator",
            detector,
            scene,
            self.comparison_panel.capture_state(self.scene_tabs.currentIndex() == 2),
            self.simulator.draft,
            self.simulator.result_reference,
            panel_view_state(self.simulator.detector) or self.simulator._pending_detector_state,
            self.simulator.native.draft,
            self.simulator.native.result_reference,
            self.simulator.draft_kind.currentData(),
            tuple(
                v
                for v in self.hbn.sessions
                if any(a.acquisition_id == v.acquisition_id for a in self.project.acquisitions)
            ),
            self.sample.session,
            self.joint.session,
            self.physical.settings_json,
            self.prepared.session,
        )

    def _validate_project_admission(
        self,
        candidate: Project,
        *,
        selected_id: UUID | None = None,
        view: ProjectViewState | None = None,
        numeric_draft: NumericDraft | None = None,
    ) -> None:
        if view is None:
            view = self._capture_view()
        if selected_id is not None:
            view = replace(view, selected_acquisition_id=selected_id, detector=None, scene=None)
        view = replace(
            view,
            hbn_sessions=tuple(
                v
                for v in view.hbn_sessions
                if any(a.acquisition_id == v.acquisition_id for a in candidate.acquisitions)
            ),
        )
        project_to_document(
            ProjectDocument(
                candidate,
                view,
                numeric_draft
                if numeric_draft is not None
                else self._numeric_draft
                if self._numeric_draft is not None
                and any(
                    item.acquisition_id == self._numeric_draft.acquisition_id
                    for item in candidate.acquisitions
                )
                else None,
            ),
            self._project_path or self._recovery_path(),
            reserved_bytes=MAX_FUTURE_VIEW_BYTES,
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
        if self._numeric_draft is not None and not any(
            item.acquisition_id == self._numeric_draft.acquisition_id
            for item in self.project.acquisitions
        ):
            self._numeric_draft = None
            self._validated_numeric = None
            self._launch_snapshot = None
            self._numeric_history = SessionHistory()
            self._refresh_numeric_editor()
        self.joint.invalidate_inputs()
        self.physical.owner_changed()
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
        if self._pending_masks:
            self._deferred_mask_write = (Path(destination), recovery, explicit, adopt_destination)
            QTimer.singleShot(0, self._dispatch_pending)
            self.statusBar().showMessage("Saving after mask preparation")
            return True
        path = Path(destination).absolute()
        try:
            document = project_to_document(
                ProjectDocument(self.project, self._capture_view(), self._numeric_draft), path
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
        if not self._close_intent and self._pending_open is None and self._start_mask_work():
            return
        if self._deferred_mask_write is not None:
            path, recovery, explicit, adopt = self._deferred_mask_write
            self._deferred_mask_write = None
            self._queue_write(path, recovery=recovery, explicit=explicit, adopt_destination=adopt)
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
        if self._pending_physical is not None:
            argument, context = self._pending_physical
            self._pending_physical = None
            self._submit_physical(argument, context)
            return
        if self._pending_prepared is not None:
            operation, argument, context = self._pending_prepared
            self._pending_prepared = None
            self._submit_prepared(operation, argument, context)
            return
        if self._pending_joint is not None:
            operation, argument, context = self._pending_joint
            self._pending_joint = None
            self._submit_joint(operation, argument, context)
            return
        if self._pending_sample is not None:
            operation, argument, context = self._pending_sample
            self._pending_sample = None
            self._submit_sample(operation, argument, context)
            return
        if self._pending_hbn is not None:
            operation, argument, context = self._pending_hbn
            self._pending_hbn = None
            self._submit_hbn(operation, argument, context)
            return
        if self._pending_simulation is not None:
            operation, argument, context = self._pending_simulation
            self._pending_simulation = None
            self._submit_simulation(operation, argument, context)
            return
        if (
            self.simulator.detector._profile_pending
            and self.simulator.frame is not None
            and self.simulator.frame.quantitative
        ):
            self.simulator.request_profiles()
            if self._pending_simulation is not None:
                QTimer.singleShot(0, self._dispatch_pending)
                return
        if self._pending_setup_copy is not None:
            plan, context = self._pending_setup_copy
            self._pending_setup_copy = None
            if context == (
                self.project.project_id,
                self._revision,
                self.selected_acquisition_id,
                self._selected_project_ids(),
            ):
                self._start_setup("copy", {"plan": plan})
            elif self._setup_dialog is not None:
                self._setup_dialog.message.setText("Confirmed copy became stale; review again")
            return
        if self._pending_setup_request is not None:
            request, context = self._pending_setup_request
            self._pending_setup_request = None
            if context == (
                self.project.project_id,
                self._revision,
                self.selected_acquisition_id,
                self._selected_project_ids(),
            ):
                self._submit_setup(request, context)
            elif self._setup_dialog is not None:
                self._setup_dialog.message.setText("Pending setup became stale; review again")
            return
        if self._pending_reference is not None:
            kind, path, ids, project_id, revision = self._pending_reference
            self._pending_reference = None
            if project_id == self.project.project_id and revision == self._revision:
                argument = json.dumps(
                    {
                        "kind": kind,
                        "path": str(path),
                        "bindings": [
                            (str(acquisition_id), binding_kind, str(binding_path))
                            for acquisition_id, binding_kind, binding_path, _ in reference_bindings(
                                self.project
                            )
                        ],
                    },
                ).encode("utf-8")
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
                            128 * 1024,
                            prepare_reference,
                        )
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    self._active_kind = None
                    self._active_reference = None
                    self._invalidate_reference_request(
                        kind, path, "Reference check could not start"
                    )
                    self.statusBar().showMessage(f"Reference check could not start: {exc}")
                else:
                    self._active_generation = identity.generation
                    return
            else:
                if project_id == self.project.project_id:
                    self._invalidate_reference_request(kind, path, "Stale reference request")
                self._show_state(
                    "error",
                    "Reference not bound",
                    f"Project changed before {kind} validation. Choose {path.name} again.",
                )
                self.statusBar().showMessage(
                    f"Stale {kind} request discarded · Choose {path.name} again"
                )
        if self._deferred_import is not None:
            source, acquisition_id, mode, project_id = self._deferred_import
            self._deferred_import = None
            if project_id == self.project.project_id:
                self._submit_import(source, acquisition_id, mode=mode)
            return
        if self._start_line_work() or self._start_comparison_preparation():
            return
        self._start_next_candidate()
        self._sync_numeric_load_button()

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
        self.physical.cancel(clear=True)
        self._pending_open = (candidate, recovery)
        self._discard_confirmed = False
        for candidate_id in self._candidate_queue:
            queued = self._candidates.get(candidate_id)
            if queued is not None and queued.status == "queued":
                queued.status = "canceled"
                queued.detail = "Interrupted by project open; retry if this project remains"
        self._pending_hbn = None
        self._pending_prepared = None
        self._pending_joint = None
        self._pending_sample = None
        self._pending_reference = None
        self._pending_setup_request = None
        self._pending_setup_copy = None
        if self._active_kind in (
            "import",
            "relink",
            "batch",
            "reference",
            "reciprocal",
            "setup",
            "simulation",
            "hbn",
            "joint",
            "prepared",
            "sample",
        ):
            self._pending_hbn = None
            self._pending_prepared = None
            self._pending_joint = None
            self._pending_sample = None
            self._pending_simulation = None
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

    def _show_setup(self) -> None:
        if self._setup_dialog is None:
            self._setup_dialog = SetupDialog(self)
        self._setup_dialog.refresh()
        self._setup_dialog.show()
        self._setup_dialog.raise_()

    def _setup_context_current(self) -> bool:
        return (
            not self._close_intent
            and self._pending_open is None
            and self._setup_context
            == (
                self.project.project_id,
                self._revision,
                self.selected_acquisition_id,
                self._selected_project_ids(),
            )
        )

    def _start_setup(self, operation: str, payload: dict) -> None:
        if self._close_intent or self._pending_open is not None:
            return
        ids = self._selected_project_ids()
        if operation not in ("load_template", "save_template") and not ids:
            self.statusBar().showMessage("Select acquisitions before setup/source review")
            return
        path = self._project_path or self._recovery_path()
        try:
            document = project_to_document(
                ProjectDocument(self.project, self._capture_view(), self._numeric_draft), path
            )
            argument = json.dumps(
                {
                    "operation": operation,
                    "project_path": str(path),
                    "document": document,
                    "ids": [str(v) for v in ids],
                    "payload": payload,
                },
                allow_nan=False,
            ).encode()
        except (ValueError, ProjectFormatError) as exc:
            self.statusBar().showMessage(f"Setup not started: {exc}")
            return
        context = (self.project.project_id, self._revision, self.selected_acquisition_id, ids)
        if self.jobs.busy or self._active_kind is not None or self._write_queue:
            self._pending_setup_request = (argument, context)
            if self._setup_dialog is not None:
                self._setup_dialog.message.setText("Setup queued after the current operation")
            QTimer.singleShot(0, self._dispatch_pending)
            return
        self._submit_setup(argument, context)

    def _submit_setup(self, argument: bytes, context: tuple) -> None:
        self._setup_context = context
        self._active_kind = "setup"
        try:
            identity = self.jobs.submit(
                JobRequest(
                    self.project.project_id,
                    self.selected_acquisition_id,
                    Revisions(data=self._revision),
                    argument,
                    len(argument),
                    min(96 * 1024 * 1024, 2 * len(argument) + 128 * 1024),
                    prepare_setup,
                )
            )
            self._active_generation = identity.generation
        except (ValueError, RuntimeError, TypeError) as exc:
            self._active_kind = None
            self.statusBar().showMessage(f"Setup not started: {exc}")
        if self._setup_dialog is not None:
            self._setup_dialog.refresh()

    def _commit_setup(self, application: SetupApplication) -> None:
        if not self._setup_context_current():
            if self._setup_dialog is not None:
                self._setup_dialog.message.setText("Review became stale; no binding committed")
            return
        updated, draft = application.document.project, application.document.numeric_draft
        try:
            project_to_document(
                ProjectDocument(updated, self._capture_view(), draft),
                self._project_path or self._recovery_path(),
                reserved_bytes=MAX_FUTURE_VIEW_BYTES,
            )
            action = combined_action(
                self.project, updated, self._numeric_draft, draft, "Apply reviewed setup / storage"
            )
        except (ValueError, ProjectFormatError) as exc:
            if self._setup_dialog is not None:
                self._setup_dialog.message.setText(f"Setup not committed: {exc}")
            return
        if action is not None:
            self._numeric_history.push(action)
        self.project, self._numeric_draft = updated, draft
        self._validated_numeric = None
        self._launch_snapshot = None
        for acquisition_id, kind in application.verified_kinds:
            if kind == "osc":
                self._source_checks[acquisition_id] = SourceCheck(
                    acquisition_id, "verified", "Decoded identity checked by storage worker"
                )
                cached = self._resident_planes.get(acquisition_id)
                acquisition = next(
                    v for v in updated.acquisitions if v.acquisition_id == acquisition_id
                )
                if cached is not None and cached.decoded_sha256 == acquisition.source_sha256:
                    self._resident_planes[acquisition_id] = replace(
                        cached, source_path=acquisition.source_path
                    )
                else:
                    self._resident_planes.pop(acquisition_id, None)
                if acquisition_id == self.selected_acquisition_id:
                    self._deferred_import = (
                        acquisition.source_path,
                        acquisition_id,
                        "import",
                        updated.project_id,
                    )
            else:
                self._reference_checks[(acquisition_id, kind)] = ReferenceCheck(
                    acquisition_id,
                    kind,
                    "verified",
                    "Exact reference identity checked by storage worker",
                )
        self.refresh_project()
        if action is not None:
            self._mark_dirty()
        message = (
            "Reviewed snapshot committed as one undoable action; numeric receipts invalidated. Source files retained."
            if action is not None
            else "Saved identities revalidated; readiness refreshed without a history action."
        )
        if self._setup_dialog is not None:
            self._setup_dialog.message.setText(message)
        self.statusBar().showMessage(message)

    def _choose_relink(self) -> None:
        self._show_setup()
        if self._setup_dialog is not None:
            self._setup_dialog.kind.setCurrentIndex(0)
            self._setup_dialog.relink()

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
        self._invalidate_reference_request(kind, path, "Reference recheck pending")
        self._pending_reference = (kind, path, ids, self.project.project_id, self._revision)
        QTimer.singleShot(0, self._dispatch_pending)

    def _invalidate_reference_request(self, kind: str, path: Path, detail: str) -> None:
        self._validated_numeric = None
        self._reference_checks = invalidate_reference_checks(
            self.project, self._reference_checks, kind, path, detail
        )
        self._update_selection_label()
        self._refresh_review_table()
        self._refresh_numeric_editor()

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
            if task is not None and task[3] == self.project.project_id:
                self._invalidate_reference_request(task[0], task[1], "Stale reference result")
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
                if kind == "configuration":
                    changes["configuration_cif_path"] = value.dependent_cif_path
                    changes["configuration_cif_sha256"] = value.dependent_cif_sha256
                    if value.dependent_cif_path is None:
                        provenance.pop("configuration_cif_path", None)
                    else:
                        provenance["configuration_cif_path"] = (
                            "bounded configuration-dependent CIF identity"
                        )
                    changes["provenance"] = tuple(sorted(provenance.items()))
                if value.material_id:
                    changes["material_id"] = value.material_id
                    provenance["material_id"] = f"validated {kind} reader"
                    changes["provenance"] = tuple(sorted(provenance.items()))
                updated = updated.update_metadata(acquisition_id, **changes)
            self._validate_project_admission(updated)
        except (ProjectFormatError, ValueError) as exc:
            self._invalidate_reference_request(kind, path, "Reference binding rejected")
            self.statusBar().showMessage(f"Reference binding rejected: {exc}")
            return
        action = metadata_action(self.project, updated, f"Bind {kind} reference")
        changed = updated != self.project
        self.project = updated
        self._reference_checks = publish_reference_checks(
            updated,
            self._reference_checks,
            path,
            value.sha256,
            value.file_identity,
            value.matching_bindings,
        )
        if kind == "configuration":
            if value.dependent_cif_path is not None and value.dependent_cif_sha256 is not None:
                self._reference_checks = publish_reference_checks(
                    updated,
                    self._reference_checks,
                    value.dependent_cif_path,
                    value.dependent_cif_sha256,
                    value.dependent_cif_file_identity,
                    value.matching_dependent_bindings,
                )
            else:
                for acquisition_id in ids:
                    self._reference_checks[(acquisition_id, "configuration_cif")] = ReferenceCheck(
                        acquisition_id,
                        "configuration_cif",
                        "unverified",
                        "No dependent CIF identity recorded",
                    )
        if changed:
            self._numeric_history.push(action)
            self._mark_dirty()
        self._update_selection_label()
        self._refresh_review_table()
        self._refresh_numeric_editor()
        self.statusBar().showMessage(
            f"Validated {kind} reference and SHA-256 for {len(ids)} acquisition(s)"
        )

    def relink_selected(self, path: Path) -> None:
        self._show_setup()
        self._start_setup("relink", {"kind": "osc", "path": str(Path(path).absolute())})

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
        self.hbn.sessions = tuple(
            v
            for v in self.hbn.sessions
            if any(a.acquisition_id == v.acquisition_id for a in self.project.acquisitions)
        )
        self.hbn.refresh()
        self.sample.refresh()
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
        self.comparison_panel.set_choices(self.project.acquisitions)
        self._refresh_comparison()
        self._refresh_numeric_editor()

    def _thumbnail_icon(self, acquisition_id: UUID) -> QIcon:
        item = self._thumbnails.get(acquisition_id)
        if item is None:
            return QIcon()
        pixels, rows, columns = item
        image = QImage(pixels, columns, rows, columns, QImage.Format.Format_Grayscale8)
        return QIcon(QPixmap.fromImage(image.copy()))

    def _reference_states(self, acquisition: Acquisition) -> tuple[ReferenceCheck, ...]:
        metadata = acquisition.metadata
        states = []
        for kind, path in (
            ("cif", metadata.cif_path),
            ("configuration", metadata.configuration_path),
            ("configuration_cif", metadata.configuration_cif_path),
        ):
            if path is None and not (
                kind == "configuration_cif" and metadata.configuration_path is not None
            ):
                continue
            states.append(
                self._reference_checks.get((acquisition.acquisition_id, kind))
                or ReferenceCheck(
                    acquisition.acquisition_id,
                    kind,
                    "unverified",
                    "No current identity check; choose the reference again",
                )
            )
        return tuple(states)

    def _numeric_acquisition(self) -> Acquisition | None:
        return next(
            (
                item
                for item in self.project.acquisitions
                if item.acquisition_id == self.selected_acquisition_id
            ),
            None,
        )

    def _numeric_identity_matches(self, acquisition: Acquisition) -> bool:
        draft = self._numeric_draft
        metadata = acquisition.metadata
        return draft is not None and (
            draft.acquisition_id == acquisition.acquisition_id
            and draft.source_sha256 == acquisition.source_sha256
            and draft.configuration_path == metadata.configuration_path
            and draft.configuration_sha256 == metadata.configuration_sha256
            and draft.cif_path == metadata.configuration_cif_path
            and draft.cif_sha256 == metadata.configuration_cif_sha256
        )

    def _numeric_validated_acquisition(self) -> Acquisition:
        acquisition = self._numeric_acquisition()
        if acquisition is None:
            raise ProjectFormatError("select one admitted acquisition")
        source = self._source_checks.get(acquisition.acquisition_id)
        if source is None or source.state != "verified":
            raise ProjectFormatError("source must be verified before editing or freezing")
        for kind in ("configuration", "configuration_cif"):
            check = self._reference_checks.get((acquisition.acquisition_id, kind))
            if check is None or check.state != "verified":
                raise ProjectFormatError(f"{kind} must be verified; choose the configuration again")
        return acquisition

    def _sync_numeric_load_button(self) -> None:
        self.numeric_load_button.setEnabled(
            self._numeric_acquisition() is not None
            and not self.jobs.busy
            and self._active_kind is None
            and not self._write_queue
            and self._pending_open is None
            and not self._close_intent
        )
        self._sync_reciprocal_button()

    def _refresh_numeric_editor(self, _value: object = None) -> None:
        acquisition = self._numeric_acquisition()
        field = self.numeric_field.currentData()
        item = description(field)
        available = acquisition is not None and self._numeric_identity_matches(acquisition)
        unavailable_detail = ""
        if available:
            available = (
                self._validated_numeric == (self.project.project_id, self._numeric_draft)
                and self._source_checks.get(acquisition.acquisition_id) is not None
                and self._source_checks[acquisition.acquisition_id].state == "verified"
                and all(
                    (check := self._reference_checks.get((acquisition.acquisition_id, kind)))
                    is not None
                    and check.state == "verified"
                    for kind in ("configuration", "configuration_cif")
                )
            )
        if available:
            draft = self._numeric_draft
            assert draft is not None
            try:
                self.numeric_value.setText(displayed_value(draft, field) if item.editable else "")
            except ProjectFormatError as exc:
                unavailable_detail = str(exc)
                available = False
            if available:
                self.numeric_status.setText(
                    f"Initial draft · revision {draft.revision} · {len(draft.proposed)} proposed value(s)"
                    + (
                        f" · frozen revision {self._launch_snapshot.draft_revision}"
                        if self._launch_snapshot is not None
                        else ""
                    )
                )
        if not available:
            self.numeric_value.clear()
            self.numeric_status.setText(
                unavailable_detail
                or "Saved draft needs verified references and full validation. Select Load to validate it."
                if self._numeric_draft is not None
                else "Load a validated configuration for one acquisition."
            )
        self.numeric_description.setText(
            f"{item.scope} · {item.frame} · {item.display_unit} → {item.stored_unit} · {item.domain}"
            + (f"\n{item.reason}" if item.reason else "")
        )
        self.numeric_value.setEnabled(available and item.editable)
        self.numeric_apply_button.setEnabled(available and item.editable)
        self.numeric_revert_button.setEnabled(
            acquisition is not None
            and self._numeric_identity_matches(acquisition)
            and bool(self._numeric_draft.proposed)
        )
        self.numeric_freeze_button.setEnabled(available)
        self.numeric_undo_button.setEnabled(bool(self._numeric_history.undo_actions))
        self.numeric_redo_button.setEnabled(bool(self._numeric_history.redo_actions))
        self._sync_numeric_load_button()
        self._refresh_reciprocal_editor()

    def _reciprocal_key(self, acquisition: Acquisition | None = None) -> tuple[object, ...] | None:
        acquisition = acquisition or self._numeric_acquisition()
        if acquisition is None:
            return None
        metadata = acquisition.metadata
        source = self._source_checks.get(acquisition.acquisition_id)
        if source is None or source.state != "verified" or metadata.native_shape is None:
            return None
        if any(
            (check := self._reference_checks.get((acquisition.acquisition_id, kind))) is None
            or check.state != "verified"
            for kind in ("configuration", "configuration_cif")
        ):
            return None
        if any(
            value is None
            for value in (
                metadata.configuration_path,
                metadata.configuration_sha256,
                metadata.configuration_cif_path,
                metadata.configuration_cif_sha256,
            )
        ):
            return None
        draft = (
            self._numeric_draft
            if self._numeric_identity_matches(acquisition)
            and self._validated_numeric == (self.project.project_id, self._numeric_draft)
            else None
        )
        return (
            self.project.project_id,
            acquisition.acquisition_id,
            acquisition.source_path,
            acquisition.source_sha256,
            metadata.configuration_path,
            metadata.configuration_sha256,
            metadata.configuration_cif_path,
            metadata.configuration_cif_sha256,
            metadata.native_shape,
            metadata.incidence_rad,
            draft,
        )

    def _sync_reciprocal_button(self) -> None:
        self.reciprocal_button.setEnabled(
            self._reciprocal_key() is not None
            and not self.jobs.busy
            and self._active_kind is None
            and not self._write_queue
            and self._pending_open is None
            and not self._close_intent
        )

    def _visible_reciprocal(self) -> ReciprocalPreview | None:
        key = self._reciprocal_key()
        acquisition = self._numeric_acquisition()
        if key is None or acquisition is None:
            return None
        cached = self._reciprocal_cache.get(acquisition.acquisition_id)
        if cached is None or cached[0] != key:
            return None
        self._reciprocal_cache.move_to_end(acquisition.acquisition_id)
        return cached[1]

    def _refresh_reciprocal_editor(self) -> None:
        preview = self._visible_reciprocal()
        if self.reciprocal_view.preview is not preview:
            self._invalidate_reciprocal_pointer()
            self.reciprocal_view.set_preview(preview)
        if preview is None:
            self.reciprocal_status.setText(
                "Verified source, configuration, dependent CIF and native shape required. "
                "Map geometry on demand; an unvalidated retained draft is excluded."
                if self._reciprocal_key() is None
                else "Coverage not prepared for this acquisition and geometry revision."
            )
            self.reciprocal_cursor.setText("Pointer Q unavailable · no current geometry map")
            self.reciprocal_selection.setText("Selected Q unavailable · no current geometry map")
        else:
            saved_valid = int(preview.saved.valid.sum())
            draft_valid = int(preview.draft.valid.sum()) if preview.draft is not None else None
            self.reciprocal_status.setText(
                f"Saved baseline: {saved_valid}/{preview.grid_axis**2} grid points valid"
                + (
                    f" · draft revision {preview.draft_revision}: "
                    f"{draft_valid}/{preview.grid_axis**2} valid"
                    if draft_valid is not None
                    else " · no validated proposed geometry"
                )
                + f" · gaps are omitted · {preview.approximation}"
            )
            self._reciprocal_selection_changed()
        self._sync_reciprocal_button()
        self._sync_scene()

    def _sync_scene(self) -> None:
        preview = self._visible_reciprocal()
        acquisition_id = self.selected_acquisition_id
        prepared = (
            self._resident_planes.get(acquisition_id)
            if (acquisition_id is not None and acquisition_id == self._visible_acquisition_id)
            else None
        )
        mapping = (
            (preview.draft or preview.saved)
            if preview is not None and prepared is not None
            else None
        )
        identity = (preview.request_sha256, preview.draft_revision) if mapping is not None else None
        scene = self.scene_panel.view
        previous_acquisition = scene._image_identity[0] if scene._image_identity else None
        scene.set_scene(acquisition_id, prepared, mapping, identity)
        self.scene_panel.view.set_overlays(self.detector_panel.view.overlays, mapping)
        if mapping is not None and self._pending_scene_restore is not None:
            saved = self._pending_scene_restore
            scene.restore_camera((saved.yaw_rad, saved.pitch_rad, saved.zoom, saved.target_lab_m))
            self._pending_scene_restore = None
            if acquisition_id is not None:
                self._scene_cameras[acquisition_id] = scene.camera_state()
        elif mapping is not None and acquisition_id != previous_acquisition:
            saved_camera = self._scene_cameras.get(acquisition_id)
            if saved_camera is not None:
                scene.restore_camera(saved_camera)
        detector = self.detector_panel.view
        self.scene_panel.view.set_levels(
            detector.low_value, detector.high_value, detector.contrast_mode
        )

    def _remember_scene_camera(self) -> None:
        identity = self.scene_panel.view._image_identity
        if identity is not None and identity[0] == self.selected_acquisition_id:
            self._scene_cameras[identity[0]] = self.scene_panel.view.camera_state()
        self._mark_dirty()

    def _invalidate_reciprocal_pointer(self) -> None:
        self.reciprocal_cursor.setText("Pointer Q unavailable · geometry changed")
        previous_q = self._reciprocal_detector_q_text
        if previous_q is not None:
            label = self.detector_panel.cursor_label
            if previous_q in label.text():
                label.setText(label.text().replace(previous_q, "Q/angles unavailable", 1))
            self._reciprocal_detector_q_text = None
        previous_status = self._reciprocal_pointer_status
        if previous_status is not None:
            if self.statusBar().currentMessage() == previous_status:
                self.statusBar().showMessage("Pointer Q unavailable · geometry changed")
            self._reciprocal_pointer_status = None

    def _reciprocal_coordinate_text(self, column: float, row: float) -> tuple[str, object, str]:
        preview = self._visible_reciprocal()
        if preview is None:
            message = "Q unavailable · no current geometry map"
            return message, None, message
        mapping = preview.draft or preview.saved
        status, internal, external = mapping.cursor(column, row)
        if internal is None or external is None:
            message = f"({column:g}, {row:g}) px · Q unavailable: {status}"
            return message, None, message
        label = (
            f"draft r{preview.draft_revision}" if preview.draft is not None else "saved baseline"
        )
        text = (
            f"({column:g}, {row:g}) native px · {label} · "
            f"Q internal film/sample=({internal[0]:.4g}, {internal[1]:.4g}, {internal[2]:.4g}) Å⁻¹ · "
            f"Q external air/sample=({external[0]:.4g}, {external[1]:.4g}, {external[2]:.4g}) Å⁻¹"
        )
        compact = (
            f"({column:g},{row:g})px {label} "
            f"Q film/sample ({internal[0]:.3g},{internal[1]:.3g},{internal[2]:.3g}) "
            f"air/sample ({external[0]:.3g},{external[1]:.3g},{external[2]:.3g}) Å⁻¹"
        )
        return text, internal, compact

    def _reciprocal_cursor_changed(self, pixel: object) -> None:
        if pixel is None:
            self.reciprocal_cursor.setText("Pointer outside detector · Q unavailable")
            self._reciprocal_detector_q_text = None
            if self._visible_reciprocal() is not None and self._active_kind is None:
                self._reciprocal_pointer_status = "Pointer outside detector · Q unavailable"
                self.statusBar().showMessage(self._reciprocal_pointer_status)
            return
        column, row, _intensity = pixel
        text, _q, compact = self._reciprocal_coordinate_text(column, row)
        self.reciprocal_cursor.setText(f"Pointer {text}")
        if self._visible_reciprocal() is not None:
            self.detector_panel.cursor_label.setText(
                self.detector_panel.cursor_label.text().replace("Q/angles unavailable", text)
            )
            self._reciprocal_detector_q_text = text
        if self._visible_reciprocal() is not None and self._active_kind is None:
            self._reciprocal_pointer_status = compact
            self.statusBar().showMessage(self._reciprocal_pointer_status)

    def _reciprocal_selection_changed(self) -> None:
        if self._visible_acquisition_id != self.selected_acquisition_id:
            self.reciprocal_selection.setText("Selected Q unavailable · image not resident")
            self.reciprocal_view.set_selected_q(None)
            return
        column, row = self.detector_panel.view.crosshair
        text, q, _compact = self._reciprocal_coordinate_text(column, row)
        self.reciprocal_selection.setText(f"Selected {text}")
        self.reciprocal_view.set_selected_q(q)

    def _request_reciprocal_preview(self) -> None:
        key = self._reciprocal_key()
        acquisition = self._numeric_acquisition()
        if key is None or acquisition is None:
            self.reciprocal_status.setText("Verify this source and its configuration first.")
            return
        if (
            self.jobs.busy
            or self._active_kind is not None
            or self._write_queue
            or self._pending_open
            or self._close_intent
        ):
            self.reciprocal_status.setText("Finish the current operation before mapping geometry.")
            return
        metadata = acquisition.metadata
        draft = key[-1]
        argument = json.dumps(
            {
                "project_id": str(self.project.project_id),
                "acquisition_id": str(acquisition.acquisition_id),
                "source_sha256": acquisition.source_sha256,
                "configuration_path": str(metadata.configuration_path),
                "configuration_sha256": metadata.configuration_sha256,
                "cif_path": str(metadata.configuration_cif_path),
                "cif_sha256": metadata.configuration_cif_sha256,
                "native_shape_rc": metadata.native_shape,
                "proposed": draft.proposed if isinstance(draft, NumericDraft) else (),
                "revision": draft.revision if isinstance(draft, NumericDraft) else 0,
                "declared_incidence_rad": metadata.incidence_rad,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        active = (key, self._reciprocal_epoch, hashlib.sha256(argument).hexdigest())
        self._active_kind = "reciprocal"
        self._active_reciprocal = active
        try:
            identity = self.jobs.submit(
                JobRequest(
                    self.project.project_id,
                    acquisition.acquisition_id,
                    Revisions(),
                    argument,
                    len(argument),
                    MAX_PREVIEW_BYTES,
                    prepare_reciprocal_preview,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if self._active_kind == "reciprocal" and self._active_reciprocal == active:
                self._active_kind = None
                self._active_reciprocal = None
            self.reciprocal_status.setText(f"Coverage request unavailable: {exc}")
            return
        self._active_generation = identity.generation
        self.reciprocal_status.setText(
            "Mapping nominal reciprocal coverage on the background worker…"
        )
        self.statusBar().showMessage("Mapping selected reciprocal geometry…")

    def _reciprocal_ready(self, identity: JobIdentity, value: object) -> None:
        active = self._active_reciprocal
        self._active_reciprocal = None
        if (
            active is None
            or not isinstance(value, ReciprocalPreview)
            or identity.project_id != self.project.project_id
            or identity.acquisition_id != self.selected_acquisition_id
            or active[0] != self._reciprocal_key()
            or active[1] != self._reciprocal_epoch
            or active[2] != value.request_sha256
            or value.project_id != self.project.project_id
            or value.acquisition_id != self.selected_acquisition_id
        ):
            self.reciprocal_status.setText(
                "Obsolete coverage discarded; map current geometry again."
            )
            return
        self._reciprocal_cache[value.acquisition_id] = (active[0], value)
        self._reciprocal_cache.move_to_end(value.acquisition_id)
        while len(self._reciprocal_cache) > 2:
            self._reciprocal_cache.popitem(last=False)
        self._refresh_reciprocal_editor()
        self._refresh_comparison()
        self.statusBar().showMessage("Reciprocal coverage ready for selected geometry")

    def _load_numeric_draft(self) -> None:
        try:
            if (
                self.jobs.busy
                or self._active_kind is not None
                or self._write_queue
                or self._pending_open is not None
                or self._close_intent
            ):
                raise ProjectFormatError(
                    "finish the current operation before loading numeric values"
                )
            acquisition = self._numeric_validated_acquisition()
            metadata = acquisition.metadata
            if (
                metadata.configuration_path is None
                or metadata.configuration_sha256 is None
                or metadata.configuration_cif_path is None
                or metadata.configuration_cif_sha256 is None
            ):
                raise ProjectFormatError("configuration reference is missing")
            retained = self._numeric_draft
            matching = self._numeric_identity_matches(acquisition)
            argument = json.dumps(
                {
                    "acquisition_id": str(acquisition.acquisition_id),
                    "source_sha256": acquisition.source_sha256,
                    "configuration_path": str(metadata.configuration_path),
                    "configuration_sha256": metadata.configuration_sha256,
                    "cif_path": str(metadata.configuration_cif_path),
                    "cif_sha256": metadata.configuration_cif_sha256,
                    "proposed": retained.proposed
                    if matching and retained is not None
                    else acquisition.initial_values,
                    "revision": retained.revision if matching and retained is not None else 0,
                }
            ).encode("utf-8")
        except (OSError, UnicodeError, ProjectFormatError, ValueError) as exc:
            QMessageBox.warning(self, "Numeric draft unavailable", str(exc))
            return
        request_identity = (
            acquisition.acquisition_id,
            self._revision,
            metadata.configuration_sha256,
            retained,
        )
        self._active_kind = "numeric"
        self._active_numeric = request_identity
        try:
            identity = self.jobs.submit(
                JobRequest(
                    self.project.project_id,
                    acquisition.acquisition_id,
                    Revisions(data=self._revision),
                    argument,
                    len(argument),
                    384 * 1024,
                    prepare_numeric_draft,
                )
            )
            self._active_generation = identity.generation
        except (OSError, ValueError, RuntimeError) as exc:
            if self._active_kind == "numeric" and self._active_numeric == request_identity:
                self._active_kind = None
                self._active_numeric = None
            QMessageBox.warning(self, "Numeric draft unavailable", str(exc))
            return
        self.numeric_status.setText("Loading and validating numeric values…")
        self.statusBar().showMessage("Loading numeric values on the background worker")

    def _numeric_ready(self, identity: JobIdentity, value: object) -> None:
        active = self._active_numeric
        self._active_numeric = None
        acquisition = self._numeric_acquisition()
        if (
            active is None
            or not isinstance(value, NumericDraft)
            or acquisition is None
            or identity.project_id != self.project.project_id
            or identity.acquisition_id != active[0]
            or identity.revisions.data != self._revision
            or self._revision != active[1]
            or self._numeric_draft is not active[3]
            or value.configuration_sha256 != active[2]
            or value.acquisition_id != acquisition.acquisition_id
        ):
            self.statusBar().showMessage("Numeric draft result became stale; load it again")
            return
        metadata = acquisition.metadata
        if (
            value.source_sha256 != acquisition.source_sha256
            or value.configuration_path != metadata.configuration_path
            or value.cif_path != metadata.configuration_cif_path
            or value.cif_sha256 != metadata.configuration_cif_sha256
        ):
            self.statusBar().showMessage("Numeric reference changed; validate it again")
            return
        try:
            self._numeric_validated_acquisition()
        except ProjectFormatError as exc:
            self.statusBar().showMessage(f"Numeric reference changed: {exc}")
            return
        try:
            project_to_document(
                ProjectDocument(self.project, self._capture_view(), value),
                self._project_path or self._recovery_path(),
                reserved_bytes=MAX_FUTURE_VIEW_BYTES,
            )
        except (ProjectFormatError, ValueError) as exc:
            self.statusBar().showMessage(f"Numeric draft cannot be admitted: {exc}")
            return
        if self._numeric_draft is not None and self._numeric_identity_matches(acquisition):
            if value != self._numeric_draft:
                self.statusBar().showMessage("Numeric draft changed; load it again")
                return
            self._validated_numeric = (self.project.project_id, self._numeric_draft)
            self._refresh_numeric_editor()
            self.statusBar().showMessage("Numeric draft already loaded; proposed edits retained")
            return
        self._numeric_history.discard_draft_actions()
        self._numeric_draft = value
        self._validated_numeric = (self.project.project_id, value)
        self._launch_snapshot = None
        self._refresh_numeric_editor()
        self._mark_dirty()
        self.statusBar().showMessage("Loaded validated numeric initial values")

    def _apply_numeric_value(self) -> None:
        try:
            acquisition = self._numeric_validated_acquisition()
        except ProjectFormatError as exc:
            self.statusBar().showMessage(str(exc))
            return
        if not self._numeric_identity_matches(acquisition):
            self.statusBar().showMessage("Load the selected configuration before editing")
            return
        if self._validated_numeric != (self.project.project_id, self._numeric_draft):
            self.statusBar().showMessage("Validate the retained numeric draft before editing")
            return
        draft = self._numeric_draft
        assert draft is not None
        field = self.numeric_field.currentData()
        try:
            updated = edit_draft(draft, field, self.numeric_value.text())
            action = draft_action(draft, updated, f"Edit {description(field).label}")
            project_to_document(
                ProjectDocument(self.project, self._capture_view(), updated),
                self._project_path or self._recovery_path(),
                reserved_bytes=MAX_FUTURE_VIEW_BYTES,
            )
        except (ProjectFormatError, ValueError, OSError) as exc:
            QMessageBox.warning(self, "Initial value rejected", str(exc))
            self._refresh_numeric_editor()
            return
        if action is not None:
            self._numeric_draft = updated
            self._validated_numeric = (self.project.project_id, updated)
            self._numeric_history.push(action)
            self._refresh_numeric_editor()
            self._mark_dirty()
            self.statusBar().showMessage(f"Proposed {description(field).label}; no fit was run")

    def _revert_numeric_draft(self) -> None:
        acquisition = self._numeric_acquisition()
        draft = self._numeric_draft
        if acquisition is None or draft is None or not self._numeric_identity_matches(acquisition):
            return
        updated = replace(draft, proposed=(), revision=draft.revision + 1)
        action = draft_action(draft, updated, "Revert numeric initial values")
        if action is None:
            return
        self._numeric_draft = updated
        if self._validated_numeric == (self.project.project_id, draft):
            self._validated_numeric = (self.project.project_id, updated)
        self._numeric_history.push(action)
        self._refresh_numeric_editor()
        self._mark_dirty()

    def _freeze_numeric_draft(self) -> None:
        try:
            acquisition = self._numeric_validated_acquisition()
            if not self._numeric_identity_matches(acquisition):
                raise ProjectFormatError("numeric draft identity is stale")
            if self._validated_numeric != (self.project.project_id, self._numeric_draft):
                raise ProjectFormatError("validate the retained numeric draft before freezing")
            assert self._numeric_draft is not None
            snapshot = freeze_draft(self.project, acquisition, self._numeric_draft)
        except (OSError, ProjectFormatError, ValueError) as exc:
            QMessageBox.warning(self, "Snapshot unavailable", str(exc))
            return
        self._launch_snapshot = snapshot
        self._refresh_numeric_editor()
        self.statusBar().showMessage(
            f"Frozen immutable initial draft revision {snapshot.draft_revision}; no solver launched"
        )

    def _numeric_undo_redo(self, *, undo: bool) -> None:
        previous_draft = self._numeric_draft
        try:
            updated, draft, references, label = self._numeric_history.apply(
                self.project,
                self._numeric_draft,
                undo=undo,
                validate=lambda project, draft: project_to_document(
                    ProjectDocument(project, self._capture_view(), draft),
                    self._project_path or self._recovery_path(),
                    reserved_bytes=MAX_FUTURE_VIEW_BYTES,
                ),
            )
        except (ProjectFormatError, ValueError, OSError) as exc:
            QMessageBox.warning(self, "Undo/redo unavailable", str(exc))
            return
        self.project = updated
        self._numeric_draft = draft
        if draft is not previous_draft and draft is not None:
            self._validated_numeric = (updated.project_id, draft)
        elif draft is None:
            self._validated_numeric = None
        self._launch_snapshot = None
        for acquisition_id, kind in references:
            self._validated_numeric = None
            if kind == "osc":
                self._source_checks[acquisition_id] = SourceCheck(
                    acquisition_id, "unreadable", "Source-path undo/redo requires revalidation"
                )
                self._resident_planes.pop(acquisition_id, None)
                continue
            if kind == "initial_values":
                continue
            self._reference_checks[(acquisition_id, kind)] = ReferenceCheck(
                acquisition_id, kind, "unverified", "Metadata undo/redo requires revalidation"
            )
        self.refresh_project()
        self._mark_dirty()
        self.statusBar().showMessage(f"{'Undid' if undo else 'Redid'} {label}")

    def _commit_metadata(self, updated: Project, label: str) -> bool:
        action = metadata_action(self.project, updated, label)
        if action is None:
            return False
        self.project = updated
        self._numeric_history.push(action)
        self.refresh_project()
        self._mark_dirty()
        return True

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
            for reference in self._reference_states(acquisition):
                if reference.state != "verified":
                    status += f" · {reference.kind} {reference.state}"
            source = self._candidates.get(acquisition.acquisition_id)
            if source is not None and source.detail:
                status += f" · {source.detail}"
            references = ", ".join(
                value
                for value in (
                    metadata.material_id,
                    metadata.cif_path.name if metadata.cif_path else None,
                    metadata.configuration_path.name if metadata.configuration_path else None,
                    metadata.configuration_cif_path.name
                    if metadata.configuration_cif_path
                    else None,
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
                cell = (
                    NumericReviewItem(
                        value,
                        metadata.incidence_rad if column == 4 else metadata.exposure_s,
                    )
                    if column in (4, 5)
                    else QTableWidgetItem(value)
                )
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
                cell = (
                    NumericReviewItem(value, None) if column in (4, 5) else QTableWidgetItem(value)
                )
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
                    str(metadata.configuration_cif_path)
                    if metadata.configuration_cif_path
                    else None,
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
        reference_status = (
            "; ".join(
                f"{check.kind}: {check.state} ({check.detail})"
                for check in self._reference_states(acquisition)
            )
            or "none bound"
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
            f"Recorded metadata origin: {source_detail}\n"
            f"Current reference identity: {reference_status}\n"
            f"Choose a failed or unverified reference again; saved identities are unchanged.\n"
            f"Suggestions: {proposals}\n"
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
            try:
                self.start_import_files(Path(filename) for filename in filenames)
            except ValueError as exc:
                self._show_state("error", "Import scope rejected", str(exc))
                self.statusBar().showMessage(f"Import scope rejected: {exc}")

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
        if not paths:
            self._show_state(
                "empty", "No direct files", f"The folder {folder} has no direct files."
            )
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Review every folder candidate")
        dialog.setMinimumSize(640, 420)
        layout = QVBoxLayout(dialog)
        unsupported = sum(not path.name.lower().endswith((".osc", ".osc.gz")) for path in paths)
        summary = QLabel(
            f"Folder: {folder}\n{len(paths)} direct files; {unsupported} unsupported. "
            "No subfolders are scanned. Confirm this complete list before import."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        candidates = QListWidget()
        candidates.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for path in paths:
            supported = path.name.lower().endswith((".osc", ".osc.gz"))
            item = QListWidgetItem(f"{'OSC' if supported else 'Unsupported'} · {path.name}")
            item.setToolTip(str(path))
            candidates.addItem(item)
        layout.addWidget(candidates)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import listed files")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
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
        self._commit_metadata(updated, "Apply acquisition metadata")

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
        if self._commit_metadata(updated, "Confirm filename suggestions"):
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
        preview = QLabel(
            "\n".join(" | ".join(value[:80] for value in row)[:512] for row in parsed[:6])
        )
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
        self._commit_metadata(updated, "Map metadata table")

    def _input_reference_paths(self) -> tuple[Path, ...]:
        return tuple(
            path
            for item in self.project.acquisitions
            for path in (
                item.source_path,
                item.metadata.cif_path,
                item.metadata.configuration_path,
                item.metadata.configuration_cif_path,
            )
            if path is not None
        )

    def _inspection_identity(self) -> tuple[Acquisition, tuple[object, ...]]:
        panel = self.detector_panel
        view = panel.view
        selected = self.selected_acquisition_id
        if (
            selected is None
            or selected != self._visible_acquisition_id
            or view.image is None
            or panel._current_profiles is None
            or panel._mask_pending
            or panel._profile_pending
            or panel._profile_key != panel._query_key()
            or panel.acquisition_identity != selected
            or panel.horizontal.values is not panel._current_profiles.horizontal
            or panel.vertical.values is not panel._current_profiles.vertical
        ):
            raise ProjectFormatError("wait for one current detector image and exact profiles")
        acquisition = next(
            (item for item in self.project.acquisitions if item.acquisition_id == selected), None
        )
        if acquisition is None:
            raise ProjectFormatError("visible acquisition is no longer in the project")
        token = (
            self.project.project_id,
            selected,
            acquisition.source_sha256,
            id(view.image),
            view.data_revision,
            view.request_generation,
            panel._profile_key,
            self._capture_view(),
        )
        return acquisition, token

    def _choose_inspection_export(self) -> None:
        if self.jobs.busy or self._active_kind is not None or self._close_intent:
            QMessageBox.warning(
                self, "Inspection export unavailable", "Wait for current work to finish."
            )
            return
        try:
            acquisition, token = self._inspection_identity()
        except (ProjectFormatError, ValueError) as exc:
            QMessageBox.warning(self, "Inspection export unavailable", str(exc))
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export detector figure and exact profiles",
            str(Path.home() / "detector-inspection.png"),
            "PNG figure (*.png)",
        )
        if not filename:
            return
        try:
            current, current_token = self._inspection_identity()
            if current_token != token or current != acquisition:
                raise ProjectFormatError(
                    "detector selection or profile changed during destination choice"
                )
            if self.jobs.busy or self._active_kind is not None or self._close_intent:
                raise ProjectFormatError("another operation began during destination choice")
            protected = self._input_reference_paths()
            project_path = self._project_path or self._recovery_path()
            figure_path = external_export_destination(
                Path(filename), project_path, protected, suffix=".png"
            )
            profiles_path = external_export_destination(
                figure_path.with_name(f"{figure_path.stem}.profiles.csv"),
                project_path,
                protected,
                suffix=".csv",
            )
            if any(path.exists() or path.is_symlink() for path in (figure_path, profiles_path)):
                raise FileExistsError("choose a new figure name; both export files must be new")
            panel = self.detector_panel
            view = panel.view
            profiles = panel._current_profiles
            assert profiles is not None
            started = perf_counter()
            figure = panel.capture_inspection_figure(
                f"{acquisition.name} · native detector · {panel.profile_measure_control.currentData()}"
                f" · data revision {view.data_revision}"
            )
            if (
                view._uploaded_revision != view.data_revision
                or self._inspection_identity()[1] != token
            ):
                raise ProjectFormatError("detector changed before the figure and profiles matched")
            buffer = QBuffer()
            if not buffer.open(QIODevice.OpenModeFlag.WriteOnly) or not figure.save(buffer, "PNG"):
                raise OSError("PNG encoding failed")
            figure_png = bytes(buffer.data())
            metadata = {
                "schema": "slate.inspection_profiles.v1",
                "project_uuid": str(self.project.project_id),
                "acquisition_uuid": str(acquisition.acquisition_id),
                "acquisition_name": acquisition.name,
                "decoded_source_sha256": acquisition.source_sha256,
                "source_hash_scope": "decoded OSC header and payload",
                "data_revision": f"decoded:{acquisition.source_sha256}/panel:{view.data_revision}",
                "mask_revision": str(view.mask_revision),
                "mask_visible": str(view.show_mask),
                "mask_provenance": json.dumps(
                    acquisition.mask.provenance if acquisition.mask else ()
                ),
                "native_shape_rows_columns": json.dumps(view.image.shape),
                "profile_measure": str(panel.profile_measure_control.currentData()),
                "profile_scope": str(panel.profile_scope_control.currentData()),
                "row_bounds_half_open": json.dumps(profiles.row_bounds),
                "column_bounds_half_open": json.dumps(profiles.column_bounds),
                "roi_column_row_bounds_half_open": json.dumps(panel._roi_bounds),
                "crosshair_column_px": str(view.crosshair[0]),
                "crosshair_row_px": str(view.crosshair[1]),
                "row_width_px": str(panel.row_width_control.value()),
                "column_width_px": str(panel.column_width_control.value()),
                "coordinate_unit": "native detector pixel index",
                "missing_semantics": "support=0 is missing; sum=0 or mean=nan is not measured zero",
                "figure_file": figure_path.name,
                "figure_sha256": hashlib.sha256(figure_png).hexdigest(),
                "figure_device_pixel_ratio": format(figure.devicePixelRatio(), ".17g"),
                "figure_width_height_physical_px": json.dumps((figure.width(), figure.height())),
                "display_contrast_mode": view.contrast_mode,
                "display_low_high_counts": json.dumps((view.low_value, view.high_value)),
            }
            profiles_csv = profile_csv(
                profiles.horizontal,
                profiles.vertical,
                profiles.horizontal_support,
                profiles.vertical_support,
                metadata,
            )
            if self._inspection_identity()[1] != token:
                raise ProjectFormatError("detector changed during figure and profile capture")
            request = inspection_export_request(
                figure_path, profiles_path, figure_png, profiles_csv, project_path, protected
            )
            task = InspectionExportTask(
                self.project.project_id,
                acquisition.acquisition_id,
                view.data_revision,
                acquisition.name,
                figure_path,
                profiles_path,
                metadata["figure_sha256"],
                hashlib.sha256(profiles_csv).hexdigest(),
                (perf_counter() - started) * 1000,
            )
            self._active_export = task
            self._active_kind = "export"
            identity = self.jobs.submit(
                JobRequest(
                    task.project_id,
                    task.acquisition_id,
                    Revisions(data=task.data_revision),
                    request,
                    len(request),
                    16 * 1024,
                    publish_inspection_export,
                )
            )
            self._active_generation = identity.generation
            self.statusBar().showMessage(f"Exporting {acquisition.name} figure and exact profiles")
        except (OSError, ProjectFormatError, RuntimeError, TypeError, ValueError) as exc:
            self._active_export = None
            if self._active_kind == "export":
                self._active_kind = None
                self._active_generation = None
            QMessageBox.warning(self, "Inspection export rejected", str(exc))

    def _comparison_identity(self) -> tuple[object, ...]:
        comparison = self.comparison_panel
        if not all(comparison.ready) or any(comparison.cut_pending) or comparison.pin_pending:
            raise ProjectFormatError("Comparison preparation is incomplete")
        bindings = []
        for slot, panel in enumerate(comparison.panels):
            acquisition = comparison.acquisitions[slot]
            current = next(
                (
                    a
                    for a in self.project.acquisitions
                    if a.acquisition_id == comparison.desired[slot]
                ),
                None,
            )
            if current != acquisition or panel._mask_pending or panel._profile_pending:
                raise ProjectFormatError("Comparison source, mask or profile is obsolete")
            if panel._current_profiles is None or panel._profile_key != panel._query_key():
                raise ProjectFormatError("Comparison profiles are not current")
            if comparison.lines[slot] is not None and comparison.cut_keys[
                slot
            ] != comparison.line_key(slot):
                raise ProjectFormatError("Comparison line samples are not current")
            check = self._source_checks.get(acquisition.acquisition_id)
            if check is None or check.state != "verified":
                raise ProjectFormatError("Comparison source is not verified")
            bindings.append(
                (
                    acquisition,
                    id(panel.view.image),
                    panel.view.data_revision,
                    panel._query_key(),
                    id(panel._current_profiles),
                    comparison.cut_keys[slot],
                    id(comparison.samples[slot]),
                )
            )
        return (
            self.project.project_id,
            tuple(bindings),
            comparison.pin_identity,
            id(comparison.pin),
            comparison.capture_state(True),
        )

    def _choose_comparison_export(self) -> None:
        if self.jobs.busy or self._active_kind is not None or self._close_intent:
            self.comparison_panel.message.setText("Finish current preparation before export")
            return
        try:
            token = self._comparison_identity()
        except (ValueError, ProjectFormatError) as exc:
            self.comparison_panel.message.setText(str(exc))
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export comparison figure and exact samples",
            str(Path.home() / "detector-comparison.png"),
            "PNG figure (*.png)",
        )
        if not filename:
            return
        try:
            if (
                token != self._comparison_identity()
                or self.jobs.busy
                or self._active_kind is not None
            ):
                raise ProjectFormatError("Comparison changed during destination choice")
            protected = self._input_reference_paths()
            project_path = self._project_path or self._recovery_path()
            figure_path = external_export_destination(
                Path(filename), project_path, protected, suffix=".png"
            )
            profiles_path = external_export_destination(
                figure_path.with_name(f"{figure_path.stem}.comparison.csv"),
                project_path,
                protected,
                suffix=".csv",
            )
            if any(path.exists() or path.is_symlink() for path in (figure_path, profiles_path)):
                raise FileExistsError("Choose a new figure name; both export files must be new")
            comparison = self.comparison_panel
            started = perf_counter()
            figure = comparison.capture_figure()
            buffer = QBuffer()
            if not buffer.open(QIODevice.OpenModeFlag.WriteOnly) or not figure.save(buffer, "PNG"):
                raise OSError("PNG encoding failed")
            figure_png = bytes(buffer.data())
            figure_hash = hashlib.sha256(figure_png).hexdigest()
            metadata = {
                "schema": "slate.comparison_samples.v1",
                "project_uuid": str(self.project.project_id),
                "figure_file": figure_path.name,
                "figure_sha256": figure_hash,
                "figure_device_pixel_ratio": format(figure.devicePixelRatio(), ".17g"),
                "linked_navigation": str(comparison.link.isChecked()),
                "locked_raw_count_limits": str(comparison.lock_limits.isChecked()),
                "normalization": "none; raw native counts; no solid angle correction",
                "line_policy": "nearest native pixel center; ties to larger index; exact endpoints; actual spacing <= requested; no averaging; repeated pixels are display samples",
                "missing_semantics": "support=0 is missing; value is not a measured zero; plots break at missing support",
                "pin_status": comparison.pin_status.text(),
            }
            for slot, name in enumerate(("A", "B")):
                acquisition, panel = comparison.acquisitions[slot], comparison.panels[slot]
                view = panel.view
                line = comparison.lines[slot]
                record = {
                    "acquisition_uuid": str(acquisition.acquisition_id),
                    "acquisition_name": acquisition.name,
                    "source_path": str(acquisition.source_path),
                    "decoded_source_sha256": acquisition.source_sha256,
                    "source_hash_scope": "decoded OSC header and payload; verified at admission",
                    "native_shape_rc": view.image.shape,
                    "data_revision": view.data_revision,
                    "mask_revision": view.mask_revision,
                    "mask_provenance": acquisition.mask.provenance if acquisition.mask else (),
                    "query": panel.profile_query(),
                    "row_bounds_half_open": panel._current_profiles.row_bounds,
                    "column_bounds_half_open": panel._current_profiles.column_bounds,
                    "exposure_s": acquisition.metadata.exposure_s,
                    "detector_frame_identity": comparison.frames[slot],
                    "line": None
                    if line is None
                    else {
                        "start_column_row": line.start_column_row,
                        "end_column_row": line.end_column_row,
                        "spacing_px": line.spacing_px,
                        "revision": line.revision,
                    },
                    "display": {
                        "mode": view.contrast_mode,
                        "low_high": (view.low_value, view.high_value),
                        "zoom": view.effective_zoom(),
                        "pan_px": (view.pan.x(), view.pan.y()),
                        "scale_mode": view.scale_mode,
                    },
                }
                metadata[f"{name}.identity"] = json.dumps(record, allow_nan=False)
                metadata[f"{name}.measure"] = panel.profile_query()[6]
            if comparison.pin_identity is not None:
                pin = comparison.pin_identity
                metadata["pin.identity"] = json.dumps(
                    {
                        "acquisition_uuid": str(pin.acquisition_id),
                        "decoded_source_sha256": pin.source_sha256,
                        "native_shape_rc": pin.native_shape_rc,
                        "mask_revision": pin.mask_revision,
                        "query": pin.query,
                    },
                    allow_nan=False,
                )
            exact_csv = comparison_csv(
                tuple(p._current_profiles for p in comparison.panels),
                tuple(comparison.samples),
                comparison.pin,
                metadata,
            )
            if token != self._comparison_identity() or any(
                p.view._uploaded_revision != p.view.data_revision for p in comparison.panels
            ):
                raise ProjectFormatError(
                    "Comparison changed before figure and exact samples matched"
                )
            request = inspection_export_request(
                figure_path, profiles_path, figure_png, exact_csv, project_path, protected
            )
            acquisition = comparison.acquisitions[comparison.active.currentIndex()]
            data_revision = comparison.panels[comparison.active.currentIndex()].view.data_revision
            task = InspectionExportTask(
                self.project.project_id,
                acquisition.acquisition_id,
                data_revision,
                "comparison",
                figure_path,
                profiles_path,
                figure_hash,
                hashlib.sha256(exact_csv).hexdigest(),
                (perf_counter() - started) * 1000,
            )
            self._active_export, self._active_kind = task, "export"
            identity = self.jobs.submit(
                JobRequest(
                    task.project_id,
                    task.acquisition_id,
                    Revisions(data=task.data_revision),
                    request,
                    len(request),
                    16 * 1024,
                    publish_inspection_export,
                )
            )
            self._active_generation = identity.generation
            self.statusBar().showMessage("Exporting comparison figure and exact samples")
        except (OSError, ProjectFormatError, RuntimeError, TypeError, ValueError) as exc:
            self._active_export = None
            if self._active_kind == "export":
                self._active_kind = self._active_generation = None
            comparison.message.setText(f"Comparison export rejected: {exc}")

    def _choose_metadata_export(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export acquisition metadata",
            str(Path.home() / "acquisitions.csv"),
            "CSV (*.csv)",
        )
        if not filename:
            return
        try:
            path = external_export_destination(
                Path(filename),
                self._project_path or self._recovery_path(),
                self._input_reference_paths(),
                suffix=".csv",
            )
            path.write_text(metadata_csv(self.project), encoding="utf-8", newline="")
        except (OSError, ProjectFormatError) as exc:
            QMessageBox.warning(self, "Metadata export rejected", str(exc))
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
                if self._active_mask is not None and self._active_mask[0] == acquisition_id:
                    self.jobs.invalidate()
                self._pending_masks.pop(acquisition_id, None)
                self._mask_history.pop(acquisition_id, None)
                self._mask_cache.pop(acquisition_id, None)
                self._resident_planes.pop(acquisition_id, None)
                self._thumbnails.pop(acquisition_id, None)
                self._source_checks.pop(acquisition_id, None)
                for kind in ("cif", "configuration", "configuration_cif"):
                    self._reference_checks.pop((acquisition_id, kind), None)
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

    def _mask_gesture(self, gesture: MaskGesture) -> None:
        acquisition_id = self.selected_acquisition_id
        if (
            self._close_intent
            or acquisition_id is None
            or acquisition_id != self._visible_acquisition_id
            or acquisition_id not in self._resident_planes
        ):
            return
        pending = self._pending_masks.get(acquisition_id, ())
        pending_bytes = sum(
            256 + 128 * len(g.points) + len(str(g.import_path or ""))
            for batch in self._pending_masks.values()
            for g in batch
        )
        if (
            len(pending) >= MAX_ACTIONS
            or pending_bytes + 256 + 128 * len(gesture.points) + len(str(gesture.import_path or ""))
            > 2 * 1024 * 1024
        ):
            self.statusBar().showMessage("Mask queue is full; wait for preparation")
            return
        self._pending_masks[acquisition_id] = (*pending, gesture)
        self.detector_panel.set_mask_pending(True)
        self.detector_panel.export_button.setEnabled(False)
        self.statusBar().showMessage(
            "Preparing profiles; committed mask/profile revision remains visible"
        )
        self._refresh_mask_history()
        self._refresh_comparison()
        if self._active_kind == "mask":
            self.jobs.cancel()
        QTimer.singleShot(0, self._dispatch_pending)

    def _choose_mask(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import native Boolean inclusion mask (True includes)",
            str(Path.home()),
            "Native Boolean masks (*.npy)",
        )
        if filename:
            self.import_mask(Path(filename))

    def import_mask(self, path: Path) -> None:
        reason = self.detector_panel.mask_reason.currentData()
        if reason == 0:
            self.statusBar().showMessage("Choose an exclusion reason for additive mask import")
            return
        self._mask_gesture(MaskGesture("import", reason, import_path=Path(path).absolute()))

    def _request_mask_profiles(self) -> None:
        if (
            self._active_kind == "mask"
            and self._active_mask is not None
            and not self._active_mask[2]
        ):
            target = self._active_profile_target
            query = (
                self.comparison_panel.pin_identity.query
                if target == "pin" and self.comparison_panel.pin_identity is not None
                else self.comparison_panel.panels[target].profile_query()
                if type(target) is int
                else self.detector_panel.profile_query()
            )
            if query != self._active_profile_query:
                self.jobs.cancel()
        QTimer.singleShot(0, self._dispatch_pending)

    def _comparison_selection(self, slot: int, acquisition_id: UUID | None) -> None:
        if acquisition_id is not None:
            self._activate_acquisition(acquisition_id)
        self._refresh_comparison()
        QTimer.singleShot(0, self._dispatch_pending)

    def _comparison_plane(self, acquisition_id: UUID) -> PreparedOsc | None:
        plane = self._resident_planes.get(acquisition_id)
        if plane is not None:
            return plane
        for acquisition, retained in zip(
            self.comparison_panel.acquisitions, self.comparison_panel.planes, strict=True
        ):
            if acquisition is not None and acquisition.acquisition_id == acquisition_id:
                return retained
        return None

    def _refresh_comparison(self) -> None:
        comparison = self.comparison_panel
        acquisitions = {a.acquisition_id: a for a in self.project.acquisitions}
        for slot, wanted in enumerate(comparison.desired):
            acquisition = acquisitions.get(wanted)
            check = self._source_checks.get(wanted)
            plane = self._comparison_plane(wanted) if wanted is not None else None
            if acquisition is None or check is None or check.state != "verified":
                comparison.unavailable(slot, check.state if check else "source not verified")
                continue
            if plane is None or plane.decoded_sha256 != acquisition.source_sha256:
                comparison.unavailable(slot, "preparing native source")
                continue
            if self._pending_masks.get(wanted):
                comparison._restored_link |= comparison.link.isChecked()
                comparison.panels[slot].set_mask_pending(True)
                comparison.unavailable(slot, "preparing mask edits; prior revision retained")
                continue
            if (
                comparison.panels[slot].view.max_texture_axis is None
                or comparison.magnifier.max_texture_axis is None
            ):
                comparison.acquisitions[slot], comparison.planes[slot] = acquisition, plane
                comparison.unavailable(slot, "open Compare images to initialize the display")
                continue
            key = self._reciprocal_key(acquisition)
            cached = self._reciprocal_cache.get(wanted)
            frame = (
                detector_frame_key(cached[1].saved.instrument)
                if key is not None and cached is not None and cached[0][:10] == key[:10]
                else None
            )
            restoring = self._restoring_view
            self._restoring_view = (
                restoring
                or comparison._restore[slot] is not None
                or (comparison._restored_link or comparison._restored_limits is not None)
            )
            try:
                comparison.admit(slot, acquisition, plane, self._mask_cache.get(wanted), frame)
            finally:
                self._restoring_view = restoring
        pin = comparison.pin_identity
        if pin is not None:
            acquisition = acquisitions.get(pin.acquisition_id)
            check = self._source_checks.get(pin.acquisition_id)
            valid = acquisition is not None and acquisition.source_sha256 == pin.source_sha256
            valid = (
                valid
                and (acquisition.mask.revision if acquisition.mask else 0) == pin.mask_revision
            )
            valid = valid and check is not None and check.state == "verified"
            comparison.pin_unavailable_reason = (
                None if valid else "Source/acquisition/mask no longer matches the pinned identity"
            )
            if not valid:
                comparison.pin_pending = False
            comparison._show_pin()
        comparison.update_frames()

    def _request_comparison_frames(self) -> None:
        self._comparison_frames_pending = deque(
            dict.fromkeys(
                acquisition_id
                for acquisition_id in self.comparison_panel.desired
                if acquisition_id is not None
            )
        )
        QTimer.singleShot(0, self._dispatch_pending)

    def _start_comparison_preparation(self) -> bool:
        comparison = self.comparison_panel
        wanted = list(comparison.desired)
        if comparison.pin_pending and comparison.pin_identity is not None:
            wanted.append(comparison.pin_identity.acquisition_id)
        for acquisition_id in dict.fromkeys(wanted):
            acquisition = next(
                (a for a in self.project.acquisitions if a.acquisition_id == acquisition_id), None
            )
            check = self._source_checks.get(acquisition_id)
            if acquisition is None or check is None or check.state != "verified":
                continue
            if self._comparison_plane(acquisition_id) is None:
                self._activate_acquisition(acquisition_id)
                if self._active_kind is None and not self.jobs.busy:
                    self._submit_import(acquisition.source_path, acquisition_id)
                return True
        while self._comparison_frames_pending:
            acquisition_id = self._comparison_frames_pending.popleft()
            acquisition = next(
                (a for a in self.project.acquisitions if a.acquisition_id == acquisition_id), None
            )
            if acquisition is None or self._reciprocal_key(acquisition) is None:
                continue
            self._activate_acquisition(acquisition_id)
            self._request_reciprocal_preview()
            return self._active_kind is not None
        return False

    def _request_line(self, slot: int) -> None:
        if self._active_kind == "line" and self._active_line is not None:
            old_slot, old_key = self._active_line
            if old_slot == slot and old_key != self.comparison_panel.line_key(slot):
                self.jobs.cancel()
        QTimer.singleShot(0, self._dispatch_pending)

    def _start_line_work(self) -> bool:
        comparison = self.comparison_panel
        for slot in range(2):
            panel = comparison.panels[slot]
            key = comparison.line_key(slot)
            if not comparison.cut_pending[slot] or not comparison.ready[slot] or key is None:
                continue
            work = LineWork(panel.view.image, panel._mask_inclusion, comparison.lines[slot])
            self._active_kind = "line"
            self._active_line = (slot, key)
            try:
                identity = self.jobs.submit(
                    JobRequest(
                        self.project.project_id,
                        key[0],
                        Revisions(data=key[3], mask=key[4], model=work.definition.revision),
                        work,
                        work.argument_bytes,
                        512 * 1024,
                        prepare_line,
                    )
                )
            except (ValueError, RuntimeError, TypeError) as exc:
                self._active_kind = self._active_line = None
                comparison.cut_pending[slot] = False
                comparison.message.setText(f"Line unavailable: {exc}")
                return False
            self._active_generation = identity.generation
            return True
        return False

    def _line_ready(self, identity: JobIdentity, value: object) -> None:
        active, self._active_line = self._active_line, None
        if active is None or not isinstance(value, LineSamples):
            return
        slot, key = active
        if identity.project_id != self.project.project_id or identity.acquisition_id != key[0]:
            return
        if identity.revisions != Revisions(
            data=key[3], mask=key[4], model=value.definition.revision
        ):
            return
        self.comparison_panel.publish_cut(slot, key, value)

    def _start_mask_work(self) -> bool:
        acquisition_id = self.selected_acquisition_id
        panel = self.detector_panel
        if not (
            acquisition_id in self._resident_planes
            and (self._pending_masks.get(acquisition_id) or panel._profile_pending)
        ):
            acquisition_id = next(
                (key for key in self._pending_masks if key in self._resident_planes), None
            )
        target: int | str | None = None
        if acquisition_id is None:
            for slot, wanted in enumerate(self.comparison_panel.desired):
                p = self.comparison_panel.panels[slot]
                retained = self._comparison_plane(wanted) if wanted is not None else None
                check = self._source_checks.get(wanted)
                if (
                    retained is not None
                    and check is not None
                    and check.state == "verified"
                    and p._profile_pending
                ):
                    acquisition_id, panel, target = wanted, p, slot
                    break
        if acquisition_id is None and self.comparison_panel.pin_pending:
            pin = self.comparison_panel.pin_identity
            if pin is not None and self._comparison_plane(pin.acquisition_id) is not None:
                acquisition_id, target = pin.acquisition_id, "pin"
        if acquisition_id is None:
            return False
        acquisition = next(
            (item for item in self.project.acquisitions if item.acquisition_id == acquisition_id),
            None,
        )
        if acquisition is None:
            self._pending_masks.pop(acquisition_id, None)
            return False
        plane = self._comparison_plane(acquisition_id)
        if plane is None:
            return False
        state = acquisition.mask or NativeMask(acquisition.source_sha256, plane.native_counts.shape)
        gestures = self._pending_masks.get(acquisition_id, ())
        visible = (
            target is not None
            or acquisition_id == self._visible_acquisition_id == self.selected_acquisition_id
        )
        query = (
            panel.profile_query()
            if visible
            else (state.shape[1] // 2, state.shape[0] // 2, 1, 1, "band", None, "sum")
        )
        if target == "pin":
            pin = self.comparison_panel.pin_identity
            if pin is None or (state.revision, state.source_sha256, state.shape) != (
                pin.mask_revision,
                pin.source_sha256,
                pin.native_shape_rc,
            ):
                self.comparison_panel.pin_pending = False
                return False
            query = pin.query
        work = MaskWork(
            plane.native_counts, state, gestures, query, self._mask_cache.get(acquisition_id)
        )
        self._active_kind = "mask"
        self._active_mask = (acquisition_id, state, gestures)
        self._active_profile_target = target
        self._active_profile_query = query
        try:
            identity = self.jobs.submit(
                JobRequest(
                    self.project.project_id,
                    acquisition_id,
                    Revisions(data=1, mask=state.revision),
                    work,
                    work.argument_bytes,
                    MAX_RESULT_BYTES,
                    prepare_mask,
                )
            )
        except (ValueError, RuntimeError, TypeError) as exc:
            self._pending_masks.pop(acquisition_id, None)
            self._active_kind = self._active_mask = None
            panel.set_mask_pending(False)
            panel._profile_pending = False
            self.statusBar().showMessage(str(exc))
            return False
        self._active_generation = identity.generation
        return True

    def _mask_ready(self, identity: JobIdentity, value: object) -> None:
        active, self._active_mask = self._active_mask, None
        target, self._active_profile_target = self._active_profile_target, None
        self._active_profile_query = None
        if (
            active is None
            or not isinstance(value, PreparedMask)
            or identity.project_id != self.project.project_id
        ):
            return
        acquisition_id, base, gestures = active
        acquisition = next(
            (item for item in self.project.acquisitions if item.acquisition_id == acquisition_id),
            None,
        )
        if (
            acquisition is None
            or identity.acquisition_id != acquisition_id
            or identity.revisions != Revisions(data=1, mask=base.revision)
            or acquisition.source_sha256 != value.mask.source_sha256
            or (acquisition.mask is not None and acquisition.mask != base)
            or self._pending_masks.get(acquisition_id, ())[: len(gestures)] != gestures
        ):
            return
        candidate = replace(
            self.project,
            acquisitions=tuple(
                replace(item, mask=value.mask) if item.acquisition_id == acquisition_id else item
                for item in self.project.acquisitions
            ),
        )
        try:
            self._validate_project_admission(candidate)
        except ValueError as exc:
            self._pending_masks.pop(acquisition_id, None)
            self.detector_panel.set_mask_pending(False)
            self.detector_panel._profile_pending = False
            self.statusBar().showMessage(f"Mask was not committed: {exc}")
            return
        remainder = self._pending_masks.get(acquisition_id, ())[len(gestures) :]
        if remainder:
            self._pending_masks[acquisition_id] = remainder
        else:
            self._pending_masks.pop(acquisition_id, None)
        if value.edits:
            self.project = candidate
            history = self._mask_history.setdefault(acquisition_id, MaskHistory())
            for before, after in value.edits:
                history.push(before, after)
            self._mask_history.move_to_end(acquisition_id)
            while sum(h.storage_bytes for h in self._mask_history.values()) > 8 * 1024 * 1024:
                self._mask_history.popitem(last=False)
            self._mark_dirty()
        self._mask_cache[acquisition_id] = replace(value, edits=())
        self._mask_cache.move_to_end(acquisition_id)
        while len(self._mask_cache) > MAX_CACHED_PLANES:
            self._mask_cache.popitem(last=False)
        if acquisition_id == self.selected_acquisition_id == self._visible_acquisition_id and (
            gestures
            or value.query == self.detector_panel.profile_query()
            or self.detector_panel.view.mask_reasons is not value.reasons
        ):
            self.detector_panel.publish_mask(value)
            self.detector_panel.set_mask_pending(bool(remainder))
            self.detector_panel.export_button.setEnabled(not remainder)
        if type(target) is int:
            p = self.comparison_panel.panels[target]
            if (
                self.comparison_panel.desired[target] == acquisition_id
                and value.query == p.profile_query()
            ):
                p.publish_mask(value)
        elif target == "pin":
            pin = self.comparison_panel.pin_identity
            if pin is not None and (
                pin.acquisition_id,
                pin.source_sha256,
                pin.mask_revision,
                pin.query,
            ) == (acquisition_id, value.mask.source_sha256, value.mask.revision, value.query):
                self.comparison_panel.publish_pin(pin, value.profiles)
        self._refresh_comparison()
        self._refresh_mask_history()
        self.statusBar().showMessage(f"Mask revision {value.mask.revision}; profiles ready")

    def _refresh_mask_history(self) -> None:
        history = self._mask_history.get(self.selected_acquisition_id)
        busy = (
            bool(self._pending_masks.get(self.selected_acquisition_id))
            or self._active_kind == "mask"
        )
        self.detector_panel.mask_undo_button.setEnabled(bool(history and history.undo and not busy))
        self.detector_panel.mask_redo_button.setEnabled(bool(history and history.redo and not busy))
        self.detector_panel.mask_history_label.setText(
            f"Undo {len(history.undo) if history else 0}; redo {len(history.redo) if history else 0} · max 32 / 512 KiB"
        )
        self.detector_panel.mask_history_label.setToolTip(
            "Session history: 32 actions / 512 KiB per image; 8 MiB total. Save persists exclusions, not undo history."
        )

    def _mask_undo_redo(self, *, undo: bool) -> None:
        acquisition_id = self.selected_acquisition_id
        history = self._mask_history.get(acquisition_id)
        if not history or self._pending_masks.get(acquisition_id) or self._active_kind == "mask":
            return
        acquisition = next(
            item for item in self.project.acquisitions if item.acquisition_id == acquisition_id
        )
        if acquisition.mask is None:
            return
        try:
            updated = history.move(acquisition.mask, undo=undo)
        except ValueError as exc:
            self.statusBar().showMessage(f"Mask history change rejected: {exc}")
            return
        if updated is acquisition.mask:
            return
        candidate = replace(
            self.project,
            acquisitions=tuple(
                replace(item, mask=updated) if item.acquisition_id == acquisition_id else item
                for item in self.project.acquisitions
            ),
        )
        try:
            self._validate_project_admission(candidate)
        except ValueError as exc:
            source, destination = (
                (history.undo, history.redo) if undo else (history.redo, history.undo)
            )
            source.append(destination.pop())
            self.statusBar().showMessage(f"Mask history change rejected: {exc}")
            return
        self.project = candidate
        self._mark_dirty()
        self.detector_panel.set_mask_pending(True)
        self.detector_panel._profile_pending = True
        self.detector_panel.export_button.setEnabled(False)
        self._refresh_mask_history()
        QTimer.singleShot(0, self._dispatch_pending)

    def _cancel_current(self) -> None:
        if self._active_kind == "prepared" or self._pending_prepared is not None:
            self._pending_prepared = None
            self.prepared.status.setText("Cancel acknowledged; prior prepared state retained")
        if self._active_kind == "physical" or self._pending_physical is not None:
            self.physical.cancel(clear=True)
        joint_requested = self._active_kind == "joint" or self._pending_joint is not None
        self._pending_prepared = None
        self._pending_joint = None
        if joint_requested:
            self.joint.status.setText("Cancel acknowledged; pending work cleared; draining safely")
            self.joint.cancel_button.setEnabled(False)
            self.joint.status.repaint()
        sample_requested = self._active_kind == "sample" or self._pending_sample is not None
        self._pending_sample = None
        if sample_requested:
            self.sample.status.setText("Cancel acknowledged; pending work cleared; draining safely")
            self.sample.cancel_button.setEnabled(False)
            self.sample.status.repaint()
        hbn_requested = self._active_kind == "hbn" or self._pending_hbn is not None
        self._pending_hbn = None
        if hbn_requested:
            self.hbn.status.setText(
                "Cancel acknowledged; pending hBN requests cleared. Waiting for active cooperative drain."
            )
            self.hbn.cancel_button.setEnabled(False)
            self.hbn.status.repaint()
        if self._active_kind == "mask" and self._active_mask is not None:
            acquisition_id = self._active_mask[0]
            self._pending_masks.pop(acquisition_id, None)
            acquisition = next(
                item for item in self.project.acquisitions if item.acquisition_id == acquisition_id
            )
            if acquisition_id == self.selected_acquisition_id == self._visible_acquisition_id:
                rebuild = (
                    acquisition.mask is not None
                    and acquisition.mask.revision != self.detector_panel.view.mask_revision
                )
                self.detector_panel._profile_pending = rebuild
                self.detector_panel.set_mask_pending(rebuild)
        if self._active_kind == "batch":
            for candidate_id in self._candidate_queue:
                candidate = self._candidates.get(candidate_id)
                if candidate is not None:
                    candidate.status = "canceled"
                    candidate.detail = "Batch canceled"
            self._candidate_queue.clear()
            self._refresh_review_table()
        simulation_requested = (
            self._active_kind == "simulation" or self._pending_simulation is not None
        )
        self._pending_simulation = None
        self._pending_setup_request = None
        self._pending_setup_copy = None
        if simulation_requested:
            self.simulator.detector._profile_pending = False
        self.jobs.cancel()
        if simulation_requested:
            self.simulator.status.setText(
                "Simulation cancellation requested; queued requests discarded. Waiting for active safe stop and drain."
                if self.jobs.busy
                else "Simulation requests canceled; no queued work remains"
            )
            self.simulator.refresh()
            if self.simulator.draft_kind.currentData() == "native":
                self.simulator.cancel_button.setEnabled(False)
                self.simulator.status.repaint()

    def _submit_import(
        self, source: Path, acquisition_id: UUID, *, mode: Literal["import", "relink"] = "import"
    ) -> None:
        if (
            self._active_kind in ("save", "discard", "open")
            or self.jobs.busy
            or self._active_kind is not None
            or self._write_queue
            or self._pending_open
            or self._pending_masks
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
        previous_path = self._active_load_path
        previous_hash = self._active_load_hash
        self._active_kind = mode
        self._active_generation = None
        self._active_load_id = acquisition_id
        self._active_load_path = acquisition.source_path
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
            self._active_load_path = previous_path
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
                self._publish_plane(acquisition.acquisition_id, value, acquisition=acquisition)
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

    def _publish_plane(
        self, acquisition_id: UUID, value: PreparedOsc, *, acquisition: Acquisition | None = None
    ) -> None:
        if acquisition is None:
            acquisition = next(
                (a for a in self.project.acquisitions if a.acquisition_id == acquisition_id), None
            )
        if (
            acquisition is None
            or acquisition.acquisition_id != acquisition_id
            or acquisition.source_sha256 != value.decoded_sha256
        ):
            raise ValueError("detector publication requires the admitted acquisition identity")
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
        mask = acquisition.mask
        cached = self._mask_cache.get(acquisition_id)
        if mask is not None:
            if (
                mask.shape != value.native_counts.shape
                or mask.source_sha256 != value.decoded_sha256
            ):
                raise ValueError("Saved mask is incompatible with the admitted native source")
            if cached is not None and cached.mask == mask:
                self.detector_panel.publish_mask(cached)
            else:
                self.detector_panel.set_mask_pending(True)
                self.detector_panel._profile_pending = True
                QTimer.singleShot(0, self._dispatch_pending)
        if self._pending_masks.get(acquisition_id):
            self.detector_panel.set_mask_pending(True)
            QTimer.singleShot(0, self._dispatch_pending)
        self._refresh_mask_history()
        self._visible_details = (
            f"Native detector: {value.native_counts.shape[0]} rows x "
            f"{value.native_counts.shape[1]} columns\n"
            f"OSC header: version {value.version}, {value.byte_order} endian"
        )
        self._reciprocal_selection_changed()
        self._sync_scene()

        self._refresh_comparison()

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
            self._active_load_id != existing.acquisition_id
            or self._active_load_path != existing.source_path
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
        if saved.scene is not None:
            self._pending_scene_restore = saved.scene
            self.scene_tabs.setCurrentIndex(1 if saved.scene.visible else 0)
            self._sync_scene()
        if saved.comparison is not None and saved.comparison.visible:
            self.scene_tabs.setCurrentIndex(2)
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
        identity = self.scene_panel.view._image_identity
        if identity is not None and identity[0] == self.selected_acquisition_id:
            self._scene_cameras[identity[0]] = self.scene_panel.view.camera_state()
        self._pending_scene_restore = None
        self._reciprocal_epoch += 1
        self.selected_acquisition_id = selected_id
        self._supersede_hbn()
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
        self._refresh_numeric_editor()
        self._sync_scene()
        had_work = self.jobs.busy and self._active_kind not in (
            "save",
            "discard",
            "batch",
            "export",
        )
        if self._active_kind not in ("save", "discard", "batch", "export"):
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
        self._sync_numeric_load_button()
        terminal = state in (JobState.COMPLETED, JobState.CANCELED, JobState.FAILED)
        if summary.identity.generation != self.jobs.latest_generation:
            if terminal and summary.identity.generation == self._active_generation:
                if self._active_kind == "reference" and self._active_reference is not None:
                    task = self._active_reference
                    if task[3] == self.project.project_id:
                        self._invalidate_reference_request(
                            task[0], task[1], "Reference result discarded"
                        )
                    self._active_reference = None
                if self._active_kind == "setup" and self._setup_dialog is not None:
                    self._setup_dialog.message.setText(
                        "Setup superseded; no binding. Verified copies may remain in the reviewed data folder."
                    )
                    self._setup_dialog.cancel_button.setEnabled(False)
                if self._active_kind == "prepared":
                    self._prepared_context = None
                    self.prepared.status.setText(
                        "Superseded prepared operation; prior state retained"
                    )
                if self._active_kind == "joint":
                    self.joint.status.setText(
                        f"Superseded joint {state.value}; safe stop {summary.safe_stop_ms} ms; no late result admitted"
                    )
                    self._joint_context = None
                    self.joint.refresh()
                if self._active_kind == "sample":
                    self.sample.status.setText(
                        f"Superseded sample {state.value}; safe stop {summary.safe_stop_ms} ms; no late result admitted"
                    )
                    self._sample_context = None
                if self._active_kind == "hbn":
                    self.hbn.status.setText(
                        f"Superseded hBN {state.value}; safe stop {summary.safe_stop_ms} ms; no late result admitted"
                    )
                    self._hbn_context = None
                if self._active_kind == "simulation":
                    self.simulator.status.setText(
                        f"Superseded simulation {state.value}; safe stop {summary.safe_stop_ms} ms. No obsolete frame admitted."
                    )
                    self._simulation_operation = self._simulation_context = None
                self._active_kind = None
                self.joint.refresh()
                self.sample.refresh()
                self.hbn.refresh()
                self.simulator.refresh()
                self._active_generation = None
                self._active_load_id = None
                self._active_load_path = None
                self._active_load_hash = None
                self._active_numeric = None
                self._active_reciprocal = None
                self._active_mask = None
                self._active_line = None
                self._active_profile_target = self._active_profile_query = None
                if self._setup_dialog is not None:
                    self._setup_dialog.refresh()
                QTimer.singleShot(0, self._dispatch_pending)
            if self._obsolete_pending and not self.jobs.busy:
                self._obsolete_pending = False
                self.cancel_button.setEnabled(False)
                self._show_state("empty", "No image open", "The previous operation was discarded.")
                self.statusBar().showMessage("Prior operation discarded · Ready")
            return
        kind = self._active_kind
        if kind == "physical":
            self.physical.status.setText(
                f"Physical {state.value}: {summary.detail or 'canonical forward geometry'}"
            )
            if state in (JobState.FAILED, JobState.CANCELED):
                self._active_kind = self._active_generation = self._physical_context = None
                QTimer.singleShot(0, self._dispatch_pending)
            return
        if kind == "prepared":
            self.prepared.status.setText(f"Prepared {state.value}: {summary.detail or 'working'}")
            if state in (JobState.FAILED, JobState.CANCELED):
                self._active_kind = self._active_generation = self._prepared_context = None
                QTimer.singleShot(0, self._dispatch_pending)
            return
        if kind == "joint":
            self.joint.status.setText(
                f"Joint {state.value}: "
                + (
                    f"drained; safe stop {summary.safe_stop_ms} ms"
                    if state == JobState.CANCELED
                    else summary.detail or "working"
                )
            )
            if state in (JobState.FAILED, JobState.CANCELED):
                self._active_kind = self._active_generation = self._joint_context = None
                QTimer.singleShot(0, self._dispatch_pending)
            self.joint.refresh()
            return
        if kind == "sample":
            self.sample.status.setText(f"Sample {state.value}: {summary.detail or 'working'}")
            if state in (JobState.FAILED, JobState.CANCELED):
                self._active_kind = self._active_generation = self._sample_context = None
                QTimer.singleShot(0, self._dispatch_pending)
            self.sample.refresh()
            return
        if kind == "hbn":
            self.hbn.status.setText(f"hBN {state.value}: {summary.detail or 'working'}")
            if state in (JobState.FAILED, JobState.CANCELED):
                self._active_kind = self._active_generation = self._hbn_context = None
                QTimer.singleShot(0, self._dispatch_pending)
            self.hbn.refresh()
            return
        if kind == "simulation":
            self.simulator.status.setText(
                f"Simulation {state.value}: {summary.detail or 'working on its owning worker'}"
            )
            if state in (JobState.FAILED, JobState.CANCELED):
                if state == JobState.CANCELED and self._simulation_operation == "export":
                    self.simulator.status.setText(
                        "Export canceled safely. Completed configured figures may remain; existing files are never overwritten. Choose a new output directory before retrying figures."
                    )
                self._active_kind = self._active_generation = None
                self._simulation_operation = self._simulation_context = None
                self.simulator.detector._profile_pending = False
                QTimer.singleShot(0, self._dispatch_pending)
            self.simulator.refresh()
            return
        self._obsolete_pending = False
        self.cancel_button.setEnabled(
            kind
            in (
                "import",
                "relink",
                "batch",
                "reference",
                "open",
                "numeric",
                "reciprocal",
                "mask",
                "line",
                "setup",
            )
            and state in (JobState.QUEUED, JobState.RUNNING)
        )
        self.comparison_panel.cancel_button.setEnabled(self.cancel_button.isEnabled())
        if kind == "setup":
            if self._setup_dialog is not None:
                self._setup_dialog.cancel_button.setEnabled(
                    state in (JobState.QUEUED, JobState.RUNNING)
                )
                self._setup_dialog.message.setText(
                    f"Setup {state.value}: {summary.detail or 'working on the background worker'}"
                )
            if state in (JobState.FAILED, JobState.CANCELED):
                self._active_kind = None
                self._active_generation = None
                if self._setup_dialog is not None:
                    self._setup_dialog.message.setText(
                        f"Setup {state.value}: {summary.detail}. No new binding; completed byte-verified copies may remain in the reviewed data folder."
                    )
                    self._setup_dialog.refresh()
                QTimer.singleShot(0, self._dispatch_pending)
            return
        if kind == "line":
            if state == JobState.CANCEL_REQUESTED:
                self.statusBar().showMessage("Cancel requested; stopping line sampling")
            if state in (JobState.FAILED, JobState.CANCELED):
                active, self._active_line = self._active_line, None
                if active is not None:
                    slot, key = active
                    if key == self.comparison_panel.line_key(slot):
                        self.comparison_panel.cut_pending[slot] = False
                        self.comparison_panel.message.setText(
                            f"Line {state.value}; prior samples retained: {summary.detail}"
                        )
                self._active_kind = None
                self._active_generation = None
                QTimer.singleShot(0, self._dispatch_pending)
            return
        if kind == "mask":
            self.detector_panel.mask_cancel_button.setEnabled(
                state in (JobState.QUEUED, JobState.RUNNING)
            )
            if state == JobState.CANCEL_REQUESTED:
                self.statusBar().showMessage("Cancel requested; stopping mask preparation")
            if state in (JobState.FAILED, JobState.CANCELED):
                active = self._active_mask
                if state == JobState.FAILED and active is not None:
                    self._pending_masks.pop(active[0], None)
                    self.detector_panel._profile_pending = False
                    self.detector_panel.set_mask_pending(False)
                    target = self._active_profile_target
                    if type(target) is int:
                        p = self.comparison_panel.panels[target]
                        p._profile_pending = False
                        self.comparison_panel.unavailable(
                            target, summary.detail or "profile failure"
                        )
                    elif target == "pin":
                        self.comparison_panel.pin_pending = False
                        self.comparison_panel.pin_status.setText(
                            "Reference rebind failed; unavailable"
                        )
                self._active_mask = None
                self._active_profile_target = None
                self._active_profile_query = None
                self._active_kind = None
                self._active_generation = None
                self.statusBar().showMessage(f"Mask preparation {state.value}: {summary.detail}")
                if state == JobState.CANCELED:
                    self._refresh_comparison()
                QTimer.singleShot(0, self._dispatch_pending)
            return
        if kind == "numeric":
            if state in (JobState.FAILED, JobState.CANCELED):
                self._active_numeric = None
                self._active_kind = None
                self._active_generation = None
                self._refresh_numeric_editor()
                self.statusBar().showMessage(f"Numeric draft {state.value}: {summary.detail}")
                QTimer.singleShot(0, self._dispatch_pending)
            return
        if kind == "reciprocal":
            if state in (JobState.FAILED, JobState.CANCELED):
                self._active_reciprocal = None
                self._active_kind = None
                self._active_generation = None
                self._refresh_reciprocal_editor()
                self.reciprocal_status.setText(
                    f"Reciprocal coverage {state.value}: {summary.detail or 'no result published'}"
                )
                self.statusBar().showMessage(f"Reciprocal coverage {state.value}: {summary.detail}")
                QTimer.singleShot(0, self._dispatch_pending)
            return
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
                if self._active_reference is not None:
                    task = self._active_reference
                    if task[3] == self.project.project_id:
                        self._invalidate_reference_request(
                            task[0], task[1], f"Reference check {state.value}"
                        )
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
        if kind == "export":
            if state in (JobState.FAILED, JobState.CANCELED):
                self._active_export = None
                self._active_kind = None
                self._active_generation = None
                self.statusBar().showMessage(f"Inspection export {state.value}: {summary.detail}")
                if not self._close_intent:
                    QMessageBox.warning(
                        self,
                        "Inspection export failed",
                        summary.detail or "Export stopped before a completed write receipt.",
                    )
                QTimer.singleShot(0, self._dispatch_pending)
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
            self._active_load_path = None
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
            elif kind == "export":
                self._export_ready(identity, value)
            elif kind == "open":
                self._open_ready(identity, value)
            elif kind == "batch":
                self._batch_ready(identity, value)
            elif kind == "reference":
                self._reference_ready(identity, value)
            elif kind == "numeric":
                self._numeric_ready(identity, value)
            elif kind == "reciprocal":
                self._reciprocal_ready(identity, value)
            elif kind == "mask":
                self._mask_ready(identity, value)
            elif kind == "line":
                self._line_ready(identity, value)
            elif kind == "physical":
                if self._physical_current(identity) and isinstance(value, PhysicalResult):
                    self.physical.guard(lambda: self.physical.ready(value))
                else:
                    self.physical.status.setText(
                        "Late/stale physical completion rejected; initial values unchanged"
                    )
            elif kind == "prepared":
                if self._prepared_current(identity) and isinstance(value, PreparedWorkResult):
                    self.prepared.guard(lambda: self.prepared.ready(value))
                else:
                    self.prepared.status.setText(
                        "Stale prepared completion rejected; prior description retained"
                    )
            elif kind == "joint":
                if self._joint_current(identity) and isinstance(value, JointWorkResult):
                    self.joint.guard(lambda: self.joint.ready(value))
                else:
                    self.joint.status.setText(
                        "Late/stale joint completion rejected; prior state retained"
                    )
            elif kind == "sample":
                if self._sample_current(identity) and isinstance(value, SampleWorkResult):
                    self.sample.guard(lambda: self.sample.ready(value))
                else:
                    self.sample.status.setText(
                        "Late/stale sample completion rejected; prior state retained"
                    )
            elif kind == "hbn":
                if self._hbn_current(identity) and isinstance(value, HbnWorkResult):
                    self.hbn.guard(lambda: self.hbn.ready(value))
                else:
                    self.hbn.status.setText(
                        "Late/stale hBN completion rejected; prior state retained"
                    )
            elif kind == "simulation":
                if self._simulation_current(identity):
                    if self._simulation_operation in (
                        "load",
                        "validate",
                        "save_configuration",
                        "native_load",
                        "native_validate",
                        "native_save",
                    ):
                        self.simulator.ready(*value)
                    else:
                        self.simulator.ready(self._simulation_operation, value)
            elif kind == "setup":
                if (
                    self._setup_context_current()
                    and isinstance(value, tuple)
                    and len(value) == 2
                    and self._setup_dialog is not None
                ):
                    self._setup_dialog.ready(*value)
                elif self._setup_dialog is not None:
                    self._setup_dialog.message.setText(
                        "Stale setup completion rejected; no binding. Verified copies may remain in the reviewed folder."
                    )
            elif kind in ("import", "relink"):
                self._import_ready(identity, value)
        finally:
            self._active_kind = None
            if kind == "physical":
                self._physical_context = None
            if kind == "prepared":
                self._prepared_context = None
            if kind == "joint":
                self._joint_context = None
                self.joint.refresh()
            if kind == "sample":
                self._sample_context = None
                self.sample.refresh()
            if kind == "hbn":
                self._hbn_context = None
                self.hbn.refresh()
            if kind == "simulation":
                self._simulation_operation = self._simulation_context = None
                self.simulator.refresh()
            self._active_generation = None
            self._active_candidate_id = None
            self._active_load_id = None
            self._active_load_path = None
            self._active_load_hash = None
            self._sync_numeric_load_button()
            self.detector_panel.mask_cancel_button.setEnabled(False)
            self._refresh_mask_history()
            if self._setup_dialog is not None:
                self._setup_dialog.refresh()
            QTimer.singleShot(0, self._dispatch_pending)

    def _export_ready(self, identity: JobIdentity, value: object) -> None:
        task = self._active_export
        self._active_export = None
        if (
            task is None
            or not isinstance(value, InspectionExportReceipt)
            or identity.project_id != task.project_id
            or identity.acquisition_id != task.acquisition_id
            or identity.revisions != Revisions(data=task.data_revision)
            or value.figure != task.figure
            or value.profiles != task.profiles
            or value.figure_sha256 != task.figure_sha256
            or value.profiles_sha256 != task.profiles_sha256
        ):
            QMessageBox.warning(
                self,
                "Inspection export uncertain",
                "Completed files did not match the captured inspection revision.",
            )
            return
        self.statusBar().showMessage(
            f"Exported {task.acquisition_name}: {task.figure.name} + {task.profiles.name}"
            f" · capture/encode {task.capture_ms:.1f} ms"
        )

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
            self.simulator.restore(value.document.view, value.simulation_detail)
            self.physical.restore(value.document.view.physical_settings_json)
            self.prepared.restore(value.document.view.native_fit_session)
            self._pending_prepared = self._prepared_context = None
            self.joint.restore(value.document.view.joint_session)
            self._pending_joint = self._joint_context = None
            self.sample.restore(value.document.view.sample_session, value.sample_input_checks)
            self._pending_sample = self._sample_context = None
            self.hbn.restore(value.document.view.hbn_sessions)
            self._pending_hbn = None
            self._pending_simulation = None
            self._scene_cameras.clear()
            self._reciprocal_epoch += 1
            self._reciprocal_cache.clear()
            self._numeric_draft = value.document.numeric_draft
            self._validated_numeric = (
                (self.project.project_id, self._numeric_draft)
                if value.numeric_validated and self._numeric_draft is not None
                else None
            )
            self._numeric_history = SessionHistory()
            self._mask_history.clear()
            self._mask_cache.clear()
            self._pending_masks.clear()
            self._active_mask = None
            self._active_line = None
            self._active_profile_target = None
            self._comparison_frames_pending.clear()
            self._deferred_mask_write = None
            self._launch_snapshot = None
            self.selected_acquisition_id = value.document.view.selected_acquisition_id
            self._visible_acquisition_id = None
            self._visible_details = ""
            self._source_checks = {item.acquisition_id: item for item in value.sources}
            self._reference_checks = {
                (item.acquisition_id, item.kind): item for item in value.references
            }
            self._candidates.clear()
            self._candidate_queue.clear()
            self._batch_auto_select = False
            self._resident_planes.clear()
            self.comparison_panel.restore_state(value.document.view.comparison or ComparisonState())
            if (
                value.document.view.comparison is not None
                and value.document.view.comparison.visible
            ):
                self.scene_tabs.setCurrentIndex(2)
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
            if self._active_kind not in ("save", "discard", "export"):
                self.experiment_status.detail.setText(message)
                if self.detector_notice.isVisible():
                    self.detector_notice.setText(message)
            self.statusBar().showMessage(message)
        if self._active_kind == "physical" and self._physical_current(identity):
            self.physical.status.setText(message)
        if self._active_kind == "simulation" and self._simulation_current(identity):
            self.simulator.status.setText(message)
        if self._active_kind == "joint" and self._joint_current(identity):
            self.joint.status.setText(message)
            return
        if self._active_kind == "sample" and self._sample_current(identity):
            self.sample.status.setText(message)
        if self._active_kind == "hbn" and self._hbn_current(identity):
            self.hbn.status.setText(message)
        if self._active_kind == "setup" and self._setup_dialog is not None:
            self._setup_dialog.message.setText(message)

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
            self.physical.cancel(clear=True)
            self.physical.scene.view.release_resources()
            self.scene_panel.view.release_resources()
            self.comparison_panel.release_resources()
            self.simulator.detector.view.release_resources()
            super().closeEvent(event)
            return
        event.ignore()
        if not self._close_intent:
            self.physical.cancel(clear=True)
            self._close_intent = True
            self._pending_open = None
            self._pending_hbn = None
            self._pending_prepared = None
            self._pending_joint = None
            self._pending_sample = None
            self._pending_simulation = None
            self._deferred_import = None
            self._pending_reference = None
            self._pending_setup_request = None
            self._pending_setup_copy = None
            for candidate_id in self._candidate_queue:
                candidate = self._candidates.get(candidate_id)
                if candidate is not None and candidate.status == "queued":
                    candidate.status = "canceled"
                    candidate.detail = "Window closing"
            self._candidate_queue.clear()
            self._autosave_timer.stop()
            self._write_queue = deque(task for task in self._write_queue if task.explicit)
            if self._active_kind in (
                "import",
                "relink",
                "batch",
                "reference",
                "open",
                "reciprocal",
                "mask",
                "line",
                "simulation",
                "hbn",
                "joint",
                "prepared",
                "sample",
            ):
                self.jobs.cancel()
            self._pending_masks.clear()
            self._active_mask = None
            self.statusBar().showMessage("Close requested · Preserving accepted state")
            if (
                self.workspaces.currentIndex() == 1
                and self.simulator.draft_kind.currentData() == "native"
            ):
                self.statusBar().repaint()
        QTimer.singleShot(0, self._dispatch_pending)


def main() -> int:
    # Resolve the declared figure dependency before the interactive event loop.
    # Its first Python import otherwise holds the GIL during a requested export.
    figure_dependency_error = None
    try:
        import_module("matplotlib.backends.backend_agg")
        import_module("matplotlib.figure")
    except ImportError as exc:
        figure_dependency_error = str(exc)
    import_module("rasim_next.pipeline.configured_simulation")
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
    if figure_dependency_error is not None:
        window.simulator.figures.setChecked(False)
        window.simulator.figures.setEnabled(False)
        window.simulator.figures.setText(
            f"Configured figures unavailable: {figure_dependency_error}; numeric export remains available"
        )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
