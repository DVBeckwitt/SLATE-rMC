"""Native desktop shell for SLATE-rMC; launch with ``python interactive/slate_app.py``."""

import sys
from typing import Literal
from uuid import UUID

from job_lifecycle import JobIdentity, JobOwner, JobState, JobSummary
from project_state import Project
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


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
        self.jobs.drained.connect(self.close)
        self.cancel_button.clicked.connect(self.jobs.cancel)
        self.statusBar().showMessage("Local project · Not saved")
        self.refresh_project()

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
            "Imported detector images will appear here. File import is unavailable in this version.",
        )
        center_layout.addWidget(self.experiment_status, 1)
        import_button = QPushButton("Import files")
        import_button.setEnabled(False)
        import_button.setToolTip("File import is unavailable in this version")
        controls = QHBoxLayout()
        controls.addWidget(import_button)
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
        project_limit = QLabel("Project opening and saving are unavailable in this version.")
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
            self.selection_label.setText(f"Selected acquisition: {selected_item.text(0)}")
        else:
            self.selected_acquisition_id = None
            self.selection_label.setText("No acquisition selected")
        self.acquisition_count.setText(f"{len(self.project.acquisitions)} acquisitions")
        if self.project.acquisitions:
            self.experiment_status.set_state(
                "empty", "No image open", "Detector viewing is unavailable in this version."
            )
        self.project_tree.blockSignals(False)

    def _selection_changed(self) -> None:
        selected = self.project_tree.selectedItems()
        item = selected[0] if selected else None
        value = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        selected_id = UUID(value) if value else None
        if selected_id != self.selected_acquisition_id:
            self.jobs.invalidate()
            if self.jobs.busy:
                self.experiment_status.set_state(
                    "loading", "Stopping obsolete work", "Waiting for the prior operation to stop."
                )
        self.selected_acquisition_id = selected_id
        self.selection_label.setText(
            f"Selected acquisition: {item.text(0)}"
            if self.selected_acquisition_id
            else "No acquisition selected"
        )

    def _job_state_changed(self, summary: JobSummary) -> None:
        if summary.identity.generation != self.jobs.latest_generation:
            if (
                not self.jobs.busy
                and self.experiment_status.title.text() == "Stopping obsolete work"
            ):
                self.experiment_status.set_state(
                    "empty", "No image open", "The previous operation was discarded."
                )
            return
        state = summary.state
        self.cancel_button.setEnabled(state in (JobState.QUEUED, JobState.RUNNING))
        if state in (JobState.QUEUED, JobState.RUNNING):
            self.experiment_status.set_state("loading", "Working", "Preparing the requested data.")
        elif state == JobState.CANCEL_REQUESTED:
            self.experiment_status.set_state(
                "loading",
                "Stopping safely",
                "Waiting for the current operation to release its resources.",
            )
        elif state == JobState.FAILED:
            self.experiment_status.set_state("error", "Operation failed", summary.detail)
        elif state == JobState.CANCELED:
            self.experiment_status.set_state(
                "empty", "Operation canceled", "No result was applied."
            )
        else:
            self.experiment_status.set_state("empty", "Operation finished", "No result is open.")
        self.statusBar().showMessage(f"Operation: {state.value}")

    def _job_progress(self, identity: JobIdentity, message: str) -> None:
        if identity.generation == self.jobs.latest_generation:
            self.experiment_status.detail.setText(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.jobs.request_close():
            event.ignore()
            self.cancel_button.setEnabled(False)
            self.experiment_status.set_state(
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
