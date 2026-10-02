"""Small Simulator contrast policy over the existing detector controls."""

from project_state import FLOAT32_NORMAL_MIN, linear_display_limits, validate_display_limits
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QWidget
from simulation_widgets import NumberEdit


class SimulationContrast(QObject):
    """Auto anchors per run/mode/bin; manual and restored levels stay explicit."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.frame = None
        self.auto = True
        self.mode = "linear"
        self._run_id = None
        self._anchors = {}
        self._base = (0.0, 1.0)
        self._manual_levels = self._base
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(0.000001, 1000000)
        self.exposure.setDecimals(6)
        self.exposure.setValue(1)
        self.exposure.setSingleStep(0.25)
        self.exposure.setSuffix(" x")
        self.exposure.setMaximumWidth(110)
        self.exposure.setAccessibleName("Simulator exposure; higher reveals weaker signal")
        self.exposure.setToolTip(
            "Changes display limits only. Higher exposure lowers the upper limit and may clip bright peaks."
        )
        self.full_button = QPushButton("Full range")
        self.full_button.setToolTip(
            "Use all finite displayed extrema; strong peaks can hide weak signal."
        )
        row = panel.auto_button.parentWidget().layout()
        for name in ("low_control", "high_control"):
            old = getattr(panel, name)
            field = NumberEdit(panel)
            field.setText(old.text())
            row.replaceWidget(old, field)
            old.hide()
            old.deleteLater()
            setattr(panel, name, field)
        self.range_controls = QWidget(panel)
        range_row = QHBoxLayout(self.range_controls)
        range_row.setContentsMargins(0, 0, 0, 0)
        for widget in (panel.low_control, panel.high_control, panel.apply_button):
            row.removeWidget(widget)
        for item in panel.auto_button.parentWidget().findChildren(QLabel):
            if item.text() in ("Low", "High"):
                row.removeWidget(item)
                item.hide()
        for caption, field in (("Low", panel.low_control), ("Upper", panel.high_control)):
            label = QLabel(caption)
            label.setBuddy(field)
            range_row.addWidget(label)
            range_row.addWidget(field, 1)
        range_row.addWidget(panel.apply_button)
        self.range_button = QPushButton("Range")
        self.range_button.setCheckable(True)
        self.range_button.setToolTip("Exact manual display limits and full range")
        self.range_button.toggled.connect(self.range_controls.setVisible)
        row.addWidget(self.range_button)
        panel.layout().addWidget(self.range_controls, 2, 0, 1, 2)
        self.range_controls.hide()
        label = QLabel("Exposure")
        label.setBuddy(self.exposure)
        row.addWidget(label)
        row.addWidget(self.exposure)
        range_row.addWidget(self.full_button)
        panel.auto_button.setText("Auto 99%")
        panel.auto_button.setToolTip(
            "Linear upper limit: 99th percentile of finite positive DISPLAY cells. Held while this run refines."
        )
        panel.layout().removeWidget(panel.contrast_note)
        row.addWidget(panel.contrast_note, 1)
        panel.contrast_note.show()
        for item in panel.auto_button.parentWidget().findChildren(QLabel):
            if item.text() == "High":
                item.setText("Upper")
                item.setBuddy(panel.high_control)
        panel.high_control.setAccessibleName("Simulator displayed upper level")
        panel.mode_control.currentIndexChanged.disconnect(panel._mode_changed)
        panel.apply_button.clicked.disconnect(panel._apply_contrast)
        panel.auto_button.clicked.disconnect(panel._auto_contrast)
        panel.mode_control.currentIndexChanged.connect(self.change_mode)
        panel.apply_button.clicked.connect(self.apply_manual)
        panel.auto_button.clicked.connect(self.restore_auto)
        self.full_button.clicked.connect(self.full_range)
        self.exposure.valueChanged.connect(self.apply_exposure)
        panel.view.display_bin_changed.connect(self.refresh)

    def show_error(self, message):
        self.panel.contrast_note.setText("Display error")
        self.panel.contrast_note.setToolTip(
            message + "\nChoose valid manual levels or press Auto 99%."
        )
        self.range_button.setChecked(True)

    def restore(self, state):
        self.frame = None
        self._run_id = None
        self._anchors.clear()
        self.auto = state is None
        self.mode = "linear" if state is None else state.contrast_mode
        self._base = (0.0, 1.0) if state is None else (state.low_value, state.high_value)
        self._manual_levels = self._base
        self._reset_exposure()

    def _reset_exposure(self):
        self.exposure.blockSignals(True)
        self.exposure.setValue(1)
        self.exposure.blockSignals(False)

    def admit(self, frame):
        self.frame = frame
        if frame.run_id != self._run_id:
            self._anchors.clear()
            self._run_id = frame.run_id
        self.panel.view.set_display_levels(frame.display_levels)
        self.refresh()

    def limits(self, level, *, full=False):
        if self.mode == "positive_log":
            if not level.positive_count or level.maximum <= FLOAT32_NORMAL_MIN:
                raise ValueError(
                    "Positive log unavailable: no positive display signal in the shader range"
                )
            low = max(float(level.positive_quantiles[0]), FLOAT32_NORMAL_MIN)
            return low, max(level.maximum, low * 1.001)
        if self.mode == "signed":
            extent = max(abs(level.minimum), abs(level.maximum), FLOAT32_NORMAL_MIN * 2)
            return -extent, extent
        if full or level.minimum < 0 or not level.positive_count:
            return linear_display_limits(level.minimum, level.maximum)
        return 0.0, max(float(level.positive_quantiles[990]), FLOAT32_NORMAL_MIN * 2)

    def refresh(self, *_):
        level = self.panel.view.active_display_level
        if self.frame is None or level is None:
            return
        if self.auto:
            key = self.mode, level.bin_size
            try:
                if key not in self._anchors and (level.positive_count or level.minimum < 0):
                    self._anchors[key] = self.limits(level)
                low, high = self._anchors[key] if key in self._anchors else self.limits(level)
                self.panel.view.set_levels(low, high, mode=self.mode)
                self._base = low, high
            except ValueError as exc:
                self.show_error(str(exc))
                return
        if not self.auto:
            self.panel.view.set_levels(*self._manual_levels, mode=self.mode)
        self.panel._sync_controls()
        self._describe(level)

    def _describe(self, level):
        view = self.panel.view
        clipped = 100 * level.above_fraction(view.high_value)
        if not self.auto:
            mode = "Manual/restored levels"
        elif self.mode == "signed":
            mode = "Auto signed range (held during this run)"
        elif self.mode == "positive_log":
            mode = "Auto positive log (held during this run)"
        elif level.minimum < 0:
            mode = "Auto full range for signed data (held during this run)"
        elif not level.positive_count:
            mode = "Auto: no positive display signal yet"
        else:
            mode = "Auto 99% (held during this run)"
        negative = (
            "; negative cells hidden by log"
            if view.contrast_mode == "positive_log" and level.minimum < 0
            else ""
        )
        tiny = "; below normal shader range" if 0 < level.maximum < FLOAT32_NORMAL_MIN else ""
        description = (
            f"{mode}; ~{clipped:.2f}% of positive display cells clipped above Upper. "
            f"Display: {level.bin_size} x {level.bin_size} cells summed per bin; "
            f"{level.invalid_count} invalid bins{negative}{tiny}. Cursor/profiles/export use original cells."
        )
        self.panel.contrast_note.setWordWrap(False)
        self.panel.contrast_note.setText(
            f"{clipped:.1f}% clipped" + (" · invalid bins" if level.invalid_count else "")
        )
        self.panel.contrast_note.setToolTip(
            description
            + "\n"
            + f"Displayed upper/lower levels use {self.panel.observable_unit}, summed per bin. "
            "Clipping estimates use a bounded positive-cell CDF with 0.1-percentile steps; zeros are excluded. "
            "Auto's linear upper level uses the exact 99th percentile of positive display values. "
            "Summation changes presentation resolution, not numerical integration or scientific samples. "
            "Nonfinite/excluded cells contribute no signal; an entirely invalid bin is checkerboard. "
            "The clipping percentage describes positive cells; signed negative values can also clip below Low. "
            "Manual limits remain the same display-sum units across zoom; Auto adapts to a newly selected bin size."
        )

    def change_mode(self):
        previous = self.mode
        self.mode = self.panel.mode_control.currentData()
        if self.auto:
            self.refresh()
            return
        view = self.panel.view
        try:
            if self.mode == "signed" and not view.low_value < 0 < view.high_value:
                raise ValueError("Signed display needs Low < 0 < Upper")
            view.set_levels(view.low_value, view.high_value, mode=self.mode)
        except ValueError as exc:
            self.mode = previous
            self.panel._sync_controls()
            self.show_error(f"{exc}; choose valid manual levels or press Auto 99%.")
        else:
            self.refresh()

    def apply_manual(self):
        try:
            low, high = (
                float(self.panel.low_control.exact_text()),
                float(self.panel.high_control.exact_text()),
            )
            mode = self.panel.mode_control.currentData()
            if mode == "signed" and not low < 0 < high:
                raise ValueError("Signed display needs Low < 0 < Upper")
            self.panel.view.set_levels(low, high, mode=mode)
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self.auto = False
        self.mode = self.panel.view.contrast_mode
        self._base = self._manual_levels = low, high
        self._reset_exposure()
        self.refresh()

    def apply_exposure(self, exposure):
        low, high = self._base
        changed = (
            (low / exposure, high / exposure)
            if self.mode == "signed"
            else (low, low + (high - low) / exposure)
        )
        try:
            validate_display_limits(*changed, self.mode)
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self.auto = False
        self._manual_levels = changed
        self.panel.view.set_levels(*changed, mode=self.mode)
        self.refresh()

    def restore_auto(self):
        self.auto = True
        self._anchors.clear()
        self.mode = self.panel.mode_control.currentData()
        self._reset_exposure()
        self.refresh()

    def full_range(self):
        level = self.panel.view.active_display_level
        if level is None:
            return
        try:
            low, high = self.limits(level, full=True)
            self.panel.view.set_levels(low, high, mode=self.mode)
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self.auto = False
        self._base = self._manual_levels = low, high
        self._reset_exposure()
        self.refresh()
