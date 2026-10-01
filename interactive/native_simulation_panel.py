"""Searchable native fields and independent native history within the shared simulator."""

import json
from dataclasses import replace

import numpy as np
from native_simulation_state import native_draft_document
from parameter_state import FieldChange, SessionHistory, _action
from project_state import ProjectFormatError
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def native_field_values(value, path=()):
    """Keep numeric vectors/matrices together; expand structured records into fields."""
    if isinstance(value, dict):
        for name, item in value.items():
            yield from native_field_values(item, (*path, name))
    elif isinstance(value, list) and any(isinstance(v, dict) for v in value):
        for index, item in enumerate(value):
            yield from native_field_values(item, (*path, index))
    else:
        yield path, value


def native_field_unit(path):
    name = str(path[-1])
    if "Ainv" in name:
        return "angstrom^-1"
    if "A2" in name:
        return "angstrom^2"
    if "attenuation" in name:
        return "metre^-1"
    if name.endswith("_A") or "wavelength" in name:
        return "angstrom"
    if name.endswith("_rad"):
        return "radian"
    if name.endswith("_m") or "translation_m" in path:
        return "metre"
    if "px" in name or "shape_rc" in name:
        return "detector-native pixel; arrays row,column"
    return "dimensionless / declared identifier"


class NativeValueEditor(QWidget):
    def __init__(self, value, changed, parent=None):
        super().__init__(parent)
        self.original = value
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if type(value) is bool:
            self.control = QCheckBox("Enabled")
            self.control.setChecked(value)
            self.control.toggled.connect(changed)
        elif (
            isinstance(value, list)
            and value
            and all(not isinstance(v, (dict, list)) for v in value)
        ):
            self.control = QTableWidget(len(value), 1)
            self.control.setHorizontalHeaderLabels(["Declared value"])
            for r, v in enumerate(value):
                self.control.setItem(r, 0, QTableWidgetItem(json.dumps(v)))
            self.control.setMaximumHeight(min(210, 58 + len(value) * 24))
            self.control.cellChanged.connect(changed)
        elif isinstance(value, list) and value and all(isinstance(v, list) for v in value):
            self.control = QTableWidget(len(value), max(len(v) for v in value))
            for r, row in enumerate(value):
                for c, v in enumerate(row):
                    self.control.setItem(r, c, QTableWidgetItem(json.dumps(v)))
            self.control.setMaximumHeight(150)
            self.control.cellChanged.connect(changed)
        else:
            self.control = QLineEdit(value if isinstance(value, str) else json.dumps(value))
            self.control.editingFinished.connect(changed)
        layout.addWidget(self.control)

    def value(self):
        if isinstance(self.control, QCheckBox):
            return self.control.isChecked()
        if isinstance(self.control, QTableWidget):
            rows = [
                [
                    json.loads(self.control.item(r, c).text())
                    for c in range(self.control.columnCount())
                ]
                for r in range(self.control.rowCount())
            ]
            return [v[0] for v in rows] if self.control.columnCount() == 1 else rows
        text = self.control.text()
        return text if isinstance(self.original, str) else json.loads(text)


class NativeDraftPanel(QWidget):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.simulator = simulator
        self.draft = None
        self.validated = None
        self.result_reference = None
        self.history = SessionHistory()
        self._restoring = False
        self.editors = {}
        self._record = None
        self._forms = {}
        self._group_order = []
        layout = QVBoxLayout(self)
        self.help = QLabel(
            "Load an expanded native physical JSON for Bi2Se3, Bi2Te3, GD1, SiD1, Clean1 or B4. No observation pack or fit is required. Bi disorder and Pb reflectivity are unsupported."
        )
        self.help.setWordWrap(True)
        layout.addWidget(self.help)
        form = QFormLayout()
        self.repeats = QLineEdit("1")
        self.bin_size = QLineEdit("1")
        self.proposal = QLineEdit("null")
        self.surface_fractions = QLineEdit("null")
        self.phase_fractions = QLineEdit("null")
        self.proposal.setToolTip(
            "null uses physical mosaic; otherwise [sigma_rad, gamma_rad, probability]"
        )
        form.addRow("Positive integer repeats (Bi cells / Pb layers)", self.repeats)
        form.addRow(
            "Integrated rectangle width (native pixels; 1 = full resolution)", self.bin_size
        )
        form.addRow("Numerical proposal mosaic (explicit or null)", self.proposal)
        form.addRow(
            "Exact declared surface fractions (or null for model shares)", self.surface_fractions
        )
        form.addRow(
            "Exact declared phase fractions (or null for model shares)", self.phase_fractions
        )
        for control in (
            self.repeats,
            self.bin_size,
            self.proposal,
            self.surface_fractions,
            self.phase_fractions,
        ):
            control.editingFinished.connect(self.propose)
        layout.addLayout(form)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search native names, units or owners")
        layout.addWidget(self.search)
        self.groups = QTabWidget()
        layout.addWidget(self.groups, 1)
        self.groups.currentChanged.connect(self._show_group)
        self.search.textChanged.connect(self.filter_fields)

    def populate(self, draft):
        self._restoring = True
        self.groups.blockSignals(True)
        try:
            self._record = json.loads(draft.physics_json)
            while self.groups.count():
                w = self.groups.widget(0)
                self.groups.removeTab(0)
                w.deleteLater()
            self.editors = {}
            self._forms = {}
            self._group_order = []
            entries = [
                (("specimen", name), value, unit)
                for name, value, unit in zip(
                    draft.parameter_names,
                    draft.parameter_values,
                    draft.parameter_units,
                    strict=True,
                )
            ]
            entries.extend(
                (path, value, native_field_unit(path))
                for path, value in native_field_values(self._record)
            )
            for path, value, unit in entries:
                group = str(path[0])
                self.editors[path] = (value, unit, None, None)
                if group not in self._forms:
                    content = QWidget()
                    form = QFormLayout(content)
                    scroll = QScrollArea()
                    scroll.setWidgetResizable(True)
                    scroll.setWidget(content)
                    self.groups.addTab(scroll, group.replace("_", " ").title())
                    self._forms[group] = form
                    self._group_order.append(group)
            self.repeats.setText(str(draft.coherent_repeats))
            self.bin_size.setText(str(draft.bin_size_px))
            self.proposal.setText(json.dumps(draft.proposal_mosaic))
            self.surface_fractions.setText(json.dumps(draft.surface_fractions))
            self.phase_fractions.setText(json.dumps(draft.phase_fractions))
            self.help.setText(
                f"{draft.recipe}: {len(draft.parameter_names)} canonical specimen coordinates; source/metres, radians and angstrom units are explicit. CPU deterministic detector integration; one numerical worker, nested BLAS=1; no native CUDA or auxiliary route. Specimen coordinates override their baseline crystal/site fields; N*c + extra defines physical thickness. Exact closed fractions preserve tiny remainders that model shares cannot reconstruct; editing their shares clears the exact declaration. Baseline phase/parent rosters and normalization remain declared. LHS uses sample_count/seed; Gaussian quadrature uses divergence/wavelength orders. Local-m0 source order requires the Bi composite. Nominal angular integration ignores panel resolution; explicit axial meshes own axial nodes. Bi disorder and Pb reflectivity unsupported."
            )
        finally:
            self.groups.blockSignals(False)
            self._restoring = False
        self._show_group(self.groups.currentIndex())
        self.filter_fields(self.search.text())

    def _show_group(self, index):
        if not 0 <= index < len(self._group_order):
            return
        group = self._group_order[index]
        self._restoring = True
        try:
            for path, (value, unit, label, editor) in tuple(self.editors.items()):
                if path[0] != group or editor is not None:
                    continue
                editor = NativeValueEditor(value, lambda *_: self.propose(), self)
                editor.control.setObjectName("native_" + ".".join(map(str, path)))
                label = QLabel(
                    ".".join(map(str, path[1:] or path))
                    + "\n"
                    + unit
                    + "; canonical constructor validates domain"
                )
                label.setWordWrap(True)
                if path in (("schema",), ("sample_id",), ("rod_catalog_revision",)):
                    editor.setEnabled(False)
                self._forms[group].addRow(label, editor)
                self.editors[path] = (value, unit, label, editor)
        finally:
            self._restoring = False
        self.filter_fields(self.search.text())

    def filter_fields(self, text):
        text = text.casefold()
        visible = set()
        for path, (_, unit, label, editor) in self.editors.items():
            shown = text in (".".join(map(str, path)) + " " + unit).casefold()
            if shown:
                visible.add(str(path[0]))
            if editor is not None:
                label.setVisible(shown)
                editor.setVisible(shown)
        for index, group in enumerate(self._group_order):
            self.groups.setTabVisible(index, group in visible)

    def propose(self):
        if self._restoring or self.draft is None:
            return False
        try:
            record = json.loads(self.draft.physics_json)
            values = dict(zip(self.draft.parameter_names, self.draft.parameter_values, strict=True))
            for path, (_, _, _, editor) in self.editors.items():
                if editor is None or not editor.isEnabled():
                    continue
                value = editor.value()
                if path[0] == "specimen":
                    values[path[1]] = float(value)
                    continue
                parent = record
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
            from rasim_next.fitting.pb_native import simplex_shares

            surface = json.loads(self.surface_fractions.text())
            phases = json.loads(self.phase_fractions.text())
            surface = None if surface is None else tuple(surface)
            phases = None if phases is None else tuple(phases)
            surface_names = ("surface_fraction_0", "surface_1_share_of_remainder")
            phase_names = tuple(
                n
                for n in self.draft.parameter_names
                if n.startswith("phase_") and "parent" not in n and n.endswith("share_of_remainder")
            )
            for names, fractions, prior in (
                (surface_names, surface, self.draft.surface_fractions),
                (phase_names, phases, self.draft.phase_fractions),
            ):
                if fractions is not None and fractions != prior:
                    for name, value in zip(names, simplex_shares(fractions), strict=True):
                        values[name] = float(value)
                elif fractions is not None and not np.array_equal(
                    simplex_shares(fractions), [values[n] for n in names]
                ):
                    if names == surface_names:
                        surface = None
                    else:
                        phases = None
            proposal = json.loads(self.proposal.text())
            updated = replace(
                self.draft,
                revision=self.draft.revision + 1,
                physics_json=json.dumps(record, sort_keys=True, allow_nan=False),
                parameter_values=tuple(values[n] for n in self.draft.parameter_names),
                coherent_repeats=int(self.repeats.text()),
                bin_size_px=int(self.bin_size.text()),
                proposal_mosaic=None if proposal is None else tuple(proposal),
                surface_fractions=surface,
                phase_fractions=phases,
            )
            if replace(updated, revision=self.draft.revision) == self.draft:
                return True
            return self.adopt(
                updated, "Edit independent native draft", validated=False, repopulate=False
            )
        except (ValueError, TypeError, ProjectFormatError) as exc:
            self.validated = None
            self.simulator.status.setText(f"Invalid native value: {exc}")
            self.simulator.refresh()
            return False

    def can_persist(self, draft, reference):
        view = replace(
            self.simulator.shell._capture_view(),
            native_simulation_draft=draft,
            native_simulation_result=reference,
        )
        try:
            self.simulator.shell._validate_project_admission(
                self.simulator.shell.project, view=view
            )
        except ProjectFormatError as exc:
            self.simulator.status.setText(
                f"Native change rejected; prior savable state retained: {exc}"
            )
            return False
        return True

    def adopt(self, draft, label, *, validated, record_history=True, repopulate=True):
        if not self.can_persist(draft, self.result_reference):
            return False
        if record_history and self.draft is not None and self.draft != draft:
            self.history.push(
                _action(label, [FieldChange(draft.draft_id, "native_draft", self.draft, draft)])
            )
        changed = self.draft != draft
        self.draft = draft
        self.validated = draft if validated else None
        self.simulator._supersede()
        if repopulate and changed:
            self.populate(draft)
        elif changed:
            self._record = json.loads(draft.physics_json)
            current = dict(native_field_values(self._record))
            current.update(
                {
                    ("specimen", n): v
                    for n, v in zip(draft.parameter_names, draft.parameter_values, strict=True)
                }
            )
            self._restoring = True
            try:
                for path, (_value, unit, label, editor) in tuple(self.editors.items()):
                    if path not in current:
                        continue
                    value = current[path]
                    self.editors[path] = (value, unit, label, editor)
                    if editor is not None and isinstance(editor.control, QLineEdit):
                        editor.control.setText(
                            value if isinstance(value, str) else json.dumps(value)
                        )
                        editor.original = value
                self.surface_fractions.setText(json.dumps(draft.surface_fractions))
                self.phase_fractions.setText(json.dumps(draft.phase_fractions))
            finally:
                self._restoring = False
        self.simulator.shell._mark_dirty()
        self.simulator.refresh()
        self.simulator.status.setText(
            f"{label}; {draft.recipe} revision {draft.revision}. Validate before running."
            if not validated
            else f"Canonical native draft admitted: {draft.recipe} revision {draft.revision}; SHA256 {draft.physics_sha256}. Run is explicit."
        )
        return True

    def history_step(self, undo):
        source = self.history.undo_actions if undo else self.history.redo_actions
        if not source or self.draft is None:
            return
        change = source[-1].changes[0]
        expected, restored = (
            (change.after, change.before) if undo else (change.before, change.after)
        )
        if replace(self.draft, revision=expected.revision) != expected:
            self.simulator.status.setText("Native history conflicts with current draft")
            return
        updated = replace(restored, revision=self.draft.revision + 1)
        if not self.can_persist(updated, self.result_reference):
            return
        action = source.pop()
        (self.history.redo_actions if undo else self.history.undo_actions).append(action)
        self.draft = updated
        self.validated = None
        self.populate(updated)
        self.simulator._supersede()
        self.simulator.shell._mark_dirty()
        self.simulator.refresh()

    def load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load independent native physics", "", "Native physics (*.json)"
        )
        if path:
            self.simulator._supersede()
            self.simulator.shell._request_simulation(
                "native_load", json.dumps({"operation": "native_load", "path": path}).encode()
            )

    def validate(self):
        if self.propose():
            self.simulator.shell._request_simulation(
                "native_validate",
                json.dumps(
                    {"operation": "native_validate", "draft": native_draft_document(self.draft)}
                ).encode(),
            )
