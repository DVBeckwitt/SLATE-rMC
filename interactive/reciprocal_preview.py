"""Bounded nominal reciprocal coverage for the desktop inspection view."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np
from job_lifecycle import JobControl, JobResult
from parameter_state import configured_draft
from project_state import MAX_NUMERIC_BASELINE_BYTES, NumericDraft, ProjectFormatError

if TYPE_CHECKING:
    from rasim_next.geometry import CompiledInstrument, IncidentTransportResult
    from rasim_next.pipeline.configured_simulation import (
        AxisRotationConfiguration,
        SimulationConfiguration,
    )
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

GRID_AXIS = 13
MAX_PREVIEW_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class ReciprocalMapping:
    """Small retained nominal inverse map; cursor calls never reread configuration."""

    instrument: CompiledInstrument
    incident: IncidentTransportResult
    ki_film_sample_Ainv: np.ndarray
    ki_air_sample_Ainv: np.ndarray
    column_px: np.ndarray
    row_px: np.ndarray
    q_sample_Ainv: np.ndarray
    valid: np.ndarray
    status: np.ndarray
    wavelength_A: float
    axis_rotations: tuple[AxisRotationConfiguration, ...]
    source_origin_lab_m: tuple[float, float, float] | None = None

    def cursor(
        self, column_px: float, row_px: float
    ) -> tuple[str, np.ndarray | None, np.ndarray | None]:
        from rasim_next.pipeline.continuous_detector import evaluate_detector_coordinates_geometry

        result = evaluate_detector_coordinates_geometry(
            np.asarray([column_px], dtype=np.float64),
            np.asarray([row_px], dtype=np.float64),
            incident=self.incident,
            instrument=self.instrument,
            ki_sample_Ainv=self.ki_film_sample_Ainv,
            include_surface_jacobian=False,
        )
        status = str(result.status[0])
        if not bool(result.valid[0]):
            return status, None, None
        return (
            status,
            result.q_sample_Ainv[0],
            result.kf_air_sample_Ainv[0] - self.ki_air_sample_Ainv,
        )


@dataclass(frozen=True, slots=True)
class ReciprocalPreview:
    request_sha256: str
    project_id: UUID
    acquisition_id: UUID
    saved: ReciprocalMapping
    draft: ReciprocalMapping | None
    draft_revision: int | None
    grid_axis: int = GRID_AXIS
    approximation: str = (
        "nominal one-ray geometry; detector grid; no source spread, intensity or fit"
    )


def _mapping(
    config: SimulationConfiguration, expected_shape_rc: tuple[int, int], control: JobControl
) -> ReciprocalMapping:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_geometry_inputs,
        build_geometry_only_ewald_context,
    )

    inputs = build_configured_geometry_inputs(config)
    if control.canceled:
        raise RuntimeError("reciprocal preview canceled")
    context = build_geometry_only_ewald_context(inputs)
    rows, columns = context.instrument.detector_shape_rc
    if (rows, columns) != expected_shape_rc:
        raise ProjectFormatError("configured detector shape differs from this native OSC")
    axis_column = np.linspace(-0.5, columns - 0.5, GRID_AXIS)
    axis_row = np.linspace(-0.5, rows - 0.5, GRID_AXIS)
    column_px, row_px = np.meshgrid(axis_column, axis_row)
    geometry = context.evaluate_detector_geometry(column_px, row_px, include_surface_jacobian=False)
    return ReciprocalMapping(
        context.instrument,
        context.incident,
        context.ki_sample_Ainv,
        context.incident.states.k_air_sample_Ainv[0],
        geometry.column_px,
        geometry.row_px,
        geometry.q_sample_Ainv,
        geometry.valid,
        geometry.status,
        float(config.source.mean_wavelength_A),
        tuple(config.instrument.axis_rotations),
        tuple(config.source.mean_origin_lab_m),
    )


def prepare_reciprocal_preview(argument: bytes, control: JobControl) -> JobResult:
    """Read exact bound bytes and prepare at most two 13-by-13 nominal maps."""
    request = json.loads(argument)
    if type(request) is not dict or set(request) != {
        "project_id",
        "acquisition_id",
        "source_sha256",
        "configuration_path",
        "configuration_sha256",
        "cif_path",
        "cif_sha256",
        "native_shape_rc",
        "proposed",
        "revision",
        "declared_incidence_rad",
    }:
        raise ProjectFormatError("invalid reciprocal preview request")
    shape = request["native_shape_rc"]
    if (
        type(shape) is not list
        or len(shape) != 2
        or any(type(v) is not int or not 1 <= v <= 16384 for v in shape)
    ):
        raise ProjectFormatError("reciprocal preview needs a native detector shape")
    path = Path(request["configuration_path"])
    with path.open("rb") as handle:
        encoded = handle.read(MAX_NUMERIC_BASELINE_BYTES + 1)
    if not encoded or len(encoded) > MAX_NUMERIC_BASELINE_BYTES:
        raise ProjectFormatError("reciprocal configuration exceeds 64 KiB")
    if hashlib.sha256(encoded).hexdigest() != request["configuration_sha256"]:
        raise ProjectFormatError("reciprocal configuration changed after validation")
    baseline = NumericDraft(
        UUID(request["acquisition_id"]),
        request["source_sha256"],
        path,
        request["configuration_sha256"],
        Path(request["cif_path"]),
        request["cif_sha256"],
        encoded.decode("utf-8"),
    )
    saved_config = configured_draft(baseline)
    if request["declared_incidence_rad"] is not None:
        rotations = saved_config.instrument.axis_rotations
        if len(rotations) != 1 or not np.isclose(
            np.deg2rad(rotations[0].angle_deg),
            request["declared_incidence_rad"],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ProjectFormatError(
                "commanded incidence is not the single configured axis angle; select matching geometry"
            )
    control.report("Mapping saved nominal reciprocal coverage")
    saved = _mapping(saved_config, tuple(shape), control)
    draft = None
    revision = None
    if request["proposed"]:
        if control.canceled:
            raise RuntimeError("reciprocal preview canceled")
        candidate = NumericDraft(
            baseline.acquisition_id,
            baseline.source_sha256,
            path,
            baseline.configuration_sha256,
            baseline.cif_path,
            baseline.cif_sha256,
            baseline.baseline_yaml,
            tuple(tuple(entry) for entry in request["proposed"]),
            request["revision"],
        )
        control.report("Mapping proposed nominal reciprocal coverage")
        draft = _mapping(configured_draft(candidate), tuple(shape), control)
        revision = candidate.revision
    if control.canceled:
        raise RuntimeError("reciprocal preview canceled")
    preview = ReciprocalPreview(
        hashlib.sha256(argument).hexdigest(),
        UUID(request["project_id"]),
        baseline.acquisition_id,
        saved,
        draft,
        revision,
    )
    arrays = (
        array
        for mapping in (saved, draft)
        if mapping is not None
        for array in (
            mapping.column_px,
            mapping.row_px,
            mapping.q_sample_Ainv,
            mapping.valid,
            mapping.status,
        )
    )
    resident = sum(array.nbytes for array in arrays) + 8192
    if resident > MAX_PREVIEW_BYTES:
        raise ProjectFormatError("reciprocal preview exceeds its 256 KiB result budget")
    return JobResult(preview, resident)


class ReciprocalCoverageView(QWidget):
    """Retained two-curve mesh projection of sample-frame Q parallel versus Qz."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(145)
        self.preview: ReciprocalPreview | None = None
        self.selected_q_sample_Ainv: np.ndarray | None = None

    def set_preview(self, preview: ReciprocalPreview | None) -> None:
        self.preview = preview
        self.selected_q_sample_Ainv = None
        self.update()

    def set_selected_q(self, q_sample_Ainv: np.ndarray | None) -> None:
        self.selected_q_sample_Ainv = q_sample_Ainv
        self.update()

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#18232b"))
        preview = self.preview
        if preview is None:
            painter.setPen(QColor("#b7c5cf"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Coverage unavailable")
            return
        mappings = (preview.saved,) + ((preview.draft,) if preview.draft is not None else ())
        valid_q = [mapping.q_sample_Ainv[mapping.valid] for mapping in mappings]
        valid_q = [values for values in valid_q if values.size]
        if not valid_q:
            painter.setPen(QColor("#b7c5cf"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No valid detector coverage"
            )
            return
        all_q = np.concatenate(valid_q)
        radial = np.linalg.norm(all_q[:, :2], axis=1)
        qz = all_q[:, 2]
        left, right = float(radial.min()), float(radial.max())
        bottom, top = float(qz.min()), float(qz.max())
        if right == left:
            right += 1.0
        if top == bottom:
            top += 1.0
        x0, y0, width, height = (
            38.0,
            9.0,
            max(1.0, self.width() - 48.0),
            max(1.0, self.height() - 31.0),
        )

        painter.setPen(QPen(QColor("#465d68"), 1))
        painter.drawRect(int(x0), int(y0), int(width), int(height))
        for mapping, color in zip(mappings, ("#61d5bd", "#e9ae75")[: len(mappings)], strict=True):
            radius = np.hypot(mapping.q_sample_Ainv[..., 0], mapping.q_sample_Ainv[..., 1])
            x = x0 + (radius - left) / (right - left) * width
            y = y0 + (top - mapping.q_sample_Ainv[..., 2]) / (top - bottom) * height
            painter.setPen(QPen(QColor(color), 1))
            for row in range(GRID_AXIS):
                for column in range(GRID_AXIS):
                    if not mapping.valid[row, column]:
                        continue
                    here = QPointF(float(x[row, column]), float(y[row, column]))
                    if column + 1 < GRID_AXIS and mapping.valid[row, column + 1]:
                        painter.drawLine(
                            here, QPointF(float(x[row, column + 1]), float(y[row, column + 1]))
                        )
                    if row + 1 < GRID_AXIS and mapping.valid[row + 1, column]:
                        painter.drawLine(
                            here, QPointF(float(x[row + 1, column]), float(y[row + 1, column]))
                        )
        if self.selected_q_sample_Ainv is not None:
            marker = QPointF(
                x0
                + (float(np.hypot(*self.selected_q_sample_Ainv[:2])) - left)
                / (right - left)
                * width,
                y0 + (top - float(self.selected_q_sample_Ainv[2])) / (top - bottom) * height,
            )
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawEllipse(marker, 4.0, 4.0)
        painter.setPen(QColor("#b7c5cf"))
        painter.drawText(3, 17, "Qz")
        painter.drawText(int(x0), self.height() - 4, "Qparallel (sample frame, Å⁻¹)")
