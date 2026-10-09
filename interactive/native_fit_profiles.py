"""Measured prepared-native profile inspection; values stay in their frozen count measure."""

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


@dataclass(frozen=True, slots=True)
class NativeProfiles:
    raw_count: np.ndarray
    background_count: np.ndarray
    corrected_count: np.ndarray
    standard_error_count: np.ndarray
    valid: np.ndarray
    row_ids: tuple[str, ...]
    input_sha256: str
    covariance_count2: np.ndarray

    def __post_init__(self):
        count = len(self.row_ids)
        if (
            type(self.input_sha256) is not str
            or len(self.input_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.input_sha256)
            or any(type(v) is not str or not 0 < len(v) <= 128 for v in self.row_ids)
        ):
            raise ValueError("Prepared profiles require bounded frozen row and input identities")
        if not 0 < count <= 4096 or len(set(self.row_ids)) != count:
            raise ValueError("Prepared profile requires at most 4096 unique frozen row IDs")
        for name in (
            "raw_count",
            "background_count",
            "corrected_count",
            "standard_error_count",
            "valid",
        ):
            values = np.array(getattr(self, name), copy=True)
            if (
                values.shape != (count,)
                or values.dtype.kind not in ("b", "i", "u", "f")
                or not np.all(np.isfinite(values))
            ):
                raise ValueError("Prepared profile arrays must be finite aligned rows")
            if name == "valid" and values.dtype.kind != "b":
                raise ValueError("Prepared profile validity must be boolean")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        covariance = np.array(self.covariance_count2, dtype=float, copy=True)
        if covariance.shape != (count, count) or not np.all(np.isfinite(covariance)):
            raise ValueError("Prepared covariance must match the frozen row roster")
        covariance.setflags(write=False)
        object.__setattr__(self, "covariance_count2", covariance)
        if np.any(self.standard_error_count < 0):
            raise ValueError("Prepared profile standard errors must be nonnegative")

    @property
    def nbytes(self):
        return (
            sum(
                v.nbytes
                for v in (
                    self.raw_count,
                    self.background_count,
                    self.corrected_count,
                    self.standard_error_count,
                    self.valid,
                )
            )
            + sum(len(v.encode()) + 64 for v in self.row_ids)
            + self.covariance_count2.nbytes
            + 1024
        )


class NativeProfilesView(QWidget):
    """Frozen measured profiles; displayed row order is neither a fit nor rebinning."""

    def __init__(self):
        super().__init__()
        self.profiles: NativeProfiles | None = None
        self.setMinimumSize(400, 200)

    def set_profiles(self, profiles):
        self.profiles = profiles
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#19242c"))
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor("#d5e0e9"))
        profiles = self.profiles
        if profiles is None:
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Load a prepared native experiment to inspect frozen measured profiles",
            )
            return
        rect = QRectF(48, 34, max(1, self.width() - 65), max(1, self.height() - 65))
        lines = [
            (profiles.raw_count, "#dfbc78"),
            (profiles.background_count, "#ca8aac"),
            (profiles.corrected_count, "#78dcd2"),
        ]
        lower = (
            min(0.0, *(float(v[profiles.valid].min()) for v, _ in lines))
            if profiles.valid.any()
            else 0.0
        )
        upper = (
            max(1.0, *(float(v[profiles.valid].max()) for v, _ in lines))
            if profiles.valid.any()
            else 1.0
        )
        scale = max(upper - lower, 1e-12)
        painter.drawRect(rect)
        painter.drawText(
            8, 20, "Frozen row order · raw / background / signed corrected counts · valid rows only"
        )
        for values, color in lines:
            painter.setPen(QPen(QColor(color), 1))
            previous = None
            for i, (value, valid) in enumerate(zip(values, profiles.valid, strict=True)):
                if not valid:
                    previous = None
                    continue
                point = QPointF(
                    rect.left() + i * rect.width() / max(1, len(values) - 1),
                    rect.bottom() - (value - lower) * rect.height() / scale,
                )
                if previous is not None:
                    painter.drawLine(previous, point)
                previous = point
        painter.setPen(QColor("#d5e0e9"))
        painter.drawText(
            5,
            self.height() - 8,
            f"Rows 0-{len(profiles.row_ids) - 1}; counts {lower:.4g}…{upper:.4g}. Input {profiles.input_sha256[:12]}",
        )
