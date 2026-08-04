"""Detector acceptance shared by specular and off-specular fitting stages."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class DetectorHorizonAcceptance:
    """Detector-domain acceptance for m=0 data and off-specular observations."""

    offspecular_air_exit_guard_rad: float

    def __post_init__(self) -> None:
        guard = float(self.offspecular_air_exit_guard_rad)
        if not math.isfinite(guard) or not 0.0 <= guard <= math.pi / 2.0:
            raise ValueError("off-specular air-exit guard must be a finite physical elevation")
        object.__setattr__(self, "offspecular_air_exit_guard_rad", guard)

    @staticmethod
    def _mask(air_exit_rad: ArrayLike, guard_rad: float) -> NDArray[np.bool_]:
        supplied = np.asarray(air_exit_rad)
        if np.iscomplexobj(supplied):
            raise ValueError("air-exit angles must be real")
        air_exit = np.asarray(supplied, dtype=np.float64)
        return np.asarray(
            np.isfinite(air_exit) & (air_exit > 0.0) & (air_exit >= guard_rad),
            dtype=np.bool_,
        )

    def m0_detector_mask(self, mean_surface_air_exit_rad: ArrayLike) -> NDArray[np.bool_]:
        """Keep every finite m=0 detector coordinate; model validity is evaluated later."""

        supplied = np.asarray(mean_surface_air_exit_rad)
        if np.iscomplexobj(supplied):
            raise ValueError("air-exit angles must be real")
        return np.asarray(np.isfinite(np.asarray(supplied, dtype=np.float64)), dtype=np.bool_)

    def offspecular_mask(self, air_exit_rad: ArrayLike) -> NDArray[np.bool_]:
        """Accept off-specular events above their independent clearance guard."""

        return self._mask(air_exit_rad, self.offspecular_air_exit_guard_rad)
