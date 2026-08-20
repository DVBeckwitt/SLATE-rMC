"""Planar-interface modes and scalar optical weights."""

from rasim_next.optics.attenuation import (
    DETECTOR_PATH_ATTENUATION_MODEL_ID,
    INCIDENT_ILLUMINATED_PATH_MODEL_ID,
    external_path_attenuation,
    incident_illuminated_path_weight,
    mode_decay_constant,
    path_attenuation,
    scalar_optical_weight,
    uniform_depth_attenuation,
)
from rasim_next.optics.refraction import (
    ExitMode,
    IncidentMode,
    solve_exit_mode,
    solve_incident_mode,
)

__all__ = [
    "DETECTOR_PATH_ATTENUATION_MODEL_ID",
    "INCIDENT_ILLUMINATED_PATH_MODEL_ID",
    "ExitMode",
    "IncidentMode",
    "external_path_attenuation",
    "incident_illuminated_path_weight",
    "mode_decay_constant",
    "path_attenuation",
    "scalar_optical_weight",
    "solve_exit_mode",
    "solve_incident_mode",
    "uniform_depth_attenuation",
]
