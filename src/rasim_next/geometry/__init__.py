"""Rigid instrument geometry and ray transport."""

from rasim_next.geometry.angles import (
    AngleFrame,
    DetectorAngles,
    DetectorCoordinates,
    angles_to_detector_coordinates,
    detector_coordinates_to_angles,
)
from rasim_next.geometry.detector import (
    DetectorProjection,
    DetectorRay,
    detector_coordinate_to_ray,
    project_detector_ray,
)
from rasim_next.geometry.instrument import (
    AxisRotation,
    CompiledInstrument,
    InstrumentConfiguration,
    compile_instrument,
)
from rasim_next.geometry.sample import SampleIntersection, intersect_sample_ray
from rasim_next.geometry.transport import (
    EventTransportResult,
    IncidentTransportResult,
    build_incident_states,
    transport_scattering_events,
)

__all__ = [
    "AngleFrame",
    "AxisRotation",
    "CompiledInstrument",
    "DetectorAngles",
    "DetectorCoordinates",
    "DetectorProjection",
    "DetectorRay",
    "EventTransportResult",
    "IncidentTransportResult",
    "InstrumentConfiguration",
    "SampleIntersection",
    "angles_to_detector_coordinates",
    "build_incident_states",
    "compile_instrument",
    "detector_coordinate_to_ray",
    "detector_coordinates_to_angles",
    "intersect_sample_ray",
    "project_detector_ray",
    "transport_scattering_events",
]
