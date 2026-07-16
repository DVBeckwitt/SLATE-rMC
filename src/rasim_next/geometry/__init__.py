"""Rigid instrument geometry and ray transport."""

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
    "AxisRotation",
    "CompiledInstrument",
    "DetectorProjection",
    "DetectorRay",
    "EventTransportResult",
    "IncidentTransportResult",
    "InstrumentConfiguration",
    "SampleIntersection",
    "build_incident_states",
    "compile_instrument",
    "detector_coordinate_to_ray",
    "intersect_sample_ray",
    "project_detector_ray",
    "transport_scattering_events",
]
