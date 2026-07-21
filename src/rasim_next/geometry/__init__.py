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
    DetectorProjectionBatch,
    DetectorRay,
    detector_coordinate_to_ray,
    project_detector_ray,
    project_detector_rays,
)
from rasim_next.geometry.instrument import (
    AxisRotation,
    CompiledInstrument,
    InstrumentConfiguration,
    compile_instrument,
    compose_intrinsic_xy_rotation,
)
from rasim_next.geometry.sample import SampleIntersection, intersect_sample_ray
from rasim_next.geometry.transport import (
    IncidentTransportResult,
    build_incident_states,
)

__all__ = [
    "AngleFrame",
    "AxisRotation",
    "CompiledInstrument",
    "DetectorAngles",
    "DetectorCoordinates",
    "DetectorProjection",
    "DetectorProjectionBatch",
    "DetectorRay",
    "IncidentTransportResult",
    "InstrumentConfiguration",
    "SampleIntersection",
    "angles_to_detector_coordinates",
    "build_incident_states",
    "compile_instrument",
    "compose_intrinsic_xy_rotation",
    "detector_coordinate_to_ray",
    "detector_coordinates_to_angles",
    "intersect_sample_ray",
    "project_detector_ray",
    "project_detector_rays",
]
