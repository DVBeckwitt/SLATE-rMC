"""Vectorized incident transport with explicit aligned validity and optional traces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from rasim_next.core.contracts import (
    DetectorHitBatch,
    IncidentSampleBatch,
    IncidentStateBatch,
    MaterialOptics,
    OutgoingWaveBatch,
    ScatteringEventBatch,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.validity import ValidityCode
from rasim_next.geometry.detector import _project_detector_rays
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.sample import _intersect_sample_rays
from rasim_next.optics.attenuation import (
    mode_decay_constant,
    scalar_optical_weight,
    uniform_depth_attenuation,
)
from rasim_next.optics.refraction import (
    _solve_exit_mode_arrays,
    _solve_incident_mode_arrays,
)
from rasim_next.proof.traces import Measure, QuantityKind, TraceRecord

_MODEL_VERSION = "geometry-optics-v1"
_PROVENANCE = "T02 detector-native geometry and planar-interface optics"

type _TraceStage = tuple[
    str,
    NDArray[np.generic],
    str,
    str,
    Measure,
    QuantityKind,
]


def _statuses(value: Iterable[object], size: int) -> tuple[ValidityCode, ...]:
    supplied = tuple(value)
    if len(supplied) != size:
        raise ValueError(f"status must contain {size} values")
    if all(isinstance(item, ValidityCode) for item in supplied):
        return cast(tuple[ValidityCode, ...], supplied)
    return tuple(map(ValidityCode, supplied))


@dataclass(frozen=True, slots=True)
class IncidentTransportResult:
    """Shared incident states plus aligned first-failure codes and opt-in traces."""

    states: IncidentStateBatch
    status: tuple[ValidityCode, ...]
    wavelength_A: NDArray[np.float64]
    traces: tuple[TraceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.states, IncidentStateBatch):
            raise TypeError("states must be an IncidentStateBatch")
        size = self.states.incident_state_id.size
        status = _statuses(self.status, size)
        if not np.array_equal(
            self.states.valid,
            np.asarray(status, dtype="U16") == ValidityCode.VALID,
        ):
            raise ValueError("incident valid flags must agree with status")
        object.__setattr__(self, "status", status)
        wavelength = np.array(self.wavelength_A, dtype=np.float64, copy=True, order="C")
        if (
            wavelength.shape != (size,)
            or not np.all(np.isfinite(wavelength))
            or np.any(wavelength <= 0.0)
        ):
            raise ValueError(f"wavelength_A must contain {size} finite positive values")
        wavelength.setflags(write=False)
        object.__setattr__(self, "wavelength_A", wavelength)
        traces = tuple(self.traces)
        if any(not isinstance(record, TraceRecord) for record in traces):
            raise TypeError("traces must contain TraceRecord values")
        object.__setattr__(self, "traces", traces)


@dataclass(frozen=True, slots=True)
class EventTransportResult:
    """Shared outgoing waves and hits with separate aligned first-failure codes."""

    outgoing_waves: OutgoingWaveBatch
    detector_hits: DetectorHitBatch
    outgoing_status: tuple[ValidityCode, ...]
    detector_status: tuple[ValidityCode, ...]
    traces: tuple[TraceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.outgoing_waves, OutgoingWaveBatch):
            raise TypeError("outgoing_waves must be an OutgoingWaveBatch")
        if not isinstance(self.detector_hits, DetectorHitBatch):
            raise TypeError("detector_hits must be a DetectorHitBatch")
        if not np.array_equal(self.outgoing_waves.event_id, self.detector_hits.event_id):
            raise ValueError("outgoing and detector event IDs must align")
        size = self.outgoing_waves.event_id.size
        outgoing_status = _statuses(self.outgoing_status, size)
        detector_status = _statuses(self.detector_status, size)
        expected_outgoing = np.asarray(outgoing_status, dtype="U16") == ValidityCode.VALID
        expected_detector = np.asarray(detector_status, dtype="U16") == ValidityCode.VALID
        if not np.array_equal(self.outgoing_waves.valid, expected_outgoing):
            raise ValueError("outgoing valid flags must agree with outgoing_status")
        if not np.array_equal(self.detector_hits.valid, expected_detector):
            raise ValueError("detector valid flags must agree with detector_status")
        object.__setattr__(self, "outgoing_status", outgoing_status)
        object.__setattr__(self, "detector_status", detector_status)
        traces = tuple(self.traces)
        if any(not isinstance(record, TraceRecord) for record in traces):
            raise TypeError("traces must contain TraceRecord values")
        object.__setattr__(self, "traces", traces)


def _trace_records(
    case_prefix: str | None,
    identity_name: str,
    identities: NDArray[np.int64],
    stages: tuple[_TraceStage, ...],
) -> tuple[TraceRecord, ...]:
    if case_prefix is None:
        return ()
    if not case_prefix:
        raise ValueError("trace_case_id must be nonempty when supplied")
    records: list[TraceRecord] = []
    for row, identity in enumerate(identities):
        case_id = f"{case_prefix}.{identity_name}={int(identity)}"
        for stage_id, values, unit, frame, measure, quantity_kind in stages:
            records.append(
                TraceRecord(
                    case_id=case_id,
                    stage_id=stage_id,
                    value=values[row],
                    unit=unit,
                    frame=frame,
                    measure=measure,
                    quantity_kind=quantity_kind,
                    model_version=_MODEL_VERSION,
                    provenance=_PROVENANCE,
                )
            )
    return tuple(records)


def build_incident_states(
    samples: IncidentSampleBatch,
    material: MaterialOptics,
    instrument: CompiledInstrument,
    *,
    trace_case_id: str | None = None,
) -> IncidentTransportResult:
    """Intersect and refract an incident batch without changing source probability mass."""

    if not isinstance(samples, IncidentSampleBatch):
        raise TypeError("samples must be an IncidentSampleBatch")
    if not isinstance(material, MaterialOptics):
        raise TypeError("material must be MaterialOptics")
    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be a CompiledInstrument")

    size = samples.incident_sample_id.size
    intersections = _intersect_sample_rays(
        samples.origin_lab_m,
        samples.direction_lab,
        lab_from_sample=instrument.lab_from_sample,
        sample_width_m=instrument.sample_width_m,
        sample_length_m=instrument.sample_length_m,
    )
    status = intersections.status.copy()
    geometry_valid = status == ValidityCode.VALID
    direction_sample = instrument.lab_from_sample.inverse().apply_vector(samples.direction_lab)

    modes = _solve_incident_mode_arrays(
        direction_sample,
        samples.wavelength_A,
        material,
    )
    optical_failure = geometry_valid & (modes.status != ValidityCode.VALID)
    status[optical_failure] = modes.status[optical_failure]
    valid = status == ValidityCode.VALID

    intersection_lab_m = np.zeros((size, 3), dtype=np.float64)
    direction_output = np.zeros((size, 3), dtype=np.float64)
    k_air_output = np.zeros((size, 3), dtype=np.float64)
    k_parallel_output = np.zeros((size, 3), dtype=np.float64)
    k_film_output = np.zeros((size, 3), dtype=np.float64)
    kz_film_output = np.zeros(size, dtype=np.complex128)
    entrance_output = np.zeros(size, dtype=np.complex128)
    intersection_lab_m[geometry_valid] = intersections.point_lab_m[geometry_valid]
    direction_output[geometry_valid] = direction_sample[geometry_valid]
    k_air_output[geometry_valid] = modes.k_air_sample_Ainv[geometry_valid]
    k_parallel_output[geometry_valid] = modes.k_parallel_sample_Ainv[geometry_valid]
    k_film_output[geometry_valid] = modes.k_film_phase_sample_Ainv[geometry_valid]
    kz_film_output[geometry_valid] = modes.kz_film_Ainv[geometry_valid]
    entrance_output[valid] = modes.entrance_amplitude[valid]
    footprint_acceptance = intersections.footprint_acceptance

    states = IncidentStateBatch(
        incident_state_id=samples.incident_sample_id,
        incident_sample_id=samples.incident_sample_id,
        sample_intersection_lab_m=intersection_lab_m,
        direction_sample=direction_output,
        k_air_sample_Ainv=k_air_output,
        k_film_phase_sample_Ainv=k_film_output,
        kz_film_Ainv=kz_film_output,
        entrance_amplitude=entrance_output,
        footprint_acceptance=footprint_acceptance,
        source_weight=samples.source_weight,
        valid=valid,
    )
    traces = _trace_records(
        trace_case_id,
        "incident_sample_id",
        states.incident_sample_id,
        (
            (
                "geometry.sample_intersection",
                states.sample_intersection_lab_m,
                "m",
                FrameId.LAB,
                Measure.NONE,
                QuantityKind.POINT,
            ),
            (
                "geometry.footprint_acceptance",
                states.footprint_acceptance,
                "1",
                FrameId.NONE,
                Measure.NONE,
                QuantityKind.SCALAR,
            ),
            (
                "optics.ki_air_sample",
                states.k_air_sample_Ainv,
                "angstrom^-1",
                FrameId.SAMPLE,
                Measure.NONE,
                QuantityKind.VECTOR,
            ),
            (
                "optics.ki_parallel_sample",
                k_parallel_output,
                "angstrom^-1",
                FrameId.SAMPLE,
                Measure.NONE,
                QuantityKind.VECTOR,
            ),
            (
                "optics.kz_incident_film",
                states.kz_film_Ainv,
                "angstrom^-1",
                FrameId.SAMPLE,
                Measure.NONE,
                QuantityKind.SCALAR,
            ),
            (
                "optics.entrance_amplitude",
                states.entrance_amplitude,
                "1",
                FrameId.SAMPLE,
                Measure.NONE,
                QuantityKind.AMPLITUDE,
            ),
            (
                "measurement.source_weight",
                states.source_weight,
                "1",
                FrameId.NONE,
                Measure.PROBABILITY_MASS,
                QuantityKind.SCALAR,
            ),
        ),
    )
    return IncidentTransportResult(
        states,
        tuple(status),
        samples.wavelength_A,
        traces,
    )


def _join_incident_states(
    events: ScatteringEventBatch,
    incident: IncidentTransportResult,
) -> NDArray[np.intp]:
    state_ids = incident.states.incident_state_id
    order = np.argsort(state_ids)
    sorted_ids = state_ids[order]
    positions = np.searchsorted(sorted_ids, events.incident_state_id)
    bounded = positions < sorted_ids.size
    matched = np.zeros(events.event_id.size, dtype=np.bool_)
    matched[bounded] = sorted_ids[positions[bounded]] == events.incident_state_id[bounded]
    if not np.all(matched):
        missing = int(events.incident_state_id[np.flatnonzero(~matched)[0]])
        raise ValueError(f"event references unknown incident_state_id {missing}")
    rows = np.asarray(order[positions], dtype=np.intp)
    if not np.array_equal(events.wavelength_A, incident.wavelength_A[rows]):
        raise ValueError("event wavelength must exactly match its incident state wavelength")
    return rows


def transport_scattering_events(
    events: ScatteringEventBatch,
    incident: IncidentTransportResult,
    material: MaterialOptics,
    instrument: CompiledInstrument,
    *,
    trace_case_id: str | None = None,
) -> EventTransportResult:
    """Refract events into air and project valid waves to continuous detector coordinates."""

    if not isinstance(events, ScatteringEventBatch):
        raise TypeError("events must be a ScatteringEventBatch")
    if not isinstance(incident, IncidentTransportResult):
        raise TypeError("incident must be an IncidentTransportResult")
    if not isinstance(material, MaterialOptics):
        raise TypeError("material must be MaterialOptics")
    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be a CompiledInstrument")

    size = events.event_id.size
    incident_rows = _join_incident_states(events, incident)
    k0_Ainv = 2.0 * np.pi / events.wavelength_A
    exit_modes = _solve_exit_mode_arrays(
        events.kf_film_phase_sample_Ainv,
        events.wavelength_A,
        material,
    )

    outgoing_status = np.full(size, ValidityCode.VALID, dtype="U16")
    outgoing_status[~events.valid] = ValidityCode.NO_SOLUTION
    incident_valid = incident.states.valid[incident_rows]
    outgoing_status[events.valid & ~incident_valid] = ValidityCode.NO_SOLUTION
    candidate = (outgoing_status == ValidityCode.VALID) & (exit_modes.status != ValidityCode.VALID)
    outgoing_status[candidate] = exit_modes.status[candidate]
    outgoing_valid = outgoing_status == ValidityCode.VALID

    kf_air_lab_Ainv = instrument.lab_from_sample.apply_vector(exit_modes.k_air_phase_sample_Ainv)
    incident_direction = np.where(
        incident.states.direction_sample[incident_rows, 2] >= 0.0,
        1,
        -1,
    )
    kappa_incident_Ainv = np.asarray(
        mode_decay_constant(
            incident.states.kz_film_Ainv[incident_rows],
            incident_direction,
        ),
        dtype=np.float64,
    )
    kappa_exit_Ainv = np.asarray(
        mode_decay_constant(
            exit_modes.kz_film_Ainv,
            exit_modes.propagation_direction,
        ),
        dtype=np.float64,
    )
    attenuation_weight = np.asarray(
        uniform_depth_attenuation(
            kappa_incident_Ainv,
            kappa_exit_Ainv,
            instrument.film_thickness_A,
        ),
        dtype=np.float64,
    )
    optical_weight = np.asarray(
        scalar_optical_weight(
            incident.states.entrance_amplitude[incident_rows],
            exit_modes.exit_amplitude,
            attenuation_weight,
        ),
        dtype=np.float64,
    )

    kf_air_output = np.zeros((size, 3), dtype=np.float64)
    exit_amplitude_output = np.zeros(size, dtype=np.complex128)
    attenuation_output = np.zeros(size, dtype=np.float64)
    optical_output = np.zeros(size, dtype=np.float64)
    kf_air_output[outgoing_valid] = kf_air_lab_Ainv[outgoing_valid]
    exit_amplitude_output[outgoing_valid] = exit_modes.exit_amplitude[outgoing_valid]
    attenuation_output[outgoing_valid] = attenuation_weight[outgoing_valid]
    optical_output[outgoing_valid] = optical_weight[outgoing_valid]
    outgoing_waves = OutgoingWaveBatch(
        event_id=events.event_id,
        kf_air_lab_Ainv=kf_air_output,
        exit_amplitude=exit_amplitude_output,
        attenuation_weight=attenuation_output,
        optical_weight=optical_output,
        valid=outgoing_valid,
    )

    origin_lab_m = incident.states.sample_intersection_lab_m[incident_rows]
    detector_status = outgoing_status.copy()
    column_output = np.zeros(size, dtype=np.float64)
    row_output = np.zeros(size, dtype=np.float64)
    solid_angle_output = np.zeros(size, dtype=np.float64)
    detector_intersection_lab_m = np.zeros((size, 3), dtype=np.float64)
    outgoing_rows = np.flatnonzero(outgoing_valid)
    if outgoing_rows.size:
        projections = _project_detector_rays(
            origin_lab_m[outgoing_rows],
            kf_air_lab_Ainv[outgoing_rows] / k0_Ainv[outgoing_rows, None],
            instrument,
        )
        detector_status[outgoing_rows] = projections.status
        column_output[outgoing_rows] = projections.column_px
        row_output[outgoing_rows] = projections.row_px
        solid_angle_output[outgoing_rows] = projections.pixel_solid_angle_sr
        detector_intersection_lab_m[outgoing_rows] = projections.point_lab_m
    detector_valid = detector_status == ValidityCode.VALID
    detector_hits = DetectorHitBatch(
        event_id=events.event_id,
        column_px=column_output,
        row_px=row_output,
        pixel_solid_angle_sr=solid_angle_output,
        valid=detector_valid,
    )

    kz_air_output = np.zeros(size, dtype=np.complex128)
    kappa_incident_output = np.zeros(size, dtype=np.float64)
    kappa_exit_output = np.zeros(size, dtype=np.float64)
    kz_air_output[outgoing_valid] = exit_modes.kz_air_Ainv[outgoing_valid]
    kappa_incident_output[outgoing_valid] = kappa_incident_Ainv[outgoing_valid]
    kappa_exit_output[outgoing_valid] = kappa_exit_Ainv[outgoing_valid]
    traces = _trace_records(
        trace_case_id,
        "event_id",
        events.event_id,
        (
            (
                "optics.kf_film_sample",
                events.kf_film_phase_sample_Ainv,
                "angstrom^-1",
                FrameId.SAMPLE,
                Measure.NONE,
                QuantityKind.VECTOR,
            ),
            (
                "optics.kz_exit_air",
                kz_air_output,
                "angstrom^-1",
                FrameId.SAMPLE,
                Measure.NONE,
                QuantityKind.SCALAR,
            ),
            (
                "optics.exit_amplitude",
                outgoing_waves.exit_amplitude,
                "1",
                FrameId.SAMPLE,
                Measure.NONE,
                QuantityKind.AMPLITUDE,
            ),
            (
                "optics.kappa_incident",
                kappa_incident_output,
                "angstrom^-1",
                FrameId.SAMPLE,
                Measure.NONE,
                QuantityKind.SCALAR,
            ),
            (
                "optics.kappa_exit",
                kappa_exit_output,
                "angstrom^-1",
                FrameId.SAMPLE,
                Measure.NONE,
                QuantityKind.SCALAR,
            ),
            (
                "optics.uniform_depth_attenuation",
                outgoing_waves.attenuation_weight,
                "1",
                FrameId.NONE,
                Measure.NONE,
                QuantityKind.INTENSITY,
            ),
            (
                "geometry.kf_air_lab",
                outgoing_waves.kf_air_lab_Ainv,
                "angstrom^-1",
                FrameId.LAB,
                Measure.NONE,
                QuantityKind.VECTOR,
            ),
            (
                "geometry.detector_intersection",
                detector_intersection_lab_m,
                "m",
                FrameId.LAB,
                Measure.NONE,
                QuantityKind.POINT,
            ),
            (
                "geometry.detector_column_px",
                detector_hits.column_px,
                "px",
                FrameId.DETECTOR,
                Measure.NONE,
                QuantityKind.SCALAR,
            ),
            (
                "geometry.detector_row_px",
                detector_hits.row_px,
                "px",
                FrameId.DETECTOR,
                Measure.NONE,
                QuantityKind.SCALAR,
            ),
            (
                "measurement.optical_weight",
                outgoing_waves.optical_weight,
                "1",
                FrameId.NONE,
                Measure.NONE,
                QuantityKind.INTENSITY,
            ),
            (
                "measurement.pixel_solid_angle",
                detector_hits.pixel_solid_angle_sr,
                "sr",
                FrameId.DETECTOR,
                Measure.NONE,
                QuantityKind.SCALAR,
            ),
        ),
    )
    return EventTransportResult(
        outgoing_waves,
        detector_hits,
        tuple(outgoing_status),
        tuple(detector_status),
        traces,
    )
