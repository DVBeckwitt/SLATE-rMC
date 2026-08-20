"""Vectorized incident transport with explicit validity and optional traces."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rasim_next.core.contracts import (
    IncidentSampleBatch,
    IncidentStateBatch,
    MaterialOptics,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.traces import Measure, QuantityKind, TraceRecord
from rasim_next.core.validity import ValidityCode
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.sample import _intersect_sample_rays
from rasim_next.optics.refraction import _solve_incident_mode_arrays

_MODEL_VERSION = "geometry-optics-v1"
_PROVENANCE = "T02 detector-native geometry and planar-interface optics"
_INCIDENT_MODEL_ID = "one_transmitted_channel.v1"

type _TraceStage = tuple[
    str,
    NDArray[np.generic],
    str,
    str,
    Measure,
    QuantityKind,
]


@dataclass(frozen=True, slots=True)
class IncidentTransportResult:
    """Authoritative incident states plus opt-in traces."""

    states: IncidentStateBatch
    traces: tuple[TraceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.states, IncidentStateBatch):
            raise TypeError("states must be an IncidentStateBatch")
        traces = tuple(self.traces)
        if any(not isinstance(record, TraceRecord) for record in traces):
            raise TypeError("traces must contain TraceRecord values")
        object.__setattr__(self, "traces", traces)


def _trace_records(
    case_prefix: str | None,
    identity_name: str,
    identities: NDArray[np.int64],
    stages: tuple[_TraceStage, ...],
    provenance: str = _PROVENANCE,
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
                    provenance=provenance,
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
        sample_from_lab=instrument.sample_from_lab,
        sample_support_model_id=instrument.sample_support_model_id,
        sample_width_m=instrument.sample_width_m,
        sample_length_m=instrument.sample_length_m,
    )
    status = intersections.status.copy()
    geometry_rows = np.flatnonzero(status == ValidityCode.VALID)

    intersection_lab_m = np.zeros((size, 3), dtype=np.float64)
    direction_output = np.zeros((size, 3), dtype=np.float64)
    k_air_output = np.zeros((size, 3), dtype=np.float64)
    k_film_output = np.zeros((size, 3), dtype=np.float64)
    kz_film_output = np.zeros(size, dtype=np.complex128)
    entrance_output = np.zeros(size, dtype=np.complex128)
    modes = None
    if geometry_rows.size:
        modes = _solve_incident_mode_arrays(
            intersections.direction_sample[geometry_rows],
            samples.wavelength_A[geometry_rows],
            material,
        )
        optical_valid = modes.status == ValidityCode.VALID
        status[geometry_rows[~optical_valid]] = modes.status[~optical_valid]
        valid_mode_rows = np.flatnonzero(optical_valid)
        valid_rows = geometry_rows[valid_mode_rows]

        intersection_lab_m[geometry_rows] = intersections.point_lab_m[geometry_rows]
        direction_output[geometry_rows] = intersections.direction_sample[geometry_rows]
        k_air_output[geometry_rows] = modes.k_air_sample_Ainv
        k_film_output[valid_rows] = modes.k_film_phase_sample_Ainv[valid_mode_rows]
        kz_film_output[valid_rows] = modes.kz_film_Ainv[valid_mode_rows]
        entrance_output[valid_rows] = modes.entrance_amplitude[valid_mode_rows]

    states = IncidentStateBatch(
        incident_state_id=samples.incident_sample_id,
        incident_sample_id=samples.incident_sample_id,
        source_origin_lab_m=samples.origin_lab_m,
        source_direction_lab=samples.direction_lab,
        sample_intersection_lab_m=intersection_lab_m,
        direction_sample=direction_output,
        k_air_sample_Ainv=k_air_output,
        k_film_phase_sample_Ainv=k_film_output,
        kz_film_Ainv=kz_film_output,
        entrance_amplitude=entrance_output,
        footprint_acceptance=intersections.footprint_acceptance,
        source_weight=samples.source_weight,
        wavelength_A=samples.wavelength_A,
        polarization_state_id=samples.polarization_state_id,
        status=tuple(map(ValidityCode, status)),
        valid=status == ValidityCode.VALID,
        source_sampling_model_id=samples.source_sampling_model_id,
        source_rng_model_id=samples.source_rng_model_id,
        source_seed=samples.source_seed,
        source_parameter_provenance=samples.source_parameter_provenance,
        source_parameter_revision=samples.source_parameter_revision,
        source_weight_revision=samples.source_weight_revision,
        source_revision=samples.source_revision,
        sample_geometry_revision=instrument.sample_geometry_revision,
        material_revision=material.material_revision,
        incident_model_id=_INCIDENT_MODEL_ID,
    )
    if trace_case_id is None:
        return IncidentTransportResult(states)

    trace_provenance = json.dumps(
        {
            "incident_model_id": states.incident_model_id,
            "material_revision": states.material_revision,
            "sample_geometry_revision": states.sample_geometry_revision,
            "scientific_provenance": _PROVENANCE,
            "source_parameter_revision": states.source_parameter_revision,
            "source_revision": states.source_revision,
            "source_rng_model_id": states.source_rng_model_id,
            "source_sampling_model_id": states.source_sampling_model_id,
            "source_seed": states.source_seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    trace_records: list[TraceRecord] = []
    zero_parallel = np.zeros(3, dtype=np.float64)
    compact_row = 0
    for row in range(size):
        if compact_row < geometry_rows.size and row == int(geometry_rows[compact_row]):
            assert modes is not None
            parallel_value = modes.k_parallel_sample_Ainv[compact_row]
            compact_row += 1
        else:
            parallel_value = zero_parallel
        trace_records.extend(
            _trace_records(
                trace_case_id,
                "incident_sample_id",
                states.incident_sample_id[row : row + 1],
                (
                    (
                        "geometry.sample_intersection",
                        states.sample_intersection_lab_m[row : row + 1],
                        "m",
                        FrameId.LAB,
                        Measure.NONE,
                        QuantityKind.POINT,
                    ),
                    (
                        "geometry.footprint_acceptance",
                        states.footprint_acceptance[row : row + 1],
                        "1",
                        FrameId.NONE,
                        Measure.NONE,
                        QuantityKind.SCALAR,
                    ),
                    (
                        "optics.ki_air_sample",
                        states.k_air_sample_Ainv[row : row + 1],
                        "angstrom^-1",
                        FrameId.SAMPLE,
                        Measure.NONE,
                        QuantityKind.VECTOR,
                    ),
                    (
                        "optics.ki_parallel_sample",
                        parallel_value[None, :],
                        "angstrom^-1",
                        FrameId.SAMPLE,
                        Measure.NONE,
                        QuantityKind.VECTOR,
                    ),
                    (
                        "optics.kz_incident_film",
                        states.kz_film_Ainv[row : row + 1],
                        "angstrom^-1",
                        FrameId.SAMPLE,
                        Measure.NONE,
                        QuantityKind.SCALAR,
                    ),
                    (
                        "optics.entrance_amplitude",
                        states.entrance_amplitude[row : row + 1],
                        "1",
                        FrameId.SAMPLE,
                        Measure.NONE,
                        QuantityKind.AMPLITUDE,
                    ),
                    (
                        "sampling.source_empirical_mass",
                        states.source_weight[row : row + 1],
                        "1",
                        FrameId.NONE,
                        Measure.PROBABILITY_MASS,
                        QuantityKind.SCALAR,
                    ),
                ),
                trace_provenance,
            )
        )
    return IncidentTransportResult(states, tuple(trace_records))
