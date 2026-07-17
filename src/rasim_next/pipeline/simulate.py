"""Generic ordered-crystal composition to a detector-native image."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rasim_next.core.contracts import (
    DetectorHitBatch,
    EventIntensityResult,
    IncidentSampleBatch,
    MaterialOptics,
    OutgoingWaveBatch,
    RodCatalog,
    RodQueryBatch,
    ScatteringEventBatch,
)
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.transport import (
    EventTransportResult,
    IncidentTransportResult,
    build_incident_states,
    transport_scattering_events,
)
from rasim_next.materials.crystal import CrystalStructure
from rasim_next.ordered.amplitudes import ordered_event_result
from rasim_next.pipeline.intersections import (
    IntersectionSupport,
    _build_intersection_support_for_rods,
    _detector_complete_rods,
    _slice_rods,
)
from rasim_next.pipeline.selection import (
    CandidateMassSummary,
    CandidatePool,
    SelectedCandidateBatch,
    _validate_selection_request,
    build_candidate_pool,
)
from rasim_next.render.deposition import DepositionResult, deposit_bilinear
from rasim_next.sampling.mosaic import MosaicOrientationBatch

_ORDERED_EVENT_CHUNK_SIZE = 100_000
_MAX_ATTEMPTS_PER_ROD_CHUNK = 250_000
_UNITY_POLARIZATION_POLICY = "UNITY_APPROXIMATION"
BranchArray = NDArray[np.int8]


@dataclass(frozen=True, slots=True)
class OrderedSimulationResult:
    """Complete rods, compact selected events, and detector-native mass."""

    incident: IncidentTransportResult
    rods: RodCatalog
    mass_summary: CandidateMassSummary
    selected_events: ScatteringEventBatch
    selected_transport: EventTransportResult
    selected_intersection_branch_id: BranchArray
    selection: SelectedCandidateBatch
    deposition: DepositionResult
    phase_population_weight: float
    polarization_policy_id: str
    polarization_provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.incident, IncidentTransportResult):
            raise TypeError("incident must be an IncidentTransportResult")
        if not isinstance(self.rods, RodCatalog):
            raise TypeError("rods must be a RodCatalog")
        if not isinstance(self.mass_summary, CandidateMassSummary):
            raise TypeError("mass_summary must be a CandidateMassSummary")
        if not isinstance(self.selected_events, ScatteringEventBatch):
            raise TypeError("selected_events must be a ScatteringEventBatch")
        if not isinstance(self.selected_transport, EventTransportResult):
            raise TypeError("selected_transport must be an EventTransportResult")
        if not isinstance(self.selection, SelectedCandidateBatch):
            raise TypeError("selection must be a SelectedCandidateBatch")
        if not isinstance(self.deposition, DepositionResult):
            raise TypeError("deposition must be a DepositionResult")
        branch = np.asarray(self.selected_intersection_branch_id)
        if (
            branch.dtype.kind not in "iu"
            or branch.shape != self.selected_events.event_id.shape
            or np.any((branch < 0) | (branch > 2))
        ):
            raise ValueError("selected_intersection_branch_id must align and contain 0, 1, or 2")
        compact_branch = np.array(branch, dtype=np.int8, copy=True, order="C")
        selected_row = self.selection.candidate_row
        if (
            np.any(selected_row >= self.selected_events.event_id.size)
            or not np.array_equal(
                self.selected_events.event_id[selected_row],
                self.selection.event_id,
            )
            or not np.array_equal(
                compact_branch[selected_row],
                self.selection.intersection_branch_id,
            )
            or not np.array_equal(
                self.selected_transport.detector_hits.event_id,
                self.selected_events.event_id,
            )
            or not np.all(self.selected_transport.detector_hits.valid)
        ):
            raise ValueError("compact selected events, transport, and draws must align")
        compact_branch.setflags(write=False)
        object.__setattr__(self, "selected_intersection_branch_id", compact_branch)


def _event_query(support: IntersectionSupport) -> RodQueryBatch:
    events = support.event_build.events
    rods = support.rods
    rod_order = np.argsort(rods.rod_id)
    sorted_rod_id = rods.rod_id[rod_order]
    position = np.searchsorted(sorted_rod_id, events.rod_id)
    if np.any(position >= sorted_rod_id.size) or not np.array_equal(
        sorted_rod_id[position], events.rod_id
    ):
        raise ValueError("scattering events reference unknown rod_id values")
    rod_row = rod_order[position]
    return RodQueryBatch(
        event_id=events.event_id,
        rod_id=events.rod_id,
        phase_id=tuple(rods.phase_id[int(row)] for row in rod_row),
        h=rods.h[rod_row],
        k=rods.k[rod_row],
        q_sample_normal_Ainv=events.q_sample_normal_Ainv,
        l_coordinate=events.l_coordinate,
        wavelength_A=events.wavelength_A,
    )


def _ordered_intensity(
    crystal: CrystalStructure,
    support: IntersectionSupport,
    query: RodQueryBatch,
) -> EventIntensityResult:
    event_count = query.event_id.size
    if event_count <= _ORDERED_EVENT_CHUNK_SIZE:
        return ordered_event_result(crystal, support.rods, query)

    scattering_strength = np.empty(event_count, dtype=np.float64)
    metadata = None
    for start in range(0, event_count, _ORDERED_EVENT_CHUNK_SIZE):
        stop = min(start + _ORDERED_EVENT_CHUNK_SIZE, event_count)
        chunk_query = RodQueryBatch(
            event_id=query.event_id[start:stop],
            rod_id=query.rod_id[start:stop],
            phase_id=query.phase_id[start:stop],
            h=query.h[start:stop],
            k=query.k[start:stop],
            q_sample_normal_Ainv=query.q_sample_normal_Ainv[start:stop],
            l_coordinate=query.l_coordinate[start:stop],
            wavelength_A=query.wavelength_A[start:stop],
        )
        chunk = ordered_event_result(crystal, support.rods, chunk_query)
        if not np.array_equal(chunk.event_id, query.event_id[start:stop]):
            raise RuntimeError("ordered chunk result changed event alignment")
        scattering_strength[start:stop] = chunk.scattering_strength_A2
        chunk_metadata = (
            chunk.model_id,
            chunk.model_component_id,
            chunk.population_group_id,
            chunk.normalization,
        )
        if metadata is None:
            metadata = chunk_metadata
        elif chunk_metadata != metadata:
            raise RuntimeError("ordered chunk result changed intensity metadata")
    del chunk, chunk_query
    if metadata is None:
        raise RuntimeError("ordered chunk evaluation produced no result")
    return EventIntensityResult(
        event_id=query.event_id,
        scattering_strength_A2=scattering_strength,
        model_id=metadata[0],
        model_component_id=metadata[1],
        population_group_id=metadata[2],
        normalization=metadata[3],
    )


def _candidate_chunk(
    *,
    crystal: CrystalStructure,
    incident_samples: IncidentSampleBatch,
    incident: IncidentTransportResult,
    material: MaterialOptics,
    instrument: CompiledInstrument,
    rods: RodCatalog,
    orientations: MosaicOrientationBatch,
    population: float,
) -> CandidatePool:
    support = _build_intersection_support_for_rods(
        incident_samples=incident_samples,
        incident_states=incident.states,
        rods=rods,
        orientations=orientations,
        sample_from_crystal=instrument.sample_from_crystal,
    )
    query = _event_query(support)
    intensity = _ordered_intensity(crystal, support, query)
    transport = transport_scattering_events(
        support.event_build.events,
        incident,
        material,
        instrument,
    )
    event_id = support.event_build.events.event_id
    return build_candidate_pool(
        support=support,
        incident_states=incident.states,
        intensity=intensity,
        transport=transport,
        population_event_id=event_id,
        population_weight=np.full(event_id.shape, population),
        polarization_event_id=event_id,
        polarization_weight=np.ones(event_id.shape),
    )


def _chunk_mass_rows(
    pool: CandidatePool,
    state_id: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    events = pool.support.event_build.events
    pool_state_id = events.incident_state_id[pool.event_row]
    pool_branch = pool.support.intersection_branch_id[pool.event_row]
    detector_count = np.empty(state_id.size, dtype=np.int64)
    positive_count = np.empty(state_id.size, dtype=np.int64)
    total_mass = np.zeros(state_id.size, dtype=np.float64)
    branch_mass = np.zeros((state_id.size, 3), dtype=np.float64)
    for state_row, state_value in enumerate(state_id):
        rows = np.flatnonzero(pool_state_id == state_value)
        detector_count[state_row] = rows.size
        positive_rows = rows[pool.candidate_mass_A2[rows] > 0.0]
        positive_count[state_row] = positive_rows.size
        if positive_rows.size == 0:
            continue
        mass = pool.candidate_mass_A2[positive_rows]
        total_mass[state_row] = np.cumsum(mass, dtype=np.float64)[-1]
        branch = pool_branch[positive_rows]
        for branch_id in range(3):
            values = mass[branch == branch_id]
            if values.size:
                branch_mass[state_row, branch_id] = np.cumsum(values, dtype=np.float64)[-1]
    return detector_count, positive_count, total_mass, branch_mass


def _compact_chunk_rows(
    pool: CandidatePool,
    event_rows: NDArray[np.int64],
    event_prefix: int,
) -> tuple[ScatteringEventBatch, EventTransportResult, BranchArray]:
    events = pool.support.event_build.events
    transport = pool.transport
    global_event_id = events.event_id[event_rows] + event_prefix
    selected_events = ScatteringEventBatch(
        event_id=global_event_id,
        incident_state_id=events.incident_state_id[event_rows],
        orientation_id=events.orientation_id[event_rows],
        rod_id=events.rod_id[event_rows],
        wavelength_A=events.wavelength_A[event_rows],
        q_internal_sample_Ainv=events.q_internal_sample_Ainv[event_rows],
        q_sample_normal_Ainv=events.q_sample_normal_Ainv[event_rows],
        l_coordinate=events.l_coordinate[event_rows],
        kf_film_phase_sample_Ainv=events.kf_film_phase_sample_Ainv[event_rows],
        reciprocal_weight=events.reciprocal_weight[event_rows],
        ewald_residual_Ainv=events.ewald_residual_Ainv[event_rows],
        status=tuple(events.status[int(row)] for row in event_rows),
        valid=events.valid[event_rows],
    )
    outgoing = transport.outgoing_waves
    hits = transport.detector_hits
    selected_transport = EventTransportResult(
        outgoing_waves=OutgoingWaveBatch(
            event_id=global_event_id,
            kf_air_lab_Ainv=outgoing.kf_air_lab_Ainv[event_rows],
            exit_amplitude=outgoing.exit_amplitude[event_rows],
            attenuation_weight=outgoing.attenuation_weight[event_rows],
            optical_weight=outgoing.optical_weight[event_rows],
            valid=outgoing.valid[event_rows],
        ),
        detector_hits=DetectorHitBatch(
            event_id=global_event_id,
            column_px=hits.column_px[event_rows],
            row_px=hits.row_px[event_rows],
            pixel_solid_angle_sr=hits.pixel_solid_angle_sr[event_rows],
            valid=hits.valid[event_rows],
        ),
        outgoing_status=tuple(transport.outgoing_status[int(row)] for row in event_rows),
        detector_status=tuple(transport.detector_status[int(row)] for row in event_rows),
    )
    branch = np.array(
        pool.support.intersection_branch_id[event_rows],
        dtype=np.int8,
        copy=True,
        order="C",
    )
    branch.setflags(write=False)
    return selected_events, selected_transport, branch


def _merge_compact_chunks(
    pieces: list[tuple[ScatteringEventBatch, EventTransportResult, BranchArray]],
) -> tuple[ScatteringEventBatch, EventTransportResult, BranchArray]:
    if len(pieces) == 1:
        return pieces[0]
    event_pieces = [piece[0] for piece in pieces]
    transport_pieces = [piece[1] for piece in pieces]
    event_id = np.concatenate([piece.event_id for piece in event_pieces])
    events = ScatteringEventBatch(
        event_id=event_id,
        incident_state_id=np.concatenate([piece.incident_state_id for piece in event_pieces]),
        orientation_id=np.concatenate([piece.orientation_id for piece in event_pieces]),
        rod_id=np.concatenate([piece.rod_id for piece in event_pieces]),
        wavelength_A=np.concatenate([piece.wavelength_A for piece in event_pieces]),
        q_internal_sample_Ainv=np.concatenate(
            [piece.q_internal_sample_Ainv for piece in event_pieces]
        ),
        q_sample_normal_Ainv=np.concatenate(
            [piece.q_sample_normal_Ainv for piece in event_pieces]
        ),
        l_coordinate=np.concatenate([piece.l_coordinate for piece in event_pieces]),
        kf_film_phase_sample_Ainv=np.concatenate(
            [piece.kf_film_phase_sample_Ainv for piece in event_pieces]
        ),
        reciprocal_weight=np.concatenate([piece.reciprocal_weight for piece in event_pieces]),
        ewald_residual_Ainv=np.concatenate(
            [piece.ewald_residual_Ainv for piece in event_pieces]
        ),
        status=tuple(status for piece in event_pieces for status in piece.status),
        valid=np.concatenate([piece.valid for piece in event_pieces]),
    )
    outgoing_pieces = [piece.outgoing_waves for piece in transport_pieces]
    hit_pieces = [piece.detector_hits for piece in transport_pieces]
    transport = EventTransportResult(
        outgoing_waves=OutgoingWaveBatch(
            event_id=event_id,
            kf_air_lab_Ainv=np.concatenate(
                [piece.kf_air_lab_Ainv for piece in outgoing_pieces]
            ),
            exit_amplitude=np.concatenate([piece.exit_amplitude for piece in outgoing_pieces]),
            attenuation_weight=np.concatenate(
                [piece.attenuation_weight for piece in outgoing_pieces]
            ),
            optical_weight=np.concatenate([piece.optical_weight for piece in outgoing_pieces]),
            valid=np.concatenate([piece.valid for piece in outgoing_pieces]),
        ),
        detector_hits=DetectorHitBatch(
            event_id=event_id,
            column_px=np.concatenate([piece.column_px for piece in hit_pieces]),
            row_px=np.concatenate([piece.row_px for piece in hit_pieces]),
            pixel_solid_angle_sr=np.concatenate(
                [piece.pixel_solid_angle_sr for piece in hit_pieces]
            ),
            valid=np.concatenate([piece.valid for piece in hit_pieces]),
        ),
        outgoing_status=tuple(
            status for piece in transport_pieces for status in piece.outgoing_status
        ),
        detector_status=tuple(
            status for piece in transport_pieces for status in piece.detector_status
        ),
    )
    branch = np.concatenate([piece[2] for piece in pieces])
    branch.setflags(write=False)
    return events, transport, branch


def simulate_ordered(
    *,
    crystal: CrystalStructure,
    incident_samples: IncidentSampleBatch,
    material: MaterialOptics,
    instrument: CompiledInstrument,
    orientations: MosaicOrientationBatch,
    phase_population_weight: float,
    polarization_policy_id: str,
    polarization_provenance: str,
    selection_seed: int,
    draw_count: int,
) -> OrderedSimulationResult:
    """Select from two streamed all-rod passes and deposit one compact event batch."""

    supplied_population = np.asarray(phase_population_weight)
    if supplied_population.shape != () or supplied_population.dtype.kind not in "iuf":
        raise ValueError("phase_population_weight must be a positive finite scalar")
    try:
        population = float(supplied_population)
    except (TypeError, ValueError) as error:
        raise ValueError("phase_population_weight must be a positive finite scalar") from error
    if not np.isfinite(population) or population <= 0.0:
        raise ValueError("phase_population_weight must be a positive finite scalar")
    if polarization_policy_id != _UNITY_POLARIZATION_POLICY:
        raise ValueError("polarization_policy_id must be exactly UNITY_APPROXIMATION")
    if not isinstance(polarization_provenance, str) or not polarization_provenance.strip():
        raise ValueError("polarization_provenance must be a nonempty string")
    if not isinstance(incident_samples, IncidentSampleBatch):
        raise TypeError("incident_samples must be an IncidentSampleBatch")
    if any(
        state_id != _UNITY_POLARIZATION_POLICY
        for state_id in incident_samples.polarization_state_id
    ):
        raise ValueError("every incident sample must use UNITY_APPROXIMATION polarization")
    generator_seed, draws_per_state = _validate_selection_request(
        selection_seed,
        draw_count,
    )

    incident = build_incident_states(incident_samples, material, instrument)
    rods = _detector_complete_rods(crystal, incident.states)
    valid_state_id = incident.states.incident_state_id[incident.states.valid]
    positive_orientation_count = int(np.count_nonzero(orientations.probability_mass > 0.0))
    attempts_per_rod = int(valid_state_id.size * positive_orientation_count)
    if attempts_per_rod <= 0 or attempts_per_rod > _MAX_ATTEMPTS_PER_ROD_CHUNK:
        raise ValueError("one rod exceeds the fixed streamed intersection attempt budget")
    rods_per_chunk = max(1, _MAX_ATTEMPTS_PER_ROD_CHUNK // attempts_per_rod)
    chunk_starts = np.arange(0, rods.rod_id.size, rods_per_chunk, dtype=np.int64)
    chunk_count = chunk_starts.size
    state_count = valid_state_id.size
    event_prefix = np.empty(chunk_count, dtype=np.int64)
    emitted_count = np.empty(chunk_count, dtype=np.int64)
    attempt_count = np.empty(chunk_count, dtype=np.int64)
    detector_count = np.empty((chunk_count, state_count), dtype=np.int64)
    positive_count = np.empty((chunk_count, state_count), dtype=np.int64)
    chunk_mass = np.empty((chunk_count, state_count), dtype=np.float64)
    chunk_branch_mass = np.empty((chunk_count, state_count, 3), dtype=np.float64)

    next_event_prefix = 0
    for chunk_index, start_value in enumerate(chunk_starts):
        start = int(start_value)
        stop = min(start + rods_per_chunk, rods.rod_id.size)
        chunk_rods = _slice_rods(rods, start, stop)
        pool = _candidate_chunk(
            crystal=crystal,
            incident_samples=incident_samples,
            incident=incident,
            material=material,
            instrument=instrument,
            rods=chunk_rods,
            orientations=orientations,
            population=population,
        )
        local_events = pool.support.event_build.events
        event_prefix[chunk_index] = next_event_prefix
        emitted_count[chunk_index] = local_events.event_id.size
        attempt_count[chunk_index] = pool.support.event_build.status.attempt_id.size
        (
            detector_count[chunk_index],
            positive_count[chunk_index],
            chunk_mass[chunk_index],
            chunk_branch_mass[chunk_index],
        ) = _chunk_mass_rows(pool, valid_state_id)
        next_event_prefix += int(emitted_count[chunk_index])
        if next_event_prefix > np.iinfo(np.int64).max:
            raise ValueError("streamed global event IDs exceed int64 capacity")
        del local_events, pool, chunk_rods

    cumulative_chunk_mass = np.cumsum(chunk_mass, axis=0, dtype=np.float64)
    total_mass = cumulative_chunk_mass[-1]
    if not np.all(np.isfinite(total_mass)) or np.any(total_mass <= 0.0):
        bad_state = int(valid_state_id[np.flatnonzero(~np.isfinite(total_mass) | (total_mass <= 0.0))[0]])
        raise ValueError(
            f"incident_state_id {bad_state} has no detector-valid positive candidate mass"
        )
    total_branch_mass = np.cumsum(chunk_branch_mass, axis=0, dtype=np.float64)[-1]
    mass_summary = CandidateMassSummary(
        incident_state_id=valid_state_id,
        detector_valid_count=np.sum(detector_count, axis=0, dtype=np.int64),
        positive_count=np.sum(positive_count, axis=0, dtype=np.int64),
        total_mass_A2=total_mass,
        branch_mass_A2=total_branch_mass,
        analytic_attempt_count=sum(map(int, attempt_count)),
        emitted_event_count=next_event_prefix,
    )

    generator = np.random.default_rng(generator_seed)
    targets = generator.random((state_count, draws_per_state)) * total_mass[:, None]
    targets = np.minimum(targets, np.nextafter(total_mass[:, None], -np.inf))
    target_chunk = np.empty(targets.shape, dtype=np.int64)
    for state_row in range(state_count):
        target_chunk[state_row] = np.searchsorted(
            cumulative_chunk_mass[:, state_row],
            targets[state_row],
            side="right",
        )
    if np.any(target_chunk >= chunk_count):
        raise RuntimeError("streamed CDF target exceeded the final rod chunk")

    selected_count = state_count * draws_per_state
    compact_row = np.full(selected_count, -1, dtype=np.int64)
    selected_event_id = np.full(selected_count, -1, dtype=np.int64)
    selected_branch = np.full(selected_count, -1, dtype=np.int8)
    selected_state_id = np.repeat(valid_state_id, draws_per_state)
    assigned_mass = np.repeat(total_mass / draws_per_state, draws_per_state)
    compact_pieces: list[tuple[ScatteringEventBatch, EventTransportResult, BranchArray]] = []
    next_compact_row = 0

    for chunk_index_value in np.unique(target_chunk):
        chunk_index = int(chunk_index_value)
        start = int(chunk_starts[chunk_index])
        stop = min(start + rods_per_chunk, rods.rod_id.size)
        chunk_rods = _slice_rods(rods, start, stop)
        pool = _candidate_chunk(
            crystal=crystal,
            incident_samples=incident_samples,
            incident=incident,
            material=material,
            instrument=instrument,
            rods=chunk_rods,
            orientations=orientations,
            population=population,
        )
        local_events = pool.support.event_build.events
        observed = _chunk_mass_rows(pool, valid_state_id)
        if (
            pool.support.event_build.status.attempt_id.size != attempt_count[chunk_index]
            or local_events.event_id.size != emitted_count[chunk_index]
            or not np.array_equal(observed[0], detector_count[chunk_index])
            or not np.array_equal(observed[1], positive_count[chunk_index])
            or not np.array_equal(observed[2], chunk_mass[chunk_index])
            or not np.array_equal(observed[3], chunk_branch_mass[chunk_index])
        ):
            raise RuntimeError("streamed candidate chunk changed between selection passes")

        pool_state_id = local_events.incident_state_id[pool.event_row]
        selected_slots: list[NDArray[np.int64]] = []
        selected_local_rows: list[NDArray[np.int64]] = []
        for state_row, state_value in enumerate(valid_state_id):
            draw_index = np.flatnonzero(target_chunk[state_row] == chunk_index)
            if draw_index.size == 0:
                continue
            state_pool_row = np.flatnonzero(pool_state_id == state_value)
            positive_pool_row = state_pool_row[pool.candidate_mass_A2[state_pool_row] > 0.0]
            cumulative = np.cumsum(
                pool.candidate_mass_A2[positive_pool_row],
                dtype=np.float64,
            )
            previous_mass = (
                0.0
                if chunk_index == 0
                else float(cumulative_chunk_mass[chunk_index - 1, state_row])
            )
            local_target = targets[state_row, draw_index] - previous_mass
            chosen = np.searchsorted(cumulative, local_target, side="right")
            if np.any(chosen >= positive_pool_row.size):
                raise RuntimeError("streamed target exceeded its selected rod chunk")
            event_rows = pool.event_row[positive_pool_row[chosen]]
            if not np.all(local_events.incident_state_id[event_rows] == state_value):
                raise RuntimeError("streamed target changed incident-state identity")
            selected_slots.append(state_row * draws_per_state + draw_index)
            selected_local_rows.append(event_rows)

        slot = np.concatenate(selected_slots)
        local_row = np.concatenate(selected_local_rows)
        unique_local_row, inverse = np.unique(local_row, return_inverse=True)
        piece = _compact_chunk_rows(
            pool,
            unique_local_row,
            int(event_prefix[chunk_index]),
        )
        compact_row[slot] = next_compact_row + inverse
        selected_event_id[slot] = piece[0].event_id[inverse]
        selected_branch[slot] = piece[2][inverse]
        compact_pieces.append(piece)
        next_compact_row += unique_local_row.size
        del local_events, pool, chunk_rods

    if np.any(compact_row < 0) or np.any(selected_event_id < 0) or np.any(selected_branch < 0):
        raise RuntimeError("streamed selection did not resolve every RNG draw")
    selected_events, selected_transport, compact_branch = _merge_compact_chunks(
        compact_pieces
    )
    selection = SelectedCandidateBatch(
        candidate_row=compact_row,
        event_id=selected_event_id,
        incident_state_id=selected_state_id,
        intersection_branch_id=selected_branch,
        assigned_mass_A2=assigned_mass,
    )
    deposition = deposit_bilinear(
        selected_transport.detector_hits,
        event_row=selection.candidate_row,
        event_id=selection.event_id,
        assigned_mass_A2=selection.assigned_mass_A2,
        detector_shape_rc=instrument.detector_shape_rc,
    )
    return OrderedSimulationResult(
        incident=incident,
        rods=rods,
        mass_summary=mass_summary,
        selected_events=selected_events,
        selected_transport=selected_transport,
        selected_intersection_branch_id=compact_branch,
        selection=selection,
        deposition=deposition,
        phase_population_weight=population,
        polarization_policy_id=polarization_policy_id,
        polarization_provenance=polarization_provenance,
    )
