"""Complete detector-valid candidate mass and seeded inverse-CDF selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import EventIntensityResult, IncidentStateBatch
from rasim_next.geometry.transport import EventTransportResult
from rasim_next.pipeline.intersections import IntersectionSupport

IntArray = NDArray[np.int64]
BranchArray = NDArray[np.int8]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CandidatePool:
    """Authoritative inputs plus every detector-valid event row and its mass."""

    support: IntersectionSupport
    incident_states: IncidentStateBatch
    intensity: EventIntensityResult
    transport: EventTransportResult
    event_row: IntArray
    candidate_mass_A2: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.support, IntersectionSupport):
            raise TypeError("support must be an IntersectionSupport")
        if not isinstance(self.incident_states, IncidentStateBatch):
            raise TypeError("incident_states must be an IncidentStateBatch")
        if not isinstance(self.intensity, EventIntensityResult):
            raise TypeError("intensity must be an EventIntensityResult")
        if not isinstance(self.transport, EventTransportResult):
            raise TypeError("transport must be an EventTransportResult")

        events = self.support.event_build.events
        event_id = events.event_id
        if not np.array_equal(event_id, self.intensity.event_id) or not np.array_equal(
            event_id, self.transport.outgoing_waves.event_id
        ):
            raise ValueError("support, intensity, and transport event IDs must align exactly")

        supplied_rows = np.asarray(self.event_row)
        if supplied_rows.dtype.kind not in "iu":
            raise ValueError("event_row must contain integers")
        event_row = np.array(supplied_rows, dtype=np.int64, copy=True, order="C")
        expected_rows = np.flatnonzero(self.transport.detector_hits.valid)
        if not np.array_equal(event_row, expected_rows):
            raise ValueError("event_row must contain every detector-valid event exactly once")

        supplied_mass = np.asarray(self.candidate_mass_A2)
        if np.iscomplexobj(supplied_mass):
            raise ValueError("candidate_mass_A2 must be real")
        candidate_mass = np.array(supplied_mass, dtype=np.float64, copy=True, order="C")
        if (
            candidate_mass.shape != event_row.shape
            or not np.all(np.isfinite(candidate_mass))
            or np.any(candidate_mass < 0.0)
        ):
            raise ValueError("candidate_mass_A2 must be finite, nonnegative, and event-row aligned")
        event_row.setflags(write=False)
        candidate_mass.setflags(write=False)
        object.__setattr__(self, "event_row", event_row)
        object.__setattr__(self, "candidate_mass_A2", candidate_mass)


@dataclass(frozen=True, slots=True)
class CandidateMassSummary:
    """Per-state once-only candidate mass retained by streamed selection."""

    incident_state_id: IntArray
    detector_valid_count: IntArray
    positive_count: IntArray
    total_mass_A2: FloatArray
    branch_mass_A2: FloatArray
    analytic_attempt_count: int
    emitted_event_count: int

    def __post_init__(self) -> None:
        supplied_state = np.asarray(self.incident_state_id)
        if supplied_state.dtype.kind not in "iu" or supplied_state.ndim != 1:
            raise ValueError("incident_state_id must be a one-dimensional integer array")
        state_id = np.array(supplied_state, dtype=np.int64, copy=True, order="C")
        size = state_id.size
        if np.any(state_id < 0) or np.unique(state_id).size != size:
            raise ValueError("incident_state_id must contain unique nonnegative values")

        counts: dict[str, IntArray] = {}
        for name in ("detector_valid_count", "positive_count"):
            supplied = np.asarray(getattr(self, name))
            if supplied.dtype.kind not in "iu" or supplied.shape != (size,):
                raise ValueError(f"{name} must be an incident-state-aligned integer array")
            value = np.array(supplied, dtype=np.int64, copy=True, order="C")
            if np.any(value < 0):
                raise ValueError(f"{name} must be nonnegative")
            counts[name] = value
        if np.any(counts["positive_count"] > counts["detector_valid_count"]):
            raise ValueError("positive_count cannot exceed detector_valid_count")

        total = np.array(self.total_mass_A2, dtype=np.float64, copy=True, order="C")
        branch = np.array(self.branch_mass_A2, dtype=np.float64, copy=True, order="C")
        if (
            total.shape != (size,)
            or branch.shape != (size, 3)
            or not np.all(np.isfinite(total))
            or not np.all(np.isfinite(branch))
            or np.any(total < 0.0)
            or np.any(branch < 0.0)
            or not np.allclose(
                np.sum(branch, axis=1, dtype=np.float64),
                total,
                rtol=32.0 * np.finfo(np.float64).eps,
                atol=0.0,
            )
        ):
            raise ValueError("candidate mass totals and branch masses must be finite and aligned")
        aggregates: list[int] = []
        for name in ("analytic_attempt_count", "emitted_event_count"):
            supplied = getattr(self, name)
            if isinstance(supplied, bool) or not isinstance(supplied, (int, np.integer)):
                raise ValueError(f"{name} must be a nonnegative integer")
            value = int(supplied)
            if value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            aggregates.append(value)

        for array in (state_id, *counts.values(), total, branch):
            array.setflags(write=False)
        object.__setattr__(self, "incident_state_id", state_id)
        for name, value in counts.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "total_mass_A2", total)
        object.__setattr__(self, "branch_mass_A2", branch)
        object.__setattr__(self, "analytic_attempt_count", aggregates[0])
        object.__setattr__(self, "emitted_event_count", aggregates[1])


@dataclass(frozen=True, slots=True)
class SelectedCandidateBatch:
    """Selected compact event rows with equal per-ray assigned mass."""

    candidate_row: IntArray
    event_id: IntArray
    incident_state_id: IntArray
    intersection_branch_id: BranchArray
    assigned_mass_A2: FloatArray

    def __post_init__(self) -> None:
        supplied_row = np.asarray(self.candidate_row)
        if supplied_row.dtype.kind not in "iu" or supplied_row.ndim != 1:
            raise ValueError("candidate_row must be a one-dimensional integer array")
        size = supplied_row.size
        arrays: dict[str, NDArray[np.generic]] = {}
        for name, dtype in (
            ("candidate_row", np.dtype(np.int64)),
            ("event_id", np.dtype(np.int64)),
            ("incident_state_id", np.dtype(np.int64)),
            ("intersection_branch_id", np.dtype(np.int8)),
        ):
            supplied = np.asarray(getattr(self, name))
            if supplied.dtype.kind not in "iu" or supplied.shape != (size,):
                raise ValueError(f"{name} must be an aligned integer array")
            array = np.array(supplied, dtype=dtype, copy=True, order="C")
            array.setflags(write=False)
            arrays[name] = array
        if (
            np.any(arrays["candidate_row"] < 0)
            or np.any(arrays["event_id"] < 0)
            or np.any(arrays["incident_state_id"] < 0)
            or np.any(
                (np.asarray(self.intersection_branch_id) < 0)
                | (np.asarray(self.intersection_branch_id) > 2)
            )
        ):
            raise ValueError("selected identities and branch IDs are out of range")

        supplied_mass = np.asarray(self.assigned_mass_A2)
        if np.iscomplexobj(supplied_mass):
            raise ValueError("assigned_mass_A2 must be real")
        assigned_mass = np.array(supplied_mass, dtype=np.float64, copy=True, order="C")
        if (
            assigned_mass.shape != (size,)
            or not np.all(np.isfinite(assigned_mass))
            or np.any(assigned_mass < 0.0)
        ):
            raise ValueError("assigned_mass_A2 must be finite, nonnegative, and aligned")
        assigned_mass.setflags(write=False)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "assigned_mass_A2", assigned_mass)


def _validate_selection_request(seed: int, draw_count: int) -> tuple[int, int]:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        isinstance(draw_count, bool)
        or not isinstance(draw_count, (int, np.integer))
        or draw_count <= 0
    ):
        raise ValueError("draw_count must be a positive integer")
    return int(seed), int(draw_count)


def build_candidate_pool(
    *,
    support: IntersectionSupport,
    incident_states: IncidentStateBatch,
    intensity: EventIntensityResult,
    transport: EventTransportResult,
    population_event_id: ArrayLike,
    population_weight: ArrayLike,
    polarization_event_id: ArrayLike,
    polarization_weight: ArrayLike,
) -> CandidatePool:
    """Assemble each detector-valid event's once-only physical candidate mass."""

    if not isinstance(support, IntersectionSupport):
        raise TypeError("support must be an IntersectionSupport")
    if not isinstance(incident_states, IncidentStateBatch):
        raise TypeError("incident_states must be an IncidentStateBatch")
    if not isinstance(intensity, EventIntensityResult):
        raise TypeError("intensity must be an EventIntensityResult")
    if not isinstance(transport, EventTransportResult):
        raise TypeError("transport must be an EventTransportResult")

    events = support.event_build.events
    event_id = events.event_id
    if not np.array_equal(event_id, intensity.event_id) or not np.array_equal(
        event_id, transport.outgoing_waves.event_id
    ):
        raise ValueError("support, intensity, and transport event IDs must align exactly")
    event_row = np.flatnonzero(transport.detector_hits.valid)
    if np.any(transport.detector_hits.valid & ~transport.outgoing_waves.valid):
        raise ValueError("detector-valid events must have valid outgoing transport")

    weights: list[FloatArray] = []
    for name, supplied_id, supplied_weight in (
        ("population", population_event_id, population_weight),
        ("polarization", polarization_event_id, polarization_weight),
    ):
        ids = np.asarray(supplied_id)
        if ids.dtype.kind not in "iu":
            raise ValueError(f"{name}_event_id must contain integers")
        aligned_id = np.array(ids, dtype=np.int64, copy=True)
        if not np.array_equal(event_id, aligned_id):
            raise ValueError(f"{name} weights must align exactly by event ID")
        supplied = np.asarray(supplied_weight)
        if np.iscomplexobj(supplied):
            raise ValueError(f"{name}_weight must be real")
        weight = np.array(supplied, dtype=np.float64, copy=True, order="C")
        if (
            weight.shape != event_id.shape
            or not np.all(np.isfinite(weight))
            or np.any(weight < 0.0)
        ):
            raise ValueError(f"{name}_weight must be finite, nonnegative, and event aligned")
        weights.append(weight)
    population, polarization = weights

    state_order = np.argsort(incident_states.incident_state_id)
    sorted_state_id = incident_states.incident_state_id[state_order]
    candidate_state_id = events.incident_state_id[event_row]
    state_position = np.searchsorted(sorted_state_id, candidate_state_id)
    if np.any(state_position >= sorted_state_id.size) or not np.array_equal(
        sorted_state_id[state_position], candidate_state_id
    ):
        raise ValueError("events reference unknown incident_state_id values")
    state_row = state_order[state_position]
    if np.any(~incident_states.valid[state_row]):
        raise ValueError("events must reference valid incident states")

    factors = (
        incident_states.source_weight[state_row],
        events.reciprocal_weight[event_row],
        population[event_row],
        intensity.scattering_strength_A2[event_row],
        transport.outgoing_waves.optical_weight[event_row],
        incident_states.footprint_acceptance[state_row],
        polarization[event_row],
    )
    candidate_mass = np.zeros(event_row.size, dtype=np.float64)
    positive = np.logical_and.reduce(tuple(factor > 0.0 for factor in factors))
    if np.any(positive):
        product = np.ones(np.count_nonzero(positive), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for factor in factors:
                product *= factor[positive]
        if not np.all(np.isfinite(product)):
            raise ValueError("candidate mass is not finite in float64")
        candidate_mass[positive] = product

    return CandidatePool(
        support=support,
        incident_states=incident_states,
        intensity=intensity,
        transport=transport,
        event_row=event_row,
        candidate_mass_A2=candidate_mass,
    )


def select_candidates(
    pool: CandidatePool,
    *,
    seed: int,
    draw_count: int,
) -> SelectedCandidateBatch:
    """Draw the requested count per valid incident state from its one all-rod pool."""

    if not isinstance(pool, CandidatePool):
        raise TypeError("pool must be a CandidatePool")
    generator_seed, draws_per_ray = _validate_selection_request(seed, draw_count)

    valid_state_id = pool.incident_states.incident_state_id[pool.incident_states.valid]
    selected_count = valid_state_id.size * draws_per_ray
    candidate_row = np.empty(selected_count, dtype=np.int64)
    event_id = np.empty(selected_count, dtype=np.int64)
    incident_state_id = np.empty(selected_count, dtype=np.int64)
    branch_id = np.empty(selected_count, dtype=np.int8)
    assigned_mass = np.empty(selected_count, dtype=np.float64)
    events = pool.support.event_build.events
    pool_state_id = events.incident_state_id[pool.event_row]
    generator = np.random.default_rng(generator_seed)

    cursor = 0
    for state_id_value in valid_state_id:
        state_id = int(state_id_value)
        state_pool_row = np.flatnonzero(pool_state_id == state_id)
        positive_pool_row = state_pool_row[pool.candidate_mass_A2[state_pool_row] > 0.0]
        if positive_pool_row.size == 0:
            raise ValueError(
                f"incident_state_id {state_id} has no detector-valid positive candidate mass"
            )
        state_mass = pool.candidate_mass_A2[positive_pool_row]
        cumulative = np.cumsum(state_mass, dtype=np.float64)
        total_mass = float(cumulative[-1])
        mass_per_draw = total_mass / draws_per_ray
        if (
            not np.isfinite(total_mass)
            or total_mass <= 0.0
            or mass_per_draw <= 0.0
        ):
            raise ValueError(f"incident_state_id {state_id} has no finite positive cumulative mass")
        targets = np.minimum(
            generator.random(draws_per_ray) * total_mass,
            np.nextafter(total_mass, -np.inf),
        )
        selected_pool_row = positive_pool_row[
            np.searchsorted(cumulative, targets, side="right")
        ]
        selected_event_row = pool.event_row[selected_pool_row]
        stop = cursor + draws_per_ray
        candidate_row[cursor:stop] = selected_event_row
        event_id[cursor:stop] = events.event_id[selected_event_row]
        incident_state_id[cursor:stop] = events.incident_state_id[selected_event_row]
        branch_id[cursor:stop] = pool.support.intersection_branch_id[selected_event_row]
        assigned_mass[cursor:stop] = mass_per_draw
        cursor = stop

    return SelectedCandidateBatch(
        candidate_row=candidate_row,
        event_id=event_id,
        incident_state_id=incident_state_id,
        intersection_branch_id=branch_id,
        assigned_mass_A2=assigned_mass,
    )
