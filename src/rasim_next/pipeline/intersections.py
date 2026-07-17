"""Detector-complete rod support and Ewald-root branch provenance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import IncidentSampleBatch, IncidentStateBatch, RodCatalog
from rasim_next.core.transforms import RigidTransform
from rasim_next.materials.crystal import CrystalStructure
from rasim_next.reciprocal.events import EventBuildResult, build_scattering_events
from rasim_next.reciprocal.ewald import EwaldRootStatus
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.reciprocal.rods import build_rod_catalog
from rasim_next.sampling.mosaic import MosaicOrientationBatch

BranchArray = NDArray[np.int8]


@dataclass(frozen=True, slots=True)
class IntersectionSupport:
    """Individual rods, their analytic events, and event-aligned root identities."""

    rods: RodCatalog
    event_build: EventBuildResult
    intersection_branch_id: BranchArray

    def __post_init__(self) -> None:
        if not isinstance(self.rods, RodCatalog):
            raise TypeError("rods must be a RodCatalog")
        if not isinstance(self.event_build, EventBuildResult):
            raise TypeError("event_build must be an EventBuildResult")
        supplied = np.asarray(self.intersection_branch_id)
        if supplied.dtype.kind not in "iu":
            raise ValueError("intersection_branch_id must contain integers")
        event_count = self.event_build.events.event_id.size
        if supplied.shape != (event_count,) or np.any((supplied < 0) | (supplied > 2)):
            raise ValueError("intersection_branch_id must align with events and contain 0, 1, or 2")
        branch = np.array(supplied, dtype=np.int8, copy=True, order="C")
        branch.setflags(write=False)
        object.__setattr__(self, "intersection_branch_id", branch)


def _outward_hk_extent(
    q_max_Ainv: float,
    diagonal_upper: float,
    determinant_lower: float,
    metric_scale_exponent: int,
) -> float:
    diagonal_mantissa, diagonal_exponent = math.frexp(diagonal_upper)
    determinant_mantissa, determinant_exponent = math.frexp(determinant_lower)
    ratio_mantissa = float(
        np.nextafter(diagonal_mantissa / determinant_mantissa, np.inf)
    )
    ratio_exponent = diagonal_exponent - determinant_exponent - metric_scale_exponent
    if ratio_exponent % 2:
        ratio_mantissa *= 2.0
        ratio_exponent -= 1
    square_root_mantissa = float(np.nextafter(math.sqrt(ratio_mantissa), np.inf))
    q_mantissa, q_exponent = math.frexp(q_max_Ainv)
    extent_mantissa = float(np.nextafter(q_mantissa * square_root_mantissa, np.inf))
    try:
        extent = math.ldexp(extent_mantissa, q_exponent + ratio_exponent // 2)
    except OverflowError:
        return math.inf
    return float(np.nextafter(extent, np.inf))


def _symmetric_hk_bounds(
    incident_states: IncidentStateBatch,
    inplane_metric_Ainv2: ArrayLike,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Enclose every integer rod satisfying ``Qr <= 2*max(|ki|)``."""

    if not isinstance(incident_states, IncidentStateBatch):
        raise TypeError("incident_states must be an IncidentStateBatch")
    valid_wavevectors = incident_states.k_film_phase_sample_Ainv[incident_states.valid]
    if valid_wavevectors.shape[0] == 0:
        raise ValueError("at least one valid incident state is required")
    norm_upper = max(
        float(np.nextafter(math.hypot(*map(float, wavevector)), np.inf))
        for wavevector in valid_wavevectors
    )
    if not np.isfinite(norm_upper) or norm_upper <= 0.0:
        raise ValueError("valid incident phase wavevectors must have finite positive norm")
    with np.errstate(over="ignore"):
        doubled = 2.0 * norm_upper
    if not np.isfinite(doubled):
        raise ValueError("incident phase wavevectors are too large for a finite rod bound")
    q_max_Ainv = float(np.nextafter(doubled, np.inf))

    supplied = np.asarray(inplane_metric_Ainv2)
    if np.iscomplexobj(supplied):
        raise ValueError("in-plane reciprocal metric must be real")
    metric = np.array(supplied, dtype=np.float64, copy=True)
    if metric.shape != (2, 2) or not np.all(np.isfinite(metric)):
        raise ValueError("in-plane reciprocal metric must be a finite 2 by 2 matrix")
    metric_scale = float(np.max(np.abs(metric)))
    if metric_scale == 0.0:
        raise ValueError("in-plane reciprocal metric must be positive definite")
    _, metric_scale_exponent = math.frexp(metric_scale)
    metric_scale_exponent -= 1
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled_metric = np.ldexp(metric, -metric_scale_exponent)
    if not np.all(np.isfinite(scaled_metric)):
        raise ValueError("in-plane reciprocal metric is not certifiable in float64")
    scaled_metric_max = float(np.max(np.abs(scaled_metric)))
    symmetry_tolerance = 64.0 * np.finfo(np.float64).eps * scaled_metric_max
    if abs(float(scaled_metric[0, 1] - scaled_metric[1, 0])) > symmetry_tolerance:
        raise ValueError("in-plane reciprocal metric must be symmetric")

    diagonal_0 = float(np.nextafter(scaled_metric[0, 0], -np.inf))
    diagonal_1 = float(np.nextafter(scaled_metric[1, 1], -np.inf))
    if diagonal_0 <= 0.0 or diagonal_1 <= 0.0:
        raise ValueError("in-plane reciprocal metric must be positive definite")
    off_diagonal = float(
        np.nextafter(
            max(abs(float(scaled_metric[0, 1])), abs(float(scaled_metric[1, 0]))),
            np.inf,
        )
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        diagonal_product = diagonal_0 * diagonal_1
        off_diagonal_squared = off_diagonal * off_diagonal
    if not np.isfinite(diagonal_product) or not np.isfinite(off_diagonal_squared):
        raise ValueError("in-plane reciprocal metric is not certifiable in float64")
    determinant_lower = float(
        np.nextafter(
            np.nextafter(diagonal_product, -np.inf)
            - np.nextafter(off_diagonal_squared, np.inf),
            -np.inf,
        )
    )
    if not np.isfinite(determinant_lower) or determinant_lower <= 0.0:
        raise ValueError("in-plane reciprocal metric must be positive definite in float64")

    diagonal_upper = float(
        np.nextafter(max(float(scaled_metric[0, 0]), float(scaled_metric[1, 1])), np.inf)
    )
    extent_upper = _outward_hk_extent(
        q_max_Ainv,
        diagonal_upper,
        determinant_lower,
        metric_scale_exponent,
    )
    if not np.isfinite(extent_upper):
        raise ValueError("detector-complete rod bound is not finite")
    half_width = math.floor(extent_upper)
    if half_width > np.iinfo(np.int32).max:
        raise ValueError("detector-complete rod bound exceeds RodCatalog integer capacity")
    side = 2 * half_width + 1
    rod_count = side * side
    catalog_capacity = min(np.iinfo(np.intp).max, np.iinfo(np.int64).max)
    if rod_count > catalog_capacity:
        raise ValueError("detector-complete rod catalog exceeds representational capacity")
    bounds = (-half_width, half_width)
    return bounds, bounds


def _intersection_branch_ids(rods: RodCatalog, event_build: EventBuildResult) -> BranchArray:
    events = event_build.events
    status = event_build.status
    h = rods.h.astype(np.int64)
    k = rods.k.astype(np.int64)
    maximum_index = max(
        int(np.max(np.abs(h), initial=0)),
        int(np.max(np.abs(k), initial=0)),
    )
    if maximum_index > math.isqrt(np.iinfo(np.int64).max // 3):
        raise ValueError("Miller indices exceed int64 m capacity")
    m = h * h + h * k + k * k
    for family_key, family_m in zip(rods.family_key, m, strict=True):
        if family_key != f"hex:m={int(family_m)}":
            raise ValueError("intersection branches require exact hexagonal family_key metadata")
    rod_order = np.argsort(rods.rod_id)
    sorted_rod_id = rods.rod_id[rod_order]

    branch = np.full(events.event_id.size, -1, dtype=np.int8)
    event_cursor = 0
    event_count = events.event_id.size
    for attempt_row in range(status.attempt_id.size):
        emitted = int(status.emitted_root_count[attempt_row])
        event_stop = event_cursor + emitted
        if event_stop > event_count:
            raise ValueError("analytic root counts exceed the emitted event batch")
        state_id = int(status.incident_state_id[attempt_row])
        rod_id = int(status.rod_id[attempt_row])
        orientation_id = int(status.orientation_id[attempt_row])
        event_slice = slice(event_cursor, event_stop)
        if not (
            np.all(events.incident_state_id[event_slice] == state_id)
            and np.all(events.rod_id[event_slice] == rod_id)
            and np.all(events.orientation_id[event_slice] == orientation_id)
        ):
            raise ValueError("emitted events do not match authoritative analytic attempt order")

        rod_position = int(np.searchsorted(sorted_rod_id, rod_id))
        if rod_position >= sorted_rod_id.size or int(sorted_rod_id[rod_position]) != rod_id:
            raise ValueError(f"analytic attempt references unknown rod_id {rod_id}")
        rod_m = int(m[int(rod_order[rod_position])])
        direct = int(status.direct_beam_root_count[attempt_row])
        root_status = status.root_status[attempt_row]
        if rod_m == 0:
            if root_status is EwaldRootStatus.TANGENT and emitted == 0 and direct == 1:
                pass
            elif root_status is EwaldRootStatus.TWO_ROOT and emitted == 1 and direct == 1:
                branch[event_cursor] = 0
            else:
                raise ValueError(f"invalid m=0 root classification for attempt {attempt_row}")
        elif root_status in (EwaldRootStatus.NO_ROOT, EwaldRootStatus.TANGENT):
            if emitted != 0 or direct != 0:
                raise ValueError(f"invalid m!=0 root classification for attempt {attempt_row}")
        elif root_status is EwaldRootStatus.TWO_ROOT and emitted == 2 and direct == 0:
            first_l = float(events.l_coordinate[event_cursor])
            second_l = float(events.l_coordinate[event_cursor + 1])
            if first_l < second_l:
                branch[event_cursor : event_cursor + 2] = (1, 2)
            elif second_l < first_l:
                branch[event_cursor : event_cursor + 2] = (2, 1)
            else:
                raise ValueError(f"ambiguous signed-L roots for analytic attempt {attempt_row}")
        else:
            raise ValueError(f"invalid m!=0 root classification for attempt {attempt_row}")
        event_cursor = event_stop

    if event_cursor != event_count:
        raise ValueError("emitted event batch has no matching analytic attempt status")
    if np.any(branch < 0):
        raise ValueError("every emitted event must receive one intersection branch identity")
    branch.setflags(write=False)
    return branch


def _detector_complete_rods(
    crystal: CrystalStructure,
    incident_states: IncidentStateBatch,
) -> RodCatalog:
    lattice = ReciprocalLattice.from_crystal(crystal)
    h_bounds, k_bounds = _symmetric_hk_bounds(
        incident_states,
        lattice.inplane_metric_Ainv2,
    )
    return build_rod_catalog(crystal, h_bounds=h_bounds, k_bounds=k_bounds)


def _slice_rods(rods: RodCatalog, start: int, stop: int) -> RodCatalog:
    rows = slice(start, stop)
    return RodCatalog(
        rod_id=rods.rod_id[rows],
        phase_id=rods.phase_id[rows],
        h=rods.h[rows],
        k=rods.k[rows],
        family_id=rods.family_id[rows],
        family_key=rods.family_key[rows],
        qr_Ainv=rods.qr_Ainv[rows],
        reciprocal_basis_Ainv=rods.reciprocal_basis_Ainv,
        symmetry_metadata=rods.symmetry_metadata[rows],
    )


def _build_intersection_support_for_rods(
    *,
    incident_samples: IncidentSampleBatch,
    incident_states: IncidentStateBatch,
    rods: RodCatalog,
    orientations: MosaicOrientationBatch,
    sample_from_crystal: RigidTransform,
) -> IntersectionSupport:
    event_build = build_scattering_events(
        incident_samples=incident_samples,
        incident_states=incident_states,
        rods=rods,
        orientations=orientations,
        sample_from_crystal=sample_from_crystal,
    )
    return IntersectionSupport(
        rods=rods,
        event_build=event_build,
        intersection_branch_id=_intersection_branch_ids(rods, event_build),
    )


def build_intersection_support(
    *,
    crystal: CrystalStructure,
    incident_samples: IncidentSampleBatch,
    incident_states: IncidentStateBatch,
    orientations: MosaicOrientationBatch,
    sample_from_crystal: RigidTransform,
) -> IntersectionSupport:
    """Build all dynamically reachable individual rods and their branch-labelled events."""

    rods = _detector_complete_rods(crystal, incident_states)
    return _build_intersection_support_for_rods(
        incident_samples=incident_samples,
        incident_states=incident_states,
        rods=rods,
        orientations=orientations,
        sample_from_crystal=sample_from_crystal,
    )
