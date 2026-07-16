"""Stable assembly of reciprocal-space scattering events."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import (
    IncidentSampleBatch,
    IncidentStateBatch,
    RodCatalog,
    ScatteringEventBatch,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.validity import ValidityCode
from rasim_next.reciprocal.ewald import EwaldRootStatus, solve_continuous_rod_ewald
from rasim_next.sampling.mosaic import MosaicOrientationBatch

IntArray = NDArray[np.int64]
ByteArray = NDArray[np.int8]
FloatArray = NDArray[np.float64]
_AttemptContext = tuple[int, float, int, int, float, FloatArray, FloatArray, FloatArray]


def _integer_array(value: ArrayLike, dtype: np.dtype, size: int, name: str) -> NDArray[np.integer]:
    supplied = np.asarray(value)
    if supplied.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    array = np.array(value, dtype=dtype, copy=True)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class EwaldStatusBatch:
    """One explicit root classification for each valid state/rod/orientation attempt."""

    attempt_id: IntArray
    incident_state_id: IntArray
    rod_id: IntArray
    orientation_id: IntArray
    root_status: tuple[EwaldRootStatus, ...]
    emitted_root_count: ByteArray
    direct_beam_root_count: ByteArray

    def __post_init__(self) -> None:
        supplied_ids = np.asarray(self.attempt_id)
        if supplied_ids.ndim != 1:
            raise ValueError("attempt_id must be one-dimensional")
        ids = _integer_array(self.attempt_id, np.dtype(np.int64), supplied_ids.size, "attempt_id")
        size = ids.size
        if np.any(ids < 0) or np.unique(ids).size != size:
            raise ValueError("attempt_id must be nonnegative and unique")
        object.__setattr__(self, "attempt_id", ids)
        for name, dtype in (
            ("incident_state_id", np.dtype(np.int64)),
            ("rod_id", np.dtype(np.int64)),
            ("orientation_id", np.dtype(np.int64)),
            ("emitted_root_count", np.dtype(np.int8)),
            ("direct_beam_root_count", np.dtype(np.int8)),
        ):
            object.__setattr__(self, name, _integer_array(getattr(self, name), dtype, size, name))
        self._validate_statuses(size)

    def _validate_statuses(self, size: int) -> None:
        statuses = self.root_status
        if not (
            isinstance(statuses, tuple)
            and all(isinstance(status, EwaldRootStatus) for status in statuses)
        ):
            statuses = tuple(EwaldRootStatus(status) for status in statuses)
        if len(statuses) != size:
            raise ValueError("root_status must align with attempt_id")
        if np.any(self.emitted_root_count < 0) or np.any(self.direct_beam_root_count < 0):
            raise ValueError("root counts must be nonnegative")
        for status, emitted, direct in zip(
            statuses,
            self.emitted_root_count,
            self.direct_beam_root_count,
            strict=True,
        ):
            classified = int(emitted) + int(direct)
            if (
                (status is EwaldRootStatus.NO_ROOT and classified != 0)
                or (status is EwaldRootStatus.TANGENT and (emitted != 0 or direct > 1))
                or (status is EwaldRootStatus.TWO_ROOT and classified != 2)
            ):
                raise ValueError("root counts are inconsistent with root_status")
        object.__setattr__(self, "root_status", statuses)

    @classmethod
    def _from_builder_owned_arrays(
        cls,
        *,
        attempt_id: IntArray,
        incident_state_id: IntArray,
        rod_id: IntArray,
        orientation_id: IntArray,
        root_status: tuple[EwaldRootStatus, ...],
        emitted_root_count: ByteArray,
        direct_beam_root_count: ByteArray,
    ) -> EwaldStatusBatch:
        """Adopt already validated builder arrays without attempt-sized copies."""

        batch = object.__new__(cls)
        arrays = {
            "attempt_id": (attempt_id, np.dtype(np.int64)),
            "incident_state_id": (incident_state_id, np.dtype(np.int64)),
            "rod_id": (rod_id, np.dtype(np.int64)),
            "orientation_id": (orientation_id, np.dtype(np.int64)),
            "emitted_root_count": (emitted_root_count, np.dtype(np.int8)),
            "direct_beam_root_count": (direct_beam_root_count, np.dtype(np.int8)),
        }
        size = attempt_id.size
        for name, (array, dtype) in arrays.items():
            if array.shape != (size,) or array.dtype != dtype or not array.flags.owndata:
                raise ValueError(f"builder-owned {name} has invalid storage")
            array.setflags(write=False)
            object.__setattr__(batch, name, array)
        object.__setattr__(batch, "root_status", root_status)
        batch._validate_statuses(size)
        return batch


@dataclass(frozen=True, slots=True)
class EventBuildResult:
    events: ScatteringEventBatch
    status: EwaldStatusBatch


def _iter_attempt_contexts(
    *,
    incident_states: IncidentStateBatch,
    sample_wavelength: dict[int, float],
    rods: RodCatalog,
    basis_b1_crystal: FloatArray,
    basis_b2_crystal: FloatArray,
    b3_hat_crystal: FloatArray,
    orientations: MosaicOrientationBatch,
    crystal_to_sample_rotation: FloatArray,
) -> Iterator[_AttemptContext]:
    for state_index, valid in enumerate(incident_states.valid):
        if not valid:
            continue
        state_id = int(incident_states.incident_state_id[state_index])
        sample_id = int(incident_states.incident_sample_id[state_index])
        incident = incident_states.k_film_phase_sample_Ainv[state_index]
        wavelength = sample_wavelength[sample_id]
        for rod_index, rod_id_value in enumerate(rods.rod_id):
            rod_id = int(rod_id_value)
            rod_q0_crystal = (
                int(rods.h[rod_index]) * basis_b1_crystal
                + int(rods.k[rod_index]) * basis_b2_crystal
            )
            for orientation_index, orientation_mass_value in enumerate(
                orientations.probability_mass
            ):
                orientation_mass = float(orientation_mass_value)
                if orientation_mass == 0.0:
                    continue
                rotation = orientations.rotation_crystal[orientation_index]
                q0_sample = crystal_to_sample_rotation @ (rotation @ rod_q0_crystal)
                direction = (rotation @ b3_hat_crystal) @ crystal_to_sample_rotation.T
                yield (
                    state_id,
                    wavelength,
                    rod_id,
                    int(orientations.orientation_id[orientation_index]),
                    orientation_mass,
                    incident,
                    q0_sample,
                    direction,
                )


def _classify_attempts(
    contexts: Iterable[_AttemptContext],
    *,
    attempt_count: int,
    b3_norm_Ainv: float,
) -> tuple[IntArray, IntArray, IntArray, ByteArray, ByteArray, tuple[EwaldRootStatus, ...]]:
    incident_state_id = np.empty(attempt_count, dtype=np.int64)
    rod_id = np.empty(attempt_count, dtype=np.int64)
    orientation_id = np.empty(attempt_count, dtype=np.int64)
    emitted_root_count = np.empty(attempt_count, dtype=np.int8)
    direct_beam_root_count = np.empty(attempt_count, dtype=np.int8)

    def statuses() -> Iterator[EwaldRootStatus]:
        for attempt_index, context in enumerate(contexts):
            (
                state_id,
                _,
                attempt_rod_id,
                attempt_orientation_id,
                _,
                incident,
                q0,
                direction,
            ) = context
            roots = solve_continuous_rod_ewald(
                ki_sample_Ainv=incident,
                q0_sample_Ainv=q0,
                d_hat_sample=direction,
                b3_norm_Ainv=b3_norm_Ainv,
            )
            incident_state_id[attempt_index] = state_id
            rod_id[attempt_index] = attempt_rod_id
            orientation_id[attempt_index] = attempt_orientation_id
            emitted_root_count[attempt_index] = len(roots.emittable_roots)
            direct_beam_root_count[attempt_index] = roots.direct_beam_root_count
            yield roots.status

    return (
        incident_state_id,
        rod_id,
        orientation_id,
        emitted_root_count,
        direct_beam_root_count,
        tuple(statuses()),
    )


def build_scattering_events(
    *,
    incident_samples: IncidentSampleBatch,
    incident_states: IncidentStateBatch,
    rods: RodCatalog,
    orientations: MosaicOrientationBatch,
    sample_from_crystal: RigidTransform,
) -> EventBuildResult:
    """Build elastic events; reciprocal weight is orientation mass times coarea Jacobian."""

    if (
        sample_from_crystal.source_frame is not FrameId.CRYSTAL
        or sample_from_crystal.target_frame is not FrameId.SAMPLE
    ):
        raise ValueError("sample_from_crystal must map CRYSTAL to SAMPLE")
    if not np.array_equal(orientations.reciprocal_basis_Ainv, rods.reciprocal_basis_Ainv):
        raise ValueError("orientations and rods must use the same reciprocal basis")

    sample_wavelength = {
        int(sample_id): float(wavelength)
        for sample_id, wavelength in zip(
            incident_samples.incident_sample_id,
            incident_samples.wavelength_A,
            strict=True,
        )
    }
    missing = sorted(set(map(int, incident_states.incident_sample_id)) - sample_wavelength.keys())
    if missing:
        raise ValueError(f"incident states reference unknown incident_sample_id values: {missing}")

    basis = rods.reciprocal_basis_Ainv
    crystal_to_sample_rotation = sample_from_crystal.rotation
    b3_norm = float(np.linalg.norm(basis[:, 2]))
    b3_hat_crystal = basis[:, 2] / b3_norm
    attempt_count = int(
        np.count_nonzero(incident_states.valid)
        * rods.rod_id.size
        * np.count_nonzero(orientations.probability_mass > 0.0)
    )
    (
        attempt_state_id,
        attempt_rod_id,
        attempt_orientation_id,
        emitted_counts,
        direct_counts,
        root_status,
    ) = _classify_attempts(
        _iter_attempt_contexts(
            incident_states=incident_states,
            sample_wavelength=sample_wavelength,
            rods=rods,
            basis_b1_crystal=basis[:, 0],
            basis_b2_crystal=basis[:, 1],
            b3_hat_crystal=b3_hat_crystal,
            orientations=orientations,
            crystal_to_sample_rotation=crystal_to_sample_rotation,
        ),
        attempt_count=attempt_count,
        b3_norm_Ainv=b3_norm,
    )
    event_count = int(emitted_counts.sum(dtype=np.int64))
    status = EwaldStatusBatch._from_builder_owned_arrays(
        attempt_id=np.arange(attempt_count, dtype=np.int64),
        incident_state_id=attempt_state_id,
        rod_id=attempt_rod_id,
        orientation_id=attempt_orientation_id,
        root_status=root_status,
        emitted_root_count=emitted_counts,
        direct_beam_root_count=direct_counts,
    )

    event_state_id = np.empty(event_count, dtype=np.int64)
    event_orientation_id = np.empty(event_count, dtype=np.int64)
    event_rod_id = np.empty(event_count, dtype=np.int64)
    event_wavelength = np.empty(event_count, dtype=np.float64)
    event_q = np.empty((event_count, 3), dtype=np.float64)
    event_l = np.empty(event_count, dtype=np.float64)
    event_kf = np.empty((event_count, 3), dtype=np.float64)
    event_weight = np.empty(event_count, dtype=np.float64)
    event_residual = np.empty(event_count, dtype=np.float64)
    next_event = 0
    contexts = _iter_attempt_contexts(
        incident_states=incident_states,
        sample_wavelength=sample_wavelength,
        rods=rods,
        basis_b1_crystal=basis[:, 0],
        basis_b2_crystal=basis[:, 1],
        b3_hat_crystal=b3_hat_crystal,
        orientations=orientations,
        crystal_to_sample_rotation=crystal_to_sample_rotation,
    )
    for attempt_index, context in enumerate(contexts):
        emitted_count = int(emitted_counts[attempt_index])
        if emitted_count == 0:
            continue
        state_id, wavelength, rod_id, orientation_id, orientation_mass, incident, q0, direction = (
            context
        )
        roots = solve_continuous_rod_ewald(
            ki_sample_Ainv=incident,
            q0_sample_Ainv=q0,
            d_hat_sample=direction,
            b3_norm_Ainv=b3_norm,
        )
        if len(roots.emittable_roots) != emitted_count:
            raise RuntimeError("Ewald root count changed between count and fill passes")
        for root in roots.emittable_roots:
            event_state_id[next_event] = state_id
            event_orientation_id[next_event] = orientation_id
            event_rod_id[next_event] = rod_id
            event_wavelength[next_event] = wavelength
            event_q[next_event] = root.q_sample_Ainv
            event_l[next_event] = root.l_coordinate
            event_kf[next_event] = root.kf_sample_Ainv
            event_weight[next_event] = orientation_mass * root.coarea_jacobian
            event_residual[next_event] = root.ewald_residual_Ainv
            next_event += 1
    if next_event != event_count:
        raise RuntimeError("Ewald event fill count does not match classified roots")

    events = ScatteringEventBatch(
        event_id=np.arange(event_count, dtype=np.int64),
        incident_state_id=event_state_id,
        orientation_id=event_orientation_id,
        rod_id=event_rod_id,
        wavelength_A=event_wavelength,
        q_internal_sample_Ainv=event_q,
        q_sample_normal_Ainv=event_q[:, 2],
        l_coordinate=event_l,
        kf_film_phase_sample_Ainv=event_kf,
        reciprocal_weight=event_weight,
        ewald_residual_Ainv=event_residual,
        status=(ValidityCode.VALID,) * event_count,
        valid=np.ones(event_count, dtype=np.bool_),
    )
    return EventBuildResult(events=events, status=status)
