"""Stable assembly of reciprocal-space scattering events."""

from __future__ import annotations

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
from rasim_next.reciprocal.ewald import EwaldRootStatus, solve_continuous_rod_ewald
from rasim_next.sampling.mosaic import MosaicOrientationBatch
from rasim_next.sampling.source import UNITY_SCALAR_POLARIZATION

IntArray = NDArray[np.int64]
ByteArray = NDArray[np.int8]


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
        statuses = tuple(EwaldRootStatus(status) for status in self.root_status)
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


@dataclass(frozen=True, slots=True)
class EventBuildResult:
    events: ScatteringEventBatch
    status: EwaldStatusBatch


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
    if any(
        state_id != UNITY_SCALAR_POLARIZATION for state_id in incident_samples.polarization_state_id
    ):
        raise ValueError('incident samples must declare polarization_state_id="unity_scalar"')
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
    q0_crystal = rods.h[:, None] * basis[:, 0][None, :] + rods.k[:, None] * basis[:, 1][None, :]
    d_hat_sample = (orientations.rotation_crystal @ b3_hat_crystal) @ crystal_to_sample_rotation.T

    event_capacity = (
        2
        * int(np.count_nonzero(incident_states.valid))
        * rods.rod_id.size
        * int(np.count_nonzero(orientations.probability_mass > 0.0))
    )
    event_state_id = np.empty(event_capacity, dtype=np.int64)
    event_rod_id = np.empty(event_capacity, dtype=np.int64)
    event_wavelength = np.empty(event_capacity, dtype=np.float64)
    event_q = np.empty((event_capacity, 3), dtype=np.float64)
    event_l = np.empty(event_capacity, dtype=np.float64)
    event_kf = np.empty((event_capacity, 3), dtype=np.float64)
    event_weight = np.empty(event_capacity, dtype=np.float64)
    event_residual = np.empty(event_capacity, dtype=np.float64)
    event_count = 0
    attempt_state_id: list[int] = []
    attempt_rod_id: list[int] = []
    attempt_orientation_id: list[int] = []
    statuses: list[EwaldRootStatus] = []
    emitted_counts: list[int] = []
    direct_counts: list[int] = []

    for state_id_value, sample_id_value, incident, valid in zip(
        incident_states.incident_state_id,
        incident_states.incident_sample_id,
        incident_states.k_film_phase_sample_Ainv,
        incident_states.valid,
        strict=True,
    ):
        if not valid:
            continue
        state_id = int(state_id_value)
        wavelength = sample_wavelength[int(sample_id_value)]
        for rod_id_value, rod_q0_crystal in zip(rods.rod_id, q0_crystal, strict=True):
            rod_id = int(rod_id_value)
            for orientation_id_value, orientation_mass_value, rotation, direction in zip(
                orientations.orientation_id,
                orientations.probability_mass,
                orientations.rotation_crystal,
                d_hat_sample,
                strict=True,
            ):
                orientation_mass = float(orientation_mass_value)
                if orientation_mass == 0.0:
                    continue
                q0_sample = crystal_to_sample_rotation @ (rotation @ rod_q0_crystal)
                roots = solve_continuous_rod_ewald(
                    ki_sample_Ainv=incident,
                    q0_sample_Ainv=q0_sample,
                    d_hat_sample=direction,
                    b3_norm_Ainv=b3_norm,
                )
                attempt_state_id.append(state_id)
                attempt_rod_id.append(rod_id)
                attempt_orientation_id.append(int(orientation_id_value))
                statuses.append(roots.status)
                emitted_counts.append(len(roots.emittable_roots))
                direct_counts.append(roots.direct_beam_root_count)
                for root in roots.emittable_roots:
                    event_state_id[event_count] = state_id
                    event_rod_id[event_count] = rod_id
                    event_wavelength[event_count] = wavelength
                    event_q[event_count] = root.q_sample_Ainv
                    event_l[event_count] = root.l_coordinate
                    event_kf[event_count] = root.kf_sample_Ainv
                    event_weight[event_count] = orientation_mass * root.coarea_jacobian
                    event_residual[event_count] = root.ewald_residual_Ainv
                    event_count += 1

    event_q_array = event_q[:event_count]
    events = ScatteringEventBatch(
        event_id=np.arange(event_count, dtype=np.int64),
        incident_state_id=event_state_id[:event_count],
        rod_id=event_rod_id[:event_count],
        wavelength_A=event_wavelength[:event_count],
        q_internal_sample_Ainv=event_q_array,
        qz_Ainv=event_q_array[:, 2],
        l_coordinate=event_l[:event_count],
        kf_film_phase_sample_Ainv=event_kf[:event_count],
        reciprocal_weight=event_weight[:event_count],
        ewald_residual_Ainv=event_residual[:event_count],
        valid=np.ones(event_count, dtype=np.bool_),
    )
    attempt_count = len(statuses)
    status = EwaldStatusBatch(
        attempt_id=np.arange(attempt_count, dtype=np.int64),
        incident_state_id=np.asarray(attempt_state_id, dtype=np.int64),
        rod_id=np.asarray(attempt_rod_id, dtype=np.int64),
        orientation_id=np.asarray(attempt_orientation_id, dtype=np.int64),
        root_status=tuple(statuses),
        emitted_root_count=np.asarray(emitted_counts, dtype=np.int8),
        direct_beam_root_count=np.asarray(direct_counts, dtype=np.int8),
    )
    return EventBuildResult(events=events, status=status)
