from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from rasim_next.core.contracts import (
    IncidentSampleBatch,
    IncidentStateBatch,
    MaterialOptics,
    RodQueryBatch,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.validity import ValidityCode
from rasim_next.geometry.instrument import InstrumentConfiguration, compile_instrument
from rasim_next.geometry.transport import IncidentTransportResult, transport_scattering_events
from rasim_next.materials.crystal import CrystalSite, CrystalStructure
from rasim_next.ordered.amplitudes import ordered_event_result
from rasim_next.pipeline.intersections import build_intersection_support
from rasim_next.pipeline.selection import build_candidate_pool, select_candidates
from rasim_next.pipeline.simulate import simulate_ordered
from rasim_next.reciprocal.events import build_scattering_events
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.reciprocal.rods import build_rod_catalog
from rasim_next.render.deposition import deposit_bilinear
from rasim_next.sampling.mosaic import MosaicOrientationBatch


def _fixture() -> dict[str, object]:
    reciprocal_basis = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.0, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    direct_basis = 2.0 * np.pi * np.linalg.inv(reciprocal_basis).T
    crystal = CrystalStructure(
        phase_id="hexagonal-fixture",
        spacegroup_hm="P 1",
        direct_basis_A=direct_basis,
        volume_A3=float(np.linalg.det(direct_basis)),
        sites=(
            CrystalSite(
                source_label="X1",
                species="C",
                element="C",
                charge=0,
                occupancy=1.0,
                fractional=(0.0, 0.0, 0.0),
                u_iso_A2=0.0,
                source_multiplicity=1,
            ),
        ),
        source_path=Path("synthetic-integration.cif"),
        provenance="compact integration fixture",
    )
    reciprocal_basis = ReciprocalLattice.from_crystal(crystal).basis_Ainv
    wavelength_A = np.pi / 2.0
    incident_wavevector = np.array([-1.0, np.sqrt(14.75), 0.5])
    incident_direction = incident_wavevector / 4.0
    samples = IncidentSampleBatch(
        incident_sample_id=np.array([3]),
        origin_lab_m=np.zeros((1, 3)),
        direction_lab=incident_direction[None, :],
        wavelength_A=np.array([wavelength_A]),
        source_weight=np.ones(1),
        polarization_state_id=("UNITY_APPROXIMATION",),
        correlation_model="explicit_fixture",
    )
    states = IncidentStateBatch(
        incident_state_id=np.array([3]),
        incident_sample_id=np.array([3]),
        sample_intersection_lab_m=np.zeros((1, 3)),
        direction_sample=incident_direction[None, :],
        k_air_sample_Ainv=incident_wavevector[None, :],
        k_film_phase_sample_Ainv=incident_wavevector[None, :],
        kz_film_Ainv=np.array([0.5 + 0.0j]),
        entrance_amplitude=np.ones(1, dtype=np.complex128),
        footprint_acceptance=np.ones(1),
        source_weight=np.ones(1),
        valid=np.ones(1, dtype=np.bool_),
    )
    incident = IncidentTransportResult(
        states=states,
        status=(ValidityCode.VALID,),
        wavelength_A=np.array([wavelength_A]),
    )
    identity = np.eye(3)
    zero = np.zeros(3)
    detector_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    instrument = compile_instrument(
        InstrumentConfiguration(
            axis_rotations=(),
            lab_from_goniometer_zero=RigidTransform(
                identity, zero, FrameId.GONIOMETER, FrameId.LAB
            ),
            goniometer_from_sample=RigidTransform(
                identity, zero, FrameId.SAMPLE, FrameId.GONIOMETER
            ),
            sample_from_crystal=RigidTransform(identity, zero, FrameId.CRYSTAL, FrameId.SAMPLE),
            lab_from_detector=RigidTransform(
                detector_rotation, np.array([0.0, 1.0, 0.0]), FrameId.DETECTOR, FrameId.LAB
            ),
            detector_shape_rc=(7, 7),
            detector_row_pitch_m=0.1,
            detector_column_pitch_m=0.1,
            detector_reference_coordinate_px=(3.0, 4.5),
            sample_width_m=1.0,
            sample_length_m=1.0,
            film_thickness_A=0.0,
        )
    )
    material = MaterialOptics(
        material_id="vacuum",
        wavelength_A=np.array([wavelength_A]),
        n_complex=np.ones(1, dtype=np.complex128),
        delta=np.zeros(1),
        beta=np.zeros(1),
        mu_Ainv=np.zeros(1),
        provenance="analytic vacuum fixture",
    )
    orientations = MosaicOrientationBatch(
        orientation_id=np.array([7]),
        alpha_rad=np.zeros(1),
        azimuth_rad=np.zeros(1),
        rotation_crystal=identity[None, :, :],
        probability_mass=np.ones(1),
        reciprocal_basis_Ainv=reciprocal_basis,
        model_id="manuscript_axisymmetric_v1",
    )
    return {
        "crystal": crystal,
        "samples": samples,
        "states": states,
        "incident": incident,
        "instrument": instrument,
        "material": material,
        "orientations": orientations,
    }


def _event_rows(
    rods: object, event_build: object, transported: object
) -> dict[tuple[object, ...], tuple[object, ...]]:
    events = event_build.events
    hits = transported.detector_hits
    row_by_rod = {int(rod_id): row for row, rod_id in enumerate(rods.rod_id)}
    rows: dict[tuple[object, ...], tuple[object, ...]] = {}
    for event_row, rod_id in enumerate(events.rod_id):
        rod_row = row_by_rod[int(rod_id)]
        key = (
            rods.phase_id[rod_row],
            int(rods.h[rod_row]),
            int(rods.k[rod_row]),
            int(events.incident_state_id[event_row]),
            int(events.orientation_id[event_row]),
            float(events.l_coordinate[event_row]).hex(),
        )
        if key in rows:
            raise AssertionError(f"duplicate physical event key: {key}")
        rows[key] = (
            events.q_internal_sample_Ainv[event_row],
            events.kf_film_phase_sample_Ainv[event_row],
            hits.column_px[event_row],
            hits.row_px[event_row],
            transported.detector_status[event_row],
        )
    return rows


def test_dynamic_all_rod_support_matches_oversized_detector_oracle() -> None:
    fixture = _fixture()
    support = build_intersection_support(
        crystal=fixture["crystal"],
        incident_samples=fixture["samples"],
        incident_states=fixture["states"],
        orientations=fixture["orientations"],
        sample_from_crystal=fixture["instrument"].sample_from_crystal,
    )
    bounds = max(int(np.max(np.abs(support.rods.h))), int(np.max(np.abs(support.rods.k))))
    assert bounds == 9
    actual_hk = {
        (int(h_value), int(k_value))
        for h_value, k_value in zip(support.rods.h, support.rods.k, strict=True)
    }
    expected_hk = {(h_value, k_value) for h_value in range(-9, 10) for k_value in range(-9, 10)}
    assert actual_hk == expected_hk
    assert support.rods.rod_id.size == 19 * 19
    assert np.any((support.rods.h == 9) & (support.rods.k == -4))
    oracle_half_width = 11
    oracle_rods = build_rod_catalog(
        fixture["crystal"],
        h_bounds=(-oracle_half_width, oracle_half_width),
        k_bounds=(-oracle_half_width, oracle_half_width),
    )
    assert oracle_rods.rod_id.size > support.rods.rod_id.size
    oracle_events = build_scattering_events(
        incident_samples=fixture["samples"],
        incident_states=fixture["states"],
        rods=oracle_rods,
        orientations=fixture["orientations"],
        sample_from_crystal=fixture["instrument"].sample_from_crystal,
    )
    transported = transport_scattering_events(
        support.event_build.events,
        fixture["incident"],
        fixture["material"],
        fixture["instrument"],
    )
    oracle_transport = transport_scattering_events(
        oracle_events.events,
        fixture["incident"],
        fixture["material"],
        fixture["instrument"],
    )
    actual_rows = _event_rows(support.rods, support.event_build, transported)
    oracle_rows = _event_rows(oracle_rods, oracle_events, oracle_transport)
    assert actual_rows.keys() == oracle_rows.keys()
    for key in actual_rows:
        actual = actual_rows[key]
        expected = oracle_rows[key]
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[2:] == expected[2:]


def test_branches_use_signed_l_before_detector_clipping() -> None:
    fixture = _fixture()
    polarization_provenance = "explicit unity approximation for a fixture without polarization data"
    simulation = simulate_ordered(
        crystal=fixture["crystal"],
        incident_samples=fixture["samples"],
        material=fixture["material"],
        instrument=fixture["instrument"],
        orientations=fixture["orientations"],
        phase_population_weight=0.5,
        polarization_policy_id="UNITY_APPROXIMATION",
        polarization_provenance=polarization_provenance,
        selection_seed=20260716,
        draw_count=40,
    )
    support = build_intersection_support(
        crystal=fixture["crystal"],
        incident_samples=fixture["samples"],
        incident_states=simulation.incident.states,
        orientations=fixture["orientations"],
        sample_from_crystal=fixture["instrument"].sample_from_crystal,
    )
    events = support.event_build.events
    rod_order = np.argsort(support.rods.rod_id)
    sorted_rod_id = support.rods.rod_id[rod_order]
    rod_position = np.searchsorted(sorted_rod_id, events.rod_id)
    assert np.array_equal(sorted_rod_id[rod_position], events.rod_id)
    rod_row = rod_order[rod_position]
    query = RodQueryBatch(
        event_id=events.event_id,
        rod_id=events.rod_id,
        phase_id=tuple(support.rods.phase_id[int(row)] for row in rod_row),
        h=support.rods.h[rod_row],
        k=support.rods.k[rod_row],
        q_sample_normal_Ainv=events.q_sample_normal_Ainv,
        l_coordinate=events.l_coordinate,
        wavelength_A=events.wavelength_A,
    )
    intensity = ordered_event_result(fixture["crystal"], support.rods, query)
    transported = transport_scattering_events(
        events,
        simulation.incident,
        fixture["material"],
        fixture["instrument"],
    )
    states = simulation.incident.states
    event_count = events.event_id.size
    population_weight = np.full(event_count, simulation.phase_population_weight)
    polarization_weight = np.ones(event_count)
    pool = build_candidate_pool(
        support=support,
        incident_states=states,
        intensity=intensity,
        transport=transported,
        population_event_id=events.event_id,
        population_weight=population_weight,
        polarization_event_id=events.event_id,
        polarization_weight=polarization_weight,
    )
    oracle_selection = select_candidates(pool, seed=20260716, draw_count=40)

    assert (
        simulation.phase_population_weight,
        simulation.polarization_policy_id,
        simulation.polarization_provenance,
    ) == (0.5, "UNITY_APPROXIMATION", polarization_provenance)
    np.testing.assert_array_equal(
        np.column_stack((simulation.rods.rod_id, simulation.rods.h, simulation.rods.k)),
        np.column_stack((support.rods.rod_id, support.rods.h, support.rods.k)),
    )

    row_by_rod = {int(rod_id): row for row, rod_id in enumerate(support.rods.rod_id)}
    event_hk = np.array(
        [
            (
                support.rods.h[row_by_rod[int(rod)]],
                support.rods.k[row_by_rod[int(rod)]],
            )
            for rod in events.rod_id
        ]
    )
    selected = np.flatnonzero(
        ((event_hk[:, 0] == 0) & (event_hk[:, 1] == 0))
        | ((event_hk[:, 0] == 1) & (event_hk[:, 1] == 0))
    )
    selected = selected[np.lexsort((events.l_coordinate[selected], event_hk[selected, 0]))]
    np.testing.assert_allclose(
        events.l_coordinate[selected],
        [-1.0, -0.5 - np.sqrt(1.25), -0.5 + np.sqrt(1.25)],
        rtol=0.0,
        atol=4.0e-15,
    )
    np.testing.assert_array_equal(support.intersection_branch_id[selected], [0, 1, 2])
    np.testing.assert_array_equal(transported.detector_hits.valid[selected], [True, False, True])
    np.testing.assert_array_equal(
        support.intersection_branch_id[selected][transported.detector_hits.valid[selected]],
        [0, 2],
    )
    assert np.any(events.l_coordinate < 0.0)

    state_by_id = {
        int(state_id): row for row, state_id in enumerate(states.incident_state_id)
    }
    state_row = np.asarray(
        [state_by_id[int(state_id)] for state_id in events.incident_state_id],
        dtype=np.intp,
    )
    expected_mass = (
        states.source_weight[state_row]
        * events.reciprocal_weight
        * population_weight
        * intensity.scattering_strength_A2
        * transported.outgoing_waves.optical_weight
        * states.footprint_acceptance[state_row]
        * polarization_weight
    )
    np.testing.assert_allclose(pool.candidate_mass_A2, expected_mass[pool.event_row])

    selected_fields = (
        "event_id",
        "incident_state_id",
        "intersection_branch_id",
        "assigned_mass_A2",
    )
    for field in selected_fields:
        np.testing.assert_array_equal(
            getattr(simulation.selection, field),
            getattr(oracle_selection, field),
        )

    detector_rows = pool.event_row
    positive = pool.candidate_mass_A2 > 0.0
    pool_branch = support.intersection_branch_id[detector_rows]
    branch_mass = np.array(
        [
            np.sum(pool.candidate_mass_A2[pool_branch == branch_id], dtype=np.float64)
            for branch_id in range(3)
        ]
    )
    total_mass = float(np.sum(pool.candidate_mass_A2, dtype=np.float64))
    summary = simulation.mass_summary
    np.testing.assert_array_equal(summary.incident_state_id, [3])
    np.testing.assert_array_equal(summary.detector_valid_count, [detector_rows.size])
    np.testing.assert_array_equal(summary.positive_count, [np.count_nonzero(positive)])
    np.testing.assert_allclose(summary.total_mass_A2, [total_mass], rtol=2.0e-15)
    np.testing.assert_allclose(summary.branch_mass_A2, branch_mass[None, :], rtol=2.0e-15)
    assert summary.analytic_attempt_count == support.event_build.status.attempt_id.size
    assert summary.emitted_event_count == events.event_id.size

    compact_events = simulation.selected_events
    compact_transport = simulation.selected_transport
    compact_branch = simulation.selected_intersection_branch_id
    selection = simulation.selection
    unique_event_id = np.unique(selection.event_id)
    assert compact_events.event_id.size == unique_event_id.size
    assert unique_event_id.size == np.unique(selection.candidate_row).size < selection.event_id.size
    np.testing.assert_array_equal(
        compact_events.event_id[selection.candidate_row],
        selection.event_id,
    )
    np.testing.assert_array_equal(
        compact_branch[selection.candidate_row],
        selection.intersection_branch_id,
    )
    event_order = np.argsort(events.event_id)
    event_position = np.searchsorted(events.event_id[event_order], compact_events.event_id)
    oracle_row = event_order[event_position]
    np.testing.assert_array_equal(
        np.column_stack(
            (
                compact_events.incident_state_id,
                compact_events.rod_id,
                compact_events.orientation_id,
            )
        ),
        np.column_stack(
            (
                events.incident_state_id[oracle_row],
                events.rod_id[oracle_row],
                events.orientation_id[oracle_row],
            )
        ),
    )
    np.testing.assert_array_equal(
        np.column_stack(
            (
                compact_events.q_internal_sample_Ainv,
                compact_events.kf_film_phase_sample_Ainv,
                compact_events.l_coordinate,
            )
        ),
        np.column_stack(
            (
                events.q_internal_sample_Ainv[oracle_row],
                events.kf_film_phase_sample_Ainv[oracle_row],
                events.l_coordinate[oracle_row],
            )
        ),
    )
    np.testing.assert_array_equal(
        np.column_stack(
            (
                compact_transport.detector_hits.event_id,
                compact_transport.detector_hits.column_px,
                compact_transport.detector_hits.row_px,
                compact_transport.detector_hits.valid,
            )
        ),
        np.column_stack(
            (
                transported.detector_hits.event_id[oracle_row],
                transported.detector_hits.column_px[oracle_row],
                transported.detector_hits.row_px[oracle_row],
                transported.detector_hits.valid[oracle_row],
            )
        ),
    )
    np.testing.assert_array_equal(
        compact_transport.outgoing_waves.event_id,
        compact_events.event_id,
    )
    np.testing.assert_allclose(
        simulation.deposition.deposited_mass_A2 + simulation.deposition.clipped_mass_A2,
        np.sum(selection.assigned_mass_A2, dtype=np.float64),
        rtol=2.0e-15,
    )

    replay = simulate_ordered(
        crystal=fixture["crystal"],
        incident_samples=fixture["samples"],
        material=fixture["material"],
        instrument=fixture["instrument"],
        orientations=fixture["orientations"],
        phase_population_weight=0.5,
        polarization_policy_id="UNITY_APPROXIMATION",
        polarization_provenance=polarization_provenance,
        selection_seed=20260716,
        draw_count=40,
    )
    for field in (*selected_fields, "candidate_row"):
        np.testing.assert_array_equal(getattr(replay.selection, field), getattr(selection, field))
    np.testing.assert_array_equal(replay.deposition.image_A2, simulation.deposition.image_A2)

    doubled = simulate_ordered(
        crystal=fixture["crystal"],
        incident_samples=fixture["samples"],
        material=fixture["material"],
        instrument=fixture["instrument"],
        orientations=fixture["orientations"],
        phase_population_weight=1.0,
        polarization_policy_id="UNITY_APPROXIMATION",
        polarization_provenance=polarization_provenance,
        selection_seed=20260716,
        draw_count=40,
    )
    np.testing.assert_array_equal(doubled.selection.event_id, selection.event_id)
    np.testing.assert_array_equal(doubled.mass_summary.total_mass_A2, 2.0 * summary.total_mass_A2)
    np.testing.assert_array_equal(doubled.mass_summary.branch_mass_A2, 2.0 * summary.branch_mass_A2)
    np.testing.assert_array_equal(
        doubled.selection.assigned_mass_A2,
        2.0 * selection.assigned_mass_A2,
    )
    np.testing.assert_allclose(
        doubled.deposition.image_A2,
        2.0 * simulation.deposition.image_A2,
        rtol=2.0e-15,
    )

    changed_solid_angle = replace(
        transported,
        detector_hits=replace(
            transported.detector_hits,
            pixel_solid_angle_sr=transported.detector_hits.pixel_solid_angle_sr + 7.0,
        ),
    )
    solid_angle_pool = build_candidate_pool(
        support=support,
        incident_states=states,
        intensity=intensity,
        transport=changed_solid_angle,
        population_event_id=events.event_id,
        population_weight=population_weight,
        polarization_event_id=events.event_id,
        polarization_weight=polarization_weight,
    )
    np.testing.assert_array_equal(solid_angle_pool.candidate_mass_A2, pool.candidate_mass_A2)
    solid_angle_selection = select_candidates(
        solid_angle_pool,
        seed=20260716,
        draw_count=40,
    )
    for field in selected_fields:
        np.testing.assert_array_equal(
            getattr(solid_angle_selection, field),
            getattr(oracle_selection, field),
        )

    deposition_column = compact_transport.detector_hits.column_px.copy()
    deposition_row = compact_transport.detector_hits.row_px.copy()
    deposition_column[selection.candidate_row] = 1.25
    deposition_row[selection.candidate_row] = 2.5
    edge_event_row = int(np.unique(selection.candidate_row)[0])
    deposition_column[edge_event_row] = -0.5
    deposition_row[edge_event_row] = -0.5
    deposition_hits = replace(
        compact_transport.detector_hits,
        column_px=deposition_column,
        row_px=deposition_row,
    )
    deposited = deposit_bilinear(
        deposition_hits,
        event_row=selection.candidate_row,
        event_id=selection.event_id,
        assigned_mass_A2=selection.assigned_mass_A2,
        detector_shape_rc=(5, 7),
    )
    edge_selected = selection.candidate_row == edge_event_row
    edge_mass = float(np.sum(selection.assigned_mass_A2[edge_selected]))
    interior_mass = float(np.sum(selection.assigned_mass_A2[~edge_selected]))
    expected_image = np.zeros((5, 7))
    expected_image[0, 0] = 0.25 * edge_mass
    expected_image[2, 1] = 0.375 * interior_mass
    expected_image[2, 2] = 0.125 * interior_mass
    expected_image[3, 1] = 0.375 * interior_mass
    expected_image[3, 2] = 0.125 * interior_mass
    np.testing.assert_allclose(deposited.image_A2, expected_image, rtol=2.0e-15)
    assert deposited.image_A2.shape == (5, 7)
    assert deposited.deposited_mass_A2 == float(np.sum(deposited.image_A2))
    np.testing.assert_allclose(deposited.clipped_mass_A2, 0.75 * edge_mass, rtol=2.0e-15)
    np.testing.assert_allclose(
        deposited.deposited_mass_A2 + deposited.clipped_mass_A2,
        np.sum(selection.assigned_mass_A2),
        rtol=2.0e-15,
    )

    changed_deposition_hits = replace(
        deposition_hits,
        pixel_solid_angle_sr=deposition_hits.pixel_solid_angle_sr + 11.0,
    )
    changed_deposition = deposit_bilinear(
        changed_deposition_hits,
        event_row=selection.candidate_row,
        event_id=selection.event_id,
        assigned_mass_A2=selection.assigned_mass_A2,
        detector_shape_rc=(5, 7),
    )
    np.testing.assert_array_equal(changed_deposition.image_A2, deposited.image_A2)
    assert changed_deposition.deposited_mass_A2 == deposited.deposited_mass_A2
    assert changed_deposition.clipped_mass_A2 == deposited.clipped_mass_A2
