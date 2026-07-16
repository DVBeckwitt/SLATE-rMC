from __future__ import annotations

import cmath
import gzip
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rasim_next.core.contracts import (
    IncidentSampleBatch,
    MaterialOptics,
    ScatteringEventBatch,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.validity import ValidityCode
from rasim_next.geometry import (
    AxisRotation,
    InstrumentConfiguration,
    build_incident_states,
    compile_instrument,
    detector_coordinate_to_ray,
    intersect_sample_ray,
    project_detector_ray,
    transport_scattering_events,
)
from rasim_next.io.orientation import detector_native_to_raw
from rasim_next.io.osc import OscFormatError, read_osc
from rasim_next.optics import (
    mode_decay_constant,
    path_attenuation,
    scalar_optical_weight,
    solve_exit_mode,
    solve_incident_mode,
    uniform_depth_attenuation,
)

ROOT = Path(__file__).resolve().parents[1]


def _configuration() -> InstrumentConfiguration:
    identity = np.eye(3)
    zero = np.zeros(3)
    return InstrumentConfiguration(
        axis_rotations=(),
        lab_from_goniometer_zero=RigidTransform(identity, zero, FrameId.GONIOMETER, FrameId.LAB),
        goniometer_from_sample=RigidTransform(identity, zero, FrameId.SAMPLE, FrameId.GONIOMETER),
        sample_from_crystal=RigidTransform(identity, zero, FrameId.CRYSTAL, FrameId.SAMPLE),
        lab_from_detector=RigidTransform(identity, [0.0, 0.0, 1.0], FrameId.DETECTOR, FrameId.LAB),
        detector_shape_rc=(11, 7),
        detector_row_pitch_m=2.0e-4,
        detector_column_pitch_m=1.0e-4,
        detector_reference_coordinate_px=(3.0, 5.0),
        sample_width_m=4.0e-4,
        sample_length_m=6.0e-4,
        film_thickness_A=500.0,
    )


def _material(wavelength_A: float = 1.54) -> MaterialOptics:
    return MaterialOptics(
        material_id="absorbing-film",
        wavelength_A=np.array([wavelength_A]),
        n_complex=np.array([0.999979 + 3.2e-7j]),
        delta=np.array([2.1e-5]),
        beta=np.array([3.2e-7]),
        mu_Ainv=np.array([1.0e-5]),
        provenance="compact permanent fixture",
    )


def test_osc_decoding_and_native_orientation(tmp_path: Path) -> None:
    big_path = ROOT / "examples/common/osc/non_square_big_endian.osc"
    little_path = ROOT / "examples/common/osc/non_square_little_endian.osc"
    big = read_osc(big_path)
    little = read_osc(little_path)

    assert (big.metadata.version, big.metadata.byte_order) == (1, "big")
    assert (little.metadata.version, little.metadata.byte_order) == (20, "little")
    assert big.metadata.raw_shape == (7, 11)
    np.testing.assert_array_equal(big.raw_counts, little.raw_counts)
    np.testing.assert_array_equal(
        detector_native_to_raw(big.detector_native_counts),
        big.raw_counts,
    )
    assert big.detector_native_counts.shape == (11, 7)
    assert big.raw_counts[4, 3] == 2222
    assert 1_048_544 in big.raw_counts
    assert big.raw_counts.dtype == np.int32
    assert not big.raw_counts.flags.writeable
    assert not big.detector_native_counts.flags.writeable

    content = big_path.read_bytes()
    compressed_path = tmp_path / "synthetic.osc.gz"
    compressed_path.write_bytes(gzip.compress(content, mtime=0))
    np.testing.assert_array_equal(
        read_osc(compressed_path).raw_counts,
        big.raw_counts,
    )
    for name, payload in {
        "bad_signature.osc": b"NOPE!" + content[5:],
        "truncated.osc": content[:-2],
        "extra.osc": content + b"\x00\x00",
    }.items():
        path = tmp_path / name
        path.write_bytes(payload)
        with pytest.raises(OscFormatError):
            read_osc(path)


def test_rigid_sample_and_detector_geometry() -> None:
    configuration = _configuration()
    pivot_z = np.array([1.0, 0.0, 0.0])
    pivot_x = np.array([0.0, 2.0, 0.0])
    rotations = (
        AxisRotation([0.0, 0.0, 1.0], np.pi / 2.0, pivot_z),
        AxisRotation([1.0, 0.0, 0.0], np.pi / 2.0, pivot_x),
    )
    point = np.array([2.0, 0.0, 1.0])
    rotation_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    expected = pivot_z + rotation_z @ (point - pivot_z)
    expected = pivot_x + rotation_x @ (expected - pivot_x)
    ordered = compile_instrument(replace(configuration, axis_rotations=rotations))
    reversed_order = compile_instrument(replace(configuration, axis_rotations=rotations[::-1]))
    np.testing.assert_allclose(
        ordered.lab_from_goniometer.apply_point(point),
        expected,
        rtol=0.0,
        atol=2e-12,
    )
    assert not np.allclose(
        reversed_order.lab_from_goniometer.apply_point(point),
        expected,
        rtol=0.0,
        atol=2e-12,
    )


def test_sample_and_detector_statuses_and_round_trip() -> None:
    configuration = _configuration()
    instrument = compile_instrument(configuration)
    valid = intersect_sample_ray(
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        lab_from_sample=instrument.lab_from_sample,
        sample_width_m=instrument.sample_width_m,
        sample_length_m=instrument.sample_length_m,
    )
    assert valid.status is ValidityCode.VALID
    assert valid.ray_distance_m == pytest.approx(1.0)
    np.testing.assert_array_equal(valid.point_lab_m, np.zeros(3))

    for origin, direction, status in (
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], ValidityCode.PARALLEL),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], ValidityCode.BACKWARD),
        ([2.1e-4, 0.0, 1.0], [0.0, 0.0, -1.0], ValidityCode.OUTSIDE_SUPPORT),
    ):
        result = intersect_sample_ray(
            origin,
            direction,
            lab_from_sample=instrument.lab_from_sample,
            sample_width_m=instrument.sample_width_m,
            sample_length_m=instrument.sample_length_m,
        )
        assert result.status is status
        assert result.footprint_acceptance == 0.0

    direct = project_detector_ray(np.zeros(3), [0.0, 0.0, 1.0], instrument)
    assert direct.status is ValidityCode.VALID
    assert (direct.column_px, direct.row_px) == pytest.approx((3.0, 5.0))
    assert direct.pixel_solid_angle_sr == pytest.approx(2.0e-8)

    for origin, direction, status in (
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], ValidityCode.PARALLEL),
        ([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], ValidityCode.BACKWARD),
        ([1.0e-3, 0.0, 0.0], [0.0, 0.0, 1.0], ValidityCode.OUTSIDE_SUPPORT),
    ):
        projection = project_detector_ray(origin, direction, instrument)
        assert projection.status is status
        assert projection.ray_distance_m == 0.0
        assert projection.pixel_solid_angle_sr == 0.0

    edge_ray = detector_coordinate_to_ray(
        6.5,
        10.5,
        origin_lab_m=np.zeros(3),
        instrument=instrument,
    )
    edge_hit = project_detector_ray(np.zeros(3), edge_ray.direction_lab, instrument)
    assert edge_hit.status is ValidityCode.VALID
    assert (edge_hit.column_px, edge_hit.row_px) == pytest.approx((6.5, 10.5), abs=1e-9)

    detector_center = instrument.lab_from_detector.apply_point(np.zeros(3))
    detector_normal = instrument.lab_from_detector.apply_vector([0.0, 0.0, 1.0])
    near = detector_coordinate_to_ray(
        3.0,
        5.0,
        origin_lab_m=detector_center - 5e-13 * detector_normal,
        instrument=instrument,
    )
    assert near.status is ValidityCode.VALID
    assert 0.0 < near.ray_distance_m <= 1e-12


def test_refraction_and_attenuation_equations() -> None:
    k0_Ainv = 4.078420201221933
    wavelength_A = 2.0 * np.pi / k0_Ainv
    film = _material(wavelength_A)
    alpha_rad = np.deg2rad(0.05)
    mode = solve_incident_mode(
        [np.cos(alpha_rad), 0.0, np.sin(alpha_rad)],
        wavelength_A,
        film,
    )
    radicand = (film.n_complex[0] * k0_Ainv) ** 2 - (k0_Ainv * np.cos(alpha_rad)) ** 2
    oracle_kz = cmath.sqrt(radicand)
    if oracle_kz.imag < 0.0:
        oracle_kz = -oracle_kz
    oracle_amplitude = 2.0 * mode.kz_air_Ainv / (mode.kz_air_Ainv + oracle_kz)
    assert mode.status is ValidityCode.VALID
    assert mode.kz_film_Ainv == pytest.approx(oracle_kz, rel=2e-12, abs=5e-13)
    assert mode.entrance_amplitude == pytest.approx(oracle_amplitude, rel=2e-12, abs=5e-13)
    dispersion = mode.kz_film_Ainv**2 + np.dot(
        mode.k_parallel_sample_Ainv,
        mode.k_parallel_sample_Ainv,
    )
    assert dispersion == pytest.approx(
        (film.n_complex[0] * k0_Ainv) ** 2,
        rel=2e-12,
        abs=5e-13,
    )

    vacuum = MaterialOptics(
        material_id="vacuum",
        wavelength_A=np.array([wavelength_A]),
        n_complex=np.array([1.0 + 0.0j]),
        delta=np.array([0.0]),
        beta=np.array([0.0]),
        mu_Ainv=np.array([0.0]),
        provenance="analytic n=1 fixture",
    )
    equal_medium = solve_incident_mode([0.6, 0.0, -0.8], wavelength_A, vacuum)
    assert equal_medium.entrance_amplitude == pytest.approx(1.0 + 0.0j)
    assert equal_medium.propagation_direction == -1
    assert (
        solve_exit_mode([4.0, 2.0, 1.0], wavelength_A, vacuum).status
        is ValidityCode.NON_PROPAGATING
    )
    with pytest.raises(ValueError, match="exact wavelength"):
        solve_incident_mode([0.0, 0.0, 1.0], wavelength_A + 1e-12, film)

    kappa_i = np.array([1e-5, 3e-5, 1e-4])
    kappa_f = np.array([2e-5, 5e-5, 2e-4])
    thickness_A = 500.0
    exponent = 2.0 * (kappa_i + kappa_f) * thickness_A
    expected = -np.expm1(-exponent) / exponent
    np.testing.assert_allclose(
        uniform_depth_attenuation(kappa_i, kappa_f, thickness_A),
        expected,
        rtol=2e-12,
        atol=5e-13,
    )
    assert uniform_depth_attenuation(0.0, 0.0, thickness_A) == 1.0
    assert path_attenuation(1e-5, 2e-5, 100.0, 200.0) == pytest.approx(np.exp(-0.01))
    assert scalar_optical_weight(2.0 + 1.0j, 0.5 - 0.25j, 0.8) == pytest.approx(
        abs((2.0 + 1.0j) * (0.5 - 0.25j)) ** 2 * 0.8
    )


def test_transport_preserves_identity_factors_and_first_failure() -> None:
    wavelength_A = 1.54
    instrument = compile_instrument(_configuration())
    material = _material(wavelength_A)
    samples = IncidentSampleBatch(
        incident_sample_id=np.array([30, 10]),
        origin_lab_m=np.array([[0.0, 0.0, 1.0], [2.1e-4, 0.0, 1.0]]),
        direction_lab=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        wavelength_A=np.full(2, wavelength_A),
        source_weight=np.array([0.8, 0.2]),
        polarization_state_id=("p30", "p10"),
        correlation_model="independent",
    )
    incident = build_incident_states(
        samples,
        material,
        instrument,
        trace_case_id="transport",
    )
    assert incident.status == (
        ValidityCode.VALID,
        ValidityCode.OUTSIDE_SUPPORT,
    )
    np.testing.assert_array_equal(incident.states.incident_state_id, [30, 10])
    np.testing.assert_array_equal(incident.states.source_weight, [0.8, 0.2])

    film_normal_Ainv = abs(float(incident.states.kz_film_Ainv[0].real))
    events = ScatteringEventBatch(
        event_id=np.array([200, 100]),
        incident_state_id=np.array([30, 10]),
        rod_id=np.array([2, 1]),
        wavelength_A=np.full(2, wavelength_A),
        q_internal_sample_Ainv=np.zeros((2, 3)),
        qz_Ainv=np.zeros(2),
        l_coordinate=np.zeros(2),
        kf_film_phase_sample_Ainv=np.array(
            [[0.0, 0.0, film_normal_Ainv], [0.0, 0.0, film_normal_Ainv]]
        ),
        reciprocal_weight=np.array([0.25, 0.75]),
        ewald_residual_Ainv=np.zeros(2),
        valid=np.ones(2, dtype=bool),
    )
    transported = transport_scattering_events(
        events,
        incident,
        material,
        instrument,
        trace_case_id="transport",
    )
    with pytest.raises(ValueError, match="incident_state_id 999"):
        transport_scattering_events(
            replace(events, incident_state_id=np.array([999, 10])),
            incident,
            material,
            instrument,
        )
    with pytest.raises(ValueError, match="event wavelength"):
        transport_scattering_events(
            replace(events, wavelength_A=np.array([1.0, wavelength_A])),
            incident,
            material,
            instrument,
        )
    assert transported.outgoing_status == (
        ValidityCode.VALID,
        ValidityCode.NO_SOLUTION,
    )
    assert transported.detector_status == (
        ValidityCode.VALID,
        ValidityCode.NO_SOLUTION,
    )
    np.testing.assert_array_equal(transported.outgoing_waves.event_id, events.event_id)
    np.testing.assert_array_equal(transported.detector_hits.event_id, events.event_id)
    assert (transported.detector_hits.column_px[0], transported.detector_hits.row_px[0]) == (
        3.0,
        5.0,
    )

    incident_mode = solve_incident_mode([0.0, 0.0, -1.0], wavelength_A, material)
    exit_mode = solve_exit_mode(
        [0.0, 0.0, film_normal_Ainv],
        wavelength_A,
        material,
    )
    attenuation = uniform_depth_attenuation(
        mode_decay_constant(
            incident_mode.kz_film_Ainv,
            incident_mode.propagation_direction,
        ),
        mode_decay_constant(exit_mode.kz_film_Ainv, exit_mode.propagation_direction),
        instrument.film_thickness_A,
    )
    expected_optical = scalar_optical_weight(
        incident_mode.entrance_amplitude,
        exit_mode.exit_amplitude,
        attenuation,
    )
    assert transported.outgoing_waves.optical_weight[0] == pytest.approx(expected_optical)
    assert {
        "optics.kz_exit_air",
        "optics.uniform_depth_attenuation",
        "geometry.detector_column_px",
        "measurement.optical_weight",
        "measurement.pixel_solid_angle",
    } <= {record.stage_id for record in transported.traces}
