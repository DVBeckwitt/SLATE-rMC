from __future__ import annotations

import cmath
import gzip
import json
import math
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import xraydb

from rasim_next.core.contracts import (
    SAMPLE_INTERSECTION_MODEL_ID,
    IncidentSampleBatch,
    IncidentStateBatch,
    MaterialOptics,
    canonical_revision_sha256,
)
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.core.validity import ValidityCode
from rasim_next.geometry import (
    AngleFrame,
    AxisRotation,
    DetectorAngles,
    DetectorProjectionBatch,
    InstrumentConfiguration,
    angles_to_detector_coordinate_area_measure,
    angles_to_detector_coordinates,
    build_incident_states,
    compile_instrument,
    detector_coordinate_to_ray,
    detector_coordinates_to_angles,
    detector_path_linear_attenuation_at_wavelength_m_inv,
    intersect_sample_ray,
    project_detector_ray,
    project_detector_rays,
)
from rasim_next.io.orientation import detector_native_to_raw
from rasim_next.io.osc import OscFormatError, read_osc
from rasim_next.materials import crystal_with_direct_basis, material_optics, read_crystal
from rasim_next.materials.optics import HC_EV_A, atomic_scattering_factor_e
from rasim_next.optics import (
    external_path_attenuation,
    incident_illuminated_path_weight,
    path_attenuation,
    scalar_optical_weight,
    solve_exit_mode,
    solve_incident_mode,
    uniform_depth_attenuation,
)
from rasim_next.sampling.source import sample_gaussian_source_rays

ROOT = Path(__file__).resolve().parents[1]


def test_atomic_factors_batch_exactly_within_chantler_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chantler_energy_eV = np.asarray(
        xraydb.chantler_energies("Bi"),
        dtype=np.float64,
    )
    interval = int(np.searchsorted(chantler_energy_eV, 8047.8, side="right") - 1)
    left_width = chantler_energy_eV[interval + 1] - chantler_energy_eV[interval]
    right_width = chantler_energy_eV[interval + 2] - chantler_energy_eV[interval + 1]
    energy_eV = np.array(
        [
            chantler_energy_eV[interval] + 0.2 * left_width,
            chantler_energy_eV[interval] + 0.6 * left_width,
            chantler_energy_eV[interval + 1] + 0.2 * right_width,
        ]
    )
    wavelength_A = HC_EV_A / energy_eV
    evaluated_energy_eV = HC_EV_A / wavelength_A
    q_magnitude_Ainv = np.array([0.1, 0.2, 0.3])

    scalar_f1 = xraydb.f1_chantler
    scalar_f2 = xraydb.f2_chantler
    f0 = np.asarray(
        xraydb.f0("Bi", q_magnitude_Ainv / (4.0 * np.pi)),
        dtype=np.float64,
    )
    scalar_f1_values = np.asarray(
        [scalar_f1("Bi", float(energy)) for energy in evaluated_energy_eV], dtype=np.float64
    )
    scalar_f2_values = np.asarray(
        [scalar_f2("Bi", float(energy)) for energy in evaluated_energy_eV], dtype=np.float64
    )
    expected = np.asarray(f0 + scalar_f1_values + 1.0j * scalar_f2_values, dtype=np.complex128)
    f1_batch_sizes: list[int] = []
    f2_batch_sizes: list[int] = []

    def tracked_f1(element: str, energy: object) -> object:
        f1_batch_sizes.append(np.asarray(energy).size)
        return scalar_f1(element, energy)

    def tracked_f2(element: str, energy: object) -> object:
        f2_batch_sizes.append(np.asarray(energy).size)
        return scalar_f2(element, energy)

    monkeypatch.setattr(xraydb, "f1_chantler", tracked_f1)
    monkeypatch.setattr(xraydb, "f2_chantler", tracked_f2)

    actual, mapping = atomic_scattering_factor_e(
        species="Bi",
        element="Bi",
        charge=0,
        q_magnitude_Ainv=q_magnitude_Ainv,
        wavelength_A=wavelength_A,
    )

    np.testing.assert_array_equal(actual, expected)
    assert mapping == "Bi->Bi"
    assert sorted(f1_batch_sizes) == [1, 2]
    assert sorted(f2_batch_sizes) == [1, 2]


def test_crystal_direct_basis_rebuild_preserves_fractional_structure_and_recomputes_volume() -> (
    None
):
    crystal = read_crystal(
        ROOT / "examples" / "bi2se3" / "structures" / "Bi2Se3_vesta.cif",
        phase_id="bi2se3",
    )
    deformation = np.diag(np.exp((5.0e-4, 5.0e-4, -2.0e-4)))
    direct_basis = crystal.direct_basis_A @ deformation

    rebuilt = crystal_with_direct_basis(
        crystal,
        direct_basis,
        provenance="regularized lattice sensitivity",
    )

    np.testing.assert_array_equal(rebuilt.direct_basis_A, direct_basis)
    assert rebuilt.volume_A3 == pytest.approx(float(np.linalg.det(direct_basis)), abs=1.0e-12)
    assert rebuilt.sites == crystal.sites
    assert rebuilt.phase_id == crystal.phase_id
    assert rebuilt.source_path == crystal.source_path
    assert rebuilt.provenance.endswith("regularized lattice sensitivity")


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
        sample_support_model_id="finite_rectangle.v1",
        sample_width_m=4.0e-4,
        sample_length_m=6.0e-4,
        film_thickness_A=500.0,
    )


def _material(wavelength_A: float = 1.54) -> MaterialOptics:
    return MaterialOptics(
        material_id="absorbing-film",
        wavelength_A=np.array([wavelength_A]),
        n_complex=np.array([0.999979 + 3.2e-7j]),
        provenance="compact permanent fixture",
    )


def _explicit_source_batch(
    *,
    incident_sample_id: np.ndarray,
    origin_lab_m: np.ndarray,
    direction_lab: np.ndarray,
    wavelength_A: np.ndarray,
    polarization_state_id: tuple[str, ...],
) -> IncidentSampleBatch:
    size = incident_sample_id.size
    weights = np.full(size, 1.0 / size)
    provenance = "geometry transport permanent fixture.v1"
    return IncidentSampleBatch(
        incident_sample_id=incident_sample_id,
        origin_lab_m=origin_lab_m,
        direction_lab=direction_lab,
        wavelength_A=wavelength_A,
        source_weight=weights,
        polarization_state_id=polarization_state_id,
        source_sampling_model_id="explicit_test_source.v1",
        source_rng_model_id="no_rng.v1",
        source_seed=0,
        source_parameter_provenance=provenance,
    )


def _angle_frame(origin_lab_m: np.ndarray | list[float] | None = None) -> AngleFrame:
    return AngleFrame(
        origin_lab_m=np.zeros(3) if origin_lab_m is None else origin_lab_m,
        row_down_lab=np.array([0.0, 1.0, 0.0]),
        column_right_lab=np.array([1.0, 0.0, 0.0]),
        direct_beam_lab=np.array([0.0, 0.0, 1.0]),
        revision="flat-reference-v1",
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
        ordered.lab_from_sample.apply_point(point),
        expected,
        rtol=0.0,
        atol=2e-12,
    )
    assert not np.allclose(
        reversed_order.lab_from_sample.apply_point(point),
        expected,
        rtol=0.0,
        atol=2e-12,
    )
    assert not hasattr(ordered, "lab_from_goniometer")
    assert not hasattr(ordered, "lab_from_crystal")
    expected_inverse = ordered.lab_from_sample.inverse()
    np.testing.assert_array_equal(ordered.sample_from_lab.rotation, expected_inverse.rotation)
    np.testing.assert_array_equal(
        ordered.sample_from_lab.translation_m,
        expected_inverse.translation_m,
    )
    for identity in (
        ordered.lab_from_sample.compose(ordered.sample_from_lab),
        ordered.sample_from_lab.compose(ordered.lab_from_sample),
    ):
        np.testing.assert_allclose(identity.rotation, np.eye(3), rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(identity.translation_m, np.zeros(3), rtol=0.0, atol=1e-12)
    with pytest.raises(TypeError, match="init=False"):
        replace(
            ordered,
            sample_from_lab=RigidTransform(
                np.eye(3),
                np.zeros(3),
                FrameId.LAB,
                FrameId.SAMPLE,
            ),
        )

    origins_sample_m = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [2.1e-4, 0.0, 1.0]]
    )
    directions_sample = np.array(
        [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    origins_lab_m = ordered.lab_from_sample.apply_point(origins_sample_m)
    directions_lab = ordered.lab_from_sample.apply_vector(directions_sample)
    samples = _explicit_source_batch(
        incident_sample_id=np.array([11, 7, 19, 3]),
        origin_lab_m=origins_lab_m,
        direction_lab=directions_lab,
        wavelength_A=np.full(4, 1.54),
        polarization_state_id=("a", "b", "c", "d"),
    )
    batch = build_incident_states(samples, _material(), ordered).states
    scalar = tuple(
        intersect_sample_ray(
            origin,
            direction,
            lab_from_sample=ordered.lab_from_sample,
            sample_support_model_id=ordered.sample_support_model_id,
            sample_width_m=ordered.sample_width_m,
            sample_length_m=ordered.sample_length_m,
        )
        for origin, direction in zip(origins_lab_m, directions_lab, strict=True)
    )
    assert batch.status == tuple(result.status for result in scalar)
    np.testing.assert_allclose(
        batch.sample_intersection_lab_m,
        np.stack([result.point_lab_m for result in scalar]),
        rtol=0.0,
        atol=1e-12,
    )
    assert batch.status == (
        ValidityCode.VALID,
        ValidityCode.BACKWARD,
        ValidityCode.PARALLEL,
        ValidityCode.OUTSIDE_SUPPORT,
    )
    np.testing.assert_allclose(batch.direction_sample[0], directions_sample[0], atol=1e-12)
    scalar_mode = solve_incident_mode(directions_sample[0], 1.54, _material())
    np.testing.assert_allclose(
        batch.k_film_phase_sample_Ainv[0],
        scalar_mode.k_film_phase_sample_Ainv,
        rtol=0.0,
        atol=2e-15,
    )


def test_sample_and_detector_statuses_and_round_trip() -> None:
    configuration = _configuration()
    instrument = compile_instrument(configuration)
    valid = intersect_sample_ray(
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        lab_from_sample=instrument.lab_from_sample,
        sample_support_model_id=instrument.sample_support_model_id,
        sample_width_m=instrument.sample_width_m,
        sample_length_m=instrument.sample_length_m,
    )
    assert valid.status is ValidityCode.VALID
    assert valid.ray_distance_m == pytest.approx(1.0)
    np.testing.assert_array_equal(valid.point_lab_m, np.zeros(3))

    threshold_outside = np.nextafter(1e-14, np.inf)
    threshold_inside = 1e-14
    near_positive = np.array([np.sqrt(1.0 - threshold_outside**2), 0.0, threshold_outside])
    near_negative = np.array([np.sqrt(1.0 - threshold_outside**2), 0.0, -threshold_outside])
    for origin, direction, status in (
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], ValidityCode.PARALLEL),
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], ValidityCode.PARALLEL),
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], ValidityCode.PARALLEL),
        ([0.0, 0.0, 0.0], [1.0, 0.0, threshold_inside], ValidityCode.PARALLEL),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], ValidityCode.BACKWARD),
        ([2.1e-4, 0.0, 1.0], [0.0, 0.0, -1.0], ValidityCode.OUTSIDE_SUPPORT),
        ([0.0, 0.0, -threshold_outside * 1e-5], near_positive, ValidityCode.VALID),
        ([0.0, 0.0, threshold_outside * 1e-5], near_negative, ValidityCode.VALID),
    ):
        direction = np.asarray(direction, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        result = intersect_sample_ray(
            origin,
            direction,
            lab_from_sample=instrument.lab_from_sample,
            sample_support_model_id=instrument.sample_support_model_id,
            sample_width_m=instrument.sample_width_m,
            sample_length_m=instrument.sample_length_m,
        )
        assert result.status is status
        assert result.footprint_acceptance == float(status is ValidityCode.VALID)
        if status is ValidityCode.PARALLEL:
            np.testing.assert_array_equal(result.point_lab_m, np.zeros(3))
            np.testing.assert_array_equal(result.point_sample_m, np.zeros(3))
            assert result.ray_distance_m == 0.0

    unbounded = compile_instrument(
        replace(
            configuration,
            sample_support_model_id="unbounded_plane.v1",
            sample_width_m=None,
            sample_length_m=None,
        )
    )
    unbounded_hit = intersect_sample_ray(
        [2.1e-4, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        lab_from_sample=unbounded.lab_from_sample,
        sample_support_model_id=unbounded.sample_support_model_id,
        sample_width_m=unbounded.sample_width_m,
        sample_length_m=unbounded.sample_length_m,
    )
    assert unbounded_hit.status is ValidityCode.VALID
    assert unbounded_hit.footprint_acceptance == 1.0
    assert (unbounded.sample_width_m, unbounded.sample_length_m) == (None, None)

    for support_model_id, width_m, length_m in (
        ("finite_rectangle.v1", None, 1.0),
        ("finite_rectangle.v1", 0.0, 1.0),
        ("unbounded_plane.v1", 1.0, None),
        ("unknown.v1", None, None),
    ):
        with pytest.raises(ValueError, match=r"sample|finite_rectangle|unbounded_plane"):
            replace(
                configuration,
                sample_support_model_id=support_model_id,
                sample_width_m=width_m,
                sample_length_m=length_m,
            )

    direct = project_detector_ray(np.zeros(3), [0.0, 0.0, 1.0], instrument)
    assert direct.status is ValidityCode.VALID
    assert (direct.column_px, direct.row_px) == pytest.approx((3.0, 5.0))
    assert direct.pixel_solid_angle_sr == pytest.approx(2.0e-8)

    for origin, direction, status in (
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], ValidityCode.PARALLEL),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], ValidityCode.PARALLEL),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], ValidityCode.NO_SOLUTION),
        ([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], ValidityCode.BACKWARD),
        ([0.0, 0.0, 2.0], [0.0, 0.0, -1.0], ValidityCode.BACKWARD),
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

    back_facing = detector_coordinate_to_ray(
        3.0,
        5.0,
        origin_lab_m=detector_center + 0.01 * detector_normal,
        instrument=instrument,
    )
    assert back_facing.status is ValidityCode.BACKWARD
    assert back_facing.ray_distance_m == 0.0

    back_facing_angles = detector_coordinates_to_angles(
        [3.0],
        [5.0],
        instrument=instrument,
        angle_frame=_angle_frame(detector_center + 0.01 * detector_normal),
    )
    assert back_facing_angles.status[0] == ValidityCode.BACKWARD
    assert not back_facing_angles.valid[0]


def test_batched_detector_projection_matches_scalar_rays() -> None:
    instrument = compile_instrument(_configuration())
    origins = np.zeros((4, 3), dtype=np.float64)
    directions = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (1.0e-3, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    batch = project_detector_rays(origins, directions, instrument)
    scalar = tuple(
        project_detector_ray(origin, direction, instrument)
        for origin, direction in zip(origins, directions, strict=True)
    )

    assert tuple(batch.status) == tuple(item.status for item in scalar)
    np.testing.assert_array_equal(
        batch.valid,
        np.asarray([item.status is ValidityCode.VALID for item in scalar]),
    )
    np.testing.assert_allclose(batch.point_lab_m, [item.point_lab_m for item in scalar])
    np.testing.assert_allclose(batch.column_px, [item.column_px for item in scalar])
    np.testing.assert_allclose(batch.row_px, [item.row_px for item in scalar])
    np.testing.assert_allclose(batch.ray_distance_m, [item.ray_distance_m for item in scalar])
    np.testing.assert_allclose(
        batch.pixel_solid_angle_sr,
        [item.pixel_solid_angle_sr for item in scalar],
    )
    for value in (
        batch.point_lab_m,
        batch.column_px,
        batch.row_px,
        batch.ray_distance_m,
        batch.pixel_solid_angle_sr,
        batch.valid,
        batch.status,
    ):
        assert not value.flags.writeable

    preserved_status = DetectorProjectionBatch(
        point_lab_m=np.zeros((1, 3)),
        column_px=np.zeros(1),
        row_px=np.zeros(1),
        ray_distance_m=np.zeros(1),
        pixel_solid_angle_sr=np.zeros(1),
        status=np.asarray([ValidityCode.RESIDUAL_EXCEEDED]),
    )
    assert preserved_status.status[0] == ValidityCode.RESIDUAL_EXCEEDED


def test_detector_angles_cardinals_wrap_and_direct_beam() -> None:
    instrument = compile_instrument(_configuration())
    frame = _angle_frame()
    columns = np.array([3.0, 3.0, 5.0, 3.0, 1.0])
    rows = np.array([5.0, 4.0, 5.0, 6.0, 5.0])

    angles = detector_coordinates_to_angles(
        columns,
        rows,
        instrument=instrument,
        angle_frame=frame,
    )
    radial_angle = np.arctan2(2.0e-4, 1.0)
    np.testing.assert_allclose(
        angles.two_theta_rad,
        [0.0, radial_angle, radial_angle, radial_angle, radial_angle],
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        angles.chi_raw_rad,
        [0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, -np.pi],
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        angles.phi_rad,
        [0.0, 0.0, -np.pi / 2.0, -np.pi, np.pi / 2.0],
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_array_equal(angles.valid, np.ones(5, dtype=bool))
    np.testing.assert_array_equal(angles.azimuth_valid, [False, True, True, True, True])
    np.testing.assert_array_equal(
        angles.status,
        np.full(5, ValidityCode.VALID, dtype="U16"),
    )

    recovered = angles_to_detector_coordinates(
        angles.two_theta_rad,
        angles.phi_rad,
        instrument=instrument,
        angle_frame=frame,
    )
    np.testing.assert_allclose(recovered.column_px, columns, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(recovered.row_px, rows, rtol=0.0, atol=2e-12)
    np.testing.assert_array_equal(recovered.valid, np.ones(5, dtype=bool))

    across_seam = detector_coordinates_to_angles(
        [3.0 - 1e-6, 3.0 + 1e-6],
        [6.0, 6.0],
        instrument=instrument,
        angle_frame=frame,
    )
    assert 0.0 < across_seam.phi_rad[0] < np.pi
    assert -np.pi < across_seam.phi_rad[1] < 0.0
    assert not angles.two_theta_rad.flags.writeable
    assert not recovered.column_px.flags.writeable


def test_detector_angles_tilted_forward_inverse_oracles() -> None:
    tilt_rad = 0.37
    cosine = np.cos(tilt_rad)
    sine = np.sin(tilt_rad)
    detector_rotation = np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    detector_translation_m = np.array([0.12, -0.04, 0.8])
    configuration = replace(
        _configuration(),
        lab_from_detector=RigidTransform(
            detector_rotation,
            detector_translation_m,
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
    )
    instrument = compile_instrument(configuration)
    frame = _angle_frame()
    columns = np.array([[-0.5, 1.25, 6.5], [3.0, -0.5, 6.5]])
    rows = np.array([[-0.5, 8.75, -0.5], [5.0, 10.5, 10.5]])

    local_points_m = np.stack(
        (
            (columns - 3.0) * 1.0e-4,
            (rows - 5.0) * 2.0e-4,
            np.zeros_like(columns),
        ),
        axis=-1,
    )
    lab_points_m = local_points_m @ detector_rotation.T + detector_translation_m
    directions_lab = lab_points_m / np.linalg.norm(lab_points_m, axis=-1, keepdims=True)
    oracle_two_theta = np.arctan2(
        np.hypot(directions_lab[..., 0], directions_lab[..., 1]),
        directions_lab[..., 2],
    )
    oracle_chi = np.arctan2(directions_lab[..., 1], directions_lab[..., 0])
    oracle_phi = (-np.pi / 2.0 - oracle_chi + np.pi) % (2.0 * np.pi) - np.pi

    angles = detector_coordinates_to_angles(
        columns,
        rows,
        instrument=instrument,
        angle_frame=frame,
    )
    np.testing.assert_allclose(angles.two_theta_rad, oracle_two_theta, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(angles.chi_raw_rad, oracle_chi, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(angles.phi_rad, oracle_phi, rtol=0.0, atol=2e-14)

    recovered = angles_to_detector_coordinates(
        angles.two_theta_rad,
        angles.phi_rad + 4.0 * np.pi,
        instrument=instrument,
        angle_frame=frame,
    )
    np.testing.assert_allclose(recovered.column_px, columns, rtol=0.0, atol=8e-12)
    np.testing.assert_allclose(recovered.row_px, rows, rtol=0.0, atol=8e-12)
    np.testing.assert_array_equal(recovered.valid, np.ones(columns.shape, dtype=bool))

    reconstructed = detector_coordinates_to_angles(
        recovered.column_px,
        recovered.row_px,
        instrument=instrument,
        angle_frame=frame,
    )
    np.testing.assert_allclose(
        reconstructed.two_theta_rad,
        angles.two_theta_rad,
        rtol=0.0,
        atol=2e-14,
    )
    circular_phi_error = np.arctan2(
        np.sin(reconstructed.phi_rad - angles.phi_rad),
        np.cos(reconstructed.phi_rad - angles.phi_rad),
    )
    np.testing.assert_allclose(circular_phi_error, 0.0, rtol=0.0, atol=2e-14)


def test_detector_angle_inverse_exposes_the_coordinate_jacobian() -> None:
    tilt_y_rad = math.radians(-5.0)
    tilt_x_rad = math.radians(7.0)
    rotation_y = np.array(
        [
            [np.cos(tilt_y_rad), 0.0, np.sin(tilt_y_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(tilt_y_rad), 0.0, np.cos(tilt_y_rad)],
        ]
    )
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt_x_rad), -np.sin(tilt_x_rad)],
            [0.0, np.sin(tilt_x_rad), np.cos(tilt_x_rad)],
        ]
    )
    configuration = replace(
        _configuration(),
        lab_from_detector=RigidTransform(
            rotation_x @ rotation_y,
            [0.0, 0.0, 1.0],
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        detector_shape_rc=(1001, 1001),
        detector_row_pitch_m=1.1e-3,
        detector_column_pitch_m=7.0e-4,
        detector_reference_coordinate_px=(500.0, 500.0),
    )
    instrument = compile_instrument(configuration)
    frame = _angle_frame()
    target_columns = np.array([220.1, 501.2, 781.4])
    target_rows = np.array([263.3, 517.1, 742.7])
    target_angles = detector_coordinates_to_angles(
        target_columns,
        target_rows,
        instrument=instrument,
        angle_frame=frame,
    )
    assert np.all(target_angles.valid)

    inverse_measure = angles_to_detector_coordinate_area_measure(
        target_angles.two_theta_rad,
        target_angles.phi_rad,
        instrument=instrument,
        angle_frame=frame,
    )
    sine = np.sin(target_angles.two_theta_rad)
    directions_lab = (
        (-sine * np.cos(target_angles.phi_rad))[:, None] * frame.row_down_lab
        + (-sine * np.sin(target_angles.phi_rad))[:, None] * frame.column_right_lab
        + np.cos(target_angles.two_theta_rad)[:, None] * frame.direct_beam_lab
    )
    projections = project_detector_rays(
        np.broadcast_to(frame.origin_lab_m, directions_lab.shape),
        directions_lab,
        instrument,
    )
    expected_jacobian = sine / projections.pixel_solid_angle_sr
    step_rad = 2.0e-7
    theta_plus = angles_to_detector_coordinates(
        target_angles.two_theta_rad + step_rad,
        target_angles.phi_rad,
        instrument=instrument,
        angle_frame=frame,
    )
    theta_minus = angles_to_detector_coordinates(
        target_angles.two_theta_rad - step_rad,
        target_angles.phi_rad,
        instrument=instrument,
        angle_frame=frame,
    )
    phi_plus = angles_to_detector_coordinates(
        target_angles.two_theta_rad,
        target_angles.phi_rad + step_rad,
        instrument=instrument,
        angle_frame=frame,
    )
    phi_minus = angles_to_detector_coordinates(
        target_angles.two_theta_rad,
        target_angles.phi_rad - step_rad,
        instrument=instrument,
        angle_frame=frame,
    )
    dc_dtheta = (theta_plus.column_px - theta_minus.column_px) / (2.0 * step_rad)
    dr_dtheta = (theta_plus.row_px - theta_minus.row_px) / (2.0 * step_rad)
    dc_dphi = (phi_plus.column_px - phi_minus.column_px) / (2.0 * step_rad)
    dr_dphi = (phi_plus.row_px - phi_minus.row_px) / (2.0 * step_rad)
    finite_difference_jacobian = np.abs(dc_dtheta * dr_dphi - dc_dphi * dr_dtheta)

    np.testing.assert_allclose(
        inverse_measure.detector_area_jacobian_px2_per_rad2,
        expected_jacobian,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        inverse_measure.detector_area_jacobian_px2_per_rad2,
        finite_difference_jacobian,
        rtol=2.0e-9,
        atol=0.0,
    )
    periodic = angles_to_detector_coordinate_area_measure(
        target_angles.two_theta_rad,
        target_angles.phi_rad + 2.0 * np.pi,
        instrument=instrument,
        angle_frame=frame,
    )
    np.testing.assert_allclose(
        periodic.detector_area_jacobian_px2_per_rad2,
        inverse_measure.detector_area_jacobian_px2_per_rad2,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert not inverse_measure.detector_area_jacobian_px2_per_rad2.flags.writeable

    flat_instrument = compile_instrument(_configuration())
    pole = angles_to_detector_coordinate_area_measure(
        [0.0, np.finfo(np.float64).eps],
        [1.7, -0.4],
        instrument=flat_instrument,
        angle_frame=frame,
    )
    assert np.all(pole.coordinates.valid)
    np.testing.assert_array_equal(pole.detector_area_jacobian_px2_per_rad2, 0.0)


def test_detector_angle_support_frame_and_inverse_statuses() -> None:
    instrument = compile_instrument(_configuration())
    frame = _angle_frame()
    columns = np.array(
        [
            -0.5,
            6.5,
            np.nextafter(-0.5, -np.inf),
            np.nextafter(6.5, np.inf),
            3.0,
            3.0,
        ]
    )
    rows = np.array(
        [
            5.0,
            5.0,
            5.0,
            5.0,
            np.nextafter(-0.5, -np.inf),
            np.nextafter(10.5, np.inf),
        ]
    )
    forward = detector_coordinates_to_angles(
        columns,
        rows,
        instrument=instrument,
        angle_frame=frame,
    )
    np.testing.assert_array_equal(
        forward.status,
        [
            ValidityCode.VALID,
            ValidityCode.VALID,
            ValidityCode.OUTSIDE_SUPPORT,
            ValidityCode.OUTSIDE_SUPPORT,
            ValidityCode.OUTSIDE_SUPPORT,
            ValidityCode.OUTSIDE_SUPPORT,
        ],
    )

    edge_theta = np.arctan2(3.5e-4, 1.0)
    inverse = angles_to_detector_coordinates(
        [np.pi / 2.0, np.pi, edge_theta, np.arctan2(3.500001e-4, 1.0)],
        [-np.pi / 2.0, 0.0, -np.pi / 2.0, -np.pi / 2.0],
        instrument=instrument,
        angle_frame=frame,
    )
    np.testing.assert_array_equal(
        inverse.status,
        [
            ValidityCode.PARALLEL,
            ValidityCode.BACKWARD,
            ValidityCode.VALID,
            ValidityCode.OUTSIDE_SUPPORT,
        ],
    )
    assert inverse.column_px[2] == pytest.approx(6.5, abs=2e-12)

    coplanar_frame = _angle_frame(np.array([0.0, 0.0, 1.0]))
    coplanar_forward = detector_coordinates_to_angles(
        [3.0],
        [5.0],
        instrument=instrument,
        angle_frame=coplanar_frame,
    )
    coplanar_inverse = angles_to_detector_coordinates(
        [0.1],
        [0.0],
        instrument=instrument,
        angle_frame=coplanar_frame,
    )
    assert coplanar_forward.status[0] == ValidityCode.NO_SOLUTION
    assert coplanar_inverse.status[0] == ValidityCode.NO_SOLUTION

    with pytest.raises(ValueError, match="right-handed detector-image basis"):
        AngleFrame(
            origin_lab_m=np.zeros(3),
            row_down_lab=[0.0, 1.0, 0.0],
            column_right_lab=[-1.0, 0.0, 0.0],
            direct_beam_lab=[0.0, 0.0, 1.0],
            revision="wrong-handed",
        )
    with pytest.raises(ValueError, match="two_theta_rad"):
        angles_to_detector_coordinates(
            [-1e-12],
            [0.0],
            instrument=instrument,
            angle_frame=frame,
        )

    retained_status = DetectorAngles(
        [0.0],
        [0.0],
        [0.0],
        [False],
        [False],
        [ValidityCode.RESIDUAL_EXCEEDED],
    )
    assert retained_status.status[0] == ValidityCode.RESIDUAL_EXCEEDED
    with pytest.raises(ValueError, match="unknown validity"):
        replace(retained_status, status=np.array(["BOGUS"]))


def test_refraction_and_attenuation_equations() -> None:
    with pytest.raises(ValueError, match="unity detector path medium"):
        compile_instrument(
            replace(
                _configuration(),
                detector_path_linear_attenuation_m_inv=1.2,
            )
        )
    air_instrument = compile_instrument(
        replace(
            _configuration(),
            detector_path_medium_id="standard_dry_air_sensitivity.v1",
            detector_path_linear_attenuation_m_inv=1.2,
        )
    )
    assert air_instrument.detector_path_medium_id == "standard_dry_air_sensitivity.v1"
    assert air_instrument.detector_path_linear_attenuation_m_inv == 1.2
    table_instrument = compile_instrument(
        replace(
            _configuration(),
            detector_path_medium_id="standard_dry_air_wavelength_table_sensitivity.v1",
            detector_path_wavelength_A=(1.540592925, 1.544427),
            detector_path_linear_attenuation_m_inv_by_wavelength=(
                1.194448861306956,
                1.2033257750023947,
            ),
        )
    )
    generated_table = replace(
        _configuration(),
        detector_path_medium_id="standard_dry_air_wavelength_table_sensitivity.v1",
        detector_path_wavelength_A=(value for value in (1.540592925, 1.544427)),
        detector_path_linear_attenuation_m_inv_by_wavelength=(
            value for value in (1.194448861306956, 1.2033257750023947)
        ),
    )
    assert generated_table.detector_path_wavelength_A == (1.540592925, 1.544427)
    assert generated_table.detector_path_linear_attenuation_m_inv_by_wavelength == (
        1.194448861306956,
        1.2033257750023947,
    )
    assert (
        detector_path_linear_attenuation_at_wavelength_m_inv(
            air_instrument,
            1.7,
        )
        == 1.2
    )
    assert (
        detector_path_linear_attenuation_at_wavelength_m_inv(
            table_instrument,
            1.540592925,
        )
        == 1.194448861306956
    )
    assert (
        detector_path_linear_attenuation_at_wavelength_m_inv(
            table_instrument,
            1.544427,
        )
        == 1.2033257750023947
    )
    with pytest.raises(ValueError, match="does not contain the exact wavelength"):
        detector_path_linear_attenuation_at_wavelength_m_inv(table_instrument, 1.541)
    for change in (
        {
            "detector_path_medium_id": "air.v1",
            "detector_path_wavelength_A": (1.54,),
        },
        {
            "detector_path_medium_id": "air.v1",
            "detector_path_wavelength_A": (1.54, 1.53),
            "detector_path_linear_attenuation_m_inv_by_wavelength": (1.0, 1.1),
        },
        {
            "detector_path_medium_id": "air.v1",
            "detector_path_linear_attenuation_m_inv": 1.0,
            "detector_path_wavelength_A": (1.54,),
            "detector_path_linear_attenuation_m_inv_by_wavelength": (1.0,),
        },
        {
            "detector_path_wavelength_A": (1.54,),
            "detector_path_linear_attenuation_m_inv_by_wavelength": (1.0,),
        },
        {
            "detector_path_medium_id": "air.v1",
            "detector_path_linear_attenuation_m_inv": True,
        },
        {
            "detector_path_medium_id": "air.v1",
            "detector_path_wavelength_A": (1.54,),
            "detector_path_linear_attenuation_m_inv_by_wavelength": (True,),
        },
        {
            "detector_path_medium_id": "air.v1",
            "detector_path_wavelength_A": (True,),
            "detector_path_linear_attenuation_m_inv_by_wavelength": (1.0,),
        },
        {
            "detector_path_medium_id": "air.v1",
            "detector_path_wavelength_A": (np.bool_(True),),
            "detector_path_linear_attenuation_m_inv_by_wavelength": (1.0,),
        },
        {
            "detector_path_medium_id": "air.v1",
            "detector_path_wavelength_A": (1.54,),
            "detector_path_linear_attenuation_m_inv_by_wavelength": (np.complex128(1.0 + 0.1j),),
        },
    ):
        with pytest.raises(ValueError):
            replace(_configuration(), **change)
    for invalid_wavelength in (True, np.bool_(True), np.complex128(1.54 + 0.1j)):
        with pytest.raises(ValueError, match="real number"):
            detector_path_linear_attenuation_at_wavelength_m_inv(
                table_instrument,
                invalid_wavelength,
            )
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
    assert incident_illuminated_path_weight([0.0, 0.0, -1.0]) == 1.0
    five_deg = incident_illuminated_path_weight(
        [math.cos(math.radians(5.0)), 0.0, -math.sin(math.radians(5.0))]
    )
    twenty_deg = incident_illuminated_path_weight(
        [math.cos(math.radians(20.0)), 0.0, -math.sin(math.radians(20.0))]
    )
    assert five_deg / twenty_deg == pytest.approx(
        math.sin(math.radians(20.0)) / math.sin(math.radians(5.0)),
        rel=2.0e-15,
    )
    with pytest.raises(ValueError, match="nonzero sample-normal"):
        incident_illuminated_path_weight([1.0, 0.0, 0.0])
    assert path_attenuation(1e-5, 2e-5, 100.0, 200.0) == pytest.approx(np.exp(-0.01))
    np.testing.assert_array_equal(
        external_path_attenuation(0.0, [0.075, 0.15]),
        np.ones(2),
    )
    np.testing.assert_allclose(
        external_path_attenuation(1.2, [0.075, 0.15]),
        np.exp(-1.2 * np.asarray((0.075, 0.15))),
        rtol=2.0e-15,
    )
    assert scalar_optical_weight(2.0 + 1.0j, 0.5 - 0.25j, 0.8) == pytest.approx(
        abs((2.0 + 1.0j) * (0.5 - 0.25j)) ** 2 * 0.8
    )


def test_incident_transport_preserves_identity_and_first_failure() -> None:
    wavelength_A = 1.54
    instrument = compile_instrument(_configuration())
    material = _material(wavelength_A)
    mixed_samples = _explicit_source_batch(
        incident_sample_id=np.array([42, 8]),
        origin_lab_m=np.array([[0.0, 0.0, 1.0], [2.1e-4, 0.0, 1.0]]),
        direction_lab=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        wavelength_A=np.array([wavelength_A, 1.73]),
        polarization_state_id=("mixed-valid", "mixed-missed"),
    )
    mixed = build_incident_states(mixed_samples, material, instrument).states
    assert mixed.status == (ValidityCode.VALID, ValidityCode.OUTSIDE_SUPPORT)
    np.testing.assert_array_equal(mixed.incident_state_id, [42, 8])
    np.testing.assert_array_equal(mixed.wavelength_A, [wavelength_A, 1.73])
    np.testing.assert_array_equal(mixed.source_weight, [0.5, 0.5])
    np.testing.assert_array_equal(mixed.k_film_phase_sample_Ainv[1], np.zeros(3))

    all_invalid_samples = _explicit_source_batch(
        incident_sample_id=np.array([91, 17]),
        origin_lab_m=np.array([[2.1e-4, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        direction_lab=np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
        wavelength_A=np.array([1.71, 1.73]),
        polarization_state_id=("missed", "coplanar"),
    )
    all_invalid = build_incident_states(all_invalid_samples, material, instrument).states
    assert all_invalid.status == (ValidityCode.OUTSIDE_SUPPORT, ValidityCode.PARALLEL)
    np.testing.assert_array_equal(all_invalid.incident_state_id, [91, 17])
    np.testing.assert_array_equal(all_invalid.wavelength_A, [1.71, 1.73])
    np.testing.assert_array_equal(all_invalid.source_weight, [0.5, 0.5])
    for payload in (
        all_invalid.sample_intersection_lab_m,
        all_invalid.direction_sample,
        all_invalid.k_air_sample_Ainv,
        all_invalid.k_film_phase_sample_Ainv,
        all_invalid.kz_film_Ainv,
        all_invalid.entrance_amplitude,
        all_invalid.footprint_acceptance,
    ):
        assert np.all(payload == 0.0)
    samples = _explicit_source_batch(
        incident_sample_id=np.array([30, 10]),
        origin_lab_m=np.array([[0.0, 0.0, 1.0], [2.1e-4, 0.0, 1.0]]),
        direction_lab=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        wavelength_A=np.full(2, wavelength_A),
        polarization_state_id=("p30", "p10"),
    )
    incident = build_incident_states(
        samples,
        material,
        instrument,
        trace_case_id="transport",
    )
    assert incident.states.status == (
        ValidityCode.VALID,
        ValidityCode.OUTSIDE_SUPPORT,
    )
    np.testing.assert_array_equal(incident.states.incident_state_id, [30, 10])
    np.testing.assert_array_equal(incident.states.source_weight, [0.5, 0.5])
    assert incident.states.wavelength_A[1] == wavelength_A
    assert incident.states.polarization_state_id[1] == "p10"
    for payload in (
        incident.states.sample_intersection_lab_m[1],
        incident.states.direction_sample[1],
        incident.states.k_air_sample_Ainv[1],
        incident.states.k_film_phase_sample_Ainv[1],
        incident.states.kz_film_Ainv[1],
        incident.states.entrance_amplitude[1],
        incident.states.footprint_acceptance[1],
    ):
        assert np.all(payload == 0.0)
    incident_parallel_records = tuple(
        record for record in incident.traces if record.stage_id == "optics.ki_parallel_sample"
    )
    expected_parallel = incident.states.k_air_sample_Ainv.copy()
    expected_parallel[:, 2] = 0.0
    np.testing.assert_array_equal(
        np.stack([record.value for record in incident_parallel_records]),
        expected_parallel,
    )
    assert {(record.unit, record.frame) for record in incident_parallel_records} == {
        ("angstrom^-1", FrameId.SAMPLE)
    }

    assert {
        "geometry.sample_intersection",
        "optics.ki_air_sample",
        "optics.ki_parallel_sample",
        "optics.kz_incident_film",
        "optics.entrance_amplitude",
        "sampling.source_empirical_mass",
    } <= {record.stage_id for record in incident.traces}


def test_public_monochromatic_multi_ray_source_uses_one_exact_material_row() -> None:
    wavelength_A = 1.540592925
    samples = sample_gaussian_source_rays(
        mean_origin_lab_m=np.array([0.0, 0.0, 1.0]),
        mean_direction_lab=np.array([0.0, 0.0, -1.0]),
        transverse_axes_lab=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        spatial_sigma_m=np.zeros(2),
        divergence_sigma_rad=np.zeros(2),
        mean_wavelength_A=wavelength_A,
        wavelength_sigma_A=0.0,
        sample_count=3,
        seed=17,
        polarization_state_id="UNITY_APPROXIMATION",
    )
    crystal = read_crystal(
        ROOT / "examples" / "bi2se3" / "structures" / "Bi2Se3_vesta.cif",
        phase_id="bi2se3-monochromatic",
    )
    material = material_optics(crystal, samples.wavelength_A)
    assert material.wavelength_A.tolist() == [wavelength_A]

    incident = build_incident_states(samples, material, compile_instrument(_configuration()))
    assert incident.states.status == (ValidityCode.VALID,) * 3
    np.testing.assert_array_equal(incident.states.wavelength_A, np.full(3, wavelength_A))
    assert np.all(incident.states.valid)


def test_compiled_sample_geometry_revision_contract_and_transport_ownership() -> None:
    configuration = _configuration()
    instrument = compile_instrument(configuration)
    expected_revision = canonical_revision_sha256(
        ("intersection_model_id", SAMPLE_INTERSECTION_MODEL_ID),
        ("lab_from_sample_rotation", instrument.lab_from_sample.rotation),
        ("lab_from_sample_source_frame", "sample"),
        ("lab_from_sample_target_frame", "lab"),
        ("lab_from_sample_translation_m", instrument.lab_from_sample.translation_m),
        ("sample_entrance_revision_schema", "sample_entrance_revision.v2"),
        ("sample_length_m", instrument.sample_length_m),
        ("sample_support_model_id", instrument.sample_support_model_id),
        ("sample_width_m", instrument.sample_width_m),
    )
    assert instrument.sample_geometry_revision == expected_revision

    samples = _explicit_source_batch(
        incident_sample_id=np.array([0]),
        origin_lab_m=np.array([[0.0, 0.0, 1.0]]),
        direction_lab=np.array([[0.0, 0.0, -1.0]]),
        wavelength_A=np.array([1.54]),
        polarization_state_id=("UNITY_APPROXIMATION",),
    )
    states = build_incident_states(samples, _material(), instrument).states
    assert states.sample_geometry_revision == instrument.sample_geometry_revision
    assert states.status == (ValidityCode.VALID,)

    shifted_instrument = compile_instrument(
        replace(
            configuration,
            goniometer_from_sample=RigidTransform(
                np.eye(3),
                np.array([3.0e-4, 0.0, 0.0]),
                FrameId.SAMPLE,
                FrameId.GONIOMETER,
            ),
        )
    )
    shifted_states = build_incident_states(samples, _material(), shifted_instrument).states
    assert shifted_instrument.sample_geometry_revision != instrument.sample_geometry_revision
    assert shifted_states.sample_geometry_revision == shifted_instrument.sample_geometry_revision
    assert shifted_states.status == (ValidityCode.OUTSIDE_SUPPORT,)


def test_unbounded_sample_revision_ignores_only_tangent_origin_translation() -> None:
    rotation = np.column_stack(
        (
            np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
            np.array([-1.0, 1.0, 2.0]) / np.sqrt(6.0),
            np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0),
        )
    )
    normal_lab = rotation[:, 2]
    plane_offset_m = 1.0e-8
    base_translation_m = plane_offset_m * normal_lab
    configuration = replace(
        _configuration(),
        goniometer_from_sample=RigidTransform(
            rotation,
            base_translation_m,
            FrameId.SAMPLE,
            FrameId.GONIOMETER,
        ),
        sample_support_model_id="unbounded_plane.v1",
        sample_width_m=None,
        sample_length_m=None,
    )
    tangent_configurations = tuple(
        replace(
            configuration,
            goniometer_from_sample=RigidTransform(
                rotation,
                base_translation_m + shift_m,
                FrameId.SAMPLE,
                FrameId.GONIOMETER,
            ),
        )
        for shift_m in (0.23456789 * rotation[:, 0], -0.34567891 * rotation[:, 1])
    )
    normal_configuration = replace(
        configuration,
        goniometer_from_sample=RigidTransform(
            rotation,
            base_translation_m + 2.0e-12 * normal_lab,
            FrameId.SAMPLE,
            FrameId.GONIOMETER,
        ),
    )
    instrument = compile_instrument(configuration)
    tangent_instruments = tuple(compile_instrument(item) for item in tangent_configurations)
    normal_instrument = compile_instrument(normal_configuration)

    expected_revision = canonical_revision_sha256(
        ("intersection_model_id", SAMPLE_INTERSECTION_MODEL_ID),
        ("lab_from_sample_rotation", instrument.lab_from_sample.rotation),
        ("lab_from_sample_source_frame", "sample"),
        ("lab_from_sample_target_frame", "lab"),
        ("sample_entrance_revision_schema", "sample_entrance_revision.v2"),
        ("sample_plane_signed_normal_offset_lab_m", plane_offset_m),
        ("sample_support_model_id", instrument.sample_support_model_id),
    )
    assert instrument.sample_geometry_revision == expected_revision
    assert all(item.sample_geometry_revision == expected_revision for item in tangent_instruments)
    assert normal_instrument.sample_geometry_revision != expected_revision

    samples = _explicit_source_batch(
        incident_sample_id=np.array([0]),
        origin_lab_m=np.asarray([normal_lab]),
        direction_lab=np.asarray([-normal_lab]),
        wavelength_A=np.array([1.54]),
        polarization_state_id=("UNITY_APPROXIMATION",),
    )
    material = _material()
    states = build_incident_states(samples, material, instrument).states
    tangent_states = tuple(
        build_incident_states(samples, material, item).states for item in tangent_instruments
    )
    normal_states = build_incident_states(samples, material, normal_instrument).states

    assert states.status == normal_states.status == (ValidityCode.VALID,)
    for tangent_state in tangent_states:
        assert tangent_state.status == states.status
        np.testing.assert_allclose(
            tangent_state.sample_intersection_lab_m,
            states.sample_intersection_lab_m,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_array_equal(
            tangent_state.k_film_phase_sample_Ainv,
            states.k_film_phase_sample_Ainv,
        )
    assert not np.allclose(
        normal_states.sample_intersection_lab_m,
        states.sample_intersection_lab_m,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_incident_revision_ownership_and_excluded_instrument_fields() -> None:
    wavelength_A = 1.540592925
    source_arguments = {
        "mean_origin_lab_m": np.array([0.0, 0.0, 1.0]),
        "mean_direction_lab": np.array([0.0, 0.0, -1.0]),
        "transverse_axes_lab": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "spatial_sigma_m": np.zeros(2),
        "divergence_sigma_rad": np.zeros(2),
        "mean_wavelength_A": wavelength_A,
        "wavelength_sigma_A": 0.0,
        "sample_count": 3,
        "polarization_state_id": "UNITY_APPROXIMATION",
    }
    samples = sample_gaussian_source_rays(**source_arguments, seed=17)
    material = _material(wavelength_A)
    configuration = _configuration()

    baseline = build_incident_states(
        samples,
        material,
        compile_instrument(configuration),
        trace_case_id="revision",
    ).states
    repeated = build_incident_states(samples, material, compile_instrument(configuration)).states

    def revisions(states: IncidentStateBatch) -> tuple[str, ...]:
        return (
            states.source_parameter_revision,
            states.source_revision,
            states.sample_geometry_revision,
            states.material_revision,
            states.incident_model_id,
        )

    assert revisions(repeated) == revisions(baseline)
    assert baseline.material_revision == material.material_revision
    assert repeated.status == baseline.status
    np.testing.assert_array_equal(
        repeated.k_film_phase_sample_Ainv, baseline.k_film_phase_sample_Ainv
    )
    np.testing.assert_array_equal(repeated.entrance_amplitude, baseline.entrance_amplitude)
    assert baseline.source_revision == samples.source_revision
    assert baseline.incident_sample_id.tolist() == samples.incident_sample_id.tolist()
    assert baseline.polarization_state_id == samples.polarization_state_id

    changed_source = build_incident_states(
        sample_gaussian_source_rays(**source_arguments, seed=18),
        material,
        compile_instrument(configuration),
    ).states
    assert {
        index
        for index, (before, after) in enumerate(
            zip(revisions(baseline), revisions(changed_source), strict=True)
        )
        if before != after
    } == {1}
    np.testing.assert_array_equal(
        changed_source.k_film_phase_sample_Ainv, baseline.k_film_phase_sample_Ainv
    )

    unbounded = build_incident_states(
        samples,
        material,
        compile_instrument(
            replace(
                configuration,
                sample_support_model_id="unbounded_plane.v1",
                sample_width_m=None,
                sample_length_m=None,
            )
        ),
    ).states
    assert {
        index
        for index, (before, after) in enumerate(
            zip(revisions(baseline), revisions(unbounded), strict=True)
        )
        if before != after
    } == {2}

    changed_material_contract = replace(
        material,
        n_complex=material.n_complex + (-1.0e-8 + 2.0e-9j),
    )
    changed_material = build_incident_states(
        samples,
        changed_material_contract,
        compile_instrument(configuration),
    ).states
    assert {
        index
        for index, (before, after) in enumerate(
            zip(revisions(baseline), revisions(changed_material), strict=True)
        )
        if before != after
    } == {3}

    excluded_configurations = (
        replace(configuration, detector_reference_coordinate_px=(4.0, 6.0)),
        replace(
            configuration,
            sample_from_crystal=RigidTransform(
                np.eye(3),
                np.array([1.0e-4, 0.0, 0.0]),
                FrameId.CRYSTAL,
                FrameId.SAMPLE,
            ),
        ),
        replace(configuration, film_thickness_A=900.0),
    )
    for excluded_configuration in excluded_configurations:
        excluded = build_incident_states(
            samples,
            material,
            compile_instrument(excluded_configuration),
        ).states
        assert revisions(excluded) == revisions(baseline)
        np.testing.assert_array_equal(
            excluded.k_film_phase_sample_Ainv, baseline.k_film_phase_sample_Ainv
        )

    untraced = build_incident_states(
        samples,
        material,
        compile_instrument(configuration),
    )
    assert untraced.traces == ()

    traced = build_incident_states(
        samples,
        material,
        compile_instrument(configuration),
        trace_case_id="revision",
    )
    for field in fields(IncidentStateBatch):
        untraced_value = getattr(untraced.states, field.name)
        traced_value = getattr(traced.states, field.name)
        if isinstance(untraced_value, np.ndarray):
            np.testing.assert_array_equal(traced_value, untraced_value)
        else:
            assert traced_value == untraced_value
    provenance = json.loads(traced.traces[0].provenance)
    assert provenance["source_revision"] == baseline.source_revision
    assert provenance["sample_geometry_revision"] == baseline.sample_geometry_revision
    assert provenance["material_revision"] == baseline.material_revision
    assert provenance["incident_model_id"] == "one_transmitted_channel.v1"
    assert provenance["scientific_provenance"] == (
        "T02 detector-native geometry and planar-interface optics"
    )
