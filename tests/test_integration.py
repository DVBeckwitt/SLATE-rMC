from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial import ConvexHull, QhullError

from rasim_next.core.contracts import DetectorHitBatch
from rasim_next.core.frames import FrameId
from rasim_next.core.transforms import RigidTransform
from rasim_next.geometry import (
    AngleFrame,
    InstrumentConfiguration,
    compile_instrument,
    detector_coordinates_to_angles,
)
from rasim_next.measurement import (
    AngleBinGrid,
    compile_detector_angle_projector,
    project_normalized_angle_field,
    to_increasing_phi,
)
from rasim_next.render.deposition import deposit_bilinear


def _instrument(
    *,
    shape_rc: tuple[int, int] = (3, 4),
    reference_cr: tuple[float, float] = (1.5, 1.0),
    detector_rotation: np.ndarray | None = None,
) -> object:
    identity = np.eye(3)
    zero = np.zeros(3)
    rotation = identity if detector_rotation is None else detector_rotation
    configuration = InstrumentConfiguration(
        axis_rotations=(),
        lab_from_goniometer_zero=RigidTransform(identity, zero, FrameId.GONIOMETER, FrameId.LAB),
        goniometer_from_sample=RigidTransform(identity, zero, FrameId.SAMPLE, FrameId.GONIOMETER),
        sample_from_crystal=RigidTransform(identity, zero, FrameId.CRYSTAL, FrameId.SAMPLE),
        lab_from_detector=RigidTransform(
            rotation,
            [1.1e-3, -0.7e-3, 0.82],
            FrameId.DETECTOR,
            FrameId.LAB,
        ),
        detector_shape_rc=shape_rc,
        detector_row_pitch_m=3.1e-4,
        detector_column_pitch_m=1.7e-4,
        detector_reference_coordinate_px=reference_cr,
        sample_width_m=4.0e-4,
        sample_length_m=6.0e-4,
        film_thickness_A=500.0,
    )
    return compile_instrument(configuration)


def _frame(origin_lab_m: np.ndarray | list[float] | None = None) -> AngleFrame:
    return AngleFrame(
        origin_lab_m=np.zeros(3) if origin_lab_m is None else origin_lab_m,
        row_down_lab=np.array([0.0, 1.0, 0.0]),
        column_right_lab=np.array([1.0, 0.0, 0.0]),
        direct_beam_lab=np.array([0.0, 0.0, 1.0]),
        revision="integration-angle-frame.v1",
    )


def _full_grid(instrument: object, frame: AngleFrame, *, radial_bins: int = 5) -> AngleBinGrid:
    rows, columns = instrument.detector_shape_rc
    corner_column, corner_row = np.meshgrid(
        np.arange(columns + 1, dtype=np.float64) - 0.5,
        np.arange(rows + 1, dtype=np.float64) - 0.5,
    )
    angles = detector_coordinates_to_angles(
        corner_column,
        corner_row,
        instrument=instrument,
        angle_frame=frame,
    )
    assert np.all(angles.valid)
    theta_max = float(np.max(angles.two_theta_rad))
    return AngleBinGrid(
        two_theta_edges_rad=np.linspace(0.0, np.nextafter(theta_max, np.inf), radial_bins + 1),
        chi_raw_edges_rad=np.linspace(-np.pi, np.pi, 17),
        revision="integration-grid.v1",
    )


def _unwrap(raw_chi: np.ndarray) -> np.ndarray:
    result = np.array(raw_chi, dtype=np.float64, copy=True)
    for index in range(1, result.size):
        delta = (raw_chi[index] - raw_chi[index - 1] + np.pi) % (2.0 * np.pi) - np.pi
        result[index] = result[index - 1] + delta
    return result


def _cross(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _inside_convex(point: np.ndarray, polygon: np.ndarray) -> bool:
    edge = np.roll(polygon, -1, axis=0) - polygon
    offset = point - polygon
    crosses = edge[:, 0] * offset[:, 1] - edge[:, 1] * offset[:, 0]
    return bool(np.all(crosses >= -2e-14) or np.all(crosses <= 2e-14))


def _segment_intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray | None:
    first_delta = first_end - first_start
    second_delta = second_end - second_start
    denominator = _cross(first_delta, second_delta)
    if abs(denominator) <= 1e-18:
        return None
    offset = second_start - first_start
    first_fraction = _cross(offset, second_delta) / denominator
    second_fraction = _cross(offset, first_delta) / denominator
    if -2e-14 <= first_fraction <= 1.0 + 2e-14 and -2e-14 <= second_fraction <= 1.0 + 2e-14:
        return first_start + first_fraction * first_delta
    return None


def _independent_intersection_area(polygon: np.ndarray, rectangle: np.ndarray) -> float:
    lower = rectangle[0]
    upper = rectangle[2]
    points: list[np.ndarray] = []
    for point in polygon:
        if np.all(point >= lower - 2e-14) and np.all(point <= upper + 2e-14):
            points.append(point)
    for point in rectangle:
        if _inside_convex(point, polygon):
            points.append(point)
    for subject_index in range(polygon.shape[0]):
        subject_start = polygon[subject_index]
        subject_end = polygon[(subject_index + 1) % polygon.shape[0]]
        for clip_index in range(4):
            intersection = _segment_intersection(
                subject_start,
                subject_end,
                rectangle[clip_index],
                rectangle[(clip_index + 1) % 4],
            )
            if intersection is not None:
                points.append(intersection)
    unique = [
        point
        for index, point in enumerate(points)
        if not any(np.linalg.norm(point - earlier) <= 2e-13 for earlier in points[:index])
    ]
    if len(unique) < 3:
        return 0.0
    try:
        return float(ConvexHull(np.asarray(unique)).volume)
    except QhullError:
        return 0.0


def _independent_pixel_weights(
    column: int,
    row: int,
    *,
    instrument: object,
    frame: AngleFrame,
    grid: AngleBinGrid,
    pole_cr: tuple[float, float] | None = None,
) -> np.ndarray:
    corner_column = np.array([column - 0.5, column + 0.5, column + 0.5, column - 0.5])
    corner_row = np.array([row - 0.5, row - 0.5, row + 0.5, row + 0.5])
    angles = detector_coordinates_to_angles(
        corner_column,
        corner_row,
        instrument=instrument,
        angle_frame=frame,
    )
    physical_corners = np.column_stack((corner_column, corner_row))
    contains_pole = pole_cr is not None and (
        column - 0.5 <= pole_cr[0] <= column + 0.5 and row - 0.5 <= pole_cr[1] <= row + 0.5
    )
    pieces = []
    if contains_pole:
        pole = np.asarray(pole_cr)
        for first in range(4):
            second = (first + 1) % 4
            if (
                abs(
                    _cross(
                        physical_corners[first] - pole,
                        physical_corners[second] - pole,
                    )
                )
                <= 1e-15
            ):
                continue
            assert angles.azimuth_valid[first] and angles.azimuth_valid[second]
            chi = _unwrap(angles.chi_raw_rad[[first, second]])
            pieces.append(
                np.array(
                    [
                        [0.0, chi[0]],
                        [angles.two_theta_rad[first], chi[0]],
                        [angles.two_theta_rad[second], chi[1]],
                        [0.0, chi[1]],
                    ]
                )
            )
    else:
        assert np.all(angles.azimuth_valid)
        for indices in ((0, 1, 2), (0, 2, 3)):
            index = np.asarray(indices)
            pieces.append(
                np.column_stack((angles.two_theta_rad[index], _unwrap(angles.chi_raw_rad[index])))
            )
    piece_area = math.fsum(float(ConvexHull(piece).volume) for piece in pieces)
    expected = np.zeros(grid.shape, dtype=np.float64)
    period = 2.0 * np.pi
    for chi_bin in range(grid.shape[0]):
        for theta_bin in range(grid.shape[1]):
            theta_lower = grid.two_theta_edges_rad[theta_bin]
            theta_upper = grid.two_theta_edges_rad[theta_bin + 1]
            for shift in range(-2, 3):
                chi_lower = grid.chi_raw_edges_rad[chi_bin] + shift * period
                chi_upper = grid.chi_raw_edges_rad[chi_bin + 1] + shift * period
                rectangle = np.array(
                    [
                        [theta_lower, chi_lower],
                        [theta_upper, chi_lower],
                        [theta_upper, chi_upper],
                        [theta_lower, chi_upper],
                    ]
                )
                expected[chi_bin, theta_bin] += math.fsum(
                    _independent_intersection_area(piece, rectangle) for piece in pieces
                )
    return expected / piece_area


def _dense_projector(projector: object) -> np.ndarray:
    bins = int(np.prod(projector.grid.shape))
    pixels = int(np.prod(projector.instrument.detector_shape_rc))
    dense = np.zeros((bins, pixels), dtype=np.float64)
    np.add.at(
        dense, (projector.coverage_bin_index, projector.coverage_pixel_index), projector.weight
    )
    return dense


def test_sparse_projector_matches_independent_polygon_oracle_across_seam() -> None:
    angle = np.deg2rad(7.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    instrument = _instrument(shape_rc=(2, 3), detector_rotation=rotation)
    frame = _frame([4.0e-3, -0.7e-3, 0.0])
    grid = _full_grid(instrument, frame, radial_bins=4)
    near_canonical_theta = grid.two_theta_edges_rad.copy()
    near_canonical_chi = grid.chi_raw_edges_rad.copy()
    near_canonical_theta[0] = np.nextafter(0.0, np.inf)
    near_canonical_chi[[0, -1]] = np.nextafter([-np.pi, np.pi], 0.0)
    canonicalized = AngleBinGrid(near_canonical_theta, near_canonical_chi, "canonicalized.v1")
    assert canonicalized.two_theta_edges_rad[0] == 0.0
    np.testing.assert_array_equal(canonicalized.chi_raw_edges_rad[[0, -1]], [-np.pi, np.pi])
    projector = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=grid,
    )
    dense = _dense_projector(projector)

    seam_pixel: int | None = None

    for row in range(2):
        for column in range(3):
            pixel = row * 3 + column
            pixel_angles = detector_coordinates_to_angles(
                [column - 0.5, column + 0.5, column + 0.5, column - 0.5],
                [row - 0.5, row - 0.5, row + 0.5, row + 0.5],
                instrument=instrument,
                angle_frame=frame,
            )
            if np.ptp(pixel_angles.chi_raw_rad) > np.pi:
                seam_pixel = pixel
            expected = _independent_pixel_weights(
                column,
                row,
                instrument=instrument,
                frame=frame,
                grid=grid,
            )
            np.testing.assert_allclose(
                dense[:, pixel].reshape(grid.shape),
                expected,
                rtol=3e-11,
                atol=3e-13,
            )
    assert seam_pixel is not None
    seam_column = dense[:, seam_pixel].reshape(grid.shape)
    assert np.any(seam_column[0] > 0.0)
    assert np.any(seam_column[-1] > 0.0)
    np.testing.assert_allclose(np.sum(dense, axis=0), 1.0, rtol=0.0, atol=3e-12)
    np.testing.assert_allclose(projector.lost_support, 0.0, rtol=0.0, atol=3e-12)
    pair_key = projector.coverage_pixel_index * math.prod(grid.shape) + projector.coverage_bin_index
    assert np.all(np.diff(pair_key) > 0)
    assert np.all(projector.weight > 0.0)
    assert not projector.weight.flags.writeable
    assert not projector.coverage_bin_index.flags.writeable
    assert not projector.coverage_pixel_index.flags.writeable

    with pytest.raises(ValueError, match="cache_key"):
        replace(projector, cache_key="stale.v1")
    split_revision_a = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=replace(frame, revision="ab"),
        grid=replace(grid, revision="c"),
    )
    split_revision_b = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=replace(frame, revision="a"),
        grid=replace(grid, revision="bc"),
    )
    changed_mask = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=grid,
        detector_valid_mask=np.array([[False, True, True], [True, True, True]]),
    )
    assert (
        len(
            {
                projector.cache_key,
                split_revision_a.cache_key,
                split_revision_b.cache_key,
                changed_mask.cache_key,
            }
        )
        == 4
    )


def test_accepted_bilinear_deposition_precedes_angle_reduction_at_all_edge_strips() -> None:
    instrument = _instrument(shape_rc=(2, 3))
    frame = _frame([4.0e-3, -0.7e-3, 0.0])
    grid = _full_grid(instrument, frame, radial_bins=3)
    projector = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=grid,
    )
    column_px = np.array([-0.25, 2.25, 0.75, 1.75, 1.0, 1.3])
    row_px = np.array([0.25, 0.75, -0.25, 1.25, 1.0, 0.4])
    event_id = np.arange(column_px.size, dtype=np.int64)
    hits = DetectorHitBatch(
        event_id=event_id,
        column_px=column_px,
        row_px=row_px,
        pixel_solid_angle_sr=np.zeros(column_px.size),
        valid=np.ones(column_px.size, dtype=np.bool_),
    )
    mass = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    exposure = np.array([1.0, 4.0, 2.0, 3.0, 5.0, 6.0])
    deposited = deposit_bilinear(
        hits,
        event_row=event_id,
        event_id=event_id,
        assigned_mass_A2=mass,
        detector_shape_rc=(2, 3),
    )
    deposited_exposure = deposit_bilinear(
        hits,
        event_row=event_id,
        event_id=event_id,
        assigned_mass_A2=exposure,
        detector_shape_rc=(2, 3),
    )

    direct_image = np.array([[2.0625, 8.2725, 2.9025], [0.375, 15.9525, 7.185]])
    np.testing.assert_allclose(deposited.image_A2, direct_image, rtol=0.0, atol=1e-15)
    assert deposited.clipped_mass_A2 == pytest.approx(4.25, rel=0.0, abs=2e-15)
    assert deposited.deposited_mass_A2 + deposited.clipped_mass_A2 == pytest.approx(mass.sum())

    matrix = _dense_projector(projector)
    expected_S = (matrix @ direct_image.ravel()).reshape(grid.shape)
    expected_N = (matrix @ deposited_exposure.image_A2.ravel()).reshape(grid.shape)
    expected_I = np.zeros(grid.shape)
    np.divide(expected_S, expected_N, out=expected_I, where=expected_N > 0.0)
    field = project_normalized_angle_field(
        projector,
        deposited.image_A2,
        deposited_exposure.image_A2,
    )
    np.testing.assert_allclose(field.S, expected_S, rtol=3e-11, atol=3e-13)
    np.testing.assert_allclose(field.N, expected_N, rtol=3e-11, atol=3e-13)
    np.testing.assert_allclose(field.I, expected_I, rtol=4e-11, atol=3e-13)


@pytest.mark.parametrize("reference_cr", [(1.0, 1.0), (1.5, 1.0), (1.5, 1.5)])
def test_direct_beam_pole_ties_cover_full_detector_support(
    reference_cr: tuple[float, float],
) -> None:
    base = _instrument(shape_rc=(3, 3), reference_cr=reference_cr)
    instrument = replace(
        base,
        lab_from_detector=RigidTransform(
            np.eye(3), [0.0, 0.0, 0.82], FrameId.DETECTOR, FrameId.LAB
        ),
    )
    frame = _frame()
    projector = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=_full_grid(instrument, frame),
    )
    dense = _dense_projector(projector)
    np.testing.assert_allclose(projector.lost_support, 0.0, rtol=0.0, atol=4e-12)
    np.testing.assert_allclose(
        np.sum(dense, axis=0) + projector.lost_support.ravel(),
        1.0,
        rtol=0.0,
        atol=4e-12,
    )
    touching = [
        row * 3 + column
        for row in range(3)
        for column in range(3)
        if column - 0.5 <= reference_cr[0] <= column + 0.5
        and row - 0.5 <= reference_cr[1] <= row + 0.5
    ]
    assert len(touching) in (1, 2, 4)
    supports = []
    for pixel in touching:
        row, column = divmod(pixel, 3)
        expected = _independent_pixel_weights(
            column,
            row,
            instrument=instrument,
            frame=frame,
            grid=projector.grid,
            pole_cr=reference_cr,
        )
        actual = dense[:, pixel].reshape(projector.grid.shape)
        np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=4e-13)
        supports.append(np.flatnonzero(np.any(actual > 0.0, axis=1)))
    assert all(support.size == projector.grid.shape[0] // len(touching) for support in supports)
    np.testing.assert_array_equal(
        np.unique(np.concatenate(supports)),
        np.arange(projector.grid.shape[0]),
    )

    if reference_cr == (1.5, 1.0):
        for shifted_column in (np.nextafter(1.5, -np.inf), np.nextafter(1.5, np.inf)):
            shifted = replace(
                instrument,
                detector_reference_coordinate_px=(shifted_column, 1.0),
            )
            shifted_projector = compile_detector_angle_projector(
                instrument=shifted,
                angle_frame=frame,
                grid=projector.grid,
            )
            np.testing.assert_allclose(
                _dense_projector(shifted_projector),
                dense,
                rtol=0.0,
                atol=4e-12,
            )


def test_normalized_field_freezes_masks_losses_divide_order_and_phi_permutation() -> None:
    instrument = _instrument(shape_rc=(2, 3))
    frame = _frame([4.0e-3, -0.7e-3, 0.0])
    full_grid = _full_grid(instrument, frame, radial_bins=2)
    detector_mask = np.array([[True, True, False], [True, True, True]])
    angle_mask = np.ones(full_grid.shape, dtype=np.bool_)
    angle_mask[0, 0] = False
    projector = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=full_grid,
        detector_valid_mask=detector_mask,
        angle_bin_valid_mask=angle_mask,
    )
    signal = np.array([[2.0, 9.0, 7.0], [5.0, 4.0, 11.0]])
    normalization = np.array([[1.0, 3.0, 2.0], [5.0, 2.0, 4.0]])
    field = project_normalized_angle_field(projector, signal, normalization)
    dense = _dense_projector(projector)
    expected_signal = (dense @ np.where(detector_mask, signal, 0.0).ravel()).reshape(
        full_grid.shape
    )
    expected_normalization = (dense @ np.where(detector_mask, normalization, 0.0).ravel()).reshape(
        full_grid.shape
    )
    angle_excluded_signal = float(expected_signal[~angle_mask].sum())
    angle_excluded_normalization = float(expected_normalization[~angle_mask].sum())
    expected_signal[~angle_mask] = 0.0
    expected_normalization[~angle_mask] = 0.0
    expected_valid = angle_mask & (expected_normalization > 0.0)
    expected_intensity = np.zeros(full_grid.shape)
    np.divide(
        expected_signal,
        expected_normalization,
        out=expected_intensity,
        where=expected_valid,
    )

    np.testing.assert_allclose(field.S, expected_signal, rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(field.N, expected_normalization, rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(field.I, expected_intensity, rtol=0.0, atol=3e-14)
    np.testing.assert_array_equal(field.valid, expected_valid)
    assert field.detector_mask_excluded_signal == pytest.approx(7.0)
    assert field.detector_mask_excluded_normalization == pytest.approx(2.0)
    assert field.angle_mask_excluded_signal == pytest.approx(angle_excluded_signal)
    assert field.angle_mask_excluded_normalization == pytest.approx(angle_excluded_normalization)
    assert (
        field.S.sum() + field.angle_mask_excluded_signal + field.angular_lost_signal
        == pytest.approx(signal[detector_mask].sum(), abs=2e-12)
    )
    assert (
        field.N.sum() + field.angle_mask_excluded_normalization + field.angular_lost_normalization
        == pytest.approx(normalization[detector_mask].sum(), abs=2e-12)
    )

    ratio_first = np.zeros_like(signal)
    np.divide(signal, normalization, out=ratio_first, where=normalization > 0.0)
    wrong = (dense @ ratio_first.ravel()).reshape(full_grid.shape)
    assert np.max(np.abs(wrong[expected_valid] - field.I[expected_valid])) > 1e-3

    phi = to_increasing_phi(field)
    assert np.any(full_grid.chi_raw_edges_rad == np.pi / 2.0)
    mapped_phi = (-np.pi / 2.0 - full_grid.chi_raw_centers_rad + np.pi) % (2.0 * np.pi) - np.pi
    permutation = np.argsort(mapped_phi, kind="stable")
    assert np.all(np.diff(phi.grid.phi_centers_rad) > 0.0)
    np.testing.assert_allclose(
        phi.grid.phi_centers_rad,
        mapped_phi[permutation],
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_array_equal(phi.S, field.S[permutation])
    np.testing.assert_array_equal(phi.N, field.N[permutation])
    np.testing.assert_array_equal(phi.I, field.I[permutation])
    np.testing.assert_array_equal(phi.valid, field.valid[permutation])
    np.testing.assert_array_equal(
        phi.angle_bin_valid_mask,
        field.angle_bin_valid_mask[permutation],
    )
    assert phi.projector_cache_key == field.projector_cache_key == projector.cache_key
    assert phi.angular_lost_signal == field.angular_lost_signal
    for array_value in (field.S, phi.valid, projector.weight):
        assert not array_value.flags.writeable

    zero = project_normalized_angle_field(projector, signal, np.zeros_like(normalization))
    assert not np.any(zero.valid)
    assert not np.any(zero.I)
    np.testing.assert_array_equal(zero.S, field.S)
    np.testing.assert_array_equal(zero.N, np.zeros_like(field.N))
    with pytest.raises(ValueError, match="nonnegative"):
        project_normalized_angle_field(projector, -signal, normalization)

    clipped_grid = AngleBinGrid(
        two_theta_edges_rad=np.linspace(
            0.0,
            0.55 * full_grid.two_theta_edges_rad[-1],
            3,
        ),
        chi_raw_edges_rad=full_grid.chi_raw_edges_rad,
        revision="clipped-grid.v1",
    )
    clipped = compile_detector_angle_projector(
        instrument=instrument,
        angle_frame=frame,
        grid=clipped_grid,
    )
    clipped_field = project_normalized_angle_field(clipped, signal, normalization)
    assert np.any(clipped.lost_support > 0.0)
    clipped_dense = _dense_projector(clipped)
    oracle_column = _independent_pixel_weights(
        0,
        0,
        instrument=instrument,
        frame=frame,
        grid=clipped_grid,
    )
    np.testing.assert_allclose(
        clipped_dense[:, 0].reshape(clipped_grid.shape),
        oracle_column,
        rtol=3e-11,
        atol=3e-13,
    )
    assert clipped.lost_support[0, 0] == pytest.approx(
        1.0 - float(oracle_column.sum()),
        abs=3e-12,
    )
    assert float(clipped_dense[:, 0].sum()) < 1.0
    assert clipped_field.S.sum() + clipped_field.angular_lost_signal == pytest.approx(
        signal.sum(), abs=2e-12
    )
    assert clipped_field.N.sum() + clipped_field.angular_lost_normalization == pytest.approx(
        normalization.sum(), abs=2e-12
    )
