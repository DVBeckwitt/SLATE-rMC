from __future__ import annotations

import numpy as np

from rasim_next.measurement.continuous_regions import (
    ContinuousDetectorChartAreaMeasure,
    ContinuousRegionQuadrature,
    compile_continuous_rectangle_quadrature,
    compile_native_pixel_region_projection,
)
from rasim_next.measurement.reciprocal_profiles import (
    LayeredReciprocalFrame,
    ReciprocalProfileRegion,
    accumulate_binned_samples,
    reciprocal_profile_membership,
)
from rasim_next.measurement.region_observations import (
    OffSpecularBand,
    OffSpecularBandLayout,
    SpecularAngularProfileRegion,
    offspecular_region_membership,
    specular_angular_membership,
)


class _AffineDetectorChart:
    revision = "affine-detector-chart.test.v1"

    def map_detector_area(
        self,
        first_coordinate: np.ndarray,
        second_coordinate: np.ndarray,
    ) -> ContinuousDetectorChartAreaMeasure:
        return ContinuousDetectorChartAreaMeasure(
            column_px=2.0 * first_coordinate + second_coordinate,
            row_px=first_coordinate - 3.0 * second_coordinate,
            detector_area_jacobian_px2_per_chart2=np.full(first_coordinate.shape, 7.0),
            valid=np.ones(first_coordinate.shape, dtype=np.bool_),
        )


def test_continuous_rectangle_quadrature_integrates_rows_without_native_pixels() -> None:
    quadrature = compile_continuous_rectangle_quadrature(
        chart=_AffineDetectorChart(),
        first_coordinate_bounds=np.asarray(((0.0, 1.0), (1.0, 2.0))),
        second_coordinate_bounds=np.asarray(((-1.0, 1.0), (0.0, 3.0))),
        observation_row=np.asarray((1, 0), dtype=np.int64),
        observation_count=2,
        gauss_order=3,
        background_coordinate_axis=0,
    )

    density = 5.0 + 2.0 * quadrature.column_px - quadrature.row_px
    np.testing.assert_allclose(
        quadrature.integrate_density(density),
        (357.0, 91.0),
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(quadrature.observation_measure_px2, (21.0, 14.0))
    np.testing.assert_allclose(quadrature.observation_background_coordinate, (1.5, 0.5))
    assert not hasattr(quadrature, "flat_pixel_index")
    assert np.any(quadrature.column_px != np.round(quadrature.column_px))
    assert not quadrature.column_px.flags.writeable


def test_native_pixel_projection_propagates_fractional_count_covariance() -> None:
    quadrature = ContinuousRegionQuadrature(
        column_px=np.asarray((0.0, 0.1, -0.1, 1.0)),
        row_px=np.zeros(4),
        detector_area_weight_px2=np.asarray((0.25, 0.25, 0.5, 1.0)),
        observation_row=np.asarray((0, 0, 1, 1)),
        background_coordinate=np.zeros(4),
        observation_count=2,
        chart_revision="native-pixel-projection.test.v1",
    )
    projection = compile_native_pixel_region_projection(quadrature, (1, 2))
    mass, covariance = projection.integrate_counts(np.asarray(((4, 9),)))

    np.testing.assert_allclose(projection.observation_measure_px2, (0.5, 1.5))
    np.testing.assert_allclose(mass, (2.0, 11.0))
    np.testing.assert_allclose(covariance, ((1.0, 1.0), (1.0, 10.0)))
    constant_mass, _ = projection.integrate_counts(np.full((1, 2), 7.0))
    np.testing.assert_allclose(constant_mass, 7.0 * projection.observation_measure_px2)
    _, empty_covariance = projection.integrate_counts(np.zeros((1, 2)))
    np.testing.assert_allclose(empty_covariance, ((0.25, 0.25), (0.25, 1.25)))
    assert projection.flat_pixel_index.tolist() == [0, 1]


def test_continuous_rectangle_quadrature_refines_each_chart_axis_independently() -> None:
    quadrature = compile_continuous_rectangle_quadrature(
        chart=_AffineDetectorChart(),
        first_coordinate_bounds=np.asarray(((0.0, 1.0),)),
        second_coordinate_bounds=np.asarray(((-1.0, 1.0),)),
        observation_row=np.asarray((0,), dtype=np.int64),
        observation_count=1,
        gauss_order=3,
        background_coordinate_axis=0,
        subdivision_count=(2, 5),
    )

    assert quadrature.column_px.size == 3 * 2 * 3 * 5
    np.testing.assert_allclose(quadrature.observation_measure_px2, (14.0,))
    recovered_first = (3.0 * quadrature.column_px + quadrature.row_px) / 7.0
    recovered_second = (quadrature.column_px - 2.0 * quadrature.row_px) / 7.0
    assert np.unique(np.round(recovered_first, decimals=14)).size == 3 * 2
    assert np.unique(np.round(recovered_second, decimals=14)).size == 3 * 5


def test_continuous_rectangle_quadrature_regularizes_an_interior_radial_fold() -> None:
    quadrature = compile_continuous_rectangle_quadrature(
        chart=_AffineDetectorChart(),
        first_coordinate_bounds=np.asarray(((1.0, 3.0),)),
        second_coordinate_bounds=np.asarray(((-1.0, 2.0),)),
        observation_row=np.asarray((0,), dtype=np.int64),
        observation_count=1,
        gauss_order=12,
        background_coordinate_axis=0,
        subdivision_count=(4, 2),
        first_coordinate_squared_fold_center=np.asarray((2.0,)),
    )

    # The transformed nodes integrate the same continuous chart area and first
    # moment while placing no node exactly on the integrable fold.
    np.testing.assert_allclose(quadrature.observation_measure_px2, (42.0,), rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        quadrature.observation_background_coordinate,
        (2.0,),
        rtol=0.0,
        atol=2.0e-12,
    )
    recovered_first = (3.0 * quadrature.column_px + quadrature.row_px) / 7.0
    assert np.all(recovered_first != 2.0)


def test_continuous_rectangle_quadrature_rejects_a_fold_outside_the_interval() -> None:
    with np.testing.assert_raises_regex(ValueError, "fold center"):
        compile_continuous_rectangle_quadrature(
            chart=_AffineDetectorChart(),
            first_coordinate_bounds=np.asarray(((1.0, 3.0),)),
            second_coordinate_bounds=np.asarray(((-1.0, 2.0),)),
            observation_row=np.asarray((0,), dtype=np.int64),
            observation_count=1,
            gauss_order=3,
            background_coordinate_axis=0,
            first_coordinate_squared_fold_center=np.asarray((4.0,)),
        )


def test_layered_reciprocal_frame_maps_sample_q_without_axis_assumptions() -> None:
    reciprocal_basis_Ainv = np.array(
        ((2.0, 0.0, 0.4), (0.0, 3.0, -0.2), (0.0, 0.0, 4.0)),
        dtype=np.float64,
    )
    sample_from_crystal = np.array(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    frame = LayeredReciprocalFrame(
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        sample_from_crystal_rotation=sample_from_crystal,
        axial_basis_index=2,
    )
    hkl = np.array(((1.0, 1.0, 2.5), (0.0, 0.0, 4.0)), dtype=np.float64)
    q_crystal = hkl @ reciprocal_basis_Ainv.T
    q_sample = q_crystal @ sample_from_crystal.T

    qr_Ainv, axial_coordinate = frame.coordinates(q_sample)

    axial_axis = reciprocal_basis_Ainv[:, 2]
    axial_axis /= np.linalg.norm(axial_axis)
    expected_qr = np.linalg.norm(
        q_crystal - (q_crystal @ axial_axis)[:, None] * axial_axis,
        axis=1,
    )
    np.testing.assert_allclose(qr_Ainv, expected_qr, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(axial_coordinate, hkl[:, 2], rtol=0.0, atol=2.0e-15)
    assert not qr_Ainv.flags.writeable
    assert not axial_coordinate.flags.writeable


def test_reciprocal_profile_membership_uses_one_branch_and_declared_sidebands() -> None:
    region = ReciprocalProfileRegion(
        identity="m1+",
        qr_center_Ainv=1.0,
        qr_half_width_Ainv=0.2,
        axial_bin_edges=np.array((2.0, 3.0, 4.0)),
        detector_column_interval_px=(0.0, 5.0),
        sideband_gap_Ainv=0.1,
        sideband_width_Ainv=0.2,
    )
    qr_Ainv = np.array((1.0, 1.0, 0.6, 1.4, 1.0, 1.0, 1.0))
    axial = np.array((2.0, 3.2, 2.2, 3.4, 4.0, 2.4, 2.4))
    column = np.array((1.0, 4.0, 2.0, 3.0, 2.0, 5.0, 1.0))
    valid = np.array((True, True, True, True, True, True, False))

    membership = reciprocal_profile_membership(
        qr_Ainv,
        axial,
        column,
        valid,
        region=region,
    )

    np.testing.assert_array_equal(membership.signal_bin_index, (0, 1, -1, -1, 1, -1, -1))
    np.testing.assert_array_equal(
        membership.sideband_bin_index,
        (-1, -1, 0, 1, -1, -1, -1),
    )
    assert not membership.signal_bin_index.flags.writeable
    assert not membership.sideband_bin_index.flags.writeable


def test_accumulate_binned_samples_keeps_signal_and_measure_separate() -> None:
    integral = accumulate_binned_samples(
        values=np.array((2.0, 5.0, 11.0, 100.0)),
        measure_weights=np.array((0.5, 1.5, 2.0, 7.0)),
        bin_index=np.array((0, 0, 1, -1)),
        bin_count=2,
    )

    np.testing.assert_allclose(integral.signal_sum, (8.5, 22.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(integral.measure_sum, (2.0, 2.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(integral.mean, (4.25, 11.0), rtol=0.0, atol=0.0)
    assert not integral.signal_sum.flags.writeable
    assert not integral.measure_sum.flags.writeable
    assert not integral.mean.flags.writeable


def test_specular_profile_membership_bins_two_theta_and_selects_phi_anchors() -> None:
    region = SpecularAngularProfileRegion(
        identity="m0",
        two_theta_bin_edges_rad=np.array((0.0, 0.1, 0.2)),
        phi_signal_interval_rad=(-0.05, 0.05),
        phi_background_intervals_rad=((-0.15, -0.08), (0.08, 0.15)),
    )
    membership = specular_angular_membership(
        two_theta_rad=np.array((0.02, 0.12, 0.12, 0.12, 0.12)),
        phi_rad=np.array((0.0, 0.04, -0.10, 0.10, 0.3)),
        valid=np.ones(5, dtype=np.bool_),
        region=region,
    )

    np.testing.assert_array_equal(membership.signal_bin_index, (0, 1, -1, -1, -1))
    np.testing.assert_array_equal(membership.background_bin_index, (-1, -1, 1, 1, -1))


def test_offspecular_layout_rejects_background_that_contains_another_signal() -> None:
    with np.testing.assert_raises_regex(ValueError, "background.*m4"):
        OffSpecularBandLayout(
            axial_bin_edges=np.array((2.0, 3.0, 4.0)),
            detector_column_interval_px=(-0.5, 100.0),
            signal_bands=(
                OffSpecularBand("m3", (2.8, 3.2)),
                OffSpecularBand("m4", (3.3, 3.7)),
            ),
            background_intervals_Ainv=((3.4, 3.5),),
        )


def test_offspecular_joint_layout_keeps_signal_and_clean_anchors_distinct() -> None:
    layout = OffSpecularBandLayout(
        axial_bin_edges=np.array((2.0, 3.0, 4.0)),
        detector_column_interval_px=(-0.5, 100.0),
        signal_bands=(
            OffSpecularBand("m3", (2.8, 3.2)),
            OffSpecularBand("m4", (3.3, 3.7)),
        ),
        background_intervals_Ainv=((2.5, 2.7), (3.8, 4.0)),
    )
    membership = offspecular_region_membership(
        qr_Ainv=np.array((3.0, 3.5, 2.6, 3.9, 3.0)),
        axial_coordinate=np.array((2.2, 3.2, 2.2, 3.2, 4.0)),
        detector_column_px=np.array((10.0, 10.0, 10.0, 10.0, 10.0)),
        valid=np.ones(5, dtype=np.bool_),
        layout=layout,
    )

    np.testing.assert_array_equal(membership.signal_bin_index["m3"], (0, -1, -1, -1, 1))
    np.testing.assert_array_equal(membership.signal_bin_index["m4"], (-1, 1, -1, -1, -1))
    np.testing.assert_array_equal(membership.background_bin_index, (-1, -1, 0, 1, -1))
