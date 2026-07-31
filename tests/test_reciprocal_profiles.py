from __future__ import annotations

import numpy as np

from rasim_next.measurement.reciprocal_profiles import (
    LayeredReciprocalFrame,
    ReciprocalProfileRegion,
    accumulate_binned_samples,
    reciprocal_profile_membership,
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
