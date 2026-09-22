from __future__ import annotations

import numpy as np

from rasim_next.fitting.joint_geometry import (
    GLOBAL_PARAMETER_NAMES,
    JOINT_GEOMETRY_PARAMETER_NAMES,
    LOCAL_PARAMETER_NAMES,
    NUISANCE_PARAMETER_NAMES,
    JointGeometryState,
)


def test_joint_geometry_parameter_partition_includes_two_independent_pbi2_specimens() -> None:
    assert GLOBAL_PARAMETER_NAMES == (
        "detector_column_tilt_rad",
        "detector_row_tilt_rad",
        "beam_center_column_px",
        "beam_center_row_px",
        "goniometer_axis_pitch_rad",
        "goniometer_axis_yaw_rad",
        "goniometer_pivot_pitch_offset_m",
        "goniometer_pivot_yaw_offset_m",
        "incidence_angle_delta_rad",
    )
    assert LOCAL_PARAMETER_NAMES == (
        "bi2se3_sample_y_tilt_rad",
        "bi2se3_zs_m",
        "bi2te3_sample_x_tilt_rad",
        "bi2te3_sample_y_tilt_rad",
        "bi2te3_zs_m",
        "pbi2_y1_sample_x_tilt_rad",
        "pbi2_y1_sample_y_tilt_rad",
        "pbi2_y1_zs_m",
        "pbi2_y2_sample_x_tilt_rad",
        "pbi2_y2_sample_y_tilt_rad",
        "pbi2_y2_zs_m",
    )
    assert NUISANCE_PARAMETER_NAMES == ("hbn_calibrant_distance_m",)
    assert len(JOINT_GEOMETRY_PARAMETER_NAMES) == 21
    assert len(set(JOINT_GEOMETRY_PARAMETER_NAMES)) == 21

    values = np.arange(len(JOINT_GEOMETRY_PARAMETER_NAMES), dtype=np.float64)
    state = JointGeometryState.from_array(values)
    np.testing.assert_array_equal(state.as_array(), values)
