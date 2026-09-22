from __future__ import annotations

from pathlib import Path

import numpy as np

from rasim_next.fitting.hbn import (
    HbnRingObservations,
    evaluate_hbn_residual_px,
    fit_hbn_detector_calibration,
)
from rasim_next.io.osc import read_osc


def test_automatic_hbn_trace_qualifies_tracked_calibrant_without_clicks() -> None:
    root = Path(__file__).resolve().parents[1]
    hbn_directory = root / "examples" / "calibration" / "hbn"
    counts = read_osc(hbn_directory / "hBN_calibrant_5m.osc.gz").detector_native_counts
    dark = read_osc(hbn_directory / "darkImg.osc.gz").detector_native_counts
    base_rotation = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))

    observations, calibration = fit_hbn_detector_calibration(
        counts,
        dark,
        base_detector_rotation=base_rotation,
        beam_direction_lab=(0.0, 1.0, 0.0),
        detector_column_pitch_m=1.0e-4,
        detector_row_pitch_m=1.0e-4,
        initial_beam_center_px=(1453.12, 1596.422),
        initial_calibrant_distance_m=0.074,
    )

    assert isinstance(observations, HbnRingObservations)
    assert calibration.success
    assert calibration.jacobian_rank == 5
    assert calibration.scaled_jacobian_condition < 25.0
    assert calibration.residual_rms_px < 1.0
    assert calibration.residual_max_px < 4.0
    assert all(count >= 16 for count in calibration.ring_point_count)
    assert all(coverage >= 0.44 for coverage in calibration.ring_angular_coverage_fraction)
    expected = np.asarray((-0.0066992, -0.0234628, 1452.6944, 1596.7065, 0.0759661))
    tolerance = np.asarray((2e-5, 2e-5, 0.03, 0.03, 3e-6))
    assert np.all(np.abs(calibration.values - expected) <= tolerance)
    residual = evaluate_hbn_residual_px(
        calibration.values,
        observations,
        base_detector_rotation=base_rotation,
        beam_direction_lab=(0.0, 1.0, 0.0),
        detector_column_pitch_m=1.0e-4,
        detector_row_pitch_m=1.0e-4,
    )
    np.testing.assert_allclose(
        np.sqrt(np.mean(residual**2)),
        calibration.residual_rms_px,
        rtol=0.0,
        atol=1e-12,
    )
