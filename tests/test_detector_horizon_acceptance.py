import numpy as np

from rasim_next.fitting.specular import DetectorHorizonAcceptance


def test_detector_horizon_acceptance_keeps_all_finite_m0_coordinates() -> None:
    acceptance = DetectorHorizonAcceptance(
        offspecular_air_exit_guard_rad=np.deg2rad(1.0),
    )
    air_exit_rad = np.deg2rad(np.array([-0.1, 0.0, 0.4, 0.9, 1.1]))

    np.testing.assert_array_equal(
        acceptance.offspecular_mask(air_exit_rad),
        np.array([False, False, False, False, True]),
    )
    np.testing.assert_array_equal(
        acceptance.m0_detector_mask(air_exit_rad),
        np.ones(air_exit_rad.shape, dtype=np.bool_),
    )
