from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "interactive" / "ewald_sphere_viewer.py"
SPEC = importlib.util.spec_from_file_location("ewald_sphere_viewer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VIEWER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VIEWER
SPEC.loader.exec_module(VIEWER)


def test_embedded_intensity_texture_mosaic_settings_are_declared() -> None:
    assert VIEWER.GAUSSIAN_SIGMA_DEG == 2.0
    assert VIEWER.LORENTZIAN_HWHM_DEG == 0.2
    assert VIEWER.LORENTZIAN_PROBABILITY == 0.1


def test_ki_frame_is_right_handed_and_aligned_with_incident_wavevector() -> None:
    ki = VIEWER.incident_wavevector_Ainv(15.0)
    basis = VIEWER.ki_frame_basis(ki)

    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1.0e-14)
    np.testing.assert_allclose(np.linalg.det(basis), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(basis[:, 2], ki / np.linalg.norm(ki), atol=1.0e-14)


def test_cylinder_intersection_points_satisfy_both_surfaces() -> None:
    ki = VIEWER.incident_wavevector_Ainv(10.0)
    radius_Ainv = 1.75
    branches = VIEWER.cylinder_ewald_intersections_Ainv(radius_Ainv, ki, sample_count=721)
    points = np.concatenate([branch[np.all(np.isfinite(branch), axis=1)] for branch in branches])

    np.testing.assert_allclose(
        points[:, 0] ** 2 + points[:, 1] ** 2,
        radius_Ainv**2,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        np.linalg.norm(points + ki, axis=1),
        VIEWER.WAVE_NUMBER_AINV,
        atol=2.0e-13,
    )


def test_central_rod_has_direct_and_regular_ewald_intersections() -> None:
    ki = VIEWER.incident_wavevector_Ainv(5.0)
    points = VIEWER.central_rod_intersections_Ainv(ki)

    np.testing.assert_allclose(points[0], np.zeros(3), atol=0.0)
    np.testing.assert_allclose(points[1], np.array([0.0, 0.0, -2.0 * ki[2]]), atol=0.0)
    np.testing.assert_allclose(
        np.linalg.norm(points + ki, axis=1),
        VIEWER.WAVE_NUMBER_AINV,
        atol=2.0e-14,
    )


def test_positive_z_segments_exclude_nonpositive_points() -> None:
    points = np.array(
        (
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 1.0),
            (2.0, 0.0, 2.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 3.0),
            (5.0, 0.0, 4.0),
        )
    )

    segments = VIEWER.positive_z_segments(points)

    assert len(segments) == 2
    assert all(np.all(segment[:, 2] > 0.0) for segment in segments)
