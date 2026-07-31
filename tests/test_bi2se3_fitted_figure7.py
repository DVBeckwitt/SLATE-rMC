from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "figures" / "render_bi2se3_fitted_figure7.py"
CONFIG = (
    Path(__file__).parents[1] / "examples" / "bi2se3" / "experiment" / "figure7_recreation.toml"
)
SPEC = importlib.util.spec_from_file_location("render_bi2se3_fitted_figure7", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FIGURE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FIGURE
SPEC.loader.exec_module(FIGURE)


def test_fit_observable_arrays_use_saved_forward_predictions_without_smoothing() -> None:
    document = {
        "fit": {
            "profiles": [
                {
                    "dataset_id": "view-10",
                    "family_m": 1,
                    "integer_L": 5,
                    "root_side_branch_id": 2,
                    "transferred_observed_signal_density_A2_per_rad2": 120.0,
                    "fitted_signal_density_A2_per_rad2": 3.0,
                    "relative_residual": -0.975,
                },
                {
                    "dataset_id": "view-5",
                    "family_m": 0,
                    "integer_L": 3,
                    "root_side_branch_id": None,
                    "transferred_observed_signal_density_A2_per_rad2": 20.0,
                    "fitted_signal_density_A2_per_rad2": 18.0,
                    "relative_residual": -0.1,
                },
            ]
        }
    }

    arrays = FIGURE.fit_observable_arrays(document, dataset_order=("view-5", "view-10"))

    np.testing.assert_array_equal(arrays["observed_signal"], (20.0, 120.0))
    np.testing.assert_array_equal(arrays["fitted_signal"], (18.0, 3.0))
    np.testing.assert_allclose(arrays["relative_residual"], (-0.1, -0.975))
    assert not np.array_equal(arrays["observed_signal"], arrays["fitted_signal"])


def test_detector_region_mask_uses_declared_half_width_and_l_interval() -> None:
    qr_Ainv = np.array([[0.79, 0.80, 1.00, 1.20, 1.21], [1.00, 1.00, 1.00, 1.00, 1.00]])
    L = np.array([[5.0, 5.0, 5.0, 5.0, 5.0], [1.99, 2.0, 8.0, 8.01, 4.0]])
    valid = np.ones_like(qr_Ainv, dtype=bool)
    valid[0, 2] = False

    mask = FIGURE.detector_region_mask(
        qr_Ainv,
        L,
        valid,
        target_qr_Ainv=1.0,
        qr_half_width_Ainv=0.2,
        minimum_L=2.0,
        maximum_L=8.0,
    )

    np.testing.assert_array_equal(
        mask,
        (
            (False, True, False, True, False),
            (False, True, True, False, True),
        ),
    )


def test_tracked_figure_recipe_freezes_the_ambiguous_legacy_regions() -> None:
    settings = FIGURE.load_figure_settings(CONFIG)

    assert settings.material_id == "Bi2Se3"
    assert settings.display_dataset_id == "Bi2Se3-5deg"
    assert settings.crop_shape_rc == (1596, 3000)
    assert settings.profile_shape_rc == (3000, 3000)
    assert tuple(region.family_m for region in settings.regions) == (0, 1, 3, 4)
    assert settings.regions[0].qr_half_width_Ainv == 0.05
    assert settings.regions[1].qr_half_width_Ainv == 0.2125
    assert tuple((region.h, region.k) for region in settings.regions) == (
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
    )
    assert len(settings.labels) == 7
    assert tuple(profile.identity for profile in settings.profiles) == (
        "m1_minus",
        "m1_plus",
        "m3_minus",
        "m3_plus",
        "m4_minus",
        "m4_plus",
        "m0",
    )


def test_figure_recipe_rejects_nonintegral_detector_extents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_text = CONFIG.read_text(encoding="utf-8").replace(
        "crop_rows = 1596", "crop_rows = 1596.5"
    )
    monkeypatch.setattr(Path, "read_text", lambda _path, encoding=None: invalid_text)

    with pytest.raises(TypeError, match="crop_rows must be an integer"):
        FIGURE.load_figure_settings(CONFIG)


def test_detector_cubature_nodes_preserve_bounded_tile_area() -> None:
    candidate = np.zeros((4, 6), dtype=bool)
    candidate[0, 0] = True

    column, row, area = FIGURE._detector_cubature_nodes(
        (4, 6),
        tile_size_px=2,
        gauss_order=2,
        candidate_pixel_mask=candidate,
    )

    assert column.shape == row.shape == area.shape == (16,)
    assert np.all((column > -0.5) & (column < 3.5))
    assert np.all((row > -0.5) & (row < 3.5))
    assert np.sum(area) == pytest.approx(16.0, rel=0.0, abs=2.0e-15)
