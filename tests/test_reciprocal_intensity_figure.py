from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "figures" / "render_bi2se3_reciprocal_intensity.py"
SPEC = importlib.util.spec_from_file_location("render_bi2se3_reciprocal_intensity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FIGURE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FIGURE
SPEC.loader.exec_module(FIGURE)


def test_final_reciprocal_figure_defaults_are_locked() -> None:
    settings = FIGURE.ReciprocalFigureSettings()

    assert settings.incidence_deg == 10.0
    assert settings.gaussian_sigma_deg == 2.0
    assert settings.lorentzian_hwhm_deg == 0.2
    assert settings.lorentzian_probability == 0.1
    assert FIGURE.AXIAL_QUADRATURE_POINT_COUNT == 32000


def test_trapezoid_weights_integrate_constant_and_linear_fields() -> None:
    nodes = np.array((-1.0, -0.25, 0.5, 2.0), dtype=np.float64)
    weights = FIGURE._trapezoid_weights(nodes)

    assert np.sum(weights) == pytest.approx(nodes[-1] - nodes[0])
    assert np.sum(weights * nodes) == pytest.approx(0.5 * (nodes[-1] ** 2 - nodes[0] ** 2))


def test_mosaic_and_structure_masses_are_combined_multiplicatively() -> None:
    mosaic_mass = np.array((0.2, 0.3, 0.5), dtype=np.float64)
    structure_mass_A = np.array((2.0, 5.0), dtype=np.float64)

    combined_A = FIGURE._mosaic_structure_mass_A(mosaic_mass, structure_mass_A)

    np.testing.assert_array_equal(combined_A, np.outer(mosaic_mass, structure_mass_A))
    np.testing.assert_allclose(np.sum(combined_A, axis=0), structure_mass_A)


def test_dense_axial_rule_matches_converged_bi2se3_sf_integrals() -> None:
    settings = FIGURE.ReciprocalFigureSettings(worker_count=1)
    inputs = FIGURE._configured_inputs(FIGURE.DEFAULT_CONFIG, settings)
    rods_by_family = FIGURE._physical_rods_by_family(inputs)
    b3_norm = float(np.linalg.norm(inputs.reciprocal.basis_Ainv[:, 2]))
    k_Ainv = 2.0 * math.pi / inputs.config.source.mean_wavelength_A
    # Independent 128-point Gauss rules on every integer-L panel; 64, 96, and 192 agree.
    oracle_integral_A = {
        0: 4.804898117228715e-3,
        1: 2.338262333177435e-2,
        3: 1.699439557667317e-2,
    }

    for family_m, rods in rods_by_family.items():
        axial_Ainv, axial_weight_A = FIGURE._structure_quadrature_axial_Ainv(inputs, rods[0])
        lower_Ainv, upper_Ainv = inputs.bragg_space.rod_u_bounds_Ainv(rods[0])
        assert np.all(np.diff(axial_Ainv) > 0.0)
        assert np.all(axial_weight_A > 0.0)
        assert np.sum(axial_weight_A) == pytest.approx(
            upper_Ainv - lower_Ainv,
            rel=2.0e-14,
        )
        strength_A2 = np.zeros_like(axial_Ainv)
        for rod in rods:
            strength_A2 += rod.population * inputs.strength.evaluate_profile(
                rod=rod,
                L=axial_Ainv / b3_norm,
                k_norm_Ainv=k_Ainv,
            )
        integrated_A = float(np.sum(strength_A2 * axial_weight_A))

        assert integrated_A == pytest.approx(oracle_integral_A[family_m], rel=5.0e-3)


def test_annular_density_conserves_deposited_structure_mass() -> None:
    radial_edges = np.array((0.0, 0.5, 1.5, 2.0), dtype=np.float64)
    axial_edges = np.array((0.0, 0.25, 1.0), dtype=np.float64)
    mass_A = np.arange(1.0, 7.0, dtype=np.float64).reshape(3, 2)

    density_A4 = FIGURE._annular_density_A4(mass_A, radial_edges, axial_edges)
    annular_area_Ainv2 = math.pi * np.diff(radial_edges**2)
    volume_Ainv3 = annular_area_Ainv2[:, None] * np.diff(axial_edges)[None, :]

    assert np.sum(density_A4 * volume_Ainv3) == pytest.approx(np.sum(mass_A))


def test_display_smoothing_preserves_annular_structure_mass() -> None:
    radial_edges = np.linspace(0.0, 2.0, 9)
    axial_edges = np.linspace(0.0, 3.0, 11)
    mass_A = np.zeros((8, 10), dtype=np.float64)
    mass_A[0, 0] = 2.0
    mass_A[3, 4] = 5.0
    mass_A[-1, -1] = 7.0

    density_A4 = FIGURE._smoothed_annular_display_density_A4(
        mass_A,
        radial_edges,
        axial_edges,
    )
    annular_area_Ainv2 = math.pi * np.diff(radial_edges**2)
    volume_Ainv3 = annular_area_Ainv2[:, None] * np.diff(axial_edges)[None, :]

    assert np.sum(density_A4 * volume_Ainv3) == pytest.approx(np.sum(mass_A), rel=2.0e-14)


def test_output_inventory_is_preflighted_before_rendering(tmp_path: Path) -> None:
    outputs = FIGURE._requested_output_paths(tmp_path, "all", ("png", "pdf"))
    assert len(outputs) == 4
    outputs[-1].touch()

    with pytest.raises(FileExistsError, match=outputs[-1].name):
        FIGURE._preflight_output_paths(
            outputs,
            tmp_path / "display-cache.npz",
            overwrite=False,
        )
    with pytest.raises(ValueError, match="cache path must not alias"):
        FIGURE._preflight_output_paths(outputs, outputs[0], overwrite=True)


def test_reciprocal_volume_rejects_fabricated_m2_family() -> None:
    radial = np.array((0.5, 1.5), dtype=np.float64)
    axial = np.array((0.5, 1.5), dtype=np.float64)
    density = np.ones((3, 2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="physical families m=0,1,3"):
        FIGURE.ReciprocalDisplayVolume(
            radial_centers_Ainv=radial,
            axial_centers_Ainv=axial,
            family_display_density_A4=density,
            family_m=np.array((0, 1, 2), dtype=np.int64),
            family_display_peaks_A4=np.ones(3, dtype=np.float64),
        )
