from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rasim_next.fitting import (
    SHARED_GEOMETRY_PARAMETER_NAMES,
    FixedLatticeState,
    FixedMosaicState,
    FixedPositionState,
    SharedGeometryCorrections,
    build_fixed_experiment_series,
    fixed_position_from_fit_record,
    zero_sum_helmert_basis,
)
from rasim_next.materials import read_crystal
from rasim_next.pipeline.configured_simulation import (
    load_simulation_config,
    rebind_configured_simulation_instrument,
)

ROOT = Path(__file__).resolve().parents[1]


def _position() -> FixedPositionState:
    trims = (math.radians(-0.05), math.radians(0.02), math.radians(0.03))
    image_ids = ("five", "ten", "fifteen")
    trim_by_image_id = dict(zip(image_ids, trims, strict=True))
    canonical_trims = tuple(trim_by_image_id[image_id] for image_id in sorted(image_ids))
    contrasts = tuple(zero_sum_helmert_basis(3).T @ np.asarray(canonical_trims, dtype=np.float64))
    return FixedPositionState(
        artifact_revision=f"sha256-{'a' * 64}",
        corrections=SharedGeometryCorrections.zero(),
        incidence_angle_delta_rad=math.radians(0.4),
        commanded_incidence_angles_rad=tuple(map(math.radians, (5.0, 10.0, 15.0))),
        beam_center_column_row_px=(1453.12, 1596.422),
        incidence_angle_image_ids=image_ids,
        incidence_angle_trim_rad=trims,
        incidence_angle_trim_contrast_rad=contrasts,
        incidence_angle_trim_prior_sigma_rad=math.radians(0.15),
        incidence_angle_trim_contrast_half_span_rad=math.radians(0.35),
    )


def test_fixed_position_round_trip_preserves_shared_delta_and_zero_sum_trims() -> None:
    position = _position()

    restored = FixedPositionState.from_record(position.to_record())

    assert restored == position
    assert math.isclose(
        math.fsum(restored.incidence_angle_trim_rad),
        0.0,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    )
    np.testing.assert_allclose(
        restored.effective_incidence_angles_rad,
        tuple(
            commanded + math.radians(0.4) + trim
            for commanded, trim in zip(
                position.commanded_incidence_angles_rad,
                position.incidence_angle_trim_rad,
                strict=True,
            )
        ),
        rtol=0.0,
        atol=2.0e-14,
    )


def test_fixed_experiment_applies_calibrated_detector_center_and_distance() -> None:
    config = load_simulation_config(ROOT / "configs" / "bi2se3_r3_simulation.yaml")
    crystal = read_crystal(
        config.material.cif_path,
        phase_id=config.material.phase_id,
        expected_sha256=config.cif_sha256,
    )
    lattice = FixedLatticeState.implicit_cif(crystal.direct_basis_A)
    nominal_position = _position()
    calibrated_position = replace(
        nominal_position,
        beam_center_column_row_px=(1455.62, 1594.922),
        detector_calibration_active=True,
        detector_plane_normal_offset_m=1.5e-3,
    )

    nominal = build_fixed_experiment_series(
        config,
        position=nominal_position,
        fixed_lattice=lattice,
        source_sample_count=1,
        gaussian_sigma_rad=math.radians(1.0),
        lorentzian_half_width_rad=math.radians(0.5),
        lorentzian_probability=0.0,
    )[0]
    calibrated = build_fixed_experiment_series(
        config,
        position=FixedPositionState.from_record(calibrated_position.to_record()),
        fixed_lattice=lattice,
        source_sample_count=1,
        gaussian_sigma_rad=math.radians(1.0),
        lorentzian_half_width_rad=math.radians(0.5),
        lorentzian_probability=0.0,
    )[0]

    assert calibrated.instrument.detector_reference_coordinate_px == (
        calibrated_position.beam_center_column_row_px
    )
    normal = nominal.instrument.lab_from_detector.rotation[:, 2]
    translation_delta = (
        calibrated.instrument.lab_from_detector.translation_m
        - nominal.instrument.lab_from_detector.translation_m
    )
    assert float(normal @ translation_delta) == pytest.approx(1.5e-3, abs=1.0e-16)
    np.testing.assert_allclose(
        translation_delta - 1.5e-3 * normal,
        0.0,
        rtol=0.0,
        atol=1.0e-16,
    )


def test_raw_geometry_fit_record_is_the_shared_position_handoff(tmp_path: Path) -> None:
    manifest = tmp_path / "geometry.yaml"
    manifest.write_text("schema_version: rasim-osc-geometry-fit-v1\n", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    position = _position()
    trim_by_id = dict(
        zip(
            position.incidence_angle_image_ids,
            position.incidence_angle_trim_rad,
            strict=True,
        )
    )
    record = {
        "schema": "rasim-osc-geometry-fit-result-v6",
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "indexed_manifest_hash": "frozen-selection",
        "run_completed": True,
        "fixed_position": position.to_record(),
        "fit": {
            "success": True,
            "jacobian_parameter_names": ["geometry", "common_delta", "trim_1", "trim_2"],
            "jacobian_rank": 4,
            "incidence_angle_delta_fitted": True,
            "incidence_angle_delta_rad": position.incidence_angle_delta_rad,
            "incidence_angle_trim_by_image_id_rad": trim_by_id,
            "corrections": {
                name: float(value)
                for name, value in zip(
                    SHARED_GEOMETRY_PARAMETER_NAMES,
                    position.corrections.as_array(),
                    strict=True,
                )
            },
        },
        "qualification": {"requested": False, "accepted": False},
        "root_audit": {"classification": "SAME"},
        "outer_audit": {"classification": "SAME"},
    }

    restored, status, selection = fixed_position_from_fit_record(
        record,
        expected_manifest_path=manifest,
        expected_manifest_sha256=manifest_sha256,
    )

    assert restored == position
    assert status == "POSITION_MODEL_LIMITED"
    assert selection == "frozen-selection"

    tampered = json.loads(json.dumps(record))
    tampered["fixed_position"]["detector_plane_normal_offset_m"] = 0.123
    with pytest.raises(ValueError, match="detector-calibration provenance"):
        fixed_position_from_fit_record(
            tampered,
            expected_manifest_path=manifest,
            expected_manifest_sha256=manifest_sha256,
        )

    record["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="position result"):
        fixed_position_from_fit_record(
            record,
            expected_manifest_path=manifest,
            expected_manifest_sha256=manifest_sha256,
        )


def test_legacy_fixed_position_cannot_silently_change_detector_center() -> None:
    config = load_simulation_config(ROOT / "configs" / "bi2se3_r3_simulation.yaml")
    crystal = read_crystal(
        config.material.cif_path,
        phase_id=config.material.phase_id,
        expected_sha256=config.cif_sha256,
    )
    lattice = FixedLatticeState.implicit_cif(crystal.direct_basis_A)
    unproven_center = replace(
        _position(),
        beam_center_column_row_px=(1454.12, 1596.422),
    )

    with pytest.raises(ValueError, match="without provenance"):
        build_fixed_experiment_series(
            config,
            position=unproven_center,
            fixed_lattice=lattice,
            source_sample_count=1,
            gaussian_sigma_rad=math.radians(1.0),
            lorentzian_half_width_rad=math.radians(0.5),
            lorentzian_probability=0.0,
        )


def test_fixed_position_rejects_inconsistent_or_out_of_bounds_helmert_state() -> None:
    position = _position()
    inconsistent = list(position.incidence_angle_trim_rad)
    inconsistent[0] += math.radians(0.01)
    inconsistent[2] -= math.radians(0.01)

    with pytest.raises(ValueError, match="Helmert"):
        FixedPositionState(
            artifact_revision=position.artifact_revision,
            corrections=position.corrections,
            incidence_angle_delta_rad=position.incidence_angle_delta_rad,
            commanded_incidence_angles_rad=position.commanded_incidence_angles_rad,
            beam_center_column_row_px=position.beam_center_column_row_px,
            incidence_angle_image_ids=position.incidence_angle_image_ids,
            incidence_angle_trim_rad=tuple(inconsistent),
            incidence_angle_trim_contrast_rad=position.incidence_angle_trim_contrast_rad,
            incidence_angle_trim_prior_sigma_rad=(position.incidence_angle_trim_prior_sigma_rad),
            incidence_angle_trim_contrast_half_span_rad=(
                position.incidence_angle_trim_contrast_half_span_rad
            ),
        )

    with pytest.raises(ValueError, match="fitted bounds"):
        FixedPositionState(
            artifact_revision=position.artifact_revision,
            corrections=position.corrections,
            incidence_angle_delta_rad=position.incidence_angle_delta_rad,
            commanded_incidence_angles_rad=position.commanded_incidence_angles_rad,
            beam_center_column_row_px=position.beam_center_column_row_px,
            incidence_angle_image_ids=position.incidence_angle_image_ids,
            incidence_angle_trim_rad=position.incidence_angle_trim_rad,
            incidence_angle_trim_contrast_rad=position.incidence_angle_trim_contrast_rad,
            incidence_angle_trim_prior_sigma_rad=(position.incidence_angle_trim_prior_sigma_rad),
            incidence_angle_trim_contrast_half_span_rad=1.0e-8,
        )


def test_fixed_position_rejects_nonstring_image_ids() -> None:
    with pytest.raises(ValueError, match="image IDs"):
        FixedPositionState(
            artifact_revision=f"sha256-{'a' * 64}",
            corrections=SharedGeometryCorrections.zero(),
            incidence_angle_delta_rad=0.0,
            commanded_incidence_angles_rad=(math.radians(5.0),),
            beam_center_column_row_px=(1453.12, 1596.422),
            incidence_angle_image_ids=(None,),  # type: ignore[arg-type]
            incidence_angle_trim_rad=(0.0,),
            incidence_angle_trim_contrast_rad=(),
            incidence_angle_trim_prior_sigma_rad=math.radians(0.1),
            incidence_angle_trim_contrast_half_span_rad=math.radians(0.2),
        )


def test_fixed_mosaic_state_is_strict_and_allows_zero_width_only_for_inactive_component() -> None:
    state = FixedMosaicState(
        gaussian_sigma_deg=1.0,
        lorentzian_hwhm_deg=0.0,
        lorentzian_probability=0.0,
        provenance="provided prior",
    )
    assert FixedMosaicState.from_record(state.to_record()) == state
    with pytest.raises(ValueError, match="zero mosaic atom mass"):
        FixedMosaicState(
            gaussian_sigma_deg=0.0,
            lorentzian_hwhm_deg=0.5,
            lorentzian_probability=0.2,
            provenance="invalid mixed prior",
        )


@pytest.mark.parametrize("material", ("bi2se3", "bi2te3"))
def test_tracked_material_mosaic_state_satisfies_the_strict_handoff(material: str) -> None:
    path = ROOT / "examples" / material / "experiment" / "fixed_mosaic.json"
    record = json.loads(path.read_text(encoding="utf-8"))

    state = FixedMosaicState.from_record(record)

    assert state.to_record() == record


@pytest.mark.parametrize(
    "config_name",
    ("bi2se3_r3_simulation.yaml", "bi2te3_r3_simulation.yaml"),
)
def test_fixed_experiment_series_reuses_immutable_physics_across_materials(
    config_name: str,
) -> None:
    config = load_simulation_config(ROOT / "configs" / config_name)
    crystal = read_crystal(
        config.material.cif_path,
        phase_id=config.material.phase_id,
        expected_sha256=config.cif_sha256,
    )
    lattice = FixedLatticeState.implicit_cif(crystal.direct_basis_A)

    series = build_fixed_experiment_series(
        config,
        position=_position(),
        fixed_lattice=lattice,
        source_sample_count=2,
        gaussian_sigma_rad=math.radians(1.1),
        lorentzian_half_width_rad=math.radians(0.5),
        lorentzian_probability=0.3,
    )

    assert len(series) == 3
    assert all(item.commanded_instrument_rebindable is False for item in series)
    with pytest.raises(ValueError, match="disabled after fixed geometry"):
        rebind_configured_simulation_instrument(series[0], series[1].config)
    np.testing.assert_allclose(
        tuple(item.config.instrument.axis_rotations[0].angle_deg for item in series),
        (5.35, 10.42, 15.43),
        rtol=0.0,
        atol=1.0e-12,
    )
    for item in series[1:]:
        assert item.samples is series[0].samples
        assert item.crystal is series[0].crystal
        assert item.material is series[0].material
        assert item.reciprocal is series[0].reciprocal
        assert item.rods is series[0].rods
        assert item.strength is series[0].strength
        assert item.mosaic is series[0].mosaic
        assert item.samples.source_revision == series[0].samples.source_revision
        np.testing.assert_array_equal(item.samples.direction_lab, series[0].samples.direction_lab)
        np.testing.assert_array_equal(item.samples.wavelength_A, series[0].samples.wavelength_A)
