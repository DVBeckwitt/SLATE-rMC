from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from rasim_next.fitting.fixed_experiment import build_fixed_experiment_series
from rasim_next.fitting.fixed_lattice import FixedLatticeState
from rasim_next.fitting.geometry import (
    ExactTagGeometryModel,
    IntegerLMarkerKey,
    IntegerLMarkerObservations,
)
from rasim_next.fitting.indexed_series import IndexedGeometryImage
from rasim_next.fitting.joint_geometry import (
    DEFAULT_FIXED_REFERENCE_PARAMETERS,
    GLOBAL_PARAMETER_NAMES,
    JOINT_GEOMETRY_PARAMETER_NAMES,
    LOCAL_PARAMETER_NAMES,
    NUISANCE_PARAMETER_NAMES,
    JointGeometryState,
    _absolute_instrument_and_model,
    joint_beam_origin_lab_m,
)
from rasim_next.fitting.joint_geometry_handoff import (
    load_joint_geometry_handoff,
    save_joint_geometry_handoff,
)
from rasim_next.geometry.instrument import compose_intrinsic_xy_rotation
from rasim_next.materials import read_crystal
from rasim_next.pipeline.configured_simulation import (
    build_configured_geometry_inputs,
    load_simulation_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _report(state: JointGeometryState, detector_base: object) -> dict[str, object]:
    instrument = detector_base.instrument
    axis = instrument.axis_rotations[0]
    fixed = dict(DEFAULT_FIXED_REFERENCE_PARAMETERS)

    def section(names: tuple[str, ...]) -> dict[str, object]:
        return {
            name: {
                "value": getattr(state, name),
                "confidence_qualified": None if name in fixed else True,
                "active_bound": False,
                "role": "fixed_reference" if name in fixed else "fitted",
            }
            for name in names
        }

    global_parameters = section(GLOBAL_PARAMETER_NAMES)
    global_parameters["z_b_m"] = {"value": 0.0}
    return {
        "schema_version": "rasim-joint-hbn-crystal-geometry-fit-result-v3",
        "success": True,
        "confidence_qualified": True,
        "qualification_failures": [],
        "global": global_parameters,
        "local": section(LOCAL_PARAMETER_NAMES),
        "nuisance": section(NUISANCE_PARAMETER_NAMES),
        "static": {
            "detector_shape_rc": instrument.detector_shape_rc,
            "detector_row_pitch_m": instrument.detector_row_pitch_m,
            "detector_column_pitch_m": instrument.detector_column_pitch_m,
            "detector_reference_coordinate_px": instrument.detector_reference_coordinate_px,
            "detector_plane_reference_translation_lab_m": instrument.lab_from_detector.translation_m,
            "detector_base_rotation_lab_from_detector": instrument.lab_from_detector.rotation,
            "beam_direction_lab": detector_base.source.mean_direction_lab,
            "goniometer_nominal_axis_lab": axis.axis_lab,
            "goniometer_nominal_pivot_lab_m": axis.pivot_lab_m,
            "bi2se3_sample_x_tilt_rad": 0.0,
        },
        "derived": {"beam_origin_lab_m": joint_beam_origin_lab_m(state, detector_base)},
        "crystalline_metrics": {
            "per_image": [{"specimen_id": "bi2te3", "image_id": "synthetic-5deg"}]
        },
    }


def test_saved_joint_handoff_rebuilds_source_and_rejects_double_geometry(tmp_path: Path) -> None:
    base_path = ROOT / "configs/bi2se3_simulation.yaml"
    specimen_path = tmp_path / "bi2te3_simulation.yaml"
    specimen_document = yaml.safe_load((ROOT / "configs/bi2te3_simulation.yaml").read_text())
    copied_cif = tmp_path / "bi2te3.cif"
    copied_cif.write_bytes(
        (ROOT / "examples/bi2te3/structures/Bi2Te3_cod_9011962.cif").read_bytes()
    )
    specimen_document["material"]["cif_path"] = str(copied_cif)
    specimen_path.write_text(yaml.safe_dump(specimen_document), encoding="utf-8")
    detector_base = load_simulation_config(base_path)
    specimen = load_simulation_config(specimen_path)
    state = replace(
        JointGeometryState.from_array(np.zeros(len(JOINT_GEOMETRY_PARAMETER_NAMES))),
        detector_column_tilt_rad=math.radians(-0.39),
        detector_row_tilt_rad=math.radians(-1.34),
        beam_center_column_px=1452.75,
        beam_center_row_px=1596.73,
        bi2te3_sample_x_tilt_rad=math.radians(-0.42),
        bi2te3_sample_y_tilt_rad=math.radians(0.32),
        bi2te3_zs_m=21.5e-6,
    )
    report = _report(state, detector_base)
    report_path = tmp_path / "joint.json"
    report_path.write_text(
        json.dumps(report, default=lambda value: np.asarray(value).tolist()), encoding="utf-8"
    )
    osc_path = tmp_path / "synthetic.osc.gz"
    osc_path.write_bytes(b"geometry identity only")
    manifest_path = tmp_path / "geometry.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "rasim-osc-geometry-fit-v1",
                "simulation_config": str(specimen_path),
                "incidence_axis_index": 0,
                "images": [
                    {
                        "image_id": "synthetic-5deg",
                        "osc_path": str(osc_path),
                        "axis_rotation_angles_deg": [5.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "handoff.json"
    save_joint_geometry_handoff(
        checkpoint,
        report_path=report_path,
        geometry_manifest_path=manifest_path,
        detector_base_config_path=base_path,
        specimen_id="bi2te3",
    )
    checkpoint_record = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_record["status"] == "GEOMETRY_ONLY"
    assert checkpoint_record["mosaic_qualified"] is False
    checkpoint_record["mosaic_qualified"] = True
    checkpoint.write_text(json.dumps(checkpoint_record), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported joint geometry handoff"):
        load_joint_geometry_handoff(checkpoint)
    checkpoint_record["mosaic_qualified"] = False
    checkpoint.write_text(json.dumps(checkpoint_record), encoding="utf-8")
    handoff = load_joint_geometry_handoff(checkpoint)
    crystal = read_crystal(
        specimen.material.cif_path,
        phase_id=specimen.material.phase_id,
        expected_sha256=specimen.cif_sha256,
    )
    fixed = build_fixed_experiment_series(
        handoff.config,
        position=handoff.position,
        fixed_lattice=FixedLatticeState.implicit_cif(crystal.direct_basis_A),
        source_sample_count=1,
        gaussian_sigma_rad=math.radians(1.0),
        lorentzian_half_width_rad=math.radians(0.5),
        lorentzian_probability=0.5,
    )[0]
    original_model = ExactTagGeometryModel(build_configured_geometry_inputs(specimen))
    keys = (IntegerLMarkerKey(1, 10, 2, -1, (-1, 0)),)
    nominal = original_model.predict_integer_l_tags(keys)
    image = IndexedGeometryImage(
        image_id="synthetic-5deg",
        commanded_angle_rad=math.radians(5.0),
        model=original_model,
        observations=IntegerLMarkerObservations(
            keys=keys,
            coordinates_px=nominal.coordinates_px,
            covariance_px2=np.eye(2)[None, :, :],
            reference_wavelength_A=original_model.reference_wavelength_A,
        ),
    )
    oracle_model, oracle_instrument, _ = _absolute_instrument_and_model(
        "bi2te3",
        image,
        state,
        base_detector_rotation=np.asarray(detector_base.instrument.lab_from_detector.rotation),
    )
    rebuilt_model = ExactTagGeometryModel(build_configured_geometry_inputs(fixed.config))
    np.testing.assert_allclose(
        rebuilt_model.inputs.samples.origin_lab_m,
        oracle_model.inputs.samples.origin_lab_m,
        rtol=0.0,
        atol=2.0e-15,
    )
    expected = oracle_model.predict_integer_l_tags(
        keys, instrument=oracle_instrument
    ).coordinates_px
    actual = rebuilt_model.predict_integer_l_tags(keys, instrument=fixed.instrument).coordinates_px
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-11)

    wrong_source = replace(
        fixed.config,
        source=replace(fixed.config.source, mean_origin_lab_m=(0.001, -0.02, 0.0)),
    )
    wrong_model = ExactTagGeometryModel(build_configured_geometry_inputs(wrong_source))
    assert (
        np.max(
            np.abs(
                wrong_model.predict_integer_l_tags(keys, instrument=fixed.instrument).coordinates_px
                - expected
            )
        )
        > 0.1
    )
    wrong_reference = replace(
        fixed.instrument,
        detector_reference_coordinate_px=(
            fixed.instrument.detector_reference_coordinate_px[0] - 0.3724,
            fixed.instrument.detector_reference_coordinate_px[1] + 0.3092,
        ),
    )
    assert (
        np.max(
            np.abs(
                rebuilt_model.predict_integer_l_tags(
                    keys, instrument=wrong_reference
                ).coordinates_px
                - expected
            )
        )
        > 0.1
    )
    wrong_tilt = replace(
        fixed.instrument,
        lab_from_detector=replace(
            fixed.instrument.lab_from_detector,
            rotation=compose_intrinsic_xy_rotation(
                fixed.instrument.lab_from_detector.rotation,
                state.detector_column_tilt_rad,
                state.detector_row_tilt_rad,
            ),
        ),
    )
    assert (
        np.max(
            np.abs(
                rebuilt_model.predict_integer_l_tags(keys, instrument=wrong_tilt).coordinates_px
                - expected
            )
        )
        > 0.1
    )

    copied_cif.write_bytes(copied_cif.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="specimen CIF bytes changed"):
        load_joint_geometry_handoff(checkpoint)
    copied_cif.write_bytes(copied_cif.read_bytes()[:-1])

    changed = json.loads(report_path.read_text(encoding="utf-8"))
    changed["derived"]["beam_origin_lab_m"][0] += 1.0e-3
    report_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="bytes changed"):
        load_joint_geometry_handoff(checkpoint)
    with pytest.raises(ValueError, match="derived beam origin"):
        save_joint_geometry_handoff(
            tmp_path / "inconsistent.json",
            report_path=report_path,
            geometry_manifest_path=manifest_path,
            detector_base_config_path=base_path,
            specimen_id="bi2te3",
        )
