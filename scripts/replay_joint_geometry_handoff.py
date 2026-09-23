"""Save and replay a qualified joint geometry handoff on its OSC images."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rasim_next.fitting.fixed_experiment import build_fixed_experiment_series  # noqa: E402
from rasim_next.fitting.fixed_lattice import FixedLatticeState  # noqa: E402
from rasim_next.fitting.geometry import ExactTagGeometryModel  # noqa: E402
from rasim_next.fitting.indexed_series import IntegerLMarkerObservations  # noqa: E402
from rasim_next.fitting.joint_geometry import (  # noqa: E402
    _absolute_instrument_and_model,
)
from rasim_next.fitting.joint_geometry_handoff import (  # noqa: E402
    load_joint_geometry_handoff,
    qualified_joint_geometry_state,
    save_joint_geometry_handoff,
)
from rasim_next.materials import read_crystal  # noqa: E402
from rasim_next.pipeline.configured_simulation import (  # noqa: E402
    build_configured_geometry_inputs,
    load_simulation_config,
)
from rasim_next.selection.osc_series import (  # noqa: E402
    index_osc_geometry_series,
    load_osc_geometry_series,
)


def replay(
    report_path: Path,
    detector_base_path: Path,
    geometry_manifest_path: Path,
    checkpoint_path: Path,
) -> dict[str, object]:
    run = index_osc_geometry_series(load_osc_geometry_series(geometry_manifest_path))
    if run.indexed_images is None:
        raise ValueError("OSC indexing did not retain fit-ready geometry images")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    state, _ = qualified_joint_geometry_state(report)
    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.exists():
        raise FileExistsError(checkpoint_path)
    candidate = checkpoint_path.with_name(f".{checkpoint_path.name}.{uuid4().hex}.candidate")
    try:
        save_joint_geometry_handoff(
            candidate,
            report_path=report_path,
            geometry_manifest_path=geometry_manifest_path,
            detector_base_config_path=detector_base_path,
            specimen_id="bi2te3",
        )
        handoff = load_joint_geometry_handoff(candidate)
        crystal = read_crystal(
            handoff.config.material.cif_path,
            phase_id=handoff.config.material.phase_id,
            expected_sha256=handoff.config.cif_sha256,
        )
        series = build_fixed_experiment_series(
            handoff.config,
            position=handoff.position,
            fixed_lattice=FixedLatticeState.implicit_cif(crystal.direct_basis_A),
            source_sample_count=1,
            gaussian_sigma_rad=math.radians(handoff.config.mosaic.gaussian_sigma_deg),
            lorentzian_half_width_rad=math.radians(handoff.config.mosaic.lorentzian_hwhm_deg),
            lorentzian_probability=handoff.config.mosaic.lorentzian_probability,
        )
        base_rotation = np.asarray(report["static"]["detector_base_rotation_lab_from_detector"])
        report_metrics = {
            metric["image_id"]: metric
            for metric in report["crystalline_metrics"]["per_image"]
            if metric["specimen_id"] == "bi2te3"
        }
        base_source = load_simulation_config(detector_base_path).source
        beam_direction = np.asarray(base_source.mean_direction_lab, dtype=np.float64)
        beam_direction /= np.linalg.norm(beam_direction)
        image_metrics = []
        for image, fixed in zip(run.indexed_images, series, strict=True):
            oracle_model, oracle_instrument, oracle_beam_origin = _absolute_instrument_and_model(
                "bi2te3", image, state, base_detector_rotation=base_rotation
            )
            rebuilt_model = ExactTagGeometryModel(build_configured_geometry_inputs(fixed.config))
            if not np.allclose(
                rebuilt_model.inputs.samples.origin_lab_m,
                oracle_model.inputs.samples.origin_lab_m,
                rtol=0.0,
                atol=2.0e-15,
            ):
                raise AssertionError("rebuilt source origin differs from the joint source")
            if not np.allclose(
                handoff.config.source.mean_origin_lab_m,
                oracle_beam_origin,
                rtol=0.0,
                atol=2.0e-15,
            ):
                raise AssertionError("checkpoint beam line differs from the joint fit")
            reference_column, reference_row = oracle_instrument.detector_reference_coordinate_px
            detector_offset = np.asarray(
                (
                    (state.beam_center_column_px - reference_column)
                    * oracle_instrument.detector_column_pitch_m,
                    (state.beam_center_row_px - reference_row)
                    * oracle_instrument.detector_row_pitch_m,
                    0.0,
                )
            )
            beam_hit = (
                oracle_instrument.lab_from_detector.translation_m
                + oracle_instrument.lab_from_detector.rotation @ detector_offset
            )
            if (
                np.linalg.norm(np.cross(beam_hit - oracle_beam_origin, beam_direction)) > 2.0e-15
                or abs(
                    np.dot(
                        oracle_beam_origin - np.asarray(base_source.mean_origin_lab_m),
                        beam_direction,
                    )
                )
                > 2.0e-15
            ):
                raise AssertionError("fitted beam line misses the declared detector beam center")
            observations = image.observations
            if isinstance(observations, IntegerLMarkerObservations):
                oracle = oracle_model.predict_integer_l_tags(
                    observations.keys, instrument=oracle_instrument
                )
                rebuilt = rebuilt_model.predict_integer_l_tags(
                    observations.keys, instrument=fixed.instrument
                )
                count = len(observations.keys)
            else:
                oracle = oracle_model.predict_layer_l_tags(
                    observations.definitions, instrument=oracle_instrument
                )
                rebuilt = rebuilt_model.predict_layer_l_tags(
                    observations.definitions, instrument=fixed.instrument
                )
                count = len(observations.definitions)
            if not (np.all(oracle.active_panel) and np.all(rebuilt.active_panel)):
                raise AssertionError("an agreed geometry site is outside the active detector panel")
            if not (
                np.all(np.isfinite(oracle.coordinates_px))
                and np.all(np.isfinite(rebuilt.coordinates_px))
            ):
                raise AssertionError("an agreed geometry site has nonfinite coordinates")
            difference = np.linalg.norm(rebuilt.coordinates_px - oracle.coordinates_px, axis=1)
            residual = np.linalg.norm(rebuilt.coordinates_px - observations.coordinates_px, axis=1)
            recorded = report_metrics[image.image_id]
            rms = float(np.sqrt(np.mean(residual**2)))
            maximum = float(np.max(residual))
            if (
                count != recorded["site_count"]
                or not np.isclose(rms, recorded["site_rms_px"], rtol=0.0, atol=2.0e-9)
                or not np.isclose(maximum, recorded["site_max_px"], rtol=0.0, atol=2.0e-9)
            ):
                raise AssertionError("replayed sites disagree with the qualified report metrics")
            image_metrics.append(
                {
                    "image_id": image.image_id,
                    "site_count": count,
                    "prediction_max_difference_px": float(np.max(difference)),
                    "site_rms_px": rms,
                    "site_max_px": maximum,
                }
            )
        if max(item["prediction_max_difference_px"] for item in image_metrics) > 2.0e-11:
            raise AssertionError("joint geometry handoff changed detector predictions")
        candidate.replace(checkpoint_path)
    finally:
        candidate.unlink(missing_ok=True)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "site_count": sum(item["site_count"] for item in image_metrics),
        "per_image": image_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("detector_base", type=Path)
    parser.add_argument("geometry_manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    tracemalloc.start()
    started = perf_counter()
    result = replay(
        arguments.report,
        arguments.detector_base,
        arguments.geometry_manifest,
        arguments.checkpoint,
    )
    result["wall_time_s"] = perf_counter() - started
    result["peak_traced_bytes"] = tracemalloc.get_traced_memory()[1]
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
