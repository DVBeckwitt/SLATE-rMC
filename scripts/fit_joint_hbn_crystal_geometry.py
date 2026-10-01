"""Automatically fit shared hBN and crystalline-specimen geometry."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rasim_next.fitting.geometry import ExactTagGeometryModel  # noqa: E402
from rasim_next.fitting.hbn import (  # noqa: E402
    fit_hbn_detector_calibration,
)
from rasim_next.fitting.indexed_series import IndexedGeometryImage  # noqa: E402
from rasim_next.fitting.joint_geometry import (  # noqa: E402
    fit_joint_geometry,
)
from rasim_next.fitting.joint_geometry_report import (  # noqa: E402
    joint_geometry_report,
    joint_geometry_static,
)
from rasim_next.io.json_publication import publish_json_document  # noqa: E402
from rasim_next.io.osc import read_osc  # noqa: E402
from rasim_next.pipeline.configured_simulation import (  # noqa: E402
    build_configured_geometry_inputs,
    load_simulation_config,
    load_strict_yaml_mapping,
)
from rasim_next.selection.osc_series import (  # noqa: E402
    OscGeometryIndexingRun,
    index_osc_geometry_series,
    load_osc_geometry_series,
)


def _mapping(value: object, name: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return value


def _path(base: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty path")
    result = (base / value).resolve()
    if not result.is_file():
        raise ValueError(f"{name} does not exist: {result}")
    return result


def _write_external_json(destination: Path, payload: dict[str, object]) -> Path:
    path = destination.resolve()
    if path == ROOT or path.is_relative_to(ROOT):
        raise ValueError("joint geometry artifact must be written outside the repository")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return publish_json_document(path, payload)


def _replicated_sparse_images(
    run: OscGeometryIndexingRun,
    *,
    minimum_sites_per_image: int = 2,
) -> tuple[IndexedGeometryImage, ...]:
    images = []
    for declared, inputs in zip(run.series.images, run.geometry_inputs, strict=True):
        observations = run.selection.replicated_key_observations_for(declared.image_id)
        if len(observations.keys) < minimum_sites_per_image:
            raise RuntimeError(
                f"{declared.image_id} retained {len(observations.keys)} replicated exact keys; "
                f"at least {minimum_sites_per_image} are required"
            )
        images.append(
            IndexedGeometryImage(
                image_id=declared.image_id,
                commanded_angle_rad=math.radians(
                    declared.axis_rotation_angles_deg[run.series.incidence_axis_index]
                ),
                model=ExactTagGeometryModel(inputs),
                observations=observations,
            )
        )
    return tuple(images)


def run(manifest_path: Path) -> dict[str, object]:
    manifest = load_strict_yaml_mapping(manifest_path)
    required_keys = {
        "schema_version",
        "detector_base_simulation_config",
        "hbn",
        "bi2se3_series",
        "bi2te3_series",
    }
    optional_keys = {"pbi2_parent", "pbi2_y1_series", "pbi2_y2_series"}
    if (
        not isinstance(manifest, dict)
        or not required_keys <= set(manifest)
        or set(manifest) - required_keys - optional_keys
    ):
        raise ValueError("joint geometry manifest has missing or unknown fields")
    root = manifest
    if root["schema_version"] != "rasim-joint-hbn-crystal-geometry-v2":
        raise ValueError("unsupported joint geometry schema_version")
    has_pbi2 = "pbi2_y1_series" in root or "pbi2_y2_series" in root
    if (has_pbi2 and root.get("pbi2_parent") != "2H") or (not has_pbi2 and "pbi2_parent" in root):
        raise ValueError("pbi2_parent must be '2H' exactly when a PbI2 series is supplied")
    hbn = _mapping(
        root["hbn"],
        "hbn",
        {
            "osc_path",
            "dark_path",
            "initial_beam_center_px",
            "initial_calibrant_distance_m",
        },
    )
    base_directory = manifest_path.parent
    base_config_path = _path(
        base_directory,
        root["detector_base_simulation_config"],
        "detector_base_simulation_config",
    )
    hbn_path = _path(base_directory, hbn["osc_path"], "hbn.osc_path")
    dark_path = _path(base_directory, hbn["dark_path"], "hbn.dark_path")
    bi2se3_path = _path(base_directory, root["bi2se3_series"], "bi2se3_series")
    bi2te3_path = _path(base_directory, root["bi2te3_series"], "bi2te3_series")
    pbi2_y1_path = (
        _path(base_directory, root["pbi2_y1_series"], "pbi2_y1_series")
        if "pbi2_y1_series" in root
        else None
    )
    pbi2_y2_path = (
        _path(base_directory, root["pbi2_y2_series"], "pbi2_y2_series")
        if "pbi2_y2_series" in root
        else None
    )
    center = np.asarray(hbn["initial_beam_center_px"], dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("hbn.initial_beam_center_px must contain finite column and row")

    base_config = load_simulation_config(base_config_path)
    base_inputs = build_configured_geometry_inputs(base_config)
    base_instrument = base_inputs.instrument
    hbn_image = read_osc(hbn_path).detector_native_counts
    dark_image = read_osc(dark_path).detector_native_counts
    hbn_observations, hbn_calibration = fit_hbn_detector_calibration(
        hbn_image,
        dark_image,
        base_detector_rotation=base_instrument.lab_from_detector.rotation,
        beam_direction_lab=base_config.source.mean_direction_lab,
        detector_column_pitch_m=base_instrument.detector_column_pitch_m,
        detector_row_pitch_m=base_instrument.detector_row_pitch_m,
        initial_beam_center_px=(float(center[0]), float(center[1])),
        initial_calibrant_distance_m=float(hbn["initial_calibrant_distance_m"]),
    )
    se3_run = index_osc_geometry_series(load_osc_geometry_series(bi2se3_path))
    te3_run = index_osc_geometry_series(load_osc_geometry_series(bi2te3_path))
    if se3_run.indexed_images is None or te3_run.indexed_images is None:
        raise RuntimeError("both BiX series must retain fit-ready multi-L observations")
    y1_images = (
        _replicated_sparse_images(index_osc_geometry_series(load_osc_geometry_series(pbi2_y1_path)))
        if pbi2_y1_path is not None
        else ()
    )
    y2_images = (
        _replicated_sparse_images(index_osc_geometry_series(load_osc_geometry_series(pbi2_y2_path)))
        if pbi2_y2_path is not None
        else ()
    )
    result = fit_joint_geometry(
        hbn_observations=hbn_observations,
        hbn_calibration=hbn_calibration,
        bi2se3_images=se3_run.indexed_images,
        bi2te3_images=te3_run.indexed_images,
        pbi2_y1_images=y1_images,
        pbi2_y2_images=y2_images,
        base_detector_rotation=base_instrument.lab_from_detector.rotation,
    )
    static = joint_geometry_static(
        base_config, base_instrument, pbi2_parent=root.get("pbi2_parent")
    )
    return joint_geometry_report(
        result,
        static=static,
        hbn_calibration=hbn_calibration,
        bi2se3_images=se3_run.indexed_images,
        bi2te3_images=te3_run.indexed_images,
        pbi2_y1_images=y1_images,
        pbi2_y2_images=y2_images,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=ROOT / "configs" / "joint_hbn_crystal_geometry_fit.yaml",
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    payload = run(arguments.manifest.resolve())
    if arguments.destination is not None:
        payload["artifact_path"] = str(_write_external_json(arguments.destination, payload))
    if arguments.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["confidence_qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
