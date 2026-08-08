"""Compose the immutable position/lattice/mosaic checkpoint used by intensity fitting."""
# ruff: noqa: E402  # Direct execution bootstraps the repository source tree below.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rasim_next.fitting import (
    FIXED_EXPERIMENT_STATE_SCHEMA_VERSION,
    FixedLatticeState,
    FixedMosaicState,
    FixedPositionState,
    build_fixed_experiment_series,
    fixed_lattice_from_fit_record,
    fixed_position_from_fit_record,
)
from rasim_next.io.osc import read_osc
from rasim_next.materials import read_crystal
from rasim_next.pipeline.configured_simulation import (
    configured_rod_catalog_revision,
    load_simulation_config,
)
from rasim_next.selection import load_osc_geometry_series


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _mapping_file(path: Path, *, format_name: str) -> dict[str, Any]:
    resolved = path.resolve()
    if format_name == "json":
        value = json.loads(resolved.read_text(encoding="utf-8"))
    elif format_name == "toml":
        value = tomllib.loads(resolved.read_text(encoding="utf-8"))
    else:  # pragma: no cover - internal call contract
        raise ValueError(f"unsupported mapping format {format_name!r}")
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a mapping")
    return value


def _position_state(
    path: Path,
    *,
    geometry_manifest_path: Path,
) -> tuple[FixedPositionState, str]:
    document = _mapping_file(path, format_name="json")
    expected_manifest = geometry_manifest_path.resolve()
    if document.get("schema_version") == "rasim-layered-position-fit-v2":
        manifest_identity = document.get("provenance", {}).get("manifest", {})
        if (
            document.get("accepted") is not True
            or document.get("intensity_evaluated") is not False
            or document.get("model_pixelized") is not False
            or manifest_identity.get("sha256") != _sha256(expected_manifest)
        ):
            raise ValueError("provided modular position artifact is not qualified")
        position = FixedPositionState.from_record(document.get("fixed_position"))
        if document.get("scientific_revision") != position.artifact_revision:
            raise ValueError("modular position artifact revision changed")
        status = str(document.get("status", ""))
        if status not in {"POSITION_FIT", "POSITION_MODEL_LIMITED"}:
            raise ValueError("modular position artifact status is invalid")
        return position, status
    position, status, _ = fixed_position_from_fit_record(
        document,
        expected_manifest_path=expected_manifest,
        expected_manifest_sha256=_sha256(expected_manifest),
    )
    return position, status


def _geometry_contract(config: Any) -> tuple[object, ...]:
    """Return only configuration fields that own geometry and reciprocal indexing."""

    return (
        config.cif_sha256,
        config.material.phase_id,
        config.source,
        replace(config.instrument, film_thickness_A=0.0),
        config.bragg,
    )


def _lattice_state(
    path: Path | None,
    *,
    position_path: Path,
    reference_direct_basis_A: np.ndarray,
) -> FixedLatticeState:
    if path is None:
        return FixedLatticeState.implicit_cif(reference_direct_basis_A)
    resolved = path.resolve()
    document = _mapping_file(resolved, format_name="json")
    return fixed_lattice_from_fit_record(
        document,
        artifact_path=str(resolved),
        artifact_sha256=_sha256(resolved),
        position_artifact_sha256=_sha256(position_path.resolve()),
        reference_direct_basis_A=reference_direct_basis_A,
    )


def _source_state_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("source_state_count must be a positive integer")
    return value


def compose_fixed_experiment(
    *,
    position_path: Path,
    geometry_manifest_path: Path,
    recipe_path: Path,
    mosaic_state_path: Path,
    lattice_path: Path | None,
    source_state_count: int,
    destination: Path,
) -> Path:
    """Validate all predecessor identities and write one resumable fixed checkpoint."""

    count = _source_state_count(source_state_count)
    destination = destination.resolve()
    if destination == ROOT or destination.is_relative_to(ROOT):
        raise ValueError("fixed-experiment artifacts must be written outside the repository")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    position_path = position_path.resolve()
    geometry_manifest_path = geometry_manifest_path.resolve()
    position, position_status = _position_state(
        position_path,
        geometry_manifest_path=geometry_manifest_path,
    )
    recipe_path = recipe_path.resolve()
    recipe = _mapping_file(recipe_path, format_name="toml")
    material_id = recipe.get("material_id")
    dataset_ids = recipe.get("dataset_ids")
    recipe_detector = recipe.get("detector")
    if (
        recipe.get("schema_version") != "rasim-fitted-figure7-recreation-v3"
        or not isinstance(material_id, str)
        or not material_id
        or not isinstance(dataset_ids, list)
        or not dataset_ids
        or any(not isinstance(value, str) or not value for value in dataset_ids)
        or len(set(dataset_ids)) != len(dataset_ids)
        or not isinstance(recipe_detector, Mapping)
    ):
        raise ValueError("recipe material and dataset roster are invalid")
    simulation_config_path = (recipe_path.parent / str(recipe["simulation_config"])).resolve()
    config = load_simulation_config(simulation_config_path)

    detector_shape = tuple(recipe_detector.get("native_shape_rc", ()))
    if (
        len(detector_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in detector_shape)
        or detector_shape != tuple(config.instrument.detector_shape_rc)
    ):
        raise ValueError("recipe detector shape differs from the simulation configuration")

    geometry_series = load_osc_geometry_series(geometry_manifest_path)
    geometry_config = load_simulation_config(geometry_series.config_path)
    images = geometry_series.images
    axis_index = geometry_series.incidence_axis_index
    if (
        _geometry_contract(geometry_config) != _geometry_contract(config)
        or len(images) != len(dataset_ids)
        or any(
            len(image.axis_rotation_angles_deg) != len(config.instrument.axis_rotations)
            for image in images
        )
    ):
        raise ValueError("geometry manifest differs from the recipe series")
    image_ids = tuple(image.image_id for image in images)
    incidence_deg = tuple(image.axis_rotation_angles_deg[axis_index] for image in images)
    if position.incidence_angle_image_ids and image_ids != position.incidence_angle_image_ids:
        raise ValueError("geometry manifest order differs from the fitted position trims")
    if len(position.commanded_incidence_angles_rad) != len(incidence_deg) or not np.allclose(
        position.commanded_incidence_angles_rad,
        np.radians(incidence_deg),
        rtol=0.0,
        atol=2.0e-14,
    ):
        raise ValueError("geometry manifest commanded angles differ from the position fit")

    reference_crystal = read_crystal(
        config.material.cif_path,
        phase_id=config.material.phase_id,
        expected_sha256=config.cif_sha256,
    )
    lattice = _lattice_state(
        lattice_path,
        position_path=position_path,
        reference_direct_basis_A=reference_crystal.direct_basis_A,
    )
    mosaic_state_path = mosaic_state_path.resolve()
    mosaic = FixedMosaicState.from_record(_mapping_file(mosaic_state_path, format_name="json"))
    series = build_fixed_experiment_series(
        config,
        position=position,
        fixed_lattice=lattice,
        source_sample_count=count,
        gaussian_sigma_rad=math.radians(mosaic.gaussian_sigma_deg),
        lorentzian_half_width_rad=math.radians(mosaic.lorentzian_hwhm_deg),
        lorentzian_probability=mosaic.lorentzian_probability,
    )

    datasets: list[dict[str, Any]] = []
    for dataset_id, image_id, angle_deg, image in zip(
        dataset_ids,
        image_ids,
        incidence_deg,
        images,
        strict=True,
    ):
        osc_path = image.osc_path
        counts = read_osc(osc_path).detector_native_counts
        if tuple(counts.shape) != detector_shape:
            raise ValueError(f"OSC detector shape differs for dataset {dataset_id!r}")
        datasets.append(
            {
                "dataset_id": dataset_id,
                "position_image_id": image_id,
                "incidence_angle_deg": angle_deg,
                "osc": _identity(osc_path),
                "detector_native_bytes_sha256": hashlib.sha256(
                    counts.tobytes(order="C")
                ).hexdigest(),
                "detector_native_dtype": str(counts.dtype),
                "detector_native_shape_rc": list(counts.shape),
            }
        )

    mosaic_identity = _identity(mosaic_state_path)
    document = {
        "schema_version": FIXED_EXPERIMENT_STATE_SCHEMA_VERSION,
        "material_id": material_id,
        "status": "FIXED_EXPERIMENT",
        "position_status": position_status,
        "source_state_count": count,
        "source_revision": str(series[0].samples.source_revision),
        "simulation_config": _identity(simulation_config_path),
        "cif_sha256": str(series[0].config.cif_sha256),
        "rod_catalog_revision": configured_rod_catalog_revision(series[0]),
        "fixed_position": position.to_record(),
        "fixed_lattice": lattice.to_record(),
        "fixed_mosaic": mosaic.to_record(),
        "datasets": datasets,
        "model_pixelized": False,
        "smoothing_applied": False,
        "provenance": {
            "position": _identity(position_path),
            "lattice": None if lattice_path is None else _identity(lattice_path),
            "mosaic": mosaic_identity,
            "geometry_manifest": _identity(geometry_manifest_path),
            "recipe": _identity(recipe_path),
            "adapter": _identity(Path(__file__)),
        },
    }
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--mosaic-state", type=Path, required=True)
    parser.add_argument("--lattice", type=Path)
    parser.add_argument("--source-state-count", type=int, default=250)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(
        compose_fixed_experiment(
            position_path=arguments.position,
            geometry_manifest_path=arguments.geometry_manifest,
            recipe_path=arguments.recipe,
            mosaic_state_path=arguments.mosaic_state,
            lattice_path=arguments.lattice,
            source_state_count=arguments.source_state_count,
            destination=arguments.destination,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
