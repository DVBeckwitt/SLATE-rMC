"""Replay the accepted Bi2Se3 and Bi2Te3 staged fits from hash-bound cases."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import io
import json
import math
import platform
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from packaging.markers import InvalidMarker, Marker

from rasim_next.core.staged_fit import (
    STAGED_FIT_STAGE_SCHEMA_VERSION,
)
from rasim_next.core.staged_fit import (
    staged_fit_scientific_revision as scientific_revision,
)

ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_VERSION = "rasim-staged-fit-replay-v2"
_STAGE_SCHEMA_VERSION = STAGED_FIT_STAGE_SCHEMA_VERSION
_CERTIFICATE_SCHEMA_VERSION = "rasim-staged-fit-replay-certificate-v2"
_STAGES = ("geometry", "mosaic", "ordered_intensity", "render")
_SHARED_GEOMETRY_PARAMETER_NAMES = (
    "detector_column_tilt_rad",
    "detector_row_tilt_rad",
    "sample_normal_x_tilt_rad",
    "sample_normal_y_tilt_rad",
    "goniometer_axis_pitch_rad",
    "goniometer_axis_yaw_rad",
    "sample_plane_normal_offset_m",
    "goniometer_pivot_pitch_offset_m",
    "goniometer_pivot_yaw_offset_m",
)
_SHA256_PREFIX = "sha256-"
_MOSAIC_COMPONENT_CACHE_SCHEMA = "rasim-mosaic-component-checkpoint-v1"
_STAGE_RESULT_KEYS = {
    "case_id",
    "stage_case_sha256",
    "execution_backend",
    "material_id",
    "runtime",
    "schema_version",
    "scientific_revision",
    "scientific_summary",
    "source_revision",
    "source_seed",
    "source_state_count",
    "stage",
    "state",
    "upstream_scientific_revision",
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "case_id",
    "material_id",
    "classification",
    "source_state_count",
    "source_seed",
    "incidence_angles_deg",
    "files",
    "geometry",
    "mosaic",
    "ordered_intensity",
    "render",
    "expected",
    "tolerances",
}
_TOLERANCE_KEYS = {
    "geometry_correction_absolute",
    "geometry_metric_absolute",
    "mosaic_parameter_absolute",
    "mosaic_objective_absolute",
    "ordered_parameter_absolute",
    "ordered_objective_absolute",
}
_PROJECT_PACKAGE = "rasim-next"
_RENDER_RUNTIME_PACKAGES = ("pillow",)
_STAGE_TOLERANCE_NAMES = {
    "geometry": ("geometry_correction_absolute", "geometry_metric_absolute"),
    "mosaic": ("mosaic_parameter_absolute", "mosaic_objective_absolute"),
    "ordered_intensity": ("ordered_parameter_absolute", "ordered_objective_absolute"),
    "render": (),
}


class ReplayMismatchError(RuntimeError):
    """Raised when a replay changes a frozen scientific result."""


@dataclass(frozen=True, slots=True)
class ReplayCase:
    path: Path
    repository_root: Path
    case_id: str
    material_id: str
    classification: str
    source_state_count: int
    source_seed: int
    incidence_angles_deg: tuple[float, ...]
    input_paths: dict[str, Path]
    file_records: tuple[dict[str, Any], ...]
    stage_config: dict[str, dict[str, Any]]
    expected_scientific_summary: dict[str, Any]
    tolerances: dict[str, float]
    runtime_identity: dict[str, Any]


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _python_minor_bounds(requirement: str) -> tuple[tuple[int, int], tuple[int, int]]:
    tokens = tuple(item.strip() for item in requirement.split(","))
    if len(tokens) != 2 or not tokens[0].startswith(">=") or not tokens[1].startswith("<"):
        raise ValueError("environment lock has an unsupported requires-python expression")
    try:
        lower = tuple(int(item) for item in tokens[0][2:].split("."))
        upper = tuple(int(item) for item in tokens[1][1:].split("."))
    except ValueError as error:
        raise ValueError("environment lock has an invalid requires-python expression") from error
    if len(lower) != 2 or len(upper) != 2 or lower >= upper:
        raise ValueError("environment lock must bound Python by ordered major.minor versions")
    return lower, upper


def _validated_runtime_identity(
    environment_lock: Path,
    *,
    expected_sha256: str,
    include_render: bool = False,
) -> dict[str, Any]:
    """Require the executing dependency closure to match one hash-bound lock snapshot."""

    payload = environment_lock.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"environment lock content hash changed: {environment_lock}")
    document = tomllib.loads(payload.decode("utf-8"))
    requirement = document.get("requires-python")
    records = document.get("package")
    if not isinstance(requirement, str) or not isinstance(records, list):
        raise ValueError("environment lock lacks Python or package metadata")
    lower, upper = _python_minor_bounds(requirement)
    python_minor = tuple(sys.version_info[:2])
    if not lower <= python_minor < upper:
        raise RuntimeError(
            f"Python {platform.python_version()} is outside locked requirement {requirement}; "
            "run the replay with `uv run --frozen`"
        )

    package_records: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("environment lock package records must be tables")
        name = record.get("name")
        if not isinstance(name, str) or name in package_records:
            raise ValueError("environment lock package names must be unique strings")
        package_records[name] = record
    try:
        project_record = package_records[_PROJECT_PACKAGE]
    except KeyError as error:
        raise ValueError(f"environment lock lacks {_PROJECT_PACKAGE}") from error

    def applicable_dependencies(record: dict[str, Any]) -> list[str]:
        names = []
        for dependency in record.get("dependencies", ()):
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
                raise ValueError("environment lock dependencies must have package names")
            marker = dependency.get("marker")
            if marker is not None:
                if not isinstance(marker, str):
                    raise ValueError("environment lock dependency marker must be a string")
                try:
                    applies = Marker(marker).evaluate()
                except InvalidMarker as error:
                    raise ValueError("environment lock dependency marker is invalid") from error
                if not applies:
                    continue
            names.append(dependency["name"])
        return names

    pending = applicable_dependencies(project_record)
    if include_render:
        pending.extend(_RENDER_RUNTIME_PACKAGES)
    locked_versions: dict[str, str] = {}
    while pending:
        name = pending.pop()
        if not isinstance(name, str) or name in locked_versions:
            continue
        try:
            record = package_records[name]
        except KeyError as error:
            raise ValueError(f"environment lock lacks runtime package {name}") from error
        version = record.get("version")
        if not isinstance(version, str):
            raise ValueError(f"environment lock does not version runtime package {name}")
        locked_versions[name] = version
        pending.extend(applicable_dependencies(record))

    installed_versions: dict[str, str] = {}
    remedy = "uv run --frozen --extra visualization" if include_render else "uv run --frozen"
    for name in sorted(locked_versions):
        try:
            installed = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"runtime package {name} is absent; run the replay with `{remedy}`"
            ) from error
        locked = locked_versions[name]
        if installed != locked:
            raise RuntimeError(
                f"runtime package {name} is {installed}, but uv.lock requires {locked}; "
                f"run the replay with `{remedy}`"
            )
        installed_versions[name] = installed
    if include_render:
        try:
            importlib.import_module("PIL.Image")
        except ImportError as error:
            raise RuntimeError(
                "PIL.Image cannot be imported; run the replay with "
                "`uv run --frozen --extra visualization`"
            ) from error
    return {
        "environment_lock_sha256": actual_sha256,
        "environment_lock_requires_python": requirement,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "packages": installed_versions,
    }


def _strict_keys(record: dict[str, Any], expected: set[str], label: str) -> None:
    unknown = set(record) - expected
    missing = expected - set(record)
    if unknown:
        raise ValueError(f"{label} contains unknown key {sorted(unknown)[0]!r}")
    if missing:
        raise ValueError(f"{label} is missing key {sorted(missing)[0]!r}")


def _validate_file_records(
    records: object,
    *,
    case_directory: Path,
    repository_root: Path,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Path], dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError("files must be a nonempty array of tables")
    parsed: list[dict[str, Any]] = []
    input_paths: dict[str, Path] = {}
    for index, value in enumerate(records):
        if not isinstance(value, dict):
            raise ValueError(f"files[{index}] must be a table")
        kind = value.get("kind", "file")
        expected_keys = {"role", "path", "sha256"}
        if kind == "osc":
            expected_keys |= {
                "kind",
                "detector_native_bytes_sha256",
                "detector_native_shape_rc",
                "detector_native_dtype",
            }
        elif kind != "file":
            raise ValueError(f"files[{index}].kind is unsupported")
        _strict_keys(value, expected_keys, f"files[{index}]")
        role = value["role"]
        path_text = value["path"]
        if not isinstance(role, str) or not role or role in input_paths:
            raise ValueError(f"files[{index}].role must be unique and nonempty")
        if not isinstance(path_text, str) or not path_text:
            raise ValueError(f"files[{index}].path must be nonempty")
        relative_path = Path(path_text)
        if relative_path.is_absolute():
            raise ValueError(f"files[{index}].path must be relative to the case")
        resolved = (case_directory / relative_path).resolve()
        if not resolved.is_relative_to(repository_root):
            raise ValueError(f"files[{index}].path resolves outside the repository")
        if not _is_sha256(value["sha256"]):
            raise ValueError(f"files[{index}].sha256 must be SHA-256")
        input_paths[role] = resolved
        parsed.append({**value, "kind": kind})

    try:
        environment_record = next(
            record for record in parsed if record["role"] == "environment_lock"
        )
    except StopIteration as error:
        raise ValueError("files must declare an environment_lock role") from error
    environment_lock = input_paths["environment_lock"]
    if not environment_lock.is_file():
        raise FileNotFoundError(environment_lock)
    runtime_identity = _validated_runtime_identity(
        environment_lock,
        expected_sha256=str(environment_record["sha256"]),
    )

    for index, record in enumerate(parsed):
        path = input_paths[str(record["role"])]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = (
            runtime_identity["environment_lock_sha256"]
            if record["role"] == "environment_lock"
            else _sha256(path)
        )
        if actual_sha256 != record["sha256"]:
            raise ValueError(f"files[{index}] content hash changed: {path}")
        if record["kind"] != "osc":
            continue
        from rasim_next.io.osc import read_osc

        counts = read_osc(path).detector_native_counts
        expected_shape = tuple(int(item) for item in record["detector_native_shape_rc"])
        if counts.shape != expected_shape or str(counts.dtype) != record["detector_native_dtype"]:
            raise ValueError(f"files[{index}] decoded OSC layout changed: {path}")
        native_sha256 = hashlib.sha256(counts.tobytes(order="C")).hexdigest()
        if native_sha256 != record["detector_native_bytes_sha256"]:
            raise ValueError(f"files[{index}] decoded OSC values changed: {path}")
    return tuple(parsed), input_paths, runtime_identity


def _nested_input_path(
    container: Path,
    value: object,
    *,
    label: str,
    repository_root: Path,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative")
    resolved = (container.parent / relative).resolve()
    if not resolved.is_relative_to(repository_root):
        raise ValueError(f"{label} resolves outside the repository")
    return resolved


def _require_nested_input(
    container: Path,
    value: object,
    *,
    expected: Path,
    label: str,
    repository_root: Path,
) -> None:
    actual = _nested_input_path(
        container,
        value,
        label=label,
        repository_root=repository_root,
    )
    if actual != expected:
        raise ValueError(f"{label} does not resolve to its declared replay role")


def _yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    return document


def _validate_consumed_input_paths(
    material_id: str,
    *,
    input_paths: dict[str, Path],
    stages: dict[str, dict[str, Any]],
    repository_root: Path,
) -> None:
    """Bind every nested path consumed by a replay to its declared hashed role."""

    geometry = stages["geometry"]
    if geometry["series_role"] != "geometry_series":
        raise ValueError("geometry.series_role must be geometry_series")
    series_path = input_paths["geometry_series"]
    series = _yaml_mapping(series_path, "geometry series")
    _require_nested_input(
        series_path,
        series.get("simulation_config"),
        expected=input_paths["simulation_config"],
        label="geometry series simulation_config",
        repository_root=repository_root,
    )
    expected_osc_roles = {5.0: "osc_5deg", 10.0: "osc_10deg", 15.0: "osc_15deg"}
    images = series.get("images")
    if not isinstance(images, list):
        raise ValueError("geometry series images must be a list")
    seen_roles = set()
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError(f"geometry series images[{index}] must be a mapping")
        angles = image.get("axis_rotation_angles_deg")
        if not isinstance(angles, list) or len(angles) != 1:
            raise ValueError(f"geometry series images[{index}] must have one incidence angle")
        try:
            role = expected_osc_roles[float(angles[0])]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"geometry series images[{index}] has an unexpected angle") from error
        if role in seen_roles:
            raise ValueError(f"geometry series repeats {role}")
        _require_nested_input(
            series_path,
            image.get("osc_path"),
            expected=input_paths[role],
            label=f"geometry series images[{index}].osc_path",
            repository_root=repository_root,
        )
        seen_roles.add(role)
    if seen_roles != set(expected_osc_roles.values()):
        raise ValueError("geometry series must consume all three declared OSC roles")

    simulation_path = input_paths["simulation_config"]
    simulation = _yaml_mapping(simulation_path, "simulation config")
    material = simulation.get("material")
    if not isinstance(material, dict):
        raise ValueError("simulation config material must be a mapping")
    _require_nested_input(
        simulation_path,
        material.get("cif_path"),
        expected=input_paths["cif"],
        label="simulation config material.cif_path",
        repository_root=repository_root,
    )

    if material_id == "Bi2Te3":
        if geometry["catalog_role"] != "indexed_catalog":
            raise ValueError("geometry.catalog_role must be indexed_catalog")
        mosaic = stages["mosaic"]
        if (
            mosaic["catalog_role"] != "indexed_catalog"
            or mosaic["dark_role"] != "dark"
            or tuple(mosaic["osc_roles"]) != ("osc_5deg", "osc_10deg", "osc_15deg")
        ):
            raise ValueError("Bi2Te3 mosaic roles changed their declared replay inputs")
        return

    mosaic = stages["mosaic"]
    ordered = stages["ordered_intensity"]
    if (
        mosaic["case_role"] != "mosaic_case"
        or mosaic["measured_profile_policy_role"] != "measured_profile_policy"
        or ordered["case_role"] != "ordered_intensity_case"
    ):
        raise ValueError("Bi2Se3 stage roles changed their declared replay inputs")
    mosaic_path = input_paths["mosaic_case"]
    mosaic_document = tomllib.loads(mosaic_path.read_text(encoding="utf-8"))
    _require_nested_input(
        mosaic_path,
        mosaic_document.get("simulation_config"),
        expected=simulation_path,
        label="mosaic case simulation_config",
        repository_root=repository_root,
    )
    nonzero_provenance = mosaic_document.get("nonzero_centroid_provenance")
    m0_provenance = mosaic_document.get("m0_centroid_provenance")
    if not isinstance(nonzero_provenance, dict) or not isinstance(m0_provenance, dict):
        raise ValueError("mosaic case centroid provenance must be mappings")
    _require_nested_input(
        mosaic_path,
        nonzero_provenance.get("series_config"),
        expected=series_path,
        label="mosaic case nonzero series_config",
        repository_root=repository_root,
    )
    _require_nested_input(
        mosaic_path,
        m0_provenance.get("source_file"),
        expected=input_paths["legacy_peak_observations"],
        label="mosaic case m0 source_file",
        repository_root=repository_root,
    )
    observations = mosaic_document.get("m0_observations")
    if not isinstance(observations, list):
        raise ValueError("mosaic case m0_observations must be a list")
    seen_roles = set()
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError(f"mosaic case m0_observations[{index}] must be a mapping")
        try:
            role = expected_osc_roles[float(observation["incidence_angle_deg"])]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"mosaic case m0_observations[{index}] has an unexpected angle"
            ) from error
        if role in seen_roles:
            raise ValueError(f"mosaic case repeats {role}")
        _require_nested_input(
            mosaic_path,
            observation.get("osc_file"),
            expected=input_paths[role],
            label=f"mosaic case m0_observations[{index}].osc_file",
            repository_root=repository_root,
        )
        seen_roles.add(role)
    if seen_roles != set(expected_osc_roles.values()):
        raise ValueError("mosaic case must consume all three declared OSC roles")

    ordered_path = input_paths["ordered_intensity_case"]
    ordered_document = tomllib.loads(ordered_path.read_text(encoding="utf-8"))
    if ordered_document.get("schema_version") != "rasim-measured-ordered-intensity-fit-v1":
        raise ValueError("Bi2Se3 ordered-intensity case must declare the measured fit schema")
    _require_nested_input(
        ordered_path,
        ordered_document.get("mosaic_case"),
        expected=mosaic_path,
        label="ordered-intensity case mosaic_case",
        repository_root=repository_root,
    )
    policy_path = input_paths["measured_profile_policy"]
    policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("base_case_sha256") != _sha256(mosaic_path):
        raise ValueError("measured profile policy does not bind the declared mosaic case")


def _validate_stage_config(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stages: dict[str, dict[str, Any]] = {}
    for stage in _STAGES:
        value = document[stage]
        if not isinstance(value, dict) or not value:
            raise ValueError(f"{stage} must be a nonempty table")
        stages[stage] = value
    schemas = {
        "Bi2Se3": {
            "geometry": {
                "benchmark",
                "fit_incidence_angle_delta",
                "fitted_parameter_names",
                "fixed_parameter_names",
                "heldout_integer_l",
                "incidence_angle_delta_half_span_deg",
                "initial_corrections",
                "initial_incidence_angle_delta_rad",
                "input_roles",
                "selection_mode",
                "series_role",
                "source_state_count",
            },
            "mosaic": {
                "case_role",
                "execution_source_state_count",
                "implementation",
                "input_roles",
                "measured_profile_policy_role",
                "observation_mode",
                "render_images",
            },
            "ordered_intensity": {
                "active_parameters",
                "case_role",
                "claim_boundary",
                "execution_source_state_count",
                "fit_atomic_positions",
                "implementation",
                "input_roles",
                "lower_bounds",
                "maximum_function_evaluations",
                "multistarts",
                "occupancy_ratio_reference",
                "parameter_scales",
                "upper_bounds",
            },
            "render": {"enabled", "input_roles", "reason"},
        },
        "Bi2Te3": {
            "geometry": {
                "benchmark",
                "bounds_model",
                "catalog_manifest_revision",
                "catalog_role",
                "fitted_parameter_names",
                "fixed_parameter_names",
                "heldout_integer_l",
                "historical_selection_revision",
                "input_roles",
                "multistart_fraction",
                "multistart_pattern",
                "selection_mode",
                "series_role",
                "source_state_count",
            },
            "mosaic": {
                "catalog_role",
                "coarse_width_count",
                "dark_role",
                "dark_subtraction_model",
                "dataset_ids",
                "execution_source_state_count",
                "extra_nonzero_profiles",
                "gaussian_sigma_bounds_deg",
                "implementation",
                "input_roles",
                "lorentzian_hwhm_bounds_deg",
                "m0_landmark_maximum_distance_px",
                "m0_phi_gauss_order",
                "m0_phi_half_width_deg",
                "m0_starts_10deg",
                "m0_starts_15deg",
                "m0_starts_5deg",
                "m0_two_theta_gauss_order",
                "m1_integer_l_10deg",
                "m1_integer_l_15deg",
                "m1_integer_l_5deg",
                "maximum_sensitivity_condition",
                "minimum_excess_energy_over_side_scatter",
                "near_optimal_objective_delta",
                "nonzero_phi_gauss_order",
                "nonzero_phi_half_width_deg",
                "nonzero_two_theta_gauss_order",
                "nuisance_background_model",
                "osc_roles",
                "phi_bin_count",
                "refinement_levels",
                "refinement_width_count",
                "sideband_two_theta_offsets_deg",
                "two_theta_half_width_deg",
            },
            "ordered_intensity": {
                "active_parameters",
                "claim_boundary",
                "execution_source_state_count",
                "fit_atomic_positions",
                "implementation",
                "input_roles",
                "lower_bounds",
                "maximum_function_evaluations",
                "multistarts",
                "occupancy_ratio_reference",
                "parameter_scales",
                "upper_bounds",
            },
            "render": {
                "cuda_coordinate_chunk",
                "cuda_state_block_count",
                "enabled",
                "image_size",
                "input_roles",
                "raw_display_model",
                "simulation_display_model",
                "source_state_count",
            },
        },
    }
    for stage, keys in schemas[str(document["material_id"])].items():
        _strict_keys(stages[stage], keys, stage)
        input_roles = stages[stage]["input_roles"]
        if (
            not isinstance(input_roles, list)
            or not input_roles
            or any(not isinstance(role, str) or not role for role in input_roles)
            or len(set(input_roles)) != len(input_roles)
            or "environment_lock" not in input_roles
        ):
            raise ValueError(f"{stage}.input_roles must be unique and include environment_lock")
    geometry = stages["geometry"]
    fitted = tuple(geometry.get("fitted_parameter_names", ()))
    fixed = tuple(geometry.get("fixed_parameter_names", ()))
    if not fitted or len(set((*fitted, *fixed))) != len((*fitted, *fixed)):
        raise ValueError("geometry fitted/fixed parameter names must be disjoint and unique")
    if int(geometry.get("source_state_count", 0)) != 1:
        raise ValueError("geometry replay must use one ideal source state")
    for stage in ("mosaic", "ordered_intensity"):
        source_count = stages[stage].get("execution_source_state_count")
        if source_count != document["source_state_count"]:
            raise ValueError(f"{stage} source count must match the case")
    if bool(stages["ordered_intensity"]["fit_atomic_positions"]):
        raise ValueError("accepted ordered-intensity replay keeps atomic positions frozen")
    if document["material_id"] == "Bi2Se3":
        expected_fitted = tuple(
            name for name in _SHARED_GEOMETRY_PARAMETER_NAMES if name != "sample_normal_x_tilt_rad"
        )
        initial_corrections = tuple(float(value) for value in geometry["initial_corrections"])
        incidence_half_span_deg = float(geometry["incidence_angle_delta_half_span_deg"])
        initial_incidence_delta = float(geometry["initial_incidence_angle_delta_rad"])
        if (
            geometry["selection_mode"] != "position_free_discovery"
            or tuple(fitted) != expected_fitted
            or tuple(fixed) != ("sample_normal_x_tilt_rad",)
            or not bool(geometry["fit_incidence_angle_delta"])
            or incidence_half_span_deg != 0.5
            or len(initial_corrections) != len(_SHARED_GEOMETRY_PARAMETER_NAMES)
            or any(not math.isfinite(value) for value in initial_corrections)
            or initial_corrections[2] != 0.0
            or not math.isfinite(initial_incidence_delta)
            or abs(initial_incidence_delta) > math.radians(incidence_half_span_deg)
            or stages["mosaic"]["implementation"] != "bi2se3_measured_profiles_v2"
            or stages["mosaic"]["observation_mode"] != "osc"
            or bool(stages["mosaic"]["render_images"])
            or stages["ordered_intensity"]["implementation"]
            != "measured_transferred_mosaic_amplitudes_v1"
            or stages["ordered_intensity"]["occupancy_ratio_reference"] != "bi_occupancy"
            or tuple(stages["ordered_intensity"]["active_parameters"])
            != ("se1_over_bi", "se2_over_bi", "u_radial_A2", "u_normal_A2")
            or tuple(float(value) for value in stages["ordered_intensity"]["parameter_scales"])
            != (1.0, 1.0, 0.1, 0.1)
            or bool(stages["render"]["enabled"])
        ):
            raise ValueError("Bi2Se3 replay changed an accepted stage implementation")
        ordered = stages["ordered_intensity"]
        lower = tuple(float(value) for value in ordered["lower_bounds"])
        upper = tuple(float(value) for value in ordered["upper_bounds"])
        multistarts = tuple(tuple(float(value) for value in row) for row in ordered["multistarts"])
        if (
            len(lower) != 4
            or len(upper) != 4
            or any(not math.isfinite(value) for value in (*lower, *upper))
            or any(minimum >= maximum for minimum, maximum in zip(lower, upper, strict=True))
            or not multistarts
            or any(
                len(row) != 4
                or any(not math.isfinite(value) for value in row)
                or any(
                    value < minimum or value > maximum
                    for value, minimum, maximum in zip(row, lower, upper, strict=True)
                )
                for row in multistarts
            )
            or isinstance(ordered["maximum_function_evaluations"], bool)
            or int(ordered["maximum_function_evaluations"]) < 1
        ):
            raise ValueError("Bi2Se3 ordered-intensity bounds or multistarts are invalid")
    else:
        if (
            geometry["selection_mode"] != "position_free_discovery_with_frozen_catalog_audit"
            or geometry["bounds_model"] != "rasim_multi_angle_pose.v1"
            or float(geometry["multistart_fraction"]) != 0.08
            or tuple(float(value) for value in geometry["multistart_pattern"])
            != (0.0, 0.0, 0.6, -0.4, 0.7, -0.5, 0.3, 0.9, -0.7)
            or stages["mosaic"]["implementation"] != "measured_angle_profiles_v1"
            or stages["mosaic"]["nuisance_background_model"] != "local_phi_constant.v1"
            or stages["mosaic"]["dark_subtraction_model"]
            != "project_raw_and_dark_separately_then_subtract_signal.v1"
            or stages["ordered_intensity"]["implementation"]
            != "measured_transferred_mosaic_amplitudes_v1"
            or stages["ordered_intensity"]["occupancy_ratio_reference"] != "bi_occupancy"
            or tuple(stages["ordered_intensity"]["active_parameters"])
            != ("te1_over_bi", "te2_over_bi", "u_radial_A2", "u_normal_A2")
            or tuple(float(value) for value in stages["ordered_intensity"]["parameter_scales"])
            != (1.0, 1.0, 0.1, 0.1)
            or not bool(stages["render"]["enabled"])
            or int(stages["render"]["source_state_count"]) != document["source_state_count"]
            or int(stages["render"]["image_size"]) != 3000
        ):
            raise ValueError("Bi2Te3 replay changed an accepted stage implementation")
        for name in ("cuda_coordinate_chunk", "cuda_state_block_count"):
            value = stages["render"][name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"render.{name} must be a positive integer")
    return stages


def _expected_summary(document: dict[str, Any]) -> dict[str, Any]:
    expected = document["expected"]
    if not isinstance(expected, dict):
        raise ValueError("expected must be a table")
    geometry_keys = {
        "active_bounds",
        "classification",
        "corrections",
        "fitted_parameter_names",
        "fixed_parameter_names",
        "per_incidence_profile_count",
        "rank",
        "selection_revision",
        "site_max_px",
        "site_rms_px",
    }
    profile_keys = {
        "classification",
        "m0_profile_count",
        "m0_profile_identities",
        "objective",
        "parameters",
        "profile_count",
        "profile_identities",
        "rank",
    }
    ordered_keys = profile_keys | {"active_bounds", "claim_boundary"}
    schemas = {
        "Bi2Se3": {
            "geometry": geometry_keys
            | {
                "commanded_incidence_angles_rad",
                "effective_incidence_angles_rad",
                "incidence_angle_delta_rad",
                "incidence_angle_image_ids",
                "jacobian_parameter_names",
            },
            "mosaic": profile_keys | {"per_incidence_profile_count"},
            "ordered_intensity": ordered_keys
            | {
                "adequacy",
                "m0_relative_residual_rms",
                "m1_relative_residual_rms",
                "relative_residual_rms",
            },
        },
        "Bi2Te3": {
            "geometry": geometry_keys | {"frozen_catalog_sha256"},
            "mosaic": profile_keys | {"per_incidence_profile_count"},
            "ordered_intensity": ordered_keys,
            "render": {
                "classification",
                "m0_rod_count",
                "raw_decoded_pixel_sha256",
                "rod_count",
                "simulated_decoded_pixel_sha256",
                "source_state_count",
            },
        },
    }
    stage_schemas = schemas[str(document["material_id"])]
    _strict_keys(expected, {"source_revision", *stage_schemas}, "expected")
    for stage, keys in stage_schemas.items():
        stage_expected = expected[stage]
        if not isinstance(stage_expected, dict):
            raise ValueError(f"expected.{stage} must be a table")
        _strict_keys(stage_expected, keys, f"expected.{stage}")
    pending = [("expected", expected)]
    while pending:
        path, value = pending.pop()
        if isinstance(value, dict):
            pending.extend((f"{path}.{key}", item) for key, item in value.items())
        elif isinstance(value, list):
            pending.extend((f"{path}[{index}]", item) for index, item in enumerate(value))
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
    if not _is_sha256(expected["source_revision"]):
        raise ValueError("expected.source_revision must be SHA-256 without a prefix")
    result = {
        "case_id": document["case_id"],
        "material_id": document["material_id"],
        "source_state_count": document["source_state_count"],
        "source_revision": expected["source_revision"],
        "geometry": expected["geometry"],
        "mosaic": expected["mosaic"],
        "ordered_intensity": expected["ordered_intensity"],
    }
    if "render" in stage_schemas:
        result["render"] = expected["render"]
    return result


def load_replay_case(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> ReplayCase:
    """Load a strict, repository-relative, hash-complete replay case."""

    resolved_path = path.resolve()
    root = ROOT if repository_root is None else repository_root.resolve()
    document = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("replay case must be a TOML table")
    _strict_keys(document, _TOP_LEVEL_KEYS, "replay case")
    if document["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("unsupported staged-fit replay schema")
    for name in ("case_id", "classification"):
        if not isinstance(document[name], str) or not document[name]:
            raise ValueError(f"{name} must be nonempty")
    if document["material_id"] not in {"Bi2Se3", "Bi2Te3"}:
        raise ValueError("material_id must be Bi2Se3 or Bi2Te3")
    if document["source_state_count"] != 250 or document["source_seed"] != 1729:
        raise ValueError("accepted replay requires exactly 250 source states with seed 1729")
    incidences = tuple(float(value) for value in document["incidence_angles_deg"])
    if incidences != (5.0, 10.0, 15.0):
        raise ValueError("accepted replay requires incidences 5, 10, and 15 degrees")
    stages = _validate_stage_config(document)
    expected_summary = _expected_summary(document)
    file_records, input_paths, runtime_identity = _validate_file_records(
        document["files"],
        case_directory=resolved_path.parent,
        repository_root=root,
    )
    expected_roles = {
        "Bi2Se3": {
            "cif",
            "environment_lock",
            "geometry_series",
            "legacy_peak_observations",
            "measured_profile_policy",
            "mosaic_case",
            "ordered_intensity_case",
            "osc_10deg",
            "osc_15deg",
            "osc_5deg",
            "simulation_config",
        },
        "Bi2Te3": {
            "cif",
            "dark",
            "environment_lock",
            "geometry_series",
            "indexed_catalog",
            "osc_10deg",
            "osc_15deg",
            "osc_5deg",
            "simulation_config",
        },
    }
    if set(input_paths) != expected_roles[str(document["material_id"])]:
        raise ValueError("replay file roles do not match the accepted material case")
    for stage, config in stages.items():
        unknown_roles = set(config["input_roles"]) - set(input_paths)
        if unknown_roles:
            raise ValueError(
                f"{stage}.input_roles contains unknown role {sorted(unknown_roles)[0]!r}"
            )
    _validate_consumed_input_paths(
        str(document["material_id"]),
        input_paths=input_paths,
        stages=stages,
        repository_root=root,
    )
    if document["material_id"] == "Bi2Te3":
        catalog = json.loads(input_paths["indexed_catalog"].read_text(encoding="utf-8"))
        if catalog.get("manifest_hash") != stages["geometry"]["catalog_manifest_revision"]:
            raise ValueError("frozen Bi2Te3 catalog manifest revision changed")
    tolerances = document["tolerances"]
    if not isinstance(tolerances, dict):
        raise ValueError("tolerances must be a table")
    _strict_keys(tolerances, _TOLERANCE_KEYS, "tolerances")
    parsed_tolerances = {name: float(value) for name, value in tolerances.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in parsed_tolerances.values()):
        raise ValueError("replay tolerances must be finite and nonnegative")
    return ReplayCase(
        path=resolved_path,
        repository_root=root,
        case_id=str(document["case_id"]),
        material_id=str(document["material_id"]),
        classification=str(document["classification"]),
        source_state_count=int(document["source_state_count"]),
        source_seed=int(document["source_seed"]),
        incidence_angles_deg=incidences,
        input_paths=input_paths,
        file_records=file_records,
        stage_config=stages,
        expected_scientific_summary=expected_summary,
        tolerances=parsed_tolerances,
        runtime_identity=runtime_identity,
    )


def _stage_case_sha256(case: ReplayCase, stage: str) -> str:
    """Hash only the case fields and declared files consumed by one stage."""

    if stage not in _STAGES:
        raise ValueError(f"unsupported replay stage {stage!r}")
    input_roles = frozenset(str(role) for role in case.stage_config[stage]["input_roles"])
    file_records = sorted(
        (record for record in case.file_records if str(record["role"]) in input_roles),
        key=lambda record: str(record["role"]),
    )
    if {str(record["role"]) for record in file_records} != input_roles:
        raise ValueError(f"{stage} stage input-role records are incomplete")
    encoded = json.dumps(
        {
            "schema": "rasim-staged-fit-stage-case-v1",
            "case_id": case.case_id,
            "material_id": case.material_id,
            "source_state_count": case.source_state_count,
            "source_seed": case.source_seed,
            "incidence_angles_deg": case.incidence_angles_deg,
            "stage": stage,
            "stage_config": case.stage_config[stage],
            "expected_source_revision": case.expected_scientific_summary["source_revision"],
            "expected_stage": case.expected_scientific_summary.get(stage),
            "tolerances": {name: case.tolerances[name] for name in _STAGE_TOLERANCE_NAMES[stage]},
            "files": file_records,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mismatch(path: str, expected: object, actual: object) -> None:
    raise ReplayMismatchError(f"{path} changed: expected {expected!r}, observed {actual!r}")


def _exact(
    actual: dict[str, Any], expected: dict[str, Any], path: str, names: tuple[str, ...]
) -> None:
    for name in names:
        if name not in actual or actual[name] != expected[name]:
            _mismatch(f"{path}.{name}", expected.get(name), actual.get(name))


def _close_scalar(actual: object, expected: object, tolerance: float, path: str) -> None:
    observed = float(actual)
    target = float(expected)
    if (
        not math.isfinite(observed)
        or not math.isfinite(target)
        or abs(observed - target) > tolerance
    ):
        _mismatch(path, target, observed)


def _close_vector(actual: object, expected: object, tolerance: float, path: str) -> None:
    observed = tuple(float(value) for value in actual)  # type: ignore[arg-type]
    target = tuple(float(value) for value in expected)  # type: ignore[arg-type]
    if len(observed) != len(target):
        _mismatch(path, target, observed)
    for index, (left, right) in enumerate(zip(observed, target, strict=True)):
        _close_scalar(left, right, tolerance, f"{path}[{index}]")


def verify_scientific_summary(
    case: ReplayCase,
    actual: dict[str, Any],
    *,
    through: str | None = None,
) -> None:
    """Verify exact identities and tolerance-bound numerical fit results."""

    terminal = (
        _STAGES.index(through)
        if through is not None
        else max(_STAGES.index(name) for name in _STAGES if name in actual)
    )
    expected = case.expected_scientific_summary
    _exact(
        actual,
        expected,
        "summary",
        ("case_id", "material_id", "source_state_count", "source_revision"),
    )
    geometry = actual.get("geometry", {})
    expected_geometry = expected["geometry"]
    _exact(
        geometry,
        expected_geometry,
        "geometry",
        (
            "classification",
            "selection_revision",
            "fitted_parameter_names",
            "fixed_parameter_names",
            *(
                (
                    "jacobian_parameter_names",
                    "incidence_angle_image_ids",
                )
                if case.material_id == "Bi2Se3"
                else ()
            ),
            "rank",
            "active_bounds",
            "per_incidence_profile_count",
        ),
    )
    if "frozen_catalog_sha256" in expected_geometry:
        _exact(
            geometry,
            expected_geometry,
            "geometry",
            ("frozen_catalog_sha256",),
        )
    _close_vector(
        geometry.get("corrections", ()),
        expected_geometry["corrections"],
        case.tolerances["geometry_correction_absolute"],
        "geometry.corrections",
    )
    if case.material_id == "Bi2Se3":
        _close_scalar(
            geometry.get("incidence_angle_delta_rad"),
            expected_geometry["incidence_angle_delta_rad"],
            case.tolerances["geometry_correction_absolute"],
            "geometry.incidence_angle_delta_rad",
        )
        for name in ("commanded_incidence_angles_rad", "effective_incidence_angles_rad"):
            _close_vector(
                geometry.get(name, ()),
                expected_geometry[name],
                case.tolerances["geometry_correction_absolute"],
                f"geometry.{name}",
            )
    for name in ("site_rms_px", "site_max_px"):
        _close_scalar(
            geometry.get(name),
            expected_geometry[name],
            case.tolerances["geometry_metric_absolute"],
            f"geometry.{name}",
        )

    if terminal == 0:
        return
    mosaic = actual.get("mosaic", {})
    expected_mosaic = expected["mosaic"]
    _exact(
        mosaic,
        expected_mosaic,
        "mosaic",
        (
            "classification",
            "rank",
            "profile_count",
            "m0_profile_count",
            "per_incidence_profile_count",
            "profile_identities",
            "m0_profile_identities",
        ),
    )
    _close_vector(
        mosaic.get("parameters", ()),
        expected_mosaic["parameters"],
        case.tolerances["mosaic_parameter_absolute"],
        "mosaic.parameters",
    )
    _close_scalar(
        mosaic.get("objective"),
        expected_mosaic["objective"],
        case.tolerances["mosaic_objective_absolute"],
        "mosaic.objective",
    )

    if terminal == 1:
        return
    ordered = actual.get("ordered_intensity", {})
    expected_ordered = expected["ordered_intensity"]
    _exact(
        ordered,
        expected_ordered,
        "ordered_intensity",
        (
            "classification",
            "claim_boundary",
            "rank",
            "profile_count",
            "m0_profile_count",
            "active_bounds",
            "profile_identities",
            "m0_profile_identities",
        ),
    )
    _close_vector(
        ordered.get("parameters", ()),
        expected_ordered["parameters"],
        case.tolerances["ordered_parameter_absolute"],
        "ordered_intensity.parameters",
    )
    _close_scalar(
        ordered.get("objective"),
        expected_ordered["objective"],
        case.tolerances["ordered_objective_absolute"],
        "ordered_intensity.objective",
    )
    if terminal >= 3 and "render" in expected:
        _exact(
            actual.get("render", {}),
            expected["render"],
            "render",
            (
                "classification",
                "source_state_count",
                "rod_count",
                "m0_rod_count",
                "raw_decoded_pixel_sha256",
                "simulated_decoded_pixel_sha256",
            ),
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_script_module(name: str, filename: str) -> Any:
    path = ROOT / "scripts" / filename
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load replay implementation {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _source_revision(
    case: ReplayCase,
    *,
    source_state_count: int | None = None,
) -> str:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        load_simulation_config,
    )

    config = load_simulation_config(case.input_paths["simulation_config"])
    sample_count = case.source_state_count if source_state_count is None else source_state_count
    configured = build_configured_simulation_inputs(
        replace(config, source=replace(config.source, sample_count=sample_count))
    )
    if configured.samples.source_seed != case.source_seed:
        raise ValueError("configured source seed does not match the replay case")
    if configured.samples.incident_sample_id.size != sample_count:
        raise RuntimeError("configured source did not produce the requested state count")
    return configured.samples.source_revision


def _stage_source_identity(
    case: ReplayCase,
    stage: str,
    *,
    case_source_revision: str,
) -> tuple[int, int, str]:
    if stage != "geometry":
        return case.source_state_count, case.source_seed, case_source_revision
    from rasim_next.pipeline.configured_simulation import (
        load_simulation_config,
        sample_configured_nominal_geometry_source,
    )

    source_state_count = int(case.stage_config["geometry"]["source_state_count"])
    if source_state_count != 1:
        raise ValueError("geometry stage must declare one nominal geometry companion")
    config = load_simulation_config(case.input_paths["simulation_config"])
    nominal = sample_configured_nominal_geometry_source(config.source)
    return (
        source_state_count,
        nominal.source_seed,
        nominal.source_revision,
    )


def _stage_result(
    stage: str,
    *,
    case: ReplayCase,
    upstream: dict[str, Any] | None,
    backend: str,
    summary: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    upstream_revision = None if upstream is None else upstream["scientific_revision"]
    case_source_revision = _source_revision(case)
    source_state_count, source_seed, source_revision = _stage_source_identity(
        case,
        stage,
        case_source_revision=case_source_revision,
    )
    result = {
        "schema_version": _STAGE_SCHEMA_VERSION,
        "stage": stage,
        "case_id": case.case_id,
        "material_id": case.material_id,
        "stage_case_sha256": _stage_case_sha256(case, stage),
        "execution_backend": backend,
        "runtime": case.runtime_identity,
        "source_state_count": source_state_count,
        "source_seed": source_seed,
        "source_revision": source_revision,
        "upstream_scientific_revision": upstream_revision,
        "scientific_summary": {
            "case_id": case.case_id,
            "material_id": case.material_id,
            "source_state_count": case.source_state_count,
            "source_revision": case_source_revision,
            stage: summary,
        },
        "state": state,
    }
    result["scientific_revision"] = scientific_revision(stage, result)
    return result


_ORDERED_STAGE = _load_script_module(
    "staged_fit_ordered_intensity",
    "staged_fit_ordered_intensity.py",
)
_identity_text = _ORDERED_STAGE.profile_identity_text
_profile_summary = _ORDERED_STAGE.profile_summary
_measured_profile_scales = _ORDERED_STAGE.measured_profile_scales
_transferred_profile_signal = _ORDERED_STAGE.transferred_profile_signal


def _bi2se3_mosaic_artifact_projection(document: dict[str, Any]) -> dict[str, Any]:
    if (
        document.get("schema_version") != "rasim-bi2se3-real-mosaic-fit-v3"
        or document.get("status") != "MODEL_LIMITED_EFFECTIVE_RADIAL_MOSAIC_ESTIMATE"
    ):
        raise ValueError("mosaic stage artifact has an unsupported scientific contract")
    recovered = document.get("recovered_effective_distribution")
    fit = document.get("fit")
    profiles = fit.get("profiles") if isinstance(fit, dict) else None
    if (
        not isinstance(recovered, dict)
        or not isinstance(profiles, list)
        or not all(isinstance(record, dict) for record in profiles)
    ):
        raise ValueError("mosaic stage artifact lacks its fitted profile state")
    parameters = [
        float(recovered["gaussian_sigma_deg"]),
        float(recovered["lorentzian_hwhm_deg"]),
        float(recovered["lorentzian_probability"]),
    ]
    identities, m0_identities, counts = _profile_summary(profiles)
    observations = document.get("observations")
    measured_policy = (
        observations.get("measured_profile_policy") if isinstance(observations, dict) else None
    )
    selection = (
        measured_policy.get("profile_selection") if isinstance(measured_policy, dict) else None
    )
    if not isinstance(selection, list) or not all(isinstance(record, dict) for record in selection):
        raise ValueError("mosaic stage artifact lacks its measured profile selection")
    eligible_identities = sorted(
        _identity_text(record) for record in selection if record.get("fit_eligible") is True
    )
    if eligible_identities != identities:
        raise ValueError("mosaic stage artifact changed its fit-eligible profile selection")
    measured_scales = _measured_profile_scales(document)
    if set(measured_scales) != set(identities):
        raise ValueError("mosaic stage artifact changed its fitted profile amplitudes")
    profile_scales = [
        {"identity": identity, "nuisance_peak_scale": measured_scales[identity]}
        for identity in identities
    ]
    fixed_position = document.get("fixed_geometry")
    source_model = document.get("source_model")
    provenance = document.get("provenance")
    if not all(isinstance(value, dict) for value in (fixed_position, source_model, provenance)):
        raise ValueError("mosaic stage artifact lacks its bound position or provenance")
    return {
        "state": {
            "fixed_position": fixed_position,
            "parameters": parameters,
            "profile_identities": identities,
            "profile_scales": profile_scales,
        },
        "summary": {
            "classification": document["status"],
            "parameters": parameters,
            "objective": float(fit["objective"]),
            "rank": int(fit["sensitivity_rank"]),
            "profile_count": len(identities),
            "m0_profile_count": len(m0_identities),
            "per_incidence_profile_count": counts,
            "profile_identities": identities,
            "m0_profile_identities": m0_identities,
        },
        "source_model": source_model,
        "provenance": provenance,
    }


_bi2se3_ordered_artifact_projection = _ORDERED_STAGE.project_bi2se3_ordered_artifact
_validated_bi2se3_ordered_cached_artifact = _ORDERED_STAGE.validate_cached_bi2se3_ordered_artifact


def _bi2se3_geometry(case: ReplayCase) -> tuple[dict[str, Any], dict[str, Any]]:
    runner = _load_script_module("staged_fit_bi2se3_geometry", "fit_osc_geometry.py")
    config = case.stage_config["geometry"]
    result = runner.fit_osc_geometry_series(
        case.input_paths[str(config["series_role"])],
        heldout_integer_l=tuple(int(value) for value in config["heldout_integer_l"]),
        benchmark=bool(config["benchmark"]),
        fitted_parameter_names=tuple(config["fitted_parameter_names"]),
        fit_incidence_angle_delta=bool(config["fit_incidence_angle_delta"]),
        incidence_angle_delta_half_span_deg=float(config["incidence_angle_delta_half_span_deg"]),
        initial=runner.SharedGeometryCorrections.from_array(config["initial_corrections"]),
        initial_incidence_angle_delta_rad=float(config["initial_incidence_angle_delta_rad"]),
    )
    corrections = result["fit"]["corrections"]
    values = [float(corrections[name]) for name in runner.SHARED_GEOMETRY_PARAMETER_NAMES]
    counts = [
        int(result["image_site_counts"][image_id]) for image_id in result["image_site_counts"]
    ]
    incidence_records = list(result["incidence_angle_correction"]["images"])
    summary = {
        "classification": case.expected_scientific_summary["geometry"]["classification"],
        "selection_revision": result["indexed_manifest_hash"],
        "fitted_parameter_names": list(result["fit"]["fitted_parameter_names"]),
        "fixed_parameter_names": list(result["fit"]["fixed_parameter_names"]),
        "jacobian_parameter_names": list(result["fit"]["jacobian_parameter_names"]),
        "corrections": values,
        "incidence_angle_delta_rad": float(result["fit"]["incidence_angle_delta_rad"]),
        "incidence_angle_image_ids": [str(item["image_id"]) for item in incidence_records],
        "commanded_incidence_angles_rad": [
            float(item["commanded_angle_rad"]) for item in incidence_records
        ],
        "effective_incidence_angles_rad": [
            float(item["effective_angle_rad"]) for item in incidence_records
        ],
        "rank": int(result["fit"]["jacobian_rank"]),
        "active_bounds": list(result["fit"]["active_bounds"]),
        "site_rms_px": float(result["post_fit"]["site_rms_px"]),
        "site_max_px": float(result["post_fit"]["site_max_px"]),
        "per_incidence_profile_count": counts,
    }
    evidence_keys = (
        "schema",
        "indexed_manifest_hash",
        "run_completed",
        "source_state_policy",
        "source_state_count_per_image",
        "baseline",
        "cross_validation",
        "fit",
        "post_fit",
        "selected_start_index",
        "multi_start",
        "pairwise_multistart_prediction_separation",
        "root_audit",
        "outer_audit",
        "qualification",
        "benchmark",
        "geometry_setup_wall_time_seconds",
        "indexing_pass_wall_times_seconds",
        "indexing_wall_time_seconds",
        "initial_start_fit_wall_time_seconds",
        "primary_fit_wall_time_seconds",
    )
    return summary, {
        "corrections": values,
        "incidence_angle_delta_rad": summary["incidence_angle_delta_rad"],
        "incidence_angles": incidence_records,
        "fit_evidence": {name: result[name] for name in evidence_keys},
    }


def _bi2te3_geometry(case: ReplayCase) -> tuple[dict[str, Any], dict[str, Any]]:
    runner = _load_script_module("staged_fit_bi2te3_geometry", "fit_osc_geometry.py")
    stage = case.stage_config["geometry"]
    result = runner.fit_osc_geometry_series(
        case.input_paths[str(stage["series_role"])],
        heldout_integer_l=tuple(int(value) for value in stage["heldout_integer_l"]),
        benchmark=bool(stage["benchmark"]),
        fitted_parameter_names=tuple(stage["fitted_parameter_names"]),
    )
    corrections = result["fit"]["corrections"]
    values = [float(corrections[name]) for name in runner.SHARED_GEOMETRY_PARAMETER_NAMES]
    summary = {
        "classification": case.expected_scientific_summary["geometry"]["classification"],
        "selection_revision": result["indexed_manifest_hash"],
        "frozen_catalog_sha256": _sha256(case.input_paths[str(stage["catalog_role"])]),
        "fitted_parameter_names": list(result["fit"]["fitted_parameter_names"]),
        "fixed_parameter_names": list(result["fit"]["fixed_parameter_names"]),
        "corrections": values,
        "rank": int(result["fit"]["jacobian_rank"]),
        "active_bounds": list(result["fit"]["active_bounds"]),
        "site_rms_px": float(result["post_fit"]["site_rms_px"]),
        "site_max_px": float(result["post_fit"]["site_max_px"]),
        "per_incidence_profile_count": [
            int(result["image_site_counts"][image_id]) for image_id in result["image_site_counts"]
        ],
    }
    return summary, {"corrections": values}


def _run_geometry_stage(
    *,
    case: ReplayCase,
    upstream: dict[str, Any] | None,
    backend: str,
    output_directory: Path,
) -> dict[str, Any]:
    del output_directory
    if upstream is not None:
        raise ValueError("geometry stage cannot have an upstream result")
    summary, state = (
        _bi2se3_geometry(case) if case.material_id == "Bi2Se3" else _bi2te3_geometry(case)
    )
    return _stage_result(
        "geometry",
        case=case,
        upstream=None,
        backend=backend,
        summary=summary,
        state=state,
    )


def _bi2se3_position_projection(
    case: ReplayCase,
    geometry_stage: dict[str, Any],
) -> dict[str, Any]:
    from rasim_next.pipeline.configured_simulation import load_simulation_config

    delta_rad = float(geometry_stage["state"]["incidence_angle_delta_rad"])
    commanded_deg = [float(value) for value in case.incidence_angles_deg]
    beam_center = load_simulation_config(
        case.input_paths["simulation_config"]
    ).instrument.detector_reference_coordinate_px
    return {
        "position_artifact_revision": geometry_stage["scientific_revision"],
        "corrections": {
            name: float(value)
            for name, value in zip(
                _SHARED_GEOMETRY_PARAMETER_NAMES,
                geometry_stage["state"]["corrections"],
                strict=True,
            )
        },
        "incidence_angle_model_id": "commanded_angle_plus_common_delta.v1",
        "incidence_angle_delta_rad": delta_rad,
        "commanded_incidence_angles_deg": commanded_deg,
        "effective_incidence_angles_deg": [
            value + math.degrees(delta_rad) for value in commanded_deg
        ],
        "beam_center_column_row_px": [float(value) for value in beam_center],
        "geometry_parameters_fitted_here": False,
    }


def _bi2se3_mosaic(
    case: ReplayCase,
    upstream: dict[str, Any],
    backend: str,
    output_directory: Path,
) -> dict[str, Any]:
    stage = case.stage_config["mosaic"]
    expected_corrections = case.expected_scientific_summary["geometry"]["corrections"]
    _close_vector(
        upstream["state"]["corrections"],
        expected_corrections,
        case.tolerances["geometry_correction_absolute"],
        "geometry.corrections",
    )
    _close_scalar(
        upstream["state"]["incidence_angle_delta_rad"],
        case.expected_scientific_summary["geometry"]["incidence_angle_delta_rad"],
        case.tolerances["geometry_correction_absolute"],
        "geometry.incidence_angle_delta_rad",
    )
    runner = _load_script_module("staged_fit_bi2se3_mosaic", "recover_bi2se3_mosaic.py")
    artifact_directory = output_directory / "mosaic_artifacts"
    arguments = [
        "--case",
        str(case.input_paths[str(stage["case_role"])]),
        "--measured-profile-policy",
        str(case.input_paths[str(stage["measured_profile_policy_role"])]),
        "--output-directory",
        str(artifact_directory),
        "--observation-mode",
        str(stage["observation_mode"]),
        "--source-sample-count",
        str(case.source_state_count),
        "--execution-backend",
        backend,
        "--position-artifact",
        str(output_directory / "geometry.json"),
        "--skip-images",
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        runner.main(arguments)
    artifact = artifact_directory / "bi2se3_real_mosaic_fit.json"
    document = json.loads(artifact.read_text(encoding="utf-8"))
    projection = _bi2se3_mosaic_artifact_projection(document)
    expected_fixed_geometry = _bi2se3_position_projection(case, upstream)
    if projection["state"]["fixed_position"] != expected_fixed_geometry:
        raise RuntimeError("Bi2Se3 mosaic artifact changed its upstream position state")
    source_model = projection["source_model"]
    if (
        source_model.get("sample_count") != case.source_state_count
        or source_model.get("source_seed") != case.source_seed
        or source_model.get("source_revision") != _source_revision(case)
        or source_model.get("reduction")
        != "one_incoherent_weighted_detector_function_per_incidence.v1"
    ):
        raise RuntimeError("Bi2Se3 mosaic artifact changed its source realization")
    if projection["provenance"].get("case_sha256") != _sha256(
        case.input_paths[str(stage["case_role"])]
    ):
        raise RuntimeError("Bi2Se3 mosaic artifact changed its immutable case")
    summary = projection["summary"]
    state = {
        "artifact": str(artifact),
        "artifact_sha256": _sha256(artifact),
        **projection["state"],
    }
    return _stage_result(
        "mosaic",
        case=case,
        upstream=upstream,
        backend=backend,
        summary=summary,
        state=state,
    )


def _bi2te3_fixed_inputs(
    case: ReplayCase,
    position: Any,
    fixed_lattice: Any,
    *,
    simulation_config_path: Path | None = None,
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    from rasim_next.fitting import (
        FixedLatticeState,
        FixedPositionState,
        build_fixed_experiment_series,
    )
    from rasim_next.pipeline.configured_simulation import (
        build_configured_geometry_inputs,
        load_simulation_config,
    )

    if not isinstance(position, FixedPositionState):
        raise TypeError("position must be FixedPositionState")
    if not isinstance(fixed_lattice, FixedLatticeState):
        raise TypeError("fixed_lattice must be FixedLatticeState")
    commanded_rad = tuple(math.radians(value) for value in case.incidence_angles_deg)
    if position.commanded_incidence_angles_rad != commanded_rad:
        raise ValueError("position incidence series differs from the mosaic case")

    config = load_simulation_config(
        case.input_paths["simulation_config"]
        if simulation_config_path is None
        else simulation_config_path
    )
    mosaic_arguments = {
        "gaussian_sigma_rad": math.radians(config.mosaic.gaussian_sigma_deg),
        "lorentzian_half_width_rad": math.radians(config.mosaic.lorentzian_hwhm_deg),
        "lorentzian_probability": config.mosaic.lorentzian_probability,
    }
    series = build_fixed_experiment_series(
        config,
        position=position,
        fixed_lattice=fixed_lattice,
        source_sample_count=case.source_state_count,
        **mosaic_arguments,
    )
    nominal_series = tuple(
        replace(
            build_configured_geometry_inputs(
                inputs.config,
                direct_basis_A=fixed_lattice.direct_basis_override_A,
            ),
            instrument=inputs.instrument,
        )
        for inputs in series
    )
    return series[0], series, nominal_series


def _bi2te3_fixed_state_from_stage(
    state: dict[str, Any],
    *,
    case: ReplayCase | None = None,
) -> tuple[Any, Any, Path]:
    """Restore the exact position, lattice, and config bound by a Bi2Te3 stage."""

    from rasim_next.fitting import FixedLatticeState, FixedPositionState
    from rasim_next.materials import read_crystal
    from rasim_next.pipeline.configured_simulation import load_simulation_config

    fixed_position = FixedPositionState.from_record(state.get("fixed_position"))
    fixed_lattice_record = state.get("fixed_lattice")
    config_record = state.get("simulation_config")
    if not isinstance(config_record, dict) or set(config_record) != {"path", "sha256"}:
        raise ValueError("Bi2Te3 stage simulation config identity is invalid")
    config_path = Path(str(config_record["path"])).resolve()
    if case is not None and config_path != case.input_paths["simulation_config"].resolve():
        raise ValueError("Bi2Te3 stage changed its simulation config path")
    if not config_path.is_file() or _sha256(config_path) != config_record["sha256"]:
        raise ValueError("Bi2Te3 stage simulation config changed")
    config = load_simulation_config(config_path)
    crystal = read_crystal(
        config.material.cif_path,
        phase_id=config.material.phase_id,
        expected_sha256=config.cif_sha256,
    )
    fixed_lattice = FixedLatticeState.from_record(
        fixed_lattice_record,
        reference_direct_basis_A=crystal.direct_basis_A,
    )
    return fixed_position, fixed_lattice, config_path


def _validate_bi2te3_fixed_state_handoff(
    case: ReplayCase,
    *,
    stage: str,
    state: dict[str, Any],
    upstream: dict[str, Any] | None,
) -> None:
    """Bind persisted Bi2Te3 geometry state to its exact upstream and CIF inputs."""

    from rasim_next.fitting import FixedLatticeState, FixedPositionState
    from rasim_next.fitting.indexed_series import SharedGeometryCorrections
    from rasim_next.materials import read_crystal
    from rasim_next.pipeline.configured_simulation import load_simulation_config

    fixed_position, fixed_lattice, config_path = _bi2te3_fixed_state_from_stage(
        state,
        case=case,
    )
    if stage == "mosaic":
        if upstream is None or upstream.get("stage") != "geometry":
            raise ValueError("Bi2Te3 mosaic stage requires its geometry artifact")
        config = load_simulation_config(config_path)
        expected_position = FixedPositionState(
            artifact_revision=upstream["scientific_revision"],
            corrections=SharedGeometryCorrections.from_array(upstream["state"]["corrections"]),
            incidence_angle_delta_rad=0.0,
            commanded_incidence_angles_rad=tuple(
                math.radians(value) for value in case.incidence_angles_deg
            ),
            beam_center_column_row_px=tuple(
                float(value) for value in config.instrument.detector_reference_coordinate_px
            ),
        )
        crystal = read_crystal(
            config.material.cif_path,
            phase_id=config.material.phase_id,
            expected_sha256=config.cif_sha256,
        )
        expected_lattice = FixedLatticeState.implicit_cif(crystal.direct_basis_A)
        if fixed_position.to_record() != expected_position.to_record():
            raise ValueError("Bi2Te3 mosaic stage changed its geometry position handoff")
        if fixed_lattice.to_record() != expected_lattice.to_record():
            raise ValueError("Bi2Te3 mosaic stage changed its CIF lattice handoff")
        return
    if stage == "ordered_intensity":
        if upstream is None or upstream.get("stage") != "mosaic":
            raise ValueError("Bi2Te3 ordered-intensity stage requires its mosaic artifact")
        for name in ("fixed_position", "fixed_lattice", "simulation_config"):
            if state.get(name) != upstream["state"].get(name):
                raise ValueError(f"Bi2Te3 ordered-intensity stage changed its {name} handoff")
        return
    raise ValueError(f"Bi2Te3 fixed-state handoff is not defined for stage {stage!r}")


def _local_peak_centroid(
    counts: Any,
    dark: Any,
    column_px: float,
    row_px: float,
    *,
    radius_px: float = 12.0,
) -> tuple[float, float]:
    import numpy as np

    row0 = max(0, math.floor(row_px - radius_px))
    row1 = min(counts.shape[0], math.ceil(row_px + radius_px + 1.0))
    column0 = max(0, math.floor(column_px - radius_px))
    column1 = min(counts.shape[1], math.ceil(column_px + radius_px + 1.0))
    rows, columns = np.mgrid[row0:row1, column0:column1]
    radial_px = np.hypot(columns - column_px, rows - row_px)
    corrected = counts[row0:row1, column0:column1].astype(np.float64) - dark[
        row0:row1, column0:column1
    ].astype(np.float64)
    background = float(np.median(corrected[(radial_px >= 9.0) & (radial_px <= radius_px)]))
    weight = np.maximum(corrected - background, 0.0)
    weight[radial_px > 8.0] = 0.0
    total = float(np.sum(weight))
    if total <= 0.0:
        raise RuntimeError("m=0 centroid has no positive local excess")
    return float(np.sum(columns * weight) / total), float(np.sum(rows * weight) / total)


def _bi2te3_profile_definitions(
    case: ReplayCase,
    nominal_series: tuple[Any, ...],
    evaluation_series: tuple[Any, ...],
    dark_counts: Any,
    mosaic_runner: Any,
) -> tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...], list[dict[str, Any]]]:
    import numpy as np

    from rasim_next.fitting import (
        ExactTagGeometryModel,
        MosaicProfileDefinition,
        MosaicProfileIdentity,
        MosaicReflectionGroupKey,
        probe_ordered_intensity_inverse_boundary_bins,
    )
    from rasim_next.geometry import detector_coordinates_to_angles
    from rasim_next.io.osc import read_osc
    from rasim_next.pipeline.configured_simulation import (
        build_nominal_ewald_context,
        build_source_averaged_structure_detector,
        evaluate_nominal_integer_l_markers,
    )
    from rasim_next.selection import build_osc_angle_frame, load_osc_geometry_series

    stage = case.stage_config["mosaic"]
    geometry_series = load_osc_geometry_series(case.input_paths["geometry_series"])
    image_id_by_angle = {
        float(image.axis_rotation_angles_deg[geometry_series.incidence_axis_index]): image.image_id
        for image in geometry_series.images
    }
    catalog = json.loads(case.input_paths[str(stage["catalog_role"])].read_text(encoding="utf-8"))
    rows = catalog["rows"]
    selected: dict[float, list[dict[str, Any]]] = {angle: [] for angle in case.incidence_angles_deg}
    for angle in case.incidence_angles_deg:
        allowed_l = set(int(value) for value in stage[f"m1_integer_l_{angle:g}deg"])
        selected[angle].extend(
            row
            for row in rows
            if row["image_id"] == image_id_by_angle[angle]
            and int(row["family_m"]) == 1
            and int(row["integer_L"]) in allowed_l
        )
    for record in stage.get("extra_nonzero_profiles", []):
        angle = float(record["incidence_angle_deg"])
        selected[angle].append(
            {
                "family_m": int(record["family_m"]),
                "integer_L": int(record["integer_L"]),
                "branch": 2,
                "root_sign": int(record["root_sign"]),
                "column_px": float(record["column_px"]),
                "row_px": float(record["row_px"]),
            }
        )

    frames = []
    datasets = []
    m0_audit: list[dict[str, Any]] = []
    for angle, inputs, evaluation_inputs, osc_role, dataset_id in zip(
        case.incidence_angles_deg,
        nominal_series,
        evaluation_series,
        stage["osc_roles"],
        stage["dataset_ids"],
        strict=True,
    ):
        effective_incidence_rad = math.radians(
            inputs.config.instrument.axis_rotations[geometry_series.incidence_axis_index].angle_deg
        )
        context = build_nominal_ewald_context(evaluation_inputs)
        topology_detector = build_source_averaged_structure_detector(evaluation_inputs)
        frame = build_osc_angle_frame(
            mean_direction_lab=inputs.config.source.mean_direction_lab,
            instrument=inputs.instrument,
            sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
            revision=f"bi2te3-fixed-geometry-{angle:g}deg.v1",
        )
        frames.append(frame)
        markers = evaluate_nominal_integer_l_markers(context)
        marker_map: dict[tuple[int, int, int], int] = {}
        for index in range(markers.family_m.size):
            family_m = int(markers.family_m[index])
            root_sign = int(markers.root_sign[index])
            if family_m == 0 or root_sign == 0 or int(markers.branch[index]) != 2:
                continue
            key = (family_m, int(markers.integer_L[index]), root_sign)
            if key in marker_map:
                raise RuntimeError(f"nonunique nominal marker {key}")
            marker_map[key] = index
        definitions = []
        counts = read_osc(case.input_paths[str(osc_role)]).detector_native_counts
        for row in selected[angle]:
            key = (int(row["family_m"]), int(row["integer_L"]), int(row["root_sign"]))
            if key not in marker_map:
                raise RuntimeError(f"measured nonzero peak lacks a nominal marker: {angle}, {key}")
            marker_index = marker_map[key]
            coordinates = detector_coordinates_to_angles(
                np.asarray([row["column_px"]]),
                np.asarray([row["row_px"]]),
                instrument=inputs.instrument,
                angle_frame=frame,
            )
            if not bool(coordinates.valid[0] & coordinates.azimuth_valid[0]):
                raise RuntimeError(f"invalid observed angle coordinate: {angle}, {key}")
            group = MosaicReflectionGroupKey(
                group_id=f"bi2te3:m={key[0]}:L={key[1]}",
                rod_catalog_revision=mosaic_runner.configured_rod_catalog_revision(
                    evaluation_inputs
                ),
                member_rod_hk=markers.contributing_rod_hk[marker_index],
                branch_mode="EXPLICIT_NONZERO",
                layered_family_m=key[0],
                layered_integer_L=key[1],
            )
            definitions.append(
                MosaicProfileDefinition(
                    identity=MosaicProfileIdentity(
                        dataset_id=str(dataset_id),
                        incidence_angle_rad=effective_incidence_rad,
                        group_key=group,
                        branch_id=1 if key[2] < 0 else 2,
                        analytic_branch_id=2,
                    ),
                    center_two_theta_rad=float(coordinates.two_theta_rad[0]),
                    center_phi_rad=float(coordinates.phi_rad[0]),
                    two_theta_half_width_rad=math.radians(float(stage["two_theta_half_width_deg"])),
                    phi_half_width_rad=math.radians(float(stage["nonzero_phi_half_width_deg"])),
                    phi_bin_count=int(stage["phi_bin_count"]),
                    two_theta_gauss_order=int(stage["nonzero_two_theta_gauss_order"]),
                    phi_gauss_order=int(stage["nonzero_phi_gauss_order"]),
                )
            )
        geometry_model = ExactTagGeometryModel(inputs)
        m0_rods = tuple(rod for rod in inputs.rods if rod.family_m == 0)
        if len(m0_rods) != 1:
            raise RuntimeError("expected exactly one physical m=0 rod")
        for raw_start in stage[f"m0_starts_{angle:g}deg"]:
            integer_l, start_column, start_row = (
                int(raw_start[0]),
                float(raw_start[1]),
                float(raw_start[2]),
            )
            column_px, row_px = _local_peak_centroid(
                counts,
                dark_counts,
                start_column,
                start_row,
            )
            observed = detector_coordinates_to_angles(
                np.asarray([column_px]),
                np.asarray([row_px]),
                instrument=inputs.instrument,
                angle_frame=frame,
            )
            prediction = geometry_model.predict_m0_minimum_tilt_exact_l_landmarks((integer_l,))
            distance_px = float(np.linalg.norm(prediction.coordinates_px[0] - (column_px, row_px)))
            accepted = distance_px <= float(stage["m0_landmark_maximum_distance_px"])
            m0_audit.append(
                {
                    "dataset_id": str(dataset_id),
                    "family_m": 0,
                    "integer_L": integer_l,
                    "fixed_geometry_distance_px": distance_px,
                    "accepted": accepted,
                }
            )
            if not accepted:
                continue
            group = MosaicReflectionGroupKey(
                group_id=f"bi2te3:m=0:L={integer_l}",
                rod_catalog_revision=mosaic_runner.configured_rod_catalog_revision(
                    evaluation_inputs
                ),
                member_rod_hk=((m0_rods[0].h, m0_rods[0].k),),
                branch_mode="COLLAPSED_00L",
                layered_family_m=0,
                layered_integer_L=integer_l,
            )
            definitions.append(
                MosaicProfileDefinition(
                    identity=MosaicProfileIdentity(
                        dataset_id=str(dataset_id),
                        incidence_angle_rad=effective_incidence_rad,
                        group_key=group,
                        branch_id=None,
                        analytic_branch_id=0,
                    ),
                    center_two_theta_rad=float(observed.two_theta_rad[0]),
                    center_phi_rad=float(observed.phi_rad[0]),
                    two_theta_half_width_rad=math.radians(float(stage["two_theta_half_width_deg"])),
                    phi_half_width_rad=math.radians(float(stage["m0_phi_half_width_deg"])),
                    phi_bin_count=int(stage["phi_bin_count"]),
                    two_theta_gauss_order=int(stage["m0_two_theta_gauss_order"]),
                    phi_gauss_order=int(stage["m0_phi_gauss_order"]),
                )
            )
        datasets.append(
            probe_ordered_intensity_inverse_boundary_bins(
                topology_detector,
                angle_frame=frame,
                definitions=tuple(definitions),
            )
        )
    return tuple(frames), tuple(datasets), m0_audit


def _bi2te3_observations(
    case: ReplayCase,
    series: tuple[Any, ...],
    frames: tuple[Any, ...],
    definitions: tuple[tuple[Any, ...], ...],
    model: Any,
    dark_counts: Any,
    profile_revision: str,
    mosaic_runner: Any,
) -> tuple[Any, tuple[tuple[Any, ...], ...], list[dict[str, Any]]]:
    import numpy as np

    from rasim_next.fitting import MosaicProfileSet
    from rasim_next.io.osc import read_osc
    from rasim_next.measurement import compile_detector_profile_projector, project_detector_profiles
    from rasim_next.selection import detector_valid_mask_from_counts

    stage = case.stage_config["mosaic"]
    selected_definitions = []
    signals = []
    normalizations = []
    valid_masks = []
    phi_edges = []
    two_theta_bounds = []
    frame_revisions = []
    selection_records: list[dict[str, Any]] = []
    start = 0
    sideband_offsets = tuple(
        math.radians(float(value)) for value in stage["sideband_two_theta_offsets_deg"]
    )
    for inputs, frame, dataset_definitions, osc_role in zip(
        series,
        frames,
        definitions,
        stage["osc_roles"],
        strict=True,
    ):
        stop = start + len(dataset_definitions)
        model_valid = model.valid[start:stop]
        model_signal = model.signal[start:stop]
        counts = read_osc(case.input_paths[str(osc_role)]).detector_native_counts
        detector_mask = detector_valid_mask_from_counts(counts)
        projector = compile_detector_profile_projector(
            instrument=inputs.instrument,
            angle_frame=frame,
            two_theta_bounds_rad=model.two_theta_bounds_rad[start:stop],
            phi_bin_edges_rad=model.phi_bin_edges_rad[start:stop],
            detector_valid_mask=detector_mask,
            profile_bin_valid_mask=model_valid,
        )
        raw = project_detector_profiles(projector, counts)
        dark = project_detector_profiles(projector, dark_counts)
        raw_side = mosaic_runner._sample_profile_centerline_sidebands(
            detector_counts=counts,
            detector_valid_mask=detector_mask,
            instrument=inputs.instrument,
            angle_frame=frame,
            two_theta_bounds_rad=model.two_theta_bounds_rad[start:stop],
            phi_bin_edges_rad=model.phi_bin_edges_rad[start:stop],
            offsets_rad=sideband_offsets,
        )
        dark_side = mosaic_runner._sample_profile_centerline_sidebands(
            detector_counts=dark_counts,
            detector_valid_mask=detector_mask,
            instrument=inputs.instrument,
            angle_frame=frame,
            two_theta_bounds_rad=model.two_theta_bounds_rad[start:stop],
            phi_bin_edges_rad=model.phi_bin_edges_rad[start:stop],
            offsets_rad=sideband_offsets,
        )
        side = raw_side - dark_side
        background = np.median(side, axis=0)
        scatter = 1.4826 * np.median(np.abs(side - background[None, ...]), axis=0)
        corrected_signal = raw.S - dark.S
        corrected_intensity = np.zeros_like(corrected_signal)
        np.divide(corrected_signal, raw.N, out=corrected_intensity, where=model_valid)
        keep = []
        for local_index, definition in enumerate(dataset_definitions):
            active = model_valid[local_index]
            excess = np.maximum(
                corrected_intensity[local_index, active] - background[local_index, active],
                0.0,
            )
            noise = np.maximum(scatter[local_index, active], 1.0)
            significance = float(np.linalg.norm(excess) / np.linalg.norm(noise))
            support = float(np.sum(model_signal[local_index, active]))
            accepted = bool(
                significance >= float(stage["minimum_excess_energy_over_side_scatter"])
                and support > np.finfo(np.float64).tiny
            )
            identity = definition.identity
            selection_records.append(
                {
                    "dataset_id": identity.dataset_id,
                    "family_m": identity.group_key.layered_family_m,
                    "integer_L": identity.group_key.layered_integer_L,
                    "root_side_branch_id": identity.branch_id,
                    "dark_subtracted_excess_significance": significance,
                    "source_averaged_modeled_signal_A2": support,
                    "accepted": accepted,
                }
            )
            if not accepted:
                continue
            keep.append(local_index)
            minimum = float(np.min(corrected_intensity[local_index, active]))
            nonnegative_offset = max(0.0, -minimum) + 1.0e-9
            restored = corrected_signal[local_index] + nonnegative_offset * raw.N[local_index]
            restored[~active] = 0.0
            signals.append(restored)
            normalizations.append(raw.N[local_index])
            valid_masks.append(active)
            phi_edges.append(model.phi_bin_edges_rad[start + local_index])
            two_theta_bounds.append(model.two_theta_bounds_rad[start + local_index])
            frame_revisions.append(frame.revision)
        selected_definitions.append(tuple(dataset_definitions[index] for index in keep))
        start = stop
    observation_revision = (
        "sha256-"
        + hashlib.sha256(
            json.dumps(selection_records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    observations = MosaicProfileSet(
        identities=tuple(
            definition.identity for dataset in selected_definitions for definition in dataset
        ),
        signal=np.asarray(signals),
        normalization=np.asarray(normalizations),
        valid=np.asarray(valid_masks),
        profile_revision=profile_revision,
        phi_bin_edges_rad=np.asarray(phi_edges),
        two_theta_bounds_rad=np.asarray(two_theta_bounds),
        angle_frame_revisions=tuple(frame_revisions),
        source_revision=None,
        observation_revision=observation_revision,
    )
    return observations, tuple(selected_definitions), selection_records


def _recover_global_mosaic_alias(error: Any) -> tuple[Any, list[dict[str, float | None]]]:
    """Retain a fitted representative only for an explicit, locally identified alias."""

    result = error.candidate_result
    if error.reason != "global_alias" or result is None:
        raise error
    matching_sets = [
        parameter_set
        for parameter_set in error.competing_parameter_sets
        if parameter_set.gaussian_sigma_rad == result.gaussian_sigma_rad
        and parameter_set.lorentzian_half_width_rad == result.lorentzian_half_width_rad
        and parameter_set.lorentzian_probability == result.lorentzian_probability
    ]
    if len(matching_sets) != 1 or not math.isclose(
        matching_sets[0].objective,
        result.objective,
        rel_tol=2.0e-12,
        abs_tol=2.0e-14 * max(1.0, result.objective),
    ):
        raise RuntimeError("global mosaic alias does not contain its fitted representative")
    return result, [
        {
            "gaussian_sigma_deg": (
                None
                if parameter_set.gaussian_sigma_rad is None
                else math.degrees(parameter_set.gaussian_sigma_rad)
            ),
            "lorentzian_hwhm_deg": (
                None
                if parameter_set.lorentzian_half_width_rad is None
                else math.degrees(parameter_set.lorentzian_half_width_rad)
            ),
            "lorentzian_probability": parameter_set.lorentzian_probability,
            "objective": parameter_set.objective,
        }
        for parameter_set in error.competing_parameter_sets
    ]


def _checkpointed_mosaic_component_evaluator(
    evaluator: Callable[[float], Any],
    *,
    cache_directory: Path,
    component_kind: str,
    cache_revision: str,
    observations: Any,
    source_revision: str,
    execution_backend: str,
) -> Callable[[float], Any]:
    """Persist exact continuous component profiles after every completed evaluation."""

    import numpy as np

    from rasim_next.fitting import MosaicProfileSet

    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if component_kind not in {"gaussian", "lorentzian"}:
        raise ValueError("component_kind must be gaussian or lorentzian")
    if not isinstance(observations, MosaicProfileSet):
        raise TypeError("observations must be a MosaicProfileSet")
    for name, value in (
        ("cache_revision", cache_revision),
        ("source_revision", source_revision),
        ("execution_backend", execution_backend),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty string")
    cache_path = cache_directory / "mosaic_component_profiles.checkpoint.npz"
    keys = {
        "schema_version",
        "cache_revision",
        "profile_revision",
        "source_revision",
        "execution_backend",
        "execution_device",
        "component_kind",
        "width_rad",
        "signal",
        "normalization",
        "valid",
    }

    def load_records() -> tuple[dict[tuple[str, float], MosaicProfileSet], str | None]:
        if not cache_path.is_file():
            return {}, None
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                if set(archive.files) != keys:
                    raise ValueError("component checkpoint has unexpected fields")
                metadata = {
                    name: str(archive[name].item())
                    for name in (
                        "schema_version",
                        "cache_revision",
                        "profile_revision",
                        "source_revision",
                        "execution_backend",
                        "execution_device",
                    )
                }
                if metadata != {
                    "schema_version": _MOSAIC_COMPONENT_CACHE_SCHEMA,
                    "cache_revision": cache_revision,
                    "profile_revision": observations.profile_revision,
                    "source_revision": source_revision,
                    "execution_backend": execution_backend,
                    "execution_device": metadata["execution_device"],
                }:
                    raise ValueError("component checkpoint provenance changed")
                kinds = np.asarray(archive["component_kind"])
                widths = np.asarray(archive["width_rad"], dtype=np.float64)
                signals = np.asarray(archive["signal"], dtype=np.float64)
                normalizations = np.asarray(archive["normalization"], dtype=np.float64)
                valid_masks = np.asarray(archive["valid"], dtype=np.bool_)
                count = widths.size
                expected_shape = (count, *observations.signal.shape)
                if (
                    kinds.shape != (count,)
                    or signals.shape != expected_shape
                    or normalizations.shape != expected_shape
                    or valid_masks.shape != expected_shape
                    or np.any(~np.isfinite(widths))
                    or np.any(widths <= 0.0)
                    or any(str(kind) not in {"gaussian", "lorentzian"} for kind in kinds)
                ):
                    raise ValueError("component checkpoint arrays are malformed")
                records: dict[tuple[str, float], MosaicProfileSet] = {}
                device = metadata["execution_device"] or None
                for index, (kind, width) in enumerate(zip(kinds, widths, strict=True)):
                    key = (str(kind), float(width))
                    if key in records:
                        raise ValueError("component checkpoint contains duplicate profiles")
                    records[key] = MosaicProfileSet(
                        identities=observations.identities,
                        signal=signals[index],
                        normalization=normalizations[index],
                        valid=valid_masks[index],
                        profile_revision=observations.profile_revision,
                        phi_bin_edges_rad=observations.phi_bin_edges_rad,
                        two_theta_bounds_rad=observations.two_theta_bounds_rad,
                        angle_frame_revisions=observations.angle_frame_revisions,
                        source_revision=source_revision,
                        execution_backend=execution_backend,
                        execution_device=device,
                    )
                return records, device
        except (KeyError, OSError, TypeError, ValueError):
            return {}, None

    def validate(profile: Any) -> MosaicProfileSet:
        if not isinstance(profile, MosaicProfileSet):
            raise TypeError("component evaluator must return MosaicProfileSet")
        if (
            profile.identities != observations.identities
            or profile.profile_revision != observations.profile_revision
            or profile.angle_frame_revisions != observations.angle_frame_revisions
            or profile.source_revision != source_revision
            or profile.execution_backend != execution_backend
            or profile.observation_revision is not None
            or not np.array_equal(profile.phi_bin_edges_rad, observations.phi_bin_edges_rad)
            or not np.array_equal(profile.two_theta_bounds_rad, observations.two_theta_bounds_rad)
        ):
            raise ValueError("component profile changed the frozen fit identity")
        return profile

    def write_records(
        records: dict[tuple[str, float], MosaicProfileSet],
        execution_device: str | None,
    ) -> None:
        ordered = sorted(records.items(), key=lambda item: (item[0][0], item[0][1]))
        cache_directory.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.npz")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema_version=np.asarray(_MOSAIC_COMPONENT_CACHE_SCHEMA),
                cache_revision=np.asarray(cache_revision),
                profile_revision=np.asarray(observations.profile_revision),
                source_revision=np.asarray(source_revision),
                execution_backend=np.asarray(execution_backend),
                execution_device=np.asarray(execution_device or ""),
                component_kind=np.asarray([key[0] for key, _ in ordered]),
                width_rad=np.asarray([key[1] for key, _ in ordered], dtype=np.float64),
                signal=np.stack([profile.signal for _, profile in ordered]),
                normalization=np.stack([profile.normalization for _, profile in ordered]),
                valid=np.stack([profile.valid for _, profile in ordered]),
            )
        temporary.replace(cache_path)

    def checkpointed(width_rad: float) -> MosaicProfileSet:
        width = float(width_rad)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("component width must be finite and positive")
        records, execution_device = load_records()
        key = (component_kind, width)
        if key in records:
            return records[key]
        profile = validate(evaluator(width))
        if execution_device is not None and profile.execution_device != execution_device:
            raise ValueError("component evaluations changed execution device")
        records[key] = profile
        write_records(records, profile.execution_device)
        return profile

    return checkpointed


def _bi2te3_mosaic(
    case: ReplayCase,
    upstream: dict[str, Any],
    backend: str,
    output_directory: Path,
    *,
    fixed_position: Any | None = None,
    fixed_lattice: Any | None = None,
    simulation_config_path: Path | None = None,
) -> dict[str, Any]:
    if backend != "cuda":
        raise ValueError("the accepted Bi2Te3 mosaic replay is CUDA-qualified only")
    from rasim_next.fitting import (
        ORDERED_INTENSITY_TOPOLOGY_PROBE_REVISION,
        FixedLatticeState,
        FixedPositionState,
        MosaicIdentifiabilityError,
    )
    from rasim_next.fitting.indexed_series import SharedGeometryCorrections
    from rasim_next.io.osc import read_osc
    from rasim_next.materials import read_crystal
    from rasim_next.pipeline.configured_simulation import load_simulation_config

    mosaic_runner = _load_script_module(
        "staged_fit_bi2te3_mosaic_physics", "recover_bi2se3_mosaic.py"
    )
    config_path = (
        case.input_paths["simulation_config"]
        if simulation_config_path is None
        else simulation_config_path.resolve()
    )
    config = load_simulation_config(config_path)
    if fixed_position is None:
        fixed_position_record = upstream["state"].get("fixed_position")
        if fixed_position_record is not None:
            fixed_position = FixedPositionState.from_record(fixed_position_record)
        else:
            corrections = [float(value) for value in upstream["state"]["corrections"]]
            fixed_position = FixedPositionState(
                artifact_revision=upstream["scientific_revision"],
                corrections=SharedGeometryCorrections.from_array(corrections),
                incidence_angle_delta_rad=float(
                    upstream["state"].get("incidence_angle_delta_rad", 0.0)
                ),
                commanded_incidence_angles_rad=tuple(
                    math.radians(value) for value in case.incidence_angles_deg
                ),
                beam_center_column_row_px=tuple(
                    float(value) for value in config.instrument.detector_reference_coordinate_px
                ),
            )
    if not isinstance(fixed_position, FixedPositionState):
        raise TypeError("fixed_position must be FixedPositionState")
    if fixed_lattice is None:
        crystal = read_crystal(
            config.material.cif_path,
            phase_id=config.material.phase_id,
            expected_sha256=config.cif_sha256,
        )
        fixed_lattice = FixedLatticeState.implicit_cif(crystal.direct_basis_A)
    if not isinstance(fixed_lattice, FixedLatticeState):
        raise TypeError("fixed_lattice must be FixedLatticeState")
    base, series, nominal_series = _bi2te3_fixed_inputs(
        case,
        fixed_position,
        fixed_lattice,
        simulation_config_path=config_path,
    )
    if base.samples.source_revision != _source_revision(case):
        raise RuntimeError("Bi2Te3 mosaic replay changed its source realization")
    stage = case.stage_config["mosaic"]
    dark_counts = read_osc(case.input_paths[str(stage["dark_role"])]).detector_native_counts
    frames, definitions, _m0_audit = _bi2te3_profile_definitions(
        case,
        nominal_series,
        series,
        dark_counts,
        mosaic_runner,
    )
    definition_payload = [
        {
            "dataset_id": definition.identity.dataset_id,
            "family_m": definition.identity.group_key.layered_family_m,
            "integer_L": definition.identity.group_key.layered_integer_L,
            "root_side_branch_id": definition.identity.branch_id,
            "analytic_branch_id": definition.identity.analytic_branch_id,
            "center_two_theta_rad": definition.center_two_theta_rad,
            "center_phi_rad": definition.center_phi_rad,
            "excluded_phi_bin_indices": list(definition.excluded_phi_bin_indices),
        }
        for dataset in definitions
        for definition in dataset
    ]
    profile_revision = (
        "sha256-"
        + hashlib.sha256(
            json.dumps(
                {
                    "geometry_revision": fixed_position.artifact_revision,
                    "fixed_lattice": fixed_lattice.to_record(),
                    "simulation_config_sha256": _sha256(config_path),
                    "physics_revision": base.config.physics_revision,
                    "cif_sha256": base.config.cif_sha256,
                    "source_revision": base.samples.source_revision,
                    "topology_probe_revision": (ORDERED_INTENSITY_TOPOLOGY_PROBE_REVISION),
                    "catalog_sha256": _sha256(case.input_paths[str(stage["catalog_role"])]),
                    "definitions": definition_payload,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    physics, profile_geometry = mosaic_runner._profile_forward_contexts(base, series)
    layout, _ = mosaic_runner._evaluate_profile_series(
        physics,
        profile_geometry,
        frames,
        definitions,
        mosaic_runner._mosaic_parameters(
            gaussian_sigma_rad=math.radians(1.0),
            lorentzian_half_width_rad=1.0,
            lorentzian_probability=0.0,
            context=physics,
        ),
        profile_revision=profile_revision,
        execution_backend=backend,
    )
    observations, fitted_definitions, _selection = _bi2te3_observations(
        case,
        series,
        frames,
        definitions,
        layout,
        dark_counts,
        profile_revision,
        mosaic_runner,
    )
    gaussian_evaluator, lorentzian_evaluator = mosaic_runner._component_profile_evaluators(
        physics=physics,
        geometry=profile_geometry,
        frames=frames,
        definitions=fitted_definitions,
        profile_revision=profile_revision,
        execution_backend=backend,
    )
    cache_revision = (
        _SHA256_PREFIX
        + hashlib.sha256(
            json.dumps(
                {
                    "schema_version": _MOSAIC_COMPONENT_CACHE_SCHEMA,
                    "stage_case_sha256": _stage_case_sha256(case, "mosaic"),
                    "profile_revision": profile_revision,
                    "source_revision": base.samples.source_revision,
                    "execution_backend": layout.execution_backend,
                    "execution_device": layout.execution_device,
                    "implementation_sha256": {
                        "mosaic_runner": _sha256(Path(mosaic_runner.__file__)),
                        "mosaic_core": _sha256(ROOT / "src/rasim_next/fitting/mosaic.py"),
                        "source_averaged_detector": _sha256(
                            ROOT / "src/rasim_next/pipeline/source_averaged_detector.py"
                        ),
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    cache_directory = output_directory / ".mosaic_component_cache"
    gaussian = _checkpointed_mosaic_component_evaluator(
        gaussian_evaluator,
        cache_directory=cache_directory,
        component_kind="gaussian",
        cache_revision=cache_revision,
        observations=observations,
        source_revision=base.samples.source_revision,
        execution_backend=str(layout.execution_backend),
    )
    lorentzian = _checkpointed_mosaic_component_evaluator(
        lorentzian_evaluator,
        cache_directory=cache_directory,
        component_kind="lorentzian",
        cache_revision=cache_revision,
        observations=observations,
        source_revision=base.samples.source_revision,
        execution_backend=str(layout.execution_backend),
    )
    alias_parameter_sets: list[dict[str, float | None]] = []
    try:
        search, _ = mosaic_runner._fit_profiles(
            observations=observations,
            evaluate_gaussian_profile=gaussian,
            evaluate_lorentzian_profile=lorentzian,
            search_config={
                name: stage[name]
                for name in (
                    "gaussian_sigma_bounds_deg",
                    "lorentzian_hwhm_bounds_deg",
                    "coarse_width_count",
                    "refinement_width_count",
                    "refinement_levels",
                    "near_optimal_objective_delta",
                    "maximum_sensitivity_condition",
                )
            },
            nuisance_basis=mosaic_runner._constant_profile_background_basis(observations),
        )
        result = search.fit
    except MosaicIdentifiabilityError as error:
        result, alias_parameter_sets = _recover_global_mosaic_alias(error)
    records = [
        {
            "dataset_id": identity.dataset_id,
            "family_m": identity.group_key.layered_family_m,
            "integer_L": identity.group_key.layered_integer_L,
            "root_side_branch_id": identity.branch_id,
            "analytic_branch_id": identity.analytic_branch_id,
            "nuisance_peak_scale": float(result.profile_scales[index]),
        }
        for index, identity in enumerate(result.profile_identities)
    ]
    identities, m0_identities, counts = _profile_summary(records)
    parameters = mosaic_runner._parameter_summary(result)
    parameter_values = [
        float(parameters["gaussian_sigma_deg"]),
        float(parameters["lorentzian_hwhm_deg"]),
        float(parameters["lorentzian_probability"]),
    ]
    classification = (
        "MODEL_LIMITED_EFFECTIVE_RADIAL_MOSAIC_INTERVAL"
        if alias_parameter_sets
        else "MODEL_LIMITED_EFFECTIVE_RADIAL_MOSAIC_ESTIMATE"
    )
    summary = {
        "classification": classification,
        "parameters": parameter_values,
        "objective": float(result.objective),
        "rank": int(result.sensitivity_rank),
        "sensitivity_condition": float(result.sensitivity_condition),
        "global_alias_count": len(alias_parameter_sets),
        "profile_count": len(identities),
        "m0_profile_count": len(m0_identities),
        "per_incidence_profile_count": counts,
        "profile_identities": identities,
        "m0_profile_identities": m0_identities,
    }
    state = {
        "parameters": parameter_values,
        "profile_records": records,
        "profile_scales": result.profile_scales.tolist(),
        "profile_revision": profile_revision,
        "fixed_position": fixed_position.to_record(),
        "fixed_lattice": fixed_lattice.to_record(),
        "simulation_config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "component_profile_checkpoint_revision": cache_revision,
        "identifiability": {
            "classification": classification,
            "representative_is_fitted_solution": True,
            "competing_parameter_sets": alias_parameter_sets,
            "sensitivity_singular_values": result.sensitivity_singular_values.tolist(),
            "sensitivity_rank": int(result.sensitivity_rank),
            "sensitivity_condition": float(result.sensitivity_condition),
        },
    }
    return _stage_result(
        "mosaic",
        case=case,
        upstream=upstream,
        backend=backend,
        summary=summary,
        state=state,
    )


def _run_mosaic_stage(
    *,
    case: ReplayCase,
    upstream: dict[str, Any] | None,
    backend: str,
    output_directory: Path,
) -> dict[str, Any]:
    if upstream is None or upstream.get("stage") != "geometry":
        raise ValueError("mosaic stage requires the geometry result")
    if case.material_id == "Bi2Se3":
        return _bi2se3_mosaic(case, upstream, backend, output_directory)
    return _bi2te3_mosaic(case, upstream, backend, output_directory)


def _bi2se3_ordered_intensity(
    case: ReplayCase,
    upstream: dict[str, Any],
    backend: str,
    output_directory: Path,
) -> dict[str, Any]:
    if backend != "cuda":
        raise ValueError(
            "the accepted Bi2Se3 measured ordered-intensity fit is CUDA-qualified only"
        )
    from rasim_next.fitting import (
        ordered_intensity_structure_model_revision,
    )
    from rasim_next.pipeline.configured_simulation import configured_rod_catalog_revision

    stage = case.stage_config["ordered_intensity"]
    runner = _load_script_module(
        "staged_fit_bi2se3_ordered",
        "recover_bi2se3_ordered_intensity.py",
    )
    mosaic_artifact = Path(upstream["state"]["artifact"])
    if _sha256(mosaic_artifact) != upstream["state"]["artifact_sha256"]:
        raise RuntimeError("Bi2Se3 mosaic artifact changed before ordered fitting")
    mosaic_document = json.loads(mosaic_artifact.read_text(encoding="utf-8"))
    mosaic_projection = _bi2se3_mosaic_artifact_projection(mosaic_document)
    if mosaic_projection["state"] != {
        name: upstream["state"][name]
        for name in ("fixed_position", "parameters", "profile_identities", "profile_scales")
    }:
        raise RuntimeError(
            "Bi2Se3 mosaic artifact changed its scientific state before ordered fitting"
        )

    ordered_case_path = case.input_paths[str(stage["case_role"])]
    ordered_case = tomllib.loads(ordered_case_path.read_text(encoding="utf-8"))
    _strict_keys(
        ordered_case,
        {
            "background_inheritance",
            "mosaic_case",
            "observation_model",
            "profiles",
            "response_validation",
            "schema_version",
        },
        "Bi2Se3 measured ordered-intensity case",
    )
    if ordered_case["schema_version"] != "rasim-measured-ordered-intensity-fit-v1":
        raise ValueError("unsupported Bi2Se3 measured ordered-intensity case")
    profile_config = ordered_case["profiles"]
    response_validation = ordered_case["response_validation"]
    if not isinstance(profile_config, dict) or not isinstance(response_validation, dict):
        raise ValueError("measured ordered-intensity profiles and validation must be tables")
    _strict_keys(
        profile_config,
        {
            "expected_catalog_revision",
            "expected_m0_count",
            "expected_nonzero_count",
            "expected_total_count",
            "phi_bin_count",
            "phi_half_width_deg",
            "response_phi_gauss_order",
            "response_two_theta_gauss_order",
            "supported_m0_integer_L_10deg",
            "supported_m0_integer_L_15deg",
            "supported_m0_integer_L_5deg",
            "two_theta_half_width_deg",
        },
        "Bi2Se3 measured ordered-intensity profiles",
    )
    _strict_keys(
        response_validation,
        {"maximum_interpolation_relative_error"},
        "Bi2Se3 measured ordered-intensity response validation",
    )
    interpolation_limit = float(response_validation["maximum_interpolation_relative_error"])
    if not math.isfinite(interpolation_limit) or interpolation_limit <= 0.0:
        raise ValueError("ordered-intensity interpolation tolerance must be positive and finite")

    mosaic_case_path = case.input_paths["mosaic_case"]
    mosaic_case = tomllib.loads(mosaic_case_path.read_text(encoding="utf-8"))
    prepared = runner.prepare_measured_ordered_inputs(
        mosaic_document,
        mosaic_case_path=mosaic_case_path,
        mosaic_case=mosaic_case,
        profile_config=profile_config,
        source_sample_count=case.source_state_count,
    )
    if prepared.source_revision != _source_revision(case):
        raise RuntimeError("Bi2Se3 ordered fit changed its source realization")
    series = prepared.series
    if (
        series[0].samples.source_revision != prepared.source_revision
        or series[0].samples.source_seed != case.source_seed
        or _sha256(series[0].config.material.cif_path) != prepared.cif_sha256
    ):
        raise RuntimeError("Bi2Se3 ordered fit changed its source or structure provenance")
    measured_scales = _measured_profile_scales(mosaic_document)
    baseline = prepared.baseline_parameters
    expected_source_model = {
        "reduction": "one_incoherent_weighted_detector_function_per_incidence.v1",
        "sample_count": case.source_state_count,
        "source_revision": series[0].samples.source_revision,
        "source_sampling_model_id": series[0].samples.source_sampling_model_id,
        "source_rng_model_id": series[0].samples.source_rng_model_id,
        "source_seed": series[0].samples.source_seed,
        "spatial_sigma_m": list(series[0].config.source.spatial_sigma_m),
        "divergence_sigma_rad": list(series[0].config.source.divergence_sigma_rad),
        "wavelength_sigma_A": series[0].config.source.wavelength_sigma_A,
    }
    artifact = output_directory / "ordered_intensity_artifacts" / "bi2se3_ordered.json"
    expected_provenance = {
        "ordered_case_sha256": _sha256(ordered_case_path),
        "mosaic_case_sha256": _sha256(mosaic_case_path),
        "upstream_mosaic_result_sha256": upstream["state"]["artifact_sha256"],
        "measured_profile_policy_sha256": _sha256(case.input_paths["measured_profile_policy"]),
        "cif_sha256": prepared.cif_sha256,
        "rod_catalog_revision": configured_rod_catalog_revision(series[0]),
        "structure_model_revision": ordered_intensity_structure_model_revision(series[0].strength),
        "osc_sha256": {
            role: _sha256(case.input_paths[role]) for role in ("osc_5deg", "osc_10deg", "osc_15deg")
        },
    }
    expected_profile_scales = [
        {"identity": identity, "nuisance_peak_scale": measured_scales[identity]}
        for identity in sorted(measured_scales)
    ]

    if artifact.is_file():
        document = json.loads(artifact.read_text(encoding="utf-8"))
        _validated_bi2se3_ordered_cached_artifact(
            document,
            case=case,
            expected_provenance=expected_provenance,
            expected_fixed_mosaic=prepared.mosaic_parameters,
            expected_fixed_position=prepared.fixed_position_record,
            expected_source_model=expected_source_model,
            expected_profile_scales=expected_profile_scales,
            expected_bi_fractional_z=baseline.bi_fractional_z,
            expected_se2_fractional_z=baseline.se2_fractional_z,
        )
    else:
        document = _ORDERED_STAGE.fit_prepared_bi2se3_measured_ordered_document(
            prepared,
            mosaic_document,
            backend=backend,
            stage=stage,
            source_state_count=case.source_state_count,
            interpolation_limit=interpolation_limit,
            source_model=expected_source_model,
            observation_model=ordered_case["observation_model"],
            background_inheritance=ordered_case["background_inheritance"],
            provenance=expected_provenance,
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        _write_json(artifact, document)

    projection = _bi2se3_ordered_artifact_projection(document, case=case)
    if projection["state"]["fixed_position"] != upstream["state"]["fixed_position"]:
        raise RuntimeError("Bi2Se3 ordered artifact changed its upstream position state")
    source_model = projection["source_model"]
    if (
        source_model.get("sample_count") != case.source_state_count
        or source_model.get("source_seed") != case.source_seed
        or source_model.get("source_revision") != _source_revision(case)
        or source_model.get("reduction")
        != "one_incoherent_weighted_detector_function_per_incidence.v1"
    ):
        raise RuntimeError("Bi2Se3 ordered artifact changed its source realization")
    mosaic_summary = upstream["scientific_summary"]["mosaic"]
    if (
        projection["summary"]["profile_identities"] != mosaic_summary["profile_identities"]
        or projection["summary"]["m0_profile_identities"] != mosaic_summary["m0_profile_identities"]
    ):
        raise RuntimeError("Bi2Se3 ordered artifact changed its upstream profile selection")
    expected_mosaic = dict(
        zip(
            ("gaussian_sigma_deg", "lorentzian_hwhm_deg", "lorentzian_probability"),
            upstream["state"]["parameters"],
            strict=True,
        )
    )
    if projection["fixed_mosaic"] != expected_mosaic:
        raise RuntimeError("Bi2Se3 ordered artifact changed its upstream mosaic parameters")
    if projection["provenance"] != expected_provenance:
        raise RuntimeError("Bi2Se3 ordered artifact changed its upstream provenance")
    state = {
        "artifact": str(artifact),
        "artifact_sha256": _sha256(artifact),
        **projection["state"],
    }
    return _stage_result(
        "ordered_intensity",
        case=case,
        upstream=upstream,
        backend=backend,
        summary=projection["summary"],
        state=state,
    )


def _bi2te3_ordered_intensity(
    case: ReplayCase,
    upstream: dict[str, Any],
    backend: str,
    output_directory: Path,
) -> dict[str, Any]:
    del output_directory
    if backend != "cuda":
        raise ValueError("the accepted Bi2Te3 ordered-intensity replay is CUDA-qualified only")
    import numpy as np

    from rasim_next.fitting import (
        OrderedIntensityPeakCenterObservations,
        compile_source_averaged_ordered_intensity_response,
        fit_ordered_intensity_series,
    )
    from rasim_next.io.osc import read_osc
    from rasim_next.ordered import Bi2X3QuintupleLayerParameters

    mosaic_runner = _load_script_module(
        "staged_fit_bi2te3_ordered_physics", "recover_bi2se3_mosaic.py"
    )
    mosaic_state = upstream["state"]
    fixed_position, fixed_lattice, config_path = _bi2te3_fixed_state_from_stage(
        mosaic_state,
        case=case,
    )
    base, series, nominal_series = _bi2te3_fixed_inputs(
        case,
        fixed_position,
        fixed_lattice,
        simulation_config_path=config_path,
    )
    if base.samples.source_revision != _source_revision(case):
        raise RuntimeError("Bi2Te3 ordered-intensity replay changed its source realization")
    mosaic_stage = case.stage_config["mosaic"]
    dark_counts = read_osc(case.input_paths[str(mosaic_stage["dark_role"])]).detector_native_counts
    frames, all_definitions, _ = _bi2te3_profile_definitions(
        case,
        nominal_series,
        series,
        dark_counts,
        mosaic_runner,
    )

    def definition_key(value: Any) -> tuple[str, int, int, int | None]:
        if isinstance(value, dict):
            return (
                str(value["dataset_id"]),
                int(value["family_m"]),
                int(value["integer_L"]),
                value.get("root_side_branch_id"),
            )
        identity = value.identity
        return (
            identity.dataset_id,
            identity.group_key.layered_family_m,
            identity.group_key.layered_integer_L,
            identity.branch_id,
        )

    definition_by_key = {
        definition_key(definition): definition
        for dataset in all_definitions
        for definition in dataset
    }
    profile_records = list(mosaic_state["profile_records"])
    profile_scales = np.asarray(mosaic_state["profile_scales"], dtype=np.float64)
    if profile_scales.shape != (len(profile_records),):
        raise RuntimeError("mosaic profile scales do not align with their identities")
    fitted_definitions = tuple(
        definition_by_key[definition_key(record)]
        for record, scale in zip(profile_records, profile_scales, strict=True)
        if scale > 0.0
    )
    if not fitted_definitions:
        raise RuntimeError("no positive mosaic profile amplitudes remain")
    definitions_by_dataset = []
    scales_by_dataset = []
    for dataset_id in mosaic_stage["dataset_ids"]:
        mask = np.asarray(
            [record["dataset_id"] == dataset_id for record in profile_records],
            dtype=np.bool_,
        ) & (profile_scales > 0.0)
        definitions_by_dataset.append(
            tuple(
                definition_by_key[definition_key(record)]
                for record, keep in zip(profile_records, mask, strict=True)
                if keep
            )
        )
        scales_by_dataset.append(profile_scales[mask])
    if any(not values for values in definitions_by_dataset):
        raise RuntimeError("ordered-intensity profiles do not cover every incidence")

    physics, profile_geometry = mosaic_runner._profile_forward_contexts(base, series)
    mosaic_parameters = mosaic_runner._mosaic_parameters(
        gaussian_sigma_rad=math.radians(float(mosaic_state["parameters"][0])),
        lorentzian_half_width_rad=math.radians(float(mosaic_state["parameters"][1])),
        lorentzian_probability=float(mosaic_state["parameters"][2]),
        context=physics,
    )
    detectors = mosaic_runner._detector_series(physics, profile_geometry, mosaic_parameters)
    baseline = Bi2X3QuintupleLayerParameters.from_crystal(base.crystal)
    if tuple(base.strength.site_labels) != ("Bi", "Te1", "Te2"):
        raise RuntimeError(f"unexpected Bi2Te3 site roles: {base.strength.site_labels}")
    responses = []
    observations = []
    for detector, frame, definitions, scales in zip(
        detectors,
        frames,
        definitions_by_dataset,
        scales_by_dataset,
        strict=True,
    ):
        response = compile_source_averaged_ordered_intensity_response(
            detector,
            angle_frame=frame,
            definitions=definitions,
            execution_backend=backend,
        )
        baseline_signal = response.predict_signal_density_A2_per_rad2(baseline)
        transferred_signal = scales * baseline_signal
        if np.any(transferred_signal <= 0.0) or not np.all(np.isfinite(transferred_signal)):
            raise FloatingPointError("transferred real-OSC peak signal is not positive and finite")
        responses.append(response)
        observations.append(
            OrderedIntensityPeakCenterObservations(
                dataset_id=response.dataset_id,
                observable_revision=response.observable_revision,
                signal_density_A2_per_rad2=transferred_signal,
            )
        )
    stage = case.stage_config["ordered_intensity"]
    canonical_active = (
        "se1_occupancy",
        "se2_occupancy",
        "u_radial_A2",
        "u_normal_A2",
    )
    lower = tuple(float(value) for value in stage["lower_bounds"])
    upper = tuple(float(value) for value in stage["upper_bounds"])
    bounds = {
        name: (minimum, maximum)
        for name, minimum, maximum in zip(canonical_active, lower, upper, strict=True)
    }
    fits = []
    failures = []
    for start_index, initial in enumerate(stage["multistarts"]):
        try:
            fit = fit_ordered_intensity_series(
                tuple(responses),
                tuple(observations),
                base_strength=base.strength,
                active_parameter_names=canonical_active,
                initial_parameters=replace(
                    baseline,
                    bi_occupancy=1.0,
                    se1_occupancy=float(initial[0]),
                    se2_occupancy=float(initial[1]),
                    u_radial_A2=float(initial[2]),
                    u_normal_A2=float(initial[3]),
                ),
                relative_scale_mode=True,
                active_parameter_bounds=bounds,
                maximum_function_evaluations=int(stage["maximum_function_evaluations"]),
                required_source_state_count=case.source_state_count,
                required_source_revision=base.samples.source_revision,
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            failures.append({"start_index": start_index, "message": str(error)})
        else:
            fits.append(fit)
    if not fits:
        raise RuntimeError(f"all Bi2Te3 structure multistarts failed: {failures}")
    result = min(fits, key=lambda fit: fit.objective)
    if result.occupancy_ratio_reference != "bi_occupancy":
        raise RuntimeError("relative fit changed the frozen Bi occupancy gauge")
    if result.occupancy_ratios is None:
        raise RuntimeError("relative fit did not report occupancy ratios")
    oracle_error = _ORDERED_STAGE.cached_vs_fresh_ordered_response_max_relative_error(
        tuple(detectors),
        tuple(zip(frames, definitions_by_dataset, strict=True)),
        tuple(result.predicted_signal_density_A2_per_rad2),
        result.structure_representative,
        backend=backend,
    )
    records = [
        {
            "dataset_id": definition.identity.dataset_id,
            "family_m": definition.identity.group_key.layered_family_m,
            "integer_L": definition.identity.group_key.layered_integer_L,
            "root_side_branch_id": definition.identity.branch_id,
            "analytic_branch_id": definition.identity.analytic_branch_id,
        }
        for definitions in definitions_by_dataset
        for definition in definitions
    ]
    identities, m0_identities, _ = _profile_summary(records)
    representative = result.structure_representative
    parameter_values = [
        float(result.occupancy_ratios[1]),
        float(result.occupancy_ratios[2]),
        float(representative.u_radial_A2),
        float(representative.u_normal_A2),
    ]
    summary = {
        "classification": "MODEL_LIMITED_REAL_OSC_STRUCTURE_ESTIMATE_NO_ORACLE",
        "claim_boundary": str(stage["claim_boundary"]),
        "parameters": parameter_values,
        "objective": float(result.objective),
        "rank": int(result.sensitivity_rank),
        "profile_count": len(identities),
        "m0_profile_count": len(m0_identities),
        "active_bounds": result.active_bounds.tolist(),
        "profile_identities": identities,
        "m0_profile_identities": m0_identities,
    }
    structure_record = {
        "bi_fractional_z": representative.bi_fractional_z,
        "te2_fractional_z": representative.se2_fractional_z,
        "bi_occupancy": representative.bi_occupancy,
        "te1_occupancy": representative.se1_occupancy,
        "te2_occupancy": representative.se2_occupancy,
        "u_radial_A2": representative.u_radial_A2,
        "u_normal_A2": representative.u_normal_A2,
    }
    state = {
        "parameters": parameter_values,
        "structure_representative": structure_record,
        "fixed_position": mosaic_state["fixed_position"],
        "fixed_lattice": mosaic_state["fixed_lattice"],
        "simulation_config": mosaic_state["simulation_config"],
        "mosaic_parameters": list(mosaic_state["parameters"]),
        "profile_records": records,
        "cached_vs_fresh_maximum_relative_error": oracle_error,
    }
    return _stage_result(
        "ordered_intensity",
        case=case,
        upstream=upstream,
        backend=backend,
        summary=summary,
        state=state,
    )


def _run_ordered_intensity_stage(
    *,
    case: ReplayCase,
    upstream: dict[str, Any] | None,
    backend: str,
    output_directory: Path,
) -> dict[str, Any]:
    if upstream is None or upstream.get("stage") != "mosaic":
        raise ValueError("ordered-intensity stage requires the mosaic result")
    if case.material_id == "Bi2Se3":
        return _bi2se3_ordered_intensity(case, upstream, backend, output_directory)
    return _bi2te3_ordered_intensity(case, upstream, backend, output_directory)


def _bi2te3_render(
    case: ReplayCase,
    upstream: dict[str, Any],
    backend: str,
    output_directory: Path,
) -> dict[str, Any]:
    if backend != "cuda":
        raise ValueError("the accepted Bi2Te3 render replay is CUDA-qualified only")
    import numpy as np
    from PIL import Image

    from rasim_next.io.osc import read_osc
    from rasim_next.ordered import Bi2X3QuintupleLayerParameters
    from rasim_next.pipeline.configured_simulation import sample_detector_pixel_center_density

    render = case.stage_config["render"]
    if not bool(render["enabled"]):
        raise ValueError("the replay case does not enable a current render")
    mosaic_runner = _load_script_module(
        "staged_fit_bi2te3_render_physics", "recover_bi2se3_mosaic.py"
    )
    state = upstream["state"]
    fixed_position, fixed_lattice, config_path = _bi2te3_fixed_state_from_stage(
        state,
        case=case,
    )
    base, series, _ = _bi2te3_fixed_inputs(
        case,
        fixed_position,
        fixed_lattice,
        simulation_config_path=config_path,
    )
    physics, profile_geometry = mosaic_runner._profile_forward_contexts(base, series)
    mosaic_parameters = mosaic_runner._mosaic_parameters(
        gaussian_sigma_rad=math.radians(float(state["mosaic_parameters"][0])),
        lorentzian_half_width_rad=math.radians(float(state["mosaic_parameters"][1])),
        lorentzian_probability=float(state["mosaic_parameters"][2]),
        context=physics,
    )
    baseline = Bi2X3QuintupleLayerParameters.from_crystal(base.crystal)
    structure = state["structure_representative"]
    fitted = replace(
        baseline,
        bi_fractional_z=float(structure["bi_fractional_z"]),
        se2_fractional_z=float(structure["te2_fractional_z"]),
        bi_occupancy=float(structure["bi_occupancy"]),
        se1_occupancy=float(structure["te1_occupancy"]),
        se2_occupancy=float(structure["te2_occupancy"]),
        u_radial_A2=float(structure["u_radial_A2"]),
        u_normal_A2=float(structure["u_normal_A2"]),
    )
    if (
        fitted.bi_fractional_z != baseline.bi_fractional_z
        or fitted.se2_fractional_z != baseline.se2_fractional_z
    ):
        raise RuntimeError("Bi2Te3 render moved a frozen atomic coordinate")
    strength = replace(base.strength, structure_parameters=fitted)
    detectors = tuple(
        detector.with_maximum_state_block_count(
            int(render["cuda_state_block_count"])
        ).rebind_physics(strength_model=strength)
        for detector in mosaic_runner._detector_series(
            physics,
            profile_geometry,
            mosaic_parameters,
        )
    )
    simulated = [
        sample_detector_pixel_center_density(
            detector,
            execution_backend=backend,
            cuda_coordinate_chunk_size=int(render["cuda_coordinate_chunk"]),
        ).image_A2_per_px2
        for detector in detectors
    ]
    mosaic_stage = case.stage_config["mosaic"]
    raw = [
        read_osc(case.input_paths[str(role)]).detector_native_counts
        for role in mosaic_stage["osc_roles"]
    ]
    raw_high = max(float(np.percentile(image, 99.995)) for image in raw)
    raw_high = max(raw_high, 1.0)
    raw_display = [np.log1p(np.maximum(image, 0.0)) / math.log1p(raw_high) for image in raw]
    positive = [image[image > 0.0] for image in simulated]
    if any(values.size == 0 for values in positive):
        raise FloatingPointError("a simulated replay image contains no positive density")
    sim_low = min(float(np.percentile(values, 1.0)) for values in positive)
    sim_high = max(float(np.percentile(values, 99.995)) for values in positive)
    sim_low = max(sim_low, sim_high * 1.0e-10, np.finfo(np.float64).tiny)
    logarithmic_range = math.log(sim_high / sim_low)
    simulated_display = [
        np.log(np.maximum(image, sim_low) / sim_low) / logarithmic_range for image in simulated
    ]
    artifact_directory = output_directory / "render_artifacts"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    raw_hashes = []
    simulated_hashes = []
    paths = []
    artifact_identities = []
    for angle, raw_values, simulated_values in zip(
        case.incidence_angles_deg,
        raw_display,
        simulated_display,
        strict=True,
    ):
        raw_pixels = np.asarray(
            np.rint(255.0 * np.clip(raw_values, 0.0, 1.0)),
            dtype=np.uint8,
        )
        simulated_pixels = np.asarray(
            np.rint(255.0 * np.clip(simulated_values, 0.0, 1.0)),
            dtype=np.uint8,
        )
        raw_path = artifact_directory / f"raw_osc_{angle:g}deg_3000x3000.png"
        simulated_path = artifact_directory / f"simulated_250ki_{angle:g}deg_3000x3000.png"
        Image.fromarray(raw_pixels, mode="L").save(raw_path, optimize=True)
        Image.fromarray(simulated_pixels, mode="L").save(simulated_path, optimize=True)
        raw_hashes.append(hashlib.sha256(raw_pixels.tobytes(order="C")).hexdigest())
        simulated_hashes.append(hashlib.sha256(simulated_pixels.tobytes(order="C")).hexdigest())
        paths.extend((str(raw_path), str(simulated_path)))
        artifact_identities.extend(
            (
                {
                    "decoded_mode": "L",
                    "decoded_size": [int(raw_pixels.shape[1]), int(raw_pixels.shape[0])],
                    "decoded_pixel_sha256": raw_hashes[-1],
                },
                {
                    "decoded_mode": "L",
                    "decoded_size": [
                        int(simulated_pixels.shape[1]),
                        int(simulated_pixels.shape[0]),
                    ],
                    "decoded_pixel_sha256": simulated_hashes[-1],
                },
            )
        )
    summary = {
        "classification": "MODEL_LIMITED_FORWARD_IMAGES_NO_COUNT_CALIBRATION",
        "source_state_count": case.source_state_count,
        "rod_count": len(detectors[0].rods),
        "m0_rod_count": sum(rod.family_m == 0 for rod in detectors[0].rods),
        "raw_decoded_pixel_sha256": raw_hashes,
        "simulated_decoded_pixel_sha256": simulated_hashes,
    }
    return _stage_result(
        "render",
        case=case,
        upstream=upstream,
        backend=backend,
        summary=summary,
        state={"artifact": paths, "artifact_identity": artifact_identities},
    )


def _run_render_stage(
    *,
    case: ReplayCase,
    upstream: dict[str, Any] | None,
    backend: str,
    output_directory: Path,
) -> dict[str, Any]:
    if upstream is None or upstream.get("stage") != "ordered_intensity":
        raise ValueError("render stage requires the ordered-intensity result")
    if case.material_id != "Bi2Te3":
        raise ValueError(str(case.stage_config["render"].get("reason", "render is disabled")))
    return _bi2te3_render(case, upstream, backend, output_directory)


def _validate_stage_envelope(
    case: ReplayCase,
    *,
    stage: str,
    result: dict[str, Any],
    backend: str,
    runtime_identity: dict[str, Any],
) -> None:
    case_source_revision = _source_revision(case)
    source_state_count, source_seed, source_revision = _stage_source_identity(
        case,
        stage,
        case_source_revision=case_source_revision,
    )
    expected_envelope = {
        "schema_version": _STAGE_SCHEMA_VERSION,
        "stage": stage,
        "case_id": case.case_id,
        "material_id": case.material_id,
        "stage_case_sha256": _stage_case_sha256(case, stage),
        "execution_backend": backend,
        "runtime": runtime_identity,
        "source_state_count": source_state_count,
        "source_seed": source_seed,
        "source_revision": source_revision,
    }
    for name, expected in expected_envelope.items():
        if result.get(name) != expected:
            raise ValueError(f"{stage} stage result changed {name}")
    state = result.get("state")
    if not isinstance(state, dict):
        raise ValueError(f"{stage} stage result lacks scientific state")
    if case.material_id == "Bi2Se3" and stage == "geometry":
        _strict_keys(
            state,
            {"corrections", "fit_evidence", "incidence_angle_delta_rad", "incidence_angles"},
            "Bi2Se3 geometry state",
        )
        summary = result.get("scientific_summary")
        geometry = summary.get("geometry") if isinstance(summary, dict) else None
        if not isinstance(geometry, dict):
            raise ValueError("Bi2Se3 geometry stage lacks its scientific summary")
        corrections = state["corrections"]
        delta_rad = float(state["incidence_angle_delta_rad"])
        angles = state["incidence_angles"]
        fit_evidence = state["fit_evidence"]
        qualification = (
            fit_evidence.get("qualification") if isinstance(fit_evidence, dict) else None
        )
        image_ids = tuple(geometry.get("incidence_angle_image_ids", ()))
        if (
            corrections != geometry.get("corrections")
            or not isinstance(corrections, list)
            or len(corrections) != len(_SHARED_GEOMETRY_PARAMETER_NAMES)
            or any(not math.isfinite(float(value)) for value in corrections)
            or float(corrections[2]) != 0.0
            or not math.isfinite(delta_rad)
            or delta_rad != geometry.get("incidence_angle_delta_rad")
            or not isinstance(angles, list)
            or len(angles) != len(case.incidence_angles_deg)
            or len(image_ids) != len(case.incidence_angles_deg)
            or not isinstance(fit_evidence, dict)
            or not isinstance(qualification, dict)
            or not bool(qualification.get("accepted"))
        ):
            raise ValueError("Bi2Se3 geometry state is incomplete or inconsistent")
        for index, (record, commanded_deg, image_id) in enumerate(
            zip(angles, case.incidence_angles_deg, image_ids, strict=True)
        ):
            if not isinstance(record, dict):
                raise ValueError(f"Bi2Se3 geometry incidence record {index} is invalid")
            _strict_keys(
                record,
                {"commanded_angle_rad", "effective_angle_rad", "image_id"},
                f"Bi2Se3 geometry incidence record {index}",
            )
            commanded_rad = math.radians(commanded_deg)
            if (
                record["image_id"] != image_id
                or float(record["commanded_angle_rad"]) != commanded_rad
                or float(record["effective_angle_rad"]) != commanded_rad + delta_rad
            ):
                raise ValueError("Bi2Se3 geometry state does not use one common angle delta")
    if case.material_id == "Bi2Se3" and stage == "mosaic":
        _strict_keys(
            state,
            {
                "artifact",
                "artifact_sha256",
                "fixed_position",
                "parameters",
                "profile_identities",
                "profile_scales",
            },
            "Bi2Se3 mosaic state",
        )
    if case.material_id == "Bi2Se3" and stage == "ordered_intensity":
        _strict_keys(
            state,
            {
                "artifact",
                "artifact_sha256",
                "fixed_position",
                "parameters",
                "profile_scales",
                "structure_representative",
            },
            "Bi2Se3 ordered-intensity state",
        )
    if case.material_id == "Bi2Te3" and stage == "geometry":
        _strict_keys(state, {"corrections"}, "Bi2Te3 geometry state")
        summary = result.get("scientific_summary")
        geometry = summary.get("geometry") if isinstance(summary, dict) else None
        if not isinstance(geometry, dict) or state["corrections"] != geometry.get("corrections"):
            raise ValueError("Bi2Te3 geometry state changed its fitted corrections")
    if case.material_id == "Bi2Te3" and stage == "mosaic":
        _strict_keys(
            state,
            {
                "component_profile_checkpoint_revision",
                "fixed_lattice",
                "fixed_position",
                "identifiability",
                "parameters",
                "profile_records",
                "profile_revision",
                "profile_scales",
                "simulation_config",
            },
            "Bi2Te3 mosaic state",
        )
    if case.material_id == "Bi2Te3" and stage == "ordered_intensity":
        _strict_keys(
            state,
            {
                "cached_vs_fresh_maximum_relative_error",
                "fixed_lattice",
                "fixed_position",
                "mosaic_parameters",
                "parameters",
                "profile_records",
                "simulation_config",
                "structure_representative",
            },
            "Bi2Te3 ordered-intensity state",
        )
    artifact = state.get("artifact")
    artifact_sha256 = state.get("artifact_sha256")
    artifact_contract = {
        ("Bi2Se3", "mosaic"): "json",
        ("Bi2Se3", "ordered_intensity"): "json",
        ("Bi2Te3", "render"): "images",
    }.get((case.material_id, stage))
    if artifact_contract is None:
        if any(name in state for name in ("artifact", "artifact_sha256", "artifact_identity")):
            raise ValueError(f"{stage} stage result has an unexpected external result")
        return
    if artifact_contract == "json":
        if not isinstance(artifact, str):
            raise ValueError(f"{stage} stage result lacks its external JSON result")
        if not isinstance(artifact_sha256, str):
            raise ValueError(f"{stage} stage result lacks its external result hash")
        artifact_path = Path(artifact)
        if not artifact_path.is_file() or _sha256(artifact_path) != artifact_sha256:
            raise ValueError(f"{stage} stage result changed its external result")
        if case.material_id == "Bi2Se3" and stage in {"mosaic", "ordered_intensity"}:
            artifact_document = json.loads(artifact_path.read_text(encoding="utf-8"))
            if stage == "mosaic":
                projection = _bi2se3_mosaic_artifact_projection(artifact_document)
                projected_state = {
                    name: state[name]
                    for name in (
                        "fixed_position",
                        "parameters",
                        "profile_identities",
                        "profile_scales",
                    )
                }
            else:
                projection = _bi2se3_ordered_artifact_projection(
                    artifact_document,
                    case=case,
                )
                projected_state = {
                    name: state[name]
                    for name in (
                        "fixed_position",
                        "parameters",
                        "profile_scales",
                        "structure_representative",
                    )
                }
            if projection["state"] != projected_state:
                raise ValueError(f"{stage} stage external artifact changed its scientific state")
            summary = result.get("scientific_summary")
            stage_summary = summary.get(stage) if isinstance(summary, dict) else None
            if projection["summary"] != stage_summary:
                raise ValueError(f"{stage} stage external artifact changed its scientific summary")
            source_model = projection["source_model"]
            if (
                source_model.get("sample_count") != source_state_count
                or source_model.get("source_seed") != source_seed
                or source_model.get("source_revision") != source_revision
                or source_model.get("reduction")
                != "one_incoherent_weighted_detector_function_per_incidence.v1"
            ):
                raise ValueError(f"{stage} stage external artifact changed its source identity")
            provenance = projection["provenance"]
            stage_config = case.stage_config[stage]
            if stage == "mosaic":
                provenance_matches = provenance.get("case_sha256") == _sha256(
                    case.input_paths[str(stage_config["case_role"])]
                )
            else:
                provenance_matches = provenance.get("ordered_case_sha256") == _sha256(
                    case.input_paths[str(stage_config["case_role"])]
                ) and provenance.get("mosaic_case_sha256") == _sha256(
                    case.input_paths["mosaic_case"]
                )
            if not provenance_matches:
                raise ValueError(f"{stage} stage external artifact changed its case provenance")
        return
    if artifact_contract == "images":
        identities = state.get("artifact_identity")
        if (
            not isinstance(artifact, list)
            or not artifact
            or not isinstance(identities, list)
            or len(artifact) != len(identities)
        ):
            raise ValueError(f"{stage} stage result identities are incomplete")
        from PIL import Image

        for index, (path_text, identity) in enumerate(zip(artifact, identities, strict=True)):
            if not isinstance(path_text, str) or not isinstance(identity, dict):
                raise ValueError(f"{stage} stage result identity {index} is invalid")
            _strict_keys(
                identity,
                {"decoded_mode", "decoded_size", "decoded_pixel_sha256"},
                f"{stage} stage result identity {index}",
            )
            artifact_path = Path(path_text)
            if not artifact_path.is_file():
                raise ValueError(f"{stage} stage result artifact {index} is missing")
            with Image.open(artifact_path) as image:
                image.load()
                decoded_mode = image.mode
                decoded_size = list(image.size)
                decoded_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
            if (
                decoded_mode != identity["decoded_mode"]
                or decoded_size != identity["decoded_size"]
                or decoded_sha256 != identity["decoded_pixel_sha256"]
            ):
                raise ValueError(
                    f"{stage} stage result artifact {index} changed its decoded pixels"
                )
        return
    raise AssertionError(f"unsupported resume artifact contract {artifact_contract!r}")


def _validate_stage_result(
    case: ReplayCase,
    *,
    stage: str,
    result: dict[str, Any],
    upstream: dict[str, Any] | None,
    backend: str,
    runtime_identity: dict[str, Any],
) -> None:
    _strict_keys(result, _STAGE_RESULT_KEYS, f"{stage} stage result")
    revision = result.get("scientific_revision")
    if not isinstance(revision, str) or not revision.startswith(_SHA256_PREFIX):
        raise ValueError(f"{stage} stage lacks a scientific revision")
    revision_payload = {
        name: value for name, value in result.items() if name != "scientific_revision"
    }
    if revision != scientific_revision(stage, revision_payload):
        raise ValueError(f"{stage} stage result changed its scientific revision")
    _validate_stage_envelope(
        case,
        stage=stage,
        result=result,
        backend=backend,
        runtime_identity=runtime_identity,
    )
    expected_upstream = None if upstream is None else upstream["scientific_revision"]
    if result.get("upstream_scientific_revision") != expected_upstream:
        raise ValueError(f"{stage} stage changed its upstream scientific revision")
    if case.material_id == "Bi2Se3" and stage == "mosaic":
        if upstream is None or upstream.get("stage") != "geometry":
            raise ValueError("Bi2Se3 mosaic stage requires its geometry artifact")
        fixed_position = result["state"].get("fixed_position")
        expected_position_projection = _bi2se3_position_projection(case, upstream)
        if fixed_position != expected_position_projection:
            raise ValueError("Bi2Se3 mosaic stage changed its geometry position handoff")
    if case.material_id == "Bi2Se3" and stage == "ordered_intensity":
        if upstream is None or upstream.get("stage") != "mosaic":
            raise ValueError("Bi2Se3 ordered-intensity stage requires its mosaic artifact")
        mosaic_position = upstream["state"].get("fixed_position")
        ordered_position = result["state"].get("fixed_position")
        if not isinstance(mosaic_position, dict) or ordered_position != mosaic_position:
            raise ValueError("Bi2Se3 ordered-intensity stage changed its position handoff")
        artifact_document = json.loads(
            Path(result["state"]["artifact"]).read_text(encoding="utf-8")
        )
        projection = _bi2se3_ordered_artifact_projection(
            artifact_document,
            case=case,
        )
        expected_mosaic = dict(
            zip(
                ("gaussian_sigma_deg", "lorentzian_hwhm_deg", "lorentzian_probability"),
                upstream["state"]["parameters"],
                strict=True,
            )
        )
        upstream_summary = upstream["scientific_summary"]["mosaic"]
        if projection["fixed_mosaic"] != expected_mosaic:
            raise ValueError("Bi2Se3 ordered-intensity artifact changed its mosaic handoff")
        if (
            projection["summary"]["profile_identities"] != upstream_summary["profile_identities"]
            or projection["summary"]["m0_profile_identities"]
            != upstream_summary["m0_profile_identities"]
        ):
            raise ValueError("Bi2Se3 ordered-intensity artifact changed its profile handoff")
        if projection["state"]["profile_scales"] != upstream["state"].get("profile_scales"):
            raise ValueError(
                "Bi2Se3 ordered-intensity artifact changed its profile amplitude handoff"
            )
        if (
            projection["provenance"].get("upstream_mosaic_result_sha256")
            != upstream["state"]["artifact_sha256"]
        ):
            raise ValueError("Bi2Se3 ordered-intensity artifact changed its mosaic provenance")
    if case.material_id == "Bi2Te3" and stage in {"mosaic", "ordered_intensity"}:
        _validate_bi2te3_fixed_state_handoff(
            case,
            stage=stage,
            state=result["state"],
            upstream=upstream,
        )


def _combined_scientific_summary(
    case: ReplayCase,
    stages: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summaries = []
    for stage, result in stages.items():
        summary = result.get("scientific_summary")
        if not isinstance(summary, dict) or stage not in summary:
            raise ValueError(f"{stage} stage lacks its scientific summary")
        summaries.append(summary)
    if not summaries:
        raise ValueError("a replay certificate requires at least one stage")
    combined = {
        "case_id": case.case_id,
        "material_id": case.material_id,
        "source_state_count": case.source_state_count,
        "source_revision": summaries[-1].get("source_revision"),
    }
    for stage, summary in zip(stages, summaries, strict=True):
        combined[stage] = summary[stage]
    return combined


def run_replay(
    case: ReplayCase,
    *,
    output_directory: Path,
    backend: str,
    through: str = "ordered_intensity",
    verify: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Run ordered stages and bind each one to the previous scientific revision."""

    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be cpu or cuda")
    if through not in _STAGES:
        raise ValueError(f"through must be one of {', '.join(_STAGES)}")
    reloaded_case = load_replay_case(case.path, repository_root=case.repository_root)
    if reloaded_case != case:
        raise ValueError("replay case or loaded input mappings changed after loading")
    case = reloaded_case
    terminal = _STAGES.index(through)
    if case.material_id == "Bi2Te3" and backend == "cpu" and terminal >= _STAGES.index("mosaic"):
        raise ValueError(
            "the accepted Bi2Te3 mosaic, ordered-intensity, and render replay is CUDA-qualified only"
        )
    if through == "render" and not bool(case.stage_config["render"].get("enabled")):
        raise ValueError(str(case.stage_config["render"].get("reason", "render is disabled")))
    environment_record = next(
        record for record in case.file_records if record["role"] == "environment_lock"
    )
    runtime_identity = _validated_runtime_identity(
        case.input_paths["environment_lock"],
        expected_sha256=str(environment_record["sha256"]),
        include_render=through == "render",
    )
    base_runtime_identity = {
        **runtime_identity,
        "packages": {
            name: version
            for name, version in runtime_identity["packages"].items()
            if name not in _RENDER_RUNTIME_PACKAGES
        },
    }
    output = output_directory.resolve()
    if output == case.repository_root or output.is_relative_to(case.repository_root):
        raise ValueError("replay output directory must be outside the repository")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("replay output directory must be empty unless --resume is used")
    output.mkdir(parents=True, exist_ok=True)
    runners = (
        _run_geometry_stage,
        _run_mosaic_stage,
        _run_ordered_intensity_stage,
        _run_render_stage,
    )
    stages: dict[str, dict[str, Any]] = {}
    upstream: dict[str, Any] | None = None
    for stage_name, runner in zip(_STAGES[: terminal + 1], runners[: terminal + 1], strict=True):
        stage_runtime_identity = (
            runtime_identity if stage_name == "render" else base_runtime_identity
        )
        stage_case = replace(case, runtime_identity=stage_runtime_identity)
        stage_path = output / f"{stage_name}.json"
        resumed = resume and stage_path.is_file()
        if resumed:
            stage_result = json.loads(stage_path.read_text(encoding="utf-8"))
        else:
            stage_result = runner(
                case=stage_case,
                upstream=upstream,
                backend=backend,
                output_directory=output,
            )
            if not isinstance(stage_result, dict):
                raise TypeError(f"{stage_name} stage did not return a mapping")
        _validate_stage_result(
            case,
            stage=stage_name,
            result=stage_result,
            upstream=upstream,
            backend=backend,
            runtime_identity=stage_runtime_identity,
        )
        candidate_stages = {**stages, stage_name: stage_result}
        if verify:
            partial_summary = _combined_scientific_summary(case, candidate_stages)
            verify_scientific_summary(case, partial_summary, through=stage_name)
        if not resumed:
            _write_json(stage_path, stage_result)
        stages[stage_name] = stage_result
        upstream = stage_result
    certificate = {
        "schema_version": _CERTIFICATE_SCHEMA_VERSION,
        "case_id": case.case_id,
        "material_id": case.material_id,
        "case_sha256": _sha256(case.path),
        "backend": backend,
        "verification_runtime": runtime_identity,
        "through": through,
        "stage_scientific_revisions": {
            name: result["scientific_revision"] for name, result in stages.items()
        },
        "stages": stages,
    }
    if verify:
        combined = _combined_scientific_summary(case, stages)
        verify_scientific_summary(case, combined, through=through)
        certificate["scientific_summary"] = combined
        certificate["verified"] = True
    else:
        certificate["verified"] = False
    _write_json(output / "replay_certificate.json", certificate)
    return certificate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--through", choices=_STAGES, default="ordered_intensity")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    arguments = parser.parse_args(argv)
    case = load_replay_case(arguments.case)
    if arguments.inputs_only:
        print(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "material_id": case.material_id,
                    "source_state_count": case.source_state_count,
                    "runtime": case.runtime_identity,
                    "input_sha256": {
                        role: _sha256(path) for role, path in sorted(case.input_paths.items())
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.output_directory is None:
        parser.error("--output-directory is required unless --inputs-only is used")
    result = run_replay(
        case,
        output_directory=arguments.output_directory,
        backend=arguments.backend,
        through=arguments.through,
        verify=not arguments.no_verify,
        resume=arguments.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
