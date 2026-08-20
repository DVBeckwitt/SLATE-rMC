"""Fit Bi2Se3 mosaic profiles jointly from synthetic or measured 5/10/15-degree views."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from painted_ewald import MosaicParameters, Rod, wrapped_mosaic_line_density_rad_inv
from rasim_next.core.contracts import MaterialOptics
from rasim_next.core.staged_fit import verify_staged_fit_stage_artifact
from rasim_next.fitting import (
    SHARED_GEOMETRY_PARAMETER_NAMES,
    SOURCE_AVERAGED_PROFILE_SUPPORT_GATE_REVISION,
    ExactTagGeometryModel,
    MosaicIdentifiabilityError,
    MosaicProfileDefinition,
    MosaicProfileFitResult,
    MosaicProfileIdentity,
    MosaicProfileNuisanceBasis,
    MosaicProfileSearchResult,
    MosaicProfileSet,
    MosaicReflectionGroupKey,
    SharedGeometryCorrections,
    apply_shared_geometry_corrections,
    evaluate_continuous_mosaic_profiles,
    fit_mosaic_component_profiles,
    fit_refined_mosaic_component_profiles,
    source_averaged_profile_has_support,
)
from rasim_next.geometry import (
    AngleFrame,
    angles_to_detector_coordinates,
    build_incident_states,
    detector_coordinates_to_angles,
)
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.io.osc import read_osc
from rasim_next.measurement import (
    compile_detector_profile_projector,
    project_detector_profiles,
)
from rasim_next.pipeline.bragg_space import Bi2X3FiniteStackStrength
from rasim_next.pipeline.configured_simulation import (
    ConfiguredGeometryInputs,
    ConfiguredSimulationInputs,
    build_configured_geometry_inputs,
    build_configured_simulation_inputs,
    build_nominal_ewald_context,
    configured_rod_catalog_revision,
    evaluate_nominal_integer_l_markers,
    integrate_detector_macrobins,
    load_simulation_config,
    rebind_configured_geometry_instrument,
)
from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure
from rasim_next.selection import (
    DETECTOR_VALID_MASK_REVISION,
    ExpectedM0Peak,
    M0PeakEvidence,
    M0PeakEvidencePolicy,
    build_osc_angle_frame,
    detector_valid_mask_from_counts,
    evaluate_expected_m0_peak_evidence,
)

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = ROOT / "examples" / "bi2se3" / "experiment" / "mosaic_fit_truth.toml"
DEFAULT_MEASURED_PROFILE_POLICY = (
    ROOT / "examples" / "bi2se3" / "experiment" / "mosaic_fit_measured_policy.toml"
)
_ACCEPTED_INDEXED_MANIFEST_SHA256 = (
    "1de21e03a801fa38390ef5280133666474bfd969377024ef6dd4fb34e40f3132"
)
_ACCEPTED_FIXED_GEOMETRY_CORRECTIONS = (
    -0.00449943391543809,
    -0.023011840834110096,
    0.007325549165585003,
    0.01565915036224212,
    -0.015187627126648262,
    -0.015250740996104902,
    4.954992196687335e-05,
    9.999999689372352e-05,
    -2.7732760149498375e-05,
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array_bytes_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


@dataclass(frozen=True, slots=True)
class _FixedPositionState:
    artifact_revision: str
    corrections: SharedGeometryCorrections
    incidence_angle_delta_rad: float

    def __post_init__(self) -> None:
        revision = str(self.artifact_revision)
        if (
            not revision.startswith("sha256-")
            or len(revision) != 71
            or any(character not in "0123456789abcdef" for character in revision[7:])
        ):
            raise ValueError("position artifact revision must be a sha256- prefixed digest")
        if not isinstance(self.corrections, SharedGeometryCorrections):
            raise TypeError("position corrections must be SharedGeometryCorrections")
        delta_rad = float(self.incidence_angle_delta_rad)
        if not math.isfinite(delta_rad):
            raise ValueError("incidence_angle_delta_rad must be finite")
        object.__setattr__(self, "artifact_revision", revision)
        object.__setattr__(self, "incidence_angle_delta_rad", delta_rad)


@dataclass(frozen=True, slots=True)
class _ProfilePhysicsContext:
    reciprocal_basis_Ainv: FloatArray
    crystal_to_sample: FloatArray
    rods: tuple[Rod, ...]
    rod_catalog_revision: str
    strength: Bi2X3FiniteStackStrength
    material: MaterialOptics
    phase_population_weight: float
    polarization_weight: float
    worker_count: int
    alpha_panel_count: int
    alpha_gauss_order: int
    azimuth_count: int
    azimuth_phase_rad: float


@dataclass(frozen=True, slots=True)
class _ProfileGeometryContext:
    incident: IncidentTransportResult
    instrument: CompiledInstrument


@dataclass(frozen=True, slots=True)
class _MeasuredProfilePolicy:
    sha256: str
    minimum_excess_energy_over_side_scatter: float
    sideband_two_theta_offsets_rad: tuple[float, ...]
    excluded_profile_keys: frozenset[tuple[str, int, int, int, int | None]]
    excluded_profiles: tuple[dict[str, object], ...]


def _profile_forward_contexts(
    base: ConfiguredSimulationInputs,
    series: tuple[ConfiguredSimulationInputs, ...],
) -> tuple[_ProfilePhysicsContext, tuple[_ProfileGeometryContext, ...]]:
    mosaic = base.config.mosaic
    physics = _ProfilePhysicsContext(
        reciprocal_basis_Ainv=base.reciprocal.basis_Ainv,
        crystal_to_sample=base.instrument.sample_from_crystal.rotation,
        rods=base.rods,
        rod_catalog_revision=configured_rod_catalog_revision(base),
        strength=base.strength,
        material=base.material,
        phase_population_weight=base.config.weights.phase_population,
        polarization_weight=base.config.weights.polarization,
        worker_count=base.config.numerics.worker_count,
        alpha_panel_count=mosaic.alpha_panel_count,
        alpha_gauss_order=mosaic.alpha_gauss_order,
        azimuth_count=mosaic.azimuth_count,
        azimuth_phase_rad=math.radians(mosaic.azimuth_phase_deg),
    )
    geometry = tuple(
        _ProfileGeometryContext(incident=item.incident, instrument=item.instrument)
        for item in series
    )
    return physics, geometry


def _source_model_record(base: ConfiguredSimulationInputs) -> dict[str, object]:
    source = base.config.source
    samples = base.samples
    return {
        "reduction": "one_incoherent_weighted_detector_function_per_incidence.v1",
        "sample_count": int(samples.incident_sample_id.size),
        "mean_origin_lab_m": list(source.mean_origin_lab_m),
        "mean_direction_lab": list(source.mean_direction_lab),
        "spatial_sigma_m": list(source.spatial_sigma_m),
        "divergence_sigma_rad": list(source.divergence_sigma_rad),
        "mean_wavelength_A": source.mean_wavelength_A,
        "wavelength_model_id": source.wavelength_model_id,
        "wavelength_sigma_A": source.wavelength_sigma_A,
        "line_wavelength_A": list(source.line_wavelength_A),
        "line_probability": list(source.line_probability),
        "common_line_sigma_A": source.common_line_sigma_A,
        "position_divergence_correlation": list(source.position_divergence_correlation),
        "sampled_wavelength_range_A": [
            float(np.min(samples.wavelength_A)),
            float(np.max(samples.wavelength_A)),
        ],
        "source_seed": samples.source_seed,
        "source_sampling_model_id": samples.source_sampling_model_id,
        "source_rng_model_id": samples.source_rng_model_id,
        "source_parameter_revision": samples.source_parameter_revision,
        "source_revision": samples.source_revision,
        "nominal_geometry_companion": {
            "sample_count": 1,
            "policy": "source-center-zero-divergence-mean-wavelength.v1",
            "contributes_detector_intensity": False,
        },
    }


def _external_directory(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or resolved.is_relative_to(ROOT):
        raise ValueError(f"output directory resolves inside the repository: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _case(path: Path) -> tuple[dict[str, Any], bytes, str]:
    case_bytes = path.read_bytes()
    case = tomllib.loads(case_bytes.decode("utf-8"))
    if case.get("schema_version") != "rasim-mosaic-recovery-v2":
        raise ValueError("unsupported mosaic recovery schema")
    required = {
        "simulation_config",
        "incidence_angles_deg",
        "shared_geometry_corrections",
        "geometry_manifest_sha256",
        "nonzero_centroid_provenance",
        "m0_centroid_provenance",
        "m0_observations",
        "truth",
        "profiles",
        "validation",
        "acceptance",
        "search",
        "render",
    }
    missing = required - set(case)
    if missing:
        raise ValueError(f"mosaic recovery case is missing {sorted(missing)[0]!r}")
    incidences = tuple(float(value) for value in case["incidence_angles_deg"])
    if incidences != (5.0, 10.0, 15.0):
        raise ValueError("the Bi2Se3 proof case requires incidences (5, 10, 15) degrees")
    for name in (
        "truth_two_theta_gauss_order",
        "truth_phi_gauss_order",
        "m0_truth_two_theta_gauss_order",
        "m0_truth_phi_gauss_order",
    ):
        order = case["validation"].get(name)
        if isinstance(order, bool) or not isinstance(order, int) or order < 2 or order % 2:
            raise ValueError(f"validation.{name} must be an even integer of at least two")
    normalization_probe_width = float(
        case["validation"].get("normalization_probe_gaussian_sigma_deg", math.nan)
    )
    if not math.isfinite(normalization_probe_width) or normalization_probe_width <= 0.0:
        raise ValueError("validation.normalization_probe_gaussian_sigma_deg must be positive")
    nuisance_scale_bounds = tuple(
        float(value) for value in case["validation"].get("unknown_intensity_scale_bounds", ())
    )
    if (
        len(nuisance_scale_bounds) != 2
        or not all(math.isfinite(value) and value > 0.0 for value in nuisance_scale_bounds)
        or nuisance_scale_bounds[0] >= nuisance_scale_bounds[1]
    ):
        raise ValueError(
            "validation.unknown_intensity_scale_bounds must be two increasing positive values"
        )
    nuisance_scale_revision = case["validation"].get("unknown_intensity_scale_revision")
    if not isinstance(nuisance_scale_revision, str) or not nuisance_scale_revision:
        raise ValueError("validation.unknown_intensity_scale_revision must be a nonempty string")
    corrections = tuple(float(value) for value in case["shared_geometry_corrections"])
    if len(corrections) != 9 or not all(math.isfinite(value) for value in corrections):
        raise ValueError("shared_geometry_corrections must contain nine finite values")
    if corrections != _ACCEPTED_FIXED_GEOMETRY_CORRECTIONS:
        raise ValueError("the tracked proof case changed the accepted nine-coordinate geometry")
    if case["geometry_manifest_sha256"] != _ACCEPTED_INDEXED_MANIFEST_SHA256:
        raise ValueError("the tracked proof case changed the accepted indexed manifest")
    profile_config = case["profiles"]
    candidate_integer_l = tuple(profile_config.get("m0_structure_candidate_integer_L", ()))
    if (
        not candidate_integer_l
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in candidate_integer_l
        )
        or tuple(sorted(set(candidate_integer_l))) != candidate_integer_l
    ):
        raise ValueError(
            "profiles.m0_structure_candidate_integer_L must be sorted unique positive integers"
        )
    divisor = profile_config.get("m0_reduced_order_divisor")
    if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor <= 0:
        raise ValueError("profiles.m0_reduced_order_divisor must be a positive integer")
    geometry_excluded_bins = _geometry_excluded_bin_map(profile_config)
    excluded_counts = {
        dataset_id: sum(key[0] == dataset_id for key in geometry_excluded_bins)
        for dataset_id in ("Bi2Se3-5deg", "Bi2Se3-10deg", "Bi2Se3-15deg")
    }
    if (
        excluded_counts != {"Bi2Se3-5deg": 10, "Bi2Se3-10deg": 8, "Bi2Se3-15deg": 8}
        or len(geometry_excluded_bins) != 26
        or sum(len(indices) for indices in geometry_excluded_bins.values()) != 43
        or any(
            key[1] != 1 or key[3] != 2 or key[4] not in {1, 2} or key[5] != (-1, 0)
            for key in geometry_excluded_bins
        )
    ):
        raise ValueError("the Bi2Se3 proof requires the frozen 26-profile, 43-bin topology mask")
    M0PeakEvidencePolicy(
        core_radius_px=profile_config.get("m0_evidence_core_radius_px", math.nan),
        background_inner_radius_px=profile_config.get(
            "m0_evidence_background_inner_radius_px", math.nan
        ),
        background_outer_radius_px=profile_config.get(
            "m0_evidence_background_outer_radius_px", math.nan
        ),
        minimum_peak_z=profile_config.get("m0_evidence_minimum_peak_z", math.nan),
        minimum_integrated_z=profile_config.get("m0_evidence_minimum_integrated_z", math.nan),
        minimum_valid_fraction=profile_config.get("m0_evidence_minimum_valid_fraction", math.nan),
        mad_scale=profile_config.get("m0_evidence_mad_scale", math.nan),
        sigma_floor_counts=profile_config.get("m0_evidence_sigma_floor_counts", math.nan),
        excess_pixel_z=profile_config.get("m0_evidence_excess_pixel_z", math.nan),
        minimum_excess_pixel_count=profile_config.get("m0_evidence_minimum_excess_pixel_count", 0),
        revision=profile_config.get("m0_evidence_revision", ""),
    )
    maximum_anchor_distance_px = float(
        profile_config.get("m0_maximum_observed_anchor_distance_px", math.nan)
    )
    if not math.isfinite(maximum_anchor_distance_px) or maximum_anchor_distance_px <= 0.0:
        raise ValueError("profiles.m0_maximum_observed_anchor_distance_px must be positive")
    nonzero_maximum_anchor_distance_px = float(
        profile_config.get("nonzero_maximum_observed_anchor_distance_px", math.nan)
    )
    if (
        not math.isfinite(nonzero_maximum_anchor_distance_px)
        or nonzero_maximum_anchor_distance_px <= 0.0
    ):
        raise ValueError("profiles.nonzero_maximum_observed_anchor_distance_px must be positive")
    observations = case["m0_observations"]
    nonzero_centroid_provenance = case["nonzero_centroid_provenance"]
    expected_nonzero_provenance_keys = {
        "series_config",
        "series_config_sha256",
        "selection_manifest_revision",
        "coordinate_convention",
        "selection_role",
    }
    if (
        not isinstance(nonzero_centroid_provenance, dict)
        or set(nonzero_centroid_provenance) != expected_nonzero_provenance_keys
        or any(
            not isinstance(value, str) or not value
            for value in nonzero_centroid_provenance.values()
        )
        or nonzero_centroid_provenance["selection_manifest_revision"]
        != f"sha256-{_ACCEPTED_INDEXED_MANIFEST_SHA256}"
    ):
        raise ValueError("nonzero_centroid_provenance does not match the strict schema")
    nonzero_series_path = (path.parent / nonzero_centroid_provenance["series_config"]).resolve()
    with nonzero_series_path.open("rb") as stream:
        nonzero_series_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if nonzero_series_hash != nonzero_centroid_provenance["series_config_sha256"]:
        raise ValueError("the frozen nonzero indexing-series config hash changed")
    centroid_provenance = case["m0_centroid_provenance"]
    expected_centroid_provenance_keys = {
        "source_file",
        "source_file_sha256",
        "coordinate_convention",
        "legacy_classification",
        "first_divergence",
        "selection_role",
    }
    if (
        not isinstance(centroid_provenance, dict)
        or set(centroid_provenance) != expected_centroid_provenance_keys
        or any(not isinstance(value, str) or not value for value in centroid_provenance.values())
        or centroid_provenance["legacy_classification"] != "CORRECTED"
    ):
        raise ValueError("m0_centroid_provenance does not match the strict schema")
    centroid_source_path = (path.parent / centroid_provenance["source_file"]).resolve()
    with centroid_source_path.open("rb") as stream:
        centroid_source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if centroid_source_hash != centroid_provenance["source_file_sha256"]:
        raise ValueError("the frozen m=0 centroid source hash changed")
    if not isinstance(observations, list) or len(observations) != len(incidences):
        raise ValueError("m0_observations must contain one record per incidence")
    expected_dataset_ids = tuple(f"Bi2Se3-{value:g}deg" for value in incidences)
    for index, (record, incidence, dataset_id) in enumerate(
        zip(observations, incidences, expected_dataset_ids, strict=True)
    ):
        required_record = {
            "dataset_id",
            "incidence_angle_deg",
            "osc_file",
            "osc_file_sha256",
            "detector_native_dtype",
            "detector_native_shape_rc",
            "detector_native_bytes_sha256",
            "indexing_discovery_revision",
            "indexed_nonzero_peaks",
            "unsupported_observed_peaks",
            "observed_peaks",
        }
        if not isinstance(record, dict) or set(record) != required_record:
            raise ValueError(f"m0_observations[{index}] keys do not match the strict schema")
        if record["dataset_id"] != dataset_id or float(record["incidence_angle_deg"]) != incidence:
            raise ValueError(f"m0_observations[{index}] dataset identity is inconsistent")
        if not isinstance(record["osc_file"], str) or not record["osc_file"]:
            raise ValueError(f"m0_observations[{index}].osc_file must be nonempty")
        for hash_name in ("osc_file_sha256", "detector_native_bytes_sha256"):
            image_hash = record[hash_name]
            try:
                valid_hash = (
                    isinstance(image_hash, str)
                    and len(image_hash) == 64
                    and int(image_hash, 16) >= 0
                )
            except ValueError:
                valid_hash = False
            if not valid_hash:
                raise ValueError(f"m0_observations[{index}].{hash_name} must be SHA-256")
        if record["detector_native_dtype"] != "int32":
            raise ValueError(f"m0_observations[{index}].detector_native_dtype must be int32")
        native_shape = record["detector_native_shape_rc"]
        if (
            not isinstance(native_shape, list)
            or len(native_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in native_shape
            )
        ):
            raise ValueError(f"m0_observations[{index}].detector_native_shape_rc is invalid")
        observed_peaks = record["observed_peaks"]
        if not isinstance(observed_peaks, list) or any(
            not isinstance(peak, dict)
            or set(peak) != {"reduced_order", "integer_L", "column_px", "row_px"}
            for peak in observed_peaks
        ):
            raise ValueError(f"m0_observations[{index}].observed_peaks is invalid")
        peak_integer_l = tuple(peak["integer_L"] for peak in observed_peaks)
        peak_orders = tuple(peak["reduced_order"] for peak in observed_peaks)
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (*peak_integer_l, *peak_orders)
            )
            or tuple(sorted(set(peak_integer_l))) != peak_integer_l
            or len(set(peak_orders)) != len(peak_orders)
            or any(
                integer_l != divisor * order
                for integer_l, order in zip(peak_integer_l, peak_orders, strict=True)
            )
            or not set(peak_integer_l).issubset(candidate_integer_l)
            or any(
                not math.isfinite(float(peak[name]))
                for peak in observed_peaks
                for name in ("column_px", "row_px")
            )
        ):
            raise ValueError(f"m0_observations[{index}].observed_peaks identities are invalid")
        discovery_revision = record["indexing_discovery_revision"]
        if (
            not isinstance(discovery_revision, str)
            or not discovery_revision.startswith("sha256-")
            or len(discovery_revision) != 71
        ):
            raise ValueError(f"m0_observations[{index}].indexing_discovery_revision is invalid")
        indexed_nonzero = record["indexed_nonzero_peaks"]
        nonzero_fields = {
            "family_m",
            "integer_L",
            "analytic_branch_id",
            "root_side_branch_id",
            "representative_rod_hk",
            "column_px",
            "row_px",
        }
        if not isinstance(indexed_nonzero, list) or len(indexed_nonzero) != (10, 8, 8)[index]:
            raise ValueError(f"m0_observations[{index}].indexed_nonzero_peaks has the wrong count")
        nonzero_identities: list[tuple[int, int, int, int, tuple[int, int]]] = []
        for peak in indexed_nonzero:
            rod_hk = peak.get("representative_rod_hk") if isinstance(peak, dict) else None
            if (
                not isinstance(peak, dict)
                or set(peak) != nonzero_fields
                or isinstance(peak["family_m"], bool)
                or not isinstance(peak["family_m"], int)
                or peak["family_m"] <= 0
                or isinstance(peak["integer_L"], bool)
                or not isinstance(peak["integer_L"], int)
                or peak["integer_L"] == 0
                or peak["analytic_branch_id"] not in {1, 2}
                or peak["root_side_branch_id"] not in {1, 2}
                or not isinstance(rod_hk, list)
                or len(rod_hk) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in rod_hk)
                or not math.isfinite(float(peak["column_px"]))
                or not math.isfinite(float(peak["row_px"]))
            ):
                raise ValueError(f"m0_observations[{index}].indexed_nonzero_peaks is invalid")
            nonzero_identities.append(
                (
                    int(peak["family_m"]),
                    int(peak["integer_L"]),
                    int(peak["analytic_branch_id"]),
                    int(peak["root_side_branch_id"]),
                    (int(rod_hk[0]), int(rod_hk[1])),
                )
            )
        if len(set(nonzero_identities)) != len(nonzero_identities):
            raise ValueError(f"m0_observations[{index}].indexed_nonzero_peaks repeats an identity")
        unsupported_peaks = record["unsupported_observed_peaks"]
        if not isinstance(unsupported_peaks, list) or any(
            not isinstance(peak, dict)
            or set(peak) != {"integer_L", "column_px", "row_px", "reason"}
            or isinstance(peak["integer_L"], bool)
            or not isinstance(peak["integer_L"], int)
            or peak["integer_L"] <= 0
            or not math.isfinite(float(peak["column_px"]))
            or not math.isfinite(float(peak["row_px"]))
            or peak["reason"] != "UNSUPPORTED_BY_TOP_EXIT_MINIMUM_TILT_FORWARD_MODEL"
            for peak in unsupported_peaks
        ):
            raise ValueError(f"m0_observations[{index}].unsupported_observed_peaks is invalid")
        unsupported_integer_l = {int(peak["integer_L"]) for peak in unsupported_peaks}
        if (
            len(unsupported_integer_l) != len(unsupported_peaks)
            or not unsupported_integer_l.issubset(candidate_integer_l)
            or unsupported_integer_l & set(peak_integer_l)
        ):
            raise ValueError(
                f"m0_observations[{index}].unsupported_observed_peaks repeats an identity"
            )
    return case, case_bytes, hashlib.sha256(case_bytes).hexdigest()


def _measured_profile_policy(
    path: Path,
    *,
    case_sha256: str,
    definitions: tuple[tuple[MosaicProfileDefinition, ...], ...],
) -> _MeasuredProfilePolicy:
    policy_bytes = path.read_bytes()
    policy = tomllib.loads(policy_bytes.decode("utf-8"))
    if policy.get("schema_version") != "rasim-measured-mosaic-profile-policy-v1":
        raise ValueError("unsupported measured mosaic profile policy schema")
    if policy.get("base_case_sha256") != case_sha256:
        raise ValueError("measured profile policy does not bind the selected base case")
    threshold = float(policy.get("minimum_excess_energy_over_side_scatter", math.nan))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("measured profile significance threshold must be positive and finite")
    raw_offsets = policy.get("sideband_two_theta_offsets_deg")
    if (
        not isinstance(raw_offsets, list)
        or len(raw_offsets) < 2
        or any(not math.isfinite(float(value)) or float(value) == 0.0 for value in raw_offsets)
    ):
        raise ValueError("measured profile sideband offsets must be finite and nonzero")
    offsets_deg = tuple(float(value) for value in raw_offsets)
    if (
        len(set(offsets_deg)) != len(offsets_deg)
        or offsets_deg != tuple(sorted(offsets_deg))
        or not (offsets_deg[0] < 0.0 < offsets_deg[-1])
    ):
        raise ValueError("measured profile sideband offsets must be unique and ordered")

    available = {
        (
            definition.identity.dataset_id,
            definition.identity.group_key.layered_family_m,
            definition.identity.group_key.layered_integer_L,
            definition.identity.analytic_branch_id,
            definition.identity.branch_id,
        )
        for dataset in definitions
        for definition in dataset
    }
    audit: list[dict[str, object]] = []
    seen: set[tuple[str, int, int, int, int | None]] = set()
    required_keys = {
        "dataset_id",
        "family_m",
        "integer_L",
        "analytic_branch_id",
        "reason",
    }
    records = policy.get("excluded_profiles", [])
    if not isinstance(records, list):
        raise ValueError("measured excluded-profile records must be a list")
    for record in records:
        if not isinstance(record, dict) or frozenset(record) not in {
            frozenset(required_keys),
            frozenset(required_keys | {"root_side_branch_id"}),
        }:
            raise ValueError("measured excluded-profile records have an invalid layout")
        raw_branch_id = record.get("root_side_branch_id")
        key = (
            str(record["dataset_id"]),
            int(record["family_m"]),
            int(record["integer_L"]),
            int(record["analytic_branch_id"]),
            None if raw_branch_id is None else int(raw_branch_id),
        )
        if key in seen or key not in available:
            raise ValueError("measured excluded-profile record is repeated or has no profile")
        seen.add(key)
        reason = str(record["reason"])
        if reason != "USER_AUTHORIZED_SECONDARY_LOBE":
            raise ValueError("measured profile exclusion lacks the accepted reason")
        audit.append(
            {
                "dataset_id": key[0],
                "family_m": key[1],
                "integer_L": key[2],
                "analytic_branch_id": key[3],
                "root_side_branch_id": key[4],
                "reason": reason,
            }
        )
    return _MeasuredProfilePolicy(
        sha256=hashlib.sha256(policy_bytes).hexdigest(),
        minimum_excess_energy_over_side_scatter=threshold,
        sideband_two_theta_offsets_rad=tuple(math.radians(value) for value in offsets_deg),
        excluded_profile_keys=frozenset(seen),
        excluded_profiles=tuple(audit),
    )


def _mosaic_parameters(
    *,
    gaussian_sigma_rad: float,
    lorentzian_half_width_rad: float,
    lorentzian_probability: float,
    context: _ProfilePhysicsContext,
) -> MosaicParameters:
    return MosaicParameters(
        gaussian_sigma_rad=gaussian_sigma_rad,
        lorentzian_half_width_rad=lorentzian_half_width_rad,
        lorentzian_probability=lorentzian_probability,
        alpha_panel_count=context.alpha_panel_count,
        alpha_gauss_order=context.alpha_gauss_order,
        azimuth_count=context.azimuth_count,
        azimuth_phase_rad=context.azimuth_phase_rad,
    )


def _fixed_geometry_inputs(
    case_path: Path,
    case: dict[str, Any],
    *,
    source_sample_count: int,
    position: _FixedPositionState,
) -> tuple[
    ConfiguredSimulationInputs,
    tuple[ConfiguredSimulationInputs, ...],
    tuple[ConfiguredGeometryInputs, ...],
]:
    if isinstance(source_sample_count, bool) or source_sample_count < 1:
        raise ValueError("source_sample_count must be a positive integer")
    if not isinstance(position, _FixedPositionState):
        raise TypeError("position must be _FixedPositionState")
    delta_rad = position.incidence_angle_delta_rad
    corrections = position.corrections
    incidence_delta_deg = math.degrees(delta_rad)
    config_path = (case_path.parent / str(case["simulation_config"])).resolve()
    config = load_simulation_config(config_path)
    first_incidence_deg = float(case["incidence_angles_deg"][0]) + incidence_delta_deg
    config = replace(
        config,
        source=replace(
            config.source,
            sample_count=int(source_sample_count),
        ),
        instrument=replace(
            config.instrument,
            axis_rotations=tuple(
                replace(axis, angle_deg=first_incidence_deg)
                for axis in config.instrument.axis_rotations
            ),
        ),
    )
    base = build_configured_simulation_inputs(config)
    geometry_base = build_configured_geometry_inputs(config)
    series: list[ConfiguredSimulationInputs] = []
    nominal_series: list[ConfiguredGeometryInputs] = []
    for commanded_incidence_deg in case["incidence_angles_deg"]:
        effective_incidence_deg = float(commanded_incidence_deg) + incidence_delta_deg
        angle_config = replace(
            config,
            instrument=replace(
                config.instrument,
                axis_rotations=tuple(
                    replace(axis, angle_deg=effective_incidence_deg)
                    for axis in config.instrument.axis_rotations
                ),
            ),
        )
        geometry = rebind_configured_geometry_instrument(geometry_base, angle_config)
        corrected_instrument = apply_shared_geometry_corrections(
            geometry.instrument,
            angle_config.instrument.axis_rotations,
            corrections,
        )
        incident = build_incident_states(base.samples, base.material, corrected_instrument)
        if incident.states.incident_state_id.size != source_sample_count or not bool(
            np.all(incident.states.valid)
        ):
            raise RuntimeError("every configured source state must produce a valid incident state")
        series.append(
            replace(
                base,
                config=angle_config,
                instrument=corrected_instrument,
                incident=incident,
            )
        )
        nominal_series.append(
            replace(
                geometry,
                instrument=corrected_instrument,
            )
        )
    return base, tuple(series), tuple(nominal_series)


def _fixed_geometry_record(
    case: dict[str, Any],
    base: ConfiguredSimulationInputs,
    position: _FixedPositionState,
) -> dict[str, object]:
    commanded_deg = tuple(float(value) for value in case["incidence_angles_deg"])
    delta_rad = position.incidence_angle_delta_rad
    effective_deg = tuple(value + math.degrees(delta_rad) for value in commanded_deg)
    return {
        "position_artifact_revision": position.artifact_revision,
        "corrections": {
            name: float(value)
            for name, value in zip(
                SHARED_GEOMETRY_PARAMETER_NAMES,
                position.corrections.as_array(),
                strict=True,
            )
        },
        "incidence_angle_model_id": "commanded_angle_plus_common_delta.v1",
        "incidence_angle_delta_rad": delta_rad,
        "commanded_incidence_angles_deg": commanded_deg,
        "effective_incidence_angles_deg": effective_deg,
        "beam_center_column_row_px": tuple(
            float(value) for value in base.config.instrument.detector_reference_coordinate_px
        ),
        "geometry_parameters_fitted_here": False,
    }


def _case_fixed_position_state(case: dict[str, Any]) -> _FixedPositionState:
    return _FixedPositionState(
        artifact_revision=f"sha256-{case['geometry_manifest_sha256']}",
        corrections=SharedGeometryCorrections.from_array(case["shared_geometry_corrections"]),
        incidence_angle_delta_rad=0.0,
    )


def _position_artifact_state(
    path: Path,
    case: dict[str, Any],
) -> _FixedPositionState:
    document = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("position artifact must contain one JSON object")
    revision = verify_staged_fit_stage_artifact(document, expected_stage="geometry")
    if document.get("material_id") != "Bi2Se3":
        raise ValueError("position artifact is not a Bi2Se3 geometry result")
    state = document.get("state")
    if not isinstance(state, dict) or set(state) != {
        "corrections",
        "fit_evidence",
        "incidence_angle_delta_rad",
        "incidence_angles",
    }:
        raise ValueError("position artifact has an invalid geometry state")
    corrections_record = state["corrections"]
    if (
        not isinstance(corrections_record, list)
        or len(corrections_record) != len(SHARED_GEOMETRY_PARAMETER_NAMES)
        or any(not math.isfinite(float(value)) for value in corrections_record)
        or float(corrections_record[2]) != 0.0
    ):
        raise ValueError("position artifact has invalid shared geometry corrections")
    delta_rad = float(state["incidence_angle_delta_rad"])
    if not math.isfinite(delta_rad):
        raise ValueError("position artifact has a nonfinite common incidence-angle delta")
    incidence_records = state["incidence_angles"]
    commanded_deg = tuple(float(value) for value in case["incidence_angles_deg"])
    if not isinstance(incidence_records, list) or len(incidence_records) != len(commanded_deg):
        raise ValueError("position artifact has an incomplete incidence-angle series")
    for index, (record, angle_deg) in enumerate(zip(incidence_records, commanded_deg, strict=True)):
        if not isinstance(record, dict) or set(record) != {
            "commanded_angle_rad",
            "effective_angle_rad",
            "image_id",
        }:
            raise ValueError(f"position artifact incidence record {index} is invalid")
        commanded_rad = math.radians(angle_deg)
        if (
            float(record["commanded_angle_rad"]) != commanded_rad
            or float(record["effective_angle_rad"]) != commanded_rad + delta_rad
        ):
            raise ValueError("position artifact does not apply one common incidence-angle delta")
    fit_evidence = state["fit_evidence"]
    qualification = fit_evidence.get("qualification") if isinstance(fit_evidence, dict) else None
    if (
        not isinstance(qualification, dict)
        or not bool(qualification.get("accepted"))
        or fit_evidence.get("indexed_manifest_hash") != f"sha256-{case['geometry_manifest_sha256']}"
    ):
        raise ValueError("position artifact is not qualified for downstream fitting")
    summary = document.get("scientific_summary")
    geometry_summary = summary.get("geometry") if isinstance(summary, dict) else None
    if (
        not isinstance(geometry_summary, dict)
        or geometry_summary.get("corrections") != corrections_record
        or geometry_summary.get("incidence_angle_delta_rad") != delta_rad
    ):
        raise ValueError("position artifact state disagrees with its scientific summary")
    return _FixedPositionState(
        artifact_revision=revision,
        corrections=SharedGeometryCorrections.from_array(corrections_record),
        incidence_angle_delta_rad=delta_rad,
    )


def _geometry_excluded_bin_map(
    profile_config: dict[str, Any],
) -> dict[tuple[str, int, int, int, int, tuple[int, int]], tuple[int, ...]]:
    for name in ("excluded_bin_policy", "excluded_bin_topology"):
        value = profile_config.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"profiles.{name} must be a nonempty string")
    phi_bin_count = profile_config.get("phi_bin_count")
    if isinstance(phi_bin_count, bool) or not isinstance(phi_bin_count, int):
        raise ValueError("profiles.phi_bin_count must be an integer")
    records = tuple(profile_config.get("geometry_excluded_phi_bins", ()))
    result: dict[tuple[str, int, int, int, int, tuple[int, int]], tuple[int, ...]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("profiles.geometry_excluded_phi_bins must contain tables")
        dataset_id = record.get("dataset_id")
        family_m = record.get("family_m")
        integer_l = record.get("integer_L")
        analytic_branch_id = record.get("analytic_branch_id")
        root_side_branch_id = record.get("root_side_branch_id")
        representative_rod_hk = tuple(record.get("representative_rod_hk", ()))
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("each excluded-bin record requires a dataset_id")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (family_m, integer_l, analytic_branch_id, root_side_branch_id)
        ):
            raise ValueError("excluded-bin reflection identities must be integers")
        if len(representative_rod_hk) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in representative_rod_hk
        ):
            raise ValueError("excluded-bin representative_rod_hk must contain two integers")
        key = (
            dataset_id,
            int(family_m),
            int(integer_l),
            int(analytic_branch_id),
            int(root_side_branch_id),
            (int(representative_rod_hk[0]), int(representative_rod_hk[1])),
        )
        indices = tuple(record.get("indices", ()))
        if (
            not indices
            or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
            or indices != tuple(sorted(set(indices)))
            or any(index < 0 or index >= phi_bin_count for index in indices)
        ):
            raise ValueError(
                "excluded-bin indices must be sorted, unique, nonempty, and inside the profile"
            )
        if key in result:
            raise ValueError(f"duplicate excluded-bin reflection identity: {key}")
        result[key] = tuple(int(index) for index in indices)
    declared_sha256 = profile_config.get("excluded_bin_topology_sha256")
    if (
        not isinstance(declared_sha256, str)
        or len(declared_sha256) != 64
        or any(character not in "0123456789abcdef" for character in declared_sha256)
    ):
        raise ValueError("profiles.excluded_bin_topology_sha256 must be lowercase SHA-256")
    canonical_payload = {
        "excluded_bin_policy": profile_config["excluded_bin_policy"],
        "excluded_bin_topology": profile_config["excluded_bin_topology"],
        "phi_bin_count": phi_bin_count,
        "records": [
            {
                "dataset_id": key[0],
                "family_m": key[1],
                "integer_L": key[2],
                "analytic_branch_id": key[3],
                "root_side_branch_id": key[4],
                "representative_rod_hk": list(key[5]),
                "indices": list(indices),
            }
            for key, indices in sorted(result.items())
        ],
    }
    actual_sha256 = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if actual_sha256 != declared_sha256:
        raise ValueError("the frozen excluded-bin topology SHA-256 changed")
    return result


def _profile_definitions(
    inputs: ConfiguredGeometryInputs,
    *,
    source_inputs: ConfiguredSimulationInputs,
    include_nominally_unsupported_m0: bool,
    case_path: Path,
    incidence_deg: float,
    osc_observation: dict[str, Any],
    nonzero_centroid_provenance: dict[str, str],
    centroid_provenance: dict[str, str],
    profile_config: dict[str, Any],
    rod_catalog_revision: str,
) -> tuple[
    AngleFrame,
    tuple[MosaicProfileDefinition, ...],
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    effective_incidence_deg = float(inputs.config.instrument.axis_rotations[0].angle_deg)
    frame_revision = (
        f"fixed-nine-coordinate-{incidence_deg:g}deg.v1"
        if effective_incidence_deg == incidence_deg
        else (
            "fixed-position-state-"
            f"commanded-{incidence_deg:g}deg-effective-{effective_incidence_deg:.12g}deg.v2"
        )
    )
    context = build_nominal_ewald_context(source_inputs)
    frame = build_osc_angle_frame(
        mean_direction_lab=inputs.config.source.mean_direction_lab,
        instrument=inputs.instrument,
        sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
        revision=frame_revision,
    )
    markers = evaluate_nominal_integer_l_markers(context)
    indexed_nonzero_peaks = tuple(osc_observation["indexed_nonzero_peaks"])
    indexed_nonzero_angles = detector_coordinates_to_angles(
        np.asarray([float(item["column_px"]) for item in indexed_nonzero_peaks]),
        np.asarray([float(item["row_px"]) for item in indexed_nonzero_peaks]),
        instrument=inputs.instrument,
        angle_frame=frame,
    )
    if not np.all(indexed_nonzero_angles.valid & indexed_nonzero_angles.azimuth_valid):
        raise RuntimeError("a frozen indexed nonzero-m centroid has no valid angle coordinate")
    common = {
        "two_theta_half_width_rad": math.radians(float(profile_config["two_theta_half_width_deg"])),
        "phi_bin_count": int(profile_config["phi_bin_count"]),
    }
    nonzero_quadrature = {
        "two_theta_gauss_order": int(profile_config["two_theta_gauss_order"]),
        "phi_gauss_order": int(profile_config["phi_gauss_order"]),
    }
    m0_quadrature = {
        "two_theta_gauss_order": int(profile_config["m0_two_theta_gauss_order"]),
        "phi_gauss_order": int(profile_config["m0_phi_gauss_order"]),
    }
    dataset_id = f"Bi2Se3-{incidence_deg:g}deg"
    geometry_excluded_bins = _geometry_excluded_bin_map(profile_config)
    dataset_excluded_keys = {key for key in geometry_excluded_bins if key[0] == dataset_id}
    used_excluded_keys: set[tuple[str, int, int, int, int, tuple[int, int]]] = set()
    marker_index_by_key: dict[tuple[int, int, int, int, tuple[int, int]], int] = {}
    for marker_index in range(markers.column_px.size):
        root_sign = int(markers.root_sign[marker_index])
        if int(markers.family_m[marker_index]) == 0 or root_sign == 0:
            continue
        marker_key = (
            int(markers.family_m[marker_index]),
            int(markers.integer_L[marker_index]),
            int(markers.branch[marker_index]),
            1 if root_sign < 0 else 2,
            markers.contributing_rod_hk[marker_index][0],
        )
        if marker_key in marker_index_by_key:
            raise RuntimeError(f"nominal marker identity is not unique: {marker_key}")
        marker_index_by_key[marker_key] = marker_index

    nonzero_candidates: list[MosaicProfileDefinition] = []
    exclusions: list[dict[str, object]] = []
    selected_marker_indices: set[int] = set()
    nonzero_anchor_records: list[dict[str, object]] = []
    maximum_nonzero_anchor_distance = float(
        profile_config["nonzero_maximum_observed_anchor_distance_px"]
    )
    for observed_index, observed in enumerate(indexed_nonzero_peaks):
        marker_key = (
            int(observed["family_m"]),
            int(observed["integer_L"]),
            int(observed["analytic_branch_id"]),
            int(observed["root_side_branch_id"]),
            tuple(int(value) for value in observed["representative_rod_hk"]),
        )
        try:
            index = marker_index_by_key[marker_key]
        except KeyError as error:
            raise RuntimeError(
                f"indexed nonzero identity has no nominal marker: {marker_key}"
            ) from error
        selected_marker_indices.add(index)
        family_m, integer_l, analytic_branch, root_side_branch, _ = marker_key
        excluded_key = (dataset_id, *marker_key)
        try:
            excluded_phi_bin_indices = geometry_excluded_bins[excluded_key]
        except KeyError as error:
            raise RuntimeError(
                f"indexed nonzero profile lacks a frozen topology mask: {excluded_key}"
            ) from error
        used_excluded_keys.add(excluded_key)
        anchor_distance = math.hypot(
            float(observed["column_px"]) - float(markers.column_px[index]),
            float(observed["row_px"]) - float(markers.row_px[index]),
        )
        if anchor_distance > maximum_nonzero_anchor_distance:
            raise RuntimeError(
                f"indexed m={family_m}, L={integer_l}, side={root_side_branch} lies "
                f"{anchor_distance:.6g} px from its fixed-geometry marker"
            )
        nonzero_candidates.append(
            MosaicProfileDefinition(
                identity=MosaicProfileIdentity(
                    dataset_id=f"Bi2Se3-{incidence_deg:g}deg",
                    incidence_angle_rad=math.radians(effective_incidence_deg),
                    group_key=MosaicReflectionGroupKey(
                        group_id=f"Bi2Se3:m={family_m}:L={integer_l}",
                        rod_catalog_revision=rod_catalog_revision,
                        member_rod_hk=markers.contributing_rod_hk[index],
                        branch_mode="EXPLICIT_NONZERO",
                        layered_family_m=family_m,
                        layered_integer_L=integer_l,
                    ),
                    branch_id=root_side_branch,
                    analytic_branch_id=analytic_branch,
                ),
                center_two_theta_rad=float(indexed_nonzero_angles.two_theta_rad[observed_index]),
                center_phi_rad=float(indexed_nonzero_angles.phi_rad[observed_index]),
                phi_half_width_rad=math.radians(
                    float(profile_config["nonzero_phi_half_width_deg"])
                ),
                **common,
                **nonzero_quadrature,
                excluded_phi_bin_indices=excluded_phi_bin_indices,
            )
        )
        nonzero_anchor_records.append(
            {
                **dict(observed),
                "member_rod_hk": [list(value) for value in markers.contributing_rod_hk[index]],
                "predicted_column_px": float(markers.column_px[index]),
                "predicted_row_px": float(markers.row_px[index]),
                "observed_to_marker_distance_px": anchor_distance,
            }
        )

    for marker_key, marker_index in marker_index_by_key.items():
        if marker_index not in selected_marker_indices:
            family_m, integer_l, analytic_branch, root_side_branch, _ = marker_key
            exclusions.append(
                {
                    "dataset_id": f"Bi2Se3-{incidence_deg:g}deg",
                    "family_m": family_m,
                    "integer_L": integer_l,
                    "analytic_branch_id": analytic_branch,
                    "root_side_branch_id": root_side_branch,
                    "reason": "NOT_IN_FROZEN_OSC_INDEX_SELECTION",
                }
            )

    if used_excluded_keys != dataset_excluded_keys:
        unused = tuple(sorted(dataset_excluded_keys - used_excluded_keys))
        raise RuntimeError(f"frozen topology masks do not match indexed profiles: {unused}")

    definitions: list[MosaicProfileDefinition] = list(nonzero_candidates)

    model = ExactTagGeometryModel(inputs)
    wavevector_magnitude_Ainv = (
        float(
            np.max(np.linalg.norm(source_inputs.incident.states.k_film_phase_sample_Ainv, axis=1))
        )
        if include_nominally_unsupported_m0
        else float(np.linalg.norm(context.ki_sample_Ainv))
    )
    reciprocal_c_Ainv = float(np.linalg.norm(context.reciprocal_basis_Ainv[:, 2]))
    kinematic_l_limit_float = 2.0 * wavevector_magnitude_Ainv / reciprocal_c_Ainv
    kinematic_l_limit = math.floor(
        kinematic_l_limit_float
        + 4096.0 * np.finfo(np.float64).eps * max(kinematic_l_limit_float, 1.0)
    )
    candidate_integer_l = tuple(
        int(value) for value in profile_config["m0_structure_candidate_integer_L"]
    )
    reduced_order_divisor = int(profile_config["m0_reduced_order_divisor"])
    expected_candidates = tuple(
        range(reduced_order_divisor, kinematic_l_limit + 1, reduced_order_divisor)
    )
    if candidate_integer_l != expected_candidates:
        raise RuntimeError(
            "the Bi2Se3 m=0 structure-candidate list must exhaust positive 00(3n) orders"
        )
    m0 = model.predict_m0_minimum_tilt_exact_l_landmarks(candidate_integer_l)
    m0_angles = detector_coordinates_to_angles(
        m0.coordinates_px[:, 0],
        m0.coordinates_px[:, 1],
        instrument=inputs.instrument,
        angle_frame=frame,
    )
    m0_rod_objects = tuple(rod for rod in inputs.rods if rod.family_m == 0)
    if len(m0_rod_objects) != 1:
        raise RuntimeError("the configured proof requires one physical m=0 rod")
    m0_rod = m0_rod_objects[0]
    m0_rods = ((m0_rod.h, m0_rod.k),)
    evidence_policy = M0PeakEvidencePolicy(
        core_radius_px=float(profile_config["m0_evidence_core_radius_px"]),
        background_inner_radius_px=float(profile_config["m0_evidence_background_inner_radius_px"]),
        background_outer_radius_px=float(profile_config["m0_evidence_background_outer_radius_px"]),
        minimum_peak_z=float(profile_config["m0_evidence_minimum_peak_z"]),
        minimum_integrated_z=float(profile_config["m0_evidence_minimum_integrated_z"]),
        minimum_valid_fraction=float(profile_config["m0_evidence_minimum_valid_fraction"]),
        mad_scale=float(profile_config["m0_evidence_mad_scale"]),
        sigma_floor_counts=float(profile_config["m0_evidence_sigma_floor_counts"]),
        excess_pixel_z=float(profile_config["m0_evidence_excess_pixel_z"]),
        minimum_excess_pixel_count=int(profile_config["m0_evidence_minimum_excess_pixel_count"]),
        revision=str(profile_config["m0_evidence_revision"]),
    )
    forward_candidates = tuple(
        ExpectedM0Peak(
            integer_L=int(integer_l_value),
            column_px=float(m0.coordinates_px[index, 0]),
            row_px=float(m0.coordinates_px[index, 1]),
        )
        for index, integer_l_value in enumerate(m0.integer_L)
        if str(m0.detector_status[index]) == "VALID"
        and bool(m0_angles.valid[index] & m0_angles.azimuth_valid[index])
    )
    osc_path = (case_path.parent / str(osc_observation["osc_file"])).resolve()
    with osc_path.open("rb") as stream:
        osc_file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if osc_file_hash != osc_observation["osc_file_sha256"]:
        raise RuntimeError("the m=0 OSC file hash changed")
    detector_counts = read_osc(osc_path).detector_native_counts
    expected_shape = tuple(int(value) for value in osc_observation["detector_native_shape_rc"])
    if (
        detector_counts.shape != inputs.instrument.detector_shape_rc
        or detector_counts.shape != expected_shape
        or detector_counts.dtype.name != osc_observation["detector_native_dtype"]
    ):
        raise RuntimeError("the m=0 OSC detector-native shape or dtype changed")
    detector_hash = hashlib.sha256(
        memoryview(np.ascontiguousarray(detector_counts)).cast("B")
    ).hexdigest()
    if detector_hash != osc_observation["detector_native_bytes_sha256"]:
        raise RuntimeError("the m=0 OSC detector-native data hash changed")
    detector_mask = detector_valid_mask_from_counts(detector_counts)
    detector_mask_hash = hashlib.sha256(
        memoryview(np.ascontiguousarray(detector_mask)).cast("B")
    ).hexdigest()
    predicted_evidence = evaluate_expected_m0_peak_evidence(
        detector_counts,
        candidates=forward_candidates,
        detector_valid_mask=detector_mask,
        policy=evidence_policy,
    )
    predicted_evidence_by_l = {item.integer_L: item for item in predicted_evidence}
    observed_peaks = tuple(osc_observation["observed_peaks"])
    observed_integer_l = tuple(int(item["integer_L"]) for item in observed_peaks)
    measured_supported_integer_l = tuple(
        item.integer_L for item in predicted_evidence if item.accepted
    )
    if observed_integer_l != measured_supported_integer_l:
        raise RuntimeError(
            "the frozen observed m=0 identities disagree with raw OSC significance: "
            f"observed={observed_integer_l}, significant={measured_supported_integer_l}"
        )
    observed_candidates = tuple(
        ExpectedM0Peak(
            integer_L=int(item["integer_L"]),
            column_px=float(item["column_px"]),
            row_px=float(item["row_px"]),
        )
        for item in observed_peaks
    )
    observed_evidence = evaluate_expected_m0_peak_evidence(
        detector_counts,
        candidates=observed_candidates,
        detector_valid_mask=detector_mask,
        policy=evidence_policy,
    )
    if not all(item.accepted for item in observed_evidence):
        raise RuntimeError("a frozen observed m=0 centroid lacks significant raw OSC support")
    unsupported_observed_peaks = tuple(osc_observation["unsupported_observed_peaks"])
    m0_index_by_l = {int(value): index for index, value in enumerate(m0.integer_L)}
    for item in unsupported_observed_peaks:
        status = str(m0.detector_status[m0_index_by_l[int(item["integer_L"])]])
        if status == "VALID":
            raise RuntimeError(
                "an unsupported observed m=0 peak is valid in the fixed forward landmark model"
            )
    unsupported_candidates = tuple(
        ExpectedM0Peak(
            integer_L=int(item["integer_L"]),
            column_px=float(item["column_px"]),
            row_px=float(item["row_px"]),
        )
        for item in unsupported_observed_peaks
    )
    unsupported_evidence = evaluate_expected_m0_peak_evidence(
        detector_counts,
        candidates=unsupported_candidates,
        detector_valid_mask=detector_mask,
        policy=evidence_policy,
    )
    if not all(item.accepted for item in unsupported_evidence):
        raise RuntimeError("a forward-unsupported observed m=0 peak lacks raw OSC significance")
    provisional_candidates = (
        observed_candidates + unsupported_candidates
        if include_nominally_unsupported_m0
        else observed_candidates
    )
    provisional_integer_l = {item.integer_L for item in provisional_candidates}
    provisional_angles = detector_coordinates_to_angles(
        np.asarray([item.column_px for item in provisional_candidates]),
        np.asarray([item.row_px for item in provisional_candidates]),
        instrument=inputs.instrument,
        angle_frame=frame,
    )
    if not np.all(provisional_angles.valid & provisional_angles.azimuth_valid):
        raise RuntimeError(
            "a provisional raw-significant m=0 centroid has no valid angle coordinate"
        )
    maximum_anchor_distance = float(profile_config["m0_maximum_observed_anchor_distance_px"])
    observed_anchor_distance_by_l: dict[int, float] = {}
    for observed in observed_candidates:
        prediction_index = m0_index_by_l[observed.integer_L]
        distance = math.hypot(
            observed.column_px - float(m0.coordinates_px[prediction_index, 0]),
            observed.row_px - float(m0.coordinates_px[prediction_index, 1]),
        )
        observed_anchor_distance_by_l[observed.integer_L] = distance
        if distance > maximum_anchor_distance:
            raise RuntimeError(
                f"observed m=0 L={observed.integer_L} lies {distance:.6g} px from its "
                "fixed-geometry landmark"
            )

    for observed_index, observed in enumerate(provisional_candidates):
        integer_l_value = observed.integer_L
        definitions.append(
            MosaicProfileDefinition(
                identity=MosaicProfileIdentity(
                    dataset_id=f"Bi2Se3-{incidence_deg:g}deg",
                    incidence_angle_rad=math.radians(effective_incidence_deg),
                    group_key=MosaicReflectionGroupKey(
                        group_id=f"Bi2Se3:m=0:L={integer_l_value}",
                        rod_catalog_revision=rod_catalog_revision,
                        member_rod_hk=m0_rods,
                        branch_mode="COLLAPSED_00L",
                        layered_family_m=0,
                        layered_integer_L=integer_l_value,
                    ),
                    branch_id=None,
                    analytic_branch_id=0,
                ),
                center_two_theta_rad=float(provisional_angles.two_theta_rad[observed_index]),
                center_phi_rad=float(provisional_angles.phi_rad[observed_index]),
                phi_half_width_rad=math.radians(float(profile_config["m0_phi_half_width_deg"])),
                **common,
                **m0_quadrature,
            )
        )

    def evidence_record(item: M0PeakEvidence) -> dict[str, object]:
        return {
            "classification": item.classification,
            "core_pixel_count": item.core_pixel_count,
            "background_pixel_count": item.background_pixel_count,
            "excess_pixel_count": item.excess_pixel_count,
            "background_counts": item.background_counts,
            "robust_sigma_counts": item.robust_sigma_counts,
            "maximum_core_counts": item.maximum_core_counts,
            "peak_z": item.peak_z,
            "integrated_z": item.integrated_z,
        }

    candidate_records: list[dict[str, object]] = []
    unsupported_evidence_by_l = {item.integer_L: item for item in unsupported_evidence}
    strength_k_norm = float(np.linalg.norm(context.ki_sample_Ainv))
    for index, integer_l_value in enumerate(m0.integer_L):
        integer_l = int(integer_l_value)
        status = str(m0.detector_status[index])
        raw_evidence = predicted_evidence_by_l.get(integer_l)
        record: dict[str, object] = {
            "integer_L": integer_l,
            "reduced_order": integer_l // reduced_order_divisor,
            "structure_strength_A2": source_inputs.strength.evaluate(
                rod=m0_rod,
                L=float(integer_l),
                k_norm_Ainv=strength_k_norm,
            ),
            "minimum_tilt_status": status,
            "predicted_column_px": float(m0.coordinates_px[index, 0]),
            "predicted_row_px": float(m0.coordinates_px[index, 1]),
            "angle_valid": bool(m0_angles.valid[index]),
            "azimuth_valid": bool(m0_angles.azimuth_valid[index]),
            "minimum_tilt_alpha_deg": math.degrees(float(m0.alpha_rad[index])),
            "raw_evidence": None if raw_evidence is None else evidence_record(raw_evidence),
            "unsupported_observed_raw_evidence": (
                None
                if integer_l not in unsupported_evidence_by_l
                else evidence_record(unsupported_evidence_by_l[integer_l])
            ),
            "forward_supported_candidate": integer_l in observed_integer_l,
            **(
                {"combined_source_gate_candidate": integer_l in provisional_integer_l}
                if include_nominally_unsupported_m0
                else {}
            ),
        }
        candidate_records.append(record)
        if integer_l not in provisional_integer_l:
            if status != "VALID":
                exclusion_reason = status
            elif not bool(m0_angles.valid[index]):
                exclusion_reason = "ANGLE_UNDEFINED"
            elif not bool(m0_angles.azimuth_valid[index]):
                exclusion_reason = "AZIMUTH_UNDEFINED"
            elif raw_evidence is None:
                exclusion_reason = "NO_RAW_EVIDENCE"
            else:
                exclusion_reason = raw_evidence.classification
            exclusions.append(
                {
                    "dataset_id": f"Bi2Se3-{incidence_deg:g}deg",
                    "family_m": 0,
                    "integer_L": integer_l,
                    "reason": exclusion_reason,
                }
            )
    observed_evidence_by_l = {item.integer_L: item for item in observed_evidence}
    m0_support_audit = {
        "dataset_id": f"Bi2Se3-{incidence_deg:g}deg",
        "indexed_nonzero_selection": {
            "centroid_provenance": dict(nonzero_centroid_provenance),
            "discovery_revision": str(osc_observation["indexing_discovery_revision"]),
            "selected_count": len(nonzero_anchor_records),
            "maximum_observed_anchor_distance_px": maximum_nonzero_anchor_distance,
            "selected_peaks": nonzero_anchor_records,
        },
        "kinematic_positive_integer_l_bounds": [1, kinematic_l_limit],
        "structure_candidate_integer_L": list(candidate_integer_l),
        "candidate_policy": "configured_crystallographic_00_3n_orders.v1",
        "forward_landmark_policy": m0.landmark_policy,
        "raw_evidence_policy": {
            "core_radius_px": evidence_policy.core_radius_px,
            "background_inner_radius_px": evidence_policy.background_inner_radius_px,
            "background_outer_radius_px": evidence_policy.background_outer_radius_px,
            "minimum_peak_z": evidence_policy.minimum_peak_z,
            "minimum_integrated_z": evidence_policy.minimum_integrated_z,
            "minimum_valid_fraction": evidence_policy.minimum_valid_fraction,
            "mad_scale": evidence_policy.mad_scale,
            "sigma_floor_counts": evidence_policy.sigma_floor_counts,
            "excess_pixel_z": evidence_policy.excess_pixel_z,
            "minimum_excess_pixel_count": evidence_policy.minimum_excess_pixel_count,
            "revision": evidence_policy.revision,
            "detector_valid_mask_revision": DETECTOR_VALID_MASK_REVISION,
            "maximum_observed_anchor_distance_px": maximum_anchor_distance,
        },
        "osc_path": str(osc_observation["osc_file"]),
        "osc_file_sha256": osc_file_hash,
        "detector_native_dtype": detector_counts.dtype.name,
        "detector_native_shape_rc": list(detector_counts.shape),
        "detector_native_bytes_sha256": detector_hash,
        "detector_valid_mask_sha256": detector_mask_hash,
        "centroid_provenance": dict(centroid_provenance),
        "candidate_records": candidate_records,
        "observed_peaks": [
            {
                **dict(item),
                "observed_to_landmark_distance_px": observed_anchor_distance_by_l[
                    int(item["integer_L"])
                ],
                "raw_evidence": evidence_record(observed_evidence_by_l[int(item["integer_L"])]),
            }
            for item in observed_peaks
        ],
        "raw_significant_unsupported_forward_peaks": [
            {
                **dict(item),
                "classification": (
                    "RAW_SIGNIFICANT_NOMINAL_LANDMARK_UNSUPPORTED_PENDING_COMBINED_SOURCE_GATE"
                    if include_nominally_unsupported_m0
                    else "RAW_SIGNIFICANT_UNSUPPORTED_FORWARD_CHANNEL"
                ),
                **(
                    {"combined_source_gate_candidate": True}
                    if include_nominally_unsupported_m0
                    else {"fit_eligible": False}
                ),
                "raw_evidence": evidence_record(unsupported_evidence_by_l[int(item["integer_L"])]),
            }
            for item in unsupported_observed_peaks
        ],
        "forward_supported_candidate_integer_L": list(observed_integer_l),
        **(
            {
                "combined_source_gate_candidate_integer_L": [
                    item.integer_L for item in provisional_candidates
                ]
            }
            if include_nominally_unsupported_m0
            else {}
        ),
        "signed_l_policy": "POSITIVE_ABS_L_WITH_COLLAPSED_INVERSE_ORIENTATIONS",
        "zero_l_policy": "DIRECT_BEAM_NOT_BRAGG_PEAK",
    }
    definitions.sort(
        key=lambda item: (
            item.identity.group_key.group_id,
            item.identity.analytic_branch_id,
            -1 if item.identity.branch_id is None else item.identity.branch_id,
        )
    )
    return frame, tuple(definitions), tuple(exclusions), m0_support_audit


def _profile_revision(
    definitions: tuple[tuple[MosaicProfileDefinition, ...], ...],
    frames: tuple[AngleFrame, ...],
    exclusions: tuple[dict[str, object], ...],
    m0_support_audits: tuple[dict[str, object], ...],
    case: dict[str, Any],
    simulation_config_sha256: str,
    configured_physics_revision: str,
    cif_sha256: str,
    rod_catalog_revision: str,
    source_revision: str,
    fixed_geometry: dict[str, object],
    measured_profile_policy_sha256: str | None = None,
) -> str:
    payload = {
        "fixed_geometry": fixed_geometry,
        "simulation_config_sha256": simulation_config_sha256,
        "configured_physics_revision": configured_physics_revision,
        "cif_sha256": cif_sha256,
        "rod_catalog_revision": rod_catalog_revision,
        "source_policy": "seeded-empirical-source-ensemble.v1",
        "source_revision": source_revision,
        "excluded_bin_policy": case["profiles"]["excluded_bin_policy"],
        "excluded_bin_topology": case["profiles"]["excluded_bin_topology"],
        "selection_policy": (
            "nominal-nonzero-paired-roots-plus-structure-forward-raw-significant-00L.v1"
        ),
        "selection_exclusions": exclusions,
        "m0_support_audits": m0_support_audits,
        "angle_frames": [
            {
                "revision": frame.revision,
                "origin_lab_m": frame.origin_lab_m.tolist(),
                "row_down_lab": frame.row_down_lab.tolist(),
                "column_right_lab": frame.column_right_lab.tolist(),
                "direct_beam_lab": frame.direct_beam_lab.tolist(),
            }
            for frame in frames
        ],
        "profiles": [
            {
                "dataset_id": item.identity.dataset_id,
                "incidence_angle_rad": item.identity.incidence_angle_rad,
                "group_id": item.identity.group_key.group_id,
                "rod_catalog_revision": item.identity.group_key.rod_catalog_revision,
                "member_rod_hk": item.identity.group_key.member_rod_hk,
                "branch_mode": item.identity.group_key.branch_mode,
                "layered_family_m": item.identity.group_key.layered_family_m,
                "layered_integer_L": item.identity.group_key.layered_integer_L,
                "analytic_branch_id": item.identity.analytic_branch_id,
                "branch_id": item.identity.branch_id,
                "center_two_theta_rad": item.center_two_theta_rad,
                "center_phi_rad": item.center_phi_rad,
                "two_theta_half_width_rad": item.two_theta_half_width_rad,
                "phi_half_width_rad": item.phi_half_width_rad,
                "phi_bin_count": item.phi_bin_count,
                "two_theta_gauss_order": item.two_theta_gauss_order,
                "phi_gauss_order": item.phi_gauss_order,
                "excluded_phi_bin_indices": item.excluded_phi_bin_indices,
            }
            for dataset in definitions
            for item in dataset
        ],
    }
    if measured_profile_policy_sha256 is not None:
        payload["measured_profile_policy_sha256"] = measured_profile_policy_sha256
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256-{hashlib.sha256(encoded).hexdigest()}"


def _detector_series(
    physics: _ProfilePhysicsContext,
    geometry: tuple[_ProfileGeometryContext, ...],
    mosaic: MosaicParameters,
    *,
    rods: tuple[Rod, ...] | None = None,
) -> tuple[SourceAveragedDetectorEwaldMeasure, ...]:
    first = geometry[0]
    detector = SourceAveragedDetectorEwaldMeasure(
        reciprocal_basis_Ainv=physics.reciprocal_basis_Ainv,
        crystal_to_sample=physics.crystal_to_sample,
        rods=physics.rods if rods is None else rods,
        rod_catalog_revision=physics.rod_catalog_revision,
        mosaic=mosaic,
        strength_model=physics.strength,
        incident=first.incident,
        material=physics.material,
        instrument=first.instrument,
        phase_population_weight=physics.phase_population_weight,
        polarization_weight=physics.polarization_weight,
        worker_count=physics.worker_count,
    )
    return (
        detector,
        *(
            detector.rebind_geometry(incident=item.incident, instrument=item.instrument)
            for item in geometry[1:]
        ),
    )


def _profile_rods(
    rods: tuple[Rod, ...],
    definitions: tuple[tuple[MosaicProfileDefinition, ...], ...],
) -> tuple[Rod, ...]:
    required_hk = {
        member_hk
        for dataset in definitions
        for definition in dataset
        for member_hk in definition.identity.group_key.member_rod_hk
    }
    selected = tuple(rod for rod in rods if (rod.h, rod.k) in required_hk)
    selected_hk = {(rod.h, rod.k) for rod in selected}
    if selected_hk != required_hk:
        raise ValueError(f"profile definitions reference absent rods: {required_hk - selected_hk}")
    return selected


def _combine_profile_sets(
    sets: tuple[MosaicProfileSet, ...],
    profile_revision: str,
) -> MosaicProfileSet:
    if not sets or any(item.profile_revision != profile_revision for item in sets):
        raise ValueError("profile sets must share the frozen revision")
    source_revisions = {item.source_revision for item in sets}
    if len(source_revisions) != 1:
        raise ValueError("profile sets must share one source profile revision")
    execution_provenance = {(item.execution_backend, item.execution_device) for item in sets}
    if len(execution_provenance) != 1:
        raise ValueError("profile sets must share one execution backend and device")
    execution_backend, execution_device = next(iter(execution_provenance))
    return MosaicProfileSet(
        identities=tuple(identity for item in sets for identity in item.identities),
        signal=np.concatenate([item.signal for item in sets], axis=0),
        normalization=np.concatenate([item.normalization for item in sets], axis=0),
        valid=np.concatenate([item.valid for item in sets], axis=0),
        profile_revision=profile_revision,
        phi_bin_edges_rad=np.concatenate([item.phi_bin_edges_rad for item in sets], axis=0),
        two_theta_bounds_rad=np.concatenate(
            [item.two_theta_bounds_rad for item in sets],
            axis=0,
        ),
        angle_frame_revisions=tuple(
            revision for item in sets for revision in item.angle_frame_revisions
        ),
        source_revision=next(iter(source_revisions)),
        execution_backend=execution_backend,
        execution_device=execution_device,
    )


def _affine_profile_background_basis(
    observations: MosaicProfileSet,
    *,
    revision: str = "local-phi-constant-plus-linear.v1",
) -> MosaicProfileNuisanceBasis:
    centers = 0.5 * (observations.phi_bin_edges_rad[:, :-1] + observations.phi_bin_edges_rad[:, 1:])
    midpoint = 0.5 * (
        observations.phi_bin_edges_rad[:, :1] + observations.phi_bin_edges_rad[:, -1:]
    )
    half_span = 0.5 * (
        observations.phi_bin_edges_rad[:, -1:] - observations.phi_bin_edges_rad[:, :1]
    )
    coordinate = (centers - midpoint) / half_span
    basis = np.stack((np.ones(coordinate.shape), coordinate), axis=-1)
    basis[~observations.valid] = 0.0
    return MosaicProfileNuisanceBasis(
        identities=observations.identities,
        basis=basis,
        valid=observations.valid,
        revision=revision,
        profile_revision=observations.profile_revision,
    )


def _constant_profile_background_basis(
    observations: MosaicProfileSet,
) -> MosaicProfileNuisanceBasis:
    basis = np.ones((*observations.valid.shape, 1), dtype=np.float64)
    basis[~observations.valid] = 0.0
    return MosaicProfileNuisanceBasis(
        identities=observations.identities,
        basis=basis,
        valid=observations.valid,
        revision="local-phi-constant.v1",
        profile_revision=observations.profile_revision,
    )


_MEASURED_PROFILE_SELECTION_SAMPLER_REVISION = (
    "detector-native-bilinear-profile-centerline-sidebands.v1"
)


def _sample_profile_centerline_sidebands(
    *,
    detector_counts: NDArray[np.int32],
    detector_valid_mask: NDArray[np.bool_],
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    two_theta_bounds_rad: FloatArray,
    phi_bin_edges_rad: FloatArray,
    offsets_rad: tuple[float, ...],
) -> FloatArray:
    phi_center = 0.5 * (phi_bin_edges_rad[:, :-1] + phi_bin_edges_rad[:, 1:])
    theta_center = np.mean(two_theta_bounds_rad, axis=1)
    shape = (len(offsets_rad), theta_center.size, phi_center.shape[1])
    theta = (
        np.broadcast_to(theta_center[None, :, None], shape)
        + np.asarray(
            offsets_rad,
            dtype=np.float64,
        )[:, None, None]
    )
    phi = np.broadcast_to(phi_center[None, :, :], shape)
    coordinates = angles_to_detector_coordinates(
        theta,
        phi,
        instrument=instrument,
        angle_frame=angle_frame,
    )
    if not np.all(coordinates.valid):
        raise RuntimeError("measured profile sideband centerline leaves the active detector")
    sampled_valid = map_coordinates(
        detector_valid_mask,
        [coordinates.row_px, coordinates.column_px],
        order=0,
        mode="constant",
        cval=False,
        prefilter=False,
    )
    if not np.all(sampled_valid):
        raise RuntimeError("measured profile sideband centerline crosses masked detector pixels")
    sampled = map_coordinates(
        detector_counts,
        [coordinates.row_px, coordinates.column_px],
        output=np.float64,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    if not np.all(np.isfinite(sampled)) or np.any(sampled < 0.0):
        raise RuntimeError("measured profile sideband sampling produced invalid counts")
    return np.asarray(sampled, dtype=np.float64)


def _real_osc_profile_observations(
    *,
    case_path: Path,
    case: dict[str, Any],
    series: tuple[ConfiguredSimulationInputs, ...],
    frames: tuple[AngleFrame, ...],
    definitions: tuple[tuple[MosaicProfileDefinition, ...], ...],
    model_layout: MosaicProfileSet,
    profile_revision: str,
    measured_policy: _MeasuredProfilePolicy,
) -> tuple[
    MosaicProfileSet,
    MosaicProfileNuisanceBasis,
    tuple[dict[str, object], ...],
    tuple[tuple[MosaicProfileDefinition, ...], ...],
    tuple[dict[str, object], ...],
]:
    signal_parts: list[FloatArray] = []
    normalization_parts: list[FloatArray] = []
    audit: list[dict[str, object]] = []
    selection_audit: list[dict[str, object]] = []
    selected_definitions: list[tuple[MosaicProfileDefinition, ...]] = []
    selected_identities: list[MosaicProfileIdentity] = []
    selected_valid: list[NDArray[np.bool_]] = []
    selected_phi_edges: list[FloatArray] = []
    selected_theta_bounds: list[FloatArray] = []
    selected_frame_revisions: list[str] = []
    projector_keys: list[str] = []
    start = 0
    for inputs, frame, dataset_definitions, record in zip(
        series,
        frames,
        definitions,
        case["m0_observations"],
        strict=True,
    ):
        stop = start + len(dataset_definitions)
        model_valid = model_layout.valid[start:stop]
        model_signal = model_layout.signal[start:stop]
        phi_edges = model_layout.phi_bin_edges_rad[start:stop]
        theta_bounds = model_layout.two_theta_bounds_rad[start:stop]
        osc_path = (case_path.parent / str(record["osc_file"])).resolve()
        compressed_sha256 = _sha256(osc_path)
        if compressed_sha256 != record["osc_file_sha256"]:
            raise RuntimeError(f"OSC content changed: {osc_path}")
        counts = read_osc(osc_path).detector_native_counts
        expected_shape = tuple(int(value) for value in record["detector_native_shape_rc"])
        if counts.shape != expected_shape or str(counts.dtype) != record["detector_native_dtype"]:
            raise RuntimeError(f"OSC detector-native layout changed: {osc_path}")
        native_sha256 = _array_bytes_sha256(counts)
        if native_sha256 != record["detector_native_bytes_sha256"]:
            raise RuntimeError(f"OSC detector-native values changed: {osc_path}")
        detector_mask = detector_valid_mask_from_counts(counts)
        projector = compile_detector_profile_projector(
            instrument=inputs.instrument,
            angle_frame=frame,
            two_theta_bounds_rad=theta_bounds,
            phi_bin_edges_rad=phi_edges,
            detector_valid_mask=detector_mask,
            profile_bin_valid_mask=model_valid,
        )
        profiles = project_detector_profiles(projector, counts)
        if not np.array_equal(profiles.valid, model_valid):
            raise RuntimeError(
                f"OSC detector support changed the frozen model profile validity for "
                f"{record['dataset_id']}"
            )
        stacked_side = _sample_profile_centerline_sidebands(
            detector_counts=counts,
            detector_valid_mask=detector_mask,
            instrument=inputs.instrument,
            angle_frame=frame,
            two_theta_bounds_rad=theta_bounds,
            phi_bin_edges_rad=phi_edges,
            offsets_rad=measured_policy.sideband_two_theta_offsets_rad,
        )
        background = np.median(stacked_side, axis=0)
        side_scatter = 1.4826 * np.median(
            np.abs(stacked_side - background[None, ...]),
            axis=0,
        )
        central_intensity = profiles.I
        selected_local: list[int] = []
        for local_index, definition in enumerate(dataset_definitions):
            active = model_valid[local_index]
            positive_excess = np.maximum(
                central_intensity[local_index, active] - background[local_index, active],
                0.0,
            )
            noise = np.maximum(side_scatter[local_index, active], 1.0)
            significance = float(np.linalg.norm(positive_excess) / np.linalg.norm(noise))
            identity = definition.identity
            key = (
                identity.dataset_id,
                int(identity.group_key.layered_family_m),
                int(identity.group_key.layered_integer_L),
                identity.analytic_branch_id,
                identity.branch_id,
            )
            secondary_lobe = key in measured_policy.excluded_profile_keys
            weak = significance < measured_policy.minimum_excess_energy_over_side_scatter
            modeled_signal = float(np.sum(model_signal[local_index, active], dtype=np.float64))
            modeled_support = source_averaged_profile_has_support(
                family_m=key[1],
                profile_signal_A2=modeled_signal,
            )
            fitted = bool(modeled_support and not (secondary_lobe or weak))
            if fitted:
                selected_local.append(local_index)
            selection_audit.append(
                {
                    "dataset_id": identity.dataset_id,
                    "family_m": key[1],
                    "integer_L": key[2],
                    "analytic_branch_id": key[3],
                    "root_side_branch_id": key[4],
                    "excess_energy_over_side_scatter": significance,
                    "minimum_required": (measured_policy.minimum_excess_energy_over_side_scatter),
                    "source_averaged_modeled_signal_A2": modeled_signal,
                    "source_averaged_modeled_support": modeled_support,
                    "fit_eligible": fitted,
                    "classification": (
                        "SOURCE_AVERAGED_FORWARD_MODEL_UNSUPPORTED"
                        if not modeled_support
                        else "USER_AUTHORIZED_SECONDARY_LOBE"
                        if secondary_lobe
                        else "WEAK_LOCAL_PROFILE"
                        if weak
                        else "FITTED_SIGNIFICANT_PROFILE"
                    ),
                }
            )
        if not selected_local:
            raise RuntimeError(
                f"measured profile gate removed every {record['dataset_id']} profile"
            )
        local_indices = np.asarray(selected_local, dtype=np.int64)
        selected_dataset_definitions = tuple(dataset_definitions[index] for index in selected_local)
        selected_definitions.append(selected_dataset_definitions)
        selected_identities.extend(item.identity for item in selected_dataset_definitions)
        signal_parts.append(profiles.S[local_indices])
        normalization_parts.append(profiles.N[local_indices])
        selected_valid.extend(model_valid[local_indices])
        selected_phi_edges.extend(phi_edges[local_indices])
        selected_theta_bounds.extend(theta_bounds[local_indices])
        selected_frame_revisions.extend(frame.revision for _ in selected_local)
        projector_keys.append(projector.cache_key)
        audit.append(
            {
                "dataset_id": record["dataset_id"],
                "incidence_angle_deg": float(record["incidence_angle_deg"]),
                "osc_path": str(osc_path),
                "osc_file_sha256": compressed_sha256,
                "detector_native_bytes_sha256": native_sha256,
                "detector_native_shape_rc": list(counts.shape),
                "count_range": [int(np.min(counts)), int(np.max(counts))],
                "detector_valid_fraction": float(np.mean(detector_mask)),
                "candidate_profile_count": len(dataset_definitions),
                "fitted_profile_count": len(selected_local),
                "projector_cache_key": projector.cache_key,
                "profile_selection_sampler_revision": (
                    _MEASURED_PROFILE_SELECTION_SAMPLER_REVISION
                ),
                "projector_coverage_record_count": int(projector.weight.size),
                "profile_pixel_bounds_cr": projector.profile_pixel_bounds_cr.tolist(),
            }
        )
        start = stop
    if start != len(model_layout.identities):
        raise RuntimeError("OSC dataset partitions do not cover the model profile layout")
    observation_payload = {
        "contract": "three-osc-exact-cropped-profile-observation.v1",
        "detector_valid_mask_revision": DETECTOR_VALID_MASK_REVISION,
        "profile_revision": profile_revision,
        "measured_profile_policy_sha256": measured_policy.sha256,
        "selection_audit": selection_audit,
        "projector_cache_keys": projector_keys,
        "profile_selection_sampler_revision": (_MEASURED_PROFILE_SELECTION_SAMPLER_REVISION),
        "source_averaged_modeled_support_gate_revision": (
            SOURCE_AVERAGED_PROFILE_SUPPORT_GATE_REVISION
        ),
        "osc_file_sha256": [item["osc_file_sha256"] for item in audit],
        "detector_native_bytes_sha256": [item["detector_native_bytes_sha256"] for item in audit],
    }
    observation_revision = (
        "sha256-"
        + hashlib.sha256(
            json.dumps(observation_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    )
    observations = MosaicProfileSet(
        identities=tuple(selected_identities),
        signal=np.concatenate(signal_parts, axis=0),
        normalization=np.concatenate(normalization_parts, axis=0),
        valid=np.asarray(selected_valid, dtype=np.bool_),
        profile_revision=profile_revision,
        phi_bin_edges_rad=np.asarray(selected_phi_edges, dtype=np.float64),
        two_theta_bounds_rad=np.asarray(selected_theta_bounds, dtype=np.float64),
        angle_frame_revisions=tuple(selected_frame_revisions),
        source_revision=None,
        observation_revision=observation_revision,
    )
    return (
        observations,
        _constant_profile_background_basis(observations),
        tuple(audit),
        tuple(selected_definitions),
        tuple(selection_audit),
    )


def _deterministic_unknown_profile_intensity_scales(
    identities: tuple[MosaicProfileIdentity, ...],
    validation_config: dict[str, Any],
) -> dict[MosaicProfileIdentity, float]:
    lower, upper = (float(value) for value in validation_config["unknown_intensity_scale_bounds"])
    revision = str(validation_config["unknown_intensity_scale_revision"])
    result: dict[MosaicProfileIdentity, float] = {}
    for identity in identities:
        group = identity.group_key
        payload = json.dumps(
            {
                "revision": revision,
                "dataset_id": identity.dataset_id,
                "incidence_angle_rad": identity.incidence_angle_rad,
                "group_id": group.group_id,
                "rod_catalog_revision": group.rod_catalog_revision,
                "member_rod_hk": group.member_rod_hk,
                "branch_mode": group.branch_mode,
                "layered_family_m": group.layered_family_m,
                "layered_integer_L": group.layered_integer_L,
                "analytic_branch_id": identity.analytic_branch_id,
                "root_side_branch_id": identity.branch_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        unit_coordinate = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(
            (1 << 64) - 1
        )
        result[identity] = math.exp(
            math.log(lower) + unit_coordinate * (math.log(upper) - math.log(lower))
        )
    if len(result) < 2 or len(set(result.values())) != len(result):
        raise RuntimeError("deterministic nuisance intensity scales are not distinct")
    return result


def _evaluate_profile_series(
    physics: _ProfilePhysicsContext,
    geometry: tuple[_ProfileGeometryContext, ...],
    frames: tuple[AngleFrame, ...],
    definitions: tuple[tuple[MosaicProfileDefinition, ...], ...],
    mosaic: MosaicParameters,
    *,
    profile_revision: str,
    execution_backend: str,
) -> tuple[MosaicProfileSet, tuple[SourceAveragedDetectorEwaldMeasure, ...]]:
    detectors = _detector_series(
        physics,
        geometry,
        mosaic,
        rods=_profile_rods(physics.rods, definitions),
    )
    profiles = _evaluate_bound_profile_series(
        detectors,
        frames,
        definitions,
        profile_revision=profile_revision,
        execution_backend=execution_backend,
    )
    return profiles, detectors


def _evaluate_bound_profile_series(
    detectors: tuple[SourceAveragedDetectorEwaldMeasure, ...],
    frames: tuple[AngleFrame, ...],
    definitions: tuple[tuple[MosaicProfileDefinition, ...], ...],
    *,
    profile_revision: str,
    execution_backend: str,
) -> MosaicProfileSet:
    sets = tuple(
        evaluate_continuous_mosaic_profiles(
            detector,
            angle_frame=frame,
            definitions=dataset_definitions,
            profile_revision=profile_revision,
            execution_backend=execution_backend,
        )
        for detector, frame, dataset_definitions in zip(
            detectors,
            frames,
            definitions,
            strict=True,
        )
    )
    return _combine_profile_sets(sets, profile_revision)


def _component_profile_evaluators(
    *,
    physics: _ProfilePhysicsContext,
    geometry: tuple[_ProfileGeometryContext, ...],
    frames: tuple[AngleFrame, ...],
    definitions: tuple[tuple[MosaicProfileDefinition, ...], ...],
    profile_revision: str,
    execution_backend: str,
) -> tuple[Callable[[float], MosaicProfileSet], Callable[[float], MosaicProfileSet]]:
    backend = execution_backend
    detector_templates = _detector_series(
        physics,
        geometry,
        _mosaic_parameters(
            gaussian_sigma_rad=1.0,
            lorentzian_half_width_rad=1.0,
            lorentzian_probability=0.5,
            context=physics,
        ),
        rods=_profile_rods(physics.rods, definitions),
    )

    def evaluate(mosaic: MosaicParameters) -> MosaicProfileSet:
        return _evaluate_bound_profile_series(
            tuple(detector.rebind_physics(mosaic=mosaic) for detector in detector_templates),
            frames,
            definitions,
            profile_revision=profile_revision,
            execution_backend=backend,
        )

    def gaussian_profile(width_rad: float) -> MosaicProfileSet:
        return evaluate(
            _mosaic_parameters(
                gaussian_sigma_rad=width_rad,
                lorentzian_half_width_rad=1.0,
                lorentzian_probability=0.0,
                context=physics,
            )
        )

    def lorentzian_profile(width_rad: float) -> MosaicProfileSet:
        return evaluate(
            _mosaic_parameters(
                gaussian_sigma_rad=1.0,
                lorentzian_half_width_rad=width_rad,
                lorentzian_probability=1.0,
                context=physics,
            )
        )

    return gaussian_profile, lorentzian_profile


def _fit_profiles(
    *,
    observations: MosaicProfileSet,
    evaluate_gaussian_profile: Callable[[float], MosaicProfileSet],
    evaluate_lorentzian_profile: Callable[[float], MosaicProfileSet],
    search_config: dict[str, Any],
    nuisance_basis: MosaicProfileNuisanceBasis | None = None,
) -> tuple[MosaicProfileSearchResult, list[dict[str, object]]]:
    search = search_config

    search_result = fit_refined_mosaic_component_profiles(
        observations,
        evaluate_gaussian_profile=evaluate_gaussian_profile,
        evaluate_lorentzian_profile=evaluate_lorentzian_profile,
        gaussian_sigma_bounds_rad=np.radians(search["gaussian_sigma_bounds_deg"]),
        lorentzian_half_width_bounds_rad=np.radians(search["lorentzian_hwhm_bounds_deg"]),
        coarse_width_count=int(search["coarse_width_count"]),
        refinement_width_count=int(search["refinement_width_count"]),
        refinement_levels=int(search["refinement_levels"]),
        near_optimal_objective_delta=float(search["near_optimal_objective_delta"]),
        nuisance_basis=nuisance_basis,
        maximum_sensitivity_condition=float(search["maximum_sensitivity_condition"]),
    )
    history = [
        {
            "level": step.level,
            "gaussian_width_count": step.gaussian_width_count,
            "lorentzian_width_count": step.lorentzian_width_count,
            "retained_cell_count": step.retained_cell_count,
            "gaussian_sigma_deg": (
                None
                if step.best_gaussian_sigma_rad is None
                else math.degrees(step.best_gaussian_sigma_rad)
            ),
            "lorentzian_hwhm_deg": (
                None
                if step.best_lorentzian_half_width_rad is None
                else math.degrees(step.best_lorentzian_half_width_rad)
            ),
            "lorentzian_probability": step.best_lorentzian_probability,
            "objective": step.best_objective,
        }
        for step in search_result.refinement_steps
    ]
    return search_result, history


def _distribution_metrics(
    case: dict[str, Any],
    result: MosaicProfileFitResult,
) -> dict[str, object]:
    alpha = np.linspace(0.0, math.pi, 200_001)
    truth = case["truth"]
    truth_parameters = MosaicParameters(
        math.radians(float(truth["gaussian_sigma_deg"])),
        math.radians(float(truth["lorentzian_hwhm_deg"])),
        float(truth["lorentzian_probability"]),
    )
    if result.gaussian_sigma_rad is None or result.lorentzian_half_width_rad is None:
        raise RuntimeError("the prescribed interior truth unexpectedly selected a boundary model")
    recovered_parameters = MosaicParameters(
        result.gaussian_sigma_rad,
        result.lorentzian_half_width_rad,
        result.lorentzian_probability,
    )
    truth_density = 2.0 * wrapped_mosaic_line_density_rad_inv(alpha, truth_parameters)
    recovered_density = 2.0 * wrapped_mosaic_line_density_rad_inv(alpha, recovered_parameters)
    difference = recovered_density - truth_density
    total_variation = 0.5 * float(np.trapezoid(np.abs(difference), alpha))
    step = alpha[1] - alpha[0]
    truth_cdf = np.concatenate(
        ([0.0], np.cumsum(0.5 * (truth_density[:-1] + truth_density[1:]) * step))
    )
    recovered_cdf = np.concatenate(
        ([0.0], np.cumsum(0.5 * (recovered_density[:-1] + recovered_density[1:]) * step))
    )
    wasserstein_rad = float(np.trapezoid(np.abs(recovered_cdf - truth_cdf), alpha))

    def quantiles(cdf: FloatArray) -> dict[str, float]:
        return {
            f"q{int(probability * 100):02d}_deg": math.degrees(
                float(np.interp(probability, cdf, alpha))
            )
            for probability in (0.5, 0.9, 0.99)
        }

    truth_quantile = quantiles(truth_cdf)
    recovered_quantile = quantiles(recovered_cdf)
    return {
        "total_variation": total_variation,
        "wasserstein_1_deg": math.degrees(wasserstein_rad),
        "maximum_density_error_rad_inv": float(np.max(np.abs(difference))),
        "truth_quantiles": truth_quantile,
        "recovered_quantiles": recovered_quantile,
        "quantile_error_deg": {
            key: recovered_quantile[key] - truth_quantile[key] for key in truth_quantile
        },
        "alpha_rad": alpha,
        "truth_density_rad_inv": truth_density,
        "recovered_density_rad_inv": recovered_density,
    }


def _write_distribution_plot(
    output_directory: Path,
    metrics: dict[str, object],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path = output_directory / "mosaic_distribution_recovery.png"
    alpha_deg = np.degrees(metrics["alpha_rad"])
    figure, axis = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    axis.semilogy(alpha_deg, metrics["truth_density_rad_inv"], label="truth", linewidth=2.0)
    axis.semilogy(
        alpha_deg,
        metrics["recovered_density_rad_inv"],
        "--",
        label="recovered",
        linewidth=1.5,
    )
    axis.set_xlim(0.0, 35.0)
    axis.set_ylim(1.0e-5, None)
    axis.set_xlabel(r"folded mosaic tilt $\alpha$ (degree)")
    axis.set_ylabel(r"probability density (rad$^{-1}$)")
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _render_images(
    detectors: tuple[SourceAveragedDetectorEwaldMeasure, ...],
    incidences_deg: tuple[float, ...],
    *,
    output_directory: Path,
    render_config: dict[str, Any],
    execution_backend: str,
) -> tuple[
    tuple[FloatArray, ...],
    tuple[Path, ...],
    tuple[tuple[str, str | None], ...],
    float,
]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    start = perf_counter()
    images: list[FloatArray] = []
    execution_provenance: list[tuple[str, str | None]] = []
    for detector in detectors:
        rendered = integrate_detector_macrobins(
            detector,
            bin_size_px=1,
            gauss_order=int(render_config["pixel_gauss_order"]),
            execution_backend=execution_backend,
        )
        if rendered.image_A2.shape != (int(render_config["image_size"]),) * 2:
            raise RuntimeError("rendered detector image does not match the requested size")
        images.append(np.asarray(rendered.image_A2))
        execution_provenance.append((rendered.execution_backend, rendered.execution_device))
    high = max(float(np.max(image)) for image in images)
    if high <= 0.0:
        raise RuntimeError("rendered detector images contain no positive intensity")
    low = high * 1.0e-10
    paths: list[Path] = []
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad(cmap(0.0))
    for incidence_deg, image in zip(incidences_deg, images, strict=True):
        path = output_directory / f"bi2se3_{incidence_deg:g}deg_3000x3000.png"
        positive = image > 0.0
        display = np.zeros(image.shape, dtype=np.float32)
        np.log(image, out=display, where=positive)
        display -= math.log(low)
        display /= math.log(high / low)
        np.clip(display, 0.0, 1.0, out=display)
        plt.imsave(
            path,
            np.ma.array(display, mask=~positive),
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            origin="upper",
        )
        paths.append(path)
    return tuple(images), tuple(paths), tuple(execution_provenance), perf_counter() - start


def _fitted_distribution(result: MosaicProfileFitResult) -> tuple[FloatArray, FloatArray]:
    alpha = np.linspace(0.0, math.pi, 200_001)
    parameters = MosaicParameters(
        1.0 if result.gaussian_sigma_rad is None else result.gaussian_sigma_rad,
        1.0 if result.lorentzian_half_width_rad is None else result.lorentzian_half_width_rad,
        result.lorentzian_probability,
    )
    density = 2.0 * wrapped_mosaic_line_density_rad_inv(alpha, parameters)
    return alpha, density


def _parameter_summary(result: MosaicProfileFitResult) -> dict[str, float | None]:
    return {
        "gaussian_sigma_deg": (
            None if result.gaussian_sigma_rad is None else math.degrees(result.gaussian_sigma_rad)
        ),
        "gaussian_fwhm_deg": (
            None
            if result.gaussian_sigma_rad is None
            else math.degrees(result.gaussian_sigma_rad) * 2.0 * math.sqrt(2.0 * math.log(2.0))
        ),
        "lorentzian_hwhm_deg": (
            None
            if result.lorentzian_half_width_rad is None
            else math.degrees(result.lorentzian_half_width_rad)
        ),
        "lorentzian_fwhm_deg": (
            None
            if result.lorentzian_half_width_rad is None
            else 2.0 * math.degrees(result.lorentzian_half_width_rad)
        ),
        "lorentzian_probability": result.lorentzian_probability,
    }


def _run_real_osc_fit(
    *,
    case_path: Path,
    case: dict[str, Any],
    case_sha256: str,
    output_directory: Path,
    base: ConfiguredSimulationInputs,
    series: tuple[ConfiguredSimulationInputs, ...],
    profile_physics: _ProfilePhysicsContext,
    profile_geometry: tuple[_ProfileGeometryContext, ...],
    frames: tuple[AngleFrame, ...],
    definitions: tuple[tuple[MosaicProfileDefinition, ...], ...],
    profile_revision: str,
    profile_exclusions: tuple[dict[str, object], ...],
    m0_support_audits: tuple[dict[str, object], ...],
    measured_profile_policy_path: Path,
    measured_profile_policy: _MeasuredProfilePolicy,
    execution_backend: str,
    setup_seconds: float,
    total_start: float,
    fixed_geometry: dict[str, object],
) -> None:
    observation_start = perf_counter()
    model_layout, _ = _evaluate_profile_series(
        profile_physics,
        profile_geometry,
        frames,
        definitions,
        _mosaic_parameters(
            gaussian_sigma_rad=math.radians(
                float(case["validation"]["normalization_probe_gaussian_sigma_deg"])
            ),
            lorentzian_half_width_rad=1.0,
            lorentzian_probability=0.0,
            context=profile_physics,
        ),
        profile_revision=profile_revision,
        execution_backend=execution_backend,
    )
    (
        observations,
        nuisance_basis,
        osc_audit,
        fitted_definitions,
        measured_selection_audit,
    ) = _real_osc_profile_observations(
        case_path=case_path,
        case=case,
        series=series,
        frames=frames,
        definitions=definitions,
        model_layout=model_layout,
        profile_revision=profile_revision,
        measured_policy=measured_profile_policy,
    )
    observation_seconds = perf_counter() - observation_start

    gaussian_evaluator, lorentzian_evaluator = _component_profile_evaluators(
        physics=profile_physics,
        geometry=profile_geometry,
        frames=frames,
        definitions=fitted_definitions,
        profile_revision=profile_revision,
        execution_backend=execution_backend,
    )
    fit_start = perf_counter()
    search_result, refinement_history = _fit_profiles(
        observations=observations,
        evaluate_gaussian_profile=gaussian_evaluator,
        evaluate_lorentzian_profile=lorentzian_evaluator,
        search_config=case["search"],
        nuisance_basis=nuisance_basis,
    )
    result = search_result.fit
    fit_seconds = perf_counter() - fit_start

    robustness_start = perf_counter()
    affine_background: dict[str, object]
    try:
        affine_result = fit_mosaic_component_profiles(
            search_result.bank,
            nuisance_basis=_affine_profile_background_basis(observations),
            maximum_sensitivity_condition=float(case["search"]["maximum_sensitivity_condition"]),
        )
        affine_background = {
            "status": "FIT",
            "parameters": _parameter_summary(affine_result),
            "objective": affine_result.objective,
            "delta_from_constant": {
                key: (
                    None
                    if value is None or _parameter_summary(result)[key] is None
                    else value - float(_parameter_summary(result)[key])
                )
                for key, value in _parameter_summary(affine_result).items()
                if key
                in {
                    "gaussian_sigma_deg",
                    "lorentzian_hwhm_deg",
                    "lorentzian_probability",
                }
            },
        }
    except (MosaicIdentifiabilityError, ValueError) as error:
        affine_background = {
            "status": "NOT_IDENTIFIABLE",
            "reason": str(error),
        }
    robustness_seconds = perf_counter() - robustness_start

    m0_mask = np.asarray(
        [
            identity.group_key.branch_mode == "COLLAPSED_00L"
            for identity in result.profile_identities
        ],
        dtype=np.bool_,
    )
    fitted_m0_by_dataset = {
        dataset_id: sorted(
            int(identity.group_key.layered_integer_L)
            for identity in result.profile_identities
            if identity.dataset_id == dataset_id
            and identity.group_key.branch_mode == "COLLAPSED_00L"
        )
        for dataset_id in sorted({identity.dataset_id for identity in result.profile_identities})
    }
    candidate_m0_by_dataset = {
        str(item["dataset_id"]): set(
            item.get(
                "combined_source_gate_candidate_integer_L",
                item["forward_supported_candidate_integer_L"],
            )
        )
        for item in m0_support_audits
    }
    if any(
        not set(fitted).issubset(candidate_m0_by_dataset[dataset_id])
        for dataset_id, fitted in fitted_m0_by_dataset.items()
    ):
        raise RuntimeError("fitted m=0 identities escaped the combined-source candidate set")
    profile_records: list[dict[str, object]] = []
    background_prediction = result.predicted_total_intensity - result.predicted_intensity
    for index, identity in enumerate(result.profile_identities):
        active = observations.valid[index]
        observed = observations.intensity[index, active]
        total_prediction = result.predicted_total_intensity[index, active]
        group = identity.group_key
        profile_records.append(
            {
                "profile_index": index,
                "dataset_id": identity.dataset_id,
                "group_id": group.group_id,
                "family_m": group.layered_family_m,
                "integer_L": group.layered_integer_L,
                "analytic_branch_id": identity.analytic_branch_id,
                "root_side_branch_id": identity.branch_id,
                "valid_bin_count": int(np.count_nonzero(active)),
                "relative_projected_shape_residual": float(
                    result.profile_relative_l2_residual[index]
                ),
                "nuisance_peak_scale": float(result.profile_scales[index]),
                "signed_background_coefficients": result.background_coefficients[index].tolist(),
                "observed_count_density_range": [
                    float(np.min(observed)),
                    float(np.max(observed)),
                ],
                "peak_prediction_range": [
                    float(np.min(result.predicted_intensity[index, active])),
                    float(np.max(result.predicted_intensity[index, active])),
                ],
                "total_prediction_rmse_counts_per_px": float(
                    np.sqrt(np.mean((total_prediction - observed) ** 2))
                ),
            }
        )

    zero_scale_profile_indices = np.flatnonzero(result.profile_scales <= 0.0)
    high_residual_profile_indices = np.flatnonzero(result.profile_relative_l2_residual >= 0.65)
    active_total_prediction = result.predicted_total_intensity[observations.valid]
    negative_total_prediction_bin_count = int(np.count_nonzero(active_total_prediction < 0.0))

    def residual_summary(mask: NDArray[np.bool_]) -> dict[str, object]:
        selected = np.flatnonzero(mask)
        residual = result.profile_relative_l2_residual[selected]
        worst = int(selected[int(np.argmax(residual))])
        return {
            "count": int(selected.size),
            "median": float(np.median(residual)),
            "root_mean_square": float(np.sqrt(np.mean(residual * residual))),
            "maximum": float(result.profile_relative_l2_residual[worst]),
            "worst_profile_index": worst,
        }

    alpha, density = _fitted_distribution(result)
    component_execution = {
        (component.profile.execution_backend, component.profile.execution_device)
        for component in (
            *search_result.bank.gaussian_profiles,
            *search_result.bank.lorentzian_profiles,
        )
    }
    if len(component_execution) != 1:
        raise RuntimeError("mosaic component bank changed execution backend or device")
    component_execution_backend, component_execution_device = next(iter(component_execution))
    diagnostic_path = output_directory / "bi2se3_real_mosaic_fit.ra_diag.npz"
    np.savez_compressed(
        diagnostic_path,
        profile_observed_signal=observations.signal,
        profile_observed_normalization=observations.normalization,
        profile_observed_intensity=observations.intensity,
        profile_valid=observations.valid,
        profile_phi_bin_edges_rad=observations.phi_bin_edges_rad,
        profile_two_theta_bounds_rad=observations.two_theta_bounds_rad,
        profile_predicted_peak_intensity=result.predicted_intensity,
        profile_predicted_background_intensity=background_prediction,
        profile_predicted_total_intensity=result.predicted_total_intensity,
        profile_total_residual_intensity=(
            observations.intensity - result.predicted_total_intensity
        ),
        profile_peak_scales=result.profile_scales,
        profile_background_coefficients=result.background_coefficients,
        profile_relative_projected_shape_residual=result.profile_relative_l2_residual,
        width_pair_objective=result.width_pair_objective,
        width_pair_eta=result.width_pair_eta,
        distribution_alpha_rad=alpha,
        distribution_density_rad_inv=density,
    )
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    manifest = {
        "schema_version": "rasim-bi2se3-real-mosaic-fit-v3",
        "status": "MODEL_LIMITED_EFFECTIVE_RADIAL_MOSAIC_ESTIMATE",
        "legacy_classification": "NO_ORACLE",
        "interpretation": (
            "A common axisymmetric probability mixture of one wrapped Gaussian and one wrapped "
            "Lorentzian, folded through the fixed nine-coordinate geometry. It is not a Voigt "
            "convolution. Without a calibrated source/detector resolution function, this "
            "effective radial envelope is not uniquely attributable to intrinsic sample mosaic."
        ),
        "recovered_effective_distribution": _parameter_summary(result),
        "fit": {
            "objective": result.objective,
            "weighting_id": result.weighting_id,
            "eta_search_id": result.eta_search_id,
            "nuisance_basis_revision": result.nuisance_basis_revision,
            "active_parameter_names": list(result.active_parameter_names),
            "sensitivity_rank": result.sensitivity_rank,
            "sensitivity_condition": result.sensitivity_condition,
            "sensitivity_singular_values": result.sensitivity_singular_values.tolist(),
            "profile_count": len(result.profile_identities),
            "m0_profile_count": int(np.count_nonzero(m0_mask)),
            "nonzero_m_profile_count": int(np.count_nonzero(~m0_mask)),
            "residual_summary": {
                "all": residual_summary(np.ones(m0_mask.shape, dtype=np.bool_)),
                "m0": residual_summary(m0_mask),
                "nonzero_m": residual_summary(~m0_mask),
                "by_dataset": {
                    dataset_id: residual_summary(
                        np.asarray(
                            [
                                identity.dataset_id == dataset_id
                                for identity in result.profile_identities
                            ],
                            dtype=np.bool_,
                        )
                    )
                    for dataset_id in sorted(
                        {identity.dataset_id for identity in result.profile_identities}
                    )
                },
            },
            "profiles": profile_records,
            "refinement_history": refinement_history,
            "adequacy": {
                "all_profiles_have_positive_fitted_amplitude": bool(
                    zero_scale_profile_indices.size == 0
                ),
                "zero_amplitude_profile_indices": zero_scale_profile_indices.tolist(),
                "profile_residual_at_least_0p65_indices": (high_residual_profile_indices.tolist()),
                "signed_detrended_total_prediction_nonnegative": bool(
                    negative_total_prediction_bin_count == 0
                ),
                "negative_signed_detrended_total_bin_count": (negative_total_prediction_bin_count),
                "minimum_signed_detrended_total_counts_per_px": float(
                    np.min(active_total_prediction)
                ),
                "model_limitations_present": True,
            },
        },
        "robustness": {
            "affine_instead_of_constant_background": affine_background,
            "scope": (
                "Deterministic model-form sensitivity only; no statistical confidence interval "
                "is claimed because OSC covariance and instrument PSF are not calibrated."
            ),
        },
        "fixed_geometry": fixed_geometry,
        "observations": {
            "joint_dataset_count": len(osc_audit),
            "datasets": list(osc_audit),
            "observation_revision": observations.observation_revision,
            "profile_revision": profile_revision,
            "detector_valid_mask_revision": DETECTOR_VALID_MASK_REVISION,
            "intensity_policy": (
                "one independent nonnegative amplitude per profile; equal normalized shape "
                "leverage; one signed constant offset projected before fitting"
            ),
            "candidate_profile_count": len(model_layout.identities),
            "fitted_profile_count": len(observations.identities),
            "measured_profile_policy": {
                "path": str(measured_profile_policy_path),
                "sha256": measured_profile_policy.sha256,
                "minimum_excess_energy_over_side_scatter": (
                    measured_profile_policy.minimum_excess_energy_over_side_scatter
                ),
                "sideband_two_theta_offsets_deg": [
                    math.degrees(value)
                    for value in measured_profile_policy.sideband_two_theta_offsets_rad
                ],
                "selection_sampler_revision": (_MEASURED_PROFILE_SELECTION_SAMPLER_REVISION),
                "source_averaged_modeled_support_gate_revision": (
                    SOURCE_AVERAGED_PROFILE_SUPPORT_GATE_REVISION
                ),
                "selection_measure": (
                    "Selection-only detector-native bilinear centerline samples; the fitted "
                    "central profiles use exact physical-pixel polygon overlap."
                ),
                "explicitly_excluded_profiles": list(measured_profile_policy.excluded_profiles),
                "profile_selection": list(measured_selection_audit),
                "scope": (
                    "Entire weak profiles and both user-authorized 10-degree secondary-lobe "
                    "profiles are excluded before fitting; no profile is partially trimmed."
                ),
            },
            "excluded_profile_atoms": list(profile_exclusions),
            "m0_support_audits": list(m0_support_audits),
            "fitted_m0_integer_L_by_dataset": fitted_m0_by_dataset,
            "unsupported_observed_peaks": [
                {
                    "dataset_id": record["dataset_id"],
                    "peaks": record["unsupported_observed_peaks"],
                }
                for record in case["m0_observations"]
            ],
        },
        "source_model": _source_model_record(base),
        "simulation_execution": {
            "backend": component_execution_backend,
            "device": component_execution_device,
        },
        "provenance": {
            "case_sha256": case_sha256,
            "cif_sha256": base.config.cif_sha256,
            "physics_revision": base.config.physics_revision,
            "rod_catalog_revision": configured_rod_catalog_revision(base),
        },
        "artifact": str(diagnostic_path),
        "timing_seconds": {
            "setup": setup_seconds,
            "osc_projection_and_model_layout": observation_seconds,
            "component_bank_and_constant_background_fit": fit_seconds,
            "affine_background_robustness": robustness_seconds,
            "total": perf_counter() - total_start,
        },
        "python_tracemalloc": {
            "current_bytes": current_bytes,
            "peak_bytes": peak_bytes,
        },
    }
    manifest_path = output_directory / "bi2se3_real_mosaic_fit.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tracemalloc.stop()
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "recovered_effective_distribution": manifest["recovered_effective_distribution"],
                "fit": {
                    key: manifest["fit"][key]
                    for key in (
                        "objective",
                        "sensitivity_rank",
                        "sensitivity_condition",
                        "profile_count",
                        "m0_profile_count",
                        "nonzero_m_profile_count",
                        "residual_summary",
                    )
                },
                "robustness": manifest["robustness"],
                "artifact": manifest["artifact"],
                "manifest": str(manifest_path),
                "timing_seconds": manifest["timing_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit a fixed-geometry 5/10/15-degree Bi2Se3 mosaic triplet."
    )
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--observation-mode",
        choices=("synthetic", "osc"),
        default="synthetic",
        help="fit the planted synthetic proof or the three measured OSC images",
    )
    parser.add_argument(
        "--measured-profile-policy",
        type=Path,
        default=DEFAULT_MEASURED_PROFILE_POLICY,
        help="case-bound explicit measured-profile exclusions used only in OSC mode",
    )
    parser.add_argument(
        "--source-sample-count",
        type=int,
        default=250,
        help="number of deterministic source states summed into each detector function",
    )
    parser.add_argument(
        "--execution-backend",
        choices=("cpu", "cuda"),
        help="override the configured profile and render backend",
    )
    parser.add_argument(
        "--position-artifact",
        type=Path,
        help="verified geometry.json from the completed upstream position fit",
    )
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args(argv)
    case_path = args.case.resolve()
    case, case_bytes, case_sha256 = _case(case_path)
    if args.observation_mode == "osc" and args.position_artifact is None:
        raise ValueError("measured OSC mosaic fitting requires --position-artifact")
    execution_backend = args.execution_backend or str(case["profiles"]["execution_backend"])
    tracemalloc.start()
    total_start = perf_counter()

    setup_start = perf_counter()
    position = (
        _case_fixed_position_state(case)
        if args.position_artifact is None
        else _position_artifact_state(args.position_artifact, case)
    )
    output_directory = _external_directory(args.output_directory)
    base, series, nominal_series = _fixed_geometry_inputs(
        case_path,
        case,
        source_sample_count=args.source_sample_count,
        position=position,
    )
    fixed_geometry = _fixed_geometry_record(
        case,
        base,
        position,
    )
    profile_physics, profile_geometry = _profile_forward_contexts(base, series)
    rod_catalog_revision = configured_rod_catalog_revision(base)
    profile_pairs = tuple(
        _profile_definitions(
            inputs,
            source_inputs=source_inputs,
            include_nominally_unsupported_m0=args.observation_mode == "osc",
            case_path=case_path,
            incidence_deg=float(incidence_deg),
            osc_observation=osc_observation,
            nonzero_centroid_provenance=case["nonzero_centroid_provenance"],
            centroid_provenance=case["m0_centroid_provenance"],
            profile_config=case["profiles"],
            rod_catalog_revision=rod_catalog_revision,
        )
        for inputs, source_inputs, incidence_deg, osc_observation in zip(
            nominal_series,
            series,
            case["incidence_angles_deg"],
            case["m0_observations"],
            strict=True,
        )
    )
    frames = tuple(item[0] for item in profile_pairs)
    definitions = tuple(item[1] for item in profile_pairs)
    measured_profile_policy_path = args.measured_profile_policy.resolve()
    measured_profile_policy: _MeasuredProfilePolicy | None = None
    if args.observation_mode == "osc":
        measured_profile_policy = _measured_profile_policy(
            measured_profile_policy_path,
            case_sha256=case_sha256,
            definitions=definitions,
        )
    truth_definitions = tuple(
        tuple(
            replace(
                definition,
                two_theta_gauss_order=int(
                    case["validation"][
                        (
                            "m0_truth_two_theta_gauss_order"
                            if definition.identity.group_key.branch_mode == "COLLAPSED_00L"
                            else "truth_two_theta_gauss_order"
                        )
                    ]
                ),
                phi_gauss_order=int(
                    case["validation"][
                        (
                            "m0_truth_phi_gauss_order"
                            if definition.identity.group_key.branch_mode == "COLLAPSED_00L"
                            else "truth_phi_gauss_order"
                        )
                    ]
                ),
            )
            for definition in dataset
        )
        for dataset in definitions
    )
    profile_exclusions = tuple(exclusion for item in profile_pairs for exclusion in item[2])
    m0_support_audits = tuple(item[3] for item in profile_pairs)
    simulation_config_path = (case_path.parent / str(case["simulation_config"])).resolve()
    simulation_config_sha256 = _sha256(simulation_config_path)
    profile_revision = _profile_revision(
        definitions,
        frames,
        profile_exclusions,
        m0_support_audits,
        case,
        simulation_config_sha256,
        base.config.physics_revision,
        base.config.cif_sha256,
        rod_catalog_revision,
        base.samples.source_revision,
        fixed_geometry,
        None if measured_profile_policy is None else measured_profile_policy.sha256,
    )
    truth_profile_revision = _profile_revision(
        truth_definitions,
        frames,
        profile_exclusions,
        m0_support_audits,
        case,
        simulation_config_sha256,
        base.config.physics_revision,
        base.config.cif_sha256,
        rod_catalog_revision,
        base.samples.source_revision,
        fixed_geometry,
    )
    setup_seconds = perf_counter() - setup_start

    if args.observation_mode == "osc":
        if measured_profile_policy is None:
            raise RuntimeError("OSC mode lacks a measured profile policy")
        _run_real_osc_fit(
            case_path=case_path,
            case=case,
            case_sha256=case_sha256,
            output_directory=output_directory,
            base=base,
            series=series,
            profile_physics=profile_physics,
            profile_geometry=profile_geometry,
            frames=frames,
            definitions=definitions,
            profile_revision=profile_revision,
            profile_exclusions=profile_exclusions,
            m0_support_audits=m0_support_audits,
            measured_profile_policy_path=measured_profile_policy_path,
            measured_profile_policy=measured_profile_policy,
            execution_backend=execution_backend,
            setup_seconds=setup_seconds,
            total_start=total_start,
            fixed_geometry=fixed_geometry,
        )
        return

    truth = case["truth"]
    truth_mosaic = _mosaic_parameters(
        gaussian_sigma_rad=math.radians(float(truth["gaussian_sigma_deg"])),
        lorentzian_half_width_rad=math.radians(float(truth["lorentzian_hwhm_deg"])),
        lorentzian_probability=float(truth["lorentzian_probability"]),
        context=profile_physics,
    )
    truth_start = perf_counter()
    independent_truth_profiles, truth_detectors = _evaluate_profile_series(
        profile_physics,
        profile_geometry,
        frames,
        truth_definitions,
        truth_mosaic,
        profile_revision=truth_profile_revision,
        execution_backend=execution_backend,
    )
    m0_profile_mask = np.asarray(
        [
            identity.group_key.branch_mode == "COLLAPSED_00L"
            for identity in independent_truth_profiles.identities
        ],
        dtype=np.bool_,
    )
    positive_truth_signal = np.any(
        independent_truth_profiles.valid & (independent_truth_profiles.signal > 0.0),
        axis=1,
    )
    if not np.all(positive_truth_signal[m0_profile_mask]):
        missing = tuple(
            independent_truth_profiles.identities[index].group_key.group_id
            for index in np.flatnonzero(m0_profile_mask & ~positive_truth_signal)
        )
        raise RuntimeError(f"detector-visible m=0 profiles lack positive truth signal: {missing}")
    normalization_probe, _ = _evaluate_profile_series(
        profile_physics,
        profile_geometry,
        frames,
        definitions,
        _mosaic_parameters(
            gaussian_sigma_rad=math.radians(
                float(case["validation"]["normalization_probe_gaussian_sigma_deg"])
            ),
            lorentzian_half_width_rad=1.0,
            lorentzian_probability=0.0,
            context=profile_physics,
        ),
        profile_revision=profile_revision,
        execution_backend=execution_backend,
    )
    common_valid = independent_truth_profiles.valid & normalization_probe.valid
    if independent_truth_profiles.source_revision != normalization_probe.source_revision:
        raise RuntimeError("truth and fit quadrature changed the configured source realization")
    planted_profile_scales = _deterministic_unknown_profile_intensity_scales(
        normalization_probe.identities,
        case["validation"],
    )
    planted_profile_scale = np.asarray(
        [planted_profile_scales[identity] for identity in normalization_probe.identities]
    )
    observed_signal = np.zeros(normalization_probe.signal.shape, dtype=np.float64)
    observed_normalization = np.zeros(normalization_probe.normalization.shape, dtype=np.float64)
    observed_normalization[common_valid] = normalization_probe.normalization[common_valid]
    observed_signal[common_valid] = (
        independent_truth_profiles.intensity * planted_profile_scale[:, None]
    )[common_valid] * observed_normalization[common_valid]
    observations = MosaicProfileSet(
        identities=normalization_probe.identities,
        signal=observed_signal,
        normalization=observed_normalization,
        valid=common_valid,
        profile_revision=profile_revision,
        phi_bin_edges_rad=normalization_probe.phi_bin_edges_rad,
        two_theta_bounds_rad=normalization_probe.two_theta_bounds_rad,
        angle_frame_revisions=normalization_probe.angle_frame_revisions,
        source_revision=normalization_probe.source_revision,
        execution_backend=normalization_probe.execution_backend,
        execution_device=normalization_probe.execution_device,
    )
    truth_profile_seconds = perf_counter() - truth_start

    gaussian_evaluator, lorentzian_evaluator = _component_profile_evaluators(
        physics=profile_physics,
        geometry=profile_geometry,
        frames=frames,
        definitions=definitions,
        profile_revision=profile_revision,
        execution_backend=execution_backend,
    )
    fit_start = perf_counter()
    search_result, refinement_history = _fit_profiles(
        observations=observations,
        evaluate_gaussian_profile=gaussian_evaluator,
        evaluate_lorentzian_profile=lorentzian_evaluator,
        search_config=case["search"],
    )
    result = search_result.fit
    fit_seconds = perf_counter() - fit_start
    normalization_invariance_error = max(
        float(np.max(np.abs(component.profile.normalization - observations.normalization)))
        for component in (
            *search_result.bank.gaussian_profiles,
            *search_result.bank.lorentzian_profiles,
        )
    )

    component_truth_start = perf_counter()
    exact_gaussian, _ = _evaluate_profile_series(
        profile_physics,
        profile_geometry,
        frames,
        truth_definitions,
        _mosaic_parameters(
            gaussian_sigma_rad=math.radians(float(truth["gaussian_sigma_deg"])),
            lorentzian_half_width_rad=1.0,
            lorentzian_probability=0.0,
            context=profile_physics,
        ),
        profile_revision=truth_profile_revision,
        execution_backend=execution_backend,
    )
    exact_lorentzian, _ = _evaluate_profile_series(
        profile_physics,
        profile_geometry,
        frames,
        truth_definitions,
        _mosaic_parameters(
            gaussian_sigma_rad=1.0,
            lorentzian_half_width_rad=math.radians(float(truth["lorentzian_hwhm_deg"])),
            lorentzian_probability=1.0,
            context=profile_physics,
        ),
        profile_revision=truth_profile_revision,
        execution_backend=execution_backend,
    )
    truth_component_seconds = perf_counter() - component_truth_start
    eta_truth = float(truth["lorentzian_probability"])
    direct_component_signal = (
        1.0 - eta_truth
    ) * exact_gaussian.signal + eta_truth * exact_lorentzian.signal
    component_difference = direct_component_signal - independent_truth_profiles.signal
    maximum_truth_signal = float(np.max(independent_truth_profiles.signal))
    decomposition_relative_error = float(
        np.max(np.abs(component_difference)) / maximum_truth_signal
    )

    distribution = _distribution_metrics(case, result)
    distribution_path: Path | None = None
    images: tuple[FloatArray, ...] = ()
    image_paths: tuple[Path, ...] = ()
    render_execution: tuple[tuple[str, str | None], ...] = ()
    render_seconds = 0.0
    if not args.skip_images:
        distribution_path = _write_distribution_plot(output_directory, distribution)
        images, image_paths, render_execution, render_seconds = _render_images(
            truth_detectors,
            tuple(float(value) for value in fixed_geometry["effective_incidence_angles_deg"]),
            output_directory=output_directory,
            render_config=case["render"],
            execution_backend=execution_backend,
        )

    diagnostic_path = output_directory / "bi2se3_mosaic_recovery.ra_diag.npz"
    arrays: dict[str, object] = {
        "profile_observed_signal": observations.signal,
        "profile_observed_intensity": observations.intensity,
        "profile_normalization": observations.normalization,
        "profile_valid": observations.valid,
        "profile_planted_unknown_intensity_scale": planted_profile_scale,
        "independent_truth_profile_signal": independent_truth_profiles.signal,
        "independent_truth_profile_intensity": independent_truth_profiles.intensity,
        "independent_truth_profile_normalization": independent_truth_profiles.normalization,
        "profile_predicted_intensity": result.predicted_intensity,
        "profile_relative_l2_residual": result.profile_relative_l2_residual,
        "width_pair_objective": result.width_pair_objective,
        "width_pair_eta": result.width_pair_eta,
        "distribution_alpha_rad": distribution["alpha_rad"],
        "distribution_truth_density_rad_inv": distribution["truth_density_rad_inv"],
        "distribution_recovered_density_rad_inv": distribution["recovered_density_rad_inv"],
    }
    if images:
        arrays.update(
            {
                f"detector_{incidence_deg:g}deg_A2": image
                for incidence_deg, image in zip(
                    case["incidence_angles_deg"],
                    images,
                    strict=True,
                )
            }
        )
    np.savez_compressed(diagnostic_path, **arrays)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    compact_distribution = {
        key: value
        for key, value in distribution.items()
        if key
        not in {
            "alpha_rad",
            "truth_density_rad_inv",
            "recovered_density_rad_inv",
        }
    }
    truth_values = {
        "gaussian_sigma_deg": float(truth["gaussian_sigma_deg"]),
        "lorentzian_hwhm_deg": float(truth["lorentzian_hwhm_deg"]),
        "lorentzian_probability": eta_truth,
    }
    recovered_values = {
        "gaussian_sigma_deg": math.degrees(result.gaussian_sigma_rad),
        "lorentzian_hwhm_deg": math.degrees(result.lorentzian_half_width_rad),
        "lorentzian_probability": result.lorentzian_probability,
    }
    parameter_error = {key: recovered_values[key] - truth_values[key] for key in truth_values}
    profile_residual = result.profile_relative_l2_residual
    m0_result_mask = np.asarray(
        [
            identity.group_key.branch_mode == "COLLAPSED_00L"
            for identity in result.profile_identities
        ],
        dtype=np.bool_,
    )

    def profile_residual_summary(indices: NDArray[np.bool_]) -> dict[str, object]:
        selected_indices = np.flatnonzero(indices)
        if not selected_indices.size:
            raise RuntimeError("profile residual summary requires at least one profile")
        selected_residual = profile_residual[selected_indices]
        worst_index = int(selected_indices[int(np.argmax(selected_residual))])
        worst_identity = result.profile_identities[worst_index]
        return {
            "count": int(selected_indices.size),
            "maximum": float(profile_residual[worst_index]),
            "median": float(np.median(selected_residual)),
            "root_mean_square": float(np.sqrt(np.mean(selected_residual**2))),
            "worst_profile": {
                "dataset_id": worst_identity.dataset_id,
                "group_id": worst_identity.group_key.group_id,
                "layered_family_m": worst_identity.group_key.layered_family_m,
                "layered_integer_L": worst_identity.group_key.layered_integer_L,
                "analytic_branch_id": worst_identity.analytic_branch_id,
                "root_side_branch_id": worst_identity.branch_id,
            },
        }

    all_profile_mask = np.ones(profile_residual.shape, dtype=np.bool_)
    profile_shape_residual = {
        "definition": "sqrt(sum_valid((prediction-observation)^2)/sum_valid(observation^2))",
        "all": profile_residual_summary(all_profile_mask),
        "m0": profile_residual_summary(m0_result_mask),
        "nonzero_m": profile_residual_summary(~m0_result_mask),
        "by_dataset": {
            dataset_id: profile_residual_summary(
                np.asarray(
                    [identity.dataset_id == dataset_id for identity in result.profile_identities],
                    dtype=np.bool_,
                )
            )
            for dataset_id in sorted(
                {identity.dataset_id for identity in result.profile_identities}
            )
        },
    }
    acceptance_config = case["acceptance"]
    acceptance_checks = {
        "gaussian_sigma": abs(parameter_error["gaussian_sigma_deg"])
        <= float(acceptance_config["maximum_gaussian_sigma_error_deg"]),
        "lorentzian_hwhm": abs(parameter_error["lorentzian_hwhm_deg"])
        <= float(acceptance_config["maximum_lorentzian_hwhm_error_deg"]),
        "lorentzian_probability": abs(parameter_error["lorentzian_probability"])
        <= float(acceptance_config["maximum_lorentzian_probability_error"]),
        "total_variation": float(distribution["total_variation"])
        <= float(acceptance_config["maximum_total_variation"]),
        "wasserstein_1": float(distribution["wasserstein_1_deg"])
        <= float(acceptance_config["maximum_wasserstein_1_deg"]),
        "component_decomposition": decomposition_relative_error
        <= float(acceptance_config["maximum_component_decomposition_relative_error"]),
        "normalization_invariance": normalization_invariance_error == 0.0,
        "sensitivity": result.sensitivity_rank == len(result.active_parameter_names)
        and result.sensitivity_condition
        <= float(acceptance_config["maximum_sensitivity_condition"]),
        "every_profile_shape": float(np.max(profile_residual))
        <= float(acceptance_config["maximum_profile_relative_l2_residual"]),
    }
    accepted = all(acceptance_checks.values())
    if case_path.read_bytes() != case_bytes:
        raise RuntimeError("the mosaic recovery case changed while the proof was running")
    definition_by_identity = {
        definition.identity: definition
        for dataset_definitions in definitions
        for definition in dataset_definitions
    }
    try:
        case_repository_path = case_path.relative_to(ROOT).as_posix()
    except ValueError:
        case_repository_path = None
    manifest = {
        "schema_version": "rasim-mosaic-recovery-result-v3",
        "case": {
            "repository_path": case_repository_path,
            "resolved_path": str(case_path),
            "sha256": case_sha256,
        },
        "status": ("ACCEPTED_IDENTIFIABLE_RECOVERY" if accepted else "FAILED_ACCEPTANCE_GATE"),
        "truth": truth_values,
        "recovered": recovered_values,
        "error": parameter_error,
        "acceptance": {
            "accepted": accepted,
            "checks": acceptance_checks,
            "thresholds": acceptance_config,
        },
        "distribution": compact_distribution,
        "fit": {
            "objective": result.objective,
            "weighting_id": result.weighting_id,
            "eta_search_id": result.eta_search_id,
            "active_parameter_names": list(result.active_parameter_names),
            "sensitivity_rank": result.sensitivity_rank,
            "sensitivity_condition": result.sensitivity_condition,
            "sensitivity_singular_values": result.sensitivity_singular_values.tolist(),
            "profile_shape_residual": profile_shape_residual,
            "reflection_group_count": len(
                {identity.group_key for identity in result.profile_identities}
            ),
            "nuisance_profile_scale_count": len(result.profile_scales),
            "nuisance_profile_scales": [
                {
                    "dataset_id": identity.dataset_id,
                    "incidence_angle_rad": identity.incidence_angle_rad,
                    "analytic_branch_id": identity.analytic_branch_id,
                    "root_side_branch_id": identity.branch_id,
                    "group_id": identity.group_key.group_id,
                    "rod_catalog_revision": identity.group_key.rod_catalog_revision,
                    "member_rod_hk": [list(rod_hk) for rod_hk in identity.group_key.member_rod_hk],
                    "branch_mode": identity.group_key.branch_mode,
                    "layered_family_m": identity.group_key.layered_family_m,
                    "layered_integer_L": identity.group_key.layered_integer_L,
                    "planted_unknown_intensity_scale": planted_profile_scales[identity],
                    "nuisance_intensity_scale": float(scale),
                    "recovered_to_planted_scale_ratio": (
                        float(scale) / planted_profile_scales[identity]
                    ),
                }
                for identity, scale in zip(
                    result.profile_identities,
                    result.profile_scales,
                    strict=True,
                )
            ],
            "profile_count": len(observations.identities),
            "unknown_intensity_scale_revision": str(
                case["validation"]["unknown_intensity_scale_revision"]
            ),
            "planted_unknown_intensity_scale_range": [
                min(planted_profile_scales.values()),
                max(planted_profile_scales.values()),
            ],
            "profile_layout": [
                {
                    "profile_index": index,
                    "dataset_id": identity.dataset_id,
                    "incidence_angle_rad": identity.incidence_angle_rad,
                    "group_id": identity.group_key.group_id,
                    "member_rod_hk": [list(rod_hk) for rod_hk in identity.group_key.member_rod_hk],
                    "branch_mode": identity.group_key.branch_mode,
                    "analytic_branch_id": identity.analytic_branch_id,
                    "root_side_branch_id": identity.branch_id,
                    "excluded_phi_bin_indices": list(
                        definition_by_identity[identity].excluded_phi_bin_indices
                    ),
                    "two_theta_bounds_rad": two_theta_bounds.tolist(),
                    "phi_bin_edges_rad": phi_edges.tolist(),
                    "angle_frame_revision": frame_revision,
                    "relative_l2_residual": float(relative_l2_residual),
                }
                for index, (
                    identity,
                    two_theta_bounds,
                    phi_edges,
                    frame_revision,
                    relative_l2_residual,
                ) in enumerate(
                    zip(
                        observations.identities,
                        observations.two_theta_bounds_rad,
                        observations.phi_bin_edges_rad,
                        observations.angle_frame_revisions,
                        result.profile_relative_l2_residual,
                        strict=True,
                    )
                )
            ],
            "m0_profile_count": sum(
                identity.group_key.branch_mode == "COLLAPSED_00L"
                for identity in observations.identities
            ),
            "nonzero_profile_count": sum(
                identity.group_key.branch_mode == "EXPLICIT_NONZERO"
                for identity in observations.identities
            ),
            "excluded_profile_atoms": list(profile_exclusions),
            "m0_support_audits": list(m0_support_audits),
            "profile_revision": profile_revision,
            "independent_truth_profile_revision": truth_profile_revision,
            "observation_source_revision": observations.source_revision,
            "profile_quadrature": {
                "excluded_bin_policy": str(case["profiles"]["excluded_bin_policy"]),
                "excluded_bin_topology": str(case["profiles"]["excluded_bin_topology"]),
                "excluded_profile_count": sum(
                    bool(definition.excluded_phi_bin_indices)
                    for definition in definition_by_identity.values()
                ),
                "excluded_bin_count": sum(
                    len(definition.excluded_phi_bin_indices)
                    for definition in definition_by_identity.values()
                ),
                "minimum_retained_bins_per_profile": min(
                    definition.phi_bin_count - len(definition.excluded_phi_bin_indices)
                    for definition in definition_by_identity.values()
                ),
                "nonzero_fit_two_theta_gauss_order": int(case["profiles"]["two_theta_gauss_order"]),
                "nonzero_fit_phi_gauss_order": int(case["profiles"]["phi_gauss_order"]),
                "m0_fit_two_theta_gauss_order": int(case["profiles"]["m0_two_theta_gauss_order"]),
                "m0_fit_phi_gauss_order": int(case["profiles"]["m0_phi_gauss_order"]),
                "nonzero_truth_two_theta_gauss_order": int(
                    case["validation"]["truth_two_theta_gauss_order"]
                ),
                "nonzero_truth_phi_gauss_order": int(case["validation"]["truth_phi_gauss_order"]),
                "m0_truth_two_theta_gauss_order": int(
                    case["validation"]["m0_truth_two_theta_gauss_order"]
                ),
                "m0_truth_phi_gauss_order": int(case["validation"]["m0_truth_phi_gauss_order"]),
                "observation_reexpression": (
                    "independent_truth_intensity_times_fit_bin_normalization.v1"
                ),
            },
            "profile_execution_backend": observations.execution_backend,
            "profile_execution_device": observations.execution_device,
            "sanitized_fit_inputs": [
                "observed_profile_signal",
                "observed_profile_normalization",
                "frozen_profile_identity",
                "component_width_bounds",
                "component_profile_evaluators",
            ],
            "boundary_activation_probe_width_rad": {
                "gaussian": result.gaussian_activation_probe_width_rad,
                "lorentzian": result.lorentzian_activation_probe_width_rad,
            },
            "direct_mixed_component_relative_error": decomposition_relative_error,
            "maximum_component_normalization_difference": normalization_invariance_error,
            "refinement_history": refinement_history,
        },
        "fixed_geometry": fixed_geometry,
        "source": _source_model_record(base),
        "mosaic_orientation_quadrature": {
            "alpha_panel_count": profile_physics.alpha_panel_count,
            "alpha_gauss_order": profile_physics.alpha_gauss_order,
            "azimuth_count": profile_physics.azimuth_count,
            "azimuth_phase_rad": profile_physics.azimuth_phase_rad,
        },
        "material_provenance": {
            "phase_id": base.config.material.phase_id,
            "cif_sha256": base.config.cif_sha256,
            "configured_physics_revision": base.config.physics_revision,
            "rod_catalog_revision": rod_catalog_revision,
            "simulation_config_sha256": simulation_config_sha256,
        },
        "render": {
            "skipped": args.skip_images,
            "model_role": "configured_truth_forward_visualization",
            "uses_recovered_mosaic_parameters": False,
            "applies_planted_nuisance_intensity_scales": False,
            "shape_rc": list(base.instrument.detector_shape_rc),
            "rod_count": len(base.rods),
            "family_m": sorted({rod.family_m for rod in base.rods}),
            "root_policy": "all_retained_roots.v1",
            "scope": "full_configured_all_rod_all_root_forward_field",
            "fit_m0_membership_controls_render": False,
            "includes_m0": any(rod.family_m == 0 for rod in base.rods),
            "measure_id": "raw_detector_macrobin_fixed_quadrature_estimate_A2.v1",
            "pixel_gauss_order": int(case["render"]["pixel_gauss_order"]),
            "image_paths": [str(path) for path in image_paths],
            "execution": (
                [
                    {
                        "incidence_deg": incidence_deg,
                        "backend": backend,
                        "device": device,
                    }
                    for incidence_deg, (backend, device) in zip(
                        case["incidence_angles_deg"],
                        render_execution,
                        strict=True,
                    )
                ]
                if render_execution
                else []
            ),
        },
        "artifacts": {
            "diagnostic_npz": str(diagnostic_path),
            "distribution_plot": (None if distribution_path is None else str(distribution_path)),
        },
        "timing_seconds": {
            "setup": setup_seconds,
            "direct_truth_profiles": truth_profile_seconds,
            "postfit_truth_component_profiles": truth_component_seconds,
            "component_bank_and_fit": fit_seconds,
            "render_images": render_seconds,
            "total": perf_counter() - total_start,
        },
        "python_tracemalloc_peak_bytes": peak_bytes,
    }
    manifest_path = output_directory / "bi2se3_mosaic_recovery.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": accepted,
                "truth": manifest["truth"],
                "recovered": manifest["recovered"],
                "error": manifest["error"],
                "distribution": manifest["distribution"],
                "artifacts": manifest["artifacts"],
                "fit": {
                    key: manifest["fit"][key]
                    for key in (
                        "objective",
                        "sensitivity_rank",
                        "sensitivity_condition",
                        "reflection_group_count",
                        "profile_count",
                        "m0_profile_count",
                        "nonzero_profile_count",
                        "planted_unknown_intensity_scale_range",
                        "profile_execution_backend",
                        "profile_execution_device",
                    )
                },
                "python_tracemalloc_peak_bytes": peak_bytes,
                "timing_seconds": manifest["timing_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
