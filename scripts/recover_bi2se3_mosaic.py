"""Recover a chosen Bi2Se3 mosaic law from three fixed-geometry synthetic views."""

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

from painted_ewald import MosaicParameters, Rod, wrapped_mosaic_line_density_rad_inv
from rasim_next.core.contracts import MaterialOptics
from rasim_next.fitting import (
    SHARED_GEOMETRY_PARAMETER_NAMES,
    ExactTagGeometryModel,
    MosaicProfileDefinition,
    MosaicProfileFitResult,
    MosaicProfileIdentity,
    MosaicProfileSearchResult,
    MosaicProfileSet,
    MosaicReflectionGroupKey,
    SharedGeometryCorrections,
    apply_shared_geometry_corrections,
    evaluate_continuous_mosaic_profiles,
    fit_refined_mosaic_component_profiles,
)
from rasim_next.geometry import AngleFrame, build_incident_states, detector_coordinates_to_angles
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.io.osc import read_osc
from rasim_next.pipeline.bragg_space import Bi2Se3TwoHStrength
from rasim_next.pipeline.configured_simulation import (
    ConfiguredGeometryInputs,
    ConfiguredSimulationInputs,
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


@dataclass(frozen=True, slots=True)
class _ProfilePhysicsContext:
    reciprocal_basis_Ainv: FloatArray
    crystal_to_sample: FloatArray
    rods: tuple[Rod, ...]
    rod_catalog_revision: str
    strength: Bi2Se3TwoHStrength
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
) -> tuple[ConfiguredSimulationInputs, tuple[ConfiguredSimulationInputs, ...]]:
    config_path = (case_path.parent / str(case["simulation_config"])).resolve()
    config = load_simulation_config(config_path)
    config = replace(
        config,
        source=replace(
            config.source,
            spatial_sigma_m=(0.0, 0.0),
            divergence_sigma_rad=(0.0, 0.0),
            wavelength_sigma_A=0.0,
            sample_count=1,
        ),
        instrument=replace(
            config.instrument,
            axis_rotations=tuple(
                replace(axis, angle_deg=float(case["incidence_angles_deg"][0]))
                for axis in config.instrument.axis_rotations
            ),
        ),
    )
    base = build_configured_simulation_inputs(config)
    geometry_base = ConfiguredGeometryInputs(
        config=base.config,
        samples=base.samples,
        instrument=base.instrument,
        crystal=base.crystal,
        material=base.material,
        reciprocal=base.reciprocal,
        rods=base.rods,
    )
    corrections = SharedGeometryCorrections.from_array(case["shared_geometry_corrections"])
    series: list[ConfiguredSimulationInputs] = []
    for incidence_deg in case["incidence_angles_deg"]:
        angle_config = replace(
            config,
            instrument=replace(
                config.instrument,
                axis_rotations=tuple(
                    replace(axis, angle_deg=float(incidence_deg))
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
        if incident.states.incident_state_id.size != 1 or not bool(incident.states.valid[0]):
            raise RuntimeError("the fixed-geometry proof requires one valid ideal incident state")
        series.append(
            replace(
                base,
                config=angle_config,
                instrument=corrected_instrument,
                incident=incident,
            )
        )
    return base, tuple(series)


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
    inputs: ConfiguredSimulationInputs,
    *,
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
    context = build_nominal_ewald_context(inputs)
    frame = build_osc_angle_frame(
        mean_direction_lab=inputs.config.source.mean_direction_lab,
        instrument=inputs.instrument,
        sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
        revision=f"fixed-nine-coordinate-{incidence_deg:g}deg.v1",
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
                    incidence_angle_rad=math.radians(incidence_deg),
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

    model = ExactTagGeometryModel(
        ConfiguredGeometryInputs(
            config=inputs.config,
            samples=inputs.samples,
            instrument=inputs.instrument,
            crystal=inputs.crystal,
            material=inputs.material,
            reciprocal=inputs.reciprocal,
            rods=inputs.rods,
        )
    )
    wavevector_magnitude_Ainv = float(np.linalg.norm(context.ki_sample_Ainv))
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
    observed_angles = detector_coordinates_to_angles(
        np.asarray([item.column_px for item in observed_candidates]),
        np.asarray([item.row_px for item in observed_candidates]),
        instrument=inputs.instrument,
        angle_frame=frame,
    )
    if not np.all(observed_angles.valid & observed_angles.azimuth_valid):
        raise RuntimeError("a frozen observed m=0 centroid has no valid angle coordinate")
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

    for observed_index, observed in enumerate(observed_candidates):
        integer_l_value = observed.integer_L
        definitions.append(
            MosaicProfileDefinition(
                identity=MosaicProfileIdentity(
                    dataset_id=f"Bi2Se3-{incidence_deg:g}deg",
                    incidence_angle_rad=math.radians(incidence_deg),
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
                center_two_theta_rad=float(observed_angles.two_theta_rad[observed_index]),
                center_phi_rad=float(observed_angles.phi_rad[observed_index]),
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
    strength_k_norm = float(np.linalg.norm(inputs.incident.states.k_film_phase_sample_Ainv[0]))
    for index, integer_l_value in enumerate(m0.integer_L):
        integer_l = int(integer_l_value)
        status = str(m0.detector_status[index])
        raw_evidence = predicted_evidence_by_l.get(integer_l)
        record: dict[str, object] = {
            "integer_L": integer_l,
            "reduced_order": integer_l // reduced_order_divisor,
            "structure_strength_A2": inputs.strength.evaluate(
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
            "selected": integer_l in observed_integer_l,
        }
        candidate_records.append(record)
        if integer_l not in observed_integer_l:
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
                "classification": "RAW_SIGNIFICANT_UNSUPPORTED_FORWARD_CHANNEL",
                "fit_eligible": False,
                "raw_evidence": evidence_record(unsupported_evidence_by_l[int(item["integer_L"])]),
            }
            for item in unsupported_observed_peaks
        ],
        "fitted_integer_L": list(observed_integer_l),
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
) -> str:
    payload = {
        "geometry_manifest_sha256": case["geometry_manifest_sha256"],
        "simulation_config_sha256": simulation_config_sha256,
        "configured_physics_revision": configured_physics_revision,
        "cif_sha256": cif_sha256,
        "rod_catalog_revision": rod_catalog_revision,
        "shared_geometry_corrections": case["shared_geometry_corrections"],
        "source_policy": "one-center-zero-divergence-mean-wavelength.v1",
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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256-{hashlib.sha256(encoded).hexdigest()}"


def _detector_series(
    physics: _ProfilePhysicsContext,
    geometry: tuple[_ProfileGeometryContext, ...],
    mosaic: MosaicParameters,
) -> tuple[SourceAveragedDetectorEwaldMeasure, ...]:
    first = geometry[0]
    detector = SourceAveragedDetectorEwaldMeasure(
        reciprocal_basis_Ainv=physics.reciprocal_basis_Ainv,
        crystal_to_sample=physics.crystal_to_sample,
        rods=physics.rods,
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
    detectors = _detector_series(physics, geometry, mosaic)
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
    return _combine_profile_sets(sets, profile_revision), detectors


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

    def gaussian_profile(width_rad: float) -> MosaicProfileSet:
        profiles, _ = _evaluate_profile_series(
            physics,
            geometry,
            frames,
            definitions,
            _mosaic_parameters(
                gaussian_sigma_rad=width_rad,
                lorentzian_half_width_rad=1.0,
                lorentzian_probability=0.0,
                context=physics,
            ),
            profile_revision=profile_revision,
            execution_backend=backend,
        )
        return profiles

    def lorentzian_profile(width_rad: float) -> MosaicProfileSet:
        profiles, _ = _evaluate_profile_series(
            physics,
            geometry,
            frames,
            definitions,
            _mosaic_parameters(
                gaussian_sigma_rad=1.0,
                lorentzian_half_width_rad=width_rad,
                lorentzian_probability=1.0,
                context=physics,
            ),
            profile_revision=profile_revision,
            execution_backend=backend,
        )
        return profiles

    return gaussian_profile, lorentzian_profile


def _fit_profiles(
    *,
    observations: MosaicProfileSet,
    evaluate_gaussian_profile: Callable[[float], MosaicProfileSet],
    evaluate_lorentzian_profile: Callable[[float], MosaicProfileSet],
    search_config: dict[str, Any],
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
            execution_backend=str(render_config["execution_backend"]),
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit a fixed-geometry 5/10/15-degree synthetic Bi2Se3 mosaic triplet."
    )
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args(argv)
    case_path = args.case.resolve()
    output_directory = _external_directory(args.output_directory)
    case, case_bytes, case_sha256 = _case(case_path)
    tracemalloc.start()
    total_start = perf_counter()

    setup_start = perf_counter()
    base, series = _fixed_geometry_inputs(case_path, case)
    profile_physics, profile_geometry = _profile_forward_contexts(base, series)
    rod_catalog_revision = configured_rod_catalog_revision(base)
    profile_pairs = tuple(
        _profile_definitions(
            inputs,
            case_path=case_path,
            incidence_deg=float(incidence_deg),
            osc_observation=osc_observation,
            nonzero_centroid_provenance=case["nonzero_centroid_provenance"],
            centroid_provenance=case["m0_centroid_provenance"],
            profile_config=case["profiles"],
            rod_catalog_revision=rod_catalog_revision,
        )
        for inputs, incidence_deg, osc_observation in zip(
            series,
            case["incidence_angles_deg"],
            case["m0_observations"],
            strict=True,
        )
    )
    frames = tuple(item[0] for item in profile_pairs)
    definitions = tuple(item[1] for item in profile_pairs)
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
    simulation_config_sha256 = hashlib.sha256(simulation_config_path.read_bytes()).hexdigest()
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
    )
    setup_seconds = perf_counter() - setup_start

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
        execution_backend=str(case["profiles"]["execution_backend"]),
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
        execution_backend=str(case["profiles"]["execution_backend"]),
    )
    common_valid = independent_truth_profiles.valid & normalization_probe.valid
    if independent_truth_profiles.source_revision != normalization_probe.source_revision:
        raise RuntimeError("truth and fit quadrature changed the ideal source realization")
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
        execution_backend=str(case["profiles"]["execution_backend"]),
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
        execution_backend=str(case["profiles"]["execution_backend"]),
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
        execution_backend=str(case["profiles"]["execution_backend"]),
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
            tuple(float(value) for value in case["incidence_angles_deg"]),
            output_directory=output_directory,
            render_config=case["render"],
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
        "schema_version": "rasim-mosaic-recovery-result-v2",
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
        "fixed_geometry": {
            "manifest_sha256": case["geometry_manifest_sha256"],
            "corrections": {
                name: value
                for name, value in zip(
                    SHARED_GEOMETRY_PARAMETER_NAMES,
                    case["shared_geometry_corrections"],
                    strict=True,
                )
            },
            "beam_center_column_row_px": list(
                base.config.instrument.detector_reference_coordinate_px
            ),
        },
        "source": {
            "sample_count": 1,
            "spatial_sigma_m": [0.0, 0.0],
            "divergence_sigma_rad": [0.0, 0.0],
            "wavelength_sigma_A": 0.0,
            "wavelength_A": base.config.source.mean_wavelength_A,
        },
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
