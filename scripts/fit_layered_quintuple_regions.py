"""Prepare and fit matched detector-native regions for layered quintuple materials."""
# ruff: noqa: E402  # Direct script execution bootstraps repository import roots below.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import replace
from importlib.metadata import version
from itertools import pairwise
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src",):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rasim_next.fitting import (
    FIXED_EXPERIMENT_STATE_SCHEMA_VERSION,
    DetectorHorizonAcceptance,
    FixedLatticeState,
    FixedMatchedRegionBackground,
    FixedMosaicState,
    FixedPositionState,
    IntegratedPeakAreaProjection,
    MatchedRegionObservations,
    RadialBackgroundProfiles,
    RadialBackgroundState,
    build_fixed_experiment_series,
    condition_matched_region_background_from_anchors,
    condition_matched_region_model_from_anchors,
    fit_matched_regions,
    fit_shared_radial_background,
)
from rasim_next.geometry import (
    angles_to_detector_coordinate_area_measure,
    detector_coordinates_to_angles,
)
from rasim_next.io.osc import read_osc
from rasim_next.measurement import (
    ContinuousDetectorChartAreaMeasure,
    ContinuousRegionQuadrature,
    LayeredReciprocalFrame,
    NativePixelRegionProjection,
    OffSpecularBand,
    OffSpecularBandLayout,
    SpecularAngularProfileRegion,
    compile_continuous_rectangle_quadrature,
    compile_native_pixel_region_projection,
    integrate_shared_native_pixel_field,
    offspecular_region_membership,
    specular_angular_membership,
)
from rasim_next.ordered import (
    SiteDisplacementProfile,
    TransverseIsotropicSiteDisplacement,
    quintuple_layer_site_labels,
)
from rasim_next.pipeline.configured_simulation import (
    build_nominal_ewald_context,
    build_source_averaged_detector,
    configured_rod_catalog_revision,
    load_simulation_config,
)
from rasim_next.pipeline.continuous_detector import SampleQIntensityEnvelope
from rasim_next.pipeline.continuous_fold import (
    ContinuousFoldBand,
    ContinuousFoldCorrectionPlan,
    apply_continuous_fold_correction_plan,
    compile_continuous_fold_correction_plan,
)
from rasim_next.pipeline.reciprocal_detector_chart import (
    LayeredReciprocalDetectorAreaChart,
)
from rasim_next.proof.diagnostics import write_diagnostic
from rasim_next.reflectivity import (
    FIXED_EXTERNAL_QZ_INTERFACE,
    LOCAL_LAMELLA_INTERFACE,
    ParrattStitchStack,
)
from rasim_next.selection import build_osc_angle_frame
from rasim_next.stacking import Parent, RichEpsilonModel

FAMILIES = (0, 1, 3, 4)
STRUCTURE_PARAMETER_NAMES = (
    "bi_delta_z_fractional",
    "outer_chalcogen_delta_z_fractional",
    "outer_chalcogen_vacancy_fraction",
    "intensity_envelope_u_radial_A2",
    "intensity_envelope_u_normal_A2",
)
STRUCTURE_PARAMETER_COUNT = len(STRUCTURE_PARAMETER_NAMES)
STRUCTURE_STAGE_PARAMETERS = {
    "A": STRUCTURE_PARAMETER_NAMES[:2],
    "B": STRUCTURE_PARAMETER_NAMES[2:3],
    "C": STRUCTURE_PARAMETER_NAMES[3:5],
    "joint": STRUCTURE_PARAMETER_NAMES,
}
PROJECTION_CONVERGENCE_METRICS = (
    "mass_relative_l2",
    "support_relative_l2",
    "covariance_relative_frobenius",
    "maximum_row_mass_standardized_error",
    "maximum_row_support_relative_error",
    "maximum_covariance_row_relative_l2",
)
MEASURED_PROJECTION_METHOD = "piecewise_constant_native_pixel_field_continuous_chart_projection.v2"
NATIVE_PIXEL_CENTER_METHOD = "frozen_native_pixel_center_raw_minus_scaled_dark.v1"
PREPARED_SCHEMA = "rasim-layered-quintuple-matched-regions-v4"
BACKGROUND_SCHEMA = "rasim-shared-radial-background-v4"
FIT_SCHEMA = "rasim-layered-quintuple-matched-region-fit-v14"
FIT_PROGRESS_SCHEMA = "rasim-layered-quintuple-matched-fit-progress-v11"
PROFILE_SCHEMA = "rasim-layered-quintuple-matched-figure-profiles-v14"
PROFILE_PROGRESS_SCHEMA = "rasim-layered-quintuple-matched-profile-progress-v7"
PROFILE_PARAMETER_REPLAY_METHOD = "frozen_fit_parameter_profile_replay.v1"
PROFILE_PARAMETER_REPLAY_EVIDENCE = "FROZEN_FIT_PARAMETER_REPLAY"
SPECULAR_INTERFACE_ASSUMPTIONS = (
    LOCAL_LAMELLA_INTERFACE,
    FIXED_EXTERNAL_QZ_INTERFACE,
)
PROFILE_VECTOR_ARRAY_NAMES = (
    "profile_identity",
    "profile_bin_index",
    "profile_valid",
    "profile_display_L",
    "profile_display_qz_Ainv",
    "profile_selection_coordinate",
    "profile_selection_coordinate_kind",
    "profile_measured_signal_density",
    "profile_measured_radial_signal_density",
    "profile_model_signal_density",
)
M0_SIGNAL_ONLY_ARRAY_NAMES = (
    "m0_signal_only_bin_index",
    "m0_signal_only_two_theta_deg",
    "m0_signal_only_measured_density",
    "m0_signal_only_model_plus_background_density",
    "m0_signal_only_support_px2",
    "m0_signal_only_valid",
)
PROFILE_RENDER_ARRAY_NAMES = (
    *PROFILE_VECTOR_ARRAY_NAMES,
    *M0_SIGNAL_ONLY_ARRAY_NAMES,
    "display_detector_counts",
    "display_full_region_code",
    "display_fit_region_code",
)
RADIAL_BACKGROUND_MODEL = (
    "C_d + A_d * (1-exp(-(r/r_in)^p_in)) * exp(-(r/r_out)^p_out); "
    "shared r_in,p_in,r_out,p_out and per-dataset A_d,C_d"
)
DARK_CORRECTION_MODEL = "scaled_dark_subtraction_no_clip.v1"
PEAK_AREA_OBJECTIVE = "background_conditioned_integrated_peak_count_mass.v1"
FAULT_FREE_THREE_R_MODEL = "rich_epsilon_parent_3r_exact_zero_fault.v1"
FIXED_EXPERIMENT_SCHEMA = FIXED_EXPERIMENT_STATE_SCHEMA_VERSION


def _dark_scale_basis_is_valid(scale: float, scale_basis: object) -> bool:
    if scale == 0.0:
        return scale_basis == "no_acquisition_matched_dark.v1"
    return scale_basis == "matched_exposure_assumed.v1"


def _dark_covariance_model(scale: float) -> str:
    return (
        "no_dark_contribution.v1"
        if scale == 0.0
        else "shared_independent_poisson_dark_across_datasets.v1"
    )


def _detector_horizon_acceptance(policy: Any) -> DetectorHorizonAcceptance:
    """Resolve the explicit off-specular detector horizon policy."""

    if not isinstance(policy, dict):
        raise ValueError("horizon_gate must be a mapping")
    try:
        return DetectorHorizonAcceptance(
            offspecular_air_exit_guard_rad=math.radians(
                float(policy["offspecular_air_exit_guard_deg"])
            )
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("horizon_gate requires a finite nonnegative off-specular guard") from None


def _diffraction_peak_horizon_policy(policy: Any) -> tuple[float, float, float, str]:
    """Resolve the explicit diffraction-peak clearance policy."""

    if not isinstance(policy, dict):
        raise ValueError("horizon_gate must be a mapping")
    try:
        return (
            float(policy["diffraction_peak_air_exit_guard_deg"]),
            float(policy["diffraction_peak_selection_center_cutoff_deg"]),
            float(policy["maximum_sub_guard_model_mass_fraction"]),
            str(policy["diffraction_peak_selection_rule"]),
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("horizon_gate requires an explicit diffraction-peak policy") from None


class _OscAngleDetectorAreaChart:
    __slots__ = (
        "angle_frame",
        "horizon_acceptance",
        "instrument",
        "nominal_context",
        "revision",
    )

    def __init__(
        self,
        *,
        instrument: Any,
        angle_frame: Any,
        nominal_context: Any,
        horizon_acceptance: DetectorHorizonAcceptance,
        revision: str,
    ) -> None:
        self.instrument = instrument
        self.angle_frame = angle_frame
        self.nominal_context = nominal_context
        self.horizon_acceptance = horizon_acceptance
        self.revision = str(revision)

    def map_detector_area(
        self,
        two_theta_rad: np.ndarray,
        phi_rad: np.ndarray,
    ) -> ContinuousDetectorChartAreaMeasure:
        angular = angles_to_detector_coordinate_area_measure(
            two_theta_rad,
            phi_rad,
            instrument=self.instrument,
            angle_frame=self.angle_frame,
        )
        coordinates = angular.coordinates
        geometry = self.nominal_context.evaluate_detector_geometry(
            coordinates.column_px,
            coordinates.row_px,
            include_surface_jacobian=False,
        )
        kf_air = np.asarray(geometry.kf_air_sample_Ainv)
        air_exit = np.arctan2(kf_air[..., 2], np.hypot(kf_air[..., 0], kf_air[..., 1]))
        horizon_valid = self.horizon_acceptance.m0_detector_mask(air_exit)
        valid = coordinates.valid & horizon_valid
        return ContinuousDetectorChartAreaMeasure(
            column_px=coordinates.column_px,
            row_px=coordinates.row_px,
            detector_area_jacobian_px2_per_chart2=np.where(
                valid,
                angular.detector_area_jacobian_px2_per_rad2,
                0.0,
            ),
            valid=valid,
        )


class _DatasetContinuousRegionPlan(NamedTuple):
    dataset_index: int
    dataset_id: str
    global_observation_row: np.ndarray
    quadrature: ContinuousRegionQuadrature
    fold_bands: tuple[ContinuousFoldBand, ...]
    rectangle_count: int


class _DatasetNativePixelCenterPlan(NamedTuple):
    dataset_index: int
    dataset_id: str
    global_observation_row: np.ndarray
    projection: NativePixelRegionProjection


def _intersected_intervals(
    base_interval: tuple[float, float],
    fit_intervals: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    result = []
    for lower, upper in fit_intervals:
        selected_lower = max(float(base_interval[0]), float(lower))
        selected_upper = min(float(base_interval[1]), float(upper))
        if selected_lower < selected_upper:
            result.append((selected_lower, selected_upper))
    return tuple(result)


def _merged_intervals(
    intervals: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    merged: list[list[float]] = []
    for lower, upper in sorted((float(a), float(b)) for a, b in intervals):
        if not merged or lower > merged[-1][1]:
            merged.append([lower, upper])
        else:
            merged[-1][1] = max(merged[-1][1], upper)
    return tuple((lower, upper) for lower, upper in merged)


def _rod_family_radial_fold_center_Ainv(
    inputs: Any,
    frame: LayeredReciprocalFrame,
    *,
    family_m: int,
    radial_interval_Ainv: tuple[float, float],
) -> float:
    """Return the unique rod radius represented by one fitted radial band."""

    lower, upper = radial_interval_Ainv
    radii = []
    for rod in inputs.rods:
        if int(rod.family_m) != family_m:
            continue
        q_crystal_Ainv = inputs.reciprocal.basis_Ainv @ np.asarray(
            (float(rod.h), float(rod.k), 0.0),
            dtype=np.float64,
        )
        q_sample_Ainv = inputs.instrument.sample_from_crystal.rotation @ q_crystal_Ainv
        radius, _ = frame.coordinates(q_sample_Ainv[None, :])
        value = float(radius[0])
        if lower < value < upper:
            radii.append(value)
    if not radii:
        raise ValueError(
            f"radial signal band {radial_interval_Ainv!r} contains no m={family_m} rod fold"
        )
    values = np.asarray(radii, dtype=np.float64)
    center = float(np.mean(values))
    tolerance = 2.0e-12 * max(1.0, abs(center))
    if float(np.max(np.abs(values - center))) > tolerance:
        raise ValueError(
            f"radial signal band {radial_interval_Ainv!r} contains multiple m={family_m} folds"
        )
    return center


def _manifest_fit_intervals(
    manifest: dict[str, Any],
    *,
    dataset_id: str,
    profile_identity: str,
    angular: bool,
) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    for peak in manifest["fit_peak_catalog"]:
        if peak["dataset_id"] != dataset_id or peak["profile_identity"] != profile_identity:
            continue
        centers = peak.get("centers", (peak.get("center"),))
        half_width = float(peak["half_width"])
        for center in centers:
            interval = (float(center) - half_width, float(center) + half_width)
            intervals.append(tuple(np.deg2rad(interval)) if angular else interval)
    return tuple(intervals)


def _peak_area_projection_revision(
    catalog: Sequence[dict[str, Any]],
    source_signal_peak_index: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"background-conditioned-integrated-peak-area-projection.v1\0")
    digest.update(json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(source_signal_peak_index, dtype=np.int64).tobytes())
    return f"sha256-{digest.hexdigest()}.integrated-peak-area.v1"


def _integrated_peak_area_projection(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> tuple[IntegratedPeakAreaProjection, list[dict[str, Any]]]:
    """Map every conditioned signal bin to one declared trusted peak area."""

    candidates: list[dict[str, Any]] = []
    dataset_ids = tuple(str(value) for value in manifest["dataset_ids"])
    reciprocal = (
        2.0
        * np.pi
        * np.linalg.inv(
            np.asarray(manifest["fixed_lattice"]["active_direct_basis_A"], dtype=np.float64)
        ).T
    )
    b3_norm_Ainv = float(np.linalg.norm(reciprocal[:, 2]))
    for peak in manifest["fit_peak_catalog"]:
        family = int(peak["family_m"])
        centers = tuple(float(value) for value in peak.get("centers", (peak.get("center"),)))
        for center in centers:
            expanded_identity = str(peak["identity"])
            if len(centers) > 1:
                expanded_identity += f"@{center:.12g}"
            for side in ("center",) if family == 0 else ("plus", "minus"):
                peak_id = expanded_identity if family == 0 else f"{expanded_identity}:{side}"
                half_width = float(peak["half_width"])
                bounds = (center - half_width, center + half_width)
                candidates.append(
                    {
                        "peak_id": peak_id,
                        "source_recipe_identity": str(peak["identity"]),
                        "dataset_id": str(peak["dataset_id"]),
                        "dataset_index": dataset_ids.index(str(peak["dataset_id"])),
                        "family_m": family,
                        "profile_identity": str(peak["profile_identity"]),
                        "side": side,
                        "coordinate_kind": str(peak["coordinate_kind"]),
                        "center": center,
                        "half_width": half_width,
                        "coordinate_bounds": bounds,
                        "qz_bounds_Ainv": (
                            None if family == 0 else [b3_norm_Ainv * value for value in bounds]
                        ),
                    }
                )

    m0_edges = np.asarray(manifest["m0_region"]["two_theta_bin_edges_rad"], dtype=np.float64)
    layout_by_group = {str(record["group"]): record for record in manifest["offspecular_layouts"]}
    signal_rows = np.flatnonzero(~np.asarray(arrays["is_background"]))
    source_candidate = np.empty(signal_rows.size, dtype=np.int64)
    for source_index, row in enumerate(signal_rows):
        dataset_id = str(arrays["dataset_id"][row])
        family = int(arrays["signal_family_m"][row])
        group = str(arrays["group"][row])
        band = str(arrays["band"][row])
        bin_index = int(arrays["bin_index"][row])
        if family == 0:
            row_bounds = tuple(np.rad2deg(m0_edges[bin_index : bin_index + 2]))
            side = "center"
        else:
            axial_edges = np.asarray(layout_by_group[group]["axial_bin_edges"], dtype=np.float64)
            row_bounds = tuple(float(value) for value in axial_edges[bin_index : bin_index + 2])
            side = band.rsplit("_", 1)[-1]
            if side not in {"plus", "minus"}:
                raise ValueError("off-specular signal band lacks its signed detector side")
        matched = [
            index
            for index, candidate in enumerate(candidates)
            if candidate["dataset_id"] == dataset_id
            and candidate["family_m"] == family
            and candidate["side"] == side
            and max(row_bounds[0], candidate["coordinate_bounds"][0])
            < min(row_bounds[1], candidate["coordinate_bounds"][1])
        ]
        if len(matched) != 1:
            raise ValueError(f"signal row {int(row)} maps to {len(matched)} integrated peak areas")
        source_candidate[source_index] = matched[0]

    used_candidate = set(int(value) for value in source_candidate)
    missing_candidate = [
        candidates[index]["peak_id"]
        for index in range(len(candidates))
        if index not in used_candidate
    ]
    if missing_candidate:
        raise ValueError(
            "declared integrated peak areas have no retained signal rows: "
            + ", ".join(missing_candidate)
        )
    source_peak = source_candidate
    catalog = []
    for peak_index, candidate in enumerate(candidates):
        record = dict(candidate)
        record["source_signal_rows"] = signal_rows[source_peak == peak_index].tolist()
        record["coordinate_bounds"] = list(record["coordinate_bounds"])
        catalog.append(record)
    revision = _peak_area_projection_revision(catalog, source_peak)
    projection = IntegratedPeakAreaProjection(
        peak_ids=tuple(record["peak_id"] for record in catalog),
        peak_dataset_index=np.asarray(
            [record["dataset_index"] for record in catalog], dtype=np.int64
        ),
        peak_signal_family=np.asarray([record["family_m"] for record in catalog], dtype=np.int64),
        source_signal_peak_index=source_peak,
        revision=revision,
    )
    projection.aggregation_matrix(
        MatchedRegionObservations(
            dataset_ids=dataset_ids,
            dataset_index=arrays["dataset_index"],
            block_index=arrays["block_index"],
            signal_family=arrays["signal_family_m"],
            is_background=arrays["is_background"],
            count_mass=arrays["count_sum"],
            support_px2=arrays["support_px2"],
            background_coordinate=arrays["coordinate_mean"],
            required_signal_families=FAMILIES,
        )
    )
    return projection, catalog


def _prepared_peak_area_projection(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    observations: MatchedRegionObservations,
) -> IntegratedPeakAreaProjection:
    catalog = manifest.get("integrated_peak_catalog")
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("prepared diagnostic lacks integrated peak areas")
    source_peak = np.asarray(arrays["source_signal_peak_index"], dtype=np.int64)
    signal_rows = np.flatnonzero(~np.asarray(observations.is_background))
    if (
        manifest.get("source_signal_peak_index_sha256") != _array_sha256(source_peak)
        or manifest.get("peak_area_projection_revision")
        != _peak_area_projection_revision(catalog, source_peak)
        or any(
            record.get("source_signal_rows") != signal_rows[source_peak == peak_index].tolist()
            for peak_index, record in enumerate(catalog)
        )
    ):
        raise ValueError("prepared integrated peak-area mapping changed")
    projection = IntegratedPeakAreaProjection(
        peak_ids=tuple(str(record["peak_id"]) for record in catalog),
        peak_dataset_index=np.asarray(
            [record["dataset_index"] for record in catalog], dtype=np.int64
        ),
        peak_signal_family=np.asarray([record["family_m"] for record in catalog], dtype=np.int64),
        source_signal_peak_index=source_peak,
        revision=str(manifest.get("peak_area_projection_revision", "")),
    )
    projection.aggregation_matrix(observations)
    return projection


def _compile_dataset_continuous_region_plan(
    inputs: Any,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    *,
    dataset_index: int,
    gauss_order: int,
    subdivision_count: int,
    offspecular_axial_refinement: int,
    offspecular_radial_transform: str,
    offspecular_signal_minimum_radial_nodes_per_side: int,
    m0_phi_subdivision_count: int,
    apply_fit_windows: bool = True,
) -> _DatasetContinuousRegionPlan:
    if (
        isinstance(offspecular_axial_refinement, bool)
        or int(offspecular_axial_refinement) < 1
        or offspecular_radial_transform != "squared_fold_coordinate.v1"
        or isinstance(offspecular_signal_minimum_radial_nodes_per_side, bool)
        or int(offspecular_signal_minimum_radial_nodes_per_side) < 1
        or isinstance(m0_phi_subdivision_count, bool)
        or int(m0_phi_subdivision_count) < 1
    ):
        raise ValueError("off-specular continuous quadrature settings are invalid")
    offspecular_axial_refinement = int(offspecular_axial_refinement)
    m0_phi_subdivision_count = int(m0_phi_subdivision_count)
    signal_radial_subdivisions = max(
        int(subdivision_count),
        math.ceil(int(offspecular_signal_minimum_radial_nodes_per_side) / int(gauss_order)),
    )
    dataset_id = str(manifest["dataset_ids"][dataset_index])
    global_rows = np.flatnonzero(arrays["dataset_index"] == dataset_index)
    if not global_rows.size:
        raise ValueError(f"dataset {dataset_id!r} has no retained observations")
    dataset_local_row = {int(row): index for index, row in enumerate(global_rows)}
    context = build_nominal_ewald_context(inputs)
    frame = LayeredReciprocalFrame(
        reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
        sample_from_crystal_rotation=inputs.instrument.sample_from_crystal.rotation,
        axial_basis_index=2,
    )
    angle_frame = build_osc_angle_frame(
        mean_direction_lab=inputs.config.source.mean_direction_lab,
        instrument=inputs.instrument,
        sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
        revision=f"continuous-matched-regions-{dataset_index}.v1",
    )
    horizon_acceptance = _detector_horizon_acceptance(manifest["horizon_gate"])
    angle_chart = _OscAngleDetectorAreaChart(
        instrument=inputs.instrument,
        angle_frame=angle_frame,
        nominal_context=context,
        horizon_acceptance=horizon_acceptance,
        revision=(
            f"continuous-osc-angle-area.{dataset_index}."
            f"{inputs.instrument.sample_geometry_revision}.{angle_frame.revision}"
        ),
    )
    layout_by_group = {str(layout["group"]): layout for layout in manifest["offspecular_layouts"]}
    part_columns: list[np.ndarray] = []
    part_rows: list[np.ndarray] = []
    part_weights: list[np.ndarray] = []
    part_observations: list[np.ndarray] = []
    part_coordinates: list[np.ndarray] = []
    part_revisions: list[str] = []
    fold_bands: list[ContinuousFoldBand] = []
    rectangle_count = 0

    def append_part(
        *,
        chart: Any,
        first_bounds: list[tuple[float, float]],
        second_bounds: list[tuple[float, float]],
        global_observation_rows: list[int],
        background_coordinate_axis: int,
        subdivision_count_by_axis: tuple[int, int],
        first_coordinate_squared_fold_centers: list[float] | None = None,
    ) -> ContinuousRegionQuadrature | None:
        nonlocal rectangle_count
        if not first_bounds:
            return None
        unique_rows = tuple(dict.fromkeys(global_observation_rows))
        row_lookup = {row: index for index, row in enumerate(unique_rows)}
        part = compile_continuous_rectangle_quadrature(
            chart=chart,
            first_coordinate_bounds=np.asarray(first_bounds, dtype=np.float64),
            second_coordinate_bounds=np.asarray(second_bounds, dtype=np.float64),
            observation_row=np.asarray(
                [row_lookup[row] for row in global_observation_rows],
                dtype=np.int64,
            ),
            observation_count=len(unique_rows),
            gauss_order=gauss_order,
            background_coordinate_axis=background_coordinate_axis,
            subdivision_count=subdivision_count_by_axis,
            first_coordinate_squared_fold_center=(
                None
                if first_coordinate_squared_fold_centers is None
                else np.asarray(first_coordinate_squared_fold_centers, dtype=np.float64)
            ),
        )
        local_map = np.asarray(
            [dataset_local_row[row] for row in unique_rows],
            dtype=np.int64,
        )
        dataset_part = ContinuousRegionQuadrature(
            column_px=part.column_px,
            row_px=part.row_px,
            detector_area_weight_px2=part.detector_area_weight_px2,
            observation_row=local_map[np.asarray(part.observation_row)],
            background_coordinate=part.background_coordinate,
            observation_count=global_rows.size,
            chart_revision=part.chart_revision,
        )
        part_columns.append(np.asarray(dataset_part.column_px))
        part_rows.append(np.asarray(dataset_part.row_px))
        part_weights.append(np.asarray(dataset_part.detector_area_weight_px2))
        part_observations.append(np.asarray(dataset_part.observation_row))
        part_coordinates.append(np.asarray(dataset_part.background_coordinate))
        part_revisions.append(dataset_part.quadrature_revision)
        rectangle_count += len(first_bounds)
        return dataset_part

    m0_record = manifest["m0_region"]
    m0_edges = np.asarray(m0_record["two_theta_bin_edges_rad"], dtype=np.float64)
    m0_first: list[tuple[float, float]] = []
    m0_second: list[tuple[float, float]] = []
    m0_rows: list[int] = []
    m0_fit_intervals = (
        _manifest_fit_intervals(
            manifest,
            dataset_id=dataset_id,
            profile_identity="m0",
            angular=True,
        )
        if apply_fit_windows
        else ((float(m0_edges[0]), float(m0_edges[-1])),)
    )
    for global_row in global_rows:
        if str(arrays["group"][global_row]) != "m0":
            continue
        bin_index = int(arrays["bin_index"][global_row])
        theta_intervals = _intersected_intervals(
            (float(m0_edges[bin_index]), float(m0_edges[bin_index + 1])),
            m0_fit_intervals,
        )
        band = str(arrays["band"][global_row])
        if band == "signal":
            phi_interval = tuple(float(value) for value in m0_record["phi_signal_interval_rad"])
        elif band.startswith("background_"):
            anchor_index = int(band.rsplit("_", 1)[1])
            phi_interval = tuple(
                float(value) for value in m0_record["phi_background_intervals_rad"][anchor_index]
            )
        else:
            raise ValueError(f"unsupported m=0 band {band!r}")
        for theta_interval in theta_intervals:
            m0_first.append(theta_interval)
            m0_second.append(phi_interval)
            m0_rows.append(int(global_row))
    append_part(
        chart=angle_chart,
        first_bounds=m0_first,
        second_bounds=m0_second,
        global_observation_rows=m0_rows,
        background_coordinate_axis=1,
        subdivision_count_by_axis=(subdivision_count, m0_phi_subdivision_count),
    )

    for group, layout in layout_by_group.items():
        group_rows = [int(row) for row in global_rows if str(arrays["group"][row]) == group]
        if not group_rows:
            continue
        chart = LayeredReciprocalDetectorAreaChart(
            detector_measure=context.geometry,
            reciprocal_frame=frame,
            detector_column_interval_px=tuple(
                float(value) for value in layout["detector_column_interval_px"]
            ),
            air_exit_guard_rad=horizon_acceptance.offspecular_air_exit_guard_rad,
        )
        axial_edges = np.asarray(layout["axial_bin_edges"], dtype=np.float64)
        signal_intervals = {
            str(record["identity"]): tuple(float(value) for value in record["qr_interval_Ainv"])
            for record in layout["signal_bands"]
        }
        signal_fold_centers = {
            identity: _rod_family_radial_fold_center_Ainv(
                inputs,
                frame,
                family_m=int(identity.split("_", 1)[0][1:]),
                radial_interval_Ainv=interval,
            )
            for identity, interval in signal_intervals.items()
        }
        background_intervals = tuple(
            tuple(float(value) for value in interval)
            for interval in layout["background_intervals_Ainv"]
        )
        represented_profiles = tuple(
            f"m{int(str(record['identity']).split('_', 1)[0][1:])}"
            for record in layout["signal_bands"]
        )
        signal_parts = {
            identity: {
                "first": [],
                "second": [],
                "rows": [],
                "fold_centers": [],
            }
            for identity in signal_intervals
        }
        background_first_bounds: list[tuple[float, float]] = []
        background_second_bounds: list[tuple[float, float]] = []
        background_observation_rows: list[int] = []
        for global_row in group_rows:
            band = str(arrays["band"][global_row])
            bin_index = int(arrays["bin_index"][global_row])
            axial_interval = (float(axial_edges[bin_index]), float(axial_edges[bin_index + 1]))
            if band in signal_intervals:
                qr_interval = signal_intervals[band]
                profiles = (f"m{int(band.split('_', 1)[0][1:])}",)
                signal_part = signal_parts[band]
                target_first = signal_part["first"]
                target_second = signal_part["second"]
                target_rows = signal_part["rows"]
                fold_center = signal_fold_centers[band]
            elif band.startswith("background_"):
                anchor_index = int(band.rsplit("_", 1)[1])
                qr_interval = background_intervals[anchor_index]
                profiles = represented_profiles
                target_first = background_first_bounds
                target_second = background_second_bounds
                target_rows = background_observation_rows
                fold_center = None
            else:
                raise ValueError(f"unsupported off-specular band {band!r}")
            fit_intervals = (
                _merged_intervals(
                    tuple(
                        interval
                        for profile in profiles
                        for interval in _manifest_fit_intervals(
                            manifest,
                            dataset_id=dataset_id,
                            profile_identity=profile,
                            angular=False,
                        )
                    )
                )
                if apply_fit_windows
                else ((float(axial_edges[0]), float(axial_edges[-1])),)
            )
            if not fit_intervals:
                raise ValueError(
                    f"prepared row {global_row} has no fitted L interval for {dataset_id!r}"
                )
            for selected_axial in _intersected_intervals(axial_interval, fit_intervals):
                target_first.append(qr_interval)
                target_second.append(selected_axial)
                target_rows.append(global_row)
                if fold_center is not None:
                    signal_part["fold_centers"].append(fold_center)
        for identity, signal_part in signal_parts.items():
            signal_first = signal_part["first"]
            signal_second = signal_part["second"]
            signal_rows = signal_part["rows"]
            signal_quadrature = append_part(
                chart=chart,
                first_bounds=signal_first,
                second_bounds=signal_second,
                global_observation_rows=signal_rows,
                background_coordinate_axis=0,
                subdivision_count_by_axis=(
                    signal_radial_subdivisions,
                    offspecular_axial_refinement * subdivision_count,
                ),
                first_coordinate_squared_fold_centers=signal_part["fold_centers"],
            )
            if signal_quadrature is not None:
                family_m = int(identity.split("_", 1)[0][1:])
                fold_bands.append(
                    ContinuousFoldBand(
                        chart=chart,
                        radial_interval_Ainv=signal_intervals[identity],
                        axial_bounds=np.asarray(signal_second, dtype=np.float64),
                        observation_row=np.asarray(
                            [dataset_local_row[row] for row in signal_rows],
                            dtype=np.int64,
                        ),
                        observation_count=global_rows.size,
                        rod_h_k=np.asarray(
                            [(rod.h, rod.k) for rod in inputs.rods if rod.family_m == family_m],
                            dtype=np.int64,
                        ),
                        base_quadrature=signal_quadrature,
                        gauss_order=gauss_order,
                        radial_subdivisions=signal_radial_subdivisions,
                        axial_subdivisions=(offspecular_axial_refinement * subdivision_count),
                    )
                )
        append_part(
            chart=chart,
            first_bounds=background_first_bounds,
            second_bounds=background_second_bounds,
            global_observation_rows=background_observation_rows,
            background_coordinate_axis=0,
            subdivision_count_by_axis=(
                subdivision_count,
                offspecular_axial_refinement * subdivision_count,
            ),
        )

    quadrature_revision = (
        "sha256-"
        + hashlib.sha256(
            json.dumps(
                {
                    "definition_id": "mixed_continuous_detector_chart_quadrature.v2",
                    "dataset_id": dataset_id,
                    "gauss_order": int(gauss_order),
                    "subdivision_count": int(subdivision_count),
                    "offspecular_axial_refinement": offspecular_axial_refinement,
                    "offspecular_radial_transform": offspecular_radial_transform,
                    "offspecular_signal_minimum_radial_nodes_per_side": int(
                        offspecular_signal_minimum_radial_nodes_per_side
                    ),
                    "apply_fit_windows": bool(apply_fit_windows),
                    "parts": part_revisions,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    quadrature = ContinuousRegionQuadrature(
        column_px=np.concatenate(part_columns),
        row_px=np.concatenate(part_rows),
        detector_area_weight_px2=np.concatenate(part_weights),
        observation_row=np.concatenate(part_observations),
        background_coordinate=np.concatenate(part_coordinates),
        observation_count=global_rows.size,
        chart_revision=quadrature_revision,
    )
    return _DatasetContinuousRegionPlan(
        dataset_index=dataset_index,
        dataset_id=dataset_id,
        global_observation_row=global_rows,
        quadrature=quadrature,
        fold_bands=tuple(fold_bands),
        rectangle_count=rectangle_count,
    )


def _update_hash_from_file(digest: Any, path: Path) -> None:
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    _update_hash_from_file(digest, path)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _raw_array_bytes_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _relative_l2(
    reference: np.ndarray,
    candidate: np.ndarray,
    selected: np.ndarray,
) -> float:
    denominator = max(np.linalg.norm(reference[selected]), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(reference[selected] - candidate[selected]) / denominator)


def _nonnegative_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_canonical_revision(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and value.startswith("sha256-")
        and len(value) >= 71
        and _is_sha256(value[7:71])
    )


def _fitted_rod_roster(inputs: Any) -> tuple[tuple[int, int, int, float], ...]:
    return tuple(
        (int(rod.h), int(rod.k), int(rod.family_m), float(rod.population))
        for rod in inputs.rods
        if rod.family_m in FAMILIES
    )


def _rod_roster_sha256(roster: Sequence[Sequence[int | float]]) -> str:
    return hashlib.sha256(
        json.dumps(
            tuple(tuple(value) for value in roster), separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()


def _background_model_identity_is_admissible(
    background_model: Any,
    *,
    expected_dataset_ids: Sequence[str],
) -> bool:
    if not isinstance(background_model, dict):
        return False
    dataset_ids = tuple(str(value) for value in expected_dataset_ids)
    count_by_dataset = background_model.get("excluded_flat_pixel_count_by_dataset")
    hash_by_dataset = background_model.get("excluded_flat_pixel_sha256_by_dataset")
    if not isinstance(count_by_dataset, dict) or not isinstance(hash_by_dataset, dict):
        return False
    return bool(
        set(count_by_dataset) == set(dataset_ids)
        and set(hash_by_dataset) == set(dataset_ids)
        and all(
            isinstance(count_by_dataset[dataset_id], int)
            and not isinstance(count_by_dataset[dataset_id], bool)
            and count_by_dataset[dataset_id] > 0
            for dataset_id in dataset_ids
        )
        and all(_is_sha256(hash_by_dataset[dataset_id]) for dataset_id in dataset_ids)
        and _is_sha256(background_model.get("state_revision"))
        and _is_canonical_revision(background_model.get("conditioned_revision"))
        and all(
            _is_sha256(background_model.get(name))
            for name in (
                "artifact_sha256",
                "parameter_vector_sha256",
                "parameter_covariance_sha256",
                "radial_mass_sha256",
                "radial_covariance_sha256",
                "conditioned_mass_sha256",
                "conditioned_covariance_sha256",
                "anchor_projection_sha256",
            )
        )
    )


def _native_projection_convergence(
    *,
    coarse_count_mass: np.ndarray,
    coarse_support_px2: np.ndarray,
    coarse_count_covariance: np.ndarray,
    refined_count_mass: np.ndarray,
    refined_support_px2: np.ndarray,
    refined_count_covariance: np.ndarray,
    dataset_index: np.ndarray,
    dataset_ids: Sequence[str],
    maximum_relative_l2: float,
) -> dict[str, Any]:
    """Summarize measured-projection refinement without hiding weak rows."""

    coarse_mass = np.asarray(coarse_count_mass, dtype=np.float64)
    refined_mass = np.asarray(refined_count_mass, dtype=np.float64)
    coarse_support = np.asarray(coarse_support_px2, dtype=np.float64)
    refined_support = np.asarray(refined_support_px2, dtype=np.float64)
    coarse_covariance = np.asarray(coarse_count_covariance, dtype=np.float64)
    refined_covariance = np.asarray(refined_count_covariance, dtype=np.float64)
    dataset = np.asarray(dataset_index, dtype=np.int64)
    ids = tuple(str(value) for value in dataset_ids)
    row_count = refined_mass.size
    limit = float(maximum_relative_l2)
    if (
        coarse_mass.shape != (row_count,)
        or coarse_support.shape != (row_count,)
        or refined_support.shape != (row_count,)
        or dataset.shape != (row_count,)
        or coarse_covariance.shape != (row_count, row_count)
        or refined_covariance.shape != (row_count, row_count)
        or not ids
        or len(set(ids)) != len(ids)
        or np.any((dataset < 0) | (dataset >= len(ids)))
        or not math.isfinite(limit)
        or limit <= 0.0
        or any(
            np.any(~np.isfinite(value))
            for value in (
                coarse_mass,
                refined_mass,
                coarse_support,
                refined_support,
                coarse_covariance,
                refined_covariance,
            )
        )
    ):
        raise ValueError("native projection convergence inputs are invalid or misaligned")
    support_masks_equal = bool(np.array_equal(coarse_support > 0.0, refined_support > 0.0))

    def relative_norm(candidate: np.ndarray, reference: np.ndarray) -> float:
        return float(
            np.linalg.norm(candidate - reference)
            / max(np.linalg.norm(reference), np.finfo(np.float64).tiny)
        )

    by_dataset: dict[str, dict[str, float]] = {}
    for dataset_number, dataset_id in enumerate(ids):
        selected = np.flatnonzero(dataset == dataset_number)
        if not selected.size:
            raise ValueError("every projection dataset must contain observation rows")
        candidate_covariance = coarse_covariance[np.ix_(selected, selected)]
        reference_covariance = refined_covariance[np.ix_(selected, selected)]
        reference_variance = np.maximum(
            np.diag(reference_covariance),
            np.finfo(np.float64).tiny,
        )
        row_mass_scale = np.maximum(
            np.abs(refined_mass[selected]),
            np.sqrt(reference_variance),
        )
        row_support_scale = np.maximum(
            np.abs(refined_support[selected]),
            np.finfo(np.float64).tiny,
        )
        covariance_row_error = np.linalg.norm(
            candidate_covariance - reference_covariance,
            axis=1,
        ) / np.maximum(
            np.linalg.norm(reference_covariance, axis=1),
            np.finfo(np.float64).tiny,
        )
        by_dataset[dataset_id] = {
            "mass_relative_l2": relative_norm(
                coarse_mass[selected],
                refined_mass[selected],
            ),
            "support_relative_l2": relative_norm(
                coarse_support[selected],
                refined_support[selected],
            ),
            "covariance_relative_frobenius": relative_norm(
                candidate_covariance,
                reference_covariance,
            ),
            "maximum_row_mass_standardized_error": float(
                np.max(np.abs(coarse_mass[selected] - refined_mass[selected]) / row_mass_scale)
            ),
            "maximum_row_support_relative_error": float(
                np.max(
                    np.abs(coarse_support[selected] - refined_support[selected]) / row_support_scale
                )
            ),
            "maximum_covariance_row_relative_l2": float(np.max(covariance_row_error)),
        }
    converged = support_masks_equal and all(
        math.isfinite(value) and 0.0 <= value <= limit
        for metrics in by_dataset.values()
        for value in metrics.values()
    )
    return {
        "status": "COMPLETE",
        "converged": converged,
        "support_masks_equal": support_masks_equal,
        "maximum_relative_l2": limit,
        "by_dataset": by_dataset,
    }


def _integrated_area_projection_converged(summary: dict[str, Any]) -> bool:
    """Accept refinement when integrated masses and their supports have converged.

    Covariance refinement remains reported, while the refined covariance itself is
    authoritative for whitening the fit objective.
    """

    required_metrics = (
        "mass_relative_l2",
        "support_relative_l2",
        "maximum_row_mass_standardized_error",
        "maximum_row_support_relative_error",
    )
    limit = float(summary["maximum_relative_l2"])
    return bool(summary["support_masks_equal"]) and all(
        math.isfinite(value) and 0.0 <= value <= limit
        for metrics in summary["by_dataset"].values()
        for name in required_metrics
        for value in (float(metrics[name]),)
    )


def _display_profile_projection_converged(summary: dict[str, Any]) -> bool:
    """Accept a display projection from its pooled count mass and support.

    Full-branch displays contain weak edge bins whose individual relative errors
    are intentionally retained as diagnostics.  They do not define the fitted
    integrated-peak observable.
    """

    required_metrics = ("mass_relative_l2", "support_relative_l2")
    limit = float(summary["maximum_relative_l2"])
    return bool(summary["support_masks_equal"]) and all(
        math.isfinite(value) and 0.0 <= value <= limit
        for metrics in summary["by_dataset"].values()
        for name in required_metrics
        for value in (float(metrics[name]),)
    )


def _projection_convergence_is_admissible(
    evidence: Any,
    *,
    expected_dataset_ids: Sequence[str],
    expected_limit: float,
) -> bool:
    """Fail closed on the compact measured-projection refinement evidence."""

    try:
        by_dataset = evidence["by_dataset"]
        limit = float(evidence["maximum_relative_l2"])
        values = {
            str(dataset_id): {str(name): float(value) for name, value in metrics.items()}
            for dataset_id, metrics in by_dataset.items()
        }
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False
    expected_ids = tuple(str(value) for value in expected_dataset_ids)
    acceptance_measure = evidence.get("acceptance_measure")
    if acceptance_measure == "integrated_peak_count_mass_and_support.v1":
        covariance_policy = "refined_covariance_is_authoritative_for_objective_whitening"
        accepted = _integrated_area_projection_converged(evidence)
    elif acceptance_measure == "display_profile_pooled_count_mass_and_support.v1":
        covariance_policy = "refined_covariance_is_authoritative_for_display"
        accepted = _display_profile_projection_converged(evidence)
    else:
        return False
    return bool(
        evidence.get("status") == "COMPLETE"
        and evidence.get("converged") is True
        and evidence.get("support_masks_equal") is True
        and isinstance(evidence.get("covariance_refinement_converged"), bool)
        and evidence.get("covariance_policy") == covariance_policy
        and limit == float(expected_limit)
        and set(values) == set(expected_ids)
        and all(set(metrics) == set(PROJECTION_CONVERGENCE_METRICS) for metrics in values.values())
        and all(
            math.isfinite(value) and value >= 0.0
            for metrics in values.values()
            for value in metrics.values()
        )
        and accepted
    )


def _conditioned_model_cubature_errors(
    fit_model_mass: np.ndarray,
    oracle_model_mass: np.ndarray,
    *,
    fixed_background: FixedMatchedRegionBackground,
    signal_family_m: np.ndarray,
    is_background: np.ndarray,
    peak_aggregation: np.ndarray | None = None,
    peak_family_m: np.ndarray | None = None,
) -> tuple[float, dict[str, float], float]:
    """Compare cubature at the exact anchor-conditioned objective measure."""

    fit_conditioned = condition_matched_region_model_from_anchors(
        fixed_background,
        fit_model_mass,
    )
    oracle_conditioned = condition_matched_region_model_from_anchors(
        fixed_background,
        oracle_model_mass,
    )
    family_by_row = np.asarray(signal_family_m, dtype=np.int64)
    background = np.asarray(is_background, dtype=np.bool_)

    if peak_aggregation is None:
        fit_objective = fit_conditioned
        oracle_objective = oracle_conditioned
        objective_family = family_by_row
        objective_row = ~background
    else:
        aggregation = np.asarray(peak_aggregation, dtype=np.float64)
        objective_family = np.asarray(peak_family_m, dtype=np.int64)
        if (
            aggregation.ndim != 2
            or aggregation.shape[1] != int(np.count_nonzero(~background))
            or objective_family.shape != (aggregation.shape[0],)
        ):
            raise ValueError("integrated peak cubature projection is misaligned")
        fit_objective = aggregation @ fit_conditioned[~background]
        oracle_objective = aggregation @ oracle_conditioned[~background]
        objective_row = np.ones(aggregation.shape[0], dtype=np.bool_)

    by_family = {
        str(family): _relative_l2(
            oracle_objective,
            fit_objective,
            objective_family == family,
        )
        for family in FAMILIES
    }
    return (
        _relative_l2(oracle_objective, fit_objective, objective_row),
        by_family,
        _relative_l2(
            np.asarray(oracle_model_mass, dtype=np.float64),
            np.asarray(fit_model_mass, dtype=np.float64),
            background,
        ),
    )


def _file_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _dark_correction_from_recipe(
    recipe_path: Path,
    recipe: dict[str, Any],
    *,
    detector_shape_rc: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    settings = recipe["dark_correction"]
    path = (recipe_path.parent / str(settings["path"])).resolve()
    identity = _file_identity(path)
    counts = read_osc(path).detector_native_counts
    if counts.shape != detector_shape_rc:
        raise ValueError("dark OSC shape differs from the fitted detector")
    native_sha256 = _raw_array_bytes_sha256(counts)
    scale = _nonnegative_real(settings["scale"], "dark scale")
    record = {
        "model_id": DARK_CORRECTION_MODEL,
        "path": str(path),
        "file_sha256": identity["sha256"],
        "detector_native_bytes_sha256": native_sha256,
        "detector_native_shape_rc": list(counts.shape),
        "detector_native_dtype": str(counts.dtype),
        "scale": scale,
        "scale_basis": str(settings["scale_basis"]),
        "negative_values_clipped": False,
        "smoothing_applied": False,
        "covariance_model": _dark_covariance_model(scale),
    }
    return counts, record


def _verified_dark_counts(manifest: dict[str, Any]) -> tuple[np.ndarray, float]:
    record = manifest.get("dark_correction")
    if not isinstance(record, dict):
        raise ValueError("prepared diagnostic lacks its declared dark correction")
    scale = _nonnegative_real(record.get("scale"), "dark scale")
    if (
        record.get("model_id") != DARK_CORRECTION_MODEL
        or record.get("negative_values_clipped") is not False
        or record.get("smoothing_applied") is not False
        or record.get("covariance_model") != _dark_covariance_model(scale)
        or not _dark_scale_basis_is_valid(scale, record.get("scale_basis"))
    ):
        raise ValueError("prepared diagnostic lacks its declared dark correction")
    path = Path(str(record.get("path"))).resolve()
    if _sha256(path) != record.get("file_sha256"):
        raise ValueError("dark OSC changed after preparation")
    counts = read_osc(path).detector_native_counts
    if (
        list(counts.shape) != record.get("detector_native_shape_rc")
        or str(counts.dtype) != record.get("detector_native_dtype")
        or _raw_array_bytes_sha256(counts) != record.get("detector_native_bytes_sha256")
    ):
        raise ValueError("decoded dark OSC changed after preparation")
    return counts, scale


def _implementation_identity() -> dict[str, Any]:
    """Hash the complete numerical implementation imported by this adapter."""

    package_roots = (ROOT / "src" / "painted_ewald", ROOT / "src" / "rasim_next")
    runtime = {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "numba": version("numba"),
    }
    files = {
        Path(__file__).resolve(),
        *(path.resolve() for package_root in package_roots for path in package_root.rglob("*.py")),
    }
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.as_posix()):
        relative = path.relative_to(ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        _update_hash_from_file(digest, path)
        digest.update(b"\0")
    digest.update(json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode("ascii"))
    return {
        "path": str(ROOT),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "scope": (
            "matched-region adapter, every painted_ewald and rasim_next Python source, and "
            "numerical runtime versions"
        ),
        "runtime": runtime,
    }


def _require_unchanged(identity: dict[str, str], *, role: str) -> None:
    path = Path(identity["path"])
    if not path.is_file() or _sha256(path) != identity["sha256"]:
        raise RuntimeError(f"{role} changed during the calculation")


def _verified_recorded_file_identity(record: Any, *, role: str) -> dict[str, str]:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError(f"{role} identity is missing")
    actual = _file_identity(Path(record["path"]))
    if actual["sha256"] != record.get("sha256"):
        raise ValueError(f"{role} changed after artifact creation")
    return actual


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _external_file(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or resolved.is_relative_to(ROOT):
        raise ValueError(f"artifact path resolves inside the repository: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _validated_recipe(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != "rasim-fitted-figure7-recreation-v3":
        raise ValueError("unsupported Figure-7 recipe")
    material_id = document.get("material_id")
    if not isinstance(material_id, str) or not material_id.strip():
        raise ValueError("Figure-7 recipe requires a material_id")
    dataset_ids = tuple(document.get("dataset_ids", ()))
    if len(dataset_ids) != 3 or len(set(dataset_ids)) != 3:
        raise ValueError("Figure-7 recipe requires three distinct datasets")
    if document.get("display_dataset_id") not in dataset_ids:
        raise ValueError("display_dataset_id must name one fitted dataset")
    _data_projection_qualification_required(document)
    stitch = document.get("parratt_stitch")
    if stitch is not None:
        if (
            not isinstance(stitch, dict)
            or stitch.get("model_id") != "empirical_parratt_kinematic_strength.v1"
            or stitch.get("scope") != "m0_only"
            or stitch.get("interface_assumption")
            not in {
                "local_lamella_follows_mosaic.v1",
                "fixed_external_qz_m0_strength.v1",
            }
        ):
            raise ValueError("Parratt stitch interface assumption is unsupported")
        try:
            substrate = complex(
                float(stitch["substrate_refractive_index_real"]),
                float(stitch["substrate_refractive_index_imag"]),
            )
            top = float(stitch["top_roughness_A"])
            bottom = float(stitch["bottom_roughness_A"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Parratt stitch optical-stack values must be explicit") from None
        ParrattStitchStack(
            substrate_refractive_index=substrate,
            top_roughness_A=top,
            bottom_roughness_A=bottom,
            interface_assumption=str(stitch["interface_assumption"]),
        )
    dark = document.get("dark_correction")
    try:
        dark_scale = _nonnegative_real(
            dark.get("scale") if isinstance(dark, dict) else None,
            "dark scale",
        )
    except ValueError:
        dark_scale = math.nan
    if (
        not isinstance(dark, dict)
        or dark.get("model_id") != DARK_CORRECTION_MODEL
        or not isinstance(dark.get("path"), str)
        or not dark["path"]
        or not math.isfinite(dark_scale)
        or dark_scale < 0.0
        or not _dark_scale_basis_is_valid(dark_scale, dark.get("scale_basis"))
    ):
        raise ValueError("Figure-7 recipe requires an explicit no-clip dark correction")
    peaks = document.get("fit_peak")
    if not isinstance(peaks, list) or not peaks:
        raise ValueError("Figure-7 recipe requires a frozen fit_peak catalog")
    represented = {int(item["family_m"]) for item in peaks}
    if represented != set(FAMILIES):
        raise ValueError("fit_peak catalog must jointly represent m=0,1,3,4")
    peak_identities: set[tuple[str, str]] = set()
    for peak in peaks:
        dataset_id = str(peak.get("dataset_id", ""))
        family = int(peak.get("family_m", -1))
        identity = str(peak.get("identity", ""))
        if dataset_id not in dataset_ids:
            raise ValueError("fit_peak dataset_id must name one fitted dataset")
        if family not in FAMILIES or peak.get("profile_identity") != f"m{family}":
            raise ValueError("fit_peak profile_identity must match family_m")
        expected_coordinate = "two_theta_deg" if family == 0 else "L"
        if peak.get("coordinate_kind") != expected_coordinate:
            raise ValueError(
                f"fit_peak coordinate_kind must be {expected_coordinate!r} for m={family}"
            )
        if not identity or (dataset_id, identity) in peak_identities:
            raise ValueError("fit_peak identities must be nonempty and unique per dataset")
        centers = peak.get("centers", (peak.get("center"),))
        try:
            center_values = np.asarray(tuple(centers), dtype=np.float64)
            half_width = float(peak.get("half_width"))
        except (TypeError, ValueError):
            raise ValueError("fit_peak centers and half_width must be finite") from None
        if (
            center_values.ndim != 1
            or not center_values.size
            or np.any(~np.isfinite(center_values))
            or len(np.unique(center_values)) != center_values.size
            or not math.isfinite(half_width)
            or half_width <= 0.0
        ):
            raise ValueError("fit_peak centers must be unique and finite with positive half_width")
        peak_identities.add((dataset_id, identity))
    for excluded in document.get("excluded_peak", ()):
        if excluded.get("dataset_id") not in dataset_ids:
            raise ValueError("excluded_peak dataset_id must name one fitted dataset")
        if excluded.get("reason") not in {"below exit horizon", "horizon-clearance gate"}:
            raise ValueError("excluded_peak reason must state the frozen horizon policy")
    m0_region = document.get("m0_region")
    if (
        not isinstance(m0_region, dict)
        or len(m0_region.get("phi_background_intervals_deg", ())) != 2
    ):
        raise ValueError("m0_region requires exactly two affine background anchors")
    offspecular_groups = document.get("offspecular_group")
    if not isinstance(offspecular_groups, list) or any(
        len(group.get("background_intervals_Ainv", ())) != 2 for group in offspecular_groups
    ):
        raise ValueError("each offspecular group requires exactly two affine anchors")
    horizon_gate = document.get("horizon_gate")
    _detector_horizon_acceptance(horizon_gate)
    (
        diffraction_peak_guard,
        selection_cutoff,
        maximum_sub_guard_fraction,
        selection_rule,
    ) = _diffraction_peak_horizon_policy(horizon_gate)
    if (
        not math.isfinite(diffraction_peak_guard)
        or diffraction_peak_guard < 0.0
        or not math.isfinite(selection_cutoff)
        or selection_cutoff <= diffraction_peak_guard
        or not math.isfinite(maximum_sub_guard_fraction)
        or not 0.0 <= maximum_sub_guard_fraction < 1.0
        or selection_rule != "alpha_f_center_minus_3_gaussian_sigma_at_least_guard.v1"
    ):
        raise ValueError("horizon_gate diffraction-peak policy is invalid")
    profile_cubature = document.get("profile_cubature")
    if (
        not isinstance(profile_cubature, dict)
        or int(profile_cubature.get("fit_gauss_order", 0)) < 2
        or int(profile_cubature.get("fold_fit_subdivisions", 0)) <= 0
        or int(profile_cubature.get("minimum_valid_bins_per_profile", 0)) <= 0
    ):
        raise ValueError("profile_cubature settings are invalid")
    model_cubature = document.get("model_cubature")
    model_tolerance = float(
        model_cubature.get("maximum_oracle_relative_l2", 0.0)
        if isinstance(model_cubature, dict)
        else 0.0
    )
    if (
        not isinstance(model_cubature, dict)
        or int(model_cubature.get("fit_gauss_order", 0)) < 2
        or int(model_cubature.get("oracle_gauss_order", 0))
        <= int(model_cubature.get("fit_gauss_order", 0))
        or int(model_cubature.get("fold_fit_subdivisions", 0)) <= 0
        or int(model_cubature.get("fold_oracle_subdivisions", 0))
        <= int(model_cubature.get("fold_fit_subdivisions", 0))
        or int(model_cubature.get("cuda_coordinate_chunk_size", 0)) <= 0
        or int(model_cubature.get("maximum_state_block_count", 0)) <= 0
        or not math.isfinite(model_tolerance)
        or model_tolerance <= 0.0
    ):
        raise ValueError("model_cubature settings are invalid")
    return document


def _trusted_model_cubature(recipe: dict[str, Any]) -> dict[str, int | float]:
    """Return the artifact-facing cubature contract from a verified recipe."""

    settings = recipe["model_cubature"]
    return {
        "fit_gauss_order": int(settings["fit_gauss_order"]),
        "oracle_gauss_order": int(settings["oracle_gauss_order"]),
        "fit_subdivision_count": int(settings["fold_fit_subdivisions"]),
        "oracle_subdivision_count": int(settings["fold_oracle_subdivisions"]),
        "maximum_relative_l2": float(settings["maximum_oracle_relative_l2"]),
    }


def _data_projection_qualification_required(recipe: dict[str, Any]) -> bool:
    """Return whether measured counts require a continuous-projection refinement gate."""

    policy = recipe.get("observation_policy", "continuous_projection_required.v1")
    if policy == "continuous_projection_required.v1":
        return True
    if policy == NATIVE_PIXEL_CENTER_METHOD:
        return False
    raise ValueError("unsupported measured observation policy")


def _uses_native_pixel_center_observations(recipe: dict[str, Any]) -> bool:
    return recipe.get("observation_policy") == NATIVE_PIXEL_CENTER_METHOD


def _joint_cubature_is_admissible(
    evidence: Any,
    *,
    trusted_recipe: dict[str, Any],
) -> bool:
    try:
        trusted = _trusted_model_cubature(trusted_recipe)
        maximum = float(trusted["maximum_relative_l2"])
        global_error = float(evidence["relative_l2"])
        anchor_error = float(evidence["relative_l2_background_anchor_rows"])
        family_errors = {
            str(key): float(value) for key, value in evidence["relative_l2_by_family_m"].items()
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return bool(
        evidence.get("performed") is True
        and evidence.get("status") == "COMPLETE"
        and all(evidence.get(name) == value for name, value in trusted.items())
        and set(family_errors) == {str(value) for value in FAMILIES}
        and all(
            math.isfinite(value) and 0.0 <= value <= maximum
            for value in (global_error, *family_errors.values())
        )
        and math.isfinite(anchor_error)
        and anchor_error >= 0.0
    )


def _data_projection_is_admissible(
    evidence: Any,
    *,
    trusted_recipe: dict[str, Any],
    expected_dataset_ids: Sequence[str],
) -> bool:
    if _uses_native_pixel_center_observations(trusted_recipe):
        try:
            projection_revisions = tuple(evidence["projection_revisions"])
            selected_pair_count = int(evidence["selected_pixel_region_pair_count"])
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return bool(
            evidence.get("method") == NATIVE_PIXEL_CENTER_METHOD
            and evidence.get("qualification_role") == "authoritative_frozen_observation"
            and evidence.get("status") == "COMPLETE"
            and evidence.get("projection_performed") is False
            and evidence.get("refinement_oracle") == "NOT_APPLICABLE_EXACT_MEMBERSHIP"
            and evidence.get("smoothing_applied") is False
            and evidence.get("diffraction_model_pixelized") is False
            and len(projection_revisions) == len(tuple(expected_dataset_ids))
            and all(_is_sha256(value) for value in projection_revisions)
            and selected_pair_count > 0
            and all(
                _is_sha256(evidence.get(name))
                for name in (
                    "selected_dataset_index_sha256",
                    "selected_flat_pixel_index_sha256",
                    "selected_observation_row_sha256",
                    "count_mass_sha256",
                    "support_px2_sha256",
                    "count_covariance_sha256",
                )
            )
        )
    try:
        trusted = _trusted_model_cubature(trusted_recipe)
        maximum = float(trusted["maximum_relative_l2"])
        family_errors = {
            str(key): float(value) for key, value in evidence["relative_l2_by_family_m"].items()
        }
        anchor_error = float(evidence["relative_l2_background_anchor_rows"])
        fit_revisions = tuple(evidence["fit_projection_revisions"])
        oracle_revisions = tuple(evidence["oracle_projection_revisions"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    dataset_ids = tuple(str(value) for value in expected_dataset_ids)
    qualification_required = _data_projection_qualification_required(trusted_recipe)
    common = bool(
        evidence.get("method") == MEASURED_PROJECTION_METHOD
        and evidence.get("status") == "COMPLETE"
        and evidence.get("smoothing_applied") is False
        and evidence.get("diffraction_model_pixelized") is False
        and all(evidence.get(name) == value for name, value in trusted.items())
        and set(family_errors) == {str(value) for value in FAMILIES}
        and all(math.isfinite(value) and value >= 0.0 for value in family_errors.values())
        and math.isfinite(anchor_error)
        and anchor_error >= 0.0
        and len(fit_revisions) == len(dataset_ids)
        and len(oracle_revisions) == len(dataset_ids)
        and all(
            isinstance(value, str) and len(value) == 64
            for value in (*fit_revisions, *oracle_revisions)
        )
    )
    if not common:
        return False
    if qualification_required:
        return bool(
            evidence.get("qualification_role", "required_release_gate") == "required_release_gate"
            and evidence.get("converged") is True
            and all(value <= maximum for value in family_errors.values())
            and _projection_convergence_is_admissible(
                evidence,
                expected_dataset_ids=dataset_ids,
                expected_limit=maximum,
            )
        )
    return evidence.get("qualification_role") == "diagnostic_only"


def _integrated_peak_objective_is_admissible(
    document: dict[str, Any],
    execution: dict[str, Any],
) -> bool:
    evidence = document.get("integrated_peak_areas", {})
    try:
        peak_ids = tuple(str(value) for value in evidence["peak_ids"])
        dataset = np.asarray(evidence["dataset_index"], dtype=np.int64)
        family = np.asarray(evidence["family_m"], dtype=np.int64)
        observed = np.asarray(
            evidence["observed_background_subtracted_count_mass"],
            dtype=np.float64,
        )
        fitted = np.asarray(evidence["fitted_count_mass"], dtype=np.float64)
        revision = str(evidence["projection_revision"])
        mapping_sha256 = str(evidence["source_signal_peak_index_sha256"])
    except (KeyError, TypeError, ValueError):
        return False
    shape = (len(peak_ids),)
    dataset_count = len(document.get("dataset_scales", {}))
    return bool(
        document.get("objective_measure") == PEAK_AREA_OBJECTIVE
        and peak_ids
        and len(set(peak_ids)) == len(peak_ids)
        and dataset.shape == shape
        and family.shape == shape
        and observed.shape == shape
        and fitted.shape == shape
        and dataset_count > 0
        and np.all((dataset >= 0) & (dataset < dataset_count))
        and set(family.tolist()) == set(FAMILIES)
        and np.all(np.isfinite(observed))
        and np.all(np.isfinite(fitted))
        and revision.startswith("sha256-")
        and revision.endswith(".integrated-peak-area.v1")
        and _is_sha256(mapping_sha256)
        and execution.get("objective_measure") == PEAK_AREA_OBJECTIVE
        and execution.get("peak_area_projection_revision") == revision
        and execution.get("source_signal_peak_index_sha256") == mapping_sha256
    )


def _native_observation_execution_is_admissible(
    evidence: Any,
    execution: Any,
) -> bool:
    """Bind exact native-count evidence to the evaluated stage identity."""

    if not isinstance(evidence, dict) or not isinstance(execution, dict):
        return False
    return bool(
        execution.get("measured_observation_method") == evidence.get("method")
        and execution.get("measured_projection_performed") is False
        and execution.get("measured_projection_revisions") == evidence.get("projection_revisions")
        and execution.get("selected_pixel_region_pair_count")
        == evidence.get("selected_pixel_region_pair_count")
        and execution.get("selected_dataset_index_sha256")
        == evidence.get("selected_dataset_index_sha256")
        and execution.get("selected_flat_pixel_index_sha256")
        == evidence.get("selected_flat_pixel_index_sha256")
        and execution.get("selected_observation_row_sha256")
        == evidence.get("selected_observation_row_sha256")
        and execution.get("observed_count_mass_sha256") == evidence.get("count_mass_sha256")
        and execution.get("observed_support_px2_sha256") == evidence.get("support_px2_sha256")
        and execution.get("observed_count_covariance_sha256")
        == evidence.get("count_covariance_sha256")
    )


def _fixed_displacement_gauge_is_admissible(structure: Any) -> bool:
    if not isinstance(structure, dict):
        return False
    profile = structure.get("site_displacement_profile", {})
    sites = profile.get("sites") if isinstance(profile, dict) else None
    try:
        scale = float(structure["site_adp_scale"])
        site_values = np.asarray(
            [
                (
                    site["reference_u_radial_A2"],
                    site["reference_u_normal_A2"],
                    site["fitted_u_radial_A2"],
                    site["fitted_u_normal_A2"],
                )
                for site in sites
            ],
            dtype=np.float64,
        )
        labels = tuple(str(site["source_label"]) for site in sites)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        scale == 1.0
        and structure.get("site_adp_refinement_status") == "fixed_literature_reference"
        and structure.get("displacement_gauge")
        == {
            "model_id": "fixed_site_adp_plus_regularized_sample_q_envelope.v1",
            "site_adp_common_mode": "fixed_reference",
            "sample_q_envelope_mode": "fit_zero_centered_regularized",
        }
        and profile.get("model_id") == "transverse_isotropic_site_reference_fixed.v1"
        and isinstance(profile.get("provenance"), str)
        and bool(profile["provenance"])
        and site_values.shape == (3, 4)
        and len(set(labels)) == 3
        and np.all(np.isfinite(site_values))
        and np.all(site_values >= 0.0)
        and np.array_equal(site_values[:, :2], site_values[:, 2:])
    )


def _vacancy_structure_representative_is_admissible(structure: Any) -> bool:
    """Validate the persisted vacancy coordinate and its derived site fractions."""

    if not isinstance(structure, dict):
        return False
    try:
        vacancy = float(structure["outer_chalcogen_vacancy_fraction"])
        outer_occupancy = float(structure["outer_chalcogen_occupancy"])
        outer_fraction = float(structure["outer_chalcogen_fraction"])
        antisite = float(structure["outer_bi_antisite_fraction"])
        bi_occupancy = float(structure["bi_occupancy"])
        central_occupancy = float(structure["central_chalcogen_occupancy"])
    except (KeyError, TypeError, ValueError):
        return False
    expected_outer = 1.0 - vacancy
    return bool(
        structure.get("occupancy_rule") == "outer_site_chalcogen_plus_vacancy.v1"
        and all(
            math.isfinite(value)
            for value in (
                vacancy,
                outer_occupancy,
                outer_fraction,
                antisite,
                bi_occupancy,
                central_occupancy,
            )
        )
        and 0.0 <= vacancy <= 1.0
        and antisite == 0.0
        and bi_occupancy == 1.0
        and central_occupancy == 1.0
        and outer_occupancy == expected_outer
        and outer_fraction == expected_outer
    )


def _fit_start_is_admissible(
    document: dict[str, Any],
    *,
    expected_stage: str,
    active_parameter_names: tuple[str, ...],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    execution_identity: dict[str, Any],
    final_parameters: np.ndarray,
) -> bool:
    """Validate the policy-bound optimizer start independently of the producer."""

    policy = document.get("execution_policy")
    optimizer = document.get("optimizer")
    provenance = document.get("provenance")
    if not isinstance(optimizer, dict) or not isinstance(provenance, dict):
        return False
    fit_start = optimizer.get("fit_start")
    if not isinstance(fit_start, dict):
        return False
    kind = fit_start.get("kind")
    if policy == "seeded_joint_only.v1":
        if expected_stage != "joint" or kind not in {"explicit", "progress_restart"}:
            return False
    elif policy == "staged_A_B_C_joint.v1":
        if expected_stage not in {"A", "B", "C", "joint"} or kind not in {
            "default",
            "progress_restart",
        }:
            return False
    else:
        return False
    try:
        active_index = np.asarray(
            [STRUCTURE_PARAMETER_NAMES.index(name) for name in active_parameter_names],
            dtype=np.int64,
        )
        initial = np.asarray(fit_start["initial_parameters"], dtype=np.float64)
        initial_full = np.asarray(fit_start["initial_full_parameters"], dtype=np.float64)
        maximum_evaluations = fit_start["maximum_function_evaluations"]
    except (KeyError, TypeError, ValueError):
        return False
    if (
        initial.shape != (active_index.size,)
        or initial_full.shape != (STRUCTURE_PARAMETER_COUNT,)
        or np.any(~np.isfinite(initial))
        or np.any(~np.isfinite(initial_full))
        or np.any(initial_full < lower_bounds)
        or np.any(initial_full > upper_bounds)
        or not np.array_equal(initial, initial_full[active_index])
        or fit_start.get("initial_parameters_sha256") != _array_sha256(initial)
        or isinstance(maximum_evaluations, bool)
        or not isinstance(maximum_evaluations, int)
        or maximum_evaluations <= 0
    ):
        return False
    frozen_index = np.asarray(
        [index for index in range(STRUCTURE_PARAMETER_COUNT) if index not in set(active_index)],
        dtype=np.int64,
    )
    if final_parameters.shape != (STRUCTURE_PARAMETER_COUNT,) or not np.array_equal(
        final_parameters[frozen_index], initial_full[frozen_index]
    ):
        return False
    source = fit_start.get("source_artifact")
    if kind == "progress_restart":
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or not source["path"]
            or not _is_sha256(source.get("sha256"))
            or fit_start.get("semantics")
            != "restart from a completely evaluated parameter vector; optimizer state is not continued"
        ):
            return False
    elif source is not None or fit_start.get("semantics") != "new optimizer run":
        return False
    return provenance.get("fit_run_identity") == {
        "model_execution": execution_identity,
        "fit_start": fit_start,
    }


def _restart_progress_is_admissible(
    document: dict[str, Any],
    *,
    expected_execution_identity: dict[str, Any],
    expected_initial_parameters: np.ndarray,
    expected_initial_full_parameters: np.ndarray,
) -> bool:
    """Validate the exact completed vector used as a same-stage restart."""

    try:
        active = np.asarray(document["active_parameters"], dtype=np.float64)
        full = np.asarray(document["full_parameters"], dtype=np.float64)
        completed = document["completed_model_evaluations"]
        run_identity = document["fit_run_identity"]
        devices = document["devices"]
        backends = document["evaluated_backends"]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        document.get("schema_version") == FIT_PROGRESS_SCHEMA
        and document.get("execution_identity") == expected_execution_identity
        and active.shape == expected_initial_parameters.shape
        and full.shape == expected_initial_full_parameters.shape
        and np.all(np.isfinite(active))
        and np.all(np.isfinite(full))
        and np.array_equal(active, expected_initial_parameters)
        and np.array_equal(full, expected_initial_full_parameters)
        and isinstance(completed, int)
        and not isinstance(completed, bool)
        and completed > 0
        and isinstance(run_identity, dict)
        and run_identity.get("model_execution") == expected_execution_identity
        and isinstance(run_identity.get("fit_start"), dict)
        and isinstance(devices, list)
        and bool(devices)
        and all(isinstance(value, str) and value for value in devices)
        and isinstance(backends, list)
        and bool(backends)
        and all(isinstance(value, str) and value for value in backends)
    )


def stage_fit_document_is_admissible(
    document: dict[str, Any],
    *,
    expected_stage: str,
    expected_active_parameter_names: Sequence[str],
    trusted_recipe: dict[str, Any],
    diagnostic_sha256: str,
    recipe_sha256: str,
    fit_plan_sha256: str,
    adapter_sha256: str,
    implementation_sha256: str,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    parameter_scales: Sequence[float],
    sensitivity_relative_tolerance: float,
    bound_proximity_in_parameter_scales: float,
    maximum_sensitivity_condition: float,
    allow_model_limited_joint: bool = False,
) -> bool:
    """Return whether one staged structure artifact can initialize its successor."""

    numerical = document.get("numerical_convergence", {})
    sensitivity = document.get("sensitivity", {})
    cubature = document.get("cubature_oracle", {})
    data_projection = document.get("data_projection", {})
    background_model = document.get("background_model", {})
    provenance = document.get("provenance", {})
    execution = provenance.get("execution_identity", {})
    active = tuple(expected_active_parameter_names)
    frozen = tuple(name for name in STRUCTURE_PARAMETER_NAMES if name not in active)
    try:
        rank = int(sensitivity.get("rank"))
        numerical_rank = int(sensitivity.get("numerical_rank"))
        parameter_count = int(sensitivity.get("parameter_count"))
        condition = float(sensitivity.get("condition"))
        relative_tolerance = float(sensitivity.get("relative_tolerance"))
        parameters = np.asarray(document.get("full_parameter_vector"), dtype=np.float64)
        lower = np.asarray(lower_bounds, dtype=np.float64)
        upper = np.asarray(upper_bounds, dtype=np.float64)
        scales = np.asarray(parameter_scales, dtype=np.float64)
        active_index = np.asarray(
            [STRUCTURE_PARAMETER_NAMES.index(name) for name in active],
            dtype=np.int64,
        )
        expected_tolerance = float(sensitivity_relative_tolerance)
        bound_proximity = float(bound_proximity_in_parameter_scales)
        maximum_condition = float(maximum_sensitivity_condition)
        regularization = document.get("regularization", {})
        recorded_scales = np.asarray(regularization.get("parameter_scales"), dtype=np.float64)
        recorded_lower = np.asarray(regularization.get("lower_bounds"), dtype=np.float64)
        recorded_upper = np.asarray(regularization.get("upper_bounds"), dtype=np.float64)
        recorded_bound_proximity = float(regularization.get("bound_proximity_in_parameter_scales"))
        structure = document.get("structure_representative", {})
        reported_parameters = np.asarray(
            [structure[name] for name in STRUCTURE_PARAMETER_NAMES],
            dtype=np.float64,
        )
        trusted_cubature = _trusted_model_cubature(trusted_recipe)
        trusted_dataset_ids = tuple(str(value) for value in trusted_recipe["dataset_ids"])
        dataset_scales = document["dataset_scales"]
        if not isinstance(dataset_scales, dict):
            raise TypeError("dataset scales must be a mapping")
        if set(dataset_scales) != set(trusted_dataset_ids):
            raise ValueError("dataset scales differ from the trusted dataset roster")
        dataset_ids = trusted_dataset_ids
        dataset_scale_values = tuple(float(dataset_scales[value]) for value in dataset_ids)
        rod_roster = tuple(tuple(value) for value in document["model_rod_roster_h_k_m_population"])
        rod_roster_sha256 = _rod_roster_sha256(rod_roster)
        rod_families = {int(value[2]) for value in rod_roster}
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False
    if (
        parameters.shape != (STRUCTURE_PARAMETER_COUNT,)
        or lower.shape != parameters.shape
        or upper.shape != parameters.shape
        or scales.shape != parameters.shape
        or np.any(~np.isfinite(parameters))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        return False
    scaled_bound_distance = np.minimum(
        (parameters[active_index] - lower[active_index]) / scales[active_index],
        (upper[active_index] - parameters[active_index]) / scales[active_index],
    )
    expected_status = "FIT" if expected_stage == "joint" else "STAGE_CONDITIONED"
    model_limited = document.get("status") == "MODEL_LIMITED_FIT" and (
        expected_stage != "joint" or allow_model_limited_joint
    )
    expected_bound_parameters = tuple(
        name
        for name, distance in zip(active, scaled_bound_distance, strict=True)
        if distance <= bound_proximity
    )
    reported_bound_parameters = tuple(document.get("parameters_on_bounds", ()))
    initializer_cubature_not_run = (
        expected_stage != "joint"
        and cubature.get("performed") is False
        and cubature.get("status") == "NOT_RUN_INITIALIZER"
        and cubature.get("relative_l2") is None
        and cubature.get("relative_l2_by_family_m") == {}
        and "relative_l2_background_anchor_rows" in cubature
        and cubature.get("relative_l2_background_anchor_rows") is None
        and numerical.get("cubature_converged_by_family") == "NOT_RUN"
    )
    cubature_contract_matches = all(
        cubature.get(name) == value for name, value in trusted_cubature.items()
    )
    execution_cubature_matches = all(
        execution.get(name) == trusted_cubature[name]
        for name in (
            "fit_gauss_order",
            "oracle_gauss_order",
            "fit_subdivision_count",
            "oracle_subdivision_count",
        )
    )
    joint_cubature_state = bool(
        expected_stage == "joint"
        and _joint_cubature_is_admissible(cubature, trusted_recipe=trusted_recipe)
    )
    return bool(
        document.get("schema_version") == FIT_SCHEMA
        and expected_stage in {"A", "B", "C", "joint"}
        and document.get("stage") == expected_stage
        and (document.get("status") == expected_status or model_limited)
        and tuple(document.get("active_parameter_names", ())) == active
        and tuple(document.get("frozen_parameter_names", ())) == frozen
        and document.get("optimizer", {}).get("success") is True
        and dataset_ids == trusted_dataset_ids
        and all(math.isfinite(value) and value > 0.0 for value in dataset_scale_values)
        and int(document.get("model_rod_count", -1)) == len(rod_roster)
        and document.get("model_rod_scope") == "fitted_families_m_0_1_3_4"
        and document.get("model_rod_roster_sha256") == rod_roster_sha256
        and rod_families == set(FAMILIES)
        and document.get("stacking_model") == _fault_free_three_r_definition()
        and _fixed_displacement_gauge_is_admissible(structure)
        and _vacancy_structure_representative_is_admissible(structure)
        and execution.get("rod_scope") == "families_m_0_1_3_4"
        and execution.get("rod_count") == len(rod_roster)
        and execution.get("rod_roster_sha256") == rod_roster_sha256
        and numerical.get("converged") is (not model_limited)
        and numerical.get("optimizer_converged") is True
        and numerical.get("identifiable") is True
        and (initializer_cubature_not_run or joint_cubature_state)
        and cubature_contract_matches
        and execution_cubature_matches
        and (expected_stage != "joint" or numerical.get("cubature_converged_by_family") is True)
        and rank == len(active)
        and numerical_rank == len(active)
        and parameter_count == len(active)
        and tuple(sensitivity.get("parameter_names", ())) == active
        and np.array_equal(
            np.asarray(sensitivity.get("parameter_scales"), dtype=np.float64),
            scales[active_index],
        )
        and sensitivity.get("data_only") is True
        and sensitivity.get("parameter_scaled") is True
        and math.isfinite(condition)
        and math.isfinite(maximum_condition)
        and 1.0 <= condition <= maximum_condition
        and math.isfinite(relative_tolerance)
        and relative_tolerance == expected_tolerance
        and np.all(parameters >= lower)
        and np.all(parameters <= upper)
        and np.array_equal(reported_parameters, parameters)
        and (
            (
                model_limited
                and bool(expected_bound_parameters)
                and reported_bound_parameters == expected_bound_parameters
            )
            or (
                not model_limited
                and np.all(scaled_bound_distance > bound_proximity)
                and reported_bound_parameters == ()
            )
        )
        and recorded_scales.shape == parameters.shape
        and recorded_lower.shape == parameters.shape
        and recorded_upper.shape == parameters.shape
        and np.array_equal(recorded_scales, scales)
        and np.array_equal(recorded_lower, lower)
        and np.array_equal(recorded_upper, upper)
        and recorded_bound_proximity == bound_proximity
        and document.get("diagnostic_sha256") == diagnostic_sha256
        and provenance.get("recipe", {}).get("sha256") == recipe_sha256
        and provenance.get("fit_plan", {}).get("sha256") == fit_plan_sha256
        and provenance.get("fit_adapter", {}).get("sha256") == adapter_sha256
        and provenance.get("implementation", {}).get("sha256") == implementation_sha256
        and execution.get("diagnostic_sha256") == diagnostic_sha256
        and execution.get("recipe_sha256") == recipe_sha256
        and execution.get("fit_plan_sha256") == fit_plan_sha256
        and execution.get("fit_adapter_sha256") == adapter_sha256
        and execution.get("implementation_sha256") == implementation_sha256
        and execution.get("background_artifact_sha256")
        == provenance.get("background", {}).get("sha256")
        == background_model.get("artifact_sha256")
        and execution.get("radial_background_state_revision")
        == background_model.get("state_revision")
        and execution.get("radial_background_parameter_vector_sha256")
        == background_model.get("parameter_vector_sha256")
        and execution.get("radial_background_parameter_covariance_sha256")
        == background_model.get("parameter_covariance_sha256")
        and execution.get("radial_background_mass_sha256")
        == background_model.get("radial_mass_sha256")
        and execution.get("radial_background_covariance_sha256")
        == background_model.get("radial_covariance_sha256")
        and execution.get("background_excluded_flat_pixel_count_by_dataset")
        == background_model.get("excluded_flat_pixel_count_by_dataset")
        and execution.get("background_excluded_flat_pixel_sha256_by_dataset")
        == background_model.get("excluded_flat_pixel_sha256_by_dataset")
        and _background_model_identity_is_admissible(
            background_model,
            expected_dataset_ids=dataset_ids,
        )
        and execution.get("conditioned_background_revision")
        == background_model.get("conditioned_revision")
        and execution.get("conditioned_background_mass_sha256")
        == background_model.get("conditioned_mass_sha256")
        and execution.get("conditioned_background_covariance_sha256")
        == background_model.get("conditioned_covariance_sha256")
        and execution.get("conditioned_model_anchor_projection_sha256")
        == background_model.get("anchor_projection_sha256")
        and execution.get("model_measure") == "continuous_detector_chart_area"
        and execution.get("stage") == expected_stage
        and tuple(execution.get("active_parameter_names", ())) == active
        and tuple(execution.get("frozen_parameter_names", ())) == frozen
        and _fit_start_is_admissible(
            document,
            expected_stage=expected_stage,
            active_parameter_names=active,
            lower_bounds=lower,
            upper_bounds=upper,
            execution_identity=execution,
            final_parameters=parameters,
        )
        and document.get("model_pixelized") is False
        and document.get("model_measure") == "continuous_detector_chart_area"
        and document.get("smoothing_applied") is False
        and _integrated_peak_objective_is_admissible(document, execution)
        and _data_projection_is_admissible(
            data_projection,
            trusted_recipe=trusted_recipe,
            expected_dataset_ids=dataset_ids,
        )
        and (
            not _uses_native_pixel_center_observations(trusted_recipe)
            or _native_observation_execution_is_admissible(data_projection, execution)
        )
    )


def fit_document_is_admissible(
    document: dict[str, Any],
    *,
    trusted_recipe: dict[str, Any],
    recipe_sha256: str,
    adapter_sha256: str,
    fit_plan_sha256: str,
    implementation_sha256: str,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    parameter_scales: Sequence[float],
    sensitivity_relative_tolerance: float,
    bound_proximity_in_parameter_scales: float,
    maximum_sensitivity_condition: float,
    expected_execution_policy: str = "seeded_joint_only.v1",
    allow_model_limited_joint: bool = False,
) -> bool:
    """Return whether a saved joint fit is locally admissible under the expected policy."""

    try:
        diagnostic_sha256 = str(document["diagnostic_sha256"])
    except (KeyError, TypeError, ValueError):
        return False
    return document.get(
        "execution_policy"
    ) == expected_execution_policy and stage_fit_document_is_admissible(
        document,
        expected_stage="joint",
        expected_active_parameter_names=STRUCTURE_PARAMETER_NAMES,
        trusted_recipe=trusted_recipe,
        diagnostic_sha256=diagnostic_sha256,
        recipe_sha256=recipe_sha256,
        fit_plan_sha256=fit_plan_sha256,
        adapter_sha256=adapter_sha256,
        implementation_sha256=implementation_sha256,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        parameter_scales=parameter_scales,
        sensitivity_relative_tolerance=sensitivity_relative_tolerance,
        bound_proximity_in_parameter_scales=bound_proximity_in_parameter_scales,
        maximum_sensitivity_condition=maximum_sensitivity_condition,
        allow_model_limited_joint=allow_model_limited_joint,
    )


def _qualify_stage_chain_documents(
    document: dict[str, Any],
    identity: dict[str, str],
    *,
    expected_stage: str,
    trusted_recipe: dict[str, Any],
    fit_plan: dict[str, Any],
    diagnostic_sha256: str,
    recipe_sha256: str,
    fit_plan_sha256: str,
    adapter_sha256: str,
    implementation_sha256: str,
    load_predecessor: Callable[[Any, str], tuple[dict[str, Any], dict[str, str]]],
    load_restart: Callable[[Any, str], tuple[dict[str, Any], dict[str, str]]],
    allow_model_limited_joint: bool = False,
) -> tuple[tuple[dict[str, Any], dict[str, str]], ...]:
    """Qualify exact staged documents independently of their storage backend."""

    lower = np.asarray(fit_plan["lower_bounds"], dtype=np.float64)
    upper = np.asarray(fit_plan["upper_bounds"], dtype=np.float64)
    scales = np.asarray(fit_plan["parameter_scales"], dtype=np.float64)
    sensitivity_tolerance = float(fit_plan["sensitivity_relative_tolerance"])
    bound_proximity = float(fit_plan["bound_proximity_in_parameter_scales"])
    maximum_condition = float(fit_plan["maximum_sensitivity_condition"])
    execution_policy = str(fit_plan["execution_policy"])
    continuous_quadrature = fit_plan["continuous_quadrature"]
    expected_axial_refinement = int(continuous_quadrature["offspecular_axial_refinement"])
    expected_radial_transform = str(continuous_quadrature["offspecular_radial_transform"])
    expected_minimum_radial_nodes = int(
        continuous_quadrature["offspecular_signal_minimum_radial_nodes_per_side"]
    )
    current_document = document
    current_identity = identity
    current_stage = expected_stage
    child_document: dict[str, Any] | None = None
    seen_paths: set[str] = set()
    chain: list[tuple[dict[str, Any], dict[str, str]]] = []
    while True:
        if current_identity["path"] in seen_paths:
            raise ValueError("structure fit predecessor chain contains a cycle")
        seen_paths.add(current_identity["path"])
        active = tuple(fit_plan["stage"][current_stage]["active_parameters"])
        current_execution = current_document.get("provenance", {}).get("execution_identity", {})
        if (
            current_document.get("execution_policy") != execution_policy
            or current_execution.get("offspecular_axial_refinement") != expected_axial_refinement
            or current_execution.get("offspecular_radial_transform") != expected_radial_transform
            or current_execution.get("offspecular_signal_minimum_radial_nodes_per_side")
            != expected_minimum_radial_nodes
            or not stage_fit_document_is_admissible(
                current_document,
                expected_stage=current_stage,
                expected_active_parameter_names=active,
                trusted_recipe=trusted_recipe,
                diagnostic_sha256=diagnostic_sha256,
                recipe_sha256=recipe_sha256,
                fit_plan_sha256=fit_plan_sha256,
                adapter_sha256=adapter_sha256,
                implementation_sha256=implementation_sha256,
                lower_bounds=lower,
                upper_bounds=upper,
                parameter_scales=scales,
                sensitivity_relative_tolerance=sensitivity_tolerance,
                bound_proximity_in_parameter_scales=bound_proximity,
                maximum_sensitivity_condition=maximum_condition,
                allow_model_limited_joint=(allow_model_limited_joint and current_stage == "joint"),
            )
        ):
            raise ValueError(f"structure stage {current_stage} is not qualified")
        fit_start = current_document["optimizer"]["fit_start"]
        if fit_start.get("kind") == "progress_restart":
            restart_document, restart_identity = load_restart(
                fit_start.get("source_artifact"),
                current_stage,
            )
            if restart_identity != fit_start.get(
                "source_artifact"
            ) or not _restart_progress_is_admissible(
                restart_document,
                expected_execution_identity=current_execution,
                expected_initial_parameters=np.asarray(
                    fit_start["initial_parameters"], dtype=np.float64
                ),
                expected_initial_full_parameters=np.asarray(
                    fit_start["initial_full_parameters"], dtype=np.float64
                ),
            ):
                raise ValueError(f"structure stage {current_stage} restart source is not qualified")
        if child_document is not None:
            try:
                child_start = np.asarray(
                    child_document["optimizer"]["fit_start"]["initial_full_parameters"],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("structure fit child start is missing") from error
            child_stage = str(child_document.get("stage"))
            child_active = set(fit_plan["stage"][child_stage]["active_parameters"])
            child_frozen_index = np.asarray(
                [
                    index
                    for index, name in enumerate(STRUCTURE_PARAMETER_NAMES)
                    if name not in child_active
                ],
                dtype=np.int64,
            )
            predecessor_parameters = _fit_structure_vector(current_document)
            if not np.array_equal(
                child_start[child_frozen_index],
                predecessor_parameters[child_frozen_index],
            ):
                raise ValueError("structure fit child did not inherit its frozen predecessor state")
            child_start_kind = child_document["optimizer"]["fit_start"].get("kind")
            if child_start_kind == "default" and not np.array_equal(
                child_start,
                predecessor_parameters,
            ):
                raise ValueError("structure fit child did not start from its predecessor")
            child_final = _fit_structure_vector(child_document)
            if not np.array_equal(
                child_final[child_frozen_index],
                child_start[child_frozen_index],
            ):
                raise ValueError("structure fit child changed a frozen parameter")
        chain.append((current_document, current_identity))
        predecessor_stage = fit_plan["stage"][current_stage].get("predecessor")
        predecessor_record = current_document.get("provenance", {}).get("predecessor")
        if predecessor_stage is None:
            if predecessor_record is not None:
                raise ValueError("structure stage A must not name a predecessor")
            initial_full = np.asarray(
                current_document["optimizer"]["fit_start"]["initial_full_parameters"],
                dtype=np.float64,
            )
            baseline = np.asarray(fit_plan["baseline_parameters"], dtype=np.float64)
            start_kind = current_document["optimizer"]["fit_start"].get("kind")
            if start_kind == "default" and not np.array_equal(initial_full, baseline):
                raise ValueError("structure stage A did not start from the plan baseline")
            if start_kind == "progress_restart":
                active_names = set(fit_plan["stage"][current_stage]["active_parameters"])
                frozen_index = np.asarray(
                    [
                        index
                        for index, name in enumerate(STRUCTURE_PARAMETER_NAMES)
                        if name not in active_names
                    ],
                    dtype=np.int64,
                )
                if not np.array_equal(initial_full[frozen_index], baseline[frozen_index]):
                    raise ValueError(
                        "structure stage A restart changed a frozen baseline parameter"
                    )
            break
        current_document, current_identity = load_predecessor(
            predecessor_record,
            str(predecessor_stage),
        )
        child_document = chain[-1][0]
        current_stage = str(predecessor_stage)
    for index, (stage_document, _) in enumerate(chain):
        expected_chain = [identity for _, identity in chain[index + 1 :]]
        expected_predecessor = expected_chain[0] if expected_chain else None
        provenance = stage_document.get("provenance", {})
        execution = provenance.get("execution_identity", {})
        if (
            provenance.get("predecessor") != expected_predecessor
            or provenance.get("predecessor_chain") != expected_chain
            or execution.get("predecessor_sha256")
            != (None if expected_predecessor is None else expected_predecessor["sha256"])
            or execution.get("predecessor_chain_sha256")
            != [identity["sha256"] for identity in expected_chain]
        ):
            raise ValueError("structure fit predecessor lineage is internally inconsistent")
    objective_binding = (
        chain[0][0].get("provenance", {}).get("background"),
        chain[0][0].get("background_model"),
        chain[0][0].get("data_projection"),
        chain[0][0].get("model_rod_roster_sha256"),
    )
    if any(
        (
            stage_document.get("provenance", {}).get("background"),
            stage_document.get("background_model"),
            stage_document.get("data_projection"),
            stage_document.get("model_rod_roster_sha256"),
        )
        != objective_binding
        for stage_document, _ in chain[1:]
    ):
        raise ValueError("structure stages do not share one frozen observation objective")
    return tuple(chain)


def _load_qualified_stage_chain(
    path: Path,
    *,
    expected_stage: str,
    trusted_recipe: dict[str, Any],
    fit_plan: dict[str, Any],
    diagnostic_sha256: str,
    recipe_sha256: str,
    fit_plan_sha256: str,
    adapter_sha256: str,
    implementation_sha256: str,
    allow_model_limited_joint: bool = False,
) -> tuple[tuple[dict[str, Any], dict[str, str]], ...]:
    """Load and qualify one exact, acyclic staged-fit predecessor chain."""

    identity = _file_identity(path)
    _require_unchanged(identity, role=f"structure stage {expected_stage}")
    document = json.loads(Path(identity["path"]).read_text(encoding="utf-8"))

    def load_predecessor(
        record: Any,
        stage: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        predecessor_identity = _verified_recorded_file_identity(
            record,
            role=f"structure stage {stage}",
        )
        predecessor_document = json.loads(
            Path(predecessor_identity["path"]).read_text(encoding="utf-8")
        )
        return predecessor_document, predecessor_identity

    def load_restart(
        record: Any,
        stage: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        restart_identity = _verified_recorded_file_identity(
            record,
            role=f"structure stage {stage} restart",
        )
        restart_document = json.loads(Path(restart_identity["path"]).read_text(encoding="utf-8"))
        return restart_document, restart_identity

    return _qualify_stage_chain_documents(
        document,
        identity,
        expected_stage=expected_stage,
        trusted_recipe=trusted_recipe,
        fit_plan=fit_plan,
        diagnostic_sha256=diagnostic_sha256,
        recipe_sha256=recipe_sha256,
        fit_plan_sha256=fit_plan_sha256,
        adapter_sha256=adapter_sha256,
        implementation_sha256=implementation_sha256,
        load_predecessor=load_predecessor,
        load_restart=load_restart,
        allow_model_limited_joint=allow_model_limited_joint,
    )


def _profile_evidence_policy(
    *,
    fit_document: dict[str, Any],
    trusted_recipe: dict[str, Any],
    recipe_sha256: str,
    adapter_sha256: str,
    fit_plan_sha256: str,
    implementation_sha256: str,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    parameter_scales: Sequence[float],
    sensitivity_relative_tolerance: float,
    bound_proximity_in_parameter_scales: float,
    maximum_sensitivity_condition: float,
    expected_dataset_ids: Sequence[str],
    predecessor_chain_complete: bool,
    expected_execution_policy: str = "seeded_joint_only.v1",
) -> dict[str, Any]:
    provenance = fit_document.get("provenance", {})
    execution = provenance.get("execution_identity", {})
    structure = fit_document.get("structure_representative", {})
    scales = fit_document.get("dataset_scales", {})
    try:
        structure_values = tuple(float(structure[name]) for name in STRUCTURE_PARAMETER_NAMES)
        dataset_ids = tuple(str(value) for value in expected_dataset_ids)
        if not isinstance(scales, dict) or set(scales) != set(dataset_ids):
            raise ValueError("fit dataset scales differ from the prepared datasets")
        scale_values = tuple(float(scales[value]) for value in dataset_ids)
    except (AttributeError, KeyError, TypeError, ValueError):
        structure_values = ()
        scale_values = ()
    fit_is_complete = predecessor_chain_complete and fit_document_is_admissible(
        fit_document,
        trusted_recipe=trusted_recipe,
        recipe_sha256=recipe_sha256,
        adapter_sha256=adapter_sha256,
        fit_plan_sha256=fit_plan_sha256,
        implementation_sha256=implementation_sha256,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        parameter_scales=parameter_scales,
        sensitivity_relative_tolerance=sensitivity_relative_tolerance,
        bound_proximity_in_parameter_scales=bound_proximity_in_parameter_scales,
        maximum_sensitivity_condition=maximum_sensitivity_condition,
        expected_execution_policy=expected_execution_policy,
        allow_model_limited_joint=True,
    )
    if (
        not fit_is_complete
        or fit_document.get("model_pixelized") is not False
        or fit_document.get("smoothing_applied") is not False
        or execution.get("rod_scope") != "families_m_0_1_3_4"
        or execution.get("model_measure") != "continuous_detector_chart_area"
        or len(structure_values) != STRUCTURE_PARAMETER_COUNT
        or not all(math.isfinite(value) for value in structure_values)
        or not 0.0 <= structure_values[2] <= 1.0
        or not all(value >= 0.0 for value in structure_values[3:])
        or not scale_values
        or not all(math.isfinite(value) and value > 0.0 for value in scale_values)
    ):
        raise ValueError("FIT_CONDITIONED profiles require a complete continuous fitted-scope fit")
    return {
        "evidence_level": "FIT_CONDITIONED",
        "publication_ready": False,
        "all_rod_validation": "NOT_REQUIRED_FOR_RENDER",
        "model_rod_scope": "fitted_families_m_0_1_3_4",
        "pass_names": ("fit",),
        "result_pass_name": "fit",
    }


def _load_recipe(path: Path) -> dict[str, Any]:
    recipe_path = path.resolve()
    return _validated_recipe(tomllib.loads(recipe_path.read_text(encoding="utf-8")))


def _recipe_parratt_stitch(recipe: dict[str, Any]) -> ParrattStitchStack | None:
    values = recipe.get("parratt_stitch")
    if values is None:
        return None
    return ParrattStitchStack(
        substrate_refractive_index=complex(
            float(values["substrate_refractive_index_real"]),
            float(values["substrate_refractive_index_imag"]),
        ),
        top_roughness_A=float(values["top_roughness_A"]),
        bottom_roughness_A=float(values["bottom_roughness_A"]),
        model_id=str(values["model_id"]),
        interface_assumption=str(values["interface_assumption"]),
    )


def _profile_recipe_with_specular_interface(
    recipe: dict[str, Any],
    interface_assumption: str | None,
) -> dict[str, Any]:
    """Return the fit recipe or its one-field profile-only stitch override."""

    if interface_assumption is None:
        return recipe
    if interface_assumption not in SPECULAR_INTERFACE_ASSUMPTIONS:
        raise ValueError("unsupported profile specular interface assumption")
    stitch = recipe.get("parratt_stitch")
    if not isinstance(stitch, dict):
        raise ValueError("profile specular replay requires a fitted Parratt stitch")
    return {
        **recipe,
        "parratt_stitch": {
            **stitch,
            "interface_assumption": interface_assumption,
        },
    }


def _validate_position_dataset_binding(
    position: FixedPositionState,
    datasets: Sequence[dict[str, Any]],
) -> tuple[float, ...]:
    """Bind per-image trims to the exact OSC ordering used downstream."""

    incidences = tuple(float(item["incidence_angle_deg"]) for item in datasets)
    commanded = np.asarray(position.commanded_incidence_angles_rad, dtype=np.float64)
    if commanded.shape != (len(datasets),) or not np.allclose(
        commanded,
        np.radians(incidences),
        rtol=0.0,
        atol=2.0e-14,
    ):
        raise ValueError("fixed-experiment commanded incidences differ from its dataset roster")
    if position.incidence_angle_image_ids:
        declared_ids = tuple(item.get("position_image_id") for item in datasets)
        if any(value is not None for value in declared_ids) and not all(
            isinstance(value, str) and value for value in declared_ids
        ):
            raise ValueError("fixed-experiment position image IDs are only partially declared")
        if all(value is None for value in declared_ids):
            position_image_ids = tuple(
                Path(str(item["osc"]["path"])).name.removesuffix(".osc.gz") for item in datasets
            )
        else:
            position_image_ids = tuple(str(value) for value in declared_ids)
        if position_image_ids != position.incidence_angle_image_ids:
            raise ValueError("fixed-experiment OSC order differs from the fitted position trims")
    return incidences


def _fixed_experiment_inputs(
    path: Path,
    *,
    expected_material_id: str,
    expected_dataset_ids: Sequence[str],
    expected_source_state_count: int,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    dict[str, str],
    tuple[Any, ...],
    Any,
    FixedLatticeState,
    dict[str, float],
    list[dict[str, Any]],
    str,
    str,
    str,
]:
    """Validate and rebuild one material-neutral position/mosaic checkpoint."""

    fixed_state_path = path.resolve()
    identity = _file_identity(fixed_state_path)
    document = json.loads(fixed_state_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("fixed-experiment state must be a mapping")
    if document.get("schema_version") != FIXED_EXPERIMENT_SCHEMA:
        raise ValueError("unsupported fixed-experiment state")
    if document.get("material_id") != expected_material_id:
        raise ValueError("fixed-experiment material differs from the recipe")
    if document.get("status") != "FIXED_EXPERIMENT" or document.get("position_status") not in {
        "POSITION_QUALIFIED",
        "POSITION_MODEL_LIMITED",
    }:
        raise ValueError("fixed-experiment predecessor state is incomplete")
    if document.get("model_pixelized") is not False:
        raise ValueError("fixed-experiment model must remain continuous")
    if document.get("smoothing_applied") is not False:
        raise ValueError("fixed-experiment data/model handoff must remain unsmoothed")
    recorded_source_count = document.get("source_state_count")
    if (
        isinstance(recorded_source_count, bool)
        or not isinstance(recorded_source_count, int)
        or recorded_source_count != expected_source_state_count
    ):
        raise ValueError("fixed-experiment source count differs from the requested count")

    config_record = document.get("simulation_config")
    if not isinstance(config_record, dict) or set(config_record) != {"path", "sha256"}:
        raise ValueError("fixed-experiment simulation configuration identity is invalid")
    simulation_identity = _verified_recorded_file_identity(
        config_record,
        role="fixed-experiment simulation configuration",
    )
    simulation_config_path = Path(simulation_identity["path"])

    datasets = document.get("datasets")
    if (
        not isinstance(datasets, list)
        or len(datasets) != len(expected_dataset_ids)
        or any(not isinstance(item, dict) for item in datasets)
    ):
        raise ValueError("fixed-experiment dataset roster is invalid")
    dataset_ids = tuple(str(item.get("dataset_id", "")) for item in datasets)
    if dataset_ids != tuple(expected_dataset_ids):
        raise ValueError("fixed-experiment dataset order differs from the recipe")
    observations: list[dict[str, Any]] = []
    for item in datasets:
        osc = item.get("osc")
        if not isinstance(osc, dict) or set(osc) != {"path", "sha256"}:
            raise ValueError("fixed-experiment OSC identity is invalid")
        osc_identity = _verified_recorded_file_identity(
            osc,
            role=f"fixed-experiment OSC {item['dataset_id']}",
        )
        osc_path = Path(osc_identity["path"])
        observations.append(
            {
                "dataset_id": str(item["dataset_id"]),
                "incidence_angle_deg": float(item["incidence_angle_deg"]),
                "osc_file": str(osc_path),
                "osc_file_sha256": str(osc["sha256"]),
                "detector_native_bytes_sha256": str(item["detector_native_bytes_sha256"]),
                "detector_native_dtype": str(item["detector_native_dtype"]),
                "detector_native_shape_rc": list(item["detector_native_shape_rc"]),
            }
        )

    position = FixedPositionState.from_record(document.get("fixed_position"))
    _validate_position_dataset_binding(position, datasets)
    lattice_record = document.get("fixed_lattice")
    if not isinstance(lattice_record, dict):
        raise ValueError("fixed-experiment lattice state is missing")
    lattice = FixedLatticeState.from_record(
        lattice_record,
        reference_direct_basis_A=lattice_record.get("reference_direct_basis_A"),
    )
    mosaic = FixedMosaicState.from_record(document.get("fixed_mosaic"))
    mosaic_parameters = {
        "gaussian_sigma_deg": mosaic.gaussian_sigma_deg,
        "lorentzian_hwhm_deg": mosaic.lorentzian_hwhm_deg,
        "lorentzian_probability": mosaic.lorentzian_probability,
    }

    series = build_fixed_experiment_series(
        load_simulation_config(simulation_config_path),
        position=position,
        fixed_lattice=lattice,
        source_sample_count=expected_source_state_count,
        gaussian_sigma_rad=math.radians(mosaic_parameters["gaussian_sigma_deg"]),
        lorentzian_half_width_rad=math.radians(mosaic_parameters["lorentzian_hwhm_deg"]),
        lorentzian_probability=mosaic_parameters["lorentzian_probability"],
    )
    source_revision = str(series[0].samples.source_revision)
    cif_sha256 = str(series[0].config.cif_sha256)
    rod_catalog_revision = configured_rod_catalog_revision(series[0])
    if (
        document.get("source_revision") != source_revision
        or document.get("cif_sha256") != cif_sha256
        or document.get("rod_catalog_revision") != rod_catalog_revision
        or not np.array_equal(
            series[0].crystal.direct_basis_A,
            lattice.active_direct_basis_A,
        )
    ):
        raise ValueError("fixed-experiment rebuilt physics differs from its checkpoint")
    return (
        document,
        identity,
        simulation_identity,
        series,
        position,
        lattice,
        mosaic_parameters,
        observations,
        source_revision,
        cif_sha256,
        rod_catalog_revision,
    )


def _validated_fit_plan(document: dict[str, Any]) -> dict[str, Any]:
    schema_version = document.get("schema_version")
    if schema_version not in {
        "rasim-layered-quintuple-structure-fit-plan-v7",
        "rasim-layered-quintuple-structure-fit-plan-v8",
    }:
        raise ValueError("unsupported layered-quintuple structure fit plan")
    if not isinstance(document.get("material_id"), str) or not document["material_id"]:
        raise ValueError("structure fit plan requires a material_id")
    if tuple(document.get("parameter_names", ())) != STRUCTURE_PARAMETER_NAMES:
        raise ValueError("structure fit parameter_names changed")
    vectors: dict[str, np.ndarray] = {}
    for name in (
        "baseline_parameters",
        "lower_bounds",
        "upper_bounds",
        "parameter_scales",
        "prior_mean",
        "prior_sigma",
    ):
        values = np.asarray(document.get(name, ()), dtype=np.float64)
        if values.shape != (STRUCTURE_PARAMETER_COUNT,) or np.any(~np.isfinite(values)):
            raise ValueError(f"{name} must contain five finite values")
        vectors[name] = values
    if (
        np.any(vectors["lower_bounds"] >= vectors["upper_bounds"])
        or np.any(vectors["parameter_scales"] <= 0.0)
        or np.any(vectors["prior_sigma"] <= 0.0)
        or np.any(vectors["baseline_parameters"] < vectors["lower_bounds"])
        or np.any(vectors["baseline_parameters"] > vectors["upper_bounds"])
        or np.any(vectors["prior_mean"] < vectors["lower_bounds"])
        or np.any(vectors["prior_mean"] > vectors["upper_bounds"])
    ):
        raise ValueError("structure fit bounds, scales, priors, or baseline are invalid")
    if (
        vectors["lower_bounds"][2] < 0.0
        or vectors["upper_bounds"][2] > 1.0
        or np.any(vectors["lower_bounds"][3:] < 0.0)
    ):
        raise ValueError("occupancy and event-envelope bounds are physically invalid")
    displacement = document.get("site_displacement_profile")
    site_records = displacement.get("site") if isinstance(displacement, dict) else None
    expected_roles = ("bi", "central_chalcogen", "outer_chalcogen")
    if (
        not isinstance(displacement, dict)
        or displacement.get("model_id") != "transverse_isotropic_site_reference_fixed.v1"
        or not isinstance(displacement.get("provenance"), str)
        or not displacement["provenance"]
        or not isinstance(site_records, list)
        or tuple(record.get("role") for record in site_records) != expected_roles
    ):
        raise ValueError("site_displacement_profile must declare the three ordered site roles")
    for record in site_records:
        values = np.asarray(
            (record.get("u_radial_A2"), record.get("u_normal_A2")),
            dtype=np.float64,
        )
        if values.shape != (2,) or np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("site displacement components must be finite and nonnegative")
    gauge = document.get("displacement_gauge")
    if (
        not isinstance(gauge, dict)
        or gauge.get("model_id") != "fixed_site_adp_plus_regularized_sample_q_envelope.v1"
        or gauge.get("site_adp_common_mode") != "fixed_reference"
        or gauge.get("sample_q_envelope_mode") != "fit_zero_centered_regularized"
        or not np.array_equal(vectors["prior_mean"][3:], np.zeros(2))
        or np.any(vectors["prior_sigma"][3:] <= 0.0)
    ):
        raise ValueError("structure fit plan has an invalid site-ADP/global-envelope gauge")
    sensitivity_relative_tolerance = float(document.get("sensitivity_relative_tolerance", 0.0))
    maximum_sensitivity_condition = float(document.get("maximum_sensitivity_condition", 0.0))
    bound_proximity = float(document.get("bound_proximity_in_parameter_scales", 0.0))
    if (
        not math.isfinite(sensitivity_relative_tolerance)
        or not 0.0 < sensitivity_relative_tolerance < 1.0
        or not math.isfinite(maximum_sensitivity_condition)
        or maximum_sensitivity_condition < 1.0
        or not math.isclose(
            maximum_sensitivity_condition * sensitivity_relative_tolerance,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isfinite(bound_proximity)
        or not 0.0 < bound_proximity < 1.0
    ):
        raise ValueError("structure fit identifiability or bound-proximity policy is invalid")
    quadrature = document.get("continuous_quadrature")
    if not isinstance(quadrature, dict):
        raise ValueError("structure fit plan requires continuous_quadrature settings")
    axial_refinement = quadrature.get("offspecular_axial_refinement")
    radial_transform = quadrature.get("offspecular_radial_transform")
    minimum_radial_nodes = quadrature.get("offspecular_signal_minimum_radial_nodes_per_side")
    if (
        isinstance(axial_refinement, bool)
        or not isinstance(axial_refinement, (int, np.integer))
        or int(axial_refinement) < 1
        or radial_transform != "squared_fold_coordinate.v1"
        or isinstance(minimum_radial_nodes, bool)
        or not isinstance(minimum_radial_nodes, (int, np.integer))
        or int(minimum_radial_nodes) < 1
    ):
        raise ValueError("continuous_quadrature settings are invalid")
    stages = document.get("stage")
    if schema_version == "rasim-layered-quintuple-structure-fit-plan-v7":
        expected = STRUCTURE_STAGE_PARAMETERS
        predecessors = {"A": None, "B": "A", "C": "B", "joint": "C"}
        execution_policy = "staged_A_B_C_joint.v1"
        if (
            document.get("execution_policy", execution_policy) != execution_policy
            or not isinstance(stages, dict)
            or tuple(stages) != tuple(expected)
        ):
            raise ValueError("structure fit stages must be A, B, C, joint in order")
    else:
        expected = {"joint": STRUCTURE_PARAMETER_NAMES}
        predecessors = {"joint": None}
        execution_policy = "seeded_joint_only.v1"
        if (
            document.get("execution_policy") != execution_policy
            or not isinstance(stages, dict)
            or tuple(stages) != ("joint",)
        ):
            raise ValueError("seeded joint-only fit plan must contain only the joint stage")
    for stage, active in expected.items():
        record = stages.get(stage)
        if (
            not isinstance(record, dict)
            or tuple(record.get("active_parameters", ())) != active
            or record.get("predecessor") != predecessors[stage]
        ):
            raise ValueError(f"structure fit stage {stage!r} changed")
    document["execution_policy"] = execution_policy
    return document


def _load_fit_plan(path: Path) -> dict[str, Any]:
    plan_path = path.resolve()
    return _validated_fit_plan(tomllib.loads(plan_path.read_text(encoding="utf-8")))


def _require_declared_fit_start(
    execution_policy: str,
    *,
    initial_parameters_override: Sequence[float] | None,
    resume_path: Path | None,
) -> None:
    if execution_policy == "staged_A_B_C_joint.v1" and initial_parameters_override is not None:
        raise ValueError("staged fit starts are fixed by the baseline or predecessor")
    if (
        execution_policy == "seeded_joint_only.v1"
        and initial_parameters_override is None
        and resume_path is None
    ):
        raise ValueError("seeded joint-only fit requires explicit initial parameters or a restart")


def _m0_region(recipe: dict[str, Any]) -> SpecularAngularProfileRegion:
    settings = recipe["m0_region"]
    minimum = float(settings["two_theta_minimum_deg"])
    maximum = float(settings["two_theta_maximum_deg"])
    width = float(settings["two_theta_bin_width_deg"])
    bin_count = round((maximum - minimum) / width)
    edges = minimum + width * np.arange(bin_count + 1, dtype=np.float64)
    if not np.isclose(edges[-1], maximum, rtol=0.0, atol=1.0e-12):
        raise ValueError("m0 two-theta bounds must contain an integer number of bins")
    return SpecularAngularProfileRegion(
        identity="m0",
        two_theta_bin_edges_rad=np.deg2rad(edges),
        phi_signal_interval_rad=tuple(np.deg2rad(settings["phi_signal_interval_deg"])),
        phi_background_intervals_rad=tuple(
            tuple(np.deg2rad(interval)) for interval in settings["phi_background_intervals_deg"]
        ),
    )


def _offspecular_layouts(
    recipe: dict[str, Any],
    beam_center_column_px: float,
    detector_column_count: int,
) -> tuple[tuple[str, OffSpecularBandLayout], ...]:
    sides = (
        ("plus", (-0.5, beam_center_column_px)),
        ("minus", (beam_center_column_px, float(detector_column_count) - 0.5)),
    )
    result: list[tuple[str, OffSpecularBandLayout]] = []
    for group in recipe["offspecular_group"]:
        minimum = float(group["L_minimum"])
        maximum = float(group["L_maximum"])
        width = float(group["L_bin_width"])
        edges = minimum + width * np.arange(
            round((maximum - minimum) / width) + 1,
            dtype=np.float64,
        )
        families = tuple(int(value) for value in group["families_m"])
        intervals = tuple(tuple(map(float, value)) for value in group["signal_bands_Ainv"])
        if len(families) != len(intervals):
            raise ValueError("offspecular families and signal bands must align")
        for side, columns in sides:
            result.append(
                (
                    f"{group['identity']}_{side}",
                    OffSpecularBandLayout(
                        axial_bin_edges=edges,
                        detector_column_interval_px=columns,
                        signal_bands=tuple(
                            OffSpecularBand(f"m{family}_{side}", interval)
                            for family, interval in zip(families, intervals, strict=True)
                        ),
                        background_intervals_Ainv=tuple(
                            tuple(map(float, value)) for value in group["background_intervals_Ainv"]
                        ),
                    ),
                )
            )
    return tuple(result)


def _serialized_regions(
    m0: SpecularAngularProfileRegion,
    layouts: tuple[tuple[str, OffSpecularBandLayout], ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    m0_record = {
        "identity": m0.identity,
        "two_theta_bin_edges_rad": np.asarray(m0.two_theta_bin_edges_rad).tolist(),
        "phi_signal_interval_rad": list(m0.phi_signal_interval_rad),
        "phi_background_intervals_rad": [
            list(interval) for interval in m0.phi_background_intervals_rad
        ],
    }
    layout_records = []
    for group, layout in layouts:
        layout_records.append(
            {
                "group": group,
                "axial_bin_edges": np.asarray(layout.axial_bin_edges).tolist(),
                "detector_column_interval_px": list(layout.detector_column_interval_px),
                "signal_bands": [
                    {
                        "identity": band.identity,
                        "qr_interval_Ainv": list(band.qr_interval_Ainv),
                    }
                    for band in layout.signal_bands
                ],
                "background_intervals_Ainv": [
                    list(interval) for interval in layout.background_intervals_Ainv
                ],
            }
        )
    return m0_record, layout_records


def _fit_window_mask(
    recipe: dict[str, Any],
    *,
    dataset_id: str,
    profile_identity: str,
    coordinate: np.ndarray,
) -> np.ndarray:
    selected = np.zeros(coordinate.shape, dtype=np.bool_)
    found = False
    for peak in recipe["fit_peak"]:
        if peak["dataset_id"] != dataset_id or peak["profile_identity"] != profile_identity:
            continue
        found = True
        centers = peak.get("centers", (peak.get("center"),))
        half_width = float(peak["half_width"])
        for center in centers:
            center_value = float(center)
            selected |= (coordinate >= center_value - half_width) & (
                coordinate <= center_value + half_width
            )
    if not found:
        return selected
    return selected


def _row_catalog(
    dataset_count: int,
    m0: SpecularAngularProfileRegion,
    layouts: tuple[tuple[str, OffSpecularBandLayout], ...],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str, int], int]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[int, str, str, int], int] = {}
    for dataset in range(dataset_count):
        for band in ("signal", "background_0", "background_1"):
            for bin_index in range(m0.bin_count):
                lookup[(dataset, "m0", band, bin_index)] = len(rows)
                rows.append(
                    {
                        "dataset": dataset,
                        "group": "m0",
                        "band": band,
                        "bin": bin_index,
                        "signal_family": 0 if band == "signal" else -1,
                        "coordinate_kind": "phi_rad",
                    }
                )
        for group, layout in layouts:
            bands = tuple(item.identity for item in layout.signal_bands) + tuple(
                f"background_{index}" for index in range(len(layout.background_intervals_Ainv))
            )
            for band in bands:
                family = (
                    int(band[1])
                    if band.startswith("m") and len(band) > 1 and band[1].isdigit()
                    else -1
                )
                for bin_index in range(layout.bin_count):
                    lookup[(dataset, group, band, bin_index)] = len(rows)
                    rows.append(
                        {
                            "dataset": dataset,
                            "group": group,
                            "band": band,
                            "bin": bin_index,
                            "signal_family": family,
                            "coordinate_kind": "qr_Ainv",
                        }
                    )
    return rows, lookup


def _add_members(
    *,
    mask_index: np.ndarray,
    group: str,
    band: str,
    dataset_index: int,
    lookup: dict[tuple[int, str, str, int], int],
    flat_pixel: np.ndarray,
    counts: np.ndarray,
    coordinate: np.ndarray,
    axial_L: np.ndarray,
    qz_Ainv: np.ndarray,
    count_sum: np.ndarray,
    support: np.ndarray,
    coordinate_sum: np.ndarray,
    axial_sum: np.ndarray,
    qz_sum: np.ndarray | None,
    selected_pixels: list[np.ndarray],
    selected_rows: list[np.ndarray],
    display_code: np.ndarray | None,
    code: int,
) -> None:
    selected = mask_index >= 0
    if not np.any(selected):
        return
    bins = mask_index[selected]
    row_id = np.fromiter(
        (lookup[(dataset_index, group, band, int(value))] for value in bins),
        dtype=np.int64,
        count=bins.size,
    )
    np.add.at(count_sum, row_id, counts[selected])
    np.add.at(support, row_id, 1.0)
    np.add.at(coordinate_sum, row_id, coordinate[selected])
    np.add.at(axial_sum, row_id, axial_L[selected])
    if qz_sum is not None:
        np.add.at(qz_sum, row_id, qz_Ainv[selected])
    selected_pixels.append(flat_pixel[selected])
    selected_rows.append(row_id)
    if display_code is not None:
        display_code[flat_pixel[selected]] = code


def _prepare_dataset_membership(
    inputs: Any,
    counts: np.ndarray,
    *,
    recipe: dict[str, Any],
    dataset_id: str,
    dataset_index: int,
    lookup: dict[tuple[int, str, str, int], int],
    m0: SpecularAngularProfileRegion,
    layouts: tuple[tuple[str, OffSpecularBandLayout], ...],
    count_sum: np.ndarray,
    support: np.ndarray,
    coordinate_sum: np.ndarray,
    axial_sum: np.ndarray,
    display_code: np.ndarray | None,
    row_chunk_size: int,
    qz_sum: np.ndarray | None = None,
    apply_fit_windows: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    context = build_nominal_ewald_context(inputs)
    frame = LayeredReciprocalFrame(
        reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
        sample_from_crystal_rotation=inputs.instrument.sample_from_crystal.rotation,
        axial_basis_index=2,
    )
    angle_frame = build_osc_angle_frame(
        mean_direction_lab=inputs.config.source.mean_direction_lab,
        instrument=inputs.instrument,
        sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
        revision=f"matched-regions-{dataset_index}.v1",
    )
    horizon_acceptance = _detector_horizon_acceptance(recipe["horizon_gate"])
    mean_wavelength_A = float(inputs.config.source.mean_wavelength_A)
    axial_basis_magnitude_Ainv = float(
        np.linalg.norm(np.asarray(inputs.reciprocal.basis_Ainv, dtype=np.float64)[:, 2])
    )
    rows, columns = counts.shape
    column = np.arange(columns, dtype=np.float64)
    selected_pixels: list[np.ndarray] = []
    selected_rows: list[np.ndarray] = []
    for row_start in range(0, rows, row_chunk_size):
        row_stop = min(rows, row_start + row_chunk_size)
        row = np.arange(row_start, row_stop, dtype=np.float64)
        column_grid, row_grid = np.broadcast_arrays(column[None, :], row[:, None])
        geometry = context.evaluate_detector_geometry(
            column_grid,
            row_grid,
            include_surface_jacobian=False,
        )
        qr_Ainv, axial_L = frame.coordinates(geometry.q_sample_Ainv)
        angles = detector_coordinates_to_angles(
            column_grid,
            row_grid,
            instrument=inputs.instrument,
            angle_frame=angle_frame,
        )
        kf_air = np.asarray(geometry.kf_air_sample_Ainv)
        air_exit_angle_rad = np.arctan2(
            kf_air[..., 2],
            np.hypot(kf_air[..., 0], kf_air[..., 1]),
        )
        angular_valid = angles.valid & angles.azimuth_valid
        specular_horizon_valid = horizon_acceptance.m0_detector_mask(air_exit_angle_rad)
        specular_valid = angular_valid & specular_horizon_valid
        offspecular_valid = (
            geometry.valid & angular_valid & horizon_acceptance.offspecular_mask(air_exit_angle_rad)
        )
        flat_pixel = (
            row_grid.astype(np.int64, copy=False) * columns
            + column_grid.astype(np.int64, copy=False)
        ).reshape(-1)
        flat_counts = counts[row_start:row_stop].reshape(-1).astype(np.float64)
        flat_L = np.asarray(axial_L).reshape(-1)
        flat_qz_Ainv = np.asarray(geometry.q_sample_Ainv)[..., 2].reshape(-1)
        flat_qr = np.asarray(qr_Ainv).reshape(-1)
        flat_phi = np.asarray(angles.phi_rad).reshape(-1)
        flat_two_theta_deg = np.rad2deg(np.asarray(angles.two_theta_rad).reshape(-1))
        flat_local_m0_qz_Ainv = (4.0 * np.pi / mean_wavelength_A) * np.sin(
            0.5 * np.asarray(angles.two_theta_rad).reshape(-1)
        )
        flat_local_m0_L = flat_local_m0_qz_Ainv / axial_basis_magnitude_Ainv
        m0_fit_window = (
            _fit_window_mask(
                recipe,
                dataset_id=dataset_id,
                profile_identity="m0",
                coordinate=flat_two_theta_deg,
            )
            if apply_fit_windows
            else np.ones(flat_two_theta_deg.shape, dtype=np.bool_)
        )
        m0_membership = specular_angular_membership(
            two_theta_rad=np.asarray(angles.two_theta_rad).reshape(-1),
            phi_rad=flat_phi,
            valid=np.asarray(specular_valid).reshape(-1),
            region=m0,
        )
        _add_members(
            mask_index=np.where(m0_fit_window, m0_membership.signal_bin_index, -1),
            group="m0",
            band="signal",
            dataset_index=dataset_index,
            lookup=lookup,
            flat_pixel=flat_pixel,
            counts=flat_counts,
            coordinate=flat_phi,
            axial_L=flat_local_m0_L,
            qz_Ainv=flat_local_m0_qz_Ainv,
            count_sum=count_sum,
            support=support,
            coordinate_sum=coordinate_sum,
            axial_sum=axial_sum,
            qz_sum=qz_sum,
            selected_pixels=selected_pixels,
            selected_rows=selected_rows,
            display_code=display_code,
            code=1,
        )
        for anchor_index, (lower, upper) in enumerate(m0.phi_background_intervals_rad):
            anchor = np.where(
                (m0_membership.background_bin_index >= 0)
                & m0_fit_window
                & (flat_phi >= lower)
                & (flat_phi < upper),
                m0_membership.background_bin_index,
                -1,
            )
            _add_members(
                mask_index=anchor,
                group="m0",
                band=f"background_{anchor_index}",
                dataset_index=dataset_index,
                lookup=lookup,
                flat_pixel=flat_pixel,
                counts=flat_counts,
                coordinate=flat_phi,
                axial_L=flat_local_m0_L,
                qz_Ainv=flat_local_m0_qz_Ainv,
                count_sum=count_sum,
                support=support,
                coordinate_sum=coordinate_sum,
                axial_sum=axial_sum,
                qz_sum=qz_sum,
                selected_pixels=selected_pixels,
                selected_rows=selected_rows,
                display_code=display_code,
                code=2,
            )
        flat_column = column_grid.reshape(-1)
        flat_valid = np.asarray(offspecular_valid).reshape(-1)
        for group, layout in layouts:
            membership = offspecular_region_membership(
                qr_Ainv=flat_qr,
                axial_coordinate=flat_L,
                detector_column_px=flat_column,
                valid=flat_valid,
                layout=layout,
            )
            for band in layout.signal_bands:
                family = int(band.identity[1])
                fit_window = (
                    _fit_window_mask(
                        recipe,
                        dataset_id=dataset_id,
                        profile_identity=f"m{family}",
                        coordinate=flat_L,
                    )
                    if apply_fit_windows
                    else np.ones(flat_L.shape, dtype=np.bool_)
                )
                _add_members(
                    mask_index=np.where(
                        fit_window,
                        membership.signal_bin_index[band.identity],
                        -1,
                    ),
                    group=group,
                    band=band.identity,
                    dataset_index=dataset_index,
                    lookup=lookup,
                    flat_pixel=flat_pixel,
                    counts=flat_counts,
                    coordinate=flat_qr,
                    axial_L=flat_L,
                    qz_Ainv=flat_qz_Ainv,
                    count_sum=count_sum,
                    support=support,
                    coordinate_sum=coordinate_sum,
                    axial_sum=axial_sum,
                    qz_sum=qz_sum,
                    selected_pixels=selected_pixels,
                    selected_rows=selected_rows,
                    display_code=display_code,
                    code={1: 3, 3: 5, 4: 6}[family],
                )
            group_fit_window = np.ones(flat_L.shape, dtype=np.bool_)
            if apply_fit_windows:
                group_fit_window.fill(False)
                for band in layout.signal_bands:
                    group_fit_window |= _fit_window_mask(
                        recipe,
                        dataset_id=dataset_id,
                        profile_identity=f"m{int(band.identity[1])}",
                        coordinate=flat_L,
                    )
            for anchor_index, (lower, upper) in enumerate(layout.background_intervals_Ainv):
                anchor = np.where(
                    (membership.background_bin_index >= 0)
                    & group_fit_window
                    & (flat_qr >= lower)
                    & (flat_qr < upper),
                    membership.background_bin_index,
                    -1,
                )
                _add_members(
                    mask_index=anchor,
                    group=group,
                    band=f"background_{anchor_index}",
                    dataset_index=dataset_index,
                    lookup=lookup,
                    flat_pixel=flat_pixel,
                    counts=flat_counts,
                    coordinate=flat_qr,
                    axial_L=flat_L,
                    qz_Ainv=flat_qz_Ainv,
                    count_sum=count_sum,
                    support=support,
                    coordinate_sum=coordinate_sum,
                    axial_sum=axial_sum,
                    qz_sum=qz_sum,
                    selected_pixels=selected_pixels,
                    selected_rows=selected_rows,
                    display_code=display_code,
                    code=4 if group.startswith("m1") else 7,
                )
    if not selected_pixels:
        raise ValueError(f"dataset {dataset_id!r} has no selected fit pixels")
    pixels = np.concatenate(selected_pixels)
    observation_rows = np.concatenate(selected_rows)
    if np.unique(pixels).size != pixels.size:
        raise ValueError(f"dataset {dataset_id!r} has overlapping fitted regions")
    return pixels, observation_rows


def prepare(
    *,
    fixed_state_path: Path,
    recipe_path: Path,
    destination: Path,
    source_state_count: int,
    row_chunk_size: int,
) -> Path:
    destination = _external_file(destination)
    if destination.exists():
        raise FileExistsError(destination)
    adapter_identity = _file_identity(Path(__file__))
    recipe_path = recipe_path.resolve()
    recipe_identity = _file_identity(recipe_path)
    recipe = _load_recipe(recipe_path)
    simulation_config_path = (recipe_path.parent / str(recipe["simulation_config"])).resolve()
    simulation_config_path = simulation_config_path.resolve()
    (
        _,
        fixed_state_identity,
        simulation_config_identity,
        series,
        fixed_position,
        fixed_lattice,
        mosaic_parameters,
        observations,
        source_revision,
        cif_sha256,
        rod_catalog_revision,
    ) = _fixed_experiment_inputs(
        fixed_state_path,
        expected_material_id=str(recipe["material_id"]),
        expected_dataset_ids=tuple(str(value) for value in recipe["dataset_ids"]),
        expected_source_state_count=source_state_count,
    )
    if simulation_config_identity != _file_identity(simulation_config_path):
        raise ValueError("recipe and fixed state name different simulation configurations")
    model_input_identities = {"fixed_state": fixed_state_identity}
    dataset_ids = tuple(str(item["dataset_id"]) for item in observations)
    if dataset_ids != tuple(recipe["dataset_ids"]):
        raise ValueError("recipe dataset order differs from the verified OSC series")
    m0 = _m0_region(recipe)
    layouts = _offspecular_layouts(
        recipe,
        fixed_position.beam_center_column_row_px[0],
        int(recipe["detector"]["native_shape_rc"][1]),
    )
    row_records, lookup = _row_catalog(len(series), m0, layouts)
    row_count = len(row_records)
    count_sum = np.zeros(row_count, dtype=np.float64)
    support = np.zeros(row_count, dtype=np.float64)
    coordinate_sum = np.zeros(row_count, dtype=np.float64)
    axial_sum = np.zeros(row_count, dtype=np.float64)
    native_shape = tuple(int(value) for value in recipe["detector"]["native_shape_rc"])
    dark_counts, dark_correction = _dark_correction_from_recipe(
        recipe_path,
        recipe,
        detector_shape_rc=native_shape,
    )
    display_counts: np.ndarray | None = None
    display_code = np.zeros(math.prod(native_shape), dtype=np.uint8)
    selected_dataset_blocks: list[np.ndarray] = []
    selected_pixel_blocks: list[np.ndarray] = []
    selected_row_blocks: list[np.ndarray] = []
    osc_provenance: list[dict[str, Any]] = []
    display_dataset_id = str(recipe["display_dataset_id"])
    display_rows = int(recipe["detector"]["display_rows"])
    start = perf_counter()
    for dataset_index, (inputs, observation) in enumerate(zip(series, observations, strict=True)):
        osc_path = Path(str(observation["osc_file"])).resolve()
        compressed_sha256 = _sha256(osc_path)
        if compressed_sha256 != observation["osc_file_sha256"]:
            raise ValueError(f"OSC file hash changed for {dataset_ids[dataset_index]!r}")
        counts = read_osc(osc_path).detector_native_counts
        native_sha256 = hashlib.sha256(counts.tobytes(order="C")).hexdigest()
        if (
            native_sha256 != observation["detector_native_bytes_sha256"]
            or counts.dtype != np.dtype(str(observation["detector_native_dtype"]))
            or list(counts.shape) != observation["detector_native_shape_rc"]
            or counts.shape != native_shape
        ):
            raise ValueError(f"decoded OSC data changed for {dataset_ids[dataset_index]!r}")
        if dataset_ids[dataset_index] == display_dataset_id:
            display_counts = (
                counts[:display_rows].astype(np.float64)
                - dark_correction["scale"] * dark_counts[:display_rows]
            )
        selected_pixel, selected_row = _prepare_dataset_membership(
            inputs,
            counts,
            recipe=recipe,
            dataset_id=dataset_ids[dataset_index],
            dataset_index=dataset_index,
            lookup=lookup,
            m0=m0,
            layouts=layouts,
            count_sum=count_sum,
            support=support,
            coordinate_sum=coordinate_sum,
            axial_sum=axial_sum,
            display_code=(
                display_code if dataset_ids[dataset_index] == display_dataset_id else None
            ),
            row_chunk_size=row_chunk_size,
        )
        selected_dataset_blocks.append(np.full(selected_pixel.shape, dataset_index, dtype=np.int64))
        selected_pixel_blocks.append(selected_pixel)
        selected_row_blocks.append(selected_row)
        osc_provenance.append(
            {
                "dataset_id": dataset_ids[dataset_index],
                "osc_path": str(osc_path),
                "osc_file_sha256": compressed_sha256,
                "detector_native_bytes_sha256": native_sha256,
            }
        )
    if display_counts is None:
        raise RuntimeError("no display OSC was prepared")
    supported = support > 0.0
    complete = np.zeros(row_count, dtype=np.bool_)
    incomplete_blocks: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str, int], list[int]] = {}
    for index, record in enumerate(row_records):
        if supported[index]:
            grouped.setdefault(
                (int(record["dataset"]), str(record["group"]), int(record["bin"])),
                [],
            ).append(index)
    for key, indices in grouped.items():
        bands = {str(row_records[index]["band"]) for index in indices}
        background_complete = {"background_0", "background_1"}.issubset(bands)
        signal_present = any(not band.startswith("background_") for band in bands)
        if background_complete and signal_present:
            complete[indices] = True
        else:
            incomplete_blocks.append(
                {"dataset": key[0], "group": key[1], "bin": key[2], "bands": sorted(bands)}
            )
    retained_index = np.flatnonzero(complete)
    if not retained_index.size:
        raise ValueError("peak selection produced no complete signal/background blocks")
    old_to_new = np.full(row_count, -1, dtype=np.int64)
    old_to_new[retained_index] = np.arange(retained_index.size, dtype=np.int64)
    selected_dataset = np.concatenate(selected_dataset_blocks)
    selected_pixel = np.concatenate(selected_pixel_blocks)
    selected_old_row = np.concatenate(selected_row_blocks)
    selected_new_row = old_to_new[selected_old_row]
    selected_complete = selected_new_row >= 0
    display_dataset_index = dataset_ids.index(display_dataset_id)
    display_code[
        selected_pixel[(selected_dataset == display_dataset_index) & ~selected_complete]
    ] = 0
    selected_dataset = selected_dataset[selected_complete]
    selected_pixel = selected_pixel[selected_complete]
    selected_new_row = selected_new_row[selected_complete]
    records = [row_records[index] for index in retained_index]
    group_names = tuple(str(item["group"]) for item in records)
    band_names = tuple(str(item["band"]) for item in records)
    coordinate_kind = tuple(str(item["coordinate_kind"]) for item in records)
    block_lookup: dict[tuple[int, str, int], int] = {}
    block_index: list[int] = []
    for item in records:
        key = (int(item["dataset"]), str(item["group"]), int(item["bin"]))
        block_index.append(block_lookup.setdefault(key, len(block_lookup)))
    arrays: dict[str, np.ndarray] = {
        "dataset_id": np.asarray([dataset_ids[int(item["dataset"])] for item in records]),
        "dataset_index": np.asarray([item["dataset"] for item in records], dtype=np.int64),
        "group": np.asarray(group_names),
        "band": np.asarray(band_names),
        "bin_index": np.asarray([item["bin"] for item in records], dtype=np.int64),
        "signal_family_m": np.asarray([item["signal_family"] for item in records], dtype=np.int64),
        "is_background": np.asarray(
            [str(item["band"]).startswith("background_") for item in records],
            dtype=np.bool_,
        ),
        "block_index": np.asarray(block_index, dtype=np.int64),
        "coordinate_kind": np.asarray(coordinate_kind),
        "count_sum": count_sum[retained_index],
        "support_px2": support[retained_index],
        "coordinate_mean": coordinate_sum[retained_index] / support[retained_index],
        "L_mean": axial_sum[retained_index] / support[retained_index],
        "selected_dataset_index": selected_dataset,
        "selected_flat_pixel_index": selected_pixel,
        "selected_observation_row": selected_new_row,
        "display_detector_counts": display_counts,
        "display_region_code": display_code.reshape(native_shape)[:display_rows],
    }
    for dataset_index, dataset_id in enumerate(dataset_ids):
        represented = set(
            arrays["signal_family_m"][
                (arrays["dataset_index"] == dataset_index) & ~arrays["is_background"]
            ].tolist()
        )
        if not represented:
            raise ValueError(f"dataset {dataset_id!r} retained no fitted signal")
    MatchedRegionObservations(
        dataset_ids=dataset_ids,
        dataset_index=arrays["dataset_index"],
        block_index=arrays["block_index"],
        signal_family=arrays["signal_family_m"],
        is_background=arrays["is_background"],
        count_mass=arrays["count_sum"],
        support_px2=arrays["support_px2"],
        background_coordinate=arrays["coordinate_mean"],
        required_signal_families=FAMILIES,
    )
    m0_record, layout_records = _serialized_regions(m0, layouts)
    peak_projection, peak_catalog = _integrated_peak_area_projection(
        arrays,
        {
            "dataset_ids": dataset_ids,
            "fit_peak_catalog": recipe["fit_peak"],
            "m0_region": m0_record,
            "offspecular_layouts": layout_records,
            "fixed_lattice": fixed_lattice.to_record(),
        },
    )
    arrays["source_signal_peak_index"] = np.asarray(
        peak_projection.source_signal_peak_index,
        dtype=np.int64,
    )
    stacking_model = _fault_free_three_r_record(series)
    guarded_identities = {
        "preparation adapter": adapter_identity,
        "recipe": recipe_identity,
        "simulation configuration": simulation_config_identity,
        "dark OSC": {
            "path": dark_correction["path"],
            "sha256": dark_correction["file_sha256"],
        },
        **{name.replace("_", " "): value for name, value in model_input_identities.items()},
    }
    for role, identity in guarded_identities.items():
        _require_unchanged(identity, role=role)
    provenance = {
        "simulation_config": simulation_config_identity["path"],
        "simulation_config_sha256": simulation_config_identity["sha256"],
        "recipe": recipe_identity["path"],
        "recipe_sha256": recipe_identity["sha256"],
        "preparation_adapter": adapter_identity["path"],
        "preparation_adapter_sha256": adapter_identity["sha256"],
    }
    for name, identity in model_input_identities.items():
        provenance[name] = identity["path"]
        provenance[f"{name}_sha256"] = identity["sha256"]
    manifest = {
        "schema_version": PREPARED_SCHEMA,
        "material_id": str(recipe["material_id"]),
        "dataset_ids": dataset_ids,
        "families_m": FAMILIES,
        "source_state_count": source_state_count,
        "source_revision": source_revision,
        "cif_sha256": cif_sha256,
        "fixed_position": fixed_position.to_record(),
        "fixed_lattice": fixed_lattice.to_record(),
        "rod_catalog_revision": rod_catalog_revision,
        "stacking_model": stacking_model,
        "mosaic": mosaic_parameters,
        "m0_region": m0_record,
        "offspecular_layouts": layout_records,
        "fit_peak_catalog": recipe["fit_peak"],
        "objective_measure": PEAK_AREA_OBJECTIVE,
        "integrated_peak_catalog": peak_catalog,
        "peak_area_projection_revision": peak_projection.revision,
        "source_signal_peak_index_sha256": _array_sha256(arrays["source_signal_peak_index"]),
        "excluded_peak_catalog": recipe.get("excluded_peak", []),
        "horizon_gate": recipe["horizon_gate"],
        "dark_correction": dark_correction,
        "incomplete_blocks_excluded": incomplete_blocks,
        "observation_contract": (
            "frozen native-pixel-center raw-minus-scaled-dark count masses with exact "
            "unit-membership support and "
            + (
                "no dark covariance contribution"
                if float(dark_correction["scale"]) == 0.0
                else "full shared-dark overlap covariance"
            )
            + "; every candidate "
            "diffraction model remains an unrasterized continuous chart integral"
            if _uses_native_pixel_center_observations(recipe)
            else "verified native counts projected over continuous phi/two-theta or Qr/L "
            "rectangles with full count covariance"
        ),
        "observation_method": (
            NATIVE_PIXEL_CENTER_METHOD
            if _uses_native_pixel_center_observations(recipe)
            else MEASURED_PROJECTION_METHOD
        ),
        "native_pixel_membership": {
            "selected_pixel_region_pair_count": int(arrays["selected_flat_pixel_index"].size),
            "selected_dataset_index_sha256": _array_sha256(arrays["selected_dataset_index"]),
            "selected_flat_pixel_index_sha256": _array_sha256(arrays["selected_flat_pixel_index"]),
            "selected_observation_row_sha256": _array_sha256(arrays["selected_observation_row"]),
            "raw_count_mass_sha256": _array_sha256(arrays["count_sum"]),
            "support_px2_sha256": _array_sha256(arrays["support_px2"]),
        },
        "model_pixelized": False,
        "smoothing_applied": False,
        "osc_provenance": osc_provenance,
        "provenance": provenance,
        "preparation_seconds": perf_counter() - start,
    }
    return write_diagnostic(destination, arrays=arrays, manifest=manifest, repository_root=ROOT)


def _resolved_diagnostic_path(path: Path) -> Path:
    diagnostic = path.resolve()
    if diagnostic == ROOT or diagnostic.is_relative_to(ROOT):
        raise ValueError("diagnostic must be outside the repository")
    if not diagnostic.name.endswith(".ra_diag.npz") or not diagnostic.is_file():
        raise ValueError("invalid diagnostic path")
    return diagnostic


def _decoded_diagnostic_manifest(
    archive: Any,
    *,
    expected_schema: str,
) -> dict[str, Any]:
    if "manifest_json" not in archive.files:
        raise ValueError("diagnostic lacks an embedded manifest")
    encoded = np.asarray(archive["manifest_json"])
    if encoded.dtype != np.uint8 or encoded.ndim != 1:
        raise ValueError("diagnostic has an invalid embedded manifest")
    try:
        manifest = json.loads(encoded.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("diagnostic has an invalid embedded manifest") from error
    if not isinstance(manifest, dict):
        raise ValueError("diagnostic manifest must be a JSON object")
    if manifest.get("schema_version") != expected_schema:
        raise ValueError("unsupported diagnostic schema")
    return manifest


def _load_diagnostic(
    path: Path,
    *,
    expected_schema: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    diagnostic = _resolved_diagnostic_path(path)
    with np.load(diagnostic, allow_pickle=False) as archive:
        if any(archive[name].dtype.hasobject for name in archive.files):
            raise ValueError("diagnostic may not contain object arrays")
        manifest = _decoded_diagnostic_manifest(archive, expected_schema=expected_schema)
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "manifest_json"
        }
    return arrays, manifest


def _load_diagnostic_manifest(
    path: Path,
    *,
    expected_schema: str,
) -> dict[str, Any]:
    diagnostic = _resolved_diagnostic_path(path)
    with np.load(diagnostic, allow_pickle=False) as archive:
        return _decoded_diagnostic_manifest(archive, expected_schema=expected_schema)


def _load_prepared(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return _load_diagnostic(path, expected_schema=PREPARED_SCHEMA)


def _load_prepared_manifest(path: Path) -> dict[str, Any]:
    return _load_diagnostic_manifest(path, expected_schema=PREPARED_SCHEMA)


def _radial_background_parameter_names(dataset_ids: Sequence[str]) -> tuple[str, ...]:
    return (
        "inner_scale_px",
        "inner_power",
        "outer_scale_px",
        "outer_power",
        *(f"amplitude_count_per_px:{value}" for value in dataset_ids),
        *(f"pedestal_count_per_px:{value}" for value in dataset_ids),
    )


def _background_exclusion_pixel_index(
    center_selected_flat_pixel_index: np.ndarray,
    continuous_projection_flat_pixel_index: np.ndarray,
) -> np.ndarray:
    """Return every measured pixel that may contribute to a fitted region."""

    center_selected = np.asarray(center_selected_flat_pixel_index, dtype=np.int64)
    continuous_selected = np.asarray(continuous_projection_flat_pixel_index, dtype=np.int64)
    if (
        center_selected.ndim != 1
        or continuous_selected.ndim != 1
        or np.any(center_selected < 0)
        or np.any(continuous_selected < 0)
    ):
        raise ValueError("background exclusion pixels must be nonnegative vectors")
    result = np.union1d(center_selected, continuous_selected).astype(np.int64, copy=False)
    result.setflags(write=False)
    return result


def _robust_radial_cells(
    *,
    counts: np.ndarray,
    beam_center_column_row_px: tuple[float, float],
    excluded_flat_pixel_index: np.ndarray,
    sample_stride: int,
    radial_bin_width_px: float,
    azimuth_sector_count: int,
    minimum_radius_px: float,
    maximum_radius_px: float,
    border_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = counts.shape
    sample_row = np.arange(border_px, rows - border_px, sample_stride, dtype=np.int64)
    sample_column = np.arange(border_px, columns - border_px, sample_stride, dtype=np.int64)
    column_grid, row_grid = np.meshgrid(sample_column, sample_row)
    flat_pixel = (row_grid * columns + column_grid).reshape(-1)
    column = column_grid.reshape(-1).astype(np.float64)
    row = row_grid.reshape(-1).astype(np.float64)
    value = counts[row_grid, column_grid].reshape(-1).astype(np.float64)
    beam_column, beam_row = beam_center_column_row_px
    delta_column = column - beam_column
    delta_row = row - beam_row
    radius = np.hypot(delta_column, delta_row)
    azimuth = np.mod(np.arctan2(delta_row, delta_column), 2.0 * np.pi)
    sector = np.floor(azimuth * azimuth_sector_count / (2.0 * np.pi)).astype(np.int64)
    excluded = np.sort(np.asarray(excluded_flat_pixel_index, dtype=np.int64))
    location = np.searchsorted(excluded, flat_pixel)
    selected_region = np.zeros(flat_pixel.size, dtype=np.bool_)
    inside = location < excluded.size
    selected_region[inside] = excluded[location[inside]] == flat_pixel[inside]
    valid = (
        (radius >= minimum_radius_px)
        & (radius < maximum_radius_px)
        & np.isfinite(value)
        & ~selected_region
    )
    radius = radius[valid]
    value = value[valid]
    sector = sector[valid]
    radial_bin = np.floor((radius - minimum_radius_px) / radial_bin_width_px).astype(np.int64)
    radial_bin_count = math.ceil((maximum_radius_px - minimum_radius_px) / radial_bin_width_px)
    key = sector * radial_bin_count + radial_bin
    order = np.argsort(key, kind="stable")
    key = key[order]
    radius = radius[order]
    value = value[order]
    boundary = np.concatenate(([0], np.flatnonzero(np.diff(key)) + 1, [key.size]))
    cell_radius: list[float] = []
    cell_sector: list[int] = []
    cell_density: list[float] = []
    cell_support: list[float] = []
    for start, stop in pairwise(boundary):
        if stop - start < 12:
            continue
        cell_value = value[start:stop]
        median = float(np.median(cell_value))
        mad = 1.4826 * float(np.median(np.abs(cell_value - median)))
        robust_sigma = max(mad, math.sqrt(max(median, 1.0)), 1.0)
        retained = np.abs(cell_value - median) <= 3.5 * robust_sigma
        if np.count_nonzero(retained) < 10:
            continue
        cell_radius.append(float(np.mean(radius[start:stop][retained])))
        cell_sector.append(int(key[start] // radial_bin_count))
        cell_density.append(float(np.mean(cell_value[retained])))
        cell_support.append(float(np.count_nonzero(retained) * sample_stride**2))
    return tuple(
        np.asarray(value, dtype=dtype)
        for value, dtype in (
            (cell_radius, np.float64),
            (cell_sector, np.int64),
            (cell_density, np.float64),
            (cell_support, np.float64),
        )
    )


def calibrate_radial_background(
    *,
    diagnostic_path: Path,
    fit_plan_path: Path,
    destination: Path,
    sample_stride: int = 3,
    radial_bin_width_px: float = 10.0,
    azimuth_sector_count: int = 16,
    minimum_radius_px: float = 45.0,
    maximum_radius_px: float = 1100.0,
    border_px: int = 12,
) -> Path:
    """Fit one shared empirical halo from background-only detector sectors."""

    destination = _external_file(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if sample_stride < 1 or azimuth_sector_count < 8 or border_px < 0:
        raise ValueError("radial background sampling policy is invalid")
    adapter_identity = _file_identity(Path(__file__))
    implementation_identity = _implementation_identity()
    diagnostic_identity = _file_identity(diagnostic_path)
    arrays, manifest = _load_prepared(diagnostic_path)
    dark_counts, dark_scale = _verified_dark_counts(manifest)
    dataset_ids = tuple(str(value) for value in manifest["dataset_ids"])
    fit_plan_identity = _file_identity(fit_plan_path)
    fit_plan = _load_fit_plan(fit_plan_path)
    if fit_plan["material_id"] != manifest["material_id"]:
        raise ValueError("background fit plan material differs from the prepared experiment")
    recipe_path = _verified_provenance_path(manifest, "recipe")
    recipe_identity = _file_identity(recipe_path)
    recipe = _load_recipe(recipe_path)
    continuous_quadrature = fit_plan["continuous_quadrature"]
    series = _rebuilt_series(manifest)
    native_observations = _uses_native_pixel_center_observations(recipe)
    if native_observations:
        detector_shape = tuple(int(value) for value in recipe["detector"]["native_shape_rc"])
        native_center_plans = _native_pixel_center_plans(
            arrays,
            dataset_ids=dataset_ids,
            detector_shape_rc=detector_shape,
        )
        oracle_region_plans: tuple[_DatasetContinuousRegionPlan, ...] = ()
    else:
        native_center_plans = ()
        oracle_region_plans = tuple(
            _compile_dataset_continuous_region_plan(
                inputs,
                arrays,
                manifest,
                dataset_index=dataset_index,
                gauss_order=int(recipe["model_cubature"]["oracle_gauss_order"]),
                subdivision_count=int(recipe["model_cubature"]["fold_oracle_subdivisions"]),
                offspecular_axial_refinement=int(
                    continuous_quadrature["offspecular_axial_refinement"]
                ),
                offspecular_radial_transform=str(
                    continuous_quadrature["offspecular_radial_transform"]
                ),
                offspecular_signal_minimum_radial_nodes_per_side=int(
                    continuous_quadrature["offspecular_signal_minimum_radial_nodes_per_side"]
                ),
                m0_phi_subdivision_count=int(recipe["model_cubature"]["fold_oracle_subdivisions"]),
            )
            for dataset_index, inputs in enumerate(series)
        )
    beam_center = tuple(
        float(value) for value in manifest["fixed_position"]["beam_center_column_row_px"]
    )
    profile_dataset: list[np.ndarray] = []
    profile_radius: list[np.ndarray] = []
    profile_sector: list[np.ndarray] = []
    profile_density: list[np.ndarray] = []
    profile_support: list[np.ndarray] = []
    osc_identities: list[dict[str, str]] = []
    observation_support_revisions: list[str] = []
    excluded_pixel_counts: dict[str, int] = {}
    excluded_pixel_sha256: dict[str, str] = {}
    for dataset_index, provenance in enumerate(manifest["osc_provenance"]):
        osc_path = Path(str(provenance["osc_path"])).resolve()
        identity = _file_identity(osc_path)
        if identity["sha256"] != provenance["osc_file_sha256"]:
            raise ValueError("OSC file changed after matched-region preparation")
        counts = read_osc(osc_path).detector_native_counts
        native_sha256 = hashlib.sha256(counts.tobytes(order="C")).hexdigest()
        if native_sha256 != provenance["detector_native_bytes_sha256"]:
            raise ValueError("decoded OSC counts changed after matched-region preparation")
        center_selected = arrays["selected_flat_pixel_index"][
            arrays["selected_dataset_index"] == dataset_index
        ]
        if native_observations:
            projection = native_center_plans[dataset_index].projection
            selected = np.unique(center_selected).astype(np.int64, copy=False)
        else:
            projection = compile_native_pixel_region_projection(
                oracle_region_plans[dataset_index].quadrature,
                counts.shape,
            )
            selected = _background_exclusion_pixel_index(
                center_selected,
                projection.flat_pixel_index,
            )
        radius, sector, density, support = _robust_radial_cells(
            counts=counts.astype(np.float64) - dark_scale * dark_counts,
            beam_center_column_row_px=beam_center,
            excluded_flat_pixel_index=selected,
            sample_stride=sample_stride,
            radial_bin_width_px=radial_bin_width_px,
            azimuth_sector_count=azimuth_sector_count,
            minimum_radius_px=minimum_radius_px,
            maximum_radius_px=maximum_radius_px,
            border_px=border_px,
        )
        profile_dataset.append(np.full(radius.size, dataset_index, dtype=np.int64))
        profile_radius.append(radius)
        profile_sector.append(sector)
        profile_density.append(density)
        profile_support.append(support)
        osc_identities.append(identity)
        observation_support_revisions.append(projection.projection_revision)
        excluded_pixel_counts[dataset_ids[dataset_index]] = int(selected.size)
        excluded_pixel_sha256[dataset_ids[dataset_index]] = _array_sha256(selected)
    dataset_index = np.concatenate(profile_dataset)
    radius = np.concatenate(profile_radius)
    sector = np.concatenate(profile_sector)
    density = np.concatenate(profile_density)
    support = np.concatenate(profile_support)
    is_training = sector % 4 != 0
    profiles = RadialBackgroundProfiles(
        dataset_ids=dataset_ids,
        dataset_index=dataset_index,
        radius_px=radius,
        azimuth_sector_index=sector,
        density_count_per_px=density,
        support_px2=support,
        is_training=is_training,
    )
    result = fit_shared_radial_background(profiles)
    if not result.success:
        raise RuntimeError(f"radial background fit failed: {result.optimizer_message}")
    metrics_by_dataset: dict[str, dict[str, float]] = {}
    residual = result.residual_density_count_per_px
    for dataset_index_value, dataset_id in enumerate(dataset_ids):
        heldout = (dataset_index == dataset_index_value) & ~is_training
        metrics_by_dataset[dataset_id] = {
            "heldout_rmse_count_per_px": float(np.sqrt(np.mean(residual[heldout] ** 2))),
            "heldout_mae_count_per_px": float(np.mean(np.abs(residual[heldout]))),
            "heldout_bias_count_per_px": float(np.mean(residual[heldout])),
        }
    artifact_arrays = {
        "dataset_index": dataset_index,
        "radius_px": radius,
        "azimuth_sector_index": sector,
        "density_count_per_px": density,
        "support_px2": support,
        "is_training": is_training,
        "parameter_vector": result.state.parameter_vector,
        "parameter_covariance": result.state.parameter_covariance,
        "fitted_density_count_per_px": result.fitted_density_count_per_px,
        "residual_density_count_per_px": residual,
    }
    guarded_identities = {
        "background adapter": adapter_identity,
        "prepared diagnostic": diagnostic_identity,
        "fit plan": fit_plan_identity,
        "recipe": recipe_identity,
        "dark OSC": {
            "path": str(manifest["dark_correction"]["path"]),
            "sha256": str(manifest["dark_correction"]["file_sha256"]),
        },
        **{
            f"OSC dataset {dataset_id}": identity
            for dataset_id, identity in zip(dataset_ids, osc_identities, strict=True)
        },
    }
    for role, identity in guarded_identities.items():
        _require_unchanged(identity, role=role)
    if _implementation_identity() != implementation_identity:
        raise RuntimeError("scientific implementation changed during background calibration")
    artifact_manifest = {
        "schema_version": BACKGROUND_SCHEMA,
        "status": "MODEL_LIMITED_EMPIRICAL_RADIAL_BACKGROUND",
        "publication_ready": False,
        "dataset_ids": dataset_ids,
        "beam_center_column_row_px": beam_center,
        "state_revision": result.state.revision,
        "parameter_vector_sha256": _array_sha256(result.state.parameter_vector),
        "parameter_covariance_sha256": _array_sha256(result.state.parameter_covariance),
        "parameter_names": _radial_background_parameter_names(dataset_ids),
        "parameter_vector": result.state.parameter_vector.tolist(),
        "model": RADIAL_BACKGROUND_MODEL,
        "sampling": {
            "native_pixel_stride": sample_stride,
            "radial_bin_width_px": radial_bin_width_px,
            "azimuth_sector_count": azimuth_sector_count,
            "heldout_sector_rule": "sector_index_mod_4_equals_0",
            "minimum_radius_px": minimum_radius_px,
            "maximum_radius_px": maximum_radius_px,
            "border_px": border_px,
            "signal_and_anchor_pixels_excluded": True,
            "observation_support_method": (
                NATIVE_PIXEL_CENTER_METHOD if native_observations else MEASURED_PROJECTION_METHOD
            ),
            "observation_support_revisions": observation_support_revisions,
            "excluded_flat_pixel_count_by_dataset": excluded_pixel_counts,
            "excluded_flat_pixel_sha256_by_dataset": excluded_pixel_sha256,
            "cell_estimator": "mean_after_symmetric_3.5_sigma_median_MAD_clipping",
        },
        "metrics": {
            "training_rmse_count_per_px": result.training_rmse_count_per_px,
            "heldout_rmse_count_per_px": result.heldout_rmse_count_per_px,
            "heldout_mae_count_per_px": result.heldout_mae_count_per_px,
            "heldout_bias_count_per_px": result.heldout_bias_count_per_px,
            "by_dataset": metrics_by_dataset,
        },
        "background_measure": "empirical_detector_native_counts_per_pixel",
        "background_calibration_uses_native_pixels": True,
        "diffraction_model_pixelized": False,
        "dark_applied": True,
        "dark_correction": manifest["dark_correction"],
        "smoothing_applied": False,
        "provenance": {
            "prepared_diagnostic": diagnostic_identity,
            "background_adapter": adapter_identity,
            "implementation": implementation_identity,
            "recipe": recipe_identity,
            "fit_plan": fit_plan_identity,
            "osc": osc_identities,
            "position_revision": manifest["fixed_position"]["position_artifact_revision"],
            "rod_catalog_revision": manifest.get("rod_catalog_revision"),
        },
    }
    return write_diagnostic(
        destination,
        arrays=artifact_arrays,
        manifest=artifact_manifest,
        repository_root=ROOT,
    )


def _load_radial_background(
    path: Path,
    *,
    diagnostic_identity: dict[str, str],
    dataset_ids: Sequence[str],
    beam_center_column_row_px: tuple[float, float],
    fit_plan_identity: dict[str, str],
    observation_support_revisions: Sequence[str],
    observation_support_method: str,
    excluded_flat_pixel_index_by_dataset: dict[str, np.ndarray] | None = None,
    expected_background_adapter_identity: dict[str, Any] | None = None,
    expected_implementation_identity: dict[str, Any] | None = None,
) -> tuple[RadialBackgroundState, dict[str, str], dict[str, Any]]:
    identity = _file_identity(path)
    arrays, manifest = _load_diagnostic(path, expected_schema=BACKGROUND_SCHEMA)
    expected_ids = tuple(str(value) for value in dataset_ids)
    provenance = manifest.get("provenance", {})
    sampling = manifest.get("sampling", {})
    background_adapter_identity = (
        _file_identity(Path(__file__))
        if expected_background_adapter_identity is None
        else expected_background_adapter_identity
    )
    implementation_identity = (
        _implementation_identity()
        if expected_implementation_identity is None
        else expected_implementation_identity
    )
    if excluded_flat_pixel_index_by_dataset is not None and set(
        excluded_flat_pixel_index_by_dataset
    ) != set(expected_ids):
        raise ValueError("background exclusion roster differs from the prepared datasets")
    expected_excluded_count = (
        None
        if excluded_flat_pixel_index_by_dataset is None
        else {
            dataset_id: int(np.unique(excluded_flat_pixel_index_by_dataset[dataset_id]).size)
            for dataset_id in expected_ids
        }
    )
    expected_excluded_sha256 = (
        None
        if excluded_flat_pixel_index_by_dataset is None
        else {
            dataset_id: _array_sha256(
                np.unique(excluded_flat_pixel_index_by_dataset[dataset_id]).astype(
                    np.int64,
                    copy=False,
                )
            )
            for dataset_id in expected_ids
        }
    )
    if (
        tuple(manifest.get("dataset_ids", ())) != expected_ids
        or provenance.get("prepared_diagnostic") != diagnostic_identity
        or provenance.get("fit_plan") != fit_plan_identity
        or provenance.get("background_adapter") != background_adapter_identity
        or provenance.get("implementation") != implementation_identity
        or _verified_recorded_file_identity(provenance.get("recipe"), role="background recipe")
        != provenance.get("recipe")
        or manifest.get("status") != "MODEL_LIMITED_EMPIRICAL_RADIAL_BACKGROUND"
        or manifest.get("model") != RADIAL_BACKGROUND_MODEL
        or manifest.get("background_measure") != "empirical_detector_native_counts_per_pixel"
        or manifest.get("background_calibration_uses_native_pixels") is not True
        or manifest.get("diffraction_model_pixelized") is not False
        or manifest.get("dark_applied") is not True
        or manifest.get("dark_correction")
        != _load_prepared_manifest(Path(diagnostic_identity["path"])).get("dark_correction")
        or manifest.get("smoothing_applied") is not False
        or tuple(manifest.get("beam_center_column_row_px", ()))
        != tuple(float(value) for value in beam_center_column_row_px)
        or tuple(sampling.get("observation_support_revisions", ()))
        != tuple(observation_support_revisions)
        or sampling.get("observation_support_method") != observation_support_method
        or sampling.get("signal_and_anchor_pixels_excluded") is not True
        or set(sampling.get("excluded_flat_pixel_count_by_dataset", {})) != set(expected_ids)
        or set(sampling.get("excluded_flat_pixel_sha256_by_dataset", {})) != set(expected_ids)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in sampling.get("excluded_flat_pixel_count_by_dataset", {}).values()
        )
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in sampling.get("excluded_flat_pixel_sha256_by_dataset", {}).values()
        )
        or (
            expected_excluded_count is not None
            and sampling.get("excluded_flat_pixel_count_by_dataset") != expected_excluded_count
        )
        or (
            expected_excluded_sha256 is not None
            and sampling.get("excluded_flat_pixel_sha256_by_dataset") != expected_excluded_sha256
        )
        or tuple(manifest.get("parameter_names", ()))
        != _radial_background_parameter_names(expected_ids)
    ):
        raise ValueError("radial background artifact does not belong to this prepared series")
    state = RadialBackgroundState.from_parameter_vector(
        expected_ids,
        arrays["parameter_vector"],
        parameter_covariance=arrays["parameter_covariance"],
    )
    if (
        state.revision != manifest.get("state_revision")
        or not np.array_equal(state.parameter_vector, np.asarray(manifest["parameter_vector"]))
        or manifest.get("parameter_vector_sha256") != _array_sha256(state.parameter_vector)
        or manifest.get("parameter_covariance_sha256") != _array_sha256(state.parameter_covariance)
    ):
        raise ValueError("radial background state and manifest disagree")
    return state, identity, manifest


def _fixed_background_from_continuous_plans(
    *,
    state: RadialBackgroundState,
    plans: Sequence[_DatasetContinuousRegionPlan],
    observation_count: int,
    beam_center_column_row_px: tuple[float, float],
) -> FixedMatchedRegionBackground:
    beam_column, beam_row = beam_center_column_row_px
    parameter_count = state.parameter_vector.size
    mass = np.zeros(observation_count, dtype=np.float64)
    mass_jacobian = np.zeros((observation_count, parameter_count), dtype=np.float64)
    quadrature_revisions: list[str] = []
    for plan in plans:
        quadrature = plan.quadrature
        global_row = np.asarray(plan.global_observation_row, dtype=np.int64)
        dataset_index = state.dataset_ids.index(plan.dataset_id)
        node_dataset = np.full(quadrature.column_px.size, dataset_index, dtype=np.int64)
        radius = np.hypot(
            np.asarray(quadrature.column_px) - beam_column,
            np.asarray(quadrature.row_px) - beam_row,
        )
        density = state.count_density(node_dataset, radius)
        density_jacobian = state.count_density_parameter_jacobian(node_dataset, radius)
        mass[global_row] = quadrature.integrate_density(density)
        for parameter_index in range(parameter_count):
            mass_jacobian[global_row, parameter_index] = np.bincount(
                quadrature.observation_row,
                weights=(
                    quadrature.detector_area_weight_px2 * density_jacobian[:, parameter_index]
                ),
                minlength=quadrature.observation_count,
            )
        quadrature_revisions.append(quadrature.quadrature_revision)
    covariance = mass_jacobian @ state.parameter_covariance @ mass_jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    revision = hashlib.sha256()
    revision.update(b"continuous-region-radial-background.v2\0")
    revision.update(state.revision.encode("utf-8"))
    revision.update(np.ascontiguousarray(beam_center_column_row_px, dtype=np.float64).tobytes())
    for quadrature_revision in quadrature_revisions:
        revision.update(b"\0")
        revision.update(quadrature_revision.encode("utf-8"))
    revision.update(np.ascontiguousarray(mass_jacobian).tobytes())
    revision.update(np.ascontiguousarray(mass).tobytes())
    revision.update(np.ascontiguousarray(covariance).tobytes())
    return FixedMatchedRegionBackground(
        count_mass=mass,
        covariance_count2=covariance,
        revision=f"sha256-{revision.hexdigest()}.continuous-radial.v2",
    )


def _fixed_background_from_native_pixel_center_plans(
    *,
    state: RadialBackgroundState,
    plans: Sequence[_DatasetNativePixelCenterPlan],
    observation_count: int,
    beam_center_column_row_px: tuple[float, float],
) -> FixedMatchedRegionBackground:
    """Evaluate the data-only radial baseline on the exact measured pixel centers."""

    beam_column, beam_row = beam_center_column_row_px
    parameter_count = state.parameter_vector.size
    mass = np.zeros(observation_count, dtype=np.float64)
    mass_jacobian = np.zeros((observation_count, parameter_count), dtype=np.float64)
    projection_revisions: list[str] = []
    for plan in plans:
        projection = plan.projection
        rows, columns = projection.detector_shape_rc
        del rows
        flat_pixel = np.asarray(projection.flat_pixel_index, dtype=np.int64)
        column = (flat_pixel % columns).astype(np.float64)
        row = (flat_pixel // columns).astype(np.float64)
        radius = np.hypot(column - beam_column, row - beam_row)
        dataset_index = state.dataset_ids.index(plan.dataset_id)
        node_dataset = np.full(flat_pixel.size, dataset_index, dtype=np.int64)
        density = state.count_density(node_dataset, radius)
        density_jacobian = state.count_density_parameter_jacobian(node_dataset, radius)
        local_row = np.asarray(projection.observation_row, dtype=np.int64)
        pixel_column = np.asarray(projection.pixel_column_index, dtype=np.int64)
        weight = np.asarray(projection.detector_area_weight_px2, dtype=np.float64)
        global_row = np.asarray(plan.global_observation_row, dtype=np.int64)
        mass[global_row] = np.bincount(
            local_row,
            weights=weight * density[pixel_column],
            minlength=projection.observation_count,
        )
        for parameter_index in range(parameter_count):
            mass_jacobian[global_row, parameter_index] = np.bincount(
                local_row,
                weights=weight * density_jacobian[pixel_column, parameter_index],
                minlength=projection.observation_count,
            )
        projection_revisions.append(projection.projection_revision)
    covariance = mass_jacobian @ state.parameter_covariance @ mass_jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    revision = hashlib.sha256()
    revision.update(b"native-pixel-center-radial-background.v1\0")
    revision.update(state.revision.encode("utf-8"))
    revision.update(np.ascontiguousarray(beam_center_column_row_px, dtype=np.float64).tobytes())
    for projection_revision in projection_revisions:
        revision.update(b"\0")
        revision.update(projection_revision.encode("utf-8"))
    revision.update(np.ascontiguousarray(mass_jacobian).tobytes())
    revision.update(np.ascontiguousarray(mass).tobytes())
    revision.update(np.ascontiguousarray(covariance).tobytes())
    return FixedMatchedRegionBackground(
        count_mass=mass,
        covariance_count2=covariance,
        revision=f"sha256-{revision.hexdigest()}.native-center-radial.v1",
    )


def _continuous_plan_reciprocal_coordinate_moments(
    inputs: Any,
    plan: _DatasetContinuousRegionPlan,
    *,
    m0_observation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    context = build_nominal_ewald_context(inputs)
    geometry = context.evaluate_detector_geometry(
        plan.quadrature.column_px,
        plan.quadrature.row_px,
        include_surface_jacobian=False,
    )
    observation_is_m0 = np.asarray(m0_observation, dtype=np.bool_)
    if observation_is_m0.shape != (plan.quadrature.observation_count,):
        raise ValueError("m=0 observation mask must align with the continuous plan")
    row = np.asarray(plan.quadrature.observation_row)
    node_is_m0 = observation_is_m0[row]
    if np.any(~node_is_m0 & ~np.asarray(geometry.valid)):
        raise FloatingPointError("off-specular profile nodes left valid detector geometry")
    frame = LayeredReciprocalFrame(
        reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
        sample_from_crystal_rotation=inputs.instrument.sample_from_crystal.rotation,
        axial_basis_index=2,
    )
    _, axial_L = frame.coordinates(geometry.q_sample_Ainv)
    axial_L = np.asarray(axial_L, dtype=np.float64)
    qz_Ainv = np.asarray(geometry.q_sample_Ainv, dtype=np.float64)[..., 2].copy()
    if np.any(node_is_m0):
        angle_frame = build_osc_angle_frame(
            mean_direction_lab=inputs.config.source.mean_direction_lab,
            instrument=inputs.instrument,
            sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
            revision=f"continuous-profile-m0-{plan.dataset_index}.v1",
        )
        angles = detector_coordinates_to_angles(
            plan.quadrature.column_px,
            plan.quadrature.row_px,
            instrument=inputs.instrument,
            angle_frame=angle_frame,
        )
        if np.any(node_is_m0 & ~(angles.valid & angles.azimuth_valid)):
            raise FloatingPointError("m=0 profile nodes left the detector angle chart")
        local_qz_Ainv = (4.0 * np.pi / float(inputs.config.source.mean_wavelength_A)) * np.sin(
            0.5 * np.asarray(angles.two_theta_rad)
        )
        axial_basis_magnitude_Ainv = float(
            np.linalg.norm(np.asarray(inputs.reciprocal.basis_Ainv, dtype=np.float64)[:, 2])
        )
        axial_L = np.array(axial_L, copy=True)
        axial_L[node_is_m0] = local_qz_Ainv[node_is_m0] / axial_basis_magnitude_Ainv
        qz_Ainv[node_is_m0] = local_qz_Ainv[node_is_m0]
    weight = np.asarray(plan.quadrature.detector_area_weight_px2)
    axial_mass = np.bincount(
        row,
        weights=weight * np.asarray(axial_L),
        minlength=plan.quadrature.observation_count,
    ).astype(np.float64, copy=False)
    qz_mass = np.bincount(
        row,
        weights=weight * qz_Ainv,
        minlength=plan.quadrature.observation_count,
    ).astype(np.float64, copy=False)
    return axial_mass, qz_mass


def _region_display_code(group: str, band: str, family_m: int) -> int:
    if group == "m0":
        return 2 if band.startswith("background_") else 1
    if band.startswith("background_"):
        return 4 if group.startswith("m1") else 7
    try:
        return {1: 3, 3: 5, 4: 6}[family_m]
    except KeyError as error:
        raise ValueError(f"unsupported fitted family m={family_m}") from error


def _profile_manifest_is_admissible(
    manifest: dict[str, Any],
    *,
    trusted_recipe: dict[str, Any],
    prepared_dark_correction: dict[str, Any],
) -> bool:
    try:
        roster = tuple(tuple(value) for value in manifest["model_rod_roster_h_k_m_population"])
        roster_sha256 = hashlib.sha256(
            json.dumps(roster, separators=(",", ":"), allow_nan=False).encode("ascii")
        ).hexdigest()
        families = {int(record[2]) for record in roster}
        cubature = manifest["profile_cubature"]
        measured_projection = manifest["measured_data_projection"]
        fitted_region_cubature = cubature["fit_artifact_fitted_region_oracle"]
        fitted_data_projection = cubature["fit_artifact_data_projection"]
        refinement_oracle = measured_projection["refinement_oracle"]
        provenance = manifest["provenance"]
        execution = provenance["execution_identity"]
        provenance_fit_plan_sha256 = str(provenance["fit_plan"]["sha256"])
        provenance_implementation_sha256 = str(provenance["implementation"]["sha256"])
        provenance_fit_chain_sha256 = [
            str(identity["sha256"]) for identity in provenance["fit_chain"]
        ]
        background_model = manifest["background_model"]
        fit_origin_adapter = str(provenance["fit_origin_adapter_sha256"])
        fit_origin_implementation = str(provenance["fit_origin_implementation_sha256"])
        profile_adapter = str(provenance["profile_adapter_sha256"])
        profile_implementation = str(provenance["implementation"]["sha256"])
        evaluator_hashes = (
            fit_origin_adapter,
            fit_origin_implementation,
            profile_adapter,
            profile_implementation,
        )
        trusted_model = _trusted_model_cubature(trusted_recipe)
        expected_dataset_ids = tuple(str(value) for value in trusted_recipe["dataset_ids"])
        display_dataset_id = str(trusted_recipe["display_dataset_id"])
        dark = manifest["dark_correction"]
        m0_signal_only = manifest["m0_signal_only_display"]
        supplemental_bin_count = m0_signal_only["supplemental_bin_count"]
        fit_parameter_replay = manifest.get("fit_parameter_replay")
        if fit_parameter_replay is not None and not isinstance(fit_parameter_replay, dict):
            raise TypeError("fit parameter replay must be a mapping")
        effective_trusted_recipe = _profile_recipe_with_specular_interface(
            trusted_recipe,
            (
                None
                if fit_parameter_replay is None
                else str(fit_parameter_replay["profile_interface_assumption"])
            ),
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if (
        isinstance(supplemental_bin_count, bool)
        or not isinstance(supplemental_bin_count, int)
        or supplemental_bin_count < 0
    ):
        return False
    native_fit_binding = bool(
        fit_origin_adapter == profile_adapter
        and fit_origin_implementation == profile_implementation
    )
    parameter_replay_admissible = False
    if fit_parameter_replay is not None:
        try:
            fit_stitch = trusted_recipe["parratt_stitch"]
            replay_scales = fit_parameter_replay["dataset_scales"]
            replay_scale_values = np.asarray(
                [float(replay_scales[dataset_id]) for dataset_id in expected_dataset_ids],
                dtype=np.float64,
            )
            structure_parameters = np.asarray(
                execution["structure_parameters"],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            parameter_replay_admissible = False
        else:
            parameter_replay_admissible = bool(
                fit_parameter_replay.get("method") == PROFILE_PARAMETER_REPLAY_METHOD
                and fit_parameter_replay.get("status") == "COMPLETE"
                and fit_parameter_replay.get("fit_role")
                == "frozen_parameter_and_dataset_scale_source_only"
                and fit_parameter_replay.get("profile_role")
                == "continuous_profile_recalculation_without_optimization"
                and fit_parameter_replay.get("scope") == "m0_specular_interface_assumption_only"
                and fit_parameter_replay.get("fit_parameters_reused") is True
                and fit_parameter_replay.get("optimizer_executed") is False
                and fit_parameter_replay.get("fit_reexecuted") is False
                and fit_parameter_replay.get("profile_reexecuted") is True
                and fit_parameter_replay.get("objective_requalified_under_profile_model") is False
                and fit_parameter_replay.get("dataset_scales_requalified") is False
                and fit_parameter_replay.get("source_fit_sha256")
                == provenance.get("fit_sha256")
                == execution.get("fit_sha256")
                and fit_parameter_replay.get("source_fit_status") == manifest.get("fit_status")
                and fit_parameter_replay.get("structure_parameter_vector_sha256")
                == _array_sha256(structure_parameters)
                and set(replay_scales) == set(expected_dataset_ids)
                and replay_scale_values.shape == (len(expected_dataset_ids),)
                and np.all(np.isfinite(replay_scale_values))
                and np.all(replay_scale_values > 0.0)
                and fit_parameter_replay.get("dataset_scale_vector_sha256")
                == _array_sha256(replay_scale_values)
                and float(manifest.get("dataset_scale", math.nan))
                == float(replay_scales[display_dataset_id])
                and fit_parameter_replay.get("fit_interface_assumption")
                == fit_stitch.get("interface_assumption")
                and fit_parameter_replay.get("profile_interface_assumption")
                in SPECULAR_INTERFACE_ASSUMPTIONS
                and fit_parameter_replay.get("profile_interface_assumption")
                != fit_parameter_replay.get("fit_interface_assumption")
                and execution.get("fit_parameter_replay") == fit_parameter_replay
                and execution.get("profile_adapter_sha256") == profile_adapter
                and execution.get("fit_origin_adapter_sha256") == fit_origin_adapter
                and execution.get("fit_origin_implementation_sha256") == fit_origin_implementation
            )
    fit_binding_admissible = (
        parameter_replay_admissible if fit_parameter_replay is not None else native_fit_binding
    )
    native_observation = _uses_native_pixel_center_observations(trusted_recipe)
    measured_projection_qualification_admissible = bool(
        (
            measured_projection.get("qualification_role") == "authoritative_frozen_observation"
            and measured_projection.get("status") == "COMPLETE"
            and measured_projection.get("projection_performed") is False
            and measured_projection.get("refinement_oracle") == "NOT_APPLICABLE_EXACT_MEMBERSHIP"
        )
        if native_observation
        else (
            measured_projection.get("qualification_role", "required_release_gate")
            == "required_release_gate"
            and _projection_convergence_is_admissible(
                refinement_oracle,
                expected_dataset_ids=(display_dataset_id,),
                expected_limit=float(trusted_model["maximum_relative_l2"]),
            )
        )
    )
    measured_projection_contract_admissible = bool(
        (
            measured_projection.get("method") == NATIVE_PIXEL_CENTER_METHOD
            and measured_projection.get("model_pixelized") is False
            and _is_sha256(measured_projection.get("projection_revision"))
            and isinstance(measured_projection.get("selected_pixel_region_pair_count"), int)
            and not isinstance(measured_projection.get("selected_pixel_region_pair_count"), bool)
            and measured_projection["selected_pixel_region_pair_count"] > 0
            and all(
                _is_sha256(measured_projection.get(name))
                for name in (
                    "selected_flat_pixel_index_sha256",
                    "selected_observation_row_sha256",
                    "count_mass_sha256",
                    "support_px2_sha256",
                    "count_covariance_sha256",
                )
            )
            and execution.get("measured_observation_method") == NATIVE_PIXEL_CENTER_METHOD
            and execution.get("measured_projection_performed") is False
            and execution.get("measured_projection_revision")
            == measured_projection.get("projection_revision")
            and execution.get("measured_selected_pixel_region_pair_count")
            == measured_projection.get("selected_pixel_region_pair_count")
            and execution.get("measured_selected_flat_pixel_index_sha256")
            == measured_projection.get("selected_flat_pixel_index_sha256")
            and execution.get("measured_selected_observation_row_sha256")
            == measured_projection.get("selected_observation_row_sha256")
            and execution.get("measured_count_mass_sha256")
            == measured_projection.get("count_mass_sha256")
            and execution.get("measured_support_px2_sha256")
            == measured_projection.get("support_px2_sha256")
            and execution.get("measured_count_covariance_sha256")
            == measured_projection.get("count_covariance_sha256")
        )
        if native_observation
        else (
            measured_projection.get("method") == MEASURED_PROJECTION_METHOD
            and measured_projection.get("gauss_order") == trusted_model["oracle_gauss_order"]
            and measured_projection.get("subdivision_count")
            == trusted_model["oracle_subdivision_count"]
            and measured_projection.get("m0_phi_subdivision_count")
            == trusted_model["oracle_subdivision_count"]
            and all(
                _is_sha256(measured_projection.get(name))
                for name in (
                    "coarse_projection_revision",
                    "projection_revision",
                    "display_fit_projection_revision",
                    "quadrature_revision",
                    "count_mass_sha256",
                    "count_covariance_sha256",
                )
            )
        )
    )
    m0_signal_only_hashes_admissible = all(
        _is_sha256(m0_signal_only.get(name))
        for name in (
            "count_mass_sha256",
            "count_covariance_sha256",
            "radial_background_mass_sha256",
            "model_mass_sha256",
            "detector_overlay_flat_pixel_sha256",
        )
    )
    m0_signal_only_common = bool(
        m0_signal_only.get("fit_role") == "not_used_in_fit_objective_or_parameter_estimation"
        and m0_signal_only.get("background_conditioning")
        == "fixed_radial_only_no_adjacent_sideband_conditioning"
        and m0_signal_only.get("gauss_order") == trusted_model["oracle_gauss_order"]
        and m0_signal_only.get("subdivision_count") == trusted_model["oracle_subdivision_count"]
        and (
            m0_signal_only.get("measured_phi_subdivision_count") is None
            if native_observation
            else m0_signal_only.get("measured_phi_subdivision_count")
            == trusted_model["oracle_subdivision_count"]
        )
        and m0_signal_only.get("model_phi_subdivision_count")
        == trusted_model["oracle_subdivision_count"]
        and m0_signal_only.get("smoothing_applied") is False
        and m0_signal_only.get("model_pixelized") is False
        and m0_signal_only_hashes_admissible
        and execution.get("m0_signal_only_bin_count") == supplemental_bin_count
        and execution.get("m0_signal_only_projection_revision")
        == m0_signal_only.get("projection_revision")
        and execution.get("m0_signal_only_measured_quadrature_revision")
        == m0_signal_only.get("measured_quadrature_revision")
        and execution.get("m0_signal_only_model_quadrature_revision")
        == m0_signal_only.get("model_quadrature_revision")
        and execution.get("m0_signal_only_count_mass_sha256")
        == m0_signal_only.get("count_mass_sha256")
        and execution.get("m0_signal_only_count_covariance_sha256")
        == m0_signal_only.get("count_covariance_sha256")
        and execution.get("m0_signal_only_radial_background_mass_sha256")
        == m0_signal_only.get("radial_background_mass_sha256")
        and execution.get("m0_signal_only_model_mass_sha256")
        == m0_signal_only.get("model_mass_sha256")
    )
    if supplemental_bin_count:
        m0_signal_only_admissible = bool(
            m0_signal_only_common
            and m0_signal_only.get("status") == "DISPLAY_ONLY_INCOMPLETE_SIDEBAND_SUPPLEMENT"
            and m0_signal_only.get("projection_method")
            == (NATIVE_PIXEL_CENTER_METHOD if native_observation else MEASURED_PROJECTION_METHOD)
            and _is_sha256(m0_signal_only.get("projection_revision"))
            and _is_sha256(m0_signal_only.get("model_quadrature_revision"))
            and (
                m0_signal_only.get("measured_quadrature_revision") is None
                if native_observation
                else _is_sha256(m0_signal_only.get("measured_quadrature_revision"))
            )
            and isinstance(m0_signal_only.get("measured_continuous_node_count"), int)
            and not isinstance(m0_signal_only.get("measured_continuous_node_count"), bool)
            and m0_signal_only["measured_continuous_node_count"]
            == (0 if native_observation else m0_signal_only["measured_continuous_node_count"])
            and (native_observation or m0_signal_only["measured_continuous_node_count"] > 0)
            and isinstance(m0_signal_only.get("model_continuous_node_count"), int)
            and not isinstance(m0_signal_only.get("model_continuous_node_count"), bool)
            and m0_signal_only["model_continuous_node_count"] > 0
            and isinstance(m0_signal_only.get("detector_overlay_flat_pixel_count"), int)
            and not isinstance(m0_signal_only.get("detector_overlay_flat_pixel_count"), bool)
            and m0_signal_only["detector_overlay_flat_pixel_count"] > 0
            and isinstance(m0_signal_only.get("execution"), dict)
            and m0_signal_only["execution"].get("model_measure") == "continuous_detector_chart_area"
        )
    else:
        m0_signal_only_admissible = bool(
            m0_signal_only_common
            and m0_signal_only.get("status") == "NOT_REQUIRED_ALL_M0_BINS_CONDITIONED"
            and m0_signal_only.get("projection_method") is None
            and m0_signal_only.get("projection_revision") is None
            and m0_signal_only.get("measured_quadrature_revision") is None
            and m0_signal_only.get("model_quadrature_revision") is None
            and m0_signal_only.get("measured_continuous_node_count") == 0
            and m0_signal_only.get("model_continuous_node_count") == 0
            and m0_signal_only.get("detector_overlay_flat_pixel_count") == 0
            and m0_signal_only.get("execution") is None
        )
    common = bool(
        manifest.get("computationally_valid") is True
        and manifest.get("model_pixelized") is False
        and manifest.get("smoothing_applied") is False
        and int(manifest.get("model_rod_count", -1)) == len(roster)
        and manifest.get("model_rod_roster_sha256") == roster_sha256
        and manifest.get("fit_model_rod_roster_sha256") == roster_sha256
        and families == set(FAMILIES)
        and all(_is_sha256(value) for value in evaluator_hashes)
        and fit_binding_admissible
        and manifest.get("figure_recipe") == effective_trusted_recipe
        and manifest.get("stacking_model") == _fault_free_three_r_definition()
        and _fixed_displacement_gauge_is_admissible(manifest.get("structure_representative"))
        and _vacancy_structure_representative_is_admissible(
            manifest.get("structure_representative")
        )
        and dark == prepared_dark_correction
        and dark.get("model_id") == DARK_CORRECTION_MODEL
        and dark.get("negative_values_clipped") is False
        and dark.get("smoothing_applied") is False
        and dark.get("covariance_model")
        == _dark_covariance_model(float(dark.get("scale", math.nan)))
        and _dark_scale_basis_is_valid(
            float(dark.get("scale", math.nan)),
            dark.get("scale_basis"),
        )
        and dark.get("scale_basis") == trusted_recipe["dark_correction"].get("scale_basis")
        and _is_sha256(dark.get("file_sha256"))
        and _is_sha256(dark.get("detector_native_bytes_sha256"))
        and isinstance(dark.get("detector_native_shape_rc"), list)
        and len(dark["detector_native_shape_rc"]) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in dark["detector_native_shape_rc"]
        )
        and isinstance(dark.get("detector_native_dtype"), str)
        and bool(dark["detector_native_dtype"])
        and float(dark.get("scale", math.nan)) == float(trusted_recipe["dark_correction"]["scale"])
        and cubature.get("settings") == trusted_recipe["profile_cubature"]
        and execution.get("fit_gauss_order")
        == int(trusted_recipe["profile_cubature"]["fit_gauss_order"])
        and execution.get("fit_subdivision_count")
        == int(trusted_recipe["profile_cubature"]["fold_fit_subdivisions"])
        and execution.get("m0_model_phi_subdivision_count")
        == int(trusted_recipe["profile_cubature"]["fold_fit_subdivisions"])
        and execution.get("rod_scope") == "fitted_families_m_0_1_3_4"
        and execution.get("rod_count") == len(roster)
        and execution.get("rod_roster_sha256") == roster_sha256
        and execution.get("diagnostic_sha256") == provenance.get("fit_diagnostic_sha256")
        and execution.get("fit_sha256") == provenance.get("fit_sha256")
        and execution.get("recipe_sha256") == provenance.get("recipe_sha256")
        and execution.get("fit_plan_sha256") == provenance_fit_plan_sha256
        and execution.get("implementation_sha256") == provenance_implementation_sha256
        and execution.get("osc_sha256") == provenance.get("osc_sha256")
        and execution.get("dark_osc_sha256")
        == provenance.get("dark_osc_sha256")
        == dark.get("file_sha256")
        and execution.get("dark_scale") == float(dark["scale"])
        and execution.get("fit_chain_sha256") == provenance_fit_chain_sha256
        and _joint_cubature_is_admissible(
            fitted_region_cubature,
            trusted_recipe=trusted_recipe,
        )
        and _data_projection_is_admissible(
            fitted_data_projection,
            trusted_recipe=trusted_recipe,
            expected_dataset_ids=expected_dataset_ids,
        )
        and measured_projection_contract_admissible
        and measured_projection.get("smoothing_applied") is False
        and cubature.get("m0_model_phi_subdivision_count")
        == int(trusted_recipe["profile_cubature"]["fold_fit_subdivisions"])
        and m0_signal_only_admissible
        and measured_projection_qualification_admissible
        and provenance.get("background_sha256") == background_model.get("artifact_sha256")
        and execution.get("background_artifact_sha256") == background_model.get("artifact_sha256")
        and execution.get("radial_background_state_revision")
        == background_model.get("state_revision")
        and execution.get("radial_background_parameter_vector_sha256")
        == background_model.get("parameter_vector_sha256")
        and execution.get("radial_background_parameter_covariance_sha256")
        == background_model.get("parameter_covariance_sha256")
        and execution.get("radial_background_mass_sha256")
        == background_model.get("radial_mass_sha256")
        and execution.get("radial_background_covariance_sha256")
        == background_model.get("radial_covariance_sha256")
        and execution.get("background_excluded_flat_pixel_count_by_dataset")
        == background_model.get("excluded_flat_pixel_count_by_dataset")
        and execution.get("background_excluded_flat_pixel_sha256_by_dataset")
        == background_model.get("excluded_flat_pixel_sha256_by_dataset")
        and _background_model_identity_is_admissible(
            background_model,
            expected_dataset_ids=expected_dataset_ids,
        )
        and execution.get("conditioned_background_revision")
        == background_model.get("conditioned_revision")
        and execution.get("conditioned_background_mass_sha256")
        == background_model.get("conditioned_mass_sha256")
        and execution.get("conditioned_background_covariance_sha256")
        == background_model.get("conditioned_covariance_sha256")
        and execution.get("conditioned_model_anchor_projection_sha256")
        == background_model.get("anchor_projection_sha256")
    )
    return bool(
        common
        and manifest.get("evidence_level")
        == (
            PROFILE_PARAMETER_REPLAY_EVIDENCE
            if fit_parameter_replay is not None
            else "FIT_CONDITIONED"
        )
        and manifest.get("publication_ready") is False
        and manifest.get("fit_status") in {"FIT", "MODEL_LIMITED_FIT"}
        and manifest.get("all_rod_validation") == "NOT_REQUIRED_FOR_RENDER"
        and manifest.get("rod_scope_validation_status") == "NOT_RUN"
        and manifest.get("model_rod_scope") == "fitted_families_m_0_1_3_4"
        and cubature.get("evaluated_passes") == ["fit"]
        and cubature.get("full_profile_oracle_performed") is False
    )


def _validate_profile_selection_coordinates(arrays: dict[str, np.ndarray]) -> None:
    """Require the declared profile abscissa: 2theta for m=0 and L otherwise."""

    try:
        identities = np.asarray(arrays["profile_identity"])
        coordinate = np.asarray(arrays["profile_selection_coordinate"])
        kinds = np.asarray(arrays["profile_selection_coordinate_kind"])
        valid = np.asarray(arrays["profile_valid"], dtype=np.bool_)
    except KeyError as error:
        raise ValueError("profile selection coordinate arrays are incomplete") from error
    if (
        identities.ndim != 1
        or coordinate.shape != identities.shape
        or kinds.shape != identities.shape
        or valid.shape != identities.shape
        or set(np.unique(identities))
        != {"m0", "m1_minus", "m1_plus", "m3_minus", "m3_plus", "m4_minus", "m4_plus"}
        or np.any(~np.isfinite(coordinate[valid]))
    ):
        raise ValueError("profile selection coordinate arrays are inconsistent")
    expected = np.where(identities == "m0", "two_theta_deg", "L")
    if np.any(kinds != expected):
        raise ValueError("profile selection coordinate kind does not match its family")


def _load_profiles(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, manifest = _load_diagnostic(path, expected_schema=PROFILE_SCHEMA)
    provenance = manifest.get("provenance", {})
    recipe_path = _verified_provenance_path(manifest, "recipe")
    if _sha256(recipe_path) != provenance.get("recipe_sha256"):
        raise ValueError("profile recipe changed after profile generation")
    verified_inputs: dict[str, dict[str, str]] = {}
    # Rendering consumes the frozen, hash-checked arrays. Producer code and the fit plan remain
    # provenance rather than compatibility gates so display-only updates never require a refit.
    for role, path_name, hash_name in (
        ("prepared diagnostic", "fit_diagnostic", "fit_diagnostic_sha256"),
        ("fit artifact", "fit", "fit_sha256"),
        ("radial background", "background", "background_sha256"),
        ("display OSC", "osc", "osc_sha256"),
        ("dark OSC", "dark_osc", "dark_osc_sha256"),
    ):
        verified_inputs[role] = _verified_recorded_file_identity(
            {"path": provenance.get(path_name), "sha256": provenance.get(hash_name)},
            role=role,
        )
    for role, record in provenance.get("model_inputs", {}).items():
        _verified_recorded_file_identity(record, role=f"model input {role}")
    for index, record in enumerate(provenance.get("fit_chain", ())):
        _verified_recorded_file_identity(record, role=f"fit chain stage {index}")
    fit_document = json.loads(
        Path(verified_inputs["fit artifact"]["path"]).read_text(encoding="utf-8")
    )
    prepared_manifest = _load_prepared_manifest(
        Path(verified_inputs["prepared diagnostic"]["path"])
    )
    prepared_dark_correction = prepared_manifest.get("dark_correction")
    if not isinstance(prepared_dark_correction, dict):
        raise ValueError("prepared diagnostic lacks a dark-correction record")
    if (
        fit_document.get("model_rod_roster_sha256") != manifest.get("fit_model_rod_roster_sha256")
        or fit_document.get("structure_representative") != manifest.get("structure_representative")
        or fit_document.get("stacking_model") != manifest.get("stacking_model")
    ):
        raise ValueError("profile scientific state does not match its fit artifact")
    fit_parameter_replay = manifest.get("fit_parameter_replay")
    if fit_parameter_replay is not None:
        fit_scales = {
            str(dataset_id): float(value)
            for dataset_id, value in fit_document.get("dataset_scales", {}).items()
        }
        fit_stitch = fit_document.get("specular_stitch")
        if (
            not isinstance(fit_parameter_replay, dict)
            or not isinstance(fit_stitch, dict)
            or fit_parameter_replay.get("source_fit_sha256")
            != verified_inputs["fit artifact"]["sha256"]
            or fit_parameter_replay.get("structure_parameter_vector_sha256")
            != _array_sha256(_fit_structure_vector(fit_document))
            or fit_parameter_replay.get("dataset_scales") != fit_scales
            or fit_parameter_replay.get("fit_interface_assumption")
            != fit_stitch.get("interface_assumption")
        ):
            raise ValueError("profile parameter replay does not match its source fit")
    trusted_recipe = _load_recipe(recipe_path)
    missing = [name for name in PROFILE_RENDER_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"profile diagnostic lacks required arrays {missing}")
    vector_shape = arrays[PROFILE_VECTOR_ARRAY_NAMES[0]].shape
    m0_signal_only_shape = arrays[M0_SIGNAL_ONLY_ARRAY_NAMES[0]].shape
    render_hashes = manifest.get("render_array_sha256")
    _validate_profile_selection_coordinates(arrays)
    if (
        len(vector_shape) != 1
        or any(arrays[name].shape != vector_shape for name in PROFILE_VECTOR_ARRAY_NAMES)
        or len(m0_signal_only_shape) != 1
        or any(arrays[name].shape != m0_signal_only_shape for name in M0_SIGNAL_ONLY_ARRAY_NAMES)
        or manifest.get("m0_signal_only_display", {}).get("supplemental_bin_count")
        != m0_signal_only_shape[0]
        or arrays["display_detector_counts"].ndim != 2
        or arrays["display_full_region_code"].shape != arrays["display_detector_counts"].shape
        or arrays["display_fit_region_code"].shape != arrays["display_detector_counts"].shape
        or not isinstance(render_hashes, dict)
        or set(render_hashes) != set(PROFILE_RENDER_ARRAY_NAMES)
        or any(
            not _is_sha256(render_hashes[name])
            or render_hashes[name] != _array_sha256(arrays[name])
            for name in PROFILE_RENDER_ARRAY_NAMES
        )
        or not _profile_manifest_is_admissible(
            manifest,
            trusted_recipe=trusted_recipe,
            prepared_dark_correction=prepared_dark_correction,
        )
    ):
        raise ValueError("profile diagnostic violates the rendering contract")
    return arrays, manifest


def _site_displacement_profile(
    strength: Any,
    fit_plan: dict[str, Any],
) -> SiteDisplacementProfile:
    bi_label, center_label, outer_label = quintuple_layer_site_labels(strength.crystal)
    labels_by_role = {
        "bi": bi_label,
        "central_chalcogen": center_label,
        "outer_chalcogen": outer_label,
    }
    settings = fit_plan["site_displacement_profile"]
    return SiteDisplacementProfile(
        sites=tuple(
            TransverseIsotropicSiteDisplacement(
                source_label=labels_by_role[str(record["role"])],
                u_radial_A2=float(record["u_radial_A2"]),
                u_normal_A2=float(record["u_normal_A2"]),
            )
            for record in settings["site"]
        ),
        scale=1.0,
        provenance=str(settings["provenance"]),
    )


def _candidate_strength(
    strength: Any,
    parameters: np.ndarray,
    *,
    fit_plan: dict[str, Any],
) -> Any:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (STRUCTURE_PARAMETER_COUNT,) or np.any(~np.isfinite(values)):
        raise ValueError("structure parameters must contain five finite values")
    baseline = strength.structure_parameters
    candidate_parameters = replace(
        baseline,
        bi_fractional_z=baseline.bi_fractional_z + float(values[0]),
        se2_fractional_z=baseline.se2_fractional_z + float(values[1]),
        bi_occupancy=1.0,
        se1_occupancy=1.0,
        se2_occupancy=1.0 - float(values[2]),
        outer_bi_antisite_fraction=0.0,
        u_radial_A2=0.0,
        u_normal_A2=0.0,
    )
    return replace(
        strength,
        structure_parameters=candidate_parameters,
        site_displacement_profile=_site_displacement_profile(strength, fit_plan),
    )


def _candidate_intensity_envelope(parameters: np.ndarray) -> SampleQIntensityEnvelope:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (STRUCTURE_PARAMETER_COUNT,) or np.any(~np.isfinite(values)):
        raise ValueError("structure parameters must contain five finite values")
    return SampleQIntensityEnvelope(
        u_radial_A2=float(values[3]),
        u_normal_A2=float(values[4]),
    )


def _fit_structure_vector(document: dict[str, Any]) -> np.ndarray:
    values = np.asarray(document.get("full_parameter_vector", ()), dtype=np.float64)
    if values.shape != (STRUCTURE_PARAMETER_COUNT,) or np.any(~np.isfinite(values)):
        raise ValueError("fit artifact lacks one finite five-coordinate structure state")
    return values


def _verified_provenance_path(
    manifest: dict[str, Any],
    name: str,
) -> Path:
    provenance = manifest["provenance"]
    expected_sha256 = str(provenance[f"{name}_sha256"])
    path = Path(provenance[name]).resolve()
    if path.is_file() and _sha256(path) == expected_sha256:
        return path
    raise ValueError(f"prepared {name} changed after observation freezing")


def _qualified_prepared_lattice(
    manifest: dict[str, Any],
    fixed_lattice: FixedLatticeState,
    rod_catalog_revision: str,
) -> dict[str, Any]:
    """Bind an adopted lattice and its rebuilt rod catalog to one preparation."""
    record = fixed_lattice.to_record()
    prepared_record = manifest.get("fixed_lattice")
    prepared_rod_revision = manifest.get("rod_catalog_revision")
    if prepared_record != record:
        raise ValueError("rebuilt lattice differs from the prepared observation")
    if prepared_rod_revision != rod_catalog_revision:
        raise ValueError("rebuilt rod catalog differs from the prepared observation")
    return record


MODEL_INPUT_NAMES = ("fixed_state", "simulation_config")


def _fault_free_three_r_definition() -> dict[str, Any]:
    return {
        "model_id": FAULT_FREE_THREE_R_MODEL,
        "parent": Parent.THREE_R.value,
        "shared_disorder_epsilon": 0.0,
        "transition_probabilities": [0.0, 0.0, 1.0, 0.0, 0.0],
        "initial_population": "plus_only",
        "evaluation": "generalized_transition_law_with_exact_one_hot_optimized_limit",
    }


def _fault_free_three_r_record(series: Sequence[Any]) -> dict[str, Any]:
    """Prove that every dataset uses the exact 3R limit of the stacking model."""

    laws: list[np.ndarray] = []
    for inputs in series:
        strength = inputs.strength
        if strength.parent is not Parent.THREE_R or strength.shared_disorder_epsilon != 0.0:
            raise ValueError("this fit requires fault-free 3R through the stacking model")
        laws.append(
            RichEpsilonModel(
                Parent.THREE_R,
                strength.shared_disorder_epsilon,
            )
            .transition_law()
            .as_array()
        )
    expected = np.asarray((0.0, 0.0, 1.0, 0.0, 0.0), dtype=np.float64)
    if any(not np.array_equal(law, expected) for law in laws):
        raise RuntimeError("fault-free 3R transition law changed")
    return _fault_free_three_r_definition()


def _rebuilt_series(manifest: dict[str, Any]) -> tuple[Any, ...]:
    for name in (*MODEL_INPUT_NAMES, "recipe"):
        _verified_provenance_path(manifest, name)
    (
        _,
        _,
        _,
        series,
        fixed_position,
        fixed_lattice,
        mosaic_parameters,
        _,
        source_revision,
        cif_sha256,
        rod_catalog_revision,
    ) = _fixed_experiment_inputs(
        _verified_provenance_path(manifest, "fixed_state"),
        expected_material_id=str(manifest["material_id"]),
        expected_dataset_ids=tuple(str(value) for value in manifest["dataset_ids"]),
        expected_source_state_count=int(manifest["source_state_count"]),
    )
    if (
        source_revision != manifest["source_revision"]
        or cif_sha256 != manifest["cif_sha256"]
        or mosaic_parameters != manifest["mosaic"]
        or fixed_position.to_record() != manifest["fixed_position"]
    ):
        raise ValueError("rebuilt fixed experiment differs from the prepared observation")
    _qualified_prepared_lattice(manifest, fixed_lattice, rod_catalog_revision)
    if manifest.get("stacking_model") != _fault_free_three_r_record(series):
        raise ValueError("rebuilt stacking model differs from the prepared observation")
    return series


def _prepared_lattice_record(
    manifest: dict[str, Any],
    series: Sequence[Any],
) -> dict[str, Any]:
    del series
    record = manifest.get("fixed_lattice")
    if not isinstance(record, dict):
        raise ValueError("prepared diagnostic lacks its fixed lattice state")
    return dict(record)


def _verified_osc_counts(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    counts_by_dataset: dict[str, np.ndarray] = {}
    for record in manifest["osc_provenance"]:
        dataset_id = str(record["dataset_id"])
        osc_path = Path(str(record["osc_path"])).resolve()
        if _sha256(osc_path) != record["osc_file_sha256"]:
            raise ValueError(f"OSC file changed for {dataset_id!r}")
        counts = read_osc(osc_path).detector_native_counts
        if (
            counts.ndim != 2
            or hashlib.sha256(counts.tobytes(order="C")).hexdigest()
            != record["detector_native_bytes_sha256"]
        ):
            raise ValueError(f"decoded OSC counts changed for {dataset_id!r}")
        counts_by_dataset[dataset_id] = counts
    expected = tuple(str(value) for value in manifest["dataset_ids"])
    if tuple(counts_by_dataset) != expected:
        raise ValueError("OSC provenance order differs from the prepared dataset order")
    return counts_by_dataset


def _native_pixel_center_plans(
    arrays: dict[str, np.ndarray],
    *,
    dataset_ids: Sequence[str],
    detector_shape_rc: tuple[int, int],
) -> tuple[_DatasetNativePixelCenterPlan, ...]:
    """Compile the frozen unit-weight native-pixel memberships."""

    dataset_ids = tuple(str(value) for value in dataset_ids)
    row_dataset = np.asarray(arrays["dataset_index"], dtype=np.int64)
    member_dataset = np.asarray(arrays["selected_dataset_index"], dtype=np.int64)
    member_pixel = np.asarray(arrays["selected_flat_pixel_index"], dtype=np.int64)
    member_row = np.asarray(arrays["selected_observation_row"], dtype=np.int64)
    row_count = row_dataset.size
    member_shape = member_dataset.shape
    if (
        row_dataset.ndim != 1
        or member_dataset.ndim != 1
        or member_pixel.shape != member_shape
        or member_row.shape != member_shape
        or np.any((row_dataset < 0) | (row_dataset >= len(dataset_ids)))
        or np.any((member_dataset < 0) | (member_dataset >= len(dataset_ids)))
        or np.any((member_row < 0) | (member_row >= row_count))
        or np.any(member_pixel < 0)
        or np.any(row_dataset[member_row] != member_dataset)
    ):
        raise ValueError("native-pixel-center membership arrays are inconsistent")
    plans: list[_DatasetNativePixelCenterPlan] = []
    for dataset_index, dataset_id in enumerate(dataset_ids):
        global_row = np.flatnonzero(row_dataset == dataset_index)
        selected = member_dataset == dataset_index
        if not global_row.size or not np.any(selected):
            raise ValueError(f"dataset {dataset_id!r} lacks native-pixel-center support")
        global_to_local = np.full(row_count, -1, dtype=np.int64)
        global_to_local[global_row] = np.arange(global_row.size, dtype=np.int64)
        local_row = global_to_local[member_row[selected]]
        flat_pixel = member_pixel[selected]
        pair_key = local_row * math.prod(detector_shape_rc) + flat_pixel
        if (
            np.unique(pair_key).size != pair_key.size
            or np.unique(flat_pixel).size != flat_pixel.size
        ):
            raise ValueError("native-pixel-center membership contains overlapping fitted regions")
        used_pixel, pixel_column = np.unique(flat_pixel, return_inverse=True)
        plans.append(
            _DatasetNativePixelCenterPlan(
                dataset_index=dataset_index,
                dataset_id=dataset_id,
                global_observation_row=global_row,
                projection=NativePixelRegionProjection(
                    detector_shape_rc=detector_shape_rc,
                    flat_pixel_index=used_pixel,
                    observation_row=local_row,
                    pixel_column_index=pixel_column,
                    detector_area_weight_px2=np.ones(local_row.size, dtype=np.float64),
                    observation_count=global_row.size,
                    quadrature_revision=NATIVE_PIXEL_CENTER_METHOD,
                ),
            )
        )
    global_order = np.concatenate([plan.global_observation_row for plan in plans])
    if not np.array_equal(np.sort(global_order), np.arange(row_count)):
        raise ValueError("native-pixel-center plans do not partition the observation rows")
    return tuple(plans)


def _native_pixel_center_count_statistics(
    arrays: dict[str, np.ndarray],
    *,
    dataset_ids: Sequence[str],
    counts_by_dataset: dict[str, np.ndarray],
    dark_counts: np.ndarray,
    dark_scale: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[_DatasetNativePixelCenterPlan, ...],
]:
    """Reduce raw-minus-dark counts on the frozen native pixel-center support."""

    dataset_ids = tuple(str(value) for value in dataset_ids)
    if tuple(counts_by_dataset) != dataset_ids:
        raise ValueError("native-pixel-center count roster differs from the dataset roster")
    shapes = {np.asarray(counts_by_dataset[value]).shape for value in dataset_ids}
    if len(shapes) != 1:
        raise ValueError("native-pixel-center datasets must share one detector shape")
    detector_shape = tuple(int(value) for value in next(iter(shapes)))
    dark = np.asarray(dark_counts, dtype=np.float64)
    if (
        len(detector_shape) != 2
        or dark.shape != detector_shape
        or np.any(~np.isfinite(dark))
        or np.any(dark < 0.0)
        or not math.isfinite(float(dark_scale))
        or float(dark_scale) < 0.0
    ):
        raise ValueError("native-pixel-center dark correction is invalid")
    plans = _native_pixel_center_plans(
        arrays,
        dataset_ids=dataset_ids,
        detector_shape_rc=detector_shape,
    )
    row_count = np.asarray(arrays["dataset_index"]).size
    raw_mass = np.zeros(row_count, dtype=np.float64)
    covariance = np.zeros((row_count, row_count), dtype=np.float64)
    support = np.zeros(row_count, dtype=np.float64)
    for plan in plans:
        local_mass, local_covariance = plan.projection.integrate_counts(
            counts_by_dataset[plan.dataset_id]
        )
        global_row = np.asarray(plan.global_observation_row, dtype=np.int64)
        raw_mass[global_row] = local_mass
        covariance[np.ix_(global_row, global_row)] = local_covariance
        support[global_row] = plan.projection.observation_measure_px2
    expected_raw_mass = np.asarray(arrays["count_sum"], dtype=np.float64)
    expected_support = np.asarray(arrays["support_px2"], dtype=np.float64)
    if not np.array_equal(raw_mass, expected_raw_mass):
        raise ValueError("native-pixel-center raw counts differ from the prepared count sums")
    if not np.array_equal(support, expected_support):
        raise ValueError("native-pixel-center support differs from the prepared membership")
    dark_mass, dark_covariance = integrate_shared_native_pixel_field(
        [plan.projection for plan in plans],
        dark,
        np.maximum(dark, 1.0),
    )
    global_order = np.concatenate([plan.global_observation_row for plan in plans])
    corrected_mass = np.array(raw_mass, copy=True)
    corrected_mass[global_order] -= float(dark_scale) * dark_mass
    covariance[np.ix_(global_order, global_order)] += float(dark_scale) ** 2 * dark_covariance
    return corrected_mass, covariance, support, plans


def _native_pixel_center_observations(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    counts_by_dataset: dict[str, np.ndarray],
) -> tuple[MatchedRegionObservations, tuple[_DatasetNativePixelCenterPlan, ...]]:
    dark_counts, dark_scale = _verified_dark_counts(manifest)
    count_mass, covariance, support, plans = _native_pixel_center_count_statistics(
        arrays,
        dataset_ids=manifest["dataset_ids"],
        counts_by_dataset=counts_by_dataset,
        dark_counts=dark_counts,
        dark_scale=dark_scale,
    )
    return (
        MatchedRegionObservations(
            dataset_ids=tuple(str(value) for value in manifest["dataset_ids"]),
            dataset_index=arrays["dataset_index"],
            block_index=arrays["block_index"],
            signal_family=arrays["signal_family_m"],
            is_background=arrays["is_background"],
            count_mass=count_mass,
            support_px2=support,
            background_coordinate=arrays["coordinate_mean"],
            required_signal_families=FAMILIES,
            count_covariance_count2=covariance,
        ),
        plans,
    )


def _continuous_matched_observations(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    plans: Sequence[_DatasetContinuousRegionPlan],
    counts_by_dataset: dict[str, np.ndarray],
) -> tuple[MatchedRegionObservations, tuple[str, ...], tuple[np.ndarray, ...]]:
    row_count = arrays["count_sum"].size
    count_mass = np.zeros(row_count, dtype=np.float64)
    count_covariance = np.zeros((row_count, row_count), dtype=np.float64)
    support = np.zeros(row_count, dtype=np.float64)
    coordinate = np.zeros(row_count, dtype=np.float64)
    covered = np.zeros(row_count, dtype=np.bool_)
    projection_revisions: list[str] = []
    projection_pixels: list[np.ndarray] = []
    projections = []
    for plan in plans:
        global_row = np.asarray(plan.global_observation_row, dtype=np.int64)
        counts = counts_by_dataset[plan.dataset_id]
        projection = compile_native_pixel_region_projection(plan.quadrature, counts.shape)
        local_mass, local_covariance = projection.integrate_counts(counts)
        count_mass[global_row] = local_mass
        count_covariance[np.ix_(global_row, global_row)] = local_covariance
        support[global_row] = projection.observation_measure_px2
        coordinate[global_row] = plan.quadrature.observation_background_coordinate
        covered[global_row] = True
        projection_revisions.append(projection.projection_revision)
        projection_pixels.append(np.asarray(projection.flat_pixel_index, dtype=np.int64))
        projections.append(projection)
    dark_counts, dark_scale = _verified_dark_counts(manifest)
    dark_local_mass, dark_local_covariance = integrate_shared_native_pixel_field(
        projections,
        dark_counts,
        np.maximum(dark_counts, 1.0),
    )
    global_order = np.concatenate(
        [np.asarray(plan.global_observation_row, dtype=np.int64) for plan in plans]
    )
    if not np.array_equal(np.sort(global_order), np.arange(row_count)):
        raise ValueError("continuous dataset plans do not partition the observation rows")
    count_mass[global_order] -= dark_scale * dark_local_mass
    count_covariance[np.ix_(global_order, global_order)] += (
        dark_scale * dark_scale * dark_local_covariance
    )
    if not np.all(covered) or np.any(support <= 0.0):
        raise ValueError("continuous quadrature must positively cover every fitted observation")
    return (
        MatchedRegionObservations(
            dataset_ids=tuple(str(value) for value in manifest["dataset_ids"]),
            dataset_index=arrays["dataset_index"],
            block_index=arrays["block_index"],
            signal_family=arrays["signal_family_m"],
            is_background=arrays["is_background"],
            count_mass=count_mass,
            support_px2=support,
            background_coordinate=coordinate,
            required_signal_families=FAMILIES,
            count_covariance_count2=count_covariance,
        ),
        tuple(projection_revisions),
        tuple(projection_pixels),
    )


def _integrate_continuous_region_mass(
    detector: Any,
    plan: _DatasetContinuousRegionPlan,
    fold_plan: ContinuousFoldCorrectionPlan | None,
    *,
    row_count: int,
    execution_backend: str,
    cuda_coordinate_chunk_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = perf_counter()
    evaluated = detector.evaluate_detector_density_all_roots(
        plan.quadrature.column_px,
        plan.quadrature.row_px,
        execution_backend=execution_backend,
        cuda_coordinate_chunk_size=(
            cuda_coordinate_chunk_size if execution_backend == "cuda" else None
        ),
    )
    density = np.asarray(evaluated.density_A2_per_px2)
    if np.any(~np.isfinite(density)):
        raise FloatingPointError(
            "continuous quadrature encountered a singular model node; subdivide its chart cell"
        )
    local_mass = plan.quadrature.integrate_density(density)
    fold_correction = None
    if fold_plan is not None:
        fold_correction = apply_continuous_fold_correction_plan(
            detector,
            fold_plan,
            execution_backend=execution_backend,
            cuda_coordinate_chunk_size=(
                cuda_coordinate_chunk_size if execution_backend == "cuda" else None
            ),
        )
        local_mass = local_mass + np.asarray(fold_correction.correction_A2)
    model_mass = np.zeros(row_count, dtype=np.float64)
    model_mass[np.asarray(plan.global_observation_row, dtype=np.int64)] = local_mass
    device = evaluated.execution_device
    return model_mass, {
        "model_measure": "continuous_detector_chart_area",
        "quadrature_revision": plan.quadrature.quadrature_revision,
        "chart_revision": plan.quadrature.chart_revision,
        "rectangle_count": plan.rectangle_count,
        "continuous_node_count": int(plan.quadrature.column_px.size),
        "continuous_fold_plan_sha256": (None if fold_plan is None else fold_plan.plan_sha256),
        "continuous_fold_entry_count": (
            0 if fold_correction is None else fold_correction.crossing_entry_count
        ),
        "continuous_fold_baseline_node_count": (
            0 if fold_correction is None else fold_correction.baseline_node_count
        ),
        "continuous_fold_transformed_node_count": (
            0 if fold_correction is None else fold_correction.transformed_node_count
        ),
        "caustic_node_count": int(np.count_nonzero(evaluated.caustic)),
        "source_state_count": int(evaluated.source_state_count),
        "root_policy": evaluated.root_policy,
        "evaluated_backends": sorted(
            {
                str(evaluated.execution_backend),
                *(() if fold_correction is None else (str(fold_correction.execution_backend),)),
            }
        ),
        "devices": sorted(
            {
                *(() if device is None else (str(device),)),
                *(
                    ()
                    if fold_correction is None or fold_correction.execution_device is None
                    else (str(fold_correction.execution_device),)
                ),
            }
        ),
        "elapsed_seconds": perf_counter() - started,
    }


def _quadrature_restricted_to_observations(
    quadrature: ContinuousRegionQuadrature,
    observation_rows: np.ndarray,
) -> ContinuousRegionQuadrature:
    selected = np.isin(quadrature.observation_row, observation_rows)
    if not np.any(selected):
        raise ValueError("continuous quadrature restriction removed every node")
    return ContinuousRegionQuadrature(
        column_px=quadrature.column_px[selected],
        row_px=quadrature.row_px[selected],
        detector_area_weight_px2=quadrature.detector_area_weight_px2[selected],
        observation_row=quadrature.observation_row[selected],
        background_coordinate=quadrature.background_coordinate[selected],
        observation_count=quadrature.observation_count,
        chart_revision=quadrature.chart_revision,
    )


def _fold_band_restricted_to_observations(
    band: ContinuousFoldBand,
    observation_rows: np.ndarray,
) -> ContinuousFoldBand:
    selected_interval = np.isin(band.observation_row, observation_rows)
    if not np.any(selected_interval):
        raise ValueError("continuous fold-band restriction removed every interval")
    return ContinuousFoldBand(
        chart=band.chart,
        radial_interval_Ainv=band.radial_interval_Ainv,
        axial_bounds=band.axial_bounds[selected_interval],
        observation_row=band.observation_row[selected_interval],
        observation_count=band.observation_count,
        rod_h_k=band.rod_h_k,
        base_quadrature=_quadrature_restricted_to_observations(
            band.base_quadrature,
            observation_rows,
        ),
        gauss_order=band.gauss_order,
        radial_subdivisions=band.radial_subdivisions,
        axial_subdivisions=band.axial_subdivisions,
    )


def _invalid_continuous_fold_observations(
    plan: _DatasetContinuousRegionPlan,
) -> dict[int, str]:
    """Locate exact full-profile rows outside the invertible continuous chart."""

    invalid: dict[int, str] = {}
    for band in plan.fold_bands:
        node, _ = np.polynomial.legendre.leggauss(band.gauss_order)
        subcell = np.arange(band.axial_subdivisions, dtype=np.float64)
        unit_node = (
            (subcell[:, None] + 0.5 + 0.5 * node[None, :]) / band.axial_subdivisions
        ).reshape(-1)
        axial_bounds = np.asarray(band.axial_bounds, dtype=np.float64)
        axial_node = axial_bounds[:, :1] + np.diff(axial_bounds, axis=1) * unit_node[None, :]
        interval_valid = np.ones(axial_bounds.shape[0], dtype=np.bool_)
        radial_lower, radial_upper = band.radial_interval_Ainv
        for radial_Ainv in (radial_lower, 0.5 * (radial_lower + radial_upper), radial_upper):
            mapped = band.chart.map_detector_area(
                np.full(axial_node.shape, radial_Ainv, dtype=np.float64),
                axial_node,
            )
            interval_valid &= np.all(mapped.valid, axis=1)
        for observation in np.unique(band.observation_row[~interval_valid]):
            invalid[int(observation)] = "continuous fold chart left its valid detector domain"
    return invalid


def _drop_continuous_plan_observations(
    plan: _DatasetContinuousRegionPlan,
    invalid_observations: set[int],
) -> _DatasetContinuousRegionPlan:
    if not invalid_observations:
        return plan
    invalid = np.asarray(sorted(invalid_observations), dtype=np.int64)
    retained_node = ~np.isin(plan.quadrature.observation_row, invalid)
    if not np.any(retained_node):
        raise RuntimeError("continuous support screening removed every profile node")
    quadrature = ContinuousRegionQuadrature(
        column_px=plan.quadrature.column_px[retained_node],
        row_px=plan.quadrature.row_px[retained_node],
        detector_area_weight_px2=plan.quadrature.detector_area_weight_px2[retained_node],
        observation_row=plan.quadrature.observation_row[retained_node],
        background_coordinate=plan.quadrature.background_coordinate[retained_node],
        observation_count=plan.quadrature.observation_count,
        chart_revision=plan.quadrature.chart_revision,
    )
    fold_bands: list[ContinuousFoldBand] = []
    for band in plan.fold_bands:
        retained_interval = ~np.isin(band.observation_row, invalid)
        if np.any(retained_interval):
            retained_observations = np.unique(band.observation_row[retained_interval])
            fold_bands.append(_fold_band_restricted_to_observations(band, retained_observations))
    return _DatasetContinuousRegionPlan(
        dataset_index=plan.dataset_index,
        dataset_id=plan.dataset_id,
        global_observation_row=plan.global_observation_row,
        quadrature=quadrature,
        fold_bands=tuple(fold_bands),
        rectangle_count=plan.rectangle_count - len(invalid_observations),
    )


def _drop_optional_continuous_plan_observations(
    plan: _DatasetContinuousRegionPlan,
    invalid_observations: set[int],
) -> _DatasetContinuousRegionPlan | None:
    if not set(range(plan.quadrature.observation_count)).difference(invalid_observations):
        return None
    return _drop_continuous_plan_observations(plan, invalid_observations)


def fit(
    *,
    diagnostic_path: Path,
    background_path: Path,
    destination: Path,
    execution_backend: str,
    maximum_function_evaluations: int,
    fit_plan_path: Path,
    stage: str = "joint",
    predecessor_path: Path | None = None,
    resume_path: Path | None = None,
    initial_parameters_override: Sequence[float] | None = None,
) -> dict[str, Any]:
    destination = _external_file(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if resume_path is not None and initial_parameters_override is not None:
        raise ValueError("resume and explicit initial parameters are mutually exclusive")
    adapter_identity = _file_identity(Path(__file__))
    implementation_identity = _implementation_identity()
    diagnostic_identity = _file_identity(diagnostic_path)
    arrays, manifest = _load_prepared(diagnostic_path)
    recipe_path = _verified_provenance_path(manifest, "recipe")
    recipe_identity = _file_identity(recipe_path)
    fit_plan_identity = _file_identity(fit_plan_path)
    fit_plan = _load_fit_plan(fit_plan_path)
    execution_policy = str(fit_plan.get("execution_policy", "staged_A_B_C_joint.v1"))
    if fit_plan["material_id"] != manifest["material_id"]:
        raise ValueError("structure fit plan material differs from the prepared experiment")
    continuous_quadrature = fit_plan["continuous_quadrature"]
    offspecular_axial_refinement = int(continuous_quadrature["offspecular_axial_refinement"])
    offspecular_radial_transform = str(continuous_quadrature["offspecular_radial_transform"])
    offspecular_signal_minimum_radial_nodes_per_side = int(
        continuous_quadrature["offspecular_signal_minimum_radial_nodes_per_side"]
    )
    if stage not in fit_plan["stage"]:
        raise ValueError(f"unsupported structure fit stage {stage!r}")
    active_parameter_names = tuple(fit_plan["stage"][stage]["active_parameters"])
    active_index = np.asarray(
        [STRUCTURE_PARAMETER_NAMES.index(name) for name in active_parameter_names],
        dtype=np.int64,
    )
    frozen_parameter_names = tuple(
        name for name in STRUCTURE_PARAMETER_NAMES if name not in active_parameter_names
    )
    lower_full = np.asarray(fit_plan["lower_bounds"], dtype=np.float64)
    upper_full = np.asarray(fit_plan["upper_bounds"], dtype=np.float64)
    scale_full = np.asarray(fit_plan["parameter_scales"], dtype=np.float64)
    prior_mean_full = np.asarray(fit_plan["prior_mean"], dtype=np.float64)
    prior_sigma_full = np.asarray(fit_plan["prior_sigma"], dtype=np.float64)
    sensitivity_relative_tolerance = float(fit_plan["sensitivity_relative_tolerance"])
    maximum_sensitivity_condition = float(fit_plan["maximum_sensitivity_condition"])
    bound_proximity = float(fit_plan["bound_proximity_in_parameter_scales"])
    expected_predecessor_stage = fit_plan["stage"][stage].get("predecessor")
    if (predecessor_path is None) != (expected_predecessor_stage is None):
        raise ValueError(f"stage {stage!r} predecessor requirement was not satisfied")
    _require_declared_fit_start(
        execution_policy,
        initial_parameters_override=initial_parameters_override,
        resume_path=resume_path,
    )
    series = _rebuilt_series(manifest)
    fit_rod_roster = _fitted_rod_roster(series[0])
    if (
        not fit_rod_roster
        or {value[2] for value in fit_rod_roster} != set(FAMILIES)
        or any(_fitted_rod_roster(inputs) != fit_rod_roster for inputs in series[1:])
    ):
        raise ValueError("fitted rod roster differs across the prepared OSC series")
    fit_rod_roster_sha256 = _rod_roster_sha256(fit_rod_roster)
    fixed_lattice_record = _prepared_lattice_record(manifest, series)
    recipe = _load_recipe(recipe_path)
    specular_stitch = _recipe_parratt_stitch(recipe)
    model_input_identities = {
        name: _file_identity(_verified_provenance_path(manifest, name))
        for name in MODEL_INPUT_NAMES
    }
    maximum_blocks = int(recipe["model_cubature"]["maximum_state_block_count"])
    detectors = tuple(
        build_source_averaged_detector(inputs)
        .restrict_rods(tuple(rod for rod in inputs.rods if rod.family_m in FAMILIES))
        .with_specular_stitch(specular_stitch)
        .with_maximum_state_block_count(maximum_blocks)
        for inputs in series
    )
    row_count = arrays["count_sum"].size
    cuda_chunk_size = int(recipe["model_cubature"]["cuda_coordinate_chunk_size"])
    fit_order = int(recipe["model_cubature"]["fit_gauss_order"])
    fit_subdivisions = int(recipe["model_cubature"]["fold_fit_subdivisions"])
    oracle_order = int(recipe["model_cubature"]["oracle_gauss_order"])
    oracle_subdivisions = int(recipe["model_cubature"]["fold_oracle_subdivisions"])
    fit_region_plans = tuple(
        _compile_dataset_continuous_region_plan(
            inputs,
            arrays,
            manifest,
            dataset_index=dataset_index,
            gauss_order=fit_order,
            subdivision_count=fit_subdivisions,
            offspecular_axial_refinement=offspecular_axial_refinement,
            offspecular_radial_transform=offspecular_radial_transform,
            offspecular_signal_minimum_radial_nodes_per_side=(
                offspecular_signal_minimum_radial_nodes_per_side
            ),
            m0_phi_subdivision_count=fit_subdivisions,
        )
        for dataset_index, inputs in enumerate(series)
    )
    oracle_region_plans = tuple(
        _compile_dataset_continuous_region_plan(
            inputs,
            arrays,
            manifest,
            dataset_index=dataset_index,
            gauss_order=oracle_order,
            subdivision_count=oracle_subdivisions,
            offspecular_axial_refinement=offspecular_axial_refinement,
            offspecular_radial_transform=offspecular_radial_transform,
            offspecular_signal_minimum_radial_nodes_per_side=(
                offspecular_signal_minimum_radial_nodes_per_side
            ),
            m0_phi_subdivision_count=oracle_subdivisions,
        )
        for dataset_index, inputs in enumerate(series)
    )
    fit_fold_plans = tuple(
        compile_continuous_fold_correction_plan(detector, region_plan.fold_bands)
        for detector, region_plan in zip(detectors, fit_region_plans, strict=True)
    )
    oracle_fold_plans = (
        tuple(
            compile_continuous_fold_correction_plan(detector, region_plan.fold_bands)
            for detector, region_plan in zip(detectors, oracle_region_plans, strict=True)
        )
        if stage == "joint"
        else fit_fold_plans
    )
    counts_by_dataset = _verified_osc_counts(manifest)
    native_observations = _uses_native_pixel_center_observations(recipe)
    if native_observations:
        observations, native_center_plans = _native_pixel_center_observations(
            arrays,
            manifest,
            counts_by_dataset,
        )
        fit_observations = observations
        fit_projection_revisions = tuple(
            plan.projection.projection_revision for plan in native_center_plans
        )
        oracle_projection_revisions = fit_projection_revisions
        oracle_projection_pixels = tuple(
            np.asarray(plan.projection.flat_pixel_index, dtype=np.int64)
            for plan in native_center_plans
        )
    else:
        fit_observations, fit_projection_revisions, _ = _continuous_matched_observations(
            arrays,
            manifest,
            fit_region_plans,
            counts_by_dataset,
        )
        (
            observations,
            oracle_projection_revisions,
            oracle_projection_pixels,
        ) = _continuous_matched_observations(
            arrays,
            manifest,
            oracle_region_plans,
            counts_by_dataset,
        )
    peak_area_projection = _prepared_peak_area_projection(arrays, manifest, observations)
    peak_aggregation = peak_area_projection.aggregation_matrix(observations)
    peak_area_projection.aggregation_matrix(fit_observations)
    signal_row = ~np.asarray(observations.is_background)
    signal_index = np.flatnonzero(signal_row)
    fit_peak_count = peak_aggregation @ np.asarray(fit_observations.count_mass)[signal_row]
    oracle_peak_count = peak_aggregation @ np.asarray(observations.count_mass)[signal_row]
    fit_peak_support = peak_aggregation @ np.asarray(fit_observations.support_px2)[signal_row]
    oracle_peak_support = peak_aggregation @ np.asarray(observations.support_px2)[signal_row]
    fit_peak_covariance = (
        peak_aggregation
        @ np.asarray(fit_observations.count_covariance_count2)[np.ix_(signal_index, signal_index)]
        @ peak_aggregation.T
    )
    oracle_peak_covariance = (
        peak_aggregation
        @ np.asarray(observations.count_covariance_count2)[np.ix_(signal_index, signal_index)]
        @ peak_aggregation.T
    )
    data_projection_relative_l2_by_family: dict[str, float] = {}
    data_projection_relative_l2_background_anchor_rows: float | None = None

    data_projection_qualification_required = _data_projection_qualification_required(recipe)
    maximum_data_projection_error = float(recipe["model_cubature"]["maximum_oracle_relative_l2"])
    if native_observations:
        data_projection_evidence = {
            "method": NATIVE_PIXEL_CENTER_METHOD,
            "qualification_role": "authoritative_frozen_observation",
            "status": "COMPLETE",
            "projection_performed": False,
            "refinement_oracle": "NOT_APPLICABLE_EXACT_MEMBERSHIP",
            "projection_revisions": list(oracle_projection_revisions),
            "selected_pixel_region_pair_count": int(arrays["selected_flat_pixel_index"].size),
            "selected_dataset_index_sha256": _array_sha256(arrays["selected_dataset_index"]),
            "selected_flat_pixel_index_sha256": _array_sha256(arrays["selected_flat_pixel_index"]),
            "selected_observation_row_sha256": _array_sha256(arrays["selected_observation_row"]),
            "raw_count_mass_sha256": _array_sha256(arrays["count_sum"]),
            "count_mass_sha256": _array_sha256(np.asarray(observations.count_mass)),
            "support_px2_sha256": _array_sha256(np.asarray(observations.support_px2)),
            "count_covariance_sha256": _array_sha256(
                np.asarray(observations.count_covariance_count2)
            ),
            "objective_measure": PEAK_AREA_OBJECTIVE,
            "smoothing_applied": False,
            "diffraction_model_pixelized": False,
        }
    else:
        for family in FAMILIES:
            data_projection_relative_l2_by_family[str(family)] = _relative_l2(
                oracle_peak_count,
                fit_peak_count,
                np.asarray(peak_area_projection.peak_signal_family) == family,
            )
        data_projection_relative_l2_background_anchor_rows = _relative_l2(
            np.asarray(observations.count_mass),
            np.asarray(fit_observations.count_mass),
            np.asarray(arrays["is_background"], dtype=np.bool_),
        )
        fine_row_data_projection_convergence = _native_projection_convergence(
            coarse_count_mass=np.asarray(fit_observations.count_mass),
            coarse_support_px2=np.asarray(fit_observations.support_px2),
            coarse_count_covariance=np.asarray(fit_observations.count_covariance_count2),
            refined_count_mass=np.asarray(observations.count_mass),
            refined_support_px2=np.asarray(observations.support_px2),
            refined_count_covariance=np.asarray(observations.count_covariance_count2),
            dataset_index=np.asarray(observations.dataset_index),
            dataset_ids=observations.dataset_ids,
            maximum_relative_l2=maximum_data_projection_error,
        )
        data_projection_convergence = _native_projection_convergence(
            coarse_count_mass=fit_peak_count,
            coarse_support_px2=fit_peak_support,
            coarse_count_covariance=fit_peak_covariance,
            refined_count_mass=oracle_peak_count,
            refined_support_px2=oracle_peak_support,
            refined_count_covariance=oracle_peak_covariance,
            dataset_index=np.asarray(peak_area_projection.peak_dataset_index),
            dataset_ids=observations.dataset_ids,
            maximum_relative_l2=maximum_data_projection_error,
        )
        covariance_refinement_converged = bool(data_projection_convergence["converged"])
        data_projection_convergence = {
            **data_projection_convergence,
            "converged": _integrated_area_projection_converged(data_projection_convergence),
            "acceptance_measure": "integrated_peak_count_mass_and_support.v1",
            "covariance_refinement_converged": covariance_refinement_converged,
            "covariance_policy": "refined_covariance_is_authoritative_for_objective_whitening",
        }
        if data_projection_qualification_required and not data_projection_convergence["converged"]:
            raise FloatingPointError(
                "native-pixel integrated peak-area projection did not converge; "
                f"refine cubature: {data_projection_convergence}"
            )
        data_projection_evidence = {
            "method": MEASURED_PROJECTION_METHOD,
            "qualification_role": "required_release_gate",
            **data_projection_convergence,
            "fit_projection_revisions": list(fit_projection_revisions),
            "oracle_projection_revisions": list(oracle_projection_revisions),
            "fit_gauss_order": fit_order,
            "oracle_gauss_order": oracle_order,
            "fit_subdivision_count": fit_subdivisions,
            "oracle_subdivision_count": oracle_subdivisions,
            "relative_l2_by_family_m": data_projection_relative_l2_by_family,
            "relative_l2_background_anchor_rows": (
                data_projection_relative_l2_background_anchor_rows
            ),
            "maximum_relative_l2": maximum_data_projection_error,
            "objective_measure": PEAK_AREA_OBJECTIVE,
            "fine_row_refinement": fine_row_data_projection_convergence,
            "smoothing_applied": False,
            "diffraction_model_pixelized": False,
        }
    background_state, background_identity, background_manifest = _load_radial_background(
        background_path,
        diagnostic_identity=diagnostic_identity,
        dataset_ids=manifest["dataset_ids"],
        beam_center_column_row_px=tuple(
            float(value) for value in manifest["fixed_position"]["beam_center_column_row_px"]
        ),
        fit_plan_identity=fit_plan_identity,
        observation_support_revisions=oracle_projection_revisions,
        observation_support_method=(
            NATIVE_PIXEL_CENTER_METHOD if native_observations else MEASURED_PROJECTION_METHOD
        ),
        excluded_flat_pixel_index_by_dataset={
            dataset_id: _background_exclusion_pixel_index(
                arrays["selected_flat_pixel_index"][
                    arrays["selected_dataset_index"] == dataset_index
                ],
                oracle_projection_pixels[dataset_index],
            )
            for dataset_index, dataset_id in enumerate(manifest["dataset_ids"])
        },
    )
    radial_background = (
        _fixed_background_from_native_pixel_center_plans(
            state=background_state,
            plans=native_center_plans,
            observation_count=row_count,
            beam_center_column_row_px=tuple(
                manifest["fixed_position"]["beam_center_column_row_px"]
            ),
        )
        if native_observations
        else _fixed_background_from_continuous_plans(
            state=background_state,
            plans=oracle_region_plans,
            observation_count=row_count,
            beam_center_column_row_px=tuple(
                manifest["fixed_position"]["beam_center_column_row_px"]
            ),
        )
    )
    fixed_background = condition_matched_region_background_from_anchors(
        observations,
        radial_background,
    )
    execution_records: list[dict[str, Any]] = []
    cache_parameters: np.ndarray | None = None
    cache_model: np.ndarray | None = None
    expected_devices: tuple[str, ...] | None = None
    expected_evaluated_backends: tuple[str, ...] | None = None
    requested_maximum_evaluations = int(maximum_function_evaluations)
    if requested_maximum_evaluations <= 0:
        raise ValueError("maximum function evaluations must be positive")
    initial_full = np.asarray(fit_plan["baseline_parameters"], dtype=np.float64)
    predecessor_identity: dict[str, str] | None = None
    predecessor_chain_identities: list[dict[str, str]] = []
    if predecessor_path is not None:
        predecessor_identity = _file_identity(predecessor_path)
        predecessor_chain = _load_qualified_stage_chain(
            predecessor_path,
            expected_stage=str(expected_predecessor_stage),
            trusted_recipe=recipe,
            fit_plan=fit_plan,
            diagnostic_sha256=diagnostic_identity["sha256"],
            recipe_sha256=recipe_identity["sha256"],
            fit_plan_sha256=fit_plan_identity["sha256"],
            adapter_sha256=adapter_identity["sha256"],
            implementation_sha256=implementation_identity["sha256"],
        )
        predecessor_document = predecessor_chain[0][0]
        if any(
            document.get("provenance", {}).get("background") != background_identity
            for document, _ in predecessor_chain
        ):
            raise ValueError("structure predecessor chain used a different radial background")
        predecessor_chain_identities = [identity for _, identity in predecessor_chain]
        initial_full = _fit_structure_vector(predecessor_document)
    initial_parameters = initial_full[active_index]
    execution_identity = {
        "diagnostic_sha256": diagnostic_identity["sha256"],
        "background_sha256": background_identity["sha256"],
        "recipe_sha256": recipe_identity["sha256"],
        "fit_plan_sha256": fit_plan_identity["sha256"],
        "fit_adapter_sha256": adapter_identity["sha256"],
        "implementation_sha256": implementation_identity["sha256"],
        "stage": stage,
        "active_parameter_names": list(active_parameter_names),
        "frozen_parameter_names": list(frozen_parameter_names),
        "predecessor_sha256": (
            None if predecessor_identity is None else predecessor_identity["sha256"]
        ),
        "predecessor_chain_sha256": [
            identity["sha256"] for identity in predecessor_chain_identities
        ],
        "execution_backend": execution_backend,
        "model_measure": "continuous_detector_chart_area",
        "background_artifact_sha256": background_identity["sha256"],
        "radial_background_state_revision": background_state.revision,
        "radial_background_parameter_vector_sha256": _array_sha256(
            background_state.parameter_vector
        ),
        "radial_background_parameter_covariance_sha256": _array_sha256(
            background_state.parameter_covariance
        ),
        "radial_background_mass_sha256": _array_sha256(radial_background.count_mass),
        "radial_background_covariance_sha256": _array_sha256(radial_background.covariance_count2),
        "background_excluded_flat_pixel_count_by_dataset": background_manifest["sampling"][
            "excluded_flat_pixel_count_by_dataset"
        ],
        "background_excluded_flat_pixel_sha256_by_dataset": background_manifest["sampling"][
            "excluded_flat_pixel_sha256_by_dataset"
        ],
        "conditioned_background_revision": fixed_background.revision,
        "conditioned_background_mass_sha256": _array_sha256(fixed_background.count_mass),
        "conditioned_background_covariance_sha256": _array_sha256(
            fixed_background.covariance_count2
        ),
        "conditioned_model_anchor_projection_sha256": _array_sha256(
            fixed_background.anchor_projection
        ),
        "observed_count_mass_sha256": _array_sha256(observations.count_mass),
        "observed_support_px2_sha256": _array_sha256(observations.support_px2),
        "observed_count_covariance_sha256": _array_sha256(observations.count_covariance_count2),
        "measured_observation_method": data_projection_evidence["method"],
        "measured_projection_performed": data_projection_evidence.get(
            "projection_performed",
            True,
        ),
        "measured_projection_revisions": data_projection_evidence.get(
            "projection_revisions",
            [],
        ),
        "selected_pixel_region_pair_count": data_projection_evidence.get(
            "selected_pixel_region_pair_count",
            0,
        ),
        "selected_dataset_index_sha256": data_projection_evidence.get(
            "selected_dataset_index_sha256"
        ),
        "selected_flat_pixel_index_sha256": data_projection_evidence.get(
            "selected_flat_pixel_index_sha256"
        ),
        "selected_observation_row_sha256": data_projection_evidence.get(
            "selected_observation_row_sha256"
        ),
        "objective_measure": PEAK_AREA_OBJECTIVE,
        "peak_area_projection_revision": peak_area_projection.revision,
        "source_signal_peak_index_sha256": _array_sha256(
            peak_area_projection.source_signal_peak_index
        ),
        "fit_native_pixel_projection_revision": list(fit_projection_revisions),
        "oracle_native_pixel_projection_revision": list(oracle_projection_revisions),
        "native_pixel_projection_relative_l2_by_family_m": (data_projection_relative_l2_by_family),
        "native_pixel_projection_relative_l2_background_anchor_rows": (
            data_projection_relative_l2_background_anchor_rows
        ),
        "fit_gauss_order": fit_order,
        "fit_subdivision_count": fit_subdivisions,
        "offspecular_axial_refinement": offspecular_axial_refinement,
        "offspecular_radial_transform": offspecular_radial_transform,
        "offspecular_signal_minimum_radial_nodes_per_side": (
            offspecular_signal_minimum_radial_nodes_per_side
        ),
        "fit_quadrature_revision": [
            plan.quadrature.quadrature_revision for plan in fit_region_plans
        ],
        "fit_continuous_node_count": [
            int(plan.quadrature.column_px.size) for plan in fit_region_plans
        ],
        "fit_continuous_fold_plan_sha256": [
            None if plan is None else plan.plan_sha256 for plan in fit_fold_plans
        ],
        "fit_continuous_fold_node_count": [
            0 if plan is None else int(np.asarray(plan.column_px).size) for plan in fit_fold_plans
        ],
        "oracle_gauss_order": oracle_order,
        "oracle_subdivision_count": oracle_subdivisions,
        "oracle_quadrature_revision": [
            plan.quadrature.quadrature_revision for plan in oracle_region_plans
        ],
        "oracle_continuous_fold_plan_sha256": [
            None if plan is None else plan.plan_sha256 for plan in oracle_fold_plans
        ],
        "cuda_coordinate_chunk_size": cuda_chunk_size,
        "maximum_state_block_count": maximum_blocks,
        "rod_scope": "families_m_0_1_3_4",
        "rod_count": len(fit_rod_roster),
        "rod_roster_sha256": fit_rod_roster_sha256,
        "model_input_sha256": {
            name: identity["sha256"] for name, identity in model_input_identities.items()
        },
    }
    fit_input_identities = {
        "fit_adapter": adapter_identity,
        "prepared_diagnostic": diagnostic_identity,
        "background": background_identity,
        "recipe": recipe_identity,
        "fit_plan": fit_plan_identity,
        **model_input_identities,
    }
    if predecessor_identity is not None:
        fit_input_identities["predecessor"] = predecessor_identity
        for index, identity in enumerate(predecessor_chain_identities):
            fit_input_identities[f"predecessor_chain_{index}"] = identity

    def verify_fit_inputs() -> None:
        for role, identity in fit_input_identities.items():
            _require_unchanged(identity, role=role.replace("_", " "))
        if _implementation_identity() != implementation_identity:
            raise RuntimeError("scientific implementation changed during the calculation")

    start_kind = "default"
    start_source: dict[str, str] | None = None
    if initial_parameters_override is not None:
        initial_parameters = np.asarray(initial_parameters_override, dtype=np.float64)
        start_kind = "explicit"
    progress_path = destination.with_name(f"{destination.stem}.progress.json")
    if resume_path is not None:
        resolved_resume = resume_path.resolve()
        resume_identity = _file_identity(resolved_resume)
        resume_document = json.loads(resolved_resume.read_text(encoding="utf-8"))
        _require_unchanged(resume_identity, role="fit restart artifact")
        start_source = resume_identity
        if resume_document.get("schema_version") == FIT_PROGRESS_SCHEMA:
            if resume_document.get("execution_identity") != execution_identity:
                raise ValueError("fit progress execution identity changed")
            initial_parameters = np.asarray(resume_document["active_parameters"], dtype=np.float64)
            expected_devices = tuple(str(value) for value in resume_document["devices"])
            expected_evaluated_backends = tuple(
                str(value) for value in resume_document["evaluated_backends"]
            )
            progress_path = destination.with_name(
                f"{destination.stem}.resume-{resume_identity['sha256'][:12]}.progress.json"
            )
            start_kind = "progress_restart"
            fit_input_identities["fit_restart_artifact"] = resume_identity
        else:
            raise ValueError("fit resume requires one same-stage progress artifact")
    if initial_parameters.shape != (active_index.size,):
        raise ValueError("fit start must contain one value per active stage parameter")
    if np.any(initial_parameters < lower_full[active_index]) or np.any(
        initial_parameters > upper_full[active_index]
    ):
        raise ValueError("fit start lies outside the declared stage bounds")
    if progress_path.exists():
        raise FileExistsError(f"fit progress destination already exists: {progress_path}")
    start_full_parameters = np.array(initial_full, copy=True)
    start_full_parameters[active_index] = initial_parameters
    fit_start = {
        "kind": start_kind,
        "initial_parameters": initial_parameters.tolist(),
        "initial_full_parameters": start_full_parameters.tolist(),
        "initial_parameters_sha256": _array_sha256(initial_parameters),
        "maximum_function_evaluations": requested_maximum_evaluations,
        "source_artifact": start_source,
        "semantics": (
            "restart from a completely evaluated parameter vector; optimizer state is not continued"
            if start_source is not None
            else "new optimizer run"
        ),
    }
    fit_run_identity = {
        "model_execution": execution_identity,
        "fit_start": fit_start,
    }

    def expanded_parameters(parameters: np.ndarray) -> np.ndarray:
        active = np.asarray(parameters, dtype=np.float64)
        if active.shape != (active_index.size,):
            raise ValueError("candidate parameters do not match the active structure stage")
        full = np.array(initial_full, copy=True)
        full[active_index] = active
        return full

    def evaluate(
        parameters: np.ndarray,
        *,
        region_plans: Sequence[_DatasetContinuousRegionPlan],
        fold_plans: Sequence[ContinuousFoldCorrectionPlan | None],
        use_cache: bool,
    ) -> np.ndarray:
        nonlocal cache_parameters, cache_model, expected_devices, expected_evaluated_backends
        if (
            use_cache
            and cache_parameters is not None
            and np.array_equal(parameters, cache_parameters)
            and cache_model is not None
        ):
            return cache_model
        model_mass = np.zeros(row_count, dtype=np.float64)
        full_parameters = expanded_parameters(parameters)
        for dataset_index, (detector, region_plan, fold_plan) in enumerate(
            zip(detectors, region_plans, fold_plans, strict=True)
        ):
            rebound = detector.rebind_physics(
                strength_model=_candidate_strength(
                    detector.strength_model,
                    full_parameters,
                    fit_plan=fit_plan,
                ),
                intensity_envelope=_candidate_intensity_envelope(full_parameters),
            )
            dataset_mass, execution = _integrate_continuous_region_mass(
                rebound,
                region_plan,
                fold_plan,
                row_count=row_count,
                execution_backend=execution_backend,
                cuda_coordinate_chunk_size=cuda_chunk_size,
            )
            model_mass += dataset_mass
            execution["dataset_id"] = manifest["dataset_ids"][dataset_index]
            execution["evaluation_index"] = len(execution_records)
            execution_records.append(execution)
        if use_cache:
            current_records = execution_records[-len(detectors) :]
            current_devices = tuple(
                sorted({value for record in current_records for value in record["devices"]})
            )
            current_backends = tuple(
                sorted(
                    {value for record in current_records for value in record["evaluated_backends"]}
                )
            )
            if expected_devices is not None and current_devices != expected_devices:
                raise RuntimeError("fit resume execution device changed")
            if (
                expected_evaluated_backends is not None
                and current_backends != expected_evaluated_backends
            ):
                raise RuntimeError("fit resume evaluated backend changed")
            expected_devices = current_devices
            expected_evaluated_backends = current_backends
            cache_parameters = np.array(parameters, copy=True)
            cache_model = np.array(model_mass, copy=True)
            progress_document = {
                "schema_version": FIT_PROGRESS_SCHEMA,
                "execution_identity": execution_identity,
                "fit_run_identity": fit_run_identity,
                "active_parameters": np.asarray(parameters, dtype=np.float64).tolist(),
                "full_parameters": full_parameters.tolist(),
                "completed_model_evaluations": len(execution_records) // len(detectors),
                "devices": list(current_devices),
                "evaluated_backends": list(current_backends),
            }
            verify_fit_inputs()
            _write_json_atomic(progress_path, progress_document)
        return model_mass

    result = fit_matched_regions(
        observations,
        lambda parameters: evaluate(
            parameters,
            region_plans=fit_region_plans,
            fold_plans=fit_fold_plans,
            use_cache=True,
        ),
        parameter_names=active_parameter_names,
        initial_parameters=(initial_parameters,),
        lower_bounds=lower_full[active_index],
        upper_bounds=upper_full[active_index],
        parameter_scales=scale_full[active_index],
        prior_residual=lambda parameters: (
            (parameters - prior_mean_full[active_index]) / prior_sigma_full[active_index]
        ),
        sensitivity_relative_tolerance=sensitivity_relative_tolerance,
        maximum_function_evaluations=requested_maximum_evaluations,
        fixed_background=fixed_background,
        peak_area_projection=peak_area_projection,
    )
    fitted_full = expanded_parameters(result.parameters)
    fit_model_unscaled = evaluate(
        result.parameters,
        region_plans=fit_region_plans,
        fold_plans=fit_fold_plans,
        use_cache=True,
    )
    if stage == "joint":
        oracle_model_unscaled = evaluate(
            result.parameters,
            region_plans=oracle_region_plans,
            fold_plans=oracle_fold_plans,
            use_cache=False,
        )
        (
            oracle_relative_l2,
            cubature_relative_l2_by_family,
            relative_l2_background_anchor_rows,
        ) = _conditioned_model_cubature_errors(
            fit_model_unscaled,
            oracle_model_unscaled,
            fixed_background=fixed_background,
            signal_family_m=arrays["signal_family_m"],
            is_background=arrays["is_background"],
            peak_aggregation=peak_aggregation,
            peak_family_m=peak_area_projection.peak_signal_family,
        )
    else:
        oracle_model_unscaled = None
        oracle_relative_l2 = None
        cubature_relative_l2_by_family = {}
        relative_l2_background_anchor_rows = None
    maximum_oracle_error = float(recipe["model_cubature"]["maximum_oracle_relative_l2"])
    baseline_structure = series[0].strength.structure_parameters
    fitted_displacement_profile = _site_displacement_profile(series[0].strength, fit_plan)
    structure = {
        "bi_fractional_z": baseline_structure.bi_fractional_z + float(fitted_full[0]),
        "outer_chalcogen_fractional_z": baseline_structure.se2_fractional_z + float(fitted_full[1]),
        "bi_delta_z_fractional": float(fitted_full[0]),
        "outer_chalcogen_delta_z_fractional": float(fitted_full[1]),
        "bi_occupancy": 1.0,
        "central_chalcogen_occupancy": 1.0,
        "outer_chalcogen_occupancy": 1.0 - float(fitted_full[2]),
        "outer_chalcogen_vacancy_fraction": float(fitted_full[2]),
        "outer_bi_antisite_fraction": 0.0,
        "outer_chalcogen_fraction": 1.0 - float(fitted_full[2]),
        "occupancy_rule": "outer_site_chalcogen_plus_vacancy.v1",
        "intensity_envelope_u_radial_A2": float(fitted_full[3]),
        "intensity_envelope_u_normal_A2": float(fitted_full[4]),
        "intensity_envelope_model": "exp(-U_r*Q_r^2-U_z*Q_z^2).v1",
        "site_adp_scale": 1.0,
        "site_adp_refinement_status": "fixed_literature_reference",
        "displacement_gauge": fit_plan["displacement_gauge"],
        "site_displacement_profile": {
            "model_id": fit_plan["site_displacement_profile"]["model_id"],
            "provenance": fitted_displacement_profile.provenance,
            "sites": [
                {
                    "source_label": site.source_label,
                    "reference_u_radial_A2": site.u_radial_A2,
                    "reference_u_normal_A2": site.u_normal_A2,
                    "fitted_u_radial_A2": fitted_displacement_profile.components_A2(
                        site.source_label
                    )[0],
                    "fitted_u_normal_A2": fitted_displacement_profile.components_A2(
                        site.source_label
                    )[1],
                }
                for site in fitted_displacement_profile.sites
            ],
        },
    }
    family_residual = {}
    for family in FAMILIES:
        selected = result.objective_signal_family == family
        family_residual[str(family)] = float(
            np.sqrt(np.mean(result.weighted_residual[selected] ** 2))
        )
    peak_aggregation = peak_area_projection.aggregation_matrix(observations)
    signal_row = ~np.asarray(observations.is_background)
    observed_peak_count = peak_aggregation @ (
        np.asarray(observations.count_mass)[signal_row]
        - np.asarray(fixed_background.count_mass)[signal_row]
    )
    fitted_peak_count = (
        peak_aggregation @ np.asarray(result.fitted_objective_model_mass)[signal_row]
    )
    parameters_on_bounds = [
        name
        for name, value, minimum, maximum, scale in zip(
            active_parameter_names,
            result.parameters,
            lower_full[active_index],
            upper_full[active_index],
            scale_full[active_index],
            strict=True,
        )
        if min((value - minimum) / scale, (maximum - value) / scale) <= bound_proximity
    ]
    cubature_converged = (
        all(value <= maximum_oracle_error for value in cubature_relative_l2_by_family.values())
        and relative_l2_background_anchor_rows is not None
        and math.isfinite(relative_l2_background_anchor_rows)
        and relative_l2_background_anchor_rows >= 0.0
        if stage == "joint"
        else None
    )
    numerically_converged = (
        result.success
        and result.sensitivity_rank == len(active_parameter_names)
        and math.isfinite(result.sensitivity_condition)
        and result.sensitivity_condition <= maximum_sensitivity_condition
        and (stage != "joint" or cubature_converged is True)
        and not parameters_on_bounds
    )
    document = {
        "schema_version": FIT_SCHEMA,
        "stage": stage,
        "execution_policy": execution_policy,
        "status": (
            ("FIT" if stage == "joint" else "STAGE_CONDITIONED")
            if numerically_converged
            else "MODEL_LIMITED_FIT"
        ),
        "active_parameter_names": list(active_parameter_names),
        "frozen_parameter_names": list(frozen_parameter_names),
        "full_parameter_vector": fitted_full.tolist(),
        "structure_representative": structure,
        "fixed_lattice": fixed_lattice_record,
        "stacking_model": manifest["stacking_model"],
        "specular_stitch": recipe.get("parratt_stitch"),
        "model_rod_scope": "fitted_families_m_0_1_3_4",
        "model_rod_count": len(fit_rod_roster),
        "model_rod_roster_h_k_m_population": fit_rod_roster,
        "model_rod_roster_sha256": fit_rod_roster_sha256,
        "dataset_scales": {
            dataset_id: float(result.dataset_scales[index])
            for index, dataset_id in enumerate(manifest["dataset_ids"])
        },
        "objective_half_chi_squared": result.objective_half_chi_squared,
        "data_objective_half_chi_squared": result.data_objective_half_chi_squared,
        "prior_objective_half_chi_squared": result.prior_objective_half_chi_squared,
        "prior_weighted_residual": result.prior_weighted_residual.tolist(),
        "weighted_residual_rms": float(
            np.sqrt(np.mean(result.weighted_residual[result.fitted_signal_row] ** 2))
        ),
        "weighted_residual_rms_by_family_m": family_residual,
        "objective_measure": PEAK_AREA_OBJECTIVE,
        "integrated_peak_areas": {
            "projection_revision": peak_area_projection.revision,
            "peak_ids": list(result.objective_ids),
            "dataset_index": result.objective_dataset_index.tolist(),
            "family_m": result.objective_signal_family.tolist(),
            "observed_background_subtracted_count_mass": observed_peak_count.tolist(),
            "fitted_count_mass": fitted_peak_count.tolist(),
            "source_signal_peak_index_sha256": _array_sha256(
                peak_area_projection.source_signal_peak_index
            ),
            "conditioning_order": (
                "dark subtraction, radial baseline plus adjacent-anchor conditioning, "
                "trusted-peak area sum, full-covariance whitening"
            ),
        },
        "sensitivity": {
            "rank": result.sensitivity_rank,
            "numerical_rank": result.sensitivity_numerical_rank,
            "parameter_count": len(active_parameter_names),
            "condition": result.sensitivity_condition,
            "maximum_condition": maximum_sensitivity_condition,
            "relative_tolerance": result.sensitivity_relative_tolerance,
            "parameter_names": list(active_parameter_names),
            "parameter_scales": scale_full[active_index].tolist(),
            "singular_values": result.jacobian_singular_values.tolist(),
            "parameter_correlation": result.parameter_correlation.tolist(),
            "data_only": True,
            "parameter_scaled": True,
            "residual_weighting": (
                "trusted integrated peak count masses; full projected regularized count covariance "
                "plus a frozen shared radial baseline, both transformed by the fixed adjacent-sideband "
                "conditioning in every fitted block before peak-area summation"
            ),
        },
        "penalized_sensitivity": {
            "rank": result.penalized_sensitivity_rank,
            "numerical_rank": result.penalized_sensitivity_numerical_rank,
            "parameter_count": len(active_parameter_names),
            "condition": result.penalized_sensitivity_condition,
            "singular_values": result.penalized_jacobian_singular_values.tolist(),
        },
        "regularization": {
            "interpretation": fit_plan["prior_interpretation"],
            "parameter_names": list(STRUCTURE_PARAMETER_NAMES),
            "mean": prior_mean_full.tolist(),
            "sigma": prior_sigma_full.tolist(),
            "parameter_scales": scale_full.tolist(),
            "lower_bounds": lower_full.tolist(),
            "upper_bounds": upper_full.tolist(),
            "bound_proximity_in_parameter_scales": bound_proximity,
        },
        "parameters_on_bounds": parameters_on_bounds,
        "numerical_convergence": {
            "converged": numerically_converged,
            "optimizer_converged": result.success,
            "identifiable": result.sensitivity_rank == len(active_parameter_names),
            "cubature_converged_by_family": (cubature_converged if stage == "joint" else "NOT_RUN"),
        },
        "data_projection": data_projection_evidence,
        "model_adequacy": "reported_by_residuals_not_inferred_from_optimizer_status",
        "fitted_model_count": result.fitted_model_mass.tolist(),
        "fitted_objective_model_count": result.fitted_objective_model_mass.tolist(),
        "fitted_background_count": result.fitted_background_mass.tolist(),
        "weighted_residual": result.weighted_residual.tolist(),
        "cubature_oracle": {
            "performed": stage == "joint",
            "status": "COMPLETE" if stage == "joint" else "NOT_RUN_INITIALIZER",
            "fit_gauss_order": fit_order,
            "oracle_gauss_order": oracle_order,
            "fit_subdivision_count": fit_subdivisions,
            "oracle_subdivision_count": oracle_subdivisions,
            "offspecular_axial_refinement": offspecular_axial_refinement,
            "offspecular_radial_transform": offspecular_radial_transform,
            "offspecular_signal_minimum_radial_nodes_per_side": (
                offspecular_signal_minimum_radial_nodes_per_side
            ),
            "relative_l2": oracle_relative_l2,
            "relative_l2_by_family_m": cubature_relative_l2_by_family,
            "relative_l2_background_anchor_rows": relative_l2_background_anchor_rows,
            "maximum_relative_l2": maximum_oracle_error,
        },
        "optimizer": {
            "success": result.success,
            "message": result.optimizer_message,
            "function_evaluations": result.function_evaluations,
            "requested_maximum_function_evaluations": requested_maximum_evaluations,
            "fit_start": fit_start,
        },
        "execution": execution_records,
        "progress_artifact": str(progress_path),
        "diagnostic": diagnostic_identity["path"],
        "diagnostic_sha256": diagnostic_identity["sha256"],
        "provenance": {
            "diagnostic": diagnostic_identity,
            "background": background_identity,
            "recipe": recipe_identity,
            "fit_plan": fit_plan_identity,
            "predecessor": predecessor_identity,
            "predecessor_chain": predecessor_chain_identities,
            "fit_adapter": adapter_identity,
            "implementation": implementation_identity,
            "execution_identity": execution_identity,
            "fit_run_identity": fit_run_identity,
        },
        "fit_contract": (
            "one five-coordinate structure shared by every retained m=0 and m!=0 peak in all "
            "three OSCs; "
            + (
                "joint activation from one explicit or same-stage restart vector under the "
                "seeded joint-only policy; "
                if execution_policy == "seeded_joint_only.v1"
                else "staged Wyckoff, constrained outer-site vacancy, one physical "
                "two-component global intensity envelope with fixed anisotropic site ADPs, "
                "then joint activation; "
            )
            + "one scale per OSC shared across its families; exact candidate-model reevaluation "
            "as continuous detector-chart area integrals in phi/2theta for m=0 and signed-side "
            "Qr/L for m!=0; "
            + (
                "measured count masses and covariance are exact reductions over frozen "
                "native-pixel-center memberships with no smoothing;"
                if native_observations
                else "verified native-pixel counts are a piecewise-constant measured field "
                "projected through the same rectangles with no smoothing;"
            )
            + " one frozen shared "
            "radial detector baseline calibrated only from "
            "background sectors and conditioned on each block's two adjacent sidebands; the "
            "same fixed adjacent-sideband operator is applied to candidate diffraction before anchor rows "
            "are excluded, with their uncertainty propagated; conditioned fine bins are summed into "
            "declared trusted peak areas before full-covariance whitening"
        ),
        "background_model": {
            "status": background_manifest["status"],
            "artifact_sha256": background_identity["sha256"],
            "state_revision": background_state.revision,
            "parameter_vector_sha256": _array_sha256(background_state.parameter_vector),
            "parameter_covariance_sha256": _array_sha256(background_state.parameter_covariance),
            "radial_mass_sha256": _array_sha256(radial_background.count_mass),
            "radial_covariance_sha256": _array_sha256(radial_background.covariance_count2),
            "excluded_flat_pixel_count_by_dataset": background_manifest["sampling"][
                "excluded_flat_pixel_count_by_dataset"
            ],
            "excluded_flat_pixel_sha256_by_dataset": background_manifest["sampling"][
                "excluded_flat_pixel_sha256_by_dataset"
            ],
            "conditioned_revision": fixed_background.revision,
            "conditioned_mass_sha256": _array_sha256(fixed_background.count_mass),
            "conditioned_covariance_sha256": _array_sha256(fixed_background.covariance_count2),
            "anchor_projection_sha256": _array_sha256(fixed_background.anchor_projection),
            "residual_conditioning": "adjacent-affine-two-anchor-model-projected.v2",
            "parameter_names": background_manifest["parameter_names"],
            "parameter_vector": background_state.parameter_vector.tolist(),
            "metrics": background_manifest["metrics"],
        },
        "model_measure": "continuous_detector_chart_area",
        "smoothing_applied": False,
        "model_pixelized": False,
    }
    verify_fit_inputs()
    _write_json_atomic(destination, document)
    return document


def _project_full_profiles(
    row_records: list[dict[str, Any]],
    *,
    count_sum: np.ndarray,
    count_support: np.ndarray,
    count_coordinate_mass: np.ndarray,
    count_axial_mass: np.ndarray,
    count_qz_mass: np.ndarray,
    model: dict[str, np.ndarray],
    dataset_scale: float,
    m0: SpecularAngularProfileRegion,
) -> dict[str, np.ndarray]:
    output: dict[str, list[Any]] = {
        "identity": [],
        "family_m": [],
        "bin_index": [],
        "display_L": [],
        "display_qz_Ainv": [],
        "selection_coordinate": [],
        "selection_coordinate_kind": [],
        "measured_signal_density": [],
        "model_signal_density": [],
        "background_density": [],
        "valid": [],
    }

    for signal_row, record in enumerate(row_records):
        band = str(record["band"])
        if band.startswith("background_"):
            continue
        bin_index = int(record["bin"])
        data_support = count_support[signal_row]
        model_support = model["support_px2"][signal_row]
        background_mass = model["background_mass"][signal_row]
        measured_valid = bool(
            data_support > 0.0
            and np.isfinite(count_sum[signal_row])
            and np.isfinite(background_mass)
        )
        model_valid = bool(model_support > 0.0 and np.isfinite(model["model_mass"][signal_row]))
        measured = (
            (count_sum[signal_row] - background_mass) / data_support
            if measured_valid
            else float("nan")
        )
        modeled = (
            dataset_scale * model["model_mass"][signal_row] / model_support
            if model_valid
            else float("nan")
        )
        background_density = background_mass / data_support if measured_valid else float("nan")
        display_L = (
            count_axial_mass[signal_row] / data_support if data_support > 0.0 else float("nan")
        )
        display_qz_Ainv = (
            count_qz_mass[signal_row] / data_support if data_support > 0.0 else float("nan")
        )
        family = int(record["signal_family"])
        if family == 0:
            edges = np.asarray(m0.two_theta_bin_edges_rad)
            selection_coordinate = math.degrees(0.5 * (edges[bin_index] + edges[bin_index + 1]))
            coordinate_kind = "two_theta_deg"
            identity = "m0"
        else:
            selection_coordinate = display_L
            coordinate_kind = "L"
            identity = band
        output["identity"].append(identity)
        output["family_m"].append(family)
        output["bin_index"].append(bin_index)
        output["display_L"].append(display_L)
        output["display_qz_Ainv"].append(display_qz_Ainv)
        output["selection_coordinate"].append(selection_coordinate)
        output["selection_coordinate_kind"].append(coordinate_kind)
        output["measured_signal_density"].append(measured)
        output["model_signal_density"].append(modeled)
        output["background_density"].append(background_density)
        output["valid"].append(measured_valid and model_valid)
    return {
        "identity": np.asarray(output["identity"]),
        "family_m": np.asarray(output["family_m"], dtype=np.int64),
        "bin_index": np.asarray(output["bin_index"], dtype=np.int64),
        "display_L": np.asarray(output["display_L"], dtype=np.float64),
        "display_qz_Ainv": np.asarray(output["display_qz_Ainv"], dtype=np.float64),
        "selection_coordinate": np.asarray(output["selection_coordinate"], dtype=np.float64),
        "selection_coordinate_kind": np.asarray(output["selection_coordinate_kind"]),
        "measured_signal_density": np.asarray(output["measured_signal_density"], dtype=np.float64),
        "model_signal_density": np.asarray(output["model_signal_density"], dtype=np.float64),
        "background_density": np.asarray(output["background_density"], dtype=np.float64),
        "valid": np.asarray(output["valid"], dtype=np.bool_),
    }


def _m0_signal_only_display_mask(
    signal_only_bin_index: np.ndarray,
    signal_only_valid: np.ndarray,
    profile_identity: np.ndarray,
    profile_bin_index: np.ndarray,
    profile_valid: np.ndarray,
) -> np.ndarray:
    """Select signal-only m=0 bins only where conditioned comparison is unavailable."""

    signal_bins = np.asarray(signal_only_bin_index, dtype=np.int64)
    signal_valid = np.asarray(signal_only_valid, dtype=np.bool_)
    identities = np.asarray(profile_identity)
    profile_bins = np.asarray(profile_bin_index, dtype=np.int64)
    conditioned_valid = np.asarray(profile_valid, dtype=np.bool_)
    if signal_bins.ndim != 1 or signal_valid.shape != signal_bins.shape:
        raise ValueError("m=0 signal-only arrays must be aligned one-dimensional vectors")
    if not (
        identities.ndim == 1
        and profile_bins.shape == identities.shape
        and conditioned_valid.shape == identities.shape
    ):
        raise ValueError("conditioned profile arrays must be aligned one-dimensional vectors")
    conditioned_m0_bins = profile_bins[(identities == "m0") & conditioned_valid]
    return signal_valid & ~np.isin(signal_bins, conditioned_m0_bins)


def prepare_profiles(
    *,
    diagnostic_path: Path,
    background_path: Path,
    fit_path: Path,
    destination: Path,
    execution_backend: str,
    row_chunk_size: int,
    specular_interface_assumption: str | None = None,
) -> Path:
    destination = _external_file(destination)
    if destination.exists():
        raise FileExistsError(destination)
    adapter_identity = _file_identity(Path(__file__))
    implementation_identity = _implementation_identity()
    diagnostic_identity = _file_identity(diagnostic_path)
    fit_identity = _file_identity(fit_path)
    fit_arrays, manifest = _load_prepared(diagnostic_path)
    fit_document = json.loads(fit_path.resolve().read_text(encoding="utf-8"))
    if fit_document.get("schema_version") != FIT_SCHEMA:
        raise ValueError("unsupported matched-region fit artifact")
    if fit_document.get("diagnostic_sha256") != diagnostic_identity["sha256"]:
        raise ValueError("fit artifact does not belong to the supplied diagnostic")
    recipe_path = _verified_provenance_path(manifest, "recipe")
    recipe_identity = _file_identity(recipe_path)
    recipe = _load_recipe(recipe_path)
    profile_recipe = _profile_recipe_with_specular_interface(
        recipe,
        specular_interface_assumption,
    )
    parameter_replay_requested = specular_interface_assumption is not None
    fit_provenance = fit_document.get("provenance", {})
    fit_plan_identity = _verified_recorded_file_identity(
        fit_document.get("provenance", {}).get("fit_plan"),
        role="structure fit plan",
    )
    fit_plan = _load_fit_plan(Path(fit_plan_identity["path"]))
    series = _rebuilt_series(manifest)
    native_observations = _uses_native_pixel_center_observations(recipe)
    data_projection_record = fit_document.get("data_projection", {})
    observation_support_revisions = data_projection_record.get(
        ("projection_revisions" if native_observations else "oracle_projection_revisions"),
        (),
    )
    background_state, background_identity, background_manifest = _load_radial_background(
        background_path,
        diagnostic_identity=diagnostic_identity,
        dataset_ids=manifest["dataset_ids"],
        beam_center_column_row_px=tuple(
            float(value) for value in manifest["fixed_position"]["beam_center_column_row_px"]
        ),
        fit_plan_identity=fit_plan_identity,
        observation_support_revisions=observation_support_revisions,
        observation_support_method=(
            NATIVE_PIXEL_CENTER_METHOD if native_observations else MEASURED_PROJECTION_METHOD
        ),
        expected_background_adapter_identity=(
            fit_provenance.get("fit_adapter") if parameter_replay_requested else None
        ),
        expected_implementation_identity=(
            fit_provenance.get("implementation") if parameter_replay_requested else None
        ),
    )
    if fit_document.get("provenance", {}).get("background") != background_identity:
        raise ValueError("fit artifact used a different radial background")
    continuous_quadrature = fit_plan["continuous_quadrature"]
    offspecular_axial_refinement = int(continuous_quadrature["offspecular_axial_refinement"])
    offspecular_radial_transform = str(continuous_quadrature["offspecular_radial_transform"])
    offspecular_signal_minimum_radial_nodes_per_side = int(
        continuous_quadrature["offspecular_signal_minimum_radial_nodes_per_side"]
    )
    lower_full = np.asarray(fit_plan["lower_bounds"], dtype=np.float64)
    upper_full = np.asarray(fit_plan["upper_bounds"], dtype=np.float64)
    scale_full = np.asarray(fit_plan["parameter_scales"], dtype=np.float64)
    sensitivity_relative_tolerance = float(fit_plan["sensitivity_relative_tolerance"])
    bound_proximity = float(fit_plan["bound_proximity_in_parameter_scales"])
    maximum_sensitivity_condition = float(fit_plan["maximum_sensitivity_condition"])
    recorded_adapter_sha256 = fit_provenance.get("fit_adapter", {}).get("sha256")
    recorded_implementation_sha256 = fit_provenance.get("implementation", {}).get("sha256")
    qualification_adapter_sha256 = (
        recorded_adapter_sha256 if parameter_replay_requested else adapter_identity["sha256"]
    )
    qualification_implementation_sha256 = (
        recorded_implementation_sha256
        if parameter_replay_requested
        else implementation_identity["sha256"]
    )
    fit_chain = _load_qualified_stage_chain(
        fit_path,
        expected_stage="joint",
        trusted_recipe=recipe,
        fit_plan=fit_plan,
        diagnostic_sha256=diagnostic_identity["sha256"],
        recipe_sha256=recipe_identity["sha256"],
        fit_plan_sha256=fit_plan_identity["sha256"],
        adapter_sha256=qualification_adapter_sha256,
        implementation_sha256=qualification_implementation_sha256,
        allow_model_limited_joint=True,
    )
    policy = _profile_evidence_policy(
        fit_document=fit_document,
        trusted_recipe=recipe,
        recipe_sha256=recipe_identity["sha256"],
        adapter_sha256=qualification_adapter_sha256,
        fit_plan_sha256=fit_plan_identity["sha256"],
        implementation_sha256=qualification_implementation_sha256,
        lower_bounds=lower_full,
        upper_bounds=upper_full,
        parameter_scales=scale_full,
        sensitivity_relative_tolerance=sensitivity_relative_tolerance,
        bound_proximity_in_parameter_scales=bound_proximity,
        maximum_sensitivity_condition=maximum_sensitivity_condition,
        expected_dataset_ids=manifest["dataset_ids"],
        predecessor_chain_complete=bool(fit_chain),
        expected_execution_policy=str(fit_plan["execution_policy"]),
    )
    fixed_lattice_record = _prepared_lattice_record(manifest, series)
    dataset_ids = tuple(str(value) for value in manifest["dataset_ids"])
    display_dataset_id = str(recipe["display_dataset_id"])
    display_index = dataset_ids.index(display_dataset_id)
    inputs = series[display_index]
    osc_record = next(
        item for item in manifest["osc_provenance"] if item["dataset_id"] == display_dataset_id
    )
    osc_path = Path(osc_record["osc_path"]).resolve()
    osc_identity = _file_identity(osc_path)
    if osc_identity["sha256"] != osc_record["osc_file_sha256"]:
        raise ValueError("display OSC compressed bytes changed")
    counts = read_osc(osc_path).detector_native_counts
    if (
        hashlib.sha256(counts.tobytes(order="C")).hexdigest()
        != osc_record["detector_native_bytes_sha256"]
    ):
        raise ValueError("display OSC native counts changed")
    dark_counts, dark_scale = _verified_dark_counts(manifest)
    if dark_counts.shape != counts.shape:
        raise ValueError("display OSC and dark OSC shapes differ")
    dark_identity = {
        "path": str(manifest["dark_correction"]["path"]),
        "sha256": str(manifest["dark_correction"]["file_sha256"]),
    }
    m0 = _m0_region(recipe)
    beam_center_column_px = float(manifest["fixed_position"]["beam_center_column_row_px"][0])
    layouts = _offspecular_layouts(recipe, beam_center_column_px, counts.shape[1])
    row_records, lookup = _row_catalog(1, m0, layouts)
    m0_signal_catalog = tuple(
        {**record, "source_observation_row": row_index}
        for row_index, record in enumerate(row_records)
        if str(record["group"]) == "m0" and str(record["band"]) == "signal"
    )
    if not m0_signal_catalog:
        raise RuntimeError("display profile catalog lacks m=0 signal bins")
    row_count = len(row_records)
    count_sum = np.zeros(row_count, dtype=np.float64)
    count_support = np.zeros(row_count, dtype=np.float64)
    count_coordinate_mass = np.zeros(row_count, dtype=np.float64)
    count_axial_mass = np.zeros(row_count, dtype=np.float64)
    count_qz_mass = np.zeros(row_count, dtype=np.float64)
    full_region_code = np.zeros(counts.size, dtype=np.uint8)
    selected_pixel, selected_row = _prepare_dataset_membership(
        inputs,
        counts,
        recipe=recipe,
        dataset_id=display_dataset_id,
        dataset_index=0,
        lookup=lookup,
        m0=m0,
        layouts=layouts,
        count_sum=count_sum,
        support=count_support,
        coordinate_sum=count_coordinate_mass,
        axial_sum=count_axial_mass,
        qz_sum=count_qz_mass,
        display_code=full_region_code,
        row_chunk_size=row_chunk_size,
        apply_fit_windows=False,
    )
    full_count_sum = np.array(count_sum, copy=True)
    full_count_support = np.array(count_support, copy=True)
    full_selected_pixel = np.array(selected_pixel, copy=True)
    full_selected_row = np.array(selected_row, copy=True)
    supported = count_support > 0.0
    complete = np.zeros(row_count, dtype=np.bool_)
    grouped_rows: dict[tuple[str, int], list[int]] = {}
    for row_index, record in enumerate(row_records):
        if supported[row_index]:
            grouped_rows.setdefault((str(record["group"]), int(record["bin"])), []).append(
                row_index
            )
    for indices in grouped_rows.values():
        bands = {str(row_records[index]["band"]) for index in indices}
        if {"background_0", "background_1"}.issubset(bands) and any(
            not band.startswith("background_") for band in bands
        ):
            complete[indices] = True
    retained_row = np.flatnonzero(complete)
    if not retained_row.size:
        raise ValueError("display profile selection has no complete signal/background blocks")
    old_to_new = np.full(row_count, -1, dtype=np.int64)
    old_to_new[retained_row] = np.arange(retained_row.size, dtype=np.int64)
    remapped_selected_row = old_to_new[selected_row]
    retained_selected = remapped_selected_row >= 0
    selected_pixel = selected_pixel[retained_selected]
    selected_row = remapped_selected_row[retained_selected]
    row_records = [row_records[index] for index in retained_row]
    count_sum = count_sum[retained_row]
    count_support = count_support[retained_row]
    count_coordinate_mass = count_coordinate_mass[retained_row]
    count_axial_mass = count_axial_mass[retained_row]
    count_qz_mass = count_qz_mass[retained_row]
    row_count = retained_row.size
    parameters = _fit_structure_vector(fit_document)
    fit_parameter_replay: dict[str, Any] | None = None
    if parameter_replay_requested:
        fit_stitch = recipe.get("parratt_stitch")
        profile_stitch = profile_recipe.get("parratt_stitch")
        if not isinstance(fit_stitch, dict) or not isinstance(profile_stitch, dict):
            raise ValueError("profile parameter replay requires explicit fit and profile stitches")
        if fit_stitch.get("interface_assumption") == profile_stitch.get("interface_assumption"):
            raise ValueError("profile parameter replay must change the m=0 interface assumption")
        replay_scales = {
            dataset_id: float(fit_document["dataset_scales"][dataset_id])
            for dataset_id in dataset_ids
        }
        fit_parameter_replay = {
            "method": PROFILE_PARAMETER_REPLAY_METHOD,
            "status": "COMPLETE",
            "fit_role": "frozen_parameter_and_dataset_scale_source_only",
            "profile_role": "continuous_profile_recalculation_without_optimization",
            "scope": "m0_specular_interface_assumption_only",
            "fit_parameters_reused": True,
            "optimizer_executed": False,
            "fit_reexecuted": False,
            "profile_reexecuted": True,
            "objective_requalified_under_profile_model": False,
            "dataset_scales_requalified": False,
            "source_fit_sha256": fit_identity["sha256"],
            "source_fit_status": fit_document.get("status"),
            "structure_parameter_vector_sha256": _array_sha256(parameters),
            "dataset_scales": replay_scales,
            "dataset_scale_vector_sha256": _array_sha256(
                np.asarray([replay_scales[dataset_id] for dataset_id in dataset_ids])
            ),
            "fit_interface_assumption": str(fit_stitch["interface_assumption"]),
            "profile_interface_assumption": str(profile_stitch["interface_assumption"]),
        }
        policy = {
            **policy,
            "evidence_level": PROFILE_PARAMETER_REPLAY_EVIDENCE,
            "publication_ready": False,
        }
    structure = fit_document["structure_representative"]
    maximum_blocks = int(recipe["model_cubature"]["maximum_state_block_count"])
    profile_rods = tuple(rod for rod in inputs.rods if rod.family_m in FAMILIES)
    if (
        tuple(manifest["families_m"]) != FAMILIES
        or not profile_rods
        or {rod.family_m for rod in profile_rods} != set(FAMILIES)
    ):
        raise RuntimeError("FIT_CONDITIONED rod reconstruction changed")
    rod_roster = _fitted_rod_roster(inputs)
    rod_roster_sha256 = _rod_roster_sha256(rod_roster)
    if rod_roster_sha256 != fit_document.get("model_rod_roster_sha256"):
        raise RuntimeError("profile rod roster differs from the qualified fit")
    specular_stitch = _recipe_parratt_stitch(profile_recipe)
    detector = build_source_averaged_detector(inputs)
    if len(profile_rods) != len(inputs.rods):
        detector = detector.restrict_rods(profile_rods)
    detector = detector.rebind_physics(
        strength_model=_candidate_strength(
            detector.strength_model,
            parameters,
            fit_plan=fit_plan,
        ),
        intensity_envelope=_candidate_intensity_envelope(parameters),
    )
    detector = detector.with_specular_stitch(specular_stitch)
    detector = detector.with_maximum_state_block_count(maximum_blocks)
    profile_cubature = recipe["profile_cubature"]
    cuda_chunk_size = int(recipe["model_cubature"]["cuda_coordinate_chunk_size"])
    profile_row_arrays = {
        "dataset_index": np.zeros(row_count, dtype=np.int64),
        "group": np.asarray([record["group"] for record in row_records]),
        "band": np.asarray([record["band"] for record in row_records]),
        "bin_index": np.asarray([record["bin"] for record in row_records], dtype=np.int64),
    }
    profile_plan_manifest = {**manifest, "dataset_ids": [display_dataset_id]}
    pass_specs = {
        "fit": (
            int(profile_cubature["fit_gauss_order"]),
            int(profile_cubature["fold_fit_subdivisions"]),
        ),
    }
    raw_profile_plans = {
        name: _compile_dataset_continuous_region_plan(
            inputs,
            profile_row_arrays,
            profile_plan_manifest,
            dataset_index=0,
            gauss_order=pass_specs[name][0],
            subdivision_count=pass_specs[name][1],
            offspecular_axial_refinement=offspecular_axial_refinement,
            offspecular_radial_transform=offspecular_radial_transform,
            offspecular_signal_minimum_radial_nodes_per_side=(
                offspecular_signal_minimum_radial_nodes_per_side
            ),
            m0_phi_subdivision_count=int(profile_cubature["fold_fit_subdivisions"]),
            apply_fit_windows=False,
        )
        for name in policy["pass_names"]
    }
    raw_measured_profile_plan = (
        None
        if native_observations
        else _compile_dataset_continuous_region_plan(
            inputs,
            profile_row_arrays,
            profile_plan_manifest,
            dataset_index=0,
            gauss_order=int(recipe["model_cubature"]["oracle_gauss_order"]),
            subdivision_count=int(recipe["model_cubature"]["fold_oracle_subdivisions"]),
            offspecular_axial_refinement=offspecular_axial_refinement,
            offspecular_radial_transform=offspecular_radial_transform,
            offspecular_signal_minimum_radial_nodes_per_side=(
                offspecular_signal_minimum_radial_nodes_per_side
            ),
            m0_phi_subdivision_count=int(recipe["model_cubature"]["fold_oracle_subdivisions"]),
            apply_fit_windows=False,
        )
    )
    raw_coarse_measured_profile_plan = (
        None
        if native_observations
        else _compile_dataset_continuous_region_plan(
            inputs,
            profile_row_arrays,
            profile_plan_manifest,
            dataset_index=0,
            gauss_order=int(profile_cubature["fit_gauss_order"]),
            subdivision_count=int(profile_cubature["fold_fit_subdivisions"]),
            offspecular_axial_refinement=offspecular_axial_refinement,
            offspecular_radial_transform=offspecular_radial_transform,
            offspecular_signal_minimum_radial_nodes_per_side=(
                offspecular_signal_minimum_radial_nodes_per_side
            ),
            m0_phi_subdivision_count=int(profile_cubature["fold_fit_subdivisions"]),
            apply_fit_windows=False,
        )
    )
    invalid_fold_observations_by_pass = {
        name: _invalid_continuous_fold_observations(plan)
        for name, plan in raw_profile_plans.items()
    }
    plans_for_support_screening = dict(raw_profile_plans)
    if raw_measured_profile_plan is not None and raw_coarse_measured_profile_plan is not None:
        invalid_fold_observations_by_pass["measured_data"] = _invalid_continuous_fold_observations(
            raw_measured_profile_plan
        )
        invalid_fold_observations_by_pass["coarse_measured_data"] = (
            _invalid_continuous_fold_observations(raw_coarse_measured_profile_plan)
        )
        plans_for_support_screening.update(
            measured_data=raw_measured_profile_plan,
            coarse_measured_data=raw_coarse_measured_profile_plan,
        )
    for name, plan in plans_for_support_screening.items():
        for observation in np.flatnonzero(~plan.quadrature.observation_covered):
            invalid_fold_observations_by_pass[name].setdefault(
                int(observation),
                "continuous detector region has zero area",
            )
    invalid_fold_observations = {
        observation
        for invalid_by_observation in invalid_fold_observations_by_pass.values()
        for observation in invalid_by_observation
    }
    retained_signal_blocks = {
        (str(record["group"]), int(record["bin"]))
        for observation, record in enumerate(row_records)
        if not str(record["band"]).startswith("background_")
        and observation not in invalid_fold_observations
    }
    valid_anchor_bands_by_block: dict[tuple[str, int], set[str]] = {}
    for observation, record in enumerate(row_records):
        if (
            str(record["band"]).startswith("background_")
            and observation not in invalid_fold_observations
        ):
            key = (str(record["group"]), int(record["bin"]))
            valid_anchor_bands_by_block.setdefault(key, set()).add(str(record["band"]))
    complete_blocks = {
        key
        for key in retained_signal_blocks
        if {"background_0", "background_1"}.issubset(valid_anchor_bands_by_block.get(key, set()))
    }
    conditioned_m0_bins = {bin_index for group, bin_index in complete_blocks if group == "m0"}
    m0_signal_only_records = tuple(
        record for record in m0_signal_catalog if int(record["bin"]) not in conditioned_m0_bins
    )
    m0_signal_only_row_count = len(m0_signal_only_records)
    m0_signal_only_plan: _DatasetContinuousRegionPlan | None = None
    if m0_signal_only_records:
        m0_signal_only_row_arrays = {
            "dataset_index": np.zeros(m0_signal_only_row_count, dtype=np.int64),
            "group": np.asarray([record["group"] for record in m0_signal_only_records]),
            "band": np.asarray([record["band"] for record in m0_signal_only_records]),
            "bin_index": np.asarray(
                [record["bin"] for record in m0_signal_only_records],
                dtype=np.int64,
            ),
        }
        raw_m0_signal_only_plan = _compile_dataset_continuous_region_plan(
            inputs,
            m0_signal_only_row_arrays,
            profile_plan_manifest,
            dataset_index=0,
            gauss_order=int(recipe["model_cubature"]["oracle_gauss_order"]),
            subdivision_count=int(recipe["model_cubature"]["fold_oracle_subdivisions"]),
            offspecular_axial_refinement=offspecular_axial_refinement,
            offspecular_radial_transform=offspecular_radial_transform,
            offspecular_signal_minimum_radial_nodes_per_side=(
                offspecular_signal_minimum_radial_nodes_per_side
            ),
            m0_phi_subdivision_count=int(recipe["model_cubature"]["fold_oracle_subdivisions"]),
            apply_fit_windows=False,
        )
        invalid_m0_signal_only_observations = set(
            _invalid_continuous_fold_observations(raw_m0_signal_only_plan)
        ) | set(
            int(value)
            for value in np.flatnonzero(~raw_m0_signal_only_plan.quadrature.observation_covered)
        )
        m0_signal_only_plan = _drop_optional_continuous_plan_observations(
            raw_m0_signal_only_plan,
            invalid_m0_signal_only_observations,
        )
        if m0_signal_only_plan is None:
            m0_signal_only_records = ()
            m0_signal_only_row_count = 0
    incomplete_block_observations = {
        observation
        for observation, record in enumerate(row_records)
        if (str(record["group"]), int(record["bin"])) not in complete_blocks
    }
    unsupported_observations = invalid_fold_observations | incomplete_block_observations
    profile_plans = {
        name: _drop_continuous_plan_observations(plan, unsupported_observations)
        for name, plan in raw_profile_plans.items()
    }
    measured_profile_plan = (
        None
        if raw_measured_profile_plan is None
        else _drop_continuous_plan_observations(
            raw_measured_profile_plan,
            unsupported_observations,
        )
    )
    coarse_measured_profile_plan = (
        None
        if raw_coarse_measured_profile_plan is None
        else _drop_continuous_plan_observations(
            raw_coarse_measured_profile_plan,
            unsupported_observations,
        )
    )
    profile_fold_plans = {
        name: compile_continuous_fold_correction_plan(detector, plan.fold_bands)
        for name, plan in profile_plans.items()
    }
    support_exclusions = []
    for observation in sorted(invalid_fold_observations):
        record = row_records[observation]
        support_exclusions.append(
            {
                "profile_identity": str(record["band"]),
                "family_m": int(record["signal_family"]),
                "bin_index": int(record["bin"]),
                "reason": "continuous fold chart outside valid detector domain",
                "detail_by_pass": {
                    name: invalid_by_observation[observation]
                    for name, invalid_by_observation in invalid_fold_observations_by_pass.items()
                    if observation in invalid_by_observation
                },
            }
        )

    def project_dark_corrected_counts(
        projection: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_mass, raw_covariance = projection.integrate_counts(counts)
        dark_mass, dark_covariance = projection.integrate_field(
            dark_counts,
            np.maximum(dark_counts, 1.0),
        )
        return (
            raw_mass - dark_scale * dark_mass,
            raw_covariance + dark_scale * dark_scale * dark_covariance,
        )

    coarse_measured_projection = None
    measured_projection = None
    m0_signal_only_projection = None
    data_projection_qualification_required = _data_projection_qualification_required(recipe)
    native_profile_plans: tuple[_DatasetNativePixelCenterPlan, ...] = ()
    native_m0_signal_only_plans: tuple[_DatasetNativePixelCenterPlan, ...] = ()
    m0_signal_only_count_mass = np.empty(0, dtype=np.float64)
    m0_signal_only_count_covariance = np.empty((0, 0), dtype=np.float64)
    m0_signal_only_support = np.empty(0, dtype=np.float64)
    m0_signal_only_flat_pixel = np.empty(0, dtype=np.int64)
    if native_observations:
        retained_native_member = ~np.isin(
            selected_row, np.asarray(sorted(unsupported_observations))
        )
        selected_pixel = selected_pixel[retained_native_member]
        selected_row = selected_row[retained_native_member]
        data_supported = np.ones(row_count, dtype=np.bool_)
        if unsupported_observations:
            data_supported[np.asarray(sorted(unsupported_observations), dtype=np.int64)] = False
        raw_count_sum = np.where(data_supported, count_sum, 0.0)
        raw_count_support = np.where(data_supported, count_support, 0.0)
        count_coordinate_mass = np.where(data_supported, count_coordinate_mass, 0.0)
        count_axial_mass = np.where(data_supported, count_axial_mass, 0.0)
        count_qz_mass = np.where(data_supported, count_qz_mass, 0.0)
        native_profile_arrays = {
            "dataset_index": np.zeros(row_count, dtype=np.int64),
            "count_sum": raw_count_sum,
            "support_px2": raw_count_support,
            "selected_dataset_index": np.zeros(selected_pixel.size, dtype=np.int64),
            "selected_flat_pixel_index": selected_pixel,
            "selected_observation_row": selected_row,
        }
        (
            count_sum,
            count_covariance,
            count_support,
            native_profile_plans,
        ) = _native_pixel_center_count_statistics(
            native_profile_arrays,
            dataset_ids=(display_dataset_id,),
            counts_by_dataset={display_dataset_id: counts},
            dark_counts=dark_counts,
            dark_scale=dark_scale,
        )
        measured_projection = native_profile_plans[0].projection
        measured_projection_refinement: dict[str, Any] = {
            "status": "NOT_APPLICABLE_EXACT_MEMBERSHIP",
            "converged": True,
            "projection_performed": False,
        }
        if m0_signal_only_plan is not None and m0_signal_only_records:
            source_rows = np.asarray(
                [record["source_observation_row"] for record in m0_signal_only_records],
                dtype=np.int64,
            )
            source_to_local = np.full(full_count_sum.size, -1, dtype=np.int64)
            source_to_local[source_rows] = np.arange(source_rows.size, dtype=np.int64)
            source_member = source_to_local[full_selected_row] >= 0
            m0_signal_only_flat_pixel = full_selected_pixel[source_member]
            m0_signal_only_selected_row = source_to_local[full_selected_row[source_member]]
            m0_signal_only_arrays = {
                "dataset_index": np.zeros(source_rows.size, dtype=np.int64),
                "count_sum": full_count_sum[source_rows],
                "support_px2": full_count_support[source_rows],
                "selected_dataset_index": np.zeros(
                    m0_signal_only_flat_pixel.size,
                    dtype=np.int64,
                ),
                "selected_flat_pixel_index": m0_signal_only_flat_pixel,
                "selected_observation_row": m0_signal_only_selected_row,
            }
            (
                m0_signal_only_count_mass,
                m0_signal_only_count_covariance,
                m0_signal_only_support,
                native_m0_signal_only_plans,
            ) = _native_pixel_center_count_statistics(
                m0_signal_only_arrays,
                dataset_ids=(display_dataset_id,),
                counts_by_dataset={display_dataset_id: counts},
                dark_counts=dark_counts,
                dark_scale=dark_scale,
            )
            m0_signal_only_flat_pixel = np.unique(m0_signal_only_flat_pixel)
    else:
        if measured_profile_plan is None or coarse_measured_profile_plan is None:
            raise RuntimeError("continuous measured-profile plans were not compiled")
        coarse_measured_projection = compile_native_pixel_region_projection(
            coarse_measured_profile_plan.quadrature,
            counts.shape,
        )
        coarse_count_sum, coarse_count_covariance = project_dark_corrected_counts(
            coarse_measured_projection
        )
        coarse_count_support = coarse_measured_projection.observation_measure_px2
        measured_projection = compile_native_pixel_region_projection(
            measured_profile_plan.quadrature,
            counts.shape,
        )
        count_sum, count_covariance = project_dark_corrected_counts(measured_projection)
        count_support = measured_projection.observation_measure_px2
        if m0_signal_only_plan is not None:
            m0_signal_only_projection = compile_native_pixel_region_projection(
                m0_signal_only_plan.quadrature,
                counts.shape,
            )
            m0_signal_only_count_mass, m0_signal_only_count_covariance = (
                project_dark_corrected_counts(m0_signal_only_projection)
            )
            m0_signal_only_support = m0_signal_only_projection.observation_measure_px2
            m0_signal_only_flat_pixel = np.unique(
                np.asarray(m0_signal_only_projection.flat_pixel_index, dtype=np.int64)[
                    np.asarray(m0_signal_only_projection.pixel_column_index, dtype=np.int64)
                ]
            )
        measured_projection_refinement = _native_projection_convergence(
            coarse_count_mass=coarse_count_sum,
            coarse_support_px2=coarse_count_support,
            coarse_count_covariance=coarse_count_covariance,
            refined_count_mass=count_sum,
            refined_support_px2=count_support,
            refined_count_covariance=count_covariance,
            dataset_index=np.zeros(row_count, dtype=np.int64),
            dataset_ids=(display_dataset_id,),
            maximum_relative_l2=float(recipe["model_cubature"]["maximum_oracle_relative_l2"]),
        )
        covariance_refinement_converged = bool(measured_projection_refinement["converged"])
        measured_projection_refinement = {
            **measured_projection_refinement,
            "converged": _display_profile_projection_converged(measured_projection_refinement),
            "acceptance_measure": "display_profile_pooled_count_mass_and_support.v1",
            "covariance_refinement_converged": covariance_refinement_converged,
            "covariance_policy": "refined_covariance_is_authoritative_for_display",
        }
        data_projection_qualification_required = _data_projection_qualification_required(recipe)
        if (
            data_projection_qualification_required
            and not measured_projection_refinement["converged"]
        ):
            raise FloatingPointError(
                "display native-pixel continuous-region projection did not converge; "
                f"refine cubature: {measured_projection_refinement}"
            )
        count_coordinate_mass = (
            count_support * measured_profile_plan.quadrature.observation_background_coordinate
        )
        count_axial_mass, count_qz_mass = _continuous_plan_reciprocal_coordinate_moments(
            inputs,
            measured_profile_plan,
            m0_observation=np.asarray(
                [str(record["group"]) == "m0" for record in row_records],
                dtype=np.bool_,
            ),
        )
        selected_row = np.asarray(measured_projection.observation_row, dtype=np.int64)
        selected_pixel = np.asarray(measured_projection.flat_pixel_index)[
            np.asarray(measured_projection.pixel_column_index, dtype=np.int64)
        ]
    region_code_by_row = np.asarray(
        [
            _region_display_code(
                str(record["group"]),
                str(record["band"]),
                int(record["signal_family"]),
            )
            for record in row_records
        ],
        dtype=np.uint8,
    )
    full_region_code.fill(0)
    np.maximum.at(full_region_code, selected_pixel, region_code_by_row[selected_row])
    np.maximum.at(full_region_code, m0_signal_only_flat_pixel, np.uint8(1))
    if native_observations:
        display_fit_projection = None
        display_fit_member = np.asarray(fit_arrays["selected_dataset_index"]) == display_index
        display_fit_pair_pixel = np.asarray(fit_arrays["selected_flat_pixel_index"])[
            display_fit_member
        ]
        display_fit_global_row = np.asarray(fit_arrays["selected_observation_row"])[
            display_fit_member
        ]
    else:
        display_fit_plan = _compile_dataset_continuous_region_plan(
            inputs,
            fit_arrays,
            manifest,
            dataset_index=display_index,
            gauss_order=int(recipe["model_cubature"]["oracle_gauss_order"]),
            subdivision_count=int(recipe["model_cubature"]["fold_oracle_subdivisions"]),
            offspecular_axial_refinement=offspecular_axial_refinement,
            offspecular_radial_transform=offspecular_radial_transform,
            offspecular_signal_minimum_radial_nodes_per_side=(
                offspecular_signal_minimum_radial_nodes_per_side
            ),
            m0_phi_subdivision_count=int(recipe["model_cubature"]["fold_oracle_subdivisions"]),
        )
        display_fit_projection = compile_native_pixel_region_projection(
            display_fit_plan.quadrature,
            counts.shape,
        )
        display_fit_pair_pixel = np.asarray(display_fit_projection.flat_pixel_index)[
            np.asarray(display_fit_projection.pixel_column_index, dtype=np.int64)
        ]
        display_fit_global_row = np.asarray(display_fit_plan.global_observation_row)[
            np.asarray(display_fit_projection.observation_row, dtype=np.int64)
        ]
    display_fit_code = np.zeros(counts.size, dtype=np.uint8)
    display_fit_code_by_global_row = np.asarray(
        [
            _region_display_code(str(group), str(band), int(family))
            for group, band, family in zip(
                fit_arrays["group"],
                fit_arrays["band"],
                fit_arrays["signal_family_m"],
                strict=True,
            )
        ],
        dtype=np.uint8,
    )
    np.maximum.at(
        display_fit_code,
        display_fit_pair_pixel,
        display_fit_code_by_global_row[display_fit_global_row],
    )
    radial_profile_background = (
        _fixed_background_from_native_pixel_center_plans(
            state=background_state,
            plans=native_profile_plans,
            observation_count=row_count,
            beam_center_column_row_px=tuple(
                manifest["fixed_position"]["beam_center_column_row_px"]
            ),
        )
        if native_observations
        else _fixed_background_from_continuous_plans(
            state=background_state,
            plans=(measured_profile_plan,),
            observation_count=row_count,
            beam_center_column_row_px=tuple(
                manifest["fixed_position"]["beam_center_column_row_px"]
            ),
        )
    )
    m0_signal_only_background_mass = np.empty(0, dtype=np.float64)
    if m0_signal_only_plan is not None:
        m0_signal_only_background_mass = (
            _fixed_background_from_native_pixel_center_plans(
                state=background_state,
                plans=native_m0_signal_only_plans,
                observation_count=m0_signal_only_row_count,
                beam_center_column_row_px=tuple(
                    manifest["fixed_position"]["beam_center_column_row_px"]
                ),
            ).count_mass
            if native_observations
            else _fixed_background_from_continuous_plans(
                state=background_state,
                plans=(m0_signal_only_plan,),
                observation_count=m0_signal_only_row_count,
                beam_center_column_row_px=tuple(
                    manifest["fixed_position"]["beam_center_column_row_px"]
                ),
            ).count_mass
        )
    supported_row = np.flatnonzero(count_support > 0.0)
    supported_records = [row_records[index] for index in supported_row]
    block_by_key: dict[tuple[str, int], int] = {}
    supported_block_index = np.empty(supported_row.size, dtype=np.int64)
    for local_index, record in enumerate(supported_records):
        key = (str(record["group"]), int(record["bin"]))
        supported_block_index[local_index] = block_by_key.setdefault(key, len(block_by_key))
    supported_is_background = np.asarray(
        [str(record["band"]).startswith("background_") for record in supported_records],
        dtype=np.bool_,
    )
    supported_signal_family = np.asarray(
        [
            -1 if background else int(record["signal_family"])
            for record, background in zip(
                supported_records,
                supported_is_background,
                strict=True,
            )
        ],
        dtype=np.int64,
    )
    profile_observations = MatchedRegionObservations(
        dataset_ids=(display_dataset_id,),
        dataset_index=np.zeros(supported_row.size, dtype=np.int64),
        block_index=supported_block_index,
        signal_family=supported_signal_family,
        is_background=supported_is_background,
        count_mass=count_sum[supported_row],
        support_px2=count_support[supported_row],
        background_coordinate=(count_coordinate_mass[supported_row] / count_support[supported_row]),
        required_signal_families=FAMILIES,
        count_covariance_count2=count_covariance[np.ix_(supported_row, supported_row)],
    )
    conditioned_profile_background = condition_matched_region_background_from_anchors(
        profile_observations,
        FixedMatchedRegionBackground(
            count_mass=radial_profile_background.count_mass[supported_row],
            covariance_count2=radial_profile_background.covariance_count2[
                np.ix_(supported_row, supported_row)
            ],
            revision=radial_profile_background.revision,
        ),
    )
    conditioned_profile_background_mass = np.asarray(
        radial_profile_background.count_mass,
        dtype=np.float64,
    ).copy()
    conditioned_profile_background_mass[supported_row] = conditioned_profile_background.count_mass
    conditioned_profile_background_covariance = np.asarray(
        radial_profile_background.covariance_count2,
        dtype=np.float64,
    ).copy()
    conditioned_profile_background_covariance[np.ix_(supported_row, supported_row)] = (
        conditioned_profile_background.covariance_count2
    )
    dataset_scale = float(fit_document["dataset_scales"][display_dataset_id])
    model_input_identities = {
        name: _file_identity(_verified_provenance_path(manifest, name))
        for name in MODEL_INPUT_NAMES
    }
    guarded_inputs = {
        "profile_adapter": adapter_identity,
        "prepared_diagnostic": diagnostic_identity,
        "fit_artifact": fit_identity,
        "background": background_identity,
        "recipe": recipe_identity,
        "fit_plan": fit_plan_identity,
        "osc": osc_identity,
        "dark_osc": dark_identity,
        **model_input_identities,
    }
    for stage_document, stage_identity in fit_chain:
        guarded_inputs[f"fit_stage_{stage_document['stage']}"] = stage_identity

    def verify_inputs() -> None:
        for role, identity in guarded_inputs.items():
            _require_unchanged(identity, role=role.replace("_", " "))
        if _implementation_identity() != implementation_identity:
            raise RuntimeError("scientific implementation changed during profile generation")

    measured_execution_identity = {
        "measured_observation_method": (
            NATIVE_PIXEL_CENTER_METHOD if native_observations else MEASURED_PROJECTION_METHOD
        ),
        "measured_projection_performed": not native_observations,
        "coarse_measured_projection_revision": (
            None
            if coarse_measured_projection is None
            else coarse_measured_projection.projection_revision
        ),
        "measured_projection_revision": measured_projection.projection_revision,
        "measured_quadrature_revision": (
            None
            if measured_profile_plan is None
            else measured_profile_plan.quadrature.quadrature_revision
        ),
        "measured_continuous_node_count": (
            0
            if measured_profile_plan is None
            else int(measured_profile_plan.quadrature.column_px.size)
        ),
        "display_fit_projection_revision": (
            None if display_fit_projection is None else display_fit_projection.projection_revision
        ),
        "display_fit_native_membership_sha256": (
            _array_sha256(display_fit_pair_pixel) if native_observations else None
        ),
        "measured_selected_pixel_region_pair_count": (
            int(selected_pixel.size) if native_observations else None
        ),
        "measured_selected_flat_pixel_index_sha256": (
            _array_sha256(selected_pixel) if native_observations else None
        ),
        "measured_selected_observation_row_sha256": (
            _array_sha256(selected_row) if native_observations else None
        ),
        "measured_count_mass_sha256": _array_sha256(count_sum),
        "measured_support_px2_sha256": _array_sha256(count_support),
        "measured_count_covariance_sha256": _array_sha256(count_covariance),
    }
    execution_identity = {
        "diagnostic_sha256": diagnostic_identity["sha256"],
        "fit_sha256": fit_identity["sha256"],
        "evidence_level": policy["evidence_level"],
        "recipe_sha256": recipe_identity["sha256"],
        "fit_plan_sha256": fit_plan_identity["sha256"],
        "implementation_sha256": implementation_identity["sha256"],
        "fit_chain_sha256": [identity["sha256"] for _, identity in fit_chain],
        "profile_adapter_sha256": adapter_identity["sha256"],
        "fit_origin_adapter_sha256": recorded_adapter_sha256,
        "fit_origin_implementation_sha256": recorded_implementation_sha256,
        "fit_parameter_replay": fit_parameter_replay,
        "osc_sha256": osc_identity["sha256"],
        "dark_osc_sha256": dark_identity["sha256"],
        "dark_scale": dark_scale,
        "selected_pixel_sha256": _array_sha256(selected_pixel),
        "selected_observation_row_sha256": _array_sha256(selected_row),
        "selected_pixel_count": int(selected_pixel.size),
        "selected_unique_pixel_count": int(np.unique(selected_pixel).size),
        **measured_execution_identity,
        "m0_signal_only_projection_revision": (
            native_m0_signal_only_plans[0].projection.projection_revision
            if native_m0_signal_only_plans
            else (
                None
                if m0_signal_only_projection is None
                else m0_signal_only_projection.projection_revision
            )
        ),
        "m0_signal_only_measured_quadrature_revision": (
            None
            if native_observations or m0_signal_only_plan is None
            else m0_signal_only_plan.quadrature.quadrature_revision
        ),
        "m0_signal_only_model_quadrature_revision": (
            None
            if m0_signal_only_plan is None
            else m0_signal_only_plan.quadrature.quadrature_revision
        ),
        "m0_signal_only_bin_count": m0_signal_only_row_count,
        "m0_signal_only_count_mass_sha256": _array_sha256(m0_signal_only_count_mass),
        "m0_signal_only_count_covariance_sha256": _array_sha256(m0_signal_only_count_covariance),
        "m0_signal_only_radial_background_mass_sha256": _array_sha256(
            m0_signal_only_background_mass
        ),
        "row_count": row_count,
        "background_artifact_sha256": background_identity["sha256"],
        "radial_background_state_revision": background_state.revision,
        "radial_background_parameter_vector_sha256": _array_sha256(
            background_state.parameter_vector
        ),
        "radial_background_parameter_covariance_sha256": _array_sha256(
            background_state.parameter_covariance
        ),
        "radial_background_mass_sha256": _array_sha256(radial_profile_background.count_mass),
        "radial_background_covariance_sha256": _array_sha256(
            radial_profile_background.covariance_count2
        ),
        "background_excluded_flat_pixel_count_by_dataset": background_manifest["sampling"][
            "excluded_flat_pixel_count_by_dataset"
        ],
        "background_excluded_flat_pixel_sha256_by_dataset": background_manifest["sampling"][
            "excluded_flat_pixel_sha256_by_dataset"
        ],
        "conditioned_background_revision": conditioned_profile_background.revision,
        "conditioned_background_mass_sha256": _array_sha256(conditioned_profile_background_mass),
        "conditioned_background_covariance_sha256": _array_sha256(
            conditioned_profile_background_covariance
        ),
        "conditioned_model_anchor_projection_sha256": _array_sha256(
            conditioned_profile_background.anchor_projection
        ),
        "structure_parameters": parameters.tolist(),
        "dataset_scale": dataset_scale,
        "execution_backend": execution_backend,
        "model_measure": "continuous_detector_chart_area",
        "fit_gauss_order": int(profile_cubature["fit_gauss_order"]),
        "fit_subdivision_count": int(profile_cubature["fold_fit_subdivisions"]),
        "m0_model_phi_subdivision_count": int(profile_cubature["fold_fit_subdivisions"]),
        "quadrature_revision": {
            name: plan.quadrature.quadrature_revision for name, plan in profile_plans.items()
        },
        "continuous_node_count": {
            name: int(plan.quadrature.column_px.size) for name, plan in profile_plans.items()
        },
        "continuous_fold_plan_sha256": {
            name: None if plan is None else plan.plan_sha256
            for name, plan in profile_fold_plans.items()
        },
        "continuous_support_exclusions": support_exclusions,
        "cuda_coordinate_chunk_size": cuda_chunk_size,
        "maximum_state_block_count": maximum_blocks,
        "row_chunk_size": int(row_chunk_size),
        "rod_scope": policy["model_rod_scope"],
        "rod_count": len(profile_rods),
        "rod_roster_sha256": rod_roster_sha256,
        "evaluated_passes": list(policy["pass_names"]),
        "model_input_sha256": {
            name: identity["sha256"] for name, identity in model_input_identities.items()
        },
    }
    progress_path = _external_file(destination.with_name(f"{destination.stem}.progress.json"))
    if progress_path.exists():
        raise FileExistsError(f"profile progress already exists: {progress_path}")
    progress_document = {
        "schema_version": PROFILE_PROGRESS_SCHEMA,
        "execution_identity": execution_identity,
        "devices": None,
        "evaluated_backends": None,
        "completed_passes": {},
        "active_pass": None,
    }
    verify_inputs()
    _write_json_atomic(progress_path, progress_document)
    pass_results: dict[str, dict[str, np.ndarray]] = {}
    executions: dict[str, dict[str, Any]] = {}
    for name in policy["pass_names"]:
        verify_inputs()
        plan = profile_plans[name]
        model_mass, executions[name] = _integrate_continuous_region_mass(
            detector,
            plan,
            profile_fold_plans[name],
            row_count=row_count,
            execution_backend=execution_backend,
            cuda_coordinate_chunk_size=cuda_chunk_size,
        )
        support = plan.quadrature.observation_measure_px2
        coordinate_mass = support * plan.quadrature.observation_background_coordinate
        objective_model_mass = model_mass.copy()
        objective_model_mass[supported_row] = condition_matched_region_model_from_anchors(
            conditioned_profile_background,
            model_mass[supported_row],
        )
        progress_plan = plan
        pass_results[name] = {
            "model_mass": objective_model_mass,
            "physical_model_mass": model_mass,
            "background_mass": conditioned_profile_background_mass,
            "support_px2": support,
            "coordinate_mass": coordinate_mass,
            "axial_mass": np.zeros(row_count, dtype=np.float64),
        }
        current_devices = sorted(set(executions[name]["devices"]))
        current_backends = sorted(set(executions[name]["evaluated_backends"]))
        if (
            progress_document["devices"] is not None
            and progress_document["devices"] != current_devices
        ):
            raise RuntimeError("profile execution device changed")
        if (
            progress_document["evaluated_backends"] is not None
            and progress_document["evaluated_backends"] != current_backends
        ):
            raise RuntimeError("profile execution backend changed")
        progress_document["devices"] = current_devices
        progress_document["evaluated_backends"] = current_backends
        progress_document["completed_passes"][name] = {
            "quadrature_revision": progress_plan.quadrature.quadrature_revision,
            "continuous_node_count": int(progress_plan.quadrature.column_px.size),
            "model_mass_sha256": _array_sha256(model_mass),
        }
        progress_document["active_pass"] = None
        verify_inputs()
        _write_json_atomic(progress_path, progress_document)
    m0_signal_only_model_mass = np.empty(0, dtype=np.float64)
    m0_signal_only_execution: dict[str, Any] | None = None
    if m0_signal_only_plan is not None:
        if m0_signal_only_plan.fold_bands:
            raise RuntimeError("m=0 signal-only display unexpectedly compiled fold corrections")
        verify_inputs()
        m0_signal_only_model_mass, m0_signal_only_execution = _integrate_continuous_region_mass(
            detector,
            m0_signal_only_plan,
            None,
            row_count=m0_signal_only_row_count,
            execution_backend=execution_backend,
            cuda_coordinate_chunk_size=cuda_chunk_size,
        )
        if sorted(set(m0_signal_only_execution["devices"])) != progress_document[
            "devices"
        ] or not set(m0_signal_only_execution["evaluated_backends"]).issubset(
            progress_document["evaluated_backends"]
        ):
            raise RuntimeError("m=0 signal-only display execution changed device or backend")
        progress_document["completed_passes"]["m0_signal_only_display"] = {
            "quadrature_revision": m0_signal_only_plan.quadrature.quadrature_revision,
            "continuous_node_count": int(m0_signal_only_plan.quadrature.column_px.size),
            "model_mass_sha256": _array_sha256(m0_signal_only_model_mass),
        }
    execution_identity["m0_signal_only_model_mass_sha256"] = _array_sha256(
        m0_signal_only_model_mass
    )
    verify_inputs()
    _write_json_atomic(progress_path, progress_document)
    projected = {
        name: _project_full_profiles(
            row_records,
            count_sum=count_sum,
            count_support=count_support,
            count_coordinate_mass=count_coordinate_mass,
            count_axial_mass=count_axial_mass,
            count_qz_mass=count_qz_mass,
            model=model,
            dataset_scale=dataset_scale,
            m0=m0,
        )
        for name, model in pass_results.items()
    }
    radial_projected = {
        name: _project_full_profiles(
            row_records,
            count_sum=count_sum,
            count_support=count_support,
            count_coordinate_mass=count_coordinate_mass,
            count_axial_mass=count_axial_mass,
            count_qz_mass=count_qz_mass,
            model={
                **model,
                "background_mass": np.asarray(radial_profile_background.count_mass),
            },
            dataset_scale=dataset_scale,
            m0=m0,
        )
        for name, model in pass_results.items()
    }
    minimum_valid_bins = int(profile_cubature["minimum_valid_bins_per_profile"])
    expected_profile_identities = {
        "m0",
        "m1_minus",
        "m1_plus",
        "m3_minus",
        "m3_plus",
        "m4_minus",
        "m4_plus",
    }
    fit_profiles = projected["fit"]
    if set(np.unique(fit_profiles["identity"])) != expected_profile_identities:
        raise RuntimeError("fitted profile identity roster changed")
    for identity in expected_profile_identities:
        selected = (fit_profiles["identity"] == identity) & fit_profiles["valid"]
        if int(np.count_nonzero(selected)) < minimum_valid_bins or any(
            np.any(~np.isfinite(fit_profiles[name][selected]))
            for name in (
                "display_L",
                "display_qz_Ainv",
                "selection_coordinate",
                "measured_signal_density",
                "model_signal_density",
            )
        ):
            raise RuntimeError(f"profile {identity!r} lacks finite fitted support")
    convergence: dict[str, float] = {}
    profile_arrays = projected[policy["result_pass_name"]]
    axial_reciprocal_magnitude_Ainv = float(
        np.linalg.norm(np.asarray(inputs.reciprocal.basis_Ainv, dtype=np.float64)[:, 2])
    )
    if not math.isfinite(axial_reciprocal_magnitude_Ainv) or axial_reciprocal_magnitude_Ainv <= 0.0:
        raise RuntimeError("accepted lattice has an invalid axial reciprocal-basis magnitude")
    m0_signal_only_bin_index = np.asarray(
        [record["bin"] for record in m0_signal_only_records],
        dtype=np.int64,
    )
    m0_edges_rad = np.asarray(m0.two_theta_bin_edges_rad, dtype=np.float64)
    m0_signal_only_two_theta_deg = np.rad2deg(
        0.5 * (m0_edges_rad[m0_signal_only_bin_index] + m0_edges_rad[m0_signal_only_bin_index + 1])
    )
    m0_signal_only_measured_density = np.full(
        m0_signal_only_row_count,
        np.nan,
        dtype=np.float64,
    )
    m0_signal_only_model_plus_background_density = np.full(
        m0_signal_only_row_count,
        np.nan,
        dtype=np.float64,
    )
    m0_signal_only_model_support = (
        np.empty(0, dtype=np.float64)
        if m0_signal_only_plan is None
        else np.asarray(m0_signal_only_plan.quadrature.observation_measure_px2)
    )
    m0_signal_only_valid = (
        (m0_signal_only_support > 0.0)
        & (m0_signal_only_model_support > 0.0)
        & np.isfinite(m0_signal_only_count_mass)
        & np.isfinite(m0_signal_only_model_mass)
        & np.isfinite(m0_signal_only_background_mass)
    )
    m0_signal_only_measured_density[m0_signal_only_valid] = (
        m0_signal_only_count_mass[m0_signal_only_valid]
        / m0_signal_only_support[m0_signal_only_valid]
    )
    m0_signal_only_model_plus_background_density[m0_signal_only_valid] = (
        dataset_scale * m0_signal_only_model_mass[m0_signal_only_valid]
        + m0_signal_only_background_mass[m0_signal_only_valid]
    ) / m0_signal_only_support[m0_signal_only_valid]
    configured_display_rows = int(recipe["detector"]["display_rows"])
    supplemental_display_rows = (
        int(np.max(m0_signal_only_flat_pixel) // counts.shape[1]) + 1
        if m0_signal_only_flat_pixel.size
        else 0
    )
    display_rows = min(
        counts.shape[0],
        max(configured_display_rows, supplemental_display_rows),
    )
    arrays = {
        **{f"profile_{name}": value for name, value in profile_arrays.items()},
        "profile_measured_radial_signal_density": radial_projected[policy["result_pass_name"]][
            "measured_signal_density"
        ],
        "profile_model_fit_signal_density": projected["fit"]["model_signal_density"],
        "row_group": np.asarray([record["group"] for record in row_records]),
        "row_band": np.asarray([record["band"] for record in row_records]),
        "row_bin_index": np.asarray([record["bin"] for record in row_records], dtype=np.int64),
        "row_signal_family_m": np.asarray(
            [record["signal_family"] for record in row_records], dtype=np.int64
        ),
        "row_count_sum": count_sum,
        "row_count_support_px2": count_support,
        "row_count_coordinate_mass": count_coordinate_mass,
        "row_count_axial_mass": count_axial_mass,
        "row_count_qz_mass": count_qz_mass,
        "selected_flat_pixel_index": selected_pixel,
        "selected_observation_row": selected_row,
        "m0_signal_only_bin_index": m0_signal_only_bin_index,
        "m0_signal_only_two_theta_deg": m0_signal_only_two_theta_deg,
        "m0_signal_only_measured_density": m0_signal_only_measured_density,
        "m0_signal_only_model_plus_background_density": (
            m0_signal_only_model_plus_background_density
        ),
        "m0_signal_only_support_px2": m0_signal_only_support,
        "m0_signal_only_valid": m0_signal_only_valid,
        "display_detector_counts": (
            counts[:display_rows].astype(np.float64) - dark_scale * dark_counts[:display_rows]
        ),
        "display_full_region_code": full_region_code.reshape(counts.shape)[:display_rows],
        "display_fit_region_code": display_fit_code.reshape(counts.shape)[:display_rows],
    }
    for pass_name, result_arrays in pass_results.items():
        for array_name, value in result_arrays.items():
            arrays[f"row_model_{pass_name}_{array_name.removeprefix('model_')}"] = value
    render_array_sha256 = {name: _array_sha256(arrays[name]) for name in PROFILE_RENDER_ARRAY_NAMES}
    completed_progress_identity = _file_identity(progress_path)
    measured_data_record = (
        {
            "method": NATIVE_PIXEL_CENTER_METHOD,
            "qualification_role": "authoritative_frozen_observation",
            "status": "COMPLETE",
            "projection_performed": False,
            "projection_revision": measured_projection.projection_revision,
            "selected_pixel_region_pair_count": int(selected_pixel.size),
            "selected_flat_pixel_index_sha256": _array_sha256(selected_pixel),
            "selected_observation_row_sha256": _array_sha256(selected_row),
            "count_mass_sha256": _array_sha256(count_sum),
            "support_px2_sha256": _array_sha256(count_support),
            "count_covariance_sha256": _array_sha256(count_covariance),
            "refinement_oracle": "NOT_APPLICABLE_EXACT_MEMBERSHIP",
            "smoothing_applied": False,
            "model_pixelized": False,
        }
        if native_observations
        else {
            "method": MEASURED_PROJECTION_METHOD,
            "qualification_role": "required_release_gate",
            "coarse_projection_revision": coarse_measured_projection.projection_revision,
            "projection_revision": measured_projection.projection_revision,
            "display_fit_projection_revision": display_fit_projection.projection_revision,
            "count_mass_sha256": _array_sha256(count_sum),
            "count_covariance_sha256": _array_sha256(count_covariance),
            "quadrature_revision": measured_profile_plan.quadrature.quadrature_revision,
            "gauss_order": int(recipe["model_cubature"]["oracle_gauss_order"]),
            "subdivision_count": int(recipe["model_cubature"]["fold_oracle_subdivisions"]),
            "m0_phi_subdivision_count": int(recipe["model_cubature"]["fold_oracle_subdivisions"]),
            "continuous_node_count": int(measured_profile_plan.quadrature.column_px.size),
            "refinement_oracle": measured_projection_refinement,
            "smoothing_applied": False,
            "model_pixelized": False,
        }
    )
    profile_manifest = {
        "schema_version": PROFILE_SCHEMA,
        "material_id": manifest["material_id"],
        "display_dataset_id": display_dataset_id,
        "dataset_scale": dataset_scale,
        "fit_status": fit_document.get("status"),
        "fit_parameter_replay": fit_parameter_replay,
        "evidence_level": policy["evidence_level"],
        "publication_ready": policy["publication_ready"],
        "all_rod_validation": policy["all_rod_validation"],
        "rod_scope_validation_status": "NOT_RUN",
        "computationally_valid": True,
        "model_rod_scope": policy["model_rod_scope"],
        "model_rod_count": len(profile_rods),
        "model_rod_roster_h_k_m_population": rod_roster,
        "model_rod_roster_sha256": rod_roster_sha256,
        "fit_model_rod_roster_sha256": fit_document["model_rod_roster_sha256"],
        "render_array_sha256": render_array_sha256,
        "continuous_chart_scope": (
            "complete_detector_visible_horizon_valid_branches_with_invalid_edge_bins_excluded"
        ),
        "fit_parameters_on_bounds": fit_document.get("parameters_on_bounds", []),
        "fixed_position": manifest["fixed_position"],
        "fixed_lattice": fixed_lattice_record,
        "structure_representative": structure,
        "stacking_model": manifest["stacking_model"],
        "dark_correction": manifest["dark_correction"],
        "figure_recipe": profile_recipe,
        "profile_contract": (
            "verified raw-minus-scaled-dark native-pixel-center count masses and unit-membership "
            "support are compared with unrasterized continuous phi/two-theta or signed-side Qr/L "
            "model integrals; a data-only native-center radial detector baseline conditioned on the two "
            "adjacent sidebands is subtracted from measured signal regions, and the identical "
            "fixed sideband-conditioning operator is applied to the model before comparison; no family or "
            "panel renormalization; "
            "m=0 bins lacking complete sidebands are retained only as a separately declared, "
            "display-only signal-region measurement"
            + (
                "; the profile model reuses the frozen fit parameters and dataset scales while "
                "recalculating only the declared m=0 interface convention; no optimizer or "
                "objective requalification is performed"
                if fit_parameter_replay is not None
                else ""
            )
        ),
        "display_projection_relation_to_fit": (
            "The plotted background-corrected signal uses the same detector-native affine "
            "sideband conditioner as the fit over the wider display catalog, and the plotted "
            "model has the same anchor projection; anchor rows do not enter either result."
        ),
        "m0_signal_only_display": {
            "status": (
                "DISPLAY_ONLY_INCOMPLETE_SIDEBAND_SUPPLEMENT"
                if m0_signal_only_row_count
                else "NOT_REQUIRED_ALL_M0_BINS_CONDITIONED"
            ),
            "supplemental_bin_count": m0_signal_only_row_count,
            "fit_role": "not_used_in_fit_objective_or_parameter_estimation",
            "selection_rule": (
                "render only valid m=0 bins without a valid two-sideband-conditioned comparison"
            ),
            "data_measure": (
                "raw-minus-scaled-dark signal-region count mass per continuous detector area"
            ),
            "model_measure": (
                "dataset-scaled unified m=0 model plus fixed radial-background count mass per "
                "continuous detector area"
            ),
            "background_conditioning": "fixed_radial_only_no_adjacent_sideband_conditioning",
            "radial_background_calibration_radius_px": [
                float(background_manifest["sampling"]["minimum_radius_px"]),
                float(background_manifest["sampling"]["maximum_radius_px"]),
            ],
            "inside_calibration_radius_policy": (
                "radial background is extrapolated; direct-beam, beamstop, and air-scatter "
                "contributions are not separately identified"
            ),
            "native_pixel_overlap_covariance_retained": True,
            "projection_method": (
                NATIVE_PIXEL_CENTER_METHOD
                if native_m0_signal_only_plans
                else (MEASURED_PROJECTION_METHOD if m0_signal_only_projection is not None else None)
            ),
            "projection_revision": (
                native_m0_signal_only_plans[0].projection.projection_revision
                if native_m0_signal_only_plans
                else (
                    None
                    if m0_signal_only_projection is None
                    else m0_signal_only_projection.projection_revision
                )
            ),
            "measured_quadrature_revision": (
                None
                if native_observations or m0_signal_only_plan is None
                else m0_signal_only_plan.quadrature.quadrature_revision
            ),
            "model_quadrature_revision": (
                None
                if m0_signal_only_plan is None
                else m0_signal_only_plan.quadrature.quadrature_revision
            ),
            "measured_phi_subdivision_count": int(
                recipe["model_cubature"]["fold_oracle_subdivisions"]
            )
            if not native_observations
            else None,
            "model_phi_subdivision_count": int(
                recipe["model_cubature"]["fold_oracle_subdivisions"]
            ),
            "gauss_order": int(recipe["model_cubature"]["oracle_gauss_order"]),
            "subdivision_count": int(recipe["model_cubature"]["fold_oracle_subdivisions"]),
            "measured_continuous_node_count": int(
                0
                if native_observations or m0_signal_only_plan is None
                else m0_signal_only_plan.quadrature.column_px.size
            ),
            "model_continuous_node_count": int(
                0 if m0_signal_only_plan is None else m0_signal_only_plan.quadrature.column_px.size
            ),
            "count_mass_sha256": _array_sha256(m0_signal_only_count_mass),
            "count_covariance_sha256": _array_sha256(m0_signal_only_count_covariance),
            "radial_background_mass_sha256": _array_sha256(m0_signal_only_background_mass),
            "model_mass_sha256": _array_sha256(m0_signal_only_model_mass),
            "detector_overlay_flat_pixel_count": int(m0_signal_only_flat_pixel.size),
            "detector_overlay_flat_pixel_sha256": _array_sha256(m0_signal_only_flat_pixel),
            "configured_display_rows": configured_display_rows,
            "actual_display_rows": display_rows,
            "execution": m0_signal_only_execution,
            "smoothing_applied": False,
            "model_pixelized": False,
        },
        "background_model": {
            "status": background_manifest["status"],
            "artifact_sha256": background_identity["sha256"],
            "state_revision": background_state.revision,
            "parameter_vector_sha256": _array_sha256(background_state.parameter_vector),
            "parameter_covariance_sha256": _array_sha256(background_state.parameter_covariance),
            "radial_mass_sha256": _array_sha256(radial_profile_background.count_mass),
            "radial_covariance_sha256": _array_sha256(radial_profile_background.covariance_count2),
            "excluded_flat_pixel_count_by_dataset": background_manifest["sampling"][
                "excluded_flat_pixel_count_by_dataset"
            ],
            "excluded_flat_pixel_sha256_by_dataset": background_manifest["sampling"][
                "excluded_flat_pixel_sha256_by_dataset"
            ],
            "conditioned_revision": conditioned_profile_background.revision,
            "conditioned_mass_sha256": _array_sha256(conditioned_profile_background_mass),
            "conditioned_covariance_sha256": _array_sha256(
                conditioned_profile_background_covariance
            ),
            "anchor_projection_sha256": _array_sha256(
                conditioned_profile_background.anchor_projection
            ),
            "residual_conditioning": "adjacent-affine-two-anchor-model-projected.v2",
            "parameter_names": background_manifest["parameter_names"],
            "parameter_vector": background_state.parameter_vector.tolist(),
            "metrics": background_manifest["metrics"],
        },
        "m0_display_coordinate": "two_theta bin center in degrees",
        "axial_reciprocal_magnitude_Ainv": axial_reciprocal_magnitude_Ainv,
        "continuous_support_exclusions": support_exclusions,
        "profile_cubature": {
            "settings": profile_cubature,
            "m0_model_phi_subdivision_count": int(profile_cubature["fold_fit_subdivisions"]),
            "offspecular_axial_refinement": offspecular_axial_refinement,
            "offspecular_radial_transform": offspecular_radial_transform,
            "offspecular_signal_minimum_radial_nodes_per_side": (
                offspecular_signal_minimum_radial_nodes_per_side
            ),
            "relative_l2_by_profile": convergence,
            "evaluated_passes": list(policy["pass_names"]),
            "full_profile_oracle_performed": False,
            "fit_artifact_fitted_region_oracle": fit_document["cubature_oracle"],
            "fit_artifact_data_projection": fit_document["data_projection"],
            "execution": executions,
        },
        "measured_data_projection": measured_data_record,
        "model_measure": "continuous_detector_chart_area",
        "full_selected_native_pixel_count": int(np.unique(selected_pixel).size),
        "full_selected_pixel_region_pair_count": int(selected_pixel.size),
        "model_pixelized": False,
        "smoothing_applied": False,
        "provenance": {
            "fit_diagnostic": diagnostic_identity["path"],
            "fit_diagnostic_sha256": diagnostic_identity["sha256"],
            "background": background_identity["path"],
            "background_sha256": background_identity["sha256"],
            "fit": fit_identity["path"],
            "fit_sha256": fit_identity["sha256"],
            "recipe": recipe_identity["path"],
            "recipe_sha256": recipe_identity["sha256"],
            "fit_plan": fit_plan_identity,
            "implementation": implementation_identity,
            "fit_origin_adapter_sha256": recorded_adapter_sha256,
            "fit_origin_implementation_sha256": recorded_implementation_sha256,
            "fit_chain": [identity for _, identity in fit_chain],
            "model_inputs": model_input_identities,
            "completed_progress": completed_progress_identity,
            "osc": osc_identity["path"],
            "osc_sha256": osc_identity["sha256"],
            "dark_osc": dark_identity["path"],
            "dark_osc_sha256": dark_identity["sha256"],
            "profile_adapter": adapter_identity["path"],
            "profile_adapter_sha256": adapter_identity["sha256"],
            "execution_identity": execution_identity,
            "execution_devices": progress_document["devices"],
            "evaluated_backends": progress_document["evaluated_backends"],
        },
    }
    verify_inputs()
    result = write_diagnostic(
        destination,
        arrays=arrays,
        manifest=profile_manifest,
        repository_root=ROOT,
    )
    try:
        verify_inputs()
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    progress_path.unlink(missing_ok=True)
    return result


def _full_peak_alignment_records(
    arrays: dict[str, np.ndarray],
    recipe: dict[str, Any],
    *,
    measured_signal_density: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    identities = arrays["profile_identity"]
    coordinate = arrays["profile_selection_coordinate"]
    measured = (
        arrays["profile_measured_signal_density"]
        if measured_signal_density is None
        else np.asarray(measured_signal_density, dtype=np.float64)
    )
    if measured.shape != arrays["profile_measured_signal_density"].shape:
        raise ValueError("alignment signal density must match the profile rows")
    modeled = arrays["profile_model_signal_density"]
    valid = arrays["profile_valid"]
    display_dataset_id = str(recipe["display_dataset_id"])
    for peak in recipe["fit_peak"]:
        if str(peak["dataset_id"]) != display_dataset_id:
            continue
        family = int(peak["family_m"])
        profile_identities = ("m0",) if family == 0 else (f"m{family}_plus", f"m{family}_minus")
        centers = peak.get("centers", (peak.get("center"),))
        for identity in profile_identities:
            for center in centers:
                center_value = float(center)
                half_width = float(peak["half_width"])
                selected = (
                    (identities == identity)
                    & valid
                    & (np.abs(coordinate - center_value) <= half_width + 1.0e-12)
                )
                if np.count_nonzero(selected) < 3:
                    continue
                x = coordinate[selected]
                data_weight = np.maximum(measured[selected], 0.0)
                model_weight = np.maximum(modeled[selected], 0.0)
                data_total = float(np.sum(data_weight))
                model_total = float(np.sum(model_weight))
                positive_totals = data_total > 0.0 and model_total > 0.0
                data_centroid = (
                    float(np.sum(x * data_weight) / data_total) if data_total > 0.0 else None
                )
                model_centroid = (
                    float(np.sum(x * model_weight) / model_total) if model_total > 0.0 else None
                )
                order = np.argsort(x)
                edge = order[[0, -1]]
                data_edge_fraction = float(
                    np.sum(data_weight[edge]) / data_total if data_total > 0.0 else 1.0
                )
                model_edge_fraction = float(
                    np.sum(model_weight[edge]) / model_total if model_total > 0.0 else 1.0
                )
                records.append(
                    {
                        "peak_identity": peak["identity"],
                        "profile_identity": identity,
                        "coordinate_kind": peak["coordinate_kind"],
                        "nominal_center": center_value,
                        "measured_centroid": data_centroid,
                        "model_centroid": model_centroid,
                        "model_minus_measured": (
                            model_centroid - data_centroid
                            if data_centroid is not None and model_centroid is not None
                            else None
                        ),
                        "measured_edge_mass_fraction": data_edge_fraction,
                        "model_edge_mass_fraction": model_edge_fraction,
                        "centroid_reliable": positive_totals
                        and max(data_edge_fraction, model_edge_fraction) < 0.10,
                    }
                )
    return records


def _pixel_cell_boundary_segments(mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(mask, dtype=np.bool_)
    if selected.ndim != 2:
        raise ValueError("native-pixel mask must be two-dimensional")
    row_count, column_count = selected.shape
    horizontal = np.zeros((row_count + 1, column_count), dtype=np.bool_)
    vertical = np.zeros((row_count, column_count + 1), dtype=np.bool_)
    if row_count and column_count:
        horizontal[0] = selected[0]
        horizontal[-1] = selected[-1]
        horizontal[1:-1] = selected[:-1] != selected[1:]
        vertical[:, 0] = selected[:, 0]
        vertical[:, -1] = selected[:, -1]
        vertical[:, 1:-1] = selected[:, :-1] != selected[:, 1:]
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        padded = np.pad(values, (1, 1), constant_values=False)
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        return transitions[::2], transitions[1::2]

    for edge_row, values in enumerate(horizontal):
        starts, stops = runs(values)
        y = float(edge_row) - 0.5
        segments.extend(
            ((float(start) - 0.5, y), (float(stop) - 0.5, y))
            for start, stop in zip(starts, stops, strict=True)
        )
    for edge_column, values in enumerate(vertical.T):
        starts, stops = runs(values)
        x = float(edge_column) - 0.5
        segments.extend(
            ((x, float(start) - 0.5), (x, float(stop) - 0.5))
            for start, stop in zip(starts, stops, strict=True)
        )
    if not segments:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.asarray(segments, dtype=np.float64)


def _filled_region_row_spans(
    mask: np.ndarray,
    *,
    split_column: int | None,
) -> np.ndarray:
    """Fill display-only gaps between sampled pixels inside each ROI branch."""

    selected = np.asarray(mask, dtype=np.bool_)
    if selected.ndim != 2:
        raise ValueError("region display mask must be two-dimensional")
    column_count = selected.shape[1]
    if split_column is None:
        column_ranges = ((0, column_count),)
    else:
        if not 0 < split_column < column_count:
            raise ValueError("region display split column lies outside the detector")
        column_ranges = ((0, split_column), (split_column, column_count))
    filled = np.zeros_like(selected)
    for start, stop in column_ranges:
        branch = selected[:, start:stop]
        for row in np.flatnonzero(np.any(branch, axis=1)):
            columns = np.flatnonzero(branch[row])
            filled[row, start + columns[0] : start + columns[-1] + 1] = True
    return filled


def render(
    *,
    profile_diagnostic_path: Path,
    output_directory: Path,
) -> tuple[Path, Path, Path, Path]:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    renderer_identity = _file_identity(Path(__file__))
    profile_identity = _file_identity(profile_diagnostic_path)
    arrays, manifest = _load_profiles(profile_diagnostic_path)
    displayed_measured = np.asarray(arrays["profile_measured_signal_density"])
    residual_sideband_density = (
        np.asarray(arrays["profile_measured_radial_signal_density"]) - displayed_measured
    )
    recipe = _validated_recipe(dict(manifest["figure_recipe"]))
    material_id = str(manifest["material_id"])
    model_label = (
        "Frozen-parameter model"
        if manifest.get("fit_parameter_replay") is not None
        else "Fitted model"
    )
    material_slug = "".join(
        character.lower() if character.isalnum() else "_" for character in material_id
    ).strip("_")
    if not material_slug:
        raise ValueError("profile material_id cannot form an output name")
    material_label = material_id.replace("2", "$_2$").replace("3", "$_3$")
    display_dataset_index = tuple(recipe["dataset_ids"]).index(manifest["display_dataset_id"])
    display_incidence_deg = float(
        manifest["fixed_position"]["commanded_incidence_angles_deg"][display_dataset_index]
    )
    output = output_directory.resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        raise ValueError("figure output must be outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    png = output / f"{material_slug}_figure7_matched_model.png"
    pdf = output / f"{material_slug}_figure7_matched_model.pdf"
    alignment_path = output / f"{material_slug}_peak_alignment.json"
    output_manifest_path = output / f"{material_slug}_figure7_matched_model.json"
    temporary_png = output / f"{material_slug}_figure7_matched_model.partial.png"
    temporary_pdf = output / f"{material_slug}_figure7_matched_model.partial.pdf"
    temporary_alignment = output / f"{material_slug}_peak_alignment.partial.json"
    temporary_manifest = output / f"{material_slug}_figure7_matched_model.partial.json"
    for temporary in (
        temporary_png,
        temporary_pdf,
        temporary_alignment,
        temporary_manifest,
    ):
        temporary.unlink(missing_ok=True)
    counts = arrays["display_detector_counts"]
    full_region_code = np.array(arrays["display_full_region_code"], copy=True)
    valid_profile_keys = {
        (str(identity), int(bin_value))
        for identity, bin_value, is_valid in zip(
            arrays["profile_identity"],
            arrays["profile_bin_index"],
            arrays["profile_valid"],
            strict=True,
        )
        if is_valid
    }
    retained_signal_blocks = set()
    for group, band, bin_value in zip(
        arrays["row_group"], arrays["row_band"], arrays["row_bin_index"], strict=True
    ):
        if str(band).startswith("background_"):
            continue
        profile_key = "m0" if str(group) == "m0" else str(band)
        if (profile_key, int(bin_value)) in valid_profile_keys:
            retained_signal_blocks.add((str(group), int(bin_value)))
    unused_anchor_rows = {
        row_index
        for row_index, (group, band, bin_value) in enumerate(
            zip(arrays["row_group"], arrays["row_band"], arrays["row_bin_index"], strict=True)
        )
        if str(band).startswith("background_")
        and (str(group), int(bin_value)) not in retained_signal_blocks
    }
    if unused_anchor_rows:
        selected_row = arrays["selected_observation_row"]
        selected_pixel = arrays["selected_flat_pixel_index"]
        unused_anchor_pixel = selected_pixel[
            np.isin(selected_row, np.asarray(sorted(unused_anchor_rows), dtype=np.int64))
        ]
        unused_anchor_pixel = unused_anchor_pixel[unused_anchor_pixel < full_region_code.size]
        full_region_code.reshape(-1)[unused_anchor_pixel] = 0
    positive = counts[counts > 0]
    row_count, column_count = counts.shape
    detector_extent = (-0.5, column_count - 0.5, row_count - 0.5, -0.5)
    figure = plt.figure(figsize=(12.0, 16.0), constrained_layout=True)
    outer = figure.add_gridspec(2, 1, height_ratios=(0.42, 1.0))
    detector_axis = figure.add_subplot(outer[0, 0])
    detector_axis.imshow(
        counts,
        origin="upper",
        cmap="magma",
        vmin=float(np.percentile(positive, recipe["detector"]["low_percentile"])),
        vmax=float(np.percentile(positive, recipe["detector"]["high_percentile"])),
        interpolation="none",
        aspect="equal",
        extent=detector_extent,
    )
    colors = {1: "#56B4E9", 3: "#0072B2", 5: "#E69F00", 6: "#009E73"}
    region_split_column = math.ceil(
        float(manifest["fixed_position"]["beam_center_column_row_px"][0])
    )
    full_display_masks = {
        code: _filled_region_row_spans(
            full_region_code == code,
            split_column=None if code == 1 else region_split_column,
        )
        for code in colors
    }
    full_overlay = np.zeros((*counts.shape, 4), dtype=np.uint8)
    for code, color in colors.items():
        full_mask = full_display_masks[code]
        if np.any(full_mask):
            rgba = to_rgba(color, alpha=0.13)
            full_overlay[full_mask] = np.rint(255.0 * np.asarray(rgba)).astype(np.uint8)

    detector_axis.imshow(
        full_overlay,
        origin="upper",
        interpolation="none",
        aspect="equal",
        extent=detector_extent,
    )

    def add_exact_boundary(
        mask: np.ndarray,
        *,
        color: str,
        linestyle: str,
        linewidth: float,
        zorder: float,
    ) -> None:
        segments = _pixel_cell_boundary_segments(mask)
        if segments.size:
            detector_axis.add_collection(
                LineCollection(
                    segments,
                    colors=(color,),
                    linestyles=(linestyle,),
                    linewidths=(linewidth,),
                    capstyle="butt",
                    zorder=zorder,
                ),
                autolim=False,
            )

    for code, color in colors.items():
        full_mask = full_display_masks[code]
        if np.any(full_mask):
            add_exact_boundary(
                full_mask,
                color=color,
                linestyle="dashed",
                linewidth=0.45,
                zorder=3.0,
            )
    for label in recipe.get("label", []):
        detector_axis.text(
            float(label["column_px"]),
            float(label["row_px"]),
            str(label["text"]),
            color="white",
            fontsize=8.6,
            ha="center",
            va="center",
        )
    detector_axis.set(
        title=f"{material_label} {display_incidence_deg:g}° OSC: integration regions",
        xlabel="detector column (px)",
        ylabel="detector row (px, top-origin)",
    )
    detector_axis.set_xlim(-0.5, column_count - 0.5)
    detector_axis.set_ylim(row_count - 0.5, -0.5)
    detector_axis.legend(
        handles=(
            Line2D(
                (),
                (),
                color="white",
                linestyle="dashed",
                linewidth=0.7,
                label="integration ROI",
            ),
        ),
        facecolor="black",
        framealpha=0.65,
        labelcolor="white",
        loc="lower right",
    )
    profile_grid = outer[1, 0].subgridspec(4, 2, hspace=0.08, wspace=0.08)
    identity_rows = (
        ("m1_minus", "m1_plus"),
        ("m3_minus", "m3_plus"),
        ("m4_minus", "m4_plus"),
    )
    axes: dict[str, Any] = {}
    for row_index, pair in enumerate(identity_rows):
        for column_index, identity in enumerate(pair):
            axes[identity] = figure.add_subplot(profile_grid[row_index, column_index])
    axes["m0"] = figure.add_subplot(profile_grid[3, :])
    identities = arrays["profile_identity"]
    valid = arrays["profile_valid"]
    selection_coordinate = arrays["profile_selection_coordinate"]
    bin_index = arrays["profile_bin_index"]
    m0_signal_only_selected = _m0_signal_only_display_mask(
        arrays["m0_signal_only_bin_index"],
        arrays["m0_signal_only_valid"],
        identities,
        bin_index,
        valid,
    )

    def plot_contiguous_profile(
        axis: Any,
        x: np.ndarray,
        observed: np.ndarray,
        modeled: np.ndarray,
        bins: np.ndarray,
        *,
        logarithmic: bool,
        observed_label: str,
        modeled_label: str,
    ) -> None:
        order = np.argsort(x)
        x = x[order]
        observed = observed[order]
        modeled = modeled[order]
        bins = bins[order]
        breaks = np.flatnonzero(np.diff(bins) != 1) + 1
        for segment_index, segment in enumerate(np.split(np.arange(x.size), breaks)):
            if not segment.size:
                continue
            observed_line = np.array(observed[segment], copy=True)
            modeled_line = np.array(modeled[segment], copy=True)
            observed_selected = np.isfinite(observed_line)
            modeled_selected = np.isfinite(modeled_line)
            if logarithmic:
                observed_selected &= observed_line > 0.0
                modeled_selected &= modeled_line > 0.0
            observed_line[~observed_selected] = np.nan
            modeled_line[~modeled_selected] = np.nan
            axis.plot(
                x[segment],
                observed_line,
                color="black",
                linewidth=1.0,
                label=observed_label if segment_index == 0 else None,
            )
            axis.plot(
                x[segment],
                modeled_line,
                color="#D55E00",
                linestyle="--",
                linewidth=1.3,
                label=modeled_label if segment_index == 0 else None,
            )

    def plot_m0_signal_only_supplement(axis: Any) -> None:
        x = arrays["m0_signal_only_two_theta_deg"][m0_signal_only_selected]
        observed = arrays["m0_signal_only_measured_density"][m0_signal_only_selected]
        modeled = arrays["m0_signal_only_model_plus_background_density"][m0_signal_only_selected]
        bins = arrays["m0_signal_only_bin_index"][m0_signal_only_selected]
        order = np.argsort(x)
        x = x[order]
        observed = observed[order]
        modeled = modeled[order]
        bins = bins[order]
        breaks = np.flatnonzero(np.diff(bins) != 1) + 1
        for segment_index, segment in enumerate(np.split(np.arange(x.size), breaks)):
            if not segment.size:
                continue
            axis.plot(
                x[segment],
                observed[segment],
                color="#666666",
                linestyle=":",
                marker="o",
                markersize=2.8,
                linewidth=0.9,
                label=("Signal-only data (no sidebands)" if segment_index == 0 else None),
            )
            axis.plot(
                x[segment],
                modeled[segment],
                color="#D55E00",
                linestyle=":",
                marker="x",
                markersize=3.0,
                linewidth=1.0,
                label=(f"{model_label} + extrapolated radial BG" if segment_index == 0 else None),
            )

    for identity, axis in axes.items():
        selected = (identities == identity) & valid
        x = selection_coordinate[selected]
        observed = displayed_measured[selected]
        modeled = arrays["profile_model_signal_density"][selected]
        bins = bin_index[selected]
        if identity == "m0":
            plot_contiguous_profile(
                axis,
                x,
                observed,
                modeled,
                bins,
                logarithmic=False,
                observed_label="Sideband-conditioned data",
                modeled_label=model_label,
            )
            plot_m0_signal_only_supplement(axis)
            axis.set_yscale("symlog", linthresh=1.0)
            suffix = "; signal-only gaps marked" if np.any(m0_signal_only_selected) else ""
            axis.set_title(rf"m = 0 (integrated in $\phi$-$2\theta$${suffix})")
        else:
            plot_contiguous_profile(
                axis,
                x,
                observed,
                modeled,
                bins,
                logarithmic=False,
                observed_label="Background-subtracted data",
                modeled_label=model_label,
            )
            family, side = identity.split("_", 1)
            side_label = {"minus": "-", "plus": "+"}[side]
            axis.set_title(f"{family.replace('m', 'm = ')} {side_label}")
        if x.size:
            half_step = (
                0.5 * float(np.median(np.diff(np.unique(np.sort(x))))) if x.size > 1 else 0.0
            )
            lower_bound = 0.0 if identity == "m0" else float(np.min(x) - half_step)
            axis.set_xlim(lower_bound, float(np.max(x) + half_step))
        if identity == "m0":
            axis.set_ylabel("counts / pixel")
        elif identity.endswith("_minus"):
            axis.set_ylabel("background-subtracted counts / pixel")
        axis.grid(color="#D9D9D9", linewidth=0.5, alpha=0.7)
    for pair in identity_rows:
        pair_selected = np.isin(identities, pair) & valid
        pair_values = np.concatenate(
            (
                displayed_measured[pair_selected],
                arrays["profile_model_signal_density"][pair_selected],
            )
        )
        pair_values = pair_values[np.isfinite(pair_values)]
        if pair_values.size:
            lower = min(0.0, float(np.min(pair_values)))
            upper = float(np.max(pair_values))
            limits = (lower, upper * 1.05 if upper > 0.0 else 1.0)
            pair_coordinate = selection_coordinate[np.isin(identities, pair) & valid]
            pair_coordinate = pair_coordinate[np.isfinite(pair_coordinate)]
            pair_xlim = None
            if pair_coordinate.size:
                unique_coordinate = np.unique(np.sort(pair_coordinate))
                half_step = (
                    0.5 * float(np.median(np.diff(unique_coordinate)))
                    if unique_coordinate.size > 1
                    else 0.0
                )
                pair_xlim = (
                    float(np.min(pair_coordinate) - half_step),
                    float(np.max(pair_coordinate) + half_step),
                )
            for identity in pair:
                axes[identity].set_ylim(*limits)
                if pair_xlim is not None:
                    axes[identity].set_xlim(*pair_xlim)
    for identity in ("m4_plus", "m4_minus"):
        axes[identity].set_xlabel(r"$L$")
    axes["m0"].set_xlabel(r"$2\theta$ (deg)")
    axes["m1_minus"].legend(
        handles=(
            Line2D((), (), color="black", label="Background-subtracted data"),
            Line2D((), (), color="#D55E00", linestyle="--", label=model_label),
        ),
        frameon=False,
        loc="upper right",
    )
    axes["m0"].legend(frameon=False, loc="upper right", fontsize=7, ncol=2)
    fixed = manifest["fixed_position"]
    structure = manifest["structure_representative"]
    figure.suptitle(
        f"{manifest['evidence_level']} — fault-free configured ordered parent; "
        f"shared Δθᵢ={math.degrees(float(fixed['incidence_angle_delta_rad'])):+.4f}°; "
        f"outer-chalcogen vacancy v={structure['outer_chalcogen_vacancy_fraction']:.4f}",
        fontsize=12,
    )
    try:
        figure.savefig(temporary_png, dpi=220, metadata={"Software": "rasim_next"})
        figure.savefig(
            temporary_pdf,
            metadata={
                "Creator": "rasim_next",
                "Producer": "rasim_next",
                "CreationDate": None,
                "ModDate": None,
            },
        )
    finally:
        plt.close(figure)
    alignment = _full_peak_alignment_records(
        arrays,
        recipe,
        measured_signal_density=displayed_measured,
    )
    alignment_document = {
        "schema_version": "rasim-layered-quintuple-peak-alignment-v3",
        "profile_diagnostic_sha256": profile_identity["sha256"],
        "renderer_sha256": renderer_identity["sha256"],
        "evidence_level": manifest["evidence_level"],
        "interpretation": (
            "m=0 offsets are evaluated in two_theta and m!=0 offsets in L. Reliable "
            "centroids screen peak-position alignment; intensity mismatch at aligned "
            "centers instead implicates structure, optics, mosaic tails, or background."
        ),
        "records": alignment,
    }
    _write_json_atomic(temporary_alignment, alignment_document)
    output_document = {
        "schema_version": "rasim-layered-quintuple-matched-figure-output-v1",
        "profile_diagnostic": profile_identity,
        "renderer": renderer_identity,
        "fit_status": manifest.get("fit_status"),
        "fit_parameter_replay": manifest.get("fit_parameter_replay"),
        "profile_specular_stitch": recipe.get("parratt_stitch"),
        "computationally_valid": manifest.get("computationally_valid"),
        "evidence_level": manifest.get("evidence_level"),
        "publication_ready": bool(manifest.get("publication_ready")),
        "all_rod_validation": manifest.get("all_rod_validation"),
        "model_rod_scope": manifest.get("model_rod_scope"),
        "model_pixelized": manifest.get("model_pixelized"),
        "smoothing_applied": manifest.get("smoothing_applied"),
        "display_background_subtraction": {
            "mode": (
                "conditioned-primary-plus-declared-m0-signal-only-gaps"
                if np.any(m0_signal_only_selected)
                else "conditioned-primary"
            ),
            "radial_background_from_profile_fit": True,
            "residual_sideband_subtracted": True,
            "residual_sideband_subtraction_scope": "primary conditioned profiles only",
            "residual_sideband_model": (
                "per-bin affine density in phi for m=0 and Qr for m!=0, determined only "
                "from the two adjacent detector sidebands by the same core conditioner used "
                "in the fit"
            ),
            "diffraction_model_used_for_subtraction": False,
            "diffraction_model_sideband_conditioned_for_comparison": True,
            "smoothing_applied": False,
            "removed_density_range_count_per_px": [
                float(np.nanmin(residual_sideband_density)),
                float(np.nanmax(residual_sideband_density)),
            ],
            "m0_signal_only_supplement": {
                "fit_role": "display only",
                "bin_index": arrays["m0_signal_only_bin_index"][m0_signal_only_selected].tolist(),
                "two_theta_deg": arrays["m0_signal_only_two_theta_deg"][
                    m0_signal_only_selected
                ].tolist(),
                "data_measure": "raw-minus-scaled-dark signal-region counts per pixel",
                "model_measure": "unified physical model plus fixed radial background per pixel",
                "adjacent_sideband_conditioning": False,
                "radial_background_extrapolated_inside_radius_px": manifest[
                    "m0_signal_only_display"
                ]["radial_background_calibration_radius_px"][0],
            },
        },
        "outputs": {
            "png": {"name": png.name, "sha256": _sha256(temporary_png)},
            "pdf": {"name": pdf.name, "sha256": _sha256(temporary_pdf)},
            "peak_alignment": {
                "name": alignment_path.name,
                "sha256": _sha256(temporary_alignment),
            },
        },
    }
    _write_json_atomic(temporary_manifest, output_document)
    _require_unchanged(renderer_identity, role="renderer")
    _require_unchanged(profile_identity, role="profile diagnostic")
    temporary_png.replace(png)
    temporary_pdf.replace(pdf)
    temporary_alignment.replace(alignment_path)
    for name, path in (
        ("png", png),
        ("pdf", pdf),
        ("peak_alignment", alignment_path),
    ):
        if _sha256(path) != output_document["outputs"][name]["sha256"]:
            raise RuntimeError(f"promoted {name} hash changed")
    temporary_manifest.replace(output_manifest_path)
    return png, pdf, alignment_path, output_manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--fixed-state", type=Path, required=True)
    prepare_parser.add_argument("--recipe", type=Path, required=True)
    prepare_parser.add_argument("--destination", type=Path, required=True)
    prepare_parser.add_argument("--source-state-count", type=int, default=250)
    prepare_parser.add_argument("--row-chunk-size", type=int, default=32)
    background_parser = subparsers.add_parser("background")
    background_parser.add_argument("--diagnostic", type=Path, required=True)
    background_parser.add_argument("--fit-plan", type=Path, required=True)
    background_parser.add_argument("--destination", type=Path, required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--diagnostic", type=Path, required=True)
    fit_parser.add_argument("--background", type=Path, required=True)
    fit_parser.add_argument("--destination", type=Path, required=True)
    fit_parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    fit_parser.add_argument("--maximum-function-evaluations", type=int, default=80)
    fit_parser.add_argument("--fit-plan", type=Path, required=True)
    fit_parser.add_argument("--stage", choices=("A", "B", "C", "joint"), required=True)
    fit_parser.add_argument("--predecessor", type=Path)
    fit_start = fit_parser.add_mutually_exclusive_group()
    fit_start.add_argument("--resume", type=Path)
    fit_start.add_argument("--initial-parameters", type=float, nargs="+")
    profiles_parser = subparsers.add_parser("profiles")
    profiles_parser.add_argument("--diagnostic", type=Path, required=True)
    profiles_parser.add_argument("--background", type=Path, required=True)
    profiles_parser.add_argument("--fit", type=Path, required=True)
    profiles_parser.add_argument("--destination", type=Path, required=True)
    profiles_parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    profiles_parser.add_argument("--row-chunk-size", type=int, default=32)
    profiles_parser.add_argument(
        "--specular-interface-assumption",
        choices=SPECULAR_INTERFACE_ASSUMPTIONS,
        help=(
            "replay the frozen fit parameters and scales while recalculating profiles with "
            "the selected m=0 interface assumption; no optimizer is run"
        ),
    )
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--profile-diagnostic", type=Path, required=True)
    render_parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        path = prepare(
            fixed_state_path=arguments.fixed_state,
            recipe_path=arguments.recipe,
            destination=arguments.destination,
            source_state_count=arguments.source_state_count,
            row_chunk_size=arguments.row_chunk_size,
        )
        print(path)
    elif arguments.command == "background":
        print(
            calibrate_radial_background(
                diagnostic_path=arguments.diagnostic,
                fit_plan_path=arguments.fit_plan,
                destination=arguments.destination,
            )
        )
    elif arguments.command == "fit":
        print(
            json.dumps(
                fit(
                    diagnostic_path=arguments.diagnostic,
                    background_path=arguments.background,
                    destination=arguments.destination,
                    execution_backend=arguments.backend,
                    maximum_function_evaluations=arguments.maximum_function_evaluations,
                    fit_plan_path=arguments.fit_plan,
                    stage=arguments.stage,
                    predecessor_path=arguments.predecessor,
                    resume_path=arguments.resume,
                    initial_parameters_override=arguments.initial_parameters,
                ),
                indent=2,
            )
        )
    elif arguments.command == "profiles":
        print(
            prepare_profiles(
                diagnostic_path=arguments.diagnostic,
                background_path=arguments.background,
                fit_path=arguments.fit,
                destination=arguments.destination,
                execution_backend=arguments.backend,
                row_chunk_size=arguments.row_chunk_size,
                specular_interface_assumption=arguments.specular_interface_assumption,
            )
        )
    elif arguments.command == "render":
        print(
            "\n".join(
                map(
                    str,
                    render(
                        profile_diagnostic_path=arguments.profile_diagnostic,
                        output_directory=arguments.output_directory,
                    ),
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
