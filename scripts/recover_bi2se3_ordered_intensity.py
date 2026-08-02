"""Recover Bi2Se3 occupancies and directional displacement from three fixed views."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import tomllib
import tracemalloc
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from painted_ewald import MosaicBraggSpace
from rasim_next.fitting import (
    SHARED_GEOMETRY_PARAMETER_NAMES,
    SOURCE_AVERAGED_ORDERED_INTENSITY_RESPONSE_CONTRACT_REVISION,
    SOURCE_AVERAGED_ORDERED_INTENSITY_SIGNAL_CERTIFICATE_RELATIVE_FLOOR,
    SOURCE_AVERAGED_PROFILE_SUPPORT_GATE_REVISION,
    MosaicProfileDefinition,
    MosaicProfileIdentity,
    MosaicReflectionGroupKey,
    OrderedIntensityIdentifiabilityError,
    OrderedIntensityPeakCenterObservations,
    SharedGeometryCorrections,
    apply_shared_geometry_corrections,
    compile_source_averaged_ordered_intensity_response,
    evaluate_source_averaged_ordered_intensity_point_signal,
    fit_ordered_intensity_series,
    ordered_intensity_profile_catalog_revision,
    ordered_intensity_structure_model_revision,
    source_averaged_detector_instrument_revision,
    source_averaged_profile_has_support,
)
from rasim_next.geometry import build_incident_states, detector_coordinates_to_angles
from rasim_next.io.osc import read_osc
from rasim_next.ordered import Bi2X3QuintupleLayerParameters
from rasim_next.pipeline.configured_simulation import (
    ConfiguredSimulationInputs,
    build_configured_simulation_inputs,
    build_nominal_ewald_context,
    build_source_averaged_detector,
    configured_rod_catalog_revision,
    evaluate_nominal_integer_l_markers,
    load_simulation_config,
    sample_detector_pixel_center_density,
)
from rasim_next.selection import build_osc_angle_frame

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = ROOT / "examples" / "bi2se3" / "experiment" / "ordered_intensity_fit_truth.toml"
_REQUIRED_MEASURED_PROFILE_SELECTION_SAMPLER_REVISION = (
    "detector-native-bilinear-profile-centerline-sidebands.v1"
)
_ProfileIdentityKey = tuple[str, int, int, int, int | None]


@dataclass(frozen=True, slots=True)
class _FixedPositionState:
    artifact_revision: str
    corrections: SharedGeometryCorrections
    incidence_angle_delta_rad: float
    commanded_incidence_angles_deg: tuple[float, ...]
    effective_incidence_angles_deg: tuple[float, ...]
    beam_center_column_row_px: tuple[float, float]
    geometry_parameters_fitted_here: bool


@dataclass(frozen=True, slots=True)
class PreparedMeasuredOrderedInputs:
    """Validated fixed inputs for an independently runnable measured ordered fit."""

    series: tuple[ConfiguredSimulationInputs, ...]
    profile_catalogs: tuple[tuple[object, tuple[MosaicProfileDefinition, ...]], ...]
    anchor_counts: tuple[dict[str, int | str], ...]
    mosaic_parameters: dict[str, float]
    source_revision: str
    cif_sha256: str
    fixed_position_record: dict[str, object]
    baseline_parameters: Bi2X3QuintupleLayerParameters


def _fixed_position_state(
    fixed_geometry: object,
    mosaic_case: dict[str, Any],
) -> _FixedPositionState:
    if not isinstance(fixed_geometry, dict):
        raise ValueError("mosaic result lacks its fixed position state")
    if set(fixed_geometry) != {
        "position_artifact_revision",
        "corrections",
        "incidence_angle_model_id",
        "incidence_angle_delta_rad",
        "commanded_incidence_angles_deg",
        "effective_incidence_angles_deg",
        "beam_center_column_row_px",
        "geometry_parameters_fitted_here",
    }:
        raise ValueError("mosaic result has an invalid fixed position record")
    revision = fixed_geometry.get("position_artifact_revision")
    if (
        not isinstance(revision, str)
        or not revision.startswith("sha256-")
        or len(revision) != 71
        or any(character not in "0123456789abcdef" for character in revision[7:])
    ):
        raise ValueError("mosaic result lacks its upstream position-artifact revision")
    if fixed_geometry.get("incidence_angle_model_id") != ("commanded_angle_plus_common_delta.v1"):
        raise ValueError("mosaic result uses an unsupported incidence-angle model")
    correction_record = fixed_geometry.get("corrections")
    if not isinstance(correction_record, dict) or set(correction_record) != set(
        SHARED_GEOMETRY_PARAMETER_NAMES
    ):
        raise ValueError("mosaic result has an invalid fixed position correction record")
    corrections = SharedGeometryCorrections.from_array(
        [float(correction_record[name]) for name in SHARED_GEOMETRY_PARAMETER_NAMES]
    )
    delta_rad = float(fixed_geometry.get("incidence_angle_delta_rad"))
    if not math.isfinite(delta_rad):
        raise ValueError("mosaic result has a nonfinite common incidence-angle delta")
    commanded = tuple(
        float(value) for value in fixed_geometry.get("commanded_incidence_angles_deg", ())
    )
    effective = tuple(
        float(value) for value in fixed_geometry.get("effective_incidence_angles_deg", ())
    )
    expected_commanded = tuple(float(value) for value in mosaic_case["incidence_angles_deg"])
    expected_effective = tuple(value + math.degrees(delta_rad) for value in expected_commanded)
    if (
        commanded != expected_commanded
        or len(effective) != len(expected_effective)
        or any(
            not math.isfinite(observed)
            or not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for observed, expected in zip(effective, expected_effective, strict=True)
        )
    ):
        raise ValueError("mosaic result does not apply one common delta to the incidence series")
    beam_center = tuple(float(value) for value in fixed_geometry["beam_center_column_row_px"])
    if len(beam_center) != 2 or not all(math.isfinite(value) for value in beam_center):
        raise ValueError("mosaic result has an invalid fixed detector beam center")
    fitted_here = fixed_geometry["geometry_parameters_fitted_here"]
    if fitted_here is not False:
        raise ValueError("mosaic result must keep every position parameter fixed")
    return _FixedPositionState(
        artifact_revision=revision,
        corrections=corrections,
        incidence_angle_delta_rad=delta_rad,
        commanded_incidence_angles_deg=commanded,
        effective_incidence_angles_deg=effective,
        beam_center_column_row_px=(beam_center[0], beam_center[1]),
        geometry_parameters_fitted_here=fitted_here,
    )


def _case_fixed_position_state(
    mosaic_case_path: Path,
    mosaic_case: dict[str, Any],
) -> _FixedPositionState:
    commanded = tuple(float(value) for value in mosaic_case["incidence_angles_deg"])
    config = load_simulation_config(
        (mosaic_case_path.parent / str(mosaic_case["simulation_config"])).resolve()
    )
    beam_center = tuple(
        float(value) for value in config.instrument.detector_reference_coordinate_px
    )
    return _FixedPositionState(
        artifact_revision=f"sha256-{mosaic_case['geometry_manifest_sha256']}",
        corrections=SharedGeometryCorrections.from_array(
            mosaic_case["shared_geometry_corrections"]
        ),
        incidence_angle_delta_rad=0.0,
        commanded_incidence_angles_deg=commanded,
        effective_incidence_angles_deg=commanded,
        beam_center_column_row_px=(beam_center[0], beam_center[1]),
        geometry_parameters_fitted_here=False,
    )


def _fixed_position_record(state: _FixedPositionState) -> dict[str, object]:
    return {
        "position_artifact_revision": state.artifact_revision,
        "corrections": {
            name: float(value)
            for name, value in zip(
                SHARED_GEOMETRY_PARAMETER_NAMES,
                state.corrections.as_array(),
                strict=True,
            )
        },
        "incidence_angle_model_id": "commanded_angle_plus_common_delta.v1",
        "incidence_angle_delta_rad": state.incidence_angle_delta_rad,
        "commanded_incidence_angles_deg": list(state.commanded_incidence_angles_deg),
        "effective_incidence_angles_deg": list(state.effective_incidence_angles_deg),
        "beam_center_column_row_px": list(state.beam_center_column_row_px),
        "geometry_parameters_fitted_here": state.geometry_parameters_fitted_here,
    }


def _profile_identity_key(record: dict[str, Any]) -> _ProfileIdentityKey:
    dataset_id = record.get("dataset_id")
    values = tuple(
        record.get(name)
        for name in ("family_m", "integer_L", "analytic_branch_id", "root_side_branch_id")
    )
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or any(isinstance(value, bool) for value in values)
        or not all(isinstance(value, Integral) for value in values[:3])
        or (values[3] is not None and not isinstance(values[3], Integral))
    ):
        raise ValueError("mosaic result contains an invalid profile identity")
    return (
        dataset_id,
        int(values[0]),
        int(values[1]),
        int(values[2]),
        None if values[3] is None else int(values[3]),
    )


def _definition_identity_key(definition: MosaicProfileDefinition) -> _ProfileIdentityKey:
    identity = definition.identity
    return (
        identity.dataset_id,
        int(identity.group_key.layered_family_m),
        int(identity.group_key.layered_integer_L),
        int(identity.analytic_branch_id),
        None if identity.branch_id is None else int(identity.branch_id),
    )


def _profile_identity_records(keys: frozenset[_ProfileIdentityKey] | None) -> list[dict[str, Any]]:
    if keys is None:
        return []
    ordered = sorted(keys, key=lambda key: (*key[:4], -1 if key[4] is None else key[4]))
    return [
        {
            "dataset_id": key[0],
            "family_m": key[1],
            "integer_L": key[2],
            "analytic_branch_id": key[3],
            "root_side_branch_id": key[4],
        }
        for key in ordered
    ]


def _filter_profile_definitions(
    definitions: tuple[MosaicProfileDefinition, ...],
    eligible_keys: frozenset[_ProfileIdentityKey] | None,
) -> tuple[MosaicProfileDefinition, ...]:
    if eligible_keys is None:
        return definitions
    dataset_ids = {definition.identity.dataset_id for definition in definitions}
    requested = {key for key in eligible_keys if key[0] in dataset_ids}
    by_key = {_definition_identity_key(definition): definition for definition in definitions}
    missing = requested - set(by_key)
    if missing:
        raise ValueError(
            f"mosaic fit-eligible profile is absent from the ordered catalog: {missing}"
        )
    return tuple(
        definition
        for definition in definitions
        if _definition_identity_key(definition) in requested
    )


def _validate_eligible_profile_dataset_ids(
    eligible_keys: frozenset[_ProfileIdentityKey] | None,
    dataset_ids: frozenset[str],
) -> None:
    if eligible_keys is None:
        return
    unknown = {key[0] for key in eligible_keys} - dataset_ids
    if unknown:
        raise ValueError(f"mosaic fit-eligible profile names an unknown dataset: {unknown}")


def _validate_combined_source_profile_support(
    observations: dict[str, Any],
) -> frozenset[_ProfileIdentityKey]:
    measured_policy = observations.get("measured_profile_policy")
    selection = (
        measured_policy.get("profile_selection") if isinstance(measured_policy, dict) else None
    )
    if (
        not isinstance(measured_policy, dict)
        or measured_policy.get("selection_sampler_revision")
        != _REQUIRED_MEASURED_PROFILE_SELECTION_SAMPLER_REVISION
        or measured_policy.get("source_averaged_modeled_support_gate_revision")
        != SOURCE_AVERAGED_PROFILE_SUPPORT_GATE_REVISION
        or not isinstance(selection, list)
        or not selection
        or not all(isinstance(record, dict) for record in selection)
        or observations.get("candidate_profile_count") != len(selection)
        or observations.get("fitted_profile_count")
        != sum(record.get("fit_eligible") is True for record in selection)
    ):
        raise ValueError("mosaic result lacks the current combined-source profile-support gate")

    seen: set[_ProfileIdentityKey] = set()
    eligible: set[_ProfileIdentityKey] = set()
    for record in selection:
        identity_key = _profile_identity_key(record)
        if identity_key in seen:
            raise ValueError("mosaic result contains a duplicate profile identity")
        seen.add(identity_key)
        family_m = record.get("family_m")
        modeled_signal = record.get("source_averaged_modeled_signal_A2")
        modeled_support = record.get("source_averaged_modeled_support")
        fit_eligible = record.get("fit_eligible")
        if (
            isinstance(family_m, bool)
            or not isinstance(family_m, Integral)
            or isinstance(modeled_signal, bool)
            or not isinstance(modeled_signal, Real)
            or not math.isfinite(float(modeled_signal))
            or float(modeled_signal) < 0.0
            or not isinstance(modeled_support, bool)
            or not isinstance(fit_eligible, bool)
            or modeled_support
            != source_averaged_profile_has_support(
                family_m=int(family_m),
                profile_signal_A2=float(modeled_signal),
            )
            or (fit_eligible and not modeled_support)
        ):
            raise ValueError("mosaic result contains an invalid combined-source support audit")
        if fit_eligible:
            eligible.add(identity_key)
    return frozenset(eligible)


def _response_contract_record() -> dict[str, str | float]:
    return {
        "revision": SOURCE_AVERAGED_ORDERED_INTENSITY_RESPONSE_CONTRACT_REVISION,
        "signal_certificate_relative_floor": (
            SOURCE_AVERAGED_ORDERED_INTENSITY_SIGNAL_CERTIFICATE_RELATIVE_FLOOR
        ),
    }


def _validate_ordered_result_header(document: dict[str, Any]) -> None:
    if (
        document.get("schema_version") != "rasim-bi2se3-ordered-intensity-recovery-v4"
        or document.get("accepted") is not True
        or document.get("positions_frozen") is not True
    ):
        raise ValueError("rendering requires an accepted source-averaged ordered-intensity result")
    if document.get("response_contract") != _response_contract_record():
        raise ValueError("ordered-intensity result uses a stale response compiler contract")


def _load_case(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    case = tomllib.loads(path.read_text(encoding="utf-8"))
    if case.get("schema_version") != "rasim-ordered-intensity-recovery-v1":
        raise ValueError("unsupported ordered-intensity recovery schema")
    mosaic_case_path = (path.parent / str(case["mosaic_case"])).resolve()
    mosaic_case = tomllib.loads(mosaic_case_path.read_text(encoding="utf-8"))
    if mosaic_case.get("schema_version") != "rasim-mosaic-recovery-v2":
        raise ValueError("ordered-intensity recovery requires the accepted mosaic case")
    return case, mosaic_case_path, mosaic_case


def _validated_mosaic_result(
    document: dict[str, Any],
    *,
    mosaic_case_path: Path,
    mosaic_case: dict[str, Any],
    source_sample_count: int,
) -> tuple[
    dict[str, float],
    str,
    str,
    frozenset[_ProfileIdentityKey],
    _FixedPositionState,
]:
    if (
        document.get("schema_version") != "rasim-bi2se3-real-mosaic-fit-v3"
        or document.get("status") != "MODEL_LIMITED_EFFECTIVE_RADIAL_MOSAIC_ESTIMATE"
    ):
        raise ValueError("ordered-intensity recovery requires the accepted real-OSC mosaic result")
    fixed_position = _fixed_position_state(document.get("fixed_geometry"), mosaic_case)
    provenance = document.get("provenance")
    expected_case_sha256 = hashlib.sha256(mosaic_case_path.read_bytes()).hexdigest()
    if not isinstance(provenance, dict) or provenance.get("case_sha256") != expected_case_sha256:
        raise ValueError("mosaic result does not match the immutable mosaic case")
    config_path = (mosaic_case_path.parent / str(mosaic_case["simulation_config"])).resolve()
    base_config = load_simulation_config(config_path)
    expected_beam_center = tuple(
        float(value) for value in base_config.instrument.detector_reference_coordinate_px
    )
    if fixed_position.beam_center_column_row_px != expected_beam_center:
        raise ValueError("mosaic result fixed beam center does not match its simulation config")
    first_incidence_deg = fixed_position.effective_incidence_angles_deg[0]
    expected_mosaic_config = replace(
        base_config,
        source=replace(base_config.source, sample_count=source_sample_count),
        instrument=replace(
            base_config.instrument,
            axis_rotations=tuple(
                replace(axis, angle_deg=first_incidence_deg)
                for axis in base_config.instrument.axis_rotations
            ),
        ),
    )
    if provenance.get("physics_revision") != expected_mosaic_config.physics_revision:
        raise ValueError("mosaic result does not match the configured source/material physics")
    source_model = document.get("source_model")
    if not isinstance(source_model, dict) or (
        source_model.get("sample_count") != source_sample_count
        or source_model.get("reduction")
        != "one_incoherent_weighted_detector_function_per_incidence.v1"
    ):
        raise ValueError("mosaic result does not match the requested combined source model")
    source_revision = source_model.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("mosaic result lacks a source realization revision")
    observations = document.get("observations")
    datasets = observations.get("datasets") if isinstance(observations, dict) else None
    if not isinstance(observations, dict):
        raise ValueError("mosaic result lacks the current combined-source profile-support gate")
    eligible_profile_keys = _validate_combined_source_profile_support(observations)
    observed_incidence = (
        [float(item["incidence_angle_deg"]) for item in datasets]
        if isinstance(datasets, list) and all(isinstance(item, dict) for item in datasets)
        else None
    )
    expected_incidence = [float(value) for value in mosaic_case["incidence_angles_deg"]]
    if observed_incidence != expected_incidence:
        raise ValueError("mosaic result does not match the requested incidence series")
    cif_sha256 = provenance.get("cif_sha256")
    if not isinstance(cif_sha256, str) or not cif_sha256:
        raise ValueError("mosaic result lacks CIF provenance")
    recovered = document.get("recovered_effective_distribution")
    if not isinstance(recovered, dict):
        raise ValueError("mosaic result lacks recovered_effective_distribution")
    try:
        parameters = {
            "gaussian_sigma_deg": float(recovered["gaussian_sigma_deg"]),
            "lorentzian_hwhm_deg": float(recovered["lorentzian_hwhm_deg"]),
            "lorentzian_probability": float(recovered["lorentzian_probability"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("mosaic result has invalid recovered distribution parameters") from error
    if not all(math.isfinite(value) for value in parameters.values()):
        raise ValueError("mosaic result has nonfinite recovered distribution parameters")
    return parameters, source_revision, cif_sha256, eligible_profile_keys, fixed_position


def _fixed_inputs(
    mosaic_case_path: Path,
    mosaic_case: dict[str, Any],
    *,
    source_sample_count: int,
    mosaic_parameters: dict[str, float],
    fixed_position: _FixedPositionState,
) -> tuple[ConfiguredSimulationInputs, ...]:
    if (
        isinstance(source_sample_count, bool)
        or not isinstance(source_sample_count, int)
        or source_sample_count < 1
    ):
        raise ValueError("source_sample_count must be a positive integer")
    config_path = (mosaic_case_path.parent / str(mosaic_case["simulation_config"])).resolve()
    source_config = load_simulation_config(config_path)
    corrections = fixed_position.corrections
    series: list[ConfiguredSimulationInputs] = []
    for incidence_deg in fixed_position.effective_incidence_angles_deg:
        config = replace(
            source_config,
            source=replace(
                source_config.source,
                sample_count=source_sample_count,
            ),
            mosaic=replace(
                source_config.mosaic,
                gaussian_sigma_deg=float(mosaic_parameters["gaussian_sigma_deg"]),
                lorentzian_hwhm_deg=float(mosaic_parameters["lorentzian_hwhm_deg"]),
                lorentzian_probability=float(mosaic_parameters["lorentzian_probability"]),
            ),
            instrument=replace(
                source_config.instrument,
                axis_rotations=tuple(
                    replace(axis, angle_deg=float(incidence_deg))
                    for axis in source_config.instrument.axis_rotations
                ),
            ),
        )
        inputs = build_configured_simulation_inputs(config)
        instrument = apply_shared_geometry_corrections(
            inputs.instrument,
            config.instrument.axis_rotations,
            corrections,
        )
        incident = build_incident_states(inputs.samples, inputs.material, instrument)
        if incident.states.incident_state_id.size != source_sample_count or not bool(
            np.all(incident.states.valid)
        ):
            raise RuntimeError("every ordered-intensity source state must be valid")
        bragg_config = replace(
            inputs.bragg_space.config,
            crystal_to_sample=instrument.sample_from_crystal.rotation,
        )
        series.append(
            replace(
                inputs,
                instrument=instrument,
                incident=incident,
                bragg_space=MosaicBraggSpace(bragg_config, inputs.strength),
            )
        )
    result = tuple(series)
    first_samples = result[0].samples
    for inputs in result[1:]:
        if (
            inputs.samples.source_revision != first_samples.source_revision
            or not np.array_equal(inputs.samples.origin_lab_m, first_samples.origin_lab_m)
            or not np.array_equal(inputs.samples.direction_lab, first_samples.direction_lab)
            or not np.array_equal(inputs.samples.wavelength_A, first_samples.wavelength_A)
            or not np.array_equal(inputs.samples.source_weight, first_samples.source_weight)
        ):
            raise RuntimeError("incidence views do not share one deterministic source realization")
    return result


def _profile_definitions(
    inputs: ConfiguredSimulationInputs,
    *,
    incidence_deg: float,
    m0_observation: dict[str, Any],
    profile_config: dict[str, Any],
    two_theta_gauss_order: int,
    phi_gauss_order: int,
    eligible_profile_keys: frozenset[_ProfileIdentityKey] | None = None,
) -> tuple[object, tuple[MosaicProfileDefinition, ...]]:
    effective_incidence_deg = float(inputs.config.instrument.axis_rotations[0].angle_deg)
    frame_revision = (
        f"ordered-intensity-fixed-nine-{incidence_deg:g}deg.v1"
        if effective_incidence_deg == incidence_deg
        else (
            "ordered-intensity-fixed-position-"
            f"commanded-{incidence_deg:g}deg-effective-{effective_incidence_deg:.12g}deg.v2"
        )
    )
    context = build_nominal_ewald_context(inputs)
    frame = build_osc_angle_frame(
        mean_direction_lab=inputs.config.source.mean_direction_lab,
        instrument=inputs.instrument,
        sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
        revision=frame_revision,
    )
    markers = evaluate_nominal_integer_l_markers(context)
    selected_marker = np.flatnonzero((markers.family_m > 0) & (markers.root_sign != 0))
    marker_angles = detector_coordinates_to_angles(
        markers.column_px[selected_marker],
        markers.row_px[selected_marker],
        instrument=inputs.instrument,
        angle_frame=frame,
    )
    if not np.all(marker_angles.valid & marker_angles.azimuth_valid):
        raise RuntimeError("a detector-visible integer-L marker has invalid angles")
    dataset_id = str(m0_observation["dataset_id"])
    if eligible_profile_keys is None:
        supported_key = f"supported_m0_integer_L_{incidence_deg:g}deg"
        supported_m0 = {int(value) for value in profile_config[supported_key]}
        observed_m0 = tuple(
            item
            for item in m0_observation["observed_peaks"]
            if int(item["integer_L"]) in supported_m0
        )
        if {int(item["integer_L"]) for item in observed_m0} != supported_m0:
            raise RuntimeError("the admitted m=0 set does not match the frozen OSC evidence")
    else:
        candidate_by_integer_l = {
            int(item["integer_L"]): item
            for item in (
                *m0_observation["observed_peaks"],
                *m0_observation["unsupported_observed_peaks"],
            )
        }
        requested_m0 = {
            key[2] for key in eligible_profile_keys if key[0] == dataset_id and key[1] == 0
        }
        missing_m0 = requested_m0 - set(candidate_by_integer_l)
        if missing_m0:
            raise ValueError(
                f"mosaic fit-eligible m=0 profile is absent from OSC evidence: {missing_m0}"
            )
        observed_m0 = tuple(candidate_by_integer_l[value] for value in sorted(requested_m0))
    m0_angles = detector_coordinates_to_angles(
        np.asarray([float(item["column_px"]) for item in observed_m0]),
        np.asarray([float(item["row_px"]) for item in observed_m0]),
        instrument=inputs.instrument,
        angle_frame=frame,
    )
    if not np.all(m0_angles.valid & m0_angles.azimuth_valid):
        raise RuntimeError("a supported m=0 observation has invalid detector angles")

    revision = configured_rod_catalog_revision(inputs)
    common = {
        "two_theta_half_width_rad": math.radians(float(profile_config["two_theta_half_width_deg"])),
        "phi_half_width_rad": math.radians(float(profile_config["phi_half_width_deg"])),
        "phi_bin_count": int(profile_config["phi_bin_count"]),
        "two_theta_gauss_order": int(two_theta_gauss_order),
        "phi_gauss_order": int(phi_gauss_order),
    }
    definitions: list[MosaicProfileDefinition] = []
    for m0_index, observed in enumerate(observed_m0):
        integer_l = int(observed["integer_L"])
        definitions.append(
            MosaicProfileDefinition(
                identity=MosaicProfileIdentity(
                    dataset_id=dataset_id,
                    incidence_angle_rad=math.radians(effective_incidence_deg),
                    group_key=MosaicReflectionGroupKey(
                        group_id=f"Bi2Se3:m=0:L={integer_l}",
                        rod_catalog_revision=revision,
                        member_rod_hk=((0, 0),),
                        branch_mode="COLLAPSED_00L",
                        layered_family_m=0,
                        layered_integer_L=integer_l,
                    ),
                    branch_id=None,
                    analytic_branch_id=0,
                ),
                center_two_theta_rad=float(m0_angles.two_theta_rad[m0_index]),
                center_phi_rad=float(m0_angles.phi_rad[m0_index]),
                **common,
            )
        )
    for angle_index, marker_index in enumerate(selected_marker):
        family = int(markers.family_m[marker_index])
        integer_l = int(markers.integer_L[marker_index])
        root_sign = int(markers.root_sign[marker_index])
        definitions.append(
            MosaicProfileDefinition(
                identity=MosaicProfileIdentity(
                    dataset_id=dataset_id,
                    incidence_angle_rad=math.radians(effective_incidence_deg),
                    group_key=MosaicReflectionGroupKey(
                        group_id=f"Bi2Se3:m={family}:L={integer_l}",
                        rod_catalog_revision=revision,
                        member_rod_hk=markers.contributing_rod_hk[marker_index],
                        branch_mode="EXPLICIT_NONZERO",
                        layered_family_m=family,
                        layered_integer_L=integer_l,
                    ),
                    branch_id=1 if root_sign < 0 else 2,
                    analytic_branch_id=int(markers.branch[marker_index]),
                ),
                center_two_theta_rad=float(marker_angles.two_theta_rad[angle_index]),
                center_phi_rad=float(marker_angles.phi_rad[angle_index]),
                **common,
            )
        )
    return frame, _filter_profile_definitions(tuple(definitions), eligible_profile_keys)


def _validated_profile_catalogs(
    series: tuple[ConfiguredSimulationInputs, ...],
    *,
    incidence_angles_deg: list[float],
    m0_observations: list[dict[str, Any]],
    profile_config: dict[str, Any],
    eligible_profile_keys: frozenset[_ProfileIdentityKey] | None,
) -> tuple[
    tuple[tuple[object, tuple[MosaicProfileDefinition, ...]], ...],
    tuple[dict[str, int | str], ...],
]:
    """Build and validate the exact profile catalogs consumed by fitting and rendering."""

    observations = {float(item["incidence_angle_deg"]): item for item in m0_observations}
    dataset_ids = frozenset(str(item["dataset_id"]) for item in m0_observations)
    _validate_eligible_profile_dataset_ids(eligible_profile_keys, dataset_ids)
    expected_total = tuple(int(value) for value in profile_config["expected_total_count"])
    expected_nonzero = tuple(int(value) for value in profile_config["expected_nonzero_count"])
    expected_m0 = tuple(int(value) for value in profile_config["expected_m0_count"])
    expected_revision = tuple(str(value) for value in profile_config["expected_catalog_revision"])
    if not (
        len(expected_total)
        == len(expected_nonzero)
        == len(expected_m0)
        == len(expected_revision)
        == len(series)
        == len(incidence_angles_deg)
    ):
        raise ValueError("expected profile-count vectors must align with the incidence series")

    catalogs: list[tuple[object, tuple[MosaicProfileDefinition, ...]]] = []
    records: list[dict[str, int | str]] = []
    compiled_keys: set[_ProfileIdentityKey] = set()
    two_theta_order = int(profile_config["response_two_theta_gauss_order"])
    phi_order = int(profile_config["response_phi_gauss_order"])
    for index, (inputs, incidence_value) in enumerate(
        zip(series, incidence_angles_deg, strict=True)
    ):
        incidence_deg = float(incidence_value)
        observation = observations[incidence_deg]
        frame, definitions = _profile_definitions(
            inputs,
            incidence_deg=incidence_deg,
            m0_observation=observation,
            profile_config=profile_config,
            two_theta_gauss_order=two_theta_order,
            phi_gauss_order=phi_order,
            eligible_profile_keys=eligible_profile_keys,
        )
        definition_keys = {_definition_identity_key(item) for item in definitions}
        compiled_keys.update(definition_keys)
        m0_count = sum(
            definition.identity.group_key.layered_family_m == 0 for definition in definitions
        )
        counts = (len(definitions), len(definitions) - m0_count, m0_count)
        catalog_revision = ordered_intensity_profile_catalog_revision(definitions)
        if eligible_profile_keys is None:
            expected_counts = (expected_total[index], expected_nonzero[index], expected_m0[index])
            if catalog_revision != expected_revision[index]:
                raise RuntimeError(
                    f"incidence {incidence_deg:g} profile identity catalog changed: "
                    f"expected {expected_revision[index]}, received {catalog_revision}"
                )
        else:
            dataset_id = str(observation["dataset_id"])
            selected = {key for key in eligible_profile_keys if key[0] == dataset_id}
            selected_m0 = sum(key[1] == 0 for key in selected)
            expected_counts = (len(selected), len(selected) - selected_m0, selected_m0)
        if counts != expected_counts:
            raise RuntimeError(
                f"incidence {incidence_deg:g} profile coverage changed: "
                f"expected {expected_counts}, received {counts}"
            )
        catalogs.append((frame, definitions))
        records.append(
            {
                "total": counts[0],
                "nonzero": counts[1],
                "m0": counts[2],
                "profile_catalog_revision": catalog_revision,
            }
        )
    if eligible_profile_keys is not None and compiled_keys != set(eligible_profile_keys):
        raise RuntimeError("ordered response did not consume the complete mosaic profile selection")
    return tuple(catalogs), tuple(records)


def _validate_anchor_catalog_records(
    actual: object,
    expected: tuple[dict[str, int | str], ...],
) -> None:
    fields = ("total", "nonzero", "m0", "profile_catalog_revision")
    if (
        not isinstance(actual, list)
        or len(actual) != len(expected)
        or not all(isinstance(item, dict) for item in actual)
        or [{name: item.get(name) for name in fields} for item in actual] != list(expected)
    ):
        raise ValueError("ordered-intensity result does not match the frozen anchor catalogs")


def _parameter_record(parameters: Bi2X3QuintupleLayerParameters) -> dict[str, float]:
    return {
        "bi_fractional_z": parameters.bi_fractional_z,
        "se2_fractional_z": parameters.se2_fractional_z,
        "bi_occupancy": parameters.bi_occupancy,
        "se1_occupancy": parameters.se1_occupancy,
        "se2_occupancy": parameters.se2_occupancy,
        "u_radial_A2": parameters.u_radial_A2,
        "u_normal_A2": parameters.u_normal_A2,
    }


def prepare_measured_ordered_inputs(
    mosaic_document: dict[str, Any],
    *,
    mosaic_case_path: Path,
    mosaic_case: dict[str, Any],
    profile_config: dict[str, Any],
    source_sample_count: int,
) -> PreparedMeasuredOrderedInputs:
    """Validate one mosaic checkpoint and construct the fixed continuous fit inputs."""

    (
        mosaic_parameters,
        source_revision,
        cif_sha256,
        eligible_profile_keys,
        fixed_position,
    ) = _validated_mosaic_result(
        mosaic_document,
        mosaic_case_path=mosaic_case_path,
        mosaic_case=mosaic_case,
        source_sample_count=source_sample_count,
    )
    series = _fixed_inputs(
        mosaic_case_path,
        mosaic_case,
        source_sample_count=source_sample_count,
        mosaic_parameters=mosaic_parameters,
        fixed_position=fixed_position,
    )
    profile_catalogs, anchor_counts = _validated_profile_catalogs(
        series,
        incidence_angles_deg=mosaic_case["incidence_angles_deg"],
        m0_observations=mosaic_case["m0_observations"],
        profile_config=profile_config,
        eligible_profile_keys=eligible_profile_keys,
    )
    return PreparedMeasuredOrderedInputs(
        series=series,
        profile_catalogs=profile_catalogs,
        anchor_counts=anchor_counts,
        mosaic_parameters=mosaic_parameters,
        source_revision=source_revision,
        cif_sha256=cif_sha256,
        fixed_position_record=_fixed_position_record(fixed_position),
        baseline_parameters=Bi2X3QuintupleLayerParameters.from_crystal(series[0].crystal),
    )


def run_recovery(
    case_path: Path,
    *,
    source_sample_count: int = 250,
    mosaic_parameters: dict[str, float] | None = None,
    required_source_revision: str | None = None,
    required_cif_sha256: str | None = None,
    upstream_mosaic_result_sha256: str | None = None,
    eligible_profile_keys: frozenset[_ProfileIdentityKey] | None = None,
    fixed_position: _FixedPositionState,
    synthetic_truth_proof: bool = False,
    execution_backend: str = "cpu",
) -> dict[str, Any]:
    case, mosaic_case_path, mosaic_case = _load_case(case_path)
    if execution_backend not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be cpu or cuda")
    upstream_values = (
        mosaic_parameters,
        required_source_revision,
        required_cif_sha256,
        upstream_mosaic_result_sha256,
        eligible_profile_keys,
    )
    if synthetic_truth_proof:
        if any(value is not None for value in upstream_values):
            raise ValueError("synthetic truth proof cannot consume a mosaic result")
        active_mosaic = {
            "gaussian_sigma_deg": float(mosaic_case["truth"]["gaussian_sigma_deg"]),
            "lorentzian_hwhm_deg": float(mosaic_case["truth"]["lorentzian_hwhm_deg"]),
            "lorentzian_probability": float(mosaic_case["truth"]["lorentzian_probability"]),
        }
    else:
        if any(value is None for value in upstream_values) or mosaic_parameters is None:
            raise ValueError("ordered recovery requires a complete upstream mosaic artifact state")
        active_mosaic = {name: float(value) for name, value in mosaic_parameters.items()}
    if set(active_mosaic) != {
        "gaussian_sigma_deg",
        "lorentzian_hwhm_deg",
        "lorentzian_probability",
    }:
        raise ValueError("mosaic_parameters must contain Gaussian, Lorentzian, and eta values")
    start = perf_counter()
    tracemalloc.start()
    series = _fixed_inputs(
        mosaic_case_path,
        mosaic_case,
        source_sample_count=source_sample_count,
        mosaic_parameters=active_mosaic,
        fixed_position=fixed_position,
    )
    if required_source_revision is not None and (
        not isinstance(required_source_revision, str)
        or not required_source_revision
        or series[0].samples.source_revision != required_source_revision
    ):
        raise ValueError("mosaic result source revision does not match the rebuilt source")
    current_cif_sha256 = hashlib.sha256(series[0].config.material.cif_path.read_bytes()).hexdigest()
    if required_cif_sha256 is not None and current_cif_sha256 != required_cif_sha256:
        raise ValueError("mosaic result CIF revision does not match the rebuilt structure")
    baseline = Bi2X3QuintupleLayerParameters.from_crystal(series[0].crystal)
    truth_config = case["truth"]
    truth = replace(
        baseline,
        bi_occupancy=float(truth_config["bi_occupancy"]),
        se1_occupancy=float(truth_config["se1_occupancy"]),
        se2_occupancy=float(truth_config["se2_occupancy"]),
        u_radial_A2=float(truth_config["u_radial_A2"]),
        u_normal_A2=float(truth_config["u_normal_A2"]),
    )
    truth_strength = replace(series[0].strength, structure_parameters=truth)
    profiles = case["profiles"]
    profile_catalogs, validated_profile_counts = _validated_profile_catalogs(
        series,
        incidence_angles_deg=mosaic_case["incidence_angles_deg"],
        m0_observations=mosaic_case["m0_observations"],
        profile_config=profiles,
        eligible_profile_keys=eligible_profile_keys,
    )
    response_rows = []
    truth_signal_rows = []
    profile_counts: list[dict[str, float | int | str]] = []
    response_compile_seconds = 0.0
    truth_generation_seconds = 0.0
    accelerated_prediction_seconds = 0.0
    kernel_oracle_relative_error = 0.0
    for inputs, incidence_value, catalog, count_record in zip(
        series,
        mosaic_case["incidence_angles_deg"],
        profile_catalogs,
        validated_profile_counts,
        strict=True,
    ):
        incidence_deg = float(incidence_value)
        frame, definitions = catalog
        detector = build_source_averaged_detector(inputs)
        m0_definitions = tuple(
            definition
            for definition in definitions
            if definition.identity.group_key.layered_family_m == 0
        )
        if m0_definitions:
            m0_forward_support = evaluate_source_averaged_ordered_intensity_point_signal(
                detector,
                angle_frame=frame,
                definitions=m0_definitions,
                structure_parameters=baseline,
                execution_backend=execution_backend,
            )
            if np.any(~np.isfinite(m0_forward_support)) or np.any(
                m0_forward_support <= np.finfo(np.float64).tiny
            ):
                raise RuntimeError(
                    f"incidence {incidence_deg:g} admits an m=0 anchor without positive "
                    "source-averaged baseline support"
                )
            m0_support_range = (
                float(np.min(m0_forward_support)),
                float(np.max(m0_forward_support)),
            )
        else:
            m0_support_range = (0.0, 0.0)
        compile_start = perf_counter()
        response = compile_source_averaged_ordered_intensity_response(
            detector,
            angle_frame=frame,
            definitions=definitions,
            execution_backend=execution_backend,
        )
        response_compile_seconds += perf_counter() - compile_start
        response_rows.append(response)
        truth_start = perf_counter()
        truth_signal = evaluate_source_averaged_ordered_intensity_point_signal(
            detector,
            angle_frame=frame,
            definitions=definitions,
            structure_parameters=truth,
            execution_backend=execution_backend,
        )
        truth_generation_seconds += perf_counter() - truth_start
        truth_signal_rows.append(truth_signal)
        accelerated_start = perf_counter()
        accelerated_signal = response.predict_signal_density_A2_per_rad2(truth)
        accelerated_prediction_seconds += perf_counter() - accelerated_start
        truth_scale = max(float(np.max(truth_signal)), np.finfo(np.float64).tiny)
        truth_relative_error = np.abs(accelerated_signal - truth_signal) / np.maximum(
            truth_signal,
            1.0e-12 * truth_scale,
        )
        kernel_oracle_relative_error = max(
            kernel_oracle_relative_error,
            float(np.max(truth_relative_error)),
        )
        profile_counts.append(
            {
                **count_record,
                "invalid_or_caustic_anchors": 0,
                "m0_baseline_support_minimum_A2_per_rad2": m0_support_range[0],
                "m0_baseline_support_maximum_A2_per_rad2": m0_support_range[1],
            }
        )
        del accelerated_signal
        gc.collect()
    responses = tuple(response_rows)
    truth_signal = tuple(truth_signal_rows)
    compile_seconds = response_compile_seconds
    maximum_interpolation_relative_error = max(
        response.interpolation_validation_maximum_relative_error for response in responses
    )
    truth_observations = tuple(
        OrderedIntensityPeakCenterObservations(
            dataset_id=response.dataset_id,
            observable_revision=response.observable_revision,
            signal_density_A2_per_rad2=signal,
        )
        for response, signal in zip(responses, truth_signal, strict=True)
    )
    active_absolute = (
        "bi_occupancy",
        "se1_occupancy",
        "se2_occupancy",
        "u_radial_A2",
        "u_normal_A2",
    )
    absolute_start = perf_counter()
    absolute = fit_ordered_intensity_series(
        responses,
        truth_observations,
        base_strength=series[0].strength,
        active_parameter_names=active_absolute,
        initial_parameters=baseline,
        relative_scale_mode=False,
        required_source_state_count=source_sample_count,
        required_source_revision=series[0].samples.source_revision,
    )
    absolute_seconds = perf_counter() - absolute_start

    relative_config = case["relative_mode"]
    if relative_config["fixed_occupancy"] != "bi_occupancy":
        raise ValueError("the first relative proof fixes Bi occupancy as its declared gauge")
    synthetic_scales = np.asarray(relative_config["synthetic_image_scales"], dtype=np.float64)
    relative_signal = tuple(
        scale * signal for scale, signal in zip(synthetic_scales, truth_signal, strict=True)
    )
    relative_observations = tuple(
        OrderedIntensityPeakCenterObservations(
            dataset_id=response.dataset_id,
            observable_revision=response.observable_revision,
            signal_density_A2_per_rad2=signal,
        )
        for response, signal in zip(responses, relative_signal, strict=True)
    )
    relative_start = perf_counter()
    relative = fit_ordered_intensity_series(
        responses,
        relative_observations,
        base_strength=series[0].strength,
        active_parameter_names=(
            "se1_occupancy",
            "se2_occupancy",
            "u_radial_A2",
            "u_normal_A2",
        ),
        initial_parameters=baseline,
        relative_scale_mode=True,
        required_source_state_count=source_sample_count,
        required_source_revision=series[0].samples.source_revision,
    )
    relative_seconds = perf_counter() - relative_start

    held_out_config = case["held_out"]
    held_out_h = np.asarray(held_out_config["h"], dtype=np.int32)
    held_out_k = np.asarray(held_out_config["k"], dtype=np.int32)
    held_out_l = np.asarray(held_out_config["L"], dtype=np.float64)
    if (
        held_out_h.ndim != 1
        or held_out_h.shape != held_out_k.shape
        or held_out_h.shape != held_out_l.shape
        or held_out_h.size == 0
    ):
        raise ValueError("held-out h, k, and L must be aligned nonempty vectors")
    for response in responses:
        fitted_hkl = {
            (h, k, int(identity.group_key.layered_integer_L))
            for identity in response.identities
            for h, k in identity.group_key.member_rod_hk
            if identity.group_key.layered_integer_L is not None
        }
        for h_value, k_value, ell_value in zip(
            held_out_h,
            held_out_k,
            held_out_l,
            strict=True,
        ):
            if (
                float(ell_value).is_integer()
                and (
                    int(h_value),
                    int(k_value),
                    int(ell_value),
                )
                in fitted_hkl
            ):
                raise ValueError("a declared held-out h,k,L point occurs in a fitted response")
    k_norm = 2.0 * math.pi / series[0].config.source.mean_wavelength_A
    held_out_truth = truth_strength.evaluate_hkl(
        h=held_out_h,
        k=held_out_k,
        L=held_out_l,
        k_norm_Ainv=k_norm,
    )
    held_out_absolute = replace(
        series[0].strength,
        structure_parameters=absolute.structure_representative,
    ).evaluate_hkl(
        h=held_out_h,
        k=held_out_k,
        L=held_out_l,
        k_norm_Ainv=k_norm,
    )
    largest_truth_occupancy = max(
        truth.bi_occupancy,
        truth.se1_occupancy,
        truth.se2_occupancy,
    )
    relative_truth = replace(
        truth,
        bi_occupancy=truth.bi_occupancy / largest_truth_occupancy,
        se1_occupancy=truth.se1_occupancy / largest_truth_occupancy,
        se2_occupancy=truth.se2_occupancy / largest_truth_occupancy,
    )
    held_out_relative_truth = replace(
        series[0].strength,
        structure_parameters=relative_truth,
    ).evaluate_hkl(
        h=held_out_h,
        k=held_out_k,
        L=held_out_l,
        k_norm_Ainv=k_norm,
    )
    held_out_relative = replace(
        series[0].strength,
        structure_parameters=relative.structure_representative,
    ).evaluate_hkl(
        h=held_out_h,
        k=held_out_k,
        L=held_out_l,
        k_norm_Ainv=k_norm,
    )
    if np.any(held_out_truth <= np.finfo(np.float64).tiny) or np.any(
        held_out_relative_truth <= np.finfo(np.float64).tiny
    ):
        raise FloatingPointError("a held-out reflection has numerically zero truth strength")
    held_out_absolute_relative_error = np.abs(held_out_absolute / held_out_truth - 1.0)
    held_out_relative_relative_error = np.abs(held_out_relative / held_out_relative_truth - 1.0)
    maximum_held_out_relative_error = float(
        max(
            np.max(held_out_absolute_relative_error),
            np.max(held_out_relative_relative_error),
        )
    )
    gauge_rejected = False
    try:
        fit_ordered_intensity_series(
            responses,
            relative_observations,
            base_strength=series[0].strength,
            active_parameter_names=active_absolute,
            initial_parameters=baseline,
            relative_scale_mode=True,
            maximum_function_evaluations=1,
            required_source_state_count=source_sample_count,
            required_source_revision=series[0].samples.source_revision,
        )
    except OrderedIntensityIdentifiabilityError:
        gauge_rejected = True

    absolute_error = {
        name: abs(getattr(absolute.structure_representative, name) - getattr(truth, name))
        for name in (
            "bi_occupancy",
            "se1_occupancy",
            "se2_occupancy",
            "u_radial_A2",
            "u_normal_A2",
        )
    }
    relative_target = {
        "bi_occupancy": 1.0,
        "se1_occupancy": truth.se1_occupancy / truth.bi_occupancy,
        "se2_occupancy": truth.se2_occupancy / truth.bi_occupancy,
        "u_radial_A2": truth.u_radial_A2,
        "u_normal_A2": truth.u_normal_A2,
    }
    ratio_by_name = dict(
        zip(
            ("bi_occupancy", "se1_occupancy", "se2_occupancy"),
            relative.occupancy_ratios,
            strict=True,
        )
    )
    relative_error = {
        **{
            name: abs(ratio_by_name[name] - relative_target[name])
            for name in ("bi_occupancy", "se1_occupancy", "se2_occupancy")
        },
        "u_radial_A2": abs(relative.structure_representative.u_radial_A2 - truth.u_radial_A2),
        "u_normal_A2": abs(relative.structure_representative.u_normal_A2 - truth.u_normal_A2),
    }
    acceptance = case["acceptance"]
    maximum_absolute_occupancy_error = max(
        absolute_error[name] for name in ("bi_occupancy", "se1_occupancy", "se2_occupancy")
    )
    maximum_relative_ratio_error = max(
        relative_error[name] for name in ("se1_occupancy", "se2_occupancy")
    )
    maximum_displacement_error = max(
        absolute_error["u_radial_A2"],
        absolute_error["u_normal_A2"],
        relative_error["u_radial_A2"],
        relative_error["u_normal_A2"],
    )
    maximum_relative_peak_residual = max(
        float(np.max(np.abs(absolute.relative_residual))),
        float(np.max(np.abs(relative.relative_residual))),
    )
    peak_memory_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    accepted = (
        gauge_rejected
        and absolute.structure_representative.bi_fractional_z == baseline.bi_fractional_z
        and absolute.structure_representative.se2_fractional_z == baseline.se2_fractional_z
        and relative.structure_representative.bi_fractional_z == baseline.bi_fractional_z
        and relative.structure_representative.se2_fractional_z == baseline.se2_fractional_z
        and maximum_absolute_occupancy_error
        <= float(acceptance["maximum_absolute_occupancy_error"])
        and maximum_relative_ratio_error
        <= float(acceptance["maximum_relative_occupancy_ratio_error"])
        and maximum_displacement_error <= float(acceptance["maximum_displacement_error_A2"])
        and maximum_relative_peak_residual <= float(acceptance["maximum_relative_peak_residual"])
        and maximum_held_out_relative_error <= float(acceptance["maximum_held_out_relative_error"])
        and maximum_interpolation_relative_error
        <= float(acceptance["maximum_response_quadrature_relative_error"])
        and kernel_oracle_relative_error
        <= float(acceptance["maximum_response_quadrature_relative_error"])
        and absolute.sensitivity_condition <= float(acceptance["maximum_sensitivity_condition"])
        and relative.sensitivity_condition <= float(acceptance["maximum_sensitivity_condition"])
        and not np.any(absolute.active_bounds)
        and not np.any(relative.active_bounds)
    )
    return {
        "schema_version": "rasim-bi2se3-ordered-intensity-recovery-v4",
        "accepted": accepted,
        "positions_frozen": True,
        "provenance": {
            "ordered_case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
            "mosaic_case_sha256": hashlib.sha256(mosaic_case_path.read_bytes()).hexdigest(),
            "upstream_mosaic_result_sha256": upstream_mosaic_result_sha256,
            "cif_sha256": current_cif_sha256,
            "rod_catalog_revision": configured_rod_catalog_revision(series[0]),
            "structure_model_revision": ordered_intensity_structure_model_revision(
                series[0].strength
            ),
        },
        "truth": _parameter_record(truth),
        "absolute": {
            "fit": _parameter_record(absolute.structure_representative),
            "absolute_error": absolute_error,
            "objective": absolute.objective,
            "maximum_relative_peak_residual": float(np.max(np.abs(absolute.relative_residual))),
            "sensitivity_rank": absolute.sensitivity_rank,
            "sensitivity_condition": absolute.sensitivity_condition,
            "singular_values": absolute.sensitivity_singular_values.tolist(),
            "parameter_correlation": absolute.parameter_correlation.tolist(),
            "active_bounds": dict(
                zip(absolute.active_parameter_names, absolute.active_bounds.tolist(), strict=True)
            ),
            "function_evaluations": absolute.function_evaluations,
            "seconds": absolute_seconds,
        },
        "relative": {
            "fixed_gauge": "bi_occupancy=1",
            "occupancy_ratio_reference": relative.occupancy_ratio_reference,
            "occupancy_ratios": ratio_by_name,
            "target": relative_target,
            "admissible_structure_representative": _parameter_record(
                relative.structure_representative
            ),
            "absolute_error": relative_error,
            "dataset_scales": relative.dataset_scales.tolist(),
            "objective": relative.objective,
            "maximum_relative_peak_residual": float(np.max(np.abs(relative.relative_residual))),
            "sensitivity_rank": relative.sensitivity_rank,
            "sensitivity_condition": relative.sensitivity_condition,
            "singular_values": relative.sensitivity_singular_values.tolist(),
            "parameter_correlation": relative.parameter_correlation.tolist(),
            "active_bounds": dict(
                zip(relative.active_parameter_names, relative.active_bounds.tolist(), strict=True)
            ),
            "function_evaluations": relative.function_evaluations,
            "seconds": relative_seconds,
        },
        "common_occupancy_scale_gauge_rejected": gauge_rejected,
        "source_model": {
            "reduction": "one_incoherent_weighted_detector_function_per_incidence.v1",
            "sample_count": source_sample_count,
            "source_revision": series[0].samples.source_revision,
            "source_sampling_model_id": series[0].samples.source_sampling_model_id,
            "source_rng_model_id": series[0].samples.source_rng_model_id,
            "source_seed": series[0].samples.source_seed,
            "spatial_sigma_m": list(series[0].config.source.spatial_sigma_m),
            "divergence_sigma_rad": list(series[0].config.source.divergence_sigma_rad),
            "wavelength_sigma_A": series[0].config.source.wavelength_sigma_A,
        },
        "fixed_mosaic": active_mosaic,
        "fixed_position": _fixed_position_record(fixed_position),
        "effective_incidence_angles_deg": list(fixed_position.effective_incidence_angles_deg),
        "upstream_fit_eligible_profiles": _profile_identity_records(eligible_profile_keys),
        "truth_generation": ("fresh_source_averaged_selected_group_peak_center_signal_density.v1"),
        "observable_interpretation": (
            "synthetic selected-component peak-center recovery; not a fit to unresolved raw OSC "
            "intensity and not an integrated peak mass"
        ),
        "compact_response_oracle_maximum_relative_error": kernel_oracle_relative_error,
        "response_contract": _response_contract_record(),
        "u_normal_chebyshev_validation": {
            "node_count": 13,
            "domain_A2": [0.0, 0.1],
            "interlaced_validation_count_per_dataset": 12,
            "maximum_relative_error": maximum_interpolation_relative_error,
            "per_dataset": [
                {
                    "dataset_id": response.dataset_id,
                    "maximum_relative_error": (
                        response.interpolation_validation_maximum_relative_error
                    ),
                    "validation_node_count": response.interpolation_validation_node_count,
                }
                for response in responses
            ],
        },
        "equivalent_work_seconds": {
            "fresh_truth_prediction": truth_generation_seconds,
            "cached_truth_prediction": accelerated_prediction_seconds,
            "truth_prediction_speedup": (truth_generation_seconds / accelerated_prediction_seconds),
        },
        "response_revisions": [response.response_revision for response in responses],
        "response_execution": [
            {
                "backend": response.execution_backend,
                "device": response.execution_device,
                "instrument_revision": response.instrument_revision,
            }
            for response in responses
        ],
        "observable_revisions": [response.observable_revision for response in responses],
        "held_out_exact_model_interpolation": {
            "h": held_out_h.tolist(),
            "k": held_out_k.tolist(),
            "L": held_out_l.tolist(),
            "truth_strength_A2": held_out_truth.tolist(),
            "absolute_fit_strength_A2": held_out_absolute.tolist(),
            "absolute_relative_error": held_out_absolute_relative_error.tolist(),
            "relative_truth_representative_strength_A2": held_out_relative_truth.tolist(),
            "relative_fit_representative_strength_A2": held_out_relative.tolist(),
            "relative_relative_error": held_out_relative_relative_error.tolist(),
            "maximum_relative_error": maximum_held_out_relative_error,
        },
        "incidence_angles_deg": [float(value) for value in mosaic_case["incidence_angles_deg"]],
        "anchor_counts": profile_counts,
        "truth_signal_density_A2_per_rad2": {
            "minimum": min(float(np.min(signal)) for signal in truth_signal),
            "maximum": max(float(np.max(signal)) for signal in truth_signal),
            "dynamic_range": max(float(np.max(signal)) for signal in truth_signal)
            / min(float(np.min(signal)) for signal in truth_signal),
            "all_anchors_retained": sum(signal.size for signal in truth_signal),
        },
        "response_coefficient_counts": [
            int(response.occupancy_quadratic_chebyshev_signal_density_A2_per_rad2.size)
            for response in responses
        ],
        "compile_seconds": compile_seconds,
        "response_compile_seconds": response_compile_seconds,
        "truth_generation_seconds": truth_generation_seconds,
        "total_seconds": perf_counter() - start,
        "peak_memory_bytes": peak_memory_bytes,
    }


def render_recovered_images(
    case_path: Path,
    *,
    mosaic_result: dict[str, Any],
    ordered_result: dict[str, Any],
    output_directory: Path,
    source_sample_count: int = 250,
    execution_backend: str = "cpu",
    mosaic_result_sha256: str | None = None,
    ordered_result_sha256: str | None = None,
) -> dict[str, Any]:
    """Render raw OSCs and fitted center-sampled detector functions on one native grid."""

    resolved_output = output_directory.resolve()
    if resolved_output == ROOT or resolved_output.is_relative_to(ROOT):
        raise ValueError("render output must be outside the repository")
    if resolved_output.exists() and any(resolved_output.iterdir()):
        raise ValueError("render output directory must be empty")
    resolved_output.mkdir(parents=True, exist_ok=True)
    case, mosaic_case_path, mosaic_case = _load_case(case_path)
    (
        mosaic_parameters,
        _,
        mosaic_cif_sha256,
        eligible_profile_keys,
        fixed_position,
    ) = _validated_mosaic_result(
        mosaic_result,
        mosaic_case_path=mosaic_case_path,
        mosaic_case=mosaic_case,
        source_sample_count=source_sample_count,
    )
    _validate_ordered_result_header(ordered_result)
    if ordered_result.get("fixed_mosaic") != mosaic_parameters:
        raise ValueError("ordered-intensity result does not use the recovered mosaic parameters")
    if ordered_result.get("fixed_position") != _fixed_position_record(fixed_position):
        raise ValueError("ordered-intensity result does not preserve the fitted position state")
    if ordered_result.get("upstream_fit_eligible_profiles") != _profile_identity_records(
        eligible_profile_keys
    ):
        raise ValueError("ordered-intensity result does not preserve the mosaic profile selection")
    expected_incidence = [float(value) for value in mosaic_case["incidence_angles_deg"]]
    if ordered_result.get("incidence_angles_deg") != expected_incidence:
        raise ValueError("ordered-intensity result does not match the requested incidence series")
    absolute = ordered_result.get("absolute")
    fitted_record = absolute.get("fit") if isinstance(absolute, dict) else None
    if not isinstance(fitted_record, dict):
        raise ValueError("ordered-intensity result lacks absolute.fit")
    series = _fixed_inputs(
        mosaic_case_path,
        mosaic_case,
        source_sample_count=source_sample_count,
        mosaic_parameters=mosaic_parameters,
        fixed_position=fixed_position,
    )
    baseline = Bi2X3QuintupleLayerParameters.from_crystal(series[0].crystal)
    ordered_provenance = ordered_result.get("provenance")
    expected_ordered_provenance = {
        "ordered_case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        "mosaic_case_sha256": hashlib.sha256(mosaic_case_path.read_bytes()).hexdigest(),
        "cif_sha256": hashlib.sha256(series[0].config.material.cif_path.read_bytes()).hexdigest(),
        "rod_catalog_revision": configured_rod_catalog_revision(series[0]),
        "structure_model_revision": ordered_intensity_structure_model_revision(series[0].strength),
    }
    if not isinstance(ordered_provenance, dict) or any(
        ordered_provenance.get(name) != value for name, value in expected_ordered_provenance.items()
    ):
        raise ValueError("ordered-intensity result does not match the current fixed structure")
    if expected_ordered_provenance["cif_sha256"] != mosaic_cif_sha256:
        raise ValueError("mosaic and ordered-intensity results do not share the current CIF")
    if (
        mosaic_result_sha256 is not None
        and ordered_provenance.get("upstream_mosaic_result_sha256") != mosaic_result_sha256
    ):
        raise ValueError("ordered-intensity result does not bind the supplied mosaic artifact")
    if (
        float(fitted_record["bi_fractional_z"]) != baseline.bi_fractional_z
        or float(fitted_record["se2_fractional_z"]) != baseline.se2_fractional_z
    ):
        raise ValueError("ordered-intensity result changed a frozen Wyckoff coordinate")
    _, expected_anchor_records = _validated_profile_catalogs(
        series,
        incidence_angles_deg=mosaic_case["incidence_angles_deg"],
        m0_observations=mosaic_case["m0_observations"],
        profile_config=case["profiles"],
        eligible_profile_keys=eligible_profile_keys,
    )
    _validate_anchor_catalog_records(ordered_result.get("anchor_counts"), expected_anchor_records)
    expected_source = {
        "sample_count": source_sample_count,
        "source_revision": series[0].samples.source_revision,
        "reduction": "one_incoherent_weighted_detector_function_per_incidence.v1",
    }
    for label, document in (("mosaic", mosaic_result), ("ordered-intensity", ordered_result)):
        source_model = document.get("source_model")
        if not isinstance(source_model, dict) or any(
            source_model.get(name) != expected for name, expected in expected_source.items()
        ):
            raise ValueError(f"{label} result does not match the requested source realization")
    fitted_parameters = replace(
        Bi2X3QuintupleLayerParameters.from_crystal(series[0].crystal),
        bi_fractional_z=float(fitted_record["bi_fractional_z"]),
        se2_fractional_z=float(fitted_record["se2_fractional_z"]),
        bi_occupancy=float(fitted_record["bi_occupancy"]),
        se1_occupancy=float(fitted_record["se1_occupancy"]),
        se2_occupancy=float(fitted_record["se2_occupancy"]),
        u_radial_A2=float(fitted_record["u_radial_A2"]),
        u_normal_A2=float(fitted_record["u_normal_A2"]),
    )
    start = perf_counter()
    simulated_images: list[np.ndarray] = []
    execution: list[dict[str, object]] = []
    response_execution = ordered_result.get("response_execution")
    if not isinstance(response_execution, list) or len(response_execution) != len(series):
        raise ValueError("ordered-intensity result lacks per-incidence execution provenance")
    response_revisions = ordered_result.get("response_revisions")
    if (
        not isinstance(response_revisions, list)
        or len(response_revisions) != len(series)
        or any(not isinstance(revision, str) or not revision for revision in response_revisions)
    ):
        raise ValueError("ordered-intensity result lacks response coefficient revisions")
    for response_record, incidence_deg, inputs in zip(
        response_execution,
        mosaic_case["incidence_angles_deg"],
        series,
        strict=True,
    ):
        strength = replace(inputs.strength, structure_parameters=fitted_parameters)
        detector = build_source_averaged_detector(replace(inputs, strength=strength))
        if (
            detector.source_state_count != source_sample_count
            or detector.valid_source_state_count != source_sample_count
            or detector.incident.states.source_revision != expected_source["source_revision"]
            or np.unique(detector.incident.states.incident_state_id).size != source_sample_count
            or tuple(detector.rods) != tuple(inputs.rods)
            or not any(rod.family_m == 0 for rod in detector.rods)
        ):
            raise RuntimeError("render detector does not preserve the full combined source/rod set")
        if not isinstance(response_record, dict) or response_record.get(
            "instrument_revision"
        ) != source_averaged_detector_instrument_revision(detector):
            raise ValueError("ordered-intensity result does not match the render instrument")
        fit_response_backend = response_record.get("backend")
        fit_response_device = response_record.get("device")
        if fit_response_backend not in {
            "numba_cpu_source_averaged.v1",
            "numba_cuda_source_averaged.v1",
        } or (fit_response_backend == "numba_cuda_source_averaged.v1") != (
            isinstance(fit_response_device, str) and bool(fit_response_device)
        ):
            raise ValueError("ordered-intensity result has invalid response execution provenance")
        sampled = sample_detector_pixel_center_density(
            detector,
            execution_backend=execution_backend,
        )
        if sampled.image_A2_per_px2.shape != (3000, 3000):
            raise RuntimeError("fitted detector image is not 3000 by 3000")
        simulated_image = np.asarray(sampled.image_A2_per_px2, dtype=np.float32)
        simulated_images.append(simulated_image)
        execution.append(
            {
                "incidence_deg": float(incidence_deg),
                "backend": sampled.execution_backend,
                "device": sampled.execution_device,
                "fit_response_backend": fit_response_backend,
                "fit_response_device": fit_response_device,
                "coordinate_evaluation_count": sampled.coordinate_evaluation_count,
                "source_state_count": detector.source_state_count,
                "valid_source_state_count": detector.valid_source_state_count,
                "rod_catalog_revision": detector.rod_catalog_revision,
                "rod_hk": [[rod.h, rod.k] for rod in detector.rods],
                "family_m_counts": {
                    str(family_m): sum(rod.family_m == family_m for rod in detector.rods)
                    for family_m in sorted({rod.family_m for rod in detector.rods})
                },
                "numeric_float32_sha256": hashlib.sha256(
                    simulated_image.tobytes(order="C")
                ).hexdigest(),
            }
        )

    raw_images: list[np.ndarray] = []
    raw_provenance: list[dict[str, object]] = []
    observation_by_incidence = {
        float(observation["incidence_angle_deg"]): observation
        for observation in mosaic_case["m0_observations"]
    }
    if set(observation_by_incidence) != set(expected_incidence):
        raise ValueError("OSC observations do not match the requested incidence series")
    for incidence_deg in expected_incidence:
        observation = observation_by_incidence[incidence_deg]
        osc_path = (mosaic_case_path.parent / str(observation["osc_file"])).resolve()
        compressed_sha256 = hashlib.sha256(osc_path.read_bytes()).hexdigest()
        if compressed_sha256 != observation["osc_file_sha256"]:
            raise ValueError(f"OSC file hash changed for {incidence_deg:g} degrees")
        native_counts = read_osc(osc_path).detector_native_counts
        native_sha256 = hashlib.sha256(native_counts.tobytes(order="C")).hexdigest()
        if (
            native_sha256 != observation["detector_native_bytes_sha256"]
            or native_counts.dtype != np.dtype(str(observation["detector_native_dtype"]))
            or list(native_counts.shape) != observation["detector_native_shape_rc"]
        ):
            raise ValueError(f"decoded OSC data changed for {incidence_deg:g} degrees")
        raw = np.asarray(native_counts, dtype=np.float32)
        if raw.shape != (3000, 3000):
            raise RuntimeError("OSC detector-native image is not 3000 by 3000")
        raw_images.append(raw)
        raw_provenance.append(
            {
                "incidence_deg": incidence_deg,
                "osc_path": str(osc_path),
                "osc_file_sha256": compressed_sha256,
                "detector_native_bytes_sha256": native_sha256,
            }
        )

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    simulation_high = max(float(np.max(image)) for image in simulated_images)
    if not math.isfinite(simulation_high) or simulation_high <= 0.0:
        raise RuntimeError("fitted detector images contain no positive finite density")
    simulation_low = simulation_high * 1.0e-10
    raw_high = max(float(np.max(np.log1p(image))) for image in raw_images)
    if not math.isfinite(raw_high) or raw_high <= 0.0:
        raise RuntimeError("OSC images contain no positive finite counts")

    def simulated_display(image: np.ndarray) -> np.ndarray:
        positive = image > 0.0
        display = np.full(image.shape, math.log(simulation_low), dtype=np.float32)
        np.log(image, out=display, where=positive)
        display -= math.log(simulation_low)
        display /= math.log(simulation_high / simulation_low)
        np.clip(display, 0.0, 1.0, out=display)
        return display

    def raw_display(image: np.ndarray) -> np.ndarray:
        display = np.log1p(image, dtype=np.float64) / raw_high
        return np.asarray(np.clip(display, 0.0, 1.0), dtype=np.float32)

    simulated_display_images = [simulated_display(image) for image in simulated_images]
    raw_display_images = [raw_display(image) for image in raw_images]
    full_paths: list[str] = []
    preview_paths: list[str] = []
    for incidence_deg, display in zip(
        mosaic_case["incidence_angles_deg"],
        simulated_display_images,
        strict=True,
    ):
        full_path = resolved_output / f"simulated_{float(incidence_deg):g}deg_3000x3000.png"
        preview_path = resolved_output / f"simulated_{float(incidence_deg):g}deg_preview.png"
        plt.imsave(full_path, display, cmap="magma", vmin=0.0, vmax=1.0, origin="upper")
        plt.imsave(
            preview_path,
            display[::2, ::2],
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
        )
        full_paths.append(str(full_path))
        preview_paths.append(str(preview_path))
    raw_paths: list[str] = []
    raw_preview_paths: list[str] = []
    for incidence_deg, display in zip(
        mosaic_case["incidence_angles_deg"],
        raw_display_images,
        strict=True,
    ):
        full_path = resolved_output / f"raw_osc_{float(incidence_deg):g}deg_3000x3000.png"
        preview_path = resolved_output / f"raw_osc_{float(incidence_deg):g}deg_preview.png"
        plt.imsave(full_path, display, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
        plt.imsave(
            preview_path,
            display[::2, ::2],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
        )
        raw_paths.append(str(full_path))
        raw_preview_paths.append(str(preview_path))

    figure, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), constrained_layout=True)
    for column, incidence_deg in enumerate(mosaic_case["incidence_angles_deg"]):
        axes[0, column].imshow(raw_display_images[column], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, column].set_title(f"Raw OSC {float(incidence_deg):g}°")
        axes[1, column].imshow(
            simulated_display_images[column],
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, column].set_title(
            f"{source_sample_count}-state forward model {float(incidence_deg):g}°\n"
            "pixel-center density"
        )
        axes[0, column].set_axis_off()
        axes[1, column].set_axis_off()
    figure.suptitle(
        "Raw counts and simulated density use separate shared log scales; display only, not "
        "count-calibrated",
        fontsize=10,
    )
    comparison_path = resolved_output / f"raw_and_{source_sample_count}ki_forward_model.png"
    figure.savefig(comparison_path, dpi=200)
    plt.close(figure)
    png_paths = [
        *(Path(path) for path in full_paths),
        *(Path(path) for path in preview_paths),
        *(Path(path) for path in raw_paths),
        *(Path(path) for path in raw_preview_paths),
        comparison_path,
    ]
    png_sha256 = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in png_paths}

    manifest = {
        "schema_version": "rasim-bi2se3-source-averaged-forward-images-v1",
        "shape_rc": [3000, 3000],
        "coordinate_convention": "detector-native [row,column], origin upper",
        "simulation_measure_id": "raw_detector_coordinate_density_A2_per_px2.v1",
        "execution_backend_policy": (
            "fit-response and render backends may differ after permanent CPU/CUDA detector "
            "parity proof; both are recorded per incidence"
        ),
        "simulation_sampling_grid_id": "native_pixel_centers.v1",
        "source_state_count": source_sample_count,
        "source_revision": series[0].samples.source_revision,
        "source_reduction": "one_incoherent_weighted_detector_function_per_incidence.v1",
        "rod_count": len(series[0].rods),
        "rod_catalog_revision": configured_rod_catalog_revision(series[0]),
        "root_policy": "all_retained_roots.v1",
        "includes_m0": any(rod.family_m == 0 for rod in series[0].rods),
        "mosaic_parameters": mosaic_parameters,
        "ordered_parameters": _parameter_record(fitted_parameters),
        "mosaic_parameter_source": "model_limited_real_osc_shape_fit",
        "ordered_parameter_source": "synthetic_selected_group_peak_center_recovery",
        "raw_osc_intensity_legacy_classification": "NO_ORACLE",
        "input_result_sha256": {
            "mosaic": mosaic_result_sha256,
            "ordered_intensity": ordered_result_sha256,
        },
        "raw_simulation_comparison": (
            "display_only; independently transformed rows; not count calibrated or a raw-OSC "
            "ordered-intensity fit"
        ),
        "display": {
            "simulation": "shared natural-log range [global_max*1e-10, global_max]",
            "raw_osc": "shared log1p(counts) range [0, global_max]",
            "png_measure": "dimensionless display value after the declared row transform",
        },
        "execution": execution,
        "raw_osc_provenance": raw_provenance,
        "simulated_full_paths": full_paths,
        "simulated_preview_paths": preview_paths,
        "raw_full_paths": raw_paths,
        "raw_preview_paths": raw_preview_paths,
        "comparison_path": str(comparison_path),
        "png_sha256": png_sha256,
        "seconds": perf_counter() - start,
    }
    manifest_path = resolved_output / f"bi2se3_{source_sample_count}ki_images.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--source-sample-count", type=int, default=250)
    parser.add_argument("--mosaic-result", type=Path)
    parser.add_argument(
        "--synthetic-truth-proof",
        action="store_true",
        help="run the explicit planted-truth proof without a prior mosaic artifact",
    )
    parser.add_argument("--ordered-result", type=Path)
    parser.add_argument("--execution-backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render-directory", type=Path)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.mosaic_result is not None and arguments.synthetic_truth_proof:
        raise ValueError("--mosaic-result and --synthetic-truth-proof are mutually exclusive")
    if arguments.render_only and arguments.synthetic_truth_proof:
        raise ValueError("--render-only cannot use --synthetic-truth-proof")
    if not arguments.render_only and not (
        arguments.mosaic_result is not None or arguments.synthetic_truth_proof
    ):
        raise ValueError("ordered recovery requires --mosaic-result or --synthetic-truth-proof")
    mosaic_parameters = None
    mosaic_result_document = None
    required_source_revision = None
    required_cif_sha256 = None
    eligible_profile_keys = None
    _, mosaic_case_path, mosaic_case = _load_case(arguments.case.resolve())
    fixed_position = None
    mosaic_result_sha256 = None
    if arguments.mosaic_result is not None:
        mosaic_result_bytes = arguments.mosaic_result.resolve().read_bytes()
        mosaic_result_sha256 = hashlib.sha256(mosaic_result_bytes).hexdigest()
        mosaic_result_document = json.loads(mosaic_result_bytes)
        (
            mosaic_parameters,
            required_source_revision,
            required_cif_sha256,
            eligible_profile_keys,
            fixed_position,
        ) = _validated_mosaic_result(
            mosaic_result_document,
            mosaic_case_path=mosaic_case_path,
            mosaic_case=mosaic_case,
            source_sample_count=arguments.source_sample_count,
        )
    if arguments.synthetic_truth_proof:
        fixed_position = _case_fixed_position_state(mosaic_case_path, mosaic_case)
    if arguments.render_only:
        if (
            mosaic_result_document is None
            or arguments.ordered_result is None
            or arguments.render_directory is None
        ):
            raise ValueError(
                "--render-only requires --mosaic-result, --ordered-result, and --render-directory"
            )
        ordered_result_bytes = arguments.ordered_result.resolve().read_bytes()
        ordered_result = json.loads(ordered_result_bytes)
        rendered = render_recovered_images(
            arguments.case.resolve(),
            mosaic_result=mosaic_result_document,
            ordered_result=ordered_result,
            output_directory=arguments.render_directory,
            source_sample_count=arguments.source_sample_count,
            execution_backend=arguments.execution_backend,
            mosaic_result_sha256=mosaic_result_sha256,
            ordered_result_sha256=hashlib.sha256(ordered_result_bytes).hexdigest(),
        )
        print(json.dumps(rendered, indent=None if arguments.json else 2, sort_keys=True))
        return 0
    if fixed_position is None:
        raise RuntimeError("ordered recovery lacks its fixed position state")
    result = run_recovery(
        arguments.case.resolve(),
        source_sample_count=arguments.source_sample_count,
        mosaic_parameters=mosaic_parameters,
        required_source_revision=required_source_revision,
        required_cif_sha256=required_cif_sha256,
        upstream_mosaic_result_sha256=mosaic_result_sha256,
        eligible_profile_keys=eligible_profile_keys,
        fixed_position=fixed_position,
        synthetic_truth_proof=arguments.synthetic_truth_proof,
        execution_backend=arguments.execution_backend,
    )
    if arguments.render_directory is not None:
        if mosaic_result_document is None:
            raise ValueError("rendering a recovery requires --mosaic-result")
        result["render"] = render_recovered_images(
            arguments.case.resolve(),
            mosaic_result=mosaic_result_document,
            ordered_result=result,
            output_directory=arguments.render_directory,
            source_sample_count=arguments.source_sample_count,
            execution_backend=arguments.execution_backend,
            mosaic_result_sha256=mosaic_result_sha256,
        )
    encoded = json.dumps(result, sort_keys=True)
    if arguments.output is not None:
        output_path = arguments.output.resolve()
        if output_path == ROOT or output_path.is_relative_to(ROOT):
            raise ValueError("output must be outside the repository")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
    if arguments.json:
        print(encoded)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
