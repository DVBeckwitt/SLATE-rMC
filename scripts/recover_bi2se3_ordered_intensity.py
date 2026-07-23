"""Recover Bi2Se3 occupancies and directional displacement from three fixed views."""

from __future__ import annotations

import argparse
import gc
import json
import math
import tomllib
import tracemalloc
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from painted_ewald import MosaicBraggSpace
from rasim_next.fitting import (
    MosaicProfileDefinition,
    MosaicProfileIdentity,
    MosaicReflectionGroupKey,
    OrderedIntensityIdentifiabilityError,
    OrderedIntensityObservations,
    SharedGeometryCorrections,
    apply_shared_geometry_corrections,
    compile_ordered_intensity_response,
    fit_ordered_intensity_series,
    ordered_intensity_profile_catalog_revision,
)
from rasim_next.geometry import build_incident_states, detector_coordinates_to_angles
from rasim_next.ordered import Bi2Se3QuintupleLayerParameters
from rasim_next.pipeline.configured_simulation import (
    ConfiguredSimulationInputs,
    build_configured_simulation_inputs,
    build_nominal_ewald_context,
    configured_rod_catalog_revision,
    evaluate_nominal_integer_l_markers,
    load_simulation_config,
)
from rasim_next.selection import build_osc_angle_frame

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = ROOT / "examples" / "bi2se3" / "experiment" / "ordered_intensity_fit_truth.toml"


def _load_case(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    case = tomllib.loads(path.read_text(encoding="utf-8"))
    if case.get("schema_version") != "rasim-ordered-intensity-recovery-v1":
        raise ValueError("unsupported ordered-intensity recovery schema")
    mosaic_case_path = (path.parent / str(case["mosaic_case"])).resolve()
    mosaic_case = tomllib.loads(mosaic_case_path.read_text(encoding="utf-8"))
    if mosaic_case.get("schema_version") != "rasim-mosaic-recovery-v2":
        raise ValueError("ordered-intensity recovery requires the accepted mosaic case")
    return case, mosaic_case_path, mosaic_case


def _fixed_inputs(
    mosaic_case_path: Path,
    mosaic_case: dict[str, Any],
) -> tuple[ConfiguredSimulationInputs, ...]:
    config_path = (mosaic_case_path.parent / str(mosaic_case["simulation_config"])).resolve()
    source_config = load_simulation_config(config_path)
    truth_mosaic = mosaic_case["truth"]
    corrections = SharedGeometryCorrections.from_array(mosaic_case["shared_geometry_corrections"])
    series: list[ConfiguredSimulationInputs] = []
    for incidence_deg in mosaic_case["incidence_angles_deg"]:
        config = replace(
            source_config,
            source=replace(
                source_config.source,
                spatial_sigma_m=(0.0, 0.0),
                divergence_sigma_rad=(0.0, 0.0),
                wavelength_sigma_A=0.0,
                sample_count=1,
            ),
            mosaic=replace(
                source_config.mosaic,
                gaussian_sigma_deg=float(truth_mosaic["gaussian_sigma_deg"]),
                lorentzian_hwhm_deg=float(truth_mosaic["lorentzian_hwhm_deg"]),
                lorentzian_probability=float(truth_mosaic["lorentzian_probability"]),
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
        if incident.states.incident_state_id.size != 1 or not bool(incident.states.valid[0]):
            raise RuntimeError("ordered-intensity proof requires one valid ideal incident state")
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
    return tuple(series)


def _profile_definitions(
    inputs: ConfiguredSimulationInputs,
    *,
    incidence_deg: float,
    m0_observation: dict[str, Any],
    profile_config: dict[str, Any],
    two_theta_gauss_order: int,
    phi_gauss_order: int,
) -> tuple[object, tuple[MosaicProfileDefinition, ...]]:
    context = build_nominal_ewald_context(inputs)
    frame = build_osc_angle_frame(
        mean_direction_lab=inputs.config.source.mean_direction_lab,
        instrument=inputs.instrument,
        sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
        revision=f"ordered-intensity-fixed-nine-{incidence_deg:g}deg.v1",
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
    supported_key = f"supported_m0_integer_L_{incidence_deg:g}deg"
    supported_m0 = {int(value) for value in profile_config[supported_key]}
    observed_m0 = tuple(
        item for item in m0_observation["observed_peaks"] if int(item["integer_L"]) in supported_m0
    )
    if {int(item["integer_L"]) for item in observed_m0} != supported_m0:
        raise RuntimeError("the admitted m=0 set does not match the frozen OSC evidence")
    m0_angles = detector_coordinates_to_angles(
        np.asarray([float(item["column_px"]) for item in observed_m0]),
        np.asarray([float(item["row_px"]) for item in observed_m0]),
        instrument=inputs.instrument,
        angle_frame=frame,
    )
    if not np.all(m0_angles.valid & m0_angles.azimuth_valid):
        raise RuntimeError("a supported m=0 observation has invalid detector angles")

    dataset_id = str(m0_observation["dataset_id"])
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
                    incidence_angle_rad=math.radians(incidence_deg),
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
                    incidence_angle_rad=math.radians(incidence_deg),
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
    return frame, tuple(definitions)


def _parameter_record(parameters: Bi2Se3QuintupleLayerParameters) -> dict[str, float]:
    return {
        "bi_fractional_z": parameters.bi_fractional_z,
        "se2_fractional_z": parameters.se2_fractional_z,
        "bi_occupancy": parameters.bi_occupancy,
        "se1_occupancy": parameters.se1_occupancy,
        "se2_occupancy": parameters.se2_occupancy,
        "u_radial_A2": parameters.u_radial_A2,
        "u_normal_A2": parameters.u_normal_A2,
    }


def _integrated_occupancy_basis(
    response: Any,
    *,
    u_radial_A2: float,
    u_normal_A2: float,
) -> np.ndarray:
    """Integrate the six cached occupancy-quadratic columns at one directional U pair."""

    damping = np.exp(
        -u_radial_A2 * response.term_q_radial_squared_Ainv2
        - u_normal_A2 * response.term_q_normal_squared_Ainv2
    )
    fixed_weight = response.term_fixed_mass_per_strength * damping
    return np.column_stack(
        tuple(
            np.bincount(
                response.term_observation_index,
                weights=fixed_weight * response.term_occupancy_quadratic_strength_A2[:, column],
                minlength=len(response.identities),
            )
            for column in range(6)
        )
    )


def run_recovery(case_path: Path) -> dict[str, Any]:
    case, mosaic_case_path, mosaic_case = _load_case(case_path)
    start = perf_counter()
    tracemalloc.start()
    series = _fixed_inputs(mosaic_case_path, mosaic_case)
    baseline = Bi2Se3QuintupleLayerParameters.from_crystal(series[0].crystal)
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
    displacement_probe_A2 = (
        (0.0, 0.0),
        (truth.u_radial_A2, truth.u_normal_A2),
        (0.1, 0.0),
        (0.0, 0.1),
        (0.1, 0.1),
    )
    profiles = case["profiles"]
    observation_by_incidence = {
        float(item["incidence_angle_deg"]): item for item in mosaic_case["m0_observations"]
    }
    response_rows = []
    truth_mass_rows = []
    truth_response_revisions = []
    truth_response_term_counts = []
    response_quadrature_rows = []
    basis_quadrature_rows = []
    profile_counts: list[dict[str, int | str]] = []
    response_compile_seconds = 0.0
    truth_compile_seconds = 0.0
    truth_generation_seconds = 0.0
    direct_seconds = 0.0
    accelerated_seconds = 0.0
    kernel_oracle_relative_error = 0.0
    response_two_theta_gauss_order = int(profiles["response_two_theta_gauss_order"])
    truth_two_theta_gauss_order = int(profiles["truth_two_theta_gauss_order"])
    response_phi_gauss_order = int(profiles["response_phi_gauss_order"])
    truth_phi_gauss_order = int(profiles["truth_phi_gauss_order"])
    if (
        truth_two_theta_gauss_order <= response_two_theta_gauss_order
        or truth_phi_gauss_order <= response_phi_gauss_order
    ):
        raise ValueError("truth quadrature orders must exceed response quadrature orders")
    expected_total_count = tuple(int(value) for value in profiles["expected_total_count"])
    expected_nonzero_count = tuple(int(value) for value in profiles["expected_nonzero_count"])
    expected_m0_count = tuple(int(value) for value in profiles["expected_m0_count"])
    expected_catalog_revision = tuple(str(value) for value in profiles["expected_catalog_revision"])
    if not (
        len(expected_total_count)
        == len(expected_nonzero_count)
        == len(expected_m0_count)
        == len(expected_catalog_revision)
        == len(series)
    ):
        raise ValueError("expected profile-count vectors must align with the incidence series")
    for series_index, (inputs, incidence_value) in enumerate(
        zip(series, mosaic_case["incidence_angles_deg"], strict=True)
    ):
        incidence_deg = float(incidence_value)
        frame, definitions = _profile_definitions(
            inputs,
            incidence_deg=incidence_deg,
            m0_observation=observation_by_incidence[incidence_deg],
            profile_config=profiles,
            two_theta_gauss_order=response_two_theta_gauss_order,
            phi_gauss_order=response_phi_gauss_order,
        )
        actual_m0_count = sum(
            definition.identity.group_key.layered_family_m == 0 for definition in definitions
        )
        actual_nonzero_count = len(definitions) - actual_m0_count
        actual_counts = (len(definitions), actual_nonzero_count, actual_m0_count)
        expected_counts = (
            expected_total_count[series_index],
            expected_nonzero_count[series_index],
            expected_m0_count[series_index],
        )
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"incidence {incidence_deg:g} profile coverage changed: "
                f"expected {expected_counts}, received {actual_counts}"
            )
        catalog_revision = ordered_intensity_profile_catalog_revision(definitions)
        if catalog_revision != expected_catalog_revision[series_index]:
            raise RuntimeError(
                f"incidence {incidence_deg:g} profile identity catalog changed: "
                f"expected {expected_catalog_revision[series_index]}, received {catalog_revision}"
            )
        context = build_nominal_ewald_context(inputs)
        compile_start = perf_counter()
        response = compile_ordered_intensity_response(
            context.geometry,
            angle_frame=frame,
            definitions=definitions,
        )
        response_compile_seconds += perf_counter() - compile_start
        response_rows.append(response)
        refined_definitions = tuple(
            replace(
                definition,
                two_theta_gauss_order=truth_two_theta_gauss_order,
                phi_gauss_order=truth_phi_gauss_order,
                excluded_phi_bin_indices=excluded,
            )
            for definition, excluded in zip(
                definitions,
                response.excluded_phi_bin_indices,
                strict=True,
            )
        )
        truth_compile_start = perf_counter()
        truth_response = compile_ordered_intensity_response(
            context.geometry,
            angle_frame=frame,
            definitions=refined_definitions,
        )
        truth_compile_seconds += perf_counter() - truth_compile_start
        if (
            truth_response.identities != response.identities
            or truth_response.excluded_phi_bin_indices != response.excluded_phi_bin_indices
            or truth_response.topology_probe_revision != response.topology_probe_revision
            or truth_response.observable_revision != response.observable_revision
        ):
            raise RuntimeError("fit and truth responses do not share one frozen topology mask")
        truth_start = perf_counter()
        refined_mass = truth_response.predict_mass_direct_A2(truth_strength)
        truth_generation_seconds += perf_counter() - truth_start
        truth_mass_rows.append(refined_mass)
        truth_accelerated_mass = truth_response.predict_mass_A2(truth)
        direct_start = perf_counter()
        response_direct_mass = response.predict_mass_direct_A2(truth_strength)
        direct_seconds += perf_counter() - direct_start
        accelerated_start = perf_counter()
        response_accelerated_mass = response.predict_mass_A2(truth)
        accelerated_seconds += perf_counter() - accelerated_start
        kernel_oracle_relative_error = max(
            kernel_oracle_relative_error,
            float(np.max(np.abs(truth_accelerated_mass / refined_mass - 1.0))),
            float(np.max(np.abs(response_accelerated_mass / response_direct_mass - 1.0))),
        )
        response_relative_error = np.abs(response_direct_mass / refined_mass - 1.0)
        worst_index = int(np.argmax(response_relative_error))
        identity = response.identities[worst_index]
        response_quadrature_rows.append(
            {
                "dataset_id": response.dataset_id,
                "maximum_relative_error": float(response_relative_error[worst_index]),
                "worst_profile": {
                    "group_id": identity.group_key.group_id,
                    "family_m": identity.group_key.layered_family_m,
                    "integer_L": identity.group_key.layered_integer_L,
                    "branch_id": identity.branch_id,
                    "analytic_branch_id": identity.analytic_branch_id,
                },
            }
        )
        for u_radial_A2, u_normal_A2 in displacement_probe_A2:
            response_basis = _integrated_occupancy_basis(
                response,
                u_radial_A2=u_radial_A2,
                u_normal_A2=u_normal_A2,
            )
            truth_basis = _integrated_occupancy_basis(
                truth_response,
                u_radial_A2=u_radial_A2,
                u_normal_A2=u_normal_A2,
            )
            truth_norm = np.linalg.norm(truth_basis, axis=1)
            scale = max(float(np.max(truth_norm)), np.finfo(np.float64).tiny)
            basis_relative_error = np.linalg.norm(
                response_basis - truth_basis,
                axis=1,
            ) / np.maximum(truth_norm, 1.0e-14 * scale)
            worst_index = int(np.argmax(basis_relative_error))
            identity = response.identities[worst_index]
            basis_quadrature_rows.append(
                {
                    "dataset_id": response.dataset_id,
                    "u_radial_A2": u_radial_A2,
                    "u_normal_A2": u_normal_A2,
                    "maximum_relative_error": float(basis_relative_error[worst_index]),
                    "worst_profile_group_id": identity.group_key.group_id,
                }
            )
        truth_response_revisions.append(truth_response.response_revision)
        truth_response_term_counts.append(int(truth_response.term_L.size))
        profile_counts.append(
            {
                "total": len(definitions),
                "m0": actual_m0_count,
                "nonzero": actual_nonzero_count,
                "profile_catalog_revision": catalog_revision,
                "excluded_phi_bins": sum(
                    len(indices) for indices in response.excluded_phi_bin_indices
                ),
                "profiles_with_exclusions": sum(
                    bool(indices) for indices in response.excluded_phi_bin_indices
                ),
            }
        )
        del (
            truth_response,
            truth_accelerated_mass,
            response_direct_mass,
            response_accelerated_mass,
            response_basis,
            truth_basis,
        )
        gc.collect()
    responses = tuple(response_rows)
    truth_mass = tuple(truth_mass_rows)
    compile_seconds = response_compile_seconds + truth_compile_seconds
    maximum_response_quadrature_relative_error = max(
        row["maximum_relative_error"] for row in response_quadrature_rows
    )
    maximum_basis_quadrature_relative_error = max(
        row["maximum_relative_error"] for row in basis_quadrature_rows
    )
    truth_observations = tuple(
        OrderedIntensityObservations(
            dataset_id=response.dataset_id,
            observable_revision=response.observable_revision,
            mass_A2=mass,
        )
        for response, mass in zip(responses, truth_mass, strict=True)
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
    )
    absolute_seconds = perf_counter() - absolute_start

    relative_config = case["relative_mode"]
    if relative_config["fixed_occupancy"] != "bi_occupancy":
        raise ValueError("the first relative proof fixes Bi occupancy as its declared gauge")
    synthetic_scales = np.asarray(relative_config["synthetic_image_scales"], dtype=np.float64)
    relative_mass = tuple(
        scale * mass for scale, mass in zip(synthetic_scales, truth_mass, strict=True)
    )
    relative_observations = tuple(
        OrderedIntensityObservations(
            dataset_id=response.dataset_id,
            observable_revision=response.observable_revision,
            mass_A2=mass,
        )
        for response, mass in zip(responses, relative_mass, strict=True)
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
        rod_index_by_hk = {(rod.h, rod.k): index for index, rod in enumerate(response.rods)}
        for h_value, k_value, ell_value in zip(
            held_out_h,
            held_out_k,
            held_out_l,
            strict=True,
        ):
            rod_index = rod_index_by_hk.get((int(h_value), int(k_value)))
            if rod_index is not None and np.any(
                (response.term_rod_index == rod_index)
                & np.isclose(response.term_L, ell_value, rtol=0.0, atol=1.0e-12)
            ):
                raise ValueError("a declared held-out h,k,L point occurs in a fitted response")
    k_norm = responses[0].k_norm_Ainv
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
        and maximum_response_quadrature_relative_error
        <= float(acceptance["maximum_response_quadrature_relative_error"])
        and maximum_basis_quadrature_relative_error
        <= float(acceptance["maximum_response_quadrature_relative_error"])
        and kernel_oracle_relative_error <= 1.0e-11
        and absolute.sensitivity_condition <= float(acceptance["maximum_sensitivity_condition"])
        and relative.sensitivity_condition <= float(acceptance["maximum_sensitivity_condition"])
        and not np.any(absolute.active_bounds)
        and not np.any(relative.active_bounds)
    )
    return {
        "accepted": accepted,
        "positions_frozen": True,
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
        "truth_generation": "full_structure_strength_on_refined_frozen_detector_response.v2",
        "kernel_oracle_maximum_relative_error": kernel_oracle_relative_error,
        "response_quadrature_convergence": {
            "response_two_theta_gauss_order": response_two_theta_gauss_order,
            "truth_two_theta_gauss_order": truth_two_theta_gauss_order,
            "response_phi_gauss_order": response_phi_gauss_order,
            "truth_phi_gauss_order": truth_phi_gauss_order,
            "maximum_relative_error": maximum_response_quadrature_relative_error,
            "per_dataset": response_quadrature_rows,
            "occupancy_quadratic_basis_maximum_relative_error": (
                maximum_basis_quadrature_relative_error
            ),
            "occupancy_quadratic_basis_probes": basis_quadrature_rows,
        },
        "equivalent_work_seconds": {
            "direct_full_structure": direct_seconds,
            "cached_quadratic_q": accelerated_seconds,
            "speedup": direct_seconds / accelerated_seconds,
        },
        "response_revisions": [response.response_revision for response in responses],
        "truth_response_revisions": truth_response_revisions,
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
        "profile_counts": profile_counts,
        "truth_mass_A2": {
            "minimum": min(float(np.min(mass)) for mass in truth_mass),
            "maximum": max(float(np.max(mass)) for mass in truth_mass),
            "dynamic_range": max(float(np.max(mass)) for mass in truth_mass)
            / min(float(np.min(mass)) for mass in truth_mass),
            "all_profiles_retained": sum(mass.size for mass in truth_mass),
        },
        "response_term_counts": [int(response.term_L.size) for response in responses],
        "truth_response_term_counts": truth_response_term_counts,
        "compile_seconds": compile_seconds,
        "response_compile_seconds": response_compile_seconds,
        "truth_compile_seconds": truth_compile_seconds,
        "truth_generation_seconds": truth_generation_seconds,
        "total_seconds": perf_counter() - start,
        "peak_memory_bytes": peak_memory_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    result = run_recovery(arguments.case.resolve())
    if arguments.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
