"""Compact proof for fixed-position ordered-intensity recovery."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from painted_ewald import Rod
from rasim_next.fitting.mosaic import MosaicProfileIdentity, MosaicReflectionGroupKey
from rasim_next.fitting.ordered_intensity import (
    OrderedIntensityDatasetResponse,
    OrderedIntensityIdentifiabilityError,
    OrderedIntensityObservations,
    fit_ordered_intensity_series,
    ordered_intensity_structure_model_revision,
)
from rasim_next.ordered import Bi2X3QuintupleLayerParameters
from rasim_next.pipeline.bragg_space import Bi2X3FiniteStackStrength
from rasim_next.pipeline.configured_simulation import (
    build_configured_simulation_inputs,
    load_simulation_config,
)


def _compact_response(
    *,
    dataset_id: str,
    incidence_deg: float,
    scale: float,
    strength: Bi2X3FiniteStackStrength,
    k_norm_Ainv: float,
    fixed: Bi2X3QuintupleLayerParameters,
) -> OrderedIntensityDatasetResponse:
    rods = (Rod(-1, 0), Rod(-1, -1))
    term_rod = np.asarray((0, 0, 0, 1, 1), dtype=np.int64)
    term_l = np.asarray((4.0, 8.0, 11.0, 4.0, 5.0))
    term_h = np.asarray([rods[index].h for index in term_rod], dtype=np.int32)
    term_k = np.asarray([rods[index].k for index in term_rod], dtype=np.int32)
    quadratic = strength.fixed_position_occupancy_quadratic(
        h=term_h,
        k=term_k,
        L=term_l,
        k_norm_Ainv=k_norm_Ainv,
    )
    q_crystal = np.column_stack((term_h, term_k, term_l)) @ strength.reciprocal_basis_Ainv.T
    normal = np.cross(
        strength.crystal.direct_basis_A[:, 0],
        strength.crystal.direct_basis_A[:, 1],
    )
    normal /= np.linalg.norm(normal)
    q_normal_squared = (q_crystal @ normal) ** 2
    q_radial_squared = np.maximum(
        np.einsum("ij,ij->i", q_crystal, q_crystal, optimize=True) - q_normal_squared,
        0.0,
    )
    incidence = math.radians(incidence_deg)
    identities = tuple(
        MosaicProfileIdentity(
            dataset_id=dataset_id,
            incidence_angle_rad=incidence,
            group_key=MosaicReflectionGroupKey(
                group_id=f"proof:m={rods[rod_index].family_m}:L={ell:g}",
                rod_catalog_revision="ordered-intensity-proof-rods.v1",
                member_rod_hk=((rods[rod_index].h, rods[rod_index].k),),
                branch_mode="EXPLICIT_NONZERO",
                layered_family_m=rods[rod_index].family_m,
                layered_integer_L=int(ell),
            ),
            branch_id=2,
            analytic_branch_id=2,
        )
        for rod_index, ell in zip(term_rod, term_l, strict=True)
    )
    return OrderedIntensityDatasetResponse(
        dataset_id=dataset_id,
        incidence_angle_rad=incidence,
        identities=identities,
        rods=rods,
        term_observation_index=np.arange(term_l.size),
        term_rod_index=term_rod,
        term_L=term_l,
        term_fixed_mass_per_strength=scale * np.linspace(0.8, 1.2, term_l.size),
        term_root_sign=np.ones(term_l.size, dtype=np.int8),
        term_occupancy_quadratic_strength_A2=quadratic,
        term_q_radial_squared_Ainv2=q_radial_squared,
        term_q_normal_squared_Ainv2=q_normal_squared,
        normalization_mass_px2=np.ones(term_l.size),
        reciprocal_basis_Ainv=strength.reciprocal_basis_Ainv,
        k_norm_Ainv=k_norm_Ainv,
        fixed_structure_parameters=fixed,
        rod_catalog_revision="ordered-intensity-proof-rods.v1",
        structure_model_revision=ordered_intensity_structure_model_revision(strength),
        mosaic_model_revision="ordered-intensity-proof-mosaic.v1",
        angle_frame_revision="ordered-intensity-proof-angle-frame.v1",
        source_revision="ordered-intensity-proof-source.v1",
        sample_geometry_revision="ordered-intensity-proof-sample.v1",
        material_revision="ordered-intensity-proof-material.v1",
        observable_revision=f"ordered-intensity-proof-observable-{incidence_deg:g}.v1",
        excluded_phi_bin_indices=((),) * len(identities),
        topology_probe_revision="ordered-intensity-proof-topology.v1",
    )


def _parameter_values(parameters: Bi2X3QuintupleLayerParameters) -> np.ndarray:
    return np.asarray(
        (
            parameters.bi_occupancy,
            parameters.se1_occupancy,
            parameters.se2_occupancy,
            parameters.u_radial_A2,
            parameters.u_normal_A2,
        )
    )


def run_ordered_intensity_proof(*, allow_missing_pack: bool = False) -> dict[str, Any]:
    """Recover absolute and scale-gauged structure parameters from three views."""

    del allow_missing_pack
    root = Path(__file__).resolve().parents[3]
    inputs = build_configured_simulation_inputs(
        load_simulation_config(
            root / "configs" / "bi2se3_simulation.yaml",
            repository_root=root,
        )
    )
    strength = inputs.strength
    fixed = Bi2X3QuintupleLayerParameters.from_crystal(inputs.crystal)
    responses = tuple(
        _compact_response(
            dataset_id=f"ordered-proof-{incidence_deg:g}",
            incidence_deg=incidence_deg,
            scale=scale,
            strength=strength,
            k_norm_Ainv=inputs.bragg_space.config.k_norm_Ainv,
            fixed=fixed,
        )
        for incidence_deg, scale in zip((5.0, 10.0, 15.0), (1.0, 1.2, 1.4), strict=True)
    )
    truth = replace(
        fixed,
        bi_occupancy=0.94,
        se1_occupancy=0.78,
        se2_occupancy=0.86,
        u_radial_A2=0.007,
        u_normal_A2=0.034,
    )
    truth_strength = replace(strength, structure_parameters=truth)
    observed = tuple(response.predict_mass_direct_A2(truth_strength) for response in responses)
    observations = tuple(
        OrderedIntensityObservations(
            dataset_id=response.dataset_id,
            observable_revision=response.observable_revision,
            mass_A2=mass,
        )
        for response, mass in zip(responses, observed, strict=True)
    )
    active_absolute = (
        "bi_occupancy",
        "se1_occupancy",
        "se2_occupancy",
        "u_radial_A2",
        "u_normal_A2",
    )
    absolute = fit_ordered_intensity_series(
        responses,
        observations,
        base_strength=strength,
        active_parameter_names=active_absolute,
        initial_parameters=fixed,
        relative_scale_mode=False,
    )
    image_scale = np.asarray((0.61, 1.37, 2.23))
    relative_observed = tuple(
        scale * mass for scale, mass in zip(image_scale, observed, strict=True)
    )
    relative_observations = tuple(
        OrderedIntensityObservations(
            dataset_id=response.dataset_id,
            observable_revision=response.observable_revision,
            mass_A2=mass,
        )
        for response, mass in zip(responses, relative_observed, strict=True)
    )
    relative = fit_ordered_intensity_series(
        responses,
        relative_observations,
        base_strength=strength,
        active_parameter_names=(
            "se1_occupancy",
            "se2_occupancy",
            "u_radial_A2",
            "u_normal_A2",
        ),
        initial_parameters=fixed,
        relative_scale_mode=True,
    )
    gauge_rejected = False
    try:
        fit_ordered_intensity_series(
            responses,
            relative_observations,
            base_strength=strength,
            active_parameter_names=active_absolute,
        )
    except OrderedIntensityIdentifiabilityError:
        gauge_rejected = True

    absolute_error = float(
        np.max(
            np.abs(_parameter_values(absolute.structure_representative) - _parameter_values(truth))
        )
    )
    relative_target = np.asarray(
        (
            1.0,
            truth.se1_occupancy / truth.bi_occupancy,
            truth.se2_occupancy / truth.bi_occupancy,
        )
    )
    relative_error = float(
        max(
            np.max(np.abs(relative.occupancy_ratios - relative_target)),
            abs(relative.structure_representative.u_radial_A2 - truth.u_radial_A2),
            abs(relative.structure_representative.u_normal_A2 - truth.u_normal_A2),
        )
    )
    expected_scales = image_scale * truth.bi_occupancy**2
    scale_error = float(np.max(np.abs(relative.dataset_scales - expected_scales)))

    holdout_h = np.asarray((-1, -1), dtype=np.int32)
    holdout_k = np.asarray((0, -1), dtype=np.int32)
    holdout_l = np.asarray((5.0, 10.0))
    truth_holdout = replace(strength, structure_parameters=truth).evaluate_hkl(
        h=holdout_h,
        k=holdout_k,
        L=holdout_l,
        k_norm_Ainv=inputs.bragg_space.config.k_norm_Ainv,
    )
    fit_holdout = replace(
        strength,
        structure_parameters=absolute.structure_representative,
    ).evaluate_hkl(
        h=holdout_h,
        k=holdout_k,
        L=holdout_l,
        k_norm_Ainv=inputs.bragg_space.config.k_norm_Ainv,
    )
    holdout_relative_error = float(np.max(np.abs(fit_holdout / truth_holdout - 1.0)))
    passed = (
        gauge_rejected
        and absolute.structure_representative.bi_fractional_z == fixed.bi_fractional_z
        and absolute.structure_representative.se2_fractional_z == fixed.se2_fractional_z
        and absolute_error <= 2.0e-10
        and relative_error <= 2.0e-10
        and scale_error <= 2.0e-10
        and holdout_relative_error <= 2.0e-10
        and absolute.sensitivity_rank == 5
        and relative.sensitivity_rank == 4
        and not np.any(absolute.active_bounds)
        and not np.any(relative.active_bounds)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "positions_frozen": True,
        "absolute_max_parameter_error": absolute_error,
        "relative_max_parameter_error": relative_error,
        "relative_scale_max_error": scale_error,
        "held_out_max_relative_error": holdout_relative_error,
        "absolute_rank": absolute.sensitivity_rank,
        "relative_rank": relative.sensitivity_rank,
        "absolute_condition": absolute.sensitivity_condition,
        "relative_condition": relative.sensitivity_condition,
        "absolute_active_bounds": absolute.active_bounds.tolist(),
        "relative_active_bounds": relative.active_bounds.tolist(),
        "common_occupancy_scale_gauge_rejected": gauge_rejected,
    }


__all__ = ["run_ordered_intensity_proof"]
