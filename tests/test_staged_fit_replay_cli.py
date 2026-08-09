from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from packaging.markers import Marker

ROOT = Path(__file__).resolve().parents[1]


def _load_replay_cli() -> ModuleType:
    script = ROOT / "scripts" / "replay_staged_fit.py"
    specification = importlib.util.spec_from_file_location("staged_fit_replay_cli_test", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _stage_envelope(
    module: ModuleType,
    case,
    stage: str,
    upstream: dict | None,
    *,
    summary: dict,
    state: dict,
) -> dict:
    geometry_source_revisions = {
        "Bi2Se3": "86a5a9f191688065c8265df9bf47175a1b070115a4ce20129a321cd79d2dc416",
        "Bi2Te3": "3a68902a791bd9e28d50205b4596738b19a6ea36ce169fe4ff39441798a24659",
    }
    source_state_count = 1 if stage == "geometry" else case.source_state_count
    source_revision = (
        geometry_source_revisions[case.material_id]
        if stage == "geometry"
        else case.expected_scientific_summary["source_revision"]
    )
    payload = {
        "schema_version": module._STAGE_SCHEMA_VERSION,
        "stage": stage,
        "case_id": case.case_id,
        "material_id": case.material_id,
        "stage_case_sha256": module._stage_case_sha256(case, stage),
        "execution_backend": "cuda",
        "runtime": case.runtime_identity,
        "source_state_count": source_state_count,
        "source_seed": case.source_seed,
        "source_revision": source_revision,
        "upstream_scientific_revision": (
            None if upstream is None else upstream["scientific_revision"]
        ),
        "scientific_summary": {
            "case_id": case.case_id,
            "material_id": case.material_id,
            "source_state_count": case.source_state_count,
            "source_revision": case.expected_scientific_summary["source_revision"],
            stage: summary,
        },
        "state": state,
    }
    payload["scientific_revision"] = module.scientific_revision(stage, payload)
    return payload


def _bi2se3_geometry_state(case) -> dict:
    geometry = case.expected_scientific_summary["geometry"]
    return {
        "corrections": list(geometry["corrections"]),
        "incidence_angle_delta_rad": geometry["incidence_angle_delta_rad"],
        "incidence_angles": [
            {
                "image_id": image_id,
                "commanded_angle_rad": commanded,
                "effective_angle_rad": effective,
            }
            for image_id, commanded, effective in zip(
                geometry["incidence_angle_image_ids"],
                geometry["commanded_incidence_angles_rad"],
                geometry["effective_incidence_angles_rad"],
                strict=True,
            )
        ],
        "fit_evidence": {
            "indexed_manifest_hash": geometry["selection_revision"],
            "qualification": {"accepted": True},
        },
    }


def _bi2se3_fixed_position(case, geometry_stage: dict) -> dict:
    delta_rad = float(geometry_stage["state"]["incidence_angle_delta_rad"])
    commanded = [float(value) for value in case.incidence_angles_deg]
    return {
        "position_artifact_revision": geometry_stage["scientific_revision"],
        "corrections": {
            name: float(value)
            for name, value in zip(
                (
                    "detector_column_tilt_rad",
                    "detector_row_tilt_rad",
                    "sample_normal_x_tilt_rad",
                    "sample_normal_y_tilt_rad",
                    "goniometer_axis_pitch_rad",
                    "goniometer_axis_yaw_rad",
                    "sample_plane_normal_offset_m",
                    "goniometer_pivot_pitch_offset_m",
                    "goniometer_pivot_yaw_offset_m",
                ),
                geometry_stage["state"]["corrections"],
                strict=True,
            )
        },
        "incidence_angle_model_id": "commanded_angle_plus_common_delta.v1",
        "incidence_angle_delta_rad": delta_rad,
        "commanded_incidence_angles_deg": commanded,
        "effective_incidence_angles_deg": [value + math.degrees(delta_rad) for value in commanded],
        "beam_center_column_row_px": [1453.12, 1596.422],
        "geometry_parameters_fitted_here": False,
    }


def _profile_record(identity: str) -> dict:
    dataset_id, family_m, integer_l, analytic_branch, side = identity.split("|")
    return {
        "dataset_id": dataset_id,
        "family_m": int(family_m),
        "integer_L": int(integer_l),
        "analytic_branch_id": int(analytic_branch),
        "root_side_branch_id": None if side == "none" else int(side),
    }


def test_bi2te3_mosaic_inputs_preserve_effective_incidence_angles() -> None:
    import numpy as np

    from rasim_next.fitting import FixedLatticeState, FixedPositionState
    from rasim_next.fitting.indexed_series import (
        SharedGeometryCorrections,
        zero_sum_helmert_basis,
    )
    from rasim_next.materials import read_crystal
    from rasim_next.pipeline.configured_simulation import load_simulation_config

    module = _load_replay_cli()
    case = SimpleNamespace(
        path=ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml",
        input_paths={"simulation_config": ROOT / "configs" / "bi2te3_simulation.yaml"},
        incidence_angles_deg=(5.0, 10.0, 15.0),
        source_state_count=3,
    )
    config = load_simulation_config(case.input_paths["simulation_config"])
    image_ids = ("Bi2Te3_5m_5d", "Bi2Te3_10d_5m", "Bi2Te3_15d_5m")
    trims = (2.0e-4, -3.5e-4, 1.5e-4)
    trim_by_id = dict(zip(image_ids, trims, strict=True))
    sorted_trims = np.asarray([trim_by_id[image_id] for image_id in sorted(image_ids)])
    contrasts = zero_sum_helmert_basis(3).T @ sorted_trims
    position = FixedPositionState(
        artifact_revision=f"sha256-{'a' * 64}",
        corrections=SharedGeometryCorrections.from_array(np.zeros(9)),
        incidence_angle_delta_rad=1.4e-4,
        commanded_incidence_angles_rad=tuple(
            math.radians(value) for value in case.incidence_angles_deg
        ),
        beam_center_column_row_px=tuple(
            float(value) for value in config.instrument.detector_reference_coordinate_px
        ),
        incidence_angle_image_ids=image_ids,
        incidence_angle_trim_rad=trims,
        incidence_angle_trim_contrast_rad=tuple(float(value) for value in contrasts),
        incidence_angle_trim_prior_sigma_rad=math.radians(0.05),
        incidence_angle_trim_contrast_half_span_rad=math.radians(0.12),
    )
    crystal = read_crystal(
        config.material.cif_path,
        phase_id=config.material.phase_id,
        expected_sha256=config.cif_sha256,
    )
    lattice = FixedLatticeState.implicit_cif(crystal.direct_basis_A)

    _, series, nominal_series = module._bi2te3_fixed_inputs(
        case,
        position,
        lattice,
    )

    expected_deg = np.degrees(position.effective_incidence_angles_rad)
    for candidate in (series, nominal_series):
        actual_deg = [inputs.config.instrument.axis_rotations[0].angle_deg for inputs in candidate]
        assert np.allclose(actual_deg, expected_deg, rtol=0.0, atol=1.0e-13)


def test_replay_recovers_only_an_explicit_global_mosaic_alias() -> None:
    class AliasError(ValueError):
        pass

    module = _load_replay_cli()
    result = SimpleNamespace(
        gaussian_sigma_rad=math.radians(0.4),
        lorentzian_half_width_rad=math.radians(0.15),
        lorentzian_probability=0.7,
        objective=3.5,
    )
    competing = (
        SimpleNamespace(
            gaussian_sigma_rad=result.gaussian_sigma_rad,
            lorentzian_half_width_rad=result.lorentzian_half_width_rad,
            lorentzian_probability=result.lorentzian_probability,
            objective=result.objective,
        ),
        SimpleNamespace(
            gaussian_sigma_rad=result.gaussian_sigma_rad,
            lorentzian_half_width_rad=math.radians(0.17),
            lorentzian_probability=result.lorentzian_probability,
            objective=result.objective,
        ),
    )
    error = AliasError("global alias")
    error.reason = "global_alias"
    error.candidate_result = result
    error.competing_parameter_sets = competing

    recovered, records = module._recover_global_mosaic_alias(error)

    assert recovered is result
    assert records[0] == {
        "gaussian_sigma_deg": pytest.approx(0.4),
        "lorentzian_hwhm_deg": pytest.approx(0.15),
        "lorentzian_probability": 0.7,
        "objective": 3.5,
    }
    for reason, candidate in (("local_sensitivity", result), ("global_alias", None)):
        rejected = AliasError(reason)
        rejected.reason = reason
        rejected.candidate_result = candidate
        rejected.competing_parameter_sets = competing
        with pytest.raises(AliasError, match=reason):
            module._recover_global_mosaic_alias(rejected)


def test_mosaic_component_checkpoint_reuses_only_exact_profiles(tmp_path: Path) -> None:
    import numpy as np

    from rasim_next.fitting import (
        MosaicProfileIdentity,
        MosaicProfileSet,
        MosaicReflectionGroupKey,
    )

    module = _load_replay_cli()
    identity = MosaicProfileIdentity(
        dataset_id="osc-5deg",
        incidence_angle_rad=math.radians(5.0),
        group_key=MosaicReflectionGroupKey(
            group_id="00L-3",
            rod_catalog_revision="catalog.v1",
            member_rod_hk=((0, 0),),
            branch_mode="COLLAPSED_00L",
            layered_family_m=0,
            layered_integer_L=3,
        ),
        branch_id=None,
    )
    phi_edges = np.asarray(((-0.2, -0.1, 0.0, 0.1, 0.2),))
    two_theta_bounds = np.asarray(((0.05, 0.08),))
    valid = np.ones((1, 4), dtype=np.bool_)
    observations = MosaicProfileSet(
        identities=(identity,),
        signal=np.ones((1, 4)),
        normalization=np.ones((1, 4)),
        valid=valid,
        profile_revision="profiles.v1",
        phi_bin_edges_rad=phi_edges,
        two_theta_bounds_rad=two_theta_bounds,
        angle_frame_revisions=("frame.v1",),
        source_revision=None,
        observation_revision="observations.v1",
    )
    calls: list[float] = []

    def evaluate(width_rad: float) -> MosaicProfileSet:
        calls.append(width_rad)
        return MosaicProfileSet(
            identities=(identity,),
            signal=np.full((1, 4), width_rad),
            normalization=np.ones((1, 4)),
            valid=valid,
            profile_revision="profiles.v1",
            phi_bin_edges_rad=phi_edges,
            two_theta_bounds_rad=two_theta_bounds,
            angle_frame_revisions=("frame.v1",),
            source_revision="source.v1",
            execution_backend="numba_cpu_source_averaged.v1",
        )

    width = math.radians(0.4)
    cached = module._checkpointed_mosaic_component_evaluator(
        evaluate,
        cache_directory=tmp_path,
        component_kind="gaussian",
        cache_revision="cache.v1",
        observations=observations,
        source_revision="source.v1",
        execution_backend="numba_cpu_source_averaged.v1",
    )
    first = cached(width)
    second = cached(width)
    assert calls == [width]
    assert np.array_equal(second.signal, first.signal)

    def must_not_run(_width_rad: float) -> MosaicProfileSet:
        raise AssertionError("exact cached profile was recomputed")

    resumed = module._checkpointed_mosaic_component_evaluator(
        must_not_run,
        cache_directory=tmp_path,
        component_kind="gaussian",
        cache_revision="cache.v1",
        observations=observations,
        source_revision="source.v1",
        execution_backend="numba_cpu_source_averaged.v1",
    )
    reloaded = resumed(width)
    assert np.array_equal(reloaded.signal, first.signal)
    assert reloaded.source_revision == "source.v1"

    changed_calls: list[float] = []

    def changed(width_rad: float) -> MosaicProfileSet:
        changed_calls.append(width_rad)
        return evaluate(width_rad)

    invalidated = module._checkpointed_mosaic_component_evaluator(
        changed,
        cache_directory=tmp_path,
        component_kind="gaussian",
        cache_revision="cache.v2",
        observations=observations,
        source_revision="source.v1",
        execution_backend="numba_cpu_source_averaged.v1",
    )
    invalidated(width)
    assert changed_calls == [width]


def _mosaic_artifact_document(case, fixed_position: dict) -> dict:
    summary = case.expected_scientific_summary["mosaic"]
    profiles = [
        {**_profile_record(identity), "nuisance_peak_scale": float(index + 1)}
        for index, identity in enumerate(summary["profile_identities"])
    ]
    return {
        "schema_version": "rasim-bi2se3-real-mosaic-fit-v3",
        "status": summary["classification"],
        "recovered_effective_distribution": {
            name: value
            for name, value in zip(
                (
                    "gaussian_sigma_deg",
                    "lorentzian_hwhm_deg",
                    "lorentzian_probability",
                ),
                summary["parameters"],
                strict=True,
            )
        },
        "fit": {
            "objective": summary["objective"],
            "sensitivity_rank": summary["rank"],
            "profiles": profiles,
        },
        "fixed_geometry": fixed_position,
        "observations": {
            "measured_profile_policy": {
                "profile_selection": [{**record, "fit_eligible": True} for record in profiles]
            }
        },
        "source_model": {
            "sample_count": case.source_state_count,
            "source_seed": case.source_seed,
            "source_revision": case.expected_scientific_summary["source_revision"],
            "reduction": "one_incoherent_weighted_detector_function_per_incidence.v1",
        },
        "provenance": {
            "case_sha256": hashlib.sha256(case.input_paths["mosaic_case"].read_bytes()).hexdigest()
        },
    }


def _ordered_artifact_document(
    case,
    fixed_position: dict,
    *,
    upstream_mosaic_sha256: str,
) -> dict:
    summary = case.expected_scientific_summary["ordered_intensity"]
    stage = case.stage_config["ordered_intensity"]
    active_names = tuple(stage["active_parameters"])
    parameters = dict(zip(active_names, summary["parameters"], strict=True))
    occupancy_ratios = [
        1.0,
        float(parameters["se1_over_bi"]),
        float(parameters["se2_over_bi"]),
    ]
    occupancy_scale = max(occupancy_ratios)
    representative = {
        "bi_fractional_z": 0.4,
        "se2_fractional_z": 0.2,
        "bi_occupancy": occupancy_ratios[0] / occupancy_scale,
        "se1_occupancy": occupancy_ratios[1] / occupancy_scale,
        "se2_occupancy": occupancy_ratios[2] / occupancy_scale,
        "u_radial_A2": float(parameters["u_radial_A2"]),
        "u_normal_A2": float(parameters["u_normal_A2"]),
    }
    mosaic_parameters = case.expected_scientific_summary["mosaic"]["parameters"]
    profile_residuals = {
        "Bi2Se3-5deg|0|3|0|none": 0.02680501007149738,
        "Bi2Se3-5deg|0|6|0|none": -0.028325873760739917,
        "Bi2Se3-10deg|0|6|0|none": 0.02161772043631527,
        "Bi2Se3-10deg|1|5|2|1": -0.9946328222607105,
        "Bi2Se3-10deg|1|5|2|2": -0.9952219467189322,
        "Bi2Se3-10deg|1|10|2|1": -0.9937863383945027,
        "Bi2Se3-10deg|1|10|2|2": -0.9941493705066183,
        "Bi2Se3-15deg|0|6|0|none": 0.14116163311273988,
        "Bi2Se3-15deg|0|9|0|none": -0.03324295341909733,
        "Bi2Se3-15deg|1|5|2|1": -0.9901540883045857,
        "Bi2Se3-15deg|1|5|2|2": -0.9921841253426466,
        "Bi2Se3-15deg|1|10|2|1": -0.9871509766126202,
        "Bi2Se3-15deg|1|10|2|2": -0.9877504441783425,
        "Bi2Se3-15deg|1|11|2|1": -0.9517192022755053,
        "Bi2Se3-15deg|1|11|2|2": -0.9574784898277218,
    }
    scale_by_identity = {
        identity: float(index + 1)
        for index, identity in enumerate(
            case.expected_scientific_summary["mosaic"]["profile_identities"]
        )
    }
    profiles = [
        {
            **_profile_record(identity),
            "nuisance_peak_scale": scale_by_identity[identity],
            "relative_residual": residual,
        }
        for identity, residual in profile_residuals.items()
    ]
    family_residual = {
        str(family_m): {
            "count": len(values),
            "sum_squared": sum(value * value for value in values),
            "root_mean_square": math.sqrt(sum(value * value for value in values) / len(values)),
            "maximum_absolute": max(abs(value) for value in values),
        }
        for family_m in (0, 1)
        for values in [
            [record["relative_residual"] for record in profiles if record["family_m"] == family_m]
        ]
    }
    overall_residual_rms = float(summary["relative_residual_rms"])
    fit_recipe = {
        "revision": "bi2se3_measured_relative_structure_multistart.v1",
        "active_parameter_names": list(active_names),
        "canonical_active_parameter_names": [
            "se1_occupancy",
            "se2_occupancy",
            "u_radial_A2",
            "u_normal_A2",
        ],
        "occupancy_ratio_reference": "bi_occupancy",
        "relative_scale_mode": True,
        "lower_bounds": [float(value) for value in stage["lower_bounds"]],
        "upper_bounds": [float(value) for value in stage["upper_bounds"]],
        "parameter_scales": [float(value) for value in stage["parameter_scales"]],
        "multistarts": [[float(value) for value in initial] for initial in stage["multistarts"]],
        "initial_occupancy_gauge": "normalize_ratio_triplet_to_maximum_one.v1",
        "maximum_function_evaluations": int(stage["maximum_function_evaluations"]),
        "multistart_selection": "objective_equivalent_then_lowest_start_index.v1",
        "objective_equivalence_tolerance_factor": 512.0,
    }
    return {
        "schema_version": "rasim-bi2se3-measured-ordered-intensity-fit-v1",
        "status": summary["classification"],
        "claim_boundary": summary["claim_boundary"],
        "positions_frozen": True,
        "observation_model": (
            "measured_mosaic_nuisance_scale_times_baseline_source_averaged_peak_center_signal.v1"
        ),
        "background_inheritance": "local_phi_constant_from_mosaic_fit.v1",
        "simulated_detector_rasterization_used_in_fit": False,
        "measured_detector_observation_source": (
            "upstream_exact_pixel_overlap_profile_amplitudes.v1"
        ),
        "fit_recipe": fit_recipe,
        "response_contract": {
            "revision": "source-averaged-selected-center-occ-quadratic-chebyshev-qz-spectral.v2",
            "signal_certificate_relative_floor": 1.0e-12,
        },
        "response_validation": {
            "maximum_allowed_relative_error": 0.002,
            "maximum_interpolation_relative_error": 1.0e-12,
            "cached_vs_fresh_maximum_relative_error": 1.0e-12,
        },
        "response_execution": [
            {
                "dataset_id": dataset_id,
                "backend": "numba_cuda_source_averaged.v1",
            }
            for dataset_id in ("Bi2Se3-5deg", "Bi2Se3-10deg", "Bi2Se3-15deg")
        ],
        "fit": {
            "active_parameter_names": list(active_names),
            "occupancy_ratio_reference": "bi_occupancy",
            "parameters": parameters,
            "structure_representative": representative,
            "objective": summary["objective"],
            "residual_summary": {"root_mean_square": overall_residual_rms},
            "sensitivity_rank": summary["rank"],
            "active_bounds": dict(zip(active_names, summary["active_bounds"], strict=True)),
            "profiles": profiles,
            "multistart_failures": [],
            "multistarts": [{"start_index": index} for index, _ in enumerate(stage["multistarts"])],
        },
        "fit_adequacy": {
            "qualified": False,
            "classification": summary["adequacy"],
            "objective_measure": "unweighted_sum_squared_relative_residual.v1",
            "overall_relative_residual_rms": overall_residual_rms,
            "relative_residual_by_family": family_residual,
            "parameters_on_bounds": [
                name
                for name, active in zip(active_names, summary["active_bounds"], strict=True)
                if active
            ],
        },
        "fixed_position": fixed_position,
        "fixed_mosaic": dict(
            zip(
                (
                    "gaussian_sigma_deg",
                    "lorentzian_hwhm_deg",
                    "lorentzian_probability",
                ),
                mosaic_parameters,
                strict=True,
            )
        ),
        "source_model": {
            "sample_count": case.source_state_count,
            "source_seed": case.source_seed,
            "source_revision": case.expected_scientific_summary["source_revision"],
            "reduction": "one_incoherent_weighted_detector_function_per_incidence.v1",
        },
        "provenance": {
            "ordered_case_sha256": hashlib.sha256(
                case.input_paths["ordered_intensity_case"].read_bytes()
            ).hexdigest(),
            "mosaic_case_sha256": hashlib.sha256(
                case.input_paths["mosaic_case"].read_bytes()
            ).hexdigest(),
            "upstream_mosaic_result_sha256": upstream_mosaic_sha256,
        },
    }


def test_measured_mosaic_ingests_one_verified_position_artifact(tmp_path: Path) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"
    )
    geometry = _stage_envelope(
        module,
        case,
        "geometry",
        None,
        summary=copy.deepcopy(case.expected_scientific_summary["geometry"]),
        state=_bi2se3_geometry_state(case),
    )
    artifact = tmp_path / "geometry.json"
    artifact.write_text(json.dumps(geometry) + "\n", encoding="utf-8")
    mosaic_runner = module._load_script_module(
        "verified_position_artifact_mosaic_test",
        "recover_bi2se3_mosaic.py",
    )
    mosaic_case_path = case.input_paths["mosaic_case"]
    mosaic_case, _, _ = mosaic_runner._case(mosaic_case_path)

    position = mosaic_runner._position_artifact_state(artifact, mosaic_case)
    assert position.artifact_revision == geometry["scientific_revision"]
    assert position.incidence_angle_delta_rad == geometry["state"]["incidence_angle_delta_rad"]
    assert position.corrections.as_array().tolist() == geometry["state"]["corrections"]

    stale_revision = copy.deepcopy(geometry)
    stale_revision["state"]["corrections"][0] += 1.0e-4
    artifact.write_text(json.dumps(stale_revision) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed its scientific revision"):
        mosaic_runner._position_artifact_state(artifact, mosaic_case)

    independent_angle = copy.deepcopy(geometry)
    independent_angle["state"]["incidence_angles"][2]["effective_angle_rad"] += 1.0e-4
    independent_angle["scientific_revision"] = module.scientific_revision(
        "geometry",
        {name: value for name, value in independent_angle.items() if name != "scientific_revision"},
    )
    artifact.write_text(json.dumps(independent_angle) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one common incidence-angle delta"):
        mosaic_runner._position_artifact_state(artifact, mosaic_case)


def test_transferred_structure_observations_follow_identity_aligned_mosaic_scales() -> None:
    module = _load_replay_cli()
    profiles = [
        {
            "dataset_id": "Bi2Se3-10deg",
            "family_m": 1,
            "integer_L": 5,
            "analytic_branch_id": 2,
            "root_side_branch_id": 1,
            "nuisance_peak_scale": 0.5,
        },
        {
            "dataset_id": "Bi2Se3-5deg",
            "family_m": 0,
            "integer_L": 3,
            "analytic_branch_id": 0,
            "root_side_branch_id": None,
            "nuisance_peak_scale": 2.0,
        },
        {
            "dataset_id": "Bi2Se3-15deg",
            "family_m": 0,
            "integer_L": 9,
            "analytic_branch_id": 0,
            "root_side_branch_id": None,
            "nuisance_peak_scale": 3.0,
        },
    ]
    scales = module._measured_profile_scales({"fit": {"profiles": profiles}})
    identity_order = (
        "Bi2Se3-15deg|0|9|0|none",
        "Bi2Se3-5deg|0|3|0|none",
        "Bi2Se3-10deg|1|5|2|1",
    )

    first = module._transferred_profile_signal(
        (4.0, 10.0, 20.0),
        identity_order,
        scales,
    )
    scales["Bi2Se3-10deg|1|5|2|1"] = 0.75
    changed = module._transferred_profile_signal(
        (4.0, 10.0, 20.0),
        identity_order,
        scales,
    )

    assert first.tolist() == [12.0, 20.0, 10.0]
    assert changed.tolist() == [12.0, 20.0, 15.0]
    with pytest.raises(ValueError, match="exactly"):
        module._transferred_profile_signal(
            (10.0,),
            ("Bi2Se3-5deg|0|3|0|none",),
            scales,
        )
    scales["Bi2Se3-5deg|0|3|0|none"] = 0.0
    with pytest.raises(ValueError, match="positive and finite"):
        module._transferred_profile_signal((4.0, 10.0, 20.0), identity_order, scales)
    duplicate = {"fit": {"profiles": [profiles[0], profiles[0]]}}
    with pytest.raises(ValueError, match="duplicate"):
        module._measured_profile_scales(duplicate)


def test_existing_ordered_artifact_cache_requires_exact_scientific_identity() -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"
    )
    fixed_position = {"position_artifact_revision": "sha256-test"}
    document = _ordered_artifact_document(
        case,
        fixed_position,
        upstream_mosaic_sha256="mosaic-sha256",
    )
    expected_profile_scales = [
        {
            "identity": module._identity_text(record),
            "nuisance_peak_scale": record["nuisance_peak_scale"],
        }
        for record in sorted(document["fit"]["profiles"], key=module._identity_text)
    ]
    projection = module._validated_bi2se3_ordered_cached_artifact(
        document,
        case=case,
        expected_provenance=document["provenance"],
        expected_fixed_mosaic=document["fixed_mosaic"],
        expected_fixed_position=fixed_position,
        expected_source_model=document["source_model"],
        expected_profile_scales=expected_profile_scales,
        expected_bi_fractional_z=0.4,
        expected_se2_fractional_z=0.2,
    )
    assert projection["state"]["profile_scales"] == expected_profile_scales

    stale_profile_scales = copy.deepcopy(expected_profile_scales)
    stale_profile_scales[0]["nuisance_peak_scale"] *= 2.0
    with pytest.raises(RuntimeError, match="does not match the current inputs"):
        module._validated_bi2se3_ordered_cached_artifact(
            document,
            case=case,
            expected_provenance=document["provenance"],
            expected_fixed_mosaic=document["fixed_mosaic"],
            expected_fixed_position=fixed_position,
            expected_source_model=document["source_model"],
            expected_profile_scales=stale_profile_scales,
            expected_bi_fractional_z=0.4,
            expected_se2_fractional_z=0.2,
        )


def test_fresh_ordered_oracle_rejects_nonfinite_response(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    import rasim_next.fitting as fitting

    module = _load_replay_cli()
    monkeypatch.setattr(
        fitting,
        "evaluate_source_averaged_ordered_intensity_point_signal",
        lambda *args, **kwargs: np.asarray([np.nan]),
    )

    with pytest.raises(FloatingPointError, match="aligned positive finite vectors"):
        module._ORDERED_STAGE.cached_vs_fresh_ordered_response_max_relative_error(
            (object(),),
            ((object(), ()),),
            (np.asarray([1.0]),),
            object(),
            backend="cpu",
        )


def test_prepared_measured_ordered_inputs_compose_with_standalone_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_replay_cli()
    prepared = SimpleNamespace(
        series=("series",),
        profile_catalogs=(("frame", ("definition",)),),
        anchor_counts=({"total": 1},),
        mosaic_parameters={"gaussian_sigma_deg": 1.0},
        source_revision="source-revision",
        fixed_position_record={"position_artifact_revision": "position-revision"},
        baseline_parameters="baseline",
    )
    mosaic_document = {
        "fit": {
            "profiles": [
                {
                    "dataset_id": "Bi2Se3-5deg",
                    "family_m": 0,
                    "integer_L": 3,
                    "analytic_branch_id": 0,
                    "root_side_branch_id": None,
                    "nuisance_peak_scale": 2.5,
                }
            ]
        }
    }
    stage = {
        "active_parameters": ["se1_over_bi", "se2_over_bi", "u_radial_A2", "u_normal_A2"],
        "lower_bounds": [0.0, 0.0, 0.0, 0.0],
        "upper_bounds": [2.0, 2.0, 0.1, 0.1],
        "parameter_scales": [1.0, 1.0, 0.1, 0.1],
        "multistarts": [[1.0, 1.0, 0.0, 0.0]],
        "maximum_function_evaluations": 10,
        "claim_boundary": "test boundary",
    }
    captured = {}

    def fake_fit(inputs):
        captured["inputs"] = inputs
        return {"artifact": "standalone"}

    monkeypatch.setattr(
        module._ORDERED_STAGE,
        "fit_bi2se3_measured_ordered_document",
        fake_fit,
    )
    result = module._ORDERED_STAGE.fit_prepared_bi2se3_measured_ordered_document(
        prepared,
        mosaic_document,
        backend="cuda",
        stage=stage,
        source_state_count=250,
        interpolation_limit=1.0e-8,
        source_model={"source_revision": "source-revision"},
        observation_model={"revision": "observation"},
        background_inheritance={"revision": "background"},
        provenance={"ordered_case_sha256": "case"},
    )

    inputs = captured["inputs"]
    assert result == {"artifact": "standalone"}
    assert inputs.series == prepared.series
    assert inputs.baseline == "baseline"
    assert inputs.required_source_revision == "source-revision"
    assert inputs.measured_scales == {"Bi2Se3-5deg|0|3|0|none": 2.5}
    assert inputs.source_state_count == 250


def test_tracked_replay_cases_are_relative_and_hash_complete(tmp_path: Path) -> None:
    module = _load_replay_cli()
    cases = tuple(
        module.load_replay_case(path)
        for path in (
            ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml",
            ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml",
        )
    )

    assert tuple(case.material_id for case in cases) == ("Bi2Se3", "Bi2Te3")
    assert all(case.source_state_count == 250 for case in cases)
    assert all(case.incidence_angles_deg == (5.0, 10.0, 15.0) for case in cases)
    assert all(path.is_relative_to(ROOT) for case in cases for path in case.input_paths.values())
    bi2te3 = cases[1]
    locked_selection = bi2te3.expected_scientific_summary["geometry"]["selection_revision"]
    historical_selection = bi2te3.stage_config["geometry"]["historical_selection_revision"]
    catalog_manifest = bi2te3.stage_config["geometry"]["catalog_manifest_revision"]
    assert locked_selection == (
        "sha256-97ba51fce133a27404d4b571e40fc21413fb13e8ccc70d725486b5f12fefff51"
    )
    assert historical_selection == (
        "sha256-79f5028d1ee7d2b3bb67bcdd15d92822d2d4b6bd43ee7fe1cb0377e43941146a"
    )
    assert catalog_manifest == (
        "sha256-4f3755acdc0c3416e05cb58778d3f07ddf12d56cb49ef3b7abbdeb057d706abc"
    )

    original = (ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml").read_text(
        encoding="utf-8"
    )
    absolute = original.replace(
        'path = "../../../configs/bi2te3_simulation.yaml"',
        f'path = "{(ROOT / "configs" / "bi2te3_simulation.yaml").as_posix()}"',
        1,
    )
    bad_case = tmp_path / "absolute.toml"
    bad_case.write_text(absolute, encoding="utf-8")
    with pytest.raises(ValueError, match="relative"):
        module.load_replay_case(bad_case, repository_root=ROOT)

    unknown = tmp_path / "unknown.toml"
    unknown.write_text("unexpected = true\n" + original, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        module.load_replay_case(unknown, repository_root=ROOT)

    nested_unknown = tmp_path / "nested_unknown.toml"
    nested_unknown.write_text(
        original.replace(
            "[expected.geometry]\n",
            "[expected.geometry]\nunverified_metric = 1\n",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"expected\.geometry contains unknown key"):
        module.load_replay_case(nested_unknown, repository_root=ROOT)

    nonfinite = tmp_path / "nonfinite.toml"
    nonfinite.write_text(
        original.replace(
            "site_rms_px = 7.175090155549409",
            "site_rms_px = nan",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="site_rms_px must be finite"):
        module.load_replay_case(nonfinite, repository_root=ROOT)

    invalid_render = tmp_path / "invalid_render.toml"
    invalid_render.write_text(
        original.replace("cuda_coordinate_chunk = 200000", "cuda_coordinate_chunk = 0", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cuda_coordinate_chunk must be a positive integer"):
        module.load_replay_case(invalid_render, repository_root=ROOT)


def test_geometry_stage_case_hash_excludes_downstream_case_fields() -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"
    )
    baseline = module._stage_case_sha256(case, "geometry")

    downstream_config = copy.deepcopy(case.stage_config)
    downstream_expected = copy.deepcopy(case.expected_scientific_summary)
    downstream_tolerances = copy.deepcopy(case.tolerances)
    downstream_files = copy.deepcopy(case.file_records)
    downstream_config["mosaic"]["observation_mode"] = "synthetic"
    downstream_expected["mosaic"]["objective"] += 1.0
    downstream_tolerances["mosaic_objective_absolute"] *= 2.0
    next(record for record in downstream_files if record["role"] == "mosaic_case")["sha256"] = (
        "0" * 64
    )
    downstream_case = module.replace(
        case,
        stage_config=downstream_config,
        expected_scientific_summary=downstream_expected,
        tolerances=downstream_tolerances,
        file_records=downstream_files,
    )
    assert module._stage_case_sha256(downstream_case, "geometry") == baseline
    assert module._stage_case_sha256(downstream_case, "mosaic") != module._stage_case_sha256(
        case, "mosaic"
    )

    geometry_config = copy.deepcopy(case.stage_config)
    geometry_config["geometry"]["benchmark"] = not geometry_config["geometry"]["benchmark"]
    assert (
        module._stage_case_sha256(module.replace(case, stage_config=geometry_config), "geometry")
        != baseline
    )

    geometry_expected = copy.deepcopy(case.expected_scientific_summary)
    geometry_expected["geometry"]["site_rms_px"] += 1.0e-12
    assert (
        module._stage_case_sha256(
            module.replace(case, expected_scientific_summary=geometry_expected), "geometry"
        )
        != baseline
    )

    geometry_tolerances = copy.deepcopy(case.tolerances)
    geometry_tolerances["geometry_metric_absolute"] *= 2.0
    assert (
        module._stage_case_sha256(module.replace(case, tolerances=geometry_tolerances), "geometry")
        != baseline
    )

    geometry_files = copy.deepcopy(case.file_records)
    next(record for record in geometry_files if record["role"] == "osc_5deg")["sha256"] = "0" * 64
    assert (
        module._stage_case_sha256(module.replace(case, file_records=geometry_files), "geometry")
        != baseline
    )


def test_bi2se3_geometry_stage_requires_one_complete_common_delta_state() -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"
    )
    summary = copy.deepcopy(case.expected_scientific_summary["geometry"])
    stage = _stage_envelope(
        module,
        case,
        "geometry",
        None,
        summary=summary,
        state=_bi2se3_geometry_state(case),
    )
    module._validate_stage_result(
        case,
        stage="geometry",
        result=stage,
        upstream=None,
        backend="cuda",
        runtime_identity=case.runtime_identity,
    )

    missing_delta = copy.deepcopy(stage)
    missing_delta["state"].pop("incidence_angle_delta_rad")
    missing_delta["scientific_revision"] = module.scientific_revision(
        "geometry",
        {name: value for name, value in missing_delta.items() if name != "scientific_revision"},
    )
    with pytest.raises(ValueError, match="missing key 'incidence_angle_delta_rad'"):
        module._validate_stage_result(
            case,
            stage="geometry",
            result=missing_delta,
            upstream=None,
            backend="cuda",
            runtime_identity=case.runtime_identity,
        )

    independent_third_delta = copy.deepcopy(stage)
    independent_third_delta["state"]["incidence_angles"][2]["effective_angle_rad"] += 1.0e-4
    independent_third_delta["scientific_revision"] = module.scientific_revision(
        "geometry",
        {
            name: value
            for name, value in independent_third_delta.items()
            if name != "scientific_revision"
        },
    )
    with pytest.raises(ValueError, match="one common angle delta"):
        module._validate_stage_result(
            case,
            stage="geometry",
            result=independent_third_delta,
            upstream=None,
            backend="cuda",
            runtime_identity=case.runtime_identity,
        )


def test_replay_rejects_nested_input_path_decoys(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_replay_cli()
    safe_load = module.yaml.safe_load

    def decoy_geometry_series(payload: str):
        document = safe_load(payload)
        if isinstance(document, dict) and document.get("schema_version") == (
            "rasim-osc-geometry-fit-v1"
        ):
            document = copy.deepcopy(document)
            document["simulation_config"] = "bi2se3_simulation.yaml"
        return document

    monkeypatch.setattr(module.yaml, "safe_load", decoy_geometry_series)
    with pytest.raises(ValueError, match="does not resolve to its declared replay role"):
        module.load_replay_case(
            ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
        )


def test_replay_requires_the_locked_numerical_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_replay_cli()
    lock = ROOT / "uv.lock"
    case_path = ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"

    identity = module.load_replay_case(case_path).runtime_identity

    assert identity["environment_lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    installed_version = module.importlib_metadata.version
    assert identity["packages"]["numpy"] == installed_version("numpy")
    expected_packages = {
        "gemmi",
        "llvmlite",
        "numba",
        "numpy",
        "packaging",
        "platformdirs",
        "pyyaml",
        "scipy",
        "sqlalchemy",
        "typing-extensions",
        "xraydb",
    }
    lock_document = tomllib.loads(lock.read_text(encoding="utf-8"))
    sqlalchemy = next(
        record for record in lock_document["package"] if record["name"] == "sqlalchemy"
    )
    greenlet = next(
        dependency for dependency in sqlalchemy["dependencies"] if dependency["name"] == "greenlet"
    )
    if Marker(greenlet["marker"]).evaluate():
        expected_packages.add("greenlet")
    assert set(identity["packages"]) == expected_packages

    def mismatched_version(name: str) -> str:
        return "0.0.0" if name == "numpy" else installed_version(name)

    monkeypatch.setattr(module.importlib_metadata, "version", mismatched_version)
    with pytest.raises(RuntimeError, match=r"numpy.*uv run --frozen"):
        module.load_replay_case(case_path)


def test_runtime_identity_rechecks_the_case_bound_lock_bytes(tmp_path: Path) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"
    )
    changed_lock = tmp_path / "uv.lock"
    changed_lock.write_bytes((ROOT / "uv.lock").read_bytes() + b"\n")
    environment_record = next(
        record for record in case.file_records if record["role"] == "environment_lock"
    )

    with pytest.raises(ValueError, match="environment lock content hash changed"):
        module._validated_runtime_identity(
            changed_lock,
            expected_sha256=environment_record["sha256"],
        )


def test_replay_reloads_and_rejects_mutated_case_mappings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    case.input_paths["simulation_config"] = case.input_paths["cif"]
    calls: list[str] = []

    def forbidden(**_kwargs):
        calls.append("runner")
        raise AssertionError("a stage runner must not execute")

    monkeypatch.setattr(module, "_run_geometry_stage", forbidden)
    output = tmp_path / "mutated_case_output"
    with pytest.raises(ValueError, match="loaded input mappings changed"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="geometry",
        )

    assert calls == []
    assert not output.exists()


def test_scientific_revision_excludes_artifact_location_and_container_hash() -> None:
    module = _load_replay_cli()
    first = {
        "runtime": {"python_version": "3.13.13"},
        "state": {
            "artifact": "C:/first/result.json",
            "artifact_sha256": "0" * 64,
            "fit_evidence": {
                "benchmark": {"warm_residual_median_seconds": 1.0},
                "outer_audit": {"frozen_reindex_wall_time_seconds": 2.0},
            },
            "parameters": [1.0, 2.0],
        },
    }
    relocated = copy.deepcopy(first)
    relocated["state"]["artifact"] = "/other/machine/result.json"
    relocated["state"]["artifact_sha256"] = "f" * 64
    relocated["state"]["fit_evidence"]["benchmark"]["warm_residual_median_seconds"] = 9.0
    relocated["state"]["fit_evidence"]["outer_audit"]["frozen_reindex_wall_time_seconds"] = 10.0
    relocated["runtime"]["python_version"] = "0.0.0"

    assert module.scientific_revision("mosaic", first) == module.scientific_revision(
        "mosaic", relocated
    )


def test_replay_runs_stages_in_order_and_chains_scientific_revisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from PIL import Image

    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    calls: list[tuple[str, str | None, str]] = []

    def stage(name: str):
        def run(*, case, upstream, backend, output_directory):
            upstream_revision = None if upstream is None else upstream["scientific_revision"]
            calls.append((name, upstream_revision, backend))
            state = {}
            if name == "render":
                pixels = bytes((1, 2, 3, 4))
                artifact = tmp_path / "ordered_stage_render.png"
                Image.frombytes("L", (2, 2), pixels).save(artifact)
                state = {
                    "artifact": [str(artifact)],
                    "artifact_identity": [
                        {
                            "decoded_mode": "L",
                            "decoded_size": [2, 2],
                            "decoded_pixel_sha256": hashlib.sha256(pixels).hexdigest(),
                        }
                    ],
                }
            return _stage_envelope(
                module,
                case,
                name,
                upstream,
                summary={},
                state=state,
            )

        return run

    monkeypatch.setattr(module, "_run_geometry_stage", stage("geometry"))
    monkeypatch.setattr(module, "_run_mosaic_stage", stage("mosaic"))
    monkeypatch.setattr(module, "_run_ordered_intensity_stage", stage("ordered_intensity"))
    monkeypatch.setattr(module, "_run_render_stage", stage("render"))

    result = module.run_replay(
        case,
        output_directory=tmp_path / "result",
        backend="cuda",
        through="render",
        verify=False,
    )

    assert tuple(name for name, *_ in calls) == (
        "geometry",
        "mosaic",
        "ordered_intensity",
        "render",
    )
    assert all(backend == "cuda" for *_, backend in calls)
    assert calls[0][1] is None
    assert calls[1][1] == result["stages"]["geometry"]["scientific_revision"]
    assert calls[2][1] == result["stages"]["mosaic"]["scientific_revision"]
    assert calls[3][1] == result["stages"]["ordered_intensity"]["scientific_revision"]
    geometry = result["stages"]["geometry"]
    assert geometry["source_state_count"] == 1
    assert geometry["source_seed"] == 1729
    assert geometry["source_revision"] == (
        "3a68902a791bd9e28d50205b4596738b19a6ea36ce169fe4ff39441798a24659"
    )
    assert geometry["source_revision"] != result["stages"]["mosaic"]["source_revision"]
    assert all(
        stage["source_state_count"] == 250
        for name, stage in result["stages"].items()
        if name != "geometry"
    )
    assert all(
        stage["scientific_summary"]["source_state_count"] == 250
        for stage in result["stages"].values()
    )
    assert result["verification_runtime"]["packages"]["pillow"] == (
        module.importlib_metadata.version("pillow")
    )
    assert "pillow" not in result["stages"]["geometry"]["runtime"]["packages"]
    assert result["stages"]["render"]["runtime"]["packages"]["pillow"] == (
        module.importlib_metadata.version("pillow")
    )


def test_verified_replay_can_stop_after_mosaic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"
    )
    output = tmp_path / "through_mosaic"
    artifact = tmp_path / "mosaic_result.json"
    calls: list[str] = []

    def geometry(*, case, upstream, backend, output_directory):
        del backend, output_directory
        calls.append("geometry")
        return _stage_envelope(
            module,
            case,
            "geometry",
            upstream,
            summary=copy.deepcopy(case.expected_scientific_summary["geometry"]),
            state=_bi2se3_geometry_state(case),
        )

    def mosaic(*, case, upstream, backend, output_directory):
        del backend, output_directory
        calls.append("mosaic")
        fixed_position = _bi2se3_fixed_position(case, upstream)
        summary = copy.deepcopy(case.expected_scientific_summary["mosaic"])
        artifact_document = _mosaic_artifact_document(case, fixed_position)
        artifact.write_text(json.dumps(artifact_document) + "\n", encoding="utf-8")
        return _stage_envelope(
            module,
            case,
            "mosaic",
            upstream,
            summary=summary,
            state={
                "artifact": str(artifact),
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "fixed_position": fixed_position,
                "parameters": list(summary["parameters"]),
                "profile_identities": list(summary["profile_identities"]),
                "profile_scales": [
                    {
                        "identity": module._identity_text(record),
                        "nuisance_peak_scale": record["nuisance_peak_scale"],
                    }
                    for record in sorted(
                        artifact_document["fit"]["profiles"], key=module._identity_text
                    )
                ],
            },
        )

    def forbidden(**_kwargs):
        raise AssertionError("a stage after mosaic must not execute")

    monkeypatch.setattr(module, "_run_geometry_stage", geometry)
    monkeypatch.setattr(module, "_run_mosaic_stage", mosaic)
    monkeypatch.setattr(module, "_run_ordered_intensity_stage", forbidden)
    monkeypatch.setattr(module, "_run_render_stage", forbidden)

    result = module.run_replay(
        case,
        output_directory=output,
        backend="cuda",
        through="mosaic",
    )

    assert calls == ["geometry", "mosaic"]
    assert tuple(result["stages"]) == ("geometry", "mosaic")
    assert (output / "geometry.json").is_file()
    assert (output / "mosaic.json").is_file()
    assert (output / "replay_certificate.json").is_file()
    assert not (output / "ordered_intensity.json").exists()
    assert not (output / "render.json").exists()


def test_verified_replay_can_stop_and_resume_after_ordered_intensity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"
    )
    output = tmp_path / "through_ordered"
    mosaic_artifact = tmp_path / "mosaic_result.json"
    ordered_artifact = tmp_path / "ordered_result.json"
    calls: list[str] = []

    def profile_scales(document: dict) -> list[dict]:
        return [
            {
                "identity": module._identity_text(record),
                "nuisance_peak_scale": record["nuisance_peak_scale"],
            }
            for record in sorted(document["fit"]["profiles"], key=module._identity_text)
        ]

    def geometry(*, case, upstream, backend, output_directory):
        del backend, output_directory
        calls.append("geometry")
        return _stage_envelope(
            module,
            case,
            "geometry",
            upstream,
            summary=copy.deepcopy(case.expected_scientific_summary["geometry"]),
            state=_bi2se3_geometry_state(case),
        )

    def mosaic(*, case, upstream, backend, output_directory):
        del backend, output_directory
        calls.append("mosaic")
        summary = copy.deepcopy(case.expected_scientific_summary["mosaic"])
        fixed_position = _bi2se3_fixed_position(case, upstream)
        document = _mosaic_artifact_document(case, fixed_position)
        mosaic_artifact.write_text(json.dumps(document) + "\n", encoding="utf-8")
        return _stage_envelope(
            module,
            case,
            "mosaic",
            upstream,
            summary=summary,
            state={
                "artifact": str(mosaic_artifact),
                "artifact_sha256": hashlib.sha256(mosaic_artifact.read_bytes()).hexdigest(),
                "fixed_position": fixed_position,
                "parameters": list(summary["parameters"]),
                "profile_identities": list(summary["profile_identities"]),
                "profile_scales": profile_scales(document),
            },
        )

    def ordered(*, case, upstream, backend, output_directory):
        del backend, output_directory
        calls.append("ordered_intensity")
        summary = copy.deepcopy(case.expected_scientific_summary["ordered_intensity"])
        document = _ordered_artifact_document(
            case,
            upstream["state"]["fixed_position"],
            upstream_mosaic_sha256=upstream["state"]["artifact_sha256"],
        )
        ordered_artifact.write_text(json.dumps(document) + "\n", encoding="utf-8")
        return _stage_envelope(
            module,
            case,
            "ordered_intensity",
            upstream,
            summary=summary,
            state={
                "artifact": str(ordered_artifact),
                "artifact_sha256": hashlib.sha256(ordered_artifact.read_bytes()).hexdigest(),
                "fixed_position": upstream["state"]["fixed_position"],
                "parameters": list(summary["parameters"]),
                "profile_scales": profile_scales(document),
                "structure_representative": document["fit"]["structure_representative"],
            },
        )

    def forbidden(**_kwargs):
        raise AssertionError("a resumed or disabled stage must not execute")

    monkeypatch.setattr(module, "_run_geometry_stage", geometry)
    monkeypatch.setattr(module, "_run_mosaic_stage", mosaic)
    monkeypatch.setattr(module, "_run_ordered_intensity_stage", ordered)
    monkeypatch.setattr(module, "_run_render_stage", forbidden)

    first = module.run_replay(
        case,
        output_directory=output,
        backend="cuda",
        through="ordered_intensity",
    )
    saved_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            output / "geometry.json",
            output / "mosaic.json",
            output / "ordered_intensity.json",
            mosaic_artifact,
            ordered_artifact,
        )
    }
    assert calls == ["geometry", "mosaic", "ordered_intensity"]
    assert tuple(first["stages"]) == ("geometry", "mosaic", "ordered_intensity")
    assert first["through"] == "ordered_intensity"
    assert not (output / "render.json").exists()

    monkeypatch.setattr(module, "_run_geometry_stage", forbidden)
    monkeypatch.setattr(module, "_run_mosaic_stage", forbidden)
    monkeypatch.setattr(module, "_run_ordered_intensity_stage", forbidden)
    resumed = module.run_replay(
        case,
        output_directory=output,
        backend="cuda",
        through="ordered_intensity",
        resume=True,
    )
    assert resumed["verified"] is True
    assert calls == ["geometry", "mosaic", "ordered_intensity"]
    assert saved_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            output / "geometry.json",
            output / "mosaic.json",
            output / "ordered_intensity.json",
            mosaic_artifact,
            ordered_artifact,
        )
    }

    disabled_output = tmp_path / "disabled_render"
    with pytest.raises(ValueError, match="render was produced"):
        module.run_replay(
            case,
            output_directory=disabled_output,
            backend="cuda",
            through="render",
        )
    assert not disabled_output.exists()


def test_fresh_stage_recomputes_revision_before_persisting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )

    def changed_stage(*, case, upstream, backend, output_directory):
        del backend, output_directory
        result = _stage_envelope(
            module,
            case,
            "geometry",
            upstream,
            summary={},
            state={},
        )
        result["state"]["changed_after_revision"] = True
        return result

    monkeypatch.setattr(module, "_run_geometry_stage", changed_stage)
    output = tmp_path / "changed_fresh_stage"
    with pytest.raises(ValueError, match="changed its scientific revision"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="geometry",
            verify=False,
        )

    assert not (output / "geometry.json").exists()


def test_stage_summary_is_verified_before_persisting_or_running_downstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    geometry_summary = copy.deepcopy(case.expected_scientific_summary["geometry"])
    geometry_summary["corrections"][2] += 10.0 * case.tolerances["geometry_correction_absolute"]
    calls: list[str] = []

    def geometry(*, case, upstream, backend, output_directory):
        del backend, output_directory
        calls.append("geometry")
        return _stage_envelope(
            module,
            case,
            "geometry",
            upstream,
            summary=geometry_summary,
            state={"corrections": list(geometry_summary["corrections"])},
        )

    def forbidden(**_kwargs):
        calls.append("mosaic")
        raise AssertionError("mosaic must not execute after rejected geometry")

    monkeypatch.setattr(module, "_run_geometry_stage", geometry)
    monkeypatch.setattr(module, "_run_mosaic_stage", forbidden)
    output = tmp_path / "rejected_geometry"
    with pytest.raises(module.ReplayMismatchError, match=r"geometry\.corrections"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="mosaic",
        )

    assert calls == ["geometry"]
    assert not (output / "geometry.json").exists()


def test_bi2te3_cpu_replay_rejects_before_output_or_stage_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    calls: list[str] = []

    def forbidden(**_kwargs):
        calls.append("runner")
        raise AssertionError("a stage runner must not execute")

    for name in (
        "_run_geometry_stage",
        "_run_mosaic_stage",
        "_run_ordered_intensity_stage",
        "_run_render_stage",
    ):
        monkeypatch.setattr(module, name, forbidden)
    output = tmp_path / "cpu_output"
    with pytest.raises(ValueError, match="CUDA-qualified only"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cpu",
            through="mosaic",
        )

    assert calls == []
    assert not output.exists()


def test_render_preflight_requires_importable_pillow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    imported = module.importlib.import_module

    def fail_pillow(name: str):
        if name == "PIL.Image":
            raise ImportError("synthetic missing image module")
        return imported(name)

    monkeypatch.setattr(module.importlib, "import_module", fail_pillow)
    output = tmp_path / "render_output"
    with pytest.raises(RuntimeError, match=r"PIL\.Image.*--extra visualization"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="render",
            verify=False,
        )

    assert not output.exists()


def test_scientific_verifier_uses_tolerances_but_rejects_identity_changes() -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    actual = copy.deepcopy(case.expected_scientific_summary)

    actual["geometry"]["corrections"][2] += 0.25 * case.tolerances["geometry_correction_absolute"]
    actual["mosaic"]["parameters"][0] += 0.25 * case.tolerances["mosaic_parameter_absolute"]
    module.verify_scientific_summary(case, actual)

    wrong_source = copy.deepcopy(actual)
    wrong_source["source_revision"] = "sha256-" + "0" * 64
    with pytest.raises(module.ReplayMismatchError, match="source_revision"):
        module.verify_scientific_summary(case, wrong_source)

    wrong_profiles = copy.deepcopy(actual)
    wrong_profiles["mosaic"]["profile_identities"].pop()
    with pytest.raises(module.ReplayMismatchError, match="profile_identities"):
        module.verify_scientific_summary(case, wrong_profiles)

    wrong_m0 = copy.deepcopy(actual)
    wrong_m0["mosaic"]["m0_profile_identities"] = []
    with pytest.raises(module.ReplayMismatchError, match="m0_profile_identities"):
        module.verify_scientific_summary(case, wrong_m0)

    wrong_parameter = copy.deepcopy(actual)
    wrong_parameter["ordered_intensity"]["parameters"][1] += (
        10.0 * case.tolerances["ordered_parameter_absolute"]
    )
    with pytest.raises(module.ReplayMismatchError, match=r"ordered_intensity\.parameters"):
        module.verify_scientific_summary(case, wrong_parameter)


def test_partial_verification_and_resume_reject_tampered_scientific_state(
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    geometry_summary = copy.deepcopy(case.expected_scientific_summary["geometry"])
    partial = {
        "case_id": case.case_id,
        "material_id": case.material_id,
        "source_state_count": case.source_state_count,
        "source_revision": case.expected_scientific_summary["source_revision"],
        "geometry": geometry_summary,
    }
    module.verify_scientific_summary(case, partial, through="geometry")

    output = tmp_path / "resume"
    output.mkdir()
    stage = _stage_envelope(
        module,
        case,
        "geometry",
        None,
        summary=geometry_summary,
        state={"corrections": list(geometry_summary["corrections"])},
    )
    (output / "geometry.json").write_text(
        json.dumps(stage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = module.run_replay(
        case,
        output_directory=output,
        backend="cuda",
        through="geometry",
        resume=True,
    )
    assert result["verified"]

    changed_runtime = copy.deepcopy(stage)
    changed_runtime["runtime"]["python_version"] = "0.0.0"
    (output / "geometry.json").write_text(
        json.dumps(changed_runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="changed runtime"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="geometry",
            resume=True,
        )

    (output / "geometry.json").write_text(
        json.dumps(stage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tampered = json.loads((output / "geometry.json").read_text(encoding="utf-8"))
    tampered["state"]["corrections"][2] += 1.0e-3
    (output / "geometry.json").write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="scientific revision"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="geometry",
            resume=True,
        )


def test_resume_requires_the_external_json_artifact_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"
    )
    output = tmp_path / "resume_json"
    artifact = tmp_path / "mosaic.json"
    artifact.write_text("{}\n", encoding="utf-8")

    def runner(stage: str):
        def run(*, case, upstream, backend, output_directory):
            del backend, output_directory
            state = {}
            summary = {}
            if stage == "geometry":
                summary = copy.deepcopy(case.expected_scientific_summary["geometry"])
                state = _bi2se3_geometry_state(case)
            if stage == "mosaic":
                fixed_position = _bi2se3_fixed_position(case, upstream)
                summary = copy.deepcopy(case.expected_scientific_summary["mosaic"])
                artifact_document = _mosaic_artifact_document(case, fixed_position)
                artifact.write_text(json.dumps(artifact_document) + "\n", encoding="utf-8")
                state = {
                    "artifact": str(artifact),
                    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "fixed_position": fixed_position,
                    "parameters": list(summary["parameters"]),
                    "profile_identities": list(summary["profile_identities"]),
                    "profile_scales": [
                        {
                            "identity": module._identity_text(record),
                            "nuisance_peak_scale": record["nuisance_peak_scale"],
                        }
                        for record in sorted(
                            artifact_document["fit"]["profiles"], key=module._identity_text
                        )
                    ],
                }
            return _stage_envelope(
                module,
                case,
                stage,
                upstream,
                summary=summary,
                state=state,
            )

        return run

    monkeypatch.setattr(module, "_run_geometry_stage", runner("geometry"))
    monkeypatch.setattr(module, "_run_mosaic_stage", runner("mosaic"))
    module.run_replay(
        case,
        output_directory=output,
        backend="cuda",
        through="mosaic",
        verify=False,
    )

    mosaic_path = output / "mosaic.json"
    original_mosaic = json.loads(mosaic_path.read_text(encoding="utf-8"))
    original_artifact = artifact.read_text(encoding="utf-8")

    independent_delta = copy.deepcopy(original_mosaic)
    independent_delta["state"]["fixed_position"]["incidence_angle_delta_rad"] += 1.0e-4
    independent_artifact = json.loads(original_artifact)
    independent_artifact["fixed_geometry"] = independent_delta["state"]["fixed_position"]
    artifact.write_text(
        json.dumps(independent_artifact) + "\n",
        encoding="utf-8",
    )
    independent_delta["state"]["artifact_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    independent_delta["scientific_revision"] = module.scientific_revision(
        "mosaic",
        {name: value for name, value in independent_delta.items() if name != "scientific_revision"},
    )
    mosaic_path.write_text(
        json.dumps(independent_delta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="geometry position handoff"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="mosaic",
            verify=False,
            resume=True,
        )

    artifact.write_text(original_artifact, encoding="utf-8")
    mosaic_path.write_text(
        json.dumps(original_mosaic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    changed_science = json.loads(original_artifact)
    changed_science["recovered_effective_distribution"]["gaussian_sigma_deg"] += 0.1
    artifact.write_text(json.dumps(changed_science) + "\n", encoding="utf-8")
    updated_container_hash = copy.deepcopy(original_mosaic)
    updated_container_hash["state"]["artifact_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    mosaic_path.write_text(
        json.dumps(updated_container_hash, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="external artifact changed its scientific state"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="mosaic",
            verify=False,
            resume=True,
        )

    artifact.write_text(original_artifact, encoding="utf-8")
    mosaic_path.write_text(
        json.dumps(original_mosaic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    changed_scale = json.loads(original_artifact)
    changed_scale["fit"]["profiles"][0]["nuisance_peak_scale"] *= 2.0
    artifact.write_text(json.dumps(changed_scale) + "\n", encoding="utf-8")
    updated_container_hash = copy.deepcopy(original_mosaic)
    updated_container_hash["state"]["artifact_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    mosaic_path.write_text(
        json.dumps(updated_container_hash, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="external artifact changed its scientific state"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="mosaic",
            verify=False,
            resume=True,
        )

    artifact.write_text(original_artifact, encoding="utf-8")
    mosaic_path.write_text(
        json.dumps(original_mosaic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact.write_text('{"substituted": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed its external result"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="mosaic",
            verify=False,
            resume=True,
        )

    artifact.write_text(original_artifact, encoding="utf-8")
    mosaic = copy.deepcopy(original_mosaic)
    mosaic["state"].pop("artifact")
    mosaic["state"].pop("artifact_sha256")
    mosaic_path.write_text(
        json.dumps(mosaic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mosaic state is missing key 'artifact'"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="mosaic",
            verify=False,
            resume=True,
        )


def test_ordered_resume_binds_external_fitted_structure(tmp_path: Path) -> None:
    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"
    )
    geometry = _stage_envelope(
        module,
        case,
        "geometry",
        None,
        summary=copy.deepcopy(case.expected_scientific_summary["geometry"]),
        state=_bi2se3_geometry_state(case),
    )
    fixed_position = _bi2se3_fixed_position(case, geometry)
    mosaic_artifact = tmp_path / "mosaic-artifact.json"
    mosaic_artifact.write_text(
        json.dumps(_mosaic_artifact_document(case, fixed_position)) + "\n",
        encoding="utf-8",
    )
    mosaic_summary = copy.deepcopy(case.expected_scientific_summary["mosaic"])
    mosaic = _stage_envelope(
        module,
        case,
        "mosaic",
        geometry,
        summary=mosaic_summary,
        state={
            "artifact": str(mosaic_artifact),
            "artifact_sha256": hashlib.sha256(mosaic_artifact.read_bytes()).hexdigest(),
            "fixed_position": fixed_position,
            "parameters": list(mosaic_summary["parameters"]),
            "profile_identities": list(mosaic_summary["profile_identities"]),
            "profile_scales": [
                {
                    "identity": module._identity_text(record),
                    "nuisance_peak_scale": record["nuisance_peak_scale"],
                }
                for record in sorted(
                    json.loads(mosaic_artifact.read_text(encoding="utf-8"))["fit"]["profiles"],
                    key=module._identity_text,
                )
            ],
        },
    )
    ordered_artifact = tmp_path / "ordered-artifact.json"
    ordered_document = _ordered_artifact_document(
        case,
        fixed_position,
        upstream_mosaic_sha256=mosaic["state"]["artifact_sha256"],
    )
    ordered_artifact.write_text(json.dumps(ordered_document) + "\n", encoding="utf-8")
    ordered_summary = copy.deepcopy(case.expected_scientific_summary["ordered_intensity"])
    ordered = _stage_envelope(
        module,
        case,
        "ordered_intensity",
        mosaic,
        summary=ordered_summary,
        state={
            "artifact": str(ordered_artifact),
            "artifact_sha256": hashlib.sha256(ordered_artifact.read_bytes()).hexdigest(),
            "fixed_position": fixed_position,
            "parameters": list(ordered_summary["parameters"]),
            "profile_scales": [
                {
                    "identity": module._identity_text(record),
                    "nuisance_peak_scale": record["nuisance_peak_scale"],
                }
                for record in sorted(ordered_document["fit"]["profiles"], key=module._identity_text)
            ],
            "structure_representative": ordered_document["fit"]["structure_representative"],
        },
    )
    module._validate_stage_result(
        case,
        stage="ordered_intensity",
        result=ordered,
        upstream=mosaic,
        backend="cuda",
        runtime_identity=case.runtime_identity,
    )

    changed_artifact = copy.deepcopy(ordered_document)
    changed_artifact["fit"]["parameters"][
        case.stage_config["ordered_intensity"]["active_parameters"][0]
    ] += 0.1
    ordered_artifact.write_text(json.dumps(changed_artifact) + "\n", encoding="utf-8")
    changed_container_hash = copy.deepcopy(ordered)
    changed_container_hash["state"]["artifact_sha256"] = hashlib.sha256(
        ordered_artifact.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match=r"changed its (occupancy gauge|scientific state)"):
        module._validate_stage_result(
            case,
            stage="ordered_intensity",
            result=changed_container_hash,
            upstream=mosaic,
            backend="cuda",
            runtime_identity=case.runtime_identity,
        )

    def stale_fit_recipe(document: dict) -> None:
        document["fit_recipe"]["revision"] = "stale"

    def stale_response_contract(document: dict) -> None:
        document["response_contract"]["revision"] = "stale"

    def rasterized_simulation(document: dict) -> None:
        document["simulated_detector_rasterization_used_in_fit"] = True

    def changed_occupancy_gauge(document: dict) -> None:
        document["fit"]["occupancy_ratio_reference"] = "se1_occupancy"

    def stale_continuous_oracle(document: dict) -> None:
        document["response_validation"]["cached_vs_fresh_maximum_relative_error"] = 1.0

    for tamper in (
        stale_fit_recipe,
        stale_response_contract,
        rasterized_simulation,
        changed_occupancy_gauge,
        stale_continuous_oracle,
    ):
        changed_contract = copy.deepcopy(ordered_document)
        tamper(changed_contract)
        ordered_artifact.write_text(json.dumps(changed_contract) + "\n", encoding="utf-8")
        changed_container_hash = copy.deepcopy(ordered)
        changed_container_hash["state"]["artifact_sha256"] = hashlib.sha256(
            ordered_artifact.read_bytes()
        ).hexdigest()
        with pytest.raises(ValueError):
            module._validate_stage_result(
                case,
                stage="ordered_intensity",
                result=changed_container_hash,
                upstream=mosaic,
                backend="cuda",
                runtime_identity=case.runtime_identity,
            )

    changed_amplitude = copy.deepcopy(ordered_document)
    changed_amplitude["fit"]["profiles"][0]["nuisance_peak_scale"] *= 2.0
    ordered_artifact.write_text(json.dumps(changed_amplitude) + "\n", encoding="utf-8")
    changed_amplitude_stage = copy.deepcopy(ordered)
    changed_amplitude_stage["state"]["artifact_sha256"] = hashlib.sha256(
        ordered_artifact.read_bytes()
    ).hexdigest()
    changed_amplitude_stage["state"]["profile_scales"] = module._bi2se3_ordered_artifact_projection(
        changed_amplitude,
        case=case,
    )["state"]["profile_scales"]
    changed_amplitude_stage["scientific_revision"] = module.scientific_revision(
        "ordered_intensity",
        {
            name: value
            for name, value in changed_amplitude_stage.items()
            if name != "scientific_revision"
        },
    )
    with pytest.raises(ValueError, match="profile amplitude handoff"):
        module._validate_stage_result(
            case,
            stage="ordered_intensity",
            result=changed_amplitude_stage,
            upstream=mosaic,
            backend="cuda",
            runtime_identity=case.runtime_identity,
        )


def test_render_resume_rejects_changed_decoded_pixels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from PIL import Image

    module = _load_replay_cli()
    case = module.load_replay_case(
        ROOT / "examples" / "bi2te3" / "experiment" / "staged_fit_replay.toml"
    )
    output = tmp_path / "resume_render"
    image_path = output / "render.png"
    pixels = bytes((1, 2, 3, 4))

    def runner(stage: str):
        def run(*, case, upstream, backend, output_directory):
            state = {}
            if stage == "render":
                Image.frombytes("L", (2, 2), pixels).save(image_path)
                state = {
                    "artifact": [str(image_path)],
                    "artifact_identity": [
                        {
                            "decoded_mode": "L",
                            "decoded_size": [2, 2],
                            "decoded_pixel_sha256": hashlib.sha256(pixels).hexdigest(),
                        }
                    ],
                }
            return _stage_envelope(
                module,
                case,
                stage,
                upstream,
                summary={},
                state=state,
            )

        return run

    monkeypatch.setattr(module, "_run_geometry_stage", runner("geometry"))
    monkeypatch.setattr(module, "_run_mosaic_stage", runner("mosaic"))
    monkeypatch.setattr(module, "_run_ordered_intensity_stage", runner("ordered_intensity"))
    monkeypatch.setattr(module, "_run_render_stage", runner("render"))
    module.run_replay(
        case,
        output_directory=output,
        backend="cuda",
        through="render",
        verify=False,
    )

    Image.new("L", (2, 2), color=9).save(image_path)
    with pytest.raises(ValueError, match="decoded pixels"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="render",
            verify=False,
            resume=True,
        )

    render = json.loads((output / "render.json").read_text(encoding="utf-8"))
    render["state"].pop("artifact")
    (output / "render.json").write_text(
        json.dumps(render, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identities are incomplete"):
        module.run_replay(
            case,
            output_directory=output,
            backend="cuda",
            through="render",
            verify=False,
            resume=True,
        )
