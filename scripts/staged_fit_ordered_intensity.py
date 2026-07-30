"""Measured ordered-intensity stage helpers shared by replay and standalone fitting."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any


def profile_identity_text(record: dict[str, Any]) -> str:
    """Return the canonical serialized identity of one fitted profile."""

    dataset_id = str(record["dataset_id"])
    family_m = int(record.get("family_m", record.get("m")))
    integer_l = int(record.get("integer_L", record.get("L")))
    analytic_branch = int(record.get("analytic_branch_id", 0 if family_m == 0 else 2))
    branch = record.get("root_side_branch_id", record.get("side"))
    return (
        f"{dataset_id}|{family_m}|{integer_l}|{analytic_branch}|"
        f"{'none' if branch is None else int(branch)}"
    )


def profile_summary(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[int]]:
    """Summarize exact identities and per-incidence coverage."""

    identities = sorted(profile_identity_text(record) for record in records)
    m0 = sorted(
        profile_identity_text(record)
        for record in records
        if int(record.get("family_m", record.get("m"))) == 0
    )
    counts = [
        sum(str(record["dataset_id"]).endswith(f"-{angle:g}deg") for record in records)
        for angle in (5.0, 10.0, 15.0)
    ]
    return identities, m0, counts


def measured_profile_scales(document: dict[str, Any]) -> dict[str, float]:
    """Read positive mosaic amplitudes keyed by canonical profile identity."""

    fit = document.get("fit")
    profiles = fit.get("profiles") if isinstance(fit, dict) else None
    if (
        not isinstance(profiles, list)
        or not profiles
        or not all(isinstance(record, dict) for record in profiles)
    ):
        raise ValueError("mosaic result lacks fitted measured-profile amplitudes")
    scales: dict[str, float] = {}
    for record in profiles:
        identity = profile_identity_text(record)
        if identity in scales:
            raise ValueError("mosaic result contains a duplicate fitted profile amplitude")
        raw_scale = record.get("nuisance_peak_scale")
        if isinstance(raw_scale, bool):
            raise ValueError("mosaic fitted profile amplitude must be positive and finite")
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "mosaic fitted profile amplitude must be positive and finite"
            ) from error
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("mosaic fitted profile amplitude must be positive and finite")
        scales[identity] = scale
    return scales


def transferred_profile_signal(
    baseline_signal: Any,
    identity_texts: tuple[str, ...],
    profile_scales: dict[str, float],
) -> Any:
    """Apply measured mosaic amplitudes to matching continuous baseline signals."""

    import numpy as np

    identities = tuple(str(value) for value in identity_texts)
    signal = np.asarray(baseline_signal, dtype=np.float64)
    if (
        signal.ndim != 1
        or signal.shape != (len(identities),)
        or len(set(identities)) != len(identities)
        or not np.all(np.isfinite(signal))
        or np.any(signal <= 0.0)
        or set(profile_scales) != set(identities)
    ):
        raise ValueError("baseline signals and mosaic amplitudes must align exactly by identity")
    scales = np.asarray([profile_scales[identity] for identity in identities], dtype=np.float64)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("mosaic fitted profile amplitude must be positive and finite")
    transferred = np.ascontiguousarray(signal * scales)
    if not np.all(np.isfinite(transferred)) or np.any(transferred <= 0.0):
        raise FloatingPointError("transferred real-OSC structure signal is not positive and finite")
    transferred.setflags(write=False)
    return transferred


def relative_residual_summary_by_family(
    profiles: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Return exact residual statistics grouped by layered family."""

    grouped: dict[int, list[float]] = {}
    for record in profiles:
        try:
            family_m = int(record["family_m"])
            residual = float(record["relative_residual"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("ordered profile lacks a finite family residual") from error
        if not math.isfinite(residual):
            raise ValueError("ordered profile lacks a finite family residual")
        grouped.setdefault(family_m, []).append(residual)
    if not grouped:
        raise ValueError("ordered fit contains no family residuals")
    return {
        str(family_m): {
            "count": len(values),
            "sum_squared": sum(value * value for value in values),
            "root_mean_square": math.sqrt(sum(value * value for value in values) / len(values)),
            "maximum_absolute": max(abs(value) for value in values),
        }
        for family_m, values in sorted(grouped.items())
    }


def bi2se3_ordered_fit_recipe(stage: dict[str, Any]) -> dict[str, Any]:
    """Freeze the Bi2Se3 relative-occupancy fit policy into artifact metadata."""

    active_parameter_names = tuple(str(name) for name in stage["active_parameters"])
    return {
        "revision": "bi2se3_measured_relative_structure_multistart.v1",
        "active_parameter_names": list(active_parameter_names),
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


def source_averaged_ordered_response_contract() -> dict[str, str | float]:
    """Return the continuous response compiler contract recorded by the stage."""

    from rasim_next.fitting import (
        SOURCE_AVERAGED_ORDERED_INTENSITY_RESPONSE_CONTRACT_REVISION,
        SOURCE_AVERAGED_ORDERED_INTENSITY_SIGNAL_CERTIFICATE_RELATIVE_FLOOR,
    )

    return {
        "revision": SOURCE_AVERAGED_ORDERED_INTENSITY_RESPONSE_CONTRACT_REVISION,
        "signal_certificate_relative_floor": (
            SOURCE_AVERAGED_ORDERED_INTENSITY_SIGNAL_CERTIFICATE_RELATIVE_FLOOR
        ),
    }


def bi2se3_ordered_artifact_contract(case: Any) -> dict[str, Any]:
    """Resolve the immutable artifact contract from one replay case."""

    stage = case.stage_config["ordered_intensity"]
    ordered_case = tomllib.loads(
        case.input_paths[str(stage["case_role"])].read_text(encoding="utf-8")
    )
    if ordered_case.get("schema_version") != "rasim-measured-ordered-intensity-fit-v1":
        raise ValueError("unsupported Bi2Se3 measured ordered-intensity case")
    response_validation = ordered_case.get("response_validation")
    if not isinstance(response_validation, dict):
        raise ValueError("measured ordered-intensity case lacks response validation")
    mosaic_case = tomllib.loads(case.input_paths["mosaic_case"].read_text(encoding="utf-8"))
    observations = mosaic_case.get("m0_observations")
    if not isinstance(observations, list) or not all(
        isinstance(record, dict) and isinstance(record.get("dataset_id"), str)
        for record in observations
    ):
        raise ValueError("Bi2Se3 mosaic case lacks its dataset identities")
    return {
        "active_parameter_names": tuple(str(name) for name in stage["active_parameters"]),
        "claim_boundary": str(stage["claim_boundary"]),
        "fit_recipe": bi2se3_ordered_fit_recipe(stage),
        "response_contract": source_averaged_ordered_response_contract(),
        "observation_model": ordered_case["observation_model"],
        "background_inheritance": ordered_case["background_inheritance"],
        "maximum_interpolation_relative_error": float(
            response_validation["maximum_interpolation_relative_error"]
        ),
        "dataset_ids": {str(record["dataset_id"]) for record in observations},
        "multistart_count": len(stage["multistarts"]),
    }


def project_bi2se3_ordered_artifact(
    document: dict[str, Any],
    *,
    case: Any,
) -> dict[str, Any]:
    """Validate and project the scientific state of a measured Bi2Se3 fit artifact."""

    contract = bi2se3_ordered_artifact_contract(case)
    active_parameter_names = contract["active_parameter_names"]
    claim_boundary = contract["claim_boundary"]
    if (
        document.get("schema_version") != "rasim-bi2se3-measured-ordered-intensity-fit-v1"
        or document.get("status") != "MODEL_LIMITED_REAL_OSC_STRUCTURE_ESTIMATE_NO_ORACLE"
        or document.get("positions_frozen") is not True
        or document.get("claim_boundary") != claim_boundary
        or document.get("fit_recipe") != contract["fit_recipe"]
        or document.get("response_contract") != contract["response_contract"]
        or document.get("simulated_detector_rasterization_used_in_fit") is not False
        or "detector_pixelization_used_in_fit" in document
        or document.get("measured_detector_observation_source")
        != "upstream_exact_pixel_overlap_profile_amplitudes.v1"
        or document.get("observation_model") != contract["observation_model"]
        or document.get("background_inheritance") != contract["background_inheritance"]
    ):
        raise ValueError("ordered-intensity stage artifact has an unsupported scientific contract")
    fit = document.get("fit")
    fitted = fit.get("parameters") if isinstance(fit, dict) else None
    representative = fit.get("structure_representative") if isinstance(fit, dict) else None
    active_bounds = fit.get("active_bounds") if isinstance(fit, dict) else None
    profiles = fit.get("profiles") if isinstance(fit, dict) else None
    multistart_failures = fit.get("multistart_failures") if isinstance(fit, dict) else None
    multistarts = fit.get("multistarts") if isinstance(fit, dict) else None
    adequacy = document.get("fit_adequacy")
    response_validation = document.get("response_validation")
    response_execution = document.get("response_execution")
    if (
        not isinstance(fitted, dict)
        or not isinstance(representative, dict)
        or not isinstance(active_bounds, dict)
        or not isinstance(profiles, list)
        or not all(isinstance(record, dict) for record in profiles)
        or not isinstance(adequacy, dict)
        or adequacy.get("classification") != "UNQUALIFIED_BOUNDARY_HIGH_RESIDUAL"
        or adequacy.get("qualified") is not False
        or adequacy.get("objective_measure") != "unweighted_sum_squared_relative_residual.v1"
        or tuple(fit.get("active_parameter_names", ())) != active_parameter_names
        or fit.get("occupancy_ratio_reference") != "bi_occupancy"
        or multistart_failures != []
        or not isinstance(multistarts, list)
        or len(multistarts) != contract["multistart_count"]
        or not isinstance(response_validation, dict)
        or response_validation.get("maximum_allowed_relative_error")
        != contract["maximum_interpolation_relative_error"]
        or not isinstance(response_execution, list)
        or len(response_execution) != len(contract["dataset_ids"])
        or not all(isinstance(record, dict) for record in response_execution)
        or {record.get("dataset_id") for record in response_execution} != contract["dataset_ids"]
        or any(
            record.get("backend") != "numba_cuda_source_averaged.v1"
            for record in response_execution
        )
    ):
        raise ValueError("ordered-intensity stage artifact lacks its fitted structure state")
    try:
        interpolation_error = float(response_validation["maximum_interpolation_relative_error"])
        oracle_error = float(response_validation["cached_vs_fresh_maximum_relative_error"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "ordered-intensity artifact lacks its continuous-response proof"
        ) from error
    error_limit = contract["maximum_interpolation_relative_error"]
    if (
        not math.isfinite(interpolation_error)
        or not math.isfinite(oracle_error)
        or interpolation_error > error_limit
        or oracle_error > error_limit
    ):
        raise ValueError("ordered-intensity artifact failed its continuous-response proof")
    parameters = [float(fitted[name]) for name in active_parameter_names]
    try:
        occupancy_ratios = [
            1.0,
            float(fitted["se1_over_bi"]),
            float(fitted["se2_over_bi"]),
        ]
        occupancy_scale = max(occupancy_ratios)
        expected_occupancies = [value / occupancy_scale for value in occupancy_ratios]
        representative_values = [
            float(representative["bi_occupancy"]),
            float(representative["se1_occupancy"]),
            float(representative["se2_occupancy"]),
        ]
        representative_displacements = [
            float(representative["u_radial_A2"]),
            float(representative["u_normal_A2"]),
        ]
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("ordered-intensity artifact lacks its occupancy gauge") from error
    if (
        occupancy_scale <= 0.0
        or any(not math.isfinite(value) for value in (*parameters, *representative_values))
        or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-14)
            for actual, expected in zip(representative_values, expected_occupancies, strict=True)
        )
        or representative_displacements != parameters[2:]
    ):
        raise ValueError("ordered-intensity artifact changed its occupancy gauge")
    identities, m0_identities, _ = profile_summary(profiles)
    scales = measured_profile_scales(document)
    if set(scales) != set(identities):
        raise ValueError("ordered-intensity artifact changed its measured profile amplitudes")
    family_residual = relative_residual_summary_by_family(profiles)
    if adequacy.get("relative_residual_by_family") != family_residual:
        raise ValueError("ordered-intensity artifact changed its family residual summary")
    try:
        overall_residual_rms = float(adequacy["overall_relative_residual_rms"])
        m0_residual_rms = float(family_residual["0"]["root_mean_square"])
        m1_residual_rms = float(family_residual["1"]["root_mean_square"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ordered-intensity artifact lacks its adequacy residuals") from error
    if not all(
        math.isfinite(value) for value in (overall_residual_rms, m0_residual_rms, m1_residual_rms)
    ):
        raise ValueError("ordered-intensity artifact has nonfinite adequacy residuals")
    residual_summary = fit.get("residual_summary")
    bound_names = [name for name in active_parameter_names if bool(active_bounds[name])]
    if (
        not isinstance(residual_summary, dict)
        or float(residual_summary.get("root_mean_square", math.nan)) != overall_residual_rms
        or adequacy.get("parameters_on_bounds") != bound_names
    ):
        raise ValueError("ordered-intensity artifact changed its adequacy classification")
    profile_scales = [
        {"identity": identity, "nuisance_peak_scale": scales[identity]} for identity in identities
    ]
    fixed_position = document.get("fixed_position")
    fixed_mosaic = document.get("fixed_mosaic")
    source_model = document.get("source_model")
    provenance = document.get("provenance")
    if not all(
        isinstance(value, dict)
        for value in (fixed_position, fixed_mosaic, source_model, provenance)
    ):
        raise ValueError("ordered-intensity stage artifact lacks its upstream handoff state")
    return {
        "state": {
            "fixed_position": fixed_position,
            "parameters": parameters,
            "profile_scales": profile_scales,
            "structure_representative": representative,
        },
        "summary": {
            "classification": document["status"],
            "adequacy": adequacy["classification"],
            "claim_boundary": claim_boundary,
            "parameters": parameters,
            "objective": float(fit["objective"]),
            "rank": int(fit["sensitivity_rank"]),
            "relative_residual_rms": overall_residual_rms,
            "m0_relative_residual_rms": m0_residual_rms,
            "m1_relative_residual_rms": m1_residual_rms,
            "profile_count": len(identities),
            "m0_profile_count": len(m0_identities),
            "active_bounds": [bool(active_bounds[name]) for name in active_parameter_names],
            "profile_identities": identities,
            "m0_profile_identities": m0_identities,
        },
        "fixed_mosaic": fixed_mosaic,
        "source_model": source_model,
        "provenance": provenance,
    }


def validate_cached_bi2se3_ordered_artifact(
    document: dict[str, Any],
    *,
    case: Any,
    expected_provenance: dict[str, Any],
    expected_fixed_mosaic: dict[str, float],
    expected_fixed_position: dict[str, Any],
    expected_source_model: dict[str, Any],
    expected_profile_scales: list[dict[str, str | float]],
    expected_bi_fractional_z: float,
    expected_se2_fractional_z: float,
) -> dict[str, Any]:
    """Accept a cached artifact only when every fixed scientific input matches."""

    projection = project_bi2se3_ordered_artifact(document, case=case)
    representative = document["fit"]["structure_representative"]
    if (
        document.get("provenance") != expected_provenance
        or document.get("fixed_mosaic") != expected_fixed_mosaic
        or document.get("fixed_position") != expected_fixed_position
        or document.get("source_model") != expected_source_model
        or projection["state"]["profile_scales"] != expected_profile_scales
        or float(representative.get("bi_fractional_z", math.nan)) != expected_bi_fractional_z
        or float(representative.get("se2_fractional_z", math.nan)) != expected_se2_fractional_z
    ):
        raise RuntimeError("existing Bi2Se3 ordered artifact does not match the current inputs")
    return projection


@dataclass(frozen=True, slots=True)
class Bi2Se3MeasuredOrderedFitInputs:
    """All fixed state needed to run the measured ordered fit in isolation."""

    backend: str
    series: tuple[Any, ...]
    profile_catalogs: tuple[tuple[Any, tuple[Any, ...]], ...]
    anchor_counts: tuple[dict[str, int | str], ...]
    measured_scales: dict[str, float]
    mosaic_profiles: dict[str, dict[str, Any]]
    baseline: Any
    parameter_names: tuple[str, ...]
    canonical_active: tuple[str, ...]
    bounds: dict[str, tuple[float, float]]
    multistarts: tuple[tuple[float, ...], ...]
    maximum_function_evaluations: int
    source_state_count: int
    required_source_revision: str
    interpolation_limit: float
    claim_boundary: str
    fixed_position_record: dict[str, Any]
    fixed_mosaic: dict[str, float]
    source_model: dict[str, Any]
    fit_recipe: dict[str, Any]
    response_contract: dict[str, str | float]
    observation_model: Any
    background_inheritance: Any
    provenance: dict[str, Any]


def fit_prepared_bi2se3_measured_ordered_document(
    prepared: Any,
    mosaic_document: dict[str, Any],
    *,
    backend: str,
    stage: dict[str, Any],
    source_state_count: int,
    interpolation_limit: float,
    source_model: dict[str, Any],
    observation_model: Any,
    background_inheritance: Any,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Fit one validated prepared checkpoint without replay-specific wiring."""

    scales = measured_profile_scales(mosaic_document)
    profiles = {
        profile_identity_text(record): record for record in mosaic_document["fit"]["profiles"]
    }
    if set(profiles) != set(scales):
        raise RuntimeError("mosaic profile records changed their identity alignment")
    parameter_names = tuple(str(name) for name in stage["active_parameters"])
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
    inputs = Bi2Se3MeasuredOrderedFitInputs(
        backend=backend,
        series=prepared.series,
        profile_catalogs=prepared.profile_catalogs,
        anchor_counts=prepared.anchor_counts,
        measured_scales=scales,
        mosaic_profiles=profiles,
        baseline=prepared.baseline_parameters,
        parameter_names=parameter_names,
        canonical_active=canonical_active,
        bounds=bounds,
        multistarts=tuple(
            tuple(float(value) for value in initial) for initial in stage["multistarts"]
        ),
        maximum_function_evaluations=int(stage["maximum_function_evaluations"]),
        source_state_count=source_state_count,
        required_source_revision=prepared.source_revision,
        interpolation_limit=interpolation_limit,
        claim_boundary=str(stage["claim_boundary"]),
        fixed_position_record=prepared.fixed_position_record,
        fixed_mosaic=prepared.mosaic_parameters,
        source_model=source_model,
        fit_recipe=bi2se3_ordered_fit_recipe(stage),
        response_contract=source_averaged_ordered_response_contract(),
        observation_model=observation_model,
        background_inheritance=background_inheritance,
        provenance=provenance,
    )
    return fit_bi2se3_measured_ordered_document(inputs)


def _structure_parameter_record(parameters: Any) -> dict[str, float]:
    return {
        "bi_fractional_z": parameters.bi_fractional_z,
        "se2_fractional_z": parameters.se2_fractional_z,
        "bi_occupancy": parameters.bi_occupancy,
        "se1_occupancy": parameters.se1_occupancy,
        "se2_occupancy": parameters.se2_occupancy,
        "u_radial_A2": parameters.u_radial_A2,
        "u_normal_A2": parameters.u_normal_A2,
    }


@dataclass(frozen=True, slots=True)
class _CompiledMeasuredSeries:
    responses: tuple[Any, ...]
    observations: tuple[Any, ...]
    detectors: tuple[Any, ...]
    baseline_rows: tuple[Any, ...]
    transferred_rows: tuple[Any, ...]
    maximum_interpolation_error: float
    compile_seconds: float
    baseline_prediction_seconds: float


def _compile_measured_observations(
    inputs: Bi2Se3MeasuredOrderedFitInputs,
) -> _CompiledMeasuredSeries:
    """Compile continuous responses and transfer exact-identity mosaic amplitudes."""

    import gc

    from rasim_next.fitting import (
        OrderedIntensityPeakCenterObservations,
        compile_source_averaged_ordered_intensity_response,
    )
    from rasim_next.pipeline.configured_simulation import build_source_averaged_detector

    responses = []
    observations = []
    detectors = []
    baseline_rows = []
    transferred_rows = []
    compile_seconds = 0.0
    prediction_seconds = 0.0
    compiled_identity_set: set[str] = set()
    for series_inputs, catalog in zip(
        inputs.series,
        inputs.profile_catalogs,
        strict=True,
    ):
        frame, definitions = catalog
        detector = build_source_averaged_detector(series_inputs)
        compile_start = perf_counter()
        response = compile_source_averaged_ordered_intensity_response(
            detector,
            angle_frame=frame,
            definitions=definitions,
            execution_backend=inputs.backend,
        )
        compile_seconds += perf_counter() - compile_start
        identity_texts = tuple(
            profile_identity_text(
                {
                    "dataset_id": identity.dataset_id,
                    "family_m": identity.group_key.layered_family_m,
                    "integer_L": identity.group_key.layered_integer_L,
                    "analytic_branch_id": identity.analytic_branch_id,
                    "root_side_branch_id": identity.branch_id,
                }
            )
            for identity in response.identities
        )
        if compiled_identity_set.intersection(identity_texts):
            raise RuntimeError("ordered response repeated a measured profile identity")
        compiled_identity_set.update(identity_texts)
        prediction_start = perf_counter()
        baseline_signal = response.predict_signal_density_A2_per_rad2(inputs.baseline)
        prediction_seconds += perf_counter() - prediction_start
        transferred_signal = transferred_profile_signal(
            baseline_signal,
            identity_texts,
            {identity: inputs.measured_scales[identity] for identity in identity_texts},
        )
        detectors.append(detector)
        responses.append(response)
        baseline_rows.append(baseline_signal)
        transferred_rows.append(transferred_signal)
        observations.append(
            OrderedIntensityPeakCenterObservations(
                dataset_id=response.dataset_id,
                observable_revision=response.observable_revision,
                signal_density_A2_per_rad2=transferred_signal,
            )
        )
        gc.collect()
    if compiled_identity_set != set(inputs.measured_scales):
        raise RuntimeError("ordered response did not consume every measured mosaic amplitude")
    maximum_interpolation_error = max(
        response.interpolation_validation_maximum_relative_error for response in responses
    )
    if maximum_interpolation_error > inputs.interpolation_limit:
        raise RuntimeError("ordered response interpolation exceeded its declared tolerance")
    return _CompiledMeasuredSeries(
        responses=tuple(responses),
        observations=tuple(observations),
        detectors=tuple(detectors),
        baseline_rows=tuple(baseline_rows),
        transferred_rows=tuple(transferred_rows),
        maximum_interpolation_error=maximum_interpolation_error,
        compile_seconds=compile_seconds,
        baseline_prediction_seconds=prediction_seconds,
    )


@dataclass(frozen=True, slots=True)
class _SelectedMultistart:
    fitted_starts: tuple[tuple[int, Any], ...]
    failures: tuple[dict[str, Any], ...]
    equivalent_starts: tuple[tuple[int, Any], ...]
    selected_start_index: int
    result: Any
    objective_equivalence_tolerance: float
    fit_seconds: float


def _fit_relative_multistarts(
    inputs: Bi2Se3MeasuredOrderedFitInputs,
    compiled: _CompiledMeasuredSeries,
) -> _SelectedMultistart:
    """Fit the frozen four-coordinate ratio model from every declared start."""

    import numpy as np

    from rasim_next.fitting import fit_ordered_intensity_series

    fitted_starts: list[tuple[int, Any]] = []
    failures: list[dict[str, Any]] = []
    fit_seconds = 0.0
    for start_index, initial in enumerate(inputs.multistarts):
        fit_start = perf_counter()
        occupancy_scale = max(1.0, float(initial[0]), float(initial[1]))
        try:
            fit = fit_ordered_intensity_series(
                compiled.responses,
                compiled.observations,
                base_strength=inputs.series[0].strength,
                active_parameter_names=inputs.canonical_active,
                initial_parameters=replace(
                    inputs.baseline,
                    bi_occupancy=1.0 / occupancy_scale,
                    se1_occupancy=float(initial[0]) / occupancy_scale,
                    se2_occupancy=float(initial[1]) / occupancy_scale,
                    u_radial_A2=float(initial[2]),
                    u_normal_A2=float(initial[3]),
                ),
                relative_scale_mode=True,
                active_parameter_bounds=inputs.bounds,
                maximum_function_evaluations=inputs.maximum_function_evaluations,
                required_source_state_count=inputs.source_state_count,
                required_source_revision=inputs.required_source_revision,
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            failures.append({"start_index": start_index, "message": str(error)})
        else:
            fitted_starts.append((start_index, fit))
        fit_seconds += perf_counter() - fit_start
    if not fitted_starts:
        raise RuntimeError(f"all Bi2Se3 measured structure multistarts failed: {failures}")
    if failures:
        raise RuntimeError(f"a Bi2Se3 measured structure multistart failed: {failures}")
    minimum_objective = min(fit.objective for _, fit in fitted_starts)
    objective_equivalence_tolerance = (
        512.0
        * np.finfo(np.float64).eps
        * max(1.0, float(len(inputs.measured_scales)), abs(minimum_objective))
    )
    equivalent_starts = tuple(
        item
        for item in fitted_starts
        if item[1].objective <= minimum_objective + objective_equivalence_tolerance
    )
    selected_start_index, result = min(equivalent_starts, key=lambda item: item[0])
    if result.occupancy_ratio_reference != "bi_occupancy" or result.occupancy_ratios is None:
        raise RuntimeError("relative ordered fit changed the frozen Bi occupancy gauge")
    return _SelectedMultistart(
        fitted_starts=tuple(fitted_starts),
        failures=tuple(failures),
        equivalent_starts=equivalent_starts,
        selected_start_index=selected_start_index,
        result=result,
        objective_equivalence_tolerance=objective_equivalence_tolerance,
        fit_seconds=fit_seconds,
    )


def cached_vs_fresh_ordered_response_max_relative_error(
    detectors: tuple[Any, ...],
    catalogs: tuple[tuple[Any, tuple[Any, ...]], ...],
    cached_rows: tuple[Any, ...],
    structure_parameters: Any,
    *,
    backend: str,
) -> float:
    """Compare cached continuous responses with independent direct evaluations."""

    import numpy as np

    from rasim_next.fitting import evaluate_source_averaged_ordered_intensity_point_signal

    maximum_error = 0.0
    for detector, catalog, cached in zip(detectors, catalogs, cached_rows, strict=True):
        frame, definitions = catalog
        fresh = np.asarray(
            evaluate_source_averaged_ordered_intensity_point_signal(
                detector,
                angle_frame=frame,
                definitions=definitions,
                structure_parameters=structure_parameters,
                execution_backend=backend,
            ),
            dtype=np.float64,
        )
        cached_array = np.asarray(cached, dtype=np.float64)
        if (
            fresh.ndim != 1
            or cached_array.shape != fresh.shape
            or not np.all(np.isfinite(fresh))
            or not np.all(np.isfinite(cached_array))
            or np.any(fresh <= 0.0)
            or np.any(cached_array <= 0.0)
        ):
            raise FloatingPointError(
                "fresh and cached ordered responses must be aligned positive finite vectors"
            )
        scale = max(float(np.max(fresh)), np.finfo(np.float64).tiny)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(cached_array - fresh) / np.maximum(fresh, 1.0e-12 * scale))),
        )
    return maximum_error


def _build_bi2se3_ordered_document(
    inputs: Bi2Se3MeasuredOrderedFitInputs,
    compiled: _CompiledMeasuredSeries,
    selected: _SelectedMultistart,
    *,
    oracle_error: float,
    oracle_seconds: float,
) -> dict[str, Any]:
    """Build the one scientific artifact for a completed measured fit."""

    import numpy as np

    result = selected.result
    representative = result.structure_representative
    parameter_values = [
        float(result.occupancy_ratios[1]),
        float(result.occupancy_ratios[2]),
        float(representative.u_radial_A2),
        float(representative.u_normal_A2),
    ]
    fitted_parameters = dict(zip(inputs.parameter_names, parameter_values, strict=True))
    active_bounds = dict(zip(inputs.parameter_names, result.active_bounds.tolist(), strict=True))
    residual_offset = 0
    profile_records = []
    per_dataset_residual = []
    for response, baseline_signal, transferred_signal, predicted, dataset_scale in zip(
        compiled.responses,
        compiled.baseline_rows,
        compiled.transferred_rows,
        result.predicted_signal_density_A2_per_rad2,
        result.dataset_scales,
        strict=True,
    ):
        count = len(response.identities)
        residual = result.relative_residual[residual_offset : residual_offset + count]
        residual_offset += count
        fitted_signal = float(dataset_scale) * predicted
        per_dataset_residual.append(
            {
                "dataset_id": response.dataset_id,
                "profile_count": count,
                "root_mean_square": float(np.sqrt(np.mean(residual * residual))),
                "maximum_absolute": float(np.max(np.abs(residual))),
            }
        )
        for index, identity in enumerate(response.identities):
            record = {
                "dataset_id": identity.dataset_id,
                "family_m": identity.group_key.layered_family_m,
                "integer_L": identity.group_key.layered_integer_L,
                "analytic_branch_id": identity.analytic_branch_id,
                "root_side_branch_id": identity.branch_id,
            }
            identity_text = profile_identity_text(record)
            mosaic_record = inputs.mosaic_profiles[identity_text]
            profile_records.append(
                {
                    **record,
                    "nuisance_peak_scale": inputs.measured_scales[identity_text],
                    "inherited_signed_background_coefficients": list(
                        mosaic_record.get("signed_background_coefficients", ())
                    ),
                    "baseline_signal_density_A2_per_rad2": float(baseline_signal[index]),
                    "transferred_observed_signal_density_A2_per_rad2": float(
                        transferred_signal[index]
                    ),
                    "fitted_signal_density_A2_per_rad2": float(fitted_signal[index]),
                    "relative_residual": float(residual[index]),
                }
            )
    if residual_offset != result.relative_residual.size:
        raise RuntimeError("ordered fit residuals do not align with measured profiles")
    family_residual = relative_residual_summary_by_family(profile_records)
    overall_residual_rms = float(
        np.sqrt(np.mean(result.relative_residual * result.relative_residual))
    )

    def fit_parameter_values(fit: Any) -> list[float]:
        if fit.occupancy_ratios is None:
            raise RuntimeError("a measured structure multistart lost its occupancy gauge")
        structure = fit.structure_representative
        return [
            float(fit.occupancy_ratios[1]),
            float(fit.occupancy_ratios[2]),
            float(structure.u_radial_A2),
            float(structure.u_normal_A2),
        ]

    multistart_records = [
        {
            "start_index": start_index,
            "parameters": fit_parameter_values(fit),
            "objective": float(fit.objective),
            "function_evaluations": int(fit.function_evaluations),
            "active_bounds": fit.active_bounds.tolist(),
        }
        for start_index, fit in selected.fitted_starts
    ]
    fit_matrix = np.asarray(
        [record["parameters"] for record in multistart_records],
        dtype=np.float64,
    )
    return {
        "schema_version": "rasim-bi2se3-measured-ordered-intensity-fit-v1",
        "status": "MODEL_LIMITED_REAL_OSC_STRUCTURE_ESTIMATE_NO_ORACLE",
        "claim_boundary": inputs.claim_boundary,
        "positions_frozen": True,
        "simulated_detector_rasterization_used_in_fit": False,
        "measured_detector_observation_source": (
            "upstream_exact_pixel_overlap_profile_amplitudes.v1"
        ),
        "observation_model": inputs.observation_model,
        "background_inheritance": inputs.background_inheritance,
        "fit": {
            "active_parameter_names": list(inputs.parameter_names),
            "occupancy_ratio_reference": "bi_occupancy",
            "parameters": fitted_parameters,
            "structure_representative": _structure_parameter_record(representative),
            "objective": float(result.objective),
            "relative_residual": result.relative_residual.tolist(),
            "residual_summary": {
                "root_mean_square": overall_residual_rms,
                "maximum_absolute": float(np.max(np.abs(result.relative_residual))),
                "per_dataset": per_dataset_residual,
            },
            "dataset_scales": dict(
                zip(result.dataset_ids, result.dataset_scales.tolist(), strict=True)
            ),
            "sensitivity_rank": int(result.sensitivity_rank),
            "sensitivity_condition": float(result.sensitivity_condition),
            "sensitivity_singular_values": result.sensitivity_singular_values.tolist(),
            "parameter_correlation": result.parameter_correlation.tolist(),
            "active_bounds": active_bounds,
            "function_evaluations": int(result.function_evaluations),
            "selected_start_index": selected.selected_start_index,
            "objective_equivalence_tolerance": selected.objective_equivalence_tolerance,
            "objective_equivalent_start_indices": [
                start_index for start_index, _ in selected.equivalent_starts
            ],
            "multistarts": multistart_records,
            "multistart_failures": list(selected.failures),
            "multistart_parameter_span": (
                np.ptp(fit_matrix, axis=0).tolist()
                if fit_matrix.shape[0] > 1
                else [0.0] * len(inputs.parameter_names)
            ),
            "profiles": profile_records,
        },
        "fit_adequacy": {
            "qualified": False,
            "classification": "UNQUALIFIED_BOUNDARY_HIGH_RESIDUAL",
            "objective_measure": "unweighted_sum_squared_relative_residual.v1",
            "overall_relative_residual_rms": overall_residual_rms,
            "overall_maximum_absolute_relative_residual": float(
                np.max(np.abs(result.relative_residual))
            ),
            "relative_residual_by_family": family_residual,
            "parameters_on_bounds": [name for name, active in active_bounds.items() if active],
            "conclusion": (
                "the nearly-perfect 2H model does not reproduce the measured nonzero-family "
                "amplitudes; fitted values are an unqualified boundary estimate"
            ),
        },
        "fixed_position": inputs.fixed_position_record,
        "fixed_mosaic": inputs.fixed_mosaic,
        "source_model": inputs.source_model,
        "fit_recipe": inputs.fit_recipe,
        "response_contract": inputs.response_contract,
        "response_validation": {
            "maximum_allowed_relative_error": inputs.interpolation_limit,
            "maximum_interpolation_relative_error": compiled.maximum_interpolation_error,
            "cached_vs_fresh_maximum_relative_error": oracle_error,
            "per_dataset_interpolation_relative_error": {
                response.dataset_id: response.interpolation_validation_maximum_relative_error
                for response in compiled.responses
            },
        },
        "response_revisions": [response.response_revision for response in compiled.responses],
        "response_execution": [
            {
                "dataset_id": response.dataset_id,
                "backend": response.execution_backend,
                "device": response.execution_device,
                "instrument_revision": response.instrument_revision,
            }
            for response in compiled.responses
        ],
        "observable_revisions": [response.observable_revision for response in compiled.responses],
        "anchor_counts": list(inputs.anchor_counts),
        "provenance": inputs.provenance,
        "timing_seconds": {
            "response_compile": compiled.compile_seconds,
            "cached_baseline_prediction": compiled.baseline_prediction_seconds,
            "multistart_fit": selected.fit_seconds,
            "fresh_oracle": oracle_seconds,
        },
    }


def fit_bi2se3_measured_ordered_document(
    inputs: Bi2Se3MeasuredOrderedFitInputs,
) -> dict[str, Any]:
    """Run the complete continuous measured fit and return its standalone artifact."""

    import tracemalloc

    start = perf_counter()
    tracemalloc.start()
    try:
        compiled = _compile_measured_observations(inputs)
        selected = _fit_relative_multistarts(inputs, compiled)
        oracle_start = perf_counter()
        oracle_error = cached_vs_fresh_ordered_response_max_relative_error(
            compiled.detectors,
            inputs.profile_catalogs,
            selected.result.predicted_signal_density_A2_per_rad2,
            selected.result.structure_representative,
            backend=inputs.backend,
        )
        oracle_seconds = perf_counter() - oracle_start
        if oracle_error > inputs.interpolation_limit:
            raise RuntimeError("cached ordered response disagrees with the fresh continuous oracle")
        document = _build_bi2se3_ordered_document(
            inputs,
            compiled,
            selected,
            oracle_error=oracle_error,
            oracle_seconds=oracle_seconds,
        )
        document["timing_seconds"]["total"] = perf_counter() - start
        document["peak_memory_bytes"] = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return document
