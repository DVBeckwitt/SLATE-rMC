"""Shared physical assembly for native refinement, isolated workers and rendering."""

from dataclasses import replace

from painted_ewald import MosaicParameters
from rasim_next.fitting.bi_joint import BiJointModel
from rasim_next.fitting.bi_native import BiNativeStructureModel
from rasim_next.fitting.native_instrument import NativeInstrumentModel
from rasim_next.fitting.native_joint import NativeJointEvaluator
from rasim_next.fitting.native_search import FitParameter
from rasim_next.fitting.pb_native import PbJointModel, PbNativeStructureModel
from rasim_next.materials.optics import material_optics


def native_physics_with(original, plan, overrides):
    if set(overrides) - {"name", "integration", "spatial_quadrature_order", "source"}:
        raise ValueError("unknown numerical refinement fields")
    refinable = {
        "strength_gauss_order",
        "strength_scalar_order",
        "strength_scalar_phase_step_rad",
        "angular_initial_power",
        "pixel_error_rtol",
        "pixel_error_atol",
        "pixel_error_initial_width_rad",
        "local_m0_axial_power",
        "local_m0_angular_power",
        "local_m0_seed",
        "local_m0_peak_spacing_L",
        "local_m0_peak_half_width_L",
        "local_m0_axial_peak_coordinate",
        "local_m0_angular_resolution_fraction",
        "source_latent_radius",
        "cone_quadrature_order",
        "stitch_grid_size",
        "regular_q_bounds_Ainv",
        "local_m0_q_bounds_Ainv",
        "frozen_ewald_bounds_Ainv_rad",
    }
    if set(overrides.get("integration", {})) - refinable:
        raise ValueError("unknown integration controls; migrate obsolete ordinary-engine plans")
    if set(overrides.get("source", {})) - {
        "kind",
        "sample_count",
        "seed",
        "divergence_order",
        "wavelength_order",
        "local_m0_divergence_order",
    }:
        raise ValueError("numerical source checks cannot change physical source parameters")
    result = replace(
        original,
        integration_rule=replace(
            original.integration_rule,
            **{**plan.get("integration_override", {}), **overrides.get("integration", {})},
        ),
        spatial_quadrature_order=overrides.get(
            "spatial_quadrature_order", original.spatial_quadrature_order
        ),
    )
    local_m0_active = result.specular_stitch_stack is not None and any(
        r.h == r.k == 0 and r.population > 0 for r in result.rods
    )
    if not local_m0_active and any(
        name.startswith("local_m0_") for name in overrides.get("integration", {})
    ):
        raise ValueError("numerical overrides cannot refine inactive local-m0 controls")
    source_override = {**plan.get("source_override", {}), **overrides.get("source", {})}
    if "local_m0_divergence_order" in overrides.get("source", {}):
        if not local_m0_active:
            raise ValueError("numerical overrides cannot refine an inactive local m0 source")
        base_source = replace(original.source_definition, **plan.get("source_override", {}))
        refined_source = replace(base_source, **overrides["source"])
        before = base_source.local_m0_divergence_order or base_source.divergence_order
        after = refined_source.local_m0_divergence_order or refined_source.divergence_order
        if before == after and base_source.divergence_order == refined_source.divergence_order:
            raise ValueError("numerical overrides cannot refine an unchanged local m0 source")
    if "stitch_overlap_measure" in plan:
        if result.specular_stitch_stack is None:
            raise ValueError("a stitch overlap measure requires the declared Bi composite")
        result = replace(
            result,
            specular_stitch_stack=replace(
                result.specular_stitch_stack, overlap_measure=plan["stitch_overlap_measure"]
            ),
        )
    if source_override:
        definition = replace(result.source_definition, **source_override)
        source = definition.sample()
        result = replace(
            result,
            source_definition=definition,
            source=source,
            material=material_optics(result.structure.crystals[0], source.mean_rays.wavelength_A),
        )
    return result


def make_native_evaluator(physics, observations, plan, *, model=None):
    if observations.projection is None:
        raise ValueError("native detector evaluation requires a native pixel projection")
    parameters = tuple(FitParameter(**p) for p in plan["parameters"])
    names = tuple(p.name for p in parameters)
    model = (
        model
        if model is not None
        else (
            PbJointModel(PbNativeStructureModel(physics))
            if physics.structure.stacking_phases
            else BiJointModel(BiNativeStructureModel(physics))
        )
    )
    instrument = (
        NativeInstrumentModel(physics, plan["acquisition_id"]) if plan["fit_instrument"] else None
    )
    evaluator = NativeJointEvaluator(
        model,
        observations,
        MosaicParameters(*plan["proposal_mosaic"]),
        instrument,
        plan["workers"],
        spatial_execution=plan.get("spatial_execution", "auto"),
    )
    if evaluator.parameter_names != names:
        raise ValueError("plan must declare every physical parameter in the evaluator's order")
    if tuple(p.unit for p in parameters) != evaluator.parameter_units:
        raise ValueError("plan units differ from the physical coordinate contract")
    specimen_count = len(names) - (18 if instrument is not None else 0)
    expected_owners = ("specimen:" + physics.sample_id,) * specimen_count + (
        plan["acquisition_id"],
    ) * (len(names) - specimen_count)
    if tuple(p.owner for p in parameters) != expected_owners:
        raise ValueError("parameter ownership differs from specimen/acquisition declarations")
    return evaluator


def native_prediction_group(
    physics, observations, plan, values, repeats, *, model=None, include_execution=False
):
    """One process-local evaluator, reused across a bounded group of candidates."""
    evaluator = make_native_evaluator(physics, observations, plan, model=model)
    predictions = [evaluator.predict(row, repeats) for row in values]
    return (predictions, evaluator.spatial_executor.summary()) if include_execution else predictions
