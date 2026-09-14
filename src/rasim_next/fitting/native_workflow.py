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
    if set(overrides.get("integration", {})) - {
        "axial_power",
        "angular_power",
        "seed",
        "cone_quadrature_order",
        "stitch_grid_size",
        "axial_peak_spacing_L",
        "axial_peak_half_width_L",
        "source_latent_radius",
        "regular_q_bounds_Ainv",
        "local_m0_q_bounds_Ainv",
        "quadrature_kind",
        "maximum_axial_panel_width_Ainv",
        "angular_support",
        "frozen_ewald_bounds_Ainv_rad",
    }:
        raise ValueError(
            "numerical checks must change quadrature, not batching or rejection guards"
        )
    if set(overrides.get("source", {})) - {
        "kind",
        "sample_count",
        "seed",
        "divergence_order",
        "wavelength_order",
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
    source_override = {**plan.get("source_override", {}), **overrides.get("source", {})}
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


def native_prediction_group(physics, observations, plan, values, repeats, *, model=None):
    """One process-local evaluator, reused across a bounded group of candidates."""
    evaluator = make_native_evaluator(physics, observations, plan, model=model)
    return [evaluator.predict(row, repeats) for row in values]
