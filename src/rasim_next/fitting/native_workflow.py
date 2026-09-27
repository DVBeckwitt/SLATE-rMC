"""Shared physical assembly for native refinement, isolated workers and rendering."""

from dataclasses import replace
from time import perf_counter

import numpy as np

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
        "angular_panel_edges_rad",
        "angular_resolution_fraction",
        "angular_integration",
        "local_m0_angular_power",
        "seed",
        "cone_quadrature_order",
        "stitch_grid_size",
        "axial_peak_spacing_L",
        "axial_peak_half_width_L",
        "local_m0_axial_peak_coordinate",
        "source_latent_radius",
        "regular_q_bounds_Ainv",
        "local_m0_q_bounds_Ainv",
        "quadrature_kind",
        "maximum_axial_panel_width_Ainv",
        "local_m0_maximum_axial_panel_width_Ainv",
        "angular_support",
        "frozen_ewald_bounds_Ainv_rad",
        "axial_meshes",
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
    ignored = set()
    if result.integration_rule.axial_meshes:
        ignored.update(
            {
                "axial_peak_spacing_L",
                "axial_peak_half_width_L",
                "regular_q_bounds_Ainv",
                "local_m0_q_bounds_Ainv",
            }
        )
    if result.integration_rule.frozen_ewald_bounds_Ainv_rad is not None:
        ignored.add("angular_support")
    if (
        result.integration_rule.angular_panel_edges_rad is not None
        or result.integration_rule.angular_integration == "nominal"
    ):
        ignored.add("angular_resolution_fraction")
    else:
        base_rule = replace(original.integration_rule, **plan.get("integration_override", {}))
        if (
            base_rule.angular_panel_edges_rad is None
            and base_rule.angular_integration == "native_panels"
        ):
            for name in ("angular_power", "local_m0_angular_power"):
                before, after = getattr(base_rule, name), getattr(result.integration_rule, name)
                before = base_rule.angular_power if before is None else before
                after = result.integration_rule.angular_power if after is None else after
                if before != after and max(3, before) == max(3, after):
                    ignored.add(name)
    if ignored & set(overrides.get("integration", {})):
        raise ValueError("numerical overrides cannot refine controls ignored by the effective mesh")
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


def _seed_native_elastic_endpoints(evaluator, candidates, repeats, meshes):
    """Union exact source/channel Ewald endpoints over one finite candidate stencil."""
    from rasim_next.pipeline.conditional_detector import native_projection_bounds_px
    from rasim_next.pipeline.fiber_detector import (
        conditional_ewald_region_bounds,
        elastic_axial_cutoff_Ainv,
    )

    inserted = [[] for _ in meshes]
    for candidate in candidates:
        bound, arguments, _, _ = evaluator.bind(candidate, repeats)
        for part in bound.integration_parts():
            detector = part.detector(mosaic=evaluator.proposal_mosaic, **arguments)
            native_bounds = native_projection_bounds_px(
                evaluator.observations.projection, detector.detector_shape_rc
            )
            basis = detector.reciprocal_basis_Ainv
            normal = basis[:, 2] / np.linalg.norm(basis[:, 2])
            part_rods = {(rod.h, rod.k) for rod in part.rods}
            for index, mesh in enumerate(meshes):
                mesh_rods = set(mesh.rods_hk)
                overlap = mesh_rods & part_rods
                if not overlap:
                    continue
                if overlap != mesh_rods:
                    raise ValueError("source partitions must retain complete axial rod groups")
                radii = []
                for h, k in mesh.rods_hk:
                    anchor = h * basis[:, 0] + k * basis[:, 1]
                    radii.append(float(np.linalg.norm(anchor - (anchor @ normal) * normal)))
                radial_tolerance = 64 * np.finfo(float).eps * max(1.0, *radii)
                if not np.allclose(radii, radii[0], rtol=0, atol=radial_tolerance):
                    raise ValueError("axial mesh rods must share one exact radial group")
                states = detector.incident.states
                for source_index in np.flatnonzero(states.valid):
                    ki = (
                        (2 * np.pi / states.wavelength_A[source_index])
                        * states.direction_sample[source_index]
                        if mesh.coordinate == "external_local_m0_q"
                        else states.k_film_phase_sample_Ainv[source_index]
                    )
                    endpoint = elastic_axial_cutoff_Ainv(ki_sample_Ainv=ki, radial_Ainv=radii[0])
                    if endpoint is None:
                        continue
                    local_m0 = mesh.coordinate == "external_local_m0_q"
                    bounds = conditional_ewald_region_bounds(
                        native_bounds_px=native_bounds,
                        source=detector.source,
                        incident=detector.incident,
                        source_state_index=int(source_index),
                        instrument=detector.instrument,
                        material=detector.material,
                        source_latent_radius=detector.integration_rule.source_latent_radius,
                        local_m0=local_m0,
                    )
                    bounds = bounds[bounds[:, 1] >= bounds[:, 0]]
                    axial_upper = np.sqrt(np.maximum(0.0, bounds[:, 1] ** 2 - radii[0] ** 2))
                    if np.any(endpoint <= axial_upper):
                        inserted[index].append(endpoint)
    seeded = []
    records = []
    duplicate_tolerance_Ainv = 1e-10
    for mesh, endpoints in zip(meshes, inserted, strict=True):
        original = np.asarray(mesh.edges_Ainv)
        additions = []
        for endpoint in sorted(endpoints):
            if not original[0] < endpoint < original[-1]:
                continue
            if np.any(np.abs(original - endpoint) <= duplicate_tolerance_Ainv):
                continue
            if additions and abs(additions[-1] - endpoint) <= duplicate_tolerance_Ainv:
                continue
            additions.append(endpoint)
        edges = tuple(np.sort(np.r_[original, additions]))
        seeded.append(replace(mesh, edges_Ainv=edges))
        records.append(
            dict(
                rods_hk=mesh.rods_hk,
                coordinate=mesh.coordinate,
                inserted_edges_Ainv=additions,
            )
        )
    return tuple(seeded), records


def prepare_native_axial_meshes(
    physics, observations, plan, meshes, *, fixed_scale, maximum_panels=2048, maximum_passes=12
):
    """Prepare one physical mesh for a candidate stencil at one declared N.

    Preparation is explicit work, not an optimizer callback. The returned rule
    feeds the ordinary fitter/renderer and retains their support validation.
    """
    from rasim_next.fitting.native_acceleration import refine_axial_meshes
    from rasim_next.pipeline.fiber_detector import AxialPanelMesh

    if any(type(value) is not int or value < 1 for value in (maximum_panels, maximum_passes)):
        raise ValueError("axial adaptation requires positive integer work limits")
    if len(plan["repeat_choices"]) != 1:
        raise ValueError("prepare each N separately so its own center defines stencil contrasts")
    repeats = plan["repeat_choices"][0]
    initial = tuple(AxialPanelMesh(**m) if isinstance(m, dict) else m for m in meshes)
    if {hk for m in initial for hk in m.rods_hk} != {(r.h, r.k) for r in physics.rods}:
        raise ValueError("prepared meshes must cover exactly the unpartitioned physical rod roster")
    candidates = np.asarray(plan["qualification_candidates"])
    names = [p["name"] for p in plan["parameters"]]
    if (
        candidates.ndim != 2
        or candidates.shape[1] != len(names)
        or len(candidates) < 2
        or np.iscomplexobj(candidates)
        or np.any(~np.isfinite(candidates))
        or np.any(candidates < [p["lower"] for p in plan["parameters"]])
        or np.any(candidates > [p["upper"] for p in plan["parameters"]])
        or any(
            name not in names or np.any(candidates[:, names.index(name)] != value)
            for name, value in plan.get("fixed_parameters", {}).items()
        )
    ):
        raise ValueError("mesh qualification requires aligned candidates preserving fixed values")
    evaluator = make_native_evaluator(physics, observations, plan)
    initial, endpoint_records = _seed_native_elastic_endpoints(
        evaluator, candidates, repeats, initial
    )
    # A parent and its children can both miss a narrow detector window. Apply
    # the declared geometric resolution guard before error-driven refinement;
    # switching from CDF panels to physical panels must not discard that guard.
    seeded, panel_count = [], 0
    for mesh in initial:
        cap = physics.integration_rule.maximum_axial_panel_width_Ainv
        if mesh.coordinate == "external_local_m0_q":
            local_cap = physics.integration_rule.local_m0_maximum_axial_panel_width_Ainv
            cap = local_cap if local_cap is not None else cap
        edges = np.asarray(mesh.edges_Ainv)
        with np.errstate(over="ignore"):
            counts = (
                np.ones(len(edges) - 1)
                if cap is None
                else np.maximum(1, np.ceil(np.diff(edges) / cap))
            )
        panel_count += float(counts.sum())
        if 2 * panel_count > maximum_panels:
            raise ValueError("axial seed panel widths exceed the parent/child work budget")
        if cap is not None:
            edges = np.r_[
                np.concatenate(
                    [
                        np.linspace(left, right, int(count) + 1)[:-1]
                        for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True)
                    ]
                ),
                edges[-1],
            ]
        seeded.append(replace(mesh, edges_Ainv=tuple(edges)))
    initial = tuple(seeded)
    builds = []

    def evaluate(meshes):
        rule = replace(
            physics.integration_rule,
            axial_meshes=meshes,
            axial_power=3,
            quadrature_kind="composite_gauss",
            seed=0,
            maximum_axial_panel_width_Ainv=None,
            local_m0_maximum_axial_panel_width_Ainv=None,
        )
        evaluator = make_native_evaluator(
            replace(physics, integration_rule=rule), observations, plan
        )
        start = perf_counter()
        values = np.array([evaluator.predict_axial_panels(v, repeats) for v in candidates])
        builds.append(
            dict(
                panels=sum(len(m.edges_Ainv) - 1 for m in meshes),
                elapsed_seconds=perf_counter() - start,
                compile_count=evaluator.compile_count,
                compile_seconds=evaluator.compile_seconds,
            )
        )
        evaluator.clear_responses()
        return values

    accepted, report = refine_axial_meshes(
        evaluate,
        observations,
        initial,
        fixed_scale=fixed_scale,
        maximum_panels=maximum_panels,
        maximum_passes=maximum_passes,
        **plan["numerical_tolerances"],
    )
    report["builds"] = builds
    report["elastic_endpoint_seeding"] = endpoint_records
    rule = replace(
        physics.integration_rule,
        axial_meshes=accepted,
        axial_power=3,
        quadrature_kind="composite_gauss",
        seed=0,
        maximum_axial_panel_width_Ainv=None,
        local_m0_maximum_axial_panel_width_Ainv=None,
    )
    return rule, report
