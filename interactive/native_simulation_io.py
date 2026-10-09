"""Independent canonical native binding and whole-panel deterministic integration."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from job_lifecycle import JobResult
from metadata_review import bounded_reference_snapshot
from native_simulation_state import (
    MAX_NATIVE_PHYSICS_BYTES,
    NATIVE_RECIPES,
    NativeSimulationDraft,
    native_draft_from_document,
)
from simulation_io import (
    MAX_SIMULATION_CPU_BYTES,
    MAX_SIMULATION_GPU_BYTES,
    MAX_SIMULATION_PIXELS,
    MAX_SIMULATION_RODS,
    _prepared_frame,
    _stop,
    mean_incidence_deg,
)

MAX_NATIVE_SIMULATION_SOURCES = 256


def native_model(physics):
    from rasim_next.fitting.bi_joint import BiJointModel
    from rasim_next.fitting.bi_native import BiNativeStructureModel
    from rasim_next.fitting.pb_native import PbJointModel, PbNativeStructureModel

    if physics.sample_id not in NATIVE_RECIPES:
        raise ValueError("select one of the six admitted native recipes")
    if physics.sample_id in ("bi2se3", "bi2te3"):
        if physics.structure.stacking_phases:
            raise ValueError("native Bi disorder is unsupported")
        return BiJointModel(BiNativeStructureModel(physics))
    if not physics.structure.stacking_phases or physics.specular_stitch_stack is not None:
        raise ValueError("native Pb requires its admitted phase roster and has no reflectivity")
    return PbJointModel(PbNativeStructureModel(physics))


def native_initial_values(physics, seed=None):
    """Optional supplied physical starting values; no fitted scale or observations."""
    from rasim_next.fitting.pb_native import simplex_shares

    model = native_model(physics)
    c_A = model.atomic.reference_parameters.c_A
    if seed is None:
        repeats = max(1, int(physics.instrument.film_thickness_A / c_A))
        phases = physics.structure.stacking_phases
        seed = {
            "coherent_repeats": repeats,
            "film_thickness_nm": physics.instrument.film_thickness_A / 10,
            "gaussian_sigma_deg": 0.3,
            "lorentzian_hwhm_deg": 0.2,
            "eta": 0.1,
            "surface_fractions": (1 / 3, 1 / 3, 1 / 3),
            "phase_fractions": [1 / len(phases)] * len(phases) if phases else [1],
            "fault_parameters": {p.fault_parameter: 0 for p in phases},
        }
    n = seed["coherent_repeats"]
    if physics.structure.stacking_phases:
        values = model.initial_values(seed)
    else:
        extra = seed["film_thickness_nm"] * 10 - n * c_A
        if extra < -1e-10:
            raise ValueError("physical thickness is smaller than the coherent stack")
        stack = physics.specular_stitch_stack
        if stack is None:
            raise ValueError("Bi recipe requires an explicit specular composite")
        values = np.r_[
            model.atomic.reference_parameters.as_array(),
            np.deg2rad(seed["gaussian_sigma_deg"]),
            np.deg2rad(seed["lorentzian_hwhm_deg"]),
            seed["eta"],
            simplex_shares(seed["surface_fractions"]),
            max(0, extra),
            stack.top_roughness_A,
            stack.bottom_roughness_A,
        ]
    return model, tuple(float(v) for v in values), n


def canonical_native(draft):
    from painted_ewald import MosaicParameters
    from rasim_next.fitting.native_input import load_native_fit_physics

    record = json.loads(draft.physics_json)
    if record.get("sample_id") != draft.recipe:
        raise ValueError("physical declaration differs from the selected native recipe")
    _source_admission(record)
    physics = load_native_fit_physics(draft.physics_path, source_bytes=draft.physics_json.encode())
    model = native_model(physics)
    if (
        tuple(model.parameter_names) != draft.parameter_names
        or tuple(model.parameter_units) != draft.parameter_units
    ):
        raise ValueError("native coordinate names/order/units differ from the canonical model")
    bound, arguments, mosaic, stack = model.bind(draft.parameter_values, draft.coherent_repeats)
    from rasim_next.fitting.pb_native import simplex_shares

    values = dict(zip(draft.parameter_names, draft.parameter_values, strict=True))
    if draft.surface_fractions is not None:
        if len(draft.surface_fractions) != 3 or not np.array_equal(
            simplex_shares(draft.surface_fractions),
            [values["surface_fraction_0"], values["surface_1_share_of_remainder"]],
        ):
            raise ValueError("exact surface fractions conflict with their declared model shares")
        arguments["surface_fractions"] = draft.surface_fractions
    if draft.phase_fractions is not None:
        count = len(bound.structure.stacking_phases) or 1
        if len(draft.phase_fractions) != count or not np.array_equal(
            simplex_shares(draft.phase_fractions),
            [values[f"phase_{i}_share_of_remainder"] for i in range(count - 1)],
        ):
            raise ValueError("exact phase fractions conflict with their declared model shares")
        arguments["phase_fractions"] = draft.phase_fractions
    proposal = mosaic if draft.proposal_mosaic is None else MosaicParameters(*draft.proposal_mosaic)
    # Stack override precedes source partitioning, so local-m0 keeps the bound declaration.
    bound = replace(bound, specular_stitch_stack=stack)
    parts = bound.integration_parts()
    if any(len(p.source.mean_rays.wavelength_A) > MAX_NATIVE_SIMULATION_SOURCES for p in parts):
        raise ValueError("complete native source exceeds 256 states; no support was pruned")
    rows, columns = bound.instrument.detector_shape_rc
    if (
        rows * columns > MAX_SIMULATION_PIXELS
        or max(rows, columns) > 16384
        or rows % draft.bin_size_px
        or columns % draft.bin_size_px
    ):
        raise ValueError("panel exceeds desktop limits or bin size does not exactly divide it")
    if len(bound.rods) > MAX_SIMULATION_RODS:
        raise ValueError("complete native rod roster exceeds 2048; no pruning is permitted")
    detectors = tuple(
        replace(
            part.detector(mosaic=mosaic, **arguments),
            specular_stitch_stack=part.specular_stitch_stack,
            proposal_mosaic=proposal,
        )
        for part in parts
    )
    return (
        bound,
        arguments,
        mosaic,
        proposal,
        detectors,
        model.inactive_parameters(draft.parameter_values),
    )


def _source_admission(record):
    rule, source = record["source_rule"], record["source"]
    if rule["kind"] == "latin_hypercube":
        count = rule.get("sample_count", 32)
    else:
        order = max(rule.get("divergence_order", 6), rule.get("local_m0_divergence_order") or 0)
        divergence_axes = sum(float(v) > 0 for v in source["divergence_sigma_rad"])
        count = (
            order**divergence_axes
            * len(source["line_wavelength_A"])
            * (rule.get("wavelength_order", 6) if source["common_wavelength_sigma_A"] else 1)
        )
    if type(count) is not int or not 1 <= count <= MAX_NATIVE_SIMULATION_SOURCES:
        raise ValueError("declared complete native source exceeds the 256-state desktop cap")


def prepare_native_draft(argument, control):
    from rasim_next.fitting.native_input import load_native_fit_physics
    from rasim_next.pipeline.fiber_detector import FiberIntegrationRule

    request = json.loads(argument)
    _stop(control)
    if request["operation"] == "native_save":
        from native_simulation_state import native_draft_document
        from simulation_io import _external, _publish_bytes

        draft = native_draft_from_document(request["draft"])
        canonical_native(draft)
        path = Path(request["path"]).resolve(strict=False)
        _external(path, (draft.physics_path,))
        encoded = json.dumps(
            {"schema": "slate.native-draft.v1", "draft": native_draft_document(draft)},
            sort_keys=True,
            allow_nan=False,
        ).encode()
        _publish_bytes(path, encoded, control)
        observed, _identity = bounded_reference_snapshot(path)
        if observed != encoded:
            raise ValueError("native draft export readback differs")
        canonical_native(native_draft_from_document(json.loads(observed)["draft"]))
        return JobResult(("native_save", str(path)), 4096)
    if request["operation"] == "native_load":
        path = Path(request["path"]).resolve(strict=True)
        raw, _identity = bounded_reference_snapshot(path)
        if len(raw) > 384 * 1024:
            raise ValueError("native draft envelope exceeds 384 KiB")
        record = json.loads(raw)
        if record.get("schema") == "slate.native-draft.v1":
            if set(record) != {"schema", "draft"}:
                raise ValueError("invalid native draft envelope")
            draft = native_draft_from_document(record["draft"])
            bound = canonical_native(draft)[0]
            incidence = mean_incidence_deg(
                bound.source_definition.mean_direction_lab, bound.instrument
            )
            _stop(control)
            return JobResult(
                ("native_load", draft, incidence), len(draft.physics_json.encode()) + 16384
            )
        if len(raw) > MAX_NATIVE_PHYSICS_BYTES:
            raise ValueError("native physical input exceeds 128 KiB")
        _source_admission(record)
        physics = load_native_fit_physics(path, source_bytes=raw)
        model, values, repeats = native_initial_values(physics, request.get("physical_start"))
        # Freeze every numerical default at admission, rather than inherit future defaults.
        record["integration_rule"] = asdict(FiberIntegrationRule(**record["integration_rule"]))
        definition = asdict(physics.source_definition)
        record["source_rule"] = {
            k: definition[k]
            for k in (
                "kind",
                "sample_count",
                "seed",
                "divergence_order",
                "wavelength_order",
                "local_m0_divergence_order",
                "position_divergence_correlation",
            )
        }
        # Correlation is a physical source coordinate, regardless of the numerical rule.
        record["source"]["position_divergence_correlation"] = np.asarray(
            record["source_rule"].pop("position_divergence_correlation")
        ).tolist()
        record["spatial_quadrature_order"] = physics.spatial_quadrature_order
        draft = NativeSimulationDraft(
            uuid4(),
            0,
            physics.sample_id,
            path,
            hashlib.sha256(raw).hexdigest(),
            json.dumps(record, sort_keys=True),
            tuple(model.parameter_names),
            tuple(model.parameter_units),
            values,
            repeats,
            surface_fractions=None
            if request.get("physical_start") is None
            else tuple(request["physical_start"]["surface_fractions"]),
            phase_fractions=None
            if request.get("physical_start") is None
            else tuple(request["physical_start"]["phase_fractions"]),
        )
    else:
        draft = native_draft_from_document(request["draft"])
    control.report("Validating canonical native model, source, structure and complete geometry")
    bound = canonical_native(draft)[0]
    incidence = mean_incidence_deg(bound.source_definition.mean_direction_lab, bound.instrument)
    _stop(control)
    return JobResult(
        (request["operation"], draft, incidence), len(draft.physics_json.encode()) + 16384
    )


def native_budget(draft, bound, parts, other_cpu_bytes=0, other_gpu_bytes=0):
    native_pixels = np.prod(bound.instrument.detector_shape_rc, dtype=np.int64).item()
    pixels = native_pixels // draft.bin_size_px**2
    rule = bound.integration_rule
    # Conservative admission estimate, not a measured process-memory ceiling.
    # Regular strength panels and the stitched m0 chart have different work counts.
    model_reserve = 512 * 1024**2
    for part in parts:
        source_count = len(part.source.mean_rays.wavelength_A)
        local_rods = sum(
            rod.h == rod.k == 0 and part.specular_stitch_stack is not None for rod in part.rods
        )
        powers = [rule.angular_initial_power]
        if local_rods:
            powers.extend((rule.local_m0_axial_power, rule.local_m0_angular_power))
        if max(powers) > 24:
            raise ValueError("declared quadrature node tables exceed the desktop memory cap")
        regular_nodes = max(rule.strength_gauss_order, rule.strength_scalar_order)
        axial_nodes = (len(part.rods) - local_rods) * regular_nodes
        if local_rods:
            axial_nodes += local_rods * 2**rule.local_m0_axial_power
        model_reserve += source_count * axial_nodes * 64
    patch_reserve = max(rule.pixel_error_maximum_bytes, native_pixels * 8)
    if draft.bin_size_px > 1:
        patch_reserve += (native_pixels + pixels) * 8
    event_reserve = rule.batch_size * 2048 + rule.maximum_angular_panel_nodes * 64 + patch_reserve
    cpu = model_reserve + event_reserve + pixels * 117 + len(draft.physics_json.encode()) * 4
    # Presentation/history staging is included above; numerical work remains CPU-only.
    gpu = pixels * 4
    if (
        type(other_cpu_bytes) is not int
        or other_cpu_bytes < 0
        or type(other_gpu_bytes) is not int
        or other_gpu_bytes < 0
        or cpu + other_cpu_bytes > MAX_SIMULATION_CPU_BYTES
        or gpu + other_gpu_bytes > MAX_SIMULATION_GPU_BYTES
    ):
        raise ValueError(
            "combined native working/result/display/history budget exceeds 2 GiB CPU or 512 MiB GPU"
        )
    return {
        "cpu_reserved_bytes": cpu,
        "gpu_reserved_bytes": gpu,
        "other_cpu_bytes": other_cpu_bytes,
        "other_gpu_bytes": other_gpu_bytes,
        "model_reserved_bytes": model_reserve,
        "reservation_kind": "conservative estimate; process peak unmeasured",
        "event_reserved_bytes": event_reserve,
        "cpu_cap_bytes": MAX_SIMULATION_CPU_BYTES,
        "gpu_cap_bytes": MAX_SIMULATION_GPU_BYTES,
        "cpu_workers_requested": 1,
        "nested_blas_threads": 1,
        "gpu_allocator_peak": "unmeasured; CPU numerical route",
    }


def run_native_simulation(argument, control):
    from threadpoolctl import threadpool_limits

    request = json.loads(argument)
    draft = native_draft_from_document(request["draft"])
    start = perf_counter()
    control.report(
        "Constructing native model; setup and individual patch generation are noninterruptible"
    )
    _stop(control)
    with threadpool_limits(limits=1):
        bound, _arguments, _mosaic, proposal, parts, inactive = canonical_native(draft)
        ledger = native_budget(
            draft,
            bound,
            parts,
            request.get("other_cpu_bytes", 0),
            request.get("other_gpu_bytes", 0),
        )
        _stop(control)
        rows, columns = bound.instrument.detector_shape_rc
        shape = rows // draft.bin_size_px, columns // draft.bin_size_px
        image = np.zeros(shape, dtype=np.float64)
        run_id = str(uuid4())
        numerical = {
            "integration": asdict(bound.integration_rule),
            "source_rule": json.loads(draft.physics_json)["source_rule"],
            "spatial_quadrature_order": bound.spatial_quadrature_order,
            "bin_size_px": draft.bin_size_px,
            "batch_offset": 0,
            "proposal_mosaic": asdict(proposal),
            "cpu_workers": 1,
            "nested_blas_threads": 1,
        }
        numerical_hash = hashlib.sha256(json.dumps(numerical, sort_keys=True).encode()).hexdigest()
        measure = (
            "raw_detector_macrobin_fixed_quadrature_estimate_A2.v1"
            if draft.bin_size_px > 1
            else "raw_detector_pixel_mass_A2.v1"
        )
        coordinates = []
        for name, size in (("column", columns), ("row", rows)):
            edges = np.arange(0, size + 1, draft.bin_size_px, dtype=np.float64) - 0.5
            centers = (edges[:-1] + edges[1:]) / 2
            edges.setflags(write=False)
            centers.setflags(write=False)
            coordinates.extend(((name + "_edges_px", edges), (name + "_centers_px", centers)))
        count = 0
        part_batches = [0] * len(parts)
        setup_s = perf_counter() - start
        max_batch_s = 0.0
        metadata = {
            "recipe": draft.recipe,
            "physics_sha256": draft.physics_sha256,
            "bound_input_revision": bound.input_revision,
            "parameter_sha256": draft.parameter_sha256,
            "source_revision": bound.source.revision,
            "rod_catalog_revision": bound.rod_catalog_revision,
            "numerical_sha256": numerical_hash,
            "effective_numerics": numerical,
            "native_shape_rc": [rows, columns],
            "output_shape_rc": list(shape),
            "completed_support": "sum of yielded accepted native patches; incomplete until all parts finish",
            "completion_counter_unit": "accepted native patch",
            "part_revisions": [
                {"source": part.source.revision, "detector": part.fixed_physics_revision}
                for part in parts
            ],
            "independent_parts": len(parts),
            "inactive_parameters": inactive,
            "resource_ledger": ledger,
            "support_kind": "whole_panel_additive_native_patch_stream",
            "setup_s": setup_s,
        }
        for index, detector in enumerate(parts):
            stream = detector.iter_native_pixel_patches(
                bin_size_px=draft.bin_size_px, batch_offset=0
            )
            try:
                while True:
                    _stop(control)
                    batch_start = perf_counter()
                    try:
                        (r0, r1, c0, c1), contribution = next(stream)
                    except StopIteration:
                        break
                    max_batch_s = max(max_batch_s, perf_counter() - batch_start)
                    _stop(control)
                    image[r0:r1, c0:c1] += contribution
                    del contribution
                    count += 1
                    part_batches[index] += 1
                    control.report(
                        f"Native integration: part {index + 1}/{len(parts)}, {count} accepted patches; incomplete integral"
                    )
                    if control.take_inspection():
                        snapshot = image.copy()
                        snapshot.setflags(write=False)
                        frame = _prepared_frame(
                            draft,
                            run_id,
                            snapshot,
                            count,
                            measure,
                            "cpu",
                            True,
                            coordinates,
                            {
                                **metadata,
                                "integration_complete": False,
                                "completed_part_batches": part_batches.copy(),
                                "max_batch_s": max_batch_s,
                                "elapsed_s": perf_counter() - start,
                            },
                        )
                        control.publish(JobResult(frame, frame.nbytes))
            finally:
                stream.close()
                del stream, detector
        _stop(control)
        image.setflags(write=False)
        frame = _prepared_frame(
            draft,
            run_id,
            image,
            count,
            measure,
            "cpu",
            True,
            coordinates,
            {
                **metadata,
                "integration_complete": True,
                "completed_part_batches": part_batches,
                "max_batch_s": max_batch_s,
                "elapsed_s": perf_counter() - start,
            },
        )
        return JobResult(frame, frame.nbytes)
