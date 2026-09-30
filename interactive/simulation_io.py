"""Canonical independent simulation workers and non-executable result publication."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
import yaml
from detector_panel import BandProfiles, exact_band_profiles
from job_lifecycle import MAX_RESULT_BYTES, JobControl, JobResult
from metadata_review import MAX_REFERENCE_BYTES, bounded_reference_snapshot
from project_state import linear_display_limits
from simulation_state import (
    MAX_SIMULATION_YAML_BYTES,
    SimulationDraft,
    SimulationExportWork,
    SimulationReference,
    simulation_draft_document,
    simulation_draft_from_document,
    simulation_reference_from_document,
)

MAX_SIMULATION_CPU_BYTES = 2 * 1024**3
MAX_SIMULATION_GPU_BYTES = 512 * 1024**2
MAX_SIMULATION_PIXELS = 12_000_000
MAX_SIMULATION_SOURCES = 256
MAX_SIMULATION_RODS = 2048
MAX_AUXILIARY_BYTES = 32 * 1024**2


def _stop(control: JobControl) -> None:
    if control.canceled:
        raise RuntimeError("simulation canceled; obsolete publications discarded")


def canonical_configuration(draft: SimulationDraft):
    from rasim_next.pipeline.configured_simulation import load_simulation_config

    config = load_simulation_config(
        draft.configuration_path,
        source_bytes=draft.yaml_text.encode(),
        max_referenced_cif_bytes=MAX_REFERENCE_BYTES,
    )
    if config.material.cif_path != draft.cif_path or config.cif_sha256 != draft.cif_sha256:
        raise ValueError(
            "simulation CIF differs from the immutable draft; load or validate the new input explicitly"
        )
    return config


def prepare_simulation_draft(argument: bytes, control: JobControl) -> JobResult:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        load_simulation_config,
        load_strict_yaml_mapping,
    )

    request = json.loads(argument)
    operation = request["operation"]
    old = simulation_draft_from_document(request.get("draft"))
    _stop(control)
    if operation == "load":
        path = Path(request["path"]).resolve(strict=True)
        raw, _identity = bounded_reference_snapshot(path)
        if len(raw) > MAX_SIMULATION_YAML_BYTES:
            raise ValueError("simulation configuration exceeds 64 KiB")
        text = raw.decode("utf-8")
        imported = hashlib.sha256(raw).hexdigest()
    elif operation == "validate":
        if old is None:
            raise ValueError("load an independent configuration first")
        path, text, imported = old.configuration_path, request["yaml_text"], old.imported_sha256
        if len(text.encode()) > MAX_SIMULATION_YAML_BYTES:
            raise ValueError("simulation configuration exceeds 64 KiB")
    elif operation == "save_configuration":
        if old is None:
            raise ValueError("no independent configuration to export")
        config = canonical_configuration(old)
        path = Path(request["path"]).resolve(strict=False)
        _external(path, (old.configuration_path, old.cif_path))
        mapping = load_strict_yaml_mapping(
            old.configuration_path, source_bytes=old.yaml_text.encode()
        )
        mapping["material"]["cif_path"] = str(config.material.cif_path)
        text = yaml.safe_dump(mapping, sort_keys=False)
        _publish_bytes(path, text.encode(), control)
        observed, _ = bounded_reference_snapshot(path)
        loaded = load_simulation_config(
            path, source_bytes=observed, max_referenced_cif_bytes=MAX_REFERENCE_BYTES
        )
        if observed != text.encode() or loaded.physics_revision != config.physics_revision:
            raise ValueError("exported configuration readback differs")
        return JobResult((operation, str(path)), 4096)
    else:
        raise ValueError("unsupported independent draft operation")
    control.report("Validating complete configuration and canonical geometry/source")
    config = load_simulation_config(
        path, source_bytes=text.encode(), max_referenced_cif_bytes=MAX_REFERENCE_BYTES
    )
    if config.source.sample_count > MAX_SIMULATION_SOURCES:
        if operation == "validate":
            raise ValueError(
                "complete canonical desktop validation requires source count <=256; imported values are preserved, choose an explicit supported count"
            )
        operation = "load_limited"
    else:
        build_configured_simulation_inputs(config)
    _stop(control)
    settings = request.get("settings", {})
    result = SimulationDraft(
        old.draft_id if old else uuid4(),
        old.revision + 1 if old else 0,
        path,
        imported,
        text,
        config.material.cif_path,
        config.cif_sha256,
        settings.get("route", old.route if old else "monte_carlo"),
        settings.get("position_mode", old.position_mode if old else "sampled"),
        settings.get("draw_count", old.draw_count if old else 8),
        settings.get("detector_seed", old.detector_seed if old else 1729),
    )
    return JobResult((operation, result), len(text.encode()) + 8192)


def simulation_budget(
    config, draft: SimulationDraft, other_cpu_bytes: int = 0, other_gpu_bytes: int = 0
) -> dict:
    rows, columns = config.instrument.detector_shape_rc
    pixels = rows * columns
    workers = config.numerics.worker_count
    if pixels > MAX_SIMULATION_PIXELS or max(rows, columns) > 16384:
        raise ValueError(
            "native detector exceeds the desktop 12-million-pixel / 16384-axis cap; physical shape is never reduced"
        )
    if config.source.sample_count > MAX_SIMULATION_SOURCES or not 1 <= workers <= 4:
        raise ValueError(
            "desktop execution requires source count <=256 and an explicitly selected CPU worker limit in [1,4]; imported values are preserved"
        )
    if (
        type(other_cpu_bytes) is not int
        or other_cpu_bytes < 0
        or type(other_gpu_bytes) is not int
        or other_gpu_bytes < 0
    ):
        raise ValueError("invalid shared resource charge")
    # Raw/worker accumulators, lease/copy, current and retained quantitative snapshots,
    # profile preparation/snapshot construction transients, display/staging and packed model reserve.
    numeric = (
        pixels * (8 * (4 + 4) + 4 * 4 + 9)
        + 512 * 1024**2
        + MAX_AUXILIARY_BYTES
        + 8 * draft.draw_count
    )
    gpu = (
        pixels * (8 + 4) * 2 if config.numerics.detector_execution_backend == "cuda" else pixels * 4
    )
    if (
        numeric + other_cpu_bytes > MAX_SIMULATION_CPU_BYTES
        or gpu + other_gpu_bytes > MAX_SIMULATION_GPU_BYTES
    ):
        raise ValueError(
            "combined simulator/inspection resource budget exceeds 2 GiB CPU or 512 MiB GPU; close retained data or choose a supported explicit worker count"
        )
    return {
        "cpu_reserved_bytes": numeric,
        "other_cpu_bytes": other_cpu_bytes,
        "gpu_reserved_bytes": gpu,
        "other_gpu_bytes": other_gpu_bytes,
        "cpu_cap_bytes": MAX_SIMULATION_CPU_BYTES,
        "gpu_cap_bytes": MAX_SIMULATION_GPU_BYTES,
        "cpu_workers_requested": workers,
        "nested_blas_threads": 1,
        "publication_cap_bytes": MAX_RESULT_BYTES,
        "model_reserve_bytes": 512 * 1024**2,
    }


@dataclass(frozen=True, slots=True)
class SimulationFrame:
    draft: SimulationDraft
    run_id: str
    draw_prefix: int
    measure: str
    backend: str
    quantitative: bool
    image: np.ndarray | None
    display: np.ndarray | None
    profiles: BandProfiles | None
    full_profiles: BandProfiles | None
    low: float
    high: float
    maximum: float
    min_positive: float | None
    arrays: tuple[tuple[str, np.ndarray], ...]
    manifest: bytes

    @property
    def nbytes(self) -> int:
        arrays = [v for v in (self.image, self.display) if v is not None]
        arrays.extend(v for _, v in self.arrays)
        for profiles in (self.profiles, self.full_profiles):
            if profiles is not None:
                arrays.extend(
                    (
                        profiles.horizontal,
                        profiles.vertical,
                        profiles.horizontal_support,
                        profiles.vertical_support,
                    )
                )
        unique = {id(v): v for v in arrays}
        return (
            sum(v.nbytes for v in unique.values())
            + len(self.manifest)
            + len(self.draft.yaml_text.encode())
            + 4096
        )


def _prepared_frame(
    draft, run_id, image, prefix, measure, backend, quantitative, arrays, metadata
) -> SimulationFrame:
    profiles = full = None
    display = None
    low, high, maximum, minimum_positive = 0.0, 1.0, 0.0, None
    if image is not None:
        if quantitative:
            if image.dtype != np.float64 or image.flags.writeable:
                raise ValueError("quantitative simulation publication needs immutable float64")
            display = np.ascontiguousarray(image, dtype=np.float32)
            display.setflags(write=False)
            profiles = exact_band_profiles(
                image, column_px=image.shape[1] // 2, row_px=image.shape[0] // 2
            )
            full = exact_band_profiles(
                image, column_px=image.shape[1] // 2, row_px=image.shape[0] // 2, scope="full"
            )
        else:
            # The caller has already consumed the canonical float32 lease with exactly one copy.
            display = image
        minimum, maximum = np.inf, -np.inf
        positive = np.inf
        for start in range(0, image.shape[0], 64):
            block = image[start : start + 64]
            finite = np.isfinite(block)
            minimum = min(minimum, float(np.min(block, where=finite, initial=np.inf)))
            maximum = max(maximum, float(np.max(block, where=finite, initial=-np.inf)))
            positive = min(
                positive, float(np.min(block, where=finite & (block > 0), initial=np.inf))
            )
        if not np.isfinite(minimum):
            minimum, maximum = 0.0, 1.0
        minimum_positive = positive if np.isfinite(positive) else None
        low, high = linear_display_limits(minimum, maximum)
    metadata = {
        **metadata,
        "schema": "slate.configured-snapshot.v1",
        "draft": simulation_draft_document(draft),
        "run_id": run_id,
        "draw_prefix": prefix,
        "measure": measure,
        "backend": backend,
        "quantitative": quantitative,
        "units": (
            "angstrom^2/rad^2"
            if draft.route in ("reciprocal_space", "ewald_surface")
            else "angstrom^2/pixel^2"
            if draft.route == "pixel_centers"
            else "angstrom^2"
        ),
        "qualification": "nominal; binding fidelity only, no convergence or fitting qualification",
    }
    manifest = json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
    result = SimulationFrame(
        draft,
        run_id,
        prefix,
        measure,
        backend,
        quantitative,
        image,
        display,
        profiles,
        full,
        low,
        high,
        maximum,
        minimum_positive,
        tuple(arrays),
        manifest,
    )
    if result.nbytes > MAX_RESULT_BYTES:
        raise ValueError("snapshot and auxiliary arrays exceed the 160 MiB publication cap")
    return result


def progressive_prefixes(final: int) -> tuple[int, ...]:
    values = [1]
    current = 4
    while current < final:
        values.append(current)
        current *= 4
    return (*tuple(v for v in values if v < final), final)


def run_simulation(argument: bytes, control: JobControl) -> JobResult:
    from threadpoolctl import threadpool_limits

    from rasim_next.pipeline.beam_position import ConditionalBeamPosition
    from rasim_next.pipeline.bragg_space import Bi2X3FiniteStackStrength
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_nominal_ewald_context,
        build_source_averaged_detector,
        evaluate_nominal_ewald_surface,
        integrate_detector_macrobins,
        sample_detector_pixel_center_density,
        sample_reciprocal_space,
    )

    request = json.loads(argument)
    draft = simulation_draft_from_document(request["draft"])
    if draft is None:
        raise ValueError("no independent simulation draft")
    start = perf_counter()
    config = canonical_configuration(draft)
    ledger = simulation_budget(
        config, draft, request.get("other_cpu_bytes", 0), request.get("other_gpu_bytes", 0)
    )
    conditional = draft.position_mode == "conditional_position"
    if conditional and draft.route != "monte_carlo":
        raise ValueError(
            "conditional-position source means require their MC position integral; use sampled mode for density/display routes"
        )
    if (
        draft.route in ("monte_carlo", "pixel_centers", "macrobins")
        and not config.outputs.detector.enabled
    ):
        raise ValueError(
            "selected detector route requires outputs.detector.enabled; preserved configuration is not overridden"
        )
    if (
        draft.route in ("reciprocal_space", "ewald_surface")
        and not getattr(config.outputs, draft.route).enabled
    ):
        raise ValueError(
            "selected output route is disabled in the configuration; enable it explicitly"
        )
    if draft.route in ("reciprocal_space", "ewald_surface") and config.outputs.detector.enabled:
        raise ValueError(
            "auxiliary-only route cannot produce selected detector output; choose a detector route or explicitly deselect detector output"
        )
    control.report("Constructing canonical physics; material/setup/JIT phases are noninterruptible")
    _stop(control)
    sampler = None
    with threadpool_limits(limits=1):
        inputs = build_configured_simulation_inputs(config, conditional_source_position=conditional)
        setup_s = perf_counter() - start
        _stop(control)
        if len(inputs.rods) > MAX_SIMULATION_RODS:
            raise ValueError(
                "complete physical rod support exceeds desktop 2048-rod cap; no rods were pruned"
            )
        auxiliary = []
        if config.outputs.reciprocal_space.enabled:
            size = (
                len(inputs.bragg_space.config.rods)
                * config.numerics.reciprocal_alpha_count
                * config.numerics.reciprocal_beta_count
                * config.numerics.reciprocal_u_count
            )
            if size * 40 > MAX_AUXILIARY_BYTES:
                raise ValueError("complete reciprocal display exceeds 32 MiB auxiliary cap")
            control.report("Sampling canonical reciprocal display")
            data = sample_reciprocal_space(inputs)
            auxiliary.extend(
                (
                    ("reciprocal_q_sample_Ainv", data.q_sample_Ainv),
                    ("reciprocal_density_A2_rad2_inv", data.intensity_density_A2_rad2_inv),
                    ("reciprocal_family_m", data.family_m),
                )
            )
        _stop(control)
        if config.outputs.ewald_surface.enabled:
            size = (
                len(inputs.rods)
                * config.numerics.ewald_alpha_count
                * config.numerics.ewald_beta_count
                * 2
            )
            if size * 56 + sum(v.nbytes for _, v in auxiliary) > MAX_AUXILIARY_BYTES:
                raise ValueError("complete Ewald display exceeds 32 MiB auxiliary cap")
            control.report("Sampling canonical nominal Ewald display")
            data = evaluate_nominal_ewald_surface(
                build_nominal_ewald_context(inputs),
                alpha_count=config.numerics.ewald_alpha_count,
                beta_count=config.numerics.ewald_beta_count,
                alpha_max_deg=config.numerics.ewald_alpha_max_deg,
            )
            auxiliary.extend(
                (
                    ("ewald_q_sample_Ainv", data.q_sample_Ainv),
                    ("ewald_density_A2_rad2_inv", data.coating_intensity_density_A2_rad2_inv),
                    ("ewald_family_m", data.family_m),
                    ("ewald_branch", data.branch),
                    ("ewald_residual_Ainv", data.ewald_residual_Ainv),
                )
            )
        _stop(control)
        metadata = {
            "configuration_sha256": draft.configuration_sha256,
            "imported_configuration_sha256": draft.imported_sha256,
            "cif_sha256": config.cif_sha256,
            "source_revision": inputs.incident.states.source_revision,
            "source_seed": config.source.seed,
            "source_state_count": inputs.incident.states.source_weight.size,
            "physics_revision": config.physics_revision,
            "render_revision": config.render_revision,
            "numerics": {
                name: getattr(config.numerics, name)
                for name in config.numerics.__dataclass_fields__
            },
            "resource_ledger": ledger,
            "setup_s": setup_s,
            "auxiliary_s": perf_counter() - start - setup_s,
            "preview_lease_copy_bytes": 0,
            "noninterruptible": "canonical material/model setup, JIT, auxiliary display sampling, snapshot reductions and file publication; sampler advance has canonical cancellation boundaries",
        }
        run_id = str(uuid4())
        if draft.route in ("reciprocal_space", "ewald_surface"):
            measure = (
                "latent_bragg_density_A2_rad2_inv.v1"
                if draft.route == "reciprocal_space"
                else "detector_visible_intrinsic_ewald_latent_density_A2_rad2_inv.v1"
            )
            frame = _prepared_frame(
                draft, run_id, None, 0, measure, "canonical_numpy", True, auxiliary, metadata
            )
            return JobResult(frame, frame.nbytes)
        if not isinstance(inputs.strength, Bi2X3FiniteStackStrength):
            raise ValueError(
                "full-image detector execution requires Bi2X3FiniteStackStrength; generic-CIF YAML admission does not supply this renderer"
            )
        control.report(
            "Building complete source/rod/root detector; compilation is noninterruptible"
        )
        detector = build_source_averaged_detector(inputs)
        _stop(control)
        backend = config.numerics.detector_execution_backend
        try:
            if draft.route == "monte_carlo":
                beam = (
                    ConditionalBeamPosition.from_source(
                        config.source, source_revision=inputs.incident.states.source_revision
                    )
                    if conditional
                    else None
                )
                sampler = detector.compile_monte_carlo_sampler(
                    execution_backend=backend, seed=draft.detector_seed, beam_position=beam
                )
                for prefix in progressive_prefixes(draft.draw_count):
                    _stop(control)
                    control.report(
                        f"Sampling {draft.position_mode}: prefix {prefix}/{draft.draw_count}; {backend}"
                    )
                    if prefix == draft.draw_count or control.take_inspection():
                        estimate = sampler.advance_to(
                            prefix, cancel_requested=lambda: control.canceled
                        )
                        arrays = [
                            *auxiliary,
                            ("replicate_total_mass_A2", estimate.replicate_total_mass_A2),
                        ]
                        meta = {
                            **metadata,
                            "attempted_root_count": estimate.attempted_root_count,
                            "visible_hit_count": estimate.visible_hit_count,
                            "total_detector_mass_A2": estimate.total_detector_mass_A2,
                            "maximum_root_deposit_A2": estimate.maximum_root_deposit_A2,
                            "execution_worker_count": estimate.execution_worker_count,
                            "execution_device": estimate.execution_device,
                            "position_integration_model_id": estimate.position_integration_model_id,
                            "rng_model_id": estimate.rng_model_id,
                            "proposal_id": estimate.proposal_id,
                            "rod_catalog_revision": estimate.rod_catalog_revision,
                        }
                        frame = _prepared_frame(
                            draft,
                            run_id,
                            estimate.image_A2,
                            prefix,
                            estimate.measure_id,
                            estimate.execution_backend,
                            True,
                            arrays,
                            meta,
                        )
                    else:
                        lease = sampler.advance_preview_to(
                            prefix, cancel_requested=lambda: control.canceled
                        )
                        image = np.array(lease.image_A2, copy=True, order="C")
                        image.setflags(write=False)
                        metadata["preview_lease_copy_bytes"] += image.nbytes
                        frame = _prepared_frame(
                            draft,
                            run_id,
                            image,
                            prefix,
                            "raw_detector_pixel_mass_conditional_position_estimate_A2.v1"
                            if conditional
                            else "raw_detector_pixel_mass_monte_carlo_estimate_A2.v1",
                            lease.execution_backend,
                            False,
                            auxiliary,
                            metadata,
                        )
                        del lease
                    _stop(control)
                    if prefix != draft.draw_count:
                        control.publish(JobResult(frame, frame.nbytes))
                _stop(control)
            elif draft.route == "pixel_centers":
                data = sample_detector_pixel_center_density(detector, execution_backend=backend)
                frame = _prepared_frame(
                    draft,
                    run_id,
                    data.image_A2_per_px2,
                    0,
                    data.measure_id,
                    data.execution_backend,
                    True,
                    auxiliary,
                    metadata,
                )
            else:
                data = integrate_detector_macrobins(
                    detector,
                    bin_size_px=config.numerics.detector_macrobin_size_px,
                    gauss_order=config.numerics.detector_gauss_order,
                    execution_backend=backend,
                )
                arrays = [
                    *auxiliary,
                    ("macrobin_column_center_px", data.column_center_px),
                    ("macrobin_row_center_px", data.row_center_px),
                    ("macrobin_valid_source_count_min", data.valid_source_count_min),
                ]
                frame = _prepared_frame(
                    draft,
                    run_id,
                    data.image_A2,
                    0,
                    data.measure_id,
                    data.execution_backend,
                    True,
                    arrays,
                    metadata,
                )
            _stop(control)
            return JobResult(frame, frame.nbytes)
        finally:
            # All sampler operations and final reference release occur on this owning worker.
            if sampler is not None:
                sampler.reset()
            sampler = None
            detector = None


def prepare_simulation_profiles(work, control: JobControl) -> JobResult:
    _stop(control)
    column, row, row_width, column_width, scope, roi, measure = work.query
    value = exact_band_profiles(
        work.native,
        column_px=column,
        row_px=row,
        row_width=row_width,
        column_width=column_width,
        scope=scope,
        roi_column_row_bounds=roi,
        measure=measure,
    )
    _stop(control)
    size = sum(
        v.nbytes
        for v in (
            value.horizontal,
            value.vertical,
            value.horizontal_support,
            value.vertical_support,
        )
    )
    return JobResult((id(work.native), work.query, value), size + 4096)


def _external(path: Path, protected=()) -> None:
    if (
        any((parent / ".git").exists() for parent in (path, *path.parents))
        or path in protected
        or any(path.exists() and p.exists() and os.path.samefile(path, p) for p in protected)
    ):
        raise ValueError("choose a new result destination outside Git and source/project files")
    if path.exists() or not path.parent.is_dir():
        raise ValueError("destination must be new with an existing parent directory")


def _publish_bytes(path: Path, encoded: bytes, control: JobControl) -> None:
    temporary = path.with_name(path.name + "." + uuid4().hex + ".part")
    try:
        _stop(control)
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _stop(control)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_configured_figures(draft: SimulationDraft, arrays: dict, control: JobControl) -> None:
    from io import BytesIO

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    config = canonical_configuration(draft)
    directory = config.output_directory
    if not directory.is_dir():
        raise ValueError("create the configured external output directory before exporting figures")
    selected = []
    for name in ("reciprocal_space", "ewald_surface", "detector"):
        output = getattr(config.outputs, name)
        if not output.enabled:
            continue
        key = (
            "image"
            if name == "detector"
            else (
                "reciprocal_q_sample_Ainv" if name == "reciprocal_space" else "ewald_q_sample_Ainv"
            )
        )
        if key not in arrays:
            raise ValueError(
                f"snapshot has no selected {name} output; choose a route producing it or deselect it explicitly"
            )
        path = directory / output.filename
        _external(path, (draft.configuration_path, draft.cif_path))
        selected.append((name, path, key))
    # Check every destination before publishing any file; each write remains no-overwrite.
    for name, path, key in selected:
        _stop(control)
        figure = Figure(figsize=(7, 6), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot()
        value = arrays[key]
        if name == "detector":
            row_stride = max(1, (value.shape[0] + 699) // 700)
            column_stride = max(1, (value.shape[1] + 699) // 700)
            axis.imshow(
                value[::row_stride, ::column_stride],
                origin="upper",
                interpolation="nearest",
                extent=(0, value.shape[1], value.shape[0], 0),
            )
            axis.text(
                0.01,
                0.01,
                f"Display sample stride {column_stride} x {row_stride}; exact native values in NPZ",
                transform=axis.transAxes,
                fontsize=7,
            )
            axis.set_xlabel(
                "macrobin column index" if draft.route == "macrobins" else "native column (px)"
            )
            axis.set_ylabel(
                "macrobin row index" if draft.route == "macrobins" else "native row (px)"
            )
        else:
            valid = np.all(np.isfinite(value), axis=1)
            density_key = (
                "reciprocal_density_A2_rad2_inv"
                if name == "reciprocal_space"
                else "ewald_density_A2_rad2_inv"
            )
            plotted = axis.scatter(
                value[valid, 0], value[valid, 2], c=arrays[density_key][valid], s=0.5
            )
            figure.colorbar(
                plotted, ax=axis, label="density (angstrom^2/rad^2), linear color scale"
            )
            axis.set_xlabel("sample Qx (angstrom^-1)")
            axis.set_ylabel("sample Qz (angstrom^-1)")
        axis.set_title(f"{name}: {draft.route}; nominal display, not convergence qualified")
        encoded = BytesIO()
        canvas.print_png(encoded)
        raw = encoded.getvalue()
        _publish_bytes(path, raw, control)
        if path.read_bytes() != raw:
            raise ValueError("configured figure readback differs")
        figure.clear()


def export_simulation(work: SimulationExportWork, control: JobControl) -> JobResult:
    work.validate()
    manifest = json.loads(work.manifest)
    if not manifest["quantitative"]:
        raise ValueError(
            "Awaiting quantitative snapshot; preview leases cannot be exported as exact values"
        )
    draft = simulation_draft_from_document(manifest["draft"])
    _external(work.destination, (draft.configuration_path, draft.cif_path))
    if manifest.get("write_configured_figures", False):
        _write_configured_figures(draft, arrays=dict(work.arrays), control=control)
    temporary = work.destination.with_name(work.destination.name + "." + uuid4().hex + ".part")
    arrays = dict(work.arrays)
    try:
        _stop(control)
        with temporary.open("xb") as stream:
            np.savez(stream, manifest_utf8=np.frombuffer(work.manifest, dtype=np.uint8), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(temporary, allow_pickle=False, max_header_size=4096) as observed:
            if (
                set(observed.files) != {"manifest_utf8", *arrays}
                or observed["manifest_utf8"].tobytes() != work.manifest
            ):
                raise ValueError("result manifest readback differs")
            for name, expected in arrays.items():
                _stop(control)
                actual = observed[name]
                if (
                    actual.dtype != expected.dtype
                    or actual.shape != expected.shape
                    or not np.array_equal(actual, expected, equal_nan=True)
                ):
                    raise ValueError(f"result array readback differs: {name}")
        _stop(control)
        os.link(temporary, work.destination)
    finally:
        temporary.unlink(missing_ok=True)
    digest = _file_hash(work.destination, control)
    reference = SimulationReference(
        work.destination,
        digest,
        draft.draft_id,
        draft.revision,
        draft.configuration_sha256,
        draft.cif_sha256,
        draft.route,
        manifest["measure"],
        manifest["draw_prefix"],
        draft.detector_seed,
    )
    return JobResult(reference, 4096)


def _file_hash(path: Path, control: JobControl) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            _stop(control)
            digest.update(block)
    return digest.hexdigest()


def reopen_simulation_result(argument: bytes, control: JobControl) -> JobResult:
    reference = simulation_reference_from_document(json.loads(argument))
    if reference.path.stat().st_size > 192 * 1024**2:
        raise ValueError("saved result exceeds the external snapshot byte cap")
    # Hash and parse the same bounded open-file bytes; never reopen mutable pathname authority.
    with reference.path.open("rb") as stream:
        data = stream.read(192 * 1024**2 + 1)
    if len(data) > 192 * 1024**2 or hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError("saved simulation result identity changed")
    from io import BytesIO

    with ZipFile(BytesIO(data)) as archive:
        if sum(v.file_size for v in archive.infolist()) > 192 * 1024**2:
            raise ValueError("expanded result exceeds its byte cap")
    _stop(control)
    with np.load(BytesIO(data), allow_pickle=False, max_header_size=4096) as saved:
        raw = saved["manifest_utf8"]
        if raw.dtype != np.uint8 or raw.ndim != 1 or raw.nbytes > 512 * 1024:
            raise ValueError("invalid simulation result manifest")
        metadata = json.loads(raw.tobytes())
        draft = simulation_draft_from_document(metadata["draft"])
        if (
            metadata["schema"] != "slate.configured-snapshot.v1"
            or not metadata["quantitative"]
            or (
                draft.draft_id,
                draft.revision,
                draft.configuration_sha256,
                draft.cif_sha256,
                draft.route,
                draft.detector_seed,
                metadata["measure"],
                metadata["draw_prefix"],
            )
            != (
                reference.draft_id,
                reference.draft_revision,
                reference.configuration_sha256,
                reference.cif_sha256,
                reference.route,
                reference.detector_seed,
                reference.measure,
                reference.draw_prefix,
            )
        ):
            raise ValueError("saved simulation manifest/reference mismatch")
        arrays = []
        image = None
        for name in saved.files:
            if name == "manifest_utf8":
                continue
            array = saved[name]
            if array.dtype.kind not in "iufb" or not array.flags.c_contiguous:
                raise ValueError("saved result requires contiguous numeric arrays")
            array.setflags(write=False)
            if name == "image":
                if (
                    array.dtype != np.float64
                    or array.ndim != 2
                    or array.size > MAX_SIMULATION_PIXELS
                ):
                    raise ValueError("saved image is not bounded native float64")
                image = array
            else:
                arrays.append((name, array))
    del data
    frame = _prepared_frame(
        draft,
        metadata["run_id"],
        image,
        reference.draw_prefix,
        reference.measure,
        metadata["backend"],
        True,
        arrays,
        metadata,
    )
    _stop(control)
    return JobResult(frame, frame.nbytes)
