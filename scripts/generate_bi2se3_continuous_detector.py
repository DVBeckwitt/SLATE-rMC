"""Generate the one-ki, m=1 upper-root Bi2Se3 detector pushforward."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from painted_ewald import (
    ContinuousEwaldCoating,
    Rod,
)
from rasim_next.pipeline.bragg_space import Bi2X3FiniteStackStrength
from rasim_next.pipeline.configured_simulation import (
    build_configured_simulation_inputs,
    load_simulation_config,
)
from rasim_next.pipeline.continuous_detector import (
    DetectorEwaldMeasure,
    DetectorQuadrature,
    PixelIntegrationMethod,
)
from rasim_next.proof.diagnostics import write_diagnostic

ROOT = Path(__file__).resolve().parents[1]
M1_ROD_KEYS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))


class _MosaicInputError(ValueError):
    """Invalid user-selected continuous mosaic parameters."""


def _decode_reference_manifest(manifest_array: np.ndarray) -> dict[str, object]:
    if manifest_array.dtype != np.uint8 or manifest_array.ndim != 1:
        raise ValueError("reference diagnostic has an invalid manifest_json")
    try:
        manifest = json.loads(manifest_array.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("reference diagnostic has an invalid manifest_json") from error
    if not isinstance(manifest, dict):
        raise ValueError("reference diagnostic manifest must be a JSON object")
    return manifest


def _require_reference_identity(
    manifest: dict[str, object],
    expected_identity: dict[str, object],
) -> None:
    for field, expected_value in expected_identity.items():
        if manifest.get(field) != expected_value:
            raise ValueError(f"reference {field} does not match the active detector fixture")


def _load_reference_manifest(reference_path: Path) -> dict[str, object]:
    with np.load(reference_path, allow_pickle=False) as reference:
        if "manifest_json" not in reference.files:
            raise ValueError("reference diagnostic has no embedded manifest_json")
        return _decode_reference_manifest(np.asarray(reference["manifest_json"]))


def _validate_reference_strength(
    reference_path: Path,
    expected_identity: dict[str, object],
) -> None:
    """Require reference and active detector-comparison identities to match."""

    manifest = _load_reference_manifest(reference_path)
    _require_reference_identity(manifest, expected_identity)


def _reference_comparison_identity(
    detector: DetectorEwaldMeasure,
    strength: Bi2X3FiniteStackStrength,
    rods: Sequence[Rod],
    *,
    branch: int,
    configured_physics_revision: str,
) -> dict[str, object]:
    return {
        "reference_comparison_model_id": "one_ki_selected_rods_detector_mass.v2",
        "configured_physics_revision": configured_physics_revision,
        "result_measure_id": "raw_detector_pixel_mass_A2.v1",
        "wavelength_A": float(2.0 * np.pi / detector.coating.bragg_space.config.k_norm_Ainv),
        "cif_path": "examples/bi2se3/structures/Bi2Se3_vesta.cif",
        "strength_parent": strength.parent.value,
        "strength_structure_model_revision": strength.structure_model_revision,
        "strength_layer_count": strength.layers,
        "strength_normalization": strength.normalization.value,
        "strength_shared_disorder_epsilon": strength.shared_disorder_epsilon,
        "detector_shape_rc": list(detector.instrument.detector_shape_rc),
        "m1_rod_keys": [[rod.h, rod.k] for rod in rods],
        "branch": branch,
    }


def _load_reference_arrays(
    reference_path: Path,
    *,
    image_shape: tuple[int, int],
    per_rod_shape: tuple[int, ...],
    expected_identity: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, str, str]:
    diagnostic_digest = hashlib.sha256()
    with reference_path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            diagnostic_digest.update(block)
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as reference:
            required = {"manifest_json", "image_A2", "per_rod_detector_mass_A2"}
            if not required.issubset(reference.files):
                raise ValueError("reference diagnostic is missing detector arrays")
            manifest = _decode_reference_manifest(np.asarray(reference["manifest_json"]))
            _require_reference_identity(manifest, expected_identity)
            raw_image = np.asarray(reference["image_A2"])
            raw_per_rod = np.asarray(reference["per_rod_detector_mass_A2"])
    if raw_image.dtype != np.float64 or raw_per_rod.dtype != np.float64:
        raise ValueError("reference detector arrays must use float64 dtype")
    if np.iscomplexobj(raw_image) or np.iscomplexobj(raw_per_rod):
        raise ValueError("reference detector arrays must be real")
    reference_image = np.asarray(raw_image, dtype=np.float64)
    reference_per_rod = np.asarray(raw_per_rod, dtype=np.float64)
    if reference_image.shape != image_shape:
        raise ValueError("reference detector image shape does not match")
    if reference_per_rod.shape != per_rod_shape:
        raise ValueError("reference per-rod detector mass shape does not match")
    if (
        not np.all(np.isfinite(reference_image))
        or np.any(reference_image < 0.0)
        or float(np.sum(reference_image, dtype=np.float64)) <= 0.0
        or not np.all(np.isfinite(reference_per_rod))
        or np.any(reference_per_rod <= 0.0)
    ):
        raise ValueError("reference detector arrays must be finite with positive mass")
    expected_image_sha256 = manifest.get("image_sha256")
    actual_image_sha256 = hashlib.sha256(
        memoryview(np.ascontiguousarray(reference_image)).cast("B")
    ).hexdigest()
    if expected_image_sha256 != actual_image_sha256:
        raise ValueError("reference detector image does not match its declared SHA-256")
    if manifest.get("nonzero_pixel_count") != int(np.count_nonzero(reference_image)):
        raise ValueError("reference detector image does not match its declared nonzero count")
    declared_per_rod = np.asarray(manifest.get("per_rod_detector_mass_A2"))
    if (
        declared_per_rod.dtype.kind not in "iuf"
        or declared_per_rod.shape != per_rod_shape
        or not np.array_equal(declared_per_rod.astype(np.float64), reference_per_rod)
    ):
        raise ValueError("reference per-rod detector masses do not match their manifest")
    declared_total = manifest.get("total_detector_mass_A2")
    if isinstance(declared_total, (bool, np.bool_)) or not isinstance(
        declared_total,
        (int, float),
    ):
        raise ValueError("reference total detector mass is invalid")
    declared_total = float(declared_total)
    image_total = float(np.sum(reference_image, dtype=np.float64))
    per_rod_total = float(np.sum(reference_per_rod, dtype=np.float64))
    if (
        not math.isfinite(declared_total)
        or declared_total <= 0.0
        or not (
            math.isclose(image_total, declared_total, rel_tol=2.0e-12, abs_tol=1.0e-24)
            and math.isclose(per_rod_total, declared_total, rel_tol=2.0e-12, abs_tol=1.0e-24)
        )
    ):
        raise ValueError("reference detector mass does not satisfy declared conservation")
    return (
        reference_image,
        reference_per_rod,
        actual_image_sha256,
        diagnostic_digest.hexdigest(),
    )


def _peak_working_set_bytes() -> int | None:
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    succeeded = psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    return int(counters.PeakWorkingSetSize) if succeeded else None


def _validate_mosaic_degrees(
    *,
    gaussian_sigma_deg: float,
    lorentzian_hwhm_deg: float,
    eta: float,
) -> None:
    values = {
        "Gaussian sigma": gaussian_sigma_deg,
        "Lorentzian HWHM": lorentzian_hwhm_deg,
        "eta": eta,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise _MosaicInputError("mosaic parameters must be finite")
    if gaussian_sigma_deg < 0.0 or lorentzian_hwhm_deg < 0.0:
        raise _MosaicInputError("mosaic widths must be nonnegative")
    if not 0.0 <= eta <= 1.0:
        raise _MosaicInputError("eta must lie in [0, 1]")
    if eta < 1.0 and gaussian_sigma_deg == 0.0:
        raise _MosaicInputError("Gaussian sigma must be positive when eta is less than one")
    if eta > 0.0 and lorentzian_hwhm_deg == 0.0:
        raise _MosaicInputError("Lorentzian HWHM must be positive when eta is greater than zero")


def build_default_detector_measure(
    *,
    gaussian_sigma_deg: float | None = None,
    lorentzian_hwhm_deg: float | None = None,
    eta: float | None = None,
    layers: int | None = None,
    shared_disorder_epsilon: float | None = None,
) -> tuple[DetectorEwaldMeasure, tuple[Rod, ...], Rod, str]:
    """Build the YAML-authoritative 5-degree fixture with optional overrides."""

    config = load_simulation_config(ROOT / "configs" / "bi2se3_simulation.yaml")
    gaussian_sigma_deg = (
        config.mosaic.gaussian_sigma_deg if gaussian_sigma_deg is None else gaussian_sigma_deg
    )
    lorentzian_hwhm_deg = (
        config.mosaic.lorentzian_hwhm_deg if lorentzian_hwhm_deg is None else lorentzian_hwhm_deg
    )
    eta = config.mosaic.lorentzian_probability if eta is None else eta
    layers = config.structure_factor.layers if layers is None else layers
    shared_disorder_epsilon = (
        config.structure_factor.shared_disorder_epsilon
        if shared_disorder_epsilon is None
        else shared_disorder_epsilon
    )

    _validate_mosaic_degrees(
        gaussian_sigma_deg=gaussian_sigma_deg,
        lorentzian_hwhm_deg=lorentzian_hwhm_deg,
        eta=eta,
    )
    config = replace(
        config,
        source=replace(
            config.source,
            wavelength_model_id="gaussian.v1",
            wavelength_sigma_A=0.0,
            sample_count=1,
            line_wavelength_A=(),
            line_probability=(),
            common_line_sigma_A=0.0,
        ),
        mosaic=replace(
            config.mosaic,
            gaussian_sigma_deg=gaussian_sigma_deg,
            lorentzian_hwhm_deg=lorentzian_hwhm_deg,
            lorentzian_probability=eta,
        ),
        structure_factor=replace(
            config.structure_factor,
            layers=layers,
            shared_disorder_epsilon=shared_disorder_epsilon,
        ),
    )
    inputs = build_configured_simulation_inputs(config)
    m0_rod = next(rod for rod in inputs.bragg_space.config.rods if rod.family_m == 0)
    m1_rods = tuple(rod for rod in inputs.bragg_space.config.rods if rod.family_m == 1)
    if tuple((rod.h, rod.k) for rod in m1_rods) != M1_ROD_KEYS:
        raise RuntimeError("configured default m=1 rod ordering changed")
    coating = ContinuousEwaldCoating(
        inputs.bragg_space,
        ki_sample_Ainv=inputs.incident.states.k_film_phase_sample_Ainv[0],
    )
    return (
        DetectorEwaldMeasure(
            coating=coating,
            incident=inputs.incident,
            material=inputs.material,
            instrument=inputs.instrument,
            phase_population_weight=config.weights.phase_population,
            polarization_weight=config.weights.polarization,
        ),
        m1_rods,
        m0_rod,
        config.physics_revision,
    )


def _center_valid_mask(detector: DetectorEwaldMeasure, *, row_chunk_size: int) -> np.ndarray:
    rows, columns = detector.instrument.detector_shape_rc
    valid = np.zeros((rows, columns), dtype=np.bool_)
    column = np.arange(columns, dtype=np.float64)
    for start in range(0, rows, row_chunk_size):
        stop = min(start + row_chunk_size, rows)
        row = np.arange(start, stop, dtype=np.float64)
        column_grid, row_grid = np.broadcast_arrays(column[None, :], row[:, None])
        geometry = detector.evaluate_detector_geometry(
            column_grid,
            row_grid,
            include_surface_jacobian=False,
        )
        valid[start:stop] = geometry.valid
    return valid


def _write_figure(
    *,
    image_A2: np.ndarray,
    valid_center: np.ndarray,
    direct_beam_column_row: tuple[float, float],
    specular_column_row: tuple[float, float],
    mosaic_description: str,
    unresolved_pixel_count: int,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    positive = image_A2[image_A2 > 0.0]
    if not positive.size:
        raise RuntimeError("detector image contains no positive m=1 mass")
    linear_max = float(np.quantile(positive, 0.9995))
    log_min = float(np.quantile(positive, 0.005))
    log_max = float(np.max(positive))
    del positive
    log_cmap = matplotlib.colormaps["magma"].copy()
    log_cmap.set_bad(log_cmap(0.0))
    rows, columns = image_A2.shape
    extent = (-0.5, columns - 0.5, rows - 0.5, -0.5)
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), constrained_layout=True)
    linear = axes[0].imshow(
        image_A2,
        origin="upper",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=linear_max,
        interpolation="nearest",
        rasterized=True,
    )
    logarithmic = axes[1].imshow(
        np.ma.array(image_A2, mask=image_A2 <= 0.0, copy=False),
        origin="upper",
        extent=extent,
        cmap=log_cmap,
        norm=LogNorm(vmin=log_min, vmax=log_max),
        interpolation="nearest",
        rasterized=True,
    )
    invalid_overlay = np.ma.masked_where(
        valid_center,
        np.ones(image_A2.shape, dtype=np.uint8),
    )
    for axis in axes:
        axis.imshow(
            invalid_overlay,
            origin="upper",
            extent=extent,
            cmap="gray",
            vmin=0.0,
            vmax=2.0,
            alpha=0.32,
            interpolation="nearest",
            rasterized=True,
        )
        axis.scatter(
            *direct_beam_column_row,
            marker="+",
            s=110,
            linewidths=1.5,
            color="#00e5ff",
            label="direct-beam coordinate",
        )
        axis.scatter(
            *specular_column_row,
            marker="x",
            s=75,
            linewidths=1.4,
            color="#7cff6b",
            label="m=0 geometry (intensity excluded)",
        )
        axis.set_xlabel("detector column (pixel coordinate)")
        axis.set_ylabel("detector row (pixel coordinate)")
        axis.set_xlim(-0.5, columns - 0.5)
        axis.set_ylim(rows - 0.5, -0.5)
        axis.legend(loc="lower right", fontsize=8, framealpha=0.85)
    axes[0].set_title("Linear m=1 upper-root mass (99.95% display clip)")
    axes[1].set_title("Logarithmic m=1 upper-root mass")
    figure.colorbar(linear, ax=axes[0], label=r"raw detector mass ($\AA^2$/pixel)")
    figure.colorbar(logarithmic, ax=axes[1], label=r"raw detector mass ($\AA^2$/pixel)")
    status_prefix = (
        ""
        if unresolved_pixel_count == 0
        else f"UNRESOLVED QUADRATURE DIAGNOSTIC ({unresolved_pixel_count:,} pixels) — "
    )
    figure.suptitle(
        status_prefix
        + "Bi$_2$Se$_3$, one 5° incident state\n"
        + f"{mosaic_description}\n"
        + "detector point → $k_f$ → Q → mosaic * 2H SF → integrated native pixels"
    )
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gaussian-sigma-deg",
        type=float,
        help="override the YAML Gaussian mosaic standard deviation in degrees",
    )
    parser.add_argument(
        "--lorentzian-hwhm-deg",
        type=float,
        help="override the YAML Lorentzian mosaic half width at half maximum in degrees",
    )
    parser.add_argument(
        "--eta",
        "--lorentzian-probability",
        dest="eta",
        type=float,
        help="override the YAML Lorentzian mixture probability; 0 is Gaussian and 1 Lorentzian",
    )
    parser.add_argument(
        "--layers",
        type=int,
        help="override the YAML finite Bi2Se3 quintuple-layer count",
    )
    parser.add_argument(
        "--stacking-epsilon",
        type=float,
        help="override the YAML shared rich-parent 2H disorder probability",
    )
    parser.add_argument("--pixel-gauss-order", type=int, default=2)
    parser.add_argument("--fold-gauss-order", type=int, default=4)
    parser.add_argument("--fold-subdivision-count", type=int, default=8)
    parser.add_argument("--row-chunk-size", type=int, default=8)
    parser.add_argument(
        "--integration-method",
        choices=tuple(method.value for method in PixelIntegrationMethod),
        default=PixelIntegrationMethod.FIXED_NUMPY.value,
    )
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--absolute-tolerance-a2", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--reference-diagnostic", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument("--allow-unresolved-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    if args.numeric_only and args.output_dir is not None:
        parser.error("--numeric-only and --output-dir are mutually exclusive")
    if not args.numeric_only and args.output_dir is None:
        parser.error("provide --output-dir or --numeric-only")
    if args.layers is not None and args.layers < 1:
        parser.error("layers must be positive")
    if args.stacking_epsilon is not None and not math.isfinite(args.stacking_epsilon):
        parser.error("stacking epsilon must be finite")
    if args.stacking_epsilon is not None and not 0.0 <= args.stacking_epsilon <= 1.0:
        parser.error("stacking epsilon must lie in [0, 1]")

    fixture_start = perf_counter()
    try:
        detector, m1_rods, m0_rod, configured_physics_revision = build_default_detector_measure(
            gaussian_sigma_deg=args.gaussian_sigma_deg,
            lorentzian_hwhm_deg=args.lorentzian_hwhm_deg,
            eta=args.eta,
            layers=args.layers,
            shared_disorder_epsilon=args.stacking_epsilon,
        )
    except _MosaicInputError as error:
        parser.error(str(error))
    fixture_elapsed = perf_counter() - fixture_start
    strength_model = detector.coating.bragg_space.strength_model
    if not isinstance(strength_model, Bi2X3FiniteStackStrength):
        raise TypeError("the default fixture must use Bi2X3FiniteStackStrength")
    reference_path = (
        args.reference_diagnostic.resolve() if args.reference_diagnostic is not None else None
    )
    reference_identity = _reference_comparison_identity(
        detector,
        strength_model,
        m1_rods,
        branch=2,
        configured_physics_revision=configured_physics_revision,
    )
    if reference_path is not None:
        _validate_reference_strength(reference_path, reference_identity)
    start = perf_counter()
    integration_method = PixelIntegrationMethod(args.integration_method)
    result = detector.integrate_native_pixels(
        rods=m1_rods,
        branch=2,
        quadrature=DetectorQuadrature(
            pixel_gauss_order=args.pixel_gauss_order,
            fold_gauss_order=args.fold_gauss_order,
            fold_subdivision_count=args.fold_subdivision_count,
            row_chunk_size=args.row_chunk_size,
            method=integration_method,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance_A2=args.absolute_tolerance_a2,
            max_depth=args.max_depth,
            worker_count=args.worker_count,
        ),
    )
    elapsed = perf_counter() - start
    if (
        args.output_dir is not None
        and integration_method is PixelIntegrationMethod.ADAPTIVE_COMPILED
        and not result.adaptive_tolerance_satisfied
        and not args.allow_unresolved_diagnostic
    ):
        raise RuntimeError(
            "adaptive quadrature left unresolved pixels; increase convergence or pass "
            "--allow-unresolved-diagnostic to emit an explicitly labelled non-accepted image"
        )
    specular = detector.map_specular_geometry(
        rod=m0_rod,
        alpha_rad=0.0,
        beta_rad=0.0,
    )
    specular_coordinate = (
        float(specular.geometry.column_px),
        float(specular.geometry.row_px),
    )
    direct_beam = tuple(
        float(value) for value in detector.instrument.detector_reference_coordinate_px
    )
    bragg_config = detector.coating.bragg_space.config
    mosaic = bragg_config.mosaic
    hash_start = perf_counter()
    image_hash = hashlib.sha256(
        memoryview(np.ascontiguousarray(result.image_A2)).cast("B")
    ).hexdigest()
    hash_elapsed = perf_counter() - hash_start
    summary = {
        **reference_identity,
        "detector_column_pitch_m": detector.instrument.detector_column_pitch_m,
        "detector_distance_m": float(
            np.linalg.norm(
                detector.instrument.lab_from_detector.translation_m
                - detector.instrument.lab_from_sample.translation_m
            )
        ),
        "detector_reference_column_row_px": list(direct_beam),
        "detector_row_pitch_m": detector.instrument.detector_row_pitch_m,
        "detector_shape_rc": list(result.image_A2.shape),
        "direct_beam_column_row_px": list(direct_beam),
        "image_sha256": image_hash,
        "ki_sample_Ainv": detector.coating.ki_sample_Ainv.tolist(),
        "m0_intensity_status": specular.intensity_status.value,
        "m0_specular_column_row_px": list(specular_coordinate),
        "mosaic_gaussian_sigma_deg": math.degrees(mosaic.gaussian_sigma_rad),
        "mosaic_lorentzian_hwhm_deg": math.degrees(mosaic.lorentzian_half_width_rad),
        "mosaic_lorentzian_probability": mosaic.lorentzian_probability,
        "nonzero_pixel_count": int(np.count_nonzero(result.image_A2)),
        "fold_gauss_order": args.fold_gauss_order,
        "fold_subdivision_count": args.fold_subdivision_count,
        "fold_refined_pixel_count": result.fold_refined_pixel_count,
        "fold_refinement_centroid_shift_px": result.fold_refinement_centroid_shift_px,
        "fold_refinement_l1_A2": result.fold_refinement_l1_A2,
        "fold_refinement_relative_l1": (
            result.fold_refinement_l1_A2 / result.total_detector_mass_A2
        ),
        "adaptive_refined_pixel_count": result.adaptive_refined_pixel_count,
        "adaptive_tolerance_satisfied": result.adaptive_tolerance_satisfied,
        "adaptive_unresolved_pixel_count": result.adaptive_unresolved_pixel_count,
        "artifact_status": (
            "FIXED_PREVIEW"
            if integration_method is PixelIntegrationMethod.FIXED_NUMPY
            else (
                "ADAPTIVE_CONVERGED"
                if result.adaptive_tolerance_satisfied
                else "ADAPTIVE_UNRESOLVED_DIAGNOSTIC"
            )
        ),
        "sampled_invalid_pixel_count": result.sampled_invalid_pixel_count,
        "coordinate_evaluation_count": result.coordinate_evaluation_count,
        "estimated_l1_error_A2": result.estimated_l1_error_A2,
        "execution_backend": result.execution_backend,
        "fixture_build_wall_time_s": fixture_elapsed,
        "image_hash_wall_time_s": hash_elapsed,
        "integration_method": args.integration_method,
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance_A2": args.absolute_tolerance_a2,
        "max_depth": args.max_depth,
        "worker_count": args.worker_count,
        "integration_peak_working_set_bytes": _peak_working_set_bytes(),
        "per_rod_detector_mass_A2": result.per_rod_detector_mass_A2.tolist(),
        "pixel_gauss_order": args.pixel_gauss_order,
        "total_detector_mass_A2": result.total_detector_mass_A2,
        "integration_wall_time_s": elapsed,
    }
    if reference_path is not None:
        reference_start = perf_counter()
        (
            reference_image,
            reference_per_rod,
            reference_image_sha256,
            reference_diagnostic_sha256,
        ) = _load_reference_arrays(
            reference_path,
            image_shape=result.image_A2.shape,
            per_rod_shape=result.per_rod_detector_mass_A2.shape,
            expected_identity=reference_identity,
        )
        reference_total = float(np.sum(reference_image, dtype=np.float64))
        total_relative_error = (
            abs(result.total_detector_mass_A2 - reference_total) / reference_total
        )
        normalized_l1 = 0.0
        for row_start in range(0, result.image_A2.shape[0], 64):
            row_stop = min(row_start + 64, result.image_A2.shape[0])
            normalized_l1 += float(
                np.sum(
                    np.abs(
                        result.image_A2[row_start:row_stop] / result.total_detector_mass_A2
                        - reference_image[row_start:row_stop] / reference_total
                    ),
                    dtype=np.float64,
                )
            )
        row_coordinate = np.arange(result.image_A2.shape[0], dtype=np.float64)
        column_coordinate = np.arange(result.image_A2.shape[1], dtype=np.float64)
        result_centroid = np.asarray(
            (
                np.dot(np.sum(result.image_A2, axis=0), column_coordinate)
                / result.total_detector_mass_A2,
                np.dot(np.sum(result.image_A2, axis=1), row_coordinate)
                / result.total_detector_mass_A2,
            )
        )
        reference_centroid = np.asarray(
            (
                np.dot(np.sum(reference_image, axis=0), column_coordinate) / reference_total,
                np.dot(np.sum(reference_image, axis=1), row_coordinate) / reference_total,
            )
        )
        per_rod_relative_error = (
            np.abs(result.per_rod_detector_mass_A2 - reference_per_rod) / reference_per_rod
        )
        summary["frozen_reference"] = {
            "centroid_shift_px": float(np.linalg.norm(result_centroid - reference_centroid)),
            "maximum_per_rod_relative_error": float(np.max(per_rod_relative_error)),
            "normalized_image_l1": normalized_l1,
            "path": os.fspath(reference_path),
            "reference_diagnostic_sha256": reference_diagnostic_sha256,
            "reference_image_sha256": reference_image_sha256,
            "total_relative_error": total_relative_error,
        }
        del reference_image
        summary["reference_comparison_wall_time_s"] = perf_counter() - reference_start
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        if output_dir == ROOT or output_dir.is_relative_to(ROOT):
            raise ValueError("output directory must be outside the repository")
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / "bi2se3-5deg-m01-continuous-detector.png"
        diagnostic_path = output_dir / "bi2se3-5deg-m01-continuous-detector.ra_diag.npz"
        validity_start = perf_counter()
        valid_center = result.sampled_valid_pixel_center
        if valid_center is None:
            valid_center = _center_valid_mask(detector, row_chunk_size=args.row_chunk_size)
            summary["validity_mask_source"] = "postintegration_geometry_evaluation.v1"
        else:
            summary["validity_mask_source"] = "compiled_center_sample_reuse.v1"
        summary["validity_mask_wall_time_s"] = perf_counter() - validity_start
        figure_start = perf_counter()
        _write_figure(
            image_A2=result.image_A2,
            valid_center=valid_center,
            direct_beam_column_row=direct_beam,
            specular_column_row=specular_coordinate,
            mosaic_description=(
                f"Gaussian sigma={math.degrees(mosaic.gaussian_sigma_rad):g}°, "
                f"Lorentzian HWHM={math.degrees(mosaic.lorentzian_half_width_rad):g}°, "
                f"eta={mosaic.lorentzian_probability:g}\n"
                f"{strength_model.layers} QLs, shared 2H disorder "
                f"epsilon={strength_model.shared_disorder_epsilon:g}"
            ),
            unresolved_pixel_count=result.adaptive_unresolved_pixel_count,
            output_path=figure_path,
        )
        summary["figure_wall_time_s"] = perf_counter() - figure_start
        summary["diagnostic_path"] = os.fspath(diagnostic_path)
        summary["figure_path"] = os.fspath(figure_path)
        summary["diagnostic_storage"] = "atomic_compressed_npz_embedded_manifest.v1"
        write_diagnostic(
            diagnostic_path,
            arrays={
                "image_A2": result.image_A2,
                "valid_pixel_center": valid_center,
                "ki_sample_Ainv": detector.coating.ki_sample_Ainv,
                "per_rod_detector_mass_A2": result.per_rod_detector_mass_A2,
            },
            manifest=summary,
            repository_root=ROOT,
        )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
