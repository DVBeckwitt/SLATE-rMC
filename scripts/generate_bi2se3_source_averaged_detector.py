"""Render one detector integration of the canonical Monte Carlo source average."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
from generate_bi2se3_continuous_detector import (
    ROOT,
    _peak_working_set_bytes,
    build_default_source_averaged_detector_measure,
)

from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure
from rasim_next.proof.diagnostics import write_diagnostic

DEFAULT_SAMPLE_COUNT = 1_000
DEFAULT_DISPLAY_BIN_SIZE_PX = 8
DEFAULT_GAUSSIAN_SIGMA_DEG = 1.0
DEFAULT_LAYER_COUNT = 52
DEFAULT_SHARED_DISORDER_EPSILON = 0.001


def integrate_display_macrobins(
    detector: SourceAveragedDetectorEwaldMeasure,
    *,
    bin_size_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply one midpoint rule to detector-aligned square macrobins."""

    if isinstance(bin_size_px, bool) or not isinstance(bin_size_px, int) or bin_size_px < 1:
        raise ValueError("bin_size_px must be a positive integer")
    rows, columns = detector.instrument.detector_shape_rc
    if rows % bin_size_px or columns % bin_size_px:
        raise ValueError("bin_size_px must divide both detector dimensions exactly")
    column_px = np.arange(columns // bin_size_px, dtype=np.float64) * bin_size_px + 0.5 * (
        bin_size_px - 1
    )
    row_px = np.arange(rows // bin_size_px, dtype=np.float64) * bin_size_px + 0.5 * (
        bin_size_px - 1
    )
    column_grid, row_grid = np.meshgrid(column_px, row_px)
    start = perf_counter()
    evaluated = detector.evaluate_detector_coordinates(
        column_grid,
        row_grid,
        branch=2,
    )
    elapsed = perf_counter() - start
    if np.any(evaluated.caustic):
        raise FloatingPointError(
            "a display midpoint lies exactly on a source-state caustic; change bin size"
        )
    bin_area_px2 = float(bin_size_px**2)
    image_A2 = evaluated.density_A2_per_px2 * bin_area_px2
    per_rod_image_A2 = evaluated.per_rod_density_A2_per_px2 * bin_area_px2
    return (
        image_A2,
        per_rod_image_A2,
        evaluated.valid_source_count,
        column_px,
        elapsed,
    )


def _write_figure(
    *,
    image_A2: np.ndarray,
    valid_source_count: np.ndarray,
    detector_shape_rc: tuple[int, int],
    direct_beam_column_row_px: tuple[float, float],
    sample_count: int,
    bin_size_px: int,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    positive = image_A2[image_A2 > 0.0]
    if not positive.size:
        raise RuntimeError("source-averaged detector image contains no positive m=1 mass")
    linear_max = float(np.quantile(positive, 0.999))
    log_max = float(np.max(positive))
    log_min = log_max * 1.0e-8
    rows, columns = detector_shape_rc
    extent = (-0.5, columns - 0.5, rows - 0.5, -0.5)
    log_cmap = matplotlib.colormaps["magma"].copy()
    log_cmap.set_bad(log_cmap(0.0))
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
        np.ma.array(image_A2, mask=image_A2 < log_min, copy=False),
        origin="upper",
        extent=extent,
        cmap=log_cmap,
        norm=LogNorm(vmin=log_min, vmax=log_max),
        interpolation="nearest",
        rasterized=True,
    )
    invalid = np.ma.masked_where(
        valid_source_count > 0,
        np.ones(image_A2.shape, dtype=np.uint8),
    )
    for axis in axes:
        axis.imshow(
            invalid,
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
            *direct_beam_column_row_px,
            marker="+",
            s=110,
            linewidths=1.5,
            color="#00e5ff",
            label="nominal direct beam (m=0 intensity excluded)",
        )
        axis.set_xlabel("detector column (native pixel coordinate)")
        axis.set_ylabel("detector row (native pixel coordinate)")
        axis.set_xlim(-0.5, columns - 0.5)
        axis.set_ylim(rows - 0.5, -0.5)
        axis.legend(loc="lower right", fontsize=8, framealpha=0.85)
    axes[0].set_title("Linear m=1 upper-root mass (99.9% display clip)")
    axes[1].set_title("Logarithmic m=1 upper-root mass (8-decade floor)")
    measure_label = rf"raw detector mass ($\AA^2$/{bin_size_px}x{bin_size_px}-pixel macrobin)"
    figure.colorbar(linear, ax=axes[0], label=measure_label)
    figure.colorbar(logarithmic, ax=axes[1], label=measure_label)
    figure.suptitle(
        rf"Bi$_2$Se$_3$: {sample_count:,} Monte Carlo incident states, fixed $\theta_i=5^\circ$"
        r" | Gaussian $\sigma=1^\circ$, no Lorentzian | 52 QLs, shared 2H $\epsilon=0.001$"
        "\n"
        "state-specific detector ray → exit refraction/attenuation → Q → mosaic * SF; "
        "sum states, then one macrobin midpoint integration"
    )
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    overall_start = perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--display-bin-size-px", type=int, default=DEFAULT_DISPLAY_BIN_SIZE_PX)
    parser.add_argument(
        "--worker-count",
        type=int,
        default=min(os.cpu_count() or 1, 32),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.sample_count < 1:
        parser.error("sample count must be positive")

    output_dir = args.output_dir.resolve()
    if output_dir == ROOT or output_dir.is_relative_to(ROOT):
        raise ValueError("output directory must be outside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    build_start = perf_counter()
    detector = build_default_source_averaged_detector_measure(
        sample_count=args.sample_count,
        gaussian_sigma_deg=DEFAULT_GAUSSIAN_SIGMA_DEG,
        lorentzian_hwhm_deg=0.0,
        eta=0.0,
        layers=DEFAULT_LAYER_COUNT,
        shared_disorder_epsilon=DEFAULT_SHARED_DISORDER_EPSILON,
        worker_count=args.worker_count,
    )
    build_elapsed = perf_counter() - build_start
    image_A2, per_rod_image_A2, valid_source_count, column_px, integration_elapsed = (
        integrate_display_macrobins(
            detector,
            bin_size_px=args.display_bin_size_px,
        )
    )
    row_px = np.arange(image_A2.shape[0], dtype=np.float64) * args.display_bin_size_px + 0.5 * (
        args.display_bin_size_px - 1
    )
    direct_beam = tuple(
        float(value) for value in detector.instrument.detector_reference_coordinate_px
    )
    figure_path = output_dir / "bi2se3-1000ki-source-averaged-detector.png"
    diagnostic_path = output_dir / "bi2se3-1000ki-source-averaged-detector.ra_diag.npz"
    figure_start = perf_counter()
    _write_figure(
        image_A2=image_A2,
        valid_source_count=valid_source_count,
        detector_shape_rc=detector.instrument.detector_shape_rc,
        direct_beam_column_row_px=direct_beam,
        sample_count=args.sample_count,
        bin_size_px=args.display_bin_size_px,
        output_path=figure_path,
    )
    figure_elapsed = perf_counter() - figure_start
    summary = {
        "branch": 2,
        "cif_path": "examples/bi2se3/structures/Bi2Se3_vesta.cif",
        "outer_detector_coordinate_count": int(image_A2.size),
        "detector_shape_rc": list(detector.instrument.detector_shape_rc),
        "direct_beam_column_row_px": list(direct_beam),
        "display_bin_size_px": args.display_bin_size_px,
        "display_integration": "one_midpoint_per_detector_aligned_macrobin.v1",
        "ensemble_measure": "raw_detector_coordinate_density_A2_per_px2.v1",
        "figure_path": os.fspath(figure_path),
        "fixture_build_wall_time_s": build_elapsed,
        "gaussian_sigma_deg": DEFAULT_GAUSSIAN_SIGMA_DEG,
        "image_sha256": hashlib.sha256(
            memoryview(np.ascontiguousarray(image_A2)).cast("B")
        ).hexdigest(),
        "cold_integration_wall_time_s": integration_elapsed,
        "layer_count": DEFAULT_LAYER_COUNT,
        "lorentzian_hwhm_deg": 0.0,
        "lorentzian_probability": 0.0,
        "m0_intensity_status": "SPECULAR_INTENSITY_EXCLUDED",
        "m1_rod_keys": [[rod.h, rod.k] for rod in detector.rods],
        "nonzero_macrobin_count": int(np.count_nonzero(image_A2)),
        "per_rod_detector_mass_A2": np.sum(
            per_rod_image_A2,
            axis=(0, 1),
            dtype=np.float64,
        ).tolist(),
        "sample_angle_deg": 5.0,
        "sample_count": detector.source_state_count,
        "shared_disorder_epsilon": DEFAULT_SHARED_DISORDER_EPSILON,
        "source_revision": detector.incident.states.source_revision,
        "state_coordinate_evaluation_count": int(image_A2.size * detector.valid_source_state_count),
        "total_detector_mass_A2": float(np.sum(image_A2, dtype=np.float64)),
        "valid_source_state_count": detector.valid_source_state_count,
        "wavelength_max_A": float(np.max(detector.incident.states.wavelength_A)),
        "wavelength_mean_A": float(np.mean(detector.incident.states.wavelength_A)),
        "wavelength_min_A": float(np.min(detector.incident.states.wavelength_A)),
        "worker_count": args.worker_count,
    }
    summary["figure_wall_time_s"] = figure_elapsed
    summary["diagnostic_path"] = os.fspath(diagnostic_path)
    write_diagnostic(
        diagnostic_path,
        arrays={
            "column_px": column_px,
            "image_A2": image_A2,
            "per_rod_image_A2": per_rod_image_A2,
            "row_px": row_px,
            "valid_source_count": valid_source_count,
            "wavelength_A": detector.incident.states.wavelength_A,
        },
        manifest=summary,
        repository_root=ROOT,
    )
    summary["diagnostic_size_bytes"] = diagnostic_path.stat().st_size
    summary["end_to_end_peak_working_set_bytes"] = _peak_working_set_bytes()
    summary["end_to_end_wall_time_s"] = perf_counter() - overall_start
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
