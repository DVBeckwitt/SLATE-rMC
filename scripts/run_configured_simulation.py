"""Generate configured reciprocal-space, Ewald-patch, and detector views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np

from rasim_next.pipeline.configured_simulation import (
    DetectorMacrobinImage,
    EwaldSurfaceDisplay,
    ReciprocalSpaceDisplay,
    build_configured_simulation_inputs,
    build_nominal_ewald_context,
    build_source_averaged_detector,
    evaluate_nominal_ewald_surface,
    integrate_detector_macrobins,
    load_simulation_config,
    sample_reciprocal_space,
)
from rasim_next.proof.diagnostics import write_diagnostic

ROOT = Path(__file__).resolve().parents[1]


def _configure_matplotlib() -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as error:
        raise SystemExit(
            "This image runner requires Matplotlib; install it with "
            "`python -m pip install matplotlib`."
        ) from error
    matplotlib.use("Agg")


def _positive_log_bounds(
    values: np.ndarray,
    *,
    decades: float = 10.0,
) -> tuple[float, float] | None:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if not positive.size:
        return None
    maximum = float(np.max(positive))
    return max(float(np.min(positive)), maximum * 10.0 ** (-decades)), maximum


def _set_equal_3d_axes(axis: object, points: np.ndarray) -> None:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = 0.5 * (minimum + maximum)
    half_range = 0.5 * float(np.max(maximum - minimum))
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] - half_range, center[2] + half_range)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _render_reciprocal_space(data: ReciprocalSpaceDisplay, path: Path) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    bounds = _positive_log_bounds(data.intensity_density_A2_rad2_inv)
    visible = np.ones(data.intensity_density_A2_rad2_inv.shape, dtype=np.bool_)
    if bounds is not None:
        low, high = bounds
        visible &= data.intensity_density_A2_rad2_inv >= low
    points = data.q_sample_Ainv[visible]
    intensity = data.intensity_density_A2_rad2_inv[visible]
    figure = plt.figure(figsize=(9.2, 8.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    scatter_options: dict[str, object] = {
        "s": 1.0,
        "alpha": 0.32,
        "linewidths": 0.0,
        "rasterized": True,
    }
    if bounds is None:
        scatter_options["color"] = "#8a8a8a"
    else:
        scatter_options.update(c=intensity, cmap="magma", norm=LogNorm(vmin=low, vmax=high))
    shown = axis.scatter(points[:, 0], points[:, 1], points[:, 2], **scatter_options)
    axis.set_xlabel(r"$Q_x$ sample ($\AA^{-1}$)")
    axis.set_ylabel(r"$Q_y$ sample ($\AA^{-1}$)")
    axis.set_zlabel(r"$Q_z$ sample ($\AA^{-1}$)")
    axis.set_title(
        "Bi$_2$Se$_3$ continuous Bragg space — all elastic-reach rods\n"
        rf"nominal-source reference $\lambda={data.reference_wavelength_A:.6f}$ $\AA$; "
        r"color = mosaic $\times$ finite-2H SF"
    )
    _set_equal_3d_axes(axis, points)
    if bounds is None:
        axis.text2D(
            0.03,
            0.03,
            "configured intensity is identically zero",
            transform=axis.transAxes,
        )
    else:
        figure.colorbar(
            shown,
            ax=axis,
            pad=0.08,
            label=r"latent intensity coefficient ($\AA^2\,rad^{-2}$; log)",
        )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _render_ewald_surface(data: EwaldSurfaceDisplay, path: Path) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    bounds = _positive_log_bounds(data.coating_intensity_density_A2_rad2_inv)
    valid = np.all(np.isfinite(data.q_sample_Ainv), axis=-1)
    if bounds is not None:
        low, high = bounds
        valid &= data.coating_intensity_density_A2_rad2_inv >= low
    points = data.q_sample_Ainv[valid]
    intensity = data.coating_intensity_density_A2_rad2_inv[valid]
    figure = plt.figure(figsize=(9.2, 8.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    scatter_options: dict[str, object] = {
        "s": 5.0,
        "alpha": 0.82,
        "linewidths": 0.0,
        "rasterized": True,
    }
    if bounds is None:
        scatter_options["color"] = "#8a8a8a"
    else:
        scatter_options.update(c=intensity, cmap="magma", norm=LogNorm(vmin=low, vmax=high))
    shown = axis.scatter(points[:, 0], points[:, 1], points[:, 2], **scatter_options)
    axis.set_xlabel(r"$Q_x$ sample ($\AA^{-1}$)")
    axis.set_ylabel(r"$Q_y$ sample ($\AA^{-1}$)")
    axis.set_zlabel(r"$Q_z$ sample ($\AA^{-1}$)")
    axis.set_title(
        "Nominal detector-visible intrinsic Ewald coating — all retained roots\n"
        r"color = mosaic $\times$ per-rod finite-2H SF $\times$ one Ewald coarea factor"
    )
    _set_equal_3d_axes(axis, points)
    if bounds is None:
        axis.text2D(
            0.03,
            0.03,
            "configured intensity is identically zero",
            transform=axis.transAxes,
        )
    else:
        figure.colorbar(
            shown,
            ax=axis,
            pad=0.08,
            label=r"intrinsic latent coating density ($\AA^2\,rad^{-2}$; log)",
        )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _render_detector(
    data: DetectorMacrobinImage,
    *,
    detector_shape_rc: tuple[int, int],
    direct_beam_column_row_px: tuple[float, float],
    sample_count: int,
    bin_size_px: int,
    includes_m0: bool,
    path: Path,
) -> None:
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    bounds = _positive_log_bounds(data.image_A2, decades=8.0)
    positive = data.image_A2[data.image_A2 > 0.0]
    linear_high = float(np.quantile(positive, 0.995)) if positive.size else 1.0
    rows, columns = detector_shape_rc
    extent = (-0.5, columns - 0.5, rows - 0.5, -0.5)
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), constrained_layout=True)
    linear = axes[0].imshow(
        data.image_A2,
        origin="upper",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=linear_high,
        interpolation="nearest",
        rasterized=True,
    )
    log_cmap = matplotlib.colormaps["magma"].copy()
    log_cmap.set_bad(log_cmap(0.0))
    if bounds is None:
        logarithmic = axes[1].imshow(
            data.image_A2,
            origin="upper",
            extent=extent,
            cmap=log_cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            rasterized=True,
        )
    else:
        low, high = bounds
        logarithmic = axes[1].imshow(
            np.ma.masked_less(data.image_A2, low),
            origin="upper",
            extent=extent,
            cmap=log_cmap,
            norm=LogNorm(vmin=low, vmax=high),
            interpolation="nearest",
            rasterized=True,
        )
    invalid = np.ma.masked_where(
        data.valid_source_count_min > 0,
        np.ones(data.image_A2.shape, dtype=np.uint8),
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
            s=105,
            linewidths=1.5,
            color="#00e5ff",
            label="direct-beam coordinate",
        )
        axis.set_xlabel("detector column (native pixel coordinate)")
        axis.set_ylabel("detector row (native pixel coordinate)")
        axis.set_xlim(-0.5, columns - 0.5)
        axis.set_ylim(rows - 0.5, -0.5)
        axis.legend(loc="lower right", fontsize=8, framealpha=0.85)
    axes[0].set_title("Linear quadrature estimate (99.5% display clip)")
    axes[1].set_title(
        "Logarithmic quadrature estimate (8-decade display floor)"
        if bounds is not None
        else "Configured zero-intensity field"
    )
    label = (
        rf"raw detector quadrature estimate "
        rf"($\AA^2$/{bin_size_px}x{bin_size_px}-pixel macrobin)"
    )
    figure.colorbar(linear, ax=axes[0], label=label)
    figure.colorbar(logarithmic, ax=axes[1], label=label)
    m0_label = r" including $m=0$" if includes_m0 else ""
    figure.suptitle(
        rf"Bi$_2$Se$_3$: {sample_count:,} incoherent source states, all natural $m${m0_label}"
        "\nall retained Ewald roots; exit refraction and attenuation evaluated per ray; "
        "fixed-quadrature preview"
    )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def _external_artifact_path(directory: Path, filename: str) -> Path:
    path = directory / filename
    resolved = path.resolve()
    if resolved == ROOT or resolved.is_relative_to(ROOT):
        raise ValueError(f"artifact path resolves inside the repository: {resolved}")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the strict YAML-configured continuous Bi2Se3 simulation."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional external artifact directory override; physics is unchanged",
    )
    args = parser.parse_args(argv)
    overall_start = perf_counter()
    config = load_simulation_config(args.config, repository_root=ROOT)
    output_directory = (
        config.output_directory if args.output_dir is None else args.output_dir.resolve()
    )
    if output_directory == ROOT or output_directory.is_relative_to(ROOT):
        parser.error("output directory must be outside the repository")
    output_directory.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    build_start = perf_counter()
    inputs = build_configured_simulation_inputs(config)
    timings = {"shared_build_wall_time_s": perf_counter() - build_start}
    arrays: dict[str, np.ndarray] = {
        "source_wavelength_A": inputs.incident.states.wavelength_A,
    }
    artifact_paths: dict[str, str] = {}
    measures: dict[str, str] = {}
    detector_gap: float | None = None
    nominal_ewald_gap: float | None = None

    if config.outputs.reciprocal_space.enabled:
        start = perf_counter()
        reciprocal = sample_reciprocal_space(inputs)
        timings["reciprocal_wall_time_s"] = perf_counter() - start
        path = _external_artifact_path(
            output_directory,
            config.outputs.reciprocal_space.filename,
        )
        _render_reciprocal_space(reciprocal, path)
        artifact_paths["reciprocal_space"] = os.fspath(path)
        measures["reciprocal_space"] = reciprocal.measure_id
        arrays.update(
            {
                "reciprocal_family_m": reciprocal.family_m,
                "reciprocal_intensity_density_A2_rad2_inv": (
                    reciprocal.intensity_density_A2_rad2_inv
                ),
                "reciprocal_q_sample_Ainv": reciprocal.q_sample_Ainv,
            }
        )

    if config.outputs.ewald_surface.enabled:
        start = perf_counter()
        nominal = build_nominal_ewald_context(inputs)
        ewald = evaluate_nominal_ewald_surface(
            nominal,
            alpha_count=config.numerics.ewald_alpha_count,
            beta_count=config.numerics.ewald_beta_count,
            alpha_max_deg=config.numerics.ewald_alpha_max_deg,
        )
        timings["ewald_wall_time_s"] = perf_counter() - start
        nominal_ewald_gap = ewald.detector_visible_m0_q_gap_Ainv
        path = _external_artifact_path(output_directory, config.outputs.ewald_surface.filename)
        _render_ewald_surface(ewald, path)
        artifact_paths["ewald_surface"] = os.fspath(path)
        measures["ewald_surface"] = ewald.measure_id
        arrays.update(
            {
                "ewald_branch": ewald.branch,
                "ewald_coating_intensity_density_A2_rad2_inv": (
                    ewald.coating_intensity_density_A2_rad2_inv
                ),
                "ewald_family_m": ewald.family_m,
                "ewald_q_sample_Ainv": ewald.q_sample_Ainv,
                "ewald_residual_Ainv": ewald.ewald_residual_Ainv,
            }
        )

    detector_image: DetectorMacrobinImage | None = None
    if config.outputs.detector.enabled:
        start = perf_counter()
        detector = build_source_averaged_detector(inputs)
        detector_image = integrate_detector_macrobins(
            detector,
            bin_size_px=config.numerics.detector_macrobin_size_px,
            gauss_order=config.numerics.detector_gauss_order,
        )
        timings["detector_wall_time_s"] = perf_counter() - start
        detector_gap = detector.detector_visible_m0_q_gap_Ainv
        path = _external_artifact_path(output_directory, config.outputs.detector.filename)
        _render_detector(
            detector_image,
            detector_shape_rc=inputs.instrument.detector_shape_rc,
            direct_beam_column_row_px=inputs.instrument.detector_reference_coordinate_px,
            sample_count=inputs.samples.incident_sample_id.size,
            bin_size_px=config.numerics.detector_macrobin_size_px,
            includes_m0=any(rod.family_m == 0 for rod in inputs.rods),
            path=path,
        )
        artifact_paths["detector"] = os.fspath(path)
        measures["detector"] = detector_image.measure_id
        arrays.update(
            {
                "detector_column_center_px": detector_image.column_center_px,
                "detector_image_A2": detector_image.image_A2,
                "detector_per_rod_image_A2": detector_image.per_rod_image_A2,
                "detector_row_center_px": detector_image.row_center_px,
                "detector_valid_source_count_min": detector_image.valid_source_count_min,
            }
        )

    family_values = sorted({rod.family_m for rod in inputs.rods})
    manifest: dict[str, object] = {
        "schema_version": "rasim-configured-result-v1",
        "config_path": os.fspath(config.config_path),
        "physics_revision": config.physics_revision,
        "render_revision": config.render_revision,
        "source_revision": inputs.incident.states.source_revision,
        "enabled_artifacts": list(config.enabled_artifact_names),
        "artifact_paths": artifact_paths,
        "measures": measures,
        "sample_count": int(inputs.samples.incident_sample_id.size),
        "valid_source_state_count": int(np.count_nonzero(inputs.incident.states.valid)),
        "rod_count": len(inputs.rods),
        "rod_hk": [[rod.h, rod.k] for rod in inputs.rods],
        "family_m_values": family_values,
        "family_m_multiplicity": {
            str(family): sum(rod.family_m == family for rod in inputs.rods)
            for family in family_values
        },
        "root_policy": config.bragg.root_policy,
        "detector_visible_m0_q_gap_Ainv": (
            detector_gap if detector_gap is not None else nominal_ewald_gap
        ),
        "ensemble_detector_m0_q_gap_Ainv": detector_gap,
        "nominal_ewald_m0_q_gap_Ainv": nominal_ewald_gap,
        "m0_model": (
            "detector_visible_kinematic_00L.v1"
            if any(rod.family_m == 0 for rod in inputs.rods)
            else "NOT_INCLUDED"
        ),
        "full_shell_m0_status": (
            "EXCLUDED_DIRECT_ROOT_DIVERGENCE"
            if any(rod.family_m == 0 for rod in inputs.rods)
            else "NOT_INCLUDED"
        ),
        "specular_reflectivity_status": "SEPARATE_NOT_SUMMED",
        "detector_macrobin_size_px": config.numerics.detector_macrobin_size_px,
        "detector_gauss_order": config.numerics.detector_gauss_order,
        "detector_integration_quality": "nonquantitative_fixed_quadrature_preview.v1",
        "reciprocal_reference_wavelength_A": (2.0 * np.pi / inputs.bragg_space.config.k_norm_Ainv),
        "timings": timings,
    }
    if detector_image is not None:
        manifest.update(
            {
                "detector_coordinate_evaluation_count": (
                    detector_image.coordinate_evaluation_count
                ),
                "detector_image_sha256": _array_sha256(detector_image.image_A2),
                "total_detector_mass_A2": float(np.sum(detector_image.image_A2, dtype=np.float64)),
                "per_family_detector_mass_A2": {
                    str(family): float(
                        np.sum(
                            detector_image.per_rod_image_A2[
                                ...,
                                [rod.family_m == family for rod in inputs.rods],
                            ],
                            dtype=np.float64,
                        )
                    )
                    for family in family_values
                },
            }
        )
    manifest["simulation_and_render_wall_time_s"] = perf_counter() - overall_start
    diagnostic_path = _external_artifact_path(
        output_directory,
        "configured-simulation.ra_diag.npz",
    )
    manifest["diagnostic_path"] = os.fspath(diagnostic_path)
    write_diagnostic(
        diagnostic_path,
        arrays=arrays,
        manifest=manifest,
        repository_root=ROOT,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
