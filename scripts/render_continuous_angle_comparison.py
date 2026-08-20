"""Render matched continuous detector and ``(phi, 2theta)`` intensity fields."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from rasim_next.fitting import ContinuousDetectorGeometryModel, GeometryCorrections
from rasim_next.geometry import AngleFrame, detector_coordinates_to_angles
from rasim_next.measurement import ContinuousNormalizedAngleFunction
from rasim_next.pipeline.configured_simulation import (
    ConfiguredSimulationInputs,
    DetectorIntegerLMarkers,
    build_configured_simulation_inputs,
    build_nominal_ewald_context,
    evaluate_nominal_integer_l_markers,
    load_simulation_config,
)
from rasim_next.selection import build_osc_angle_frame

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "bi2se3_simulation.yaml"


def _external_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or resolved.is_relative_to(ROOT):
        raise ValueError(f"output path resolves inside the repository: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _nominal_angle_frame(inputs: ConfiguredSimulationInputs) -> AngleFrame:
    nominal = build_nominal_ewald_context(inputs)
    return build_osc_angle_frame(
        mean_direction_lab=inputs.config.source.mean_direction_lab,
        instrument=inputs.instrument,
        sample_intersection_lab_m=nominal.incident.states.sample_intersection_lab_m[0],
        revision="configured-nominal-source-angle-frame.v1",
    )


def _panel_two_theta_max_rad(
    inputs: ConfiguredSimulationInputs,
    angle_frame: AngleFrame,
) -> float:
    rows, columns = inputs.instrument.detector_shape_rc
    corner_column = np.asarray((-0.5, columns - 0.5, columns - 0.5, -0.5))
    corner_row = np.asarray((-0.5, -0.5, rows - 0.5, rows - 0.5))
    angles = detector_coordinates_to_angles(
        corner_column,
        corner_row,
        instrument=inputs.instrument,
        angle_frame=angle_frame,
    )
    if not np.all(angles.valid):
        raise RuntimeError("the active detector corners are not visible from the angle origin")
    return float(np.nextafter(np.max(angles.two_theta_rad), np.inf))


def _evaluate_detector_field(
    detector_function: object,
    *,
    detector_shape_rc: tuple[int, int],
    image_size: int,
    tile_row_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    rows, columns = detector_shape_rc
    column_edges = np.linspace(-0.5, columns - 0.5, image_size + 1)
    row_edges = np.linspace(-0.5, rows - 0.5, image_size + 1)
    column_px = 0.5 * (column_edges[:-1] + column_edges[1:])
    row_px = 0.5 * (row_edges[:-1] + row_edges[1:])
    image = np.zeros((image_size, image_size), dtype=np.float64)
    valid = np.zeros((image_size, image_size), dtype=np.bool_)
    for start in range(0, image_size, tile_row_count):
        stop = min(start + tile_row_count, image_size)
        column_grid, row_grid = np.broadcast_arrays(
            column_px[None, :],
            row_px[start:stop, None],
        )
        evaluated = detector_function.evaluate_detector_coordinates(column_grid, row_grid)
        image[start:stop] = evaluated.density_A2_per_px2
        valid[start:stop] = evaluated.valid_source_count > 0
    return image, valid, column_edges, row_edges


def _evaluate_angle_field(
    angle_function: ContinuousNormalizedAngleFunction,
    *,
    two_theta_max_rad: float,
    image_size: int,
    tile_row_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    two_theta_edges = np.linspace(0.0, two_theta_max_rad, image_size + 1)
    phi_edges = np.linspace(-np.pi, np.pi, image_size + 1)
    two_theta = 0.5 * (two_theta_edges[:-1] + two_theta_edges[1:])
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    image = np.zeros((image_size, image_size), dtype=np.float64)
    valid = np.zeros((image_size, image_size), dtype=np.bool_)
    for start in range(0, image_size, tile_row_count):
        stop = min(start + tile_row_count, image_size)
        two_theta_grid, phi_grid = np.broadcast_arrays(
            two_theta[None, :],
            phi[start:stop, None],
        )
        evaluated = angle_function(two_theta_grid, phi_grid)
        image[start:stop] = evaluated.intensity_A2_per_px2
        valid[start:stop] = evaluated.valid
    return image, valid, two_theta_edges, phi_edges


def _positive_log_bounds(*fields: NDArray[np.float64]) -> tuple[float, float]:
    positive = np.concatenate(tuple(field[np.isfinite(field) & (field > 0.0)] for field in fields))
    if not positive.size:
        raise RuntimeError("the continuous detector and angle fields contain no positive intensity")
    high = float(np.max(positive))
    return max(float(np.quantile(positive, 0.001)), high * 1.0e-10), high


def _plot_markers(
    axes: tuple[object, object],
    markers: DetectorIntegerLMarkers,
    marker_two_theta_deg: NDArray[np.float64],
    marker_phi_deg: NDArray[np.float64],
    *,
    detector_middle_column_px: float,
) -> None:
    import matplotlib

    groups = sorted(
        {
            (int(family), int(branch))
            for family, branch in zip(markers.family_m, markers.branch, strict=True)
        }
    )
    colors = matplotlib.colormaps["tab10"](np.linspace(0.0, 0.8, max(len(groups), 1)))
    branch_shapes = {0: "s", 1: "v", 2: "o"}
    group_index = {key: index for index, key in enumerate(groups)}
    coordinates = (
        (markers.column_px, markers.row_px),
        (marker_two_theta_deg, marker_phi_deg),
    )
    for axis, (x_value, y_value) in zip(axes, coordinates, strict=True):
        for family, branch in groups:
            selected = (markers.family_m == family) & (markers.branch == branch)
            color = colors[group_index[(family, branch)]]
            axis.scatter(
                x_value[selected],
                y_value[selected],
                marker=branch_shapes[branch],
                s=24,
                linewidths=0.9,
                facecolors="none",
                edgecolors=color,
                label=rf"$m={family}$, branch {branch}; label = $L$",
            )
    for index in range(markers.column_px.size):
        family = int(markers.family_m[index])
        branch = int(markers.branch[index])
        color = colors[group_index[(family, branch)]]
        left_detector = float(markers.column_px[index]) < detector_middle_column_px
        left_angle = float(marker_phi_deg[index]) < 0.0
        for axis, x_value, y_value, left in (
            (
                axes[0],
                float(markers.column_px[index]),
                float(markers.row_px[index]),
                left_detector,
            ),
            (
                axes[1],
                float(marker_two_theta_deg[index]),
                float(marker_phi_deg[index]),
                left_angle,
            ),
        ):
            axis.annotate(
                str(int(markers.integer_L[index])),
                (x_value, y_value),
                xytext=(-3.5 if left else 3.5, 0.0),
                textcoords="offset points",
                color=color,
                fontsize=5.5,
                ha="right" if left else "left",
                va="center",
                clip_on=True,
            )


def _render(
    *,
    detector_image: NDArray[np.float64],
    detector_valid: NDArray[np.bool_],
    detector_column_edges: NDArray[np.float64],
    detector_row_edges: NDArray[np.float64],
    angle_image: NDArray[np.float64],
    angle_valid: NDArray[np.bool_],
    two_theta_edges_rad: NDArray[np.float64],
    phi_edges_rad: NDArray[np.float64],
    markers: DetectorIntegerLMarkers,
    marker_two_theta_rad: NDArray[np.float64],
    marker_phi_rad: NDArray[np.float64],
    source_sample_count: int,
    minimum_physical_sample_count: int,
    rod_count: int,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    low, high = _positive_log_bounds(detector_image, angle_image)
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad(cmap(0.0))
    detector_display = np.where(np.isposinf(detector_image), high, detector_image)
    angle_display = np.where(np.isposinf(angle_image), high, angle_image)
    detector_masked = np.ma.array(
        detector_display,
        mask=~detector_valid | (detector_display < low),
    )
    angle_masked = np.ma.array(
        angle_display,
        mask=~angle_valid | (angle_display < low),
    )
    figure, axes_array = plt.subplots(1, 2, figsize=(18.0, 8.2), constrained_layout=True)
    axes = (axes_array[0], axes_array[1])
    detector_extent = (
        detector_column_edges[0],
        detector_column_edges[-1],
        detector_row_edges[-1],
        detector_row_edges[0],
    )
    angle_extent = tuple(
        np.degrees(
            (
                two_theta_edges_rad[0],
                two_theta_edges_rad[-1],
                phi_edges_rad[0],
                phi_edges_rad[-1],
            )
        )
    )
    shown = axes[0].imshow(
        detector_masked,
        origin="upper",
        extent=detector_extent,
        cmap=cmap,
        norm=LogNorm(vmin=low, vmax=high),
        interpolation="nearest",
        rasterized=True,
        aspect="equal",
    )
    axes[1].imshow(
        angle_masked,
        origin="lower",
        extent=angle_extent,
        cmap=cmap,
        norm=LogNorm(vmin=low, vmax=high),
        interpolation="nearest",
        rasterized=True,
        aspect="auto",
    )
    _plot_markers(
        axes,
        markers,
        np.degrees(marker_two_theta_rad),
        np.degrees(marker_phi_rad),
        detector_middle_column_px=0.5 * (detector_column_edges[0] + detector_column_edges[-1]),
    )
    axes[0].set_title("Continuous detector density")
    axes[0].set_xlabel("detector column (native pixel coordinate)")
    axes[0].set_ylabel("detector row (native pixel coordinate)")
    axes[1].set_title(r"Continuous normalized angle intensity $I=S/N$")
    axes[1].set_xlabel(r"$2\theta$ (degree)")
    axes[1].set_ylabel(r"$\phi$ (degree; increasing upward)")
    for axis in axes:
        axis.legend(loc="lower right", fontsize=7.2, framealpha=0.86)
    figure.colorbar(
        shown,
        ax=axes,
        shrink=0.92,
        label=r"raw detector-area-normalized intensity ($\AA^2$/pixel$^2$; shared log scale)",
    )
    source_label = (
        f"{source_sample_count} weighted spectral lines at mean ray geometry"
        if source_sample_count == minimum_physical_sample_count
        else f"{source_sample_count} configured physical source rows"
    )
    figure.suptitle(
        rf"Bi$_2$Se$_3$: {source_label}, {rod_count} physical rods, "
        "all retained roots\n"
        rf"each field is {detector_image.shape[0]}x{detector_image.shape[1]} center-sampled; "
        r"labels are geometry-visible exact integer-$L$, $\alpha=0$ landmarks (not raster maxima)"
    )
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render matched detector and continuous phi/2theta intensity fields."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=3000)
    parser.add_argument("--source-sample-count", type=int)
    parser.add_argument("--tile-row-count", type=int, default=16)
    args = parser.parse_args()
    if args.image_size < 2:
        parser.error("--image-size must be at least 2")
    if args.source_sample_count is not None and args.source_sample_count < 1:
        parser.error("--source-sample-count must be positive")
    if args.tile_row_count < 1:
        parser.error("--tile-row-count must be positive")
    output_path = _external_output_path(args.output)

    config = load_simulation_config(args.config.resolve())
    if (
        args.source_sample_count is not None
        and args.source_sample_count < config.source.minimum_physical_sample_count
    ):
        parser.error("--source-sample-count must realize every configured physical source line")
    source_sample_count = (
        config.source.minimum_physical_sample_count
        if args.source_sample_count is None
        else args.source_sample_count
    )
    config = replace(
        config,
        source=replace(config.source, sample_count=source_sample_count),
    )
    build_start = perf_counter()
    inputs = build_configured_simulation_inputs(config)
    detector_function = ContinuousDetectorGeometryModel(inputs).bind(GeometryCorrections.zero())
    angle_frame = _nominal_angle_frame(inputs)
    angle_function = ContinuousNormalizedAngleFunction(detector_function, angle_frame)
    markers = evaluate_nominal_integer_l_markers(build_nominal_ewald_context(inputs))
    marker_angles = detector_coordinates_to_angles(
        markers.column_px,
        markers.row_px,
        instrument=detector_function.instrument,
        angle_frame=angle_frame,
    )
    if not np.all(marker_angles.valid & marker_angles.azimuth_valid):
        raise RuntimeError("a geometry-visible detector marker has no canonical angle coordinate")
    two_theta_max_rad = _panel_two_theta_max_rad(inputs, angle_frame)
    build_seconds = perf_counter() - build_start

    detector_start = perf_counter()
    detector_image, detector_valid, detector_column_edges, detector_row_edges = (
        _evaluate_detector_field(
            detector_function,
            detector_shape_rc=inputs.instrument.detector_shape_rc,
            image_size=args.image_size,
            tile_row_count=args.tile_row_count,
        )
    )
    detector_seconds = perf_counter() - detector_start
    angle_start = perf_counter()
    angle_image, angle_valid, two_theta_edges, phi_edges = _evaluate_angle_field(
        angle_function,
        two_theta_max_rad=two_theta_max_rad,
        image_size=args.image_size,
        tile_row_count=args.tile_row_count,
    )
    angle_seconds = perf_counter() - angle_start
    _render(
        detector_image=detector_image,
        detector_valid=detector_valid,
        detector_column_edges=detector_column_edges,
        detector_row_edges=detector_row_edges,
        angle_image=angle_image,
        angle_valid=angle_valid,
        two_theta_edges_rad=two_theta_edges,
        phi_edges_rad=phi_edges,
        markers=markers,
        marker_two_theta_rad=marker_angles.two_theta_rad,
        marker_phi_rad=marker_angles.phi_rad,
        source_sample_count=source_sample_count,
        minimum_physical_sample_count=config.source.minimum_physical_sample_count,
        rod_count=len(inputs.rods),
        output_path=output_path,
    )
    print(
        f"wrote {output_path} with {markers.column_px.size} geometry-visible labels; "
        f"build={build_seconds:.3f}s detector={detector_seconds:.3f}s "
        f"angle={angle_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
