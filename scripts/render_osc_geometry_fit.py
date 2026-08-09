"""Render observed and fitted detector-native landmarks for an OSC geometry fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from rasim_next.io.osc import read_osc
from rasim_next.selection import load_osc_geometry_series

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_by_image(records: object, name: str) -> dict[str, list[dict[str, object]]]:
    if not isinstance(records, list):
        raise ValueError(f"{name} must be a list")
    result: dict[str, list[dict[str, object]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{name} records must be mappings")
        image_id = record.get("image_id")
        sites = record.get("sites")
        if not isinstance(image_id, str) or image_id in result or not isinstance(sites, list):
            raise ValueError(f"{name} image records are malformed or duplicated")
        result[image_id] = sites
    return result


def _coordinates(sites: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray([site["observed_coordinate_px"] for site in sites], dtype=np.float64)
    predicted = np.asarray([site["predicted_coordinate_px"] for site in sites], dtype=np.float64)
    error = predicted - observed
    if (
        observed.shape != (len(sites), 2)
        or predicted.shape != observed.shape
        or np.any(~np.isfinite(observed))
        or np.any(~np.isfinite(predicted))
    ):
        raise ValueError("prediction coordinates must be finite detector column/row pairs")
    return observed, predicted, error


def render_geometry_fit(
    manifest_path: str | Path,
    position_path: str | Path,
    destination: str | Path,
) -> dict[str, object]:
    """Render OSC counts plus primary and held-out detector-coordinate residuals."""

    manifest = Path(manifest_path).resolve()
    position_file = Path(position_path).resolve()
    output = Path(destination).resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        raise ValueError("geometry-fit figure must be written outside the repository")
    if output.suffix.lower() != ".png":
        raise ValueError("geometry-fit figure destination must use a .png suffix")
    if output.exists():
        raise FileExistsError(output)

    position = json.loads(position_file.read_text(encoding="utf-8"))
    if position.get("schema") != "rasim-osc-geometry-fit-result-v6":
        raise ValueError("unsupported position artifact schema")
    if position.get("manifest_sha256") != _sha256(manifest):
        raise ValueError("position artifact does not match the supplied geometry manifest")

    series = load_osc_geometry_series(manifest)
    primary = _prediction_by_image(position.get("predictions"), "predictions")
    image_ids = {record.image_id for record in series.images}
    if set(primary) != image_ids:
        raise ValueError("primary prediction image IDs do not match the geometry manifest")

    cross_validation = position.get("cross_validation")
    heldout = (
        _prediction_by_image(cross_validation.get("heldout_predictions"), "heldout_predictions")
        if isinstance(cross_validation, dict)
        else {}
    )

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    ordered_images = sorted(
        series.images,
        key=lambda record: (
            record.axis_rotation_angles_deg[series.incidence_axis_index],
            record.image_id,
        ),
    )
    figure, axes = plt.subplots(
        2,
        len(ordered_images),
        figsize=(5.4 * len(ordered_images), 9.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (3.0, 1.2)},
    )
    axes = np.asarray(axes).reshape(2, len(ordered_images))
    summaries = []
    for column, record in enumerate(ordered_images):
        counts = read_osc(record.osc_path).detector_native_counts
        display = np.log1p(np.asarray(counts, dtype=np.float32))
        finite = display[np.isfinite(display)]
        vmax = float(np.quantile(finite, 0.9999)) if finite.size else 1.0
        image_axis = axes[0, column]
        image_axis.imshow(
            display,
            cmap="gray",
            origin="upper",
            vmin=0.0,
            vmax=max(vmax, 1.0),
            interpolation="nearest",
        )

        sites = primary[record.image_id]
        observed, predicted, error = _coordinates(sites)
        for observed_site, predicted_site in zip(observed, predicted, strict=True):
            image_axis.plot(
                (observed_site[0], predicted_site[0]),
                (observed_site[1], predicted_site[1]),
                color="#FFD60A",
                linewidth=0.8,
                zorder=3,
            )
        image_axis.scatter(
            observed[:, 0],
            observed[:, 1],
            s=54,
            facecolors="none",
            edgecolors="#00E5FF",
            linewidths=1.1,
            label="observed",
            zorder=4,
        )
        image_axis.scatter(
            predicted[:, 0],
            predicted[:, 1],
            s=42,
            color="#FF453A",
            marker="x",
            linewidths=1.2,
            label="primary fit",
            zorder=5,
        )
        image_axis.set_xlim(0.0, counts.shape[1] - 1.0)
        image_axis.set_ylim(counts.shape[0] - 1.0, 0.0)
        angle = record.axis_rotation_angles_deg[series.incidence_axis_index]
        rms = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
        maximum = float(np.max(np.linalg.norm(error, axis=1)))
        image_axis.set_title(f"{record.image_id}  ({angle:g}°)\n{len(sites)} fitted landmarks")
        image_axis.set_xlabel("detector column (px)")
        image_axis.set_ylabel("detector row (px)")
        if column == 0:
            image_axis.legend(loc="upper right", framealpha=0.9)

        residual_axis = axes[1, column]
        residual_axis.axhline(0.0, color="0.75", linewidth=0.8)
        residual_axis.axvline(0.0, color="0.75", linewidth=0.8)
        for radius in (1.0, 2.0):
            residual_axis.add_patch(
                plt.Circle((0.0, 0.0), radius, fill=False, color="0.82", linewidth=0.8)
            )
        residual_axis.scatter(
            error[:, 0],
            error[:, 1],
            s=28,
            color="#FF453A",
            label="primary fit",
            zorder=3,
        )
        heldout_sites = heldout.get(record.image_id, [])
        heldout_maximum = None
        if heldout_sites:
            _, _, heldout_error = _coordinates(heldout_sites)
            heldout_maximum = float(np.max(np.linalg.norm(heldout_error, axis=1)))
            residual_axis.scatter(
                heldout_error[:, 0],
                heldout_error[:, 1],
                s=48,
                facecolors="none",
                edgecolors="#BF5AF2",
                marker="D",
                linewidths=1.2,
                label="held-out fit",
                zorder=4,
            )
        limit = max(2.25, math.ceil(maximum + 0.25))
        if heldout_maximum is not None:
            limit = max(limit, math.ceil(heldout_maximum + 0.25))
        residual_axis.set_xlim(-limit, limit)
        residual_axis.set_ylim(-limit, limit)
        residual_axis.set_aspect("equal", adjustable="box")
        residual_axis.set_xlabel("predicted - observed column (px)")
        residual_axis.set_ylabel("predicted - observed row (px)")
        residual_axis.set_title(f"primary RMS {rms:.3f} px; max {maximum:.3f} px")
        if column == 0 and heldout:
            residual_axis.legend(loc="best", framealpha=0.9)
        summaries.append(
            {
                "image_id": record.image_id,
                "site_count": len(sites),
                "primary_rms_px": rms,
                "primary_max_px": maximum,
                "heldout_site_count": len(heldout_sites),
                "heldout_max_px": heldout_maximum,
            }
        )

    figure.suptitle(
        "Detector-native OSC geometry fit: observed vs predicted positions\n"
        "training scope: fitted detector landmarks",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.png")
    try:
        figure.savefig(
            temporary,
            dpi=180,
            metadata={"Description": f"position_sha256={_sha256(position_file)}"},
        )
        plt.close(figure)
        temporary.replace(output)
    except BaseException:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
        raise
    return {
        "schema": "rasim-osc-geometry-fit-render-v1",
        "position_path": str(position_file),
        "position_sha256": _sha256(position_file),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "figure_path": str(output),
        "figure_sha256": _sha256(output),
        "images": summaries,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("position", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = render_geometry_fit(arguments.manifest, arguments.position, arguments.destination)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if arguments.json:
            print(json.dumps({"error": {"type": type(error).__name__, "message": str(error)}}))
        else:
            print(f"geometry render rejected: {error}")
        return 1
    if arguments.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result["figure_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
