"""hBN calibrant ellipse fitting and bundle persistence."""

import math
import os
import json

import numpy as np
import cv2
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from ra_sim import hbn_geometry as _hbn_geometry
from ra_sim.hbn_fitter import fitter as _hbn_backend


# ------------------------------------------------------------
# Configuration parameters
# ------------------------------------------------------------

# Number of rings / ellipses
N_ELLIPSES = 5

# Points clicked per ellipse (>=5 recommended)
# An unconstrained ellipse has five free parameters (xc, yc, a, b, theta), so
# five non-collinear points are the minimum needed to uniquely determine it
# without imposing extra assumptions.
POINTS_PER_ELLIPSE = 5

# Intensity based refinement settings
REFINE_N_ANGLES = 360  # angular sampling along ellipse
REFINE_DR = 10.0  # half width of radial search window [pixels]
REFINE_STEP = 1.0  # radial step [pixels]


# ------------------------------------------------------------
# Profile save / load
# ------------------------------------------------------------
def save_click_profile(path, ell_points_ds, img_shape):
    profile = {
        "schema": "ra_sim.hbn_click_profile",
        "version": 1,
        "image_shape": list(img_shape),
        "points": [
            [[float(x), float(y)] for (x, y) in ellipse_pts] for ellipse_pts in ell_points_ds
        ],
    }
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"Saved click profile to:\n  {path}")


def load_click_profile(path, *, expected_image_shape):
    if os.path.getsize(path) > 1024 * 1024:
        raise ValueError("hBN click profile exceeds the 1 MiB size limit")
    with open(path) as f:
        profile = json.load(f)
    if not isinstance(profile, dict) or set(profile) != {
        "schema",
        "version",
        "image_shape",
        "points",
    }:
        raise ValueError("hBN click profile must use the documented version-1 schema")
    if profile["schema"] != "ra_sim.hbn_click_profile" or profile["version"] != 1:
        raise ValueError("unsupported hBN click-profile schema or version")
    image_shape = profile["image_shape"]
    if (
        not isinstance(image_shape, list)
        or len(image_shape) != 2
        or any(type(value) is not int or value <= 0 for value in image_shape)
    ):
        raise ValueError("hBN click profile image_shape must contain positive integer dimensions")
    if tuple(image_shape) != tuple(int(value) for value in expected_image_shape[:2]):
        raise ValueError("hBN click profile image_shape does not match the current image")
    points_raw = profile["points"]
    if not isinstance(points_raw, list) or len(points_raw) != N_ELLIPSES:
        raise ValueError(f"hBN click profile must contain exactly {N_ELLIPSES} rings")
    ell_points_ds = []
    for ell in points_raw:
        if not isinstance(ell, list) or len(ell) != POINTS_PER_ELLIPSE:
            raise ValueError(
                f"each hBN click-profile ring must contain {POINTS_PER_ELLIPSE} points"
            )
        ring = []
        for point in ell:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("each hBN click-profile point must contain x and y")
            x, y = (float(point[0]), float(point[1]))
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("hBN click-profile coordinates must be finite")
            if not (0.0 <= x < image_shape[1] and 0.0 <= y < image_shape[0]):
                raise ValueError("hBN click-profile coordinate lies outside the image")
            ring.append((x, y))
        ell_points_ds.append(ring)
    print(f"Loaded click profile from:\n  {path}")
    return ell_points_ds


# ------------------------------------------------------------
# Interactive clicking with zoom
# ------------------------------------------------------------
def get_points_per_ellipse_with_zoom(
    small, n_ellipses, pts_per_ellipse, beam_center=None, n_slices=5
):
    if pts_per_ellipse < 5:
        raise ValueError("Need at least 5 points per ellipse for a stable fit.")

    ell_points = [[] for _ in range(n_ellipses)]
    current_ellipse = 0
    current_point = 0
    all_points = []

    fig, ax = plt.subplots(figsize=(6, 6))

    h, w = small.shape
    ax.imshow(small, cmap="gray", origin="upper")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    title_text = ax.set_title(
        f"Ellipse 1/{n_ellipses}  point 0/{pts_per_ellipse}  "
        f"(Left: point, Right drag: zoom, 'r': reset)"
    )

    full_xlim = (0, w)
    full_ylim = (h, 0)

    zoom_rect = Rectangle(
        (0, 0),
        1,
        1,
        edgecolor="yellow",
        facecolor="none",
        linewidth=1.0,
        visible=False,
    )
    ax.add_patch(zoom_rect)
    zooming = {"active": False, "x0": None, "y0": None}

    seed_scatter = ax.scatter([], [], s=20, c="red", marker="x")

    center_marker = None
    guide_lines = []
    beam_center = (
        tuple(beam_center)
        if beam_center is not None and len(beam_center) == 2
        else (w / 2.0, h / 2.0)
    )

    def draw_guides():
        nonlocal center_marker, guide_lines

        for ln in guide_lines:
            ln.remove()
        guide_lines = []

        if center_marker is not None:
            center_marker.remove()
            center_marker = None

        if beam_center is None or n_slices <= 0:
            return

        xc, yc = beam_center
        radius = math.hypot(w, h)
        angles = np.linspace(0.0, 2.0 * np.pi, n_slices, endpoint=False)

        for ang in angles:
            x1 = xc + radius * math.cos(ang)
            y1 = yc + radius * math.sin(ang)
            ln = ax.plot([xc, x1], [yc, y1], linestyle=":", color="cyan", alpha=0.6)[0]
            guide_lines.append(ln)

        center_marker = ax.plot(
            xc, yc, marker="+", markersize=6, markeredgewidth=1.4, color="cyan"
        )[0]

    draw_guides()

    def update_title():
        title_text.set_text(
            f"Ellipse {current_ellipse + 1}/{n_ellipses}  "
            f"point {current_point}/{pts_per_ellipse}  "
            f"(Left: point, Right drag: zoom, 'r': reset)"
        )

    def on_button_press(event):
        nonlocal current_ellipse, current_point, all_points

        if event.inaxes is not ax:
            return

        if event.button == 1:
            if event.xdata is None or event.ydata is None:
                return
            if current_ellipse >= n_ellipses:
                return

            x, y = float(event.xdata), float(event.ydata)
            ell_points[current_ellipse].append((x, y))
            all_points.append((x, y))
            xs, ys = zip(*all_points)
            seed_scatter.set_offsets(np.column_stack([xs, ys]))

            current_point += 1
            print(f"Ellipse {current_ellipse + 1}, point {current_point}: x={x:.1f}, y={y:.1f}")

            if current_point >= pts_per_ellipse:
                current_ellipse += 1
                current_point = 0
                if current_ellipse >= n_ellipses:
                    update_title()
                    fig.canvas.draw_idle()
                    plt.close(fig)
                    return

            update_title()
            fig.canvas.draw_idle()

        elif event.button == 3:
            if event.xdata is None or event.ydata is None:
                return
            zooming["active"] = True
            zooming["x0"] = event.xdata
            zooming["y0"] = event.ydata
            zoom_rect.set_visible(True)
            zoom_rect.set_xy((event.xdata, event.ydata))
            zoom_rect.set_width(0)
            zoom_rect.set_height(0)
            fig.canvas.draw_idle()

    def on_motion(event):
        if not zooming["active"]:
            return
        if event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0 = zooming["x0"]
        y0 = zooming["y0"]
        x1 = event.xdata
        y1 = event.ydata

        x_min = min(x0, x1)
        x_max = max(x0, x1)
        y_min = min(y0, y1)
        y_max = max(y0, y1)

        zoom_rect.set_xy((x_min, y_min))
        zoom_rect.set_width(x_max - x_min)
        zoom_rect.set_height(y_max - y_min)
        fig.canvas.draw_idle()

    def on_button_release(event):
        if not zooming["active"]:
            return
        if event.button != 3:
            return

        zooming["active"] = False
        zoom_rect.set_visible(False)

        if event.inaxes is not ax:
            fig.canvas.draw_idle()
            return
        if event.xdata is None or event.ydata is None:
            fig.canvas.draw_idle()
            return

        x0 = zooming["x0"]
        y0 = zooming["y0"]
        x1 = event.xdata
        y1 = event.ydata

        if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:
            fig.canvas.draw_idle()
            return

        x_min = min(x0, x1)
        x_max = max(x0, x1)
        y_min = min(y0, y1)
        y_max = max(y0, y1)

        y_lim0, y_lim1 = ax.get_ylim()
        if y_lim0 > y_lim1:
            ax.set_ylim(max(y_min, y_max), min(y_min, y_max))
        else:
            ax.set_ylim(min(y_min, y_max), max(y_min, y_max))
        ax.set_xlim(min(x_min, x_max), max(x_min, x_max))
        fig.canvas.draw_idle()

    def on_key_press(event):
        if event.key == "r":
            ax.set_xlim(*full_xlim)
            ax.set_ylim(*full_ylim)
            fig.canvas.draw_idle()

    cid_press = fig.canvas.mpl_connect("button_press_event", on_button_press)
    cid_release = fig.canvas.mpl_connect("button_release_event", on_button_release)
    cid_motion = fig.canvas.mpl_connect("motion_notify_event", on_motion)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key_press)

    plt.tight_layout()
    plt.show()

    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_release)
    fig.canvas.mpl_disconnect(cid_motion)
    fig.canvas.mpl_disconnect(cid_key)

    for i, pts in enumerate(ell_points, 1):
        print(f"Ellipse {i} collected {len(pts)} points.")

    return ell_points


# ------------------------------------------------------------
# Ellipse fitting and refinement
# ------------------------------------------------------------
def _format_ellipse_lines(ellipses):
    lines = []
    for i, e in enumerate(ellipses):
        lines.append(
            "  "
            f"{i}: xc={e['xc']:.2f}, yc={e['yc']:.2f}, "
            f"a={e['a']:.2f}, b={e['b']:.2f}, "
            f"theta={np.degrees(e['theta']):.2f} deg"
        )
    return "\n".join(lines)


def plot_ellipses(img_bgsub, ellipses, save_path=None):
    disp = _hbn_backend.build_display(_hbn_backend.make_log_image(img_bgsub), 1)
    h, w = disp.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(disp, cmap="gray", origin="upper")

    for e in ellipses:
        x_curve, y_curve = _hbn_backend.ellipse_curve(e["xc"], e["yc"], e["a"], e["b"], e["theta"])
        ax.plot(x_curve, y_curve, "r-", linewidth=1.5)
        ax.scatter(e["xc"], e["yc"], s=20, c="yellow", marker="+")
    if ellipses:
        xc0, yc0 = _hbn_backend.ellipse_center(ellipses)
        ax.scatter(xc0, yc0, s=40, c="cyan", marker="x")
        print(f"Common center estimate: xc={xc0:.2f}, yc={yc0:.2f}")

        # Show the fitted parameters alongside the real image overlay.
        lines = _format_ellipse_lines(ellipses)
        if lines:
            ax.text(
                0.01,
                0.99,
                "Ellipses (xc, yc, a, b, theta):\n" + lines,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=6),
            )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title(f"{len(ellipses)} ellipses (clicked points, intensity refined)")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"Saved overlay image with ellipses to:\n  {save_path}")
    plt.show()


def plot_tilt_correction_overlay(
    img_bgsub,
    ellipses,
    ell_points_ds,
    corrected_points,
    center,
    radii_before,
    radii_after,
    tilt_x_deg,
    tilt_y_deg,
    distance_info=None,
    save_path=None,
):
    xc, yc = center
    theta = np.linspace(0.0, 2.0 * np.pi, 720)

    fig, ax = plt.subplots(figsize=(6, 6))

    if img_bgsub is not None:
        disp = _hbn_backend.build_display(_hbn_backend.make_log_image(img_bgsub), 1)
        ax.imshow(disp, cmap="gray", origin="upper")

    all_pts = []
    for pts in ell_points_ds:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2 and pts.size > 0:
            all_pts.append(pts)
    for pts in corrected_points:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2 and pts.size > 0:
            all_pts.append(pts)

    if all_pts:
        all_xy = np.vstack(all_pts)
        xmin, xmax = np.min(all_xy[:, 0]), np.max(all_xy[:, 0])
        ymin, ymax = np.min(all_xy[:, 1]), np.max(all_xy[:, 1])
    elif img_bgsub is not None:
        height, width = np.asarray(img_bgsub).shape[:2]
        xmin, xmax = 0.0, float(width)
        ymin, ymax = 0.0, float(height)
    else:
        radius = max(
            (abs(float(value)) for ellipse in ellipses for value in (ellipse["a"], ellipse["b"])),
            default=1.0,
        )
        xmin, xmax = xc - radius, xc + radius
        ymin, ymax = yc - radius, yc + radius

    for e in ellipses:
        x_curve, y_curve = _hbn_backend.ellipse_curve(e["xc"], e["yc"], e["a"], e["b"], e["theta"])
        ax.plot(x_curve, y_curve, "r-", linewidth=1.0, label="_orig_fit")

    for pts in ell_points_ds:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2:
            ax.plot(pts[:, 0], pts[:, 1], ".", markersize=1, alpha=0.35, label="_orig_pts")

    for pts in corrected_points:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2:
            ax.plot(pts[:, 0], pts[:, 1], ".", markersize=1, alpha=0.7, label="_corr_pts")

    for r in radii_before:
        if not np.isfinite(r) or r <= 0:
            continue
        x_circ = xc + r * np.cos(theta)
        y_circ = yc + r * np.sin(theta)
        ax.plot(x_circ, y_circ, linestyle="--", linewidth=1, color="orange", label="_orig_circle")

    for r in radii_after:
        if not np.isfinite(r) or r <= 0:
            continue
        x_circ = xc + r * np.cos(theta)
        y_circ = yc + r * np.sin(theta)
        ax.plot(x_circ, y_circ, linewidth=1.2, color="cyan", label="_corr_circle")

    ax.axhline(y=yc, linestyle="-.", linewidth=1.0, color="white", label="x tilt axis")
    ax.axvline(x=xc, linestyle="-.", linewidth=1.0, color="white", label="y tilt axis")

    ax.plot(xc, yc, "x", markersize=6, color="yellow", label="center")

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)

    title = (
        f"Tilt-corrected rings on image\ntilt_x={tilt_x_deg:.2f} deg, tilt_y={tilt_y_deg:.2f} deg"
    )
    if distance_info:
        mean_m = distance_info.get("mean_m")
        basis = distance_info.get("basis")
        if mean_m is not None:
            basis_label = f" ({basis})" if basis else ""
            title += f"\nshared distance L={mean_m:.4f} m{basis_label}"
    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.legend(loc="best")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"Saved tilt-corrected overlay to:\n  {save_path}")
    plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_hbn_fit(
    osc_path,
    dark_path,
    output_dir=None,
    load_bundle=None,
    highres_refine=False,
    reclick=False,
    paths_file=None,
    load_clicks=None,
    save_clicks=None,
    clicks_only=False,
    beam_center=None,
):
    resolved = _hbn_geometry.resolve_hbn_paths(
        osc_path=osc_path,
        dark_path=dark_path,
        paths_file=paths_file,
    )

    if resolved.get("paths_file"):
        print(f"Loaded hBN paths from file:\n  {resolved['paths_file']}")

    osc_path = resolved["osc"]
    dark_path = resolved["dark"]
    configured_beam_center = resolved.get("beam_center")
    if beam_center is None and configured_beam_center is not None and load_bundle is None:
        beam_center = resolved["beam_center"]
    bundle_path_in = load_bundle
    if bundle_path_in is not None and output_dir is None:
        raise ValueError(
            "output_dir is required when loading a bundle; choose a distinct output directory."
        )

    output_dir = (
        output_dir
        or os.path.join(os.path.expanduser("~"), "Downloads")
    )
    os.makedirs(output_dir, exist_ok=True)
    out_tiff_path = os.path.join(output_dir, "hbn_bgsub.tiff")
    out_overlay_path = os.path.join(output_dir, "hbn_bgsub_ellipses.png")
    out_tilt_overlay_path = os.path.join(output_dir, "hbn_bgsub_ellipses_tilt_corrected.png")

    click_profile_in = load_clicks or resolved.get("click_profile")
    click_profile_out = None
    if save_clicks is not None or clicks_only:
        click_profile_out = save_clicks
        if click_profile_out in (None, ""):
            click_profile_out = os.path.join(output_dir, "hbn_click_profile.json")
    if (
        click_profile_in
        and click_profile_out
        and os.path.normcase(os.path.abspath(click_profile_in))
        == os.path.normcase(os.path.abspath(click_profile_out))
    ):
        raise ValueError("The output click-profile path must differ from the loaded profile path.")

    bundle_path_save = os.path.join(output_dir, "hbn_ellipse_bundle.npz")
    if bundle_path_in is not None and os.path.normcase(
        os.path.abspath(bundle_path_in)
    ) == os.path.normcase(os.path.abspath(bundle_path_save)):
        raise ValueError("The output bundle path must differ from the loaded bundle path.")

    for p in [bundle_path_save, click_profile_out]:
        if not p:
            continue
        parent = os.path.dirname(os.path.abspath(p))
        os.makedirs(parent, exist_ok=True)

    outputs = {
        "output_dir": output_dir,
        "background_subtracted": out_tiff_path,
        "overlay": out_overlay_path,
        "tilt_overlay": out_tilt_overlay_path,
        "bundle": bundle_path_save,
        "click_profile": click_profile_out,
        "ellipses": [],
        "tilt_hint": None,
        "expected_peaks": None,
        "distance_estimate_m": None,
        "tilt_correction": None,
        "aborted": False,
        "abort_reason": None,
    }

    click_profile_saved = False

    bundle_loaded = None
    center_from_bundle = None
    center_common = None
    if bundle_path_in is not None:
        bundle_loaded = _hbn_geometry.load_bundle_npz(bundle_path_in)
        center_from_bundle = bundle_loaded[8]
        if center_from_bundle is not None:
            center_common = center_from_bundle
        elif beam_center is None and configured_beam_center is not None:
            beam_center = configured_beam_center

    if beam_center is not None and len(beam_center) == 2:
        center_common = tuple(beam_center)

    # Shared output containers
    img_bgsub_out = None
    img_log_out = None
    ell_points_ds = None
    ellipses_out = None
    tilt_correction = None
    tilt_hint = None
    expected_peaks = None

    abort_reason = None
    completed = False

    try:
        if bundle_loaded is not None and not reclick:
            (
                img_bgsub_b,
                img_log_b,
                ell_points_ds,
                ellipses_b,
                _,
                tilt_correction,
                tilt_hint,
                expected_peaks,
                center_common,
            ) = bundle_loaded

            if beam_center is not None and len(beam_center) == 2:
                center_common = tuple(beam_center)

            if highres_refine:
                if osc_path is None or dark_path is None:
                    raise ValueError(
                        "Refitting a bundle requires both --osc and --dark so the background "
                        "image can be recomputed."
                    )

                print("Refitting bundle ellipses at full resolution...")
                img_bgsub_out = _hbn_backend.load_and_bgsub(osc_path, dark_path)
                img_log_out = _hbn_backend.make_log_image(img_bgsub_out)
            else:
                img_bgsub_out = img_bgsub_b
                img_log_out = img_log_b

            scale_x = img_bgsub_out.shape[1] / img_bgsub_b.shape[1]
            scale_y = img_bgsub_out.shape[0] / img_bgsub_b.shape[0]
            ell_points_ds = [
                [(float(x) * scale_x, float(y) * scale_y) for x, y in ring]
                for ring in ell_points_ds
            ]
            scaled_initial = [
                {
                    **ellipse,
                    "xc": float(ellipse["xc"]) * scale_x,
                    "yc": float(ellipse["yc"]) * scale_y,
                    "a": float(ellipse["a"]) * scale_x,
                    "b": float(ellipse["b"]) * scale_y,
                }
                for ellipse in ellipses_b
            ]
            ellipses_out = _hbn_backend.fit_ellipses(
                ell_points_ds,
                1,
                img_bgsub_out,
                REFINE_N_ANGLES,
                REFINE_DR,
                REFINE_STEP,
                initial_ellipses=scaled_initial,
                img_log=img_log_out,
            )
        else:
            if reclick and bundle_loaded is not None:
                print(
                    "Reclick requested: ignoring stored clicks in bundle and collecting new points."
                )

            if osc_path is None or dark_path is None:
                raise ValueError(
                    "Both --osc and --dark are required unless --load-bundle is used "
                    "without --highres-refine."
                )

            img_bgsub_out = _hbn_backend.load_and_bgsub(osc_path, dark_path)

            if click_profile_in and not reclick:
                if not os.path.exists(click_profile_in):
                    print(
                        "Click profile not found; collecting new clicks instead:\n  "
                        f"{click_profile_in}"
                    )
                    click_profile_in = None
                else:
                    ell_points_ds = load_click_profile(
                        click_profile_in,
                        expected_image_shape=img_bgsub_out.shape,
                    )
                    if outputs["click_profile"] is None:
                        outputs["click_profile"] = click_profile_in

            if ell_points_ds is None:
                print(
                    f"Interactive picking: collect {POINTS_PER_ELLIPSE} points on each of "
                    f"{N_ELLIPSES} rings (left click = point, right drag = zoom, 'r' = reset)."
                )
                small = _hbn_backend.build_display(_hbn_backend.make_log_image(img_bgsub_out), 1)
                guide_center = center_common or center_from_bundle
                if guide_center is not None:
                    print(
                        "Beam center guide enabled at "
                        f"x={guide_center[0]:.2f}, y={guide_center[1]:.2f}; "
                        f"drawing {N_ELLIPSES} radial guides."
                    )
                ell_points_ds = get_points_per_ellipse_with_zoom(
                    small,
                    n_ellipses=N_ELLIPSES,
                    pts_per_ellipse=POINTS_PER_ELLIPSE,
                    beam_center=guide_center,
                    n_slices=N_ELLIPSES,
                )

            if click_profile_out and ell_points_ds:
                save_click_profile(click_profile_out, ell_points_ds, img_bgsub_out.shape)
                outputs["click_profile"] = click_profile_out
                click_profile_saved = True

            expected_points = N_ELLIPSES * POINTS_PER_ELLIPSE
            collected_points = (
                0 if ell_points_ds is None else sum(len(pts) for pts in ell_points_ds)
            )

            if clicks_only:
                if collected_points < expected_points:
                    abort_reason = (
                        "Ellipse picking did not finish; collected "
                        f"{collected_points}/{expected_points} points. Skipping save."
                    )
                else:
                    print("Click-only mode requested; returning after saving collected points.")
                    outputs["clicks_only"] = True
                    outputs["ellipses"] = []
                    outputs["aborted"] = True
                    outputs["abort_reason"] = "Click-only save requested; ellipse fitting skipped."
                    return outputs

            if collected_points < expected_points:
                abort_reason = (
                    "Ellipse picking did not finish; collected "
                    f"{collected_points}/{expected_points} points. Skipping save."
                )
            else:
                ellipses_out = _hbn_backend.fit_ellipses(
                    ell_points_ds,
                    1,
                    img_bgsub_out,
                    REFINE_N_ANGLES,
                    REFINE_DR,
                    REFINE_STEP,
                )
                print(
                    f"Fitted {len(ellipses_out)} ellipses from clicked points and intensity refinement."
                )

            img_log_out = _hbn_backend.make_log_image(img_bgsub_out)

        if click_profile_out and ell_points_ds and not clicks_only and not click_profile_saved:
            save_click_profile(click_profile_out, ell_points_ds, img_bgsub_out.shape)
            outputs["click_profile"] = click_profile_out
            click_profile_saved = True

        if abort_reason:
            outputs["aborted"] = True
            outputs["abort_reason"] = abort_reason
            print(abort_reason)
            return outputs

        if ellipses_out and center_common is None:
            center_common = _hbn_backend.ellipse_center(ellipses_out)
        if center_common is None:
            raise ValueError("A finite hBN center is required for projective tilt fitting.")
        point_sets = [
            np.asarray(points, dtype=float).reshape(-1, 2)
            for points in ell_points_ds
            if len(points) >= 3
        ]
        tilt_correction = _hbn_backend.optimize_tilts_projective(
            point_sets,
            center_common,
            optimize_center=True,
            center_prior=center_common if beam_center is not None else None,
        )
        tilt_hint = _hbn_backend.enrich_projective_calibration(tilt_correction)
        center_common = tuple(float(value) for value in tilt_correction["center"])
        expected_peaks = tilt_correction["expected_peaks"]
        distance_info = tilt_correction["distance_estimate_m"]
        print(
            "Projective tilt fit completed: "
            f"tilt_x={tilt_correction['tilt_x_deg']:.4f} deg, "
            f"tilt_y={tilt_correction['tilt_y_deg']:.4f} deg, "
            f"cost={tilt_correction['cost_final']:.6e}."
        )
        print(
            "Estimated detector tilt from projective hBN fit: "
            f"Rot1={tilt_hint['rot1_rad']:.4f} rad, "
            f"Rot2={tilt_hint['rot2_rad']:.4f} rad"
        )
        if expected_peaks:
            print("Expected hBN peaks for Cu K-alpha:")
            for i, peak in enumerate(expected_peaks, 1):
                h, k, l = peak["hkl"]
                print(
                    f"  Ring {i}: hkl=({h}{k}{l}) "
                    f"d={peak['d_spacing_ang']:.4f} Angstrom "
                    f"2theta={peak['two_theta_deg']:.2f} deg"
                )
        if distance_info:
            basis_label = distance_info.get("basis", "ellipses")
            print(
                "Estimated sample-detector distance (using matched rings): "
                f"mean={distance_info['mean_m']:.4f} m "
                f"(basis={basis_label})"
            )
            for i, dist in enumerate(distance_info["per_ring_m"], 1):
                print(f"  Ring {i}: {dist:.4f} m")
        completed = True
    except KeyboardInterrupt:
        abort_reason = "hBN fitting interrupted by user; skipping save."

    if not completed:
        outputs["aborted"] = True
        outputs["abort_reason"] = abort_reason
        if abort_reason:
            print(abort_reason)
        return outputs

    print(f"Saving background-subtracted image to:\n  {out_tiff_path}")
    if not cv2.imwrite(out_tiff_path, img_bgsub_out):
        raise OSError(f"OpenCV could not write the output image: {out_tiff_path}")
    point_sigmas = [np.ones(len(points), dtype=float) for points in ell_points_ds]
    bundle_payload = _hbn_backend.build_hbn_fitter_bundle_payload(
        img_bgsub=img_bgsub_out,
        img_log_full=img_log_out,
        downsample_factor=1,
        center=center_common,
        center_source="beam_center" if beam_center is not None else "ellipse_fit",
        optim=tilt_correction,
        fit_quality={},
        points_ds=ell_points_ds,
        points_raw_ds=ell_points_ds,
        points_sigma_ds=point_sigmas,
        ellipses=ellipses_out,
        input_hbn_path=osc_path or "",
        input_dark_path=dark_path or "",
    )
    np.savez_compressed(bundle_path_save, **bundle_payload)

    plot_ellipses(img_bgsub_out, ellipses_out, save_path=out_overlay_path)

    if tilt_correction:
        corrected_points = tilt_correction.get("corrected_points", [])
        radii_before = tilt_correction.get("radii_before", [])
        radii_after = tilt_correction.get("radii_after_fit") or tilt_correction.get(
            "radii_after", []
        )
        plot_tilt_correction_overlay(
            img_bgsub_out,
            ellipses_out,
            ell_points_ds,
            corrected_points,
            center_common,
            radii_before,
            radii_after,
            tilt_correction.get("tilt_x_deg", 0.0),
            tilt_correction.get("tilt_y_deg", 0.0),
            distance_info=distance_info,
            save_path=out_tilt_overlay_path,
        )

    if ellipses_out:
        print("Fitted ellipse parameters:")
        print(_format_ellipse_lines(ellipses_out))

    outputs["ellipses"] = ellipses_out
    outputs["tilt_hint"] = tilt_hint
    outputs["expected_peaks"] = expected_peaks
    outputs["distance_estimate_m"] = distance_info
    outputs["tilt_correction"] = tilt_correction
    return outputs
